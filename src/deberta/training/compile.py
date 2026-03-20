"""Torch compile and attention-mask helpers for pretraining."""

from __future__ import annotations

import logging
import os
import types
from typing import Any

import torch

from deberta.config import ModelConfig, _normalize_sdpa_kernel
from deberta.modeling.mask_utils import normalize_keep_mask

logger = logging.getLogger(__name__)
_DOC_BLOCK_EYE_CACHE: dict[tuple[int, str, int | None], torch.Tensor] = {}
_DOC_BLOCK_CLS_KEY_CACHE: dict[tuple[int, str, int | None], torch.Tensor] = {}
_FLASH_TRUTHY = {"1", "true", "yes", "y", "on"}


def _maybe_enable_tf32(enabled: bool) -> None:
    """Configure TF32 compute policy for CUDA matmul/cudnn.

    :param bool enabled: Whether to enable TF32.
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)


def _maybe_configure_sdpa_kernels(policy: str, *, is_main: bool) -> None:
    """Configure PyTorch SDPA backend toggles on CUDA.

    :param str policy: SDPA kernel policy.
    :param bool is_main: Whether current process should emit logs.
    """
    if not torch.cuda.is_available():
        return

    policy = _normalize_sdpa_kernel(policy)

    enable_flash = True
    enable_mem_efficient = True
    enable_math = True

    if policy == "flash":
        enable_mem_efficient = False
        enable_math = False
    elif policy == "mem_efficient":
        enable_flash = False
    elif policy == "math":
        enable_flash = False
        enable_mem_efficient = False

    try:
        cuda_backend = getattr(torch.backends, "cuda", None)
        if cuda_backend is not None:
            for name, enabled in (
                ("enable_flash_sdp", enable_flash),
                ("enable_mem_efficient_sdp", enable_mem_efficient),
                ("enable_math_sdp", enable_math),
            ):
                fn = getattr(cuda_backend, name, None)
                if callable(fn):
                    fn(bool(enabled))
    except Exception:
        # Best-effort only; do not fail training if this backend API changes.
        pass

    if is_main:
        logger.info(
            "SDPA kernel policy=%s (requested flash=%s, mem_efficient=%s, math=%s).",
            policy,
            enable_flash,
            enable_mem_efficient,
            enable_math,
        )


def _bf16_runtime_sanity_check() -> bool:
    """Check whether bf16 autocast executes a tiny CUDA matmul.

    :return bool: True when a tiny bf16 autocast path succeeds.
    """
    if not torch.cuda.is_available():
        logger.error("bf16 mixed precision requested but CUDA is not available.")
        return False
    if not torch.cuda.is_bf16_supported():
        logger.error("bf16 mixed precision requested but this CUDA device reports no bf16 support.")
        return False

    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            a = torch.randn((64, 64), device="cuda", dtype=torch.float32)
            b = torch.randn((64, 64), device="cuda", dtype=torch.float32)
            c = a @ b
            _ = c.sum().item()
        return True
    except Exception:
        logger.error("bf16 autocast preflight failed.", exc_info=True)
        return False


def _resolve_compile_enabled_or_raise(requested: bool) -> bool:
    """Return compile-enabled flag, raising when torch.compile is unavailable.

    :param bool requested: Whether compile was requested by config.
    :raises RuntimeError: If compile was requested but torch.compile is unavailable.
    :return bool: True when compile should be enabled.
    """
    if not bool(requested):
        return False
    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "train.torch_compile=true requested but this PyTorch build does not expose torch.compile."
        )
    return True


def _flash_truthy_env(name: str, default: str = "0") -> bool:
    """Return whether an env var is set to a truthy value.

    :param str name: Environment variable name.
    :param str default: Default text when unset.
    :return bool: Parsed truthy value.
    """

    return os.environ.get(name, default).strip().lower() in _FLASH_TRUTHY


def _flash_int_env(name: str, default: int) -> int:
    """Parse an integer env var with fallback.

    :param str name: Environment variable name.
    :param int default: Default integer.
    :return int: Parsed value or default.
    """

    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _flash_mask_to_2d_keep_mask(attention_mask: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Extract a canonical ``(B,S)`` keep mask from rank-2/4 padding masks.

    :param torch.Tensor attention_mask: Padding-style keep mask.
    :param int seq_len: Expected sequence length.
    :raises ValueError: If the mask is not 2D or broadcast 4D.
    :return torch.Tensor: Boolean keep mask in ``(B,S)`` layout.
    """

    mask = normalize_keep_mask(attention_mask)
    if mask.ndim == 2:
        return mask[:, :seq_len]
    if mask.ndim == 4 and int(mask.shape[-2]) == 1:
        return mask[:, 0, 0, :seq_len]
    raise ValueError(f"Unsupported padding-mask shape for flash metadata: {tuple(mask.shape)}")


def _flash_is_pairwise_mask(attention_mask: torch.Tensor, *, seq_len: int) -> bool:
    """Return whether a mask carries per-query pairwise structure.

    :param torch.Tensor attention_mask: Candidate mask tensor.
    :param int seq_len: Expected query/key length.
    :return bool: True for ``(B,S,S)`` or ``(B,1,S,S)`` style masks.
    """

    mask = normalize_keep_mask(attention_mask)
    if mask.ndim == 3:
        return tuple(mask.shape[-2:]) == (int(seq_len), int(seq_len))
    if mask.ndim == 4:
        return tuple(mask.shape[-2:]) == (int(seq_len), int(seq_len))
    return False


def _flash_density_bucket(*, seq_len: int, active_tokens: int, batch_size: int) -> str:
    """Return the repo-local density bucket for one padded batch.

    :param int seq_len: Padded sequence length.
    :param int active_tokens: Total active tokens across the batch.
    :param int batch_size: Batch size.
    :return str: Density bucket label.
    """

    capacity = max(1, int(seq_len) * max(1, int(batch_size)))
    density = float(active_tokens) / float(capacity)
    if int(seq_len) >= 4096:
        return "4096_plus"
    if int(seq_len) >= 2048:
        return "2048_medium" if density >= 0.60 else "2048_sparse"
    return "1024_dense_or_medium"


def _flash_route_hint_for_padding_batch(
    *,
    seq_len: int,
    active_tokens: int,
    batch_size: int,
) -> str:
    """Select a fixed-vs-varlen route for one standard padded batch.

    :param int seq_len: Padded sequence length.
    :param int active_tokens: Total active tokens across the batch.
    :param int batch_size: Batch size.
    :return str: Either ``fixed`` or ``varlen``.
    """

    if _flash_truthy_env("FLASHDEBERTA_FORCE_VARLEN", default="0"):
        return "varlen"
    density_bucket = _flash_density_bucket(
        seq_len=int(seq_len),
        active_tokens=int(active_tokens),
        batch_size=int(batch_size),
    )
    route_by_bucket = {
        "1024_dense_or_medium": "fixed",
        "2048_medium": "varlen",
        "2048_sparse": "varlen",
        "4096_plus": "varlen",
    }
    default_varlen_min_seq_len = max(1, _flash_int_env("FLASHDEBERTA_VARLEN_MIN_SEQ_LEN", 2048))
    default_route = "varlen" if int(seq_len) >= int(default_varlen_min_seq_len) else "fixed"
    return route_by_bucket.get(density_bucket, default_route)


def prepare_flash_attention_batch_metadata(
    *,
    batch: dict[str, Any],
    backbone_type: str,
) -> tuple[dict[str, Any], str | None]:
    """Attach precomputed flash metadata and return an out-of-graph route hint.

    :param dict[str, Any] batch: Device-local batch mapping.
    :param str backbone_type: Backbone type string.
    :return tuple[dict[str, Any], str | None]: Updated batch and selected route hint.
    """

    btype = str(backbone_type).strip().lower()
    if btype != "hf_deberta_v2":
        batch.pop("flash_seq_lengths", None)
        batch.pop("flash_active_tokens", None)
        return batch, None

    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
        batch.pop("flash_seq_lengths", None)
        batch.pop("flash_active_tokens", None)
        return batch, None

    attention_mask = batch.get("attention_mask")
    seq_len = int(input_ids.shape[-1])
    if attention_mask is None:
        batch.pop("flash_seq_lengths", None)
        batch.pop("flash_active_tokens", None)
        return batch, "dense"
    if _flash_is_pairwise_mask(attention_mask, seq_len=int(seq_len)):
        batch.pop("flash_seq_lengths", None)
        batch.pop("flash_active_tokens", None)
        return batch, "pairwise"

    if not isinstance(attention_mask, torch.Tensor):
        batch.pop("flash_seq_lengths", None)
        batch.pop("flash_active_tokens", None)
        return batch, None

    keep_mask = _flash_mask_to_2d_keep_mask(attention_mask, seq_len=seq_len)
    seq_lengths = keep_mask.sum(dim=-1, dtype=torch.int32)
    active_tokens = int(seq_lengths.sum(dtype=torch.int32).item())
    route_hint = _flash_route_hint_for_padding_batch(
        seq_len=seq_len,
        active_tokens=active_tokens,
        batch_size=int(input_ids.shape[0]),
    )
    batch["flash_seq_lengths"] = seq_lengths
    batch["flash_active_tokens"] = torch.tensor(active_tokens, device=seq_lengths.device, dtype=torch.int32)
    return batch, route_hint


def _maybe_cudagraph_mark_step_begin() -> None:
    """Mark cudagraph step boundaries when the API is available.

    This is a no-op on PyTorch builds that do not expose ``torch.compiler``
    or ``cudagraph_mark_step_begin``.
    """
    try:
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
    except Exception:
        # Best-effort: keep training running if backend API shape changes.
        pass


def _resolve_compile_scope(
    *,
    requested_scope: str,
    model_cfg: ModelConfig,
    block_cross_document_attention: bool = False,
) -> tuple[str, str | None]:
    """Resolve effective compile scope for auto compile mode.

    :param str requested_scope: Requested canonical compile scope.
    :param ModelConfig model_cfg: Model configuration.
    :param bool block_cross_document_attention: Whether doc-blocking is enabled.
    :return tuple[str, str | None]: Effective scope and optional reason message.
    """
    if requested_scope != "auto":
        return requested_scope, None

    backbone_type = str(getattr(model_cfg, "backbone_type", "")).strip().lower()
    if bool(block_cross_document_attention) and backbone_type == "rope":
        return (
            "ffn",
            "auto scope selected FFN-only for rope + doc-blocking (mask shape churn under compile)",
        )
    return "backbones", None


def _compile_backbones_for_scope(
    *,
    unwrapped_model: torch.nn.Module,
    compile_scope: str,
    compile_kwargs: dict[str, Any],
) -> list[str]:
    """Compile selected generator/discriminator submodules for a requested scope.

    :param torch.nn.Module unwrapped_model: Unwrapped RTD pretrainer model.
    :param str compile_scope: Effective compile scope.
    :param dict[str, Any] compile_kwargs: Keyword arguments passed to ``torch.compile``.
    :raises RuntimeError: If required backbone submodules are missing.
    :return list[str]: Human-readable list of compiled targets.
    """
    generator = getattr(unwrapped_model, "generator", None)
    discriminator = getattr(unwrapped_model, "discriminator", None)
    if not isinstance(generator, torch.nn.Module) or not isinstance(discriminator, torch.nn.Module):
        raise RuntimeError("RTD model must expose generator and discriminator modules for compilation.")

    compiled_targets: list[str] = []

    def _compile_module_forward(*, module: torch.nn.Module, target: str) -> None:
        """Compile one module's ``forward`` method in-place.

        Keeping module identity stable avoids ``._orig_mod`` wrapper keys in saved checkpoints.

        :param torch.nn.Module module: Module whose forward should be compiled.
        :param str target: Human-readable target name for logs.
        """
        if _install_stable_backbone_compile_dispatch(
            module=module,
            compile_kwargs=compile_kwargs,
            target=target,
            compiled_targets=compiled_targets,
        ):
            return
        forward = getattr(module, "forward", None)
        if not callable(forward):
            raise RuntimeError(f"{target}.forward is required for compile scope.")
        module.forward = torch.compile(forward, **compile_kwargs)  # type: ignore[assignment]
        compiled_targets.append(target)

    def _compile_encoder_ffn(*, backbone: torch.nn.Module, branch: str) -> None:
        """Compile FFN blocks in all encoder layers.

        Supports both HF DeBERTa-v2 (``encoder.layer[i].intermediate``/``.output``)
        and RoPE (``encoder.layers[i].mlp``) module layouts.

        :param torch.nn.Module backbone: Generator/discriminator backbone.
        :param str branch: Branch label used in compiled target names.
        :raises RuntimeError: If encoder/layer/FFN modules are missing.
        """
        encoder = getattr(backbone, "encoder", None)
        if not isinstance(encoder, torch.nn.Module):
            raise RuntimeError(f"{branch}.encoder is required for ffn-only compile scope.")

        # Try HF DeBERTa-v2 convention (encoder.layer) then RoPE (encoder.layers).
        layers = getattr(encoder, "layer", None)
        layers_attr = "layer"
        if not isinstance(layers, (torch.nn.ModuleList, list, tuple)):
            layers = getattr(encoder, "layers", None)
            layers_attr = "layers"
        if not isinstance(layers, (torch.nn.ModuleList, list, tuple)):
            raise RuntimeError(f"{branch}.encoder.layer/layers is required for ffn-only compile scope.")
        if len(layers) == 0:
            raise RuntimeError(
                f"{branch}.encoder.{layers_attr} is empty; cannot apply ffn-only compile scope."
            )

        for idx, layer in enumerate(layers):
            if not isinstance(layer, torch.nn.Module):
                raise RuntimeError(f"{branch}.encoder.{layers_attr}[{idx}] is not a torch.nn.Module.")

            # HF DeBERTa-v2: intermediate + output modules.
            intermediate = getattr(layer, "intermediate", None)
            output = getattr(layer, "output", None)
            if isinstance(intermediate, torch.nn.Module) and isinstance(output, torch.nn.Module):
                pfx = f"{branch}.encoder.{layers_attr}[{idx}]"
                _compile_module_forward(module=intermediate, target=f"{pfx}.intermediate")
                _compile_module_forward(module=output, target=f"{pfx}.output")
                continue

            # RoPE: single mlp module.
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, torch.nn.Module):
                _compile_module_forward(module=mlp, target=f"{branch}.encoder.{layers_attr}[{idx}].mlp")
                continue

            raise RuntimeError(
                f"{branch}.encoder.{layers_attr}[{idx}] has no recognized FFN module "
                "(expected intermediate+output or mlp)."
            )

    if compile_scope == "backbones":
        _compile_module_forward(module=generator, target="generator")
        _compile_module_forward(module=discriminator, target="discriminator")
        return compiled_targets

    if compile_scope in {"encoder", "gen_encoder"}:
        gen_encoder = getattr(generator, "encoder", None)
        if not isinstance(gen_encoder, torch.nn.Module):
            raise RuntimeError("generator.encoder is required for encoder-only compile scope.")
        _compile_module_forward(module=gen_encoder, target="generator.encoder")

    if compile_scope in {"encoder", "disc_encoder"}:
        disc_encoder = getattr(discriminator, "encoder", None)
        if not isinstance(disc_encoder, torch.nn.Module):
            raise RuntimeError("discriminator.encoder is required for encoder-only compile scope.")
        _compile_module_forward(module=disc_encoder, target="discriminator.encoder")

    if compile_scope in {"ffn", "gen_ffn"}:
        _compile_encoder_ffn(backbone=generator, branch="generator")

    if compile_scope in {"ffn", "disc_ffn"}:
        _compile_encoder_ffn(backbone=discriminator, branch="discriminator")

    if not compiled_targets:
        raise ValueError(f"Unsupported compile scope: {compile_scope}")
    return compiled_targets


def _install_stable_backbone_compile_dispatch(
    *,
    module: torch.nn.Module,
    compile_kwargs: dict[str, Any],
    target: str,
    compiled_targets: list[str],
) -> bool:
    """Install a stable compiled dense/masked dispatcher when supported.

    Native HF DeBERTa backbones accept Python optionals in ``forward`` for
    ``attention_mask`` and output flags. Compiling that public ``forward``
    directly encourages Dynamo to guard on ``None``/bool optionals. When the
    backbone exposes resolved dense/masked helpers, compile those stable
    entrypoints instead and leave ``forward`` as a tiny Python dispatcher.

    :param torch.nn.Module module: Candidate backbone module.
    :param dict[str, Any] compile_kwargs: Keyword arguments passed to ``torch.compile``.
    :param str target: Human-readable target name for logs.
    :param list[str] compiled_targets: Accumulator for compiled target labels.
    :return bool: True when a stable dispatcher was installed.
    """

    resolve_options = getattr(module, "_resolve_forward_options", None)
    dense_hs0 = getattr(module, "_forward_dense_hs0", None)
    dense_hs1 = getattr(module, "_forward_dense_hs1", None)
    masked_hs0 = getattr(module, "_forward_masked_hs0", None)
    masked_hs1 = getattr(module, "_forward_masked_hs1", None)
    dense_forward = getattr(module, "_forward_dense_resolved", None)
    masked_forward = getattr(module, "_forward_masked_resolved", None)
    if not callable(resolve_options):
        return False

    if callable(dense_hs0) and callable(dense_hs1) and callable(masked_hs0) and callable(masked_hs1):
        dense_hs0_fn = dense_hs0
        dense_hs1_fn = dense_hs1
        masked_hs0_fn = masked_hs0
        masked_hs1_fn = masked_hs1
    else:
        if not callable(dense_forward) or not callable(masked_forward):
            return False

        def _dense_hs0_fn(
            *,
            input_ids: torch.Tensor | None = None,
            token_type_ids: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
        ) -> Any:
            """Call the generic dense helper with fixed ``hidden_states=False``.

            :param torch.Tensor | None input_ids: Optional input token ids.
            :param torch.Tensor | None token_type_ids: Optional token type ids.
            :param torch.Tensor | None position_ids: Optional position ids.
            :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
            :return Any: Dense-path backbone outputs.
            """

            return dense_forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

        def _dense_hs1_fn(
            *,
            input_ids: torch.Tensor | None = None,
            token_type_ids: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
        ) -> Any:
            """Call the generic dense helper with fixed ``hidden_states=True``.

            :param torch.Tensor | None input_ids: Optional input token ids.
            :param torch.Tensor | None token_type_ids: Optional token type ids.
            :param torch.Tensor | None position_ids: Optional position ids.
            :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
            :return Any: Dense-path backbone outputs.
            """

            return dense_forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

        def _masked_hs0_fn(
            *,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            flash_seq_lengths: torch.Tensor | None = None,
            flash_route_hint: str | None = None,
        ) -> Any:
            """Call the generic masked helper with fixed ``hidden_states=False``.

            :param torch.Tensor | None input_ids: Optional input token ids.
            :param torch.Tensor attention_mask: Attention mask tensor.
            :param torch.Tensor | None token_type_ids: Optional token type ids.
            :param torch.Tensor | None position_ids: Optional position ids.
            :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
            :param torch.Tensor | None flash_seq_lengths: Optional precomputed active lengths.
            :param str | None flash_route_hint: Optional flash routing hint.
            :return Any: Masked-path backbone outputs.
            """

            return masked_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                flash_seq_lengths=flash_seq_lengths,
                flash_route_hint=flash_route_hint,
            )

        def _masked_hs1_fn(
            *,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            flash_seq_lengths: torch.Tensor | None = None,
            flash_route_hint: str | None = None,
        ) -> Any:
            """Call the generic masked helper with fixed ``hidden_states=True``.

            :param torch.Tensor | None input_ids: Optional input token ids.
            :param torch.Tensor attention_mask: Attention mask tensor.
            :param torch.Tensor | None token_type_ids: Optional token type ids.
            :param torch.Tensor | None position_ids: Optional position ids.
            :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
            :param torch.Tensor | None flash_seq_lengths: Optional precomputed active lengths.
            :param str | None flash_route_hint: Optional flash routing hint.
            :return Any: Masked-path backbone outputs.
            """

            return masked_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                flash_seq_lengths=flash_seq_lengths,
                flash_route_hint=flash_route_hint,
            )

        dense_hs0_fn = _dense_hs0_fn
        dense_hs1_fn = _dense_hs1_fn
        masked_hs0_fn = _masked_hs0_fn
        masked_hs1_fn = _masked_hs1_fn

    def _masked_fixed_hs0_fn(
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        flash_seq_lengths: torch.Tensor | None = None,
    ) -> Any:
        """Call the masked helper with fixed-flash routing.

        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor attention_mask: Attention mask tensor.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param torch.Tensor | None flash_seq_lengths: Optional precomputed active lengths.
        :return Any: Masked-path backbone outputs.
        """

        return masked_hs0_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            flash_seq_lengths=flash_seq_lengths,
            flash_route_hint="fixed",
        )

    def _masked_fixed_hs1_fn(
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        flash_seq_lengths: torch.Tensor | None = None,
    ) -> Any:
        """Call the masked helper with fixed-flash routing and hidden states.

        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor attention_mask: Attention mask tensor.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param torch.Tensor | None flash_seq_lengths: Optional precomputed active lengths.
        :return Any: Masked-path backbone outputs.
        """

        return masked_hs1_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            flash_seq_lengths=flash_seq_lengths,
            flash_route_hint="fixed",
        )

    def _masked_varlen_hs0_fn(
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        flash_seq_lengths: torch.Tensor | None = None,
    ) -> Any:
        """Call the masked helper with varlen-flash routing.

        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor attention_mask: Attention mask tensor.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param torch.Tensor | None flash_seq_lengths: Optional precomputed active lengths.
        :return Any: Masked-path backbone outputs.
        """

        return masked_hs0_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            flash_seq_lengths=flash_seq_lengths,
            flash_route_hint="varlen",
        )

    def _masked_varlen_hs1_fn(
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        flash_seq_lengths: torch.Tensor | None = None,
    ) -> Any:
        """Call the masked helper with varlen-flash routing and hidden states.

        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor attention_mask: Attention mask tensor.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param torch.Tensor | None flash_seq_lengths: Optional precomputed active lengths.
        :return Any: Masked-path backbone outputs.
        """

        return masked_hs1_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            flash_seq_lengths=flash_seq_lengths,
            flash_route_hint="varlen",
        )

    compiled_dense = {
        False: torch.compile(dense_hs0_fn, **compile_kwargs),
        True: torch.compile(dense_hs1_fn, **compile_kwargs),
    }
    compiled_masked = {
        False: torch.compile(masked_hs0_fn, **compile_kwargs),
        True: torch.compile(masked_hs1_fn, **compile_kwargs),
    }
    compiled_masked_fixed = {
        False: torch.compile(_masked_fixed_hs0_fn, **compile_kwargs),
        True: torch.compile(_masked_fixed_hs1_fn, **compile_kwargs),
    }
    compiled_masked_varlen = {
        False: torch.compile(_masked_varlen_hs0_fn, **compile_kwargs),
        True: torch.compile(_masked_varlen_hs1_fn, **compile_kwargs),
    }

    def _dispatch_forward(
        self: torch.nn.Module,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        flash_seq_lengths: torch.Tensor | None = None,
        flash_route_hint: str | None = None,
    ) -> Any:
        """Normalize public options, then dispatch into stable compiled entrypoints.

        :param torch.nn.Module self: Backbone module instance.
        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param bool | None output_attentions: Optional attention-output flag.
        :param bool | None output_hidden_states: Optional hidden-state-output flag.
        :param bool | None return_dict: Optional return-format flag.
        :param torch.Tensor | None flash_seq_lengths: Optional precomputed active lengths for flash backends.
        :param str | None flash_route_hint: Optional flash routing hint selected outside compiled code.
        :return Any: Module outputs from either a compiled fast path or the generic resolved path.
        """

        (
            resolved_output_attentions,
            resolved_output_hidden_states,
            resolved_return_dict,
        ) = self._resolve_forward_options(  # type: ignore[attr-defined]
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # The fast compiled path specializes on the fixed training contract:
        # return_dict=True, output_attentions=False, output_hidden_states in {False, True}.
        # Other combinations are correct but uncommon in training, so keep them
        # on the uncompiled resolved path instead of exploding compile variants.
        if resolved_output_attentions or not resolved_return_dict:
            return self._forward_resolved(  # type: ignore[attr-defined]
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=resolved_output_attentions,
                output_hidden_states=resolved_output_hidden_states,
                return_dict=resolved_return_dict,
                flash_seq_lengths=flash_seq_lengths,
                flash_route_hint=flash_route_hint,
            )
        if attention_mask is None:
            return compiled_dense[resolved_output_hidden_states](
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )
        normalized_flash_route = None
        normalize_route_hint = getattr(self, "_normalize_flash_route_hint", None)
        if callable(normalize_route_hint):
            normalized_flash_route = normalize_route_hint(flash_route_hint)
        elif flash_route_hint is not None:
            normalized_flash_route = str(flash_route_hint).strip().lower() or None
        if normalized_flash_route == "fixed":
            return compiled_masked_fixed[resolved_output_hidden_states](
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                flash_seq_lengths=flash_seq_lengths,
            )
        if normalized_flash_route == "varlen":
            return compiled_masked_varlen[resolved_output_hidden_states](
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                flash_seq_lengths=flash_seq_lengths,
            )
        return compiled_masked[resolved_output_hidden_states](
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            flash_seq_lengths=flash_seq_lengths,
            flash_route_hint=normalized_flash_route,
        )

    module._compiled_forward_dense = compiled_dense
    module._compiled_forward_masked = compiled_masked
    module._compiled_forward_masked_fixed = compiled_masked_fixed
    module._compiled_forward_masked_varlen = compiled_masked_varlen
    module.forward = types.MethodType(_dispatch_forward, module)  # type: ignore[assignment]
    compiled_targets.extend(
        [
            f"{target}[dense_hs0]",
            f"{target}[dense_hs1]",
            f"{target}[masked_hs0]",
            f"{target}[masked_hs1]",
            f"{target}[masked_fixed_hs0]",
            f"{target}[masked_fixed_hs1]",
            f"{target}[masked_varlen_hs0]",
            f"{target}[masked_varlen_hs1]",
        ]
    )
    return True


def _dtype_for_mixed_precision(mode: str) -> torch.dtype:
    """Map configured mixed-precision mode to runtime compute dtype.

    :param str mode: Effective mixed-precision mode.
    :return torch.dtype: Expected activation dtype used in forward passes.
    """
    normalized = str(mode).strip().lower()
    if normalized == "bf16":
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _prefill_rotary_caches_for_compile(
    *,
    model: torch.nn.Module,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> int:
    """Prefill rotary caches on all RoPE attention modules before compile.

    :param torch.nn.Module model: Unwrapped RTD model.
    :param int seq_len: Maximum runtime sequence length.
    :param torch.device device: Runtime device.
    :param torch.dtype dtype: Runtime compute dtype.
    :return int: Number of rotary modules prefilled.
    """
    if int(seq_len) <= 0:
        return 0

    prefilled = 0
    for module in model.modules():
        rope = getattr(module, "rope", None)
        prefill = getattr(rope, "prefill_cache", None)
        if callable(prefill):
            prefill(int(seq_len), device=device, dtype=dtype)
            prefilled += 1
    return prefilled


def _build_doc_block_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Build a pairwise ``(B, S, S)`` keep-mask from document ids on-device.

    Contract:

    - Active tokens (``doc_id != 0``) attend only within the same document.
    - The diagonal encodes query activity (active ``True``, pad/inactive ``False``).
    - Inactive/pad queries get a single keep-edge to the CLS key (position 0) so SDPA
      never sees all-False rows.

    :param torch.Tensor doc_ids: Document id tensor ``(B, S)`` with 0 for padding.
    :return torch.Tensor: Bool keep-mask ``(B, S, S)``.
    """
    if doc_ids.ndim != 2:
        raise ValueError(f"doc_ids must be rank-2 (B,S); got shape={tuple(doc_ids.shape)}")

    bsz, seq_len = int(doc_ids.shape[0]), int(doc_ids.shape[1])
    device = doc_ids.device

    key = (seq_len, str(device.type), int(device.index) if device.index is not None else None)
    eye = _DOC_BLOCK_EYE_CACHE.get(key)
    if eye is None or eye.device != device or eye.shape != (seq_len, seq_len):
        eye = torch.eye(seq_len, dtype=torch.bool, device=device)
        _DOC_BLOCK_EYE_CACHE[key] = eye

    cls_key = _DOC_BLOCK_CLS_KEY_CACHE.get(key)
    if cls_key is None or cls_key.device != device or cls_key.shape != (seq_len,):
        cls_key = torch.zeros(seq_len, dtype=torch.bool, device=device)
        cls_key[0] = True
        _DOC_BLOCK_CLS_KEY_CACHE[key] = cls_key

    active = doc_ids.ne(0)  # (B,S)
    same_doc = doc_ids[:, :, None].eq(doc_ids[:, None, :])  # (B,S,S)
    keep = same_doc & active[:, :, None] & active[:, None, :]
    keep = keep | ((~active)[:, :, None] & cls_key[None, None, :])
    keep = (keep & ~eye[None, :, :]) | (eye[None, :, :] & active[:, :, None])

    if int(keep.shape[0]) != bsz:
        raise RuntimeError("doc-block mask batch dimension mismatch.")
    return keep


def _stabilize_compile_attention_mask(
    *,
    batch: dict[str, torch.Tensor],
    compile_enabled: bool,
    compile_scope: str,
    backbone_type: str,
) -> dict[str, torch.Tensor]:
    """Canonicalize attention-mask dtype for compiled attention paths.

    For HF DeBERTa-v2: normalizes existing masks to bool but does **not**
    materialize a mask when absent — the backbone's no-mask fast path handles
    ``None`` directly.

    RoPE + doc-blocking mask shape churn is handled by auto-downgrading compile
    scope to FFN in ``_resolve_compile_scope`` instead of materializing S² masks.

    :param dict[str, torch.Tensor] batch: Device-local batch mapping.
    :param bool compile_enabled: Whether torch.compile is active.
    :param str compile_scope: Effective compile scope.
    :param str backbone_type: Model backbone type.
    :return dict[str, torch.Tensor]: Possibly updated batch mapping.
    """
    if not bool(compile_enabled):
        return batch

    scope = str(compile_scope).strip().lower()
    if scope not in {"backbones", "encoder", "gen_encoder", "disc_encoder"}:
        return batch

    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        return batch

    btype = str(backbone_type).strip().lower()

    if btype == "hf_deberta_v2":
        attn = batch.get("attention_mask")
        if isinstance(attn, torch.Tensor) and attn.dtype != torch.bool:
            batch["attention_mask"] = attn.to(dtype=torch.bool)
        return batch

    return batch
