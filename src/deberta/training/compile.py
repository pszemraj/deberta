"""Torch compile and attention-mask helpers for pretraining."""

from __future__ import annotations

import dataclasses
import logging
import types
from collections.abc import Callable
from typing import Any

import torch

from deberta.config import ModelConfig, ModelHFFlashConfig, _normalize_sdpa_kernel
from deberta.modeling.flashdeberta_kernel_tuning import (
    configure_flashdeberta_kernel_overrides,
    flash_padding_route,
    flash_route_choice,
    flash_seq_bucket,
)
from deberta.modeling.flashdeberta_op_utils import device_compute_capability
from deberta.modeling.mask_utils import (
    FlashBatchMeta,
    build_doc_block_mask,
    is_pairwise_mask,
)

logger = logging.getLogger(__name__)


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


def _flash_route_hint_for_docblock_batch(
    *,
    seq_len: int,
    batch_size: int | None = None,
    flash_cfg: ModelHFFlashConfig | None = None,
    device: torch.device | None = None,
) -> str:
    """Select the doc-block flash backend for one packed batch.

    The default policy comes from the repo-local JSON route table. Measured
    packed RTD sequence buckets choose dense ``docblock_bias`` on GPUs with
    matching capability-scoped rows (shipped: ``sm_120``); other hardware
    defaults to the segment-aware ragged ``docblock`` route. The dense rows
    carry ``max_seq_len``/``max_batch_size`` bounds because the route saves a
    ``(B,H,S,S)`` bias for backward; out-of-bounds batches stay ragged. Set
    ``docblock_bias_seq_len`` to force dense at one exact length on any GPU
    regardless of bounds, or ``0`` to force ragged for ablations.

    :param int seq_len: Packed sequence length.
    :param int | None batch_size: Packed batch size, when known.
    :param ModelHFFlashConfig | None flash_cfg: Optional resolved flash config.
    :param torch.device | None device: Batch device for capability-scoped table rows.
    :return str: Either ``docblock_bias`` or ``docblock``.
    """

    override_bias_seq_len = flash_cfg.docblock_bias_seq_len if flash_cfg is not None else None
    if override_bias_seq_len is not None:
        if int(override_bias_seq_len) > 0 and int(seq_len) == int(override_bias_seq_len):
            return "docblock_bias"
        return "docblock"

    seq_bucket = flash_seq_bucket(seq_len=int(seq_len))
    table_route = flash_route_choice(
        policy="docblock",
        seq_bucket=seq_bucket,
        compute_capability=device_compute_capability(device) if device is not None else None,
        seq_len=int(seq_len),
        batch_size=int(batch_size) if batch_size is not None else None,
    )
    if table_route in {"docblock", "docblock_bias"}:
        return table_route
    return "docblock"


def _flash_meta_with_route(
    flash_meta: FlashBatchMeta | None, route_hint: str | None
) -> FlashBatchMeta | None:
    """Return metadata with a route hint override.

    :param FlashBatchMeta | None flash_meta: Existing metadata bundle.
    :param str | None route_hint: Route hint to install.
    :return FlashBatchMeta | None: Metadata bundle with the requested route.
    """

    if flash_meta is None:
        return FlashBatchMeta(route_hint=route_hint) if route_hint is not None else None
    if flash_meta.route_hint == route_hint:
        return flash_meta
    return dataclasses.replace(flash_meta, route_hint=route_hint)


def prepare_flash_attention_batch_metadata(
    *,
    batch: dict[str, Any],
    backbone_type: str,
    flash_enabled: bool = False,
    flash_cfg: ModelHFFlashConfig | None = None,
    route_device: torch.device | None = None,
) -> tuple[dict[str, Any], FlashBatchMeta | None]:
    """Select the attention route from collator-built batch metadata.

    :param dict[str, Any] batch: Device-local batch mapping.
    :param str backbone_type: Backbone type string.
    :param bool flash_enabled: Whether the active backend can consume flash metadata.
    :param ModelHFFlashConfig | None flash_cfg: Optional resolved flash config for route selection.
    :param torch.device | None route_device: Optional eventual activation device
        when preparing metadata before transfer.
    :return tuple[dict[str, Any], FlashBatchMeta | None]: Updated batch and optional metadata.
    """

    flash_meta = batch.pop("_flash_meta", None)
    if not isinstance(flash_meta, FlashBatchMeta):
        flash_meta = None

    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
        return batch, None

    btype = str(backbone_type).strip().lower()
    seq_len = int(input_ids.shape[-1])
    batch_size = int(input_ids.shape[0])
    routing_device = route_device if route_device is not None else input_ids.device
    flash_enabled = bool(flash_enabled)
    if flash_enabled and btype == "hf_deberta_v2" and flash_cfg is not None:
        configure_flashdeberta_kernel_overrides(flash_cfg.kernel_overrides_path)

    doc_ids = batch.pop("doc_ids", None)
    if isinstance(doc_ids, torch.Tensor) and doc_ids.ndim == 2:
        if btype != "hf_deberta_v2" or not flash_enabled or flash_meta is None:
            batch["attention_mask"] = build_doc_block_mask(doc_ids)
            return batch, None
        route_hint = _flash_route_hint_for_docblock_batch(
            seq_len=seq_len,
            batch_size=batch_size,
            flash_cfg=flash_cfg,
            device=routing_device,
        )
        batch["attention_mask"] = (
            build_doc_block_mask(doc_ids) if route_hint == "docblock_bias" else doc_ids.ne(0)
        )
        return (
            batch,
            dataclasses.replace(flash_meta, doc_ids=doc_ids, route_hint=route_hint),
        )

    if btype != "hf_deberta_v2" or not flash_enabled:
        return batch, None

    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        return batch, FlashBatchMeta(route_hint="dense")
    if not isinstance(attention_mask, torch.Tensor):
        return batch, None
    if is_pairwise_mask(attention_mask, query_len=seq_len, key_len=seq_len):
        return batch, None
    if flash_meta is None or flash_meta.seq_lengths is None or flash_meta.active_tokens_scalar is None:
        return batch, None

    route_hint = flash_padding_route(
        seq_len=seq_len,
        total_tokens=int(flash_meta.active_tokens_scalar),
        batch_size=batch_size,
        compute_capability=device_compute_capability(routing_device),
    )
    return batch, dataclasses.replace(flash_meta, route_hint=route_hint)


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


def _resolve_effective_compile_scope(
    *,
    train_cfg: Any,
    model_cfg: ModelConfig,
    data_cfg: Any,
    compile_enabled: bool,
) -> tuple[str, str, str | None]:
    """Resolve requested and effective compile scope for one training run.

    :param Any train_cfg: Train config carrying ``torch_compile_scope``.
    :param ModelConfig model_cfg: Model configuration.
    :param Any data_cfg: Data config carrying ``block_cross_document_attention``.
    :param bool compile_enabled: Whether torch.compile is enabled.
    :return tuple[str, str, str | None]: Requested scope, effective scope, and
        optional downgrade reason (effective == requested when compile is off).
    """
    requested_scope = str(train_cfg.compile.scope).strip().lower()
    if not compile_enabled:
        return requested_scope, requested_scope, None
    scope, reason = _resolve_compile_scope(
        requested_scope=requested_scope,
        model_cfg=model_cfg,
        block_cross_document_attention=bool(data_cfg.packing.block_cross_document_attention),
    )
    return requested_scope, scope, reason


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
    if not (
        callable(resolve_options)
        and callable(dense_hs0)
        and callable(dense_hs1)
        and callable(masked_hs0)
        and callable(masked_hs1)
    ):
        return False
    dense_hs0_fn = dense_hs0
    dense_hs1_fn = dense_hs1
    masked_hs0_fn = masked_hs0
    masked_hs1_fn = masked_hs1

    def _make_routed_masked_fn(base_fn: Callable[..., Any], route: str) -> Callable[..., Any]:
        """Bind a fixed flash route onto one stable masked entrypoint.

        Each returned closure is a distinct function object, so ``torch.compile``
        keeps one compiled artifact per (route, hidden-states) combination.

        :param Callable[..., Any] base_fn: Stable masked helper to wrap.
        :param str route: Flash route literal stamped onto ``flash_meta``.
        :return Callable[..., Any]: Route-bound masked entrypoint.
        """

        def _routed_masked_fn(
            *,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            flash_meta: FlashBatchMeta | None = None,
        ) -> Any:
            """Call the masked helper with a fixed flash route.

            :param torch.Tensor | None input_ids: Optional input token ids.
            :param torch.Tensor attention_mask: Attention mask tensor.
            :param torch.Tensor | None token_type_ids: Optional token type ids.
            :param torch.Tensor | None position_ids: Optional position ids.
            :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
            :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
            :return Any: Masked-path backbone outputs.
            """

            return base_fn(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                flash_meta=_flash_meta_with_route(flash_meta, route),
            )

        return _routed_masked_fn

    def _compile_routed_masked_pair(route: str) -> dict[bool, Any]:
        """Compile the hs0/hs1 masked entrypoints bound to one flash route.

        :param str route: Flash route literal.
        :return dict[bool, Any]: Compiled entrypoints keyed by ``output_hidden_states``.
        """

        return {
            False: torch.compile(_make_routed_masked_fn(masked_hs0_fn, route), **compile_kwargs),
            True: torch.compile(_make_routed_masked_fn(masked_hs1_fn, route), **compile_kwargs),
        }

    compiled_dense = {
        False: torch.compile(dense_hs0_fn, **compile_kwargs),
        True: torch.compile(dense_hs1_fn, **compile_kwargs),
    }
    compiled_masked = {
        False: torch.compile(masked_hs0_fn, **compile_kwargs),
        True: torch.compile(masked_hs1_fn, **compile_kwargs),
    }
    # Routes with a dedicated compiled specialization; adding a route family
    # means adding one entry here. Hints outside this mapping (or None) run the
    # generic masked entrypoint with the hint re-attached, so the model-side
    # adapter still resolves them.
    compiled_masked_routed = {
        route: _compile_routed_masked_pair(route)
        for route in ("fixed", "varlen", "docblock", "docblock_bias")
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
        flash_meta: FlashBatchMeta | None = None,
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
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
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
                flash_meta=flash_meta,
            )
        if attention_mask is None:
            return compiled_dense[resolved_output_hidden_states](
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )
        route = flash_meta.route_hint if flash_meta is not None else None
        routed = compiled_masked_routed.get(route) if route is not None else None
        if routed is not None:
            assert flash_meta is not None, "A compiled routed target requires explicit FlashBatchMeta."
            return routed[resolved_output_hidden_states](
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                flash_meta=flash_meta,
            )
        return compiled_masked[resolved_output_hidden_states](
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            flash_meta=_flash_meta_with_route(flash_meta, route),
        )

    module._compiled_forward_dense = compiled_dense
    module._compiled_forward_masked = compiled_masked
    module._compiled_forward_masked_routed = compiled_masked_routed
    module.forward = types.MethodType(_dispatch_forward, module)  # type: ignore[assignment]
    compiled_targets.extend(
        [
            f"{target}[dense_hs0]",
            f"{target}[dense_hs1]",
            f"{target}[masked_hs0]",
            f"{target}[masked_hs1]",
            *(
                f"{target}[masked_{route}_hs{int(hs)}]"
                for route in compiled_masked_routed
                for hs in (False, True)
            ),
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
