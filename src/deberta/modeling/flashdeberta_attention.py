"""FlashDeBERTa attention adapter for the native DeBERTa-v2/v3 backbone.

This revision is tuned for the current repo state and for ``torch.compile``.

What changed versus the earlier adapter
--------------------------------------
1. Stats are now debug-only and disabled by default. The prior adapter mutated
   a Python ``Counter`` inside the attention forward path, which is exactly the
   kind of Python/global state that TorchDynamo may guard on and recompile.
2. Runtime knobs are parsed once at import time, then exposed via a refresh helper
   for tests or benchmark scripts that deliberately mutate environment variables.
3. Dense-vs-varlen routing no longer inspects attention-mask contents inside the
   compiled forward path. In this repository's training loop, dense batches already
   arrive as ``attention_mask=None`` because the collator drops all-ones masks.
   Using that contract is both faster and more compile-friendly.

Important behavior
------------------
- Fixed-length flash remains the default path for dense / maskless batches.
- Varlen flash is used for padded batches when enabled by runtime policy.
- The padded varlen path now runs through an opaque custom op on CUDA so
  ``torch.compile`` does not trace into FlashDeBERTa's Python/Triton wrapper.
- Pairwise masks still fall back to eager attention for correctness.

Optional runtime knobs
----------------------
Set these before importing this module:

- ``FLASHDEBERTA_VARLEN_MIN_SEQ_LEN`` (default: ``1024``)
    Minimum sequence length required before the varlen kernel is used when a
    padding mask is present.
- ``FLASHDEBERTA_FORCE_VARLEN`` (default: ``0``)
    Force the varlen kernel whenever a padding mask is present.
- ``FLASHDEBERTA_EAGER_DENSE_MAX_SEQ_LEN`` (default: ``0`` / disabled)
    Route dense maskless batches at or below this length back to eager attention.
- ``FLASHDEBERTA_DEBUG_STATS`` (default: ``0``)
    Enable eager/debug-only path counters for benchmark scripts.
- ``FLASHDEBERTA_WARN_FALLBACKS`` (default: ``1``)
    Emit one warning per fallback reason outside compiled graphs.
"""

from __future__ import annotations

import math
import os
import warnings
from collections import Counter
from dataclasses import dataclass

import torch

from deberta.modeling.deberta_v2_native import (
    DisentangledSelfAttention as _EagerDisentangledSelfAttention,
)
from deberta.modeling.flashdeberta_fixed_op import (
    flashdeberta_fixed,
    flashdeberta_fixed_import_error,
)
from deberta.modeling.flashdeberta_varlen_op import (
    flashdeberta_compiled_varlen_available,
    flashdeberta_varlen_padded,
)
from deberta.modeling.mask_utils import normalize_keep_mask

_FLASH_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_FLASH_STATS: Counter[str] = Counter()
_TRUTHY = {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class FlashDebertaRuntimeConfig:
    """Static runtime policy for the flash adapter.

    Values are intentionally read once per process to reduce Python work and
    guard surface inside compiled attention forwards.
    """

    force_varlen: bool = False
    varlen_min_seq_len: int = 1024
    eager_dense_max_seq_len: int = 0
    enable_debug_stats: bool = False
    warn_fallbacks: bool = True


def _is_torch_compiling() -> bool:
    """Return whether execution is happening under ``torch.compile``.

    :return bool: True when inside compiled/traced execution.
    """

    if not hasattr(torch, "compiler") or not hasattr(torch.compiler, "is_compiling"):
        return False
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


def _truthy_env(name: str, default: str = "0") -> bool:
    """Parse a boolean-ish environment variable.

    :param str name: Environment variable name.
    :param str default: Default string to parse when the variable is unset.
    :return bool: Whether the value matches the repo's truthy set.
    """

    return os.environ.get(name, default).strip().lower() in _TRUTHY


def _int_env(name: str, default: int) -> int:
    """Parse an integer environment variable with safe fallback.

    :param str name: Environment variable name.
    :param int default: Default integer value.
    :return int: Parsed integer or the default on parse failure.
    """

    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _read_runtime_config_from_env() -> FlashDebertaRuntimeConfig:
    """Load runtime policy once from environment variables.

    :return FlashDebertaRuntimeConfig: Parsed runtime policy.
    """

    return FlashDebertaRuntimeConfig(
        force_varlen=_truthy_env("FLASHDEBERTA_FORCE_VARLEN", default="0"),
        varlen_min_seq_len=max(1, _int_env("FLASHDEBERTA_VARLEN_MIN_SEQ_LEN", 1024)),
        eager_dense_max_seq_len=max(0, _int_env("FLASHDEBERTA_EAGER_DENSE_MAX_SEQ_LEN", 0)),
        enable_debug_stats=_truthy_env("FLASHDEBERTA_DEBUG_STATS", default="0"),
        warn_fallbacks=_truthy_env("FLASHDEBERTA_WARN_FALLBACKS", default="1"),
    )


_RUNTIME_CONFIG = _read_runtime_config_from_env()


def refresh_flashdeberta_runtime_config_from_env() -> None:
    """Reload runtime policy from environment variables.

    This exists primarily for tests or benchmark scripts that intentionally
    mutate ``os.environ`` after the module was imported.
    """

    global _RUNTIME_CONFIG
    _RUNTIME_CONFIG = _read_runtime_config_from_env()


def flashdeberta_import_error() -> Exception | None:
    """Return the fixed-kernel import error, if any.

    The runtime patch requires the fixed kernel. The varlen kernel is optional and
    only affects padding-heavy workloads.

    :return Exception | None: Stored fixed-kernel import failure, if one occurred.
    """

    return flashdeberta_fixed_import_error()


def flashdeberta_stats_snapshot() -> dict[str, int]:
    """Return a copy of eager/debug-only flash path counters.

    These counters are intentionally disabled by default and are not intended as
    a normal-training metric source.

    :return dict[str, int]: Counter snapshot keyed by path or fallback name.
    """

    return dict(_FLASH_STATS)


def reset_flashdeberta_stats() -> None:
    """Reset eager/debug-only flash path counters."""

    _FLASH_STATS.clear()


def _record_stat(name: str, value: int = 1) -> None:
    """Increment an eager/debug-only flash path counter.

    This is a strict no-op inside compiled graphs.

    :param str name: Counter key.
    :param int value: Increment amount, defaults to ``1``.
    """

    if not _RUNTIME_CONFIG.enable_debug_stats:
        return
    if _is_torch_compiling():
        return
    _FLASH_STATS[name] += int(value)


def _mask4d_to_seqlens(attention_mask: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Convert a padding-style keep mask into per-example sequence lengths.

    Supported mask shapes:
    - ``(B, S)``
    - ``(B, 1, 1, S)``

    :param torch.Tensor attention_mask: Padding-style keep mask.
    :param int seq_len: Expected sequence length.
    :return torch.Tensor: Per-example sequence lengths with dtype ``int32``.
    """

    key_mask = _mask_to_2d_keep_mask(attention_mask, seq_len=seq_len)
    return key_mask.sum(dim=-1, dtype=torch.int32)


def _mask_to_2d_keep_mask(attention_mask: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Extract a canonical 2D key-padding mask ``(B,S)`` from a broadcast mask.

    This function intentionally rejects pairwise masks because the varlen path in
    this adapter is for standard padding masks only.

    :param torch.Tensor attention_mask: Broadcast or 2D keep mask.
    :param int seq_len: Expected sequence length.
    :raises ValueError: If the mask is not 2D or broadcast 4D.
    :return torch.Tensor: Boolean mask with shape ``(B, S)``.
    """

    mask = normalize_keep_mask(attention_mask)
    if mask.ndim == 2:
        key_mask = mask
    elif mask.ndim == 4 and int(mask.shape[-2]) == 1:
        key_mask = mask[:, 0, 0, :]
    else:
        raise ValueError(
            "FlashDeBERTa padding masks must be shaped (B,S) or (B,1,1,S) for sequence-length extraction."
        )

    if int(key_mask.shape[-1]) != int(seq_len):
        key_mask = key_mask[..., :seq_len]
    return key_mask.to(dtype=torch.bool)


def _is_pairwise_mask(attention_mask: torch.Tensor, *, query_len: int, key_len: int) -> bool:
    """Return whether a mask encodes per-query pairwise constraints.

    :param torch.Tensor attention_mask: Candidate attention mask.
    :param int query_len: Expected query length.
    :param int key_len: Expected key length.
    :return bool: True when the mask has per-query pairwise structure.
    """

    mask = normalize_keep_mask(attention_mask)
    if mask.ndim == 3:
        return tuple(mask.shape[-2:]) == (int(query_len), int(key_len))
    if mask.ndim == 4:
        return tuple(mask.shape[-2:]) == (int(query_len), int(key_len))
    return False


def _should_use_varlen(
    *,
    attention_mask: torch.Tensor | None,
    seq_len: int,
) -> bool:
    """Return whether the varlen kernel should be used for this call.

    This deliberately does not inspect tensor contents. In this repository's
    training path, the collator already drops all-ones masks, so
    ``attention_mask is None`` is the dense signal and ``attention_mask is not None``
    is the padded signal. When ``torch.compile`` is active we require the opaque
    custom-op varlen wrapper so Dynamo does not trace into FlashDeBERTa's
    Python/Triton launcher.

    :param torch.Tensor | None attention_mask: Optional attention mask.
    :param int seq_len: Sequence length for the current call.
    :return bool: True when the varlen kernel should run.
    """

    if attention_mask is None:
        return False

    if _is_torch_compiling() and not flashdeberta_compiled_varlen_available():
        return False

    if _RUNTIME_CONFIG.force_varlen:
        return True

    return int(seq_len) >= int(_RUNTIME_CONFIG.varlen_min_seq_len)


class FlashDisentangledSelfAttention(_EagerDisentangledSelfAttention):
    """FlashDeBERTa-backed variant of native disentangled self-attention."""

    _warned_reasons: set[str] = set()

    @classmethod
    def _warn_once(cls, *, reason: str, message: str) -> None:
        """Emit one warning per process for a fallback reason.

        Warnings are skipped while executing inside compiled graphs.

        :param str reason: Stable fallback key.
        :param str message: Warning text to emit.
        """

        if not _RUNTIME_CONFIG.warn_fallbacks:
            return
        if _is_torch_compiling():
            return
        if reason in cls._warned_reasons:
            return
        cls._warned_reasons.add(reason)
        warnings.warn(message, stacklevel=2)

    def _fallback_reason(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        query_states: torch.Tensor,
        rel_embeddings: torch.Tensor | None,
    ) -> tuple[str, str] | None:
        """Return the first reason this call should use eager attention.

        :param torch.Tensor hidden_states: Key/value hidden states.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param torch.Tensor query_states: Query hidden states.
        :param torch.Tensor | None rel_embeddings: Relative embedding table.
        :return tuple[str, str] | None: Fallback reason key and message, or ``None``.
        """

        query_len = int(query_states.shape[-2])
        key_len = int(hidden_states.shape[-2])
        dropout_p = float(getattr(self.dropout, "p", 0.0))

        dense_eager_limit = int(_RUNTIME_CONFIG.eager_dense_max_seq_len)
        if dense_eager_limit > 0 and attention_mask is None and key_len <= dense_eager_limit:
            return (
                "dense_short_policy",
                "FlashDeBERTa dense short-sequence policy selected eager attention for this batch.",
            )
        if attention_mask is not None and _is_pairwise_mask(
            attention_mask, query_len=query_len, key_len=key_len
        ):
            return (
                "pairwise_mask",
                "FlashDeBERTa attention does not support pairwise (B,S,S)/(B,1,S,S) masks; using eager attention.",
            )
        if "p2p" in self.pos_att_type:
            return (
                "p2p",
                "FlashDeBERTa attention does not support pos_att_type='p2p'; using eager attention.",
            )
        if self.training and dropout_p > 0.0:
            return (
                "attention_dropout",
                "FlashDeBERTa attention requires attention_probs_dropout_prob=0.0 during training; using eager attention.",
            )
        if not self.relative_attention:
            return (
                "relative_attention_disabled",
                "FlashDeBERTa integration currently targets relative-attention DeBERTa configs; using eager attention.",
            )
        if rel_embeddings is None:
            return (
                "missing_rel_embeddings",
                "FlashDeBERTa attention requires relative embeddings for the current config; using eager attention.",
            )
        if int(self.position_buckets) <= 0:
            return (
                "missing_position_buckets",
                "FlashDeBERTa attention requires config.position_buckets > 0; using eager attention.",
            )
        fixed_import_error = flashdeberta_fixed_import_error()
        if fixed_import_error is not None:
            detail = str(fixed_import_error)
            return (
                "missing_flashdeberta",
                f"FlashDeBERTa fixed-length attention is unavailable ({detail}); using eager attention.",
            )
        if hidden_states.device.type != "cuda":
            return (
                "device",
                "FlashDeBERTa attention requires CUDA tensors; using eager attention.",
            )
        if query_len != key_len:
            return (
                "query_key_length_mismatch",
                "FlashDeBERTa attention integration currently expects self-attention with matching query/key lengths; using eager attention.",
            )
        return None

    def _projected_qkv_fallback_reason(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
    ) -> tuple[str, str] | None:
        """Return whether projected QKV tensors are unsupported by the flash kernels.

        :param torch.Tensor query_layer: Projected query tensor.
        :param torch.Tensor key_layer: Projected key tensor.
        :param torch.Tensor value_layer: Projected value tensor.
        :return tuple[str, str] | None: Fallback reason key and message, or ``None``.
        """

        qkv_dtypes = {query_layer.dtype, key_layer.dtype, value_layer.dtype}
        if len(qkv_dtypes) != 1:
            return (
                "mixed_qkv_dtype",
                "FlashDeBERTa attention requires projected query/key/value tensors to share one dtype; using eager attention.",
            )

        qkv_dtype = query_layer.dtype
        if qkv_dtype not in _FLASH_SUPPORTED_DTYPES:
            return (
                "dtype",
                "FlashDeBERTa attention currently supports float16/bfloat16 projected QKV activations; using eager attention.",
            )
        return None

    def _shape_varlen(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape projection output into varlen-friendly multi-head layout.

        The padded-varlen path operates more naturally in ``(B, S, H, D)``
        layout, which lets the custom op flatten ``(B*S)`` directly and use one
        flat gather/scatter instead of per-token advanced indexing over a
        ``(B, H, S, D)`` transpose.

        :param torch.Tensor x: Projected tensor with trailing dim ``all_head_size``.
        :return torch.Tensor: Tensor with shape ``(B, S, H, D)``.
        """

        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size)

    def _flash_fixed(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor | None,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
    ) -> torch.Tensor:
        """Run the fixed-length FlashDeBERTa kernel.

        :param torch.Tensor query_layer: Projected queries in ``(B, H, S, D)`` layout.
        :param torch.Tensor key_layer: Projected keys in ``(B, H, S, D)`` layout.
        :param torch.Tensor value_layer: Projected values in ``(B, H, S, D)`` layout.
        :param torch.Tensor | None attention_mask: Optional padding mask.
        :param torch.Tensor | None pos_key: Optional c2p term.
        :param torch.Tensor | None pos_query: Optional p2c term.
        :param float sm_scale: Softmax scale.
        :return torch.Tensor: Flash output in ``(B, H, S, D)`` layout.
        """

        seq_lengths = None
        if attention_mask is not None:
            seq_lengths = _mask4d_to_seqlens(attention_mask, seq_len=int(key_layer.shape[-2]))
        if _RUNTIME_CONFIG.enable_debug_stats:
            _record_stat("flash_fixed_calls")
        return flashdeberta_fixed(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            seq_lengths=seq_lengths,
            pos_key=pos_key,
            pos_query=pos_query,
            causal=False,
            sm_scale=sm_scale,
            position_buckets=int(self.position_buckets),
            max_relative_distance=int(self.max_relative_positions),
        )

    def _flash_varlen(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
    ) -> torch.Tensor:
        """Run padded varlen FlashDeBERTa attention.

        :param torch.Tensor query_layer: Projected queries in ``(B, S, H, D)`` layout.
        :param torch.Tensor key_layer: Projected keys in ``(B, S, H, D)`` layout.
        :param torch.Tensor value_layer: Projected values in ``(B, S, H, D)`` layout.
        :param torch.Tensor attention_mask: Padding-style keep mask.
        :param torch.Tensor | None pos_key: Optional c2p term.
        :param torch.Tensor | None pos_query: Optional p2c term.
        :param float sm_scale: Softmax scale.
        :return torch.Tensor: Flash output in ``(B, S, H, D)`` layout.
        """

        seq_len = int(query_layer.shape[1])
        mask_2d = _mask_to_2d_keep_mask(attention_mask, seq_len=seq_len)
        out = flashdeberta_varlen_padded(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            attention_mask_2d=mask_2d,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            position_buckets=int(self.position_buckets),
            max_relative_distance=int(self.max_relative_positions),
            causal=False,
        )
        if _RUNTIME_CONFIG.enable_debug_stats:
            _record_stat("flash_varlen_calls")
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run flash-backed attention when the runtime contract is compatible.

        :param torch.Tensor hidden_states: Input hidden states.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param bool output_attentions: Whether to return attention probabilities.
        :param torch.Tensor | None query_states: Optional query states.
        :param torch.Tensor | None relative_pos: Optional relative-position ids.
        :param torch.Tensor | None rel_embeddings: Optional relative embedding table.
        :return tuple[torch.Tensor, torch.Tensor | None]: Attention output and optional probs.
        """

        del relative_pos  # The flash kernels compute relative positions on device.

        if query_states is None:
            query_states = hidden_states

        if _RUNTIME_CONFIG.enable_debug_stats:
            _record_stat("forward_calls")

        reason = self._fallback_reason(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            query_states=query_states,
            rel_embeddings=rel_embeddings,
        )
        if reason is not None:
            key, message = reason
            if _RUNTIME_CONFIG.enable_debug_stats:
                _record_stat("fallback_calls")
                _record_stat(f"fallback_{key}")
            self._warn_once(reason=key, message=message)
            # The encoder-level get_rel_pos patch suppresses the shared (S,S)
            # allocation globally. Unsupported correctness fallbacks rebuild
            # relative-position bias inside eager attention instead.
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                query_states=query_states,
                relative_pos=None,
                rel_embeddings=rel_embeddings,
            )

        if flashdeberta_fixed_import_error() is not None:  # pragma: no cover - guarded above
            raise RuntimeError("FlashDeBERTa operator import unexpectedly unavailable during flash forward.")

        model_dtype = hidden_states.dtype
        bsz, query_len, _ = query_states.shape
        use_varlen = _should_use_varlen(attention_mask=attention_mask, seq_len=int(hidden_states.shape[-2]))

        if use_varlen:
            query_layer = self._shape_varlen(self.query_proj(query_states))
            key_layer = self._shape_varlen(self.key_proj(hidden_states))
            value_layer = self._shape_varlen(self.value_proj(hidden_states))
        else:
            query_layer = self._shape(self.query_proj(query_states)).contiguous()
            key_layer = self._shape(self.key_proj(hidden_states)).contiguous()
            value_layer = self._shape(self.value_proj(hidden_states)).contiguous()

        projected_reason = self._projected_qkv_fallback_reason(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
        )
        if projected_reason is not None:
            key, message = projected_reason
            if _RUNTIME_CONFIG.enable_debug_stats:
                _record_stat("fallback_calls")
                _record_stat(f"fallback_{key}")
            self._warn_once(reason=key, message=message)
            # Keep the same eager fallback contract here for dtype/layout
            # mismatches instead of reviving the encoder-wide relative_pos tensor.
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                query_states=query_states,
                relative_pos=None,
                rel_embeddings=rel_embeddings,
            )

        pos_key: torch.Tensor | None = None
        pos_query: torch.Tensor | None = None
        rel_embeddings = self.pos_dropout(rel_embeddings)

        if "c2p" in self.pos_att_type:
            pos_key_layer = self._project_rel(rel_embeddings, use_query=False).to(dtype=query_layer.dtype)
            if use_varlen:
                pos_key = torch.einsum("bshd,hpd->bshp", query_layer, pos_key_layer)
            else:
                pos_key = torch.matmul(query_layer, pos_key_layer.unsqueeze(0).transpose(-1, -2))

        if "p2c" in self.pos_att_type:
            pos_query_layer = self._project_rel(rel_embeddings, use_query=True).to(dtype=key_layer.dtype)
            if use_varlen:
                pos_query = torch.einsum("bshd,hpd->bshp", key_layer, pos_query_layer)
            else:
                pos_query = torch.matmul(key_layer, pos_query_layer.unsqueeze(0).transpose(-1, -2))

        scale_factor = 1
        if pos_key is not None:
            scale_factor += 1
        if pos_query is not None:
            scale_factor += 1
        sm_scale = 1.0 / math.sqrt(float(self.attention_head_size * scale_factor))

        if _RUNTIME_CONFIG.enable_debug_stats:
            _record_stat("flash_eligible_calls")
        if use_varlen:
            output = self._flash_varlen(
                query_layer=query_layer,
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask,
                pos_key=pos_key,
                pos_query=pos_query,
                sm_scale=sm_scale,
            )
            output = output.contiguous().view(bsz, query_len, self.all_head_size)
        else:
            output = self._flash_fixed(
                query_layer=query_layer,
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask,
                pos_key=pos_key,
                pos_query=pos_query,
                sm_scale=sm_scale,
            )
            output = output.transpose(1, 2).contiguous().view(bsz, query_len, self.all_head_size)
        output = output.to(dtype=model_dtype)

        if output_attentions:
            return output, None
        return output, None


__all__ = [
    "FlashDisentangledSelfAttention",
    "_is_pairwise_mask",
    "_mask4d_to_seqlens",
    "_mask_to_2d_keep_mask",
    "flashdeberta_import_error",
    "flashdeberta_stats_snapshot",
    "refresh_flashdeberta_runtime_config_from_env",
    "reset_flashdeberta_stats",
]
