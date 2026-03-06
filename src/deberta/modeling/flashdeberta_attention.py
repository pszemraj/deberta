"""FlashDeBERTa attention adapter for the native DeBERTa-v2/v3 backbone.

This replacement extends the current branch in two ways that matter for runtime:

1. Keep the current fixed-length FlashDeBERTa path for dense / maskless batches.
2. Add the upstream-style variable-length unpadded kernel for padding-heavy runs.

Why this file exists
--------------------
The branch being audited already patches the right integration seams:
- swap ``DisentangledSelfAttention`` at runtime
- bypass encoder-side ``(S,S)`` relative-position materialization
- preserve broadcast EMD masks instead of forcing quadratic expansion

But its attention adapter currently calls only the fixed-length Triton kernel.
That means it leaves the upstream ``flash_attention_with_disentangled_varlen``
path unused, which is the biggest missing performance lever for unpacked or
otherwise padding-heavy training runs.

This file is a drop-in replacement for:
    src/deberta/modeling/flashdeberta_attention.py

Optional runtime knobs
----------------------
- ``FLASHDEBERTA_VARLEN_MIN_SEQ_LEN`` (default: ``1024``)
    Minimum sequence length required before the varlen kernel is used.
- ``FLASHDEBERTA_FORCE_VARLEN`` (default: ``0``)
    Force varlen whenever a padding mask is present.
- ``FLASHDEBERTA_EAGER_DENSE_MAX_SEQ_LEN`` (default: ``0`` / disabled)
    For benchmarking, route dense batches at or below this sequence length back
    to eager attention. This is useful when dense packed 1024 is slower than
    eager on a given GPU/software stack.

The statistics helpers are intentionally tiny so benchmark scripts can confirm
which path actually executed.
"""

from __future__ import annotations

import math
import os
import warnings
from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F

from deberta.modeling.deberta_v2_native import (
    DisentangledSelfAttention as _EagerDisentangledSelfAttention,
)
from deberta.modeling.mask_utils import normalize_keep_mask

try:
    from flashdeberta.ops.flash_attention import flash_attention_with_disentangled

    _FLASH_FIXED_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised indirectly in unit tests
    flash_attention_with_disentangled = None
    _FLASH_FIXED_IMPORT_ERROR = exc

try:
    from flashdeberta.ops.flash_attention_varlen import flash_attention_with_disentangled_varlen

    _FLASH_VARLEN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional path
    flash_attention_with_disentangled_varlen = None
    _FLASH_VARLEN_IMPORT_ERROR = exc

_FLASH_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_FLASH_STATS: Counter[str] = Counter()
_TRUTHY = {"1", "true", "yes", "y", "on"}


try:  # pragma: no cover - version-compat fallback
    _compiler_disable = torch.compiler.disable
except Exception:  # pragma: no cover

    def _compiler_disable(fn):  # type: ignore[no-redef]
        return fn


def flashdeberta_import_error() -> Exception | None:
    """Return the fixed-kernel import error, if any.

    The runtime patch requires the fixed kernel. The varlen kernel is optional and
    only affects padding-heavy workloads.
    """

    return _FLASH_FIXED_IMPORT_ERROR


def flashdeberta_stats_snapshot() -> dict[str, int]:
    """Return a copy of lightweight flash-path counters for benchmarking."""

    return dict(_FLASH_STATS)


def reset_flashdeberta_stats() -> None:
    """Reset lightweight flash-path counters for benchmarking."""

    _FLASH_STATS.clear()


def _record_stat(name: str, value: int = 1) -> None:
    """Increment a small in-process counter."""

    _FLASH_STATS[name] += int(value)


def _truthy_env(name: str, default: str = "0") -> bool:
    """Parse a boolean-ish environment variable."""

    return os.environ.get(name, default).strip().lower() in _TRUTHY


def _int_env(name: str, default: int) -> int:
    """Parse an integer environment variable with safe fallback."""

    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _mask4d_to_seqlens(attention_mask: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Convert a padding-style keep mask into per-example sequence lengths.

    Supported mask shapes:
    - ``(B, S)``
    - ``(B, 1, 1, S)``
    """

    key_mask = _mask_to_2d_keep_mask(attention_mask, seq_len=seq_len)
    return key_mask.sum(dim=-1, dtype=torch.int32)


def _mask_to_2d_keep_mask(attention_mask: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Extract a canonical 2D key-padding mask ``(B,S)`` from a broadcast mask.

    This function intentionally rejects pairwise masks because the varlen path in
    this adapter is for standard padding masks only.
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
    """Return whether a mask encodes per-query pairwise constraints."""

    mask = normalize_keep_mask(attention_mask)
    if mask.ndim == 3:
        return tuple(mask.shape[-2:]) == (int(query_len), int(key_len))
    if mask.ndim == 4:
        return tuple(mask.shape[-2:]) == (int(query_len), int(key_len))
    return False


def _is_dense_batch(attention_mask: torch.Tensor | None, *, seq_len: int) -> bool:
    """Return whether all tokens are active for all batch elements."""

    if attention_mask is None:
        return True
    try:
        key_mask = _mask_to_2d_keep_mask(attention_mask, seq_len=seq_len)
    except Exception:
        return False
    return bool(key_mask.all().item())


@_compiler_disable

def _build_unpad_metadata(mask_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build unpadding metadata for the varlen FlashDeBERTa kernel.

    Returns:
        indices: flattened non-pad token indices, shape ``(NNZ,)``
        cu_seqlens: cumulative sequence lengths, shape ``(B+1,)`` int32
        max_seqlen: maximum active tokens in any batch element, as a Python int
    """

    seqlens = mask_2d.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(mask_2d.reshape(-1), as_tuple=False).squeeze(-1)
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() > 0 else 0
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen


def _flatten_valid_tokens(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather valid tokens from a padded ``(B,S,...)`` tensor into ``(NNZ,...)``."""

    flat = tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])
    return flat.index_select(0, indices)


def _pad_valid_tokens(values: torch.Tensor, indices: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """Scatter ``(NNZ,...)`` values back into a padded ``(B,S,...)`` layout."""

    flat_out = values.new_zeros((batch_size * seq_len, *values.shape[1:]))
    flat_out = flat_out.index_copy(0, indices, values)
    return flat_out.view(batch_size, seq_len, *values.shape[1:])


def _should_use_varlen(
    *,
    attention_mask: torch.Tensor | None,
    seq_len: int,
) -> bool:
    """Return whether the varlen kernel should be used for this call."""

    if flash_attention_with_disentangled_varlen is None:
        return False
    if attention_mask is None:
        return False

    if _truthy_env("FLASHDEBERTA_FORCE_VARLEN", default="0"):
        return True

    if _is_dense_batch(attention_mask, seq_len=seq_len):
        return False

    min_seq_len = max(1, _int_env("FLASHDEBERTA_VARLEN_MIN_SEQ_LEN", 1024))
    return int(seq_len) >= int(min_seq_len)


class FlashDisentangledSelfAttention(_EagerDisentangledSelfAttention):
    """FlashDeBERTa-backed variant of native disentangled self-attention."""

    _warned_reasons: set[str] = set()

    @classmethod
    def _warn_once(cls, *, reason: str, message: str) -> None:
        """Emit one warning per process for a fallback reason."""

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
        """Return the first reason this call should use eager attention."""

        query_len = int(query_states.shape[-2])
        key_len = int(hidden_states.shape[-2])
        dropout_p = float(getattr(self.dropout, "p", 0.0))

        dense_eager_limit = max(0, _int_env("FLASHDEBERTA_EAGER_DENSE_MAX_SEQ_LEN", 0))
        if dense_eager_limit > 0 and key_len <= dense_eager_limit and _is_dense_batch(attention_mask, seq_len=key_len):
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
        if flash_attention_with_disentangled is None:
            detail = (
                str(_FLASH_FIXED_IMPORT_ERROR)
                if _FLASH_FIXED_IMPORT_ERROR is not None
                else "unknown import error"
            )
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
        """Return whether projected QKV tensors are unsupported by the flash kernels."""

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
        """Run the fixed-length FlashDeBERTa kernel."""

        seq_lengths = None
        if attention_mask is not None:
            seq_lengths = _mask4d_to_seqlens(attention_mask, seq_len=int(key_layer.shape[-2]))
        _record_stat("flash_fixed_calls")
        return flash_attention_with_disentangled(
            query_layer,
            key_layer,
            value_layer,
            seq_lengths,
            pos_key,
            pos_query,
            False,
            sm_scale,
            int(self.position_buckets),
            int(self.max_relative_positions),
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
        """Run the upstream-style unpadded varlen FlashDeBERTa kernel."""

        if flash_attention_with_disentangled_varlen is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("Varlen FlashDeBERTa operator unexpectedly unavailable.")

        bsz = int(query_layer.shape[0])
        seq_len = int(query_layer.shape[-2])
        mask_2d = _mask_to_2d_keep_mask(attention_mask, seq_len=seq_len)
        indices, cu_seqlens, max_seqlen = _build_unpad_metadata(mask_2d)

        if int(indices.numel()) == 0:
            # Degenerate all-pad batch. Training should not normally produce this, but
            # returning zeros is safer than crashing inside the kernel.
            _record_stat("flash_varlen_all_pad_batches")
            return torch.zeros_like(query_layer)

        # Kernel expects token-major tensors: (NNZ, H, D/P).
        q_padded = query_layer.permute(0, 2, 1, 3).contiguous()
        k_padded = key_layer.permute(0, 2, 1, 3).contiguous()
        v_padded = value_layer.permute(0, 2, 1, 3).contiguous()
        pos_key_padded = pos_key.permute(0, 2, 1, 3).contiguous() if pos_key is not None else None
        pos_query_padded = pos_query.permute(0, 2, 1, 3).contiguous() if pos_query is not None else None

        q_unpad = _flatten_valid_tokens(q_padded, indices)
        k_unpad = _flatten_valid_tokens(k_padded, indices)
        v_unpad = _flatten_valid_tokens(v_padded, indices)
        pos_key_unpad = _flatten_valid_tokens(pos_key_padded, indices) if pos_key_padded is not None else None
        pos_query_unpad = _flatten_valid_tokens(pos_query_padded, indices) if pos_query_padded is not None else None

        out_unpad = flash_attention_with_disentangled_varlen(
            q_unpad,
            k_unpad,
            v_unpad,
            pos_key_unpad,
            pos_query_unpad,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            False,
            sm_scale,
            int(self.position_buckets),
            int(self.max_relative_positions),
        )
        _record_stat("flash_varlen_calls")

        out_padded = _pad_valid_tokens(out_unpad, indices, bsz, seq_len)
        return out_padded.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run flash-backed attention when the runtime contract is compatible."""

        del relative_pos  # The flash kernels compute relative positions on device.

        if query_states is None:
            query_states = hidden_states

        _record_stat("forward_calls")

        reason = self._fallback_reason(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            query_states=query_states,
            rel_embeddings=rel_embeddings,
        )
        if reason is not None:
            key, message = reason
            _record_stat("fallback_calls")
            _record_stat(f"fallback_{key}")
            self._warn_once(reason=key, message=message)
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                query_states=query_states,
                relative_pos=None,
                rel_embeddings=rel_embeddings,
            )

        if flash_attention_with_disentangled is None:  # pragma: no cover - guarded above
            raise RuntimeError("FlashDeBERTa operator import unexpectedly unavailable during flash forward.")

        model_dtype = hidden_states.dtype
        bsz, query_len, _ = query_states.shape

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
            _record_stat("fallback_calls")
            _record_stat(f"fallback_{key}")
            self._warn_once(reason=key, message=message)
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
            pos_key = torch.matmul(query_layer, pos_key_layer.unsqueeze(0).transpose(-1, -2))

        if "p2c" in self.pos_att_type:
            pos_query_layer = self._project_rel(rel_embeddings, use_query=True).to(dtype=key_layer.dtype)
            pos_query = torch.matmul(key_layer, pos_query_layer.unsqueeze(0).transpose(-1, -2))

        scale_factor = 1
        if pos_key is not None:
            scale_factor += 1
        if pos_query is not None:
            scale_factor += 1
        sm_scale = 1.0 / math.sqrt(float(self.attention_head_size * scale_factor))

        _record_stat("flash_eligible_calls")
        if _should_use_varlen(attention_mask=attention_mask, seq_len=int(key_layer.shape[-2])):
            output = self._flash_varlen(
                query_layer=query_layer,
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask,
                pos_key=pos_key,
                pos_query=pos_query,
                sm_scale=sm_scale,
            )
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
    "reset_flashdeberta_stats",
]
