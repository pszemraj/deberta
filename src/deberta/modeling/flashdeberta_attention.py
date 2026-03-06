"""FlashDeBERTa attention adapter for the native DeBERTa-v2/v3 backbone.

This module provides a drop-in replacement for
``deberta.modeling.deberta_v2_native.DisentangledSelfAttention`` that routes
compatible calls through FlashDeBERTa's Triton kernel while preserving eager
fallback for unsupported cases.
"""

from __future__ import annotations

import math
import warnings

import torch

from deberta.modeling.deberta_v2_native import (
    DisentangledSelfAttention as _EagerDisentangledSelfAttention,
)
from deberta.modeling.mask_utils import normalize_keep_mask

try:
    from flashdeberta.ops.flash_attention import flash_attention_with_disentangled

    _FLASHDEBERTA_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised via patch-module tests
    flash_attention_with_disentangled = None
    _FLASHDEBERTA_IMPORT_ERROR = exc


_FLASH_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


def flashdeberta_import_error() -> Exception | None:
    """Return the stored FlashDeBERTa import error, if any.

    :return Exception | None: Import failure captured at module import time.
    """

    return _FLASHDEBERTA_IMPORT_ERROR


def _mask4d_to_seqlens(attention_mask: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Convert a padding-style keep mask into per-example sequence lengths.

    Supported shapes:
    - ``(B, S)``
    - ``(B, 1, 1, S)``

    :param torch.Tensor attention_mask: Input keep mask.
    :param int seq_len: Expected sequence length.
    :raises ValueError: If the mask encodes pairwise query/key constraints.
    :return torch.Tensor: Sequence lengths with shape ``(B,)`` and dtype ``int32``.
    """

    mask = normalize_keep_mask(attention_mask)
    if mask.ndim == 2:
        key_mask = mask
    elif mask.ndim == 4 and int(mask.shape[-2]) == 1:
        key_mask = mask[:, 0, 0, :]
    else:
        raise ValueError(
            "FlashDeBERTa sequence lengths require a padding/broadcast mask shaped (B,S) or (B,1,1,S)."
        )

    if int(key_mask.shape[-1]) != int(seq_len):
        key_mask = key_mask[..., :seq_len]
    return key_mask.sum(dim=-1, dtype=torch.int32)


def _is_pairwise_mask(attention_mask: torch.Tensor, *, query_len: int, key_len: int) -> bool:
    """Return whether a mask encodes per-query pairwise constraints.

    :param torch.Tensor attention_mask: Input keep mask.
    :param int query_len: Query sequence length.
    :param int key_len: Key sequence length.
    :return bool: True when the mask is pairwise rather than padding-style.
    """

    mask = normalize_keep_mask(attention_mask)
    if mask.ndim == 3:
        return tuple(mask.shape[-2:]) == (int(query_len), int(key_len))
    if mask.ndim == 4:
        return tuple(mask.shape[-2:]) == (int(query_len), int(key_len))
    return False


class FlashDisentangledSelfAttention(_EagerDisentangledSelfAttention):
    """FlashDeBERTa-backed variant of native disentangled self-attention."""

    _warned_reasons: set[str] = set()

    @classmethod
    def _warn_once(cls, *, reason: str, message: str) -> None:
        """Emit one warning per process for a fallback reason.

        :param str reason: Stable reason key.
        :param str message: Warning message text.
        """

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

        :param torch.Tensor hidden_states: Key/value states.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param torch.Tensor query_states: Query states.
        :param torch.Tensor | None rel_embeddings: Relative embedding table.
        :return tuple[str, str] | None: Reason key and warning message, or ``None``.
        """

        query_len = int(query_states.shape[-2])
        key_len = int(hidden_states.shape[-2])
        dropout_p = float(getattr(self.dropout, "p", 0.0))

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
                str(_FLASHDEBERTA_IMPORT_ERROR)
                if _FLASHDEBERTA_IMPORT_ERROR is not None
                else "unknown import error"
            )
            return (
                "missing_flashdeberta",
                f"FlashDeBERTa attention is unavailable ({detail}); using eager attention.",
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
        """Return whether projected QKV tensors are unsupported by the flash kernel.

        :param torch.Tensor query_layer: Projected query tensor.
        :param torch.Tensor key_layer: Projected key tensor.
        :param torch.Tensor value_layer: Projected value tensor.
        :return tuple[str, str] | None: Reason key and warning message, or ``None``.
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

        if query_states is None:
            query_states = hidden_states

        reason = self._fallback_reason(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            query_states=query_states,
            rel_embeddings=rel_embeddings,
        )
        if reason is not None:
            key, message = reason
            self._warn_once(reason=key, message=message)
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )

        if flash_attention_with_disentangled is None:  # pragma: no cover - guarded above
            raise RuntimeError("FlashDeBERTa operator import unexpectedly unavailable during flash forward.")

        model_dtype = hidden_states.dtype
        bsz, query_len, _ = query_states.shape
        key_len = int(hidden_states.shape[-2])

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
            self._warn_once(reason=key, message=message)
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                query_states=query_states,
                relative_pos=relative_pos,
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

        seq_lengths = None
        if attention_mask is not None:
            seq_lengths = _mask4d_to_seqlens(attention_mask, seq_len=key_len)

        output = flash_attention_with_disentangled(
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

        output = output.transpose(1, 2).contiguous().view(bsz, query_len, self.all_head_size)
        output = output.to(dtype=model_dtype)

        if output_attentions:
            return output, None
        return output, None


__all__ = [
    "FlashDisentangledSelfAttention",
    "_is_pairwise_mask",
    "_mask4d_to_seqlens",
    "flashdeberta_import_error",
]
