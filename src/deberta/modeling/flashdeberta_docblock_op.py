"""Compile-safe doc-block-aware FlashDeBERTa attention.

Packed doc-block batches already carry compact ``doc_ids`` metadata in this
repo. This module turns those contiguous document spans into a ragged varlen
batch, runs the existing FlashDeBERTa varlen kernels over the packed segments,
and scatters the outputs back to the original padded ``(B, S, ...)`` layout.

The attention math itself is unchanged: this is a metadata + pack/scatter layer
around the tuned varlen kernels, exposed as an opaque CUDA custom op so
``torch.compile`` does not trace into the launcher details.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any

import torch

import deberta.modeling.flashdeberta_varlen_op as _varlen_mod
from deberta.modeling.flashdeberta_segment_pack import (
    segment_pack_padded_rows,
    segment_pack_padded_rows_pair,
    segment_pack_padded_rows_triple,
    segment_unpack_padded_rows,
    segment_unpack_padded_rows_pair,
    segment_unpack_padded_rows_triple,
)

_DOCBLOCK_OP_NAMESPACE = "deberta"
_DOCBLOCK_FWD_OP_NAME = "flashdeberta_docblock"
_DOCBLOCK_BWD_OP_NAME = "flashdeberta_docblock_backward"


@dataclass
class _DocBlockForwardAuxCacheEntry:
    """Forward-side packed tensors reused by doc-block backward."""

    output_ref: weakref.ReferenceType[torch.Tensor] | None
    segment_offsets: torch.Tensor
    segment_lengths: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    total_tokens: int
    q_unpad: torch.Tensor
    k_unpad: torch.Tensor
    v_unpad: torch.Tensor
    out_unpad: torch.Tensor
    lse_unpad: torch.Tensor
    pos_key_unpad: torch.Tensor | None
    pos_query_unpad: torch.Tensor | None


_DOCBLOCK_FORWARD_AUX_CACHE: dict[int, _DocBlockForwardAuxCacheEntry] = {}


def flashdeberta_docblock_import_error() -> Exception | None:
    """Return the most relevant import failure for doc-block flash support.

    :return Exception | None: Import failure or ``None`` when the varlen kernels are available.
    """

    return _varlen_mod.flashdeberta_varlen_import_error()


def flashdeberta_compiled_docblock_available() -> bool:
    """Return whether the opaque doc-block CUDA custom op is available.

    :return bool: True when the custom-op based CUDA path is registered.
    """

    return _FLASHDEBERTA_DOCBLOCK_CUSTOM_OP is not None and _FLASHDEBERTA_DOCBLOCK_BWD_CUSTOM_OP is not None


def _lookup_registered_op(namespace: str, name: str) -> Any | None:
    """Return a previously registered custom op overload, if one exists.

    :param str namespace: Operator namespace.
    :param str name: Operator name.
    :return Any | None: Registered overload or ``None``.
    """

    ns = getattr(torch.ops, namespace, None)
    if ns is None or not hasattr(ns, name):
        return None
    op = getattr(ns, name)
    return getattr(op, "default", op)


def _active_docblock_metadata(
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Return the active prefix of one fixed-shape doc-segment metadata batch.

    :param torch.Tensor segment_offsets: Fixed-shape flat padded row offsets per segment.
    :param torch.Tensor segment_lengths: Fixed-shape per-segment lengths.
    :param torch.Tensor cu_seqlens: Fixed-shape cumulative packed offsets.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        Active segment offsets, active segment lengths, active cumulative
        offsets, number of segments, maximum segment length, and total tokens.
    """

    num_segments = int(segment_lengths.count_nonzero().item())
    if num_segments <= 0:
        empty = segment_lengths[:0]
        return segment_offsets[:0], empty, cu_seqlens[:1], 0, 0, 0
    active_offsets = segment_offsets[:num_segments]
    active_lengths = segment_lengths[:num_segments]
    active_cu_seqlens = cu_seqlens[: num_segments + 1]
    max_seqlen = int(active_lengths.max().item())
    total_tokens = int(active_cu_seqlens[-1].item())
    return active_offsets, active_lengths, active_cu_seqlens, num_segments, max_seqlen, total_tokens


def _store_forward_aux_cache(
    *,
    output_padded: torch.Tensor,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_tokens: int,
    q_unpad: torch.Tensor,
    k_unpad: torch.Tensor,
    v_unpad: torch.Tensor,
    out_unpad: torch.Tensor,
    lse_unpad: torch.Tensor,
    pos_key_unpad: torch.Tensor | None,
    pos_query_unpad: torch.Tensor | None,
) -> None:
    """Stash forward packed tensors so backward does not rebuild them.

    :param torch.Tensor output_padded: Returned padded attention output tensor.
    :param torch.Tensor segment_offsets: Flat padded row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets.
    :param int max_seqlen: Maximum segment length in the batch.
    :param int total_tokens: Total packed token count.
    :param torch.Tensor q_unpad: Packed query tensor.
    :param torch.Tensor k_unpad: Packed key tensor.
    :param torch.Tensor v_unpad: Packed value tensor.
    :param torch.Tensor out_unpad: Packed forward output tensor.
    :param torch.Tensor lse_unpad: Packed forward LSE tensor.
    :param torch.Tensor | None pos_key_unpad: Optional packed c2p tensor.
    :param torch.Tensor | None pos_query_unpad: Optional packed p2c tensor.
    """

    cache_key = id(output_padded)
    output_ref: weakref.ReferenceType[torch.Tensor] | None = None
    try:
        output_ref = weakref.ref(
            output_padded, lambda _ref, key=cache_key: _DOCBLOCK_FORWARD_AUX_CACHE.pop(key, None)
        )
    except TypeError:
        output_ref = None

    _DOCBLOCK_FORWARD_AUX_CACHE[cache_key] = _DocBlockForwardAuxCacheEntry(
        output_ref=output_ref,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        max_seqlen=int(max_seqlen),
        total_tokens=int(total_tokens),
        q_unpad=q_unpad,
        k_unpad=k_unpad,
        v_unpad=v_unpad,
        out_unpad=out_unpad,
        lse_unpad=lse_unpad,
        pos_key_unpad=pos_key_unpad,
        pos_query_unpad=pos_query_unpad,
    )


def _pop_forward_aux_cache(output_padded: torch.Tensor) -> _DocBlockForwardAuxCacheEntry | None:
    """Return and remove cached forward aux tensors for one output tensor.

    :param torch.Tensor output_padded: Padded output tensor returned by the custom op.
    :return _DocBlockForwardAuxCacheEntry | None: Cached aux entry or ``None``.
    """

    cache_key = id(output_padded)
    cached = _DOCBLOCK_FORWARD_AUX_CACHE.pop(cache_key, None)
    if cached is None:
        return None
    cached_output = cached.output_ref() if cached.output_ref is not None else None
    if cached.output_ref is not None and cached_output is not output_padded:
        return None
    return cached


def _docblock_forward_impl(
    *,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
    require_lse: bool,
    stash_backward_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run doc-block-aware attention over packed contiguous document segments.

    :param torch.Tensor query_layer: Queries in ``(B, S, H, D)`` layout.
    :param torch.Tensor key_layer: Keys in ``(B, S, H, D)`` layout.
    :param torch.Tensor value_layer: Values in ``(B, S, H, D)`` layout.
    :param torch.Tensor segment_offsets: Flat padded row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param torch.Tensor | None pos_key: Optional c2p tensor in ``(B, S, H, P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c tensor in ``(B, S, H, P)`` layout.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :param bool require_lse: Whether the caller also needs padded LSE values.
    :param bool stash_backward_cache: Whether to cache packed forward tensors for backward reuse.
    :raises RuntimeError: If the low-level varlen kernels are unavailable.
    :return tuple[torch.Tensor, torch.Tensor | None]: Padded output and optional padded LSE.
    """

    if (
        _varlen_mod._flash_attn_v2_fwd_dise_lowlevel is None
        and _varlen_mod._flash_attention_with_disentangled_varlen_highlevel is None
    ):
        detail = flashdeberta_docblock_import_error()
        raise RuntimeError(
            "FlashDeBERTa doc-block attention is unavailable."
            if detail is None
            else f"FlashDeBERTa doc-block attention is unavailable ({detail})."
        )

    batch_size = int(query_layer.shape[0])
    seq_len = int(query_layer.shape[1])
    (
        active_segment_offsets,
        active_segment_lengths,
        active_cu_seqlens,
        num_segments,
        max_seqlen,
        total_tokens,
    ) = _active_docblock_metadata(
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
    )
    att_span = int(position_buckets) if int(position_buckets) > 0 else int(max_relative_distance)

    if total_tokens == 0:
        output = torch.zeros_like(query_layer)
        if require_lse:
            lse = torch.zeros(
                (batch_size, seq_len, int(query_layer.shape[2])),
                device=query_layer.device,
                dtype=torch.float32,
            )
            return output, lse
        return output, None

    q_unpad, k_unpad, v_unpad = segment_pack_padded_rows_triple(
        query_layer,
        key_layer,
        value_layer,
        segment_offsets=active_segment_offsets,
        segment_lengths=active_segment_lengths,
        cu_seqlens=active_cu_seqlens,
        total_tokens=total_tokens,
    )
    if pos_key is not None and pos_query is not None:
        pos_key_unpad, pos_query_unpad = segment_pack_padded_rows_pair(
            pos_key,
            pos_query,
            segment_offsets=active_segment_offsets,
            segment_lengths=active_segment_lengths,
            cu_seqlens=active_cu_seqlens,
            total_tokens=total_tokens,
        )
    else:
        pos_key_unpad = (
            segment_pack_padded_rows(
                pos_key,
                segment_offsets=active_segment_offsets,
                segment_lengths=active_segment_lengths,
                cu_seqlens=active_cu_seqlens,
                total_tokens=total_tokens,
            )
            if pos_key is not None
            else None
        )
        pos_query_unpad = (
            segment_pack_padded_rows(
                pos_query,
                segment_offsets=active_segment_offsets,
                segment_lengths=active_segment_lengths,
                cu_seqlens=active_cu_seqlens,
                total_tokens=total_tokens,
            )
            if pos_query is not None
            else None
        )

    if (
        _varlen_mod._flash_attn_v2_fwd_dise_lowlevel is not None
        and _varlen_mod._get_fwd_config_lowlevel is not None
    ):
        override = _varlen_mod._varlen_kernel_override_from_env(kind="fwd")
        if override is not None:
            block_m, block_n, num_stages, num_warps = override
        else:
            block_m, block_n, num_stages, num_warps = _varlen_mod._get_fwd_config_lowlevel(
                total_tokens=int(q_unpad.shape[0]),
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                D=int(query_layer.shape[-1]),
                causal=bool(causal),
                disentangled=True,
                att_span=att_span,
            )
        out_unpad, lse_unpad = _varlen_mod._flash_attn_v2_fwd_dise_lowlevel(
            q_unpad,
            k_unpad,
            v_unpad,
            pos_key_unpad,
            pos_query_unpad,
            active_cu_seqlens,
            active_cu_seqlens,
            max_seqlen,
            max_seqlen,
            bool(causal),
            float(sm_scale),
            block_m,
            block_n,
            int(position_buckets),
            int(max_relative_distance),
            num_warps,
            num_stages,
            att_span,
        )
    else:
        out_unpad = _varlen_mod._flash_attention_with_disentangled_varlen_highlevel(
            q_unpad,
            k_unpad,
            v_unpad,
            pos_key_unpad,
            pos_query_unpad,
            active_cu_seqlens,
            active_cu_seqlens,
            max_seqlen,
            max_seqlen,
            bool(causal),
            float(sm_scale),
            int(position_buckets),
            int(max_relative_distance),
        )
        lse_unpad = None

    out_padded = segment_unpack_padded_rows(
        out_unpad,
        segment_offsets=active_segment_offsets,
        segment_lengths=active_segment_lengths,
        cu_seqlens=active_cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    if not require_lse:
        return out_padded, None
    if lse_unpad is None:
        raise RuntimeError(
            "Compiled FlashDeBERTa doc-block attention requires low-level forward primitives with LSE support."
        )
    lse_padded = segment_unpack_padded_rows(
        lse_unpad,
        segment_offsets=active_segment_offsets,
        segment_lengths=active_segment_lengths,
        cu_seqlens=active_cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
    ).contiguous()
    if stash_backward_cache:
        _store_forward_aux_cache(
            output_padded=out_padded,
            segment_offsets=active_segment_offsets,
            segment_lengths=active_segment_lengths,
            cu_seqlens=active_cu_seqlens,
            max_seqlen=max_seqlen,
            total_tokens=total_tokens,
            q_unpad=q_unpad,
            k_unpad=k_unpad,
            v_unpad=v_unpad,
            out_unpad=out_unpad,
            lse_unpad=lse_unpad,
            pos_key_unpad=pos_key_unpad,
            pos_query_unpad=pos_query_unpad,
        )
    return out_padded, lse_padded


def _docblock_backward_impl(
    *,
    grad_output: torch.Tensor,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    output_padded: torch.Tensor,
    lse_padded: torch.Tensor,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
    q_unpad: torch.Tensor | None,
    k_unpad: torch.Tensor | None,
    v_unpad: torch.Tensor | None,
    out_unpad: torch.Tensor | None,
    lse_unpad: torch.Tensor | None,
    pos_key_unpad: torch.Tensor | None,
    pos_query_unpad: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run doc-block backward and scatter packed gradients back to padded layout.

    :param torch.Tensor grad_output: Gradient of padded output in ``(B, S, H, D)`` layout.
    :param torch.Tensor query_layer: Forward queries in ``(B, S, H, D)`` layout.
    :param torch.Tensor key_layer: Forward keys in ``(B, S, H, D)`` layout.
    :param torch.Tensor value_layer: Forward values in ``(B, S, H, D)`` layout.
    :param torch.Tensor output_padded: Forward output in ``(B, S, H, D)`` layout.
    :param torch.Tensor lse_padded: Forward padded LSE in ``(B, S, H)`` layout.
    :param torch.Tensor segment_offsets: Flat padded row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param torch.Tensor | None pos_key: Optional c2p tensor in ``(B, S, H, P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c tensor in ``(B, S, H, P)`` layout.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :param torch.Tensor | None q_unpad: Optional cached packed queries.
    :param torch.Tensor | None k_unpad: Optional cached packed keys.
    :param torch.Tensor | None v_unpad: Optional cached packed values.
    :param torch.Tensor | None out_unpad: Optional cached packed forward output.
    :param torch.Tensor | None lse_unpad: Optional cached packed forward LSE.
    :param torch.Tensor | None pos_key_unpad: Optional cached packed c2p tensor.
    :param torch.Tensor | None pos_query_unpad: Optional cached packed p2c tensor.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        Gradients in the same padded layouts as the forward inputs.
    """

    batch_size = int(query_layer.shape[0])
    seq_len = int(query_layer.shape[1])
    (
        active_segment_offsets,
        active_segment_lengths,
        active_cu_seqlens,
        num_segments,
        max_seqlen,
        total_tokens,
    ) = _active_docblock_metadata(
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
    )

    if total_tokens == 0:
        dq = torch.zeros_like(query_layer)
        dk = torch.zeros_like(key_layer)
        dv = torch.zeros_like(value_layer)
        dpos_key = torch.zeros_like(pos_key) if pos_key is not None else None
        dpos_query = torch.zeros_like(pos_query) if pos_query is not None else None
        return dq, dk, dv, dpos_key, dpos_query

    if q_unpad is None or k_unpad is None or v_unpad is None:
        q_unpad, k_unpad, v_unpad = segment_pack_padded_rows_triple(
            query_layer,
            key_layer,
            value_layer,
            segment_offsets=active_segment_offsets,
            segment_lengths=active_segment_lengths,
            cu_seqlens=active_cu_seqlens,
            total_tokens=total_tokens,
        )
    if out_unpad is None:
        out_unpad = segment_pack_padded_rows(
            output_padded,
            segment_offsets=active_segment_offsets,
            segment_lengths=active_segment_lengths,
            cu_seqlens=active_cu_seqlens,
            total_tokens=total_tokens,
        )
    grad_unpad = segment_pack_padded_rows(
        grad_output,
        segment_offsets=active_segment_offsets,
        segment_lengths=active_segment_lengths,
        cu_seqlens=active_cu_seqlens,
        total_tokens=total_tokens,
    )
    delta = (out_unpad * grad_unpad).sum(dim=-1)
    if lse_unpad is None:
        lse_unpad = segment_pack_padded_rows(
            lse_padded,
            segment_offsets=active_segment_offsets,
            segment_lengths=active_segment_lengths,
            cu_seqlens=active_cu_seqlens,
            total_tokens=total_tokens,
        )
    if pos_key is not None and pos_query is not None and (pos_key_unpad is None or pos_query_unpad is None):
        pos_key_unpad, pos_query_unpad = segment_pack_padded_rows_pair(
            pos_key,
            pos_query,
            segment_offsets=active_segment_offsets,
            segment_lengths=active_segment_lengths,
            cu_seqlens=active_cu_seqlens,
            total_tokens=total_tokens,
        )
    else:
        if pos_key is not None and pos_key_unpad is None:
            pos_key_unpad = segment_pack_padded_rows(
                pos_key,
                segment_offsets=active_segment_offsets,
                segment_lengths=active_segment_lengths,
                cu_seqlens=active_cu_seqlens,
                total_tokens=total_tokens,
            )
        if pos_query is not None and pos_query_unpad is None:
            pos_query_unpad = segment_pack_padded_rows(
                pos_query,
                segment_offsets=active_segment_offsets,
                segment_lengths=active_segment_lengths,
                cu_seqlens=active_cu_seqlens,
                total_tokens=total_tokens,
            )

    dq_unpad, dk_unpad, dv_unpad, dpos_key_unpad, dpos_query_unpad = _varlen_mod._varlen_backward_raw_impl(
        q_unpad=q_unpad,
        k_unpad=k_unpad,
        v_unpad=v_unpad,
        out_unpad=out_unpad,
        grad_unpad=grad_unpad,
        lse_unpad=lse_unpad,
        delta=delta,
        pos_key_unpad=pos_key_unpad,
        pos_query_unpad=pos_query_unpad,
        cu_seqlens=active_cu_seqlens,
        batch_size=num_segments,
        seq_bound=max_seqlen,
        token_capacity=total_tokens,
        sm_scale=sm_scale,
        position_buckets=position_buckets,
        max_relative_distance=max_relative_distance,
        causal=causal,
        dense_mid_tensors=False,
    )

    dq, dk, dv = segment_unpack_padded_rows_triple(
        dq_unpad,
        dk_unpad,
        dv_unpad,
        segment_offsets=active_segment_offsets,
        segment_lengths=active_segment_lengths,
        cu_seqlens=active_cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    if dpos_key_unpad is not None and dpos_query_unpad is not None:
        dpos_key, dpos_query = segment_unpack_padded_rows_pair(
            dpos_key_unpad,
            dpos_query_unpad,
            segment_offsets=active_segment_offsets,
            segment_lengths=active_segment_lengths,
            cu_seqlens=active_cu_seqlens,
            batch_size=batch_size,
            seq_len=seq_len,
        )
    else:
        dpos_key = (
            segment_unpack_padded_rows(
                dpos_key_unpad,
                segment_offsets=active_segment_offsets,
                segment_lengths=active_segment_lengths,
                cu_seqlens=active_cu_seqlens,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            if dpos_key_unpad is not None
            else None
        )
        dpos_query = (
            segment_unpack_padded_rows(
                dpos_query_unpad,
                segment_offsets=active_segment_offsets,
                segment_lengths=active_segment_lengths,
                cu_seqlens=active_cu_seqlens,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            if dpos_query_unpad is not None
            else None
        )
    return dq, dk, dv, dpos_key, dpos_query


def _build_docblock_custom_ops() -> tuple[Any | None, Any | None]:
    """Register or retrieve the opaque doc-block-aware custom ops.

    :return tuple[Any | None, Any | None]: Forward and backward custom-op handles.
    """

    existing_forward = _lookup_registered_op(_DOCBLOCK_OP_NAMESPACE, _DOCBLOCK_FWD_OP_NAME)
    existing_backward = _lookup_registered_op(_DOCBLOCK_OP_NAMESPACE, _DOCBLOCK_BWD_OP_NAME)
    if existing_forward is not None and existing_backward is not None:
        return existing_forward, existing_backward

    if (
        _varlen_mod._flash_attn_v2_fwd_dise_lowlevel is None
        or _varlen_mod._flash_attn_v2_bwd_dise_varlen_lowlevel is None
        or not hasattr(torch, "library")
        or not hasattr(torch.library, "custom_op")
    ):
        return None, None

    @torch.library.custom_op(
        f"{_DOCBLOCK_OP_NAMESPACE}::{_DOCBLOCK_FWD_OP_NAME}",
        mutates_args=(),
        device_types="cuda",
        schema=(
            "(Tensor q, Tensor k, Tensor v, Tensor segment_offsets, Tensor segment_lengths, Tensor cu_seqlens, "
            "Tensor? pos_key, Tensor? pos_query, float sm_scale, int position_buckets, int max_relative_distance, "
            "bool causal) -> (Tensor, Tensor)"
        ),
    )
    def _forward_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        segment_offsets: torch.Tensor,
        segment_lengths: torch.Tensor,
        cu_seqlens: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run doc-block-aware forward as one opaque CUDA op.

        :param torch.Tensor q: Padded queries in ``(B, S, H, D)`` layout.
        :param torch.Tensor k: Padded keys in ``(B, S, H, D)`` layout.
        :param torch.Tensor v: Padded values in ``(B, S, H, D)`` layout.
        :param torch.Tensor segment_offsets: Flat padded row offsets per segment.
        :param torch.Tensor segment_lengths: Per-segment lengths.
        :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
        :param torch.Tensor | None pos_key: Optional c2p tensor.
        :param torch.Tensor | None pos_query: Optional p2c tensor.
        :param float sm_scale: Softmax scale.
        :param int position_buckets: Relative-position bucket count.
        :param int max_relative_distance: Maximum relative distance.
        :param bool causal: Whether causal masking is enabled.
        :return tuple[torch.Tensor, torch.Tensor]: Padded output and padded LSE tensors.
        """

        return _docblock_forward_impl(
            query_layer=q,
            key_layer=k,
            value_layer=v,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            position_buckets=position_buckets,
            max_relative_distance=max_relative_distance,
            causal=causal,
            require_lse=True,
            stash_backward_cache=True,
        )

    @torch.library.register_fake(_forward_op)
    def _forward_op_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        segment_offsets: torch.Tensor,
        segment_lengths: torch.Tensor,
        cu_seqlens: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fake forward outputs with static padded shapes.

        :param torch.Tensor q: Fake query tensor.
        :param torch.Tensor k: Fake key tensor.
        :param torch.Tensor v: Fake value tensor.
        :param torch.Tensor segment_offsets: Fake segment offsets tensor.
        :param torch.Tensor segment_lengths: Fake segment lengths tensor.
        :param torch.Tensor cu_seqlens: Fake cumulative offsets tensor.
        :param torch.Tensor | None pos_key: Fake optional c2p tensor.
        :param torch.Tensor | None pos_query: Fake optional p2c tensor.
        :param float sm_scale: Fake softmax scale.
        :param int position_buckets: Fake bucket count.
        :param int max_relative_distance: Fake maximum relative distance.
        :param bool causal: Fake causal flag.
        :return tuple[torch.Tensor, torch.Tensor]: Fake padded output and padded LSE tensors.
        """

        del (
            k,
            v,
            segment_offsets,
            segment_lengths,
            cu_seqlens,
            pos_key,
            pos_query,
            sm_scale,
            position_buckets,
            max_relative_distance,
            causal,
        )
        lse = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        return torch.empty(q.shape, device=q.device, dtype=q.dtype), lse

    @torch.library.custom_op(
        f"{_DOCBLOCK_OP_NAMESPACE}::{_DOCBLOCK_BWD_OP_NAME}",
        mutates_args=(),
        device_types="cuda",
        schema=(
            "(Tensor grad_out, Tensor q, Tensor k, Tensor v, Tensor segment_offsets, Tensor segment_lengths, "
            "Tensor cu_seqlens, Tensor out, Tensor lse, Tensor? pos_key, Tensor? pos_query, float sm_scale, "
            "int position_buckets, int max_relative_distance, bool causal) -> "
            "(Tensor, Tensor, Tensor, Tensor?, Tensor?)"
        ),
    )
    def _backward_op(
        grad_out: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        segment_offsets: torch.Tensor,
        segment_lengths: torch.Tensor,
        cu_seqlens: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Run doc-block-aware backward as one opaque CUDA op.

        :param torch.Tensor grad_out: Gradient of padded output.
        :param torch.Tensor q: Forward padded queries.
        :param torch.Tensor k: Forward padded keys.
        :param torch.Tensor v: Forward padded values.
        :param torch.Tensor segment_offsets: Flat padded row offsets per segment.
        :param torch.Tensor segment_lengths: Per-segment lengths.
        :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
        :param torch.Tensor out: Forward padded output.
        :param torch.Tensor lse: Forward padded LSE tensor.
        :param torch.Tensor | None pos_key: Optional c2p tensor.
        :param torch.Tensor | None pos_query: Optional p2c tensor.
        :param float sm_scale: Softmax scale.
        :param int position_buckets: Relative-position bucket count.
        :param int max_relative_distance: Maximum relative distance.
        :param bool causal: Whether causal masking is enabled.
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            Padded gradients for q/k/v and optional positional tensors.
        """

        return _docblock_backward_impl(
            grad_output=grad_out,
            query_layer=q,
            key_layer=k,
            value_layer=v,
            output_padded=out,
            lse_padded=lse,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            position_buckets=position_buckets,
            max_relative_distance=max_relative_distance,
            causal=causal,
            q_unpad=None,
            k_unpad=None,
            v_unpad=None,
            out_unpad=None,
            lse_unpad=None,
            pos_key_unpad=None,
            pos_query_unpad=None,
        )

    @torch.library.register_fake(_backward_op)
    def _backward_op_fake(
        grad_out: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        segment_offsets: torch.Tensor,
        segment_lengths: torch.Tensor,
        cu_seqlens: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return fake backward outputs with static padded shapes.

        :param torch.Tensor grad_out: Fake gradient tensor.
        :param torch.Tensor q: Fake query tensor.
        :param torch.Tensor k: Fake key tensor.
        :param torch.Tensor v: Fake value tensor.
        :param torch.Tensor segment_offsets: Fake segment offsets tensor.
        :param torch.Tensor segment_lengths: Fake segment lengths tensor.
        :param torch.Tensor cu_seqlens: Fake cumulative offsets tensor.
        :param torch.Tensor out: Fake output tensor.
        :param torch.Tensor lse: Fake LSE tensor.
        :param torch.Tensor | None pos_key: Fake optional c2p tensor.
        :param torch.Tensor | None pos_query: Fake optional p2c tensor.
        :param float sm_scale: Fake softmax scale.
        :param int position_buckets: Fake bucket count.
        :param int max_relative_distance: Fake maximum relative distance.
        :param bool causal: Fake causal flag.
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            Fake padded gradients for q/k/v and optional positional tensors.
        """

        del (
            grad_out,
            segment_offsets,
            segment_lengths,
            cu_seqlens,
            out,
            lse,
            sm_scale,
            position_buckets,
            max_relative_distance,
            causal,
        )
        dpos_key = (
            torch.empty(pos_key.shape, device=pos_key.device, dtype=pos_key.dtype)
            if pos_key is not None
            else None
        )
        dpos_query = (
            torch.empty(pos_query.shape, device=pos_query.device, dtype=pos_query.dtype)
            if pos_query is not None
            else None
        )
        return (
            torch.empty(q.shape, device=q.device, dtype=q.dtype),
            torch.empty(k.shape, device=k.device, dtype=k.dtype),
            torch.empty(v.shape, device=v.device, dtype=v.dtype),
            dpos_key,
            dpos_query,
        )

    def _setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Save forward inputs and outputs needed by the doc-block backward helper.

        :param Any ctx: Autograd context object.
        :param tuple[Any, ...] inputs: Forward custom-op inputs.
        :param tuple[torch.Tensor, torch.Tensor] output: Forward custom-op outputs.
        """

        (
            q,
            k,
            v,
            segment_offsets,
            segment_lengths,
            cu_seqlens,
            pos_key,
            pos_query,
            sm_scale,
            position_buckets,
            max_relative_distance,
            causal,
        ) = inputs
        out, lse = output
        saved: list[torch.Tensor] = [q, k, v, segment_offsets, segment_lengths, cu_seqlens, out, lse]
        if pos_key is not None:
            saved.append(pos_key)
        if pos_query is not None:
            saved.append(pos_query)
        ctx.has_pos_key = pos_key is not None
        ctx.has_pos_query = pos_query is not None
        ctx.save_for_backward(*saved)
        ctx.sm_scale = float(sm_scale)
        ctx.position_buckets = int(position_buckets)
        ctx.max_relative_distance = int(max_relative_distance)
        ctx.causal = bool(causal)

    def _backward(
        ctx: Any,
        grad_out: torch.Tensor | None,
        grad_lse: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Dispatch backward through the opaque doc-block backward helper.

        :param Any ctx: Autograd context populated by ``_setup_context``.
        :param torch.Tensor | None grad_out: Gradient of padded output.
        :param torch.Tensor | None grad_lse: Gradient of padded LSE output.
        :return tuple[torch.Tensor | None, ...]: Gradients for the forward custom-op inputs.
        """

        del grad_lse
        saved = list(ctx.saved_tensors)
        q, k, v, segment_offsets, segment_lengths, cu_seqlens, out, lse = saved[:8]
        next_idx = 8
        pos_key = saved[next_idx] if bool(ctx.has_pos_key) else None
        if bool(ctx.has_pos_key):
            next_idx += 1
        pos_query = saved[next_idx] if bool(ctx.has_pos_query) else None
        grad = grad_out if grad_out is not None else torch.zeros_like(out)
        cached = _pop_forward_aux_cache(out)
        if cached is not None:
            dq, dk, dv, dpos_key, dpos_query = _docblock_backward_impl(
                grad_output=grad,
                query_layer=q,
                key_layer=k,
                value_layer=v,
                output_padded=out,
                lse_padded=lse,
                segment_offsets=segment_offsets,
                segment_lengths=segment_lengths,
                cu_seqlens=cu_seqlens,
                pos_key=pos_key,
                pos_query=pos_query,
                sm_scale=ctx.sm_scale,
                position_buckets=ctx.position_buckets,
                max_relative_distance=ctx.max_relative_distance,
                causal=ctx.causal,
                q_unpad=cached.q_unpad,
                k_unpad=cached.k_unpad,
                v_unpad=cached.v_unpad,
                out_unpad=cached.out_unpad,
                lse_unpad=cached.lse_unpad,
                pos_key_unpad=cached.pos_key_unpad,
                pos_query_unpad=cached.pos_query_unpad,
            )
        else:
            dq, dk, dv, dpos_key, dpos_query = _backward_op(
                grad,
                q,
                k,
                v,
                segment_offsets,
                segment_lengths,
                cu_seqlens,
                out,
                lse,
                pos_key,
                pos_query,
                ctx.sm_scale,
                ctx.position_buckets,
                ctx.max_relative_distance,
                ctx.causal,
            )
        return dq, dk, dv, None, None, None, dpos_key, dpos_query, None, None, None, None

    torch.library.register_autograd(_forward_op, _backward, setup_context=_setup_context)
    return _forward_op, _backward_op


_FLASHDEBERTA_DOCBLOCK_CUSTOM_OP, _FLASHDEBERTA_DOCBLOCK_BWD_CUSTOM_OP = _build_docblock_custom_ops()


def flashdeberta_docblock(
    *,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
) -> torch.Tensor:
    """Run doc-block-aware FlashDeBERTa attention.

    :param torch.Tensor query_layer: Queries in ``(B, S, H, D)`` layout.
    :param torch.Tensor key_layer: Keys in ``(B, S, H, D)`` layout.
    :param torch.Tensor value_layer: Values in ``(B, S, H, D)`` layout.
    :param torch.Tensor segment_offsets: Flat padded row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param torch.Tensor | None pos_key: Optional c2p tensor.
    :param torch.Tensor | None pos_query: Optional p2c tensor.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :return torch.Tensor: Attention output in ``(B, S, H, D)`` layout.
    """

    if _FLASHDEBERTA_DOCBLOCK_CUSTOM_OP is not None and query_layer.device.type == "cuda":
        output, _ = _FLASHDEBERTA_DOCBLOCK_CUSTOM_OP(
            query_layer,
            key_layer,
            value_layer,
            segment_offsets,
            segment_lengths,
            cu_seqlens,
            pos_key,
            pos_query,
            float(sm_scale),
            int(position_buckets),
            int(max_relative_distance),
            bool(causal),
        )
        return output

    output, _ = _docblock_forward_impl(
        query_layer=query_layer,
        key_layer=key_layer,
        value_layer=value_layer,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        pos_key=pos_key,
        pos_query=pos_query,
        sm_scale=sm_scale,
        position_buckets=position_buckets,
        max_relative_distance=max_relative_distance,
        causal=causal,
        require_lse=False,
        stash_backward_cache=False,
    )
    return output


__all__ = [
    "flashdeberta_compiled_docblock_available",
    "flashdeberta_docblock",
    "flashdeberta_docblock_import_error",
]
