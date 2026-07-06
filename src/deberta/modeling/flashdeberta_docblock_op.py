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

from typing import Any

import torch

import deberta.modeling.flashdeberta_fixed_op as _fixed_mod
import deberta.modeling.flashdeberta_varlen_op as _varlen_mod
from deberta.modeling.flashdeberta_kernel_tuning import (
    FlashKernelContext,
    resolve_flash_kernel_config,
)
from deberta.modeling.flashdeberta_segment_pack import (
    segment_pack_grad_and_delta_from_padded,
    segment_pack_padded_rows,
    segment_pack_padded_rows_pair,
    segment_pack_padded_rows_triple,
    segment_unpack_padded_rows,
    segment_unpack_padded_rows_pair,
    segment_unpack_padded_rows_triple,
)

_DOCBLOCK_OP_NAMESPACE = "deberta"
_DOCBLOCK_FWD_OP_NAME = "flashdeberta_docblock_v2"
_DOCBLOCK_BWD_OP_NAME = "flashdeberta_docblock_backward_v2"


def _scalar_int(value: int | torch.Tensor, *, name: str) -> int:
    """Return a Python int from a host scalar.

    :param int | torch.Tensor value: Python integer or scalar tensor.
    :param str name: Name used in validation errors.
    :raises ValueError: If a tensor value is not scalar.
    :return int: Host integer value.
    """

    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar tensor, got shape {tuple(value.shape)}.")
        if value.device.type == "cpu":
            return int(value)
        return int(value.detach().cpu().item())
    return int(value)


def _scalar_tensor(value: int | torch.Tensor, *, name: str) -> torch.Tensor:
    """Return a CPU scalar tensor for compile-stable custom-op inputs.

    :param int | torch.Tensor value: Python integer or scalar tensor.
    :param str name: Name used in validation errors.
    :raises ValueError: If a tensor value is not scalar.
    :return torch.Tensor: CPU scalar int32 tensor.
    """

    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar tensor, got shape {tuple(value.shape)}.")
        if value.device.type == "cpu" and value.dtype == torch.int32:
            return value
        if value.device.type == "cpu":
            return value.to(dtype=torch.int32)
        return torch.tensor(int(value.detach().cpu().item()), dtype=torch.int32)
    return torch.tensor(int(value), dtype=torch.int32)


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
    num_segments: int,
    max_seqlen: int,
    total_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Return the active prefix of one fixed-shape doc-segment metadata batch.

    :param torch.Tensor segment_offsets: Fixed-shape flat padded row offsets per segment.
    :param torch.Tensor segment_lengths: Fixed-shape per-segment lengths.
    :param torch.Tensor cu_seqlens: Fixed-shape cumulative packed offsets.
    :param int num_segments: Host-side active segment count.
    :param int max_seqlen: Host-side maximum segment length.
    :param int total_tokens: Host-side total active token count.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        Active segment offsets, active segment lengths, active cumulative
        offsets, number of segments, maximum segment length, and total tokens.
    """

    num_segments = max(0, min(int(num_segments), int(segment_lengths.shape[0])))
    if num_segments <= 0:
        empty = segment_lengths[:0]
        return segment_offsets[:0], empty, cu_seqlens[:1], 0, 0, 0
    active_offsets = segment_offsets[:num_segments]
    active_lengths = segment_lengths[:num_segments]
    active_cu_seqlens = cu_seqlens[: num_segments + 1]
    return (
        active_offsets,
        active_lengths,
        active_cu_seqlens,
        num_segments,
        max(0, int(max_seqlen)),
        max(0, int(total_tokens)),
    )


def _pad_packed_aux(tensor: torch.Tensor | None, *, capacity: int) -> torch.Tensor | None:
    """Return ``tensor`` with a fixed packed-row capacity for saved aux outputs.

    :param torch.Tensor | None tensor: Packed tensor with active rows.
    :param int capacity: Desired first-dimension capacity.
    :return torch.Tensor | None: Tensor with ``capacity`` rows, or ``None``.
    """

    if tensor is None:
        return None
    capacity = max(0, int(capacity))
    if int(tensor.shape[0]) == capacity:
        return tensor
    padded = tensor.new_empty((capacity,) + tuple(tensor.shape[1:]))
    active_rows = min(int(tensor.shape[0]), capacity)
    if active_rows > 0:
        padded[:active_rows].copy_(tensor[:active_rows])
    return padded


def _active_packed_prefix(tensor: torch.Tensor | None, *, total_tokens: int) -> torch.Tensor | None:
    """Return the active packed prefix from a possibly fixed-capacity aux tensor.

    :param torch.Tensor | None tensor: Packed auxiliary tensor.
    :param int total_tokens: Active packed-token count.
    :return torch.Tensor | None: Active prefix view, or ``None``.
    """

    if tensor is None:
        return None
    total = max(0, int(total_tokens))
    if int(tensor.shape[0]) == total:
        return tensor
    return tensor[:total]


def _select_rows_or_none(tensor: torch.Tensor | None, rows: torch.Tensor) -> torch.Tensor | None:
    """Gather batch rows when the optional tensor is present.

    :param torch.Tensor | None tensor: Optional tensor whose first dim is batch.
    :param torch.Tensor rows: Row indices to gather.
    :return torch.Tensor | None: Gathered tensor or ``None``.
    """

    if tensor is None:
        return None
    if int(rows.numel()) == 0:
        return tensor[:0]
    return tensor.index_select(0, rows)


def _docblock_fixed_forward_impl(
    *,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    seq_lengths: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fixed FlashDeBERTa path on a subset of batch rows.

    :param torch.Tensor query_layer: Queries in ``(B, S, H, D)`` layout.
    :param torch.Tensor key_layer: Keys in ``(B, S, H, D)`` layout.
    :param torch.Tensor value_layer: Values in ``(B, S, H, D)`` layout.
    :param torch.Tensor seq_lengths: Per-row active lengths.
    :param torch.Tensor | None pos_key: Optional c2p tensor in ``(B, S, H, P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c tensor in ``(B, S, H, P)`` layout.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :return tuple[torch.Tensor, torch.Tensor]: Output and LSE in ``(B, S, H, *)`` layouts.
    """

    q = query_layer.transpose(1, 2).contiguous()
    k = key_layer.transpose(1, 2).contiguous()
    v = value_layer.transpose(1, 2).contiguous()
    pk = pos_key.transpose(1, 2).contiguous() if pos_key is not None else None
    pq = pos_query.transpose(1, 2).contiguous() if pos_query is not None else None
    out, lse = _fixed_mod._fixed_eager_forward_impl(
        query_layer=q,
        key_layer=k,
        value_layer=v,
        seq_lengths=seq_lengths,
        pos_key=pk,
        pos_query=pq,
        sm_scale=sm_scale,
        position_buckets=position_buckets,
        max_relative_distance=max_relative_distance,
        causal=causal,
        require_lse=True,
    )
    if lse is None:  # pragma: no cover - guarded by low-level availability
        raise RuntimeError("Doc-block fixed forward requires fixed-length LSE support.")
    return out.transpose(1, 2).contiguous(), lse.transpose(1, 2).contiguous()


def _docblock_fixed_backward_impl(
    *,
    grad_output: torch.Tensor,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    seq_lengths: torch.Tensor,
    output_padded: torch.Tensor,
    lse_padded: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run fixed-length backward on a subset of batch rows.

    :param torch.Tensor grad_output: Gradient in ``(B, S, H, D)`` layout.
    :param torch.Tensor query_layer: Queries in ``(B, S, H, D)`` layout.
    :param torch.Tensor key_layer: Keys in ``(B, S, H, D)`` layout.
    :param torch.Tensor value_layer: Values in ``(B, S, H, D)`` layout.
    :param torch.Tensor seq_lengths: Per-row active lengths.
    :param torch.Tensor output_padded: Forward output in ``(B, S, H, D)`` layout.
    :param torch.Tensor lse_padded: Forward LSE in ``(B, S, H)`` layout.
    :param torch.Tensor | None pos_key: Optional c2p tensor in ``(B, S, H, P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c tensor in ``(B, S, H, P)`` layout.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        Gradients in the same layouts as the forward inputs.
    """

    q = query_layer.transpose(1, 2).contiguous()
    k = key_layer.transpose(1, 2).contiguous()
    v = value_layer.transpose(1, 2).contiguous()
    grad = grad_output.transpose(1, 2).contiguous()
    out = output_padded.transpose(1, 2).contiguous()
    lse = lse_padded.transpose(1, 2).contiguous()
    pk = pos_key.transpose(1, 2).contiguous() if pos_key is not None else None
    pq = pos_query.transpose(1, 2).contiguous() if pos_query is not None else None
    dq, dk, dv, dpk, dpq = _fixed_mod._fixed_eager_backward_impl(
        grad_output=grad,
        query_layer=q,
        key_layer=k,
        value_layer=v,
        seq_lengths=seq_lengths,
        output=out,
        lse=lse,
        pos_key=pk,
        pos_query=pq,
        sm_scale=sm_scale,
        position_buckets=position_buckets,
        max_relative_distance=max_relative_distance,
        causal=causal,
    )
    return (
        dq.transpose(1, 2).contiguous(),
        dk.transpose(1, 2).contiguous(),
        dv.transpose(1, 2).contiguous(),
        dpk.transpose(1, 2).contiguous() if dpk is not None else None,
        dpq.transpose(1, 2).contiguous() if dpq is not None else None,
    )


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
    num_segments: int,
    max_seqlen: int,
    total_tokens: int,
    require_lse: bool,
    aux_capacity: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
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
    :param int num_segments: Host-side active segment count.
    :param int max_seqlen: Host-side maximum segment length.
    :param int total_tokens: Host-side total active token count.
    :param bool require_lse: Whether the caller also needs padded LSE values.
    :param int | None aux_capacity: Optional fixed packed-buffer capacity for
        compile-stable auxiliary outputs.
    :raises RuntimeError: If the low-level varlen kernels are unavailable.
    :return tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        Padded output, optional padded LSE, and packed forward auxiliaries.
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
        num_segments=num_segments,
        max_seqlen=max_seqlen,
        total_tokens=total_tokens,
    )
    att_span = int(position_buckets) if int(position_buckets) > 0 else int(max_relative_distance)
    packed_capacity = max(0, int(total_tokens))
    if aux_capacity is not None:
        packed_capacity = max(packed_capacity, int(aux_capacity))

    if total_tokens == 0:
        output = torch.zeros_like(query_layer)
        q_empty = query_layer.new_empty(
            (packed_capacity, int(query_layer.shape[2]), int(query_layer.shape[3]))
        )
        k_empty = key_layer.new_empty((packed_capacity, int(key_layer.shape[2]), int(key_layer.shape[3])))
        v_empty = value_layer.new_empty(
            (packed_capacity, int(value_layer.shape[2]), int(value_layer.shape[3]))
        )
        lse_empty = torch.empty(
            (packed_capacity, int(query_layer.shape[2])),
            device=query_layer.device,
            dtype=torch.float32,
        )
        pos_key_empty = (
            pos_key.new_empty((packed_capacity, int(pos_key.shape[2]), int(pos_key.shape[3])))
            if pos_key is not None
            else None
        )
        pos_query_empty = (
            pos_query.new_empty((packed_capacity, int(pos_query.shape[2]), int(pos_query.shape[3])))
            if pos_query is not None
            else None
        )
        if require_lse:
            lse = torch.zeros(
                (batch_size, seq_len, int(query_layer.shape[2])),
                device=query_layer.device,
                dtype=torch.float32,
            )
            return output, lse, q_empty, k_empty, v_empty, q_empty, lse_empty, pos_key_empty, pos_query_empty
        return output, None, q_empty, k_empty, v_empty, q_empty, lse_empty, pos_key_empty, pos_query_empty

    q_unpad, k_unpad, v_unpad = segment_pack_padded_rows_triple(
        query_layer,
        key_layer,
        value_layer,
        segment_offsets=active_segment_offsets,
        segment_lengths=active_segment_lengths,
        cu_seqlens=active_cu_seqlens,
        total_tokens=total_tokens,
        max_segment_length=max_seqlen,
    )
    if pos_key is not None and pos_query is not None:
        pos_key_unpad, pos_query_unpad = segment_pack_padded_rows_pair(
            pos_key,
            pos_query,
            segment_offsets=active_segment_offsets,
            segment_lengths=active_segment_lengths,
            cu_seqlens=active_cu_seqlens,
            total_tokens=total_tokens,
            max_segment_length=max_seqlen,
        )
    else:
        pos_key_unpad = (
            segment_pack_padded_rows(
                pos_key,
                segment_offsets=active_segment_offsets,
                segment_lengths=active_segment_lengths,
                cu_seqlens=active_cu_seqlens,
                total_tokens=total_tokens,
                max_segment_length=max_seqlen,
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
                max_segment_length=max_seqlen,
            )
            if pos_query is not None
            else None
        )

    if (
        _varlen_mod._flash_attn_v2_fwd_dise_lowlevel is not None
        and _varlen_mod._get_fwd_config_lowlevel is not None
    ):
        table_config = resolve_flash_kernel_config(
            FlashKernelContext(
                compute_capability=_varlen_mod._varlen_device_capability(query_layer.device),
                route="docblock",
                kind="fwd",
                seq_len=int(max_seqlen),
                total_tokens=int(q_unpad.shape[0]),
                batch_size=int(num_segments),
                head_dim=int(query_layer.shape[-1]),
                dtype=str(query_layer.dtype).removeprefix("torch."),
                causal=bool(causal),
                disentangled=True,
                att_span=int(att_span),
            )
        )
        if table_config is not None:
            block_m, block_n, num_stages, num_warps = table_config
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
        max_segment_length=max_seqlen,
    )
    q_aux = _pad_packed_aux(q_unpad, capacity=packed_capacity)
    k_aux = _pad_packed_aux(k_unpad, capacity=packed_capacity)
    v_aux = _pad_packed_aux(v_unpad, capacity=packed_capacity)
    out_aux = _pad_packed_aux(out_unpad, capacity=packed_capacity)
    lse_aux = _pad_packed_aux(lse_unpad, capacity=packed_capacity)
    pos_key_aux = _pad_packed_aux(pos_key_unpad, capacity=packed_capacity)
    pos_query_aux = _pad_packed_aux(pos_query_unpad, capacity=packed_capacity)
    if not require_lse:
        return (
            out_padded,
            None,
            q_aux,
            k_aux,
            v_aux,
            out_aux,
            lse_aux if lse_aux is not None else torch.empty((0,), device=query_layer.device),
            pos_key_aux,
            pos_query_aux,
        )
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
        max_segment_length=max_seqlen,
    ).contiguous()
    return out_padded, lse_padded, q_aux, k_aux, v_aux, out_aux, lse_aux, pos_key_aux, pos_query_aux


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
    num_segments: int,
    max_seqlen: int,
    total_tokens: int,
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
    :param int num_segments: Host-side active segment count.
    :param int max_seqlen: Host-side maximum segment length.
    :param int total_tokens: Host-side total active token count.
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
        num_segments=num_segments,
        max_seqlen=max_seqlen,
        total_tokens=total_tokens,
    )

    if total_tokens == 0:
        dq = torch.zeros_like(query_layer)
        dk = torch.zeros_like(key_layer)
        dv = torch.zeros_like(value_layer)
        dpos_key = torch.zeros_like(pos_key).contiguous() if pos_key is not None else None
        dpos_query = torch.zeros_like(pos_query).contiguous() if pos_query is not None else None
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
            max_segment_length=max_seqlen,
        )
    q_unpad = _active_packed_prefix(q_unpad, total_tokens=total_tokens)
    k_unpad = _active_packed_prefix(k_unpad, total_tokens=total_tokens)
    v_unpad = _active_packed_prefix(v_unpad, total_tokens=total_tokens)
    if q_unpad is None or k_unpad is None or v_unpad is None:
        raise RuntimeError("Doc-block backward expected packed q/k/v auxiliaries.")
    if out_unpad is None:
        out_unpad = segment_pack_padded_rows(
            output_padded,
            segment_offsets=active_segment_offsets,
            segment_lengths=active_segment_lengths,
            cu_seqlens=active_cu_seqlens,
            total_tokens=total_tokens,
            max_segment_length=max_seqlen,
        )
    out_unpad = _active_packed_prefix(out_unpad, total_tokens=total_tokens)
    if out_unpad is None:
        raise RuntimeError("Doc-block backward expected packed output auxiliary.")
    grad_unpad, delta = segment_pack_grad_and_delta_from_padded(
        grad_output=grad_output,
        out_unpad=out_unpad,
        segment_offsets=active_segment_offsets,
        segment_lengths=active_segment_lengths,
        cu_seqlens=active_cu_seqlens,
        total_tokens=total_tokens,
        max_segment_length=max_seqlen,
    )
    if lse_unpad is None:
        lse_unpad = segment_pack_padded_rows(
            lse_padded,
            segment_offsets=active_segment_offsets,
            segment_lengths=active_segment_lengths,
            cu_seqlens=active_cu_seqlens,
            total_tokens=total_tokens,
            max_segment_length=max_seqlen,
        )
    lse_unpad = _active_packed_prefix(lse_unpad, total_tokens=total_tokens)
    if lse_unpad is None:
        raise RuntimeError("Doc-block backward expected packed LSE auxiliary.")
    if pos_key is not None and pos_query is not None and (pos_key_unpad is None or pos_query_unpad is None):
        pos_key_unpad, pos_query_unpad = segment_pack_padded_rows_pair(
            pos_key,
            pos_query,
            segment_offsets=active_segment_offsets,
            segment_lengths=active_segment_lengths,
            cu_seqlens=active_cu_seqlens,
            total_tokens=total_tokens,
            max_segment_length=max_seqlen,
        )
    else:
        if pos_key is not None and pos_key_unpad is None:
            pos_key_unpad = segment_pack_padded_rows(
                pos_key,
                segment_offsets=active_segment_offsets,
                segment_lengths=active_segment_lengths,
                cu_seqlens=active_cu_seqlens,
                total_tokens=total_tokens,
                max_segment_length=max_seqlen,
            )
        if pos_query is not None and pos_query_unpad is None:
            pos_query_unpad = segment_pack_padded_rows(
                pos_query,
                segment_offsets=active_segment_offsets,
                segment_lengths=active_segment_lengths,
                cu_seqlens=active_cu_seqlens,
                total_tokens=total_tokens,
                max_segment_length=max_seqlen,
            )

    if pos_key is not None:
        pos_key_unpad = _active_packed_prefix(pos_key_unpad, total_tokens=total_tokens)
    if pos_query is not None:
        pos_query_unpad = _active_packed_prefix(pos_query_unpad, total_tokens=total_tokens)

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
        route="docblock",
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
        max_segment_length=max_seqlen,
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
            max_segment_length=max_seqlen,
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
                max_segment_length=max_seqlen,
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
                max_segment_length=max_seqlen,
            )
            if dpos_query_unpad is not None
            else None
        )
    return (
        dq.contiguous(),
        dk.contiguous(),
        dv.contiguous(),
        dpos_key.contiguous() if dpos_key is not None else None,
        dpos_query.contiguous() if dpos_query is not None else None,
    )


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
            "Tensor num_segments, Tensor max_seqlen, Tensor total_tokens, bool causal) -> "
            "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
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
        num_segments: torch.Tensor,
        max_seqlen: torch.Tensor,
        total_tokens: torch.Tensor,
        causal: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
        :param torch.Tensor num_segments: Host-side number of active document segments.
        :param torch.Tensor max_seqlen: Host-side maximum document segment length.
        :param torch.Tensor total_tokens: Host-side active token count.
        :param bool causal: Whether causal masking is enabled.
        :return tuple[torch.Tensor, ...]: Padded output/LSE plus packed forward auxiliaries.
        """

        output, lse, q_unpad, k_unpad, v_unpad, out_unpad, lse_unpad, pos_key_unpad, pos_query_unpad = (
            _docblock_forward_impl(
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
                num_segments=_scalar_int(num_segments, name="num_segments"),
                max_seqlen=_scalar_int(max_seqlen, name="max_seqlen"),
                total_tokens=_scalar_int(total_tokens, name="total_tokens"),
                causal=causal,
                require_lse=True,
                aux_capacity=int(q.shape[0]) * int(q.shape[1]),
            )
        )
        if lse is None:  # pragma: no cover - require_lse=True above
            raise RuntimeError("Doc-block custom op expected padded LSE output.")
        return (
            output,
            lse,
            q_unpad,
            k_unpad,
            v_unpad,
            out_unpad,
            lse_unpad,
            pos_key_unpad if pos_key_unpad is not None else q.new_empty((0,)),
            pos_query_unpad if pos_query_unpad is not None else q.new_empty((0,)),
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
        num_segments: torch.Tensor,
        max_seqlen: torch.Tensor,
        total_tokens: torch.Tensor,
        causal: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
        :param torch.Tensor num_segments: Fake host-side number of active document segments.
        :param torch.Tensor max_seqlen: Fake host-side maximum document segment length.
        :param torch.Tensor total_tokens: Fake host-side active token count.
        :param bool causal: Fake causal flag.
        :return tuple[torch.Tensor, ...]: Fake padded outputs and packed auxiliaries.
        """

        del (
            segment_offsets,
            segment_lengths,
            cu_seqlens,
            sm_scale,
            position_buckets,
            max_relative_distance,
            num_segments,
            max_seqlen,
            total_tokens,
            causal,
        )
        capacity = q.shape[0] * q.shape[1]
        packed_shape = (capacity, q.shape[2], q.shape[3])
        lse = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        lse_unpad = torch.empty((capacity, q.shape[2]), device=q.device, dtype=torch.float32)
        pos_key_aux = (
            torch.empty(
                (capacity, pos_key.shape[2], pos_key.shape[3]), device=pos_key.device, dtype=pos_key.dtype
            )
            if pos_key is not None
            else torch.empty((0,), device=q.device, dtype=q.dtype)
        )
        pos_query_aux = (
            torch.empty(
                (capacity, pos_query.shape[2], pos_query.shape[3]),
                device=pos_query.device,
                dtype=pos_query.dtype,
            )
            if pos_query is not None
            else torch.empty((0,), device=q.device, dtype=q.dtype)
        )
        return (
            torch.empty(q.shape, device=q.device, dtype=q.dtype),
            lse,
            torch.empty(packed_shape, device=q.device, dtype=q.dtype),
            torch.empty(packed_shape, device=k.device, dtype=k.dtype),
            torch.empty(packed_shape, device=v.device, dtype=v.dtype),
            torch.empty(packed_shape, device=q.device, dtype=q.dtype),
            lse_unpad,
            pos_key_aux,
            pos_query_aux,
        )

    @torch.library.custom_op(
        f"{_DOCBLOCK_OP_NAMESPACE}::{_DOCBLOCK_BWD_OP_NAME}",
        mutates_args=(),
        device_types="cuda",
        schema=(
            "(Tensor grad_out, Tensor q, Tensor k, Tensor v, Tensor segment_offsets, Tensor segment_lengths, "
            "Tensor cu_seqlens, Tensor out, Tensor lse, Tensor? pos_key, Tensor? pos_query, float sm_scale, "
            "int position_buckets, int max_relative_distance, Tensor num_segments, Tensor max_seqlen, "
            "Tensor total_tokens, bool causal, Tensor q_unpad, Tensor k_unpad, Tensor v_unpad, Tensor out_unpad, "
            "Tensor lse_unpad, Tensor? pos_key_unpad, Tensor? pos_query_unpad) -> "
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
        num_segments: torch.Tensor,
        max_seqlen: torch.Tensor,
        total_tokens: torch.Tensor,
        causal: bool,
        q_unpad: torch.Tensor,
        k_unpad: torch.Tensor,
        v_unpad: torch.Tensor,
        out_unpad: torch.Tensor,
        lse_unpad: torch.Tensor,
        pos_key_unpad: torch.Tensor | None,
        pos_query_unpad: torch.Tensor | None,
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
        :param torch.Tensor num_segments: Host-side number of active document segments.
        :param torch.Tensor max_seqlen: Host-side maximum document segment length.
        :param torch.Tensor total_tokens: Host-side active token count.
        :param bool causal: Whether causal masking is enabled.
        :param torch.Tensor q_unpad: Packed forward queries.
        :param torch.Tensor k_unpad: Packed forward keys.
        :param torch.Tensor v_unpad: Packed forward values.
        :param torch.Tensor out_unpad: Packed forward output.
        :param torch.Tensor lse_unpad: Packed forward LSE.
        :param torch.Tensor | None pos_key_unpad: Optional packed c2p tensor.
        :param torch.Tensor | None pos_query_unpad: Optional packed p2c tensor.
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
            num_segments=_scalar_int(num_segments, name="num_segments"),
            max_seqlen=_scalar_int(max_seqlen, name="max_seqlen"),
            total_tokens=_scalar_int(total_tokens, name="total_tokens"),
            causal=causal,
            q_unpad=q_unpad,
            k_unpad=k_unpad,
            v_unpad=v_unpad,
            out_unpad=out_unpad,
            lse_unpad=lse_unpad,
            pos_key_unpad=pos_key_unpad,
            pos_query_unpad=pos_query_unpad,
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
        num_segments: torch.Tensor,
        max_seqlen: torch.Tensor,
        total_tokens: torch.Tensor,
        causal: bool,
        q_unpad: torch.Tensor,
        k_unpad: torch.Tensor,
        v_unpad: torch.Tensor,
        out_unpad: torch.Tensor,
        lse_unpad: torch.Tensor,
        pos_key_unpad: torch.Tensor | None,
        pos_query_unpad: torch.Tensor | None,
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
        :param torch.Tensor num_segments: Fake host-side number of active document segments.
        :param torch.Tensor max_seqlen: Fake host-side maximum document segment length.
        :param torch.Tensor total_tokens: Fake host-side active token count.
        :param bool causal: Fake causal flag.
        :param torch.Tensor q_unpad: Fake packed query auxiliary.
        :param torch.Tensor k_unpad: Fake packed key auxiliary.
        :param torch.Tensor v_unpad: Fake packed value auxiliary.
        :param torch.Tensor out_unpad: Fake packed output auxiliary.
        :param torch.Tensor lse_unpad: Fake packed LSE auxiliary.
        :param torch.Tensor | None pos_key_unpad: Fake packed c2p auxiliary.
        :param torch.Tensor | None pos_query_unpad: Fake packed p2c auxiliary.
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
            q_unpad,
            k_unpad,
            v_unpad,
            out_unpad,
            lse_unpad,
            pos_key_unpad,
            pos_query_unpad,
            sm_scale,
            position_buckets,
            max_relative_distance,
            num_segments,
            max_seqlen,
            total_tokens,
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
        output: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
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
            num_segments,
            max_seqlen,
            total_tokens,
            causal,
        ) = inputs
        out, lse, q_unpad, k_unpad, v_unpad, out_unpad, lse_unpad, pos_key_unpad, pos_query_unpad = output
        saved: list[torch.Tensor] = [
            q,
            k,
            v,
            segment_offsets,
            segment_lengths,
            cu_seqlens,
            out,
            lse,
            num_segments,
            max_seqlen,
            total_tokens,
            q_unpad,
            k_unpad,
            v_unpad,
            out_unpad,
            lse_unpad,
        ]
        if pos_key is not None:
            saved.append(pos_key)
            saved.append(pos_key_unpad)
        if pos_query is not None:
            saved.append(pos_query)
            saved.append(pos_query_unpad)
        if hasattr(ctx, "mark_non_differentiable"):
            non_diff = [lse, q_unpad, k_unpad, v_unpad, out_unpad, lse_unpad]
            if pos_key is not None:
                non_diff.append(pos_key_unpad)
            if pos_query is not None:
                non_diff.append(pos_query_unpad)
            ctx.mark_non_differentiable(*non_diff)
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
        grad_q_unpad: torch.Tensor | None,
        grad_k_unpad: torch.Tensor | None,
        grad_v_unpad: torch.Tensor | None,
        grad_out_unpad: torch.Tensor | None,
        grad_lse_unpad: torch.Tensor | None,
        grad_pos_key_unpad: torch.Tensor | None,
        grad_pos_query_unpad: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Dispatch backward through the opaque doc-block backward helper.

        :param Any ctx: Autograd context populated by ``_setup_context``.
        :param torch.Tensor | None grad_out: Gradient of padded output.
        :param torch.Tensor | None grad_lse: Gradient of padded LSE output.
        :param torch.Tensor | None grad_q_unpad: Ignored gradient for packed query auxiliary.
        :param torch.Tensor | None grad_k_unpad: Ignored gradient for packed key auxiliary.
        :param torch.Tensor | None grad_v_unpad: Ignored gradient for packed value auxiliary.
        :param torch.Tensor | None grad_out_unpad: Ignored gradient for packed output auxiliary.
        :param torch.Tensor | None grad_lse_unpad: Ignored gradient for packed LSE auxiliary.
        :param torch.Tensor | None grad_pos_key_unpad: Ignored gradient for packed c2p auxiliary.
        :param torch.Tensor | None grad_pos_query_unpad: Ignored gradient for packed p2c auxiliary.
        :return tuple[torch.Tensor | None, ...]: Gradients for the forward custom-op inputs.
        """

        del (
            grad_lse,
            grad_q_unpad,
            grad_k_unpad,
            grad_v_unpad,
            grad_out_unpad,
            grad_lse_unpad,
            grad_pos_key_unpad,
            grad_pos_query_unpad,
        )
        saved = list(ctx.saved_tensors)
        (
            q,
            k,
            v,
            segment_offsets,
            segment_lengths,
            cu_seqlens,
            out,
            lse,
            num_segments,
            max_seqlen,
            total_tokens,
            q_unpad,
            k_unpad,
            v_unpad,
            out_unpad,
            lse_unpad,
        ) = saved[:16]
        next_idx = 16
        pos_key = saved[next_idx] if bool(ctx.has_pos_key) else None
        if bool(ctx.has_pos_key):
            next_idx += 1
            pos_key_unpad = saved[next_idx]
            next_idx += 1
        else:
            pos_key_unpad = None
        pos_query = saved[next_idx] if bool(ctx.has_pos_query) else None
        if bool(ctx.has_pos_query):
            next_idx += 1
            pos_query_unpad = saved[next_idx]
        else:
            pos_query_unpad = None
        grad = grad_out if grad_out is not None else torch.zeros_like(out)
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
            num_segments,
            max_seqlen,
            total_tokens,
            ctx.causal,
            q_unpad,
            k_unpad,
            v_unpad,
            out_unpad,
            lse_unpad,
            pos_key_unpad,
            pos_query_unpad,
        )
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            dpos_key,
            dpos_query,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

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
    num_segments: int | torch.Tensor,
    max_seqlen: int | torch.Tensor,
    total_tokens: int | torch.Tensor,
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
    :param int | torch.Tensor num_segments: Host-side active segment count.
    :param int | torch.Tensor max_seqlen: Host-side maximum segment length.
    :param int | torch.Tensor total_tokens: Host-side total active token count.
    :param bool causal: Whether causal masking is enabled.
    :return torch.Tensor: Attention output in ``(B, S, H, D)`` layout.
    """

    if _FLASHDEBERTA_DOCBLOCK_CUSTOM_OP is not None and query_layer.device.type == "cuda":
        num_segments_tensor = _scalar_tensor(num_segments, name="num_segments")
        max_seqlen_tensor = _scalar_tensor(max_seqlen, name="max_seqlen")
        total_tokens_tensor = _scalar_tensor(total_tokens, name="total_tokens")
        output, *_ = _FLASHDEBERTA_DOCBLOCK_CUSTOM_OP(
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
            num_segments_tensor,
            max_seqlen_tensor,
            total_tokens_tensor,
            bool(causal),
        )
        return output

    output, *_ = _docblock_forward_impl(
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
        num_segments=_scalar_int(num_segments, name="num_segments"),
        max_seqlen=_scalar_int(max_seqlen, name="max_seqlen"),
        total_tokens=_scalar_int(total_tokens, name="total_tokens"),
        causal=causal,
        require_lse=False,
    )
    return output


__all__ = [
    "flashdeberta_compiled_docblock_available",
    "flashdeberta_docblock",
    "flashdeberta_docblock_import_error",
]
