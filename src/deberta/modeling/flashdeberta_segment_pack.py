"""Contiguous-segment pack/unpack helpers for doc-block-aware flash attention.

Packed doc-block batches contain multiple contiguous document segments inside one
``(B, S, ...)`` tensor. Flash attention cannot consume the original block-diagonal
pairwise mask directly, but it can consume the same tokens once they are repacked
into a ragged batch of per-document segments.

This module provides small row-major Triton copy kernels, plus eager fallbacks,
for packing contiguous token ranges out of padded ``(B, S, ...)`` tensors and for
scattering packed outputs/gradients back to the original padded layout.

The public one/two/three-tensor wrappers remain explicit so their return types
and compile-visible arity stay stable. Their CUDA copy implementation is shared
through ``launch_triton_row_copy`` rather than duplicated per wrapper.
"""

from __future__ import annotations

from functools import partial

import torch

from deberta.modeling.flashdeberta_op_utils import (
    TRITON_ROW_COPY_SEGMENT_PACK,
    TRITON_ROW_COPY_SEGMENT_PACK_STRIDED,
    TRITON_ROW_COPY_SEGMENT_UNPACK,
    can_use_triton_pack,
    flatten_padded_rows,
    launch_triton_row_copy,
    require_matching_tensor_layout,
)
from deberta.modeling.flashdeberta_op_utils import (
    optional_triton_jit as _optional_triton_jit,
)

try:  # pragma: no cover - optional Triton dependency
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional Triton dependency
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

_SEGMENT_BLOCK_ROWS = 32
_SEGMENT_BLOCK_COLS = 128
_SEGMENT_GRAD_DELTA_BLOCK_ROWS = 32
_SEGMENT_NUM_WARPS = 4
_SEGMENT_NUM_STAGES = 2


def flashdeberta_segment_pack_available() -> bool:
    """Return whether Triton-backed segment pack kernels are importable.

    :return bool: True when Triton imported successfully.
    """

    return _TRITON_AVAILABLE


def _tensor_host_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
    """Return a host tuple for one CPU metadata tensor.

    :param torch.Tensor tensor: CPU metadata tensor.
    :raises RuntimeError: If a device tensor reaches the eager fallback path.
    :return tuple[int, ...]: Host integer values.
    """

    if tensor.device.type != "cpu":
        raise RuntimeError(
            "Segment-pack eager fallback requires CPU metadata; device metadata must use Triton."
        )
    return tuple(int(value) for value in tensor.tolist())


_flatten_rows = partial(flatten_padded_rows, context="Segment-pack")


def _can_use_triton_segment_pack(
    *,
    tensor: torch.Tensor,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> bool:
    """Return whether the Triton segment-pack path can be used.

    :param torch.Tensor tensor: Input or output tensor.
    :param torch.Tensor segment_offsets: Flat row offsets per segment.
    :param torch.Tensor segment_lengths: Segment lengths.
    :param torch.Tensor cu_seqlens: Packed cumulative sequence lengths.
    :return bool: True when the Triton path is usable.
    """

    return can_use_triton_pack(
        available=flashdeberta_segment_pack_available(),
        tensor=tensor,
        metadata_tensors=(segment_offsets, segment_lengths, cu_seqlens),
    )


def _can_use_triton_segment_rank4_pack(
    tensors: tuple[torch.Tensor, ...],
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> bool:
    """Return whether rank-4 inputs can use the stride-aware Triton copy mode.

    :param tuple[torch.Tensor, ...] tensors: Candidate padded rank-4 tensors.
    :param torch.Tensor segment_offsets: Flat padded-row offsets per segment.
    :param torch.Tensor segment_lengths: Active row count per segment.
    :param torch.Tensor cu_seqlens: Cumulative packed-row offsets.
    :return bool: True when the inputs and metadata share one CUDA device.
    """

    if not flashdeberta_segment_pack_available() or not tensors:
        return False
    device = tensors[0].device
    return (
        device.type == "cuda"
        and all(tensor.device == device and tensor.ndim == 4 for tensor in tensors)
        and segment_offsets.device == device
        and segment_lengths.device == device
        and cu_seqlens.device == device
    )


@_optional_triton_jit
def _pack_segment_grad_and_delta_kernel(
    grad_padded_ptr: None,
    out_unpad_ptr: None,
    grad_unpad_ptr: None,
    delta_ptr: None,
    offsets_ptr: None,
    lengths_ptr: None,
    cu_seqlens_ptr: None,
    stride_gb: None,
    stride_gs: None,
    stride_gh: None,
    stride_gd: None,
    stride_oz: None,
    stride_oh: None,
    stride_od: None,
    stride_guz: None,
    stride_guh: None,
    stride_gud: None,
    stride_dz: None,
    stride_dh: None,
    seq_len: None,
    head_dim: None,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
) -> None:
    """Pack segmented output gradients and compute packed delta in one pass.

    :param Any grad_padded_ptr: Pointer to padded output gradients.
    :param Any out_unpad_ptr: Pointer to packed forward outputs.
    :param Any grad_unpad_ptr: Pointer to packed output gradients.
    :param Any delta_ptr: Pointer to packed delta tensor.
    :param Any offsets_ptr: Pointer to flat padded row offsets per segment.
    :param Any lengths_ptr: Pointer to per-segment lengths.
    :param Any cu_seqlens_ptr: Pointer to cumulative packed offsets.
    :param Any stride_gb: Batch stride for padded gradients.
    :param Any stride_gs: Sequence stride for padded gradients.
    :param Any stride_gh: Head stride for padded gradients.
    :param Any stride_gd: Feature stride for padded gradients.
    :param Any stride_oz: Token stride for packed forward outputs.
    :param Any stride_oh: Head stride for packed forward outputs.
    :param Any stride_od: Feature stride for packed forward outputs.
    :param Any stride_guz: Token stride for packed output gradients.
    :param Any stride_guh: Head stride for packed output gradients.
    :param Any stride_gud: Feature stride for packed output gradients.
    :param Any stride_dz: Token stride for packed delta.
    :param Any stride_dh: Head stride for packed delta.
    :param Any seq_len: Padded sequence length.
    :param Any head_dim: Runtime head dimension.
    :param Any BLOCK_ROWS: Row tile size.
    :param Any BLOCK_DMODEL: Feature tile size.
    :return None: This Triton kernel writes in place.
    """

    segment_idx = tl.program_id(0)
    tile_row = tl.program_id(1)
    head_idx = tl.program_id(2)

    row_offsets = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    d_offsets = tl.arange(0, BLOCK_DMODEL)

    src_base = tl.load(offsets_ptr + segment_idx)
    seg_len = tl.load(lengths_ptr + segment_idx)
    dst_base = tl.load(cu_seqlens_ptr + segment_idx)

    src_flat_rows = src_base + row_offsets
    src_batch = src_flat_rows // seq_len
    src_seq = src_flat_rows - src_batch * seq_len
    packed_rows = dst_base + row_offsets

    mask_rows = row_offsets < seg_len
    mask_d = d_offsets < head_dim
    mask = mask_rows[:, None] & mask_d[None, :]

    grad_ptrs = grad_padded_ptr + (
        src_batch[:, None] * stride_gb
        + src_seq[:, None] * stride_gs
        + head_idx * stride_gh
        + d_offsets[None, :] * stride_gd
    )
    out_ptrs = out_unpad_ptr + (
        packed_rows[:, None] * stride_oz + head_idx * stride_oh + d_offsets[None, :] * stride_od
    )
    grad_unpad_ptrs = grad_unpad_ptr + (
        packed_rows[:, None] * stride_guz + head_idx * stride_guh + d_offsets[None, :] * stride_gud
    )

    grad = tl.load(grad_ptrs, mask=mask, other=0.0)
    out = tl.load(out_ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(grad_unpad_ptrs, grad, mask=mask)

    delta = tl.sum(out * grad.to(tl.float32), axis=1)
    delta_ptrs = delta_ptr + packed_rows * stride_dz + head_idx * stride_dh
    tl.store(delta_ptrs, delta, mask=mask_rows)


def _segment_pack_padded_rows(
    tensors: tuple[torch.Tensor, ...],
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    max_segment_length: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Pack one or more padded tensors with shared segment metadata.

    :param tuple[torch.Tensor, ...] tensors: Input tensors with shape ``(B, S, ...)``.
    :param torch.Tensor segment_offsets: Flat ``(B*S)`` source row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param int total_tokens: Total packed token count.
    :param int | None max_segment_length: Host-side maximum segment length.
    :return tuple[torch.Tensor, ...]: Packed tensors with shape ``(NNZ, ...)``.
    """

    if not tensors:
        raise ValueError("Segment-pack requires at least one input tensor")
    if len(tensors) == 1:
        shape = tuple(int(dim) for dim in tensors[0].shape)
        if len(shape) < 2:
            raise ValueError(f"Segment-pack tensor must have rank >= 2, got shape {shape}")
    else:
        shape = require_matching_tensor_layout(
            *tensors,
            context="Segment-pack padded tensors",
            minimum_rank=2,
        )
    trailing_shape = shape[2:]
    total = max(0, int(total_tokens))
    outputs = tuple(tensor.new_empty((total,) + trailing_shape) for tensor in tensors)
    if total == 0:
        return outputs
    output_flats = tuple(output.view(total, -1) for output in outputs)
    row_size = int(output_flats[0].shape[1])
    max_len = max(1, int(max_segment_length if max_segment_length is not None else total))

    if _can_use_triton_segment_rank4_pack(
        tensors,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
    ) and any(not tensor.is_contiguous() for tensor in tensors):
        num_heads = int(shape[2])
        feature_size = int(shape[3])
        launch_triton_row_copy(
            tensors,
            output_flats,
            offsets=segment_offsets,
            lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
            address_mode=TRITON_ROW_COPY_SEGMENT_PACK_STRIDED,
            seq_len=int(shape[1]),
            row_size=row_size,
            max_rows=max_len,
            num_heads=num_heads,
            feature_size=feature_size,
            block_rows=_SEGMENT_BLOCK_ROWS,
            block_cols=_SEGMENT_BLOCK_COLS,
            num_warps=_SEGMENT_NUM_WARPS,
            num_stages=_SEGMENT_NUM_STAGES,
        )
        return outputs

    contiguous_tensors = tuple(
        tensor if tensor.is_contiguous() else tensor.contiguous() for tensor in tensors
    )
    flattened = tuple(_flatten_rows(tensor)[0] for tensor in contiguous_tensors)
    seq_len = int(shape[1])

    if _can_use_triton_segment_pack(
        tensor=contiguous_tensors[0],
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
    ):
        launch_triton_row_copy(
            flattened,
            output_flats,
            offsets=segment_offsets,
            lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
            address_mode=TRITON_ROW_COPY_SEGMENT_PACK,
            seq_len=seq_len,
            row_size=row_size,
            max_rows=max_len,
            block_rows=_SEGMENT_BLOCK_ROWS,
            block_cols=_SEGMENT_BLOCK_COLS,
            num_warps=_SEGMENT_NUM_WARPS,
            num_stages=_SEGMENT_NUM_STAGES,
        )
        return outputs

    offsets_host = _tensor_host_tuple(segment_offsets)
    lengths_host = _tensor_host_tuple(segment_lengths)
    cu_host = _tensor_host_tuple(cu_seqlens)
    for idx, src_base in enumerate(offsets_host):
        seg_len = int(lengths_host[idx])
        if seg_len <= 0:
            continue
        dst_base = int(cu_host[idx])
        src_slice = slice(src_base, src_base + seg_len)
        dst_slice = slice(dst_base, dst_base + seg_len)
        for output_flat, flat in zip(output_flats, flattened, strict=True):
            output_flat[dst_slice].copy_(flat[src_slice])
    return outputs


def segment_pack_padded_rows(
    tensor: torch.Tensor,
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    max_segment_length: int | None = None,
) -> torch.Tensor:
    """Pack contiguous padded token segments into one packed tensor.

    :param torch.Tensor tensor: Input tensor with shape ``(B, S, ...)``.
    :param torch.Tensor segment_offsets: Flat ``(B*S)`` source row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param int total_tokens: Total packed token count.
    :param int | None max_segment_length: Host-side maximum segment length.
    :return torch.Tensor: Packed tensor with shape ``(NNZ, ...)``.
    """

    return _segment_pack_padded_rows(
        (tensor,),
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        total_tokens=total_tokens,
        max_segment_length=max_segment_length,
    )[0]


def segment_pack_padded_rows_pair(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    max_segment_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack two contiguous padded tensors with shared segment metadata.

    :param torch.Tensor tensor_a: First input tensor ``(B, S, ...)``.
    :param torch.Tensor tensor_b: Second input tensor ``(B, S, ...)``.
    :param torch.Tensor segment_offsets: Flat ``(B*S)`` source row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param int total_tokens: Total packed token count.
    :param int | None max_segment_length: Host-side maximum segment length.
    :return tuple[torch.Tensor, torch.Tensor]: Packed tensors ``(NNZ, ...)``.
    """

    out_a, out_b = _segment_pack_padded_rows(
        (tensor_a, tensor_b),
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        total_tokens=total_tokens,
        max_segment_length=max_segment_length,
    )
    return out_a, out_b


def segment_pack_padded_rows_triple(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_c: torch.Tensor,
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    max_segment_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack three contiguous padded tensors with shared segment metadata.

    :param torch.Tensor tensor_a: First input tensor ``(B, S, ...)``.
    :param torch.Tensor tensor_b: Second input tensor ``(B, S, ...)``.
    :param torch.Tensor tensor_c: Third input tensor ``(B, S, ...)``.
    :param torch.Tensor segment_offsets: Flat ``(B*S)`` source row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param int total_tokens: Total packed token count.
    :param int | None max_segment_length: Host-side maximum segment length.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Packed tensors ``(NNZ, ...)``.
    """

    out_a, out_b, out_c = _segment_pack_padded_rows(
        (tensor_a, tensor_b, tensor_c),
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        total_tokens=total_tokens,
        max_segment_length=max_segment_length,
    )
    return out_a, out_b, out_c


def _segment_delta_fallback(*, out_unpad: torch.Tensor, grad_unpad: torch.Tensor) -> torch.Tensor:
    """Return packed ``sum(out * grad, dim=-1)`` in fp32.

    :param torch.Tensor out_unpad: Packed forward output tensor.
    :param torch.Tensor grad_unpad: Packed output gradient tensor.
    :return torch.Tensor: Packed delta tensor in ``(NNZ, H)`` layout.
    """

    return (out_unpad.to(dtype=torch.float32) * grad_unpad.to(dtype=torch.float32)).sum(dim=-1)


def segment_pack_grad_and_delta_from_padded(
    *,
    grad_output: torch.Tensor,
    out_unpad: torch.Tensor,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    max_segment_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack segmented output gradients and build packed delta.

    :param torch.Tensor grad_output: Padded output gradient in ``(B,S,H,D)`` layout.
    :param torch.Tensor out_unpad: Packed forward output in ``(NNZ,H,D)`` layout.
    :param torch.Tensor segment_offsets: Flat padded row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param int total_tokens: Active packed-token count.
    :param int | None max_segment_length: Host-side maximum segment length.
    :return tuple[torch.Tensor, torch.Tensor]: Packed output gradient and packed delta.
    """

    total = max(0, int(total_tokens))
    grad_unpad = grad_output.new_empty((total,) + tuple(grad_output.shape[2:]))
    if total == 0:
        delta = torch.empty((0, int(grad_output.shape[2])), device=grad_output.device, dtype=torch.float32)
        return grad_unpad, delta

    can_use_triton = (
        _can_use_triton_segment_pack(
            tensor=grad_output,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
        )
        and out_unpad.device == grad_output.device
    )
    if not can_use_triton:
        grad_unpad = segment_pack_padded_rows(
            grad_output,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
            total_tokens=total,
            max_segment_length=max_segment_length,
        )
        return grad_unpad, _segment_delta_fallback(out_unpad=out_unpad, grad_unpad=grad_unpad)

    head_dim = int(grad_output.shape[-1])
    if head_dim > 256:
        grad_unpad = segment_pack_padded_rows(
            grad_output,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
            total_tokens=total,
            max_segment_length=max_segment_length,
        )
        return grad_unpad, _segment_delta_fallback(out_unpad=out_unpad, grad_unpad=grad_unpad)
    block_dmodel = max(16, triton.next_power_of_2(head_dim))
    delta = torch.empty((total, int(grad_output.shape[2])), device=grad_output.device, dtype=torch.float32)
    max_len = max(1, int(max_segment_length if max_segment_length is not None else total))
    grid = (
        int(segment_lengths.shape[0]),
        triton.cdiv(max_len, _SEGMENT_GRAD_DELTA_BLOCK_ROWS),
        int(grad_output.shape[2]),
    )
    torch.library.wrap_triton(_pack_segment_grad_and_delta_kernel)[grid](
        grad_output,
        out_unpad,
        grad_unpad,
        delta,
        segment_offsets,
        segment_lengths,
        cu_seqlens,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        out_unpad.stride(0),
        out_unpad.stride(1),
        out_unpad.stride(2),
        grad_unpad.stride(0),
        grad_unpad.stride(1),
        grad_unpad.stride(2),
        delta.stride(0),
        delta.stride(1),
        int(grad_output.shape[1]),
        head_dim,
        BLOCK_ROWS=_SEGMENT_GRAD_DELTA_BLOCK_ROWS,
        BLOCK_DMODEL=block_dmodel,
        num_warps=_SEGMENT_NUM_WARPS,
        num_stages=_SEGMENT_NUM_STAGES,
    )
    return grad_unpad, delta


def _segment_unpack_padded_rows(
    packed_tensors: tuple[torch.Tensor, ...],
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    max_segment_length: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Scatter one or more packed tensors with shared segment metadata.

    :param tuple[torch.Tensor, ...] packed_tensors: Packed tensors with shape ``(NNZ, ...)``.
    :param torch.Tensor segment_offsets: Flat padded row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param int batch_size: Output batch size.
    :param int seq_len: Output padded sequence length.
    :param int | None max_segment_length: Host-side maximum segment length.
    :return tuple[torch.Tensor, ...]: Padded tensors with shape ``(B, S, ...)``.
    """

    if not packed_tensors:
        raise ValueError("Segment-unpack requires at least one packed tensor")
    if len(packed_tensors) == 1:
        shape = tuple(int(dim) for dim in packed_tensors[0].shape)
        if not shape:
            raise ValueError("Segment-pack packed tensor must have rank >= 1")
    else:
        shape = require_matching_tensor_layout(
            *packed_tensors,
            context="Segment-pack packed tensors",
        )
    total = int(shape[0])
    trailing_shape = shape[1:]
    outputs = tuple(
        packed.new_zeros((int(batch_size), int(seq_len)) + trailing_shape) for packed in packed_tensors
    )
    if total == 0:
        return outputs
    flattened = tuple(packed.contiguous().view(total, -1) for packed in packed_tensors)
    output_flats = tuple(output.view(int(batch_size) * int(seq_len), -1) for output in outputs)
    row_size = int(flattened[0].shape[1])

    if _can_use_triton_segment_pack(
        tensor=outputs[0],
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
    ):
        max_len = max(1, int(max_segment_length if max_segment_length is not None else int(seq_len)))
        launch_triton_row_copy(
            flattened,
            output_flats,
            offsets=segment_offsets,
            lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
            address_mode=TRITON_ROW_COPY_SEGMENT_UNPACK,
            seq_len=int(seq_len),
            row_size=row_size,
            max_rows=max_len,
            block_rows=_SEGMENT_BLOCK_ROWS,
            block_cols=_SEGMENT_BLOCK_COLS,
            num_warps=_SEGMENT_NUM_WARPS,
            num_stages=_SEGMENT_NUM_STAGES,
        )
        return outputs

    offsets_host = _tensor_host_tuple(segment_offsets)
    lengths_host = _tensor_host_tuple(segment_lengths)
    cu_host = _tensor_host_tuple(cu_seqlens)
    for idx, dst_base in enumerate(offsets_host):
        seg_len = int(lengths_host[idx])
        if seg_len <= 0:
            continue
        src_base = int(cu_host[idx])
        src_slice = slice(src_base, src_base + seg_len)
        dst_slice = slice(dst_base, dst_base + seg_len)
        for output_flat, flat in zip(output_flats, flattened, strict=True):
            output_flat[dst_slice].copy_(flat[src_slice])
    return outputs


def segment_unpack_padded_rows(
    packed: torch.Tensor,
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    max_segment_length: int | None = None,
) -> torch.Tensor:
    """Scatter one packed tensor back into padded ``(B, S, ...)`` layout.

    :param torch.Tensor packed: Packed tensor with shape ``(NNZ, ...)``.
    :param torch.Tensor segment_offsets: Flat padded row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param int batch_size: Output batch size.
    :param int seq_len: Output padded sequence length.
    :param int | None max_segment_length: Host-side maximum segment length.
    :return torch.Tensor: Padded tensor with shape ``(B, S, ...)``.
    """

    return _segment_unpack_padded_rows(
        (packed,),
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
        max_segment_length=max_segment_length,
    )[0]


def segment_unpack_padded_rows_pair(
    packed_a: torch.Tensor,
    packed_b: torch.Tensor,
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    max_segment_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter two packed tensors back into padded ``(B, S, ...)`` layout.

    :param torch.Tensor packed_a: First packed tensor ``(NNZ, ...)``.
    :param torch.Tensor packed_b: Second packed tensor ``(NNZ, ...)``.
    :param torch.Tensor segment_offsets: Flat padded row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param int batch_size: Output batch size.
    :param int seq_len: Output padded sequence length.
    :param int | None max_segment_length: Host-side maximum segment length.
    :return tuple[torch.Tensor, torch.Tensor]: Padded tensors ``(B, S, ...)``.
    """

    output_a, output_b = _segment_unpack_padded_rows(
        (packed_a, packed_b),
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
        max_segment_length=max_segment_length,
    )
    return output_a, output_b


def segment_unpack_padded_rows_triple(
    packed_a: torch.Tensor,
    packed_b: torch.Tensor,
    packed_c: torch.Tensor,
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    max_segment_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter three packed tensors back into padded ``(B, S, ...)`` layout.

    :param torch.Tensor packed_a: First packed tensor ``(NNZ, ...)``.
    :param torch.Tensor packed_b: Second packed tensor ``(NNZ, ...)``.
    :param torch.Tensor packed_c: Third packed tensor ``(NNZ, ...)``.
    :param torch.Tensor segment_offsets: Flat padded row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param int batch_size: Output batch size.
    :param int seq_len: Output padded sequence length.
    :param int | None max_segment_length: Host-side maximum segment length.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Padded tensors ``(B, S, ...)``.
    """

    output_a, output_b, output_c = _segment_unpack_padded_rows(
        (packed_a, packed_b, packed_c),
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
        max_segment_length=max_segment_length,
    )
    return output_a, output_b, output_c


def segment_pack_optional_pair(
    tensor_a: torch.Tensor | None,
    tensor_b: torch.Tensor | None,
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    max_segment_length: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Pack an optional tensor pair, sharing one launch when both are present.

    :param torch.Tensor | None tensor_a: Optional first padded tensor.
    :param torch.Tensor | None tensor_b: Optional second padded tensor.
    :param torch.Tensor segment_offsets: Flat row offsets per segment.
    :param torch.Tensor segment_lengths: Segment lengths.
    :param torch.Tensor cu_seqlens: Packed cumulative sequence lengths.
    :param int total_tokens: Total packed token count.
    :param int max_segment_length: Maximum segment length.
    :return tuple[torch.Tensor | None, torch.Tensor | None]: Packed tensors, or
        ``None`` for absent inputs.
    """

    if tensor_a is not None and tensor_b is not None:
        return segment_pack_padded_rows_pair(
            tensor_a,
            tensor_b,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
            total_tokens=total_tokens,
            max_segment_length=max_segment_length,
        )
    packed = [
        segment_pack_padded_rows(
            tensor,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
            total_tokens=total_tokens,
            max_segment_length=max_segment_length,
        )
        if tensor is not None
        else None
        for tensor in (tensor_a, tensor_b)
    ]
    return packed[0], packed[1]


def segment_unpack_optional_pair(
    tensor_a: torch.Tensor | None,
    tensor_b: torch.Tensor | None,
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    max_segment_length: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Unpack an optional tensor pair, sharing one launch when both are present.

    :param torch.Tensor | None tensor_a: Optional first packed tensor.
    :param torch.Tensor | None tensor_b: Optional second packed tensor.
    :param torch.Tensor segment_offsets: Flat row offsets per segment.
    :param torch.Tensor segment_lengths: Segment lengths.
    :param torch.Tensor cu_seqlens: Packed cumulative sequence lengths.
    :param int batch_size: Padded batch size.
    :param int seq_len: Padded sequence length.
    :param int max_segment_length: Maximum segment length.
    :return tuple[torch.Tensor | None, torch.Tensor | None]: Padded tensors, or
        ``None`` for absent inputs.
    """

    if tensor_a is not None and tensor_b is not None:
        return segment_unpack_padded_rows_pair(
            tensor_a,
            tensor_b,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
            batch_size=batch_size,
            seq_len=seq_len,
            max_segment_length=max_segment_length,
        )
    unpacked = [
        segment_unpack_padded_rows(
            tensor,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
            batch_size=batch_size,
            seq_len=seq_len,
            max_segment_length=max_segment_length,
        )
        if tensor is not None
        else None
        for tensor in (tensor_a, tensor_b)
    ]
    return unpacked[0], unpacked[1]


__all__ = [
    "flashdeberta_segment_pack_available",
    "segment_pack_grad_and_delta_from_padded",
    "segment_pack_optional_pair",
    "segment_pack_padded_rows",
    "segment_pack_padded_rows_pair",
    "segment_pack_padded_rows_triple",
    "segment_unpack_optional_pair",
    "segment_unpack_padded_rows",
    "segment_unpack_padded_rows_pair",
    "segment_unpack_padded_rows_triple",
]
