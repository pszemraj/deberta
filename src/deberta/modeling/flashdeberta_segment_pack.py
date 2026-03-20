"""Contiguous-segment pack/unpack helpers for doc-block-aware flash attention.

Packed doc-block batches contain multiple contiguous document segments inside one
``(B, S, ...)`` tensor. Flash attention cannot consume the original block-diagonal
pairwise mask directly, but it can consume the same tokens once they are repacked
into a ragged batch of per-document segments.

This module provides small row-major Triton copy kernels, plus eager fallbacks,
for packing contiguous token ranges out of padded ``(B, S, ...)`` tensors and for
scattering packed outputs/gradients back to the original padded layout.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch

try:  # pragma: no cover - optional Triton dependency
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional Triton dependency
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc

_SEGMENT_BLOCK_ROWS = 32
_SEGMENT_BLOCK_COLS = 128
_SEGMENT_NUM_WARPS = 4
_SEGMENT_NUM_STAGES = 2


@dataclass
class _SegmentHostCacheEntry:
    """Cached host tuple for one device metadata tensor."""

    tensor_ref: weakref.ReferenceType[torch.Tensor] | None
    host_values: tuple[int, ...]


_SEGMENT_HOST_CACHE: dict[int, _SegmentHostCacheEntry] = {}


def flashdeberta_segment_pack_import_error() -> Exception | None:
    """Return the Triton import failure for segment-pack kernels, if any.

    :return Exception | None: Stored Triton import failure or ``None``.
    """

    return _TRITON_IMPORT_ERROR


def flashdeberta_segment_pack_available() -> bool:
    """Return whether Triton-backed segment pack kernels are importable.

    :return bool: True when Triton imported successfully.
    """

    return triton is not None and tl is not None


def _traceable_triton_kernel(kernel: object) -> object:
    """Return a traceable Triton kernel wrapper when PyTorch exposes one.

    :param object kernel: Raw Triton kernel or autotuned wrapper.
    :return object: Traceable wrapper when available, otherwise ``kernel``.
    """

    if not hasattr(torch, "library") or not hasattr(torch.library, "wrap_triton"):
        return kernel
    try:
        return torch.library.wrap_triton(kernel)
    except Exception:
        return kernel


def _tensor_host_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
    """Return a cached host tuple for one 1D metadata tensor.

    :param torch.Tensor tensor: Device metadata tensor.
    :return tuple[int, ...]: Cached host integer values.
    """

    cache_key = id(tensor)
    cached = _SEGMENT_HOST_CACHE.get(cache_key)
    if cached is not None:
        cached_tensor = cached.tensor_ref() if cached.tensor_ref is not None else None
        if cached_tensor is tensor:
            return cached.host_values
        _SEGMENT_HOST_CACHE.pop(cache_key, None)

    host_values = tuple(int(value) for value in tensor.detach().cpu().tolist())
    tensor_ref: weakref.ReferenceType[torch.Tensor] | None = None
    try:
        tensor_ref = weakref.ref(tensor, lambda _ref, key=cache_key: _SEGMENT_HOST_CACHE.pop(key, None))
    except TypeError:
        tensor_ref = None
    _SEGMENT_HOST_CACHE[cache_key] = _SegmentHostCacheEntry(tensor_ref=tensor_ref, host_values=host_values)
    return host_values


def clear_segment_pack_host_cache() -> None:
    """Clear cached host metadata tuples.

    :return None: This exists primarily for tests.
    """

    _SEGMENT_HOST_CACHE.clear()


def _flatten_rows(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], int, int]:
    """Flatten a contiguous ``(B, S, ...)`` tensor into row-major ``(B*S, F)`` form.

    :param torch.Tensor tensor: Tensor with at least two leading dims ``(B, S)``.
    :raises ValueError: If the tensor rank is less than two.
    :return tuple[torch.Tensor, tuple[int, ...], int, int]:
        Flattened tensor, trailing shape, batch size, and padded sequence length.
    """

    if tensor.ndim < 2:
        raise ValueError(f"Expected tensor with leading (B,S) dims; got shape={tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError("Segment-pack Triton path requires contiguous tensors.")
    batch_size = int(tensor.shape[0])
    seq_len = int(tensor.shape[1])
    trailing_shape = tuple(int(dim) for dim in tensor.shape[2:])
    flat = tensor.view(batch_size * seq_len, -1)
    return flat, trailing_shape, batch_size, seq_len


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

    if not flashdeberta_segment_pack_available():
        return False
    if tensor.device.type != "cuda":
        return False
    if not tensor.is_contiguous():
        return False
    if (
        segment_offsets.device != tensor.device
        or segment_lengths.device != tensor.device
        or cu_seqlens.device != tensor.device
    ):
        return False
    return True


@triton.jit
def _pack_segment_rows_kernel(
    input_ptr: None,
    output_ptr: None,
    offsets_ptr: None,
    lengths_ptr: None,
    cu_seqlens_ptr: None,
    row_size: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
) -> None:
    """Copy contiguous padded segment rows into packed row-major output.

    :param Any input_ptr: Pointer to padded input rows.
    :param Any output_ptr: Pointer to packed output rows.
    :param Any offsets_ptr: Pointer to flat source row offsets per segment.
    :param Any lengths_ptr: Pointer to per-segment lengths.
    :param Any cu_seqlens_ptr: Pointer to cumulative packed offsets.
    :param Any row_size: Flattened feature width.
    :param Any BLOCK_ROWS: Triton row tile size.
    :param Any BLOCK_COLS: Triton feature tile size.
    :return None: This Triton kernel writes in place.
    """

    segment_idx = tl.program_id(0)
    tile_row = tl.program_id(1)
    tile_col = tl.program_id(2)

    row_offsets = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = tile_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    src_base = tl.load(offsets_ptr + segment_idx)
    seg_len = tl.load(lengths_ptr + segment_idx)
    dst_base = tl.load(cu_seqlens_ptr + segment_idx)

    src_rows = src_base + row_offsets
    dst_rows = dst_base + row_offsets

    mask = (row_offsets[:, None] < seg_len) & (col_offsets[None, :] < row_size)
    src_ptrs = input_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    dst_ptrs = output_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    values = tl.load(src_ptrs, mask=mask, other=0)
    tl.store(dst_ptrs, values, mask=mask)


@triton.jit
def _pack_segment_rows_pair_kernel(
    input_a_ptr: None,
    input_b_ptr: None,
    output_a_ptr: None,
    output_b_ptr: None,
    offsets_ptr: None,
    lengths_ptr: None,
    cu_seqlens_ptr: None,
    row_size: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
) -> None:
    """Copy two contiguous padded segment tensors into packed outputs.

    :param Any input_a_ptr: Pointer to the first padded input rows.
    :param Any input_b_ptr: Pointer to the second padded input rows.
    :param Any output_a_ptr: Pointer to the first packed output rows.
    :param Any output_b_ptr: Pointer to the second packed output rows.
    :param Any offsets_ptr: Pointer to flat source row offsets per segment.
    :param Any lengths_ptr: Pointer to per-segment lengths.
    :param Any cu_seqlens_ptr: Pointer to cumulative packed offsets.
    :param Any row_size: Flattened feature width.
    :param Any BLOCK_ROWS: Triton row tile size.
    :param Any BLOCK_COLS: Triton feature tile size.
    :return None: This Triton kernel writes in place.
    """

    segment_idx = tl.program_id(0)
    tile_row = tl.program_id(1)
    tile_col = tl.program_id(2)

    row_offsets = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = tile_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    src_base = tl.load(offsets_ptr + segment_idx)
    seg_len = tl.load(lengths_ptr + segment_idx)
    dst_base = tl.load(cu_seqlens_ptr + segment_idx)

    src_rows = src_base + row_offsets
    dst_rows = dst_base + row_offsets

    mask = (row_offsets[:, None] < seg_len) & (col_offsets[None, :] < row_size)
    src_a_ptrs = input_a_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    src_b_ptrs = input_b_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    dst_a_ptrs = output_a_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    dst_b_ptrs = output_b_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    values_a = tl.load(src_a_ptrs, mask=mask, other=0)
    values_b = tl.load(src_b_ptrs, mask=mask, other=0)
    tl.store(dst_a_ptrs, values_a, mask=mask)
    tl.store(dst_b_ptrs, values_b, mask=mask)


@triton.jit
def _pack_segment_rows_triple_kernel(
    input_a_ptr: None,
    input_b_ptr: None,
    input_c_ptr: None,
    output_a_ptr: None,
    output_b_ptr: None,
    output_c_ptr: None,
    offsets_ptr: None,
    lengths_ptr: None,
    cu_seqlens_ptr: None,
    row_size: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
) -> None:
    """Copy three contiguous padded segment tensors into packed outputs.

    :param Any input_a_ptr: Pointer to the first padded input rows.
    :param Any input_b_ptr: Pointer to the second padded input rows.
    :param Any input_c_ptr: Pointer to the third padded input rows.
    :param Any output_a_ptr: Pointer to the first packed output rows.
    :param Any output_b_ptr: Pointer to the second packed output rows.
    :param Any output_c_ptr: Pointer to the third packed output rows.
    :param Any offsets_ptr: Pointer to flat source row offsets per segment.
    :param Any lengths_ptr: Pointer to per-segment lengths.
    :param Any cu_seqlens_ptr: Pointer to cumulative packed offsets.
    :param Any row_size: Flattened feature width.
    :param Any BLOCK_ROWS: Triton row tile size.
    :param Any BLOCK_COLS: Triton feature tile size.
    :return None: This Triton kernel writes in place.
    """

    segment_idx = tl.program_id(0)
    tile_row = tl.program_id(1)
    tile_col = tl.program_id(2)

    row_offsets = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = tile_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    src_base = tl.load(offsets_ptr + segment_idx)
    seg_len = tl.load(lengths_ptr + segment_idx)
    dst_base = tl.load(cu_seqlens_ptr + segment_idx)

    src_rows = src_base + row_offsets
    dst_rows = dst_base + row_offsets

    mask = (row_offsets[:, None] < seg_len) & (col_offsets[None, :] < row_size)
    src_a_ptrs = input_a_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    src_b_ptrs = input_b_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    src_c_ptrs = input_c_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    dst_a_ptrs = output_a_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    dst_b_ptrs = output_b_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    dst_c_ptrs = output_c_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    values_a = tl.load(src_a_ptrs, mask=mask, other=0)
    values_b = tl.load(src_b_ptrs, mask=mask, other=0)
    values_c = tl.load(src_c_ptrs, mask=mask, other=0)
    tl.store(dst_a_ptrs, values_a, mask=mask)
    tl.store(dst_b_ptrs, values_b, mask=mask)
    tl.store(dst_c_ptrs, values_c, mask=mask)


@triton.jit
def _unpack_segment_rows_kernel(
    input_ptr: None,
    output_ptr: None,
    offsets_ptr: None,
    lengths_ptr: None,
    cu_seqlens_ptr: None,
    row_size: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
) -> None:
    """Scatter packed rows back into padded row-major layout.

    :param Any input_ptr: Pointer to packed input rows.
    :param Any output_ptr: Pointer to padded output rows.
    :param Any offsets_ptr: Pointer to flat padded row offsets per segment.
    :param Any lengths_ptr: Pointer to per-segment lengths.
    :param Any cu_seqlens_ptr: Pointer to cumulative packed offsets.
    :param Any row_size: Flattened feature width.
    :param Any BLOCK_ROWS: Triton row tile size.
    :param Any BLOCK_COLS: Triton feature tile size.
    :return None: This Triton kernel writes in place.
    """

    segment_idx = tl.program_id(0)
    tile_row = tl.program_id(1)
    tile_col = tl.program_id(2)

    row_offsets = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = tile_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    dst_base = tl.load(offsets_ptr + segment_idx)
    seg_len = tl.load(lengths_ptr + segment_idx)
    src_base = tl.load(cu_seqlens_ptr + segment_idx)

    dst_rows = dst_base + row_offsets
    src_rows = src_base + row_offsets

    mask = (row_offsets[:, None] < seg_len) & (col_offsets[None, :] < row_size)
    src_ptrs = input_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    dst_ptrs = output_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    values = tl.load(src_ptrs, mask=mask, other=0)
    tl.store(dst_ptrs, values, mask=mask)


@triton.jit
def _unpack_segment_rows_pair_kernel(
    input_a_ptr: None,
    input_b_ptr: None,
    output_a_ptr: None,
    output_b_ptr: None,
    offsets_ptr: None,
    lengths_ptr: None,
    cu_seqlens_ptr: None,
    row_size: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
) -> None:
    """Scatter two packed tensors back into padded row-major layout.

    :param Any input_a_ptr: Pointer to the first packed input rows.
    :param Any input_b_ptr: Pointer to the second packed input rows.
    :param Any output_a_ptr: Pointer to the first padded output rows.
    :param Any output_b_ptr: Pointer to the second padded output rows.
    :param Any offsets_ptr: Pointer to flat padded row offsets per segment.
    :param Any lengths_ptr: Pointer to per-segment lengths.
    :param Any cu_seqlens_ptr: Pointer to cumulative packed offsets.
    :param Any row_size: Flattened feature width.
    :param Any BLOCK_ROWS: Triton row tile size.
    :param Any BLOCK_COLS: Triton feature tile size.
    :return None: This Triton kernel writes in place.
    """

    segment_idx = tl.program_id(0)
    tile_row = tl.program_id(1)
    tile_col = tl.program_id(2)

    row_offsets = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = tile_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    dst_base = tl.load(offsets_ptr + segment_idx)
    seg_len = tl.load(lengths_ptr + segment_idx)
    src_base = tl.load(cu_seqlens_ptr + segment_idx)

    dst_rows = dst_base + row_offsets
    src_rows = src_base + row_offsets

    mask = (row_offsets[:, None] < seg_len) & (col_offsets[None, :] < row_size)
    src_a_ptrs = input_a_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    src_b_ptrs = input_b_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    dst_a_ptrs = output_a_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    dst_b_ptrs = output_b_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    values_a = tl.load(src_a_ptrs, mask=mask, other=0)
    values_b = tl.load(src_b_ptrs, mask=mask, other=0)
    tl.store(dst_a_ptrs, values_a, mask=mask)
    tl.store(dst_b_ptrs, values_b, mask=mask)


@triton.jit
def _unpack_segment_rows_triple_kernel(
    input_a_ptr: None,
    input_b_ptr: None,
    input_c_ptr: None,
    output_a_ptr: None,
    output_b_ptr: None,
    output_c_ptr: None,
    offsets_ptr: None,
    lengths_ptr: None,
    cu_seqlens_ptr: None,
    row_size: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
) -> None:
    """Scatter three packed tensors back into padded row-major layout.

    :param Any input_a_ptr: Pointer to the first packed input rows.
    :param Any input_b_ptr: Pointer to the second packed input rows.
    :param Any input_c_ptr: Pointer to the third packed input rows.
    :param Any output_a_ptr: Pointer to the first padded output rows.
    :param Any output_b_ptr: Pointer to the second padded output rows.
    :param Any output_c_ptr: Pointer to the third padded output rows.
    :param Any offsets_ptr: Pointer to flat padded row offsets per segment.
    :param Any lengths_ptr: Pointer to per-segment lengths.
    :param Any cu_seqlens_ptr: Pointer to cumulative packed offsets.
    :param Any row_size: Flattened feature width.
    :param Any BLOCK_ROWS: Triton row tile size.
    :param Any BLOCK_COLS: Triton feature tile size.
    :return None: This Triton kernel writes in place.
    """

    segment_idx = tl.program_id(0)
    tile_row = tl.program_id(1)
    tile_col = tl.program_id(2)

    row_offsets = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = tile_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    dst_base = tl.load(offsets_ptr + segment_idx)
    seg_len = tl.load(lengths_ptr + segment_idx)
    src_base = tl.load(cu_seqlens_ptr + segment_idx)

    dst_rows = dst_base + row_offsets
    src_rows = src_base + row_offsets

    mask = (row_offsets[:, None] < seg_len) & (col_offsets[None, :] < row_size)
    src_a_ptrs = input_a_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    src_b_ptrs = input_b_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    src_c_ptrs = input_c_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    dst_a_ptrs = output_a_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    dst_b_ptrs = output_b_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    dst_c_ptrs = output_c_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    values_a = tl.load(src_a_ptrs, mask=mask, other=0)
    values_b = tl.load(src_b_ptrs, mask=mask, other=0)
    values_c = tl.load(src_c_ptrs, mask=mask, other=0)
    tl.store(dst_a_ptrs, values_a, mask=mask)
    tl.store(dst_b_ptrs, values_b, mask=mask)
    tl.store(dst_c_ptrs, values_c, mask=mask)


def segment_pack_padded_rows(
    tensor: torch.Tensor,
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    """Pack contiguous padded token segments into one packed tensor.

    :param torch.Tensor tensor: Contiguous input tensor with shape ``(B, S, ...)``.
    :param torch.Tensor segment_offsets: Flat ``(B*S)`` source row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param int total_tokens: Total packed token count.
    :return torch.Tensor: Packed tensor with shape ``(NNZ, ...)``.
    """

    flat, trailing_shape, _batch_size, _seq_len = _flatten_rows(tensor)
    total = max(0, int(total_tokens))
    output = tensor.new_empty((total,) + trailing_shape)
    if total == 0:
        return output
    out_flat = output.view(total, -1)
    row_size = int(out_flat.shape[1])

    if _can_use_triton_segment_pack(
        tensor=tensor,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
    ):
        max_len = max(1, int(segment_lengths.max().item()))
        grid = (
            int(segment_lengths.shape[0]),
            triton.cdiv(max_len, _SEGMENT_BLOCK_ROWS),
            triton.cdiv(row_size, _SEGMENT_BLOCK_COLS),
        )
        _traceable_triton_kernel(_pack_segment_rows_kernel)[grid](
            flat,
            out_flat,
            segment_offsets,
            segment_lengths,
            cu_seqlens,
            row_size,
            BLOCK_ROWS=_SEGMENT_BLOCK_ROWS,
            BLOCK_COLS=_SEGMENT_BLOCK_COLS,
            num_warps=_SEGMENT_NUM_WARPS,
            num_stages=_SEGMENT_NUM_STAGES,
        )
        return output

    offsets_host = _tensor_host_tuple(segment_offsets)
    lengths_host = _tensor_host_tuple(segment_lengths)
    cu_host = _tensor_host_tuple(cu_seqlens)
    for idx, src_base in enumerate(offsets_host):
        seg_len = int(lengths_host[idx])
        if seg_len <= 0:
            continue
        dst_base = int(cu_host[idx])
        out_flat[dst_base : dst_base + seg_len].copy_(flat[src_base : src_base + seg_len])
    return output


def segment_pack_padded_rows_pair(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack two contiguous padded tensors with shared segment metadata.

    :param torch.Tensor tensor_a: First contiguous input tensor ``(B, S, ...)``.
    :param torch.Tensor tensor_b: Second contiguous input tensor ``(B, S, ...)``.
    :param torch.Tensor segment_offsets: Flat ``(B*S)`` source row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param int total_tokens: Total packed token count.
    :return tuple[torch.Tensor, torch.Tensor]: Packed tensors ``(NNZ, ...)``.
    """

    flat_a, trailing_shape, _batch_size, _seq_len = _flatten_rows(tensor_a)
    flat_b, trailing_shape_b, _, _ = _flatten_rows(tensor_b)
    if trailing_shape_b != trailing_shape:
        raise ValueError("segment_pack_padded_rows_pair requires matching trailing shapes.")
    total = max(0, int(total_tokens))
    out_a = tensor_a.new_empty((total,) + trailing_shape)
    out_b = tensor_b.new_empty((total,) + trailing_shape)
    if total == 0:
        return out_a, out_b
    out_flat_a = out_a.view(total, -1)
    out_flat_b = out_b.view(total, -1)
    row_size = int(out_flat_a.shape[1])

    if (
        _can_use_triton_segment_pack(
            tensor=tensor_a,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
        )
        and tensor_b.is_contiguous()
        and tensor_b.device == tensor_a.device
    ):
        max_len = max(1, int(segment_lengths.max().item()))
        grid = (
            int(segment_lengths.shape[0]),
            triton.cdiv(max_len, _SEGMENT_BLOCK_ROWS),
            triton.cdiv(row_size, _SEGMENT_BLOCK_COLS),
        )
        _traceable_triton_kernel(_pack_segment_rows_pair_kernel)[grid](
            flat_a,
            flat_b,
            out_flat_a,
            out_flat_b,
            segment_offsets,
            segment_lengths,
            cu_seqlens,
            row_size,
            BLOCK_ROWS=_SEGMENT_BLOCK_ROWS,
            BLOCK_COLS=_SEGMENT_BLOCK_COLS,
            num_warps=_SEGMENT_NUM_WARPS,
            num_stages=_SEGMENT_NUM_STAGES,
        )
        return out_a, out_b

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
        out_flat_a[dst_slice].copy_(flat_a[src_slice])
        out_flat_b[dst_slice].copy_(flat_b[src_slice])
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack three contiguous padded tensors with shared segment metadata.

    :param torch.Tensor tensor_a: First contiguous input tensor ``(B, S, ...)``.
    :param torch.Tensor tensor_b: Second contiguous input tensor ``(B, S, ...)``.
    :param torch.Tensor tensor_c: Third contiguous input tensor ``(B, S, ...)``.
    :param torch.Tensor segment_offsets: Flat ``(B*S)`` source row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param int total_tokens: Total packed token count.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Packed tensors ``(NNZ, ...)``.
    """

    flat_a, trailing_shape, _batch_size, _seq_len = _flatten_rows(tensor_a)
    flat_b, trailing_shape_b, _, _ = _flatten_rows(tensor_b)
    flat_c, trailing_shape_c, _, _ = _flatten_rows(tensor_c)
    if trailing_shape_b != trailing_shape or trailing_shape_c != trailing_shape:
        raise ValueError("segment_pack_padded_rows_triple requires matching trailing shapes.")
    total = max(0, int(total_tokens))
    out_a = tensor_a.new_empty((total,) + trailing_shape)
    out_b = tensor_b.new_empty((total,) + trailing_shape)
    out_c = tensor_c.new_empty((total,) + trailing_shape)
    if total == 0:
        return out_a, out_b, out_c
    out_flat_a = out_a.view(total, -1)
    out_flat_b = out_b.view(total, -1)
    out_flat_c = out_c.view(total, -1)
    row_size = int(out_flat_a.shape[1])

    if (
        _can_use_triton_segment_pack(
            tensor=tensor_a,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
        )
        and tensor_b.is_contiguous()
        and tensor_c.is_contiguous()
    ):
        max_len = max(1, int(segment_lengths.max().item()))
        grid = (
            int(segment_lengths.shape[0]),
            triton.cdiv(max_len, _SEGMENT_BLOCK_ROWS),
            triton.cdiv(row_size, _SEGMENT_BLOCK_COLS),
        )
        _traceable_triton_kernel(_pack_segment_rows_triple_kernel)[grid](
            flat_a,
            flat_b,
            flat_c,
            out_flat_a,
            out_flat_b,
            out_flat_c,
            segment_offsets,
            segment_lengths,
            cu_seqlens,
            row_size,
            BLOCK_ROWS=_SEGMENT_BLOCK_ROWS,
            BLOCK_COLS=_SEGMENT_BLOCK_COLS,
            num_warps=_SEGMENT_NUM_WARPS,
            num_stages=_SEGMENT_NUM_STAGES,
        )
        return out_a, out_b, out_c

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
        out_flat_a[dst_slice].copy_(flat_a[src_slice])
        out_flat_b[dst_slice].copy_(flat_b[src_slice])
        out_flat_c[dst_slice].copy_(flat_c[src_slice])
    return out_a, out_b, out_c


def segment_unpack_padded_rows(
    packed: torch.Tensor,
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Scatter one packed tensor back into padded ``(B, S, ...)`` layout.

    :param torch.Tensor packed: Packed tensor with shape ``(NNZ, ...)``.
    :param torch.Tensor segment_offsets: Flat padded row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param int batch_size: Output batch size.
    :param int seq_len: Output padded sequence length.
    :return torch.Tensor: Padded tensor with shape ``(B, S, ...)``.
    """

    total = int(packed.shape[0])
    trailing_shape = tuple(int(dim) for dim in packed.shape[1:])
    flat = packed.contiguous().view(total, -1)
    output = packed.new_zeros((int(batch_size), int(seq_len)) + trailing_shape)
    if total == 0:
        return output
    out_flat = output.view(int(batch_size) * int(seq_len), -1)
    row_size = int(flat.shape[1])

    if _can_use_triton_segment_pack(
        tensor=output,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
    ):
        max_len = max(1, int(segment_lengths.max().item()))
        grid = (
            int(segment_lengths.shape[0]),
            triton.cdiv(max_len, _SEGMENT_BLOCK_ROWS),
            triton.cdiv(row_size, _SEGMENT_BLOCK_COLS),
        )
        _traceable_triton_kernel(_unpack_segment_rows_kernel)[grid](
            flat,
            out_flat,
            segment_offsets,
            segment_lengths,
            cu_seqlens,
            row_size,
            BLOCK_ROWS=_SEGMENT_BLOCK_ROWS,
            BLOCK_COLS=_SEGMENT_BLOCK_COLS,
            num_warps=_SEGMENT_NUM_WARPS,
            num_stages=_SEGMENT_NUM_STAGES,
        )
        return output

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
        out_flat[dst_slice].copy_(flat[src_slice])
    return output


def segment_unpack_padded_rows_pair(
    packed_a: torch.Tensor,
    packed_b: torch.Tensor,
    *,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter two packed tensors back into padded ``(B, S, ...)`` layout.

    :param torch.Tensor packed_a: First packed tensor ``(NNZ, ...)``.
    :param torch.Tensor packed_b: Second packed tensor ``(NNZ, ...)``.
    :param torch.Tensor segment_offsets: Flat padded row offsets per segment.
    :param torch.Tensor segment_lengths: Per-segment lengths.
    :param torch.Tensor cu_seqlens: Cumulative packed offsets per segment.
    :param int batch_size: Output batch size.
    :param int seq_len: Output padded sequence length.
    :return tuple[torch.Tensor, torch.Tensor]: Padded tensors ``(B, S, ...)``.
    """

    total = int(packed_a.shape[0])
    trailing_shape = tuple(int(dim) for dim in packed_a.shape[1:])
    flat_a = packed_a.contiguous().view(total, -1)
    flat_b = packed_b.contiguous().view(total, -1)
    output_a = packed_a.new_zeros((int(batch_size), int(seq_len)) + trailing_shape)
    output_b = packed_b.new_zeros((int(batch_size), int(seq_len)) + trailing_shape)
    if total == 0:
        return output_a, output_b
    out_flat_a = output_a.view(int(batch_size) * int(seq_len), -1)
    out_flat_b = output_b.view(int(batch_size) * int(seq_len), -1)
    row_size = int(flat_a.shape[1])

    if (
        _can_use_triton_segment_pack(
            tensor=output_a,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
        )
        and packed_b.device == packed_a.device
    ):
        max_len = max(1, int(segment_lengths.max().item()))
        grid = (
            int(segment_lengths.shape[0]),
            triton.cdiv(max_len, _SEGMENT_BLOCK_ROWS),
            triton.cdiv(row_size, _SEGMENT_BLOCK_COLS),
        )
        _traceable_triton_kernel(_unpack_segment_rows_pair_kernel)[grid](
            flat_a,
            flat_b,
            out_flat_a,
            out_flat_b,
            segment_offsets,
            segment_lengths,
            cu_seqlens,
            row_size,
            BLOCK_ROWS=_SEGMENT_BLOCK_ROWS,
            BLOCK_COLS=_SEGMENT_BLOCK_COLS,
            num_warps=_SEGMENT_NUM_WARPS,
            num_stages=_SEGMENT_NUM_STAGES,
        )
        return output_a, output_b

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
        out_flat_a[dst_slice].copy_(flat_a[src_slice])
        out_flat_b[dst_slice].copy_(flat_b[src_slice])
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
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Padded tensors ``(B, S, ...)``.
    """

    total = int(packed_a.shape[0])
    trailing_shape = tuple(int(dim) for dim in packed_a.shape[1:])
    flat_a = packed_a.contiguous().view(total, -1)
    flat_b = packed_b.contiguous().view(total, -1)
    flat_c = packed_c.contiguous().view(total, -1)
    output_a = packed_a.new_zeros((int(batch_size), int(seq_len)) + trailing_shape)
    output_b = packed_b.new_zeros((int(batch_size), int(seq_len)) + trailing_shape)
    output_c = packed_c.new_zeros((int(batch_size), int(seq_len)) + trailing_shape)
    if total == 0:
        return output_a, output_b, output_c
    out_flat_a = output_a.view(int(batch_size) * int(seq_len), -1)
    out_flat_b = output_b.view(int(batch_size) * int(seq_len), -1)
    out_flat_c = output_c.view(int(batch_size) * int(seq_len), -1)
    row_size = int(flat_a.shape[1])

    if (
        _can_use_triton_segment_pack(
            tensor=output_a,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
        )
        and packed_b.device == packed_a.device
        and packed_c.device == packed_a.device
    ):
        max_len = max(1, int(segment_lengths.max().item()))
        grid = (
            int(segment_lengths.shape[0]),
            triton.cdiv(max_len, _SEGMENT_BLOCK_ROWS),
            triton.cdiv(row_size, _SEGMENT_BLOCK_COLS),
        )
        _traceable_triton_kernel(_unpack_segment_rows_triple_kernel)[grid](
            flat_a,
            flat_b,
            flat_c,
            out_flat_a,
            out_flat_b,
            out_flat_c,
            segment_offsets,
            segment_lengths,
            cu_seqlens,
            row_size,
            BLOCK_ROWS=_SEGMENT_BLOCK_ROWS,
            BLOCK_COLS=_SEGMENT_BLOCK_COLS,
            num_warps=_SEGMENT_NUM_WARPS,
            num_stages=_SEGMENT_NUM_STAGES,
        )
        return output_a, output_b, output_c

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
        out_flat_a[dst_slice].copy_(flat_a[src_slice])
        out_flat_b[dst_slice].copy_(flat_b[src_slice])
        out_flat_c[dst_slice].copy_(flat_c[src_slice])
    return output_a, output_b, output_c


__all__ = [
    "clear_segment_pack_host_cache",
    "flashdeberta_segment_pack_available",
    "flashdeberta_segment_pack_import_error",
    "segment_pack_padded_rows",
    "segment_pack_padded_rows_pair",
    "segment_pack_padded_rows_triple",
    "segment_unpack_padded_rows",
    "segment_unpack_padded_rows_pair",
    "segment_unpack_padded_rows_triple",
]
