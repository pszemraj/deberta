"""Prefix-padding pack/unpack helpers for FlashDeBERTa varlen paths.

The repo's collator and padding-mask utilities follow standard prefix-padding
semantics: active tokens are a prefix in each sequence and padded tokens occupy
the tail. The varlen adapter can exploit that contract directly and avoid the
generic ``nonzero``/``gather``/``index_copy`` path that was previously used to
pack and repad tensors around the varlen Triton kernels.

On CUDA with Triton available, these helpers use small repo-local copy kernels.
Otherwise they fall back to simple eager prefix copies, which keeps CPU tests
and missing-Triton environments working without affecting correctness.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

try:  # pragma: no cover - optional Triton dependency
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional Triton dependency
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc

_PACK_BLOCK_ROWS = 32
_PACK_BLOCK_COLS = 128
_PACK_NUM_WARPS = 4
_PACK_NUM_STAGES = 2


def flashdeberta_prefix_pack_import_error() -> Exception | None:
    """Return the Triton import failure for prefix-pack kernels, if any.

    :return Exception | None: Stored Triton import failure or ``None``.
    """

    return _TRITON_IMPORT_ERROR


def flashdeberta_prefix_pack_available() -> bool:
    """Return whether Triton-backed prefix pack kernels are importable.

    :return bool: True when Triton was imported successfully.
    """

    return triton is not None and tl is not None


def _flatten_rows(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], int, int]:
    """Flatten a ``(B, S, ...)`` tensor into contiguous row-major ``(B*S, F)`` form.

    :param torch.Tensor tensor: Tensor with at least two leading dims ``(B, S)``.
    :raises ValueError: If the tensor rank is less than two.
    :return tuple[torch.Tensor, tuple[int, ...], int, int]:
        Flattened tensor, trailing shape, batch size, and sequence length.
    """

    if tensor.ndim < 2:
        raise ValueError(f"Expected tensor with leading (B,S) dims; got shape={tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError("Prefix-pack Triton path requires contiguous tensors.")

    batch_size = int(tensor.shape[0])
    seq_len = int(tensor.shape[1])
    trailing_shape = tuple(int(dim) for dim in tensor.shape[2:])
    flat = tensor.view(batch_size * seq_len, -1)
    return flat, trailing_shape, batch_size, seq_len


def _can_use_triton_prefix_pack(
    *,
    tensor: torch.Tensor,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> bool:
    """Return whether the Triton prefix-pack path should be used.

    :param torch.Tensor tensor: Input/output tensor.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative sequence lengths.
    :return bool: True when the Triton path is usable for this call.
    """

    if not flashdeberta_prefix_pack_available():
        return False
    if tensor.device.type != "cuda":
        return False
    if not tensor.is_contiguous():
        return False
    if seqlens.device != tensor.device or cu_seqlens.device != tensor.device:
        return False
    return True


@triton.jit
def _pack_prefix_rows_kernel(
    input_ptr: None,
    output_ptr: None,
    seqlens_ptr: None,
    cu_seqlens_ptr: None,
    seq_len: int,
    row_size: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
) -> None:
    """Copy valid prefix rows from padded ``(B*S, F)`` into packed ``(NNZ, F)``.

    :param Any input_ptr: Triton pointer to the padded row-major input tensor.
    :param Any output_ptr: Triton pointer to the packed row-major output tensor.
    :param Any seqlens_ptr: Triton pointer to per-example active lengths.
    :param Any cu_seqlens_ptr: Triton pointer to cumulative active lengths.
    :param Any seq_len: Padded sequence length.
    :param Any row_size: Flattened trailing feature width.
    :param Any BLOCK_ROWS: Triton row tile size.
    :param Any BLOCK_COLS: Triton feature tile size.
    :return None: This Triton kernel writes directly to ``output_ptr``.
    """

    batch_idx = tl.program_id(0)
    tile_row = tl.program_id(1)
    tile_col = tl.program_id(2)

    row_offsets = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = tile_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    seqlen = tl.load(seqlens_ptr + batch_idx)
    packed_base = tl.load(cu_seqlens_ptr + batch_idx)
    src_rows = batch_idx * seq_len + row_offsets
    dst_rows = packed_base + row_offsets

    mask = (row_offsets[:, None] < seqlen) & (col_offsets[None, :] < row_size)
    src_ptrs = input_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    dst_ptrs = output_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    values = tl.load(src_ptrs, mask=mask, other=0)
    tl.store(dst_ptrs, values, mask=mask)


@triton.jit
def _pack_prefix_rows_rank4_strided_kernel(
    input_ptr: None,
    output_ptr: None,
    seqlens_ptr: None,
    cu_seqlens_ptr: None,
    stride_b: int,
    stride_s: int,
    stride_h: int,
    stride_f: int,
    seq_len: int,
    num_heads: int,
    feature_size: int,
    col_tiles: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
) -> None:
    """Pack strided ``(B, S, H, F)`` tensors into contiguous ``(NNZ, H, F)``.

    :param Any input_ptr: Triton pointer to the padded strided input tensor.
    :param Any output_ptr: Triton pointer to the packed contiguous output tensor.
    :param Any seqlens_ptr: Triton pointer to per-example active lengths.
    :param Any cu_seqlens_ptr: Triton pointer to cumulative active lengths.
    :param Any stride_b: Input stride for the batch dimension.
    :param Any stride_s: Input stride for the sequence dimension.
    :param Any stride_h: Input stride for the head dimension.
    :param Any stride_f: Input stride for the per-head feature dimension.
    :param Any seq_len: Padded sequence length.
    :param Any num_heads: Number of heads in the input tensor.
    :param Any feature_size: Per-head feature width.
    :param Any col_tiles: Number of feature tiles per head.
    :param Any BLOCK_ROWS: Triton row tile size.
    :param Any BLOCK_COLS: Triton feature tile size.
    :return None: This Triton kernel writes directly to ``output_ptr``.
    """

    batch_idx = tl.program_id(0)
    tile_row = tl.program_id(1)
    tile_hf = tl.program_id(2)

    row_offsets = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    head_idx = tile_hf // col_tiles
    tile_col = tile_hf % col_tiles
    col_offsets = tile_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    seqlen = tl.load(seqlens_ptr + batch_idx)
    packed_base = tl.load(cu_seqlens_ptr + batch_idx)
    dst_row_offsets = packed_base + row_offsets

    src_ptrs = (
        input_ptr
        + batch_idx * stride_b
        + row_offsets[:, None] * stride_s
        + head_idx * stride_h
        + col_offsets[None, :] * stride_f
    )
    dst_ptrs = (
        output_ptr
        + dst_row_offsets[:, None] * (num_heads * feature_size)
        + head_idx * feature_size
        + col_offsets[None, :]
    )
    mask = (head_idx < num_heads) & (row_offsets[:, None] < seqlen) & (col_offsets[None, :] < feature_size)
    values = tl.load(src_ptrs, mask=mask, other=0)
    tl.store(dst_ptrs, values, mask=mask)


@triton.jit
def _unpack_prefix_rows_kernel(
    input_ptr: None,
    output_ptr: None,
    seqlens_ptr: None,
    cu_seqlens_ptr: None,
    seq_len: int,
    row_size: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
) -> None:
    """Copy packed rows back into padded layout with zero tail fill.

    :param Any input_ptr: Triton pointer to the packed row-major input tensor.
    :param Any output_ptr: Triton pointer to the padded row-major output tensor.
    :param Any seqlens_ptr: Triton pointer to per-example active lengths.
    :param Any cu_seqlens_ptr: Triton pointer to cumulative active lengths.
    :param Any seq_len: Padded sequence length.
    :param Any row_size: Flattened trailing feature width.
    :param Any BLOCK_ROWS: Triton row tile size.
    :param Any BLOCK_COLS: Triton feature tile size.
    :return None: This Triton kernel writes directly to ``output_ptr``.
    """

    batch_idx = tl.program_id(0)
    tile_row = tl.program_id(1)
    tile_col = tl.program_id(2)

    row_offsets = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = tile_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    seqlen = tl.load(seqlens_ptr + batch_idx)
    packed_base = tl.load(cu_seqlens_ptr + batch_idx)
    dst_rows = batch_idx * seq_len + row_offsets
    src_rows = packed_base + row_offsets

    output_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < row_size)
    input_mask = (row_offsets[:, None] < seqlen) & (col_offsets[None, :] < row_size)
    src_ptrs = input_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    dst_ptrs = output_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    values = tl.load(src_ptrs, mask=input_mask, other=0)
    tl.store(dst_ptrs, values, mask=output_mask)


@triton.jit
def _unpack_prefix_rows_pair_kernel(
    input_a_ptr: None,
    input_b_ptr: None,
    output_a_ptr: None,
    output_b_ptr: None,
    seqlens_ptr: None,
    cu_seqlens_ptr: None,
    seq_len: int,
    row_size: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
) -> None:
    """Copy two packed tensors back into padded layout with shared prefix metadata.

    :param Any input_a_ptr: Triton pointer to the first packed row-major input tensor.
    :param Any input_b_ptr: Triton pointer to the second packed row-major input tensor.
    :param Any output_a_ptr: Triton pointer to the first padded row-major output tensor.
    :param Any output_b_ptr: Triton pointer to the second padded row-major output tensor.
    :param Any seqlens_ptr: Triton pointer to per-example active lengths.
    :param Any cu_seqlens_ptr: Triton pointer to cumulative active lengths.
    :param Any seq_len: Padded sequence length.
    :param Any row_size: Flattened trailing feature width.
    :param Any BLOCK_ROWS: Triton row tile size.
    :param Any BLOCK_COLS: Triton feature tile size.
    :return None: This Triton kernel writes directly to the output pointers.
    """

    batch_idx = tl.program_id(0)
    tile_row = tl.program_id(1)
    tile_col = tl.program_id(2)

    row_offsets = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = tile_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    seqlen = tl.load(seqlens_ptr + batch_idx)
    packed_base = tl.load(cu_seqlens_ptr + batch_idx)
    dst_rows = batch_idx * seq_len + row_offsets
    src_rows = packed_base + row_offsets

    output_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < row_size)
    input_mask = (row_offsets[:, None] < seqlen) & (col_offsets[None, :] < row_size)
    src_a_ptrs = input_a_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    src_b_ptrs = input_b_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    dst_a_ptrs = output_a_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    dst_b_ptrs = output_b_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    values_a = tl.load(src_a_ptrs, mask=input_mask, other=0)
    values_b = tl.load(src_b_ptrs, mask=input_mask, other=0)
    tl.store(dst_a_ptrs, values_a, mask=output_mask)
    tl.store(dst_b_ptrs, values_b, mask=output_mask)


@triton.jit
def _unpack_prefix_rows_triple_kernel(
    input_a_ptr: None,
    input_b_ptr: None,
    input_c_ptr: None,
    output_a_ptr: None,
    output_b_ptr: None,
    output_c_ptr: None,
    seqlens_ptr: None,
    cu_seqlens_ptr: None,
    seq_len: int,
    row_size: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
) -> None:
    """Copy three packed tensors back into padded layout with shared prefix metadata.

    :param Any input_a_ptr: Triton pointer to the first packed row-major input tensor.
    :param Any input_b_ptr: Triton pointer to the second packed row-major input tensor.
    :param Any input_c_ptr: Triton pointer to the third packed row-major input tensor.
    :param Any output_a_ptr: Triton pointer to the first padded row-major output tensor.
    :param Any output_b_ptr: Triton pointer to the second padded row-major output tensor.
    :param Any output_c_ptr: Triton pointer to the third padded row-major output tensor.
    :param Any seqlens_ptr: Triton pointer to per-example active lengths.
    :param Any cu_seqlens_ptr: Triton pointer to cumulative active lengths.
    :param Any seq_len: Padded sequence length.
    :param Any row_size: Flattened trailing feature width.
    :param Any BLOCK_ROWS: Triton row tile size.
    :param Any BLOCK_COLS: Triton feature tile size.
    :return None: This Triton kernel writes directly to the output pointers.
    """

    batch_idx = tl.program_id(0)
    tile_row = tl.program_id(1)
    tile_col = tl.program_id(2)

    row_offsets = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = tile_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    seqlen = tl.load(seqlens_ptr + batch_idx)
    packed_base = tl.load(cu_seqlens_ptr + batch_idx)
    dst_rows = batch_idx * seq_len + row_offsets
    src_rows = packed_base + row_offsets

    output_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < row_size)
    input_mask = (row_offsets[:, None] < seqlen) & (col_offsets[None, :] < row_size)
    src_a_ptrs = input_a_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    src_b_ptrs = input_b_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    src_c_ptrs = input_c_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    dst_a_ptrs = output_a_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    dst_b_ptrs = output_b_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    dst_c_ptrs = output_c_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    values_a = tl.load(src_a_ptrs, mask=input_mask, other=0)
    values_b = tl.load(src_b_ptrs, mask=input_mask, other=0)
    values_c = tl.load(src_c_ptrs, mask=input_mask, other=0)
    tl.store(dst_a_ptrs, values_a, mask=output_mask)
    tl.store(dst_b_ptrs, values_b, mask=output_mask)
    tl.store(dst_c_ptrs, values_c, mask=output_mask)


def _prefix_pack_fallback(
    tensor: torch.Tensor,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Pack prefix-padded rows with a simple eager fallback.

    :param torch.Tensor tensor: Padded tensor with leading shape ``(B, S, ...)``.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative active lengths.
    :return torch.Tensor: Packed tensor with leading shape ``(NNZ, ...)``.
    """

    del cu_seqlens
    lengths = [int(v) for v in seqlens.detach().cpu().tolist()]
    parts = [tensor[b, :length] for b, length in enumerate(lengths) if length > 0]
    if not parts:
        return tensor.new_empty((0, *tensor.shape[2:]))
    return torch.cat(parts, dim=0)


def _prefix_unpack_fallback(
    values: torch.Tensor,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    trailing_shape: Sequence[int],
) -> torch.Tensor:
    """Unpack prefix rows with a simple eager fallback.

    :param torch.Tensor values: Packed tensor with leading shape ``(NNZ, ...)``.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative active lengths.
    :param int batch_size: Batch size.
    :param int seq_len: Padded sequence length.
    :param Sequence[int] trailing_shape: Original trailing tensor shape.
    :return torch.Tensor: Padded tensor with leading shape ``(B, S, ...)``.
    """

    lengths = [int(v) for v in seqlens.detach().cpu().tolist()]
    offsets = [int(v) for v in cu_seqlens.detach().cpu().tolist()]
    out = values.new_zeros((batch_size, seq_len, *trailing_shape))
    for batch_idx, length in enumerate(lengths):
        if length <= 0:
            continue
        start = offsets[batch_idx]
        end = offsets[batch_idx + 1]
        out[batch_idx, :length] = values[start:end]
    return out


def prefix_pack_padded_rows(
    tensor: torch.Tensor,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_tokens: int | None = None,
) -> torch.Tensor:
    """Pack active prefix rows from a padded tensor.

    :param torch.Tensor tensor: Padded tensor with leading shape ``(B, S, ...)``.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative active lengths.
    :param int max_seqlen: Maximum active length in the batch.
    :param int | None total_tokens: Optional precomputed number of active tokens.
    :return torch.Tensor: Packed tensor with leading shape ``(NNZ, ...)``.
    """

    if int(cu_seqlens.numel()) == 0:
        return tensor.new_empty((0, *tensor.shape[2:]))
    if total_tokens is None:
        total_tokens = int(sum(int(length) for length in seqlens.detach().cpu().tolist()))
    else:
        total_tokens = int(total_tokens)
    if total_tokens == 0:
        return tensor.new_empty((0, *tensor.shape[2:]))
    if (
        flashdeberta_prefix_pack_available()
        and tensor.device.type == "cuda"
        and not tensor.is_contiguous()
        and tensor.ndim == 4
    ):
        batch_size = int(tensor.shape[0])
        num_heads = int(tensor.shape[2])
        feature_size = int(tensor.shape[3])
        flat_output = torch.empty(
            (total_tokens, num_heads * feature_size),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        col_tiles = triton.cdiv(feature_size, _PACK_BLOCK_COLS)
        grid = (
            batch_size,
            triton.cdiv(int(max_seqlen), _PACK_BLOCK_ROWS),
            num_heads * col_tiles,
        )
        with torch.cuda.device(tensor.device):
            _pack_prefix_rows_rank4_strided_kernel[grid](
                tensor,
                flat_output,
                seqlens,
                cu_seqlens,
                tensor.stride(0),
                tensor.stride(1),
                tensor.stride(2),
                tensor.stride(3),
                int(tensor.shape[1]),
                num_heads,
                feature_size,
                col_tiles,
                BLOCK_ROWS=_PACK_BLOCK_ROWS,
                BLOCK_COLS=_PACK_BLOCK_COLS,
                num_warps=_PACK_NUM_WARPS,
                num_stages=_PACK_NUM_STAGES,
            )
        return flat_output.view(total_tokens, num_heads, feature_size)
    if flashdeberta_prefix_pack_available() and tensor.device.type == "cuda" and not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if not _can_use_triton_prefix_pack(tensor=tensor, seqlens=seqlens, cu_seqlens=cu_seqlens):
        return _prefix_pack_fallback(tensor, seqlens=seqlens, cu_seqlens=cu_seqlens)

    flat_input, trailing_shape, batch_size, seq_len = _flatten_rows(tensor)
    row_size = int(flat_input.shape[1])
    flat_output = torch.empty((total_tokens, row_size), device=tensor.device, dtype=tensor.dtype)
    grid = (
        batch_size,
        triton.cdiv(int(max_seqlen), _PACK_BLOCK_ROWS),
        triton.cdiv(row_size, _PACK_BLOCK_COLS),
    )
    with torch.cuda.device(tensor.device):
        _pack_prefix_rows_kernel[grid](
            flat_input,
            flat_output,
            seqlens,
            cu_seqlens,
            seq_len,
            row_size,
            BLOCK_ROWS=_PACK_BLOCK_ROWS,
            BLOCK_COLS=_PACK_BLOCK_COLS,
            num_warps=_PACK_NUM_WARPS,
            num_stages=_PACK_NUM_STAGES,
        )
    return flat_output.view(total_tokens, *trailing_shape)


def prefix_unpack_padded_rows(
    values: torch.Tensor,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Unpack active prefix rows back into a padded tensor.

    :param torch.Tensor values: Packed tensor with leading shape ``(NNZ, ...)``.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative active lengths.
    :param int batch_size: Batch size.
    :param int seq_len: Padded sequence length.
    :return torch.Tensor: Padded tensor with leading shape ``(B, S, ...)``.
    """

    trailing_shape = tuple(int(dim) for dim in values.shape[1:])
    if batch_size == 0 or seq_len == 0:
        return values.new_zeros((batch_size, seq_len, *trailing_shape))
    if values.numel() == 0:
        return values.new_zeros((batch_size, seq_len, *trailing_shape))
    if flashdeberta_prefix_pack_available() and values.device.type == "cuda" and not values.is_contiguous():
        values = values.contiguous()
    if not _can_use_triton_prefix_pack(tensor=values, seqlens=seqlens, cu_seqlens=cu_seqlens):
        return _prefix_unpack_fallback(
            values,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            batch_size=batch_size,
            seq_len=seq_len,
            trailing_shape=trailing_shape,
        )

    flat_values = values.view(int(values.shape[0]), -1)
    row_size = int(flat_values.shape[1])
    flat_output = torch.empty((batch_size * seq_len, row_size), device=values.device, dtype=values.dtype)
    grid = (
        batch_size,
        triton.cdiv(seq_len, _PACK_BLOCK_ROWS),
        triton.cdiv(row_size, _PACK_BLOCK_COLS),
    )
    with torch.cuda.device(values.device):
        _unpack_prefix_rows_kernel[grid](
            flat_values,
            flat_output,
            seqlens,
            cu_seqlens,
            seq_len,
            row_size,
            BLOCK_ROWS=_PACK_BLOCK_ROWS,
            BLOCK_COLS=_PACK_BLOCK_COLS,
            num_warps=_PACK_NUM_WARPS,
            num_stages=_PACK_NUM_STAGES,
        )
    return flat_output.view(batch_size, seq_len, *trailing_shape)


def _require_matching_packed_shapes(
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    values_c: torch.Tensor | None = None,
) -> tuple[int, tuple[int, ...]]:
    """Validate that packed tensors share the same leading and trailing shape.

    :param torch.Tensor values_a: First packed tensor.
    :param torch.Tensor values_b: Second packed tensor.
    :param torch.Tensor | None values_c: Optional third packed tensor.
    :raises ValueError: If packed tensor shapes do not match.
    :return tuple[int, tuple[int, ...]]: Shared packed-token count and trailing shape.
    """

    trailing_shape = tuple(int(dim) for dim in values_a.shape[1:])
    total_tokens = int(values_a.shape[0]) if int(values_a.ndim) > 0 else 0
    expected = (total_tokens, *trailing_shape)
    if tuple(int(dim) for dim in values_b.shape) != expected:
        raise ValueError(f"Packed tensor shape mismatch: expected {expected}, got {tuple(values_b.shape)}")
    if values_c is not None and tuple(int(dim) for dim in values_c.shape) != expected:
        raise ValueError(f"Packed tensor shape mismatch: expected {expected}, got {tuple(values_c.shape)}")
    return total_tokens, trailing_shape


def prefix_unpack_padded_rows_pair(
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack two packed tensors back into padded layout with shared metadata.

    :param torch.Tensor values_a: First packed tensor with leading shape ``(NNZ, ...)``.
    :param torch.Tensor values_b: Second packed tensor with leading shape ``(NNZ, ...)``.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative active lengths.
    :param int batch_size: Batch size.
    :param int seq_len: Padded sequence length.
    :return tuple[torch.Tensor, torch.Tensor]: Two padded tensors with leading shape ``(B, S, ...)``.
    """

    _, trailing_shape = _require_matching_packed_shapes(values_a, values_b)
    if batch_size == 0 or seq_len == 0 or values_a.numel() == 0:
        empty = values_a.new_zeros((batch_size, seq_len, *trailing_shape))
        return empty, values_b.new_zeros((batch_size, seq_len, *trailing_shape))
    if (
        flashdeberta_prefix_pack_available()
        and values_a.device.type == "cuda"
        and not values_a.is_contiguous()
    ):
        values_a = values_a.contiguous()
    if (
        flashdeberta_prefix_pack_available()
        and values_b.device.type == "cuda"
        and not values_b.is_contiguous()
    ):
        values_b = values_b.contiguous()
    if not _can_use_triton_prefix_pack(tensor=values_a, seqlens=seqlens, cu_seqlens=cu_seqlens):
        return (
            _prefix_unpack_fallback(
                values_a,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                batch_size=batch_size,
                seq_len=seq_len,
                trailing_shape=trailing_shape,
            ),
            _prefix_unpack_fallback(
                values_b,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                batch_size=batch_size,
                seq_len=seq_len,
                trailing_shape=trailing_shape,
            ),
        )

    flat_values_a = values_a.view(int(values_a.shape[0]), -1)
    flat_values_b = values_b.view(int(values_b.shape[0]), -1)
    row_size = int(flat_values_a.shape[1])
    flat_output_a = torch.empty(
        (batch_size * seq_len, row_size), device=values_a.device, dtype=values_a.dtype
    )
    flat_output_b = torch.empty(
        (batch_size * seq_len, row_size), device=values_b.device, dtype=values_b.dtype
    )
    grid = (
        batch_size,
        triton.cdiv(seq_len, _PACK_BLOCK_ROWS),
        triton.cdiv(row_size, _PACK_BLOCK_COLS),
    )
    with torch.cuda.device(values_a.device):
        _unpack_prefix_rows_pair_kernel[grid](
            flat_values_a,
            flat_values_b,
            flat_output_a,
            flat_output_b,
            seqlens,
            cu_seqlens,
            seq_len,
            row_size,
            BLOCK_ROWS=_PACK_BLOCK_ROWS,
            BLOCK_COLS=_PACK_BLOCK_COLS,
            num_warps=_PACK_NUM_WARPS,
            num_stages=_PACK_NUM_STAGES,
        )
    return (
        flat_output_a.view(batch_size, seq_len, *trailing_shape),
        flat_output_b.view(batch_size, seq_len, *trailing_shape),
    )


def prefix_unpack_padded_rows_triple(
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    values_c: torch.Tensor,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack three packed tensors back into padded layout with shared metadata.

    :param torch.Tensor values_a: First packed tensor with leading shape ``(NNZ, ...)``.
    :param torch.Tensor values_b: Second packed tensor with leading shape ``(NNZ, ...)``.
    :param torch.Tensor values_c: Third packed tensor with leading shape ``(NNZ, ...)``.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative active lengths.
    :param int batch_size: Batch size.
    :param int seq_len: Padded sequence length.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Three padded tensors with leading shape ``(B, S, ...)``.
    """

    _, trailing_shape = _require_matching_packed_shapes(values_a, values_b, values_c)
    if batch_size == 0 or seq_len == 0 or values_a.numel() == 0:
        empty_a = values_a.new_zeros((batch_size, seq_len, *trailing_shape))
        empty_b = values_b.new_zeros((batch_size, seq_len, *trailing_shape))
        empty_c = values_c.new_zeros((batch_size, seq_len, *trailing_shape))
        return empty_a, empty_b, empty_c
    if (
        flashdeberta_prefix_pack_available()
        and values_a.device.type == "cuda"
        and not values_a.is_contiguous()
    ):
        values_a = values_a.contiguous()
    if (
        flashdeberta_prefix_pack_available()
        and values_b.device.type == "cuda"
        and not values_b.is_contiguous()
    ):
        values_b = values_b.contiguous()
    if (
        flashdeberta_prefix_pack_available()
        and values_c.device.type == "cuda"
        and not values_c.is_contiguous()
    ):
        values_c = values_c.contiguous()
    if not _can_use_triton_prefix_pack(tensor=values_a, seqlens=seqlens, cu_seqlens=cu_seqlens):
        return (
            _prefix_unpack_fallback(
                values_a,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                batch_size=batch_size,
                seq_len=seq_len,
                trailing_shape=trailing_shape,
            ),
            _prefix_unpack_fallback(
                values_b,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                batch_size=batch_size,
                seq_len=seq_len,
                trailing_shape=trailing_shape,
            ),
            _prefix_unpack_fallback(
                values_c,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                batch_size=batch_size,
                seq_len=seq_len,
                trailing_shape=trailing_shape,
            ),
        )

    flat_values_a = values_a.view(int(values_a.shape[0]), -1)
    flat_values_b = values_b.view(int(values_b.shape[0]), -1)
    flat_values_c = values_c.view(int(values_c.shape[0]), -1)
    row_size = int(flat_values_a.shape[1])
    flat_output_a = torch.empty(
        (batch_size * seq_len, row_size), device=values_a.device, dtype=values_a.dtype
    )
    flat_output_b = torch.empty(
        (batch_size * seq_len, row_size), device=values_b.device, dtype=values_b.dtype
    )
    flat_output_c = torch.empty(
        (batch_size * seq_len, row_size), device=values_c.device, dtype=values_c.dtype
    )
    grid = (
        batch_size,
        triton.cdiv(seq_len, _PACK_BLOCK_ROWS),
        triton.cdiv(row_size, _PACK_BLOCK_COLS),
    )
    with torch.cuda.device(values_a.device):
        _unpack_prefix_rows_triple_kernel[grid](
            flat_values_a,
            flat_values_b,
            flat_values_c,
            flat_output_a,
            flat_output_b,
            flat_output_c,
            seqlens,
            cu_seqlens,
            seq_len,
            row_size,
            BLOCK_ROWS=_PACK_BLOCK_ROWS,
            BLOCK_COLS=_PACK_BLOCK_COLS,
            num_warps=_PACK_NUM_WARPS,
            num_stages=_PACK_NUM_STAGES,
        )
    return (
        flat_output_a.view(batch_size, seq_len, *trailing_shape),
        flat_output_b.view(batch_size, seq_len, *trailing_shape),
        flat_output_c.view(batch_size, seq_len, *trailing_shape),
    )


__all__ = [
    "flashdeberta_prefix_pack_available",
    "flashdeberta_prefix_pack_import_error",
    "prefix_pack_padded_rows",
    "prefix_unpack_padded_rows",
    "prefix_unpack_padded_rows_pair",
    "prefix_unpack_padded_rows_triple",
]
