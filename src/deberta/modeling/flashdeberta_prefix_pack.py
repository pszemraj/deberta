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
from functools import partial

import torch

from deberta.modeling.flashdeberta_op_utils import (
    TRITON_ROW_COPY_PREFIX_PACK,
    TRITON_ROW_COPY_PREFIX_PACK_STRIDED,
    TRITON_ROW_COPY_PREFIX_UNPACK,
    can_use_triton_pack,
    flatten_padded_rows,
    launch_triton_row_copy,
    require_matching_tensor_layout,
)

try:  # pragma: no cover - optional Triton dependency
    import triton

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional Triton dependency
    triton = None
    _TRITON_AVAILABLE = False

_PACK_BLOCK_ROWS = 32
_PACK_BLOCK_COLS = 128
_PACK_NUM_WARPS = 4
_PACK_NUM_STAGES = 2


def flashdeberta_prefix_pack_available() -> bool:
    """Return whether Triton-backed prefix pack kernels are importable.

    :return bool: True when Triton was imported successfully.
    """

    return _TRITON_AVAILABLE


_flatten_rows = partial(flatten_padded_rows, context="Prefix-pack")


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

    return can_use_triton_pack(
        available=flashdeberta_prefix_pack_available(),
        tensor=tensor,
        metadata_tensors=(seqlens, cu_seqlens),
    )


def _can_use_triton_prefix_rank4_pack(
    tensors: tuple[torch.Tensor, ...],
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> bool:
    """Return whether rank-4 inputs can use the stride-aware Triton copy mode.

    :param tuple[torch.Tensor, ...] tensors: Candidate padded rank-4 tensors.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative active lengths.
    :return bool: True when the inputs and metadata share one CUDA device.
    """

    if not flashdeberta_prefix_pack_available() or not tensors:
        return False
    device = tensors[0].device
    return (
        device.type == "cuda"
        and all(tensor.device == device and tensor.ndim == 4 for tensor in tensors)
        and seqlens.device == device
        and cu_seqlens.device == device
    )


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
        _can_use_triton_prefix_rank4_pack((tensor,), seqlens=seqlens, cu_seqlens=cu_seqlens)
        and not tensor.is_contiguous()
    ):
        num_heads = int(tensor.shape[2])
        feature_size = int(tensor.shape[3])
        flat_output = torch.empty(
            (total_tokens, num_heads * feature_size),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        launch_triton_row_copy(
            (tensor,),
            (flat_output,),
            offsets=seqlens,
            lengths=seqlens,
            cu_seqlens=cu_seqlens,
            address_mode=TRITON_ROW_COPY_PREFIX_PACK_STRIDED,
            seq_len=int(tensor.shape[1]),
            row_size=num_heads * feature_size,
            max_rows=int(max_seqlen),
            num_heads=num_heads,
            feature_size=feature_size,
            block_rows=_PACK_BLOCK_ROWS,
            block_cols=_PACK_BLOCK_COLS,
            num_warps=_PACK_NUM_WARPS,
            num_stages=_PACK_NUM_STAGES,
        )
        return flat_output.view(total_tokens, num_heads, feature_size)
    if flashdeberta_prefix_pack_available() and tensor.device.type == "cuda" and not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if not _can_use_triton_prefix_pack(tensor=tensor, seqlens=seqlens, cu_seqlens=cu_seqlens):
        return _prefix_pack_fallback(tensor, seqlens=seqlens, cu_seqlens=cu_seqlens)

    flat_input, trailing_shape, _batch_size, seq_len = _flatten_rows(tensor)
    row_size = int(flat_input.shape[1])
    flat_output = torch.empty((total_tokens, row_size), device=tensor.device, dtype=tensor.dtype)
    launch_triton_row_copy(
        (flat_input,),
        (flat_output,),
        offsets=seqlens,
        lengths=seqlens,
        cu_seqlens=cu_seqlens,
        address_mode=TRITON_ROW_COPY_PREFIX_PACK,
        seq_len=seq_len,
        row_size=row_size,
        max_rows=int(max_seqlen),
        block_rows=_PACK_BLOCK_ROWS,
        block_cols=_PACK_BLOCK_COLS,
        num_warps=_PACK_NUM_WARPS,
        num_stages=_PACK_NUM_STAGES,
    )
    return flat_output.view(total_tokens, *trailing_shape)


def prefix_pack_padded_rows_pair(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack two padded tensors with shared prefix metadata.

    :param torch.Tensor tensor_a: First padded tensor with leading shape ``(B, S, ...)``.
    :param torch.Tensor tensor_b: Second padded tensor with leading shape ``(B, S, ...)``.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative active lengths.
    :param int max_seqlen: Maximum active length in the batch.
    :param int | None total_tokens: Optional precomputed number of active tokens.
    :return tuple[torch.Tensor, torch.Tensor]: Packed tensors with leading shape ``(NNZ, ...)``.
    """

    shape = require_matching_tensor_layout(
        tensor_a, tensor_b, context="Prefix-pack padded tensors", minimum_rank=2
    )
    _batch_size, seq_len = int(shape[0]), int(shape[1])
    trailing_shape = shape[2:]
    if int(cu_seqlens.numel()) == 0:
        empty = tensor_a.new_empty((0, *trailing_shape))
        return empty, tensor_b.new_empty((0, *trailing_shape))
    if total_tokens is None:
        total_tokens = int(sum(int(length) for length in seqlens.detach().cpu().tolist()))
    else:
        total_tokens = int(total_tokens)
    if total_tokens == 0:
        empty = tensor_a.new_empty((0, *trailing_shape))
        return empty, tensor_b.new_empty((0, *trailing_shape))

    can_rank4 = _can_use_triton_prefix_rank4_pack(
        (tensor_a, tensor_b), seqlens=seqlens, cu_seqlens=cu_seqlens
    ) and (not tensor_a.is_contiguous() or not tensor_b.is_contiguous())
    if can_rank4:
        num_heads = int(tensor_a.shape[2])
        feature_size = int(tensor_a.shape[3])
        flat_output_a = torch.empty(
            (total_tokens, num_heads * feature_size),
            device=tensor_a.device,
            dtype=tensor_a.dtype,
        )
        flat_output_b = torch.empty(
            (total_tokens, num_heads * feature_size),
            device=tensor_b.device,
            dtype=tensor_b.dtype,
        )
        launch_triton_row_copy(
            (tensor_a, tensor_b),
            (flat_output_a, flat_output_b),
            offsets=seqlens,
            lengths=seqlens,
            cu_seqlens=cu_seqlens,
            address_mode=TRITON_ROW_COPY_PREFIX_PACK_STRIDED,
            seq_len=seq_len,
            row_size=num_heads * feature_size,
            max_rows=int(max_seqlen),
            num_heads=num_heads,
            feature_size=feature_size,
            block_rows=_PACK_BLOCK_ROWS,
            block_cols=_PACK_BLOCK_COLS,
            num_warps=_PACK_NUM_WARPS,
            num_stages=_PACK_NUM_STAGES,
        )
        return (
            flat_output_a.view(total_tokens, num_heads, feature_size),
            flat_output_b.view(total_tokens, num_heads, feature_size),
        )

    if (
        flashdeberta_prefix_pack_available()
        and tensor_a.device.type == "cuda"
        and not tensor_a.is_contiguous()
    ):
        tensor_a = tensor_a.contiguous()
    if (
        flashdeberta_prefix_pack_available()
        and tensor_b.device.type == "cuda"
        and not tensor_b.is_contiguous()
    ):
        tensor_b = tensor_b.contiguous()
    if not _can_use_triton_prefix_pack(tensor=tensor_a, seqlens=seqlens, cu_seqlens=cu_seqlens):
        return (
            _prefix_pack_fallback(tensor_a, seqlens=seqlens, cu_seqlens=cu_seqlens),
            _prefix_pack_fallback(tensor_b, seqlens=seqlens, cu_seqlens=cu_seqlens),
        )

    flat_a, _, _, _ = _flatten_rows(tensor_a)
    flat_b, _, _, _ = _flatten_rows(tensor_b)
    row_size = int(flat_a.shape[1])
    flat_output_a = torch.empty((total_tokens, row_size), device=tensor_a.device, dtype=tensor_a.dtype)
    flat_output_b = torch.empty((total_tokens, row_size), device=tensor_b.device, dtype=tensor_b.dtype)
    launch_triton_row_copy(
        (flat_a, flat_b),
        (flat_output_a, flat_output_b),
        offsets=seqlens,
        lengths=seqlens,
        cu_seqlens=cu_seqlens,
        address_mode=TRITON_ROW_COPY_PREFIX_PACK,
        seq_len=seq_len,
        row_size=row_size,
        max_rows=int(max_seqlen),
        block_rows=_PACK_BLOCK_ROWS,
        block_cols=_PACK_BLOCK_COLS,
        num_warps=_PACK_NUM_WARPS,
        num_stages=_PACK_NUM_STAGES,
    )
    return (
        flat_output_a.view(total_tokens, *trailing_shape),
        flat_output_b.view(total_tokens, *trailing_shape),
    )


def prefix_pack_padded_rows_triple(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_c: torch.Tensor,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack three padded tensors with shared prefix metadata.

    :param torch.Tensor tensor_a: First padded tensor with leading shape ``(B, S, ...)``.
    :param torch.Tensor tensor_b: Second padded tensor with leading shape ``(B, S, ...)``.
    :param torch.Tensor tensor_c: Third padded tensor with leading shape ``(B, S, ...)``.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative active lengths.
    :param int max_seqlen: Maximum active length in the batch.
    :param int | None total_tokens: Optional precomputed number of active tokens.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Packed tensors with leading shape ``(NNZ, ...)``.
    """

    shape = require_matching_tensor_layout(
        tensor_a,
        tensor_b,
        tensor_c,
        context="Prefix-pack padded tensors",
        minimum_rank=2,
    )
    _batch_size, seq_len = int(shape[0]), int(shape[1])
    trailing_shape = shape[2:]
    if int(cu_seqlens.numel()) == 0:
        empty_a = tensor_a.new_empty((0, *trailing_shape))
        empty_b = tensor_b.new_empty((0, *trailing_shape))
        empty_c = tensor_c.new_empty((0, *trailing_shape))
        return empty_a, empty_b, empty_c
    if total_tokens is None:
        total_tokens = int(sum(int(length) for length in seqlens.detach().cpu().tolist()))
    else:
        total_tokens = int(total_tokens)
    if total_tokens == 0:
        empty_a = tensor_a.new_empty((0, *trailing_shape))
        empty_b = tensor_b.new_empty((0, *trailing_shape))
        empty_c = tensor_c.new_empty((0, *trailing_shape))
        return empty_a, empty_b, empty_c

    can_rank4 = _can_use_triton_prefix_rank4_pack(
        (tensor_a, tensor_b, tensor_c), seqlens=seqlens, cu_seqlens=cu_seqlens
    ) and (not tensor_a.is_contiguous() or not tensor_b.is_contiguous() or not tensor_c.is_contiguous())
    if can_rank4:
        num_heads = int(tensor_a.shape[2])
        feature_size = int(tensor_a.shape[3])
        flat_output_a = torch.empty(
            (total_tokens, num_heads * feature_size),
            device=tensor_a.device,
            dtype=tensor_a.dtype,
        )
        flat_output_b = torch.empty(
            (total_tokens, num_heads * feature_size),
            device=tensor_b.device,
            dtype=tensor_b.dtype,
        )
        flat_output_c = torch.empty(
            (total_tokens, num_heads * feature_size),
            device=tensor_c.device,
            dtype=tensor_c.dtype,
        )
        launch_triton_row_copy(
            (tensor_a, tensor_b, tensor_c),
            (flat_output_a, flat_output_b, flat_output_c),
            offsets=seqlens,
            lengths=seqlens,
            cu_seqlens=cu_seqlens,
            address_mode=TRITON_ROW_COPY_PREFIX_PACK_STRIDED,
            seq_len=seq_len,
            row_size=num_heads * feature_size,
            max_rows=int(max_seqlen),
            num_heads=num_heads,
            feature_size=feature_size,
            block_rows=_PACK_BLOCK_ROWS,
            block_cols=_PACK_BLOCK_COLS,
            num_warps=_PACK_NUM_WARPS,
            num_stages=_PACK_NUM_STAGES,
        )
        return (
            flat_output_a.view(total_tokens, num_heads, feature_size),
            flat_output_b.view(total_tokens, num_heads, feature_size),
            flat_output_c.view(total_tokens, num_heads, feature_size),
        )

    if (
        flashdeberta_prefix_pack_available()
        and tensor_a.device.type == "cuda"
        and not tensor_a.is_contiguous()
    ):
        tensor_a = tensor_a.contiguous()
    if (
        flashdeberta_prefix_pack_available()
        and tensor_b.device.type == "cuda"
        and not tensor_b.is_contiguous()
    ):
        tensor_b = tensor_b.contiguous()
    if (
        flashdeberta_prefix_pack_available()
        and tensor_c.device.type == "cuda"
        and not tensor_c.is_contiguous()
    ):
        tensor_c = tensor_c.contiguous()
    if not _can_use_triton_prefix_pack(tensor=tensor_a, seqlens=seqlens, cu_seqlens=cu_seqlens):
        return (
            _prefix_pack_fallback(tensor_a, seqlens=seqlens, cu_seqlens=cu_seqlens),
            _prefix_pack_fallback(tensor_b, seqlens=seqlens, cu_seqlens=cu_seqlens),
            _prefix_pack_fallback(tensor_c, seqlens=seqlens, cu_seqlens=cu_seqlens),
        )

    flat_a, _, _, _ = _flatten_rows(tensor_a)
    flat_b, _, _, _ = _flatten_rows(tensor_b)
    flat_c, _, _, _ = _flatten_rows(tensor_c)
    row_size = int(flat_a.shape[1])
    flat_output_a = torch.empty((total_tokens, row_size), device=tensor_a.device, dtype=tensor_a.dtype)
    flat_output_b = torch.empty((total_tokens, row_size), device=tensor_b.device, dtype=tensor_b.dtype)
    flat_output_c = torch.empty((total_tokens, row_size), device=tensor_c.device, dtype=tensor_c.dtype)
    launch_triton_row_copy(
        (flat_a, flat_b, flat_c),
        (flat_output_a, flat_output_b, flat_output_c),
        offsets=seqlens,
        lengths=seqlens,
        cu_seqlens=cu_seqlens,
        address_mode=TRITON_ROW_COPY_PREFIX_PACK,
        seq_len=seq_len,
        row_size=row_size,
        max_rows=int(max_seqlen),
        block_rows=_PACK_BLOCK_ROWS,
        block_cols=_PACK_BLOCK_COLS,
        num_warps=_PACK_NUM_WARPS,
        num_stages=_PACK_NUM_STAGES,
    )
    return (
        flat_output_a.view(total_tokens, *trailing_shape),
        flat_output_b.view(total_tokens, *trailing_shape),
        flat_output_c.view(total_tokens, *trailing_shape),
    )


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
    launch_triton_row_copy(
        (flat_values,),
        (flat_output,),
        offsets=seqlens,
        lengths=seqlens,
        cu_seqlens=cu_seqlens,
        address_mode=TRITON_ROW_COPY_PREFIX_UNPACK,
        seq_len=seq_len,
        row_size=row_size,
        max_rows=seq_len,
        block_rows=_PACK_BLOCK_ROWS,
        block_cols=_PACK_BLOCK_COLS,
        num_warps=_PACK_NUM_WARPS,
        num_stages=_PACK_NUM_STAGES,
    )
    return flat_output.view(batch_size, seq_len, *trailing_shape)


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

    shape = require_matching_tensor_layout(values_a, values_b, context="Prefix-pack packed tensors")
    trailing_shape = shape[1:]
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
    launch_triton_row_copy(
        (flat_values_a, flat_values_b),
        (flat_output_a, flat_output_b),
        offsets=seqlens,
        lengths=seqlens,
        cu_seqlens=cu_seqlens,
        address_mode=TRITON_ROW_COPY_PREFIX_UNPACK,
        seq_len=seq_len,
        row_size=row_size,
        max_rows=seq_len,
        block_rows=_PACK_BLOCK_ROWS,
        block_cols=_PACK_BLOCK_COLS,
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

    shape = require_matching_tensor_layout(values_a, values_b, values_c, context="Prefix-pack packed tensors")
    trailing_shape = shape[1:]
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
    launch_triton_row_copy(
        (flat_values_a, flat_values_b, flat_values_c),
        (flat_output_a, flat_output_b, flat_output_c),
        offsets=seqlens,
        lengths=seqlens,
        cu_seqlens=cu_seqlens,
        address_mode=TRITON_ROW_COPY_PREFIX_UNPACK,
        seq_len=seq_len,
        row_size=row_size,
        max_rows=seq_len,
        block_rows=_PACK_BLOCK_ROWS,
        block_cols=_PACK_BLOCK_COLS,
        num_warps=_PACK_NUM_WARPS,
        num_stages=_PACK_NUM_STAGES,
    )
    return (
        flat_output_a.view(batch_size, seq_len, *trailing_shape),
        flat_output_b.view(batch_size, seq_len, *trailing_shape),
        flat_output_c.view(batch_size, seq_len, *trailing_shape),
    )


def prefix_pack_optional_pair(
    tensor_a: torch.Tensor | None,
    tensor_b: torch.Tensor | None,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_tokens: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Pack an optional tensor pair, sharing one launch when both are present.

    :param torch.Tensor | None tensor_a: Optional first padded tensor.
    :param torch.Tensor | None tensor_b: Optional second padded tensor.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative sequence lengths.
    :param int max_seqlen: Maximum active length.
    :param int total_tokens: Total packed token count.
    :return tuple[torch.Tensor | None, torch.Tensor | None]: Packed tensors, or
        ``None`` for absent inputs.
    """

    if tensor_a is not None and tensor_b is not None:
        return prefix_pack_padded_rows_pair(
            tensor_a,
            tensor_b,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            total_tokens=total_tokens,
        )
    packed = [
        prefix_pack_padded_rows(
            tensor,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            total_tokens=total_tokens,
        )
        if tensor is not None
        else None
        for tensor in (tensor_a, tensor_b)
    ]
    return packed[0], packed[1]


def prefix_unpack_optional_pair(
    tensor_a: torch.Tensor | None,
    tensor_b: torch.Tensor | None,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Unpack an optional tensor pair, sharing one launch when both are present.

    :param torch.Tensor | None tensor_a: Optional first packed tensor.
    :param torch.Tensor | None tensor_b: Optional second packed tensor.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative sequence lengths.
    :param int batch_size: Padded batch size.
    :param int seq_len: Padded sequence length.
    :return tuple[torch.Tensor | None, torch.Tensor | None]: Padded tensors, or
        ``None`` for absent inputs.
    """

    if tensor_a is not None and tensor_b is not None:
        return prefix_unpack_padded_rows_pair(
            tensor_a,
            tensor_b,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            batch_size=batch_size,
            seq_len=seq_len,
        )
    unpacked = [
        prefix_unpack_padded_rows(
            tensor,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        if tensor is not None
        else None
        for tensor in (tensor_a, tensor_b)
    ]
    return unpacked[0], unpacked[1]


__all__ = [
    "flashdeberta_prefix_pack_available",
    "prefix_pack_optional_pair",
    "prefix_pack_padded_rows",
    "prefix_pack_padded_rows_pair",
    "prefix_pack_padded_rows_triple",
    "prefix_unpack_optional_pair",
    "prefix_unpack_padded_rows",
    "prefix_unpack_padded_rows_pair",
    "prefix_unpack_padded_rows_triple",
]
