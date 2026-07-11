"""Prefix-layout adapters for the unified segment pack primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from deberta.modeling.flashdeberta_segment_pack import (
    flashdeberta_segment_pack_available,
    segment_pack_optional_pair,
    segment_pack_padded_rows,
    segment_pack_padded_rows_pair,
    segment_pack_padded_rows_triple,
    segment_unpack_optional_pair,
    segment_unpack_padded_rows,
    segment_unpack_padded_rows_pair,
    segment_unpack_padded_rows_triple,
)


def flashdeberta_prefix_pack_available() -> bool:
    """Return whether the unified Triton row-copy primitives are available."""

    return flashdeberta_segment_pack_available()


def _prefix_segment_metadata(
    *,
    seqlens: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Represent batch prefixes as segments with implicit row offsets.

    :param torch.Tensor seqlens: Per-example active prefix lengths.
    :param int seq_len: Padded sequence length.
    :return tuple[torch.Tensor, torch.Tensor]: Segment offsets and lengths.
    """

    batch_size = int(seqlens.numel())
    offsets = torch.arange(batch_size, device=seqlens.device, dtype=seqlens.dtype)
    return offsets.mul_(int(seq_len)), seqlens


def _resolve_total_tokens(seqlens: torch.Tensor, total_tokens: int | None) -> int:
    """Resolve the packed capacity while preserving the existing optional API.

    :param torch.Tensor seqlens: Per-example prefix lengths.
    :param int | None total_tokens: Optional precomputed total.
    :return int: Packed token count.
    """

    if total_tokens is not None:
        return int(total_tokens)
    return sum(int(length) for length in seqlens.detach().cpu().tolist())


def _pack(
    operation: Callable[..., Any],
    *tensors: torch.Tensor,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_tokens: int | None,
) -> Any:
    """Delegate one prefix pack operation to the segment implementation.

    :param Callable[..., Any] operation: Segment pack callable.
    :param torch.Tensor tensors: Padded input tensors.
    :param torch.Tensor seqlens: Per-example prefix lengths.
    :param torch.Tensor cu_seqlens: Packed cumulative lengths.
    :param int max_seqlen: Maximum active prefix length.
    :param int | None total_tokens: Optional packed token count.
    :return Any: Packed tensor or tensor tuple.
    """

    seq_len = int(tensors[0].shape[1])
    offsets, lengths = _prefix_segment_metadata(seqlens=seqlens, seq_len=seq_len)
    return operation(
        *tensors,
        segment_offsets=offsets,
        segment_lengths=lengths,
        cu_seqlens=cu_seqlens,
        total_tokens=_resolve_total_tokens(seqlens, total_tokens),
        max_segment_length=int(max_seqlen),
    )


def _unpack(
    operation: Callable[..., Any],
    *tensors: torch.Tensor,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> Any:
    """Delegate one prefix unpack operation to the segment implementation.

    :param Callable[..., Any] operation: Segment unpack callable.
    :param torch.Tensor tensors: Packed input tensors.
    :param torch.Tensor seqlens: Per-example prefix lengths.
    :param torch.Tensor cu_seqlens: Packed cumulative lengths.
    :param int batch_size: Padded batch size.
    :param int seq_len: Padded sequence length.
    :return Any: Padded tensor or tensor tuple.
    """

    offsets, lengths = _prefix_segment_metadata(seqlens=seqlens, seq_len=seq_len)
    return operation(
        *tensors,
        segment_offsets=offsets,
        segment_lengths=lengths,
        cu_seqlens=cu_seqlens,
        batch_size=int(batch_size),
        seq_len=int(seq_len),
        max_segment_length=int(seq_len),
    )


def prefix_pack_padded_rows(
    tensor: torch.Tensor,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_tokens: int | None = None,
) -> torch.Tensor:
    """Pack active prefix rows from one padded tensor.

    :param torch.Tensor tensor: Padded input tensor.
    :param torch.Tensor seqlens: Per-example prefix lengths.
    :param torch.Tensor cu_seqlens: Packed cumulative lengths.
    :param int max_seqlen: Maximum active prefix length.
    :param int | None total_tokens: Optional packed token count.
    :return torch.Tensor: Packed tensor.
    """

    return _pack(
        segment_pack_padded_rows,
        tensor,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        total_tokens=total_tokens,
    )


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

    :param torch.Tensor tensor_a: First padded tensor.
    :param torch.Tensor tensor_b: Second padded tensor.
    :param torch.Tensor seqlens: Per-example prefix lengths.
    :param torch.Tensor cu_seqlens: Packed cumulative lengths.
    :param int max_seqlen: Maximum active prefix length.
    :param int | None total_tokens: Optional packed token count.
    :return tuple[torch.Tensor, torch.Tensor]: Packed tensors.
    """

    return _pack(
        segment_pack_padded_rows_pair,
        tensor_a,
        tensor_b,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        total_tokens=total_tokens,
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

    :param torch.Tensor tensor_a: First padded tensor.
    :param torch.Tensor tensor_b: Second padded tensor.
    :param torch.Tensor tensor_c: Third padded tensor.
    :param torch.Tensor seqlens: Per-example prefix lengths.
    :param torch.Tensor cu_seqlens: Packed cumulative lengths.
    :param int max_seqlen: Maximum active prefix length.
    :param int | None total_tokens: Optional packed token count.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Packed tensors.
    """

    return _pack(
        segment_pack_padded_rows_triple,
        tensor_a,
        tensor_b,
        tensor_c,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        total_tokens=total_tokens,
    )


def prefix_unpack_padded_rows(
    packed: torch.Tensor,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Scatter one packed tensor into prefix-padded layout.

    :param torch.Tensor packed: Packed input tensor.
    :param torch.Tensor seqlens: Per-example prefix lengths.
    :param torch.Tensor cu_seqlens: Packed cumulative lengths.
    :param int batch_size: Padded batch size.
    :param int seq_len: Padded sequence length.
    :return torch.Tensor: Padded tensor.
    """

    return _unpack(
        segment_unpack_padded_rows,
        packed,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
    )


def prefix_unpack_padded_rows_pair(
    packed_a: torch.Tensor,
    packed_b: torch.Tensor,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter two packed tensors into prefix-padded layout.

    :param torch.Tensor packed_a: First packed tensor.
    :param torch.Tensor packed_b: Second packed tensor.
    :param torch.Tensor seqlens: Per-example prefix lengths.
    :param torch.Tensor cu_seqlens: Packed cumulative lengths.
    :param int batch_size: Padded batch size.
    :param int seq_len: Padded sequence length.
    :return tuple[torch.Tensor, torch.Tensor]: Padded tensors.
    """

    return _unpack(
        segment_unpack_padded_rows_pair,
        packed_a,
        packed_b,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
    )


def prefix_unpack_padded_rows_triple(
    packed_a: torch.Tensor,
    packed_b: torch.Tensor,
    packed_c: torch.Tensor,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter three packed tensors into prefix-padded layout.

    :param torch.Tensor packed_a: First packed tensor.
    :param torch.Tensor packed_b: Second packed tensor.
    :param torch.Tensor packed_c: Third packed tensor.
    :param torch.Tensor seqlens: Per-example prefix lengths.
    :param torch.Tensor cu_seqlens: Packed cumulative lengths.
    :param int batch_size: Padded batch size.
    :param int seq_len: Padded sequence length.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Padded tensors.
    """

    return _unpack(
        segment_unpack_padded_rows_triple,
        packed_a,
        packed_b,
        packed_c,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
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
    """Pack an optional tensor pair with shared prefix metadata.

    :param torch.Tensor | None tensor_a: Optional first padded tensor.
    :param torch.Tensor | None tensor_b: Optional second padded tensor.
    :param torch.Tensor seqlens: Per-example prefix lengths.
    :param torch.Tensor cu_seqlens: Packed cumulative lengths.
    :param int max_seqlen: Maximum active prefix length.
    :param int total_tokens: Packed token count.
    :return tuple[torch.Tensor | None, torch.Tensor | None]: Optional packed tensors.
    """

    reference = tensor_a if tensor_a is not None else tensor_b
    if reference is None:
        return None, None
    offsets, lengths = _prefix_segment_metadata(seqlens=seqlens, seq_len=int(reference.shape[1]))
    return segment_pack_optional_pair(
        tensor_a,
        tensor_b,
        segment_offsets=offsets,
        segment_lengths=lengths,
        cu_seqlens=cu_seqlens,
        total_tokens=int(total_tokens),
        max_segment_length=int(max_seqlen),
    )


def prefix_unpack_optional_pair(
    tensor_a: torch.Tensor | None,
    tensor_b: torch.Tensor | None,
    *,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Unpack an optional tensor pair with shared prefix metadata.

    :param torch.Tensor | None tensor_a: Optional first packed tensor.
    :param torch.Tensor | None tensor_b: Optional second packed tensor.
    :param torch.Tensor seqlens: Per-example prefix lengths.
    :param torch.Tensor cu_seqlens: Packed cumulative lengths.
    :param int batch_size: Padded batch size.
    :param int seq_len: Padded sequence length.
    :return tuple[torch.Tensor | None, torch.Tensor | None]: Optional padded tensors.
    """

    if tensor_a is None and tensor_b is None:
        return None, None
    offsets, lengths = _prefix_segment_metadata(seqlens=seqlens, seq_len=seq_len)
    return segment_unpack_optional_pair(
        tensor_a,
        tensor_b,
        segment_offsets=offsets,
        segment_lengths=lengths,
        cu_seqlens=cu_seqlens,
        batch_size=int(batch_size),
        seq_len=int(seq_len),
        max_segment_length=int(seq_len),
    )


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
