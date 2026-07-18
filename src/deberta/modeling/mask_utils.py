"""Attention-mask normalization helpers for model/runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def is_torch_compiling() -> bool:
    """Return whether execution is happening under ``torch.compile``.

    :return bool: True when inside compiled/traced execution.
    """

    return bool(torch.compiler.is_compiling())


@dataclass(frozen=True)
class FlashBatchMeta:
    """Batch-scoped FlashDeBERTa metadata.

    :param torch.Tensor | None seq_lengths: Optional per-example active prefix lengths.
        Right-padding contract: position ``i`` is active iff ``i < length``, so
        producers must derive and reconcile lengths with
        :func:`build_validated_prefix_lengths` before publishing them here.
    :param torch.Tensor | None doc_segment_offsets: Optional flat padded row offsets per doc segment.
    :param torch.Tensor | None doc_segment_lengths: Optional per-segment doc lengths.
    :param torch.Tensor | None doc_cu_seqlens: Optional cumulative packed doc offsets.
    :param torch.Tensor | None doc_ids: Optional compact document ids for eager fallback.
    :param torch.Tensor | None active_tokens_scalar: Optional CPU scalar active-token tensor for compiled routes.
    :param torch.Tensor | None doc_num_segments_scalar: Optional CPU scalar segment-count tensor for compiled routes.
    :param torch.Tensor | None doc_max_segment_length_scalar: Optional CPU scalar max-segment tensor for compiled routes.
    :param str | None route_hint: Optional flash route hint.
    """

    seq_lengths: torch.Tensor | None = None
    doc_segment_offsets: torch.Tensor | None = None
    doc_segment_lengths: torch.Tensor | None = None
    doc_cu_seqlens: torch.Tensor | None = None
    doc_ids: torch.Tensor | None = None
    active_tokens_scalar: torch.Tensor | None = None
    doc_num_segments_scalar: torch.Tensor | None = None
    doc_max_segment_length_scalar: torch.Tensor | None = None
    route_hint: str | None = None

    def is_cross_document(self) -> bool:
        """Return whether this batch carries packed cross-document semantics.

        This is the single source of truth for "is this a doc-block batch".
        Consumers that mix information across sequence positions (for example
        per-document CLS conditioning) must check it before doing so, because
        doc-block batches ship a compact 2D keep mask whose pairwise
        document-blocking semantics live in this metadata instead.

        :return bool: True when the batch routes through a doc-block path or ships doc-segment metadata.
        """

        return (
            self.route_hint in {"docblock", "docblock_bias"}
            or self.doc_ids is not None
            or self.doc_segment_offsets is not None
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> FlashBatchMeta:
        """Move kernel-consumed metadata tensors while keeping scalar tensors on CPU.

        :param torch.device | str device: Destination device.
        :param bool non_blocking: Whether tensor copies may proceed asynchronously.
        :return FlashBatchMeta: Metadata bundle for the destination device.
        """

        def _move(value: torch.Tensor | None) -> torch.Tensor | None:
            """Move one optional kernel metadata tensor.

            :param torch.Tensor | None value: Optional source tensor.
            :return torch.Tensor | None: Tensor on the destination device.
            """

            return (
                value.to(device=device, non_blocking=non_blocking)
                if isinstance(value, torch.Tensor)
                else None
            )

        return replace(
            self,
            seq_lengths=_move(self.seq_lengths),
            doc_segment_offsets=_move(self.doc_segment_offsets),
            doc_segment_lengths=_move(self.doc_segment_lengths),
            doc_cu_seqlens=_move(self.doc_cu_seqlens),
            doc_ids=_move(self.doc_ids),
        )


def normalize_keep_mask(mask: torch.Tensor, *, name: str = "attention_mask") -> torch.Tensor:
    """Normalize a keep-mask tensor to boolean without lossy float coercion.

    Keep-mask contract in this repo:
    - ``True`` / non-zero integer values mean "keep".
    - ``False`` / zero integer values mean "masked".
    - Floating-point masks are rejected to avoid ambiguous semantics
      (for example 0/1 keep masks vs additive 0/-inf masks).

    :param torch.Tensor mask: Raw mask tensor.
    :param str name: Name used in validation errors.
    :raises TypeError: If ``mask`` is not a tensor.
    :raises ValueError: If ``mask`` is floating-point.
    :return torch.Tensor: Boolean keep mask with the same shape.
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(mask)!r}.")
    if mask.dtype == torch.bool:
        return mask
    if torch.is_floating_point(mask):
        raise ValueError(
            f"{name} must be a bool/integer keep-mask tensor. "
            "Floating-point masks are ambiguous (0/1 keep vs 0/-inf additive)."
        )
    return mask.ne(0)


def mask_to_2d_keep_mask(attention_mask: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Extract a canonical ``(B,S)`` keep mask from rank-2/4 padding masks.

    Pairwise masks are rejected on purpose: callers that support per-query
    structure must branch on :func:`is_pairwise_mask` first. Length mismatches
    raise instead of slicing: the flash routes reinterpret this mask against
    the full sequence, so silent truncation or a too-short mask would change
    which positions count as active.

    :param torch.Tensor attention_mask: Padding-style keep mask.
    :param int seq_len: Expected sequence length.
    :raises ValueError: If the mask is not an exact-length 2D or broadcast 4D mask.
    :return torch.Tensor: Boolean keep mask in ``(B,S)`` layout.
    """

    mask = normalize_keep_mask(attention_mask)
    expected = int(seq_len)
    if mask.ndim == 2:
        if int(mask.shape[-1]) != expected:
            raise ValueError(
                f"Padding mask key length must exactly match seq_len={expected}; "
                f"got shape={tuple(mask.shape)}"
            )
        return mask
    if mask.ndim == 4 and int(mask.shape[1]) == 1 and int(mask.shape[-2]) == 1:
        if int(mask.shape[-1]) != expected:
            raise ValueError(
                f"Broadcast padding mask key length must exactly match seq_len={expected}; "
                f"got shape={tuple(mask.shape)}"
            )
        return mask[:, 0, 0, :]
    raise ValueError(
        f"Padding masks must be shaped (B,S) or (B,1,1,S) for 2D keep-mask extraction; "
        f"got shape={tuple(mask.shape)}"
    )


def build_validated_prefix_lengths(
    attention_mask: torch.Tensor,
    *,
    seq_len: int,
    supplied_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Derive and validate right-padded prefix lengths on CPU.

    ``attention_mask`` is authoritative. Optional precomputed lengths are only
    accepted after exact reconciliation with the mask; they never override it.
    This boundary helper intentionally rejects device tensors so validation
    cannot introduce a hidden CUDA synchronization in a compiled attention
    layer.

    :param torch.Tensor attention_mask: CPU padding keep mask.
    :param int seq_len: Exact sequence length.
    :param torch.Tensor | None supplied_lengths: Optional CPU lengths to verify.
    :raises ValueError: If tensors are not CPU-resident, the mask is not a
        right-padded prefix, or supplied lengths disagree.
    :return torch.Tensor: Mask-derived CPU int32 lengths with shape ``(B,)``.
    """

    if attention_mask.device.type != "cpu":
        raise ValueError("Flash padding metadata must be validated on CPU before device transfer.")
    keep_mask = mask_to_2d_keep_mask(attention_mask, seq_len=seq_len)
    lengths = keep_mask.sum(dim=-1, dtype=torch.int32)
    positions = torch.arange(int(seq_len), dtype=torch.int32)
    expected_keep_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
    if not torch.equal(keep_mask, expected_keep_mask):
        raise ValueError("Flash fixed/varlen attention requires a right-padded prefix mask.")

    if supplied_lengths is not None:
        if supplied_lengths.device.type != "cpu":
            raise ValueError(
                "Supplied flash sequence lengths must be validated on CPU before device transfer."
            )
        if supplied_lengths.dtype not in _INTEGER_DTYPES:
            raise ValueError(
                "Supplied flash sequence lengths disagree with their contract: integer dtype required."
            )
        supplied = supplied_lengths.to(dtype=torch.int32)
        if supplied.ndim != 1 or supplied.shape != lengths.shape or not torch.equal(supplied, lengths):
            raise ValueError(
                "Supplied flash sequence lengths disagree with attention_mask: "
                f"derived={lengths.tolist()}, supplied={supplied.tolist()}."
            )
    return lengths


def is_pairwise_mask(attention_mask: torch.Tensor, *, query_len: int, key_len: int) -> bool:
    """Return whether a mask encodes per-query pairwise constraints.

    :param torch.Tensor attention_mask: Candidate mask tensor.
    :param int query_len: Expected query length.
    :param int key_len: Expected key length.
    :return bool: True for ``(B,Q,K)`` or ``(B,*,Q,K)`` style masks.
    """

    mask = normalize_keep_mask(attention_mask)
    if mask.ndim in (3, 4):
        return tuple(mask.shape[-2:]) == (int(query_len), int(key_len))
    return False


def expand_keep_mask_to_4d(
    attention_mask: torch.Tensor,
    *,
    pairwise_2d: bool = False,
    collapse_heads: bool = True,
) -> torch.Tensor:
    """Expand a rank-2/3/4 keep mask to the canonical 4D attention layout.

    2D key-padding masks become broadcast ``(B,1,1,S)`` by default, or a full
    outer-product ``(B,1,S,S)`` pairwise mask when ``pairwise_2d`` is set (the
    original DeBERTa EMD convention). 3D pairwise masks gain a head axis and 4D
    masks are head-reduced to ``(B,1,*,S)`` by default. Callers that support
    genuine per-head constraints can preserve the head axis explicitly.

    :param torch.Tensor attention_mask: Keep mask in rank-2/3/4 layout.
    :param bool pairwise_2d: Whether 2D masks expand via outer product.
    :param bool collapse_heads: Whether multi-head 4D masks are OR-reduced.
    :raises ValueError: If the mask rank is unsupported.
    :return torch.Tensor: Boolean keep mask in ``(B,1,1,S)`` or ``(B,1,S,S)`` layout.
    """

    mask = normalize_keep_mask(attention_mask)
    if mask.ndim == 2:
        ext = mask[:, None, None, :]
        if pairwise_2d:
            return ext & ext.transpose(-1, -2)
        return ext
    if mask.ndim == 3:
        return mask[:, None, :, :]
    if mask.ndim == 4:
        if mask.shape[1] == 1 or not collapse_heads:
            return mask
        return mask.any(dim=1, keepdim=True)
    raise ValueError(f"attention_mask must be rank-2/3/4; got rank={mask.ndim}")


def reduce_keep_mask_to_2d(attention_mask: torch.Tensor, *, seq_len: int | None = None) -> torch.Tensor:
    """Reduce a rank-2/3/4 keep mask to per-token ``(B,S)`` activity.

    Pairwise masks contribute their diagonal (the diagonal encodes per-query
    activity); broadcast key-padding rows keep the full sequence axis; 4D masks
    are head-reduced first.

    :param torch.Tensor attention_mask: Keep mask in rank-2/3/4 layout.
    :param int | None seq_len: Optional sequence length to slice the key axis to.
    :raises ValueError: If the mask rank is unsupported.
    :return torch.Tensor: Boolean keep mask in ``(B,S)`` layout.
    """

    mask = normalize_keep_mask(attention_mask)
    if mask.ndim == 4:
        mask = mask[:, 0] if mask.shape[1] == 1 else mask.any(dim=1)
    if mask.ndim == 3:
        if mask.shape[-2] == 1:
            # Broadcast padding path: (B,1,S) keeps the full sequence axis.
            mask = mask[:, 0, :]
        else:
            mask = torch.diagonal(mask, dim1=-2, dim2=-1)
    if mask.ndim != 2:
        raise ValueError(f"attention_mask must be rank-2/3/4; got rank={mask.ndim}")
    if seq_len is not None and int(mask.shape[-1]) != int(seq_len):
        mask = mask[:, : int(seq_len)]
    return mask


def build_doc_block_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Build a dense pairwise keep mask from compact document ids.

    :param torch.Tensor doc_ids: Document id tensor ``(B,S)`` with ``0`` for padding.
    :raises ValueError: If ``doc_ids`` is not rank 2.
    :return torch.Tensor: Boolean keep mask ``(B,S,S)``.
    """

    if doc_ids.ndim != 2:
        raise ValueError(f"doc_ids must be rank-2 (B,S); got shape={tuple(doc_ids.shape)}")

    ids = doc_ids.to(dtype=torch.long)
    active = ids.ne(0)
    same_doc = ids[:, :, None].eq(ids[:, None, :])
    keep = same_doc & active[:, :, None] & active[:, None, :]

    seq_len = int(ids.shape[1])
    cls_key = torch.zeros((seq_len,), dtype=torch.bool, device=ids.device)
    if seq_len > 0:
        cls_key[0] = True

    keep = keep | ((~active)[:, :, None] & cls_key[None, None, :])

    return keep


def build_doc_segment_metadata(
    doc_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build contiguous document-segment metadata from compact ``doc_ids``.

    :param torch.Tensor doc_ids: Document id tensor ``(B,S)`` with ``0`` for padding.
    :raises ValueError: If ``doc_ids`` is not rank 2.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]: Flat padded segment offsets,
        segment lengths, cumulative packed offsets, and total active tokens.
    """

    if doc_ids.ndim != 2:
        raise ValueError(f"doc_ids must be rank-2 (B,S); got shape={tuple(doc_ids.shape)}")

    batch_size, seq_len = int(doc_ids.shape[0]), int(doc_ids.shape[1])
    max_segments = max(1, batch_size * seq_len)
    segment_offsets_padded = torch.zeros((max_segments,), device=doc_ids.device, dtype=torch.int32)
    segment_lengths_padded = torch.zeros((max_segments,), device=doc_ids.device, dtype=torch.int32)
    cu_seqlens_padded = torch.zeros((max_segments + 1,), device=doc_ids.device, dtype=torch.int32)

    active = doc_ids.ne(0)
    if not bool(active.any().item()):
        return segment_offsets_padded, segment_lengths_padded, cu_seqlens_padded, 0

    prev = torch.zeros_like(doc_ids)
    prev[:, 1:] = doc_ids[:, :-1]
    next_ids = torch.zeros_like(doc_ids)
    next_ids[:, :-1] = doc_ids[:, 1:]

    start_mask = active & doc_ids.ne(prev)
    end_mask = active & doc_ids.ne(next_ids)

    start_idx = start_mask.nonzero(as_tuple=False)
    end_idx = end_mask.nonzero(as_tuple=False)
    if int(start_idx.shape[0]) != int(end_idx.shape[0]):
        raise RuntimeError("doc-block segment boundary count mismatch.")

    segment_starts = start_idx[:, 1].to(dtype=torch.int32)
    segment_ends = end_idx[:, 1].to(dtype=torch.int32)
    segment_lengths = (segment_ends - segment_starts + 1).to(dtype=torch.int32)
    batch_rows = start_idx[:, 0].to(dtype=torch.int32)
    segment_offsets = batch_rows * int(seq_len) + segment_starts
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(segment_lengths, dim=0, dtype=torch.int32),
        (1, 0),
    )
    num_segments = int(segment_lengths.shape[0])
    segment_offsets_padded[:num_segments] = segment_offsets
    segment_lengths_padded[:num_segments] = segment_lengths
    cu_seqlens_padded[: num_segments + 1] = cu_seqlens
    total_tokens = int(cu_seqlens[-1].item())
    return segment_offsets_padded, segment_lengths_padded, cu_seqlens_padded, total_tokens


__all__ = [
    "FlashBatchMeta",
    "build_validated_prefix_lengths",
    "build_doc_block_mask",
    "build_doc_segment_metadata",
    "expand_keep_mask_to_4d",
    "is_pairwise_mask",
    "mask_to_2d_keep_mask",
    "normalize_keep_mask",
    "reduce_keep_mask_to_2d",
]
