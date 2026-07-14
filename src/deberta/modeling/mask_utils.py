"""Attention-mask normalization helpers for model/runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal

import torch

_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def is_torch_compiling() -> bool:
    """Return whether execution is happening under ``torch.compile``.

    :return bool: True when inside compiled/traced execution.
    """

    if not hasattr(torch, "compiler") or not hasattr(torch.compiler, "is_compiling"):
        return False
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


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
    :param int | None active_tokens_host: Optional host-side active token count.
    :param int | None doc_num_segments_host: Optional host-side active doc segment count.
    :param int | None doc_max_segment_length_host: Optional host-side max doc segment length.
    :param torch.Tensor | None active_tokens_scalar: Optional CPU scalar active-token tensor for compiled routes.
    :param torch.Tensor | None doc_num_segments_scalar: Optional CPU scalar segment-count tensor for compiled routes.
    :param torch.Tensor | None doc_max_segment_length_scalar: Optional CPU scalar max-segment tensor for compiled routes.
    :param str | None route_hint: Optional normalized flash route hint.
    :param Literal["all_active", "prefix", "docblock"] mask_contract: Static mask
        contract proven by the batch metadata producer.
    :param bool mask_contract_validated: Whether the mask and all consumed metadata
        were reconciled before entering the model. Direct construction leaves this
        false; normal training metadata preparation is responsible for attestation.
    """

    seq_lengths: torch.Tensor | None = None
    doc_segment_offsets: torch.Tensor | None = None
    doc_segment_lengths: torch.Tensor | None = None
    doc_cu_seqlens: torch.Tensor | None = None
    active_tokens_host: int | None = None
    doc_num_segments_host: int | None = None
    doc_max_segment_length_host: int | None = None
    active_tokens_scalar: torch.Tensor | None = None
    doc_num_segments_scalar: torch.Tensor | None = None
    doc_max_segment_length_scalar: torch.Tensor | None = None
    route_hint: str | None = None
    mask_contract: Literal["all_active", "prefix", "docblock"] = "all_active"
    mask_contract_validated: bool = False

    def normalized_route_hint(self) -> str | None:
        """Return the normalized route hint.

        :return str | None: Lowercase route hint or ``None``.
        """

        if self.route_hint is None:
            return None
        text = str(self.route_hint).strip().lower()
        return text if text else None

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
            self.normalized_route_hint() in {"docblock", "docblock_bias"}
            or self.doc_segment_offsets is not None
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> FlashBatchMeta:
        """Move kernel-consumed metadata tensors while keeping host scalars on CPU.

        Production training constructs metadata after moving each batch, so
        this transfer helper primarily serves standalone tools that prepare
        metadata before selecting their execution device.

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


def validate_doc_segments_against_mask(
    *,
    attention_mask: torch.Tensor,
    segment_offsets: torch.Tensor,
    segment_lengths: torch.Tensor,
    seq_len: int,
    doc_ids: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> None:
    """Validate flat doc-segment descriptors against authoritative CPU masks.

    The repo stores offsets as flat ``batch_index * seq_len + token_index``
    values rather than a padded ``(B,max_segments)`` matrix. Active descriptors
    must form one positive-length prefix, cover every active position exactly
    once, stay within a single batch row, and agree with document boundaries.

    :param torch.Tensor attention_mask: CPU padding keep mask in ``(B,S)`` form.
    :param torch.Tensor segment_offsets: Flat CPU segment offsets.
    :param torch.Tensor segment_lengths: Flat CPU segment lengths with zero tail padding.
    :param int seq_len: Exact sequence length.
    :param torch.Tensor | None doc_ids: Optional authoritative document ids.
    :param torch.Tensor | None cu_seqlens: Optional padded cumulative lengths.
    :raises ValueError: If descriptors are malformed or disagree with the mask/layout.
    """

    tensors = {
        "attention_mask": attention_mask,
        "segment_offsets": segment_offsets,
        "segment_lengths": segment_lengths,
    }
    if doc_ids is not None:
        tensors["doc_ids"] = doc_ids
    if cu_seqlens is not None:
        tensors["cu_seqlens"] = cu_seqlens
    for name, tensor in tensors.items():
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be validated on CPU before document metadata device transfer.")

    keep_mask = mask_to_2d_keep_mask(attention_mask, seq_len=seq_len)
    if segment_offsets.ndim != 1 or segment_lengths.ndim != 1:
        raise ValueError(
            "Document segment offsets and lengths must be flat 1D tensors; "
            f"got offsets={tuple(segment_offsets.shape)}, lengths={tuple(segment_lengths.shape)}."
        )
    if segment_offsets.shape != segment_lengths.shape:
        raise ValueError(
            "Document segment offsets and lengths must have identical shapes; "
            f"got offsets={tuple(segment_offsets.shape)}, lengths={tuple(segment_lengths.shape)}."
        )
    if segment_offsets.dtype not in _INTEGER_DTYPES or segment_lengths.dtype not in _INTEGER_DTYPES:
        raise ValueError("Document segment offsets and lengths must use integer dtypes.")
    if cu_seqlens is not None and cu_seqlens.dtype not in _INTEGER_DTYPES:
        raise ValueError("Document cumulative lengths must use an integer dtype.")

    batch_size = int(keep_mask.shape[0])
    offsets = segment_offsets.to(dtype=torch.int64)
    lengths = segment_lengths.to(dtype=torch.int64)
    covered = torch.zeros_like(keep_mask, dtype=torch.bool)
    if bool((offsets < 0).any().item()) or bool((lengths < 0).any().item()):
        raise ValueError("Document segment offsets and lengths must be non-negative.")
    positive = lengths > 0
    active_segments = int(positive.sum().item())
    expected_positive = torch.arange(int(lengths.numel())) < active_segments
    if not torch.equal(positive, expected_positive):
        raise ValueError("Positive document segments must precede zero-padded descriptors.")
    if bool(offsets[active_segments:].ne(0).any().item()):
        raise ValueError("Zero-length document descriptors must have offset=0.")
    ids = None
    if doc_ids is not None:
        if doc_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("doc_ids must use an integer dtype.")
        if doc_ids.ndim != 2 or doc_ids.shape != keep_mask.shape:
            raise ValueError(
                "doc_ids must match attention_mask shape; "
                f"got doc_ids={tuple(doc_ids.shape)}, mask={tuple(keep_mask.shape)}."
            )
        ids = doc_ids.to(dtype=torch.long)
        if not torch.equal(ids.ne(0), keep_mask):
            raise ValueError("doc_ids liveness disagrees with attention_mask.")

    for segment_idx in range(active_segments):
        start = int(offsets[segment_idx].item())
        length = int(lengths[segment_idx].item())
        row = start // int(seq_len)
        row_start = start % int(seq_len)
        row_end = row_start + length
        if row >= batch_size or row_end > int(seq_len):
            raise ValueError(
                f"Document segment exceeds its batch row at segment={segment_idx}: "
                f"offset={start}, length={length}, batch_size={batch_size}, seq_len={seq_len}."
            )
        if bool(covered[row, row_start:row_end].any().item()):
            raise ValueError(f"Overlapping document segments at segment={segment_idx}.")

        if ids is not None:
            segment_doc_ids = ids[row, row_start:row_end]
            doc_id = int(segment_doc_ids[0].item())
            if doc_id <= 0 or not bool(segment_doc_ids.eq(doc_id).all().item()):
                raise ValueError(f"Document segment crosses a document boundary at segment={segment_idx}.")
            if (
                row_start > 0
                and bool(keep_mask[row, row_start - 1].item())
                and int(ids[row, row_start - 1].item()) == doc_id
            ):
                raise ValueError(f"Document segment starts inside document {doc_id}.")
            if (
                row_end < int(seq_len)
                and bool(keep_mask[row, row_end].item())
                and int(ids[row, row_end].item()) == doc_id
            ):
                raise ValueError(f"Document segment ends inside document {doc_id}.")

        covered[row, row_start:row_end] = True

    if not torch.equal(covered, keep_mask):
        missing = int((keep_mask & ~covered).sum().item())
        extra = int((covered & ~keep_mask).sum().item())
        raise ValueError(
            "Document segment descriptors disagree with attention_mask: "
            f"missing_active_tokens={missing}, included_masked_tokens={extra}."
        )

    if cu_seqlens is not None:
        if cu_seqlens.ndim != 1 or int(cu_seqlens.numel()) != int(lengths.numel()) + 1:
            raise ValueError(
                "Document cumulative lengths must be flat with one more entry than segment lengths."
            )
        expected_cu = torch.zeros_like(cu_seqlens, dtype=torch.int64)
        if active_segments > 0:
            expected_cu[1 : active_segments + 1] = torch.cumsum(
                lengths[:active_segments], dim=0, dtype=torch.int64
            )
        actual_cu = cu_seqlens.to(dtype=torch.int64)
        if not torch.equal(actual_cu, expected_cu):
            raise ValueError(
                "Document cumulative lengths disagree with segment lengths: "
                f"expected={expected_cu.tolist()}, supplied={actual_cu.tolist()}."
            )


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


@lru_cache(maxsize=8)
def _doc_block_static_masks(
    seq_len: int,
    device_type: str,
    device_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached identity and fallback-CLS masks for one device shape.

    :param int seq_len: Sequence length.
    :param str device_type: Torch device type.
    :param int | None device_index: Optional device index.
    :return tuple[torch.Tensor, torch.Tensor]: Identity matrix and CLS key mask.
    """

    device = torch.device(device_type, device_index)
    eye = torch.eye(int(seq_len), dtype=torch.bool, device=device)
    cls_key = torch.zeros((int(seq_len),), dtype=torch.bool, device=device)
    if int(seq_len) > 0:
        cls_key[0] = True
    return eye, cls_key


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
    cache_key = (seq_len, str(ids.device.type), ids.device.index)
    _, cls_key = _doc_block_static_masks(*cache_key)

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


def doc_segment_metadata_host_stats(
    segment_lengths: torch.Tensor,
    *,
    active_tokens: int | None = None,
) -> tuple[int | None, int | None, int | None]:
    """Return host-side doc-segment stats for already-built metadata.

    :param torch.Tensor segment_lengths: Padded per-segment lengths.
    :param int | None active_tokens: Optional total active token count.
    :return tuple[int | None, int | None, int | None]: Active segment count, max segment length, and active tokens.
    """

    if segment_lengths.device.type != "cpu":
        return None, None, active_tokens
    active = segment_lengths[segment_lengths.ne(0)]
    if int(active.numel()) == 0:
        return 0, 0, active_tokens
    return int(active.numel()), int(active.max().item()), active_tokens


def doc_ids_from_segments(
    *,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Reconstruct compact document ids from padded segment descriptors.

    :param torch.Tensor offsets: Flat row offsets for segment starts.
    :param torch.Tensor lengths: Segment lengths with zero padding.
    :param int batch_size: Batch size.
    :param int seq_len: Sequence length.
    :return torch.Tensor: Reconstructed ``doc_ids`` tensor.
    """

    doc_ids = torch.zeros((int(batch_size), int(seq_len)), device=offsets.device, dtype=torch.long)
    active_lengths = lengths.to(dtype=torch.long).clamp_min(0)
    num_segments = int(active_lengths.count_nonzero().item())
    if num_segments == 0:
        return doc_ids
    active_offsets = offsets[:num_segments].to(dtype=torch.long)
    active_lengths = active_lengths[:num_segments]
    for idx in range(num_segments):
        offset = int(active_offsets[idx].item())
        length = int(active_lengths[idx].item())
        if length <= 0:
            continue
        row = offset // int(seq_len)
        start = offset % int(seq_len)
        end = min(int(seq_len), start + length)
        if 0 <= row < int(batch_size) and start < end:
            doc_ids[row, start:end] = int(idx + 1)
    return doc_ids


__all__ = [
    "FlashBatchMeta",
    "build_validated_prefix_lengths",
    "build_doc_block_mask",
    "build_doc_segment_metadata",
    "doc_segment_metadata_host_stats",
    "doc_ids_from_segments",
    "expand_keep_mask_to_4d",
    "is_pairwise_mask",
    "mask_to_2d_keep_mask",
    "normalize_keep_mask",
    "reduce_keep_mask_to_2d",
    "validate_doc_segments_against_mask",
]
