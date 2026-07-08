"""Attention-mask normalization helpers for model/runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_FLASH_TRUTHY = {"1", "true", "yes", "y", "on"}


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
        producers must prove the batch mask is a contiguous-prefix mask (see
        :func:`is_prefix_padding_keep_mask`) before publishing lengths here.
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
        global CLS conditioning) must check it before doing so, because
        doc-block batches ship a compact 2D keep mask whose pairwise
        document-blocking semantics live in this metadata instead.

        :return bool: True when the batch routes through a doc-block path or ships doc-segment metadata.
        """

        return (
            self.normalized_route_hint() in {"docblock", "docblock_bias"}
            or self.doc_segment_offsets is not None
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


def _flash_cfg_get(flash_cfg: Any | None, name: str, default: Any) -> Any:
    """Return one flash config value from a mapping/dataclass/object.

    :param Any | None flash_cfg: Optional config source.
    :param str name: Field name.
    :param Any default: Default value.
    :return Any: Resolved value.
    """

    if flash_cfg is None:
        return default
    if isinstance(flash_cfg, dict):
        return flash_cfg.get(name, default)
    return getattr(flash_cfg, name, default)


def _flash_cfg_bool(
    flash_cfg: Any | None,
    *,
    name: str,
    default: str,
) -> bool:
    """Resolve one boolean flash option from config or a declared default.

    :param Any | None flash_cfg: Optional config source.
    :param str name: Config field name.
    :param str default: Default text when config is absent.
    :return bool: Resolved boolean value.
    """

    if flash_cfg is None:
        return str(default).strip().lower() in _FLASH_TRUTHY
    return bool(_flash_cfg_get(flash_cfg, name, False))


def _flash_cfg_optional_int(
    flash_cfg: Any | None,
    *,
    name: str,
    default: int | None = None,
) -> int | None:
    """Resolve one optional integer flash option.

    :param Any | None flash_cfg: Optional config source.
    :param str name: Config field name.
    :param int | None default: Default value when config is absent.
    :return int | None: Resolved integer, or None when unset.
    """

    value = _flash_cfg_get(flash_cfg, name, default)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return default


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


def is_prefix_padding_keep_mask(attention_mask: torch.Tensor, *, seq_len: int) -> bool:
    """Return whether a padding keep mask is a right-padded contiguous prefix.

    The flash fixed/varlen routes compress padding masks into per-example
    prefix lengths, which only preserves eager semantics when every row keeps
    a contiguous prefix (right padding). Masks with holes or left padding must
    stay on eager attention, which honors arbitrary key-padding masks.

    :param torch.Tensor attention_mask: Padding-style keep mask ``(B,S)`` or ``(B,1,1,S)``.
    :param int seq_len: Expected sequence length.
    :raises ValueError: If the mask is not an exact-length padding mask.
    :return bool: True when each row's kept positions form a contiguous prefix.
    """

    keep = mask_to_2d_keep_mask(attention_mask, seq_len=seq_len)
    if int(keep.shape[-1]) <= 1:
        return True
    # A prefix mask never rises from False back to True along the key axis.
    rises = ~keep[:, :-1] & keep[:, 1:]
    return not bool(rises.any())


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


def expand_keep_mask_to_4d(attention_mask: torch.Tensor, *, pairwise_2d: bool = False) -> torch.Tensor:
    """Expand a rank-2/3/4 keep mask to the canonical 4D attention layout.

    2D key-padding masks become broadcast ``(B,1,1,S)`` by default, or a full
    outer-product ``(B,1,S,S)`` pairwise mask when ``pairwise_2d`` is set (the
    original DeBERTa EMD convention). 3D pairwise masks gain a head axis and 4D
    masks are head-reduced to ``(B,1,*,S)``.

    :param torch.Tensor attention_mask: Keep mask in rank-2/3/4 layout.
    :param bool pairwise_2d: Whether 2D masks expand via outer product.
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
        if mask.shape[1] == 1:
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


_DOC_BLOCK_EYE_CACHE: dict[tuple[int, str, int | None], torch.Tensor] = {}
_DOC_BLOCK_CLS_KEY_CACHE: dict[tuple[int, str, int | None], torch.Tensor] = {}
_DOC_BLOCK_CACHE_MAX_ENTRIES = 8


def _doc_block_cache_put(
    cache: dict[tuple[int, str, int | None], torch.Tensor],
    key: tuple[int, str, int | None],
    value: torch.Tensor,
) -> None:
    """Insert into a doc-block mask cache, evicting oldest entries past the cap.

    Cached entries scale with ``seq_len**2`` bytes, so variable-seq-len
    training must not grow these dicts without bound.

    :param dict[tuple[int, str, int | None], torch.Tensor] cache: Target cache.
    :param tuple[int, str, int | None] key: Cache key.
    :param torch.Tensor value: Tensor to cache.
    """

    cache[key] = value
    if len(cache) > _DOC_BLOCK_CACHE_MAX_ENTRIES:
        for stale_key in list(cache.keys())[: _DOC_BLOCK_CACHE_MAX_ENTRIES // 2]:
            cache.pop(stale_key, None)


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

    batch_size, seq_len = int(ids.shape[0]), int(ids.shape[1])
    del batch_size
    cache_key = (seq_len, str(ids.device.type), ids.device.index)
    eye = _DOC_BLOCK_EYE_CACHE.get(cache_key)
    if eye is None:
        eye = torch.eye(seq_len, dtype=torch.bool, device=ids.device)
        _doc_block_cache_put(_DOC_BLOCK_EYE_CACHE, cache_key, eye)

    keep = (keep & ~eye[None, :, :]) | (eye[None, :, :] & active[:, :, None])

    cls_key = _DOC_BLOCK_CLS_KEY_CACHE.get(cache_key)
    if cls_key is None:
        cls_key = torch.zeros((seq_len,), dtype=torch.bool, device=ids.device)
        if seq_len > 0:
            cls_key[0] = True
        _doc_block_cache_put(_DOC_BLOCK_CLS_KEY_CACHE, cache_key, cls_key)
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
    "_flash_cfg_bool",
    "_flash_cfg_get",
    "build_doc_block_mask",
    "build_doc_segment_metadata",
    "doc_segment_metadata_host_stats",
    "doc_ids_from_segments",
    "expand_keep_mask_to_4d",
    "is_pairwise_mask",
    "is_prefix_padding_keep_mask",
    "mask_to_2d_keep_mask",
    "normalize_keep_mask",
    "reduce_keep_mask_to_2d",
]
