"""Shared field names for collator-to-training batch contracts."""

from __future__ import annotations

from typing import Final, NamedTuple

__all__ = ["CPU_SCALAR_BATCH_KEYS"]


class _CpuScalarBatchKeys(NamedTuple):
    """Names of scalar tensors that must remain host-resident during transfer."""

    active_tokens: str
    doc_num_segments: str
    doc_max_seqlen: str


CPU_SCALAR_BATCH_KEYS: Final = _CpuScalarBatchKeys(
    active_tokens="flash_active_tokens_scalar",
    doc_num_segments="flash_doc_num_segments_scalar",
    doc_max_seqlen="flash_doc_max_seqlen_scalar",
)
"""Batch keys whose zero-dimensional tensors must remain on the CPU."""
