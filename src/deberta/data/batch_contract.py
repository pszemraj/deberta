"""Shared field names for collator-to-training batch contracts."""

from __future__ import annotations

from typing import Final, NamedTuple

__all__ = ["CPU_SCALAR_BATCH_KEYS", "FLASH_SCALAR_BATCH_KEYS", "HostScalarBatchKey"]


class HostScalarBatchKey(NamedTuple):
    """Paired plain-int and CPU-scalar keys for one compiled batch value."""

    host: str
    scalar: str


class _FlashScalarBatchKeys(NamedTuple):
    """Complete key pairs for FlashDeBERTa host/scalar metadata."""

    active_tokens: HostScalarBatchKey
    doc_num_segments: HostScalarBatchKey
    doc_max_seqlen: HostScalarBatchKey


class _CpuScalarBatchKeys(NamedTuple):
    """Names of scalar tensors that must remain host-resident during transfer."""

    active_tokens: str
    doc_num_segments: str
    doc_max_seqlen: str


FLASH_SCALAR_BATCH_KEYS: Final = _FlashScalarBatchKeys(
    active_tokens=HostScalarBatchKey("flash_active_tokens", "flash_active_tokens_scalar"),
    doc_num_segments=HostScalarBatchKey("flash_doc_num_segments", "flash_doc_num_segments_scalar"),
    doc_max_seqlen=HostScalarBatchKey("flash_doc_max_seqlen", "flash_doc_max_seqlen_scalar"),
)

CPU_SCALAR_BATCH_KEYS: Final = _CpuScalarBatchKeys(
    active_tokens=FLASH_SCALAR_BATCH_KEYS.active_tokens.scalar,
    doc_num_segments=FLASH_SCALAR_BATCH_KEYS.doc_num_segments.scalar,
    doc_max_seqlen=FLASH_SCALAR_BATCH_KEYS.doc_max_seqlen.scalar,
)
