"""Streaming dataset packing utilities for fixed-length RTD inputs."""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from deberta.data.retry import handle_dataset_retry_failure

logger = logging.getLogger(__name__)


@dataclass
class PackedStreamingConfig:
    """Configuration for packing raw text into fixed-length token blocks."""

    text_column_name: str
    max_seq_length: int
    seed: int
    shuffle_buffer_size: int
    block_cross_document_attention: bool = False
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0


class PackedStreamingDataset(torch.utils.data.IterableDataset):
    """Pack streaming text into fixed-length blocks across ranks and DataLoader workers.

    See [Data pipeline](../guides/data-pipeline.md#packed-streaming-path) for sample construction
    and output fields.
    """

    def __init__(
        self,
        *,
        hf_dataset: Any,
        tokenizer: Any,
        cfg: PackedStreamingConfig,
        process_index: int = 0,
        num_processes: int = 1,
    ) -> None:
        """Create a streaming packer dataset.

        :param Any hf_dataset: Source HF dataset (usually iterable).
        :param Any tokenizer: Tokenizer with cls/sep/pad token ids.
        :param PackedStreamingConfig cfg: Packing configuration.
        :param int process_index: Current distributed rank index.
        :param int num_processes: Total distributed process count.
        """
        super().__init__()
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.process_index = int(process_index)
        self.num_processes = int(num_processes)
        try:
            self._epoch = mp.Value("i", 0)
            self._epoch_shared = True
        except Exception:
            # Some restricted environments (for example sandboxed CI) may block
            # shared-memory allocations used by multiprocessing.Value.
            self._epoch = 0
            self._epoch_shared = False

        if not hasattr(tokenizer, "cls_token_id") or tokenizer.cls_token_id is None:
            raise ValueError("Tokenizer must define cls_token_id.")
        if not hasattr(tokenizer, "sep_token_id") or tokenizer.sep_token_id is None:
            raise ValueError("Tokenizer must define sep_token_id.")
        if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id.")

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch to underlying dataset when supported.

        :param int epoch: Training epoch.
        """
        if self._epoch_shared:
            with self._epoch.get_lock():
                self._epoch.value = int(max(0, epoch))
        else:
            self._epoch = int(max(0, epoch))
        set_epoch = getattr(self.hf_dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    def _current_epoch(self) -> int:
        """Read current epoch value.

        :return int: Current dataset epoch.
        """
        if self._epoch_shared:
            return int(self._epoch.value)
        return int(self._epoch)

    def _shard_dataset_for_process(self, ds: Any) -> Any:
        """Shard dataset across distributed processes.

        :param Any ds: Input dataset object.
        :return Any: Sharded dataset view.
        """
        # HF IterableDataset.__iter__ assigns its available data-source shards to DataLoader
        # workers. Sharding by worker here would apply that split twice and can request an
        # empty shard when num_workers exceeds the source's shard count.
        if self.num_processes > 1 and hasattr(ds, "shard"):
            ds = ds.shard(num_shards=self.num_processes, index=self.process_index)
        return ds

    def _new_example_iterator(self) -> Iterator[dict[str, Any]]:
        """Build a fresh iterator over shuffled and sharded examples.

        :return Iterator[dict[str, Any]]: Example iterator.
        """
        ds = self.hf_dataset
        epoch = self._current_epoch()
        shuffle_seed = int(self.cfg.seed) + int(epoch)
        if self.cfg.shuffle_buffer_size and self.cfg.shuffle_buffer_size > 0 and hasattr(ds, "shuffle"):
            try:
                # IterableDataset-style shuffle (streaming reservoir shuffle).
                ds = ds.shuffle(buffer_size=self.cfg.shuffle_buffer_size, seed=shuffle_seed)
            except TypeError:
                # Map-style Dataset.shuffle does not accept buffer_size.
                ds = ds.shuffle(seed=shuffle_seed)
        ds = self._shard_dataset_for_process(ds)
        return iter(ds)

    def _iter_examples(self) -> Iterator[dict[str, Any]]:
        """Iterate with bounded deterministic replay after transient I/O failure.

        :return Iterator[dict[str, Any]]: Example iterator.
        """
        attempts = max(1, int(self.cfg.retry_attempts))
        yielded = 0
        for attempt in range(1, attempts + 1):
            try:
                iterator = self._new_example_iterator()
                # Arbitrary shuffled iterables expose no exact seek contract.
                # Replaying the consumed prefix is therefore the only generic
                # way to resume without duplicates; the retry-attempt budget
                # bounds how many full-prefix replays one failure can trigger.
                for _ in range(yielded):
                    next(iterator)
                for example in iterator:
                    yielded += 1
                    yield example
                return
            except Exception as exc:
                handle_dataset_retry_failure(
                    exc,
                    attempt=attempt,
                    attempts=attempts,
                    backoff_seconds=self.cfg.retry_backoff_seconds,
                    on_retry=lambda failed_attempt, delay, failure, yielded=yielded: logger.warning(
                        "Dataset stream failed transiently at epoch %d after %d examples "
                        "(attempt %d/%d, retry in %.1fs): %s",
                        self._current_epoch(),
                        yielded,
                        failed_attempt,
                        attempts,
                        delay,
                        failure,
                    ),
                )

    def _normalize_raw_text(self, ex: dict[str, Any]) -> str:
        """Extract and normalize one raw text example.

        :param dict[str, Any] ex: Example mapping.
        :return str: Normalized text string.
        """
        text_key = self.cfg.text_column_name
        raw = ex.get(text_key, None)
        if raw is None:
            raise KeyError(f"Text column '{text_key}' not found. Available keys: {list(ex.keys())}")

        if isinstance(raw, (list, tuple)):
            raw = "\n".join([str(x) for x in raw])
        else:
            raw = str(raw)
        return raw

    def _tokenize_text(self, raw: str) -> list[int]:
        """Tokenize text into ids without injecting specials.

        :param str raw: Input text.
        :return list[int]: Token ids.
        """
        if not raw.strip():
            return []

        tokenized = self.tokenizer(
            raw,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = tokenized.get("input_ids", [])
        return [int(x) for x in ids]

    def _iter_tokenized_documents(self) -> Iterator[list[int]]:
        """Yield nonempty tokenized documents from the source stream.

        :return Iterator[list[int]]: Token IDs for each nonempty document.
        """
        for example in self._iter_examples():
            token_ids = self._tokenize_text(self._normalize_raw_text(example))
            if token_ids:
                yield token_ids

    def _build_example_from_chunk(self, *, chunk: list[int], max_seq: int) -> dict[str, Any]:
        """Build one fixed-length packed example from a token chunk.

        :param list[int] chunk: Chunk content without outer specials.
        :param int max_seq: Target sequence length.
        :return dict[str, Any]: Packed example.
        """
        cls_id = int(self.tokenizer.cls_token_id)
        sep_id = int(self.tokenizer.sep_token_id)
        pad_id = int(self.tokenizer.pad_token_id)

        input_ids = [cls_id] + chunk + [sep_id]
        pad_len = max_seq - len(input_ids)
        if pad_len > 0:
            input_ids.extend([pad_id] * pad_len)

        # Mark all separator tokens (including internal separators), CLS, and padding as special.
        special_ids = {cls_id, sep_id, pad_id}
        chunk_special = [1 if t in special_ids else 0 for t in chunk]
        special_tokens_mask = [1] + chunk_special + [1] + [1] * pad_len

        ex_out = {
            "input_ids": input_ids,
            "special_tokens_mask": special_tokens_mask,
        }
        if pad_len > 0:
            ex_out["attention_mask"] = [1] * (max_seq - pad_len) + [0] * pad_len
        return ex_out

    def _build_docblock_example(
        self,
        *,
        segments: list[list[int]],
        max_seq: int,
    ) -> dict[str, Any]:
        """Build one row whose document segments each own CLS and local positions.

        :param list[list[int]] segments: Lexical token chunks without specials.
        :param int max_seq: Fixed output sequence length.
        :return dict[str, Any]: Packed row with structural document ids.
        """

        cls_id = int(self.tokenizer.cls_token_id)
        sep_id = int(self.tokenizer.sep_token_id)
        pad_id = int(self.tokenizer.pad_token_id)
        special_ids = {cls_id, sep_id, pad_id}
        input_ids: list[int] = []
        special_tokens_mask: list[int] = []
        doc_ids: list[int] = []

        for document_id, segment in enumerate(segments, start=1):
            wrapped = [cls_id, *segment, sep_id]
            input_ids.extend(wrapped)
            special_tokens_mask.extend([1, *[1 if token in special_ids else 0 for token in segment], 1])
            doc_ids.extend([document_id] * len(wrapped))

        pad_len = int(max_seq) - len(input_ids)
        if pad_len < 0:
            raise RuntimeError("Doc-block segment packing exceeded max_seq_length.")
        if pad_len:
            input_ids.extend([pad_id] * pad_len)
            special_tokens_mask.extend([1] * pad_len)
            doc_ids.extend([0] * pad_len)

        output: dict[str, Any] = {
            "input_ids": input_ids,
            "special_tokens_mask": special_tokens_mask,
            "doc_ids": doc_ids,
        }
        if pad_len:
            output["attention_mask"] = [1] * (int(max_seq) - pad_len) + [0] * pad_len
        return output

    def _iter_docblock_examples(self) -> Iterator[dict[str, Any]]:
        """Yield greedily packed whole document chunks with per-segment CLS.

        Long documents are split independently at ``max_seq_length - 2`` so
        their chunk boundaries do not depend on the remaining space in a row.

        :return Iterator[dict[str, Any]]: Fixed-length doc-block examples.
        """

        max_seq = int(self.cfg.max_seq_length)
        max_content = max_seq - 2
        row_segments: list[list[int]] = []
        row_length = 0

        for token_ids in self._iter_tokenized_documents():
            for start in range(0, len(token_ids), max_content):
                segment = token_ids[start : start + max_content]
                if not segment:
                    continue
                wrapped_length = len(segment) + 2
                if row_segments and row_length + wrapped_length > max_seq:
                    yield self._build_docblock_example(segments=row_segments, max_seq=max_seq)
                    row_segments = []
                    row_length = 0
                row_segments.append(segment)
                row_length += wrapped_length
                if row_length == max_seq:
                    yield self._build_docblock_example(segments=row_segments, max_seq=max_seq)
                    row_segments = []
                    row_length = 0

        if row_segments:
            yield self._build_docblock_example(segments=row_segments, max_seq=max_seq)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        max_seq = int(self.cfg.max_seq_length)
        if max_seq < 8:
            raise ValueError("max_seq_length is too small for pretraining.")

        if bool(self.cfg.block_cross_document_attention):
            yield from self._iter_docblock_examples()
            return

        sep_id = int(self.tokenizer.sep_token_id)
        block_len = max_seq - 2

        buffer: list[int] = []
        buffer_start = 0

        def _compact_buffer_if_needed() -> None:
            """Trim consumed prefix from the rolling token buffer when beneficial."""
            nonlocal buffer
            nonlocal buffer_start
            if buffer_start <= 0:
                return
            # Compact occasionally to avoid unbounded list growth while keeping
            # O(1) amortized consumption from the left.
            if buffer_start >= 65536 or buffer_start >= (len(buffer) // 2):
                buffer = buffer[buffer_start:]
                buffer_start = 0

        for ids in self._iter_tokenized_documents():
            buffer.extend(ids)
            # Explicit doc separator to reduce cross-doc leakage.
            buffer.append(sep_id)

            # Emit fixed-length blocks.
            # We reserve 2 spots for [CLS] and final [SEP].
            while (len(buffer) - buffer_start) >= block_len:
                chunk = buffer[buffer_start : buffer_start + block_len]
                buffer_start += block_len
                # Strip leading separator tokens left over from previous document
                # boundaries to avoid degenerate [CLS, SEP, ...] empty-document
                # starts under doc-blocking.
                lead = 0
                while lead < len(chunk) and chunk[lead] == sep_id:
                    lead += 1
                if lead > 0:
                    chunk = chunk[lead:]
                if not chunk:
                    _compact_buffer_if_needed()
                    continue
                yield self._build_example_from_chunk(chunk=chunk, max_seq=max_seq)
                _compact_buffer_if_needed()

        # Flush trailing remainder instead of silently dropping it.
        #
        # We append one explicit document separator after each document, so the final
        # buffer commonly ends with SEP. Emitting that SEP-only tail would produce a
        # degenerate [CLS, SEP, SEP, PAD...] example with no training signal.
        if buffer_start > 0:
            buffer = buffer[buffer_start:]
            buffer_start = 0
        while buffer and buffer[-1] == sep_id:
            buffer.pop()
        if buffer:
            chunk = buffer[:block_len]
            lead = 0
            while lead < len(chunk) and chunk[lead] == sep_id:
                lead += 1
            if lead < len(chunk):
                yield self._build_example_from_chunk(chunk=chunk[lead:], max_seq=max_seq)


class SequentialStreamingDataset(PackedStreamingDataset):
    """One-document-per-sequence dataset (reference mode without cross-document packing)."""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        max_seq = int(self.cfg.max_seq_length)
        if max_seq < 8:
            raise ValueError("max_seq_length is too small for pretraining.")
        block_len = max_seq - 2

        for ids in self._iter_tokenized_documents():
            # Split long documents into consecutive one-document chunks.
            for i in range(0, len(ids), block_len):
                chunk = ids[i : i + block_len]
                if not chunk:
                    continue
                yield self._build_example_from_chunk(chunk=chunk, max_seq=max_seq)
