"""Streaming dataset packing utilities for fixed-length RTD inputs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PackedStreamingConfig:
    """Configuration for packing raw text into fixed-length token blocks."""

    text_column_name: str
    max_seq_length: int
    seed: int
    shuffle_buffer_size: int


class PackedStreamingDataset(torch.utils.data.IterableDataset):
    """Packs a streaming HF IterableDataset of text into fixed-length token blocks.

    Key properties:
      - streaming-first
      - sharded across *both* distributed processes and dataloader workers
      - concatenates tokenized documents and chunks into blocks

    Output examples are dicts with:
      - input_ids: List[int]
      - attention_mask: List[int]
      - special_tokens_mask: List[int]
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
        # Some streaming datasets support set_epoch() for deterministic shuffling.
        if hasattr(self.hf_dataset, "set_epoch"):
            try:
                self.hf_dataset.set_epoch(epoch)
            except Exception:
                pass

    def _shard_dataset_for_worker(self, ds: Any) -> Any:
        """Shard dataset across processes and dataloader workers.

        :param Any ds: Input dataset object.
        :return Any: Sharded dataset view.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = int(worker_info.num_workers)
            worker_id = int(worker_info.id)

        total_shards = self.num_processes * num_workers
        shard_id = self.process_index * num_workers + worker_id

        # HF IterableDataset shard selects every `total_shards`-th example.
        if hasattr(ds, "shard"):
            ds = ds.shard(num_shards=total_shards, index=shard_id)
        return ds

    def _iter_examples(self) -> Iterator[dict[str, Any]]:
        """Build an iterator over shuffled and sharded examples.

        :return Iterator[dict[str, Any]]: Example iterator.
        """
        ds = self.hf_dataset
        if self.cfg.shuffle_buffer_size and self.cfg.shuffle_buffer_size > 0 and hasattr(ds, "shuffle"):
            ds = ds.shuffle(buffer_size=self.cfg.shuffle_buffer_size, seed=self.cfg.seed)
        ds = self._shard_dataset_for_worker(ds)
        return iter(ds)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        max_seq = int(self.cfg.max_seq_length)
        if max_seq < 8:
            raise ValueError("max_seq_length is too small for pretraining.")

        cls_id = int(self.tokenizer.cls_token_id)
        sep_id = int(self.tokenizer.sep_token_id)
        pad_id = int(self.tokenizer.pad_token_id)

        text_key = self.cfg.text_column_name

        buffer: list[int] = []
        for ex in self._iter_examples():
            raw = ex.get(text_key, None)
            if raw is None:
                # Fail fast: the dataset doesn't have the expected column.
                raise KeyError(f"Text column '{text_key}' not found. Available keys: {list(ex.keys())}")

            if isinstance(raw, (list, tuple)):
                # Some datasets store pre-split lines.
                raw = "\n".join([str(x) for x in raw])
            else:
                raw = str(raw)

            if not raw.strip():
                continue

            tokenized = self.tokenizer(
                raw,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            ids = tokenized.get("input_ids", [])
            if not ids:
                continue

            buffer.extend([int(x) for x in ids])
            # Explicit doc separator to reduce cross-doc leakage.
            buffer.append(sep_id)

            # Emit fixed-length blocks.
            # We reserve 2 spots for [CLS] and final [SEP].
            block_len = max_seq - 2
            while len(buffer) >= block_len:
                chunk = buffer[:block_len]
                buffer = buffer[block_len:]

                input_ids = [cls_id] + chunk + [sep_id]
                attention_mask = [1] * len(input_ids)

                pad_len = max_seq - len(input_ids)
                if pad_len > 0:
                    input_ids.extend([pad_id] * pad_len)
                    attention_mask.extend([0] * pad_len)

                # Mask specials + pad as special to prevent MLM masking.
                # Note: chunk may contain internal [SEP] doc separators; mark them special too.
                special_ids = {cls_id, sep_id, pad_id}
                chunk_special = [1 if t in special_ids else 0 for t in chunk]
                special_tokens_mask = [1] + chunk_special + [1] + [1] * pad_len

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "special_tokens_mask": special_tokens_mask,
                }
