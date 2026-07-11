"""Masking collator implementations used by RTD pretraining."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch

from deberta.data.batch_contract import FLASH_SCALAR_BATCH_KEYS, HostScalarBatchKey
from deberta.modeling.mask_utils import (
    build_doc_segment_metadata,
    build_validated_prefix_lengths,
    doc_segment_metadata_host_stats,
    validate_doc_segments_against_mask,
)

logger = logging.getLogger(__name__)


def _set_scalar_pair(batch: dict[str, Any], key: HostScalarBatchKey, value: int) -> None:
    """Store one host integer and its CPU scalar-tensor mirror.

    :param dict[str, Any] batch: Mutable collated batch.
    :param HostScalarBatchKey key: Paired batch-key contract.
    :param int value: Scalar value to publish.
    """

    normalized = int(value)
    batch[key.host] = normalized
    batch[key.scalar] = torch.tensor(normalized, dtype=torch.int32)


@dataclass
class MLMConfig:
    """Masking configuration.

    Mask selection uses DeBERTa's windowed policy: ``max_ngram=1`` selects windowed unigrams and
    larger values enable whole-word n-grams. Replacement probabilities are conditional on token
    selection and must sum to at most one; the remainder keeps the original token.
    """

    mlm_probability: float
    mask_token_prob: float = 0.8
    random_token_prob: float = 0.1
    max_ngram: int = 1


class DebertaV3ElectraCollator:
    """Dynamic MLM masking collator suitable for RTD/ELECTRA-style pretraining.

    Produces masked ``input_ids``, MLM ``labels``, and optional attention/token-type tensors.
    Packed doc-block batches also carry ``doc_ids`` and ``flash_*`` metadata. See
    [Data pipeline](../guides/data-pipeline.md#cross-document-attention-blocking) for the complete
    batch-preparation contract.

    Replacement probabilities come from ``MLMConfig``. Training config resolution may replace
    those raw helper defaults for the selected backbone; see ``configs/config-reference.yaml``.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        cfg: MLMConfig,
        packed_sequences: bool = False,
        block_cross_document_attention: bool = True,
        emit_flash_metadata: bool = True,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        """Initialize collator state.

        :param Any tokenizer: HF tokenizer.
        :param MLMConfig cfg: Masking configuration.
        :param bool packed_sequences: Whether inputs are pre-packed with internal separators.
        :param bool block_cross_document_attention: Whether to emit compact document metadata for packed inputs.
        :param bool emit_flash_metadata: Whether to attest Flash routing metadata.
        :param int | None pad_to_multiple_of: Optional right-padding multiple.
        """
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.pad_to_multiple_of = pad_to_multiple_of
        self._packed_sequences = bool(packed_sequences)
        self._block_cross_document_attention = bool(block_cross_document_attention)
        self._emit_flash_metadata = bool(emit_flash_metadata)

        if self.tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer must define a mask token for MLM masking.")
        if float(self.cfg.mlm_probability) <= 0.0 or float(self.cfg.mlm_probability) >= 1.0:
            raise ValueError("mlm_probability must be in (0, 1).")
        if int(self.cfg.max_ngram) < 1:
            raise ValueError("max_ngram must be >= 1")

        mask_prob = float(self.cfg.mask_token_prob)
        rand_prob = float(self.cfg.random_token_prob)
        if mask_prob < 0 or rand_prob < 0 or (mask_prob + rand_prob) > 1.0:
            raise ValueError(
                "Invalid masking probabilities: mask_token_prob + random_token_prob must be <= 1."
            )

        self._special_token_ids = self._collect_special_token_ids()
        self._special_token_ids_tensor_cpu = (
            torch.tensor(sorted(self._special_token_ids), dtype=torch.long)
            if self._special_token_ids
            else None
        )
        self._special_token_ids_tensor_by_device: dict[str, torch.Tensor] = {}
        self._non_special_token_ids_cpu = self._build_non_special_token_ids()
        self._non_special_token_ids_by_device: dict[str, torch.Tensor] = {}
        self._word_boundary_scheme = self._detect_word_boundary_scheme()
        self._word_continuation_by_id = self._build_word_continuation_lookup()
        self._ngram_prob_cache: dict[tuple[int, str], torch.Tensor] = {}
        if int(self.cfg.max_ngram) > 1 and self._word_boundary_scheme == "none":
            logger.warning(
                "Tokenizer word-boundary probe returned scheme='none'; "
                "whole-word n-gram masking will degrade to conservative token-level groups."
            )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        self._validate_structural_doc_id_types(features)
        features = self._harmonize_optional_attention_masks(features)
        needs_padding = self._needs_padding(features)

        # Let tokenizer handle padding for non-packed datasets.
        pad_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "pad_to_multiple_of": self.pad_to_multiple_of,
        }
        # If no padding is needed and the dataset does not provide attention_mask, avoid
        # materializing an all-ones mask.
        if not any("attention_mask" in f for f in features) and not needs_padding:
            pad_kwargs["return_attention_mask"] = False
        batch = self.tokenizer.pad(features, **pad_kwargs)

        # Flash metadata is an internal attestation created from this collated
        # batch. Never inherit stale or user-supplied claims from dataset rows.
        for key in tuple(batch):
            if key.startswith("flash_"):
                batch.pop(key)
        batch.pop("doc_context_index", None)
        if self._packed_sequences and self._block_cross_document_attention:
            batch.pop("position_ids", None)

        if "attention_mask" not in batch and needs_padding:
            raise ValueError(
                "tokenizer.pad must return attention_mask when it adds padding; "
                "token IDs are not a valid substitute for token liveness."
            )

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        inferred_special_tokens_mask = self._infer_special_tokens_mask(batch["input_ids"])
        if special_tokens_mask is None:
            special_tokens_mask = inferred_special_tokens_mask
        else:
            # Some upstream datasets (for example packed streaming) may only tag
            # structural specials (CLS/SEP/PAD). Union with tokenizer-level specials
            # so corruption targets stay aligned with forbidden sampling ids.
            special_tokens_mask = special_tokens_mask.bool() | inferred_special_tokens_mask

        supplied_doc_ids = batch.pop("doc_ids", None)
        doc_ids = None
        if self._packed_sequences and self._block_cross_document_attention:
            if supplied_doc_ids is not None:
                if not isinstance(supplied_doc_ids, torch.Tensor):
                    raise TypeError("Packed doc_ids must be a tensor after tokenizer padding.")
                if (
                    supplied_doc_ids.dtype == torch.bool
                    or supplied_doc_ids.is_floating_point()
                    or supplied_doc_ids.is_complex()
                ):
                    raise TypeError("Packed doc_ids must use an integer dtype.")
                if supplied_doc_ids.shape != batch["input_ids"].shape:
                    raise ValueError(
                        "Packed doc_ids must match input_ids shape; "
                        f"got doc_ids={tuple(supplied_doc_ids.shape)}, "
                        f"input_ids={tuple(batch['input_ids'].shape)}."
                    )
                doc_ids = supplied_doc_ids.to(dtype=torch.long)
            else:
                doc_ids = self._compute_document_ids(
                    input_ids=batch["input_ids"],
                    special_tokens_mask=special_tokens_mask,
                    attention_mask=batch.get("attention_mask"),
                )
        if doc_ids is not None:
            batch["doc_ids"] = doc_ids
            self._attach_document_objective_metadata(batch=batch, doc_ids=doc_ids)
            if self._emit_flash_metadata:
                self._attach_flash_doc_metadata(batch=batch, doc_ids=doc_ids)
        else:
            # Packed/unpadded pretraining examples often have all-ones attention masks.
            # Drop all-ones masks so downstream can pass attention_mask=None to SDPA.
            attn = batch.get("attention_mask")
            if attn is not None:
                all_active = (
                    bool(attn.all().item()) if attn.dtype == torch.bool else bool((attn == 1).all().item())
                )
                if all_active:
                    batch.pop("attention_mask", None)
            if self._emit_flash_metadata:
                self._attach_flash_padding_metadata(batch)

        input_ids, labels = self._mask_tokens(batch["input_ids"], special_tokens_mask=special_tokens_mask)

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch

    @staticmethod
    def _validate_structural_doc_id_types(features: list[dict[str, Any]]) -> None:
        """Reject structural document ids that padding could silently coerce.

        :param list[dict[str, Any]] features: Raw dataset rows before tokenizer padding.
        :raises TypeError: If a row's document ids are not integer-valued.
        """

        for feature in features:
            values = feature.get("doc_ids")
            if values is None:
                continue
            if isinstance(values, torch.Tensor):
                invalid = values.dtype == torch.bool or values.is_floating_point() or values.is_complex()
            elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                invalid = any(isinstance(value, bool) or not isinstance(value, Integral) for value in values)
            else:
                invalid = True
            if invalid:
                raise TypeError("Packed doc_ids must use an integer dtype.")

    @staticmethod
    def _attach_flash_doc_metadata(*, batch: dict[str, Any], doc_ids: torch.Tensor) -> None:
        """Attach CPU-built flash metadata for compact doc-block batches.

        :param dict[str, Any] batch: Collated batch mapping.
        :param torch.Tensor doc_ids: Compact document ids in ``(B,S)`` layout.
        """

        attention_mask = batch.get("attention_mask")
        keep_mask = (
            attention_mask.to(dtype=torch.bool)
            if isinstance(attention_mask, torch.Tensor)
            else torch.ones_like(doc_ids, dtype=torch.bool)
        )
        segment_offsets, segment_lengths, cu_seqlens, active_tokens = build_doc_segment_metadata(doc_ids)
        validate_doc_segments_against_mask(
            attention_mask=keep_mask,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            seq_len=int(doc_ids.shape[-1]),
            doc_ids=doc_ids,
            cu_seqlens=cu_seqlens,
        )
        num_segments, max_segment_length, _ = doc_segment_metadata_host_stats(
            segment_lengths,
            active_tokens=int(active_tokens),
        )
        _set_scalar_pair(batch, FLASH_SCALAR_BATCH_KEYS.active_tokens, active_tokens)
        _set_scalar_pair(batch, FLASH_SCALAR_BATCH_KEYS.doc_num_segments, num_segments)
        _set_scalar_pair(batch, FLASH_SCALAR_BATCH_KEYS.doc_max_seqlen, max_segment_length)
        batch["flash_doc_segment_offsets"] = segment_offsets
        batch["flash_doc_segment_lengths"] = segment_lengths
        batch["flash_doc_cu_seqlens"] = cu_seqlens
        batch["flash_mask_contract"] = "docblock"
        batch["flash_mask_contract_validated"] = True

    def _attach_document_objective_metadata(
        self,
        *,
        batch: dict[str, Any],
        doc_ids: torch.Tensor,
    ) -> None:
        """Attach document-local positions and per-token CLS context indices.

        :param dict[str, Any] batch: Collated batch mapping.
        :param torch.Tensor doc_ids: Validated document ids in ``(B,S)`` layout.
        :raises ValueError: If any active segment does not have standalone CLS/SEP boundaries.
        """

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        active = (
            attention_mask.to(dtype=torch.bool)
            if isinstance(attention_mask, torch.Tensor)
            else torch.ones_like(doc_ids, dtype=torch.bool)
        )
        if not torch.equal(doc_ids.ne(0), active):
            raise ValueError("Packed doc_ids liveness disagrees with attention_mask.")

        batch_size, seq_len = doc_ids.shape
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        previous = torch.zeros_like(doc_ids)
        previous[:, 1:] = doc_ids[:, :-1]
        segment_starts = active & doc_ids.ne(previous)
        start_positions = torch.where(segment_starts, positions, torch.zeros_like(positions))
        doc_context_index = torch.cummax(start_positions, dim=-1).values

        cls_token_id = getattr(self.tokenizer, "cls_token_id", None)
        if cls_token_id is None:
            raise ValueError("Packed document segments require tokenizer.cls_token_id.")
        context_tokens = input_ids.gather(1, doc_context_index)
        if bool((active & context_tokens.ne(int(cls_token_id))).any().item()):
            raise ValueError("Every packed document segment must begin with its own CLS token.")

        sep_token_id = getattr(self.tokenizer, "sep_token_id", None)
        if sep_token_id is None:
            raise ValueError("Packed document segments require tokenizer.sep_token_id.")
        following = torch.zeros_like(doc_ids)
        following[:, :-1] = doc_ids[:, 1:]
        segment_ends = active & doc_ids.ne(following)
        if bool(input_ids[segment_ends].ne(int(sep_token_id)).any().item()):
            raise ValueError("Every packed document segment must end with its own SEP token.")

        position_ids = (positions - doc_context_index).masked_fill(~active, 0)
        batch["position_ids"] = position_ids
        batch["doc_context_index"] = doc_context_index.masked_fill(~active, 0)

    @staticmethod
    def _attach_flash_padding_metadata(batch: dict[str, Any]) -> None:
        """Attach cheap flash metadata for standard padded batches.

        :param dict[str, Any] batch: Collated batch mapping.
        """

        attention_mask = batch.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
            return
        if attention_mask.shape != batch["input_ids"].shape:
            raise ValueError(
                "attention_mask must match input_ids before flash metadata is attested; "
                f"got mask={tuple(attention_mask.shape)}, "
                f"input_ids={tuple(batch['input_ids'].shape)}."
            )
        try:
            seq_lengths = build_validated_prefix_lengths(
                attention_mask,
                seq_len=int(batch["input_ids"].shape[-1]),
            )
        except ValueError as exc:
            # Arbitrary legal keep masks remain eager. Do not publish a lossy
            # length summary or an attestation for them.
            if "right-padded prefix mask" not in str(exc):
                raise
            return
        active_tokens = int(seq_lengths.sum(dtype=torch.int32))
        batch["flash_seq_lengths"] = seq_lengths
        _set_scalar_pair(batch, FLASH_SCALAR_BATCH_KEYS.active_tokens, active_tokens)
        batch["flash_mask_contract"] = "prefix"
        batch["flash_mask_contract_validated"] = True

    def _harmonize_optional_attention_masks(self, features: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure optional ``attention_mask`` keys are consistent before tokenizer padding.

        :param list[dict[str, Any]] features: Raw feature dicts.
        :return list[dict[str, Any]]: Features with harmonized attention-mask keys.
        """
        if not features:
            return features
        if not any("attention_mask" in f for f in features):
            return features

        normalized: list[dict[str, Any]] = []
        for feature in features:
            item = dict(feature)
            if "attention_mask" not in item:
                item["attention_mask"] = [1] * len(item["input_ids"])
            normalized.append(item)
        return normalized

    def _needs_padding(self, features: list[dict[str, Any]]) -> bool:
        """Check whether tokenizer.pad will add padding.

        :param list[dict[str, Any]] features: Batch feature dicts.
        :return bool: True when any sequence will be padded.
        """
        lengths = [len(f["input_ids"]) for f in features]
        if not lengths:
            return False
        max_len = max(lengths)
        if any(seq_len != max_len for seq_len in lengths):
            return True
        if self.pad_to_multiple_of is not None and max_len % int(self.pad_to_multiple_of) != 0:
            return True
        return False

    def _collect_special_token_ids(self) -> set[int]:
        """Collect known tokenizer special token ids.

        :return set[int]: Special token ids.
        """
        out: set[int] = set()
        for sid in getattr(self.tokenizer, "all_special_ids", []):
            try:
                out.add(int(sid))
            except Exception:
                continue
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            out.add(int(pad_id))
        return out

    @staticmethod
    def _effective_vocab_size(tokenizer: Any) -> int:
        """Return the full tokenizer vocabulary size including added tokens.

        :param Any tokenizer: Tokenizer instance.
        :return int: Effective vocabulary size.
        """
        try:
            return int(len(tokenizer))
        except TypeError:
            return int(tokenizer.vocab_size)

    def _build_non_special_token_ids(self) -> torch.Tensor | None:
        """Build tensor of non-special token ids for random replacement.

        :return torch.Tensor | None: CPU tensor of non-special ids, or None.
        """
        vocab_size = self._effective_vocab_size(self.tokenizer)
        if not self._special_token_ids:
            return None

        mask = torch.ones(vocab_size, dtype=torch.bool)
        for sid in self._special_token_ids:
            if 0 <= int(sid) < vocab_size:
                mask[int(sid)] = False

        if not bool(mask.any().item()):
            return None
        return torch.arange(vocab_size, dtype=torch.long)[mask]

    def _sample_random_words(self, shape: torch.Size | tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample random replacement ids, excluding known special ids when possible.

        :param torch.Size | tuple[int, ...] shape: Output tensor shape.
        :param torch.device device: Target device.
        :return torch.Tensor: Random token ids.
        """
        if self._non_special_token_ids_cpu is None:
            return torch.randint(
                low=0, high=self._effective_vocab_size(self.tokenizer), size=shape, device=device
            )

        key = f"{device.type}:{device.index if device.index is not None else -1}"
        ids = self._non_special_token_ids_by_device.get(key)
        if ids is None:
            ids = self._non_special_token_ids_cpu.to(device=device)
            self._non_special_token_ids_by_device[key] = ids

        idx = torch.randint(low=0, high=int(ids.numel()), size=shape, device=device)
        return ids[idx]

    def _special_ids_tensor_for_device(self, device: torch.device) -> torch.Tensor | None:
        """Return cached special-token id tensor on the requested device.

        :param torch.device device: Target device.
        :return torch.Tensor | None: 1D special-token id tensor or ``None``.
        """
        if self._special_token_ids_tensor_cpu is None:
            return None
        key = f"{device.type}:{device.index if device.index is not None else -1}"
        cached = self._special_token_ids_tensor_by_device.get(key)
        if cached is None:
            cached = self._special_token_ids_tensor_cpu.to(device=device)
            self._special_token_ids_tensor_by_device[key] = cached
        return cached

    def _infer_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Infer special-token mask when dataset does not provide one.

        :param torch.Tensor input_ids: Batch token ids.
        :return torch.Tensor: Boolean special-token mask.
        """
        inferred = torch.zeros_like(input_ids, dtype=torch.bool)

        special_ids = self._special_ids_tensor_for_device(input_ids.device)
        if special_ids is not None and int(special_ids.numel()) > 0:
            inferred = torch.isin(input_ids, special_ids)
        elif hasattr(self.tokenizer, "get_special_tokens_mask"):
            try:
                rows = input_ids.detach().cpu().tolist()
                masks = [
                    self.tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True)
                    for row in rows
                ]
                inferred = torch.tensor(masks, dtype=torch.bool, device=input_ids.device)
            except Exception:
                pass

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            inferred = inferred | input_ids.eq(int(pad_id))

        return inferred

    def _detect_word_boundary_scheme(self) -> str | None:
        """Detect tokenizer continuation marker scheme once when possible.

        :return str | None: One of ``wordpiece`` / ``sentencepiece`` / ``gpt2`` or ``None``.
        """
        tokenize = getattr(self.tokenizer, "tokenize", None)
        if not callable(tokenize):
            return None

        try:
            probe = tokenize("hello world")
        except Exception:
            return None

        if not isinstance(probe, list) or not probe:
            return None
        return self._infer_word_boundary_scheme_from_tokens([str(tok) for tok in probe])

    def _build_word_continuation_lookup(self) -> list[bool] | None:
        """Build a vocabulary-indexed continuation lookup for n-gram word grouping.

        This avoids repeated ``convert_ids_to_tokens`` calls in the collator hot path.

        :return list[bool] | None: Per-id continuation flags or ``None`` if unavailable.
        """
        scheme = str(self._word_boundary_scheme or "").strip().lower()
        if scheme not in {"wordpiece", "sentencepiece", "gpt2"}:
            return None

        vocab_size = self._effective_vocab_size(self.tokenizer)
        try:
            tokens = self.tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
        except Exception:
            return None
        if not isinstance(tokens, list) or len(tokens) != int(vocab_size):
            return None

        special_tokens = set(getattr(self.tokenizer, "all_special_tokens", []))
        continuation = [False] * int(vocab_size)
        for i, tok in enumerate(tokens):
            if tok is None:
                continue
            tok = str(tok)
            if not tok or tok in special_tokens:
                continue
            if tok.startswith("##"):
                continuation[i] = True
                continue
            if scheme == "sentencepiece":
                continuation[i] = not tok.startswith("▁")
            elif scheme == "gpt2":
                continuation[i] = not tok.startswith("Ġ")
            elif scheme == "wordpiece":
                continuation[i] = False
        return continuation

    def _infer_word_boundary_scheme_from_tokens(self, tokens: Sequence[str]) -> str:
        """Infer continuation marker style from token strings.

        :param Sequence[str] tokens: Token strings.
        :return str: One of ``wordpiece`` / ``sentencepiece`` / ``gpt2`` / ``none``.
        """
        if any(tok.startswith("##") for tok in tokens):
            return "wordpiece"
        if any(tok.startswith("▁") for tok in tokens):
            return "sentencepiece"
        if any(tok.startswith("Ġ") for tok in tokens):
            return "gpt2"
        return "none"

    def _compute_document_ids(
        self,
        *,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Compute per-token document ids for cross-document attention blocking.

        Returns a compact ``(B, S)`` long tensor of document ids (1-based for active
        tokens, 0 for padding) instead of a dense ``(B, S, S)`` pairwise mask.  The
        pairwise mask is constructed on-device in the training loop to avoid CPU→GPU
        transfer of O(B*S²) data.

        :param torch.Tensor input_ids: Batch token ids of shape (B, S).
        :param torch.Tensor special_tokens_mask: Boolean special-token mask (B, S).
        :param torch.Tensor | None attention_mask: Optional 2D active-token mask (B, S).
        :return torch.Tensor | None: Document ids ``(B, S)`` long, or ``None`` when unnecessary.
        """
        if input_ids.ndim != 2:
            return None
        if special_tokens_mask.ndim != 2 or special_tokens_mask.shape != input_ids.shape:
            return None

        cls_id = getattr(self.tokenizer, "cls_token_id", None)
        sep_id = getattr(self.tokenizer, "sep_token_id", None)
        if cls_id is None or sep_id is None or input_ids.shape[1] < 3:
            return None

        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "Packed attention_mask must match input_ids shape; "
                    f"got mask={tuple(attention_mask.shape)}, input_ids={tuple(input_ids.shape)}."
                )
            active = attention_mask.to(dtype=torch.bool)
        else:
            active = torch.ones_like(input_ids, dtype=torch.bool)
        document_starts = input_ids.eq(int(cls_id)) & special_tokens_mask & active
        if not bool(document_starts.sum(dim=-1).gt(1).any().item()):
            separator_count = (input_ids.eq(int(sep_id)) & special_tokens_mask & active).sum(dim=-1)
            if bool(separator_count.gt(1).any().item()):
                raise ValueError("Cross-document packing requires every document segment to begin with CLS.")
            return None
        return document_starts.to(dtype=torch.long).cumsum(dim=-1).masked_fill(~active, 0)

    def _mask_tokens(
        self, input_ids: torch.Tensor, *, special_tokens_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply DeBERTa-style masking based on configured n-gram width.

        :param torch.Tensor input_ids: Input token ids of shape (B, S).
        :param torch.Tensor special_tokens_mask: Special-token mask of shape (B, S).
        :return tuple[torch.Tensor, torch.Tensor]: Masked ids and MLM labels.
        """
        if int(self.cfg.max_ngram) <= 1:
            return self._mask_tokens_unigram_windowed(input_ids, special_tokens_mask=special_tokens_mask)
        return self._mask_tokens_ngram(
            input_ids, special_tokens_mask=special_tokens_mask, max_ngram=int(self.cfg.max_ngram)
        )

    @staticmethod
    def _sample_windowed_unigram_indices(maskable_idx: torch.Tensor, *, mask_window: int) -> torch.Tensor:
        """Sample one mask position per DeBERTa window from sorted candidate indices.

        :param torch.Tensor maskable_idx: 1D sorted token positions.
        :param int mask_window: Window size ``int(1 / mlm_probability)``.
        :return torch.Tensor: Selected token positions.
        """
        count = int(maskable_idx.numel())
        if count <= 0:
            return maskable_idx.new_empty((0,), dtype=torch.long)
        if mask_window <= 1:
            return maskable_idx

        n_windows = int(math.ceil(float(count) / float(mask_window)))
        starts = torch.arange(n_windows, device=maskable_idx.device, dtype=torch.long) * int(mask_window)
        sizes = torch.clamp(
            torch.full((n_windows,), int(mask_window), device=maskable_idx.device, dtype=torch.long),
            max=count - starts,
        )
        offsets = torch.floor(
            torch.rand(n_windows, device=maskable_idx.device, dtype=torch.float32) * sizes.float()
        ).to(torch.long)
        selected_offsets = starts + offsets
        return maskable_idx.index_select(0, selected_offsets)

    def _resolve_masking_hyperparams(
        self, *, seq_len: int, mlm_prob: float
    ) -> tuple[float, float, float, int, int]:
        """Resolve the shared DeBERTa masking hyperparameters for one batch.

        :param int seq_len: Sequence length of the batch.
        :param float mlm_prob: Effective MLM probability.
        :return tuple[float, float, float, int, int]: mask/random/keep
            probabilities, window size, and per-sequence prediction cap.
        """
        mask_prob = float(self.cfg.mask_token_prob)
        random_prob = float(self.cfg.random_token_prob)
        keep_prob = max(0.0, 1.0 - mask_prob - random_prob)
        mask_window = max(1, int(1.0 / mlm_prob))
        max_preds_per_seq = int(math.ceil(float(seq_len) * mlm_prob / 10.0) * 10)
        return mask_prob, random_prob, keep_prob, mask_window, max_preds_per_seq

    def _apply_mask_replacement_policy(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        row: int,
        selected: torch.Tensor,
        *,
        mask_prob: float,
        random_prob: float,
        keep_prob: float,
        mask_token_id: int,
    ) -> None:
        """Apply the mask/random/keep replacement policy to selected positions.

        Mutates ``input_ids`` and ``labels`` for one batch row in place. RNG
        draw order matches the historical inline implementation exactly (one
        uniform roll per selection, then one random-word draw when needed).

        :param torch.Tensor input_ids: Mutable input ids of shape (B, S).
        :param torch.Tensor labels: Mutable MLM labels of shape (B, S).
        :param int row: Batch row index.
        :param torch.Tensor selected: Selected position indices for this row.
        :param float mask_prob: Probability of replacing with the mask token.
        :param float random_prob: Probability of replacing with a random token.
        :param float keep_prob: Probability of keeping the original token.
        :param int mask_token_id: Mask token id.
        """
        originals = input_ids[row].index_select(0, selected)
        labels[row].scatter_(0, selected, originals)

        if mask_prob >= 1.0 and random_prob <= 0.0:
            input_ids[row, selected] = mask_token_id
            return

        roll = torch.rand(int(selected.numel()), device=input_ids.device, dtype=torch.float32)
        mask_sel = roll < mask_prob
        rand_sel = roll >= (mask_prob + keep_prob)

        if bool(mask_sel.any().item()):
            input_ids[row, selected[mask_sel]] = mask_token_id
        if random_prob > 0.0 and bool(rand_sel.any().item()):
            rand_ids = self._sample_random_words((int(rand_sel.sum().item()),), device=input_ids.device)
            input_ids[row, selected[rand_sel]] = rand_ids

    def _mask_tokens_unigram_windowed(
        self, input_ids: torch.Tensor, *, special_tokens_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply DeBERTa windowed-unigram masking for ``max_ngram=1``.

        :param torch.Tensor input_ids: Input token ids of shape (B, S).
        :param torch.Tensor special_tokens_mask: Special-token mask of shape (B, S).
        :return tuple[torch.Tensor, torch.Tensor]: Masked ids and MLM labels.
        """
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        input_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)

        batch, seq_len = input_ids.shape
        mask_token_id = int(self.tokenizer.mask_token_id)
        mlm_prob = float(self.cfg.mlm_probability)

        mask_prob, random_prob, keep_prob, mask_window, max_preds_per_seq = self._resolve_masking_hyperparams(
            seq_len=seq_len, mlm_prob=mlm_prob
        )

        for b in range(batch):
            spec = special_tokens_mask[b].to(dtype=torch.bool)
            if bool(spec.all().item()):
                continue

            num_to_predict = min(max_preds_per_seq, max(1, int(round(float(seq_len) * mlm_prob))))
            maskable_idx = torch.nonzero(~spec, as_tuple=False).squeeze(-1)
            if int(maskable_idx.numel()) == 0:
                continue

            selected = self._sample_windowed_unigram_indices(maskable_idx, mask_window=mask_window)
            if int(selected.numel()) == 0:
                continue
            if int(selected.numel()) > int(num_to_predict):
                selected = selected[: int(num_to_predict)]

            self._apply_mask_replacement_policy(
                input_ids,
                labels,
                b,
                selected,
                mask_prob=mask_prob,
                random_prob=random_prob,
                keep_prob=keep_prob,
                mask_token_id=mask_token_id,
            )

        return input_ids, labels

    def _mask_tokens_ngram(
        self,
        input_ids: torch.Tensor,
        *,
        special_tokens_mask: torch.Tensor,
        max_ngram: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply DeBERTa NGramMaskGenerator-style masking.

        Selection matches the original DeBERTa policy:
          - n-gram length sampled with p(n) ∝ 1/n
          - windowed selection with ``mask_window = int(1 / mlm_probability)``
          - sequence-level target budget derived from full sequence length
        For ``max_ngram > 1``, word groups are built from tokenizer boundary heuristics.

        :param torch.Tensor input_ids: Input token ids of shape (B, S).
        :param torch.Tensor special_tokens_mask: Special-token mask of shape (B, S).
        :param int max_ngram: Maximum n-gram width.
        :return tuple[torch.Tensor, torch.Tensor]: Masked ids and MLM labels.
        """

        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        input_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)

        B, S = input_ids.shape
        mask_token_id = int(self.tokenizer.mask_token_id)
        mlm_prob = float(self.cfg.mlm_probability)
        mask_prob, random_prob, keep_prob, mask_window, max_preds_per_seq = self._resolve_masking_hyperparams(
            seq_len=S, mlm_prob=mlm_prob
        )

        # n-gram sampling distribution: p(n) ∝ 1/n
        cache_key = (
            int(max_ngram),
            f"{input_ids.device.type}:{input_ids.device.index if input_ids.device.index is not None else -1}",
        )
        probs = self._ngram_prob_cache.get(cache_key)
        if probs is None:
            probs = torch.tensor(
                [1.0 / float(n) for n in range(1, int(max_ngram) + 1)],
                dtype=torch.float32,
                device=input_ids.device,
            )
            probs = probs / probs.sum().clamp(min=1e-12)
            self._ngram_prob_cache[cache_key] = probs

        for b in range(B):
            spec = special_tokens_mask[b].to(dtype=torch.bool)
            if bool(spec.all().item()):
                continue

            num_to_predict = min(max_preds_per_seq, max(1, int(round(float(S) * mlm_prob))))
            ids = input_ids[b].tolist()
            spec_list = spec.tolist()

            groups = self._build_word_groups(ids, spec_list)
            groups = [g for g in groups if any(not bool(spec[idx]) for idx in g)]

            if not groups:
                continue

            mask_grams = [False] * len(groups)
            offset = 0
            while offset < len(groups):
                gram_n = int(torch.multinomial(probs, 1).item()) + 1
                ctx_size = min(gram_n * mask_window, len(groups) - offset)
                if ctx_size <= 0:
                    break

                m = int(torch.randint(low=0, high=ctx_size, size=(1,), device=input_ids.device).item())
                start = offset + m
                end = min(offset + m + gram_n, len(groups))
                offset = max(offset + ctx_size, end)
                for i in range(start, end):
                    mask_grams[i] = True

            selected_positions: list[int] = []
            budget = int(num_to_predict)
            max_budget = int(max_preds_per_seq)
            used = 0
            for do_mask, group in zip(mask_grams, groups, strict=True):
                if not do_mask:
                    continue
                g_len = len(group)
                if g_len <= 0:
                    continue
                # Keep whole-word integrity: never partially mask one word group.
                if used + g_len > max_budget:
                    break
                if used + g_len > budget and used > 0:
                    continue
                selected_positions.extend(group)
                used += g_len
                if used >= budget:
                    break

            if not selected_positions:
                # Fallback intentionally picks one full group (bounded by
                # max_preds_per_seq) to preserve whole-word integrity; this may
                # exceed num_to_predict for long groups and matches DeBERTa's
                # practical masking behavior.
                candidates = [
                    group
                    for do_mask, group in zip(mask_grams, groups, strict=True)
                    if do_mask and 0 < len(group) <= max_budget
                ]
                if not candidates:
                    candidates = [group for group in groups if 0 < len(group) <= max_budget]
                if candidates:
                    selected_positions.extend(candidates[0])
            if not selected_positions:
                continue

            selected = torch.tensor(selected_positions, device=input_ids.device, dtype=torch.long)
            self._apply_mask_replacement_policy(
                input_ids,
                labels,
                b,
                selected,
                mask_prob=mask_prob,
                random_prob=random_prob,
                keep_prob=keep_prob,
                mask_token_id=mask_token_id,
            )

        return input_ids, labels

    def _build_word_groups(self, ids: Sequence[int], spec: Sequence[bool]) -> list[list[int]]:
        """Group token indices into whole words.

        Heuristics:
          - WordPiece continuation: token starts with '##'
          - SentencePiece continuation: token does NOT start with '▁'
          - GPT2/RoBERTa BPE continuation: token does NOT start with 'Ġ'

        We only group contiguous indices; any special/pad positions act as hard boundaries.

        :param Sequence[int] ids: Token ids for one sequence.
        :param Sequence[bool] spec: Boolean special-token mask for one sequence.
        :return list[list[int]]: Contiguous index groups representing words.
        """
        continuation_lookup = self._word_continuation_by_id
        if continuation_lookup is not None:
            lookup_size = len(continuation_lookup)
            groups: list[list[int]] = []
            prev_i: int | None = None
            for i, (tid, is_spec) in enumerate(zip(ids, spec, strict=True)):
                if is_spec:
                    prev_i = None
                    continue
                try:
                    tid_i = int(tid)
                except Exception:
                    prev_i = None
                    continue
                if tid_i in self._special_token_ids or tid_i < 0 or tid_i >= lookup_size:
                    prev_i = None
                    continue

                is_cont = bool(continuation_lookup[tid_i])
                start_new = not groups or prev_i is None or i != prev_i + 1 or not is_cont
                if start_new:
                    groups.append([i])
                else:
                    groups[-1].append(i)
                prev_i = i
            return groups

        tokens = self.tokenizer.convert_ids_to_tokens(list(ids))
        groups: list[list[int]] = []
        prev_i: int | None = None

        special_tokens = set(getattr(self.tokenizer, "all_special_tokens", []))
        scheme = self._word_boundary_scheme
        if scheme is None:
            lexical_tokens = [
                tok
                for tok, is_spec in zip(tokens, spec, strict=True)
                if (not is_spec) and tok is not None and tok != "" and tok not in special_tokens
            ]
            scheme = self._infer_word_boundary_scheme_from_tokens(lexical_tokens)
            self._word_boundary_scheme = scheme

        def _is_continuation(tok: str) -> bool:
            """Detect whether token text continues the previous word.

            :param str tok: Token string from tokenizer.
            :return bool: True if token should join previous group.
            """
            if tok.startswith("##"):
                return True
            if scheme == "sentencepiece":
                return not tok.startswith("▁")
            if scheme == "gpt2":
                return not tok.startswith("Ġ")
            # WordPiece tokenizers often emit plain tokens for word starts and
            # reserve only '##' for continuations.
            if scheme == "wordpiece":
                return False
            # Conservative fallback: if we cannot infer continuation markers,
            # avoid over-merging unrelated adjacent tokens.
            return False

        for i, (tok, is_spec) in enumerate(zip(tokens, spec, strict=True)):
            if is_spec:
                prev_i = None
                continue

            # Treat tokenizer special tokens (e.g. [PAD]) as hard boundaries even if
            # special_tokens_mask was not provided by the dataset.
            if tok in special_tokens:
                prev_i = None
                continue

            # Some tokenizers return None/'' for unknown ids; treat as boundary.
            if tok is None or tok == "":
                prev_i = None
                continue

            start_new = False
            if not groups or prev_i is None or i != prev_i + 1:
                start_new = True
            else:
                # Adjacent: decide based on token string.
                if not _is_continuation(tok):
                    start_new = True

            if start_new:
                groups.append([i])
            else:
                groups[-1].append(i)

            prev_i = i

        return groups
