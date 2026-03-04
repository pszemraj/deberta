"""Masking collator implementations used by RTD pretraining."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class MLMConfig:
    """Masking configuration.

    Notes:
      - Mask selection uses DeBERTa's windowed n-gram policy for all max_ngram>=1.
      - max_ngram == 1 corresponds to DeBERTa windowed unigram masking (not BERT iid masking).
      - max_ngram > 1 enables whole-word n-gram grouping.
      - Replacement probabilities are conditional on *being selected for masking*.
        (i.e., they should sum to <= 1; remainder keeps the original token).
    """

    mlm_probability: float
    mask_token_prob: float = 0.8
    random_token_prob: float = 0.1
    max_ngram: int = 1


class DebertaV3ElectraCollator:
    """Dynamic MLM masking collator suitable for RTD/ELECTRA-style pretraining.

    Produces:
      - input_ids (masked)
      - labels (original token ids at masked positions, -100 elsewhere)
      - attention_mask (optional; omitted for fully-unpadded batches)
      - token_type_ids (if present)

    Notes:
      - If `special_tokens_mask` is provided, we merge it with tokenizer-inferred
        special ids so upstream partial masks cannot unprotect tokenizer specials.
      - Default behavior mirrors BERT's 80/10/10 replacement.
      - Supports optional whole-word n-gram masking (max_ngram > 1) as used by DeBERTa.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        cfg: MLMConfig,
        packed_sequences: bool = False,
        block_cross_document_attention: bool = True,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        """Initialize collator state.

        :param Any tokenizer: HF tokenizer.
        :param MLMConfig cfg: Masking configuration.
        :param bool packed_sequences: Whether inputs are pre-packed with internal separators.
        :param bool block_cross_document_attention: Whether to emit 3D doc-block masks for packed inputs.
        :param int | None pad_to_multiple_of: Optional right-padding multiple.
        """
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.pad_to_multiple_of = pad_to_multiple_of
        self._packed_sequences = bool(packed_sequences)
        self._block_cross_document_attention = bool(block_cross_document_attention)

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
        features = self._harmonize_optional_attention_masks(features)

        # Let tokenizer handle padding for non-packed datasets.
        pad_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "pad_to_multiple_of": self.pad_to_multiple_of,
        }
        # If no padding is needed and the dataset does not provide attention_mask, avoid
        # materializing an all-ones mask.
        if not any("attention_mask" in f for f in features) and not self._needs_padding(features):
            pad_kwargs["return_attention_mask"] = False
        try:
            batch = self.tokenizer.pad(features, **pad_kwargs)
        except TypeError:
            # Some minimal tokenizer stubs do not accept return_attention_mask.
            pad_kwargs.pop("return_attention_mask", None)
            batch = self.tokenizer.pad(features, **pad_kwargs)

        # Safety fallback: if tokenizer did not emit an attention mask, infer one from
        # padded input_ids when pad tokens are present.
        if "attention_mask" not in batch:
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is not None:
                attn = batch["input_ids"].ne(int(pad_token_id))
                if not bool(attn.all().item()):
                    batch["attention_mask"] = attn.long()

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        inferred_special_tokens_mask = self._infer_special_tokens_mask(batch["input_ids"])
        if special_tokens_mask is None:
            special_tokens_mask = inferred_special_tokens_mask
        else:
            # Some upstream datasets (for example packed streaming) may only tag
            # structural specials (CLS/SEP/PAD). Union with tokenizer-level specials
            # so corruption targets stay aligned with forbidden sampling ids.
            special_tokens_mask = special_tokens_mask.bool() | inferred_special_tokens_mask

        doc_ids = (
            self._compute_document_ids(
                input_ids=batch["input_ids"],
                special_tokens_mask=special_tokens_mask,
                attention_mask=batch.get("attention_mask"),
            )
            if self._packed_sequences and self._block_cross_document_attention
            else None
        )
        if doc_ids is not None:
            batch["doc_ids"] = doc_ids
        else:
            # Packed/unpadded pretraining examples often have all-ones attention masks.
            # Drop all-ones masks so downstream can pass attention_mask=None to SDPA.
            attn = batch.get("attention_mask")
            if attn is not None:
                try:
                    if attn.dtype == torch.bool:
                        all_active = bool(attn.all().item())
                    else:
                        all_active = bool((attn == 1).all().item())
                    if all_active:
                        batch.pop("attention_mask", None)
                except Exception:
                    pass

        input_ids, labels = self._mask_tokens(batch["input_ids"], special_tokens_mask=special_tokens_mask)

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch

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

    def _sample_one_random_word(self, device: torch.device) -> int:
        """Sample one random replacement id.

        :param torch.device device: Target device.
        :return int: Sampled token id.
        """
        return int(self._sample_random_words((1,), device=device)[0].item())

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

        sep_id = getattr(self.tokenizer, "sep_token_id", None)
        if sep_id is None or input_ids.shape[1] < 3:
            return None
        sep_id = int(sep_id)

        sep_positions = input_ids.eq(sep_id) & special_tokens_mask

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if attention_mask is not None and attention_mask.ndim == 2:
            active = attention_mask.to(dtype=torch.bool)
        elif pad_id is not None:
            active = input_ids.ne(int(pad_id))
        else:
            active = torch.ones_like(input_ids, dtype=torch.bool)

        # Packed batches that contain only single-document chunks have no internal
        # separators and do not need doc-blocking metadata.
        internal_sep_positions = sep_positions[:, 1:-1] & active[:, 1:-1]
        if not bool(internal_sep_positions.any().item()):
            return None

        # Collapse contiguous separator runs into one boundary increment so packed
        # "... [SEP] [SEP] ..." tails do not create phantom empty-document segments.
        sep_prev = torch.zeros_like(sep_positions)
        sep_prev[:, 1:] = sep_positions[:, :-1]
        sep_boundaries = sep_positions & (~sep_prev)
        sep_before = sep_boundaries.long().cumsum(dim=1) - sep_boundaries.long()
        doc_ids = sep_before + 1

        cls_id = getattr(self.tokenizer, "cls_token_id", None)
        if cls_id is not None:
            cls_positions = input_ids.eq(int(cls_id)) & active
            # Keep CLS in document 1 so strict packed doc-blocking stays block-diagonal.
            doc_ids = doc_ids.masked_fill(cls_positions, 1)
        if pad_id is not None:
            doc_ids = doc_ids.masked_fill(input_ids.eq(int(pad_id)), 0)

        return doc_ids

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
        if mlm_prob <= 0.0:
            return input_ids, labels

        mask_prob = float(self.cfg.mask_token_prob)
        random_prob = float(self.cfg.random_token_prob)
        keep_prob = max(0.0, 1.0 - mask_prob - random_prob)
        mask_window = max(1, int(1.0 / mlm_prob))
        max_preds_per_seq = int(math.ceil(float(seq_len) * mlm_prob / 10.0) * 10)

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

            originals = input_ids[b].index_select(0, selected)
            labels[b].scatter_(0, selected, originals)

            if mask_prob >= 1.0 and random_prob <= 0.0:
                input_ids[b, selected] = mask_token_id
                continue

            roll = torch.rand(int(selected.numel()), device=input_ids.device, dtype=torch.float32)
            mask_sel = roll < mask_prob
            rand_sel = roll >= (mask_prob + keep_prob)

            if bool(mask_sel.any().item()):
                input_ids[b, selected[mask_sel]] = mask_token_id
            if random_prob > 0.0 and bool(rand_sel.any().item()):
                rand_ids = self._sample_random_words((int(rand_sel.sum().item()),), device=input_ids.device)
                input_ids[b, selected[rand_sel]] = rand_ids

        return input_ids, labels

    def _mask_tokens_bert(
        self, input_ids: torch.Tensor, *, special_tokens_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply token-level (independent) MLM masking.

        This helper is retained for targeted unit tests and optional ablation
        workflows. The default collator path intentionally uses DeBERTa-style
        windowed masking for ``max_ngram>=1`` via ``_mask_tokens``.

        :param torch.Tensor input_ids: Input token ids of shape (B, S).
        :param torch.Tensor special_tokens_mask: Special-token mask of shape (B, S).
        :return tuple[torch.Tensor, torch.Tensor]: Masked ids and MLM labels.
        """
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        input_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)

        mask_prob = float(self.cfg.mask_token_prob)
        random_prob = float(self.cfg.random_token_prob)
        mlm_prob = float(self.cfg.mlm_probability)
        pad_token_id = self.tokenizer.pad_token_id
        mask_token_id = int(self.tokenizer.mask_token_id)

        # Use a fixed budget per sequence (rounded from maskable count) to stabilize
        # effective masked-token counts across microbatches. This executes in eager
        # collator workers (not under torch.compile graph capture).
        maskable = ~special_tokens_mask
        if pad_token_id is not None:
            maskable = maskable & input_ids.ne(int(pad_token_id))

        candidate_counts = maskable.sum(dim=1)
        num_to_mask = torch.round(candidate_counts.to(torch.float32) * mlm_prob).to(torch.long)
        num_to_mask = torch.where(
            candidate_counts > 0,
            torch.clamp(num_to_mask, min=1),
            torch.zeros_like(num_to_mask),
        )
        num_to_mask = torch.minimum(num_to_mask, candidate_counts)

        max_to_mask = int(num_to_mask.max().item()) if int(num_to_mask.numel()) > 0 else 0
        if max_to_mask <= 0:
            return input_ids, labels

        # Row-wise random ranking over candidate positions, then take per-row top-k.
        scores = torch.rand(input_ids.shape, device=input_ids.device, dtype=torch.float32)
        scores = scores.masked_fill(~maskable, 2.0)
        topk_idx = torch.topk(scores, k=max_to_mask, dim=1, largest=False).indices
        topk_rank = torch.arange(max_to_mask, device=input_ids.device).unsqueeze(0)
        topk_valid = topk_rank < num_to_mask.unsqueeze(1)

        masked_indices = torch.zeros_like(maskable)
        masked_indices.scatter_(1, topk_idx, topk_valid)

        labels[masked_indices] = input_ids[masked_indices]

        repl_roll = torch.rand(input_ids.shape, device=input_ids.device, dtype=torch.float32)
        mask_sel = masked_indices & (repl_roll < mask_prob)
        rand_sel = masked_indices & (repl_roll >= mask_prob) & (repl_roll < (mask_prob + random_prob))

        if bool(mask_sel.any().item()):
            input_ids[mask_sel] = mask_token_id
        if bool(rand_sel.any().item()):
            rand_words = self._sample_random_words(input_ids.shape, device=input_ids.device)
            input_ids[rand_sel] = rand_words[rand_sel]

        # Remaining masked positions keep original token.
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

        if int(max_ngram) <= 1:
            return self._mask_tokens_unigram_windowed(input_ids, special_tokens_mask=special_tokens_mask)

        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        input_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)

        B, S = input_ids.shape
        mask_token_id = int(self.tokenizer.mask_token_id)
        mlm_prob = float(self.cfg.mlm_probability)
        if mlm_prob <= 0.0:
            return input_ids, labels
        mask_prob = float(self.cfg.mask_token_prob)
        random_prob = float(self.cfg.random_token_prob)
        keep_prob = max(0.0, 1.0 - mask_prob - random_prob)

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

        mask_window = max(1, int(1.0 / mlm_prob))
        max_preds_per_seq = int(math.ceil(float(S) * mlm_prob / 10.0) * 10)

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
            originals = input_ids[b].index_select(0, selected)
            labels[b].scatter_(0, selected, originals)

            if mask_prob >= 1.0 and random_prob <= 0.0:
                input_ids[b, selected] = mask_token_id
                continue

            roll = torch.rand(int(selected.numel()), device=input_ids.device, dtype=torch.float32)
            mask_sel = roll < mask_prob
            rand_sel = roll >= (mask_prob + keep_prob)

            if bool(mask_sel.any().item()):
                input_ids[b, selected[mask_sel]] = mask_token_id
            if random_prob > 0.0 and bool(rand_sel.any().item()):
                rand_ids = self._sample_random_words((int(rand_sel.sum().item()),), device=input_ids.device)
                input_ids[b, selected[rand_sel]] = rand_ids

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
