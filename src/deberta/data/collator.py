from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class MLMConfig:
    """Masking configuration.

    Notes:
      - Token-level masking (BERT-style) is used when max_ngram == 1.
      - Whole-word n-gram masking is enabled when max_ngram > 1.
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
      - attention_mask
      - token_type_ids (if present)

    Notes:
      - We honor `special_tokens_mask` if provided.
      - Default behavior mirrors BERT's 80/10/10 replacement.
      - Supports optional whole-word n-gram masking (max_ngram > 1) as used by DeBERTa.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        cfg: MLMConfig,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.pad_to_multiple_of = pad_to_multiple_of

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

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Let tokenizer handle padding for non-packed datasets.
        batch = self.tokenizer.pad(
            features,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if special_tokens_mask is None:
            special_tokens_mask = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        input_ids, labels = self._mask_tokens(batch["input_ids"], special_tokens_mask=special_tokens_mask)

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch

    def _mask_tokens(
        self, input_ids: torch.Tensor, *, special_tokens_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(self.cfg.max_ngram) <= 1:
            return self._mask_tokens_bert(input_ids, special_tokens_mask=special_tokens_mask)
        return self._mask_tokens_ngram(
            input_ids, special_tokens_mask=special_tokens_mask, max_ngram=int(self.cfg.max_ngram)
        )

    def _mask_tokens_bert(
        self, input_ids: torch.Tensor, *, special_tokens_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fast token-level (independent) masking."""
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        input_ids = input_ids.clone()
        labels = input_ids.clone()

        # Build mask probability matrix.
        prob = torch.full(labels.shape, float(self.cfg.mlm_probability), device=labels.device)

        # Never mask special tokens.
        prob.masked_fill_(special_tokens_mask, 0.0)

        # Never mask padding tokens.
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            prob.masked_fill_(labels.eq(pad_token_id), 0.0)

        masked_indices = torch.bernoulli(prob).bool()

        # Labels: only compute loss on masked tokens.
        labels[~masked_indices] = -100

        mask_prob = float(self.cfg.mask_token_prob)
        random_prob = float(self.cfg.random_token_prob)

        # mask_token_indices: of masked positions, which ones become [MASK]
        mask_token_indices = (
            torch.bernoulli(torch.full(labels.shape, mask_prob, device=labels.device)).bool() & masked_indices
        )
        input_ids[mask_token_indices] = int(self.tokenizer.mask_token_id)

        # random_token_indices: of remaining masked positions, which ones become random tokens
        random_token_indices = (
            torch.bernoulli(torch.full(labels.shape, random_prob, device=labels.device)).bool()
            & masked_indices
            & ~mask_token_indices
        )
        random_words = torch.randint(
            low=0, high=int(self.tokenizer.vocab_size), size=labels.shape, device=labels.device
        )
        input_ids[random_token_indices] = random_words[random_token_indices]

        # Remaining masked positions keep original token.
        return input_ids, labels

    def _mask_tokens_ngram(
        self,
        input_ids: torch.Tensor,
        *,
        special_tokens_mask: torch.Tensor,
        max_ngram: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Whole-word n-gram masking (DeBERTa-style).

        This is slower than token-level masking, but is a closer match to DeBERTa pretraining.

        Implementation notes:
          - We group sub-tokens into "words" using tokenizer token string heuristics.
          - We sample contiguous spans of whole words with an n-gram distribution p(n) ∝ 1/n.
        """

        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        input_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)

        B, S = input_ids.shape
        pad_token_id = self.tokenizer.pad_token_id
        mask_token_id = int(self.tokenizer.mask_token_id)
        vocab_size = int(self.tokenizer.vocab_size)

        # n-gram sampling distribution: p(n) ∝ 1/n
        probs = torch.tensor([1.0 / float(n) for n in range(1, max_ngram + 1)], dtype=torch.float)
        probs = probs / probs.sum()

        for b in range(B):
            ids = input_ids[b].tolist()
            spec = special_tokens_mask[b].tolist()

            maskable: list[int] = []
            for i, (tid, is_spec) in enumerate(zip(ids, spec)):
                if is_spec:
                    continue
                if pad_token_id is not None and tid == pad_token_id:
                    continue
                maskable.append(i)

            if not maskable:
                continue

            num_to_mask = max(1, int(round(len(maskable) * float(self.cfg.mlm_probability))))
            maskable_set = set(maskable)

            word_groups = self._build_word_groups(ids, spec)
            # Keep only groups that contain at least one maskable position
            word_groups = [g for g in word_groups if any(i in maskable_set for i in g)]
            if not word_groups:
                continue

            masked: list[int] = []
            masked_set = set()

            # Try a bounded number of attempts to fill the mask budget.
            # (Avoids pathological loops on very short / heavily special-token sequences.)
            max_attempts = max(50, 10 * len(word_groups))
            attempts = 0

            while len(masked) < num_to_mask and attempts < max_attempts:
                attempts += 1

                start = int(torch.randint(low=0, high=len(word_groups), size=(1,)).item())
                n = int(torch.multinomial(probs, 1).item()) + 1

                span = word_groups[start : start + n]
                if not span:
                    continue

                # Flatten span indices and filter already-masked.
                span_indices: list[int] = []
                for g in span:
                    for idx in g:
                        if idx in maskable_set and idx not in masked_set:
                            span_indices.append(idx)

                if not span_indices:
                    continue

                # Add tokens until we hit the budget.
                for idx in span_indices:
                    if len(masked) >= num_to_mask:
                        break
                    masked.append(idx)
                    masked_set.add(idx)

            if not masked:
                continue

            # Apply replacements.
            for idx in masked:
                labels[b, idx] = ids[idx]

                r = float(torch.rand(1).item())
                if r < float(self.cfg.mask_token_prob):
                    input_ids[b, idx] = mask_token_id
                elif r < float(self.cfg.mask_token_prob) + float(self.cfg.random_token_prob):
                    input_ids[b, idx] = int(torch.randint(low=0, high=vocab_size, size=(1,)).item())
                else:
                    # Keep original.
                    pass

        return input_ids, labels

    def _build_word_groups(self, ids: Sequence[int], spec: Sequence[bool]) -> list[list[int]]:
        """Group token indices into whole words.

        Heuristics:
          - WordPiece continuation: token starts with '##'
          - SentencePiece continuation: token does NOT start with '▁'
          - GPT2/RoBERTa BPE continuation: token does NOT start with 'Ġ'

        We only group contiguous indices; any special/pad positions act as hard boundaries.
        """

        tokens = self.tokenizer.convert_ids_to_tokens(list(ids))
        groups: list[list[int]] = []
        prev_i: int | None = None

        def _is_continuation(tok: str) -> bool:
            if tok.startswith("##"):
                return True
            if tok.startswith("▁") or tok.startswith("Ġ"):
                return False
            # Fallback: treat as continuation if adjacent.
            return True

        for i, (tok, is_spec) in enumerate(zip(tokens, spec)):
            if is_spec:
                prev_i = None
                continue

            # Treat tokenizer special tokens (e.g. [PAD]) as hard boundaries even if
            # special_tokens_mask was not provided by the dataset.
            if tok in getattr(self.tokenizer, "all_special_tokens", []):
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
