from __future__ import annotations

from typing import Any

import torch

from deberta.data.collator import DebertaV3ElectraCollator, MLMConfig
from deberta.data.streaming import PackedStreamingConfig, PackedStreamingDataset


class DummyTokenizer:
    """Minimal tokenizer stub for unit tests (no network/model downloads)."""

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.mask_token_id = 3
        self.all_special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        self._id_to_tok = {
            self.pad_token_id: "[PAD]",
            self.cls_token_id: "[CLS]",
            self.sep_token_id: "[SEP]",
            self.mask_token_id: "[MASK]",
        }

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
        return_token_type_ids: bool = False,
    ) -> dict[str, list[int]]:
        # Cheap whitespace tokenizer. Map each "word" to a stable-ish ID.
        ids: list[int] = []
        for w in text.strip().split():
            # Keep ids away from special range.
            ids.append(10 + (abs(hash(w)) % (self.vocab_size - 10)))
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        out: list[str] = []
        for i in ids:
            if i in self._id_to_tok:
                out.append(self._id_to_tok[i])
            else:
                # SentencePiece-like word boundary marker.
                out.append("▁" + str(i))
        return out

    def pad(self, features: list[dict[str, Any]], return_tensors: str = "pt", pad_to_multiple_of: int | None = None):
        # Minimal padding for collator tests.
        max_len = max(len(f["input_ids"]) for f in features)
        if pad_to_multiple_of is not None:
            if max_len % pad_to_multiple_of != 0:
                max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

        batch: dict[str, list[list[int]]] = {}
        for k in features[0].keys():
            batch[k] = []
        for f in features:
            for k, v in f.items():
                if isinstance(v, list):
                    pad_val = 0
                    if k == "attention_mask":
                        pad_val = 0
                    elif k == "special_tokens_mask":
                        pad_val = 1
                    batch[k].append(v + [pad_val] * (max_len - len(v)))
                else:
                    raise TypeError(f"Unsupported feature type for {k}: {type(v)}")

        if "attention_mask" not in batch:
            batch["attention_mask"] = [[1] * len(v) + [0] * (max_len - len(v)) for v in batch["input_ids"]]

        if return_tensors == "pt":
            return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        return batch


def test_packed_streaming_marks_internal_sep_as_special():
    tok = DummyTokenizer(vocab_size=64)

    # Two docs, each 2 tokens. With max_seq=8, block_len=6, chunk will include internal seps.
    hf_dataset = [{"text": "a b"}, {"text": "c d"}]

    ds = PackedStreamingDataset(
        hf_dataset=hf_dataset,
        tokenizer=tok,
        cfg=PackedStreamingConfig(text_column_name="text", max_seq_length=8, seed=0, shuffle_buffer_size=0),
        process_index=0,
        num_processes=1,
    )

    ex = next(iter(ds))
    input_ids = ex["input_ids"]
    stm = ex["special_tokens_mask"]

    # Expect at least one internal sep.
    assert tok.sep_token_id in input_ids[1:-1]

    # All sep/cls/pad tokens should be marked special.
    for i, tid in enumerate(input_ids):
        if tid in {tok.sep_token_id, tok.cls_token_id, tok.pad_token_id}:
            assert stm[i] == 1


def test_ngram_masking_respects_specials():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.5, max_ngram=3))

    # Build a fake batch with specials.
    input_ids = torch.tensor(
        [
            [tok.cls_token_id, 11, 12, tok.sep_token_id, 13, tok.pad_token_id, tok.pad_token_id],
        ],
        dtype=torch.long,
    )
    special = torch.tensor([[1, 0, 0, 1, 0, 1, 1]], dtype=torch.bool)

    masked, labels = coll._mask_tokens_ngram(input_ids, special_tokens_mask=special, max_ngram=3)

    # Never compute loss on specials.
    assert labels[0, 0].item() == -100
    assert labels[0, 3].item() == -100
    assert labels[0, 5].item() == -100

    # Never replace specials.
    assert masked[0, 0].item() == tok.cls_token_id
    assert masked[0, 3].item() == tok.sep_token_id
    assert masked[0, 5].item() == tok.pad_token_id


def test_pretrainer_forward_smoke():
    """Requires transformers; skipped automatically if not installed."""

    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    # Tiny configs.
    disc_cfg = DebertaRoPEConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        norm_arch="post",
    )
    gen_cfg = DebertaRoPEConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        norm_arch="post",
    )

    disc = DebertaRoPEModel(disc_cfg)
    gen = DebertaRoPEModel(gen_cfg)

    model = DebertaV3RTDPretrainer(
        discriminator_backbone=disc,
        generator_backbone=gen,
        disc_config=disc_cfg,
        gen_config=gen_cfg,
        embedding_sharing="gdes",
    )

    B, S = 2, 16
    input_ids = torch.randint(low=0, high=128, size=(B, S), dtype=torch.long)
    attention_mask = torch.ones((B, S), dtype=torch.long)

    labels = torch.full((B, S), -100, dtype=torch.long)
    labels[:, 3] = input_ids[:, 3]
    labels[:, 7] = input_ids[:, 7]

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        sampling_temperature=1.0,
        gen_loss_weight=1.0,
        disc_loss_weight=50.0,
        decoupled_loss_scaling=False,
    )

    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.disc_accuracy.ndim == 0
