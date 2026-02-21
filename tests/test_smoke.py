from __future__ import annotations

from typing import Any

import pytest
import torch

from deberta.data.collator import DebertaV3ElectraCollator, MLMConfig
from deberta.data.streaming import PackedStreamingConfig, PackedStreamingDataset, SequentialStreamingDataset


class DummyTokenizer:
    """Minimal tokenizer stub for unit tests (no network/model downloads)."""

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.mask_token_id = 3
        self.all_special_ids = [self.pad_token_id, self.cls_token_id, self.sep_token_id, self.mask_token_id]
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

    def get_special_tokens_mask(
        self, token_ids_0: list[int], already_has_special_tokens: bool = True
    ) -> list[int]:
        del already_has_special_tokens
        specials = set(self.all_special_ids)
        return [1 if int(tid) in specials else 0 for tid in token_ids_0]

    def pad(
        self,
        features: list[dict[str, Any]],
        return_tensors: str = "pt",
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool = True,
    ):
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

        if "attention_mask" not in batch and return_attention_mask:
            batch["attention_mask"] = [
                [0 if tid == self.pad_token_id else 1 for tid in row] for row in batch["input_ids"]
            ]

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
    assert "attention_mask" not in ex
    input_ids = ex["input_ids"]
    stm = ex["special_tokens_mask"]

    # Expect at least one internal sep.
    assert tok.sep_token_id in input_ids[1:-1]

    # All sep/cls/pad tokens should be marked special.
    for i, tid in enumerate(input_ids):
        if tid in {tok.sep_token_id, tok.cls_token_id, tok.pad_token_id}:
            assert stm[i] == 1


def test_packed_streaming_flushes_tail_instead_of_dropping():
    tok = DummyTokenizer(vocab_size=64)
    hf_dataset = [{"text": "a b"}, {"text": "c d"}, {"text": "e"}]

    ds = PackedStreamingDataset(
        hf_dataset=hf_dataset,
        tokenizer=tok,
        cfg=PackedStreamingConfig(text_column_name="text", max_seq_length=8, seed=0, shuffle_buffer_size=0),
        process_index=0,
        num_processes=1,
    )
    rows = list(ds)
    assert len(rows) == 2
    assert "attention_mask" in rows[1]
    assert (tok.pad_token_id in rows[1]["input_ids"]) is True


def test_packed_streaming_does_not_emit_separator_only_tail_chunk():
    tok = DummyTokenizer(vocab_size=64)
    # max_seq=8 => block_len=6, so this document fills exactly one block.
    hf_dataset = [{"text": "a b c d e f"}]

    ds = PackedStreamingDataset(
        hf_dataset=hf_dataset,
        tokenizer=tok,
        cfg=PackedStreamingConfig(text_column_name="text", max_seq_length=8, seed=0, shuffle_buffer_size=0),
        process_index=0,
        num_processes=1,
    )
    rows = list(ds)

    # We should emit exactly one fully useful row, not an extra [CLS, SEP, SEP, PAD...] tail.
    assert len(rows) == 1
    ex = rows[0]
    assert ex["input_ids"][0] == tok.cls_token_id
    assert ex["input_ids"][-1] == tok.sep_token_id
    assert tok.pad_token_id not in ex["input_ids"]
    assert "attention_mask" not in ex
    assert ex["special_tokens_mask"] == [1, 0, 0, 0, 0, 0, 0, 1]


def test_packed_streaming_epoch_changes_shuffle_seed():
    class _ShuffleProbeDataset:
        def __init__(self) -> None:
            self.last_seed: int | None = None
            self._rows = [{"text": "a b c"}]

        def shuffle(self, *, buffer_size: int, seed: int):
            del buffer_size
            self.last_seed = int(seed)
            return self

        def shard(self, *, num_shards: int, index: int):
            del num_shards
            del index
            return self

        def __iter__(self):
            return iter(self._rows)

    tok = DummyTokenizer(vocab_size=64)
    probe = _ShuffleProbeDataset()
    ds = PackedStreamingDataset(
        hf_dataset=probe,
        tokenizer=tok,
        cfg=PackedStreamingConfig(text_column_name="text", max_seq_length=8, seed=7, shuffle_buffer_size=16),
        process_index=0,
        num_processes=1,
    )

    _ = list(ds)
    assert probe.last_seed == 7

    ds.set_epoch(3)
    _ = list(ds)
    assert probe.last_seed == 10


def test_sequential_streaming_splits_long_documents_without_cross_doc_packing():
    tok = DummyTokenizer(vocab_size=64)
    hf_dataset = [{"text": "a b c d e f g h i"}]

    ds = SequentialStreamingDataset(
        hf_dataset=hf_dataset,
        tokenizer=tok,
        cfg=PackedStreamingConfig(text_column_name="text", max_seq_length=8, seed=0, shuffle_buffer_size=0),
        process_index=0,
        num_processes=1,
    )
    rows = list(ds)

    # 9 lexical tokens with max_seq=8 (block_len=6) => two one-document chunks.
    assert len(rows) == 2
    for ex in rows:
        mids = ex["input_ids"][1:-1]
        # No cross-document separators in sequential one-document mode.
        assert sum(1 for tid in mids if tid == tok.sep_token_id) <= 1


def test_collator_builds_document_block_attention_mask_when_packed():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(
        tokenizer=tok,
        cfg=MLMConfig(mlm_probability=0.2, max_ngram=1),
        packed_sequences=True,
        block_cross_document_attention=True,
    )

    features = [
        {
            "input_ids": [tok.cls_token_id, 11, tok.sep_token_id, 12, 13, tok.sep_token_id],
            "special_tokens_mask": [1, 0, 1, 0, 0, 1],
        }
    ]
    batch = coll(features)
    assert "attention_mask" in batch
    attn = batch["attention_mask"]
    assert attn.ndim == 3
    assert attn.dtype == torch.bool

    # Doc 1: positions {1,2}; Doc 2: positions {3,4,5}. Cross-doc attention blocked.
    assert attn[0, 1, 3].item() == 0
    assert attn[0, 3, 1].item() == 0
    assert attn[0, 1, 2].item() == 1
    assert attn[0, 3, 5].item() == 1
    # CLS remains in the first packed document under strict block-diagonal masking.
    assert attn[0, 0, 1].item() == 1
    assert attn[0, 1, 0].item() == 1
    assert attn[0, 0, 3].item() == 0
    assert attn[0, 3, 0].item() == 0


def test_collator_treats_consecutive_internal_separators_as_single_boundary():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(
        tokenizer=tok,
        cfg=MLMConfig(mlm_probability=0.2, max_ngram=1),
        packed_sequences=True,
        block_cross_document_attention=True,
    )

    # Boundary-aligned packed chunks can produce consecutive separators.
    features = [
        {
            "input_ids": [tok.cls_token_id, 11, tok.sep_token_id, tok.sep_token_id, 12, tok.sep_token_id],
            "special_tokens_mask": [1, 0, 1, 1, 0, 1],
        }
    ]
    batch = coll(features)
    attn = batch["attention_mask"]
    assert attn.ndim == 3

    # Token in doc1 should not attend doc2 token.
    assert attn[0, 1, 4].item() == 0
    assert attn[0, 4, 1].item() == 0
    # Consecutive [SEP] should not create an extra phantom boundary.
    assert attn[0, 3, 4].item() == 1
    assert attn[0, 4, 3].item() == 1


def test_collator_skips_document_block_attention_mask_for_single_doc_packed_chunk():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(
        tokenizer=tok,
        cfg=MLMConfig(mlm_probability=0.2, max_ngram=1),
        packed_sequences=True,
        block_cross_document_attention=True,
    )

    features = [
        {
            "input_ids": [tok.cls_token_id, 11, 12, tok.sep_token_id],
            "special_tokens_mask": [1, 0, 0, 1],
        }
    ]
    batch = coll(features)
    assert "attention_mask" not in batch


def test_collator_build_drops_document_mask_when_not_packed():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.2, max_ngram=1))

    features = [
        {
            "input_ids": [tok.cls_token_id, 11, tok.sep_token_id, 12, 13, tok.sep_token_id],
            "special_tokens_mask": [1, 0, 1, 0, 0, 1],
        }
    ]
    batch = coll(features)
    assert "attention_mask" not in batch


def test_ngram_masking_fills_shortfall_from_remaining_candidates(monkeypatch: pytest.MonkeyPatch):
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.75, max_ngram=3))

    input_ids = torch.tensor([[tok.cls_token_id, 11, 12, 13, 14, tok.sep_token_id]], dtype=torch.long)
    special = torch.tensor([[1, 0, 0, 0, 0, 1]], dtype=torch.bool)

    def fake_randint(low: int, high: int, size, **kwargs):
        del kwargs
        return torch.tensor([0], dtype=torch.long)

    def fake_multinomial(input: torch.Tensor, num_samples: int, replacement: bool = False):
        return torch.tensor([0], dtype=torch.long)

    monkeypatch.setattr(torch, "randint", fake_randint)
    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    masked, labels = coll._mask_tokens_ngram(input_ids, special_tokens_mask=special, max_ngram=3)

    # With 4 maskable positions and 75% target ratio, fallback should reach 3 masked positions.
    assert int(labels.ne(-100).sum().item()) == 3


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


def test_token_level_masking_uses_fixed_budget_per_sequence():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.2, max_ngram=1))

    input_ids = torch.tensor(
        [[tok.cls_token_id, 11, 12, 13, 14, 15, tok.sep_token_id]],
        dtype=torch.long,
    )
    special = torch.tensor([[1, 0, 0, 0, 0, 0, 1]], dtype=torch.bool)

    counts = []
    for seed in range(10):
        torch.manual_seed(seed)
        _, labels = coll._mask_tokens_bert(input_ids, special_tokens_mask=special)
        counts.append(int(labels.ne(-100).sum().item()))

    assert set(counts) == {1}


def test_token_level_masking_uses_fixed_budget_for_variable_length_batch():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.4, max_ngram=1))

    features = [
        {
            "input_ids": [tok.cls_token_id, 10, 11, 12, tok.sep_token_id],
            "special_tokens_mask": [1, 0, 0, 0, 1],
        },
        {"input_ids": [tok.cls_token_id, 13, 14, tok.sep_token_id], "special_tokens_mask": [1, 0, 0, 1]},
    ]
    batch = coll(features)
    labels = batch["labels"]

    masked_counts = labels.ne(-100).sum(dim=1).tolist()
    # seq0 has 3 candidates => round(1.2)=1, seq1 has 2 candidates => round(0.8)=1
    assert masked_counts == [1, 1]


def test_ngram_wordpiece_like_tokens_do_not_overmerge_groups():
    class WordPieceLikeTokenizer(DummyTokenizer):
        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            table = {
                self.pad_token_id: "[PAD]",
                self.cls_token_id: "[CLS]",
                self.sep_token_id: "[SEP]",
                self.mask_token_id: "[MASK]",
                10: "the",
                11: "cat",
                12: "sat",
            }
            return [table.get(i, f"tok{i}") for i in ids]

    tok = WordPieceLikeTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.5, max_ngram=3))

    ids = [tok.cls_token_id, 10, 11, 12, tok.sep_token_id]
    spec = [1, 0, 0, 0, 1]
    groups = coll._build_word_groups(ids, spec)
    assert groups == [[1], [2], [3]]


def test_word_boundary_scheme_detected_once_from_tokenizer_probe():
    class ProbeWordPieceTokenizer(DummyTokenizer):
        def tokenize(self, text: str) -> list[str]:
            del text
            return ["hello", "##world"]

        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            table = {
                self.pad_token_id: "[PAD]",
                self.cls_token_id: "[CLS]",
                self.sep_token_id: "[SEP]",
                self.mask_token_id: "[MASK]",
                10: "Ġalpha",
                11: "beta",
                12: "Ġgamma",
            }
            return [table.get(i, f"tok{i}") for i in ids]

    tok = ProbeWordPieceTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.5, max_ngram=3))

    ids = [tok.cls_token_id, 10, 11, 12, tok.sep_token_id]
    spec = [1, 0, 0, 0, 1]
    groups = coll._build_word_groups(ids, spec)
    assert groups == [[1], [2], [3]]


def test_collator_drops_all_ones_attention_mask():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.2, max_ngram=1))

    features = [
        {"input_ids": [tok.cls_token_id, 11, 12, tok.sep_token_id], "special_tokens_mask": [1, 0, 0, 1]},
        {"input_ids": [tok.cls_token_id, 13, 14, tok.sep_token_id], "special_tokens_mask": [1, 0, 0, 1]},
    ]
    batch = coll(features)
    assert "attention_mask" not in batch


def test_collator_keeps_attention_mask_when_padding_present():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.2, max_ngram=1))

    features = [
        {
            "input_ids": [tok.cls_token_id, 11, tok.sep_token_id],
            "attention_mask": [1, 1, 1],
            "special_tokens_mask": [1, 0, 1],
        },
        {
            "input_ids": [tok.cls_token_id, 12, 13, tok.sep_token_id],
            "attention_mask": [1, 1, 1, 1],
            "special_tokens_mask": [1, 0, 0, 1],
        },
    ]
    batch = coll(features)
    assert "attention_mask" in batch
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert (batch["attention_mask"] == 0).any()


def test_collator_generates_attention_mask_for_variable_length_inputs():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.2, max_ngram=1))

    # No attention_mask in features; tokenizer padding is required.
    features = [
        {"input_ids": [tok.cls_token_id, 11, tok.sep_token_id], "special_tokens_mask": [1, 0, 1]},
        {"input_ids": [tok.cls_token_id, 12, 13, tok.sep_token_id], "special_tokens_mask": [1, 0, 0, 1]},
    ]
    batch = coll(features)

    assert "attention_mask" in batch
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert (batch["attention_mask"] == 0).any()


def test_collator_handles_mixed_attention_mask_keys():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.2, max_ngram=1))

    with_mask = {
        "input_ids": [tok.cls_token_id, 11, tok.sep_token_id],
        "attention_mask": [1, 1, 1],
        "special_tokens_mask": [1, 0, 1],
    }
    without_mask = {
        "input_ids": [tok.cls_token_id, 12, 13, tok.sep_token_id],
        "special_tokens_mask": [1, 0, 0, 1],
    }

    batch_a = coll([with_mask, without_mask])
    batch_b = coll([without_mask, with_mask])

    for batch in (batch_a, batch_b):
        assert "attention_mask" in batch
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        assert (batch["attention_mask"] == 0).any()


def test_collator_infers_special_tokens_mask_when_missing():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.9, max_ngram=1))

    # No special_tokens_mask provided: collator should infer specials and avoid masking them.
    features = [
        {"input_ids": [tok.cls_token_id, 11, tok.sep_token_id]},
        {"input_ids": [tok.cls_token_id, 12, 13, tok.sep_token_id]},
    ]
    batch = coll(features)

    # CLS / SEP / PAD positions are special and should not contribute MLM loss.
    assert torch.all(batch["labels"][:, 0] == -100)
    assert torch.all(batch["labels"][batch["input_ids"] == tok.sep_token_id] == -100)
    assert torch.all(batch["labels"][batch["input_ids"] == tok.pad_token_id] == -100)


def test_collator_random_replacement_avoids_special_ids():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(
        tokenizer=tok,
        cfg=MLMConfig(mlm_probability=0.999, mask_token_prob=0.0, random_token_prob=1.0, max_ngram=1),
    )

    torch.manual_seed(0)
    input_ids = torch.arange(10, 266, dtype=torch.long).view(1, -1) % tok.vocab_size
    special = torch.zeros_like(input_ids, dtype=torch.bool)

    masked, labels = coll._mask_tokens_bert(input_ids, special_tokens_mask=special)
    changed = labels.ne(-100)
    assert bool(changed.any().item())

    replaced = masked[changed]
    for sid in tok.all_special_ids:
        assert not bool((replaced == sid).any().item())


def test_self_attention_zeroes_padded_query_outputs():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPESelfAttention

    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        attention_implementation="eager",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    attn = DebertaRoPESelfAttention(cfg).eval()
    x = torch.randn((2, 6, cfg.hidden_size), dtype=torch.float32)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        out = attn(x, attention_mask)

    assert torch.allclose(out[0, 4:, :], torch.zeros_like(out[0, 4:, :]), atol=1e-6)
    assert torch.allclose(out[1, 5:, :], torch.zeros_like(out[1, 5:, :]), atol=1e-6)


def test_self_attention_has_no_internal_residual_dropout():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPESelfAttention

    torch.manual_seed(0)
    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        attention_implementation="eager",
        hidden_dropout_prob=0.9,
        attention_probs_dropout_prob=0.0,
    )
    attn = DebertaRoPESelfAttention(cfg)
    x = torch.randn((2, 6, cfg.hidden_size), dtype=torch.float32)
    attention_mask = torch.ones((2, 6), dtype=torch.long)

    with torch.no_grad():
        attn.eval()
        out_eval = attn(x, attention_mask)
        attn.train()
        out_train = attn(x, attention_mask)

    torch.testing.assert_close(out_train, out_eval, rtol=0.0, atol=0.0)


def test_self_attention_handles_pairwise_mask_rows_without_keys():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPESelfAttention

    torch.manual_seed(0)
    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        attention_implementation="eager",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    attn = DebertaRoPESelfAttention(cfg).eval()
    x = torch.randn((1, 4, cfg.hidden_size), dtype=torch.float32)
    pair_keep = torch.tensor(
        [
            [
                [1, 1, 0, 0],
                [0, 0, 0, 0],  # row with no valid keys
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ]
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        out = attn(x, pair_keep)

    assert torch.isfinite(out).all()
    assert torch.allclose(out[0, 1], torch.zeros_like(out[0, 1]), atol=1e-6)


def test_self_attention_zeroes_padded_query_outputs_for_pairwise_mask():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPESelfAttention

    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        attention_implementation="eager",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    attn = DebertaRoPESelfAttention(cfg).eval()
    x = torch.randn((1, 4, cfg.hidden_size), dtype=torch.float32)
    # Last row/col correspond to pad-like token with no valid edges.
    pair_keep = torch.tensor(
        [
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ]
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        out = attn(x, pair_keep)

    assert torch.isfinite(out).all()
    assert torch.allclose(out[0, 3], torch.zeros_like(out[0, 3]), atol=1e-6)


def test_mlp_has_no_internal_residual_dropout():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEMLP

    torch.manual_seed(0)
    for ffn_type in ("mlp", "swiglu"):
        cfg = DebertaRoPEConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
            type_vocab_size=0,
            ffn_type=ffn_type,
            hidden_dropout_prob=0.9,
            attention_probs_dropout_prob=0.0,
        )
        mlp = DebertaRoPEMLP(cfg)
        x = torch.randn((2, 6, cfg.hidden_size), dtype=torch.float32)

        with torch.no_grad():
            mlp.eval()
            out_eval = mlp(x)
            mlp.train()
            out_train = mlp(x)

        torch.testing.assert_close(out_train, out_eval, rtol=0.0, atol=0.0)


def test_self_attention_sdpa_matches_eager_with_padding_mask():
    import pytest
    import torch.nn.functional as F

    pytest.importorskip("transformers")
    if not hasattr(F, "scaled_dot_product_attention"):
        pytest.skip("torch.nn.functional.scaled_dot_product_attention is not available")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPESelfAttention

    torch.manual_seed(0)
    cfg_kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg_sdpa = DebertaRoPEConfig(attention_implementation="sdpa", **cfg_kwargs)
    cfg_eager = DebertaRoPEConfig(attention_implementation="eager", **cfg_kwargs)

    attn_sdpa = DebertaRoPESelfAttention(cfg_sdpa).eval()
    attn_eager = DebertaRoPESelfAttention(cfg_eager).eval()
    attn_eager.load_state_dict(attn_sdpa.state_dict())

    # Isolate mask behavior from rotary implementation details.
    attn_sdpa.rope = None
    attn_eager.rope = None

    x = torch.randn((2, 6, cfg_sdpa.hidden_size), dtype=torch.float32)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        out_sdpa = attn_sdpa(x, attention_mask)
        out_eager = attn_eager(x, attention_mask)

    torch.testing.assert_close(out_sdpa, out_eager, rtol=1e-5, atol=1e-6)


def test_rope_projections_respect_use_bias_config():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEMLP, DebertaRoPESelfAttention

    base_kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )

    cfg_no_bias = DebertaRoPEConfig(use_bias=False, **base_kwargs)
    attn_no_bias = DebertaRoPESelfAttention(cfg_no_bias)
    mlp_no_bias = DebertaRoPEMLP(cfg_no_bias)
    assert attn_no_bias.qkv.bias is None
    assert attn_no_bias.out_proj.bias is None
    assert mlp_no_bias.w12.bias is None
    assert mlp_no_bias.w3.bias is None

    cfg_with_bias = DebertaRoPEConfig(use_bias=True, **base_kwargs)
    attn_with_bias = DebertaRoPESelfAttention(cfg_with_bias)
    mlp_with_bias = DebertaRoPEMLP(cfg_with_bias)
    assert attn_with_bias.qkv.bias is not None
    assert attn_with_bias.out_proj.bias is not None
    assert mlp_with_bias.w12.bias is not None
    assert mlp_with_bias.w3.bias is not None


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
        ffn_type="swiglu",
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
        ffn_type="mlp",
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
    labels = torch.full((B, S), -100, dtype=torch.long)
    labels[:, 3] = input_ids[:, 3]
    labels[:, 7] = input_ids[:, 7]

    out = model(
        input_ids=input_ids,
        labels=labels,
        sampling_temperature=1.0,
        gen_loss_weight=1.0,
        disc_loss_weight=50.0,
        decoupled_loss_scaling=False,
    )

    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.disc_accuracy.ndim == 0


def test_pretrainer_sampler_avoids_configured_special_ids():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    cfg = DebertaRoPEConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        type_vocab_size=0,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        mask_token_id=3,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        norm_arch="post",
    )
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=DebertaRoPEModel(cfg),
        generator_backbone=DebertaRoPEModel(cfg),
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="none",
    ).eval()

    logits = torch.full((256, cfg.vocab_size), -10.0, dtype=torch.float32)
    logits[:, 0] = 20.0
    logits[:, 1] = 19.0
    logits[:, 2] = 18.0
    logits[:, 3] = 17.0

    with torch.no_grad():
        sampled = model._sample_generator_tokens(logits, sampling_temperature=1.0)

    for sid in (cfg.pad_token_id, cfg.cls_token_id, cfg.sep_token_id, cfg.mask_token_id):
        assert sid is not None
        assert not bool((sampled == int(sid)).any().item())


def test_masked_lm_head_tied_mode_avoids_unused_decoder_allocation():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig
    from deberta.modeling.rtd import MaskedLMHead

    cfg = DebertaRoPEConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
    )
    head = MaskedLMHead(cfg, tie_word_embeddings=True)
    assert head.decoder is None
    assert head.bias.shape == (cfg.vocab_size,)


def test_masked_lm_head_tied_mode_requires_embedding_weight():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig
    from deberta.modeling.rtd import MaskedLMHead

    cfg = DebertaRoPEConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
    )
    head = MaskedLMHead(cfg, tie_word_embeddings=True)
    hidden = torch.randn((2, cfg.hidden_size), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="requires `word_embedding_weight`"):
        _ = head(hidden)


def test_mlm_and_rtd_heads_use_layernorm_when_rmsnorm_heads_disabled():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rtd import MLMTransform, RTDHead

    class _Cfg:
        hidden_size = 16
        hidden_act = "gelu"
        hidden_dropout_prob = 0.0
        layer_norm_eps = 1.0e-5
        use_rmsnorm_heads = False

    mlm = MLMTransform(_Cfg())
    rtd = RTDHead(_Cfg())
    assert isinstance(mlm.norm, torch.nn.LayerNorm)
    assert isinstance(rtd.norm, torch.nn.LayerNorm)


def test_masked_lm_head_tied_mode_aligns_to_weight_dtype_outside_autocast():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig
    from deberta.modeling.rtd import MaskedLMHead

    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        type_vocab_size=0,
    )
    head = MaskedLMHead(cfg, tie_word_embeddings=True)
    head.transform = torch.nn.Identity()

    hidden = torch.randn((3, cfg.hidden_size), dtype=torch.float64)
    word_w = torch.randn((cfg.vocab_size, cfg.hidden_size), dtype=torch.float32)

    with torch.no_grad():
        logits = head(hidden, word_embedding_weight=word_w)

    assert logits.dtype == torch.float32


def test_rtd_head_applies_dropout_once_per_forward():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig
    from deberta.modeling.rtd import RTDHead

    class _CountingDropout(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return x

    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.0,
    )
    head = RTDHead(cfg)
    counting_dropout = _CountingDropout()
    head.dropout = counting_dropout

    hidden = torch.randn((2, 4, cfg.hidden_size), dtype=torch.float32)
    _ = head(hidden)
    assert counting_dropout.calls == 1


def test_pretrainer_raises_clear_error_when_generator_word_embeddings_cannot_be_tied():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    class _EmbeddingWithoutWeight(torch.nn.Module):
        def __init__(self, base: torch.nn.Module) -> None:
            super().__init__()
            self.base = base

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.base(input_ids)

    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        pad_token_id=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        norm_arch="post",
    )
    generator = DebertaRoPEModel(cfg)
    generator.embeddings.word_embeddings = _EmbeddingWithoutWeight(generator.embeddings.word_embeddings)
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=DebertaRoPEModel(cfg),
        generator_backbone=generator,
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="none",
    )
    input_ids = torch.tensor([[1, 7, 8, 2]], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[0, 1] = input_ids[0, 1]

    with pytest.raises(RuntimeError, match="word_embeddings must expose a `.weight`"):
        _ = model(
            input_ids=input_ids,
            attention_mask=None,
            labels=labels,
            sampling_temperature=1.0,
            gen_loss_weight=1.0,
            disc_loss_weight=50.0,
            decoupled_loss_scaling=False,
        )


def test_tied_embedding_tracks_replaced_base_module_weight():
    from deberta.modeling.rtd import _TiedEmbedding

    base = torch.nn.Embedding(16, 8, padding_idx=0)
    tied = _TiedEmbedding(
        base_embedding_module=base,
        padding_idx=0,
        detach_base=False,
        add_bias=False,
    )
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with torch.no_grad():
        out_a = tied(input_ids)
        new_weight = torch.randn_like(base.weight)
        base.weight = torch.nn.Parameter(new_weight)
        out_b = tied(input_ids)

    assert not torch.allclose(out_a, out_b)
    expected = torch.nn.functional.embedding(input_ids, new_weight, padding_idx=0)
    torch.testing.assert_close(out_b, expected, rtol=0.0, atol=0.0)


def test_tied_embedding_gdes_bias_matches_base_weight_dtype():
    from deberta.modeling.rtd import _TiedEmbedding

    base = torch.nn.Embedding(16, 8, padding_idx=0).to(dtype=torch.bfloat16)
    tied = _TiedEmbedding(
        base_embedding_module=base,
        padding_idx=0,
        detach_base=True,
        add_bias=True,
    )
    assert tied.bias is not None
    assert tied.bias.dtype == base.weight.dtype


def test_pretrainer_raises_if_tied_embeddings_become_stale_after_model_surgery():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        pad_token_id=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        norm_arch="post",
    )
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=DebertaRoPEModel(cfg),
        generator_backbone=DebertaRoPEModel(cfg),
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="es",
    )
    model.generator.embeddings.word_embeddings = torch.nn.Embedding(
        cfg.vocab_size,
        cfg.hidden_size,
        padding_idx=cfg.pad_token_id,
    )

    input_ids = torch.tensor([[1, 7, 8, 2]], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[0, 1] = input_ids[0, 1]

    with pytest.raises(RuntimeError, match="stale tied embeddings"):
        _ = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
            sampling_temperature=1.0,
            gen_loss_weight=1.0,
            disc_loss_weight=50.0,
            decoupled_loss_scaling=False,
        )


def test_rope_model_treats_missing_attention_mask_as_unpadded_contract():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel

    torch.manual_seed(0)
    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        pad_token_id=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        norm_arch="post",
    )
    model = DebertaRoPEModel(cfg).eval()

    class _CaptureEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_attention_mask: torch.Tensor | None = None

        def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
            self.seen_attention_mask = attention_mask
            return x

    input_ids = torch.tensor(
        [
            [1, 7, 8, 2, 0, 0],
            [1, 9, 10, 11, 2, 0],
        ],
        dtype=torch.long,
    )
    capture = _CaptureEncoder()
    model.encoder = capture

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=None).last_hidden_state

    assert capture.seen_attention_mask is None


def test_rope_model_accepts_positional_input_ids_call():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel

    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        norm_arch="post",
    )
    model = DebertaRoPEModel(cfg).eval()

    input_ids = torch.randint(low=0, high=cfg.vocab_size, size=(2, 6), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out_positional = model(input_ids, attention_mask=attention_mask).last_hidden_state
        out_keyword = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    torch.testing.assert_close(out_positional, out_keyword, rtol=0.0, atol=0.0)


def test_rope_model_rejects_unknown_forward_kwargs():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel

    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        norm_arch="post",
    )
    model = DebertaRoPEModel(cfg).eval()
    input_ids = torch.randint(low=0, high=cfg.vocab_size, size=(2, 6), dtype=torch.long)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _ = model(input_ids=input_ids, output_hidden_states=True)


def test_pretrainer_ignores_pad_for_disc_loss_when_attention_mask_missing():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    disc_cfg = DebertaRoPEConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=0,
        pad_token_id=0,
        ffn_type="swiglu",
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
        pad_token_id=0,
        ffn_type="mlp",
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

    input_ids = torch.tensor(
        [
            [1, 7, 8, 2, 0, 0],
            [1, 9, 10, 11, 2, 0],
        ],
        dtype=torch.long,
    )
    labels = torch.full_like(input_ids, -100)
    labels[:, 1] = input_ids[:, 1]
    labels[:, 2] = input_ids[:, 2]
    explicit_mask = input_ids.ne(0).long()

    torch.manual_seed(0)
    out_missing = model(
        input_ids=input_ids,
        attention_mask=None,
        labels=labels,
        sampling_temperature=1.0,
        gen_loss_weight=1.0,
        disc_loss_weight=50.0,
        decoupled_loss_scaling=False,
    )

    torch.manual_seed(0)
    out_explicit = model(
        input_ids=input_ids,
        attention_mask=explicit_mask,
        labels=labels,
        sampling_temperature=1.0,
        gen_loss_weight=1.0,
        disc_loss_weight=50.0,
        decoupled_loss_scaling=False,
    )

    torch.testing.assert_close(
        out_missing.disc_token_count, out_explicit.disc_token_count, rtol=0.0, atol=0.0
    )


def test_pretrainer_disc_loss_excludes_special_tokens_from_active_count():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    cfg = DebertaRoPEConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        mask_token_id=3,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        norm_arch="post",
    )
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=DebertaRoPEModel(cfg),
        generator_backbone=DebertaRoPEModel(cfg),
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="none",
    )

    input_ids = torch.tensor(
        [
            [1, 7, 8, 2, 0, 0],
            [1, 9, 10, 11, 2, 0],
        ],
        dtype=torch.long,
    )
    labels = torch.full_like(input_ids, -100)
    labels[:, 1] = input_ids[:, 1]
    labels[:, 2] = input_ids[:, 2]

    out = model(
        input_ids=input_ids,
        attention_mask=None,
        labels=labels,
        sampling_temperature=1.0,
        gen_loss_weight=1.0,
        disc_loss_weight=50.0,
        decoupled_loss_scaling=False,
    )
    expected_active = ((input_ids != 0) & (input_ids != 1) & (input_ids != 2) & (input_ids != 3)).sum()
    assert int(out.disc_token_count.item()) == int(expected_active.item())


def test_pretrainer_disc_active_keeps_masked_positions_even_if_sampled_special(monkeypatch):
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        mask_token_id=3,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        norm_arch="post",
    )
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=DebertaRoPEModel(cfg),
        generator_backbone=DebertaRoPEModel(cfg),
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="none",
    ).eval()

    input_ids = torch.tensor([[cfg.cls_token_id, 7, 8, cfg.sep_token_id]], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[0, 1] = input_ids[0, 1]

    def _force_special_sample(logits: torch.Tensor, sampling_temperature: float) -> torch.Tensor:
        del logits, sampling_temperature
        return torch.tensor([int(cfg.cls_token_id)], dtype=torch.long)

    monkeypatch.setattr(model, "_sample_generator_tokens", _force_special_sample)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
            sampling_temperature=1.0,
            gen_loss_weight=1.0,
            disc_loss_weight=50.0,
            decoupled_loss_scaling=False,
        )

    torch.testing.assert_close(out.disc_token_count, torch.tensor(2.0))


def test_rope_config_rejects_unknown_ffn_type():
    import pytest

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig

    with pytest.raises(ValueError, match="ffn_type must be one of"):
        _ = DebertaRoPEConfig(ffn_type="glu")


def test_rotary_compile_mode_bypasses_stateful_cache(monkeypatch):
    from deberta.modeling import rope as rope_mod

    rope = rope_mod.RotaryEmbedding(dim=8, base=10_000.0)
    device = torch.device("cpu")

    monkeypatch.setattr(rope_mod, "_is_torch_compiling", lambda: True)
    cos1, sin1 = rope.get_cos_sin(8, device=device, dtype=torch.float32)
    cos2, sin2 = rope.get_cos_sin(8, device=device, dtype=torch.float32)

    assert rope._cache is None
    assert cos1.data_ptr() != cos2.data_ptr()
    assert sin1.data_ptr() != sin2.data_ptr()

    monkeypatch.setattr(rope_mod, "_is_torch_compiling", lambda: False)
    _ = rope.get_cos_sin(8, device=device, dtype=torch.float32)
    assert rope._cache is not None


def test_rotary_apply_full_dim_matches_reference():
    from deberta.modeling.rope import RotaryEmbedding, _rotate_half

    rope = RotaryEmbedding(dim=8, base=10_000.0)
    q = torch.randn((2, 3, 5, 8), dtype=torch.float32)
    k = torch.randn((2, 3, 5, 8), dtype=torch.float32)

    q_out, k_out = rope.apply(q, k)

    cos, sin = rope.get_cos_sin(q.shape[-2], device=q.device, dtype=q.dtype)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_ref = (q * cos) + (_rotate_half(q) * sin)
    k_ref = (k * cos) + (_rotate_half(k) * sin)

    torch.testing.assert_close(q_out, q_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(k_out, k_ref, rtol=0.0, atol=0.0)


def test_rotary_embedding_uses_full_head_dim_for_partial_rope_pct():
    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPESelfAttention

    cfg = DebertaRoPEConfig(
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_act="gelu",
        rotary_pct=0.5,
    )
    attn = DebertaRoPESelfAttention(cfg)
    rope = attn.rope
    assert rope is not None

    # expected frequencies for the 32-wide head (not 16-wide rotary subspace)
    dim = torch.arange(0, 16, 2).float()
    expected = 1.0 / (10000.0 ** (dim / 32))
    torch.testing.assert_close(rope.inv_freq, expected)


def test_rmsnorm_matches_reference_division_form():
    from deberta.modeling.norm import RMSNorm

    torch.manual_seed(0)
    x = torch.randn((3, 5, 16), dtype=torch.float32)
    layer = RMSNorm(hidden_size=16, eps=1e-6, elementwise_affine=True).eval()

    with torch.no_grad():
        out = layer(x)
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(layer.eps).sqrt()
        ref = (x_float / rms).to(dtype=x.dtype) * layer.weight.to(dtype=x.dtype)

    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-7)
