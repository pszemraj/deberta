from __future__ import annotations

import logging
import math

import pytest
import torch
from _fakes import DummyTokenizer

from deberta.data.collator import DebertaV3ElectraCollator, MLMConfig
from deberta.data.streaming import PackedStreamingConfig, PackedStreamingDataset, SequentialStreamingDataset
from deberta.modeling.mask_utils import build_doc_block_mask


@pytest.fixture
def tiny_rope_config_factory():
    """Return a helper that builds tiny DebertaRoPEConfig instances for attention tests."""
    pytest.importorskip("transformers")
    from deberta.modeling.rope_encoder import DebertaRoPEConfig

    base = dict(
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

    def _build(**overrides):
        return DebertaRoPEConfig(**(base | dict(overrides)))

    return _build


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


def test_docblock_streaming_gives_every_segment_its_own_cls() -> None:
    tok = DummyTokenizer(vocab_size=64)
    ds = PackedStreamingDataset(
        hf_dataset=[{"text": "a b"}, {"text": "c d"}],
        tokenizer=tok,
        cfg=PackedStreamingConfig(
            text_column_name="text",
            max_seq_length=8,
            seed=0,
            shuffle_buffer_size=0,
            block_cross_document_attention=True,
        ),
        process_index=0,
        num_processes=1,
    )

    rows = list(ds)
    assert len(rows) == 1
    assert rows[0]["input_ids"][0] == tok.cls_token_id
    assert rows[0]["input_ids"][3] == tok.sep_token_id
    assert rows[0]["input_ids"][4] == tok.cls_token_id
    assert rows[0]["input_ids"][7] == tok.sep_token_id
    assert rows[0]["doc_ids"] == [1, 1, 1, 1, 2, 2, 2, 2]


def test_docblock_streaming_rows_collate_with_structural_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the normal packer-to-collator structural ``doc_ids`` path."""

    tok = DummyTokenizer(vocab_size=64)
    dataset = PackedStreamingDataset(
        hf_dataset=[{"text": "a b"}, {"text": "c d"}, {"text": "e"}],
        tokenizer=tok,
        cfg=PackedStreamingConfig(
            text_column_name="text",
            max_seq_length=8,
            seed=0,
            shuffle_buffer_size=0,
            block_cross_document_attention=True,
        ),
    )
    rows = list(dataset)
    assert len(rows) == 2
    assert "attention_mask" not in rows[0]
    assert "attention_mask" in rows[1]

    collator = DebertaV3ElectraCollator(
        tokenizer=tok,
        cfg=MLMConfig(mlm_probability=0.2),
        packed_sequences=True,
        block_cross_document_attention=True,
    )
    monkeypatch.setattr(
        collator,
        "_compute_document_ids",
        lambda **_kwargs: pytest.fail("packer-supplied doc_ids must remain authoritative"),
    )
    batch = collator(rows)

    assert torch.equal(batch["doc_ids"][0], torch.tensor([1, 1, 1, 1, 2, 2, 2, 2]))
    assert torch.equal(batch["doc_ids"][1], torch.tensor([1, 1, 1, 0, 0, 0, 0, 0]))
    assert torch.equal(batch["position_ids"][0], torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
    assert torch.equal(batch["position_ids"][1], torch.tensor([0, 1, 2, 0, 0, 0, 0, 0]))
    assert torch.equal(batch["doc_context_index"][0], torch.tensor([0, 0, 0, 0, 4, 4, 4, 4]))


def test_docblock_streaming_long_document_chunks_keep_standalone_boundaries() -> None:
    tok = DummyTokenizer(vocab_size=64)
    dataset = PackedStreamingDataset(
        hf_dataset=[{"text": "a b c d e f g h i"}, {"text": "j"}],
        tokenizer=tok,
        cfg=PackedStreamingConfig(
            text_column_name="text",
            max_seq_length=8,
            seed=0,
            shuffle_buffer_size=0,
            block_cross_document_attention=True,
        ),
    )

    for row in dataset:
        doc_ids = row["doc_ids"]
        for document_id in sorted(set(doc_ids) - {0}):
            positions = [idx for idx, value in enumerate(doc_ids) if value == document_id]
            assert row["input_ids"][positions[0]] == tok.cls_token_id
            assert row["input_ids"][positions[-1]] == tok.sep_token_id


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


def test_packed_streaming_does_not_emit_separator_tail_chunk():
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


def test_packed_streaming_flush_strips_all_trailing_separators(monkeypatch: pytest.MonkeyPatch):
    import deberta.data.streaming as streaming_mod

    class _NoopLock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    class _DummySharedInt:
        def __init__(self, _typecode: str, value: int) -> None:
            self.value = int(value)
            self._lock = _NoopLock()

        def get_lock(self) -> _NoopLock:
            return self._lock

    monkeypatch.setattr(streaming_mod.mp, "Value", _DummySharedInt)

    tok = DummyTokenizer(vocab_size=64)
    hf_dataset = [{"text": "a"}, {"text": "b"}]

    ds = PackedStreamingDataset(
        hf_dataset=hf_dataset,
        tokenizer=tok,
        cfg=PackedStreamingConfig(text_column_name="text", max_seq_length=8, seed=0, shuffle_buffer_size=0),
        process_index=0,
        num_processes=1,
    )

    # Pathological tokenizer output: lexical stream itself is separator-only.
    # Final flush should drop the entire trailing separator run.
    monkeypatch.setattr(ds, "_tokenize_text", lambda _raw: [tok.sep_token_id])
    rows = list(ds)
    assert rows == []


def test_packed_streaming_strips_leading_sep_from_chunk():
    tok = DummyTokenizer(vocab_size=64)
    # max_seq=8 => block_len=6.
    # Doc1 has 5 tokens, so buffer after doc1 = [t1..t5, SEP] (len=6).
    # First chunk = [t1..t5, SEP] — consumed exactly.
    # Doc2 has 7 tokens, buffer = [t6..t12, SEP] (len=8).
    # Second chunk = [t6..t11] (len=6), buffer remainder = [t12, SEP].
    # But we want to test the case where SEP lands at the start of a chunk.
    # Doc1 with exactly block_len-1=5 tokens: buffer = [t1..t5, SEP] (len=6).
    # That's one chunk: [t1..t5, SEP]. After _build_example_from_chunk, SEP is
    # internal and valid.
    # Better scenario: doc1 has block_len tokens (6). Buffer = [t1..t6, SEP] (7).
    # chunk1 = buffer[:6] = [t1..t6], buffer = [SEP].
    # Doc2 has block_len tokens (6). Buffer = [SEP, t7..t12, SEP] (8).
    # chunk2 = buffer[:6] = [SEP, t7..t11], which starts with SEP — exactly the bug.
    hf_dataset = [{"text": "a b c d e f"}, {"text": "g h i j k l"}]

    ds = PackedStreamingDataset(
        hf_dataset=hf_dataset,
        tokenizer=tok,
        cfg=PackedStreamingConfig(text_column_name="text", max_seq_length=8, seed=0, shuffle_buffer_size=0),
        process_index=0,
        num_processes=1,
    )
    rows = list(ds)

    for row in rows:
        content = row["input_ids"]
        # After CLS (pos 0), the first content token must not be SEP.
        assert content[0] == tok.cls_token_id
        if len(content) > 2:
            assert content[1] != tok.sep_token_id, (
                f"Chunk starts with [CLS, SEP, ...]: degenerate empty doc start. ids={content}"
            )


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


def test_collator_emits_document_ids_when_packed():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(
        tokenizer=tok,
        cfg=MLMConfig(mlm_probability=0.2, max_ngram=1),
        packed_sequences=True,
        block_cross_document_attention=True,
    )

    features = [
        {
            "input_ids": [
                tok.cls_token_id,
                11,
                tok.sep_token_id,
                tok.cls_token_id,
                12,
                tok.sep_token_id,
            ],
            "special_tokens_mask": [1, 0, 1, 1, 0, 1],
        }
    ]
    batch = coll(features)

    # Collator now emits compact doc_ids instead of dense (B,S,S) attention_mask.
    assert "attention_mask" not in batch
    assert "doc_ids" in batch
    doc_ids = batch["doc_ids"]
    assert doc_ids.ndim == 2
    assert doc_ids.dtype == torch.long

    # CLS (pos 0) is in doc 1.
    assert doc_ids[0, 0].item() == 1
    # Positions in first doc share the same id.
    assert doc_ids[0, 1].item() == doc_ids[0, 2].item()
    # Positions in second doc share a different id.
    assert doc_ids[0, 3].item() == doc_ids[0, 4].item()
    assert doc_ids[0, 3].item() == doc_ids[0, 5].item()
    # Cross-document ids differ.
    assert doc_ids[0, 1].item() != doc_ids[0, 3].item()
    assert batch["flash_active_tokens"] == 6
    assert torch.equal(batch["flash_doc_segment_offsets"][:2], torch.tensor([0, 3], dtype=torch.int32))
    assert torch.equal(batch["flash_doc_segment_lengths"][:2], torch.tensor([3, 3], dtype=torch.int32))
    assert torch.equal(batch["flash_doc_cu_seqlens"][:3], torch.tensor([0, 3, 6], dtype=torch.int32))
    assert torch.equal(batch["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 2]]))
    assert torch.equal(batch["doc_context_index"], torch.tensor([[0, 0, 0, 3, 3, 3]]))


def test_collator_rejects_legacy_packing_without_per_document_cls():
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
    with pytest.raises(ValueError, match="every document segment to begin with CLS"):
        coll(features)


@pytest.mark.parametrize(
    ("doc_ids", "expected_error"),
    [
        ([1.0, 1.0, 1.0], "integer dtype"),
        ([1, 1, 1], "end with its own SEP"),
    ],
)
def test_collator_rejects_invalid_structural_document_metadata(
    doc_ids: list[int] | list[float],
    expected_error: str,
) -> None:
    tok = DummyTokenizer(vocab_size=128)
    collator = DebertaV3ElectraCollator(
        tokenizer=tok,
        cfg=MLMConfig(mlm_probability=0.2),
        packed_sequences=True,
        block_cross_document_attention=True,
    )
    feature = {
        "input_ids": [tok.cls_token_id, 11, 12],
        "special_tokens_mask": [1, 0, 0],
        "doc_ids": doc_ids,
    }

    with pytest.raises((TypeError, ValueError), match=expected_error):
        collator([feature])


def test_collator_skips_document_ids_for_single_doc_packed_chunk():
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
    assert "doc_ids" not in batch
    assert "attention_mask" not in batch


def _assert_active_token_definitions_agree(batch, *, expected_active: int) -> None:
    """Pin every production active-token definition for one collator batch."""

    from deberta.modeling.rtd import attention_mask_to_active_tokens
    from deberta.training.compile import prepare_flash_attention_batch_metadata
    from deberta.training.loop_utils import _count_input_tokens_for_batch, _count_rtd_tokens_for_batch

    assert int(batch["doc_ids"].ne(0).sum().item()) == expected_active

    tokens_per_sec_count = _count_input_tokens_for_batch(batch)
    assert int(tokens_per_sec_count) == expected_active

    _, disc_ga_count = _count_rtd_tokens_for_batch(batch)
    assert int(disc_ga_count) == expected_active

    rtd_loss_active = attention_mask_to_active_tokens(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
    )
    assert int(rtd_loss_active.sum().item()) == expected_active
    assert torch.equal(rtd_loss_active, batch["doc_ids"].ne(0))

    # Flash prep derives its count through segment metadata (a structurally
    # different algorithm from .ne(0).sum()), so pin it explicitly too.
    prepared, meta = prepare_flash_attention_batch_metadata(
        batch=dict(batch),
        backbone_type="hf_deberta_v2",
        flash_enabled=True,
    )
    assert meta is not None
    assert not any(key.startswith("flash_") for key in prepared)
    assert int(meta.active_tokens_host) == expected_active
    assert torch.equal(prepared["position_ids"], batch["position_ids"])
    assert torch.equal(prepared["doc_context_index"], batch["doc_context_index"])


def test_active_token_definitions_agree_for_packed_docblock_batches():
    """GA weighting, tokens/sec logging, RTD loss, and flash prep must count the same tokens.

    Four call sites independently derive "active tokens" from a raw collator
    batch; a drift between any two silently skews per-token effective LR or
    logged throughput, so this pins them to each other and to doc_ids.ne(0).
    """

    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(
        tokenizer=tok,
        cfg=MLMConfig(mlm_probability=0.2, max_ngram=1),
        packed_sequences=True,
        block_cross_document_attention=True,
    )

    # Rows of different lengths force a padded tail on the shorter row.
    features = [
        {
            "input_ids": [
                tok.cls_token_id,
                11,
                tok.sep_token_id,
                tok.cls_token_id,
                13,
                tok.sep_token_id,
            ],
            "special_tokens_mask": [1, 0, 1, 1, 0, 1],
        },
        {
            "input_ids": [tok.cls_token_id, 21, 22, tok.sep_token_id],
            "special_tokens_mask": [1, 0, 0, 1],
        },
    ]
    batch = coll(features)
    assert "doc_ids" in batch
    assert batch.get("attention_mask") is not None
    _assert_active_token_definitions_agree(batch, expected_active=10)


def test_active_token_definitions_agree_for_unpadded_intra_row_packing():
    """A genuinely packed row (multiple docs, no padding) has no attention_mask at all."""

    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(
        tokenizer=tok,
        cfg=MLMConfig(mlm_probability=0.2, max_ngram=1),
        packed_sequences=True,
        block_cross_document_attention=True,
    )

    features = [
        {
            "input_ids": [
                tok.cls_token_id,
                11,
                tok.sep_token_id,
                tok.cls_token_id,
                13,
                tok.sep_token_id,
            ],
            "special_tokens_mask": [1, 0, 1, 1, 0, 1],
        }
    ]
    batch = coll(features)
    assert "doc_ids" in batch
    assert batch.get("attention_mask") is None
    _assert_active_token_definitions_agree(batch, expected_active=6)


def test_packed_document_ids_follow_attention_mask_not_token_values():
    """Active pad-valued tokens stay live while masked non-pad fillers stay dead."""

    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(
        tokenizer=tok,
        cfg=MLMConfig(mlm_probability=0.2, max_ngram=1),
        packed_sequences=True,
        block_cross_document_attention=True,
    )
    batch = coll(
        [
            {
                "input_ids": [
                    tok.cls_token_id,
                    11,
                    tok.pad_token_id,
                    tok.sep_token_id,
                    tok.cls_token_id,
                    12,
                    tok.sep_token_id,
                    99,
                ],
                "attention_mask": [1, 1, 1, 1, 1, 1, 1, 0],
                "special_tokens_mask": [1, 0, 1, 1, 1, 0, 1, 0],
            }
        ]
    )

    assert batch["doc_ids"][0, 2].item() != 0
    assert batch["doc_ids"][0, 7].item() == 0
    assert batch["flash_active_tokens"] == 7


def test_collator_replaces_dataset_flash_metadata_with_its_own_attestation():
    """Dataset columns cannot attest a mask contract on the collator's behalf."""

    coll = DebertaV3ElectraCollator(
        tokenizer=DummyTokenizer(vocab_size=128),
        cfg=MLMConfig(mlm_probability=0.2, max_ngram=1),
    )
    batch = coll(
        [
            {
                "input_ids": [1, 11, 12, 2],
                "attention_mask": [1, 0, 1, 0],
                "special_tokens_mask": [1, 0, 0, 1],
                "flash_seq_lengths": [3, 0, 0, 0],
                "flash_mask_contract_validated": [1, 1, 1, 1],
            }
        ]
    )

    assert not any(key.startswith("flash_") for key in batch)


def test_build_doc_block_mask_matches_expected_structure():
    # doc_ids: doc1=[1,1,1], doc2=[2,2], pad=[0]
    doc_ids = torch.tensor([[1, 1, 1, 2, 2, 0]], dtype=torch.long)
    mask = build_doc_block_mask(doc_ids)
    assert mask.shape == (1, 6, 6)
    assert mask.dtype == torch.bool

    # Same-doc tokens attend each other.
    assert mask[0, 0, 1].item() is True
    assert mask[0, 0, 2].item() is True
    assert mask[0, 3, 4].item() is True

    # Cross-doc tokens do NOT attend each other.
    assert mask[0, 0, 3].item() is False
    assert mask[0, 3, 0].item() is False

    # Active tokens do not attend pad keys.
    assert mask[0, 0, 5].item() is False

    # Active tokens self-attend; pad diagonal is inactive.
    for i in range(5):
        assert mask[0, i, i].item() is True
    assert mask[0, 5, 5].item() is False

    # SDPA safety: inactive pad query has a single keep-edge to CLS key.
    assert mask[0, 5, 0].item() is True
    assert int(mask[0, 5].sum().item()) == 1


def test_mask_utils_reduce_and_expand_helpers_cover_all_ranks():
    from deberta.modeling.mask_utils import expand_keep_mask_to_4d, reduce_keep_mask_to_2d

    mask_2d = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
    expected = mask_2d.bool()

    # 2D passthrough plus optional key-axis slicing.
    assert torch.equal(reduce_keep_mask_to_2d(mask_2d), expected)
    assert torch.equal(reduce_keep_mask_to_2d(mask_2d, seq_len=3), expected[:, :3])

    # Pairwise (B,S,S): diagonal encodes per-query activity.
    pairwise = expected[:, :, None] & expected[:, None, :]
    assert torch.equal(reduce_keep_mask_to_2d(pairwise), expected)

    # 4D broadcast (B,1,1,S) and head-specific (B,H,S,S) layouts.
    assert torch.equal(reduce_keep_mask_to_2d(expected[:, None, None, :]), expected)
    assert torch.equal(reduce_keep_mask_to_2d(pairwise[:, None].expand(-1, 3, -1, -1)), expected)

    # Rank-3 broadcast (B,1,S) keeps the sequence axis instead of truncating
    # to a bogus (B,1) diagonal -- this was the latent drift between the old
    # per-module copies of this reduction.
    assert torch.equal(reduce_keep_mask_to_2d(expected[:, None, :]), expected)

    # Expansion: broadcast by default, EMD outer product on request.
    assert torch.equal(expand_keep_mask_to_4d(mask_2d), expected[:, None, None, :])
    assert torch.equal(expand_keep_mask_to_4d(mask_2d, pairwise_2d=True), pairwise[:, None])
    assert torch.equal(expand_keep_mask_to_4d(pairwise), pairwise[:, None])


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


def test_ngram_masking_windowed_selection_matches_deberta_policy(monkeypatch: pytest.MonkeyPatch):
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

    # Windowed DeBERTa selection can cover each local context once under deterministic
    # sampling, yielding four masked lexical tokens in this toy sequence.
    assert int(labels.ne(-100).sum().item()) == 4


def test_ngram_masking_does_not_split_selected_word_groups():
    tok = DummyTokenizer(
        vocab_size=128,
        token_map={10: "hello", 11: "##world"},
        tokenize_output=["hello", "##world"],
        default_token_prefix="tok",
    )
    coll = DebertaV3ElectraCollator(
        tokenizer=tok,
        # S=4 and p=0.2 -> num_to_predict=1 token budget. Whole-word masking must
        # still mask the full two-piece word group (allowing bounded overshoot).
        cfg=MLMConfig(mlm_probability=0.2, mask_token_prob=1.0, random_token_prob=0.0, max_ngram=3),
    )
    input_ids = torch.tensor([[tok.cls_token_id, 10, 11, tok.sep_token_id]], dtype=torch.long)
    special = torch.tensor([[1, 0, 0, 1]], dtype=torch.bool)

    masked, labels = coll._mask_tokens_ngram(input_ids, special_tokens_mask=special, max_ngram=3)
    assert int(labels.ne(-100).sum().item()) == 2
    assert masked[0, 1].item() == tok.mask_token_id
    assert masked[0, 2].item() == tok.mask_token_id


def test_ngram_masking_samples_random_replacement_per_subtoken(monkeypatch: pytest.MonkeyPatch):
    tok = DummyTokenizer(
        vocab_size=256,
        token_map={10: "hello", 11: "##world"},
        tokenize_output=["hello", "##world"],
        default_token_prefix="tok",
    )
    coll = DebertaV3ElectraCollator(
        tokenizer=tok,
        cfg=MLMConfig(mlm_probability=0.9, mask_token_prob=0.0, random_token_prob=1.0, max_ngram=3),
    )
    input_ids = torch.tensor([[tok.cls_token_id, 10, 11, tok.sep_token_id]], dtype=torch.long)
    special = torch.tensor([[1, 0, 0, 1]], dtype=torch.bool)

    calls = {"n": 0}

    def _sample_random(shape: torch.Size | tuple[int, ...], device: torch.device) -> torch.Tensor:
        del device
        calls["n"] += 1
        n = int(shape[0]) if isinstance(shape, tuple) else int(shape[0])
        return torch.tensor([50, 51], dtype=torch.long)[:n]

    monkeypatch.setattr(coll, "_sample_random_words", _sample_random)
    masked, labels = coll._mask_tokens_ngram(input_ids, special_tokens_mask=special, max_ngram=3)

    assert calls["n"] == 1
    assert int(labels.ne(-100).sum().item()) == 2
    assert masked[0, 1].item() == 50
    assert masked[0, 2].item() == 51


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
        _, labels = coll._mask_tokens_unigram_windowed(input_ids, special_tokens_mask=special)
        counts.append(int(labels.ne(-100).sum().item()))

    assert set(counts) == {1}


def test_mask_tokens_dispatch_uses_windowed_unigram_not_ngram(monkeypatch: pytest.MonkeyPatch):
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.2, max_ngram=1))

    input_ids = torch.tensor([[tok.cls_token_id, 11, 12, tok.sep_token_id]], dtype=torch.long)
    special = torch.tensor([[1, 0, 0, 1]], dtype=torch.bool)

    calls = {"windowed": 0}

    def _windowed(_input_ids: torch.Tensor, *, special_tokens_mask: torch.Tensor):
        del special_tokens_mask
        calls["windowed"] += 1
        labels = torch.full_like(_input_ids, -100)
        return _input_ids.clone(), labels

    def _ngram(*args, **kwargs):
        del args, kwargs
        raise AssertionError("_mask_tokens_ngram should not be called from _mask_tokens dispatch")

    monkeypatch.setattr(coll, "_mask_tokens_unigram_windowed", _windowed)
    monkeypatch.setattr(coll, "_mask_tokens_ngram", _ngram)

    _ = coll._mask_tokens(input_ids, special_tokens_mask=special)
    assert int(calls["windowed"]) == 1


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
    # DeBERTa budget uses full sequence length (including specials/pad) before applying
    # eligibility filtering. In this padded batch that yields [2, 1].
    assert masked_counts == [2, 1]


def test_ngram_wordpiece_like_tokens_do_not_overmerge_groups():
    tok = DummyTokenizer(
        vocab_size=128, token_map={10: "the", 11: "cat", 12: "sat"}, default_token_prefix="tok"
    )
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.5, max_ngram=3))

    ids = [tok.cls_token_id, 10, 11, 12, tok.sep_token_id]
    spec = [1, 0, 0, 0, 1]
    groups = coll._build_word_groups(ids, spec)
    assert groups == [[1], [2], [3]]


def test_word_boundary_scheme_detected_once_from_tokenizer_probe():
    tok = DummyTokenizer(
        vocab_size=128,
        token_map={10: "Ġalpha", 11: "beta", 12: "Ġgamma"},
        tokenize_output=["hello", "##world"],
        default_token_prefix="tok",
    )
    coll = DebertaV3ElectraCollator(tokenizer=tok, cfg=MLMConfig(mlm_probability=0.5, max_ngram=3))

    ids = [tok.cls_token_id, 10, 11, 12, tok.sep_token_id]
    spec = [1, 0, 0, 0, 1]
    groups = coll._build_word_groups(ids, spec)
    assert groups == [[1], [2], [3]]


def test_collator_warns_when_word_boundary_scheme_is_none_for_ngram(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        _ = DebertaV3ElectraCollator(
            tokenizer=DummyTokenizer(
                vocab_size=128,
                token_map={10: "hello", 11: "world"},
                tokenize_output=["hello", "world"],
                default_token_prefix="tok",
            ),
            cfg=MLMConfig(mlm_probability=0.2, max_ngram=3),
        )

    assert "scheme='none'" in caplog.text


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
        assert torch.equal(batch["flash_seq_lengths"].sort().values, torch.tensor([3, 4], dtype=torch.int32))
        assert batch["flash_active_tokens"] == 7


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


def test_collator_merges_partial_special_tokens_mask_with_tokenizer_special_ids():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(
        tokenizer=tok,
        cfg=MLMConfig(mlm_probability=0.999, mask_token_prob=1.0, random_token_prob=0.0, max_ngram=1),
    )

    # Upstream packed datasets may emit a partial special_tokens_mask that only marks
    # structural specials. The collator should still protect tokenizer specials such as
    # mask_token_id even when the provided mask marks them as non-special.
    features = [
        {
            "input_ids": [tok.cls_token_id, tok.mask_token_id, tok.sep_token_id],
            "special_tokens_mask": [1, 0, 1],
        }
    ]
    batch = coll(features)
    assert int(batch["labels"][0, 1].item()) == -100


def test_collator_random_replacement_avoids_special_ids():
    tok = DummyTokenizer(vocab_size=128)
    coll = DebertaV3ElectraCollator(
        tokenizer=tok,
        cfg=MLMConfig(mlm_probability=0.999, mask_token_prob=0.0, random_token_prob=1.0, max_ngram=1),
    )

    torch.manual_seed(0)
    input_ids = torch.arange(10, 266, dtype=torch.long).view(1, -1) % tok.vocab_size
    special = torch.zeros_like(input_ids, dtype=torch.bool)

    masked, labels = coll._mask_tokens_unigram_windowed(input_ids, special_tokens_mask=special)
    changed = labels.ne(-100)
    assert bool(changed.any().item())

    replaced = masked[changed]
    for sid in tok.all_special_ids:
        assert not bool((replaced == sid).any().item())


def test_collator_random_replacement_samples_full_vocab_including_added_tokens():
    """When len(tokenizer) > tokenizer.vocab_size, random replacement must cover the full range."""
    tok = DummyTokenizer(vocab_size=100, extra_length=10)
    assert len(tok) == 110
    assert tok.vocab_size == 100

    coll = DebertaV3ElectraCollator(
        tokenizer=tok,
        cfg=MLMConfig(mlm_probability=0.999, mask_token_prob=0.0, random_token_prob=1.0, max_ngram=1),
    )

    # Non-special token ids should cover [0, 110) minus specials.
    assert coll._non_special_token_ids_cpu is not None
    non_special = coll._non_special_token_ids_cpu
    assert int(non_special.max().item()) >= 100, "Random replacement should be able to sample added-token IDs"

    # Run actual masking and verify some replacements land in the extended range.
    torch.manual_seed(42)
    input_ids = torch.arange(10, 266, dtype=torch.long).view(1, -1) % tok.vocab_size
    special = torch.zeros_like(input_ids, dtype=torch.bool)
    masked, labels = coll._mask_tokens_unigram_windowed(input_ids, special_tokens_mask=special)
    changed = labels.ne(-100)
    replaced = masked[changed]
    assert bool((replaced >= 100).any().item()), "Expected some replacement tokens from the added-token range"


def test_self_attention_has_no_internal_residual_dropout(tiny_rope_config_factory):
    from deberta.modeling.rope_encoder import DebertaRoPESelfAttention

    torch.manual_seed(0)
    cfg = tiny_rope_config_factory(
        attention_implementation="eager",
        hidden_dropout_prob=0.9,
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


@pytest.mark.parametrize(
    ("attention_implementation", "mask_case"),
    [
        ("eager", "padded_queries_2d"),
        ("eager", "all_masked_2d"),
        ("eager", "pairwise_row_without_keys"),
        ("sdpa", "pairwise_row_without_keys"),
        ("eager", "pairwise_dead_last_row"),
        ("sdpa", "pairwise_dead_last_row"),
    ],
)
def test_self_attention_mask_edge_cases(
    tiny_rope_config_factory,
    attention_implementation: str,
    mask_case: str,
):
    from deberta.modeling.rope_encoder import DebertaRoPESelfAttention

    torch.manual_seed(0)
    cfg = tiny_rope_config_factory(attention_implementation=attention_implementation)
    attn = DebertaRoPESelfAttention(cfg).eval()
    if mask_case == "padded_queries_2d":
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
        assert torch.isfinite(out).all()
        assert torch.allclose(out[0, 4:, :], torch.zeros_like(out[0, 4:, :]), atol=1e-6)
        assert torch.allclose(out[1, 5:, :], torch.zeros_like(out[1, 5:, :]), atol=1e-6)
    elif mask_case == "all_masked_2d":
        x = torch.randn((2, 4, cfg.hidden_size), dtype=torch.float32)
        attention_mask = torch.zeros((2, 4), dtype=torch.long)
        with torch.no_grad():
            out = attn(x, attention_mask)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
    elif mask_case == "pairwise_row_without_keys":
        x = torch.randn((1, 4, cfg.hidden_size), dtype=torch.float32)
        pair_keep = torch.tensor(
            [
                [
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                ]
            ],
            dtype=torch.long,
        )
        with torch.no_grad():
            out = attn(x, pair_keep)
        assert torch.isfinite(out).all(), f"{attention_implementation} produced NaN for dead query row"
        assert torch.allclose(out[0, 1], torch.zeros_like(out[0, 1]), atol=1e-6)
    elif mask_case == "pairwise_dead_last_row":
        x = torch.randn((1, 4, cfg.hidden_size), dtype=torch.float32)
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
    else:  # pragma: no cover
        raise AssertionError(f"Unexpected mask_case: {mask_case}")


def test_self_attention_uses_pairwise_diagonal_for_query_activity(tiny_rope_config_factory):
    from deberta.modeling.rope_encoder import DebertaRoPESelfAttention

    cfg = tiny_rope_config_factory(attention_implementation="eager")
    attn = DebertaRoPESelfAttention(cfg).eval()
    x = torch.randn((1, 4, cfg.hidden_size), dtype=torch.float32)

    # Row 3 has an accidental off-diagonal keep bit but diagonal=False.
    # In packed-mask semantics, query activity is encoded on the diagonal, so
    # this row must remain inactive and be zeroed after projection.
    pair_keep = torch.tensor(
        [
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
            ]
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        out = attn(x, pair_keep)

    assert torch.isfinite(out).all()
    assert torch.allclose(out[0, 3], torch.zeros_like(out[0, 3]), atol=1e-6)


def test_mlp_has_no_internal_residual_dropout():

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
    )

    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.disc_accuracy.ndim == 0


def test_rope_pretrainer_ignores_flash_metadata_boundary():
    """RoPE RTD should run when flash metadata is present at the RTD boundary."""

    pytest.importorskip("transformers")

    from deberta.modeling.mask_utils import FlashBatchMeta
    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=48,
        max_position_embeddings=32,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=DebertaRoPEModel(cfg),
        generator_backbone=DebertaRoPEModel(cfg),
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="gdes",
    )
    input_ids = torch.randint(low=0, high=64, size=(2, 8), dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[:, 2] = input_ids[:, 2]
    flash_meta = FlashBatchMeta(
        seq_lengths=torch.tensor([8, 7], dtype=torch.int32),
        active_tokens_host=15,
        route_hint="fixed",
    )

    out = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones_like(input_ids),
        flash_meta=flash_meta,
    )

    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)


def test_pretrainer_generator_phase_gates_flash_metadata_for_base_signature_backbone():
    """RTD should not pass flash metadata to backbones that do not declare it."""

    from types import SimpleNamespace

    from deberta.modeling.mask_utils import FlashBatchMeta
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    class _Embeddings(torch.nn.Module):
        def __init__(self, vocab_size: int, hidden_size: int) -> None:
            super().__init__()
            self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size)

    class _BaseSignatureBackbone(torch.nn.Module):
        def __init__(self, cfg: SimpleNamespace) -> None:
            super().__init__()
            self.cfg = cfg
            self.embeddings = _Embeddings(cfg.vocab_size, cfg.hidden_size)
            self.proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size)

        def _initialize_weights(self, module: torch.nn.Module) -> None:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            token_type_ids: torch.Tensor | None = None,
            return_dict: bool = True,
            output_hidden_states: bool = False,
        ) -> SimpleNamespace:
            del attention_mask, token_type_ids, return_dict
            hidden = self.proj(self.embeddings.word_embeddings(input_ids))
            hidden_states = (hidden,) if output_hidden_states else None
            return SimpleNamespace(last_hidden_state=hidden, hidden_states=hidden_states)

    cfg = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        embedding_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        use_rmsnorm_heads=False,
        position_biased_input=True,
        pad_token_id=0,
    )
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=_BaseSignatureBackbone(cfg),
        generator_backbone=_BaseSignatureBackbone(cfg),
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="none",
    )
    assert model._generator_accepts_flash_kwargs is False

    input_ids = torch.randint(low=1, high=32, size=(2, 6), dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[:, 2] = input_ids[:, 2]
    out = model.forward_generator_phase(
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones_like(input_ids),
        flash_meta=FlashBatchMeta(
            seq_lengths=torch.tensor([6, 6], dtype=torch.int32),
            active_tokens_host=12,
            route_hint="fixed",
        ),
    )

    assert out.has_masked_targets is True
    assert torch.isfinite(out.gen_loss_raw)


def test_pretrainer_sampler_avoids_configured_special_ids():

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
        sampled = model._gumbel_sample(
            logits,
            temperature=1.0,
            forbidden_vocab_mask=model._forbidden_sample_token_mask,
        )

    for sid in (cfg.pad_token_id, cfg.cls_token_id, cfg.sep_token_id, cfg.mask_token_id):
        assert sid is not None
        assert not bool((sampled == int(sid)).any().item())


def test_pretrainer_rejects_z_steps_as_enhanced_mask_decoder_substitute() -> None:
    pytest.importorskip("transformers")

    from deberta.modeling.deberta_v2_native import DebertaV2Config, DebertaV2Model
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    common_kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        position_biased_input=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        mask_token_id=3,
    )
    disc_cfg = DebertaV2Config(**common_kwargs, z_steps=0)
    gen_cfg = DebertaV2Config(**common_kwargs, z_steps=2)
    with pytest.raises(ValueError, match="not equivalent to backbone z_steps"):
        DebertaV3RTDPretrainer(
            discriminator_backbone=DebertaV2Model(disc_cfg),
            generator_backbone=DebertaV2Model(gen_cfg),
            disc_config=disc_cfg,
            gen_config=gen_cfg,
            embedding_sharing="none",
        )


def _make_emd_harness(
    last_layer: torch.nn.Module,
    *,
    layer_norm: torch.nn.Module | None = None,
    num_last_layer_passes: int = 1,
):
    """Build an EnhancedMaskDecoder harness around one instrumented last layer.

    The three EMD contract tests share this scaffolding and differ only in the
    last layer's recording behavior plus their assertions.
    """
    from deberta.modeling.rtd import EnhancedMaskDecoder

    class _PositionEmbeddings(torch.nn.Module):
        def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
            offsets = torch.tensor([0.25, -0.5, 1.0, -1.5], dtype=torch.float32)
            return position_ids.to(dtype=torch.float32).unsqueeze(-1) + offsets

    class _Embeddings(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.position_embeddings = _PositionEmbeddings()
            self.LayerNorm = layer_norm if layer_norm is not None else torch.nn.Identity()

    class _Encoder(torch.nn.Module):
        def __init__(self, layer: torch.nn.Module) -> None:
            super().__init__()
            self.layer = torch.nn.ModuleList([layer])

    decoder = EnhancedMaskDecoder(num_last_layer_passes=num_last_layer_passes)
    encoder = _Encoder(last_layer)
    embeddings = _Embeddings()
    kv_states = torch.arange(0, 12, dtype=torch.float32).view(1, 3, 4)
    encoder_hidden_states = [kv_states, kv_states + 1.0]
    masked_positions = torch.tensor([[False, True, False]], dtype=torch.bool)
    return decoder, encoder, embeddings, kv_states, encoder_hidden_states, masked_positions


def test_enhanced_mask_decoder_adds_raw_position_states_without_embedding_norm():

    pytest.importorskip("transformers")

    class _ForbiddenNorm(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            del x
            raise AssertionError("EMD must not apply the input embedding LayerNorm to raw positions.")

    class _LastLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_states_seen: list[torch.Tensor] = []
            self.query_states_seen: list[torch.Tensor] = []

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            *,
            output_attentions: bool = False,
            query_states: torch.Tensor | None = None,
            relative_pos: torch.Tensor | None = None,
            rel_embeddings: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, None]:
            del attention_mask, output_attentions, relative_pos, rel_embeddings
            assert query_states is not None
            self.hidden_states_seen.append(hidden_states.detach().clone())
            self.query_states_seen.append(query_states.detach().clone())
            pass_number = float(len(self.query_states_seen))
            return query_states + pass_number, None

    last_layer = _LastLayer()
    decoder, encoder, embeddings, kv_states, encoder_hidden_states, masked_positions = _make_emd_harness(
        last_layer,
        layer_norm=_ForbiddenNorm(),
        num_last_layer_passes=2,
    )
    attention_mask = torch.ones((1, 3), dtype=torch.bool)

    masked = decoder(
        encoder_hidden_states=encoder_hidden_states,
        masked_positions=masked_positions,
        attention_mask=attention_mask,
        embeddings=embeddings,
        encoder=encoder,
    )

    position_ids = torch.arange(kv_states.shape[1]).unsqueeze(0)
    raw_positions = embeddings.position_embeddings(position_ids)
    first_query = kv_states + raw_positions
    first_output = first_query + 1.0
    second_output = first_output + 2.0
    assert len(last_layer.query_states_seen) == 2
    assert len(last_layer.hidden_states_seen) == 2
    torch.testing.assert_close(last_layer.hidden_states_seen[0], kv_states, rtol=0.0, atol=0.0)
    torch.testing.assert_close(last_layer.hidden_states_seen[1], kv_states, rtol=0.0, atol=0.0)
    torch.testing.assert_close(last_layer.query_states_seen[0], first_query, rtol=0.0, atol=0.0)
    torch.testing.assert_close(last_layer.query_states_seen[1], first_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(masked, second_output[:, 1:2, :].reshape(1, 4), rtol=0.0, atol=0.0)


def test_enhanced_mask_decoder_keeps_none_attention_mask_unmaterialized():

    pytest.importorskip("transformers")

    class _LastLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_attention_mask: torch.Tensor | None = torch.ones(1, dtype=torch.bool)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None,
            *,
            output_attentions: bool = False,
            query_states: torch.Tensor | None = None,
            relative_pos: torch.Tensor | None = None,
            rel_embeddings: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, None]:
            del hidden_states, output_attentions, relative_pos, rel_embeddings
            self.seen_attention_mask = attention_mask
            assert query_states is not None
            return query_states, None

    last_layer = _LastLayer()
    decoder, encoder, embeddings, _, encoder_hidden_states, masked_positions = _make_emd_harness(last_layer)

    masked = decoder(
        encoder_hidden_states=encoder_hidden_states,
        masked_positions=masked_positions,
        attention_mask=None,
        embeddings=embeddings,
        encoder=encoder,
    )

    assert last_layer.seen_attention_mask is None
    assert tuple(masked.shape) == (1, 4)


def test_enhanced_mask_decoder_forwards_flash_metadata_to_last_layer():

    pytest.importorskip("transformers")

    from deberta.modeling.mask_utils import FlashBatchMeta

    class _LastLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: dict[str, torch.Tensor | int | str | None] = {}

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None,
            *,
            output_attentions: bool = False,
            query_states: torch.Tensor | None = None,
            relative_pos: torch.Tensor | None = None,
            rel_embeddings: torch.Tensor | None = None,
            flash_meta: FlashBatchMeta | None = None,
        ) -> tuple[torch.Tensor, None]:
            del hidden_states, attention_mask, output_attentions, relative_pos, rel_embeddings
            assert query_states is not None
            self.seen = {"flash_meta": flash_meta}
            return query_states, None

    last_layer = _LastLayer()
    decoder, encoder, embeddings, _, encoder_hidden_states, masked_positions = _make_emd_harness(last_layer)
    attention_mask = torch.tensor(
        [[[True, True, False], [True, True, False], [False, False, True]]],
        dtype=torch.bool,
    )
    flash_seq_lengths = torch.tensor([2], dtype=torch.int32)
    flash_doc_segment_offsets = torch.tensor([0, 2], dtype=torch.int32)
    flash_doc_segment_lengths = torch.tensor([2, 1], dtype=torch.int32)
    flash_doc_cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)
    flash_meta = FlashBatchMeta(
        seq_lengths=flash_seq_lengths,
        doc_segment_offsets=flash_doc_segment_offsets,
        doc_segment_lengths=flash_doc_segment_lengths,
        doc_cu_seqlens=flash_doc_cu_seqlens,
        active_tokens_host=3,
        doc_num_segments_host=2,
        doc_max_segment_length_host=2,
        route_hint="docblock_bias",
    )

    masked = decoder(
        encoder_hidden_states=encoder_hidden_states,
        masked_positions=masked_positions,
        attention_mask=attention_mask,
        embeddings=embeddings,
        encoder=encoder,
        flash_meta=flash_meta,
    )

    assert tuple(masked.shape) == (1, 4)
    seen_meta = last_layer.seen["flash_meta"]
    assert isinstance(seen_meta, FlashBatchMeta)
    assert seen_meta.normalized_route_hint() == "docblock_bias"
    assert torch.equal(seen_meta.seq_lengths, flash_seq_lengths)
    assert torch.equal(seen_meta.doc_segment_offsets, flash_doc_segment_offsets)
    assert torch.equal(seen_meta.doc_segment_lengths, flash_doc_segment_lengths)
    assert torch.equal(seen_meta.doc_cu_seqlens, flash_doc_cu_seqlens)
    assert seen_meta.active_tokens_host == 3
    assert seen_meta.doc_num_segments_host == 2
    assert seen_meta.doc_max_segment_length_host == 2


def test_masked_lm_head_has_only_tied_projection_bias():

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
    head = MaskedLMHead(cfg)
    assert not hasattr(head, "decoder")
    assert head.bias.shape == (cfg.vocab_size,)


@pytest.mark.parametrize("backbone_type", ["native", "rope"])
def test_pretrainer_initializes_only_new_heads_from_backbone_contract(backbone_type: str) -> None:
    pytest.importorskip("transformers")
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    initializer_range = 0.005
    if backbone_type == "native":
        from deberta.modeling.deberta_v2_native import DebertaV2Config, DebertaV2Model

        cfg = DebertaV2Config(
            vocab_size=96,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=16,
            type_vocab_size=0,
            initializer_range=initializer_range,
            position_biased_input=True,
        )
        generator = DebertaV2Model(cfg)
        discriminator = DebertaV2Model(cfg)
    else:
        from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel

        cfg = DebertaRoPEConfig(
            vocab_size=96,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=16,
            type_vocab_size=0,
            initializer_range=initializer_range,
        )
        generator = DebertaRoPEModel(cfg)
        discriminator = DebertaRoPEModel(cfg)

    generator_before = {name: value.detach().clone() for name, value in generator.named_parameters()}
    discriminator_before = {name: value.detach().clone() for name, value in discriminator.named_parameters()}
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=discriminator,
        generator_backbone=generator,
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="none",
    )

    for name, value in generator.named_parameters():
        torch.testing.assert_close(value, generator_before[name], rtol=0.0, atol=0.0)
    for name, value in discriminator.named_parameters():
        torch.testing.assert_close(value, discriminator_before[name], rtol=0.0, atol=0.0)

    linears = (
        model.generator_lm_head.transform.dense,
        model.discriminator_head.dense,
        model.discriminator_head.classifier,
    )
    for linear in linears:
        assert float(linear.weight.detach().std(unbiased=False)) == pytest.approx(
            initializer_range,
            rel=0.35,
        )
        assert linear.bias is not None
        assert torch.count_nonzero(linear.bias).item() == 0
    assert torch.count_nonzero(model.generator_lm_head.bias).item() == 0


def test_mlm_and_rtd_heads_use_layernorm_when_rmsnorm_heads_disabled():

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
    head = MaskedLMHead(cfg)
    head.transform = torch.nn.Identity()

    hidden = torch.randn((3, cfg.hidden_size), dtype=torch.float64)
    word_w = torch.randn((cfg.vocab_size, cfg.hidden_size), dtype=torch.float32)

    with torch.no_grad():
        logits = head(hidden, word_embedding_weight=word_w)

    assert logits.dtype == torch.float32


def test_rtd_head_does_not_apply_dropout_in_parity_path():

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
    assert counting_dropout.calls == 0


def test_rtd_head_applies_cls_conditioning_before_dense_projection():

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig
    from deberta.modeling.rtd import RTDHead

    class _Spy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: torch.Tensor | None = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.seen = x.detach().clone()
            return x

    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=16,
        type_vocab_size=0,
    )
    head = RTDHead(cfg)

    norm_spy = _Spy()
    dense_spy = _Spy()
    head.norm = norm_spy
    head.dense = dense_spy
    head.act = torch.nn.Identity()
    head.classifier = torch.nn.Linear(cfg.hidden_size, 1, bias=False)
    with torch.no_grad():
        head.classifier.weight.fill_(1.0)

    hidden = torch.arange(0, 2 * 3 * cfg.hidden_size, dtype=torch.float32).view(2, 3, cfg.hidden_size)
    logits = head(hidden)

    expected_norm_input = hidden + hidden[:, 0:1, :]
    assert norm_spy.seen is not None
    assert dense_spy.seen is not None
    torch.testing.assert_close(norm_spy.seen, expected_norm_input)
    torch.testing.assert_close(dense_spy.seen, expected_norm_input)
    assert logits.shape == (2, 3)


def test_rtd_head_gathers_per_document_cls_for_pairwise_attention_masks():

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig
    from deberta.modeling.rtd import RTDHead

    class _Spy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: torch.Tensor | None = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.seen = x.detach().clone()
            return x

    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=16,
        type_vocab_size=0,
    )
    head = RTDHead(cfg)

    norm_spy = _Spy()
    dense_spy = _Spy()
    head.norm = norm_spy
    head.dense = dense_spy
    head.act = torch.nn.Identity()
    head.classifier = torch.nn.Linear(cfg.hidden_size, 1, bias=False)
    with torch.no_grad():
        head.classifier.weight.fill_(1.0)

    hidden = torch.arange(0, 2 * 3 * cfg.hidden_size, dtype=torch.float32).view(2, 3, cfg.hidden_size)
    pairwise_mask = torch.tensor(
        [
            [[True, True, False], [True, True, False], [False, False, True]],
            [[True, True, False], [True, True, False], [False, False, True]],
        ]
    )
    with pytest.raises(RuntimeError, match="requires doc_context_index"):
        head(hidden, attention_mask=pairwise_mask)
    doc_context_index = torch.tensor([[0, 0, 2], [0, 0, 2]])
    logits = head(
        hidden,
        attention_mask=pairwise_mask,
        doc_context_index=doc_context_index,
    )

    assert norm_spy.seen is not None
    assert dense_spy.seen is not None
    context = hidden.gather(
        1,
        doc_context_index.unsqueeze(-1).expand_as(hidden),
    )
    torch.testing.assert_close(norm_spy.seen, hidden + context)
    torch.testing.assert_close(dense_spy.seen, hidden + context)
    assert logits.shape == (2, 3)


def test_rtd_head_requires_per_document_cls_for_docblock_flash_meta():
    """Ragged Flash metadata must not silently drop or globalize CLS context.

    The ragged flash `docblock` route ships a compact 2D keep mask plus
    `FlashBatchMeta` segment metadata. Position 0 of a packed row belongs to
    the first document only, so adding it globally would leak document 1 into
    every other document in the row (regression for the P1 review finding).
    """

    pytest.importorskip("transformers")

    from deberta.modeling.mask_utils import FlashBatchMeta
    from deberta.modeling.rope_encoder import DebertaRoPEConfig
    from deberta.modeling.rtd import RTDHead

    class _Spy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: torch.Tensor | None = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.seen = x.detach().clone()
            return x

    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=16,
        type_vocab_size=0,
    )
    head = RTDHead(cfg)

    norm_spy = _Spy()
    head.norm = norm_spy
    head.act = torch.nn.Identity()

    hidden = torch.arange(0, 2 * 4 * cfg.hidden_size, dtype=torch.float32).view(2, 4, cfg.hidden_size)
    keep_mask_2d = torch.ones((2, 4), dtype=torch.bool)
    docblock_meta = FlashBatchMeta(
        doc_segment_offsets=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        doc_segment_lengths=torch.tensor([2, 2, 2, 2], dtype=torch.int32),
        doc_cu_seqlens=torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32),
        active_tokens_host=8,
        doc_num_segments_host=4,
        doc_max_segment_length_host=2,
        route_hint="docblock",
    )

    with pytest.raises(RuntimeError, match="requires doc_context_index"):
        head(hidden, attention_mask=keep_mask_2d, flash_meta=docblock_meta)
    doc_context_index = torch.tensor([[0, 0, 2, 2], [0, 0, 2, 2]])
    _ = head(
        hidden,
        attention_mask=keep_mask_2d,
        doc_context_index=doc_context_index,
        flash_meta=docblock_meta,
    )
    assert norm_spy.seen is not None
    context = hidden.gather(1, doc_context_index.unsqueeze(-1).expand_as(hidden))
    torch.testing.assert_close(norm_spy.seen, hidden + context)

    # The same 2D mask without doc-block metadata keeps the original
    # single-document behavior: global CLS conditioning stays enabled.
    _ = head(hidden, attention_mask=keep_mask_2d)
    torch.testing.assert_close(norm_spy.seen, hidden + hidden[:, 0:1, :])

    assert RTDHead._requires_document_context(keep_mask_2d, flash_meta=docblock_meta) is True
    assert (
        RTDHead._requires_document_context(
            keep_mask_2d,
            flash_meta=FlashBatchMeta(route_hint="docblock_bias"),
        )
        is True
    )
    assert (
        RTDHead._requires_document_context(
            keep_mask_2d,
            flash_meta=FlashBatchMeta(route_hint="varlen"),
        )
        is False
    )


def test_packed_rtd_matches_standalone_documents_with_local_positions_and_cls() -> None:
    """Packing must preserve generator and discriminator objective semantics."""

    pytest.importorskip("transformers")
    from deberta.modeling.deberta_v2_native import DebertaV2Config, DebertaV2Model
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    torch.manual_seed(41)
    cfg = DebertaV2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
        max_relative_positions=16,
        position_buckets=-1,
        relative_attention=True,
        pos_att_type=["c2p", "p2c"],
        position_biased_input=False,
        z_steps=0,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        mask_token_id=3,
    )
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=DebertaV2Model(cfg),
        generator_backbone=DebertaV2Model(cfg),
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="none",
    ).eval()

    standalone_inputs = (
        torch.tensor([[1, 10, 3, 2]], dtype=torch.long),
        torch.tensor([[1, 20, 3, 2]], dtype=torch.long),
    )
    standalone_labels = (
        torch.tensor([[-100, -100, 12, -100]], dtype=torch.long),
        torch.tensor([[-100, -100, 22, -100]], dtype=torch.long),
    )
    standalone_positions = torch.arange(4).unsqueeze(0)
    standalone_mask = torch.ones((1, 4), dtype=torch.bool)
    packed_input = torch.cat(standalone_inputs, dim=1)
    packed_labels = torch.cat(standalone_labels, dim=1)
    packed_positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)
    packed_doc_ids = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=torch.long)
    packed_mask = build_doc_block_mask(packed_doc_ids)
    packed_context = torch.tensor([[0, 0, 0, 0, 4, 4, 4, 4]], dtype=torch.long)

    def _generator_logits(
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        masked_positions = labels.ne(-100)
        output = model.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        masked_hidden = model.enhanced_mask_decoder(
            encoder_hidden_states=output.hidden_states,
            masked_positions=masked_positions,
            attention_mask=attention_mask,
            embeddings=model.generator.embeddings,
            encoder=model.generator.encoder,
            position_ids=position_ids,
        )
        return model.generator_lm_head(
            masked_hidden,
            word_embedding_weight=model._get_generator_word_embedding_weight(),
        )

    def _discriminator_logits(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        doc_context_index: torch.Tensor | None,
    ) -> torch.Tensor:
        output = model.discriminator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        return model.discriminator_head(
            output.last_hidden_state,
            attention_mask=attention_mask,
            doc_context_index=doc_context_index,
        )

    with torch.no_grad():
        packed_gen_logits = _generator_logits(
            packed_input,
            packed_labels,
            packed_mask,
            packed_positions,
        )
        standalone_gen_logits = [
            _generator_logits(ids, labels, standalone_mask, standalone_positions)
            for ids, labels in zip(standalone_inputs, standalone_labels, strict=True)
        ]
        torch.testing.assert_close(
            packed_gen_logits,
            torch.cat(standalone_gen_logits, dim=0),
            rtol=2e-5,
            atol=2e-6,
        )

        packed_gen_phase = model.forward_generator_phase(
            input_ids=packed_input,
            attention_mask=packed_mask,
            labels=packed_labels,
            position_ids=packed_positions,
        )
        packed_targets = packed_labels[packed_labels.ne(-100)]
        packed_emd_loss = torch.nn.functional.cross_entropy(
            packed_gen_logits.float(),
            packed_targets,
        )
        torch.testing.assert_close(
            packed_gen_phase.gen_loss_raw,
            packed_emd_loss,
            rtol=2e-5,
            atol=2e-6,
        )
        standalone_gen_phases = [
            model.forward_generator_phase(
                input_ids=ids,
                attention_mask=standalone_mask,
                labels=labels,
                position_ids=standalone_positions,
            )
            for ids, labels in zip(standalone_inputs, standalone_labels, strict=True)
        ]
        torch.testing.assert_close(
            packed_gen_phase.gen_loss_raw,
            torch.stack([phase.gen_loss_raw for phase in standalone_gen_phases]).mean(),
            rtol=2e-5,
            atol=2e-6,
        )

        packed_corrupted = packed_input.clone()
        packed_corrupted[0, 2] = 30
        packed_corrupted[0, 6] = 31
        packed_disc_labels = torch.zeros_like(packed_input, dtype=torch.float32)
        packed_disc_labels[0, 2] = 1.0
        packed_disc_labels[0, 6] = 1.0
        packed_disc_logits = _discriminator_logits(
            packed_corrupted,
            packed_mask,
            packed_positions,
            packed_context,
        )
        standalone_disc_logits: list[torch.Tensor] = []
        standalone_disc_losses: list[torch.Tensor] = []
        for document_idx, ids in enumerate(standalone_inputs):
            corrupted = ids.clone()
            corrupted[0, 2] = 30 + document_idx
            labels = torch.zeros_like(ids, dtype=torch.float32)
            labels[0, 2] = 1.0
            standalone_disc_logits.append(
                _discriminator_logits(
                    corrupted,
                    standalone_mask,
                    standalone_positions,
                    None,
                )
            )
            standalone_disc_losses.append(
                model.forward_discriminator_phase(
                    input_ids=ids,
                    corrupted_input_ids=corrupted,
                    disc_labels=labels,
                    attention_mask=standalone_mask,
                    position_ids=standalone_positions,
                ).disc_loss_raw
            )

        torch.testing.assert_close(
            packed_disc_logits,
            torch.cat(standalone_disc_logits, dim=1),
            rtol=2e-5,
            atol=2e-6,
        )
        packed_disc_phase = model.forward_discriminator_phase(
            input_ids=packed_input,
            corrupted_input_ids=packed_corrupted,
            disc_labels=packed_disc_labels,
            attention_mask=packed_mask,
            position_ids=packed_positions,
            doc_context_index=packed_context,
        )
        torch.testing.assert_close(
            packed_disc_phase.disc_loss_raw,
            torch.stack(standalone_disc_losses).mean(),
            rtol=2e-5,
            atol=2e-6,
        )


def test_packed_rope_rtd_accepts_local_positions_and_matches_standalone() -> None:
    """RoPE packed training must preserve optional learned absolute positions."""

    pytest.importorskip("transformers")
    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    torch.manual_seed(43)
    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
        type_vocab_size=0,
        use_absolute_position_embeddings=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        attention_implementation="eager",
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        mask_token_id=3,
    )
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=DebertaRoPEModel(cfg),
        generator_backbone=DebertaRoPEModel(cfg),
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="none",
    ).eval()

    standalone_input = torch.tensor([[1, 10, 3, 2]], dtype=torch.long)
    standalone_labels = torch.tensor([[-100, -100, 12, -100]], dtype=torch.long)
    standalone_mask = torch.ones((1, 4), dtype=torch.bool)
    standalone_positions = torch.arange(4).unsqueeze(0)
    packed_input = torch.cat((standalone_input, standalone_input), dim=1)
    packed_labels = torch.cat((standalone_labels, standalone_labels), dim=1)
    packed_doc_ids = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=torch.long)
    packed_mask = build_doc_block_mask(packed_doc_ids)
    packed_positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)
    packed_context = torch.tensor([[0, 0, 0, 0, 4, 4, 4, 4]], dtype=torch.long)

    with torch.no_grad():
        standalone_hidden = model.generator(
            input_ids=standalone_input,
            attention_mask=standalone_mask,
            position_ids=standalone_positions,
            return_dict=True,
        ).last_hidden_state
        packed_hidden = model.generator(
            input_ids=packed_input,
            attention_mask=packed_mask,
            position_ids=packed_positions,
            return_dict=True,
        ).last_hidden_state
        expected_hidden = torch.cat((standalone_hidden, standalone_hidden), dim=1)
        torch.testing.assert_close(packed_hidden, expected_hidden, rtol=1e-5, atol=1e-6)

        row_global_hidden = model.generator(
            input_ids=packed_input,
            attention_mask=packed_mask,
            position_ids=torch.arange(8).unsqueeze(0),
            return_dict=True,
        ).last_hidden_state
        assert not torch.allclose(row_global_hidden[:, 4:], standalone_hidden)

        standalone_gen = model.forward_generator_phase(
            input_ids=standalone_input,
            attention_mask=standalone_mask,
            labels=standalone_labels,
            position_ids=standalone_positions,
        )
        packed_gen = model.forward_generator_phase(
            input_ids=packed_input,
            attention_mask=packed_mask,
            labels=packed_labels,
            position_ids=packed_positions,
        )
        torch.testing.assert_close(packed_gen.gen_loss_raw, standalone_gen.gen_loss_raw)

        standalone_corrupted = standalone_input.clone()
        standalone_corrupted[0, 2] = 30
        standalone_disc_labels = torch.zeros_like(standalone_input, dtype=torch.float32)
        standalone_disc_labels[0, 2] = 1.0
        standalone_disc = model.forward_discriminator_phase(
            input_ids=standalone_input,
            corrupted_input_ids=standalone_corrupted,
            disc_labels=standalone_disc_labels,
            attention_mask=standalone_mask,
            position_ids=standalone_positions,
        )
        packed_disc = model.forward_discriminator_phase(
            input_ids=packed_input,
            corrupted_input_ids=torch.cat((standalone_corrupted, standalone_corrupted), dim=1),
            disc_labels=torch.cat((standalone_disc_labels, standalone_disc_labels), dim=1),
            attention_mask=packed_mask,
            position_ids=packed_positions,
            doc_context_index=packed_context,
        )
        torch.testing.assert_close(packed_disc.disc_loss_raw, standalone_disc.disc_loss_raw)


def test_flash_batch_meta_is_cross_document_predicate():
    from deberta.modeling.mask_utils import FlashBatchMeta

    assert FlashBatchMeta().is_cross_document() is False
    assert FlashBatchMeta(route_hint="dense").is_cross_document() is False
    assert FlashBatchMeta(route_hint="varlen").is_cross_document() is False
    assert FlashBatchMeta(route_hint="docblock").is_cross_document() is True
    assert FlashBatchMeta(route_hint="docblock_bias").is_cross_document() is True
    assert FlashBatchMeta(route_hint=" DOCBLOCK ").is_cross_document() is True
    assert (
        FlashBatchMeta(doc_segment_offsets=torch.tensor([0, 2], dtype=torch.int32)).is_cross_document()
        is True
    )


def test_pretrainer_raises_clear_error_when_generator_word_embeddings_cannot_be_tied():

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
        )


def test_synced_buffer_embedding_tracks_sync_updates():
    from deberta.modeling.rtd import _SyncedBufferEmbedding

    init_weight = torch.randn((16, 8), dtype=torch.float32)
    tied = _SyncedBufferEmbedding(
        init_weight=init_weight,
        padding_idx=0,
    )
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with torch.no_grad():
        out_a = tied(input_ids)
        new_weight = torch.randn_like(init_weight)
        tied.sync_from(new_weight)
        out_b = tied(input_ids)

    assert isinstance(tied.base_weight, torch.nn.Parameter)
    assert tied.base_weight.requires_grad is False
    assert not torch.allclose(out_a, out_b)
    expected = torch.nn.functional.embedding(input_ids, new_weight, padding_idx=0)
    torch.testing.assert_close(out_b, expected, rtol=0.0, atol=0.0)


def test_synced_buffer_embedding_sync_updates_compiled_module():

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")

    from deberta.modeling.rtd import _SyncedBufferEmbedding

    init_weight = torch.randn((16, 8), dtype=torch.float32)
    tied = _SyncedBufferEmbedding(
        init_weight=init_weight,
        padding_idx=0,
    )
    compiled = torch.compile(tied, backend="aot_eager", mode="default", dynamic=False)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with torch.no_grad():
        out_a = compiled(input_ids)
        new_weight = torch.randn_like(init_weight)
        tied.sync_from(new_weight)
        out_b = compiled(input_ids)

    assert not torch.allclose(out_a, out_b)
    expected = torch.nn.functional.embedding(input_ids, new_weight, padding_idx=0)
    torch.testing.assert_close(out_b, expected, rtol=1e-6, atol=1e-6)


def test_synced_buffer_embedding_rejects_sharded_dtensor_sync(monkeypatch):

    from deberta.modeling import rtd as rtd_mod

    init_weight = torch.randn((16, 8), dtype=torch.float32)
    tied = rtd_mod._SyncedBufferEmbedding(
        init_weight=init_weight,
        padding_idx=0,
    )
    new_weight = torch.randn_like(init_weight)
    original_probe = rtd_mod._is_sharded_dtensor

    def _probe(tensor: torch.Tensor) -> bool:
        if tensor is tied.base_weight:
            return True
        return original_probe(tensor)

    monkeypatch.setattr(rtd_mod, "_is_sharded_dtensor", _probe)
    with pytest.raises(RuntimeError, match="sharded DTensor"):
        tied.sync_from(new_weight)


def test_synced_buffer_embedding_gdes_bias_matches_base_weight_dtype():
    from deberta.modeling.rtd import _SyncedBufferEmbedding

    init_weight = torch.randn((16, 8), dtype=torch.bfloat16)
    tied = _SyncedBufferEmbedding(
        init_weight=init_weight,
        padding_idx=0,
    )
    assert tied.bias.dtype == init_weight.dtype


def test_pretrainer_es_embedding_alias_is_static_after_model_surgery():

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
    shared_weight = model.discriminator.embeddings.word_embeddings.weight
    assert shared_weight is model.generator.embeddings.word_embeddings.weight

    model.generator.embeddings.word_embeddings = torch.nn.Embedding(
        cfg.vocab_size,
        cfg.hidden_size,
        padding_idx=cfg.pad_token_id,
    )
    # ES aliasing is resolved at init-time; replacing modules post-init does not auto-repatch.
    assert shared_weight is model.discriminator.embeddings.word_embeddings.weight
    assert shared_weight is not model.generator.embeddings.word_embeddings.weight

    input_ids = torch.tensor([[1, 7, 8, 2]], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[0, 1] = input_ids[0, 1]

    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
            sampling_temperature=1.0,
            gen_loss_weight=1.0,
            disc_loss_weight=50.0,
        )


def test_rope_model_treats_missing_attention_mask_as_unpadded_contract():

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


def test_native_hf_deberta_v2_forward_smoke():

    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DebertaV2Model

    cfg = DebertaV2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = DebertaV2Model(cfg).eval()
    input_ids = torch.randint(low=0, high=cfg.vocab_size, size=(2, 8), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        out_2d = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pair_mask = attention_mask[:, :, None] & attention_mask[:, None, :]
        out_3d = model(input_ids=input_ids, attention_mask=pair_mask).last_hidden_state

    assert out_2d.shape == (2, 8, 32)
    torch.testing.assert_close(out_2d, out_3d, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("pos_att_type", "seed"),
    [("c2p|p2c", 123), ("c2p|p2c|p2p", 321)],
    ids=["c2p_p2c", "with_p2p"],
)
def test_native_hf_deberta_v2_cached_and_stable_attention_match_dynamic(pos_att_type: str, seed: int):

    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DebertaV2Model

    cfg = DebertaV2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        relative_attention=True,
        pos_att_type=pos_att_type,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    input_ids = torch.randint(low=0, high=cfg.vocab_size, size=(2, 8), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    torch.manual_seed(seed)
    cfg.hf_attention_kernel = "dynamic"
    dynamic_model = DebertaV2Model(cfg).eval()
    snapshot = {k: v.detach().clone() for k, v in dynamic_model.state_dict().items()}

    cfg.hf_attention_kernel = "cached_bmm"
    cached_model = DebertaV2Model(cfg).eval()
    cached_model.load_state_dict(snapshot, strict=True)

    cfg.hf_attention_kernel = "stable"
    stable_model = DebertaV2Model(cfg).eval()
    stable_model.load_state_dict(snapshot, strict=True)

    with torch.no_grad():
        out_dynamic = dynamic_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out_cached = cached_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out_stable = stable_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    torch.testing.assert_close(out_dynamic, out_cached, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_dynamic, out_stable, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("term", ["c2p", "p2c"])
@pytest.mark.parametrize("kernel", ["dynamic", "cached_bmm", "stable"])
def test_native_disentangled_signed_bucket_forward_and_gradients_match_definition(
    term: str,
    kernel: str,
) -> None:
    """Each positional term must use the same canonical signed q-k bucket."""

    pytest.importorskip("transformers")
    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DisentangledSelfAttention

    torch.manual_seed(17)
    cfg = DebertaV2Config(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=4,
        max_relative_positions=4,
        position_buckets=-1,
        relative_attention=True,
        pos_att_type=term,
        share_att_key=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg.hf_attention_kernel = kernel
    attention = DisentangledSelfAttention(cfg).eval()
    attention.float()

    batch_size = 2
    seq_len = 4
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    query_actual = torch.randn((batch_size, cfg.num_attention_heads, seq_len, head_dim), requires_grad=True)
    key_actual = torch.randn_like(query_actual, requires_grad=True)
    rel_actual = torch.randn((2 * cfg.max_relative_positions, cfg.hidden_size), requires_grad=True)
    query_expected = query_actual.detach().clone().requires_grad_()
    key_expected = key_actual.detach().clone().requires_grad_()
    rel_expected = rel_actual.detach().clone().requires_grad_()
    relative_pos = torch.arange(seq_len)[:, None] - torch.arange(seq_len)[None, :]
    scale_factor = 2

    actual = attention.disentangled_attention_bias(
        query_layer=query_actual,
        key_layer=key_actual,
        relative_pos=relative_pos,
        rel_embeddings=rel_actual,
        scale_factor=scale_factor,
    )

    use_query_projection = term == "p2c"
    position_scores = attention._project_rel(
        rel_expected,
        use_query=use_query_projection,
    )
    bucket = (relative_pos + int(attention.pos_ebd_size)).clamp(
        min=0,
        max=(2 * int(attention.pos_ebd_size)) - 1,
    )
    selected_positions = position_scores[:, bucket, :]
    content = key_expected[:, :, None, :, :] if term == "p2c" else query_expected[:, :, :, None, :]
    expected = (content * selected_positions.unsqueeze(0)).sum(dim=-1)
    expected = expected / math.sqrt(float(head_dim * scale_factor))

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    loss_weight = torch.randn_like(actual)
    projection = attention.pos_query_proj if term == "p2c" else attention.pos_key_proj
    assert projection is not None
    actual_grads = torch.autograd.grad(
        (actual * loss_weight).sum(),
        (query_actual, key_actual, rel_actual, projection.weight, projection.bias),
        allow_unused=True,
    )
    expected_grads = torch.autograd.grad(
        (expected * loss_weight).sum(),
        (query_expected, key_expected, rel_expected, projection.weight, projection.bias),
        allow_unused=True,
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        if actual_grad is None or expected_grad is None:
            assert actual_grad is None and expected_grad is None
        else:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-6)


def test_native_attention_matches_transformers_reference_on_active_tokens() -> None:
    """Corrected native attention must match the independent HF implementation."""

    pytest.importorskip("transformers")
    from transformers import DebertaV2Config
    from transformers.models.deberta_v2.modeling_deberta_v2 import (
        DisentangledSelfAttention as TransformersDisentangledSelfAttention,
    )

    from deberta.modeling.deberta_v2_native import DisentangledSelfAttention

    torch.manual_seed(29)
    cfg = DebertaV2Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=8,
        max_relative_positions=8,
        position_buckets=-1,
        relative_attention=True,
        pos_att_type=["c2p", "p2c"],
        share_att_key=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg.hf_attention_kernel = "dynamic"
    reference = TransformersDisentangledSelfAttention(cfg).eval()
    native = DisentangledSelfAttention(cfg).eval()
    native.load_state_dict(reference.state_dict(), strict=True)

    hidden_states = torch.randn((2, 6, cfg.hidden_size), dtype=torch.float32)
    keep_mask = torch.tensor([[True, True, True, True, False, False], [True, True, True, True, True, False]])
    attention_mask = keep_mask[:, None, None, :]
    rel_embeddings = torch.randn((2 * cfg.max_relative_positions, cfg.hidden_size))

    with torch.no_grad():
        native_out, _ = native(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rel_embeddings=rel_embeddings,
        )
        reference_out, _ = reference(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rel_embeddings=rel_embeddings,
        )

    torch.testing.assert_close(native_out[keep_mask], reference_out[keep_mask], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("kernel", ["cached_bmm", "stable"])
def test_native_hf_deberta_v2_cached_bias_recomputes_for_new_query_key(kernel: str):
    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DisentangledSelfAttention

    cfg = DebertaV2Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        relative_attention=True,
        pos_att_type="c2p|p2c",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg.hf_attention_kernel = kernel
    attn = DisentangledSelfAttention(cfg).eval()

    bsz, nheads, qlen, klen = 2, cfg.num_attention_heads, 4, 4
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    query_a = torch.randn((bsz, nheads, qlen, head_dim), dtype=torch.float32)
    key_a = torch.randn((bsz, nheads, klen, head_dim), dtype=torch.float32)
    query_b = torch.randn((bsz, nheads, qlen, head_dim), dtype=torch.float32)
    key_b = torch.randn((bsz, nheads, klen, head_dim), dtype=torch.float32)
    rel_embeddings = torch.randn((2 * cfg.max_position_embeddings, cfg.hidden_size), dtype=torch.float32)

    with torch.no_grad():
        score_a = attn.disentangled_attention_bias(
            query_layer=query_a,
            key_layer=key_a,
            relative_pos=None,
            rel_embeddings=rel_embeddings,
            scale_factor=3,
        )
        score_b = attn.disentangled_attention_bias(
            query_layer=query_b,
            key_layer=key_b,
            relative_pos=None,
            rel_embeddings=rel_embeddings,
            scale_factor=3,
        )

    assert score_a.shape == score_b.shape
    assert not torch.allclose(score_a, score_b, rtol=0.0, atol=0.0)
    assert float((score_a - score_b).abs().max().item()) > 0.0


def test_native_hf_deberta_v2_cached_bmm_casts_relative_bias_to_query_dtype(
    monkeypatch: pytest.MonkeyPatch,
):

    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DisentangledSelfAttention

    cfg = DebertaV2Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        relative_attention=True,
        pos_att_type="c2p",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg.hf_attention_kernel = "cached_bmm"
    attn = DisentangledSelfAttention(cfg).eval()

    head_dim = cfg.hidden_size // cfg.num_attention_heads

    def _fake_project_rel(_rel_embeddings: torch.Tensor, *, use_query: bool) -> torch.Tensor:
        del _rel_embeddings, use_query
        return torch.randn(
            (cfg.num_attention_heads, 2 * cfg.max_position_embeddings, head_dim),
            dtype=torch.bfloat16,
        )

    monkeypatch.setattr(attn, "_project_rel", _fake_project_rel)

    bmm_dtypes: list[tuple[torch.dtype, torch.dtype]] = []
    original_bmm = torch.bmm

    def _checked_bmm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        bmm_dtypes.append((lhs.dtype, rhs.dtype))
        assert lhs.dtype == torch.float32
        assert rhs.dtype == torch.float32
        return original_bmm(lhs, rhs)

    monkeypatch.setattr(torch, "bmm", _checked_bmm)

    bsz, nheads, qlen, klen = 2, cfg.num_attention_heads, 4, 4
    query = torch.randn((bsz, nheads, qlen, head_dim), dtype=torch.float32)
    key = torch.randn((bsz, nheads, klen, head_dim), dtype=torch.float32)
    rel_embeddings = torch.randn((2 * cfg.max_position_embeddings, cfg.hidden_size), dtype=torch.float32)

    out = attn.disentangled_attention_bias(
        query_layer=query,
        key_layer=key,
        relative_pos=None,
        rel_embeddings=rel_embeddings,
        scale_factor=2,
    )

    assert out.shape == (bsz, nheads, qlen, klen)
    assert bmm_dtypes


def test_native_hf_deberta_v2_dynamic_bias_casts_relative_bias_to_query_dtype(
    monkeypatch: pytest.MonkeyPatch,
):

    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DisentangledSelfAttention

    cfg = DebertaV2Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        relative_attention=True,
        pos_att_type="c2p|p2c",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg.hf_attention_kernel = "dynamic"
    attn = DisentangledSelfAttention(cfg).eval()

    head_dim = cfg.hidden_size // cfg.num_attention_heads

    def _fake_project_rel(_rel_embeddings: torch.Tensor, *, use_query: bool) -> torch.Tensor:
        del _rel_embeddings, use_query
        return torch.randn(
            (cfg.num_attention_heads, 2 * cfg.max_position_embeddings, head_dim),
            dtype=torch.bfloat16,
        )

    monkeypatch.setattr(attn, "_project_rel", _fake_project_rel)

    einsum_dtypes: list[tuple[str, tuple[torch.dtype, ...]]] = []
    original_einsum = torch.einsum

    def _checked_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
        if equation in {"bhqd,hkd->bhqk", "bhkd,hqd->bhkq"}:
            einsum_dtypes.append((equation, tuple(t.dtype for t in operands)))
            assert all(t.dtype == torch.float32 for t in operands)
        return original_einsum(equation, *operands)

    monkeypatch.setattr(torch, "einsum", _checked_einsum)

    bsz, nheads, qlen, klen = 2, cfg.num_attention_heads, 4, 4
    query = torch.randn((bsz, nheads, qlen, head_dim), dtype=torch.float32)
    key = torch.randn((bsz, nheads, klen, head_dim), dtype=torch.float32)
    rel_embeddings = torch.randn((2 * cfg.max_position_embeddings, cfg.hidden_size), dtype=torch.float32)

    out = attn.disentangled_attention_bias(
        query_layer=query,
        key_layer=key,
        relative_pos=None,
        rel_embeddings=rel_embeddings,
        scale_factor=3,
    )

    assert out.shape == (bsz, nheads, qlen, klen)
    assert {eq for eq, _ in einsum_dtypes} == {"bhqd,hkd->bhqk", "bhkd,hqd->bhkq"}


def test_native_hf_deberta_v2_p2p_only_bias_is_nonzero():

    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DisentangledSelfAttention

    cfg = DebertaV2Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        relative_attention=True,
        pos_att_type="p2p",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    attn = DisentangledSelfAttention(cfg).eval()

    bsz, nheads, qlen, klen = 2, 4, 8, 8
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    query = torch.randn(bsz, nheads, qlen, head_dim)
    key = torch.randn(bsz, nheads, klen, head_dim)
    rel_embeddings = torch.randn(2 * cfg.max_position_embeddings, cfg.hidden_size)

    score = attn.disentangled_attention_bias(
        query_layer=query,
        key_layer=key,
        relative_pos=None,
        rel_embeddings=rel_embeddings,
        scale_factor=2,  # base 1 + p2p 1
    )
    assert score.shape == (bsz, nheads, qlen, klen)
    assert torch.isfinite(score).all()
    assert float(score.abs().sum().item()) > 0.0


def test_native_hf_deberta_v2_p2p_bias_respects_scale_factor():
    import math

    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DisentangledSelfAttention

    cfg = DebertaV2Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        relative_attention=True,
        pos_att_type="p2p",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    attn = DisentangledSelfAttention(cfg).eval()

    bsz, nheads, qlen, klen = 2, 4, 8, 8
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    query = torch.randn(bsz, nheads, qlen, head_dim)
    key = torch.randn(bsz, nheads, klen, head_dim)
    rel_embeddings = torch.randn(2 * cfg.max_position_embeddings, cfg.hidden_size)

    with torch.no_grad():
        score_scale_2 = attn.disentangled_attention_bias(
            query_layer=query,
            key_layer=key,
            relative_pos=None,
            rel_embeddings=rel_embeddings,
            scale_factor=2,
        )
        score_scale_8 = attn.disentangled_attention_bias(
            query_layer=query,
            key_layer=key,
            relative_pos=None,
            rel_embeddings=rel_embeddings,
            scale_factor=8,
        )

    mean_abs_2 = float(score_scale_2.abs().mean().item())
    mean_abs_8 = float(score_scale_8.abs().mean().item())
    assert mean_abs_2 > 0.0
    assert mean_abs_8 > 0.0
    ratio = mean_abs_2 / mean_abs_8
    assert ratio == pytest.approx(math.sqrt(8.0 / 2.0), rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("kernel", ["dynamic", "cached_bmm", "stable"])
def test_native_hf_deberta_v2_c2p_p2c_bias_respects_scale_factor(kernel: str):
    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DisentangledSelfAttention

    cfg = DebertaV2Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        relative_attention=True,
        pos_att_type="c2p|p2c",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg.hf_attention_kernel = kernel
    attn = DisentangledSelfAttention(cfg).eval()

    bsz, nheads, qlen, klen = 2, cfg.num_attention_heads, 8, 8
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    query = torch.randn((bsz, nheads, qlen, head_dim), dtype=torch.float32)
    key = torch.randn((bsz, nheads, klen, head_dim), dtype=torch.float32)
    rel_embeddings = torch.randn((2 * cfg.max_position_embeddings, cfg.hidden_size), dtype=torch.float32)

    with torch.no_grad():
        score_scale_3 = attn.disentangled_attention_bias(
            query_layer=query,
            key_layer=key,
            relative_pos=None,
            rel_embeddings=rel_embeddings,
            scale_factor=3,
        )
        score_scale_12 = attn.disentangled_attention_bias(
            query_layer=query,
            key_layer=key,
            relative_pos=None,
            rel_embeddings=rel_embeddings,
            scale_factor=12,
        )

    mean_abs_3 = float(score_scale_3.abs().mean().item())
    mean_abs_12 = float(score_scale_12.abs().mean().item())
    assert mean_abs_3 > 0.0
    assert mean_abs_12 > 0.0
    ratio = mean_abs_3 / mean_abs_12
    assert ratio == pytest.approx(math.sqrt(12.0 / 3.0), rel=1e-4, abs=1e-4)


def test_native_hf_deberta_v2_log_bucket_clamps_relative_positions():
    from deberta.modeling.deberta_v2_native import _make_log_bucket_position

    relative_pos = torch.tensor(
        [-999, -800, -511, -510, 0, 510, 511, 800, 999],
        dtype=torch.long,
    )
    bucket = _make_log_bucket_position(relative_pos, bucket_size=128, max_position=512)

    neg_edge = bucket[2].item()  # rel=-511
    pos_edge = bucket[6].item()  # rel=511

    assert bucket[0].item() == neg_edge
    assert bucket[1].item() == neg_edge
    assert bucket[-1].item() == pos_edge
    assert bucket[-2].item() == pos_edge


def test_native_hf_deberta_v2_rejects_invalid_attention_kernel_config():

    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DebertaV2Model

    cfg = DebertaV2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
        type_vocab_size=0,
    )

    cfg.hf_attention_kernel = "bad_kernel"
    with pytest.raises(ValueError, match="model.hf.attention_kernel must be one of"):
        _ = DebertaV2Model(cfg)


def test_native_hf_deberta_v2_stable_attention_handles_fully_masked_rows():

    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DebertaV2Model

    cfg = DebertaV2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
        relative_attention=True,
        pos_att_type="c2p|p2c",
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg.hf_attention_kernel = "stable"
    model = DebertaV2Model(cfg).eval()

    input_ids = torch.randint(low=0, high=cfg.vocab_size, size=(2, 8), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    assert torch.isfinite(out).all()


def test_native_hf_deberta_v2_stable_attention_random_masks_stay_finite():

    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DebertaV2Model

    cfg = DebertaV2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        relative_attention=True,
        pos_att_type="c2p|p2c",
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg.hf_attention_kernel = "stable"
    model = DebertaV2Model(cfg).eval()

    torch.manual_seed(7)
    input_ids = torch.randint(low=0, high=cfg.vocab_size, size=(2, 8), dtype=torch.long)
    attention_mask = torch.randint(low=0, high=2, size=input_ids.shape, dtype=torch.long).to(torch.bool)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    assert torch.isfinite(out).all()


def test_native_hf_deberta_v2_pairwise_mask_uses_diagonal_for_query_activity():
    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DisentangledSelfAttention

    cfg = DebertaV2Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
        relative_attention=False,
        pos_att_type="",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    attn = DisentangledSelfAttention(cfg).eval()
    x = torch.randn((1, 4, cfg.hidden_size), dtype=torch.float32)

    # Row 3 has a CLS fallback edge but diagonal=False; packed-mask semantics
    # require this query to stay inactive and produce zero output/probs.
    pair_keep = torch.tensor(
        [
            [
                [
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                ]
            ]
        ],
        dtype=torch.bool,
    )

    with torch.no_grad():
        out, probs = attn(x, pair_keep, output_attentions=True)

    assert probs is not None
    assert torch.isfinite(out).all()
    assert torch.isfinite(probs).all()
    assert torch.allclose(out[0, 3], torch.zeros_like(out[0, 3]), atol=1e-6)
    assert torch.allclose(probs[0, :, 3, :], torch.zeros_like(probs[0, :, 3, :]), atol=1e-6)
    assert not torch.allclose(out[0, 0], torch.zeros_like(out[0, 0]), atol=1e-6)


def test_native_hf_deberta_v2_stable_compile_step_is_finite():

    pytest.importorskip("transformers")
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DebertaV2Model

    cfg = DebertaV2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
        relative_attention=True,
        pos_att_type="c2p|p2c",
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg.hf_attention_kernel = "stable"
    input_ids = torch.randint(low=0, high=cfg.vocab_size, size=(2, 8), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    for backend in ("aot_eager", "inductor"):
        model = DebertaV2Model(cfg).train()
        compiled = torch.compile(model, backend=backend, mode="default", dynamic=False)
        out = compiled(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        loss = out.float().mean()
        loss.backward()
        grads = [p.grad for p in compiled.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all().item() for g in grads)


@pytest.mark.parametrize("relative_attention", [False, True], ids=["plain", "disentangled"])
def test_native_hf_deberta_v2_forward_with_none_mask_matches_all_ones(relative_attention: bool):

    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DebertaV2Model

    extra = {"relative_attention": True, "pos_att_type": "c2p|p2c"} if relative_attention else {}
    cfg = DebertaV2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        **extra,
    )
    model = DebertaV2Model(cfg).eval()
    input_ids = torch.randint(low=0, high=cfg.vocab_size, size=(2, 8), dtype=torch.long)
    all_ones = torch.ones_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        out_none = model(input_ids=input_ids, attention_mask=None).last_hidden_state
        out_ones = model(input_ids=input_ids, attention_mask=all_ones).last_hidden_state

    assert out_none.shape == (2, 8, 32)
    assert torch.isfinite(out_none).all()
    torch.testing.assert_close(out_none, out_ones, rtol=0.0, atol=0.0)


def test_native_hf_deberta_v2_padding_mask_avoids_quadratic_expansion():

    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DebertaV2Model

    cfg = DebertaV2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = DebertaV2Model(cfg).eval()

    # 2D padding mask → encoder should produce (B,1,1,S) not (B,1,S,S).
    mask_2d = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)
    attn_mask_4d = model.encoder.get_attention_mask(mask_2d)
    assert attn_mask_4d.shape == (1, 1, 1, 8), f"Expected (1,1,1,8), got {attn_mask_4d.shape}"

    # Full forward with padding: padded positions must be zeroed, active positions finite.
    input_ids = torch.randint(low=0, high=cfg.vocab_size, size=(1, 8), dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=mask_2d).last_hidden_state
        out_broadcast = model(input_ids=input_ids, attention_mask=attn_mask_4d).last_hidden_state
    assert out.shape == (1, 8, 32)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, out_broadcast, rtol=0.0, atol=0.0)

    # Broadcast mask must produce identical output to the old dense (B,1,S,S) expansion.
    mask_dense = (mask_2d.bool()[:, :, None] & mask_2d.bool()[:, None, :]).unsqueeze(1)
    with torch.no_grad():
        out_dense = model(input_ids=input_ids, attention_mask=mask_dense).last_hidden_state
    torch.testing.assert_close(out, out_dense, rtol=0.0, atol=0.0)


def test_native_hf_deberta_v2_rejects_conv_checkpoint_configs():

    pytest.importorskip("transformers")

    from transformers import DebertaV2Config

    from deberta.modeling.deberta_v2_native import DebertaV2Model

    cfg = DebertaV2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        conv_kernel_size=3,
    )
    with pytest.raises(ValueError, match="conv_kernel_size"):
        DebertaV2Model(cfg)


def test_rope_model_accepts_positional_input_ids_call():

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


def test_rope_keel_default_alpha_matches_paper_contract():

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEEncoder

    cfg = DebertaRoPEConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        norm_arch="keel",
        keel_alpha_init=None,
        keel_alpha_learnable=False,
    )
    encoder = DebertaRoPEEncoder(cfg).eval()
    expected_alpha = 1.0 / math.sqrt(float(2 * int(cfg.num_hidden_layers)))

    for layer in encoder.layers:
        assert float(layer.alpha1.alpha.item()) == pytest.approx(expected_alpha)
        assert float(layer.alpha2.alpha.item()) == pytest.approx(expected_alpha)
        assert not isinstance(layer.alpha1.alpha, torch.nn.Parameter)
        assert not isinstance(layer.alpha2.alpha, torch.nn.Parameter)


def test_rope_keel_learnable_alpha_is_independent_per_residual_sublayer():

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPELayer

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
        norm_arch="keel",
        keel_alpha_init=1.0,
        keel_alpha_learnable=True,
    )
    layer = DebertaRoPELayer(cfg, alpha_init=1.0).eval()

    assert isinstance(layer.alpha1.alpha, torch.nn.Parameter)
    assert isinstance(layer.alpha2.alpha, torch.nn.Parameter)
    assert layer.alpha1.alpha.data_ptr() != layer.alpha2.alpha.data_ptr()

    torch.manual_seed(0)
    x = torch.randn(2, 8, 32)
    attention_mask = torch.ones(2, 8, dtype=torch.bool)

    with torch.no_grad():
        layer.alpha1.alpha.fill_(0.0)
        layer.alpha2.alpha.fill_(0.0)
        out_00 = layer(x, attention_mask)

        layer.alpha1.alpha.fill_(5.0)
        layer.alpha2.alpha.fill_(0.0)
        out_50 = layer(x, attention_mask)

        layer.alpha1.alpha.fill_(0.0)
        layer.alpha2.alpha.fill_(5.0)
        out_05 = layer(x, attention_mask)

    assert not torch.allclose(out_00, out_50)
    assert not torch.allclose(out_00, out_05)
    assert not torch.allclose(out_50, out_05)


def test_rope_model_supports_output_hidden_states():

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

    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
        out_tuple = model(input_ids=input_ids, output_hidden_states=True, return_dict=False)

    assert out.hidden_states is not None
    assert len(out.hidden_states) == (cfg.num_hidden_layers + 1)
    assert out_tuple[1] is not None
    assert len(out_tuple[1]) == (cfg.num_hidden_layers + 1)


def test_pretrainer_missing_attention_mask_means_all_tokens_are_active():

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
    )

    torch.manual_seed(0)
    out_explicit = model(
        input_ids=input_ids,
        attention_mask=explicit_mask,
        labels=labels,
        sampling_temperature=1.0,
        gen_loss_weight=1.0,
        disc_loss_weight=50.0,
    )

    assert out_missing.disc_token_count.item() == input_ids.numel()
    assert out_explicit.disc_token_count.item() == explicit_mask.sum().item()


def test_pretrainer_disc_loss_supervises_all_tokens_without_attention_mask():

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
    )
    assert int(out.disc_token_count.item()) == input_ids.numel()


def test_pretrainer_disc_active_keeps_all_non_padding_tokens_even_if_sampled_special(monkeypatch):

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

    def _force_special_sample(
        logits: torch.Tensor,
        *,
        temperature: float,
        forbidden_vocab_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        del logits, temperature, forbidden_vocab_mask
        return torch.tensor([int(cfg.cls_token_id)], dtype=torch.long)

    monkeypatch.setattr(model, "_gumbel_sample", _force_special_sample)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
            sampling_temperature=1.0,
            gen_loss_weight=1.0,
            disc_loss_weight=50.0,
        )

    torch.testing.assert_close(out.disc_token_count, torch.tensor(4.0))


def test_rope_config_rejects_unknown_ffn_type():

    pytest.importorskip("transformers")

    from deberta.modeling.rope_encoder import DebertaRoPEConfig

    with pytest.raises(ValueError, match="ffn_type must be one of"):
        _ = DebertaRoPEConfig(ffn_type="glu")


def test_rotary_compile_mode_requires_prefilled_cache(monkeypatch):
    from deberta.modeling import rope as rope_mod

    rope = rope_mod.RotaryEmbedding(dim=8, base=10_000.0)
    device = torch.device("cpu")

    monkeypatch.setattr(rope_mod, "is_torch_compiling", lambda: True)
    with pytest.raises(RuntimeError, match="not prefilled"):
        _ = rope.get_cos_sin(8, device=device, dtype=torch.float32)

    rope.prefill_cache(8, device=device, dtype=torch.float32)
    assert rope._cache is not None
    assert rope._cache_device == rope._cache.cos.device
    cos1, sin1 = rope.get_cos_sin(8, device=device, dtype=torch.float32)
    cos2, sin2 = rope.get_cos_sin(8, device=device, dtype=torch.float32)
    assert cos1.data_ptr() == cos2.data_ptr()
    assert sin1.data_ptr() == sin2.data_ptr()

    with pytest.raises(RuntimeError, match="too short"):
        _ = rope.get_cos_sin(9, device=device, dtype=torch.float32)

    monkeypatch.setattr(rope_mod, "is_torch_compiling", lambda: False)
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
