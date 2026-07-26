from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from _config_factories import make_native_deberta_config


def test_flash_parity_rejects_empty_case_selection() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/flashdeberta_parity_test.py",
            "--case",
            "docblock_bias",
            "--no-include-docblock-bias",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "No parity cases remain" in result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for real Inductor coverage.")
def test_real_fixed_flash_attention_compiles_with_inductor_and_backward() -> None:
    """The production fixed custom op must remain opaque to real Inductor."""

    from deberta.modeling.flashdeberta_attention import (
        FlashDisentangledSelfAttention,
        flashdeberta_fixed_import_error,
    )

    if flashdeberta_fixed_import_error() is not None:
        pytest.skip("FlashDeBERTa fixed kernels are unavailable in this environment.")

    cfg = make_native_deberta_config(
        flash=True,
        vocab_size=64,
        num_attention_heads=2,
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=8,
        max_relative_positions=16,
        pos_att_type=["c2p", "p2c"],
        pad_token_id=0,
        position_biased_input=False,
    )
    attention = FlashDisentangledSelfAttention(cfg).to(device="cuda", dtype=torch.bfloat16)

    def _reject_eager(**_kwargs):
        raise AssertionError("real fixed-route compile smoke must not fall back to eager")

    attention._eager_forward_fallback = _reject_eager
    compiled = torch.compile(attention, backend="inductor", fullgraph=True, dynamic=False)
    hidden_states = torch.randn(
        (1, 16, cfg.hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    rel_embeddings = torch.randn(
        (cfg.position_buckets * 2, cfg.hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    output, probs = compiled(
        hidden_states=hidden_states,
        attention_mask=None,
        rel_embeddings=rel_embeddings,
    )
    output.float().square().mean().backward()

    assert probs is None
    assert torch.isfinite(output).all()
    assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
    assert rel_embeddings.grad is not None and torch.isfinite(rel_embeddings.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for real Inductor coverage.")
def test_real_docblock_flash_metadata_values_do_not_recompile() -> None:
    from torch._dynamo.utils import counters

    from deberta.modeling.flashdeberta_attention import (
        FlashDisentangledSelfAttention,
        flashdeberta_docblock_import_error,
    )
    from deberta.modeling.mask_utils import FlashBatchMeta, build_doc_segment_metadata

    if flashdeberta_docblock_import_error() is not None:
        pytest.skip("FlashDeBERTa doc-block kernels are unavailable in this environment.")

    cfg = make_native_deberta_config(
        flash=True,
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=8,
        max_relative_positions=16,
        pos_att_type=["c2p", "p2c"],
        pad_token_id=0,
        position_biased_input=False,
    )
    attention = FlashDisentangledSelfAttention(cfg).to(device="cuda", dtype=torch.bfloat16)

    def _reject_eager(**_kwargs):
        raise AssertionError("compiled doc-block coverage must not fall back to eager")

    attention._eager_forward_fallback = _reject_eager

    def _metadata(doc_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        offsets, lengths, cu_seqlens, active_tokens = build_doc_segment_metadata(doc_ids)
        active_lengths = lengths[lengths.ne(0)]
        return (
            doc_ids.ne(0).to(device="cuda"),
            offsets.to(device="cuda"),
            lengths.to(device="cuda"),
            cu_seqlens.to(device="cuda"),
            torch.tensor(active_tokens, dtype=torch.int32),
            torch.tensor(active_lengths.numel(), dtype=torch.int32),
            active_lengths.max().to(device="cpu"),
        )

    def _forward(
        hidden_states: torch.Tensor,
        rel_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        cu_seqlens: torch.Tensor,
        active_tokens: torch.Tensor,
        num_segments: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        output, _ = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rel_embeddings=rel_embeddings,
            flash_meta=FlashBatchMeta(
                doc_segment_offsets=offsets,
                doc_segment_lengths=lengths,
                doc_cu_seqlens=cu_seqlens,
                active_tokens_scalar=active_tokens,
                doc_num_segments_scalar=num_segments,
                doc_max_segment_length_scalar=max_seqlen,
                route_hint="docblock",
            ),
        )
        return output

    first_meta = _metadata(torch.tensor([[1] * 8 + [2] * 8], dtype=torch.long))
    second_meta = _metadata(torch.tensor([[1] * 4 + [2] * 4 + [3] * 4 + [4] * 4], dtype=torch.long))
    assert [tuple(tensor.shape) for tensor in first_meta] == [tuple(tensor.shape) for tensor in second_meta]

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(_forward, backend="inductor", fullgraph=True, dynamic=False)

    def _run(meta: tuple[torch.Tensor, ...]) -> torch.Tensor:
        hidden_states = torch.randn(
            (1, 16, cfg.hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        rel_embeddings = torch.randn(
            (cfg.position_buckets * 2, cfg.hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        output = compiled(hidden_states, rel_embeddings, *meta)
        output.float().square().mean().backward()
        return output

    first_output = _run(first_meta)
    compiled_graphs = counters["stats"]["unique_graphs"]
    assert compiled_graphs > 0
    second_output = _run(second_meta)

    assert torch.isfinite(first_output).all()
    assert torch.isfinite(second_output).all()
    assert counters["stats"]["unique_graphs"] == compiled_graphs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for real Inductor coverage.")
def test_real_fixed_flash_seq_bucket_crossing_recompiles_and_stays_correct() -> None:
    """A tuning-bucket change must recompile through Dynamo guards, never reuse the old graph.

    The density bucket is resolved host-side and crosses the compile boundary as
    ``FlashBatchMeta.seq_bucket``; the kernel launch config is looked up from it
    inside the custom op. If Dynamo ever stopped guarding on the string, a batch
    in a new bucket would silently run the previous bucket's graph.
    """

    from torch._dynamo.utils import counters

    from deberta.modeling.flashdeberta_attention import (
        FlashDisentangledSelfAttention,
        flashdeberta_fixed_import_error,
    )
    from deberta.modeling.mask_utils import FlashBatchMeta

    if flashdeberta_fixed_import_error() is not None:
        pytest.skip("FlashDeBERTa fixed kernels are unavailable in this environment.")

    cfg = make_native_deberta_config(
        flash=True,
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=8,
        max_relative_positions=16,
        pos_att_type=["c2p", "p2c"],
        pad_token_id=0,
        position_biased_input=False,
    )
    attention = FlashDisentangledSelfAttention(cfg).to(device="cuda", dtype=torch.bfloat16)

    def _reject_eager(**_kwargs):
        raise AssertionError("bucket-crossing coverage must not fall back to eager")

    attention._eager_forward_fallback = _reject_eager

    def _forward(
        hidden_states: torch.Tensor,
        rel_embeddings: torch.Tensor,
        meta: FlashBatchMeta,
    ) -> torch.Tensor:
        output, _ = attention(
            hidden_states=hidden_states,
            attention_mask=None,
            rel_embeddings=rel_embeddings,
            flash_meta=meta,
        )
        return output

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(_forward, backend="inductor", fullgraph=True, dynamic=False)
    base_hidden = torch.randn((1, 16, cfg.hidden_size), device="cuda", dtype=torch.bfloat16)
    base_rel = torch.randn((cfg.position_buckets * 2, cfg.hidden_size), device="cuda", dtype=torch.bfloat16)

    def _run(seq_bucket: str) -> torch.Tensor:
        hidden_states = base_hidden.clone().requires_grad_(True)
        rel_embeddings = base_rel.clone().requires_grad_(True)
        output = compiled(
            hidden_states,
            rel_embeddings,
            FlashBatchMeta(route_hint="fixed", seq_bucket=seq_bucket),
        )
        output.float().square().mean().backward()
        assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
        return output

    first = _run("default")
    graphs_after_first = counters["stats"]["unique_graphs"]
    assert graphs_after_first > 0

    crossed = _run("1024_exact")
    graphs_after_crossing = counters["stats"]["unique_graphs"]
    assert graphs_after_crossing > graphs_after_first

    assert torch.isfinite(first).all()
    # The bucket only selects launch geometry; the math must not move.
    torch.testing.assert_close(crossed.float(), first.float(), rtol=5e-2, atol=5e-2)

    replay = _run("default")
    assert counters["stats"]["unique_graphs"] == graphs_after_crossing
    torch.testing.assert_close(replay.float(), first.float(), rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Flash microbench.")
def test_flash_microbench_runs_flash_mode() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/flashdeberta_microbench.py",
            "--mode",
            "flash",
            "--seq-len",
            "16",
            "--batch-size",
            "1",
            "--warmup",
            "0",
            "--steps",
            "1",
            "--vocab-size",
            "64",
            "--hidden-size",
            "64",
            "--num-layers",
            "1",
            "--num-heads",
            "4",
            "--intermediate-size",
            "128",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "mode=flash" in result.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Flash RTD training.")
def test_flash_rtd_one_step_forward_backward() -> None:
    from deberta.modeling.deberta_v2_native import DebertaV2Model
    from deberta.modeling.flashdeberta_attention import (
        FlashDisentangledSelfAttention,
        flashdeberta_fixed_import_error,
    )
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    if flashdeberta_fixed_import_error() is not None:
        pytest.skip("FlashDeBERTa fixed kernels are unavailable in this environment.")

    cfg = make_native_deberta_config(
        flash=True,
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=8,
        max_relative_positions=16,
        pos_att_type=["c2p", "p2c"],
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        mask_token_id=3,
        position_biased_input=False,
    )
    cfg.hf_attention_impl = "flash"
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=DebertaV2Model(cfg),
        generator_backbone=DebertaV2Model(cfg),
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="none",
    ).to(device="cuda", dtype=torch.bfloat16)

    def _reject_eager(**_kwargs):
        raise AssertionError("Flash RTD smoke must not fall back to eager attention")

    flash_attention_modules = [
        module for module in model.modules() if isinstance(module, FlashDisentangledSelfAttention)
    ]
    assert flash_attention_modules
    for module in flash_attention_modules:
        module._eager_forward_fallback = _reject_eager
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.randint(5, cfg.vocab_size, (1, 16), device="cuda")
    labels = torch.full_like(input_ids, -100)
    labels[:, (3, 7)] = input_ids[:, (3, 7)]
    input_ids[:, (3, 7)] = cfg.mask_token_id

    output = model(
        input_ids=input_ids,
        labels=labels,
        sampling_temperature=1.0,
        gen_loss_weight=1.0,
        disc_loss_weight=50.0,
    )
    output.loss.backward()
    optimizer.step()

    assert torch.isfinite(output.loss)
    assert any(parameter.grad is not None for parameter in model.generator.parameters())
    assert any(parameter.grad is not None for parameter in model.discriminator.parameters())
