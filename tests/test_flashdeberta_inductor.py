from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for real Inductor coverage.")
def test_real_fixed_flash_attention_compiles_with_inductor_and_backward() -> None:
    """The production fixed custom op must remain opaque to real Inductor."""

    from deberta.modeling.deberta_v2_native import DebertaV2Config
    from deberta.modeling.flashdeberta_attention import (
        FlashDisentangledSelfAttention,
        flashdeberta_fixed_import_error,
    )

    if flashdeberta_fixed_import_error() is not None:
        pytest.skip("FlashDeBERTa fixed kernels are unavailable in this environment.")

    cfg = DebertaV2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=8,
        max_relative_positions=16,
        pos_att_type=["c2p", "p2c"],
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
        position_biased_input=False,
    )
    cfg.hf_flash = {}
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
