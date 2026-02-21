from __future__ import annotations

import sys
import types
import pytest

from deberta.config import ModelConfig
from deberta.modeling import builder as builder_mod


class _DummyTokenizer:
    """Tokenizer stub for backbone-config unit tests."""

    pad_token_id = 0

    def __len__(self) -> int:
        return 128


def test_build_backbone_configs_preserves_loaded_ffn_for_pretrained_rope(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")

    def _fake_from_pretrained(cls, src: str):
        del src
        return cls(ffn_type="mlp", num_hidden_layers=6)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="disc",
        ffn_type="swiglu",
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.ffn_type == "mlp"
    assert gen_cfg.ffn_type == "mlp"


def test_build_backbone_configs_applies_ffn_override_for_scratch_rope(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")

    def _fake_from_pretrained(cls, src: str):
        del src
        return cls(ffn_type="mlp", num_hidden_layers=6)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        discriminator_model_name_or_path="disc",
        ffn_type="swiglu",
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.ffn_type == "swiglu"
    assert gen_cfg.ffn_type == "swiglu"


def test_build_backbone_configs_applies_use_bias_override_for_scratch_rope(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("transformers")

    def _fake_from_pretrained(cls, src: str):
        del src
        return cls(use_bias=True, num_hidden_layers=6)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        discriminator_model_name_or_path="disc",
        use_bias=False,
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.use_bias is False
    assert gen_cfg.use_bias is False


def test_build_backbone_configs_preserves_loaded_use_bias_for_pretrained_rope(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("transformers")

    def _fake_from_pretrained(cls, src: str):
        del src
        return cls(use_bias=True, num_hidden_layers=6)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="custom-rope-checkpoint",
        use_bias=False,
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.use_bias is True
    assert gen_cfg.use_bias is True


def test_build_backbone_configs_adjusts_swiglu_intermediate_for_scratch(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")

    def _fake_from_pretrained(cls, src: str):
        del src
        return cls(ffn_type="mlp", num_hidden_layers=6, intermediate_size=3072)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        discriminator_model_name_or_path="disc",
        ffn_type="swiglu",
        swiglu_adjust_intermediate=True,
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.intermediate_size == 2048
    assert gen_cfg.intermediate_size == 2048


def test_build_backbone_configs_can_disable_swiglu_intermediate_adjustment(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")

    def _fake_from_pretrained(cls, src: str):
        del src
        return cls(ffn_type="mlp", num_hidden_layers=6, intermediate_size=3072)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        discriminator_model_name_or_path="disc",
        ffn_type="swiglu",
        swiglu_adjust_intermediate=False,
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.intermediate_size == 3072
    assert gen_cfg.intermediate_size == 3072


def test_build_backbone_configs_respects_explicit_generator_intermediate_with_swiglu_adjust(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("transformers")

    def _fake_from_pretrained(cls, src: str):
        del src
        return cls(ffn_type="mlp", num_hidden_layers=6, intermediate_size=3072)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        discriminator_model_name_or_path="disc",
        ffn_type="swiglu",
        swiglu_adjust_intermediate=True,
        generator_intermediate_size=1024,
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.intermediate_size == 2048
    assert gen_cfg.intermediate_size == 1024


def test_build_backbone_configs_preserves_explicit_generator_ffn_for_pretrained(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("transformers")

    def _fake_from_pretrained(cls, src: str):
        ffn = "mlp" if "disc" in src else "swiglu"
        return cls(ffn_type=ffn, num_hidden_layers=6)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="disc",
        generator_config_name_or_path="gen",
        ffn_type="mlp",
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.ffn_type == "mlp"
    assert gen_cfg.ffn_type == "swiglu"


def _build_hf_fake_transformers_module() -> types.ModuleType:
    """Build a fake ``transformers`` module for deterministic AutoConfig tests."""

    class _FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, src: str) -> types.SimpleNamespace:
            del src
            return types.SimpleNamespace(
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.2,
                num_hidden_layers=6,
                hidden_size=768,
                intermediate_size=3072,
                num_attention_heads=12,
            )

    module = types.ModuleType("transformers")
    module.AutoConfig = _FakeAutoConfig
    return module


def test_build_backbone_configs_hf_deberta_preserves_checkpoint_dropouts(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")

    fake_transformers = _build_hf_fake_transformers_module()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        discriminator_model_name_or_path="disc",
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.hidden_dropout_prob == pytest.approx(0.1)
    assert disc_cfg.attention_probs_dropout_prob == pytest.approx(0.2)
    assert gen_cfg.hidden_dropout_prob == pytest.approx(0.1)
    assert gen_cfg.attention_probs_dropout_prob == pytest.approx(0.2)


def test_build_backbone_configs_hf_deberta_applies_nonzero_dropout_overrides(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")

    fake_transformers = _build_hf_fake_transformers_module()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        discriminator_model_name_or_path="disc",
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.25,
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.hidden_dropout_prob == pytest.approx(0.5)
    assert disc_cfg.attention_probs_dropout_prob == pytest.approx(0.25)
    assert gen_cfg.hidden_dropout_prob == pytest.approx(0.5)
    assert gen_cfg.attention_probs_dropout_prob == pytest.approx(0.25)


def test_build_backbone_configs_rejects_invalid_model_options_early():
    pytest.importorskip("transformers")

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        norm_arch="not-valid",
    )
    with pytest.raises(ValueError, match="model.norm_arch must be one of"):
        builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=_DummyTokenizer(),
            max_position_embeddings=128,
        )


def test_build_backbone_configs_rejects_hf_deberta_sources_for_pretrained_rope():
    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="microsoft/deberta-v3-base",
    )
    with pytest.raises(ValueError, match="requires DebertaRoPE checkpoints"):
        builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=_DummyTokenizer(),
            max_position_embeddings=128,
        )
