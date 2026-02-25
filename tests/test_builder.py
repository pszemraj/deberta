from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from deberta.config import ModelConfig
from deberta.modeling import builder as builder_mod


class _DummyTokenizer:
    """Tokenizer stub for backbone-config unit tests."""

    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    mask_token_id = 3
    bos_token_id = 4
    eos_token_id = 5

    def __len__(self) -> int:
        return 50265


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


def test_build_backbone_configs_from_scratch_avoids_pretrained_config_load(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("transformers")

    called = {"count": 0}

    def _raise_if_called(cls, src: str):
        del cls
        del src
        called["count"] += 1
        raise AssertionError("from_pretrained should not be called when model.from_scratch=true")

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_raise_if_called),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        discriminator_model_name_or_path="disc",
    )
    _ = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert called["count"] == 0


def test_build_backbone_configs_propagates_tokenizer_special_ids_for_rope():
    pytest.importorskip("transformers")

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        discriminator_model_name_or_path="disc",
    )
    tokenizer = _DummyTokenizer()
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=128,
    )

    for attr in (
        "pad_token_id",
        "cls_token_id",
        "sep_token_id",
        "mask_token_id",
        "bos_token_id",
        "eos_token_id",
    ):
        expected = int(getattr(tokenizer, attr))
        assert int(getattr(disc_cfg, attr)) == expected
        assert int(getattr(gen_cfg, attr)) == expected
    assert bool(getattr(disc_cfg, "use_rmsnorm_heads", False)) is True
    assert bool(getattr(gen_cfg, "use_rmsnorm_heads", False)) is True


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


def test_scaled_swiglu_intermediate_size_rounds_to_multiple_of_128():
    pytest.importorskip("transformers")

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        discriminator_model_name_or_path="disc",
        ffn_type="swiglu",
        intermediate_size=4096,
        swiglu_adjust_intermediate=True,
    )
    disc_cfg, _ = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.intermediate_size == 2816


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
        generator_model_name_or_path="gen",
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.ffn_type == "mlp"
    assert gen_cfg.ffn_type == "swiglu"


@pytest.mark.parametrize(
    (
        "backbone_type",
        "from_scratch",
        "generator_config_name_or_path",
        "generator_model_name_or_path",
        "expected_disc_cfg_src",
        "expected_disc_weight_src",
        "expected_gen_cfg_src",
        "expected_gen_weight_src",
        "expected_gen_derived",
    ),
    [
        ("rope", True, None, None, None, None, None, None, True),
        ("rope", True, "gen_cfg", None, None, None, "gen_cfg", None, False),
        ("rope", False, None, None, "disc", "disc", None, "disc", True),
        ("rope", False, None, "gen_model", "disc", "disc", "gen_model", "gen_model", False),
        (
            "rope",
            False,
            "gen_cfg",
            "gen_model",
            "disc",
            "disc",
            "gen_cfg",
            "gen_model",
            False,
        ),
        ("hf_deberta_v2", True, None, None, "disc", None, None, None, True),
        ("hf_deberta_v2", False, None, None, "disc", "disc", None, "disc", True),
    ],
)
def test_resolve_backbone_sources_matrix(
    backbone_type: str,
    from_scratch: bool,
    generator_config_name_or_path: str | None,
    generator_model_name_or_path: str | None,
    expected_disc_cfg_src: str | None,
    expected_disc_weight_src: str | None,
    expected_gen_cfg_src: str | None,
    expected_gen_weight_src: str | None,
    expected_gen_derived: bool,
):
    cfg = ModelConfig(
        backbone_type=backbone_type,
        from_scratch=from_scratch,
        discriminator_model_name_or_path="disc",
        generator_config_name_or_path=generator_config_name_or_path,
        generator_model_name_or_path=generator_model_name_or_path,
    )
    resolved = builder_mod._resolve_backbone_sources(cfg)

    assert resolved.discriminator.config_source == expected_disc_cfg_src
    assert resolved.discriminator.weight_source == expected_disc_weight_src
    assert resolved.generator.config_source == expected_gen_cfg_src
    assert resolved.generator.weight_source == expected_gen_weight_src
    assert resolved.generator.derived_from_discriminator is expected_gen_derived


def test_validate_model_config_rejects_pretrained_generator_config_without_generator_weights():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="local-rope-disc",
        generator_config_name_or_path="local-rope-gen-config",
        generator_model_name_or_path=None,
    )
    with pytest.raises(ValueError, match="requires model.generator_model_name_or_path"):
        builder_mod.validate_model_config(cfg)


def test_build_backbone_configs_scratch_explicit_generator_config_is_authoritative(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("transformers")

    def _fake_from_pretrained(cls, src: str):
        if src == "gen_cfg":
            return cls(
                vocab_size=50265,
                hidden_size=384,
                num_hidden_layers=4,
                num_attention_heads=6,
                intermediate_size=1536,
                ffn_type="mlp",
                max_position_embeddings=256,
            )
        raise AssertionError(f"unexpected config source: {src}")

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        discriminator_model_name_or_path="disc",
        generator_config_name_or_path="gen_cfg",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        ffn_type="swiglu",
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.hidden_size == 768
    assert disc_cfg.ffn_type == "swiglu"
    assert gen_cfg.hidden_size == 384
    assert gen_cfg.num_hidden_layers == 4
    assert gen_cfg.intermediate_size == 1536
    assert gen_cfg.ffn_type == "mlp"
    assert gen_cfg.max_position_embeddings == 256


def test_build_backbone_configs_rejects_pretrained_rope_vocab_mismatch(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")

    def _fake_from_pretrained(cls, src: str):
        del src
        return cls(vocab_size=1234, num_hidden_layers=6)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="local-rope-disc",
    )
    with pytest.raises(ValueError, match="Tokenizer/checkpoint vocab mismatch for discriminator"):
        _ = builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=_DummyTokenizer(),
            max_position_embeddings=128,
        )


def test_build_backbone_configs_rejects_pretrained_rope_special_id_mismatch(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")

    def _fake_from_pretrained(cls, src: str):
        del src
        return cls(vocab_size=50265, cls_token_id=999, num_hidden_layers=6)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="local-rope-disc",
    )
    with pytest.raises(ValueError, match="Tokenizer/config special-token mismatch"):
        _ = builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=_DummyTokenizer(),
            max_position_embeddings=128,
        )


def test_build_backbone_configs_applies_explicit_pretrained_rope_overrides(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("transformers")

    def _fake_from_pretrained(cls, src: str):
        del src
        return cls(
            vocab_size=50265,
            num_hidden_layers=6,
            rope_theta=10000.0,
            rotary_pct=1.0,
            norm_arch="post",
            ffn_type="swiglu",
            use_bias=False,
        )

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="local-rope-disc",
        pretrained_rope_theta=50_000.0,
        pretrained_rotary_pct=0.5,
        pretrained_norm_arch="keel",
        pretrained_ffn_type="mlp",
        pretrained_use_bias=True,
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    for cfg in (disc_cfg, gen_cfg):
        assert cfg.rope_theta == pytest.approx(50_000.0)
        assert cfg.rotary_pct == pytest.approx(0.5)
        assert cfg.norm_arch == "keel"
        assert cfg.ffn_type == "mlp"
        assert cfg.use_bias is True


def _build_hf_fake_transformers_module() -> types.ModuleType:
    """Build a fake ``transformers`` module for deterministic AutoConfig tests."""

    class _FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, src: str) -> types.SimpleNamespace:
            del src
            return types.SimpleNamespace(
                vocab_size=50265,
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


_USE_MODEL_DEFAULT = object()


@pytest.mark.parametrize(
    (
        "from_scratch",
        "hidden_dropout_prob",
        "attention_probs_dropout_prob",
        "expected_hidden",
        "expected_attn",
    ),
    [
        (False, _USE_MODEL_DEFAULT, _USE_MODEL_DEFAULT, 0.1, 0.2),
        (True, _USE_MODEL_DEFAULT, _USE_MODEL_DEFAULT, 0.1, 0.2),
        (False, None, None, 0.1, 0.2),
        (True, None, None, 0.1, 0.2),
        (False, 0.5, 0.25, 0.5, 0.25),
        (True, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_build_backbone_configs_hf_deberta_dropout_overrides(
    monkeypatch: pytest.MonkeyPatch,
    from_scratch: bool,
    hidden_dropout_prob: float | None | object,
    attention_probs_dropout_prob: float | None | object,
    expected_hidden: float,
    expected_attn: float,
):
    pytest.importorskip("transformers")

    fake_transformers = _build_hf_fake_transformers_module()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model_kwargs: dict[str, object] = dict(
        backbone_type="hf_deberta_v2",
        from_scratch=from_scratch,
        discriminator_model_name_or_path="disc",
    )
    if hidden_dropout_prob is not _USE_MODEL_DEFAULT:
        model_kwargs["hidden_dropout_prob"] = hidden_dropout_prob
    if attention_probs_dropout_prob is not _USE_MODEL_DEFAULT:
        model_kwargs["attention_probs_dropout_prob"] = attention_probs_dropout_prob

    model_cfg = ModelConfig(**model_kwargs)
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_DummyTokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.hidden_dropout_prob == pytest.approx(expected_hidden)
    assert disc_cfg.attention_probs_dropout_prob == pytest.approx(expected_attn)
    assert gen_cfg.hidden_dropout_prob == pytest.approx(expected_hidden)
    assert gen_cfg.attention_probs_dropout_prob == pytest.approx(expected_attn)


def test_build_backbone_configs_rejects_pretrained_hf_vocab_mismatch(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")

    class _FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, src: str) -> types.SimpleNamespace:
            del src
            return types.SimpleNamespace(
                vocab_size=777,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.2,
                num_hidden_layers=6,
                hidden_size=768,
                intermediate_size=3072,
                num_attention_heads=12,
            )

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = _FakeAutoConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        discriminator_model_name_or_path="disc",
    )
    with pytest.raises(ValueError, match="Tokenizer/checkpoint vocab mismatch for discriminator"):
        _ = builder_mod.build_backbone_configs(
            model_cfg=cfg,
            tokenizer=_DummyTokenizer(),
            max_position_embeddings=128,
        )


def test_build_backbone_configs_rejects_pretrained_hf_special_id_mismatch(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")

    class _FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, src: str) -> types.SimpleNamespace:
            del src
            return types.SimpleNamespace(
                vocab_size=50265,
                cls_token_id=404,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.2,
                num_hidden_layers=6,
                hidden_size=768,
                intermediate_size=3072,
                num_attention_heads=12,
            )

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = _FakeAutoConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        discriminator_model_name_or_path="disc",
    )
    with pytest.raises(ValueError, match="Tokenizer/config special-token mismatch"):
        _ = builder_mod.build_backbone_configs(
            model_cfg=cfg,
            tokenizer=_DummyTokenizer(),
            max_position_embeddings=128,
        )


def test_build_backbones_uses_resolved_rope_weight_sources(monkeypatch: pytest.MonkeyPatch):
    called: list[tuple[str, int]] = []

    def _fake_from_pretrained(cls, src: str, config: Any):
        called.append((src, int(config.hidden_size)))
        return object()

    monkeypatch.setattr(
        builder_mod.DebertaRoPEModel,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="disc_weights",
        generator_model_name_or_path="gen_weights",
    )
    disc_cfg = builder_mod.DebertaRoPEConfig(hidden_size=768, num_hidden_layers=2)
    gen_cfg = builder_mod.DebertaRoPEConfig(hidden_size=384, num_hidden_layers=1)
    _ = builder_mod.build_backbones(model_cfg=model_cfg, disc_config=disc_cfg, gen_config=gen_cfg)

    assert called == [("disc_weights", 768), ("gen_weights", 384)]


def test_build_backbones_uses_discriminator_fallback_for_derived_pretrained_rope(
    monkeypatch: pytest.MonkeyPatch,
):
    called: list[str] = []

    def _fake_from_pretrained(cls, src: str, config: Any):
        del config
        called.append(src)
        return object()

    monkeypatch.setattr(
        builder_mod.DebertaRoPEModel,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="disc_weights",
        generator_model_name_or_path=None,
    )
    disc_cfg = builder_mod.DebertaRoPEConfig(hidden_size=768, num_hidden_layers=2)
    gen_cfg = builder_mod.DebertaRoPEConfig(hidden_size=768, num_hidden_layers=1)
    _ = builder_mod.build_backbones(model_cfg=model_cfg, disc_config=disc_cfg, gen_config=gen_cfg)

    assert called == ["disc_weights", "disc_weights"]


def test_build_backbones_uses_resolved_hf_weight_sources(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")

    called: list[tuple[str, Any]] = []

    def _fake_from_pretrained(cls, src: str, config: Any):
        del cls
        called.append((src, config))
        return object()

    monkeypatch.setattr(
        builder_mod.DebertaV2Model,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        discriminator_model_name_or_path="disc_weights",
        generator_model_name_or_path="gen_weights",
    )
    disc_cfg = types.SimpleNamespace(hidden_size=768)
    gen_cfg = types.SimpleNamespace(hidden_size=384)
    _ = builder_mod.build_backbones(model_cfg=model_cfg, disc_config=disc_cfg, gen_config=gen_cfg)

    assert called == [("disc_weights", disc_cfg), ("gen_weights", gen_cfg)]


def test_build_backbones_hf_from_scratch_uses_native_implementation():
    pytest.importorskip("transformers")
    from transformers import DebertaV2Config

    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=True,
        discriminator_model_name_or_path="disc",
    )
    disc_cfg = DebertaV2Config(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
    )
    gen_cfg = DebertaV2Config(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
    )

    disc, gen = builder_mod.build_backbones(model_cfg=model_cfg, disc_config=disc_cfg, gen_config=gen_cfg)

    assert isinstance(disc, builder_mod.DebertaV2Model)
    assert isinstance(gen, builder_mod.DebertaV2Model)


def test_build_backbones_uses_discriminator_fallback_for_derived_pretrained_hf(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("transformers")

    called: list[str] = []

    def _fake_from_pretrained(cls, src: str, config: Any):
        del cls
        del config
        called.append(src)
        return object()

    monkeypatch.setattr(
        builder_mod.DebertaV2Model,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        discriminator_model_name_or_path="disc_weights",
        generator_model_name_or_path=None,
    )
    disc_cfg = types.SimpleNamespace(hidden_size=768)
    gen_cfg = types.SimpleNamespace(hidden_size=384)
    _ = builder_mod.build_backbones(model_cfg=model_cfg, disc_config=disc_cfg, gen_config=gen_cfg)

    assert called == ["disc_weights", "disc_weights"]


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
