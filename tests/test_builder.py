from __future__ import annotations

import types
from typing import Any

import pytest
import torch
from _fakes import DummyTokenizer

from deberta.config import ModelConfig
from deberta.modeling import builder as builder_mod


@pytest.fixture(autouse=True)
def _default_pretrained_config_stub(monkeypatch: pytest.MonkeyPatch):
    """Default ``from_pretrained`` stub for RoPE config loads in builder tests."""
    pytest.importorskip("transformers")

    def _default_from_pretrained(cls, src: str):
        del src
        return cls(ffn_type="mlp", num_hidden_layers=6, intermediate_size=3072)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_default_from_pretrained),
    )


@pytest.mark.parametrize(
    ("loaded_kwargs", "model_kwargs", "expected"),
    [
        (
            {"ffn_type": "mlp", "num_hidden_layers": 6},
            {"from_scratch": False, "pretrained_discriminator_path": "disc"},
            {"ffn_type": "mlp"},
        ),
        (
            {"ffn_type": "mlp", "num_hidden_layers": 6},
            {"from_scratch": True, "pretrained_discriminator_path": "disc", "ffn_type": "swiglu"},
            {"ffn_type": "swiglu"},
        ),
        (
            {"use_bias": True, "num_hidden_layers": 6},
            {"from_scratch": True, "pretrained_discriminator_path": "disc", "use_bias": False},
            {"use_bias": False},
        ),
        (
            {"use_bias": True, "num_hidden_layers": 6},
            {
                "from_scratch": False,
                "pretrained_discriminator_path": "custom-rope-checkpoint",
                "use_bias": False,
            },
            {"use_bias": True},
        ),
        (
            {"ffn_type": "mlp", "num_hidden_layers": 6, "intermediate_size": 3072},
            {
                "from_scratch": True,
                "pretrained_discriminator_path": "disc",
                "ffn_type": "swiglu",
                "swiglu_adjust_intermediate": True,
            },
            {"intermediate_size": 2048},
        ),
    ],
)
def test_build_backbone_configs_rope_overrides(
    monkeypatch: pytest.MonkeyPatch,
    loaded_kwargs: dict[str, Any],
    model_kwargs: dict[str, Any],
    expected: dict[str, Any],
):
    pytest.importorskip("transformers")

    def _fake_from_pretrained(cls, src: str):
        del src
        return cls(**loaded_kwargs)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = ModelConfig(backbone_type="rope", **model_kwargs)
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=50265),
        max_position_embeddings=128,
    )

    for key, value in expected.items():
        assert getattr(disc_cfg, key) == value
        assert getattr(gen_cfg, key) == value


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
        pretrained_discriminator_path="disc",
    )
    _ = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=50265),
        max_position_embeddings=128,
    )

    assert called["count"] == 0


@pytest.mark.parametrize(
    ("from_scratch", "pretrained_generator_path", "tokenizer_vocab"),
    [
        (True, None, 50265),
        (False, None, 128100),
        (False, "gen_weights", 128100),
    ],
)
def test_build_backbone_configs_hf_deberta_never_loads_external_model_config(
    monkeypatch: pytest.MonkeyPatch,
    from_scratch: bool,
    pretrained_generator_path: str | None,
    tokenizer_vocab: int,
):
    pytest.importorskip("transformers")

    rope_called = {"count": 0}
    hf_cfg_called = {"count": 0}
    repo_called = {"count": 0}
    original_repo_builder = builder_mod._build_repo_hf_deberta_v2_config

    def _raise_rope_cfg_pretrained(cls, src: str):
        del cls
        del src
        rope_called["count"] += 1
        raise AssertionError("RoPE config source must not be touched for hf_deberta_v2 backbone.")

    def _raise_hf_cfg_pretrained(cls, src: str, **kwargs: Any):
        del cls
        del src
        del kwargs
        hf_cfg_called["count"] += 1
        raise AssertionError("HF config source must not be loaded; config must be repo-synthesized.")

    def _count_repo_builder(*, model_cfg: ModelConfig):
        repo_called["count"] += 1
        return original_repo_builder(model_cfg=model_cfg)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_raise_rope_cfg_pretrained),
    )
    monkeypatch.setattr(
        builder_mod.DebertaV2Config,
        "from_pretrained",
        classmethod(_raise_hf_cfg_pretrained),
    )
    monkeypatch.setattr(builder_mod, "_build_repo_hf_deberta_v2_config", _count_repo_builder)

    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=from_scratch,
        pretrained_discriminator_path=("disc_weights" if not from_scratch else ""),
        pretrained_generator_path=pretrained_generator_path,
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=tokenizer_vocab),
        max_position_embeddings=128,
    )

    assert int(repo_called["count"]) == 1
    assert int(rope_called["count"]) == 0
    assert int(hf_cfg_called["count"]) == 0
    assert int(disc_cfg.hidden_size) > 0
    assert int(gen_cfg.hidden_size) > 0


def test_build_backbone_configs_propagates_tokenizer_special_ids_for_rope():
    pytest.importorskip("transformers")

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        pretrained_discriminator_path="disc",
    )
    tokenizer = DummyTokenizer(vocab_size=50265)
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
        pretrained_discriminator_path="disc",
        ffn_type="swiglu",
        swiglu_adjust_intermediate=False,
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=50265),
        max_position_embeddings=128,
    )

    assert disc_cfg.intermediate_size == 3072
    assert gen_cfg.intermediate_size == 3072


def test_scaled_swiglu_intermediate_size_rounds_to_multiple_of_128():
    pytest.importorskip("transformers")

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        pretrained_discriminator_path="disc",
        ffn_type="swiglu",
        intermediate_size=4096,
        swiglu_adjust_intermediate=True,
    )
    disc_cfg, _ = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=50265),
        max_position_embeddings=128,
    )

    assert disc_cfg.intermediate_size == 2816


def test_derive_generator_config_uses_half_depth_for_hf_backbone():
    base_cfg = types.SimpleNamespace(num_hidden_layers=12)
    model_cfg = ModelConfig(
        profile="modern",
        backbone_type="hf_deberta_v2",
        generator_num_hidden_layers=None,
    )

    gen_cfg = builder_mod._derive_generator_config(base_cfg, model_cfg)
    assert int(gen_cfg.num_hidden_layers) == 6


def test_derive_generator_config_keeps_third_depth_default_for_non_hf_backbone():
    base_cfg = types.SimpleNamespace(num_hidden_layers=12)
    model_cfg = ModelConfig(
        profile="modern",
        backbone_type="rope",
        generator_num_hidden_layers=None,
    )

    gen_cfg = builder_mod._derive_generator_config(base_cfg, model_cfg)
    assert int(gen_cfg.num_hidden_layers) == 4


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
        pretrained_discriminator_path="disc",
        ffn_type="swiglu",
        swiglu_adjust_intermediate=True,
        generator_intermediate_size=1024,
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=50265),
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
        pretrained_discriminator_path="disc",
        pretrained_generator_path="gen",
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=50265),
        max_position_embeddings=128,
    )

    assert disc_cfg.ffn_type == "mlp"
    assert gen_cfg.ffn_type == "swiglu"


@pytest.mark.parametrize(
    (
        "backbone_type",
        "from_scratch",
        "pretrained_generator_path",
        "expected_disc_cfg_src",
        "expected_disc_weight_src",
        "expected_gen_cfg_src",
        "expected_gen_weight_src",
        "expected_gen_derived",
    ),
    [
        ("rope", True, None, None, None, None, None, True),
        ("rope", True, "gen_model", None, None, "gen_model", None, False),
        ("rope", False, None, "disc", "disc", None, "disc", True),
        ("rope", False, "gen_model", "disc", "disc", "gen_model", "gen_model", False),
        ("hf_deberta_v2", True, None, None, None, None, None, True),
        ("hf_deberta_v2", False, None, None, "disc", None, "disc", True),
        ("hf_deberta_v2", False, "gen_model", None, "disc", None, "gen_model", False),
    ],
)
def test_resolve_backbone_sources_matrix(
    backbone_type: str,
    from_scratch: bool,
    pretrained_generator_path: str | None,
    expected_disc_cfg_src: str | None,
    expected_disc_weight_src: str | None,
    expected_gen_cfg_src: str | None,
    expected_gen_weight_src: str | None,
    expected_gen_derived: bool,
):
    cfg = ModelConfig(
        backbone_type=backbone_type,
        from_scratch=from_scratch,
        pretrained_discriminator_path="disc",
        pretrained_generator_path=pretrained_generator_path,
    )
    resolved = builder_mod._resolve_backbone_sources(cfg)

    assert resolved.discriminator.config_source == expected_disc_cfg_src
    assert resolved.discriminator.weight_source == expected_disc_weight_src
    assert resolved.generator.config_source == expected_gen_cfg_src
    assert resolved.generator.weight_source == expected_gen_weight_src
    assert resolved.generator.derived_from_discriminator is expected_gen_derived


def test_validate_model_config_rejects_hf_max_position_embeddings_in_pretrained_mode():
    cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        pretrained_discriminator_path="microsoft/deberta-v3-base",
        hf_max_position_embeddings=1024,
    )
    with pytest.raises(ValueError, match="only supported when model.from_scratch=true"):
        builder_mod.validate_model_config(cfg)


def test_build_backbone_configs_scratch_explicit_generator_model_is_authoritative(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("transformers")

    def _fake_from_pretrained(cls, src: str):
        if src == "gen_model":
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
        pretrained_discriminator_path="disc",
        pretrained_generator_path="gen_model",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        ffn_type="swiglu",
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=50265),
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
        pretrained_discriminator_path="local-rope-disc",
    )
    with pytest.raises(ValueError, match="Tokenizer/checkpoint vocab mismatch for discriminator"):
        _ = builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=DummyTokenizer(vocab_size=50265),
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
        pretrained_discriminator_path="local-rope-disc",
    )
    with pytest.raises(ValueError, match="Tokenizer/config special-token mismatch"):
        _ = builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=DummyTokenizer(vocab_size=50265),
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
        pretrained_discriminator_path="local-rope-disc",
        pretrained_rope_theta=50_000.0,
        pretrained_rotary_pct=0.5,
        pretrained_norm_arch="keel",
        pretrained_ffn_type="mlp",
        pretrained_use_bias=True,
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=50265),
        max_position_embeddings=128,
    )

    for cfg in (disc_cfg, gen_cfg):
        assert cfg.rope_theta == pytest.approx(50_000.0)
        assert cfg.rotary_pct == pytest.approx(0.5)
        assert cfg.norm_arch == "keel"
        assert cfg.ffn_type == "mlp"
        assert cfg.use_bias is True


def _patch_repo_hf_base_config(
    monkeypatch: pytest.MonkeyPatch, *, config_overrides: dict[str, object] | None = None
) -> None:
    """Patch repo HF base config builder for deterministic tests."""
    overrides = dict(config_overrides or {})
    original = builder_mod._build_repo_hf_deberta_v2_config

    def _factory(*, model_cfg: ModelConfig) -> Any:
        cfg = original(model_cfg=model_cfg)
        for key, value in overrides.items():
            setattr(cfg, str(key), value)
        return cfg

    monkeypatch.setattr(builder_mod, "_build_repo_hf_deberta_v2_config", _factory)


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
        (False, _USE_MODEL_DEFAULT, _USE_MODEL_DEFAULT, 0.0, 0.0),
        (True, _USE_MODEL_DEFAULT, _USE_MODEL_DEFAULT, 0.0, 0.0),
        (False, None, None, 0.1, 0.1),
        (True, None, None, 0.1, 0.1),
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

    model_kwargs: dict[str, object] = dict(
        backbone_type="hf_deberta_v2",
        from_scratch=from_scratch,
        pretrained_discriminator_path="disc",
    )
    if hidden_dropout_prob is not _USE_MODEL_DEFAULT:
        model_kwargs["hidden_dropout_prob"] = hidden_dropout_prob
    if attention_probs_dropout_prob is not _USE_MODEL_DEFAULT:
        model_kwargs["attention_probs_dropout_prob"] = attention_probs_dropout_prob

    model_cfg = ModelConfig(**model_kwargs)
    tokenizer_vocab = 128100 if not from_scratch else 50265
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=tokenizer_vocab),
        max_position_embeddings=128,
    )

    assert disc_cfg.hidden_dropout_prob == pytest.approx(expected_hidden)
    assert disc_cfg.attention_probs_dropout_prob == pytest.approx(expected_attn)
    assert gen_cfg.hidden_dropout_prob == pytest.approx(expected_hidden)
    assert gen_cfg.attention_probs_dropout_prob == pytest.approx(expected_attn)


@pytest.mark.parametrize(
    ("hf_model_size", "hidden_size", "layers", "heads", "intermediate", "gen_z_steps"),
    [
        ("xsmall", 384, 12, 6, 1536, 2),
        ("small", 768, 6, 12, 3072, 0),
        ("base", 768, 12, 12, 3072, 0),
        ("large", 1024, 24, 16, 4096, 0),
    ],
)
def test_build_backbone_configs_hf_deberta_uses_repo_architecture_presets(
    hf_model_size: str,
    hidden_size: int,
    layers: int,
    heads: int,
    intermediate: int,
    gen_z_steps: int,
):
    pytest.importorskip("transformers")

    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=True,
        hf_model_size=hf_model_size,
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=50265),
        max_position_embeddings=128,
    )

    assert int(disc_cfg.hidden_size) == hidden_size
    assert int(disc_cfg.num_hidden_layers) == layers
    assert int(disc_cfg.num_attention_heads) == heads
    assert int(disc_cfg.intermediate_size) == intermediate
    assert int(disc_cfg.hidden_size) // int(disc_cfg.num_attention_heads) == 64
    assert int(gen_cfg.num_hidden_layers) == max(1, layers // 2)
    assert int(gen_cfg.hidden_size) == hidden_size
    assert int(gen_cfg.num_attention_heads) == heads
    assert int(gen_cfg.intermediate_size) == intermediate
    assert int(getattr(disc_cfg, "z_steps", 0)) == 0
    assert int(getattr(gen_cfg, "z_steps", 0)) == gen_z_steps


def test_build_backbone_configs_scratch_can_pad_tokenizer_vocab_to_multiple():
    pytest.importorskip("transformers")

    tokenizer = DummyTokenizer(vocab_size=500)
    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        tokenizer_allow_vocab_resize=True,
        tokenizer_vocab_multiple=128,
    )

    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=128,
    )

    assert len(tokenizer) == 512
    assert int(disc_cfg.vocab_size) == 512
    assert int(gen_cfg.vocab_size) == 512


def test_build_backbone_configs_scratch_rejects_vocab_growth_without_resize_permission():
    pytest.importorskip("transformers")

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        tokenizer_allow_vocab_resize=False,
        tokenizer_vocab_target=640,
    )
    with pytest.raises(ValueError, match="tokenizer_allow_vocab_resize=false"):
        _ = builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=DummyTokenizer(vocab_size=500),
            max_position_embeddings=128,
        )


def test_build_backbone_configs_pretrained_hf_can_auto_grow_tokenizer_to_config_vocab(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("transformers")
    _patch_repo_hf_base_config(monkeypatch, config_overrides={"vocab_size": 512})

    tokenizer = DummyTokenizer(vocab_size=500)
    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        pretrained_discriminator_path="disc",
        tokenizer_allow_vocab_resize=True,
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=128,
    )

    assert len(tokenizer) == 512
    assert int(disc_cfg.vocab_size) == 512
    assert int(gen_cfg.vocab_size) == 512


def test_build_backbone_configs_pretrained_hf_rejects_vocab_multiple_if_it_exceeds_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("transformers")
    _patch_repo_hf_base_config(monkeypatch, config_overrides={"vocab_size": 500})

    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        pretrained_discriminator_path="disc",
        tokenizer_allow_vocab_resize=True,
        tokenizer_vocab_multiple=128,
    )
    with pytest.raises(ValueError, match="tokenizer_vocab_multiple"):
        _ = builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=DummyTokenizer(vocab_size=489),
            max_position_embeddings=128,
        )


def test_build_backbone_configs_rejects_pretrained_hf_vocab_mismatch(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")
    _patch_repo_hf_base_config(monkeypatch, config_overrides={"vocab_size": 777})

    cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        pretrained_discriminator_path="disc",
    )
    with pytest.raises(ValueError, match="Tokenizer/checkpoint vocab mismatch for discriminator"):
        _ = builder_mod.build_backbone_configs(
            model_cfg=cfg,
            tokenizer=DummyTokenizer(vocab_size=50265),
            max_position_embeddings=128,
        )


def test_build_backbone_configs_rejects_pretrained_hf_special_id_mismatch(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")
    _patch_repo_hf_base_config(monkeypatch, config_overrides={"cls_token_id": 404})

    cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        pretrained_discriminator_path="disc",
    )
    with pytest.raises(ValueError, match="Tokenizer/config special-token mismatch"):
        _ = builder_mod.build_backbone_configs(
            model_cfg=cfg,
            tokenizer=DummyTokenizer(vocab_size=128100),
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
        pretrained_discriminator_path="disc_weights",
        pretrained_generator_path="gen_weights",
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
        pretrained_discriminator_path="disc_weights",
        pretrained_generator_path=None,
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
        pretrained_discriminator_path="disc_weights",
        pretrained_generator_path="gen_weights",
    )
    disc_cfg = types.SimpleNamespace(hidden_size=768)
    gen_cfg = types.SimpleNamespace(hidden_size=384)
    _ = builder_mod.build_backbones(model_cfg=model_cfg, disc_config=disc_cfg, gen_config=gen_cfg)

    assert called == [("disc_weights", disc_cfg), ("gen_weights", gen_cfg)]


def test_build_backbones_pretrained_hf_does_not_fetch_external_config(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")

    model_calls: list[str] = []
    cfg_calls = {"count": 0}

    def _fake_model_from_pretrained(cls, src: str, config: Any):
        del cls
        del config
        model_calls.append(src)
        return object()

    def _raise_cfg_from_pretrained(cls, src: str, **kwargs: Any):
        del cls
        del src
        del kwargs
        cfg_calls["count"] += 1
        raise AssertionError(
            "DebertaV2Config.from_pretrained must not be called when build_backbones receives explicit configs."
        )

    monkeypatch.setattr(
        builder_mod.DebertaV2Model,
        "from_pretrained",
        classmethod(_fake_model_from_pretrained),
    )
    monkeypatch.setattr(
        builder_mod.DebertaV2Config,
        "from_pretrained",
        classmethod(_raise_cfg_from_pretrained),
    )

    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        pretrained_discriminator_path="disc_weights",
        pretrained_generator_path="gen_weights",
    )
    disc_cfg = types.SimpleNamespace(hidden_size=768, vocab_size=128100)
    gen_cfg = types.SimpleNamespace(hidden_size=384, vocab_size=128100)
    _ = builder_mod.build_backbones(model_cfg=model_cfg, disc_config=disc_cfg, gen_config=gen_cfg)

    assert model_calls == ["disc_weights", "gen_weights"]
    assert int(cfg_calls["count"]) == 0


def test_build_backbones_hf_from_scratch_uses_native_implementation():
    pytest.importorskip("transformers")
    from transformers import DebertaV2Config

    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=True,
        pretrained_discriminator_path="disc",
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
        pretrained_discriminator_path="disc_weights",
        pretrained_generator_path=None,
    )
    disc_cfg = types.SimpleNamespace(hidden_size=768)
    gen_cfg = types.SimpleNamespace(hidden_size=384)
    _ = builder_mod.build_backbones(model_cfg=model_cfg, disc_config=disc_cfg, gen_config=gen_cfg)

    assert called == ["disc_weights", "disc_weights"]


def test_build_backbones_pretrained_rope_can_skip_pretrained_weight_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    called = {"count": 0}

    def _raise_if_called(cls, src: str, config: Any):
        del cls
        del src
        del config
        called["count"] += 1
        raise AssertionError("from_pretrained should not be called when load_pretrained_weights=false")

    monkeypatch.setattr(
        builder_mod.DebertaRoPEModel,
        "from_pretrained",
        classmethod(_raise_if_called),
    )

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        pretrained_discriminator_path="disc_weights",
    )
    disc_cfg = builder_mod.DebertaRoPEConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
    )
    gen_cfg = builder_mod.DebertaRoPEConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
    )
    disc, gen = builder_mod.build_backbones(
        model_cfg=model_cfg,
        disc_config=disc_cfg,
        gen_config=gen_cfg,
        load_pretrained_weights=False,
    )

    assert isinstance(disc, builder_mod.DebertaRoPEModel)
    assert isinstance(gen, builder_mod.DebertaRoPEModel)
    assert called["count"] == 0


def test_build_backbones_pretrained_hf_can_skip_pretrained_weight_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("transformers")
    from transformers import DebertaV2Config

    called = {"count": 0}

    def _raise_if_called(cls, src: str, config: Any):
        del cls
        del src
        del config
        called["count"] += 1
        raise AssertionError("from_pretrained should not be called when load_pretrained_weights=false")

    monkeypatch.setattr(
        builder_mod.DebertaV2Model,
        "from_pretrained",
        classmethod(_raise_if_called),
    )

    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        pretrained_discriminator_path="disc_weights",
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
    disc, gen = builder_mod.build_backbones(
        model_cfg=model_cfg,
        disc_config=disc_cfg,
        gen_config=gen_cfg,
        load_pretrained_weights=False,
    )

    assert isinstance(disc, builder_mod.DebertaV2Model)
    assert isinstance(gen, builder_mod.DebertaV2Model)
    assert called["count"] == 0


def test_native_deberta_v2_model_honors_config_z_steps():
    pytest.importorskip("transformers")
    from transformers import DebertaV2Config

    cfg = DebertaV2Config(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        z_steps=2,
    )
    model = builder_mod.DebertaV2Model(cfg)
    assert int(model.z_steps) == 2

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
    assert out.hidden_states is not None
    # num_layers + embeddings + (z_steps - 1) extra last-layer passes
    assert len(out.hidden_states) == 4


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
            tokenizer=DummyTokenizer(vocab_size=50265),
            max_position_embeddings=128,
        )


def test_build_backbone_configs_rejects_hf_deberta_sources_for_pretrained_rope():
    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        pretrained_discriminator_path="microsoft/deberta-v3-base",
    )
    with pytest.raises(ValueError, match="requires DebertaRoPE checkpoints"):
        builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=DummyTokenizer(vocab_size=50265),
            max_position_embeddings=128,
        )
