from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from _config_factories import make_model_config
from _fakes import BackboneConfigStub, DummyTokenizer

from deberta.config import ModelConfig, load_config
from deberta.modeling import builder as builder_mod


@pytest.fixture(autouse=True)
def _default_pretrained_config_stub(monkeypatch: pytest.MonkeyPatch):
    """Default ``from_pretrained`` stub for RoPE config loads in builder tests."""

    def _default_from_pretrained(cls, src: str):
        del src
        return cls(ffn_type="mlp", num_hidden_layers=6, intermediate_size=3072)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_default_from_pretrained),
    )


def _tiny_backbone_pair(backbone_type: str) -> tuple[type[torch.nn.Module], Any, Any]:
    """Return the model class and tiny discriminator/generator configs for one backbone."""

    if backbone_type == "rope":
        model_cls = builder_mod.DebertaRoPEModel
        config_cls = builder_mod.DebertaRoPEConfig
    else:
        model_cls = builder_mod.DebertaV2Model
        config_cls = builder_mod.DebertaV2Config
    common = {
        "vocab_size": 128,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "intermediate_size": 64,
    }
    return (
        model_cls,
        config_cls(**common, num_hidden_layers=2),
        config_cls(**common, num_hidden_layers=1),
    )


@pytest.mark.parametrize(
    ("loaded_kwargs", "model_kwargs", "expected"),
    [
        (
            {"ffn_type": "mlp", "num_hidden_layers": 6},
            {"from_scratch": False, "pretrained": {"discriminator_path": "disc"}},
            {"ffn_type": "mlp"},
        ),
        (
            {"ffn_type": "mlp", "num_hidden_layers": 6},
            {
                "from_scratch": True,
                "pretrained": {"discriminator_path": "disc"},
                "rope": {"ffn_type": "swiglu"},
            },
            {"ffn_type": "swiglu"},
        ),
        (
            {"use_bias": True, "num_hidden_layers": 6},
            {
                "from_scratch": True,
                "pretrained": {"discriminator_path": "disc"},
                "rope": {"use_bias": False},
            },
            {"use_bias": False},
        ),
        (
            {"use_bias": True, "num_hidden_layers": 6},
            {
                "from_scratch": False,
                "pretrained": {"discriminator_path": "custom-rope-checkpoint"},
                "rope": {"use_bias": False},
            },
            {"use_bias": True},
        ),
        (
            {"ffn_type": "mlp", "num_hidden_layers": 6, "intermediate_size": 3072},
            {
                "from_scratch": True,
                "pretrained": {"discriminator_path": "disc"},
                "rope": {"ffn_type": "swiglu", "swiglu_adjust_intermediate": True},
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

    def _fake_from_pretrained(cls, src: str):
        del src
        return cls(**loaded_kwargs)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = make_model_config(backbone_type="rope", **model_kwargs)
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

    model_cfg = make_model_config(
        backbone_type="rope", from_scratch=True, pretrained={"discriminator_path": "disc"}
    )
    _ = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=50265),
        max_position_embeddings=128,
    )

    assert called["count"] == 0


@pytest.mark.parametrize(
    (
        "from_scratch",
        "pretrained_generator_path",
        "tokenizer_vocab",
        "expected_repo_calls",
        "expected_hf_cfg_sources",
    ),
    [
        (True, None, 50265, 1, []),
        (False, None, 128100, 0, ["disc_weights"]),
        (False, "gen_weights", 128100, 0, ["disc_weights", "gen_weights"]),
        (True, "gen_weights", 50265, 1, ["gen_weights"]),
    ],
)
def test_build_backbone_configs_hf_deberta_loads_generator_config_only_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
    from_scratch: bool,
    pretrained_generator_path: str | None,
    tokenizer_vocab: int,
    expected_repo_calls: int,
    expected_hf_cfg_sources: list[str],
):

    rope_called = {"count": 0}
    hf_cfg_called = {"count": 0, "sources": []}
    repo_called = {"count": 0}
    original_repo_builder = builder_mod._build_repo_hf_deberta_v2_config

    def _raise_rope_cfg_pretrained(cls, src: str):
        del cls
        del src
        rope_called["count"] += 1
        raise AssertionError("RoPE config source must not be touched for hf_deberta_v2 backbone.")

    def _fake_hf_cfg_pretrained(cls, src: str, **kwargs: Any):
        del cls
        del kwargs
        hf_cfg_called["count"] += 1
        hf_cfg_called["sources"].append(str(src))
        return builder_mod.DebertaV2Config(
            vocab_size=128100,
            hidden_size=384,
            num_hidden_layers=5,
            num_attention_heads=6,
            intermediate_size=1536,
            max_position_embeddings=512,
            relative_attention=True,
            pos_att_type=["p2c", "c2p"],
            position_biased_input=False,
            share_att_key=True,
            type_vocab_size=0,
        )

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
        classmethod(_fake_hf_cfg_pretrained),
    )
    monkeypatch.setattr(builder_mod, "_build_repo_hf_deberta_v2_config", _count_repo_builder)

    model_cfg = make_model_config(
        backbone_type="hf_deberta_v2",
        from_scratch=from_scratch,
        pretrained={
            "discriminator_path": "disc_weights" if not from_scratch else "",
            "generator_path": pretrained_generator_path,
        },
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=tokenizer_vocab),
        max_position_embeddings=128,
    )

    assert int(repo_called["count"]) == int(expected_repo_calls)
    assert int(rope_called["count"]) == 0
    assert int(hf_cfg_called["count"]) == len(expected_hf_cfg_sources)
    if expected_hf_cfg_sources:
        assert hf_cfg_called["sources"] == [str(src) for src in expected_hf_cfg_sources]
        assert int(gen_cfg.hidden_size) == 384
        if pretrained_generator_path is not None:
            assert int(gen_cfg.num_hidden_layers) == 5
        else:
            assert int(gen_cfg.num_hidden_layers) == 2
    elif from_scratch:
        assert int(gen_cfg.hidden_size) == 768
        assert int(gen_cfg.num_hidden_layers) == 6
    if from_scratch:
        assert int(disc_cfg.hidden_size) == 768
    else:
        assert int(disc_cfg.hidden_size) == 384
    assert int(gen_cfg.hidden_size) > 0


def test_build_backbone_configs_propagates_tokenizer_special_ids_for_rope():

    model_cfg = make_model_config(
        backbone_type="rope", from_scratch=True, pretrained={"discriminator_path": "disc"}
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

    def _fake_from_pretrained(cls, src: str):
        del src
        return cls(ffn_type="mlp", num_hidden_layers=6, intermediate_size=3072)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = make_model_config(
        backbone_type="rope",
        from_scratch=True,
        pretrained={"discriminator_path": "disc"},
        rope={"ffn_type": "swiglu", "swiglu_adjust_intermediate": False},
    )
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=50265),
        max_position_embeddings=128,
    )

    assert disc_cfg.intermediate_size == 3072
    assert gen_cfg.intermediate_size == 3072


def test_scaled_swiglu_intermediate_size_rounds_to_multiple_of_128():

    model_cfg = make_model_config(
        backbone_type="rope",
        from_scratch=True,
        pretrained={"discriminator_path": "disc"},
        rope={"ffn_type": "swiglu", "intermediate_size": 4096, "swiglu_adjust_intermediate": True},
    )
    disc_cfg, _ = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=DummyTokenizer(vocab_size=50265),
        max_position_embeddings=128,
    )

    assert disc_cfg.intermediate_size == 2816


@pytest.mark.parametrize(
    ("backbone_type", "expected_layers"),
    [("hf_deberta_v2", 6), ("rope", 4)],
    ids=["hf_half_depth", "rope_third_depth"],
)
def test_derive_generator_config_uses_backbone_depth_ratio(
    backbone_type: str,
    expected_layers: int,
) -> None:
    base_cfg = BackboneConfigStub(num_hidden_layers=12)
    model_cfg = make_model_config(backbone_type=backbone_type, generator={"num_hidden_layers": None})

    gen_cfg = builder_mod._derive_generator_config(base_cfg, model_cfg)
    assert int(gen_cfg.num_hidden_layers) == expected_layers


def test_build_backbone_configs_respects_explicit_generator_intermediate_with_swiglu_adjust(
    monkeypatch: pytest.MonkeyPatch,
):

    def _fake_from_pretrained(cls, src: str):
        del src
        return cls(ffn_type="mlp", num_hidden_layers=6, intermediate_size=3072)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = make_model_config(
        backbone_type="rope",
        from_scratch=True,
        pretrained={"discriminator_path": "disc"},
        rope={"ffn_type": "swiglu", "swiglu_adjust_intermediate": True},
        generator={"intermediate_size": 1024},
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

    def _fake_from_pretrained(cls, src: str):
        ffn = "mlp" if "disc" in src else "swiglu"
        return cls(ffn_type=ffn, num_hidden_layers=6)

    monkeypatch.setattr(
        builder_mod.DebertaRoPEConfig,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    model_cfg = make_model_config(
        backbone_type="rope",
        from_scratch=False,
        pretrained={"discriminator_path": "disc", "generator_path": "gen"},
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
        ("hf_deberta_v2", False, None, "disc", "disc", None, "disc", True),
        ("hf_deberta_v2", False, "gen_model", "disc", "disc", "gen_model", "gen_model", False),
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
    cfg = make_model_config(
        backbone_type=backbone_type,
        from_scratch=from_scratch,
        pretrained={"discriminator_path": "disc", "generator_path": pretrained_generator_path},
    )
    resolved = builder_mod._resolve_backbone_sources(cfg)

    assert resolved.discriminator.config_source == expected_disc_cfg_src
    assert resolved.discriminator.weight_source == expected_disc_weight_src
    assert resolved.generator.config_source == expected_gen_cfg_src
    assert resolved.generator.weight_source == expected_gen_weight_src
    assert resolved.generator.derived_from_discriminator is expected_gen_derived


def test_validate_model_config_rejects_hf_max_position_embeddings_in_pretrained_mode():
    cfg = make_model_config(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        pretrained={"discriminator_path": "microsoft/deberta-v3-base"},
        hf={"max_position_embeddings": 1024},
    )
    with pytest.raises(ValueError, match="only supported when model.from_scratch=true"):
        builder_mod.validate_model_config(cfg)


def test_build_hf_configs_propagates_flash_runtime_policy(tmp_path: Path):

    override_path = tmp_path / "custom.json"
    override_path.write_text("{}", encoding="utf-8")

    cfg = make_model_config(
        backbone_type="hf_deberta_v2",
        hf={
            "attention_impl": "flash",
            "model_size": "xsmall",
            "flash": {
                "docblock_bias_seq_len": 0,
                "kernel_overrides_path": str(override_path),
            },
        },
    )
    tokenizer = DummyTokenizer(vocab_size=128100)

    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=cfg,
        tokenizer=tokenizer,
        max_position_embeddings=64,
    )

    for built_cfg in (disc_cfg, gen_cfg):
        assert built_cfg.hf_attention_impl == "flash"
        assert built_cfg.hf_flash["docblock_bias_seq_len"] == 0
        assert built_cfg.hf_flash["kernel_overrides_path"] == str(override_path)

    from deberta.modeling.deberta_v2_native import DebertaV2Attention
    from deberta.modeling.flashdeberta_attention import FlashDisentangledSelfAttention
    from deberta.modeling.flashdeberta_kernel_tuning import configure_flashdeberta_kernel_overrides

    try:
        assert isinstance(DebertaV2Attention(disc_cfg).self, FlashDisentangledSelfAttention)
    finally:
        configure_flashdeberta_kernel_overrides(None)


def test_shipped_flash_configs_activate_flash_for_both_backbones() -> None:

    config_paths = sorted(Path("configs/flashdeberta").glob("*.yaml"))
    assert config_paths
    for config_path in config_paths:
        cfg = load_config(config_path)
        assert cfg.model.hf.attention_impl == "flash", config_path
        disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
            model_cfg=cfg.model,
            tokenizer=DummyTokenizer(vocab_size=50265),
            max_position_embeddings=cfg.data.packing.max_seq_length,
        )
        assert disc_cfg.hf_attention_impl == "flash", config_path
        assert gen_cfg.hf_attention_impl == "flash", config_path


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"relative_attention": False}, "relative_attention=true"),
        ({"position_buckets": 0}, "position_buckets > 0"),
        ({"pos_att_type": "p2c|p2p"}, "does not support pos_att_type"),
        ({"pos_att_type": "p2c,p2p"}, "does not support pos_att_type"),
        # External checkpoints may pin a short explicit span; eager clamps
        # relative positions to it before bucketing while the flash kernels
        # do not, so flash must refuse rather than silently diverge.
        ({"max_relative_positions": 128}, "max_relative_positions to cover"),
        # A pinned span with no known position range must fail fast rather
        # than let the coverage check silently no-op. (A zero value is
        # already rejected earlier by _validate_required_max_positions;
        # None skips that check and must be caught here.)
        (
            {"max_relative_positions": 128, "max_position_embeddings": None},
            "cannot verify max_relative_positions",
        ),
    ],
)
def test_build_hf_configs_reject_flash_unsupported_materialized_configs(
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, Any],
    message: str,
):
    invalid_cfg = builder_mod._build_repo_hf_deberta_v2_config(
        model_cfg=make_model_config(backbone_type="hf_deberta_v2")
    )
    for key, value in updates.items():
        setattr(invalid_cfg, key, value)

    def _fake_from_pretrained(cls, src: str):
        del cls
        del src
        return invalid_cfg

    monkeypatch.setattr(
        builder_mod.DebertaV2Config,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )
    model_cfg = make_model_config(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        pretrained={"discriminator_path": "custom-deberta"},
        hf={"attention_impl": "flash"},
    )

    with pytest.raises(ValueError, match=message):
        builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=DummyTokenizer(vocab_size=128100),
            max_position_embeddings=64,
        )


def test_build_hf_configs_reject_flash_generator_non_power_of_two_head_dimension() -> None:
    model_cfg = make_model_config(
        backbone_type="hf_deberta_v2",
        hf={"attention_impl": "flash"},
        generator={"hidden_size": 768, "num_attention_heads": 8},
    )

    with pytest.raises(
        ValueError,
        match=(
            "generator flash attention.*positive power of two.*"
            "hidden_size=768, num_attention_heads=8, head_dim=96"
        ),
    ):
        builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=DummyTokenizer(vocab_size=128100),
            max_position_embeddings=64,
        )


def test_build_hf_configs_reject_flash_generator_head_dimension_below_16() -> None:
    model_cfg = make_model_config(
        backbone_type="hf_deberta_v2",
        hf={"attention_impl": "flash"},
        generator={"hidden_size": 32, "num_attention_heads": 4},
    )

    with pytest.raises(
        ValueError,
        match=("generator flash attention.*at least 16.*hidden_size=32, num_attention_heads=4, head_dim=8"),
    ):
        builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=DummyTokenizer(vocab_size=128100),
            max_position_embeddings=64,
        )


def test_build_backbone_configs_scratch_explicit_generator_model_is_authoritative(
    monkeypatch: pytest.MonkeyPatch,
):

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

    model_cfg = make_model_config(
        backbone_type="rope",
        from_scratch=True,
        pretrained={"discriminator_path": "disc", "generator_path": "gen_model"},
        rope={
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "ffn_type": "swiglu",
        },
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


@pytest.mark.parametrize(
    ("backbone_type", "config_overrides", "tokenizer_vocab_size", "error_match"),
    [
        (
            "rope",
            {"vocab_size": 1234},
            50265,
            "Tokenizer/checkpoint vocab mismatch for discriminator",
        ),
        (
            "rope",
            {"vocab_size": 50265, "cls_token_id": 999},
            50265,
            "Tokenizer/config special-token mismatch",
        ),
        (
            "hf_deberta_v2",
            {"vocab_size": 777},
            50265,
            "Tokenizer/checkpoint vocab mismatch for discriminator",
        ),
        (
            "hf_deberta_v2",
            {"cls_token_id": 404},
            128100,
            "Tokenizer/config special-token mismatch",
        ),
    ],
    ids=["rope_vocab_mismatch", "rope_special_id_mismatch", "hf_vocab_mismatch", "hf_special_id_mismatch"],
)
def test_build_backbone_configs_rejects_pretrained_config_contract_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    backbone_type: str,
    config_overrides: dict[str, int],
    tokenizer_vocab_size: int,
    error_match: str,
):
    pretrained_path = "disc"
    if backbone_type == "rope":
        pretrained_path = "local-rope-disc"

        def _fake_rope_from_pretrained(cls, src: str):
            del src
            return cls(num_hidden_layers=6, **config_overrides)

        monkeypatch.setattr(
            builder_mod.DebertaRoPEConfig,
            "from_pretrained",
            classmethod(_fake_rope_from_pretrained),
        )
    else:
        _patch_hf_pretrained_config_loader(monkeypatch, config_overrides=config_overrides)

    model_cfg = make_model_config(
        backbone_type=backbone_type, from_scratch=False, pretrained={"discriminator_path": pretrained_path}
    )
    with pytest.raises(ValueError, match=error_match):
        _ = builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=DummyTokenizer(vocab_size=tokenizer_vocab_size),
            max_position_embeddings=128,
        )


def test_build_backbone_configs_applies_explicit_pretrained_rope_overrides(
    monkeypatch: pytest.MonkeyPatch,
):

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

    model_cfg = make_model_config(
        backbone_type="rope",
        from_scratch=False,
        pretrained={"discriminator_path": "local-rope-disc"},
        rope={
            "pretrained.rope_theta": 50_000.0,
            "pretrained.rotary_pct": 0.5,
            "pretrained.norm_arch": "keel",
            "pretrained.ffn_type": "mlp",
            "pretrained.use_bias": True,
        },
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


def _patch_hf_pretrained_config_loader(
    monkeypatch: pytest.MonkeyPatch, *, config_overrides: dict[str, object] | None = None
) -> None:
    """Patch ``DebertaV2Config.from_pretrained`` for deterministic pretrained-HF tests."""
    overrides = dict(config_overrides or {})

    def _loader(cls, src: str, **kwargs: Any) -> Any:
        del cls
        del src
        del kwargs
        cfg = builder_mod.DebertaV2Config(
            vocab_size=128100,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            relative_attention=True,
            pos_att_type=["p2c", "c2p"],
            position_biased_input=False,
            share_att_key=True,
            type_vocab_size=0,
        )
        for key, value in overrides.items():
            setattr(cfg, str(key), value)
        return cfg

    monkeypatch.setattr(
        builder_mod.DebertaV2Config,
        "from_pretrained",
        classmethod(_loader),
    )


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
    _patch_hf_pretrained_config_loader(monkeypatch)

    model_kwargs: dict[str, object] = dict(
        backbone_type="hf_deberta_v2",
        from_scratch=from_scratch,
        pretrained={"discriminator_path": "disc"},
    )
    dropout: dict[str, object] = {}
    if hidden_dropout_prob is not _USE_MODEL_DEFAULT:
        dropout["hidden_prob"] = hidden_dropout_prob
    if attention_probs_dropout_prob is not _USE_MODEL_DEFAULT:
        dropout["attention_probs_prob"] = attention_probs_dropout_prob
    if dropout:
        model_kwargs["dropout"] = dropout

    model_cfg = make_model_config(**model_kwargs)
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
        ("xsmall", 384, 12, 6, 1536, 0),
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

    model_cfg = make_model_config(
        backbone_type="hf_deberta_v2", from_scratch=True, hf={"model_size": hf_model_size}
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


def test_build_backbone_configs_hf_deberta_rejects_explicit_generator_z_steps_for_rtd(
    monkeypatch: pytest.MonkeyPatch,
):

    def _fake_hf_cfg_pretrained(cls, src: str, **kwargs: Any):
        del cls
        del kwargs
        if str(src) != "gen_model":
            raise AssertionError(f"Unexpected HF config source: {src}")
        return builder_mod.DebertaV2Config(
            vocab_size=50265,
            hidden_size=384,
            num_hidden_layers=5,
            num_attention_heads=6,
            intermediate_size=1536,
            max_position_embeddings=512,
            relative_attention=True,
            pos_att_type=["p2c", "c2p"],
            position_biased_input=False,
            share_att_key=True,
            type_vocab_size=0,
            z_steps=7,
        )

    monkeypatch.setattr(
        builder_mod.DebertaV2Config,
        "from_pretrained",
        classmethod(_fake_hf_cfg_pretrained),
    )

    model_cfg = make_model_config(
        backbone_type="hf_deberta_v2",
        from_scratch=True,
        hf={"model_size": "xsmall"},
        pretrained={"generator_path": "gen_model"},
    )
    with pytest.raises(ValueError, match="not equivalent to generator z_steps"):
        builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=DummyTokenizer(vocab_size=50265),
            max_position_embeddings=128,
        )


def test_build_backbone_configs_scratch_can_pad_tokenizer_vocab_to_multiple():

    tokenizer = DummyTokenizer(vocab_size=500)
    model_cfg = make_model_config(
        backbone_type="rope", from_scratch=True, tokenizer={"allow_vocab_resize": True, "vocab_multiple": 128}
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

    model_cfg = make_model_config(
        backbone_type="rope", from_scratch=True, tokenizer={"allow_vocab_resize": False, "vocab_target": 640}
    )
    with pytest.raises(ValueError, match=r"model\.tokenizer\.allow_vocab_resize=false"):
        _ = builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=DummyTokenizer(vocab_size=500),
            max_position_embeddings=128,
        )


def test_build_backbone_configs_pretrained_hf_can_auto_grow_tokenizer_to_config_vocab(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_hf_pretrained_config_loader(monkeypatch, config_overrides={"vocab_size": 512})

    tokenizer = DummyTokenizer(vocab_size=500)
    model_cfg = make_model_config(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        pretrained={"discriminator_path": "disc"},
        tokenizer={"allow_vocab_resize": True},
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
    _patch_hf_pretrained_config_loader(monkeypatch, config_overrides={"vocab_size": 500})

    model_cfg = make_model_config(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        pretrained={"discriminator_path": "disc"},
        tokenizer={"allow_vocab_resize": True, "vocab_multiple": 128},
    )
    with pytest.raises(ValueError, match=r"model\.tokenizer\.vocab_multiple"):
        _ = builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=DummyTokenizer(vocab_size=489),
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

    model_cfg = make_model_config(
        backbone_type="rope",
        from_scratch=False,
        pretrained={"discriminator_path": "disc_weights", "generator_path": "gen_weights"},
    )
    disc_cfg = builder_mod.DebertaRoPEConfig(hidden_size=768, num_hidden_layers=2)
    gen_cfg = builder_mod.DebertaRoPEConfig(hidden_size=384, num_hidden_layers=1)
    _ = builder_mod.build_backbones(model_cfg=model_cfg, disc_config=disc_cfg, gen_config=gen_cfg)

    assert called == [("disc_weights", 768), ("gen_weights", 384)]


@pytest.mark.parametrize("backbone_type", ["rope", "hf_deberta_v2"])
def test_build_backbones_uses_discriminator_fallback_for_derived_pretrained_generator(
    monkeypatch: pytest.MonkeyPatch,
    backbone_type: str,
):
    called: list[str] = []

    def _fake_from_pretrained(cls, src: str, config: Any):
        del cls
        del config
        called.append(src)
        return object()

    model_cls, disc_cfg, gen_cfg = _tiny_backbone_pair(backbone_type)
    monkeypatch.setattr(model_cls, "from_pretrained", classmethod(_fake_from_pretrained))

    model_cfg = make_model_config(
        backbone_type=backbone_type,
        from_scratch=False,
        pretrained={"discriminator_path": "disc_weights", "generator_path": None},
    )
    _ = builder_mod.build_backbones(model_cfg=model_cfg, disc_config=disc_cfg, gen_config=gen_cfg)

    assert called == ["disc_weights", "disc_weights"]


def test_build_backbones_uses_resolved_hf_weight_sources(monkeypatch: pytest.MonkeyPatch):

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

    model_cfg = make_model_config(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        pretrained={"discriminator_path": "disc_weights", "generator_path": "gen_weights"},
    )
    disc_cfg = BackboneConfigStub(hidden_size=768)
    gen_cfg = BackboneConfigStub(hidden_size=384)
    _ = builder_mod.build_backbones(model_cfg=model_cfg, disc_config=disc_cfg, gen_config=gen_cfg)

    assert called == [("disc_weights", disc_cfg), ("gen_weights", gen_cfg)]


def test_build_backbones_pretrained_hf_does_not_fetch_external_config(monkeypatch: pytest.MonkeyPatch):

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

    model_cfg = make_model_config(
        backbone_type="hf_deberta_v2",
        from_scratch=False,
        pretrained={"discriminator_path": "disc_weights", "generator_path": "gen_weights"},
    )
    disc_cfg = BackboneConfigStub(hidden_size=768, vocab_size=128100)
    gen_cfg = BackboneConfigStub(hidden_size=384, vocab_size=128100)
    _ = builder_mod.build_backbones(model_cfg=model_cfg, disc_config=disc_cfg, gen_config=gen_cfg)

    assert model_calls == ["disc_weights", "gen_weights"]
    assert int(cfg_calls["count"]) == 0


def test_build_backbones_hf_from_scratch_uses_native_implementation():
    from transformers import DebertaV2Config

    model_cfg = make_model_config(
        backbone_type="hf_deberta_v2", from_scratch=True, pretrained={"discriminator_path": "disc"}
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


@pytest.mark.parametrize("backbone_type", ["rope", "hf_deberta_v2"])
def test_build_backbones_pretrained_can_skip_weight_loading(
    monkeypatch: pytest.MonkeyPatch,
    backbone_type: str,
):
    called = {"count": 0}

    def _raise_if_called(cls, src: str, config: Any):
        del cls
        del src
        del config
        called["count"] += 1
        raise AssertionError("from_pretrained should not be called when load_pretrained_weights=false")

    model_cls, disc_cfg, gen_cfg = _tiny_backbone_pair(backbone_type)
    monkeypatch.setattr(model_cls, "from_pretrained", classmethod(_raise_if_called))

    model_cfg = make_model_config(
        backbone_type=backbone_type,
        from_scratch=False,
        pretrained={"discriminator_path": "disc_weights"},
    )
    disc, gen = builder_mod.build_backbones(
        model_cfg=model_cfg,
        disc_config=disc_cfg,
        gen_config=gen_cfg,
        load_pretrained_weights=False,
    )

    assert isinstance(disc, model_cls)
    assert isinstance(gen, model_cls)
    assert called["count"] == 0


def test_native_deberta_v2_model_honors_config_z_steps():
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

    model_cfg = make_model_config(backbone_type="rope", from_scratch=True, rope={"norm_arch": "not-valid"})
    with pytest.raises(ValueError, match="model.rope.norm_arch must be one of"):
        builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=DummyTokenizer(vocab_size=50265),
            max_position_embeddings=128,
        )


def test_build_backbone_configs_rejects_hf_deberta_sources_for_pretrained_rope():
    model_cfg = make_model_config(
        backbone_type="rope",
        from_scratch=False,
        pretrained={"discriminator_path": "microsoft/deberta-v3-base"},
    )
    with pytest.raises(ValueError, match="requires DebertaRoPE checkpoints"):
        builder_mod.build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=DummyTokenizer(vocab_size=50265),
            max_position_embeddings=128,
        )


def test_build_backbone_configs_sets_tokenizer_special_ids_for_hf_configs():
    tokenizer = DummyTokenizer(vocab_size=128)

    model_cfg = make_model_config(backbone_type="hf_deberta_v2", from_scratch=True)
    disc_cfg, gen_cfg = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=128,
    )

    for cfg in (disc_cfg, gen_cfg):
        assert getattr(cfg, "pad_token_id", None) == 0
        assert getattr(cfg, "cls_token_id", None) == 1
        assert getattr(cfg, "sep_token_id", None) == 2
        assert getattr(cfg, "mask_token_id", None) == 3
        assert getattr(cfg, "bos_token_id", None) == 4
        assert getattr(cfg, "eos_token_id", None) == 5
        assert getattr(cfg, "use_rmsnorm_heads", None) is False


def test_build_backbone_configs_preserves_pretrained_rope_architecture_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    from deberta.modeling.rope_encoder import DebertaRoPEConfig

    checkpoint_cfg = DebertaRoPEConfig(
        vocab_size=32000,
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_act="gelu",
        rope_theta=50000.0,
        rotary_pct=0.5,
        use_absolute_position_embeddings=True,
        type_vocab_size=3,
        norm_arch="keel",
        norm_eps=1.0e-5,
        keel_alpha_init=9.0,
        keel_alpha_learnable=True,
        ffn_type="mlp",
        use_bias=True,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.4,
    )

    monkeypatch.setattr(
        "deberta.modeling.builder.DebertaRoPEConfig.from_pretrained",
        lambda _src: checkpoint_cfg,
    )
    tokenizer = DummyTokenizer(vocab_size=32000)

    model_cfg = make_model_config(
        backbone_type="rope",
        from_scratch=False,
        pretrained={"discriminator_path": "local-rope-disc"},
        dropout={"hidden_prob": None, "attention_probs_prob": None},
    )

    disc_cfg, _ = builder_mod.build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=512,
    )

    assert disc_cfg.rope_theta == pytest.approx(50000.0)
    assert disc_cfg.rotary_pct == pytest.approx(0.5)
    assert disc_cfg.use_absolute_position_embeddings is True
    assert disc_cfg.type_vocab_size == 3
    assert disc_cfg.norm_arch == "keel"
    assert disc_cfg.norm_eps == pytest.approx(1.0e-5)
    assert disc_cfg.keel_alpha_init == pytest.approx(9.0)
    assert disc_cfg.keel_alpha_learnable is True
    assert disc_cfg.ffn_type == "mlp"
    assert disc_cfg.use_bias is True
    assert disc_cfg.hidden_dropout_prob == pytest.approx(0.3)
    assert disc_cfg.attention_probs_dropout_prob == pytest.approx(0.4)
