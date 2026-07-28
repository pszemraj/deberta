from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from _config_factories import make_model_config, make_optim_config, make_train_config

from deberta.config import (
    Config,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    apply_dotted_override,
    load_config,
    validate_model_config,
    validate_optim_config,
    validate_train_config,
)


def test_programmatic_config_resolves_rope_profile_without_serializing_provenance() -> None:
    cfg = Config(model=ModelConfig(backbone_type="rope"))

    assert cfg.train.objective.mask_token_prob == pytest.approx(0.8)
    assert cfg.train.objective.random_token_prob == pytest.approx(0.1)
    assert cfg.train.objective.disc_loss_weight == pytest.approx(50.0)
    assert cfg.optim.lr.base == pytest.approx(5e-4)
    assert cfg.optim.adam.epsilon == pytest.approx(1e-8)
    assert cfg.optim.scheduler.warmup_steps == 1_000
    assert set(dataclasses.asdict(cfg)) == {"model", "data", "train", "optim", "logging"}


def test_config_defaults_match_training_contract() -> None:
    cfg = Config()

    assert cfg.train.max_steps == 10_000
    assert cfg.optim.scheduler.warmup_steps == 1_000
    assert cfg.optim.scheduler.warmup_steps < cfg.train.max_steps
    assert cfg.train.mixed_precision == "bf16"
    assert cfg.train.sdpa_kernel == "auto"
    assert cfg.train.token_weighted_gradient_accumulation is True
    assert cfg.train.decoupled_training is True
    assert make_train_config(decoupled_training=False).decoupled_training is False
    assert cfg.model.backbone_type == "hf_deberta_v2"
    assert cfg.model.dropout.hidden_prob == pytest.approx(0.0)
    assert cfg.model.dropout.attention_probs_prob == pytest.approx(0.0)
    assert cfg.model.hf.model_size == "base"
    assert cfg.model.rope.ffn_type == "mlp"
    assert cfg.model.rope.swiglu_adjust_intermediate is True
    assert cfg.data.packing.block_cross_document_attention is False


def test_programmatic_config_preserves_custom_profile_values() -> None:
    cfg = Config(
        model=ModelConfig(backbone_type="rope"),
        train=make_train_config(
            objective={
                "mask_token_prob": 0.6,
                "random_token_prob": 0.2,
                "disc_loss_weight": 7.0,
            }
        ),
        optim=make_optim_config(
            adam={"epsilon": 2e-7},
            scheduler={"warmup_steps": 77},
        ),
    )

    assert cfg.train.objective.mask_token_prob == pytest.approx(0.6)
    assert cfg.train.objective.random_token_prob == pytest.approx(0.2)
    assert cfg.train.objective.disc_loss_weight == pytest.approx(7.0)
    assert cfg.optim.lr.base == pytest.approx(1e-4)
    assert cfg.optim.adam.epsilon == pytest.approx(2e-7)
    assert cfg.optim.scheduler.warmup_steps == 77

    cfg = apply_dotted_override(cfg, "model.backbone_type=hf_deberta_v2")
    cfg = apply_dotted_override(cfg, "model.backbone_type=rope")
    assert cfg.train.objective.mask_token_prob == pytest.approx(0.6)
    assert cfg.train.objective.random_token_prob == pytest.approx(0.2)
    assert cfg.train.objective.disc_loss_weight == pytest.approx(7.0)
    assert cfg.optim.adam.epsilon == pytest.approx(2e-7)
    assert cfg.optim.scheduler.warmup_steps == 77


def test_programmatic_config_preserves_explicit_schema_default_profile_values() -> None:
    cfg = Config(
        model=ModelConfig(backbone_type="rope"),
        train=TrainConfig(),
        optim=OptimConfig(),
    )

    assert cfg.train.objective.mask_token_prob == pytest.approx(1.0)
    assert cfg.train.objective.random_token_prob == pytest.approx(0.0)
    assert cfg.train.objective.disc_loss_weight == pytest.approx(10.0)
    assert cfg.optim.lr.base == pytest.approx(1e-4)
    assert cfg.optim.adam.epsilon == pytest.approx(1e-6)
    assert cfg.optim.scheduler.warmup_steps == 1_000


def test_load_yaml_nested(tmp_path: Path):

    nested = tmp_path / "nested.yaml"
    nested.write_text(
        "\n".join(
            [
                "model:",
                "  backbone_type: rope",
                "  rope:",
                "    ffn_type: swiglu",
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "  packing:",
                "    max_seq_length: 128",
                "train:",
                "  checkpoint:",
                "    overwrite_output_dir: true",
                "    save_steps: 777",
                "  objective:",
                "    mlm_max_ngram: 3",
                "optim:",
                "  scheduler:",
                "    warmup_steps: 2",
                "logging:",
                "  wandb:",
                "    enabled: true",
                "    watch: all",
                "  debug:",
                "    metrics: true",
            ]
        ),
        encoding="utf-8",
    )
    cfg_nested = load_config(nested)
    assert cfg_nested.model.rope.ffn_type == "swiglu"
    assert cfg_nested.data.packing.max_seq_length == 128
    assert cfg_nested.train.checkpoint.overwrite_output_dir is True
    assert cfg_nested.train.checkpoint.save_steps == 777
    assert cfg_nested.train.objective.mlm_max_ngram == 3
    assert cfg_nested.train.mixed_precision == "bf16"
    assert cfg_nested.train.objective.mask_token_prob == pytest.approx(0.8)
    assert cfg_nested.train.objective.random_token_prob == pytest.approx(0.1)
    assert cfg_nested.train.objective.disc_loss_weight == pytest.approx(50.0)
    assert cfg_nested.optim.lr.base == pytest.approx(5e-4)
    assert cfg_nested.optim.adam.epsilon == pytest.approx(1e-8)
    assert cfg_nested.optim.scheduler.warmup_steps == 2
    assert cfg_nested.logging.wandb.enabled is True
    assert cfg_nested.logging.wandb.watch == "all"
    assert cfg_nested.logging.debug.metrics is True


def test_load_json_nested(tmp_path: Path):
    nested = tmp_path / "nested.json"
    nested.write_text(
        json.dumps(
            {
                "model": {"backbone_type": "rope", "rope": {"ffn_type": "mlp"}},
                "data": {
                    "source": {"dataset_name": "HuggingFaceFW/fineweb-edu"},
                    "packing": {"max_seq_length": 96},
                },
                "train": {
                    "objective": {
                        "mask_token_prob": 1.0,
                        "random_token_prob": 0.0,
                        "disc_loss_weight": 10.0,
                    }
                },
                "optim": {
                    "lr": {"generator": 3.0e-4},
                    "adam": {"epsilon": 1.0e-6},
                    "scheduler": {"warmup_steps": 5_000},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg_nested = load_config(nested)
    assert cfg_nested.model.rope.ffn_type == "mlp"
    assert cfg_nested.data.packing.max_seq_length == 96
    assert cfg_nested.optim.lr.generator == pytest.approx(3.0e-4)
    assert cfg_nested.train.objective.mask_token_prob == pytest.approx(1.0)
    assert cfg_nested.train.objective.random_token_prob == pytest.approx(0.0)
    assert cfg_nested.train.objective.disc_loss_weight == pytest.approx(10.0)
    assert cfg_nested.optim.adam.epsilon == pytest.approx(1e-6)
    assert cfg_nested.optim.scheduler.warmup_steps == 5_000


def test_load_yaml_hf_flash_config(tmp_path: Path):

    config_path = tmp_path / "flash.yaml"
    override_path = tmp_path / "custom.json"
    override_path.write_text("{}", encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  backbone_type: hf_deberta_v2",
                "  hf:",
                "    attention_impl: flash",
                "    flash:",
                f"      kernel_overrides_path: {override_path}",
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.model.hf.attention_impl == "flash"
    assert cfg.model.hf.flash.kernel_overrides_path == str(override_path)
    assert cfg.optim.lr.base == pytest.approx(1e-4)


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (make_model_config(generator={"hidden_size": 0}), "model.generator.hidden_size"),
        (make_model_config(rope={"rope_theta": 0.0}), "model.rope.rope_theta"),
        (make_model_config(dropout={"hidden_prob": 1.1}), "model.dropout.hidden_prob"),
        (
            make_model_config(
                hf={"attention_impl": "flash"},
                dropout={"hidden_prob": None, "attention_probs_prob": 0.0},
            ),
            "must resolve to 0.0",
        ),
    ],
)
def test_model_constraints_fail_during_validation(cfg: ModelConfig, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_model_config(cfg)


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (make_train_config(objective={"gen_loss_weight": -1.0}), "loss weights"),
        (
            make_train_config(objective={"gen_loss_weight": 0.0, "disc_loss_weight": 0.0}),
            "At least one",
        ),
    ],
)
def test_train_constraints_fail_during_validation(cfg: TrainConfig, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_train_config(cfg)


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (make_optim_config(adam={"beta1": 1.0}), "optim.adam.beta1"),
        (make_optim_config(adam={"epsilon": 0.0}), "optim.adam.epsilon"),
    ],
)
def test_optim_constraints_fail_during_validation(cfg: OptimConfig, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_optim_config(cfg)


@pytest.mark.parametrize("format_name", ["JSON", "YAML"])
def test_load_nested_unknown_top_level_key_raises(tmp_path: Path, format_name: str):
    if format_name == "YAML":
        content = "model:\n  backbone_type: rope\nunexpected_top_level_key: 1"
    else:
        content = json.dumps({"model": {"backbone_type": "rope"}, "unexpected_top_level_key": 1})
    bad = tmp_path / f"bad_nested.{format_name.lower()}"
    bad.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=f"Unknown top-level keys in nested {format_name} config"):
        load_config(bad)


@pytest.mark.parametrize(
    ("section", "removed_key", "value", "match"),
    [
        (
            "model",
            "profile",
            "modern",
            r"profile \(remove model\.profile; configure model\.backbone_type and explicit fields directly\)",
        ),
        (
            "logging",
            "backend",
            "tensorboard",
            r"backend \(remove logging\.backend; only logging\.wandb\.enabled remains for optional tracking\)",
        ),
    ],
)
def test_load_config_explains_removed_keys(
    tmp_path: Path,
    section: str,
    removed_key: str,
    value: str,
    match: str,
) -> None:
    bad = tmp_path / "removed-key.json"
    bad.write_text(json.dumps({section: {removed_key: value}}), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_config(bad)


def test_load_config_explains_removed_dense_docblock_route(tmp_path: Path) -> None:
    bad = tmp_path / "removed-flash-key.json"
    bad.write_text(
        json.dumps(
            {
                "model": {
                    "hf": {
                        "flash": {
                            "docblock_bias_seq_len": 1024,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"docblock_bias_seq_len \(remove model\.hf\.flash\.docblock_bias_seq_len",
    ):
        load_config(bad)


@pytest.mark.parametrize(
    "config_name",
    [
        "pretrain_deberta_v3_bespoke_100k.yaml",
        "pretrain_rope_fineweb_edu.yaml",
        "tiny_cpu_smoke.yaml",
        "flashdeberta/pretrain_flashdeberta_1024.yaml",
    ],
)
def test_example_yaml_configs_load(config_name: str) -> None:

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / config_name
    cfg = load_config(config_path)

    assert cfg.data.source.dataset_name is not None or cfg.data.source.data_files is not None
