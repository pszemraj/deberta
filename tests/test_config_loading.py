from __future__ import annotations

import json
from pathlib import Path

import pytest

from deberta.config import (
    ModelConfig,
    OptimConfig,
    TrainConfig,
    load_config,
    validate_data_config,
    validate_model_config,
    validate_optim_config,
    validate_train_config,
    validate_training_workflow_options,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_yaml_nested(tmp_path: Path):
    pytest.importorskip("yaml")

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
                "  objective:",
                "    mlm_max_ngram: 3",
            ]
        ),
        encoding="utf-8",
    )
    cfg_nested = load_config(nested)
    assert cfg_nested.model.rope.ffn_type == "swiglu"
    assert cfg_nested.data.packing.max_seq_length == 128
    assert cfg_nested.train.overwrite_output_dir is True
    assert cfg_nested.train.mlm_max_ngram == 3
    assert cfg_nested.train.mixed_precision == "bf16"


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
                "train": {"objective": {"disc_loss_weight": 50.0}},
                "optim": {"lr": {"generator": 3.0e-4}},
            }
        ),
        encoding="utf-8",
    )
    cfg_nested = load_config(nested)
    assert cfg_nested.model.rope.ffn_type == "mlp"
    assert cfg_nested.data.packing.max_seq_length == 96
    assert cfg_nested.optim.lr.generator == pytest.approx(3.0e-4)
    assert cfg_nested.train.objective.disc_loss_weight == pytest.approx(50.0)


def test_load_yaml_hf_flash_config(tmp_path: Path):
    pytest.importorskip("yaml")

    config_path = tmp_path / "flash.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  backbone_type: hf_deberta_v2",
                "  hf:",
                "    attention_impl: flash",
                "    flash:",
                "      force_varlen: true",
                "      varlen_min_seq_len: 4096",
                "      docblock_bias_seq_len: 0",
                "      local_bias_seq_len: 1024",
                "      local_bias_max_batch_size: 2",
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.model.hf.attention_impl == "flash"
    assert cfg.model.hf.flash.force_varlen is True
    assert cfg.model.hf.flash.varlen_min_seq_len == 4096
    assert cfg.model.hf.flash.docblock_bias_seq_len == 0
    assert cfg.model.hf.flash.local_bias_seq_len == 1024
    assert cfg.model.hf.flash.local_bias_max_batch_size == 2


def test_config_reference_loads() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "config-reference.yaml")
    assert cfg.model.backbone_type == "hf_deberta_v2"
    assert cfg.data.source.dataset_name == "HuggingFaceFW/fineweb-edu"


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (ModelConfig(generator={"hidden_size": 0}), "model.generator.hidden_size"),
        (ModelConfig(rope={"rope_theta": 0.0}), "model.rope.rope_theta"),
        (ModelConfig(dropout={"hidden_prob": 1.1}), "model.dropout.hidden_prob"),
        (
            ModelConfig(
                hf={"attention_impl": "flash"},
                dropout={"hidden_prob": None, "attention_probs_prob": 0.0},
            ),
            "explicitly set to 0.0",
        ),
    ],
)
def test_model_constraints_fail_during_validation(cfg: ModelConfig, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_model_config(cfg)


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (TrainConfig(objective={"gen_loss_weight": -1.0}), "loss weights"),
        (
            TrainConfig(objective={"gen_loss_weight": 0.0, "disc_loss_weight": 0.0}),
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
        (OptimConfig(adam={"beta1": 1.0}), "optim.adam.beta1"),
        (OptimConfig(adam={"epsilon": 0.0}), "optim.adam.epsilon"),
    ],
)
def test_optim_constraints_fail_during_validation(cfg: OptimConfig, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_optim_config(cfg)


def test_load_json_nested_unknown_top_level_key_raises(tmp_path: Path):
    bad = tmp_path / "bad_nested.json"
    bad.write_text(
        json.dumps(
            {
                "model": {"backbone_type": "rope"},
                "unexpected_top_level_key": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown top-level keys in nested JSON config"):
        load_config(bad)


def test_load_yaml_nested_unknown_top_level_key_raises(tmp_path: Path):
    pytest.importorskip("yaml")

    bad = tmp_path / "bad_nested.yaml"
    bad.write_text(
        "\n".join(
            [
                "model:",
                "  backbone_type: rope",
                "unexpected_top_level_key: 1",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown top-level keys in nested YAML config"):
        load_config(bad)


@pytest.mark.parametrize(
    "config_name",
    [
        "pretrain_hf_deberta_v2_parity_base.yaml",
        "pretrain_hf_deberta_v2_parity_small.yaml",
    ],
)
def test_parity_yaml_configs_parse_and_validate(config_name: str) -> None:
    pytest.importorskip("yaml")

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / config_name
    cfg = load_config(config_path)

    validate_model_config(cfg.model)
    validate_data_config(cfg.data)
    validate_train_config(cfg.train)
    validate_optim_config(cfg.optim)
    validate_training_workflow_options(
        data_cfg=cfg.data,
        train_cfg=cfg.train,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
    )

    assert cfg.model.backbone_type == "hf_deberta_v2"
    assert cfg.model.pretrained_discriminator_path == ""
    assert bool(cfg.train.decoupled_training) is True
