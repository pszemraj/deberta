from __future__ import annotations

import json
from pathlib import Path

import pytest

from deberta.config import (
    apply_profile_defaults,
    load_config,
    validate_data_config,
    validate_model_config,
    validate_optim_config,
    validate_train_config,
    validate_training_workflow_options,
)


def test_load_yaml_nested_and_flat(tmp_path: Path):
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

    flat = tmp_path / "flat.yaml"
    flat.write_text(
        "\n".join(
            [
                "backbone_type: rope",
                "max_seq_length: 64",
                "overwrite_output_dir: false",
                "mixed_precision: no",
                "mlm_max_ngram: 1",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown top-level keys in nested YAML config"):
        load_config(flat)


def test_load_json_nested_and_flat(tmp_path: Path):
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
    assert cfg_nested.train.generator_learning_rate == pytest.approx(3.0e-4)
    assert cfg_nested.train.disc_loss_weight == pytest.approx(50.0)

    flat = tmp_path / "flat.json"
    flat.write_text(
        json.dumps(
            {
                "backbone_type": "rope",
                "max_seq_length": 80,
                "mlm_max_ngram": 2,
                "mask_token_prob": 0.7,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown top-level keys in nested JSON config"):
        load_config(flat)


def test_load_yaml_resolves_variables(tmp_path: Path):
    pytest.importorskip("yaml")

    cfg = tmp_path / "vars.yaml"
    cfg.write_text(
        "\n".join(
            [
                "variables:",
                "  seq: 256",
                "  lr: 5e-4",
                "model:",
                "  backbone_type: hf_deberta_v2",
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "  packing:",
                "    max_seq_length: $variables.seq",
                "optim:",
                "  lr:",
                "    base: $variables.lr",
                "logging:",
                "  run_name: run-{$variables.seq}",
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_config(cfg)
    assert int(loaded.data.packing.max_seq_length) == 256
    assert float(loaded.train.learning_rate) == pytest.approx(5e-4)
    assert loaded.train.run_name == "run-256"


def test_load_yaml_variable_circular_reference_raises(tmp_path: Path):
    pytest.importorskip("yaml")

    cfg = tmp_path / "vars_cycle.yaml"
    cfg.write_text(
        "\n".join(
            [
                "variables:",
                "  a: $variables.b",
                "  b: $variables.a",
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 1",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Circular variable reference"):
        load_config(cfg)


def test_load_json_unknown_key_raises(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"unknown_field": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown top-level keys in nested JSON config"):
        load_config(bad)


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
    apply_profile_defaults(model_cfg=cfg.model, train_cfg=cfg.train, optim_cfg=cfg.optim)

    validate_model_config(cfg.model)
    validate_data_config(cfg.data)
    validate_train_config(cfg.train)
    validate_optim_config(cfg.optim)
    validate_training_workflow_options(
        data_cfg=cfg.data,
        train_cfg=cfg.train,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        logging_cfg=cfg.logging,
    )

    assert cfg.model.profile == "deberta_v3_parity"
    assert cfg.model.backbone_type == "hf_deberta_v2"
    assert cfg.model.pretrained_discriminator_path == ""
    assert bool(cfg.train.decoupled_training) is True
