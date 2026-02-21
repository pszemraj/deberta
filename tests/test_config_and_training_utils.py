from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from deberta.cli import _load_json, _load_yaml
from deberta.config import TrainConfig
from deberta.training.pretrain import (
    _build_optimizer,
    _find_latest_checkpoint,
    _normalize_mixed_precision,
    _normalize_torch_compile_mode,
    _prepare_output_dir,
    _should_force_legacy_tf32_for_compile,
)


class _TinyRTDLikeModel(torch.nn.Module):
    """Minimal module exposing generator/discriminator-style parameter names."""

    def __init__(self) -> None:
        super().__init__()
        self.generator = torch.nn.Linear(8, 8)
        self.generator_lm_head = torch.nn.Linear(8, 8)
        self.discriminator = torch.nn.Linear(8, 8)
        self.discriminator_norm = torch.nn.LayerNorm(8)


def test_load_yaml_nested_and_flat(tmp_path: Path):
    pytest.importorskip("yaml")

    nested = tmp_path / "nested.yaml"
    nested.write_text(
        "\n".join(
            [
                "model:",
                "  backbone_type: rope",
                "  ffn_type: swiglu",
                "data:",
                "  max_seq_length: 128",
                "train:",
                "  overwrite_output_dir: true",
                "  mlm_max_ngram: 3",
            ]
        ),
        encoding="utf-8",
    )
    model_nested, data_nested, train_nested = _load_yaml(nested)
    assert model_nested.ffn_type == "swiglu"
    assert data_nested.max_seq_length == 128
    assert train_nested.overwrite_output_dir is True
    assert train_nested.mlm_max_ngram == 3
    assert train_nested.mixed_precision == "bf16"

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
    _, data_flat, train_flat = _load_yaml(flat)
    assert data_flat.max_seq_length == 64
    assert train_flat.overwrite_output_dir is False
    assert train_flat.mlm_max_ngram == 1
    assert _normalize_mixed_precision(train_flat.mixed_precision) == "no"


def test_load_json_nested_and_flat(tmp_path: Path):
    nested = tmp_path / "nested.json"
    nested.write_text(
        json.dumps(
            {
                "model": {"backbone_type": "rope", "ffn_type": "mlp"},
                "data": {"max_seq_length": 96},
                "train": {"generator_learning_rate": 3.0e-4, "disc_loss_weight": 50.0},
            }
        ),
        encoding="utf-8",
    )
    model_nested, data_nested, train_nested = _load_json(nested)
    assert model_nested.ffn_type == "mlp"
    assert data_nested.max_seq_length == 96
    assert train_nested.generator_learning_rate == pytest.approx(3.0e-4)
    assert train_nested.disc_loss_weight == pytest.approx(50.0)

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
    _, data_flat, train_flat = _load_json(flat)
    assert data_flat.max_seq_length == 80
    assert train_flat.mlm_max_ngram == 2
    assert train_flat.mask_token_prob == pytest.approx(0.7)


def test_load_json_unknown_key_raises(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"unknown_field": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys in config file"):
        _load_json(bad)


def test_prepare_output_dir_respects_overwrite_and_resume(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    keep = out / "existing.txt"
    keep.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Output directory exists and is not empty"):
        _prepare_output_dir(
            output_dir=out,
            overwrite_output_dir=False,
            resume_from_checkpoint=None,
            is_main_process=True,
        )

    _prepare_output_dir(
        output_dir=out,
        overwrite_output_dir=False,
        resume_from_checkpoint="auto",
        is_main_process=True,
    )
    assert keep.exists()

    _prepare_output_dir(
        output_dir=out,
        overwrite_output_dir=True,
        resume_from_checkpoint=None,
        is_main_process=True,
    )
    assert out.exists()
    assert not any(out.iterdir())


def test_find_latest_checkpoint_picks_highest_step(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoint-2").mkdir()
    (out / "checkpoint-11").mkdir()
    (out / "checkpoint-4").mkdir()

    latest = _find_latest_checkpoint(out)
    assert latest is not None
    assert latest.name == "checkpoint-11"


def test_build_optimizer_supports_generator_specific_lr():
    model = _TinyRTDLikeModel()
    cfg = TrainConfig(learning_rate=1.0e-3, generator_learning_rate=5.0e-4, weight_decay=0.1)
    opt = _build_optimizer(model, cfg)

    lrs = {float(g["lr"]) for g in opt.param_groups}
    assert lrs == {1.0e-3, 5.0e-4}

    # We should have both decay and no-decay groups present.
    wds = {float(g["weight_decay"]) for g in opt.param_groups}
    assert wds == {0.0, 0.1}


def test_train_config_defaults_to_bf16_autocast():
    cfg = TrainConfig()
    assert cfg.mixed_precision == "bf16"


def test_normalize_mixed_precision_accepts_bool_and_synonyms():
    assert _normalize_mixed_precision("bf16") == "bf16"
    assert _normalize_mixed_precision("none") == "no"
    assert _normalize_mixed_precision(False) == "no"
    assert _normalize_mixed_precision(True) == "bf16"

    with pytest.raises(ValueError, match="train.mixed_precision must be one of: no\\|bf16"):
        _normalize_mixed_precision("fp16")


def test_normalize_torch_compile_mode_accepts_aliases_and_rejects_invalid():
    assert _normalize_torch_compile_mode("default") == "default"
    assert _normalize_torch_compile_mode("reduce_overhead") == "reduce-overhead"
    assert _normalize_torch_compile_mode("max_autotune") == "max-autotune"
    assert _normalize_torch_compile_mode("max-autotune-no-cudagraphs") == "max-autotune-no-cudagraphs"

    with pytest.raises(ValueError, match="train.torch_compile_mode must be one of"):
        _normalize_torch_compile_mode("fastest")


def test_force_legacy_tf32_for_compile_modes():
    assert _should_force_legacy_tf32_for_compile(torch_compile=False, compile_mode="max-autotune") is False
    assert _should_force_legacy_tf32_for_compile(torch_compile=True, compile_mode="default") is False
    assert _should_force_legacy_tf32_for_compile(torch_compile=True, compile_mode="max-autotune") is True
    assert (
        _should_force_legacy_tf32_for_compile(torch_compile=True, compile_mode="max-autotune-no-cudagraphs")
        is True
    )
