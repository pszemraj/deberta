from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import pytest
import torch

import deberta.cli as cli_mod
from deberta.cli import _load_json, _load_yaml
from deberta.config import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    validate_data_config,
    validate_model_config,
    validate_training_workflow_options,
)
from deberta.export_cli import _build_export_parser
from deberta.training.pretrain import (
    _build_optimizer,
    _find_latest_checkpoint,
    _load_checkpoint_data_progress,
    _normalize_mixed_precision,
    _normalize_sdpa_kernel,
    _normalize_torch_compile_mode,
    _prepare_output_dir,
    _save_checkpoint_data_progress,
    _save_training_checkpoint,
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
        _load_json(bad)


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
        _load_yaml(bad)


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


def test_checkpoint_data_progress_roundtrip(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir(parents=True, exist_ok=True)

    assert _load_checkpoint_data_progress(ckpt) is None
    _save_checkpoint_data_progress(checkpoint_dir=ckpt, consumed_micro_batches=123)
    assert _load_checkpoint_data_progress(ckpt) == 123


class _FakeAccelerator:
    def __init__(self, *, is_main_process: bool) -> None:
        self.is_main_process = bool(is_main_process)
        self.wait_count = 0
        self.save_paths: list[str] = []

    def wait_for_everyone(self) -> None:
        self.wait_count += 1

    def save_state(self, path: str) -> None:
        self.save_paths.append(path)
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        marker = "main" if self.is_main_process else "worker"
        (p / f"{marker}.txt").write_text("ok", encoding="utf-8")


def test_save_training_checkpoint_calls_collective_save_on_non_main_rank(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint-1"
    ckpt.mkdir(parents=True, exist_ok=True)

    accel = _FakeAccelerator(is_main_process=False)
    _save_training_checkpoint(
        accelerator=accel,
        checkpoint_dir=ckpt,
        output_dir=out,
        consumed_micro_batches=7,
        save_total_limit=3,
        log_label="periodic",
    )

    assert accel.save_paths == [str(ckpt)]
    assert accel.wait_count >= 3
    assert not (ckpt / "data_state.json").exists()


def test_save_training_checkpoint_writes_data_progress_on_main_rank(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint-3"

    accel = _FakeAccelerator(is_main_process=True)
    _save_training_checkpoint(
        accelerator=accel,
        checkpoint_dir=ckpt,
        output_dir=out,
        consumed_micro_batches=42,
        save_total_limit=3,
        log_label="final",
    )

    assert accel.save_paths == [str(ckpt)]
    assert _load_checkpoint_data_progress(ckpt) == 42


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
    assert cfg.sdpa_kernel == "auto"


def test_model_config_defaults_dropouts_to_zero():
    cfg = ModelConfig()
    assert cfg.hidden_dropout_prob == pytest.approx(0.0)
    assert cfg.attention_probs_dropout_prob == pytest.approx(0.0)


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


def test_normalize_sdpa_kernel_accepts_aliases_and_rejects_invalid():
    assert _normalize_sdpa_kernel("auto") == "auto"
    assert _normalize_sdpa_kernel("flashattention") == "flash"
    assert _normalize_sdpa_kernel("mem-efficient") == "mem_efficient"
    assert _normalize_sdpa_kernel("math") == "math"
    assert _normalize_sdpa_kernel("flash_only") == "flash_only"

    with pytest.raises(ValueError, match="train.sdpa_kernel must be one of"):
        _normalize_sdpa_kernel("best")


def test_main_cli_train_subcommand_loads_yaml_and_applies_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_name: HuggingFaceFW/fineweb-edu",
                "  max_seq_length: 32",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(["train", str(cfg_path), "--max_steps", "7"])

    assert "train_cfg" in seen
    assert seen["data_cfg"].max_seq_length == 32
    assert seen["train_cfg"].max_steps == 7


def test_main_cli_export_subcommand_builds_export_config(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    def _fake_run_export(cfg):
        seen["cfg"] = cfg

    monkeypatch.setattr(cli_mod, "run_export", _fake_run_export)
    cli_mod.main(
        [
            "export",
            "runs/demo/checkpoint-10",
            "--what",
            "generator",
            "--output-dir",
            "runs/demo/exported_hf",
        ]
    )

    cfg = seen["cfg"]
    assert cfg.checkpoint_dir == "runs/demo/checkpoint-10"
    assert cfg.export_what == "generator"
    assert cfg.output_dir == "runs/demo/exported_hf"


def test_train_cli_rejects_invalid_constrained_values_at_parse_time():
    parser = cli_mod._build_main_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--norm_arch", "invalid"])

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--report_to", "invalid"])


def test_export_parser_rejects_conflicting_boolean_flags():
    parser = _build_export_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["runs/demo/checkpoint-10", "--safe-serialization", "--no-safe-serialization"])

    with pytest.raises(SystemExit):
        parser.parse_args(["runs/demo/checkpoint-10", "--offload-to-cpu", "--no-offload-to-cpu"])

    with pytest.raises(SystemExit):
        parser.parse_args(["runs/demo/checkpoint-10", "--rank0-only", "--no-rank0-only"])


def test_validate_data_config_rejects_conflicting_sources():
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_data_config(
            DataConfig(
                load_from_disk="runs/saved_ds",
                dataset_name="HuggingFaceFW/fineweb-edu",
                streaming=False,
            )
        )


def test_validate_training_workflow_options_rejects_eval_knobs():
    with pytest.raises(ValueError, match="Evaluation workflow is not implemented yet"):
        validate_training_workflow_options(
            data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu", eval_split="validation"),
            train_cfg=TrainConfig(),
        )

    with pytest.raises(ValueError, match="currently unused"):
        validate_training_workflow_options(
            data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
            train_cfg=TrainConfig(per_device_eval_batch_size=8),
        )


def test_validate_training_workflow_options_rejects_flash_only_with_packing():
    with pytest.raises(ValueError, match="flash_only is not supported with data.pack_sequences=true"):
        validate_training_workflow_options(
            data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu", pack_sequences=True),
            train_cfg=TrainConfig(sdpa_kernel="flash_only"),
        )


def test_validate_model_config_rejects_rope_only_knobs_in_hf_mode():
    cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        rope_theta=50_000.0,
    )
    with pytest.raises(ValueError, match="only valid when model.backbone_type='rope'"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_derived_generator_knobs_with_explicit_generator_source():
    cfg = ModelConfig(
        backbone_type="rope",
        generator_model_name_or_path="microsoft/deberta-v3-small",
        generator_hidden_size=256,
    )
    with pytest.raises(ValueError, match="only used when deriving generator config"):
        validate_model_config(cfg)


def test_readme_cli_examples_are_parseable():
    parser = cli_mod._build_main_parser()
    examples = [
        "train configs/pretrain_rope_fineweb_edu.yaml",
        "train configs/pretrain_rope_fineweb_edu_2048.yaml",
        "train configs/pretrain_rope_fineweb_edu_4096.yaml",
        (
            "export runs/deberta_rope_rtd/checkpoint-10000 "
            "--what discriminator "
            "--output-dir runs/deberta_rope_rtd/exported_hf"
        ),
    ]

    for cmd in examples:
        parser.parse_args(shlex.split(cmd))
