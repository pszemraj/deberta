from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import logging
import re
import shlex
import sys
import types
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from _fakes import (
    DummyTokenizer,
    FakeAccelerator,
    FakeWandbRun,
    SimpleRTD,
    TinyRTDLikeModel,
    setup_pretraining_mocks,
)

import deberta.cli as cli_mod
from deberta.checkpoint_utils import (
    canonical_compile_state_key,
    load_checkpoint_model_state_dict,
    load_model_state_with_compile_key_remap,
)
from deberta.cli import _load_json, _load_yaml
from deberta.config import (
    RUN_CONFIG_SCHEMA_VERSION,
    Config,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    _looks_like_hf_deberta_checkpoint,
    _normalize_hf_attention_kernel,
    _normalize_sdpa_kernel,
    _normalize_torch_compile_backend,
    _normalize_torch_compile_mode,
    _normalize_torch_compile_scope,
    _normalize_wandb_watch,
    apply_dotted_override,
    apply_profile_defaults,
    load_config,
    load_data_config_snapshot,
    load_model_config_snapshot,
    normalize_mixed_precision,
    validate_data_config,
    validate_logging_config,
    validate_model_config,
    validate_optim_config,
    validate_train_config,
    validate_training_workflow_options,
)
from deberta.data.loading import load_hf_dataset
from deberta.export_cli import ExportArgumentDefaultsHelpFormatter, add_export_arguments
from deberta.modeling.builder import build_backbone_configs
from deberta.modeling.mask_utils import normalize_keep_mask
from deberta.modeling.rtd import attention_mask_to_active_tokens
from deberta.training.pretrain import (
    _append_metrics_jsonl_row,
    _apply_lr_mult,
    _apply_nonfinite_recovery,
    _build_decoupled_optimizers,
    _build_optimizer,
    _build_runtime_resolved_tracker_config,
    _build_training_collator,
    _coerce_dataclass_payload_types,
    _count_input_tokens_for_batch,
    _count_rtd_tokens_for_batch,
    _cycle_dataloader,
    _export_discriminator_hf,
    _export_discriminator_hf_subprocess,
    _finalize_window_metric_loss,
    _flush_loggers,
    _full_backbone_hf_inductor_warning,
    _global_grad_l2_norm,
    _has_nonfinite_grad_norm_any_rank,
    _load_resume_state_with_compile_fallback,
    _normalize_resume_consumed_micro_batches,
    _optimizer_param_order_digest,
    _partition_optimizer_params,
    _record_unscaled_lrs,
    _resolve_compile_enabled_or_raise,
    _resolve_compile_scope,
    _resolve_data_resume_policy,
    _resolve_effective_mixed_precision_or_raise,
    _resolve_window_token_denominators,
    _scale_loss_for_backward,
    _should_clip_gradients,
    _should_force_legacy_tf32_for_compile,
    _stabilize_compile_attention_mask,
    _sync_discriminator_embeddings_if_available,
    _token_weighted_micro_objective,
    _write_export_readme,
    run_pretraining_dry_run,
)
from deberta.training.run_config import (
    _build_run_metadata,
    _persist_or_validate_run_configs,
)
from deberta.training.run_management import (
    _find_latest_checkpoint,
    _load_checkpoint_data_progress,
    _load_checkpoint_progress_metadata,
    _prepare_output_dir,
    _resolve_output_dir,
    _resolve_output_dir_for_accelerator,
    _resolve_resume_checkpoint,
    _resolve_resume_checkpoint_for_accelerator,
    _save_checkpoint_data_progress,
    _save_training_checkpoint,
)
from deberta.training.tracker_utils import (
    _init_trackers,
    _setup_wandb_watch,
    _upload_wandb_original_config,
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
    model_nested, data_nested, train_nested = _load_yaml(nested)
    assert model_nested.rope.ffn_type == "swiglu"
    assert data_nested.packing.max_seq_length == 128
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
    with pytest.raises(ValueError, match="Unknown top-level keys in nested YAML config"):
        _load_yaml(flat)


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
    model_nested, data_nested, train_nested = _load_json(nested)
    assert model_nested.rope.ffn_type == "mlp"
    assert data_nested.packing.max_seq_length == 96
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
    with pytest.raises(ValueError, match="Unknown top-level keys in nested JSON config"):
        _load_json(flat)


def test_load_hf_dataset_handles_missing_cache_dir_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def _fake_load_dataset(name: str, **kwargs: Any) -> list[dict[str, str]]:
        calls.append((str(name), dict(kwargs)))
        return [{"text": "ok"}]

    fake_datasets = types.SimpleNamespace(
        load_dataset=_fake_load_dataset,
        load_from_disk=lambda _path: [],
        DatasetDict=dict,
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    cfg = DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy")
    out = load_hf_dataset(cfg=cfg, split="train", streaming=True)

    assert out == [{"text": "ok"}]
    assert calls
    assert calls[0][0] == "hf-internal-testing/librispeech_asr_dummy"
    assert "cache_dir" in calls[0][1]
    assert calls[0][1]["cache_dir"] is None


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
    _, data_cfg, train_cfg = _load_yaml(cfg)
    assert int(data_cfg.packing.max_seq_length) == 256
    assert float(train_cfg.learning_rate) == pytest.approx(5e-4)
    assert train_cfg.run_name == "run-256"


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
        _load_yaml(cfg)


def test_load_config_returns_frozen_top_level_and_sections(tmp_path: Path):
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 1",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert isinstance(cfg, Config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.train = TrainConfig(max_steps=2)  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.optim.scheduler.warmup_steps = 5  # type: ignore[misc]


def test_load_config_supports_extended_sections_and_projects_to_runtime_train(tmp_path: Path):
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
                "  checkpoint:",
                "    save_steps: 777",
                "optim:",
                "  scheduler:",
                "    warmup_steps: 222",
                "logging:",
                "  backend: none",
                "  wandb:",
                "    enabled: true",
                "    watch: all",
                "  debug:",
                "    metrics: true",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert int(cfg.train.warmup_steps) == 222
    assert int(cfg.train.save_steps) == 777
    assert str(cfg.train.report_to) == "wandb"
    assert str(cfg.train.wandb_watch) == "all"
    assert bool(cfg.train.debug_metrics) is True


def test_load_config_rejects_string_boolean_for_data_streaming(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "bad_bool.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "    streaming: 'false'",
                "train:",
                "  max_steps: 1",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="data.source.streaming must be a boolean"):
        load_config(cfg_path)


def test_load_config_rejects_string_boolean_for_token_weighted_gradient_accumulation(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "bad_bool_train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 1",
                "  token_weighted_gradient_accumulation: 'false'",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train.token_weighted_gradient_accumulation must be a boolean"):
        load_config(cfg_path)


def test_apply_dotted_override_supports_nested_section_paths() -> None:
    cfg = Config(
        data=DataConfig(source={"dataset_name": "HuggingFaceFW/fineweb-edu"}),
        train=TrainConfig(max_steps=1),
    )
    cfg2 = apply_dotted_override(cfg, "logging.wandb.watch=all")
    cfg2 = apply_dotted_override(cfg2, "optim.scheduler.warmup_steps=123")
    assert cfg2.logging.wandb.watch == "all"
    assert int(cfg2.optim.scheduler.warmup_steps) == 123
    assert int(cfg2.train.warmup_steps) == 123


def test_apply_dotted_override_preserves_existing_explicit_fields_per_section() -> None:
    cfg = Config()
    cfg = apply_dotted_override(cfg, "train.objective.mask_token_prob=0.8")
    cfg = apply_dotted_override(cfg, "train.max_steps=20")

    explicit_train_fields = set(getattr(cfg.train, "_explicit_fields", set()))
    assert "objective.mask_token_prob" in explicit_train_fields
    assert "max_steps" in explicit_train_fields

    apply_profile_defaults(model_cfg=cfg.model, train_cfg=cfg.train, optim_cfg=cfg.optim)
    assert cfg.train.mask_token_prob == pytest.approx(0.8)


def test_load_json_unknown_key_raises(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"unknown_field": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown top-level keys in nested JSON config"):
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


@pytest.mark.parametrize(
    "config_name",
    [
        "pretrain_hf_deberta_v2_parity_base.yaml",
        "pretrain_hf_deberta_v2_parity_small.yaml",
    ],
)
def test_parity_yaml_configs_parse_and_validate(
    config_name: str,
) -> None:
    pytest.importorskip("yaml")

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / config_name
    model_cfg, data_cfg, train_cfg = _load_yaml(config_path)
    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg)

    validate_model_config(model_cfg)
    validate_data_config(data_cfg)
    validate_train_config(train_cfg)
    validate_training_workflow_options(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
    )

    assert model_cfg.profile == "deberta_v3_parity"
    assert model_cfg.backbone_type == "hf_deberta_v2"
    assert model_cfg.pretrained_discriminator_path == ""
    assert bool(train_cfg.decoupled_training) is True


def test_load_model_config_snapshot_rejects_unknown_legacy_key() -> None:
    with pytest.raises(ValueError, match="Unsupported model_config.json keys"):
        load_model_config_snapshot(
            {"backbone_type": "rope", "legacy_field": 1},
            source="model_config.json",
        )


def test_load_data_config_snapshot_rejects_unknown_legacy_key() -> None:
    with pytest.raises(ValueError, match="Unsupported data_config.json keys"):
        load_data_config_snapshot(
            {"dataset_name": "HuggingFaceFW/fineweb-edu", "legacy_field": 1},
            source="data_config.json",
        )


def test_load_model_config_snapshot_rejects_missing_required_key() -> None:
    model_raw = asdict(ModelConfig())
    model_raw.pop("backbone_type")
    with pytest.raises(ValueError, match="Missing required model_config.json keys"):
        load_model_config_snapshot(model_raw, source="model_config.json")


def test_load_data_config_snapshot_rejects_missing_required_key() -> None:
    data_raw = asdict(DataConfig())
    data_raw.pop("source")
    with pytest.raises(ValueError, match="Missing required data_config.json keys"):
        load_data_config_snapshot(data_raw, source="data_config.json")


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


def test_prepare_output_dir_rejects_blank_resume_hint_on_nonempty_dir(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "existing.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Output directory exists and is not empty"):
        _prepare_output_dir(
            output_dir=out,
            overwrite_output_dir=False,
            resume_from_checkpoint="   ",
            is_main_process=True,
        )


def test_find_latest_checkpoint_picks_highest_step(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoint-2").mkdir()
    (out / "checkpoint-11").mkdir()
    (out / "checkpoint-4").mkdir()

    latest = _find_latest_checkpoint(out)
    assert latest is not None
    assert latest.name == "checkpoint-11"


def test_resolve_output_dir_auto_uses_project_and_config_stem():
    out = _resolve_output_dir(
        output_dir=None,
        project_name="deberta-train",
        config_path="configs/pretrain_rope_fineweb_edu.yaml",
    )
    assert out.parent.name == "deberta-train"
    assert re.fullmatch(r"\d{8}_\d{6}_pretrain_rope_fineweb_edu", out.name) is not None


def test_resolve_output_dir_auto_prefers_run_name():
    out = _resolve_output_dir(
        output_dir=None,
        project_name="deberta-train",
        config_path="configs/pretrain_rope_fineweb_edu.yaml",
        run_name="my run",
    )
    assert out.parent.name == "deberta-train"
    assert re.fullmatch(r"\d{8}_\d{6}_my-run", out.name) is not None


def test_resolve_output_dir_keeps_explicit_path():
    out = _resolve_output_dir(
        output_dir="runs/custom/run-01",
        project_name="ignored",
        config_path="configs/pretrain_rope_fineweb_edu.yaml",
    )
    assert out == Path("runs/custom/run-01")


def _accel_stub(*, is_main_process: bool, num_processes: int) -> Any:
    """Return a minimal accelerator-like object for broadcast helper tests."""
    return types.SimpleNamespace(is_main_process=bool(is_main_process), num_processes=int(num_processes))


def test_resolve_output_dir_for_accelerator_keeps_explicit_path():
    called = {"count": 0}

    def _fake_broadcast(payload: list[str | None], *, from_process: int = 0) -> None:
        del payload, from_process
        called["count"] += 1

    out = _resolve_output_dir_for_accelerator(
        accelerator=_accel_stub(is_main_process=False, num_processes=8),
        output_dir="runs/custom/run-02",
        project_name="ignored",
        config_path="cfg.yaml",
        broadcast_fn=_fake_broadcast,
    )
    assert out == Path("runs/custom/run-02")
    assert called["count"] == 0


def test_resolve_output_dir_for_accelerator_uses_broadcasted_auto_value():
    def _fake_broadcast(payload: list[str | None], *, from_process: int = 0) -> None:
        assert from_process == 0
        payload[0] = "runs/demo/20260101_010101_shared"

    out = _resolve_output_dir_for_accelerator(
        accelerator=_accel_stub(is_main_process=False, num_processes=2),
        output_dir=None,
        project_name="demo",
        config_path="cfg.yaml",
        broadcast_fn=_fake_broadcast,
    )
    assert out == Path("runs/demo/20260101_010101_shared")


def test_resolve_resume_checkpoint_for_accelerator_uses_rank0_broadcast_value(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)

    def _fake_broadcast(payload: list[dict[str, Any]], *, from_process: int = 0) -> None:
        assert from_process == 0
        payload[0] = {
            "ok": True,
            "value": str((out / "checkpoint-9").resolve()),
            "error_type": None,
            "error_message": None,
        }

    resolved = _resolve_resume_checkpoint_for_accelerator(
        accelerator=_accel_stub(is_main_process=False, num_processes=4),
        output_dir=out,
        resume_from_checkpoint="auto",
        broadcast_fn=_fake_broadcast,
    )
    assert resolved == str((out / "checkpoint-9").resolve())


def test_resolve_resume_checkpoint_for_accelerator_propagates_rank0_error(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)

    def _fake_broadcast(payload: list[dict[str, Any]], *, from_process: int = 0) -> None:
        assert from_process == 0
        payload[0] = {
            "ok": False,
            "value": None,
            "error_type": "ValueError",
            "error_message": "rank0 failed",
        }

    with pytest.raises(ValueError, match="rank0 failed"):
        _resolve_resume_checkpoint_for_accelerator(
            accelerator=_accel_stub(is_main_process=False, num_processes=2),
            output_dir=out,
            resume_from_checkpoint="auto",
            broadcast_fn=_fake_broadcast,
        )


def test_resolve_resume_checkpoint_auto_returns_none_when_no_checkpoint(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_resume_checkpoint(
        output_dir=out,
        resume_from_checkpoint="auto",
        is_main_process=True,
    )
    assert resolved is None


def test_resolve_resume_checkpoint_auto_rejects_non_empty_output_dir_without_checkpoints(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "stale.txt").write_text("leftover", encoding="utf-8")

    with pytest.raises(ValueError, match="resume_from_checkpoint=auto was requested"):
        _resolve_resume_checkpoint(
            output_dir=out,
            resume_from_checkpoint="auto",
            is_main_process=True,
        )


@pytest.mark.parametrize(
    ("case", "exc_type", "match"),
    [
        ("missing_path", FileNotFoundError, "checkpoint path does not exist"),
        ("not_directory", ValueError, "must point to a checkpoint directory"),
        ("missing_complete", ValueError, "missing .complete marker"),
        ("missing_data_state", ValueError, "missing/invalid data_state.json"),
        ("missing_weights", ValueError, "model weights appear missing or empty"),
    ],
)
def test_resolve_resume_checkpoint_rejects_invalid_explicit_path_states(
    tmp_path: Path,
    mock_checkpoint: Any,
    case: str,
    exc_type: type[Exception],
    match: str,
):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    if case == "missing_path":
        target = out / "checkpoint-missing"
    elif case == "not_directory":
        target = out / "not-a-dir.txt"
        target.write_text("x", encoding="utf-8")
    elif case == "missing_complete":
        target = mock_checkpoint(
            root=out,
            name="checkpoint-2",
            with_weights=True,
            with_data_state=True,
            with_complete=False,
        )
    elif case == "missing_data_state":
        target = mock_checkpoint(
            root=out,
            name="checkpoint-2",
            with_weights=True,
            with_data_state=False,
            with_complete=True,
        )
    elif case == "missing_weights":
        target = mock_checkpoint(
            root=out,
            name="checkpoint-2",
            with_weights=False,
            with_data_state=True,
            with_complete=True,
        )
    else:  # pragma: no cover
        raise AssertionError(f"Unsupported case: {case}")

    with pytest.raises(exc_type, match=match):
        _resolve_resume_checkpoint(
            output_dir=out,
            resume_from_checkpoint=str(target),
            is_main_process=True,
        )


def test_resolve_resume_checkpoint_returns_resolved_explicit_path(tmp_path: Path, mock_checkpoint: Any):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = mock_checkpoint(root=out, name="checkpoint-2", consumed_micro_batches=2)
    resolved = _resolve_resume_checkpoint(
        output_dir=out,
        resume_from_checkpoint=str(ckpt),
        is_main_process=True,
    )
    assert resolved == str(ckpt.resolve())


def test_resolve_resume_checkpoint_auto_skips_latest_non_resumable_checkpoint(
    tmp_path: Path, mock_checkpoint: Any
):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)

    ckpt1 = mock_checkpoint(root=out, name="checkpoint-1", consumed_micro_batches=10)

    # Simulate interrupted checkpoint write: directory exists but metadata was never written.
    ckpt2 = out / "checkpoint-2"
    ckpt2.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_resume_checkpoint(
        output_dir=out,
        resume_from_checkpoint="auto",
        is_main_process=True,
    )
    assert resolved == str(ckpt1.resolve())


def test_resolve_resume_checkpoint_auto_rejects_when_all_checkpoints_non_resumable(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoint-1").mkdir(parents=True, exist_ok=True)
    (out / "checkpoint-2").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="none are resumable"):
        _resolve_resume_checkpoint(
            output_dir=out,
            resume_from_checkpoint="auto",
            is_main_process=True,
        )


def test_checkpoint_data_progress_roundtrip(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir(parents=True, exist_ok=True)

    consumed, lr_mult, digest = _load_checkpoint_data_progress(ckpt)
    assert consumed is None
    assert lr_mult == 1.0
    assert digest is None

    _save_checkpoint_data_progress(checkpoint_dir=ckpt, consumed_micro_batches=123, lr_mult=0.25)
    consumed, lr_mult, digest = _load_checkpoint_data_progress(ckpt)
    assert consumed == 123
    assert abs(lr_mult - 0.25) < 1e-9
    assert digest is None  # no digest was saved

    # With optimizer param digest.
    _save_checkpoint_data_progress(
        checkpoint_dir=ckpt,
        consumed_micro_batches=200,
        lr_mult=0.5,
        optimizer_param_digest="abc123deadbeef00",
        global_step=17,
        gradient_accumulation_steps=4,
    )
    consumed, lr_mult, digest = _load_checkpoint_data_progress(ckpt)
    assert consumed == 200
    assert abs(lr_mult - 0.5) < 1e-9
    assert digest == "abc123deadbeef00"
    _, _, _, global_step, saved_ga = _load_checkpoint_progress_metadata(ckpt)
    assert global_step == 17
    assert saved_ga == 4

    # Back-compat: old checkpoints without lr_mult or digest default gracefully.
    import json

    (ckpt / "data_state.json").write_text(json.dumps({"consumed_micro_batches": 50}))
    consumed_old, lr_mult_old, digest_old = _load_checkpoint_data_progress(ckpt)
    assert consumed_old == 50
    assert lr_mult_old == 1.0
    assert digest_old is None


def test_checkpoint_data_progress_roundtrip_with_dual_optimizer_digest(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-12"
    ckpt.mkdir(parents=True, exist_ok=True)

    dual_digest = {"generator": "aaaabbbbccccdddd", "discriminator": "1111222233334444"}
    _save_checkpoint_data_progress(
        checkpoint_dir=ckpt,
        consumed_micro_batches=77,
        lr_mult=0.75,
        optimizer_param_digest=dual_digest,
        global_step=9,
        gradient_accumulation_steps=3,
    )

    consumed, lr_mult, digest = _load_checkpoint_data_progress(ckpt)
    assert consumed == 77
    assert lr_mult == pytest.approx(0.75)
    assert isinstance(digest, dict)
    assert digest == dual_digest

    _, _, digest_meta, saved_step, saved_ga = _load_checkpoint_progress_metadata(ckpt)
    assert isinstance(digest_meta, dict)
    assert digest_meta == dual_digest
    assert saved_step == 9
    assert saved_ga == 3


def test_load_checkpoint_data_progress_warns_on_invalid_json(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ckpt = tmp_path / "checkpoint-7"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "data_state.json").write_text('{"consumed_micro_batches": ', encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        consumed, lr_mult, digest = _load_checkpoint_data_progress(ckpt)

    assert consumed is None
    assert lr_mult == 1.0
    assert digest is None
    assert any("invalid data_state.json" in record.message for record in caplog.records)


def test_dump_json_is_atomic_on_serialization_failure(tmp_path: Path) -> None:
    from deberta.io_utils import dump_json

    target = tmp_path / "state.json"
    with pytest.raises(TypeError):
        dump_json({"bad": {1, 2, 3}}, target)

    assert not target.exists()
    tmp_files = list(tmp_path.glob(".*state.json.*.tmp"))
    assert not tmp_files


def test_optimizer_param_order_digest_deterministic() -> None:
    """Same model produces the same digest across calls."""
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    d1 = _optimizer_param_order_digest(m)
    d2 = _optimizer_param_order_digest(m)
    assert d1 == d2
    assert len(d1) == 16  # 16-char hex prefix


def test_optimizer_param_order_digest_changes_on_different_params() -> None:
    """Different parameter names produce a different digest."""
    m1 = torch.nn.ModuleDict({"alpha": torch.nn.Linear(4, 4), "beta": torch.nn.Linear(4, 2)})
    m2 = torch.nn.ModuleDict({"gamma": torch.nn.Linear(4, 4), "beta": torch.nn.Linear(4, 2)})
    d1 = _optimizer_param_order_digest(m1)
    d2 = _optimizer_param_order_digest(m2)
    assert d1 != d2, "Different param names must produce different digest"


def test_optimizer_param_order_digest_ignores_frozen_params() -> None:
    """Frozen parameters are excluded from the digest."""
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    d_all = _optimizer_param_order_digest(m)
    m[0].weight.requires_grad_(False)
    m[0].bias.requires_grad_(False)
    d_partial = _optimizer_param_order_digest(m)
    assert d_all != d_partial, "Freezing params changes the trainable param set and digest"


def test_partition_optimizer_params_deduplicates_shared_parameters() -> None:
    class _SharedRTDLike(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = torch.nn.Module()
            self.discriminator = torch.nn.Module()
            shared = torch.nn.Parameter(torch.randn(3, 3))
            self.generator.shared = shared
            self.discriminator.shared = shared

    model = _SharedRTDLike()
    parts = _partition_optimizer_params(model)
    ordered = ("gen_decay", "gen_no_decay", "disc_decay", "disc_no_decay")
    params = [p for key in ordered for p in parts[key]["params"]]
    names = [n for key in ordered for n in parts[key]["names"]]

    assert len(params) == 1
    assert len(names) == 1
    assert len({id(p) for p in params}) == 1


def test_optimizer_param_order_digest_matches_optimizer_group_insertion_order() -> None:
    model = TinyRTDLikeModel()
    cfg = TrainConfig()
    opt = _build_optimizer(model, cfg)

    param_to_name = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    ordered_names: list[str] = []
    for group in opt.param_groups:
        ordered_names.extend(param_to_name[id(p)] for p in group["params"])
    expected = hashlib.sha256("\n".join(ordered_names).encode()).hexdigest()[:16]

    assert _optimizer_param_order_digest(model) == expected
    assert str(opt._param_order_digest) == expected


def test_build_decoupled_optimizers_uses_branch_lrs_and_tracks_digests() -> None:
    model = TinyRTDLikeModel()
    cfg = TrainConfig(
        learning_rate=5.0e-4,
        generator_learning_rate=2.5e-4,
        weight_decay=0.01,
        mixed_precision="no",
    )
    gen_opt, disc_opt = _build_decoupled_optimizers(model, cfg, mixed_precision="no")

    assert gen_opt.param_groups
    assert disc_opt.param_groups
    for group in gen_opt.param_groups:
        assert float(group["lr"]) == pytest.approx(2.5e-4)
    for group in disc_opt.param_groups:
        assert float(group["lr"]) == pytest.approx(5.0e-4)

    assert isinstance(getattr(gen_opt, "_param_order_digest", ""), str)
    assert isinstance(getattr(disc_opt, "_param_order_digest", ""), str)
    assert len(str(gen_opt._param_order_digest)) == 16
    assert len(str(disc_opt._param_order_digest)) == 16
    assert str(gen_opt._param_order_digest) != str(disc_opt._param_order_digest)


def test_build_decoupled_optimizers_assigns_enhanced_mask_decoder_to_generator_optimizer() -> None:
    class _ModelWithEmd(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = torch.nn.Linear(8, 8)
            self.generator_lm_head = torch.nn.Linear(8, 8)
            self.enhanced_mask_decoder = torch.nn.Linear(8, 8)
            self.discriminator = torch.nn.Linear(8, 8)

    model = _ModelWithEmd()
    cfg = TrainConfig(
        learning_rate=5.0e-4,
        generator_learning_rate=2.5e-4,
        weight_decay=0.01,
        mixed_precision="no",
    )
    gen_opt, disc_opt = _build_decoupled_optimizers(model, cfg, mixed_precision="no")

    gen_param_ids = {id(p) for group in gen_opt.param_groups for p in group["params"]}
    disc_param_ids = {id(p) for group in disc_opt.param_groups for p in group["params"]}
    emd_param_ids = {id(p) for p in model.enhanced_mask_decoder.parameters() if p.requires_grad}

    assert emd_param_ids
    assert emd_param_ids.issubset(gen_param_ids)
    assert emd_param_ids.isdisjoint(disc_param_ids)


def test_save_training_checkpoint_persists_optimizer_digest(tmp_path: Path):
    """_save_training_checkpoint forwards optimizer_param_digest to data_state.json."""
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint-5"

    accel = _checkpoint_saving_accelerator(is_main_process=True)
    _save_training_checkpoint(
        accelerator=accel,
        checkpoint_dir=ckpt,
        output_dir=out,
        consumed_micro_batches=10,
        save_total_limit=3,
        log_label="test",
        optimizer_param_digest="deadbeef12345678",
    )
    _, _, digest = _load_checkpoint_data_progress(ckpt)
    assert digest == "deadbeef12345678"


def test_save_training_checkpoint_persists_dual_optimizer_digest(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint-6"

    dual_digest = {"generator": "feedfacecafebeef", "discriminator": "baadf00d12345678"}
    accel = _checkpoint_saving_accelerator(is_main_process=True)
    _save_training_checkpoint(
        accelerator=accel,
        checkpoint_dir=ckpt,
        output_dir=out,
        consumed_micro_batches=15,
        save_total_limit=3,
        log_label="test",
        optimizer_param_digest=dual_digest,
    )
    _, _, digest = _load_checkpoint_data_progress(ckpt)
    assert isinstance(digest, dict)
    assert digest == dual_digest


def test_canonical_compile_state_key_strips_orig_mod_segments() -> None:
    assert canonical_compile_state_key("generator._orig_mod.encoder.layer.0._orig_mod.weight") == (
        "generator.encoder.layer.0.weight"
    )
    assert canonical_compile_state_key("_orig_mod.generator._orig_mod.encoder.weight") == (
        "generator.encoder.weight"
    )


def test_load_model_state_with_compile_key_remap_matches_checkpoint_with_orig_mod_keys(
    tmp_path: Path,
) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model[0].weight.fill_(1.5)
        model[0].bias.fill_(-0.25)

    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    remapped = {key.replace("0.", "0._orig_mod."): value.detach().clone() for key, value in original.items()}
    torch.save(remapped, checkpoint / "model.bin")

    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()

    stats = load_model_state_with_compile_key_remap(model, checkpoint)
    assert stats == {"matched": 2}
    assert torch.allclose(model[0].weight, original["0.weight"])
    assert torch.allclose(model[0].bias, original["0.bias"])


def test_load_model_state_with_compile_key_remap_matches_top_level_orig_mod_keys(
    tmp_path: Path,
) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model[0].weight.fill_(0.5)
        model[0].bias.fill_(0.125)

    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    remapped = {f"_orig_mod.{key}": value.detach().clone() for key, value in original.items()}
    torch.save(remapped, checkpoint / "model.bin")

    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()

    stats = load_model_state_with_compile_key_remap(model, checkpoint)
    assert stats == {"matched": 2}
    assert torch.allclose(model[0].weight, original["0.weight"])
    assert torch.allclose(model[0].bias, original["0.bias"])


def test_load_model_state_with_compile_key_remap_supports_fsdp_sharded_checkpoint(
    tmp_path: Path,
) -> None:
    dcp = pytest.importorskip("torch.distributed.checkpoint")

    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    checkpoint = tmp_path / "checkpoint-1"
    shard_dir = checkpoint / "pytorch_model_fsdp_0"
    shard_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model[0].weight.fill_(1.25)
        model[0].bias.fill_(-0.75)

    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sharded_saved = {k.replace("0.", "0._orig_mod."): v.clone() for k, v in saved.items()}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to save in a single process.",
            category=UserWarning,
        )
        dcp.save(
            state_dict={"model": sharded_saved},
            checkpoint_id=str(shard_dir),
            no_dist=True,
        )

    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process.",
            category=UserWarning,
        )
        loaded_direct = load_checkpoint_model_state_dict(checkpoint, model_state_template=model.state_dict())
    assert set(loaded_direct.keys()) == set(sharded_saved.keys())

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process.",
            category=UserWarning,
        )
        stats = load_model_state_with_compile_key_remap(model, checkpoint)
    assert stats == {"matched": 2}
    assert torch.allclose(model[0].weight, saved["0.weight"])
    assert torch.allclose(model[0].bias, saved["0.bias"])


def test_load_resume_state_with_compile_fallback_retries_strict_false_and_remaps(
    tmp_path: Path,
) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model[0].weight.fill_(0.75)
        model[0].bias.fill_(0.5)
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(
        {k.replace("0.", "0._orig_mod."): v.clone() for k, v in saved.items()},
        checkpoint / "model.bin",
    )

    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()

    calls: list[tuple[str, dict[str, Any]]] = []

    def _load_state_hook(ckpt: str, kwargs: dict[str, Any]) -> None:
        calls.append((str(ckpt), dict(kwargs)))
        if kwargs.get("strict", True):
            raise RuntimeError("Error(s) in loading state_dict with _orig_mod mismatch")

    accel = FakeAccelerator(load_state_hook=_load_state_hook)
    _load_resume_state_with_compile_fallback(accel, model, str(checkpoint))

    assert len(calls) == 2
    assert calls[0][1] == {}
    assert calls[1][1] == {"strict": False}
    assert torch.allclose(model[0].weight, saved["0.weight"])
    assert torch.allclose(model[0].bias, saved["0.bias"])


def test_append_metrics_jsonl_row_appends_rows(tmp_path: Path):
    metrics_path = tmp_path / "run" / "metrics.jsonl.gz"
    _append_metrics_jsonl_row(metrics_path, {"step": 1, "loss": 0.123})
    _append_metrics_jsonl_row(metrics_path, {"step": 2, "crash": True, "crash_type": "KeyboardInterrupt"})

    with gzip.open(metrics_path, "rt", encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    row1 = json.loads(lines[1])
    assert row0["step"] == 1
    assert row1["step"] == 2
    assert row1["crash"] is True


def test_flush_loggers_suppresses_handler_flush_errors() -> None:
    class _CountingHandler(logging.Handler):
        def __init__(self, *, raise_on_flush: bool) -> None:
            super().__init__()
            self.raise_on_flush = bool(raise_on_flush)
            self.flush_calls = 0

        def emit(self, record: logging.LogRecord) -> None:
            del record

        def flush(self) -> None:
            self.flush_calls += 1
            if self.raise_on_flush:
                raise RuntimeError("boom")

    test_logger = logging.getLogger("deberta.tests.flush")
    test_logger.setLevel(logging.INFO)
    ok_handler = _CountingHandler(raise_on_flush=False)
    bad_handler = _CountingHandler(raise_on_flush=True)
    test_logger.addHandler(ok_handler)
    test_logger.addHandler(bad_handler)
    try:
        _flush_loggers()
    finally:
        test_logger.removeHandler(ok_handler)
        test_logger.removeHandler(bad_handler)

    assert ok_handler.flush_calls >= 1
    assert bad_handler.flush_calls >= 1


def test_init_trackers_passes_wandb_name_with_wrapped_signature() -> None:
    class _WrappedAccelerator:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def init_trackers(self, *args, **kwargs) -> None:
            del args
            self.calls.append(dict(kwargs))

    accel = _WrappedAccelerator()
    _init_trackers(
        accelerator=accel,
        project_name="demo-project",
        tracker_cfg={"a": 1},
        report_to="wandb",
        run_name="demo-run",
    )

    assert accel.calls
    first = accel.calls[0]
    assert first["project_name"] == "demo-project"
    assert first["init_kwargs"]["wandb"]["name"] == "demo-run"


def test_init_trackers_falls_back_without_init_kwargs(caplog: pytest.LogCaptureFixture) -> None:
    class _LegacyAccelerator:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def init_trackers(self, *, project_name: str, config: dict[str, Any]) -> None:
            self.calls.append({"project_name": project_name, "config": dict(config)})

    accel = _LegacyAccelerator()
    with caplog.at_level(logging.WARNING):
        _init_trackers(
            accelerator=accel,
            project_name="demo-project",
            tracker_cfg={"a": 1},
            report_to="wandb",
            run_name="demo-run",
        )

    assert accel.calls
    assert "rejected init_kwargs" in caplog.text


def test_setup_wandb_watch_calls_watch_with_mode_and_frequency() -> None:
    model = torch.nn.Linear(4, 4)
    run = FakeWandbRun()
    enabled = _setup_wandb_watch(
        accelerator=FakeAccelerator(),
        wandb_run=run,
        model=model,
        watch_mode="gradients",
        watch_log_freq=123,
    )
    assert enabled is True
    assert run.watch_calls
    watched_model, kwargs = run.watch_calls[0]
    assert watched_model is model
    assert kwargs["log"] == "gradients"
    assert kwargs["log_freq"] == 123


def test_setup_wandb_watch_returns_false_when_mode_none() -> None:
    run = FakeWandbRun()
    enabled = _setup_wandb_watch(
        accelerator=FakeAccelerator(),
        wandb_run=run,
        model=torch.nn.Linear(2, 2),
        watch_mode="none",
        watch_log_freq=100,
    )
    assert enabled is False
    assert run.watch_calls == []


def test_upload_wandb_original_config_stages_with_expected_filename(tmp_path: Path) -> None:
    src = tmp_path / "config_original.yaml"
    src.write_text("train:\n  max_steps: 1\n", encoding="utf-8")

    run = FakeWandbRun()
    uploaded = _upload_wandb_original_config(
        accelerator=types.SimpleNamespace(is_main_process=True),
        wandb_run=run,
        config_original_path=src,
        run_name="demo-run",
    )
    assert uploaded is True
    assert run.saved_paths
    assert run.saved_paths[0].name == "config_original.yaml"
    assert run.saved_paths[0] == src


def test_upload_wandb_original_config_uploads_resolved_and_source_files(tmp_path: Path) -> None:
    src_original = tmp_path / "config_original.yaml"
    src_original.write_text("model:\n  backbone_type: hf_deberta_v2\n", encoding="utf-8")
    src_resolved = tmp_path / "config_resolved.yaml"
    src_resolved.write_text(
        "model:\n  backbone_type: hf_deberta_v2\ntrain:\n  warmup_steps: 10000\n", encoding="utf-8"
    )
    src_source = tmp_path / "passed.yaml"
    src_source.write_text("model:\n  profile: deberta_v3_parity\n", encoding="utf-8")

    run = FakeWandbRun()
    uploaded = _upload_wandb_original_config(
        accelerator=types.SimpleNamespace(is_main_process=True),
        wandb_run=run,
        config_original_path=src_original,
        config_resolved_path=src_resolved,
        config_source_path=src_source,
        run_name="demo-run",
    )
    assert uploaded is True
    saved_names = {p.name for p in run.saved_paths}
    assert "config_original.yaml" not in saved_names
    assert "config_resolved.yaml" in saved_names
    assert "passed.yaml" in saved_names


def test_coerce_dataclass_payload_types_accepts_mapping_inputs() -> None:
    payload = {
        "alpha": 1,
        "nested": {"beta": True},
        3: "non-string-key",
    }
    coerced = _coerce_dataclass_payload_types(payload)
    assert coerced == {
        "alpha": 1,
        "nested": {"beta": True},
        "3": "non-string-key",
    }


def test_build_runtime_resolved_tracker_config_populates_effective_values_and_prunes_none() -> None:
    model_cfg = ModelConfig(
        profile="deberta_v3_parity",
        backbone_type="hf_deberta_v2",
        pretrained_discriminator_path="microsoft/deberta-v3-base",
        generator_num_hidden_layers=None,
        hidden_dropout_prob=None,
        attention_probs_dropout_prob=None,
        tokenizer_vocab_target=None,
    )
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig(
        learning_rate=5e-4,
        generator_learning_rate=-1.0,
        discriminator_learning_rate=-1.0,
    )
    optim_cfg = OptimConfig(
        learning_rate=5e-4, generator_learning_rate=-1.0, discriminator_learning_rate=-1.0
    )
    logging_cfg = LoggingConfig()
    disc_cfg = types.SimpleNamespace(
        num_hidden_layers=12,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        to_dict=lambda: {
            "num_hidden_layers": 12,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
        },
    )
    gen_cfg = types.SimpleNamespace(
        num_hidden_layers=6,
        hidden_size=384,
        intermediate_size=1536,
        num_attention_heads=6,
        to_dict=lambda: {
            "num_hidden_layers": 6,
            "hidden_size": 384,
            "intermediate_size": 1536,
            "num_attention_heads": 6,
        },
    )
    tokenizer = DummyTokenizer(vocab_size=32000)

    payload = _build_runtime_resolved_tracker_config(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        optim_cfg=optim_cfg,
        logging_cfg=logging_cfg,
        disc_config=disc_cfg,
        gen_config=gen_cfg,
        tokenizer=tokenizer,
    )

    assert payload["model"]["generator"]["num_hidden_layers"] == 6
    assert payload["model"]["generator"]["hidden_size"] == 384
    assert payload["model"]["generator"]["intermediate_size"] == 1536
    assert payload["model"]["generator"]["num_attention_heads"] == 6
    assert payload["model"]["dropout"]["hidden_prob"] == pytest.approx(0.1)
    assert payload["model"]["dropout"]["attention_probs_prob"] == pytest.approx(0.1)
    assert payload["model"]["tokenizer"]["vocab_target"] == 32000
    assert payload["optim"]["lr"]["base"] == pytest.approx(5e-4)
    assert payload["model"]["pretrained"]["discriminator_path"] == "microsoft/deberta-v3-base"
    assert "generator_path" not in payload["model"]["pretrained"]
    assert "resume_from_checkpoint" not in payload["train"]["checkpoint"]
    assert set(payload.keys()) == {"model", "data", "train", "optim", "logging"}
    assert "effective" not in payload


def test_build_runtime_resolved_tracker_config_omits_effective_backbone_payload() -> None:
    model_cfg = ModelConfig(profile="deberta_v3_parity", backbone_type="hf_deberta_v2")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig()
    disc_cfg = types.SimpleNamespace(
        to_dict=lambda: {
            "model_type": "deberta-v2",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "vocab_size": 32000,
            "max_position_embeddings": 1024,
            "pad_token_id": 3,
            "max_length": 20,
            "top_k": 50,
            "id2label": {"0": "LABEL_0"},
            "_name_or_path": "",
        },
    )
    gen_cfg = types.SimpleNamespace(
        to_dict=lambda: {
            "model_type": "deberta-v2",
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "vocab_size": 32000,
            "max_position_embeddings": 1024,
            "pad_token_id": 3,
            "max_length": 20,
            "top_k": 50,
            "id2label": {"0": "LABEL_0"},
            "_name_or_path": "",
        },
    )
    tokenizer = DummyTokenizer(vocab_size=32000)

    payload = _build_runtime_resolved_tracker_config(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        disc_config=disc_cfg,
        gen_config=gen_cfg,
        tokenizer=tokenizer,
    )

    assert set(payload.keys()) == {"model", "data", "train", "optim", "logging"}
    assert "effective" not in payload


def test_build_runtime_resolved_tracker_config_coerces_numeric_strings() -> None:
    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        hidden_size="768",  # type: ignore[arg-type]
        num_hidden_layers="12",  # type: ignore[arg-type]
        num_attention_heads="12",  # type: ignore[arg-type]
        intermediate_size="3072",  # type: ignore[arg-type]
        hidden_dropout_prob="0.0",  # type: ignore[arg-type]
        attention_probs_dropout_prob="0.0",  # type: ignore[arg-type]
    )
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu", max_seq_length="1024")  # type: ignore[arg-type]
    train_cfg = TrainConfig(
        token_weighted_gradient_accumulation="true",  # type: ignore[arg-type]
    )
    optim_cfg = OptimConfig(
        learning_rate="5e-4",  # type: ignore[arg-type]
        adam_epsilon="1e-6",  # type: ignore[arg-type]
        warmup_steps="1000",  # type: ignore[arg-type]
    )
    disc_cfg = types.SimpleNamespace(
        to_dict=lambda: {
            "num_hidden_layers": 12,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 1024,
        },
    )
    gen_cfg = types.SimpleNamespace(
        to_dict=lambda: {
            "num_hidden_layers": 6,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
        },
    )
    tokenizer = DummyTokenizer(vocab_size=32000)

    payload = _build_runtime_resolved_tracker_config(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        optim_cfg=optim_cfg,
        disc_config=disc_cfg,
        gen_config=gen_cfg,
        tokenizer=tokenizer,
    )

    assert payload["model"]["rope"]["hidden_size"] == 768
    assert payload["data"]["packing"]["max_seq_length"] == 1024
    assert payload["optim"]["scheduler"]["warmup_steps"] == 1000
    assert payload["optim"]["lr"]["base"] == pytest.approx(5e-4)
    assert payload["optim"]["adam"]["epsilon"] == pytest.approx(1e-6)
    assert payload["train"]["token_weighted_gradient_accumulation"] is True


def test_run_pretraining_keyboard_interrupt_logs_crash_and_finishes_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes import _PRETRAINING_BATCH

    saved_checkpoints: list[tuple[str, int, str]] = []

    def _fake_save_checkpoint(
        *,
        accelerator,
        checkpoint_dir,
        output_dir,
        consumed_micro_batches,
        save_total_limit,
        log_label,
        **kwargs,
    ):
        del accelerator, output_dir, save_total_limit, kwargs
        saved_checkpoints.append((str(checkpoint_dir), int(consumed_micro_batches), str(log_label)))

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        save_checkpoint_fn=_fake_save_checkpoint,
        extra_patches={"_export_discriminator_hf": lambda **kwargs: None},
    )

    # Override cycle to interrupt after first batch.
    def _interrupt_cycle(_loader, *, start_epoch: int = 0):
        del start_epoch
        yield _PRETRAINING_BATCH
        raise KeyboardInterrupt

    monkeypatch.setattr(pretrain_mod, "_cycle_dataloader", _interrupt_cycle)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=2,
        save_steps=0,
        report_to="wandb",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        torch_compile=False,
        export_hf_final=False,
    )

    with pytest.raises(KeyboardInterrupt):
        pretrain_mod.run_pretraining(
            model_cfg=ModelConfig(),
            data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
            train_cfg=train_cfg,
        )

    metrics_path = Path(train_cfg.output_dir) / "metrics.jsonl.gz"
    with gzip.open(metrics_path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f.read().splitlines()]
    assert rows and rows[-1]["crash"] is True
    assert rows[-1]["crash_type"] == "KeyboardInterrupt"
    assert int(rows[-1]["step"]) == 1

    assert saved_checkpoints
    assert saved_checkpoints[-1][0].endswith("checkpoint-1")
    assert saved_checkpoints[-1][1] == 1
    assert saved_checkpoints[-1][2] == "final"

    accel = FakeAccelerator.last_instance
    assert accel is not None
    assert accel.wandb_run.summary["crashed"] is True
    assert accel.wandb_run.summary["crash_type"] == "KeyboardInterrupt"
    assert accel.wandb_run.finished_exit_code == 130
    assert accel.wandb_run.logged
    assert accel.wandb_run.watch_calls
    assert accel.wandb_run.saved_paths
    saved_names = {p.name for p in accel.wandb_run.saved_paths}
    assert "config_original.yaml" in saved_names
    assert "config_resolved.yaml" in saved_names
    assert accel.wandb_run.watch_calls[0][1]["log"] == "gradients"
    assert accel.tracker_init_calls
    first_tracker_call = accel.tracker_init_calls[0]
    assert first_tracker_call["project_name"] == "deberta-train"
    assert first_tracker_call["init_kwargs"]["wandb"]["name"] == "run"
    assert accel.ended is False


def test_run_pretraining_logs_crash_save_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from _fakes import _PRETRAINING_BATCH

    def _fake_save_checkpoint(
        *,
        accelerator,
        checkpoint_dir,
        output_dir,
        consumed_micro_batches,
        save_total_limit,
        log_label,
        **kwargs,
    ):
        del accelerator, checkpoint_dir, output_dir, consumed_micro_batches, save_total_limit, kwargs
        if str(log_label) == "final":
            raise RuntimeError("disk full")

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        save_checkpoint_fn=_fake_save_checkpoint,
        extra_patches={"_export_discriminator_hf": lambda **kwargs: None},
    )

    def _interrupt_cycle(_loader, *, start_epoch: int = 0):
        del start_epoch
        yield _PRETRAINING_BATCH
        raise KeyboardInterrupt

    monkeypatch.setattr(pretrain_mod, "_cycle_dataloader", _interrupt_cycle)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=2,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        torch_compile=False,
        export_hf_final=False,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyboardInterrupt):
            pretrain_mod.run_pretraining(
                model_cfg=ModelConfig(),
                data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
                train_cfg=train_cfg,
            )

    assert any("Final/crash-time checkpoint save failed" in rec.message for rec in caplog.records)


def test_run_pretraining_crash_checkpoint_saves_committed_microbatch_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes import _PRETRAINING_BATCH

    saved_checkpoints: list[tuple[str, int, str]] = []

    def _fake_save_checkpoint(
        *,
        accelerator,
        checkpoint_dir,
        output_dir,
        consumed_micro_batches,
        save_total_limit,
        log_label,
        **kwargs,
    ):
        del accelerator, output_dir, save_total_limit, kwargs
        saved_checkpoints.append((str(checkpoint_dir), int(consumed_micro_batches), str(log_label)))

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        save_checkpoint_fn=_fake_save_checkpoint,
        extra_patches={"_export_discriminator_hf": lambda **kwargs: None},
    )

    # Complete one accumulation window (2 micro-batches), then interrupt in the next
    # window after fetching one more micro-batch. Final checkpoint should persist only
    # committed-step progress for checkpoint-{global_step}.
    def _interrupt_cycle(_loader, *, start_epoch: int = 0):
        del start_epoch
        yield _PRETRAINING_BATCH
        yield _PRETRAINING_BATCH
        yield _PRETRAINING_BATCH
        raise KeyboardInterrupt

    monkeypatch.setattr(pretrain_mod, "_cycle_dataloader", _interrupt_cycle)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=3,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        torch_compile=False,
        export_hf_final=False,
    )

    with pytest.raises(KeyboardInterrupt):
        pretrain_mod.run_pretraining(
            model_cfg=ModelConfig(),
            data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
            train_cfg=train_cfg,
        )

    assert saved_checkpoints
    assert saved_checkpoints[-1][0].endswith("checkpoint-1")
    assert saved_checkpoints[-1][1] == 2
    assert saved_checkpoints[-1][2] == "final"


def _write_resume_source_snapshots(run_dir: Path) -> None:
    """Write minimal config snapshots required for strict resume validation."""
    model_cfg = ModelConfig()
    data_cfg = DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy")
    train_cfg = TrainConfig()
    optim_cfg = OptimConfig()
    logging_cfg = LoggingConfig(output_dir=str(run_dir))
    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg, optim_cfg=optim_cfg)
    (run_dir / "model_config.json").write_text(
        json.dumps(asdict(model_cfg), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "data_config.json").write_text(
        json.dumps(asdict(data_cfg), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "train_config.json").write_text(
        json.dumps(asdict(train_cfg), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "optim_config.json").write_text(
        json.dumps(asdict(optim_cfg), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "logging_config.json").write_text(
        json.dumps(asdict(logging_cfg), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"config_schema_version": int(RUN_CONFIG_SCHEMA_VERSION)}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def test_run_pretraining_resume_at_max_steps_skips_data_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoint-2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_resume_source_snapshots(checkpoint_dir.parent)
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "data_state.json").write_text(
        json.dumps({"consumed_micro_batches": 50}),
        encoding="utf-8",
    )
    (checkpoint_dir / ".complete").write_text("ok\n", encoding="utf-8")

    replay_calls = {"next": 0}

    def _fail_cycle(_loader, *, start_epoch: int = 0):
        del start_epoch
        while True:
            replay_calls["next"] += 1
            raise AssertionError("resume replay should be skipped when global_step >= max_steps")
            yield {}  # pragma: no cover

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        cycle_fn=_fail_cycle,
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=2,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        resume_from_checkpoint=str(checkpoint_dir),
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )
    assert replay_calls["next"] == 0


def test_run_pretraining_resume_normalizes_legacy_partial_window_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoint-1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_resume_source_snapshots(checkpoint_dir.parent)
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "data_state.json").write_text(
        json.dumps({"consumed_micro_batches": 3}),
        encoding="utf-8",
    )
    (checkpoint_dir / ".complete").write_text("ok\n", encoding="utf-8")

    captured: dict[str, int] = {}

    def _capture_policy(*, train_cfg: Any, consumed_micro_batches: int, global_step: int):
        del train_cfg
        captured["consumed_micro_batches"] = int(consumed_micro_batches)
        captured["global_step"] = int(global_step)
        return 0, False, "captured"

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
    )
    monkeypatch.setattr(pretrain_mod, "_resolve_data_resume_policy", _capture_policy)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        resume_from_checkpoint=str(checkpoint_dir),
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    assert captured["global_step"] == 1
    assert captured["consumed_micro_batches"] == 2


def test_run_pretraining_resume_normalization_uses_save_time_ga_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoint-1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_resume_source_snapshots(checkpoint_dir.parent)
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "data_state.json").write_text(
        json.dumps(
            {
                "consumed_micro_batches": 3,
                "global_step": 1,
                "gradient_accumulation_steps": 3,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / ".complete").write_text("ok\n", encoding="utf-8")

    captured: dict[str, int] = {}

    def _capture_policy(*, train_cfg: Any, consumed_micro_batches: int, global_step: int):
        del train_cfg
        captured["consumed_micro_batches"] = int(consumed_micro_batches)
        captured["global_step"] = int(global_step)
        return 0, False, "captured"

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
    )
    monkeypatch.setattr(pretrain_mod, "_resolve_data_resume_policy", _capture_policy)

    # Current run uses GA=2, but resume normalization should use save-time GA=3.
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        resume_from_checkpoint=str(checkpoint_dir),
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    assert captured["global_step"] == 1
    assert captured["consumed_micro_batches"] == 3


def test_run_pretraining_resume_requires_data_state_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoint-2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_resume_source_snapshots(checkpoint_dir.parent)
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / ".complete").write_text("ok\n", encoding="utf-8")
    pretrain_mod = setup_pretraining_mocks(monkeypatch)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=3,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        resume_from_checkpoint=str(checkpoint_dir),
    )

    with pytest.raises(ValueError, match="missing/invalid data_state.json"):
        pretrain_mod.run_pretraining(
            model_cfg=ModelConfig(),
            data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
            train_cfg=train_cfg,
        )


def test_run_pretraining_resume_rejects_checkpoint_step_metadata_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoint-2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_resume_source_snapshots(checkpoint_dir.parent)
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "data_state.json").write_text(
        json.dumps({"consumed_micro_batches": 2, "global_step": 1}),
        encoding="utf-8",
    )
    (checkpoint_dir / ".complete").write_text("ok\n", encoding="utf-8")
    pretrain_mod = setup_pretraining_mocks(monkeypatch)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=3,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        resume_from_checkpoint=str(checkpoint_dir),
    )

    with pytest.raises(RuntimeError, match="Checkpoint step mismatch on resume"):
        pretrain_mod.run_pretraining(
            model_cfg=ModelConfig(),
            data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
            train_cfg=train_cfg,
        )


def test_run_pretraining_resume_accepts_legacy_single_digest_in_decoupled_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoint-0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_resume_source_snapshots(checkpoint_dir.parent)
    (checkpoint_dir.parent / "model_config.json").write_text(
        json.dumps(asdict(ModelConfig(backbone_type="hf_deberta_v2", embedding_sharing="gdes")), indent=2)
        + "\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "data_state.json").write_text(
        json.dumps(
            {
                "consumed_micro_batches": 0,
                "global_step": 0,
                "optimizer_param_digest": "legacydeadbeef000",
            }
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / ".complete").write_text("ok\n", encoding="utf-8")

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=SimpleRTD,
    )

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=1,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        decoupled_training=True,
        resume_from_checkpoint=str(checkpoint_dir),
    )
    (checkpoint_dir.parent / "logging_config.json").write_text(
        json.dumps(
            asdict(
                LoggingConfig(
                    output_dir=str(checkpoint_dir.parent),
                    logging_steps=1,
                    report_to="none",
                )
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        pretrain_mod.run_pretraining(
            model_cfg=ModelConfig(backbone_type="hf_deberta_v2", embedding_sharing="gdes"),
            data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
            train_cfg=train_cfg,
        )

    assert any(
        "legacy single optimizer digest while current run uses decoupled mode" in rec.message
        for rec in caplog.records
    )


def test_run_pretraining_logs_window_averaged_rtd_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=lambda **kwargs: SimpleRTD(
            behavior={
                "loss": 1.0,
                "gen_loss": [1.0, 9.0],
                "disc_loss": [10.0, 2.0],
                "disc_accuracy": [0.2, 0.8],
                "gen_token_count": [1.0, 9.0],
                "disc_token_count": [10.0, 2.0],
                "disc_positive_count": [7.0, 1.0],
            },
            **kwargs,
        ),
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=1,
        save_steps=0,
        report_to="tensorboard",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        token_weighted_gradient_accumulation=False,
        decoupled_training=False,
        torch_compile=False,
        export_hf_final=False,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    accel = FakeAccelerator.last_instance
    assert accel is not None
    step_rows = [row for row, step in accel.logged_rows if int(step or -1) == 1]
    assert step_rows
    metrics = step_rows[-1]
    assert metrics["gen_loss"] == pytest.approx(8.2, rel=0.0, abs=1e-6)
    assert metrics["disc_loss"] == pytest.approx(104.0 / 12.0, rel=0.0, abs=1e-6)
    assert metrics["disc_acc"] == pytest.approx(0.3, rel=0.0, abs=1e-6)
    assert "gen_token_count" not in metrics
    assert "disc_token_count" not in metrics
    assert metrics["disc_pos_frac"] == pytest.approx(8.0 / 12.0, rel=0.0, abs=1e-6)


@pytest.mark.parametrize(
    "scenario",
    [
        "steps_and_sync",
        "token_weighted_scaling",
        "branch_loss_weights",
        "skip_generator_step",
        "partial_disc_window_sync",
    ],
)
def test_run_pretraining_decoupled_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    call_order: list[str] = []
    step_counts = {"gen": 0, "disc": 0}
    behavior: dict[str, Any] = {}
    extra_patches: dict[str, Any] | None = None

    if scenario == "steps_and_sync":
        behavior = {
            "generator_phase_loss_scale": 1.0,
            "discriminator_phase_loss_scale": 2.0,
            "on_generator_phase": lambda _m, _i: call_order.append("gen_forward"),
            "on_discriminator_phase": lambda _m, _i: call_order.append("disc_forward"),
            "on_sync_discriminator_embeddings": lambda _m: call_order.append("sync"),
        }
        train_cfg = TrainConfig(
            output_dir=str(tmp_path / "run"),
            max_steps=1,
            logging_steps=1,
            save_steps=0,
            report_to="tensorboard",
            mixed_precision="no",
            tf32=False,
            dataloader_num_workers=0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            token_weighted_gradient_accumulation=False,
            gen_loss_weight=1.0,
            disc_loss_weight=1.0,
            torch_compile=False,
            export_hf_final=False,
            decoupled_training=True,
        )
    elif scenario == "token_weighted_scaling":
        behavior = {
            "generator_phase_loss_scale": [10.0, 20.0],
            "discriminator_phase_loss_scale": [5.0, 7.0],
            "discriminator_phase_accuracy": 0.5,
            "discriminator_phase_positive_count": 0.0,
        }
        micro_counts = iter([(1.0, 4.0), (3.0, 2.0)])

        def _count_tokens_for_microbatch(*_args: Any, **_kwargs: Any) -> tuple[float, float]:
            return next(micro_counts)

        extra_patches = {"_count_rtd_tokens_for_batch": _count_tokens_for_microbatch}
        train_cfg = TrainConfig(
            output_dir=str(tmp_path / "run"),
            max_steps=1,
            logging_steps=1,
            save_steps=0,
            report_to="none",
            mixed_precision="no",
            tf32=False,
            dataloader_num_workers=0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            token_weighted_gradient_accumulation=True,
            gen_loss_weight=1.0,
            disc_loss_weight=1.0,
            torch_compile=False,
            export_hf_final=False,
            decoupled_training=True,
        )
    elif scenario == "branch_loss_weights":
        behavior = {"generator_phase_loss_scale": 2.0, "discriminator_phase_loss_scale": 3.0}
        train_cfg = TrainConfig(
            output_dir=str(tmp_path / "run"),
            max_steps=1,
            logging_steps=0,
            save_steps=0,
            report_to="none",
            mixed_precision="no",
            tf32=False,
            dataloader_num_workers=0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            token_weighted_gradient_accumulation=False,
            gen_loss_weight=0.25,
            disc_loss_weight=4.0,
            torch_compile=False,
            export_hf_final=False,
            decoupled_training=True,
        )
    elif scenario == "skip_generator_step":
        behavior = {"generator_phase_loss_scale": 2.0, "discriminator_phase_loss_scale": 3.0}
        train_cfg = TrainConfig(
            output_dir=str(tmp_path / "run"),
            max_steps=1,
            logging_steps=0,
            save_steps=0,
            report_to="none",
            mixed_precision="no",
            tf32=False,
            dataloader_num_workers=0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            token_weighted_gradient_accumulation=False,
            gen_loss_weight=0.0,
            disc_loss_weight=1.0,
            torch_compile=False,
            export_hf_final=False,
            decoupled_training=True,
        )
    elif scenario == "partial_disc_window_sync":
        behavior = {
            "generator_phase_loss_scale": [2.0, 2.0],
            "generator_phase_token_count": [0.0, 1.0],
            "discriminator_phase_loss_scale": 3.0,
        }
        train_cfg = TrainConfig(
            output_dir=str(tmp_path / "run"),
            max_steps=1,
            logging_steps=0,
            save_steps=0,
            report_to="none",
            mixed_precision="no",
            tf32=False,
            dataloader_num_workers=0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            token_weighted_gradient_accumulation=False,
            gen_loss_weight=1.0,
            disc_loss_weight=1.0,
            torch_compile=False,
            export_hf_final=False,
            decoupled_training=True,
        )
    else:  # pragma: no cover
        raise AssertionError(f"Unsupported scenario: {scenario}")

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=lambda **kwargs: SimpleRTD(behavior=behavior, **kwargs),
        extra_patches=extra_patches,
    )
    original_build_decoupled = pretrain_mod._build_decoupled_optimizers

    def _build_logged_decoupled(model: torch.nn.Module, cfg: TrainConfig, *, mixed_precision: str = "no"):
        gen_opt, disc_opt = original_build_decoupled(model, cfg, mixed_precision=mixed_precision)
        original_gen_step = gen_opt.step
        original_disc_step = disc_opt.step

        def _gen_step(_opt_self: Any, *args: Any, **kwargs: Any):
            step_counts["gen"] += 1
            if scenario == "steps_and_sync":
                call_order.append("gen_step")
            return original_gen_step(*args, **kwargs)

        def _disc_step(_opt_self: Any, *args: Any, **kwargs: Any):
            step_counts["disc"] += 1
            if scenario == "steps_and_sync":
                call_order.append("disc_step")
            return original_disc_step(*args, **kwargs)

        gen_opt.step = types.MethodType(_gen_step, gen_opt)  # type: ignore[method-assign]
        disc_opt.step = types.MethodType(_disc_step, disc_opt)  # type: ignore[method-assign]
        return gen_opt, disc_opt

    monkeypatch.setattr(pretrain_mod, "_build_decoupled_optimizers", _build_logged_decoupled)

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope", embedding_sharing="gdes"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    accel = FakeAccelerator.last_instance
    assert accel is not None

    if scenario == "steps_and_sync":
        assert step_counts == {"gen": 1, "disc": 1}
        gen_forward_idx = call_order.index("gen_forward")
        gen_step_idx = call_order.index("gen_step")
        sync_after_gen_idx = call_order.index("sync", gen_step_idx + 1)
        disc_forward_idx = call_order.index("disc_forward")
        disc_step_idx = call_order.index("disc_step")
        assert gen_forward_idx < gen_step_idx < sync_after_gen_idx < disc_forward_idx < disc_step_idx
        step_rows = [row for row, step in accel.logged_rows if int(step or -1) == 1]
        assert step_rows
        assert "decoupled_training" not in step_rows[-1]
    elif scenario == "token_weighted_scaling":
        assert step_counts == {"gen": 1, "disc": 1}
        assert accel.calls["backward"] == pytest.approx(
            [
                10.0 * (1.0 / 4.0) * 2.0,
                20.0 * (3.0 / 4.0) * 2.0,
                5.0 * (4.0 / 6.0) * 2.0,
                7.0 * (2.0 / 6.0) * 2.0,
            ],
            rel=0.0,
            abs=1e-6,
        )
    elif scenario == "branch_loss_weights":
        assert step_counts == {"gen": 1, "disc": 1}
        assert accel.calls["backward"] == pytest.approx([2.0 * 0.25, 3.0 * 4.0], rel=0.0, abs=1e-6)
    elif scenario == "skip_generator_step":
        assert step_counts == {"gen": 0, "disc": 1}
        assert accel.calls["backward"] == pytest.approx([3.0], rel=0.0, abs=1e-6)
    elif scenario == "partial_disc_window_sync":
        assert step_counts == {"gen": 1, "disc": 1}
        model = SimpleRTD.last_instance
        assert model is not None
        assert len(model.calls["forward_discriminator_phase"]) == 2
        assert accel.calls["backward"] == pytest.approx([2.0, 2.0, 0.0, 3.0], rel=0.0, abs=1e-6)


def test_run_pretraining_decoupled_nonfinite_disc_does_not_double_step_gen_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler_steps = {"gen": 0, "disc": 0}

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=lambda **kwargs: SimpleRTD(
            behavior={
                "generator_phase_loss_scale": 1.0,
                "discriminator_phase_loss_scale": float("nan"),
            },
            **kwargs,
        ),
    )

    scheduler_build_count = 0

    def _build_counted_scheduler(optimizer: torch.optim.Optimizer, _cfg: TrainConfig):
        nonlocal scheduler_build_count
        phase = "gen" if scheduler_build_count == 0 else "disc"
        scheduler_build_count += 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        original_step = scheduler.step

        def _count_step(*args: Any, **kwargs: Any) -> Any:
            scheduler_steps[phase] += 1
            return original_step(*args, **kwargs)

        scheduler.step = _count_step  # type: ignore[method-assign]
        return scheduler

    monkeypatch.setattr(pretrain_mod, "_build_scheduler", _build_counted_scheduler)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=0,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        decoupled_training=True,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope", embedding_sharing="gdes"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    assert scheduler_steps["gen"] == 1
    assert scheduler_steps["disc"] == 0


def test_run_pretraining_decoupled_skips_discriminator_for_zero_generator_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=lambda **kwargs: SimpleRTD(
            behavior={
                "generator_phase_loss_scale": 0.0,
                "generator_phase_token_count": 0.0,
            },
            **kwargs,
        ),
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=0,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        decoupled_training=True,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope", embedding_sharing="gdes"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    model = SimpleRTD.last_instance
    assert model is not None
    assert len(model.calls.get("forward_discriminator_phase", [])) == 0


def test_run_pretraining_decoupled_routes_phase_calls_through_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=SimpleRTD,
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=0,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        decoupled_training=True,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope", embedding_sharing="gdes"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    model = SimpleRTD.last_instance
    assert model is not None
    # One generator and one discriminator phase should each route through model(...)
    # so distributed wrappers keep their forward-path bookkeeping/autocast hooks.
    assert len(model.calls.get("forward", [])) == 2
    assert len(model.calls.get("forward_generator_phase", [])) == 1
    assert len(model.calls.get("forward_discriminator_phase", [])) == 1


def test_run_pretraining_final_export_uses_subprocess_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_calls: list[tuple[str, str]] = []

    def _fake_export_subprocess(*, checkpoint_dir: Path, output_dir: Path) -> None:
        export_calls.append((str(checkpoint_dir), str(output_dir)))

    def _fake_save_checkpoint(
        *,
        accelerator: Any,
        checkpoint_dir: Path,
        output_dir: Path,
        consumed_micro_batches: int,
        save_total_limit: int,
        log_label: str,
        **kwargs: Any,
    ) -> None:
        del accelerator, output_dir, consumed_micro_batches, save_total_limit, log_label, kwargs
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        save_checkpoint_fn=_fake_save_checkpoint,
        extra_patches={"_export_discriminator_hf_subprocess": _fake_export_subprocess},
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=0,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=True,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    assert export_calls
    ckpt_path, export_path = export_calls[-1]
    assert ckpt_path.endswith("checkpoint-1")
    assert export_path.endswith("final_hf")


def _run_zero_token_weighted_case(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    debug_metrics: bool,
) -> Path:
    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=lambda **kwargs: SimpleRTD(
            behavior={
                "loss": 1.0,
                "gen_loss": 1.0,
                "disc_loss": 1.0,
                "disc_accuracy": 1.0,
                "gen_token_count": 1.0,
                "disc_token_count": 1.0,
                "disc_positive_count": 1.0,
            },
            **kwargs,
        ),
        extra_patches={"_count_rtd_tokens_for_batch": lambda *args, **kwargs: (0.0, 0.0)},
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=1,
        save_steps=0,
        report_to="tensorboard",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        token_weighted_gradient_accumulation=True,
        decoupled_training=False,
        debug_metrics=bool(debug_metrics),
        torch_compile=False,
        export_hf_final=False,
    )
    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )
    return Path(train_cfg.output_dir) / "metrics.jsonl.gz"


def _latest_zero_token_tracker_metrics() -> dict[str, Any]:
    accel = FakeAccelerator.last_instance
    assert accel is not None
    step_rows = [row for row, step in accel.logged_rows if int(step or -1) == 1]
    assert step_rows
    return step_rows[-1]


def _assert_zero_window_metrics_hidden_from_trackers(metrics: dict[str, Any]) -> None:
    assert "zero_gen_window_total" not in metrics
    assert "zero_disc_window_total" not in metrics
    assert "zero_gen_window_since_log" not in metrics
    assert "zero_disc_window_since_log" not in metrics


def _load_last_debug_metrics_row(metrics_path: Path) -> dict[str, Any]:
    with gzip.open(metrics_path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f.read().splitlines()]
    assert rows
    debug_rows = [row for row in rows if bool(row.get("debug_metrics"))]
    assert debug_rows
    return debug_rows[-1]


@pytest.mark.parametrize(
    ("debug_metrics", "expect_metrics_file", "expect_warning"),
    [
        (False, False, True),
        (True, True, False),
    ],
)
def test_run_pretraining_zero_token_weighted_metrics_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    debug_metrics: bool,
    expect_metrics_file: bool,
    expect_warning: bool,
) -> None:
    with caplog.at_level(logging.WARNING):
        metrics_path = _run_zero_token_weighted_case(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            debug_metrics=bool(debug_metrics),
        )
    _assert_zero_window_metrics_hidden_from_trackers(_latest_zero_token_tracker_metrics())

    if expect_warning:
        assert any(
            "Token-weighted GA window has zero effective tokens" in rec.message for rec in caplog.records
        )

    if expect_metrics_file:
        assert metrics_path.exists()
        last = _load_last_debug_metrics_row(metrics_path)
        assert int(last["step"]) == 1
        assert float(last["zero_gen_window_total"]) == pytest.approx(1.0, rel=0.0, abs=1e-6)
        assert float(last["zero_disc_window_total"]) == pytest.approx(1.0, rel=0.0, abs=1e-6)
        assert float(last["zero_gen_window_since_log"]) == pytest.approx(1.0, rel=0.0, abs=1e-6)
        assert float(last["zero_disc_window_since_log"]) == pytest.approx(1.0, rel=0.0, abs=1e-6)
    else:
        assert not metrics_path.exists()


def test_run_pretraining_compiles_generator_and_discriminator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compile_calls: list[tuple[Any, dict[str, Any]]] = []

    def _fake_compile(
        target: Any, *, mode: str = "default", backend: str = "inductor", dynamic: bool | None = None
    ) -> Any:
        compile_calls.append((target, {"mode": str(mode), "backend": str(backend), "dynamic": dynamic}))
        return target

    pretrain_mod = setup_pretraining_mocks(monkeypatch)
    monkeypatch.setattr(pretrain_mod.torch, "compile", _fake_compile)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=True,
        torch_compile_mode="default",
        export_hf_final=False,
    )
    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    instance = SimpleRTD.last_instance
    assert instance is not None
    assert len(compile_calls) == 2
    for i in range(2):
        assert callable(compile_calls[i][0])
        assert compile_calls[i][1] == {"mode": "default", "backend": "inductor", "dynamic": False}
    assert getattr(compile_calls[0][0], "__self__", None) is instance.generator
    assert getattr(compile_calls[1][0], "__self__", None) is instance.discriminator


def test_run_pretraining_builds_doc_block_mask_before_compile_stabilizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes import _PRETRAINING_BATCH

    pretrain_mod = setup_pretraining_mocks(monkeypatch)

    batch_with_doc_ids: dict[str, torch.Tensor] = {
        k: v.clone() for k, v in _PRETRAINING_BATCH.items() if isinstance(v, torch.Tensor)
    }
    batch_with_doc_ids["doc_ids"] = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.long)
    expected_mask = pretrain_mod._build_doc_block_mask(batch_with_doc_ids["doc_ids"])

    def _cycle_with_doc_ids(_loader: Any, *, start_epoch: int = 0):
        del _loader, start_epoch
        while True:
            yield {k: v.clone() for k, v in batch_with_doc_ids.items()}

    monkeypatch.setattr(pretrain_mod, "_cycle_dataloader", _cycle_with_doc_ids)
    monkeypatch.setattr(
        pretrain_mod.torch,
        "compile",
        lambda target, *, mode="default", backend="inductor", dynamic=None: target,
    )

    seen_masks: list[torch.Tensor] = []
    original_stabilize = pretrain_mod._stabilize_compile_attention_mask

    def _stabilize_spy(
        *,
        batch: dict[str, Any],
        compile_enabled: bool,
        compile_scope: str,
        backbone_type: str,
        block_cross_document_attention: bool = False,
    ) -> dict[str, Any]:
        mask = batch.get("attention_mask")
        if isinstance(mask, torch.Tensor):
            seen_masks.append(mask.detach().clone())
        return original_stabilize(
            batch=batch,
            compile_enabled=compile_enabled,
            compile_scope=compile_scope,
            backbone_type=backbone_type,
            block_cross_document_attention=block_cross_document_attention,
        )

    monkeypatch.setattr(pretrain_mod, "_stabilize_compile_attention_mask", _stabilize_spy)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=0,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=True,
        torch_compile_scope="backbones",
        export_hf_final=False,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope"),
        data_cfg=DataConfig(
            dataset_name="hf-internal-testing/librispeech_asr_dummy",
            block_cross_document_attention=True,
        ),
        train_cfg=train_cfg,
    )

    assert seen_masks
    assert seen_masks[0].ndim == 3
    assert seen_masks[0].dtype == torch.bool
    assert torch.equal(seen_masks[0], expected_mask)


def test_run_pretraining_hf_deberta_auto_scope_compiles_backbones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = torch.nn.Linear(2, 2)
            self.intermediate = torch.nn.Linear(2, 2)
            self.output = torch.nn.Linear(2, 2)

    class _FakeBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = torch.nn.Linear(2, 2)
            self.encoder = torch.nn.Module()
            self.encoder.layer = torch.nn.ModuleList([_FakeLayer()])

    compile_calls: list[tuple[Any, dict[str, Any]]] = []

    def _fake_compile(
        target: Any, *, mode: str = "default", backend: str = "inductor", dynamic: bool | None = None
    ) -> Any:
        compile_calls.append((target, {"mode": str(mode), "backend": str(backend), "dynamic": dynamic}))
        return target

    created_models: list[SimpleRTD] = []

    def _make_scope_rtd(**kwargs: Any) -> SimpleRTD:
        model = SimpleRTD(**kwargs)
        model.generator = _FakeBackbone()
        model.discriminator = _FakeBackbone()
        created_models.append(model)
        return model

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        rtd_cls=_make_scope_rtd,
        build_backbones_fn=lambda **kwargs: (_FakeBackbone(), _FakeBackbone()),
    )
    monkeypatch.setattr(pretrain_mod.torch, "compile", _fake_compile)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=True,
        torch_compile_mode="default",
        export_hf_final=False,
    )
    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    assert created_models
    instance = created_models[-1]
    assert instance is not None
    assert len(compile_calls) == 2
    for i in range(2):
        assert callable(compile_calls[i][0])
        assert compile_calls[i][1] == {"mode": "default", "backend": "inductor", "dynamic": False}
    assert getattr(compile_calls[0][0], "__self__", None) is instance.generator
    assert getattr(compile_calls[1][0], "__self__", None) is instance.discriminator


def test_run_pretraining_skips_nonfinite_grad_window_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        extra_patches={"_global_grad_l2_norm": lambda _model: float("inf")},
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        decoupled_training=False,
        torch_compile=False,
        export_hf_final=False,
        max_grad_norm=1.0,
    )

    with caplog.at_level(logging.WARNING):
        pretrain_mod.run_pretraining(
            model_cfg=ModelConfig(),
            data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
            train_cfg=train_cfg,
        )
    assert "nonfinite_window_skipped=1" in caplog.text


def test_run_pretraining_nonfinite_grad_norm_never_steps_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opt_ref: dict[str, Any] = {}

    class _CountingSGD(torch.optim.SGD):
        def __init__(self, params: Any, lr: float) -> None:
            super().__init__(params, lr=lr)
            self.step_calls = 0

        def step(self, closure: Any = None):  # type: ignore[override]
            self.step_calls += 1
            return super().step(closure=closure)

    def _build_optimizer(model: torch.nn.Module, _cfg: Any, **_kwargs: Any) -> torch.optim.Optimizer:
        opt = _CountingSGD(model.parameters(), lr=0.1)
        opt_ref["opt"] = opt
        return opt

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        extra_patches={
            "_global_grad_l2_norm": lambda _model: float("inf"),
            "_build_optimizer": _build_optimizer,
        },
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        decoupled_training=False,
        torch_compile=False,
        export_hf_final=False,
        max_grad_norm=1.0,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )
    opt = opt_ref.get("opt")
    assert isinstance(opt, _CountingSGD)
    assert int(opt.step_calls) == 0


def test_apply_nonfinite_recovery_ratchets_lr_mult_and_resets_state_on_interval() -> None:
    # First nonfinite event: lr_mult should halve (0.5 backoff).
    new_mult, did_reset = _apply_nonfinite_recovery(lr_mult=1.0, skip_streak=1)
    assert did_reset is False
    assert abs(new_mult - 0.5) < 1e-9

    # Second consecutive: ratchets further.
    new_mult2, _ = _apply_nonfinite_recovery(lr_mult=new_mult, skip_streak=2)
    assert abs(new_mult2 - 0.25) < 1e-9

    # At streak=4: optimizer state reset triggered.
    new_mult4, did_reset4 = _apply_nonfinite_recovery(lr_mult=new_mult2, skip_streak=4)
    assert did_reset4 is True
    assert new_mult4 >= 0.01  # floor

    # lr_mult cannot go below _NONFINITE_LR_MULT_FLOOR.
    new_mult_floor, _ = _apply_nonfinite_recovery(lr_mult=0.01, skip_streak=5)
    assert abs(new_mult_floor - 0.01) < 1e-9


def test_apply_lr_mult_scales_all_param_groups() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    _apply_lr_mult(optimizer, 0.5)
    assert abs(float(optimizer.param_groups[0]["lr"]) - 5e-4) < 1e-12


def test_apply_lr_mult_uses_unscaled_reference_without_compounding() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    _record_unscaled_lrs(optimizer, scheduler)

    _apply_lr_mult(optimizer, 0.5)
    assert float(optimizer.param_groups[0]["lr"]) == pytest.approx(5e-4, abs=1e-12)

    # Applying a new multiplier must be absolute against scheduler LR (1e-3),
    # not multiplicative against the already-scaled group LR (5e-4).
    _apply_lr_mult(optimizer, 0.25)
    assert float(optimizer.param_groups[0]["lr"]) == pytest.approx(2.5e-4, abs=1e-12)


def test_persist_or_validate_run_configs_rejects_resume_model_data_mismatch(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)

    base_model = ModelConfig(backbone_type="rope")
    base_data = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    base_train = TrainConfig()
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=base_model,
        data_cfg=base_data,
        train_cfg=base_train,
        resume_checkpoint=None,
        is_main_process=True,
    )
    run_meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert int(run_meta["config_schema_version"]) == int(RUN_CONFIG_SCHEMA_VERSION)

    changed_model = ModelConfig(backbone_type="rope", hidden_size=1024)
    with pytest.raises(ValueError, match="Resume configuration mismatch for model_config.json"):
        _persist_or_validate_run_configs(
            output_dir=out,
            model_cfg=changed_model,
            data_cfg=base_data,
            train_cfg=base_train,
            resume_checkpoint=str(out / "checkpoint-10"),
            is_main_process=True,
        )


def test_persist_or_validate_run_configs_does_not_backfill_metadata_on_failed_resume(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig(backbone_type="rope")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig()
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        is_main_process=True,
    )

    run_meta_path = out / "run_metadata.json"
    run_meta_path.unlink()
    changed_model = ModelConfig(backbone_type="rope", hidden_size=1024)

    with pytest.raises(ValueError, match="Resume configuration mismatch for model_config.json"):
        _persist_or_validate_run_configs(
            output_dir=out,
            model_cfg=changed_model,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            resume_checkpoint=str(out / "checkpoint-10"),
            is_main_process=True,
        )
    assert not run_meta_path.exists()


def test_persist_or_validate_run_configs_validates_against_resume_source_run_dir(tmp_path: Path):
    source_run = tmp_path / "source-run"
    source_run.mkdir(parents=True, exist_ok=True)
    model_cfg = ModelConfig(backbone_type="rope")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig(max_steps=10)
    _persist_or_validate_run_configs(
        output_dir=source_run,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        is_main_process=True,
    )
    checkpoint_dir = source_run / "checkpoint-10"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    new_output_dir = tmp_path / "new-run"
    new_output_dir.mkdir(parents=True, exist_ok=True)
    mismatched_model = ModelConfig(backbone_type="rope", hidden_size=1024)
    with pytest.raises(ValueError, match="Resume configuration mismatch for model_config.json"):
        _persist_or_validate_run_configs(
            output_dir=new_output_dir,
            model_cfg=mismatched_model,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            resume_checkpoint=str(checkpoint_dir),
            is_main_process=True,
        )


def test_persist_or_validate_run_configs_tracks_resume_source_when_output_dir_differs(tmp_path: Path):
    source_run = tmp_path / "source-run"
    source_run.mkdir(parents=True, exist_ok=True)
    model_cfg = ModelConfig(backbone_type="rope")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig(max_steps=10)
    _persist_or_validate_run_configs(
        output_dir=source_run,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        is_main_process=True,
    )
    checkpoint_dir = source_run / "checkpoint-10"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    new_output_dir = tmp_path / "new-run"
    new_output_dir.mkdir(parents=True, exist_ok=True)
    changed_train_cfg = TrainConfig(max_steps=25)
    _persist_or_validate_run_configs(
        output_dir=new_output_dir,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=changed_train_cfg,
        resume_checkpoint=str(checkpoint_dir),
        is_main_process=True,
    )

    resume_source = json.loads((new_output_dir / "resume_source.json").read_text(encoding="utf-8"))
    assert resume_source["resume_checkpoint"] == str(checkpoint_dir.resolve())
    assert resume_source["resume_run_dir"] == str(source_run.resolve())
    assert (new_output_dir / "model_config.json").read_text(encoding="utf-8") == (
        source_run / "model_config.json"
    ).read_text(encoding="utf-8")
    assert (new_output_dir / "train_config.json").read_text(encoding="utf-8") == (
        source_run / "train_config.json"
    ).read_text(encoding="utf-8")


def test_persist_or_validate_run_configs_allows_resume_when_only_logging_output_dir_differs(
    tmp_path: Path,
) -> None:
    source_run = tmp_path / "source-run"
    source_run.mkdir(parents=True, exist_ok=True)
    source_logging_dir = source_run / "logs"
    source_logging_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig(backbone_type="rope")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig(max_steps=10)
    source_logging_cfg = LoggingConfig(output_dir=str(source_logging_dir))
    _persist_or_validate_run_configs(
        output_dir=source_run,
        logging_output_dir=source_logging_dir,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        logging_cfg=source_logging_cfg,
        resume_checkpoint=None,
        is_main_process=True,
    )
    checkpoint_dir = source_run / "checkpoint-10"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    new_output_dir = tmp_path / "new-run"
    new_output_dir.mkdir(parents=True, exist_ok=True)
    new_logging_dir = new_output_dir / "logs"
    new_logging_dir.mkdir(parents=True, exist_ok=True)
    resumed_logging_cfg = LoggingConfig(output_dir=str(new_logging_dir))
    _persist_or_validate_run_configs(
        output_dir=new_output_dir,
        logging_output_dir=new_logging_dir,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        logging_cfg=resumed_logging_cfg,
        resume_checkpoint=str(checkpoint_dir),
        is_main_process=True,
    )

    resume_source = json.loads((new_output_dir / "resume_source.json").read_text(encoding="utf-8"))
    assert resume_source["resume_checkpoint"] == str(checkpoint_dir.resolve())
    assert resume_source["resume_run_dir"] == str(source_run.resolve())
    assert (new_output_dir / "logging_config.json").read_text(encoding="utf-8") == (
        source_run / "logging_config.json"
    ).read_text(encoding="utf-8")


def test_persist_or_validate_run_configs_preflight_rejects_conflicting_resume_snapshot(tmp_path: Path):
    source_run = tmp_path / "source-run"
    source_run.mkdir(parents=True, exist_ok=True)
    model_cfg = ModelConfig(backbone_type="rope")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig(max_steps=10)
    _persist_or_validate_run_configs(
        output_dir=source_run,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        is_main_process=True,
    )
    checkpoint_dir = source_run / "checkpoint-10"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    new_output_dir = tmp_path / "new-run"
    new_output_dir.mkdir(parents=True, exist_ok=True)
    (new_output_dir / "model_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting run snapshot"):
        _persist_or_validate_run_configs(
            output_dir=new_output_dir,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            resume_checkpoint=str(checkpoint_dir),
            is_main_process=False,
            preflight_only=True,
        )


def test_persist_or_validate_run_configs_rejects_resume_when_source_snapshots_missing(tmp_path: Path):
    source_run = tmp_path / "source-run"
    source_run.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = source_run / "checkpoint-1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="source run directory is missing required config snapshots"):
        _persist_or_validate_run_configs(
            output_dir=out,
            model_cfg=ModelConfig(backbone_type="rope"),
            data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
            train_cfg=TrainConfig(),
            resume_checkpoint=str(checkpoint_dir),
            is_main_process=True,
        )


def test_persist_or_validate_run_configs_allows_resume_when_only_inert_model_fields_change(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)

    base_model = ModelConfig(backbone_type="rope")
    base_data = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    base_train = TrainConfig()
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=base_model,
        data_cfg=base_data,
        train_cfg=base_train,
        resume_checkpoint=None,
        is_main_process=True,
    )

    inert_changed_model = ModelConfig(
        backbone_type="rope",
        hf_attention_kernel="stable",
        hf_max_position_embeddings=1024,
    )
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=inert_changed_model,
        data_cfg=base_data,
        train_cfg=base_train,
        resume_checkpoint=str(out / "checkpoint-10"),
        is_main_process=True,
    )


def test_persist_or_validate_run_configs_writes_original_and_resolved_yaml(
    tmp_path: Path,
) -> None:
    pytest.importorskip("yaml")

    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    src_cfg = tmp_path / "source.yaml"
    src_text = "model:\n  backbone_type: rope\ntrain:\n  max_steps: 7\n"
    src_cfg.write_text(src_text, encoding="utf-8")

    model_cfg = ModelConfig(backbone_type="rope")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig(max_steps=7)
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        config_path=src_cfg,
        is_main_process=True,
    )

    original_path = out / "config_original.yaml"
    resolved_path = out / "config_resolved.yaml"
    assert original_path.read_text(encoding="utf-8") == src_text
    assert resolved_path.exists()
    loaded_resolved = _load_yaml(resolved_path)
    assert loaded_resolved[0].backbone_type == model_cfg.backbone_type
    assert loaded_resolved[1].dataset_name == data_cfg.dataset_name
    assert loaded_resolved[2].max_steps == train_cfg.max_steps


def test_persist_or_validate_run_configs_preserves_existing_snapshots_on_matching_resume(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig(backbone_type="rope")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig(max_steps=10)
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        is_main_process=True,
    )
    original_train_snapshot = json.loads((out / "train_config.json").read_text(encoding="utf-8"))

    changed_train_cfg = TrainConfig(max_steps=20)
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=changed_train_cfg,
        resume_checkpoint=str(out / "checkpoint-10"),
        is_main_process=True,
    )

    resumed_train_snapshot = json.loads((out / "train_config.json").read_text(encoding="utf-8"))
    assert resumed_train_snapshot == original_train_snapshot


def test_persist_or_validate_run_configs_rejects_unknown_run_metadata_schema(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig(backbone_type="rope")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig()
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        is_main_process=True,
    )

    (out / "run_metadata.json").write_text(
        json.dumps({"config_schema_version": int(RUN_CONFIG_SCHEMA_VERSION) + 1}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported run metadata schema"):
        _persist_or_validate_run_configs(
            output_dir=out,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            resume_checkpoint=str(out / "checkpoint-10"),
            is_main_process=True,
        )


def _checkpoint_saving_accelerator(
    *,
    is_main_process: bool,
    write_weights: bool = True,
) -> FakeAccelerator:
    """Build a fake accelerator whose ``save_state`` writes checkpoint-like files."""

    accel = FakeAccelerator(is_main_process=bool(is_main_process))

    def _save_state(output_dir: str | None) -> None:
        if output_dir is None:
            return
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        if write_weights:
            (p / "model.safetensors").write_bytes(b"weights")
        marker = "main" if accel.is_main_process else "worker"
        (p / f"{marker}.txt").write_text("ok", encoding="utf-8")

    accel.save_state_hook = _save_state
    return accel


def test_save_training_checkpoint_calls_collective_save_on_non_main_rank(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint-1"
    ckpt.mkdir(parents=True, exist_ok=True)

    accel = _checkpoint_saving_accelerator(is_main_process=False)
    _save_training_checkpoint(
        accelerator=accel,
        checkpoint_dir=ckpt,
        output_dir=out,
        consumed_micro_batches=7,
        save_total_limit=3,
        log_label="periodic",
    )

    assert len(accel.calls["save_state"]) == 1
    staged = Path(str(accel.calls["save_state"][0]))
    assert staged.parent == out
    assert staged.name.startswith(f".{ckpt.name}.tmp-")
    assert len(accel.calls["wait_for_everyone"]) >= 3
    assert not (ckpt / "data_state.json").exists()


def test_save_training_checkpoint_writes_data_progress_on_main_rank(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint-3"

    accel = _checkpoint_saving_accelerator(is_main_process=True)
    _save_training_checkpoint(
        accelerator=accel,
        checkpoint_dir=ckpt,
        output_dir=out,
        consumed_micro_batches=42,
        save_total_limit=3,
        log_label="final",
    )

    assert len(accel.calls["save_state"]) == 1
    staged = Path(str(accel.calls["save_state"][0]))
    assert staged.parent == out
    assert staged.name.startswith(f".{ckpt.name}.tmp-")
    consumed, lr_mult, digest = _load_checkpoint_data_progress(ckpt)
    assert consumed == 42
    assert lr_mult == 1.0
    assert digest is None  # no digest passed
    assert (ckpt / ".complete").exists()


def test_save_training_checkpoint_rejects_overwrite_of_nonempty_checkpoint_dir(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint-8"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "stale.bin").write_bytes(b"stale")

    accel = _checkpoint_saving_accelerator(is_main_process=True)
    with pytest.raises(RuntimeError, match="Refusing to overwrite non-empty checkpoint directory"):
        _save_training_checkpoint(
            accelerator=accel,
            checkpoint_dir=ckpt,
            output_dir=out,
            consumed_micro_batches=1,
            save_total_limit=2,
            log_label="periodic",
        )


def test_save_training_checkpoint_rotates_only_after_postsave_validation(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    old_ckpt = out / "checkpoint-1"
    old_ckpt.mkdir(parents=True, exist_ok=True)
    (old_ckpt / "model.safetensors").write_bytes(b"old")
    (old_ckpt / "data_state.json").write_text(
        json.dumps({"consumed_micro_batches": 1}),
        encoding="utf-8",
    )
    (old_ckpt / ".complete").write_text("ok\n", encoding="utf-8")
    new_ckpt = out / "checkpoint-2"

    accel = _checkpoint_saving_accelerator(is_main_process=True)
    _save_training_checkpoint(
        accelerator=accel,
        checkpoint_dir=new_ckpt,
        output_dir=out,
        consumed_micro_batches=2,
        save_total_limit=1,
        log_label="periodic",
    )

    assert not old_ckpt.exists()
    assert new_ckpt.exists()
    assert (new_ckpt / ".complete").exists()


def test_save_training_checkpoint_skips_rotation_when_new_checkpoint_weights_invalid(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    old_ckpt = out / "checkpoint-1"
    old_ckpt.mkdir(parents=True, exist_ok=True)
    (old_ckpt / "model.safetensors").write_bytes(b"old")
    (old_ckpt / "data_state.json").write_text(
        json.dumps({"consumed_micro_batches": 1}),
        encoding="utf-8",
    )
    (old_ckpt / ".complete").write_text("ok\n", encoding="utf-8")
    new_ckpt = out / "checkpoint-2"

    accel = _checkpoint_saving_accelerator(is_main_process=True, write_weights=False)
    with pytest.raises(RuntimeError, match="Post-save structural validation failed"):
        _save_training_checkpoint(
            accelerator=accel,
            checkpoint_dir=new_ckpt,
            output_dir=out,
            consumed_micro_batches=2,
            save_total_limit=1,
            log_label="periodic",
        )

    assert old_ckpt.exists()
    assert not new_ckpt.exists()


def test_cycle_dataloader_advances_dataset_epoch_each_pass():
    class _EpochDataset(torch.utils.data.IterableDataset):
        def __init__(self) -> None:
            super().__init__()
            self.current_epoch = -1
            self.seen_epochs: list[int] = []

        def set_epoch(self, epoch: int) -> None:
            self.current_epoch = int(epoch)
            self.seen_epochs.append(int(epoch))

        def __iter__(self):
            yield {"epoch": torch.tensor(self.current_epoch, dtype=torch.long)}

    ds = _EpochDataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0)
    it = _cycle_dataloader(dl)

    e0 = int(next(it)["epoch"].item())
    e1 = int(next(it)["epoch"].item())
    e2 = int(next(it)["epoch"].item())

    assert (e0, e1, e2) == (0, 1, 2)
    assert ds.seen_epochs[:3] == [0, 1, 2]


def test_cycle_dataloader_honors_start_epoch_offset():
    class _EpochDataset(torch.utils.data.IterableDataset):
        def __init__(self) -> None:
            super().__init__()
            self.current_epoch = -1
            self.seen_epochs: list[int] = []

        def set_epoch(self, epoch: int) -> None:
            self.current_epoch = int(epoch)
            self.seen_epochs.append(int(epoch))

        def __iter__(self):
            yield {"epoch": torch.tensor(self.current_epoch, dtype=torch.long)}

    ds = _EpochDataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0)
    it = _cycle_dataloader(dl, start_epoch=7)

    e0 = int(next(it)["epoch"].item())
    e1 = int(next(it)["epoch"].item())
    assert (e0, e1) == (7, 8)
    assert ds.seen_epochs[:2] == [7, 8]


def test_resolve_data_resume_policy_auto_replays_when_small():
    cfg = TrainConfig(resume_data_strategy="auto", resume_replay_max_micro_batches=100)
    start_epoch, do_replay, reason = _resolve_data_resume_policy(
        train_cfg=cfg,
        consumed_micro_batches=42,
        global_step=9,
    )
    assert start_epoch == 0
    assert do_replay is True
    assert "replay" in reason


def test_resolve_data_resume_policy_auto_restarts_epoch_when_large():
    cfg = TrainConfig(resume_data_strategy="auto", resume_replay_max_micro_batches=10)
    start_epoch, do_replay, reason = _resolve_data_resume_policy(
        train_cfg=cfg,
        consumed_micro_batches=42,
        global_step=9,
    )
    assert start_epoch == 9
    assert do_replay is False
    assert "restart_epoch" in reason


def test_resolve_data_resume_policy_respects_explicit_strategy():
    replay_cfg = TrainConfig(resume_data_strategy="replay", resume_replay_max_micro_batches=0)
    start_epoch_replay, do_replay_replay, _ = _resolve_data_resume_policy(
        train_cfg=replay_cfg,
        consumed_micro_batches=999,
        global_step=3,
    )
    assert start_epoch_replay == 0
    assert do_replay_replay is True

    restart_cfg = TrainConfig(resume_data_strategy="restart_epoch", resume_replay_max_micro_batches=1_000_000)
    start_epoch_restart, do_replay_restart, _ = _resolve_data_resume_policy(
        train_cfg=restart_cfg,
        consumed_micro_batches=12,
        global_step=17,
    )
    assert start_epoch_restart == 17
    assert do_replay_restart is False


def test_normalize_resume_consumed_micro_batches_clamps_legacy_partial_window():
    consumed, reason = _normalize_resume_consumed_micro_batches(
        consumed_micro_batches=35,
        global_step=11,
        gradient_accumulation_steps=3,
    )
    assert consumed == 33
    assert reason is not None
    assert "clamped_legacy_partial_accumulation_delta" in reason


def test_normalize_resume_consumed_micro_batches_keeps_non_legacy_mismatch():
    consumed, reason = _normalize_resume_consumed_micro_batches(
        consumed_micro_batches=100,
        global_step=11,
        gradient_accumulation_steps=3,
    )
    assert consumed == 100
    assert reason is None


def test_gumbel_sample_rejects_all_forbidden_vocab_mask():
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    logits = torch.zeros((4, 8), dtype=torch.float32)
    forbidden = torch.ones(8, dtype=torch.bool)
    with pytest.raises(ValueError, match="excludes all vocabulary ids"):
        DebertaV3RTDPretrainer._gumbel_sample(
            logits,
            temperature=1.0,
            forbidden_vocab_mask=forbidden,
        )


def test_pretrainer_additional_forbidden_token_ids_extend_config_special_set() -> None:
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    class _TinyBackbone(torch.nn.Module):
        def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
            super().__init__()
            self.embeddings = types.SimpleNamespace(
                word_embeddings=torch.nn.Embedding(vocab_size, hidden_size),
            )

        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            token_type_ids: torch.Tensor | None = None,
            return_dict: bool = True,
        ) -> Any:
            del attention_mask, token_type_ids, return_dict
            hidden = self.embeddings.word_embeddings(input_ids)
            return types.SimpleNamespace(last_hidden_state=hidden)

    cfg = types.SimpleNamespace(
        vocab_size=32,
        hidden_size=8,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        norm_eps=1e-6,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        mask_token_id=3,
    )
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=_TinyBackbone(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size),
        generator_backbone=_TinyBackbone(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size),
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="none",
        additional_forbidden_token_ids=[7, 15, -1, 99],
    )

    expected = {0, 1, 2, 3, 7, 15}
    assert expected.issubset(model._forbidden_sample_token_ids)
    assert int(model._forbidden_sample_token_mask.numel()) == 32
    for tid in expected:
        assert bool(model._forbidden_sample_token_mask[tid].item())


def test_pretrainer_skips_discriminator_when_no_masked_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    class _TinyBackbone(torch.nn.Module):
        def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
            super().__init__()
            self.embeddings = types.SimpleNamespace(
                word_embeddings=torch.nn.Embedding(vocab_size, hidden_size),
            )

        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            token_type_ids: torch.Tensor | None = None,
            return_dict: bool = True,
        ) -> Any:
            del attention_mask, token_type_ids, return_dict
            hidden = self.embeddings.word_embeddings(input_ids)
            return types.SimpleNamespace(last_hidden_state=hidden)

    cfg = types.SimpleNamespace(
        vocab_size=32,
        hidden_size=8,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        norm_eps=1e-6,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        mask_token_id=3,
    )
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=_TinyBackbone(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size),
        generator_backbone=_TinyBackbone(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size),
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="none",
    )

    def _fail_if_called(**kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("discriminator.forward should not run when there are no masked tokens")

    monkeypatch.setattr(model.discriminator, "forward", _fail_if_called)

    input_ids = torch.tensor([[1, 11, 12, 2]], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
    )

    torch.testing.assert_close(out.loss, torch.zeros((), dtype=out.loss.dtype))
    torch.testing.assert_close(out.disc_loss, torch.zeros((), dtype=out.disc_loss.dtype))
    torch.testing.assert_close(out.disc_accuracy, torch.zeros((), dtype=out.disc_accuracy.dtype))
    torch.testing.assert_close(out.disc_token_count, torch.zeros((), dtype=out.disc_token_count.dtype))
    torch.testing.assert_close(out.disc_positive_count, torch.zeros((), dtype=out.disc_positive_count.dtype))


def test_export_discriminator_hf_uses_unwrapped_submodules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import deberta.training.export_helpers as export_mod

    class _Inner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.discriminator = torch.nn.Linear(2, 2)
            self.generator = torch.nn.Linear(2, 2)
            self.disc_config = object()

    inner = _Inner()

    class _Wrapped(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.module = wrapped

    wrapped = _Wrapped(inner)
    called_targets: list[torch.nn.Module] = []

    class _FakeExportModel:
        def save_pretrained(self, path: str, safe_serialization: bool = True) -> None:
            del safe_serialization
            Path(path).mkdir(parents=True, exist_ok=True)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModel = types.SimpleNamespace(from_config=lambda _cfg: _FakeExportModel())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        export_mod,
        "load_intersection_state_dict",
        lambda _model, _state: types.SimpleNamespace(missing_keys=[]),
    )
    monkeypatch.setattr(
        export_mod,
        "merge_embeddings_into_export_backbone",
        lambda **_kwargs: None,
    )

    def _unwrap_model(model: torch.nn.Module, **_kwargs: Any) -> torch.nn.Module:
        assert model is wrapped
        return inner

    def _get_state_dict(model: torch.nn.Module, *, unwrap: bool = True) -> dict[str, torch.Tensor]:
        del unwrap
        called_targets.append(model)
        return {}

    accelerator = FakeAccelerator(
        is_main_process=True,
        unwrap_model_hook=_unwrap_model,
        get_state_dict_hook=_get_state_dict,
    )
    _export_discriminator_hf(
        accelerator=accelerator,
        model=wrapped,  # type: ignore[arg-type]
        tokenizer=DummyTokenizer(),
        output_dir=tmp_path / "export",
        embedding_sharing="none",
    )

    assert called_targets == [inner.discriminator, inner.generator]


def test_export_discriminator_hf_collects_state_dicts_on_non_main_rank(tmp_path: Path) -> None:
    class _Inner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.discriminator = torch.nn.Linear(2, 2)
            self.generator = torch.nn.Linear(2, 2)
            self.disc_config = object()

    inner = _Inner()
    called_targets: list[torch.nn.Module] = []

    def _unwrap_model(model: torch.nn.Module, **_kwargs: Any) -> torch.nn.Module:
        assert model is inner
        return model

    def _get_state_dict(model: torch.nn.Module, *, unwrap: bool = True) -> dict[str, torch.Tensor]:
        del unwrap
        called_targets.append(model)
        return {}

    accelerator = FakeAccelerator(
        is_main_process=False,
        unwrap_model_hook=_unwrap_model,
        get_state_dict_hook=_get_state_dict,
    )

    _export_discriminator_hf(
        accelerator=accelerator,
        model=inner,  # type: ignore[arg-type]
        tokenizer=DummyTokenizer(),
        output_dir=tmp_path / "export",
        embedding_sharing="none",
    )

    assert called_targets == [inner.discriminator, inner.generator]
    assert not (tmp_path / "export").exists()


def test_export_discriminator_hf_subprocess_uses_allow_partial_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = ""

    def _fake_run(
        cmd: list[str],
        *,
        stdout: Any,
        stderr: Any,
        text: bool,
        check: bool,
    ) -> _Proc:
        del stdout, stderr, text, check
        calls.append(list(cmd))
        return _Proc()

    import deberta.training.export_helpers as export_mod

    monkeypatch.setattr(export_mod.subprocess, "run", _fake_run)

    _export_discriminator_hf_subprocess(
        checkpoint_dir=Path("runs/demo/checkpoint-1"),
        output_dir=Path("runs/demo/final_hf"),
    )

    assert calls
    cmd = calls[-1]
    assert cmd[0] == sys.executable
    assert cmd[1:5] == ["-m", "deberta", "export", "runs/demo/checkpoint-1"]
    assert "--allow-partial-export" in cmd


def test_write_export_readme_rope_usage_warns_auto_model_limitation(tmp_path: Path):
    out_dir = tmp_path / "rope-export"
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_export_readme(
        out_dir,
        model_cfg=ModelConfig(backbone_type="rope"),
        data_cfg=DataConfig(max_seq_length=777),
        train_cfg=TrainConfig(max_steps=100),
        embedding_sharing="gdes",
    )

    text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "DebertaRoPEModel.from_pretrained" in text
    assert 'model = AutoModel.from_pretrained("path/to/this/dir")' not in text
    assert "model_type" in text
    assert "| Max sequence length | 777 |" in text


def test_write_export_readme_hf_uses_auto_model_snippet(tmp_path: Path):
    out_dir = tmp_path / "hf-export"
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_export_readme(
        out_dir,
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
        data_cfg=DataConfig(max_seq_length=333),
        train_cfg=TrainConfig(max_steps=100),
        embedding_sharing="gdes",
    )

    text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "AutoModel.from_pretrained" in text
    assert "DebertaRoPEModel.from_pretrained" not in text
    assert "| Max sequence length | 333 |" in text


def test_write_export_readme_uses_export_config_dimensions_when_available(tmp_path: Path):
    out_dir = tmp_path / "hf-export-effective-config"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_cfg = types.SimpleNamespace(
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        max_position_embeddings=4096,
    )

    _write_export_readme(
        out_dir,
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2", hf_model_size="small"),
        export_config=export_cfg,
        data_cfg=None,
        train_cfg=TrainConfig(max_steps=100),
        embedding_sharing="gdes",
    )

    text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "# hf_deberta_v2-768h-6L-12H" in text
    assert "| Max sequence length | 4096 |" in text


def test_build_optimizer_supports_generator_specific_lr():
    model = TinyRTDLikeModel()
    cfg = TrainConfig(learning_rate=1.0e-3, generator_learning_rate=5.0e-4, weight_decay=0.1)
    opt = _build_optimizer(model, cfg)

    lrs = {float(g["lr"]) for g in opt.param_groups}
    assert lrs == {1.0e-3, 5.0e-4}

    # We should have both decay and no-decay groups present.
    wds = {float(g["weight_decay"]) for g in opt.param_groups}
    assert wds == {0.0, 0.1}


def test_build_optimizer_supports_discriminator_specific_lr():
    model = TinyRTDLikeModel()
    cfg = TrainConfig(
        learning_rate=1.0e-3,
        generator_learning_rate=5.0e-4,
        discriminator_learning_rate=2.0e-4,
        weight_decay=0.1,
    )
    opt = _build_optimizer(model, cfg)

    lrs = {float(g["lr"]) for g in opt.param_groups}
    assert lrs == {5.0e-4, 2.0e-4}


def test_build_decoupled_optimizers_support_discriminator_specific_lr():
    model = TinyRTDLikeModel()
    cfg = TrainConfig(
        learning_rate=1.0e-3,
        generator_learning_rate=5.0e-4,
        discriminator_learning_rate=2.0e-4,
        weight_decay=0.1,
        mixed_precision="no",
    )
    gen_opt, disc_opt = _build_decoupled_optimizers(model, cfg, mixed_precision="no")

    assert gen_opt.param_groups
    assert disc_opt.param_groups
    for group in gen_opt.param_groups:
        assert float(group["lr"]) == pytest.approx(5.0e-4)
    for group in disc_opt.param_groups:
        assert float(group["lr"]) == pytest.approx(2.0e-4)


def test_build_optimizer_keeps_fused_for_hf_backbones_compile_bf16(monkeypatch: pytest.MonkeyPatch):
    import deberta.training.pretrain as pretrain_mod

    model = TinyRTDLikeModel()
    cfg = TrainConfig()

    monkeypatch.setattr(pretrain_mod, "_maybe_fused_adamw_kwargs", lambda: {"fused": True})
    opt = pretrain_mod._build_optimizer(
        model,
        cfg,
        backbone_type="hf_deberta_v2",
        compile_enabled=True,
        compile_scope="backbones",
        mixed_precision="bf16",
    )

    assert bool(opt.defaults.get("fused", False)) is True


def test_build_optimizer_keeps_fused_outside_hf_compile_bf16_risk(monkeypatch: pytest.MonkeyPatch):
    import deberta.training.pretrain as pretrain_mod

    model = TinyRTDLikeModel()
    cfg = TrainConfig()

    monkeypatch.setattr(pretrain_mod, "_maybe_fused_adamw_kwargs", lambda: {"fused": True})
    opt = pretrain_mod._build_optimizer(
        model,
        cfg,
        backbone_type="rope",
        compile_enabled=True,
        compile_scope="backbones",
        mixed_precision="bf16",
    )

    assert bool(opt.defaults.get("fused", False)) is True


def test_build_optimizer_raises_adam_epsilon_floor_for_bf16():
    model = TinyRTDLikeModel()
    cfg = TrainConfig(adam_epsilon=1e-8)

    opt = _build_optimizer(model, cfg, mixed_precision="bf16")
    assert float(opt.defaults["eps"]) == pytest.approx(1e-6)

    opt_fp32 = _build_optimizer(model, cfg, mixed_precision="no")
    assert float(opt_fp32.defaults["eps"]) == pytest.approx(1e-8)


def test_config_defaults():
    """Key config defaults match design requirements."""
    train = TrainConfig()
    assert train.mixed_precision == "bf16"
    assert train.sdpa_kernel == "auto"
    assert train.token_weighted_gradient_accumulation is True

    model = ModelConfig()
    assert model.hidden_dropout_prob == pytest.approx(0.0)
    assert model.attention_probs_dropout_prob == pytest.approx(0.0)
    assert model.hf_model_size == "base"
    assert model.ffn_type == "mlp"
    assert model.swiglu_adjust_intermediate is True

    data = DataConfig()
    assert data.block_cross_document_attention is False


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("true", "bf16"),
        ("false", "no"),
        ("TRUE", "bf16"),
        ("False", "no"),
        ("yes", "bf16"),
        ("no", "no"),
        ("on", "bf16"),
        ("off", "no"),
        ("1", "bf16"),
        ("0", "no"),
        ("bfloat16", "bf16"),
        ("none", "no"),
    ],
)
def test_validate_train_config_accepts_mixed_precision_aliases(raw: str, expected: str):
    cfg = TrainConfig(mixed_precision=raw)
    validate_train_config(cfg)
    assert cfg.mixed_precision == expected


def test_validate_train_config_rejects_invalid_mixed_precision():
    cfg = TrainConfig(mixed_precision="fp16")
    with pytest.raises(ValueError, match="train.mixed_precision must be one of: bf16\\|no"):
        validate_train_config(cfg)


def test_validate_train_config_normalizes_compile_scope_and_backend_aliases():
    cfg = TrainConfig(
        torch_compile=True,
        torch_compile_scope="generator_ffn",
        torch_compile_backend="aot-eager",
    )
    validate_train_config(cfg)
    assert cfg.torch_compile_scope == "gen_ffn"
    assert cfg.torch_compile_backend == "aot_eager"


def test_validate_train_config_normalizes_wandb_watch_aliases():
    cfg = LoggingConfig(report_to="wandb", wandb_watch="weights", wandb_watch_log_freq=25)
    validate_logging_config(cfg)
    assert cfg.wandb.watch == "parameters"
    assert cfg.wandb.watch_log_freq == 25


def test_validate_train_config_rejects_invalid_wandb_watch_mode():
    cfg = LoggingConfig(wandb_watch="histogram")
    with pytest.raises(ValueError, match="logging.wandb.watch must be one of"):
        validate_logging_config(cfg)


def test_validate_train_config_rejects_non_positive_wandb_watch_log_freq():
    cfg = LoggingConfig(wandb_watch_log_freq=0)
    with pytest.raises(ValueError, match="logging.wandb.watch_log_freq must be >= 1"):
        validate_logging_config(cfg)


def test_validate_train_config_accepts_resume_data_strategy_values():
    cfg = TrainConfig(
        checkpoint={"resume_data_strategy": "restart_epoch", "resume_replay_max_micro_batches": 123}
    )
    validate_train_config(cfg)
    assert cfg.checkpoint.resume_data_strategy == "restart_epoch"
    assert cfg.checkpoint.resume_replay_max_micro_batches == 123


def test_validate_train_config_rejects_invalid_resume_data_strategy():
    cfg = TrainConfig(checkpoint={"resume_data_strategy": "fast"})
    with pytest.raises(ValueError, match="train.checkpoint.resume_data_strategy must be one of"):
        validate_train_config(cfg)


def test_validate_train_config_rejects_negative_resume_replay_threshold():
    cfg = TrainConfig(checkpoint={"resume_replay_max_micro_batches": -1})
    with pytest.raises(ValueError, match="train.checkpoint.resume_replay_max_micro_batches must be >= 0"):
        validate_train_config(cfg)


def test_token_weighted_micro_objective_matches_full_batch_normalization():
    gen_losses = [torch.tensor(0.5), torch.tensor(0.8)]
    disc_losses = [torch.tensor(0.2), torch.tensor(0.4)]
    gen_counts = [10.0, 30.0]
    disc_counts = [24.0, 16.0]

    gen_total = sum(gen_counts)
    disc_total = sum(disc_counts)
    gen_w = 1.0
    disc_w = 50.0

    micro_0 = _token_weighted_micro_objective(
        gen_loss=gen_losses[0],
        disc_loss=disc_losses[0],
        gen_count=gen_counts[0],
        disc_count=disc_counts[0],
        gen_window_tokens_per_rank=gen_total,
        disc_window_tokens_per_rank=disc_total,
        gen_loss_weight=gen_w,
        disc_loss_weight=disc_w,
    )
    micro_1 = _token_weighted_micro_objective(
        gen_loss=gen_losses[1],
        disc_loss=disc_losses[1],
        gen_count=gen_counts[1],
        disc_count=disc_counts[1],
        gen_window_tokens_per_rank=gen_total,
        disc_window_tokens_per_rank=disc_total,
        gen_loss_weight=gen_w,
        disc_loss_weight=disc_w,
    )

    combined = micro_0 + micro_1
    expected = gen_w * (
        (gen_losses[0] * gen_counts[0] + gen_losses[1] * gen_counts[1]) / gen_total
    ) + disc_w * ((disc_losses[0] * disc_counts[0] + disc_losses[1] * disc_counts[1]) / disc_total)
    torch.testing.assert_close(combined, expected)


def test_resolve_window_token_denominators_clamps_and_flags_zero_windows():
    gen_denom, disc_denom, gen_zero, disc_zero = _resolve_window_token_denominators(
        gen_window_tokens_per_rank_raw=8.0,
        disc_window_tokens_per_rank_raw=3.0,
    )
    assert gen_denom == pytest.approx(8.0)
    assert disc_denom == pytest.approx(3.0)
    assert gen_zero is False
    assert disc_zero is False

    gen_denom, disc_denom, gen_zero, disc_zero = _resolve_window_token_denominators(
        gen_window_tokens_per_rank_raw=0.0,
        disc_window_tokens_per_rank_raw=-2.0,
    )
    assert gen_denom == pytest.approx(1.0)
    assert disc_denom == pytest.approx(1.0)
    assert gen_zero is True
    assert disc_zero is True


def test_finalize_window_metric_loss_averages_non_token_weighted_windows():
    total = torch.tensor(3.0)
    out = _finalize_window_metric_loss(accumulated_loss=total, ga_steps=3, token_weighted_ga=False)
    torch.testing.assert_close(out, torch.tensor(1.0))


def test_finalize_window_metric_loss_passthrough_for_token_weighted_windows():
    total = torch.tensor(1.2345)
    out = _finalize_window_metric_loss(accumulated_loss=total, ga_steps=8, token_weighted_ga=True)
    torch.testing.assert_close(out, total)


def test_scale_loss_for_backward_passthrough_when_not_token_weighted():
    loss = torch.tensor(2.0)
    out = _scale_loss_for_backward(loss=loss, ga_steps=8, token_weighted_ga=False)
    torch.testing.assert_close(out, loss)


def test_scale_loss_for_backward_cancels_accelerate_ga_division_for_token_weighted():
    loss = torch.tensor(2.0)
    out = _scale_loss_for_backward(loss=loss, ga_steps=4, token_weighted_ga=True)
    torch.testing.assert_close(out, torch.tensor(8.0))


def test_build_training_collator_propagates_packed_sequences_flag():
    tokenizer = DummyTokenizer(vocab_size=64)
    train_cfg = TrainConfig(mlm_probability=0.2, mlm_max_ngram=2)
    collator = _build_training_collator(
        tokenizer=tokenizer,
        train_cfg=train_cfg,
        packed_sequences=True,
        block_cross_document_attention=True,
    )
    assert collator._packed_sequences is True
    assert collator._block_cross_document_attention is True


def test_should_clip_gradients_on_sync_steps():
    assert _should_clip_gradients(sync_gradients=False, max_grad_norm=1.0) is False
    assert _should_clip_gradients(sync_gradients=True, max_grad_norm=None) is False
    assert _should_clip_gradients(sync_gradients=True, max_grad_norm=0.0) is False
    assert _should_clip_gradients(sync_gradients=True, max_grad_norm=-1.0) is False
    assert _should_clip_gradients(sync_gradients=True, max_grad_norm=1.0) is True


def test_has_nonfinite_grad_norm_any_rank_uses_reduced_flag():
    accel = FakeAccelerator()

    def _reduce(_self: FakeAccelerator, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        assert reduction == "sum"
        return tensor + 1

    accel.reduce = types.MethodType(_reduce, accel)  # type: ignore[method-assign]
    assert _has_nonfinite_grad_norm_any_rank(accelerator=accel, grad_norm=1.0) is True
    assert _has_nonfinite_grad_norm_any_rank(accelerator=accel, grad_norm=float("inf")) is True


def test_sync_discriminator_embeddings_if_available_is_noop_without_hook():
    model = torch.nn.Linear(2, 2)
    _sync_discriminator_embeddings_if_available(model)


def test_sync_discriminator_embeddings_if_available_calls_hook_once():
    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def sync_discriminator_embeddings_from_generator(self) -> None:
            self.calls += 1

    model = _Model()
    _sync_discriminator_embeddings_if_available(model)
    assert model.calls == 1


def test_sync_discriminator_embeddings_if_available_skips_fsdp_summon_when_not_gdes(
    monkeypatch: pytest.MonkeyPatch,
):
    class _FakeCtx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeFSDP(torch.nn.Module):
        summon_calls = 0

        @staticmethod
        def summon_full_params(_module: torch.nn.Module, *, recurse: bool, writeback: bool):
            del recurse, writeback
            _FakeFSDP.summon_calls += 1
            return _FakeCtx()

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.embedding_sharing = "es"
            self._gdes_synced_embeddings = [object()]

        def sync_discriminator_embeddings_from_generator(self) -> None:
            self.calls += 1

    fake_fsdp = types.ModuleType("torch.distributed.fsdp")
    fake_fsdp.FullyShardedDataParallel = _FakeFSDP
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fake_fsdp)

    model = _FakeFSDP()
    _sync_discriminator_embeddings_if_available(model)

    assert model.calls == 0
    assert _FakeFSDP.summon_calls == 0


def test_sync_discriminator_embeddings_if_available_uses_non_recursive_fsdp_summon(
    monkeypatch: pytest.MonkeyPatch,
):
    class _FakeCtx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeFSDP(torch.nn.Module):
        recurse_args: list[bool] = []
        writeback_args: list[bool] = []

        @staticmethod
        def summon_full_params(_module: torch.nn.Module, *, recurse: bool, writeback: bool):
            _FakeFSDP.recurse_args.append(bool(recurse))
            _FakeFSDP.writeback_args.append(bool(writeback))
            return _FakeCtx()

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.embedding_sharing = "gdes"
            self._gdes_synced_embeddings = [object()]

        def sync_discriminator_embeddings_from_generator(self) -> None:
            self.calls += 1

    fake_fsdp = types.ModuleType("torch.distributed.fsdp")
    fake_fsdp.FullyShardedDataParallel = _FakeFSDP
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fake_fsdp)

    model = _FakeFSDP()
    _sync_discriminator_embeddings_if_available(model)

    assert model.calls == 1
    assert _FakeFSDP.recurse_args == [False]
    assert _FakeFSDP.writeback_args == [True]


def test_sync_discriminator_embeddings_if_available_propagates_fsdp_summon_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    class _FakeFSDP(torch.nn.Module):
        @staticmethod
        def summon_full_params(_module: torch.nn.Module, *, recurse: bool, writeback: bool):
            del recurse, writeback
            raise RuntimeError("boom")

        def __init__(self) -> None:
            super().__init__()
            self.embedding_sharing = "gdes"
            self._gdes_synced_embeddings = [object()]

        def sync_discriminator_embeddings_from_generator(self) -> None:
            raise AssertionError("sync hook should not run when summon_full_params fails")

    fake_fsdp = types.ModuleType("torch.distributed.fsdp")
    fake_fsdp.FullyShardedDataParallel = _FakeFSDP
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fake_fsdp)

    with pytest.raises(RuntimeError, match="boom"):
        _sync_discriminator_embeddings_if_available(_FakeFSDP())


def test_sync_discriminator_embeddings_if_available_supports_fsdp2_unshard_reshard(
    monkeypatch: pytest.MonkeyPatch,
):
    class _Handle:
        def __init__(self) -> None:
            self.wait_calls = 0

        def wait(self) -> None:
            self.wait_calls += 1

    class _FakeFSDPModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.unshard_calls = 0
            self.reshard_calls = 0
            self.handle = _Handle()
            self.embedding_sharing = "gdes"
            self._gdes_synced_embeddings = [object()]

        def unshard(self, async_op: bool = False):
            assert bool(async_op) is False
            self.unshard_calls += 1
            return self.handle

        def reshard(self) -> None:
            self.reshard_calls += 1

        def sync_discriminator_embeddings_from_generator(self) -> None:
            self.calls += 1

    class _DummyFSDP(torch.nn.Module):
        pass

    fake_fsdp = types.ModuleType("torch.distributed.fsdp")
    fake_fsdp.FullyShardedDataParallel = _DummyFSDP
    fake_fsdp.FSDPModule = _FakeFSDPModule
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fake_fsdp)

    model = _FakeFSDPModule()
    _sync_discriminator_embeddings_if_available(model)

    assert model.calls == 1
    assert model.unshard_calls == 1
    assert model.handle.wait_calls == 1
    assert model.reshard_calls == 1


def test_count_rtd_tokens_for_batch_keeps_masked_positions_active_for_discriminator():
    batch = {
        "input_ids": torch.tensor([[1, 3, 11, 2, 0]], dtype=torch.long),
        "labels": torch.tensor([[-100, 99, -100, -100, -100]], dtype=torch.long),
    }
    gen_count, disc_count = _count_rtd_tokens_for_batch(
        batch,
        pad_token_id=0,
    )
    assert gen_count == pytest.approx(1.0)
    assert disc_count == pytest.approx(4.0)


def test_count_input_tokens_for_batch_with_various_mask_shapes():
    """Token counting handles 2D, 3D, 4D masks and missing masks."""
    # 2D attention_mask → sums mask.
    batch_2d = {
        "input_ids": torch.tensor([[10, 11, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
    }
    assert _count_input_tokens_for_batch(batch_2d) == pytest.approx(2.0)

    # No mask → fallback to input numel.
    batch_no_mask = {"input_ids": torch.tensor([[10, 11, 12], [13, 14, 15]], dtype=torch.long)}
    assert _count_input_tokens_for_batch(batch_no_mask) == pytest.approx(6.0)

    # 3D pairwise mask → uses diagonal token activity.
    pair_3d = torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.long)
    batch_3d = {
        "input_ids": torch.tensor([[10, 11, 0, 0]], dtype=torch.long),
        "attention_mask": pair_3d,
    }
    assert _count_input_tokens_for_batch(batch_3d) == pytest.approx(2.0)

    # 4D pairwise mask → same logic, extra head dim.
    batch_4d = {
        "input_ids": torch.tensor([[10, 11, 0, 0]], dtype=torch.long),
        "attention_mask": pair_3d.unsqueeze(1),
    }
    assert _count_input_tokens_for_batch(batch_4d) == pytest.approx(2.0)


def test_compute_disc_active_mask_preserves_masked_non_special_tokens():
    from deberta.training.loop_utils import _compute_disc_active_mask

    mask = _compute_disc_active_mask(
        input_ids=torch.tensor([[1, 11, 2, 13, 0]], dtype=torch.long),
        labels=torch.tensor([[-100, 99, -100, 77, -100]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
        pad_token_id=0,
    )

    expected = torch.tensor([[True, True, True, True, True]], dtype=torch.bool)
    assert torch.equal(mask, expected)


def test_attention_mask_to_active_tokens_uses_diagonal_activity_with_pad_for_3d_masks():
    input_ids = torch.tensor([[11, 12, 13, 0]], dtype=torch.long)
    # Row 2 has diagonal=False but off-diagonal keep=True. Active-token recovery
    # must respect diagonal activity even when pad_token_id is available.
    # Row 3 is explicit pad and must remain inactive.
    pair_keep = torch.tensor(
        [
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        ],
        dtype=torch.bool,
    )
    active = attention_mask_to_active_tokens(
        input_ids=input_ids,
        attention_mask=pair_keep,
        pad_token_id=0,
    )
    expected = torch.tensor([[True, True, False, False]], dtype=torch.bool)
    assert torch.equal(active, expected)


def test_normalize_keep_mask_rejects_floating_masks() -> None:
    with pytest.raises(ValueError, match="Floating-point masks are ambiguous"):
        _ = normalize_keep_mask(torch.tensor([[0.0, -1.0]], dtype=torch.float32))


def test_attention_mask_to_active_tokens_rejects_floating_masks() -> None:
    input_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
    with pytest.raises(ValueError, match="Floating-point masks are ambiguous"):
        _ = attention_mask_to_active_tokens(
            input_ids=input_ids,
            attention_mask=torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float32),
            pad_token_id=None,
        )


def test_attention_mask_to_active_tokens_uses_diagonal_not_any_for_3d_no_pad():
    """Without pad_token_id, 3D fallback must use diagonal, not any(dim=-1).

    Construct a mask where off-diagonal keeps in an inactive row make
    any(dim=-1) return True but the diagonal is False.
    """
    input_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
    # Row 2: diagonal=False but has off-diagonal True → any(dim=-1) would wrongly report active.
    pair_keep = torch.tensor(
        [
            [
                [True, True, False],
                [True, True, False],
                [True, False, False],
            ]
        ],
        dtype=torch.bool,
    )
    active = attention_mask_to_active_tokens(
        input_ids=input_ids,
        attention_mask=pair_keep,
        pad_token_id=None,
    )
    # Diagonal: [True, True, False]
    expected = torch.tensor([[True, True, False]], dtype=torch.bool)
    assert torch.equal(active, expected), f"Expected diagonal-based result {expected}, got {active}"


def test_attention_mask_to_active_tokens_uses_diagonal_for_4d_no_pad():
    """4D fallback without pad_token_id must use diagonal after squeezing heads."""
    input_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
    # (B=1, H=1, S=3, S=3) — row 2 has off-diagonal True but diagonal False.
    pair_keep = torch.tensor(
        [
            [
                [
                    [True, True, False],
                    [True, True, False],
                    [True, False, False],
                ]
            ]
        ],
        dtype=torch.bool,
    )
    active = attention_mask_to_active_tokens(
        input_ids=input_ids,
        attention_mask=pair_keep,
        pad_token_id=None,
    )
    expected = torch.tensor([[True, True, False]], dtype=torch.bool)
    assert torch.equal(active, expected)


def test_attention_mask_to_active_tokens_handles_4d_broadcast_no_pad():
    input_ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    # Broadcast keep mask shape (B,1,1,S).
    broadcast_keep = torch.tensor([[[[True, True, False, False]]]], dtype=torch.bool)
    active = attention_mask_to_active_tokens(
        input_ids=input_ids,
        attention_mask=broadcast_keep,
        pad_token_id=None,
    )
    expected = torch.tensor([[True, True, False, False]], dtype=torch.bool)
    assert torch.equal(active, expected)


def test_attention_mask_to_active_tokens_uses_diagonal_activity_with_pad_for_4d_masks():
    input_ids = torch.tensor([[10, 11, 12, 0]], dtype=torch.long)
    pair_keep = torch.tensor(
        [
            [
                [
                    [True, True, False, False],
                    [True, True, False, False],
                    [True, False, False, False],
                    [True, False, False, False],
                ]
            ]
        ],
        dtype=torch.bool,
    )
    active = attention_mask_to_active_tokens(
        input_ids=input_ids,
        attention_mask=pair_keep,
        pad_token_id=0,
    )
    expected = torch.tensor([[True, True, False, False]], dtype=torch.bool)
    assert torch.equal(active, expected)


def test_build_optimizer_marks_scalar_params_as_no_decay():
    train_cfg = TrainConfig()

    class _RegressionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = torch.nn.Linear(4, 4)
            self.generator.alpha = torch.nn.Parameter(torch.tensor(1.0))
            self.generator_lm_head = torch.nn.Linear(4, 4)
            self.discriminator = torch.nn.Linear(4, 4)
            self.discriminator.alpha = torch.nn.Parameter(torch.tensor(2.0))
            self.discriminator_norm = torch.nn.LayerNorm(4)

    model = _RegressionModel()
    opt = _build_optimizer(model, train_cfg)

    def _group_for(param: torch.nn.Parameter) -> float:
        for g in opt.param_groups:
            for p in g["params"]:
                if p is param:
                    return float(g["weight_decay"])
        raise AssertionError("Parameter missing from optimizer groups")

    assert _group_for(model.generator.alpha) == pytest.approx(0.0)
    assert _group_for(model.discriminator.alpha) == pytest.approx(0.0)
    assert _group_for(model.discriminator_norm.weight) == pytest.approx(0.0)
    assert _group_for(model.generator.weight) == pytest.approx(train_cfg.weight_decay)


def test_build_optimizer_applies_decay_to_high_rank_bias_parameters():
    train_cfg = TrainConfig()

    class _BiasMatrixModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.randn(4, 4))

    class _RegressionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = torch.nn.Linear(4, 4)
            self.generator_bias = _BiasMatrixModule()
            self.generator_lm_head = torch.nn.Linear(4, 4)
            self.discriminator = torch.nn.Linear(4, 4)

    model = _RegressionModel()
    opt = _build_optimizer(model, train_cfg)

    def _group_for(param: torch.nn.Parameter) -> float:
        for g in opt.param_groups:
            for p in g["params"]:
                if p is param:
                    return float(g["weight_decay"])
        raise AssertionError("Parameter missing from optimizer groups")

    assert _group_for(model.generator_bias.bias) == pytest.approx(train_cfg.weight_decay)


def test_normalize_mixed_precision_accepts_bool_and_synonyms():
    assert normalize_mixed_precision("bf16") == "bf16"
    assert normalize_mixed_precision("none") == "no"
    assert normalize_mixed_precision(False) == "no"
    assert normalize_mixed_precision(True) == "bf16"

    with pytest.raises(ValueError, match="train.mixed_precision must be one of: bf16\\|no"):
        normalize_mixed_precision("fp16")


def test_resolve_effective_mixed_precision_or_raise_errors_for_bf16_preflight_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    import deberta.training.pretrain as pretrain_mod

    monkeypatch.setattr(pretrain_mod, "_bf16_runtime_sanity_check", lambda: False)
    with pytest.raises(RuntimeError, match="mixed_precision=no explicitly"):
        _resolve_effective_mixed_precision_or_raise("bf16")

    assert _resolve_effective_mixed_precision_or_raise("no") == "no"


def test_resolve_compile_enabled_or_raise_errors_when_torch_compile_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    import deberta.training.pretrain as pretrain_mod

    monkeypatch.delattr(pretrain_mod.torch, "compile", raising=False)
    with pytest.raises(RuntimeError, match="does not expose torch.compile"):
        _resolve_compile_enabled_or_raise(True)

    assert _resolve_compile_enabled_or_raise(False) is False


def test_normalizer_aliases_and_rejection():
    """All config normalizer functions accept documented aliases and reject unknown values."""
    cases: list[tuple[Any, list[tuple[str, str]], str, str]] = [
        (
            _normalize_torch_compile_mode,
            [
                ("default", "default"),
                ("reduce_overhead", "reduce-overhead"),
                ("max_autotune", "max-autotune"),
                ("max-autotune-no-cudagraphs", "max-autotune-no-cudagraphs"),
            ],
            "fastest",
            "train.compile.mode",
        ),
        (
            _normalize_torch_compile_scope,
            [
                ("auto", "auto"),
                ("backbone", "backbones"),
                ("full", "backbones"),
                ("encoder", "encoder"),
                ("generator_encoder", "gen_encoder"),
                ("disc-encoder", "disc_encoder"),
                ("ffn", "ffn"),
                ("generator_ffn", "gen_ffn"),
                ("disc_ffn", "disc_ffn"),
            ],
            "all",
            "train.compile.scope",
        ),
        (
            _normalize_torch_compile_backend,
            [("inductor", "inductor"), ("aot-eager", "aot_eager")],
            "xla",
            "train.compile.backend",
        ),
        (
            _normalize_wandb_watch,
            [
                ("gradients", "gradients"),
                ("grad", "gradients"),
                ("weights", "parameters"),
                ("all", "all"),
                ("off", "none"),
            ],
            "full_histograms",
            "logging.wandb.watch",
        ),
        (
            _normalize_hf_attention_kernel,
            [
                ("dynamic", "dynamic"),
                ("cache", "cached_bmm"),
                ("cached-bmm", "cached_bmm"),
                ("safe", "stable"),
                ("stable", "stable"),
            ],
            "einsum",
            "model.hf.attention_kernel",
        ),
        (
            _normalize_sdpa_kernel,
            [
                ("auto", "auto"),
                ("flashattention", "flash"),
                ("mem-efficient", "mem_efficient"),
                ("math", "math"),
            ],
            "best",
            "train.sdpa_kernel",
        ),
    ]
    for fn, valid_pairs, invalid_input, error_pattern in cases:
        for raw, expected in valid_pairs:
            assert fn(raw) == expected, f"{fn.__name__}({raw!r}) should be {expected!r}"
        with pytest.raises(ValueError, match=error_pattern):
            fn(invalid_input)


def test_force_legacy_tf32_for_compile_modes():
    assert _should_force_legacy_tf32_for_compile(torch_compile=False, compile_mode="max-autotune") is False
    assert _should_force_legacy_tf32_for_compile(torch_compile=True, compile_mode="default") is False
    assert _should_force_legacy_tf32_for_compile(torch_compile=True, compile_mode="max-autotune") is True
    assert (
        _should_force_legacy_tf32_for_compile(torch_compile=True, compile_mode="max-autotune-no-cudagraphs")
        is True
    )


def test_stabilize_compile_attention_mask_hf_deberta_v2():
    # Missing mask stays absent — backbone handles None via no-mask fast path.
    batch1 = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)}
    out1 = _stabilize_compile_attention_mask(
        batch=batch1,
        compile_enabled=True,
        compile_scope="backbones",
        backbone_type="hf_deberta_v2",
    )
    assert "attention_mask" not in out1

    # FFN scope does not inject mask.
    batch2 = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    out2 = _stabilize_compile_attention_mask(
        batch=dict(batch2),
        compile_enabled=True,
        compile_scope="ffn",
        backbone_type="hf_deberta_v2",
    )
    assert "attention_mask" not in out2

    # Non-bool mask gets converted to bool.
    batch3 = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
    }
    out3 = _stabilize_compile_attention_mask(
        batch=batch3,
        compile_enabled=True,
        compile_scope="encoder",
        backbone_type="hf_deberta_v2",
    )
    assert out3["attention_mask"].dtype == torch.bool
    assert torch.equal(out3["attention_mask"], torch.tensor([[True, True, False]], dtype=torch.bool))


def test_stabilize_compile_attention_mask_rope_doc_blocking():
    # Stabilizer is now a no-op for RoPE — mask shape churn is handled by
    # _resolve_compile_scope auto-downgrading to FFN instead.

    # RoPE + doc-blocking + compile: no mask materialization.
    batch1 = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    out1 = _stabilize_compile_attention_mask(
        batch=batch1,
        compile_enabled=True,
        compile_scope="backbones",
        backbone_type="rope",
        block_cross_document_attention=True,
    )
    assert "attention_mask" not in out1

    # RoPE without doc-blocking: no mask injection.
    batch2 = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    out2 = _stabilize_compile_attention_mask(
        batch=dict(batch2),
        compile_enabled=True,
        compile_scope="backbones",
        backbone_type="rope",
        block_cross_document_attention=False,
    )
    assert "attention_mask" not in out2

    # Compile disabled: no mask injection regardless.
    batch3 = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    out3 = _stabilize_compile_attention_mask(
        batch=dict(batch3),
        compile_enabled=False,
        compile_scope="backbones",
        backbone_type="rope",
        block_cross_document_attention=True,
    )
    assert "attention_mask" not in out3


def test_resolve_compile_scope_auto_prefers_backbones_except_rope_doc_blocking():
    scope, reason = _resolve_compile_scope(
        requested_scope="auto",
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
        compile_mode="default",
        compile_backend="inductor",
    )
    assert scope == "backbones"
    assert reason is None

    scope, reason = _resolve_compile_scope(
        requested_scope="auto",
        model_cfg=ModelConfig(backbone_type="rope"),
        compile_mode="default",
        compile_backend="inductor",
    )
    assert scope == "backbones"
    assert reason is None

    # RoPE + doc-blocking auto-downgrades to FFN to avoid mask shape churn.
    scope, reason = _resolve_compile_scope(
        requested_scope="auto",
        model_cfg=ModelConfig(backbone_type="rope"),
        compile_mode="default",
        compile_backend="inductor",
        block_cross_document_attention=True,
    )
    assert scope == "ffn"
    assert reason is not None

    scope, reason = _resolve_compile_scope(
        requested_scope="auto",
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
        compile_mode="default",
        compile_backend="aot_eager",
    )
    assert scope == "backbones"
    assert reason is None

    scope, reason = _resolve_compile_scope(
        requested_scope="backbones",
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
        compile_mode="default",
        compile_backend="inductor",
    )
    assert scope == "backbones"
    assert reason is None


def test_full_backbone_hf_inductor_warning_is_disabled():
    assert (
        _full_backbone_hf_inductor_warning(
            model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
            compile_enabled=True,
            compile_scope="backbones",
            compile_backend="inductor",
        )
        is None
    )
    assert (
        _full_backbone_hf_inductor_warning(
            model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
            compile_enabled=True,
            compile_scope="backbones",
            compile_backend="aot_eager",
        )
        is None
    )
    assert (
        _full_backbone_hf_inductor_warning(
            model_cfg=ModelConfig(backbone_type="rope"),
            compile_enabled=True,
            compile_scope="backbones",
            compile_backend="inductor",
        )
        is None
    )


def test_compile_controls_do_not_reference_environment_variables():
    import inspect

    import deberta.training.pretrain as pretrain_mod

    source = inspect.getsource(pretrain_mod)
    assert "DEBERTA_COMPILE_SCOPE" not in source
    assert "DEBERTA_COMPILE_BACKEND" not in source
    assert "DEBERTA_HF_ATTN_KERNEL" not in source


def test_main_cli_train_subcommand_loads_yaml_and_applies_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "  packing:",
                "    max_seq_length: 32",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(["train", str(cfg_path), "--train.max_steps", "7"])

    assert "train_cfg" in seen
    assert seen["data_cfg"].packing.max_seq_length == 32
    assert seen["train_cfg"].max_steps == 7
    assert seen["config_path"] == cfg_path


def test_main_cli_train_honors_explicit_yaml_warmup_value_for_hf_backbone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "model:",
                "  profile: deberta_v3_parity",
                "  backbone_type: hf_deberta_v2",
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
                "optim:",
                "  scheduler:",
                "    warmup_steps: 1000",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(["train", str(cfg_path)])

    assert "train_cfg" in seen
    assert int(seen["train_cfg"].warmup_steps) == 1000
    assert seen["config_path"] == cfg_path
    err = capsys.readouterr().err
    assert "optim.scheduler.warmup_steps" not in err


def test_main_cli_train_reports_when_file_value_is_changed_by_cli_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(["train", str(cfg_path), "--train.max_steps", "7"])

    assert "train_cfg" in seen
    assert int(seen["train_cfg"].max_steps) == 7
    err = capsys.readouterr().err
    assert "train.max_steps: 5 -> 7 (CLI override (--train.max_steps))" in err


def test_main_cli_train_reports_when_runtime_mutation_changes_loaded_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_apply_profile_defaults(*, model_cfg, train_cfg, optim_cfg):
        del model_cfg, optim_cfg
        object.__setattr__(train_cfg, "max_steps", int(train_cfg.max_steps) + 1)

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "apply_profile_defaults", _fake_apply_profile_defaults)
    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(["train", str(cfg_path)])

    assert int(seen["train_cfg"].max_steps) == 6
    err = capsys.readouterr().err
    assert "train.max_steps: 5 -> 6 (runtime normalization/defaulting)" in err


def test_main_cli_train_supports_dotted_overrides_with_type_casting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
                "optim:",
                "  adam:",
                "    epsilon: 1e-6",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(
        [
            "train",
            str(cfg_path),
            "--train.max_steps",
            "11",
            "--optim.adam.epsilon",
            "1e-5",
            "--model.hf.max_position_embeddings",
            "640",
        ]
    )

    assert int(seen["train_cfg"].max_steps) == 11
    assert float(seen["train_cfg"].adam_epsilon) == pytest.approx(1e-5)
    assert int(seen["model_cfg"].hf_max_position_embeddings) == 640
    err = capsys.readouterr().err
    assert "train.max_steps: 5 -> 11 (CLI override (--train.max_steps))" in err


def test_main_cli_train_supports_null_for_optional_numeric_dotted_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "model:",
                "  hf:",
                "    max_position_embeddings: 640",
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(
        [
            "train",
            str(cfg_path),
            "--model.hf.max_position_embeddings",
            "null",
        ]
    )

    assert seen["model_cfg"].hf_max_position_embeddings is None
    err = capsys.readouterr().err
    assert (
        "model.hf.max_position_embeddings: 640 -> None (CLI override (--model.hf.max_position_embeddings))"
    ) in err


def test_main_cli_train_supports_dotted_overrides_for_extended_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(
        [
            "train",
            str(cfg_path),
            "--optim.scheduler.warmup_steps",
            "333",
            "--logging.wandb.watch",
            "all",
        ]
    )

    assert int(seen["train_cfg"].warmup_steps) == 333
    assert seen["train_cfg"].wandb_watch == "all"


def test_main_cli_train_rejects_invalid_dotted_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_mod, "run_pretraining", lambda **_: None)
    with pytest.raises(SystemExit):
        cli_mod.main(["train", str(cfg_path), "--train.no_such_field", "1"])


def test_main_cli_train_with_config_and_preset_applies_model_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    cfg_path = tmp_path / "train.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model": {"backbone_type": "rope"},
                "data": {"source": {"dataset_name": "demo-dataset"}, "packing": {"max_seq_length": 128}},
                "train": {"max_steps": 11},
            }
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(["train", str(cfg_path), "--preset", "deberta-v3-base"])

    assert "train_cfg" in seen
    assert seen["model_cfg"].backbone_type == "hf_deberta_v2"
    assert seen["data_cfg"].source.dataset_name == "demo-dataset"
    assert seen["data_cfg"].packing.max_seq_length == 128
    assert seen["train_cfg"].max_steps == 11
    assert seen["config_path"] == cfg_path
    out = capsys.readouterr().out
    assert "model-only overrides (config file provided)" in out


def test_main_cli_train_preset_without_config_applies_defaults_and_cli_overrides(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(["train", "--preset", "deberta-v3-base", "--train.max_steps", "123"])

    assert seen["config_path"] is None
    assert seen["model_cfg"].backbone_type == "hf_deberta_v2"
    assert seen["model_cfg"].embedding_sharing == "gdes"
    assert seen["data_cfg"].source.dataset_name == "HuggingFaceFW/fineweb-edu"
    assert seen["data_cfg"].source.dataset_config_name == "default"
    assert seen["train_cfg"].max_steps == 123
    assert seen["train_cfg"].report_to == "none"
    out = capsys.readouterr().out
    assert "model+data+train+optim+logging defaults (no config file)" in out


def test_main_cli_train_dry_run_calls_preflight_and_skips_training(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*args, **kwargs):
        del args, kwargs
        raise AssertionError("run_pretraining should not be called for --dry-run")

    def _fake_dry_run(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path
        return {
            "checkpoint_output_dir": "runs/demo",
            "logging_output_dir": "runs/demo",
            "resume_checkpoint": None,
            "effective_compile_scope": "ffn",
            "sample_batch_shape": (1, 8),
            "sample_active_tokens": 8.0,
            "tokenizer_vocab_size": 32000,
        }

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    monkeypatch.setattr(cli_mod, "run_pretraining_dry_run", _fake_dry_run)

    cli_mod.main(
        [
            "train",
            "--data.source.dataset_name",
            "HuggingFaceFW/fineweb-edu",
            "--train.max_steps",
            "5",
            "--dry-run",
        ]
    )
    assert "train_cfg" in seen
    assert int(seen["train_cfg"].max_steps) == 5


def test_main_cli_train_explicit_argv_skips_fast_exit(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {"ran": False, "exit_called": False}

    def _fake_run_train(ns: argparse.Namespace, *, dotflag_map: dict[str, str]) -> None:
        del ns, dotflag_map
        seen["ran"] = True

    def _fake_exit(code: int) -> None:
        del code
        seen["exit_called"] = True
        raise AssertionError("os._exit should not be called when main(argv=...) is used.")

    monkeypatch.setattr(cli_mod, "_run_train", _fake_run_train)
    monkeypatch.setattr(cli_mod.os, "_exit", _fake_exit)

    cli_mod.main(["train", "--dry-run"])

    assert seen["ran"] is True
    assert seen["exit_called"] is False


def test_main_cli_train_implicit_argv_fast_exit_enabled(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {"ran": False}

    def _fake_run_train(ns: argparse.Namespace, *, dotflag_map: dict[str, str]) -> None:
        del ns, dotflag_map
        seen["ran"] = True

    def _fake_exit(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(cli_mod, "_run_train", _fake_run_train)
    monkeypatch.setattr(cli_mod.os, "_exit", _fake_exit)
    monkeypatch.setenv("DEBERTA_FAST_EXIT_AFTER_TRAIN", "1")
    monkeypatch.setattr(cli_mod.sys, "argv", ["deberta", "train", "--dry-run"])

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main()

    assert int(exc_info.value.code) == 0
    assert seen["ran"] is True


def test_train_parser_accepts_dry_run_flag():
    parser = cli_mod._build_main_parser()
    ns = parser.parse_args(
        [
            "train",
            "--data.source.dataset_name",
            "HuggingFaceFW/fineweb-edu",
            "--train.max_steps",
            "5",
            "--dry-run",
        ]
    )
    assert ns.command == "train"
    assert ns.dry_run is True


def test_train_parser_accepts_preset_flag():
    parser = cli_mod._build_main_parser()
    ns = parser.parse_args(["train", "--preset", "deberta-v3-base", "--dry-run"])
    assert ns.command == "train"
    assert ns.preset == "deberta-v3-base"
    assert ns.dry_run is True


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
        parser.parse_args(["train", "--model.rope.norm_arch", "invalid"])

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--logging.backend", "invalid"])


def test_export_parser_rejects_conflicting_boolean_flags():
    parser = argparse.ArgumentParser(prog="deberta export")
    add_export_arguments(parser)
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


def test_validate_train_config_rejects_overwrite_with_resume_conflict():
    with pytest.raises(ValueError, match="cannot be combined with train.checkpoint.resume_from_checkpoint"):
        validate_train_config(
            TrainConfig(
                overwrite_output_dir=True,
                resume_from_checkpoint="auto",
            )
        )


def test_validate_train_config_normalizes_blank_resume_hint_to_none():
    cfg = TrainConfig(resume_from_checkpoint="   ")
    validate_train_config(cfg)
    assert cfg.checkpoint.resume_from_checkpoint is None


def test_validate_train_config_trims_resume_hint():
    cfg = TrainConfig(resume_from_checkpoint=" auto ")
    validate_train_config(cfg)
    assert cfg.checkpoint.resume_from_checkpoint == "auto"


def test_model_config_default_backbone_is_hf_deberta_v2() -> None:
    cfg = ModelConfig()
    assert cfg.backbone_type == "hf_deberta_v2"


def test_apply_profile_defaults_applies_parity_values_when_unset() -> None:
    model_cfg = ModelConfig(profile="deberta_v3_parity")
    train_cfg = TrainConfig()
    optim_cfg = OptimConfig()

    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg, optim_cfg=optim_cfg)

    assert model_cfg.backbone_type == "hf_deberta_v2"
    assert model_cfg.embedding_sharing == "gdes"
    assert model_cfg.hf_attention_kernel == "dynamic"
    assert train_cfg.mask_token_prob == pytest.approx(1.0)
    assert train_cfg.random_token_prob == pytest.approx(0.0)
    assert train_cfg.mlm_max_ngram == 1
    assert train_cfg.disc_loss_weight == pytest.approx(10.0)
    assert optim_cfg.adam.epsilon == pytest.approx(1e-6)
    assert optim_cfg.scheduler.warmup_steps == 10_000
    assert train_cfg.token_weighted_gradient_accumulation is True


def test_apply_profile_defaults_applies_hf_parity_values_without_parity_profile() -> None:
    model_cfg = ModelConfig(profile="modern", backbone_type="hf_deberta_v2")
    train_cfg = TrainConfig()
    optim_cfg = OptimConfig()

    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg, optim_cfg=optim_cfg)

    assert train_cfg.mask_token_prob == pytest.approx(1.0)
    assert train_cfg.random_token_prob == pytest.approx(0.0)
    assert train_cfg.disc_loss_weight == pytest.approx(10.0)
    assert optim_cfg.adam.epsilon == pytest.approx(1e-6)
    assert optim_cfg.scheduler.warmup_steps == 10_000
    assert train_cfg.token_weighted_gradient_accumulation is True


def test_apply_profile_defaults_keeps_rope_modern_defaults() -> None:
    model_cfg = ModelConfig(profile="modern", backbone_type="rope")
    train_cfg = TrainConfig()
    optim_cfg = OptimConfig()

    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg, optim_cfg=optim_cfg)

    assert train_cfg.mask_token_prob == pytest.approx(0.8)
    assert train_cfg.random_token_prob == pytest.approx(0.1)
    assert train_cfg.disc_loss_weight == pytest.approx(50.0)
    assert optim_cfg.adam.epsilon == pytest.approx(1e-8)
    assert optim_cfg.scheduler.warmup_steps == 1_000
    assert train_cfg.token_weighted_gradient_accumulation is True


def test_apply_profile_defaults_keeps_explicit_non_default_values() -> None:
    model_cfg = ModelConfig(
        profile="deberta_v3_parity",
        backbone_type="hf_deberta_v2",
        embedding_sharing="none",
        hf_attention_kernel="stable",
    )
    train_cfg = TrainConfig(
        mask_token_prob=0.95,
        random_token_prob=0.03,
        mlm_max_ngram=2,
        disc_loss_weight=12.5,
        token_weighted_gradient_accumulation=False,
    )
    optim_cfg = OptimConfig(adam_epsilon=5e-6, warmup_steps=2_000)

    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg, optim_cfg=optim_cfg)

    assert model_cfg.backbone_type == "hf_deberta_v2"
    assert model_cfg.embedding_sharing == "none"
    assert model_cfg.hf_attention_kernel == "stable"
    assert train_cfg.mask_token_prob == pytest.approx(0.95)
    assert train_cfg.random_token_prob == pytest.approx(0.03)
    assert train_cfg.mlm_max_ngram == 2
    assert train_cfg.disc_loss_weight == pytest.approx(12.5)
    assert optim_cfg.adam.epsilon == pytest.approx(5e-6)
    assert optim_cfg.scheduler.warmup_steps == 2_000
    assert train_cfg.token_weighted_gradient_accumulation is False


def test_apply_profile_defaults_honors_explicit_fields_even_when_matching_raw_defaults() -> None:
    model_cfg = ModelConfig(
        profile="deberta_v3_parity",
        backbone_type="hf_deberta_v2",
        embedding_sharing="es",
        hf_attention_kernel="auto",
    )
    train_cfg = TrainConfig(
        mask_token_prob=0.8,
        random_token_prob=0.1,
        disc_loss_weight=50.0,
        token_weighted_gradient_accumulation=True,
    )
    optim_cfg = OptimConfig(adam_epsilon=1e-8, warmup_steps=1_000)
    object.__setattr__(
        model_cfg,
        "_explicit_fields",
        {"backbone_type", "embedding_sharing", "hf.attention_kernel"},
    )
    object.__setattr__(
        train_cfg,
        "_explicit_fields",
        {
            "objective.mask_token_prob",
            "objective.random_token_prob",
            "objective.disc_loss_weight",
            "token_weighted_gradient_accumulation",
        },
    )
    object.__setattr__(optim_cfg, "_explicit_fields", {"adam.epsilon", "scheduler.warmup_steps"})

    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg, optim_cfg=optim_cfg)

    assert model_cfg.backbone_type == "hf_deberta_v2"
    assert model_cfg.embedding_sharing == "es"
    assert model_cfg.hf_attention_kernel == "auto"
    assert train_cfg.mask_token_prob == pytest.approx(0.8)
    assert train_cfg.random_token_prob == pytest.approx(0.1)
    assert train_cfg.disc_loss_weight == pytest.approx(50.0)
    assert optim_cfg.adam.epsilon == pytest.approx(1e-8)
    assert optim_cfg.scheduler.warmup_steps == 1_000
    assert train_cfg.token_weighted_gradient_accumulation is True


def test_decoupled_training_defaults_true_and_allows_explicit_disable() -> None:
    assert bool(TrainConfig().decoupled_training) is True
    assert bool(TrainConfig(decoupled_training=False).decoupled_training) is False


def test_validate_train_config_rejects_non_boolean_decoupled_training() -> None:
    cfg = TrainConfig(decoupled_training=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="train.decoupled_training must be a boolean"):
        validate_train_config(cfg)


@pytest.mark.parametrize("value", [-1.0, 1.0e-4, 2.5e-3])
def test_validate_train_config_allows_discriminator_learning_rate_inherit_or_positive(value: float) -> None:
    cfg = OptimConfig(discriminator_learning_rate=value)
    validate_optim_config(cfg)


@pytest.mark.parametrize("value", [0.0, -0.5, -2.0])
def test_validate_train_config_rejects_invalid_discriminator_learning_rate(value: float) -> None:
    cfg = OptimConfig(discriminator_learning_rate=value)
    with pytest.raises(ValueError, match="optim.lr.discriminator"):
        validate_optim_config(cfg)


def test_run_pretraining_dry_run_fails_fast_for_nonempty_output_dir(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "marker.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Output directory exists and is not empty"):
        run_pretraining_dry_run(
            model_cfg=ModelConfig(),
            data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
            train_cfg=TrainConfig(output_dir=str(out_dir), overwrite_output_dir=False, max_steps=5),
            config_path=None,
        )


def test_validate_data_config_rejects_non_streaming_shuffle_buffer_above_one():
    with pytest.raises(ValueError, match="shuffle_buffer_size must be 0 or 1"):
        validate_data_config(
            DataConfig(
                dataset_name="HuggingFaceFW/fineweb-edu",
                streaming=False,
                shuffle_buffer_size=10_000,
            )
        )


def test_validate_data_config_rejects_doc_blocking_when_not_packed():
    with pytest.raises(ValueError, match="requires data.packing.enabled=true"):
        validate_data_config(
            DataConfig(
                dataset_name="HuggingFaceFW/fineweb-edu",
                pack_sequences=False,
                block_cross_document_attention=True,
            )
        )


def test_validate_data_config_warns_on_long_context_dense_doc_blocking():
    with pytest.warns(UserWarning, match="dense O\\(S\\^2\\) pairwise masks"):
        validate_data_config(
            DataConfig(
                dataset_name="HuggingFaceFW/fineweb-edu",
                pack_sequences=True,
                block_cross_document_attention=True,
                max_seq_length=4096,
            )
        )


def test_validate_training_workflow_options_rejects_flash_with_packing():
    with pytest.raises(
        ValueError, match="train.sdpa_kernel=flash is not supported with data.packing.enabled=true"
    ):
        validate_training_workflow_options(
            data_cfg=DataConfig(
                dataset_name="HuggingFaceFW/fineweb-edu",
                pack_sequences=True,
                block_cross_document_attention=True,
            ),
            train_cfg=TrainConfig(sdpa_kernel="flash"),
        )


def test_validate_training_workflow_options_allows_flash_when_packed_doc_blocking_disabled():
    validate_training_workflow_options(
        data_cfg=DataConfig(
            dataset_name="HuggingFaceFW/fineweb-edu",
            pack_sequences=True,
            block_cross_document_attention=False,
        ),
        train_cfg=TrainConfig(sdpa_kernel="flash"),
    )


def test_validate_training_workflow_options_rejects_sdpa_kernel_override_when_rope_attention_is_eager():
    with pytest.raises(ValueError, match="train.sdpa_kernel only affects rope attention"):
        validate_training_workflow_options(
            data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu", pack_sequences=False),
            train_cfg=TrainConfig(sdpa_kernel="flash"),
            model_cfg=ModelConfig(backbone_type="rope", attention_implementation="eager"),
        )


def test_validate_training_workflow_options_rejects_hf_backbone_doc_blocking_in_packed_mode():
    with pytest.raises(ValueError, match="only supported with model.backbone_type='rope'"):
        validate_training_workflow_options(
            data_cfg=DataConfig(
                dataset_name="HuggingFaceFW/fineweb-edu",
                pack_sequences=True,
                block_cross_document_attention=True,
            ),
            train_cfg=TrainConfig(),
            model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
        )


def test_validate_training_workflow_options_rejects_es_with_divergent_gen_lr():
    with pytest.raises(ValueError, match="embedding_sharing='es'"):
        validate_training_workflow_options(
            data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
            train_cfg=TrainConfig(),
            model_cfg=ModelConfig(embedding_sharing="es"),
            optim_cfg=OptimConfig(learning_rate=5e-4, generator_learning_rate=3e-4),
        )


def test_validate_training_workflow_options_allows_es_with_matching_gen_lr():
    # Explicit gen LR matching disc LR — should pass.
    validate_training_workflow_options(
        data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
        train_cfg=TrainConfig(decoupled_training=False),
        model_cfg=ModelConfig(embedding_sharing="es"),
        optim_cfg=OptimConfig(learning_rate=5e-4, generator_learning_rate=5e-4),
    )
    # Inherited gen LR (-1) — should pass.
    validate_training_workflow_options(
        data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
        train_cfg=TrainConfig(decoupled_training=False),
        model_cfg=ModelConfig(embedding_sharing="es"),
        optim_cfg=OptimConfig(learning_rate=5e-4, generator_learning_rate=-1.0),
    )


def test_validate_training_workflow_options_allows_gdes_with_divergent_gen_lr():
    # GDES handles separate LR correctly — should not raise.
    validate_training_workflow_options(
        data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
        train_cfg=TrainConfig(),
        model_cfg=ModelConfig(embedding_sharing="gdes"),
        optim_cfg=OptimConfig(learning_rate=5e-4, generator_learning_rate=3e-4),
    )


def test_validate_training_workflow_options_rejects_decoupled_with_es_embedding_sharing():
    with pytest.raises(ValueError, match="incompatible with model.embedding_sharing='es'"):
        validate_training_workflow_options(
            data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
            train_cfg=TrainConfig(decoupled_training=True),
            model_cfg=ModelConfig(backbone_type="hf_deberta_v2", embedding_sharing="es"),
        )


def test_validate_model_config_rejects_rope_knobs_in_hf_mode():
    cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        rope_theta=50_000.0,
    )
    with pytest.raises(ValueError, match="only valid when model.backbone_type='rope'"):
        validate_model_config(cfg)


def test_validate_model_config_normalizes_hf_attention_kernel_alias():
    cfg = ModelConfig(backbone_type="hf_deberta_v2", hf_attention_kernel="cache")
    validate_model_config(cfg)
    assert cfg.hf_attention_kernel == "cached_bmm"

    cfg = ModelConfig(backbone_type="hf_deberta_v2", hf_attention_kernel="safe")
    validate_model_config(cfg)
    assert cfg.hf_attention_kernel == "stable"


def test_validate_model_config_rejects_non_positive_tokenizer_vocab_multiple():
    cfg = ModelConfig(tokenizer_vocab_multiple=0)
    with pytest.raises(ValueError, match="model.tokenizer.vocab_multiple"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_non_positive_tokenizer_vocab_target():
    cfg = ModelConfig(tokenizer_vocab_target=0)
    with pytest.raises(ValueError, match="model.tokenizer.vocab_target"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_derived_generator_knobs_with_explicit_generator_source():
    cfg = ModelConfig(
        backbone_type="rope",
        pretrained_generator_path="microsoft/deberta-v3-small",
        generator_hidden_size=256,
    )
    with pytest.raises(ValueError, match="only used when deriving generator config"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_hf_max_position_embeddings_for_rope():
    cfg = ModelConfig(
        backbone_type="rope",
        hf_max_position_embeddings=1024,
    )
    with pytest.warns(UserWarning, match="model.hf.max_position_embeddings only applies"):
        validate_model_config(cfg)


def test_validate_model_config_allows_hf_max_position_embeddings_in_hf_scratch_mode():
    cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=True,
        hf_max_position_embeddings=1024,
    )
    validate_model_config(cfg)


def test_validate_model_config_rejects_pretrained_derived_generator_shape_overrides():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        pretrained_discriminator_path="local-rope-disc",
        generator_hidden_size=256,
    )
    with pytest.raises(ValueError, match="derived generator weights"):
        validate_model_config(cfg)


def test_validate_model_config_allows_pretrained_derived_generator_layer_override():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        pretrained_discriminator_path="local-rope-disc",
        generator_num_hidden_layers=4,
    )
    validate_model_config(cfg)


def test_looks_like_hf_deberta_checkpoint_avoids_local_rope_false_positive():
    assert _looks_like_hf_deberta_checkpoint("microsoft/deberta-v3-base")
    assert _looks_like_hf_deberta_checkpoint("https://huggingface.co/microsoft/deberta-v3-base")
    assert _looks_like_hf_deberta_checkpoint(
        "/home/user/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/abc"
    )
    assert not _looks_like_hf_deberta_checkpoint("runs/deberta-v3-rope/checkpoint-1000")


def test_validate_model_config_allows_local_rope_checkpoint_path_with_deberta_in_name():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        pretrained_discriminator_path="runs/deberta-v3-rope/checkpoint-1000",
    )
    validate_model_config(cfg)


def test_validate_model_config_rejects_scratch_rope_knobs_in_pretrained_mode():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        pretrained_discriminator_path="local-rope-disc",
        rope_theta=50_000.0,
    )
    with pytest.raises(ValueError, match="only affect scratch RoPE initialization"):
        validate_model_config(cfg)


def test_validate_model_config_allows_explicit_pretrained_rope_overrides():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        pretrained_discriminator_path="local-rope-disc",
        pretrained_rope_theta=50_000.0,
        pretrained_norm_arch="keel",
    )
    validate_model_config(cfg)


def test_validate_model_config_rejects_pretrained_rope_overrides_in_scratch_mode():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        pretrained_rope_theta=50_000.0,
    )
    with pytest.raises(ValueError, match="apply only when model.from_scratch=false"):
        validate_model_config(cfg)


def test_build_backbone_configs_sets_tokenizer_special_ids_for_hf_configs():
    pytest.importorskip("transformers")
    tokenizer = DummyTokenizer(vocab_size=128)

    model_cfg = ModelConfig(backbone_type="hf_deberta_v2", from_scratch=True)
    disc_cfg, gen_cfg = build_backbone_configs(
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

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        pretrained_discriminator_path="local-rope-disc",
        hidden_dropout_prob=None,
        attention_probs_dropout_prob=None,
    )

    disc_cfg, _ = build_backbone_configs(
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


def test_readme_cli_examples_are_parseable():
    parser = cli_mod._build_main_parser()
    examples = [
        "train configs/pretrain_rope_fineweb_edu.yaml",
        "train configs/pretrain_rope_fineweb_edu_2048.yaml",
        "train configs/pretrain_rope_fineweb_edu_4096.yaml",
        "train --preset deberta-v3-base --dry-run",
        (
            "export runs/deberta_rope_rtd/checkpoint-10000 "
            "--what discriminator "
            "--output-dir runs/deberta_rope_rtd/exported_hf"
        ),
    ]

    for cmd in examples:
        parser.parse_args(shlex.split(cmd))


def test_export_help_does_not_show_misleading_defaults_on_no_flags() -> None:
    parser = argparse.ArgumentParser(
        prog="deberta export",
        formatter_class=ExportArgumentDefaultsHelpFormatter,
    )
    add_export_arguments(parser)
    help_text = parser.format_help()
    assert "--safe-serialization" in help_text and "(default: True)" in help_text
    for opt in ("--no-safe-serialization", "--no-offload-to-cpu", "--no-rank0-only"):
        line = next((ln for ln in help_text.splitlines() if opt in ln), "")
        assert "(default:" not in line


def test_docs_use_current_nested_key_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    docs = [
        repo_root / "docs" / "advanced" / "torch-compile.md",
        repo_root / "docs" / "advanced" / "distributed-training.md",
        repo_root / "docs" / "guides" / "data-pipeline.md",
    ]
    merged = "\n".join(path.read_text(encoding="utf-8") for path in docs)

    stale_tokens = (
        "train.torch_compile",
        "train.torch_compile_mode",
        "train.torch_compile_scope",
        "train.torch_compile_backend",
        "train.report_to",
        "train.resume_data_strategy",
        "train.resume_replay_max_micro_batches",
        "data.block_cross_document_attention",
        "train.mlm_max_ngram",
        "train.mask_token_prob",
        "train.random_token_prob",
    )
    for token in stale_tokens:
        assert token not in merged


# ---------------------------------------------------------------------------
# Inert parameter combination warnings (Phase 1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg_kwargs,expect_warn",
    [
        # 1A: rope + non-default hf_attention_kernel → warn
        ({"backbone_type": "rope", "hf_attention_kernel": "stable"}, True),
        # 1A: rope + default hf_attention_kernel → no warn
        ({"backbone_type": "rope", "hf_attention_kernel": "dynamic"}, False),
        # 1B: post + non-default keel_alpha_init → warn
        ({"backbone_type": "rope", "norm_arch": "post", "keel_alpha_init": 5.0}, True),
        # 1B: post + non-default keel_alpha_learnable → warn
        ({"backbone_type": "rope", "norm_arch": "post", "keel_alpha_learnable": True}, True),
        # 1B: keel + keel_alpha_init → no warn
        ({"backbone_type": "rope", "norm_arch": "keel", "keel_alpha_init": 5.0}, False),
        # 1B: post + default keel params → no warn
        ({"backbone_type": "rope", "norm_arch": "post"}, False),
        # 1C: mlp + non-default swiglu_adjust_intermediate → warn
        ({"backbone_type": "rope", "ffn_type": "mlp", "swiglu_adjust_intermediate": False}, True),
        # 1C: swiglu + swiglu_adjust_intermediate → no warn
        ({"backbone_type": "rope", "ffn_type": "swiglu", "swiglu_adjust_intermediate": False}, False),
        # 1C: mlp + default swiglu_adjust_intermediate → no warn
        ({"backbone_type": "rope", "ffn_type": "mlp"}, False),
    ],
    ids=[
        "1A-rope-hf_kernel-warn",
        "1A-rope-hf_kernel-default-nowarn",
        "1B-post-keel_alpha_init-warn",
        "1B-post-keel_alpha_learnable-warn",
        "1B-keel-keel_alpha_init-nowarn",
        "1B-post-default-nowarn",
        "1C-mlp-swiglu_adjust-warn",
        "1C-swiglu-swiglu_adjust-nowarn",
        "1C-mlp-default-nowarn",
    ],
)
def test_validate_model_config_inert_param_warnings(cfg_kwargs, expect_warn):
    cfg = ModelConfig(**cfg_kwargs)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_model_config(cfg)
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    if expect_warn:
        assert len(user_warnings) >= 1, f"Expected warning for {cfg_kwargs}, got none"
    else:
        assert len(user_warnings) == 0, (
            f"Unexpected warning for {cfg_kwargs}: {[str(x.message) for x in user_warnings]}"
        )


@pytest.mark.parametrize(
    "cfg_kwargs,expect_warn",
    [
        # 1D: compile=false + non-auto scope → warn
        ({"torch_compile": False, "torch_compile_scope": "backbones"}, True),
        # 1D: compile=false + non-default mode → warn
        ({"torch_compile": False, "torch_compile_mode": "max-autotune"}, True),
        # 1D: compile=false + non-default backend → warn
        ({"torch_compile": False, "torch_compile_backend": "aot_eager"}, True),
        # 1D: report_to!=wandb + non-default watch mode → warn
        ({"report_to": "tensorboard", "wandb_watch": "all"}, True),
        # 1D: report_to!=wandb + non-default watch freq → warn
        ({"report_to": "none", "wandb_watch_log_freq": 7}, True),
        # 1D: compile=false + auto scope → no warn
        ({"torch_compile": False, "torch_compile_scope": "auto"}, False),
        # 1D: compile=false + default mode/backend/watch knobs → no warn
        ({"torch_compile": False, "report_to": "none"}, False),
        # 1D: wandb backend + watch options active → no inert warning
        ({"report_to": "wandb", "wandb_watch": "all", "wandb_watch_log_freq": 7}, False),
        # 1D: compile=true + non-auto scope → no warn
        ({"torch_compile": True, "torch_compile_scope": "backbones"}, False),
    ],
    ids=[
        "1D-nocompile-scope-warn",
        "1D-nocompile-mode-warn",
        "1D-nocompile-backend-warn",
        "1D-nowandb-watchmode-warn",
        "1D-nowandb-watchfreq-warn",
        "1D-nocompile-auto-nowarn",
        "1D-nocompile-defaults-nowarn",
        "1D-wandb-watch-active-nowarn",
        "1D-compile-scope-nowarn",
    ],
)
def test_validate_train_config_inert_param_warnings(cfg_kwargs, expect_warn):
    train_keys = {
        "torch_compile",
        "torch_compile_scope",
        "torch_compile_mode",
        "torch_compile_backend",
    }
    logging_keys = {"report_to", "wandb_watch", "wandb_watch_log_freq"}

    train_kwargs = {k: v for k, v in cfg_kwargs.items() if k in train_keys}
    logging_kwargs = {k: v for k, v in cfg_kwargs.items() if k in logging_keys}

    train_cfg = TrainConfig(**train_kwargs)
    logging_cfg = LoggingConfig(**logging_kwargs) if logging_kwargs else None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_train_config(train_cfg)
        if logging_cfg is not None:
            validate_logging_config(logging_cfg)
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    if expect_warn:
        assert len(user_warnings) >= 1, f"Expected warning for {cfg_kwargs}, got none"
    else:
        assert len(user_warnings) == 0, (
            f"Unexpected warning for {cfg_kwargs}: {[str(x.message) for x in user_warnings]}"
        )


@pytest.mark.parametrize(
    "model_kwargs,train_kwargs,expect_warn",
    [
        # 1E: hf_deberta_v2 + non-auto sdpa_kernel → warn
        ({"backbone_type": "hf_deberta_v2"}, {"sdpa_kernel": "mem_efficient"}, True),
        # 1E: hf_deberta_v2 + auto sdpa_kernel → no warn
        ({"backbone_type": "hf_deberta_v2"}, {"sdpa_kernel": "auto"}, False),
        # 1E: rope + non-auto sdpa_kernel → no warn (rope SDPA is functional)
        ({"backbone_type": "rope"}, {"sdpa_kernel": "mem_efficient"}, False),
    ],
    ids=[
        "1E-hfv2-sdpa-warn",
        "1E-hfv2-auto-nowarn",
        "1E-rope-sdpa-nowarn",
    ],
)
def test_validate_workflow_sdpa_kernel_inert_warning(model_kwargs, train_kwargs, expect_warn):
    model_cfg = ModelConfig(**model_kwargs)
    train_cfg = TrainConfig(**train_kwargs)
    data_cfg = DataConfig(data_files="dummy.txt")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_training_workflow_options(data_cfg=data_cfg, train_cfg=train_cfg, model_cfg=model_cfg)
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    if expect_warn:
        assert len(user_warnings) >= 1, (
            f"Expected warning for model={model_kwargs} train={train_kwargs}, got none"
        )
    else:
        assert len(user_warnings) == 0, (
            f"Unexpected warning for model={model_kwargs} train={train_kwargs}: "
            f"{[str(x.message) for x in user_warnings]}"
        )


# ---------------------------------------------------------------------------
# Compile scope metadata persistence (Phase 3A)
# ---------------------------------------------------------------------------


def test_build_run_metadata_includes_compile_scope():
    meta = _build_run_metadata(effective_compile_scope="backbones", compile_scope_reason="auto test")
    assert meta["effective_compile_scope"] == "backbones"
    assert meta["compile_scope_reason"] == "auto test"
    assert "config_schema_version" in meta


def test_build_run_metadata_omits_compile_scope_when_none():
    meta = _build_run_metadata()
    assert "effective_compile_scope" not in meta
    assert "compile_scope_reason" not in meta


def test_persist_or_validate_run_configs_preflight_mode_writes_no_snapshots(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=ModelConfig(),
        data_cfg=DataConfig(data_files="dummy.txt"),
        train_cfg=TrainConfig(),
        resume_checkpoint=None,
        is_main_process=True,
        preflight_only=True,
    )
    for filename in ("model_config.json", "data_config.json", "train_config.json", "run_metadata.json"):
        assert not (out / filename).exists()


def test_persist_run_configs_writes_compile_scope_to_metadata(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    model_cfg = ModelConfig()
    data_cfg = DataConfig(data_files="dummy.txt")
    train_cfg = TrainConfig()
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        is_main_process=True,
        effective_compile_scope="ffn",
        compile_scope_reason="auto scope selected FFN-only",
    )
    from deberta.io_utils import load_json_mapping

    meta = load_json_mapping(out / "run_metadata.json")
    assert meta["effective_compile_scope"] == "ffn"
    assert meta["compile_scope_reason"] == "auto scope selected FFN-only"


def test_persist_run_configs_warns_on_compile_scope_drift_on_resume(tmp_path: Path, caplog):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    model_cfg = ModelConfig()
    data_cfg = DataConfig(data_files="dummy.txt")
    train_cfg = TrainConfig()

    # Initial run persists scope=backbones.
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        is_main_process=True,
        effective_compile_scope="backbones",
    )

    # Simulate a checkpoint directory for resume.
    ckpt = out / "checkpoint-100"
    ckpt.mkdir()

    # Resume with different scope should log a warning.
    with caplog.at_level(logging.WARNING):
        _persist_or_validate_run_configs(
            output_dir=out,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            resume_checkpoint=str(ckpt),
            is_main_process=True,
            effective_compile_scope="ffn",
        )
    assert any("compile scope changed on resume" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# _global_grad_l2_norm
# ---------------------------------------------------------------------------


def test_global_grad_l2_norm_finite_model():
    model = torch.nn.Linear(4, 4, bias=False)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    norm = _global_grad_l2_norm(model)
    assert norm > 0.0
    assert torch.isfinite(torch.tensor(norm))
    # Cross-check against torch.nn.utils.clip_grad_norm_ (which returns the norm).
    ref = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
    assert abs(norm - float(ref)) < 1e-4


def test_global_grad_l2_norm_zero_grad():
    model = torch.nn.Linear(4, 4, bias=True)
    model.zero_grad()
    # Gradients are None after zero_grad with set_to_none=True (default).
    assert _global_grad_l2_norm(model) == 0.0
    # Explicitly set grads to zero tensors.
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    assert _global_grad_l2_norm(model) == 0.0


def test_global_grad_l2_norm_no_parameters():
    model = torch.nn.Module()
    assert _global_grad_l2_norm(model) == 0.0
