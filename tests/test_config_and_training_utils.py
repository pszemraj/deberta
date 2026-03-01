from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import shlex
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from _fakes import DummyTokenizer, FakeAccelerator, SimpleRTD, setup_pretraining_mocks

import deberta.cli as cli_mod
from deberta.checkpoint_utils import (
    canonical_compile_state_key,
    load_model_state_with_compile_key_remap,
)
from deberta.cli import _load_json, _load_yaml
from deberta.config import (
    RUN_CONFIG_SCHEMA_VERSION,
    DataConfig,
    ModelConfig,
    TrainConfig,
    _normalize_hf_attention_kernel,
    _normalize_sdpa_kernel,
    _normalize_torch_compile_backend,
    _normalize_torch_compile_mode,
    _normalize_torch_compile_scope,
    _normalize_wandb_watch,
    load_data_config_snapshot,
    load_model_config_snapshot,
    normalize_mixed_precision,
    validate_data_config,
    validate_model_config,
    validate_train_config,
    validate_training_workflow_options,
)
from deberta.export_cli import add_export_arguments
from deberta.modeling.builder import build_backbone_configs
from deberta.modeling.rtd import attention_mask_to_active_tokens, compute_generator_loss_term
from deberta.training.pretrain import (
    _append_metrics_jsonl_row,
    _apply_lr_mult,
    _apply_nonfinite_recovery,
    _build_optimizer,
    _build_run_metadata,
    _build_training_collator,
    _count_input_tokens_for_batch,
    _count_rtd_tokens_for_batch,
    _cycle_dataloader,
    _export_discriminator_hf,
    _finalize_window_metric_loss,
    _find_latest_checkpoint,
    _flush_loggers,
    _full_backbone_hf_inductor_warning,
    _global_grad_l2_norm,
    _has_nonfinite_grad_norm_any_rank,
    _init_trackers,
    _load_checkpoint_data_progress,
    _load_resume_state_with_compile_fallback,
    _persist_or_validate_run_configs,
    _prepare_output_dir,
    _resolve_compile_enabled_or_raise,
    _resolve_compile_scope,
    _resolve_data_resume_policy,
    _resolve_effective_mixed_precision_or_raise,
    _resolve_output_dir,
    _resolve_output_dir_for_accelerator,
    _resolve_resume_checkpoint,
    _resolve_window_token_denominators,
    _sanitize_nonfinite_gradients_,
    _save_checkpoint_data_progress,
    _save_training_checkpoint,
    _scale_loss_for_backward,
    _setup_wandb_watch,
    _should_clip_gradients,
    _should_force_legacy_tf32_for_compile,
    _stabilize_compile_attention_mask,
    _sync_discriminator_embeddings_if_available,
    _token_weighted_micro_objective,
    _upload_wandb_original_config,
    _write_export_readme,
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
    assert normalize_mixed_precision(train_flat.mixed_precision) == "no"


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


def test_resolve_output_dir_for_accelerator_keeps_explicit_path():
    class _FakeAccelerator:
        is_main_process = False
        num_processes = 8

    called = {"count": 0}

    def _fake_broadcast(payload: list[str | None], *, from_process: int = 0) -> None:
        del payload, from_process
        called["count"] += 1

    out = _resolve_output_dir_for_accelerator(
        accelerator=_FakeAccelerator(),
        output_dir="runs/custom/run-02",
        project_name="ignored",
        config_path="cfg.yaml",
        broadcast_fn=_fake_broadcast,
    )
    assert out == Path("runs/custom/run-02")
    assert called["count"] == 0


def test_resolve_output_dir_for_accelerator_uses_broadcasted_auto_value():
    class _FakeAccelerator:
        is_main_process = False
        num_processes = 2

    def _fake_broadcast(payload: list[str | None], *, from_process: int = 0) -> None:
        assert from_process == 0
        payload[0] = "runs/demo/20260101_010101_shared"

    out = _resolve_output_dir_for_accelerator(
        accelerator=_FakeAccelerator(),
        output_dir=None,
        project_name="demo",
        config_path="cfg.yaml",
        broadcast_fn=_fake_broadcast,
    )
    assert out == Path("runs/demo/20260101_010101_shared")


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


def test_resolve_resume_checkpoint_rejects_missing_explicit_path(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    missing = out / "checkpoint-missing"
    with pytest.raises(FileNotFoundError, match="checkpoint path does not exist"):
        _resolve_resume_checkpoint(
            output_dir=out,
            resume_from_checkpoint=str(missing),
            is_main_process=True,
        )


def test_resolve_resume_checkpoint_rejects_non_directory_explicit_path(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "not-a-dir.txt"
    marker.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="must point to a checkpoint directory"):
        _resolve_resume_checkpoint(
            output_dir=out,
            resume_from_checkpoint=str(marker),
            is_main_process=True,
        )


def test_resolve_resume_checkpoint_returns_resolved_explicit_path(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint-2"
    ckpt.mkdir()
    resolved = _resolve_resume_checkpoint(
        output_dir=out,
        resume_from_checkpoint=str(ckpt),
        is_main_process=True,
    )
    assert resolved == str(ckpt.resolve())


def test_checkpoint_data_progress_roundtrip(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir(parents=True, exist_ok=True)

    consumed, lr_mult = _load_checkpoint_data_progress(ckpt)
    assert consumed is None
    assert lr_mult == 1.0

    _save_checkpoint_data_progress(checkpoint_dir=ckpt, consumed_micro_batches=123, lr_mult=0.25)
    consumed, lr_mult = _load_checkpoint_data_progress(ckpt)
    assert consumed == 123
    assert abs(lr_mult - 0.25) < 1e-9

    # Back-compat: old checkpoints without lr_mult default to 1.0.
    import json

    (ckpt / "data_state.json").write_text(json.dumps({"consumed_micro_batches": 50}))
    consumed_old, lr_mult_old = _load_checkpoint_data_progress(ckpt)
    assert consumed_old == 50
    assert lr_mult_old == 1.0


def test_canonical_compile_state_key_strips_orig_mod_segments() -> None:
    assert canonical_compile_state_key("generator._orig_mod.encoder.layer.0._orig_mod.weight") == (
        "generator.encoder.layer.0.weight"
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

    class _ResumeAccelerator:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def load_state(self, ckpt: str, **kwargs: Any) -> None:
            self.calls.append((ckpt, dict(kwargs)))
            if kwargs.get("strict", True):
                raise RuntimeError("Error(s) in loading state_dict with _orig_mod mismatch")

        def unwrap_model(self, wrapped: torch.nn.Module) -> torch.nn.Module:
            return wrapped

    accel = _ResumeAccelerator()
    _load_resume_state_with_compile_fallback(accel, model, str(checkpoint))

    assert len(accel.calls) == 2
    assert accel.calls[0][1] == {}
    assert accel.calls[1][1] == {"strict": False}
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
    class _FakeRun:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.nn.Module, dict[str, Any]]] = []

        def watch(self, model: torch.nn.Module, **kwargs: Any) -> None:
            self.calls.append((model, dict(kwargs)))

    class _FakeAccelerator:
        is_main_process = True

        @staticmethod
        def unwrap_model(model: torch.nn.Module):
            return model

        @staticmethod
        def get_tracker(name: str, unwrap: bool = True):
            del name, unwrap
            return None

    model = torch.nn.Linear(4, 4)
    run = _FakeRun()
    enabled = _setup_wandb_watch(
        accelerator=_FakeAccelerator(),
        wandb_run=run,
        model=model,
        watch_mode="gradients",
        watch_log_freq=123,
    )
    assert enabled is True
    assert run.calls
    watched_model, kwargs = run.calls[0]
    assert watched_model is model
    assert kwargs["log"] == "gradients"
    assert kwargs["log_freq"] == 123


def test_setup_wandb_watch_returns_false_when_mode_none() -> None:
    class _FakeRun:
        def watch(self, model: torch.nn.Module, **kwargs: Any) -> None:  # pragma: no cover
            del model, kwargs
            raise AssertionError("watch must not be called when mode=none")

    class _FakeAccelerator:
        is_main_process = True

        @staticmethod
        def unwrap_model(model: torch.nn.Module):
            return model

    enabled = _setup_wandb_watch(
        accelerator=_FakeAccelerator(),
        wandb_run=_FakeRun(),
        model=torch.nn.Linear(2, 2),
        watch_mode="none",
        watch_log_freq=100,
    )
    assert enabled is False


def test_upload_wandb_original_config_stages_with_expected_filename(tmp_path: Path) -> None:
    src = tmp_path / "config_original.yaml"
    src.write_text("train:\n  max_steps: 1\n", encoding="utf-8")

    class _FakeRun:
        def __init__(self) -> None:
            self.saved: list[Path] = []

        def save(self, path: str, **kwargs: Any) -> None:
            del kwargs
            self.saved.append(Path(path))

    class _FakeAccelerator:
        is_main_process = True

    run = _FakeRun()
    uploaded = _upload_wandb_original_config(
        accelerator=_FakeAccelerator(),
        wandb_run=run,
        config_original_path=src,
        run_name="demo-run",
    )
    assert uploaded is True
    assert run.saved
    assert run.saved[0].name == "config_deberta_demo-run"


def test_run_pretraining_keyboard_interrupt_logs_crash_and_finishes_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes import _PRETRAINING_BATCH

    class _TrackingAccelerator(FakeAccelerator):
        last_instance: _TrackingAccelerator | None = None

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            _TrackingAccelerator.last_instance = self

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
        accelerator_cls=_TrackingAccelerator,
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

    accel = _TrackingAccelerator.last_instance
    assert accel is not None
    assert accel.wandb_run.summary["crashed"] is True
    assert accel.wandb_run.summary["crash_type"] == "KeyboardInterrupt"
    assert accel.wandb_run.finished_exit_code == 130
    assert accel.wandb_run.logged
    assert accel.wandb_run.watch_calls
    assert accel.wandb_run.saved_paths
    assert accel.wandb_run.saved_paths[0].name == "config_deberta_run"
    assert accel.wandb_run.watch_calls[0][1]["log"] == "gradients"
    assert accel.tracker_init_calls
    first_tracker_call = accel.tracker_init_calls[0]
    assert first_tracker_call["project_name"] == "deberta-train"
    assert first_tracker_call["init_kwargs"]["wandb"]["name"] == "run"
    assert accel.ended is False


def test_run_pretraining_resume_at_max_steps_skips_data_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _ResumeAccelerator(FakeAccelerator):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.loaded: list[tuple[str, dict[str, Any]]] = []

        def load_state(self, ckpt: str, **kwargs: Any) -> None:
            self.loaded.append((ckpt, dict(kwargs)))

    checkpoint_dir = tmp_path / "run" / "checkpoint-2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "data_state.json").write_text(
        json.dumps({"consumed_micro_batches": 50}),
        encoding="utf-8",
    )

    replay_calls = {"next": 0}

    def _fail_cycle(_loader, *, start_epoch: int = 0):
        del start_epoch
        while True:
            replay_calls["next"] += 1
            raise AssertionError("resume replay should be skipped when global_step >= max_steps")
            yield {}  # pragma: no cover

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=_ResumeAccelerator,
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


def test_run_pretraining_logs_window_averaged_rtd_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _TrackingAccelerator(FakeAccelerator):
        last_instance: _TrackingAccelerator | None = None

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            _TrackingAccelerator.last_instance = self

    class _WindowMetricRTD(SimpleRTD):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._call_idx = 0

        def forward(self, **kwargs: Any) -> Any:
            del kwargs
            self._call_idx += 1
            t = self.weight * 0.0 + 1.0
            if self._call_idx % 2 == 1:
                gen_loss, disc_loss, disc_acc = 1.0, 10.0, 0.2
                gen_tokens, disc_tokens, disc_pos = 1.0, 10.0, 7.0
            else:
                gen_loss, disc_loss, disc_acc = 9.0, 2.0, 0.8
                gen_tokens, disc_tokens, disc_pos = 9.0, 2.0, 1.0
            return types.SimpleNamespace(
                loss=t,
                gen_loss=torch.tensor(gen_loss, dtype=torch.float32),
                disc_loss=torch.tensor(disc_loss, dtype=torch.float32),
                disc_accuracy=torch.tensor(disc_acc, dtype=torch.float32),
                gen_token_count=torch.tensor(gen_tokens, dtype=torch.float32),
                disc_token_count=torch.tensor(disc_tokens, dtype=torch.float32),
                disc_positive_count=torch.tensor(disc_pos, dtype=torch.float32),
                gen_loss_raw=t,
                disc_loss_raw=t,
            )

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=_TrackingAccelerator,
        rtd_cls=_WindowMetricRTD,
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
        torch_compile=False,
        export_hf_final=False,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    accel = _TrackingAccelerator.last_instance
    assert accel is not None
    step_rows = [row for row, step in accel.logged_rows if int(step or -1) == 1]
    assert step_rows
    metrics = step_rows[-1]
    assert metrics["gen_loss"] == pytest.approx(8.2, rel=0.0, abs=1e-6)
    assert metrics["disc_loss"] == pytest.approx(104.0 / 12.0, rel=0.0, abs=1e-6)
    assert metrics["disc_acc"] == pytest.approx(0.3, rel=0.0, abs=1e-6)
    assert metrics["gen_token_count"] == pytest.approx(10.0, rel=0.0, abs=1e-6)
    assert metrics["disc_token_count"] == pytest.approx(12.0, rel=0.0, abs=1e-6)
    assert metrics["disc_pos_frac"] == pytest.approx(8.0 / 12.0, rel=0.0, abs=1e-6)


def test_run_pretraining_warns_and_tracks_zero_token_weighted_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _TrackingAccelerator(FakeAccelerator):
        last_instance: _TrackingAccelerator | None = None

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            _TrackingAccelerator.last_instance = self

    class _ScalarLossRTD(SimpleRTD):
        def forward(self, **kwargs: Any) -> Any:
            del kwargs
            t = self.weight.sum() * 0.0 + 1.0
            return types.SimpleNamespace(
                loss=t,
                gen_loss=t.detach(),
                disc_loss=t.detach(),
                disc_accuracy=t.detach(),
                gen_token_count=torch.tensor(1.0),
                disc_token_count=torch.tensor(1.0),
                disc_positive_count=torch.tensor(1.0),
                gen_loss_raw=t,
                disc_loss_raw=t,
            )

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=_TrackingAccelerator,
        rtd_cls=_ScalarLossRTD,
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
        torch_compile=False,
        export_hf_final=False,
    )

    with caplog.at_level(logging.WARNING):
        pretrain_mod.run_pretraining(
            model_cfg=ModelConfig(),
            data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
            train_cfg=train_cfg,
        )

    assert any("Token-weighted GA window has zero effective tokens" in rec.message for rec in caplog.records)

    accel = _TrackingAccelerator.last_instance
    assert accel is not None
    step_rows = [row for row, step in accel.logged_rows if int(step or -1) == 1]
    assert step_rows
    metrics = step_rows[-1]
    assert metrics["zero_gen_window_total"] == pytest.approx(1.0, rel=0.0, abs=1e-6)
    assert metrics["zero_disc_window_total"] == pytest.approx(1.0, rel=0.0, abs=1e-6)
    assert metrics["zero_gen_window_since_log"] == pytest.approx(1.0, rel=0.0, abs=1e-6)
    assert metrics["zero_disc_window_since_log"] == pytest.approx(1.0, rel=0.0, abs=1e-6)


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
        model_cfg=ModelConfig(),
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


def test_run_pretraining_hf_deberta_auto_scope_compiles_ffn(
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

    class _FFNScopeRTD(SimpleRTD):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.generator = _FakeBackbone()
            self.discriminator = _FakeBackbone()

    compile_calls: list[tuple[Any, dict[str, Any]]] = []

    def _fake_compile(
        target: Any, *, mode: str = "default", backend: str = "inductor", dynamic: bool | None = None
    ) -> Any:
        compile_calls.append((target, {"mode": str(mode), "backend": str(backend), "dynamic": dynamic}))
        return target

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        rtd_cls=_FFNScopeRTD,
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

    instance = _FFNScopeRTD.last_instance
    assert instance is not None
    assert len(compile_calls) == 4
    for i in range(4):
        assert callable(compile_calls[i][0])
        assert compile_calls[i][1] == {"mode": "default", "backend": "inductor", "dynamic": False}
    assert getattr(compile_calls[0][0], "__self__", None) is instance.generator.encoder.layer[0].intermediate
    assert getattr(compile_calls[1][0], "__self__", None) is instance.generator.encoder.layer[0].output
    disc_layer = instance.discriminator.encoder.layer[0]
    assert getattr(compile_calls[2][0], "__self__", None) is disc_layer.intermediate
    assert getattr(compile_calls[3][0], "__self__", None) is disc_layer.output


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


def test_sanitize_nonfinite_gradients_replaces_nan_inf_values() -> None:
    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

    model = _Tiny()
    model.w.grad = torch.tensor([float("nan"), float("inf"), -float("inf")], dtype=torch.float32)
    replaced = _sanitize_nonfinite_gradients_(model)
    assert int(replaced) == 3
    assert model.w.grad is not None
    assert torch.isfinite(model.w.grad).all()
    assert torch.equal(model.w.grad, torch.zeros_like(model.w.grad))


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
    consumed, lr_mult = _load_checkpoint_data_progress(ckpt)
    assert consumed == 42
    assert lr_mult == 1.0


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


def test_export_discriminator_hf_uses_unwrapped_submodules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import deberta.training.pretrain as pretrain_mod

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

    class _FakeAccelerator:
        is_main_process = True

        def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
            assert model is wrapped
            return inner

        def get_state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
            called_targets.append(model)
            return {}

    class _FakeExportModel:
        def save_pretrained(self, path: str, safe_serialization: bool = True) -> None:
            del safe_serialization
            Path(path).mkdir(parents=True, exist_ok=True)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModel = types.SimpleNamespace(from_config=lambda _cfg: _FakeExportModel())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        pretrain_mod,
        "load_intersection_state_dict",
        lambda _model, _state: types.SimpleNamespace(missing_keys=[]),
    )
    monkeypatch.setattr(
        pretrain_mod,
        "merge_embeddings_into_export_backbone",
        lambda **_kwargs: None,
    )

    _export_discriminator_hf(
        accelerator=_FakeAccelerator(),
        model=wrapped,  # type: ignore[arg-type]
        tokenizer=DummyTokenizer(),
        output_dir=tmp_path / "export",
        embedding_sharing="none",
    )

    assert called_targets == [inner.discriminator, inner.generator]


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


def test_build_optimizer_supports_generator_specific_lr():
    model = _TinyRTDLikeModel()
    cfg = TrainConfig(learning_rate=1.0e-3, generator_learning_rate=5.0e-4, weight_decay=0.1)
    opt = _build_optimizer(model, cfg)

    lrs = {float(g["lr"]) for g in opt.param_groups}
    assert lrs == {1.0e-3, 5.0e-4}

    # We should have both decay and no-decay groups present.
    wds = {float(g["weight_decay"]) for g in opt.param_groups}
    assert wds == {0.0, 0.1}


def test_build_optimizer_keeps_fused_for_hf_backbones_compile_bf16(monkeypatch: pytest.MonkeyPatch):
    import deberta.training.pretrain as pretrain_mod

    model = _TinyRTDLikeModel()
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

    model = _TinyRTDLikeModel()
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
    model = _TinyRTDLikeModel()
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
    assert model.hidden_dropout_prob is None
    assert model.attention_probs_dropout_prob is None
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
    cfg = TrainConfig(report_to="wandb", wandb_watch="weights", wandb_watch_log_freq=25)
    validate_train_config(cfg)
    assert cfg.wandb_watch == "parameters"
    assert cfg.wandb_watch_log_freq == 25


def test_validate_train_config_rejects_invalid_wandb_watch_mode():
    cfg = TrainConfig(wandb_watch="histogram")
    with pytest.raises(ValueError, match="train.wandb_watch must be one of"):
        validate_train_config(cfg)


def test_validate_train_config_rejects_non_positive_wandb_watch_log_freq():
    cfg = TrainConfig(wandb_watch_log_freq=0)
    with pytest.raises(ValueError, match="train.wandb_watch_log_freq must be >= 1"):
        validate_train_config(cfg)


def test_validate_train_config_accepts_resume_data_strategy_values():
    cfg = TrainConfig(resume_data_strategy="restart_epoch", resume_replay_max_micro_batches=123)
    validate_train_config(cfg)
    assert cfg.resume_data_strategy == "restart_epoch"
    assert cfg.resume_replay_max_micro_batches == 123


def test_validate_train_config_rejects_invalid_resume_data_strategy():
    cfg = TrainConfig(resume_data_strategy="fast")
    with pytest.raises(ValueError, match="train.resume_data_strategy must be one of"):
        validate_train_config(cfg)


def test_validate_train_config_rejects_negative_resume_replay_threshold():
    cfg = TrainConfig(resume_replay_max_micro_batches=-1)
    with pytest.raises(ValueError, match="train.resume_replay_max_micro_batches must be >= 0"):
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
        decoupled_loss_scaling=False,
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
        decoupled_loss_scaling=False,
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


def test_compute_generator_loss_term_matches_decoupled_formula():
    gen_loss = torch.tensor(2.0)
    disc_loss = torch.tensor(10.0)

    term = compute_generator_loss_term(
        gen_loss=gen_loss,
        disc_loss=disc_loss,
        decoupled_loss_scaling=True,
    )
    expected = (disc_loss / gen_loss) * gen_loss
    torch.testing.assert_close(term, expected)


def test_compute_generator_loss_term_passthrough_when_disabled():
    gen_loss = torch.tensor(1.5)
    disc_loss = torch.tensor(7.0)
    term = compute_generator_loss_term(
        gen_loss=gen_loss,
        disc_loss=disc_loss,
        decoupled_loss_scaling=False,
    )
    torch.testing.assert_close(term, gen_loss)


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
    class _Tokenizer:
        mask_token_id = 3
        pad_token_id = 0
        vocab_size = 64
        all_special_ids = [0, 1, 2, 3]

        def tokenize(self, text: str) -> list[str]:
            return text.split()

    train_cfg = TrainConfig(mlm_probability=0.2, mlm_max_ngram=2)
    collator = _build_training_collator(
        tokenizer=_Tokenizer(),
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
    class _FakeAccelerator:
        device = torch.device("cpu")

        def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
            assert reduction == "sum"
            # Simulate another rank reporting a non-finite norm.
            return tensor + 1

    accel = _FakeAccelerator()
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


def test_count_rtd_tokens_for_batch_keeps_masked_positions_active_for_discriminator():
    batch = {
        "input_ids": torch.tensor([[1, 3, 11, 2, 0]], dtype=torch.long),
        "labels": torch.tensor([[-100, 99, -100, -100, -100]], dtype=torch.long),
    }
    special_token_mask = torch.zeros(16, dtype=torch.bool)
    special_token_mask[torch.tensor([0, 1, 2, 3])] = True
    gen_count, disc_count = _count_rtd_tokens_for_batch(
        batch,
        special_token_mask=special_token_mask,
        pad_token_id=0,
    )
    assert gen_count == pytest.approx(1.0)
    assert disc_count == pytest.approx(2.0)


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
    from deberta.training.pretrain import _compute_disc_active_mask

    special_token_mask = torch.zeros(16, dtype=torch.bool)
    special_token_mask[torch.tensor([0, 1, 2])] = True
    mask = _compute_disc_active_mask(
        input_ids=torch.tensor([[1, 11, 2, 13, 0]], dtype=torch.long),
        labels=torch.tensor([[-100, 99, -100, 77, -100]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
        special_token_mask=special_token_mask,
        pad_token_id=0,
    )

    expected = torch.tensor([[False, True, False, True, False]], dtype=torch.bool)
    assert torch.equal(mask, expected)


def test_attention_mask_to_active_tokens_uses_pad_contract_for_3d_masks():
    input_ids = torch.tensor([[11, 12, 0]], dtype=torch.long)
    # Deliberately make the pad row look active in the pairwise mask to ensure
    # active-token recovery does not depend on O(S^2) reductions over mask rows.
    pair_keep = torch.tensor(
        [
            [
                [1, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
            ]
        ],
        dtype=torch.bool,
    )
    active = attention_mask_to_active_tokens(
        input_ids=input_ids,
        attention_mask=pair_keep,
        pad_token_id=0,
    )
    expected = torch.tensor([[True, True, False]], dtype=torch.bool)
    assert torch.equal(active, expected)


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
            "torch_compile_mode",
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
            "torch_compile_scope",
        ),
        (
            _normalize_torch_compile_backend,
            [("inductor", "inductor"), ("aot-eager", "aot_eager")],
            "xla",
            "torch_compile_backend",
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
            "wandb_watch",
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
            "hf_attention_kernel",
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
            "sdpa_kernel",
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


def test_resolve_compile_scope_uses_hf_deberta_v2_default_inductor_fallback():
    scope, reason = _resolve_compile_scope(
        requested_scope="auto",
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
        compile_mode="default",
        compile_backend="inductor",
    )
    assert scope == "ffn"
    assert reason is not None

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


def test_full_backbone_hf_inductor_warning_for_unstable_combo():
    msg = _full_backbone_hf_inductor_warning(
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
        compile_enabled=True,
        compile_scope="backbones",
        compile_backend="inductor",
    )
    assert msg is not None
    assert "ffn" in msg

    assert (
        _full_backbone_hf_inductor_warning(
            model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
            compile_enabled=True,
            compile_scope="ffn",
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
                "  dataset_name: HuggingFaceFW/fineweb-edu",
                "  max_seq_length: 32",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(["train", str(cfg_path), "--max_steps", "7"])

    assert "train_cfg" in seen
    assert seen["data_cfg"].max_seq_length == 32
    assert seen["train_cfg"].max_steps == 7
    assert seen["config_path"] == cfg_path


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
    with pytest.raises(ValueError, match="requires data.pack_sequences=true"):
        validate_data_config(
            DataConfig(
                dataset_name="HuggingFaceFW/fineweb-edu",
                pack_sequences=False,
                block_cross_document_attention=True,
            )
        )


def test_validate_training_workflow_options_rejects_flash_with_packing():
    with pytest.raises(ValueError, match="sdpa_kernel=flash is not supported with data.pack_sequences=true"):
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
            train_cfg=TrainConfig(learning_rate=5e-4, generator_learning_rate=3e-4),
            model_cfg=ModelConfig(embedding_sharing="es"),
        )


def test_validate_training_workflow_options_allows_es_with_matching_gen_lr():
    # Explicit gen LR matching disc LR — should pass.
    validate_training_workflow_options(
        data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
        train_cfg=TrainConfig(learning_rate=5e-4, generator_learning_rate=5e-4),
        model_cfg=ModelConfig(embedding_sharing="es"),
    )
    # Inherited gen LR (-1) — should pass.
    validate_training_workflow_options(
        data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
        train_cfg=TrainConfig(learning_rate=5e-4, generator_learning_rate=-1.0),
        model_cfg=ModelConfig(embedding_sharing="es"),
    )


def test_validate_training_workflow_options_allows_gdes_with_divergent_gen_lr():
    # GDES handles separate LR correctly — should not raise.
    validate_training_workflow_options(
        data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
        train_cfg=TrainConfig(learning_rate=5e-4, generator_learning_rate=3e-4),
        model_cfg=ModelConfig(embedding_sharing="gdes"),
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
    with pytest.raises(ValueError, match="tokenizer_vocab_multiple"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_non_positive_tokenizer_vocab_target():
    cfg = ModelConfig(tokenizer_vocab_target=0)
    with pytest.raises(ValueError, match="tokenizer_vocab_target"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_derived_generator_knobs_with_explicit_generator_source():
    cfg = ModelConfig(
        backbone_type="rope",
        generator_model_name_or_path="microsoft/deberta-v3-small",
        generator_hidden_size=256,
    )
    with pytest.raises(ValueError, match="only used when deriving generator config"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_hf_max_position_embeddings_for_rope():
    cfg = ModelConfig(
        backbone_type="rope",
        hf_max_position_embeddings=1024,
    )
    with pytest.warns(UserWarning, match="hf_max_position_embeddings only applies"):
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
        discriminator_model_name_or_path="local-rope-disc",
        generator_hidden_size=256,
    )
    with pytest.raises(ValueError, match="derived generator weights"):
        validate_model_config(cfg)


def test_validate_model_config_allows_pretrained_derived_generator_layer_override():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="local-rope-disc",
        generator_num_hidden_layers=4,
    )
    validate_model_config(cfg)


def test_validate_model_config_rejects_scratch_rope_knobs_in_pretrained_mode():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="local-rope-disc",
        rope_theta=50_000.0,
    )
    with pytest.raises(ValueError, match="only affect scratch RoPE initialization"):
        validate_model_config(cfg)


def test_validate_model_config_allows_explicit_pretrained_rope_overrides():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="local-rope-disc",
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


def test_build_backbone_configs_sets_tokenizer_special_ids_for_hf_configs(
    monkeypatch: pytest.MonkeyPatch,
):
    class _FakeHFConfig:
        def __init__(self) -> None:
            self.vocab_size = 64
            self.hidden_size = 32
            self.num_hidden_layers = 2
            self.num_attention_heads = 4
            self.intermediate_size = 64
            self.hidden_act = "gelu"

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = types.SimpleNamespace(from_pretrained=lambda _src: _FakeHFConfig())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    class _Tokenizer:
        pad_token_id = 0
        cls_token_id = 1
        sep_token_id = 2
        mask_token_id = 3
        bos_token_id = 4
        eos_token_id = 5

        def __len__(self) -> int:
            return 128

    model_cfg = ModelConfig(backbone_type="hf_deberta_v2", from_scratch=True)
    disc_cfg, gen_cfg = build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_Tokenizer(),
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

    class _Tokenizer:
        pad_token_id = 0
        cls_token_id = 1
        sep_token_id = 2
        mask_token_id = 3

        def __len__(self) -> int:
            return 32000

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="local-rope-disc",
    )

    disc_cfg, _ = build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_Tokenizer(),
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
        (
            "export runs/deberta_rope_rtd/checkpoint-10000 "
            "--what discriminator "
            "--output-dir runs/deberta_rope_rtd/exported_hf"
        ),
    ]

    for cmd in examples:
        parser.parse_args(shlex.split(cmd))


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
    cfg = TrainConfig(**cfg_kwargs)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_train_config(cfg)
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
