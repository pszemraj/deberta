from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import shlex
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
import torch

import deberta.cli as cli_mod
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
    _build_optimizer,
    _build_training_collator,
    _count_input_tokens_for_batch,
    _count_rtd_tokens_for_batch,
    _cycle_dataloader,
    _export_discriminator_hf,
    _finalize_window_metric_loss,
    _find_latest_checkpoint,
    _flush_loggers,
    _init_trackers,
    _load_checkpoint_data_progress,
    _persist_or_validate_run_configs,
    _prepare_output_dir,
    _resolve_compile_scope,
    _resolve_output_dir,
    _resolve_resume_checkpoint,
    _save_checkpoint_data_progress,
    _save_training_checkpoint,
    _scale_loss_for_backward,
    _should_clip_gradients,
    _should_force_legacy_tf32_for_compile,
    _sync_discriminator_embeddings_if_available,
    _token_weighted_micro_objective,
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


def test_resolve_output_dir_keeps_explicit_path():
    out = _resolve_output_dir(
        output_dir="runs/custom/run-01",
        project_name="ignored",
        config_path="configs/pretrain_rope_fineweb_edu.yaml",
    )
    assert out == Path("runs/custom/run-01")


def test_resolve_resume_checkpoint_auto_returns_none_when_no_checkpoint(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_resume_checkpoint(
        output_dir=out,
        resume_from_checkpoint="auto",
        is_main_process=True,
    )
    assert resolved is None


def test_checkpoint_data_progress_roundtrip(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir(parents=True, exist_ok=True)

    assert _load_checkpoint_data_progress(ckpt) is None
    _save_checkpoint_data_progress(checkpoint_dir=ckpt, consumed_micro_batches=123)
    assert _load_checkpoint_data_progress(ckpt) == 123


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


def test_run_pretraining_keyboard_interrupt_logs_crash_and_finishes_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deberta.training.pretrain as pretrain_mod

    class _FakeWandbRun:
        def __init__(self) -> None:
            self.summary: dict[str, Any] = {}
            self.logged: list[tuple[dict[str, Any], int | None]] = []
            self.finished_exit_code: int | None = None

        def log(self, row: dict[str, Any], step: int | None = None) -> None:
            self.logged.append((dict(row), step))

        def finish(self, exit_code: int = 0) -> None:
            self.finished_exit_code = int(exit_code)

    class _FakeAccelerator:
        last_instance: _FakeAccelerator | None = None

        def __init__(
            self, *, gradient_accumulation_steps: int, log_with: str | None, mixed_precision: str
        ) -> None:
            del gradient_accumulation_steps, log_with, mixed_precision
            self.is_main_process = True
            self.process_index = 0
            self.num_processes = 1
            self.device = torch.device("cpu")
            self.state = "fake-accelerator"
            self.logged_rows: list[tuple[dict[str, Any], int | None]] = []
            self.tracker_init_calls: list[dict[str, Any]] = []
            self.ended = False
            self.wandb_run = _FakeWandbRun()
            _FakeAccelerator.last_instance = self

        def wait_for_everyone(self) -> None:
            return None

        def prepare(self, *objs):
            return objs

        def unwrap_model(self, model):
            return model

        def no_sync(self, model):
            del model
            return nullcontext()

        def backward(self, loss: torch.Tensor) -> None:
            del loss

        def clip_grad_norm_(self, params, max_norm: float) -> None:
            del params, max_norm

        def reduce(self, tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
            del reduction
            return tensor

        def gather(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        def init_trackers(
            self,
            project_name: str,
            config: dict[str, Any],
            init_kwargs: dict[str, Any] | None = None,
        ) -> None:
            self.tracker_init_calls.append(
                {
                    "project_name": project_name,
                    "config": dict(config),
                    "init_kwargs": dict(init_kwargs or {}),
                }
            )

        def get_tracker(self, name: str, unwrap: bool = True):
            del unwrap
            if name != "wandb":
                raise KeyError(name)
            return self.wandb_run

        def log(self, row: dict[str, Any], step: int | None = None) -> None:
            self.logged_rows.append((dict(row), step))

        def load_state(self, ckpt: str) -> None:
            del ckpt

        def end_training(self) -> None:
            self.ended = True

    fake_accelerate = types.ModuleType("accelerate")
    fake_accelerate.Accelerator = _FakeAccelerator
    fake_accelerate_utils = types.ModuleType("accelerate.utils")
    fake_accelerate_utils.set_seed = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", fake_accelerate_utils)

    class _FakeTokenizer:
        pad_token_id = 0
        cls_token_id = 1
        sep_token_id = 2
        mask_token_id = 3

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: _FakeTokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    batch = {
        "input_ids": torch.tensor([[1, 3, 9, 2, 0]], dtype=torch.long),
        "labels": torch.tensor([[-100, 10, -100, -100, -100]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long),
    }

    def _fake_cycle(_loader):
        yield batch
        raise KeyboardInterrupt

    class _FakeRTD(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            del kwargs
            self.weight = torch.nn.Parameter(torch.ones(1))
            self._forbidden_sample_token_ids = {0, 1, 2, 3}
            self.disc_config = types.SimpleNamespace(pad_token_id=0)

        def forward(self, **kwargs):
            del kwargs
            t = torch.tensor(1.0)
            return types.SimpleNamespace(
                loss=t,
                gen_loss=t,
                disc_loss=t,
                disc_accuracy=t,
                gen_token_count=torch.tensor(1.0),
                disc_token_count=torch.tensor(1.0),
                gen_loss_raw=t,
                disc_loss_raw=t,
            )

    saved_checkpoints: list[tuple[str, int, str]] = []

    def _fake_save_checkpoint(
        *,
        accelerator: Any,
        checkpoint_dir: Path,
        output_dir: Path,
        consumed_micro_batches: int,
        save_total_limit: int,
        log_label: str,
    ) -> None:
        del accelerator, output_dir, save_total_limit
        saved_checkpoints.append((str(checkpoint_dir), int(consumed_micro_batches), str(log_label)))

    monkeypatch.setattr(pretrain_mod, "_bf16_runtime_sanity_check", lambda: True)
    monkeypatch.setattr(pretrain_mod, "_maybe_enable_tf32", lambda *args, **kwargs: None)
    monkeypatch.setattr(pretrain_mod, "_maybe_configure_sdpa_kernels", lambda *args, **kwargs: None)
    monkeypatch.setattr(pretrain_mod, "load_hf_dataset", lambda **kwargs: [{"text": "hello"}])
    monkeypatch.setattr(pretrain_mod, "PackedStreamingDataset", lambda **kwargs: [batch])
    monkeypatch.setattr(pretrain_mod, "SequentialStreamingDataset", lambda **kwargs: [batch])
    monkeypatch.setattr(pretrain_mod, "_build_training_collator", lambda **kwargs: lambda rows: rows[0])
    monkeypatch.setattr(
        pretrain_mod,
        "build_backbone_configs",
        lambda **kwargs: (types.SimpleNamespace(pad_token_id=0), types.SimpleNamespace()),
    )
    monkeypatch.setattr(
        pretrain_mod,
        "build_backbones",
        lambda **kwargs: (torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)),
    )
    monkeypatch.setattr(pretrain_mod, "DebertaV3RTDPretrainer", _FakeRTD)
    monkeypatch.setattr(
        pretrain_mod, "_build_optimizer", lambda model, _cfg: torch.optim.SGD(model.parameters(), lr=0.1)
    )
    monkeypatch.setattr(
        pretrain_mod,
        "_build_scheduler",
        lambda optimizer, _cfg: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0),
    )
    monkeypatch.setattr(pretrain_mod, "_cycle_dataloader", _fake_cycle)
    monkeypatch.setattr(pretrain_mod, "_move_batch_to_device", lambda b, _device: b)
    monkeypatch.setattr(pretrain_mod, "_save_training_checkpoint", _fake_save_checkpoint)
    monkeypatch.setattr(pretrain_mod, "_export_discriminator_hf", lambda **kwargs: None)

    model_cfg = ModelConfig()
    data_cfg = DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy")
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
        pretrain_mod.run_pretraining(model_cfg=model_cfg, data_cfg=data_cfg, train_cfg=train_cfg)

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

    accel = _FakeAccelerator.last_instance
    assert accel is not None
    assert accel.wandb_run.summary["crashed"] is True
    assert accel.wandb_run.summary["crash_type"] == "KeyboardInterrupt"
    assert accel.wandb_run.finished_exit_code == 130
    assert accel.wandb_run.logged
    assert accel.tracker_init_calls
    first_tracker_call = accel.tracker_init_calls[0]
    assert first_tracker_call["project_name"] == "deberta-train"
    assert first_tracker_call["init_kwargs"]["wandb"]["name"] == "run"
    assert accel.ended is False


def test_run_pretraining_compiles_only_generator_and_discriminator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deberta.training.pretrain as pretrain_mod

    compile_calls: list[tuple[torch.nn.Module, dict[str, Any]]] = []

    class _FakeAccelerator:
        def __init__(
            self, *, gradient_accumulation_steps: int, log_with: str | None, mixed_precision: str
        ) -> None:
            del gradient_accumulation_steps, log_with, mixed_precision
            self.is_main_process = True
            self.process_index = 0
            self.num_processes = 1
            self.device = torch.device("cpu")
            self.state = "fake-accelerator"

        def wait_for_everyone(self) -> None:
            return None

        def prepare(self, *objs):
            return objs

        def unwrap_model(self, model):
            return model

        def no_sync(self, model):
            del model
            return nullcontext()

        def backward(self, loss: torch.Tensor) -> None:
            del loss

        def clip_grad_norm_(self, params, max_norm: float) -> None:
            del params, max_norm

        def reduce(self, tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
            del reduction
            return tensor

        def gather(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor

    fake_accelerate = types.ModuleType("accelerate")
    fake_accelerate.Accelerator = _FakeAccelerator
    fake_accelerate_utils = types.ModuleType("accelerate.utils")
    fake_accelerate_utils.set_seed = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", fake_accelerate_utils)

    class _FakeTokenizer:
        pad_token_id = 0
        cls_token_id = 1
        sep_token_id = 2
        mask_token_id = 3

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: _FakeTokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    batch = {
        "input_ids": torch.tensor([[1, 3, 9, 2, 0]], dtype=torch.long),
        "labels": torch.tensor([[-100, 10, -100, -100, -100]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long),
    }

    def _fake_cycle(_loader):
        while True:
            yield batch

    class _FakeRTD(torch.nn.Module):
        last_instance: _FakeRTD | None = None

        def __init__(self, **kwargs) -> None:
            super().__init__()
            del kwargs
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.generator = torch.nn.Linear(2, 2)
            self.discriminator = torch.nn.Linear(2, 2)
            self._forbidden_sample_token_mask = torch.zeros(32, dtype=torch.bool)
            self.disc_config = types.SimpleNamespace(pad_token_id=0)
            _FakeRTD.last_instance = self

        def forward(self, **kwargs):
            del kwargs
            t = self.weight * 0.0 + 1.0
            return types.SimpleNamespace(
                loss=t,
                gen_loss=t.detach(),
                disc_loss=t.detach(),
                disc_accuracy=t.detach(),
                gen_token_count=torch.tensor(1.0),
                disc_token_count=torch.tensor(1.0),
                gen_loss_raw=t,
                disc_loss_raw=t,
            )

    def _fake_compile(
        module: torch.nn.Module,
        *,
        mode: str = "default",
        backend: str = "inductor",
        dynamic: bool | None = None,
    ) -> torch.nn.Module:
        compile_calls.append((module, {"mode": str(mode), "backend": str(backend), "dynamic": dynamic}))
        return module

    monkeypatch.setattr(pretrain_mod, "_bf16_runtime_sanity_check", lambda: True)
    monkeypatch.setattr(pretrain_mod, "_maybe_enable_tf32", lambda *args, **kwargs: None)
    monkeypatch.setattr(pretrain_mod, "_maybe_configure_sdpa_kernels", lambda *args, **kwargs: None)
    monkeypatch.setattr(pretrain_mod, "load_hf_dataset", lambda **kwargs: [{"text": "hello"}])
    monkeypatch.setattr(pretrain_mod, "PackedStreamingDataset", lambda **kwargs: [batch])
    monkeypatch.setattr(pretrain_mod, "SequentialStreamingDataset", lambda **kwargs: [batch])
    monkeypatch.setattr(pretrain_mod, "_build_training_collator", lambda **kwargs: lambda rows: rows[0])
    monkeypatch.setattr(
        pretrain_mod,
        "build_backbone_configs",
        lambda **kwargs: (types.SimpleNamespace(pad_token_id=0), types.SimpleNamespace()),
    )
    monkeypatch.setattr(
        pretrain_mod,
        "build_backbones",
        lambda **kwargs: (torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)),
    )
    monkeypatch.setattr(pretrain_mod, "DebertaV3RTDPretrainer", _FakeRTD)
    monkeypatch.setattr(
        pretrain_mod, "_build_optimizer", lambda model, _cfg: torch.optim.SGD(model.parameters(), lr=0.1)
    )
    monkeypatch.setattr(
        pretrain_mod,
        "_build_scheduler",
        lambda optimizer, _cfg: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0),
    )
    monkeypatch.setattr(pretrain_mod, "_cycle_dataloader", _fake_cycle)
    monkeypatch.setattr(pretrain_mod, "_move_batch_to_device", lambda b, _device: b)
    monkeypatch.setattr(pretrain_mod, "_save_training_checkpoint", lambda **kwargs: None)
    monkeypatch.setattr(pretrain_mod.torch, "compile", _fake_compile)

    model_cfg = ModelConfig()
    data_cfg = DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy")
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

    pretrain_mod.run_pretraining(model_cfg=model_cfg, data_cfg=data_cfg, train_cfg=train_cfg)

    instance = _FakeRTD.last_instance
    assert instance is not None
    assert len(compile_calls) == 2
    assert compile_calls[0][0] is instance.generator
    assert compile_calls[1][0] is instance.discriminator
    assert compile_calls[0][1]["mode"] == "default"
    assert compile_calls[1][1]["mode"] == "default"
    assert compile_calls[0][1]["backend"] == "inductor"
    assert compile_calls[1][1]["backend"] == "inductor"
    assert compile_calls[0][1]["dynamic"] is False
    assert compile_calls[1][1]["dynamic"] is False


def test_run_pretraining_hf_deberta_auto_scope_compiles_ffn_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deberta.training.pretrain as pretrain_mod

    compile_calls: list[tuple[torch.nn.Module, dict[str, Any]]] = []

    class _FakeAccelerator:
        def __init__(
            self, *, gradient_accumulation_steps: int, log_with: str | None, mixed_precision: str
        ) -> None:
            del gradient_accumulation_steps, log_with, mixed_precision
            self.is_main_process = True
            self.process_index = 0
            self.num_processes = 1
            self.device = torch.device("cpu")
            self.state = "fake-accelerator"

        def wait_for_everyone(self) -> None:
            return None

        def prepare(self, *objs):
            return objs

        def unwrap_model(self, model):
            return model

        def no_sync(self, model):
            del model
            return nullcontext()

        def backward(self, loss: torch.Tensor) -> None:
            del loss

        def clip_grad_norm_(self, params, max_norm: float) -> None:
            del params, max_norm

        def reduce(self, tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
            del reduction
            return tensor

        def gather(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor

    fake_accelerate = types.ModuleType("accelerate")
    fake_accelerate.Accelerator = _FakeAccelerator
    fake_accelerate_utils = types.ModuleType("accelerate.utils")
    fake_accelerate_utils.set_seed = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", fake_accelerate_utils)

    class _FakeTokenizer:
        pad_token_id = 0
        cls_token_id = 1
        sep_token_id = 2
        mask_token_id = 3

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: _FakeTokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    batch = {
        "input_ids": torch.tensor([[1, 3, 9, 2, 0]], dtype=torch.long),
        "labels": torch.tensor([[-100, 10, -100, -100, -100]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long),
    }

    def _fake_cycle(_loader):
        while True:
            yield batch

    class _FakeLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = torch.nn.Linear(2, 2)
            self.intermediate = torch.nn.Linear(2, 2)
            self.output = torch.nn.Linear(2, 2)

    class _FakeEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.ModuleList([_FakeLayer()])

    class _FakeBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = torch.nn.Linear(2, 2)
            self.encoder = _FakeEncoder()

    class _FakeRTD(torch.nn.Module):
        last_instance: _FakeRTD | None = None

        def __init__(self, **kwargs) -> None:
            super().__init__()
            del kwargs
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.generator = _FakeBackbone()
            self.discriminator = _FakeBackbone()
            self._forbidden_sample_token_mask = torch.zeros(32, dtype=torch.bool)
            self.disc_config = types.SimpleNamespace(pad_token_id=0)
            _FakeRTD.last_instance = self

        def forward(self, **kwargs):
            del kwargs
            t = self.weight * 0.0 + 1.0
            return types.SimpleNamespace(
                loss=t,
                gen_loss=t.detach(),
                disc_loss=t.detach(),
                disc_accuracy=t.detach(),
                gen_token_count=torch.tensor(1.0),
                disc_token_count=torch.tensor(1.0),
                gen_loss_raw=t,
                disc_loss_raw=t,
            )

    def _fake_compile(
        module: torch.nn.Module,
        *,
        mode: str = "default",
        backend: str = "inductor",
        dynamic: bool | None = None,
    ) -> torch.nn.Module:
        compile_calls.append((module, {"mode": str(mode), "backend": str(backend), "dynamic": dynamic}))
        return module

    monkeypatch.setattr(pretrain_mod, "_bf16_runtime_sanity_check", lambda: True)
    monkeypatch.setattr(pretrain_mod, "_maybe_enable_tf32", lambda *args, **kwargs: None)
    monkeypatch.setattr(pretrain_mod, "_maybe_configure_sdpa_kernels", lambda *args, **kwargs: None)
    monkeypatch.setattr(pretrain_mod, "load_hf_dataset", lambda **kwargs: [{"text": "hello"}])
    monkeypatch.setattr(pretrain_mod, "PackedStreamingDataset", lambda **kwargs: [batch])
    monkeypatch.setattr(pretrain_mod, "SequentialStreamingDataset", lambda **kwargs: [batch])
    monkeypatch.setattr(pretrain_mod, "_build_training_collator", lambda **kwargs: lambda rows: rows[0])
    monkeypatch.setattr(
        pretrain_mod,
        "build_backbone_configs",
        lambda **kwargs: (types.SimpleNamespace(pad_token_id=0), types.SimpleNamespace()),
    )
    monkeypatch.setattr(
        pretrain_mod,
        "build_backbones",
        lambda **kwargs: (_FakeBackbone(), _FakeBackbone()),
    )
    monkeypatch.setattr(pretrain_mod, "DebertaV3RTDPretrainer", _FakeRTD)
    monkeypatch.setattr(
        pretrain_mod, "_build_optimizer", lambda model, _cfg: torch.optim.SGD(model.parameters(), lr=0.1)
    )
    monkeypatch.setattr(
        pretrain_mod,
        "_build_scheduler",
        lambda optimizer, _cfg: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0),
    )
    monkeypatch.setattr(pretrain_mod, "_cycle_dataloader", _fake_cycle)
    monkeypatch.setattr(pretrain_mod, "_move_batch_to_device", lambda b, _device: b)
    monkeypatch.setattr(pretrain_mod, "_save_training_checkpoint", lambda **kwargs: None)
    monkeypatch.setattr(pretrain_mod.torch, "compile", _fake_compile)

    model_cfg = ModelConfig(backbone_type="hf_deberta_v2")
    data_cfg = DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy")
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

    pretrain_mod.run_pretraining(model_cfg=model_cfg, data_cfg=data_cfg, train_cfg=train_cfg)

    instance = _FakeRTD.last_instance
    assert instance is not None
    assert len(compile_calls) == 4
    assert compile_calls[0][0] is instance.generator.encoder.layer[0].intermediate
    assert compile_calls[1][0] is instance.generator.encoder.layer[0].output
    assert compile_calls[2][0] is instance.discriminator.encoder.layer[0].intermediate
    assert compile_calls[3][0] is instance.discriminator.encoder.layer[0].output
    assert compile_calls[0][1]["mode"] == "default"
    assert compile_calls[1][1]["mode"] == "default"
    assert compile_calls[2][1]["mode"] == "default"
    assert compile_calls[3][1]["mode"] == "default"
    assert compile_calls[0][1]["backend"] == "inductor"
    assert compile_calls[1][1]["backend"] == "inductor"
    assert compile_calls[2][1]["backend"] == "inductor"
    assert compile_calls[3][1]["backend"] == "inductor"
    assert compile_calls[0][1]["dynamic"] is False
    assert compile_calls[1][1]["dynamic"] is False
    assert compile_calls[2][1]["dynamic"] is False
    assert compile_calls[3][1]["dynamic"] is False


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
    assert _load_checkpoint_data_progress(ckpt) == 42


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

    class _FakeTokenizer:
        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

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
        tokenizer=_FakeTokenizer(),
        output_dir=tmp_path / "export",
        embedding_sharing="none",
    )

    assert called_targets == [inner.discriminator, inner.generator]


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


def test_model_config_defaults_leave_dropout_overrides_unset():
    cfg = ModelConfig()
    assert cfg.hidden_dropout_prob is None
    assert cfg.attention_probs_dropout_prob is None


def test_data_config_defaults_disable_cross_document_blocking():
    cfg = DataConfig()
    assert cfg.block_cross_document_attention is False


def test_model_config_defaults_to_adjust_swiglu_intermediate():
    cfg = ModelConfig()
    assert cfg.swiglu_adjust_intermediate is True


def test_train_config_defaults_to_token_weighted_gradient_accumulation():
    cfg = TrainConfig()
    assert cfg.token_weighted_gradient_accumulation is True


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
    cfg = TrainConfig(torch_compile_scope="generator_ffn", torch_compile_backend="aot-eager")
    validate_train_config(cfg)
    assert cfg.torch_compile_scope == "gen_ffn_only"
    assert cfg.torch_compile_backend == "aot_eager"


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


def test_should_clip_gradients_only_on_sync_steps():
    assert _should_clip_gradients(sync_gradients=False, max_grad_norm=1.0) is False
    assert _should_clip_gradients(sync_gradients=True, max_grad_norm=None) is False
    assert _should_clip_gradients(sync_gradients=True, max_grad_norm=0.0) is False
    assert _should_clip_gradients(sync_gradients=True, max_grad_norm=-1.0) is False
    assert _should_clip_gradients(sync_gradients=True, max_grad_norm=1.0) is True


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


def test_count_input_tokens_for_batch_uses_attention_mask_when_available():
    batch = {
        "input_ids": torch.tensor([[10, 11, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
    }
    assert _count_input_tokens_for_batch(batch) == pytest.approx(2.0)


def test_count_input_tokens_for_batch_falls_back_to_input_size():
    batch = {"input_ids": torch.tensor([[10, 11, 12], [13, 14, 15]], dtype=torch.long)}
    assert _count_input_tokens_for_batch(batch) == pytest.approx(6.0)


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


def test_normalize_compile_scope_accepts_aliases_and_rejects_invalid():
    assert _normalize_torch_compile_scope("auto") == "auto"
    assert _normalize_torch_compile_scope("backbone") == "backbones"
    assert _normalize_torch_compile_scope("full") == "backbones"
    assert _normalize_torch_compile_scope("encoder") == "encoder_only"
    assert _normalize_torch_compile_scope("generator_encoder") == "gen_encoder_only"
    assert _normalize_torch_compile_scope("disc-encoder") == "disc_encoder_only"
    assert _normalize_torch_compile_scope("ffn") == "ffn_only"
    assert _normalize_torch_compile_scope("generator_ffn") == "gen_ffn_only"
    assert _normalize_torch_compile_scope("disc_ffn") == "disc_ffn_only"

    with pytest.raises(ValueError, match="train.torch_compile_scope must be one of"):
        _normalize_torch_compile_scope("all")


def test_normalize_compile_backend_accepts_aliases_and_rejects_invalid():
    assert _normalize_torch_compile_backend("inductor") == "inductor"
    assert _normalize_torch_compile_backend("aot-eager") == "aot_eager"

    with pytest.raises(ValueError, match="train.torch_compile_backend must be one of"):
        _normalize_torch_compile_backend("xla")


def test_normalize_hf_attention_kernel_accepts_aliases_and_rejects_invalid():
    assert _normalize_hf_attention_kernel("dynamic") == "dynamic"
    assert _normalize_hf_attention_kernel("cache") == "cached_bmm"
    assert _normalize_hf_attention_kernel("cached-bmm") == "cached_bmm"

    with pytest.raises(ValueError, match="model.hf_attention_kernel must be one of"):
        _normalize_hf_attention_kernel("einsum")


def test_resolve_compile_scope_uses_hf_deberta_v2_default_inductor_fallback():
    scope, reason = _resolve_compile_scope(
        requested_scope="auto",
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
        compile_mode="default",
        compile_backend="inductor",
    )
    assert scope == "ffn_only"
    assert reason is not None

    scope, reason = _resolve_compile_scope(
        requested_scope="auto",
        model_cfg=ModelConfig(backbone_type="rope"),
        compile_mode="default",
        compile_backend="inductor",
    )
    assert scope == "backbones"
    assert reason is None

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


def test_validate_data_config_canonicalizes_non_streaming_shuffle_buffer_size():
    cfg = DataConfig(
        dataset_name="HuggingFaceFW/fineweb-edu",
        streaming=False,
        shuffle_buffer_size=10_000,
    )
    validate_data_config(cfg)
    assert cfg.shuffle_buffer_size == 1


def test_validate_data_config_disables_doc_blocking_when_not_packed():
    cfg = DataConfig(
        dataset_name="HuggingFaceFW/fineweb-edu",
        pack_sequences=False,
        block_cross_document_attention=True,
    )
    validate_data_config(cfg)
    assert cfg.block_cross_document_attention is False


def test_validate_training_workflow_options_rejects_flash_only_with_packing():
    with pytest.raises(ValueError, match="flash_only is not supported with data.pack_sequences=true"):
        validate_training_workflow_options(
            data_cfg=DataConfig(
                dataset_name="HuggingFaceFW/fineweb-edu",
                pack_sequences=True,
                block_cross_document_attention=True,
            ),
            train_cfg=TrainConfig(sdpa_kernel="flash_only"),
        )


def test_validate_training_workflow_options_allows_flash_only_when_packed_doc_blocking_disabled():
    validate_training_workflow_options(
        data_cfg=DataConfig(
            dataset_name="HuggingFaceFW/fineweb-edu",
            pack_sequences=True,
            block_cross_document_attention=False,
        ),
        train_cfg=TrainConfig(sdpa_kernel="flash_only"),
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


def test_validate_model_config_rejects_rope_only_knobs_in_hf_mode():
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


def test_validate_model_config_rejects_hf_attention_kernel_override_for_rope_backbone():
    cfg = ModelConfig(backbone_type="rope", hf_attention_kernel="cached_bmm")
    with pytest.raises(ValueError, match="model.hf_attention_kernel only applies when"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_derived_generator_knobs_with_explicit_generator_source():
    cfg = ModelConfig(
        backbone_type="rope",
        generator_model_name_or_path="microsoft/deberta-v3-small",
        generator_hidden_size=256,
    )
    with pytest.raises(ValueError, match="only used when deriving generator config"):
        validate_model_config(cfg)


def test_validate_model_config_requires_generator_model_source_for_pretrained_generator_config():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        discriminator_model_name_or_path="local-rope-disc",
        generator_config_name_or_path="local-rope-gen-config",
        generator_model_name_or_path=None,
    )
    with pytest.raises(ValueError, match="requires model.generator_model_name_or_path"):
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
        discriminator_config_name_or_path="local-rope-disc",
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
