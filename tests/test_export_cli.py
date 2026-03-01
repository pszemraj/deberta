from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from _fakes import DummyTokenizer

import deberta.export_cli as export_cli
from deberta.config import RUN_CONFIG_SCHEMA_VERSION


class _FakeExportBackbone:
    def __init__(self) -> None:
        self._weights: dict[str, torch.Tensor] = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._weights

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = False) -> None:
        del strict
        self._weights.update(state_dict)

    def save_pretrained(self, path: str, safe_serialization: bool = True) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        del safe_serialization
        return None


class _FakeRTDModel:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args
        del kwargs


def _write_run_layout(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_config.json").write_text(
        json.dumps(
            {
                "tokenizer_name_or_path": "dummy-tokenizer",
                "embedding_sharing": "none",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "data_config.json").write_text(
        json.dumps({"dataset_name": "dummy-dataset", "max_seq_length": 32}),
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "checkpoint-10"
    checkpoint_dir.mkdir()
    return run_dir, checkpoint_dir


def _install_export_fakes(
    *,
    monkeypatch: pytest.MonkeyPatch,
    called: dict[str, object],
    fsdp2: bool,
    provide_torch_state_dict_api: bool,
    load_state_orig_mod_mismatch: bool = False,
) -> None:
    fake_utils = types.ModuleType("accelerate.utils")
    fake_utils.DistributedType = types.SimpleNamespace(FSDP="FSDP")

    class _FakeAccelerator:
        distributed_type = fake_utils.DistributedType.FSDP
        is_fsdp2 = fsdp2

        def __init__(self) -> None:
            self.is_main_process = True

        def prepare(self, model):
            return model

        def load_state(self, _checkpoint_dir: str, **kwargs: Any) -> None:
            called["load_state_calls"] = list(called["load_state_calls"]) + [dict(kwargs)]
            if load_state_orig_mod_mismatch and kwargs.get("strict", True):
                raise RuntimeError("Error(s) in loading state_dict with _orig_mod mismatch")
            return None

        def wait_for_everyone(self) -> None:
            return None

        def get_state_dict(self, model):
            del model
            called["get_state_dict"] = int(called["get_state_dict"]) + 1
            return {
                "discriminator.weight": torch.tensor(1.0),
                "generator.weight": torch.tensor(2.0),
            }

        def unwrap_model(self, model):
            called["unwrap_model"] = int(called["unwrap_model"]) + 1
            return model

    if provide_torch_state_dict_api:

        class _FakeStateDictOptions:
            def __init__(self, **kwargs) -> None:
                called["options_kwargs"] = kwargs

        def _fake_get_model_state_dict(model, *, options=None):
            del model
            del options
            called["get_model_state_dict"] = int(called["get_model_state_dict"]) + 1
            return {
                "discriminator.weight": torch.tensor(1.0),
                "generator.weight": torch.tensor(2.0),
            }

        fake_torch_state_dict = types.ModuleType("torch.distributed.checkpoint.state_dict")
        fake_torch_state_dict.StateDictOptions = _FakeStateDictOptions
        fake_torch_state_dict.get_model_state_dict = _fake_get_model_state_dict
        monkeypatch.setitem(sys.modules, "torch.distributed.checkpoint.state_dict", fake_torch_state_dict)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: DummyTokenizer()
    )
    fake_accelerate = types.ModuleType("accelerate")
    fake_accelerate.Accelerator = _FakeAccelerator
    fake_accelerate.utils = fake_utils

    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(export_cli, "build_backbone_configs", lambda **kwargs: (object(), object()))

    def _fake_build_backbones(*args: Any, **kwargs: Any) -> tuple[object, object]:
        del args
        called["build_backbones_calls"] = list(called.get("build_backbones_calls", [])) + [dict(kwargs)]
        return object(), object()

    monkeypatch.setattr(export_cli, "build_backbones", _fake_build_backbones)
    monkeypatch.setattr(export_cli, "DebertaV3RTDPretrainer", lambda *args, **kwargs: _FakeRTDModel())
    monkeypatch.setattr(
        export_cli,
        "_build_export_backbone",
        lambda model_cfg, disc_config, gen_config, export_what: (_FakeExportBackbone(), None),
    )
    monkeypatch.setattr(export_cli, "merge_embeddings_into_export_backbone", lambda *args, **kwargs: None)


@pytest.mark.parametrize(
    ("fsdp2", "provide_torch_state_dict_api", "offload_to_cpu", "rank0"),
    [
        (False, False, True, True),
        (True, True, False, False),
    ],
)
def test_run_export_fsdp_state_dict_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fsdp2: bool,
    provide_torch_state_dict_api: bool,
    offload_to_cpu: bool,
    rank0: bool,
):
    run_dir, checkpoint_dir = _write_run_layout(tmp_path)
    called: dict[str, object] = {
        "get_state_dict": 0,
        "unwrap_model": 0,
        "get_model_state_dict": 0,
        "options_kwargs": None,
        "load_state_calls": [],
        "build_backbones_calls": [],
    }
    _install_export_fakes(
        monkeypatch=monkeypatch,
        called=called,
        fsdp2=fsdp2,
        provide_torch_state_dict_api=provide_torch_state_dict_api,
    )

    export_cli.run_export(
        export_cli.ExportConfig(
            checkpoint_dir=str(checkpoint_dir),
            run_dir=str(run_dir),
            output_dir=str(tmp_path / "exported"),
            offload_to_cpu=offload_to_cpu,
            rank0=rank0,
        )
    )

    if fsdp2 and provide_torch_state_dict_api:
        assert called["get_model_state_dict"] == 1
        assert called["get_state_dict"] == 0
        assert called["options_kwargs"] == {
            "full_state_dict": True,
            "cpu_offload": offload_to_cpu,
            "broadcast_from_rank0": rank0,
        }
    else:
        assert called["get_state_dict"] == 1
        assert called["get_model_state_dict"] == 0
        assert called["options_kwargs"] is None
    build_backbones_calls = list(called["build_backbones_calls"])
    assert build_backbones_calls
    assert build_backbones_calls[0]["load_pretrained_weights"] is False
    assert called["unwrap_model"] == 0
    assert called["load_state_calls"] == [{}]


def test_run_export_retries_compile_wrapper_mismatch_with_key_remap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, checkpoint_dir = _write_run_layout(tmp_path)
    called: dict[str, object] = {
        "get_state_dict": 0,
        "unwrap_model": 0,
        "get_model_state_dict": 0,
        "options_kwargs": None,
        "load_state_calls": [],
        "remap_calls": [],
    }
    _install_export_fakes(
        monkeypatch=monkeypatch,
        called=called,
        fsdp2=False,
        provide_torch_state_dict_api=False,
        load_state_orig_mod_mismatch=True,
    )

    def _fake_remap(model: torch.nn.Module, checkpoint_dir_for_remap: Path) -> dict[str, int]:
        called["remap_calls"] = list(called["remap_calls"]) + [(model, checkpoint_dir_for_remap)]
        return {"matched": 2, "missing": 0, "unexpected": 0}

    monkeypatch.setattr(export_cli, "load_model_state_with_compile_key_remap", _fake_remap)

    export_cli.run_export(
        export_cli.ExportConfig(
            checkpoint_dir=str(checkpoint_dir),
            run_dir=str(run_dir),
            output_dir=str(tmp_path / "exported"),
            offload_to_cpu=True,
            rank0=True,
        )
    )

    assert called["load_state_calls"] == [{}, {"strict": False}]
    assert called["unwrap_model"] == 1
    remap_calls = list(called["remap_calls"])
    assert len(remap_calls) == 1
    assert remap_calls[0][1] == checkpoint_dir


def test_run_export_non_fsdp2_torch_fsdp_uses_rank0_only_for_full_state_dict_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, checkpoint_dir = _write_run_layout(tmp_path)
    called: dict[str, object] = {
        "get_state_dict": 0,
        "unwrap_model": 0,
        "get_model_state_dict": 0,
        "options_kwargs": None,
        "load_state_calls": [],
    }
    _install_export_fakes(
        monkeypatch=monkeypatch,
        called=called,
        fsdp2=False,
        provide_torch_state_dict_api=False,
    )

    full_cfg_calls: list[dict[str, bool]] = []

    class _FakeFullStateDictConfig:
        def __init__(self, *, offload_to_cpu: bool = False, rank0_only: bool = False) -> None:
            full_cfg_calls.append(
                {
                    "offload_to_cpu": bool(offload_to_cpu),
                    "rank0_only": bool(rank0_only),
                }
            )

    class _DummyContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeFSDP:
        @staticmethod
        def state_dict_type(model, state_dict_type, cfg_full):
            del model, state_dict_type, cfg_full
            return _DummyContext()

    class _FakeFSDPModel(_FakeFSDP):
        def state_dict(self) -> dict[str, torch.Tensor]:
            return {
                "discriminator.weight": torch.tensor(1.0),
                "generator.weight": torch.tensor(2.0),
            }

    fake_fsdp_mod = types.ModuleType("torch.distributed.fsdp")
    fake_fsdp_mod.FullStateDictConfig = _FakeFullStateDictConfig
    fake_fsdp_mod.StateDictType = types.SimpleNamespace(FULL_STATE_DICT="FULL_STATE_DICT")
    fake_fsdp_mod.FullyShardedDataParallel = _FakeFSDP
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fake_fsdp_mod)
    monkeypatch.setattr(export_cli, "DebertaV3RTDPretrainer", lambda *args, **kwargs: _FakeFSDPModel())

    export_cli.run_export(
        export_cli.ExportConfig(
            checkpoint_dir=str(checkpoint_dir),
            run_dir=str(run_dir),
            output_dir=str(tmp_path / "exported"),
            offload_to_cpu=True,
            rank0=False,
        )
    )

    assert full_cfg_calls == [{"offload_to_cpu": True, "rank0_only": False}]
    assert called["get_state_dict"] == 0
    assert called["load_state_calls"] == [{}]


def test_run_export_rejects_non_empty_output_dir_before_loading_model_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, checkpoint_dir = _write_run_layout(tmp_path)
    out_dir = tmp_path / "exported"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stale.txt").write_text("x", encoding="utf-8")

    called: dict[str, object] = {
        "get_state_dict": 0,
        "unwrap_model": 0,
        "get_model_state_dict": 0,
        "options_kwargs": None,
        "load_state_calls": [],
        "build_backbones_calls": [],
    }
    _install_export_fakes(
        monkeypatch=monkeypatch,
        called=called,
        fsdp2=False,
        provide_torch_state_dict_api=False,
    )

    def _fail_model_config_load(*args: Any, **kwargs: Any) -> object:
        del args, kwargs
        raise AssertionError("model config load should not run when output-dir preflight fails")

    monkeypatch.setattr(export_cli, "load_model_config_snapshot", _fail_model_config_load)

    with pytest.raises(ValueError, match="output_dir already exists and is not empty"):
        export_cli.run_export(
            export_cli.ExportConfig(
                checkpoint_dir=str(checkpoint_dir),
                run_dir=str(run_dir),
                output_dir=str(out_dir),
            )
        )
    assert called["build_backbones_calls"] == []
    assert called["load_state_calls"] == []


def test_validate_run_metadata_if_present_accepts_missing_metadata(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    export_cli._validate_run_metadata_if_present(run_dir)


def test_validate_run_metadata_if_present_rejects_unknown_schema(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"config_schema_version": int(RUN_CONFIG_SCHEMA_VERSION) + 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported run metadata schema"):
        export_cli._validate_run_metadata_if_present(run_dir)
