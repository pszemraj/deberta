from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

import deberta.export_cli as export_cli
from deberta.config import RUN_CONFIG_SCHEMA_VERSION


class _FakeTokenizer:
    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


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

        def load_state(self, _checkpoint_dir: str) -> None:
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
        from_pretrained=lambda *args, **kwargs: _FakeTokenizer()
    )
    fake_accelerate = types.ModuleType("accelerate")
    fake_accelerate.Accelerator = _FakeAccelerator
    fake_accelerate.utils = fake_utils

    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(export_cli, "build_backbone_configs", lambda **kwargs: (object(), object()))
    monkeypatch.setattr(export_cli, "build_backbones", lambda *args, **kwargs: (object(), object()))
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
    assert called["unwrap_model"] == 0


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
