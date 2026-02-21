from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

import deberta.export_cli as export_cli


def test_run_export_prefers_accelerator_state_dict_for_fsdp_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
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

    class _FakeTokenizer:
        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    fake_utils = types.ModuleType("accelerate.utils")
    fake_utils.DistributedType = types.SimpleNamespace(FSDP="FSDP")

    called = {"get_state_dict": 0, "unwrap_model": 0}

    class _FakeAccelerator:
        distributed_type = fake_utils.DistributedType.FSDP

        def __init__(self) -> None:
            self.is_main_process = True

        def prepare(self, model):
            return model

        def load_state(self, _checkpoint_dir: str) -> None:
            return None

        def wait_for_everyone(self) -> None:
            return None

        def get_state_dict(self, model):
            called["get_state_dict"] += 1
            return {
                "discriminator.weight": torch.tensor(1.0),
                "generator.weight": torch.tensor(2.0),
            }

        def unwrap_model(self, model):
            called["unwrap_model"] += 1
            return model

    class _FakeExportBackbone:
        def __init__(self) -> None:
            self._weights = {}

        def state_dict(self):
            return self._weights

        def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = False) -> None:
            del strict
            self._weights.update(state_dict)

        def save_pretrained(self, path: str, safe_serialization: bool = True) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            del safe_serialization
            return None

    class _FakeRTDModel:
        def __init__(self, *args, **kwargs) -> None:
            del args
            del kwargs

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: _FakeTokenizer()
    )
    fake_accelerate = types.ModuleType("accelerate")
    fake_accelerate.Accelerator = _FakeAccelerator
    fake_accelerate.utils = fake_utils

    def _fake_build_backbone_configs(**kwargs):
        del kwargs
        return object(), object()

    def _fake_build_backbones(*args, **kwargs):
        del args
        del kwargs
        return object(), object()

    def _fake_rtd_pretrainer(*args, **kwargs):
        del args
        del kwargs
        return _FakeRTDModel()

    def _fake_build_export_backbone(model_cfg, disc_config, gen_config, export_what):
        del model_cfg
        del disc_config
        del gen_config
        del export_what
        return _FakeExportBackbone(), None

    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(export_cli, "build_backbone_configs", _fake_build_backbone_configs)
    monkeypatch.setattr(export_cli, "build_backbones", _fake_build_backbones)
    monkeypatch.setattr(export_cli, "DebertaV3RTDPretrainer", _fake_rtd_pretrainer)
    monkeypatch.setattr(export_cli, "_build_export_backbone", _fake_build_export_backbone)
    monkeypatch.setattr(export_cli, "merge_embeddings_into_export_backbone", lambda *args, **kwargs: None)

    export_cli.run_export(
        export_cli.ExportConfig(
            checkpoint_dir=str(checkpoint_dir),
            run_dir=str(run_dir),
            output_dir=str(tmp_path / "exported"),
        )
    )

    assert called["get_state_dict"] == 1
    assert called["unwrap_model"] == 0
