from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from _fakes import DummyTokenizer

import deberta.export_cli as export_cli
from deberta.config import RUN_CONFIG_SCHEMA_VERSION, DataConfig, ModelConfig


class _FakeExportBackbone:
    def __init__(self) -> None:
        self._weights: dict[str, torch.Tensor] = {"weight": torch.tensor(0.0)}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._weights

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = False) -> Any:
        del strict
        model_keys = set(self._weights.keys())
        source_keys = set(state_dict.keys())
        missing = sorted(model_keys - source_keys)
        unexpected = sorted(source_keys - model_keys)
        for key in model_keys & source_keys:
            self._weights[key] = state_dict[key]
        return types.SimpleNamespace(missing_keys=missing, unexpected_keys=unexpected)

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
    model_cfg = ModelConfig(
        tokenizer_name_or_path="dummy-tokenizer",
        embedding_sharing="none",
    )
    (run_dir / "model_config.json").write_text(
        json.dumps(asdict(model_cfg)),
        encoding="utf-8",
    )
    data_cfg = DataConfig(dataset_name="dummy-dataset", max_seq_length=32)
    (run_dir / "data_config.json").write_text(
        json.dumps(asdict(data_cfg)),
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "checkpoint-10"
    checkpoint_dir.mkdir()
    return run_dir, checkpoint_dir


def _new_export_call_counters() -> dict[str, object]:
    """Return a fresh mutable counter map used by export fakes."""
    return {
        "get_state_dict": 0,
        "unwrap_model": 0,
        "get_model_state_dict": 0,
        "options_kwargs": None,
        "load_state_calls": [],
        "build_backbones_calls": [],
    }


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
    called = _new_export_call_counters()
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
    called = _new_export_call_counters()
    called["remap_calls"] = []
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
    called = _new_export_call_counters()
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


def test_run_export_fails_on_partial_backbone_load_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, checkpoint_dir = _write_run_layout(tmp_path)
    called = _new_export_call_counters()
    _install_export_fakes(
        monkeypatch=monkeypatch,
        called=called,
        fsdp2=False,
        provide_torch_state_dict_api=False,
    )

    class _BrokenBackbone(_FakeExportBackbone):
        def __init__(self) -> None:
            self._weights = {"other_weight": torch.tensor(0.0)}

    monkeypatch.setattr(
        export_cli,
        "_build_export_backbone",
        lambda model_cfg, disc_config, gen_config, export_what: (_BrokenBackbone(), None),
    )

    with pytest.raises(RuntimeError, match="partial state_dict load rejected"):
        export_cli.run_export(
            export_cli.ExportConfig(
                checkpoint_dir=str(checkpoint_dir),
                run_dir=str(run_dir),
                output_dir=str(tmp_path / "exported"),
                export_what="discriminator",
            )
        )


def test_run_export_allows_partial_backbone_load_with_opt_in_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, checkpoint_dir = _write_run_layout(tmp_path)
    called = _new_export_call_counters()
    _install_export_fakes(
        monkeypatch=monkeypatch,
        called=called,
        fsdp2=False,
        provide_torch_state_dict_api=False,
    )

    class _BrokenBackbone(_FakeExportBackbone):
        def __init__(self) -> None:
            self._weights = {"other_weight": torch.tensor(0.0)}

    monkeypatch.setattr(
        export_cli,
        "_build_export_backbone",
        lambda model_cfg, disc_config, gen_config, export_what: (_BrokenBackbone(), None),
    )

    out_dir = tmp_path / "exported"
    export_cli.run_export(
        export_cli.ExportConfig(
            checkpoint_dir=str(checkpoint_dir),
            run_dir=str(run_dir),
            output_dir=str(out_dir),
            export_what="discriminator",
            allow_partial_export=True,
        )
    )
    assert (out_dir / "discriminator").exists()


def test_run_export_strict_load_allows_gdes_discriminator_embedding_key_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, checkpoint_dir = _write_run_layout(tmp_path)
    called = _new_export_call_counters()
    _install_export_fakes(
        monkeypatch=monkeypatch,
        called=called,
        fsdp2=False,
        provide_torch_state_dict_api=False,
    )

    class _EmbeddingWeightBackbone(_FakeExportBackbone):
        def __init__(self) -> None:
            self._weights = {
                "embeddings.word_embeddings.weight": torch.tensor([0.0]),
                "encoder.weight": torch.tensor([0.0]),
            }

    merge_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        export_cli,
        "_build_export_backbone",
        lambda model_cfg, disc_config, gen_config, export_what: (_EmbeddingWeightBackbone(), None),
    )
    monkeypatch.setattr(
        export_cli,
        "split_pretrainer_state_dict",
        lambda full_sd: (
            {
                "embeddings.word_embeddings.base_weight": torch.tensor([1.0]),
                "embeddings.word_embeddings.bias": torch.tensor([0.2]),
                "encoder.weight": torch.tensor([3.0]),
            },
            {"embeddings.word_embeddings.weight": torch.tensor([0.8])},
        ),
    )
    monkeypatch.setattr(
        export_cli,
        "merge_embeddings_into_export_backbone",
        lambda **kwargs: merge_calls.append(dict(kwargs)),
    )

    out_dir = tmp_path / "exported"
    export_cli.run_export(
        export_cli.ExportConfig(
            checkpoint_dir=str(checkpoint_dir),
            run_dir=str(run_dir),
            output_dir=str(out_dir),
            export_what="discriminator",
            embedding_sharing="gdes",
        )
    )

    assert (out_dir / "discriminator").exists()
    assert len(merge_calls) == 1
    assert "embeddings.word_embeddings.bias" in merge_calls[0]["disc_sd"]


def test_run_export_rejects_non_empty_output_dir_before_loading_model_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, checkpoint_dir = _write_run_layout(tmp_path)
    out_dir = tmp_path / "exported"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stale.txt").write_text("x", encoding="utf-8")

    called = _new_export_call_counters()
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


def test_run_export_strips_training_internal_keys_from_saved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, checkpoint_dir = _write_run_layout(tmp_path)
    called = _new_export_call_counters()
    _install_export_fakes(
        monkeypatch=monkeypatch,
        called=called,
        fsdp2=False,
        provide_torch_state_dict_api=False,
    )

    class _ConfigWritingBackbone(_FakeExportBackbone):
        def save_pretrained(self, path: str, safe_serialization: bool = True) -> None:
            del safe_serialization
            target = Path(path)
            target.mkdir(parents=True, exist_ok=True)
            payload = {
                "model_type": "deberta-v2",
                "hidden_size": 768,
                "hf_attention_kernel": "stable",
                "use_rmsnorm_heads": False,
                "legacy": True,
                "cls_token_id": 1,
            }
            (target / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        export_cli,
        "_build_export_backbone",
        lambda model_cfg, disc_config, gen_config, export_what: (_ConfigWritingBackbone(), None),
    )

    out_dir = tmp_path / "exported"
    export_cli.run_export(
        export_cli.ExportConfig(
            checkpoint_dir=str(checkpoint_dir),
            run_dir=str(run_dir),
            output_dir=str(out_dir),
            export_what="discriminator",
        )
    )

    config_path = out_dir / "discriminator" / "config.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "hf_attention_kernel" not in data
    assert "use_rmsnorm_heads" not in data
    assert "legacy" not in data
    assert "cls_token_id" not in data
    assert data["model_type"] == "deberta-v2"
    assert data["hidden_size"] == 768


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


def test_namespace_to_export_config_maps_allow_partial_export() -> None:
    ns = types.SimpleNamespace(
        checkpoint_dir="/tmp/checkpoint-1",
        output_dir="/tmp/exported",
        run_dir="/tmp/run",
        export_what="discriminator",
        safe_serialization=True,
        offload_to_cpu=True,
        rank0=True,
        embedding_sharing=None,
        allow_partial_export=True,
    )
    cfg = export_cli.namespace_to_export_config(ns)
    assert cfg.allow_partial_export is True
