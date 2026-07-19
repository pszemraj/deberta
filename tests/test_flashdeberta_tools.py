"""Tests for FlashDeBERTa tooling behavior."""

from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest

from deberta.config import DataConfig, DataPackingConfig, ModelConfig, ModelHFConfig, ModelHFFlashConfig


def test_bias_tuner_forces_dense_route_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    tune_mod = importlib.import_module("flashdeberta_bias_tune")

    original_model_cfg = ModelConfig(
        hf=ModelHFConfig(
            attention_impl="flash",
            flash=ModelHFFlashConfig(kernel_overrides_path="overrides.json"),
        )
    )
    cfg = types.SimpleNamespace(
        model=original_model_cfg,
        data=DataConfig(
            packing=DataPackingConfig(
                enabled=True,
                max_seq_length=768,
                block_cross_document_attention=True,
            )
        ),
    )
    args = types.SimpleNamespace(
        config="config.yaml",
        branch="discriminator",
        sample_batches=1,
        out_dir=tmp_path,
    )

    class SamplingReached(Exception):
        """Stop the tuner after validating its sampling configuration."""

    def _sample_flash_batches(**kwargs: object) -> None:
        sampling_model_cfg = kwargs["model_cfg"]
        assert isinstance(sampling_model_cfg, ModelConfig)
        assert sampling_model_cfg.hf.flash.docblock_bias_seq_len == 768
        assert sampling_model_cfg.hf.flash.kernel_overrides_path == "overrides.json"
        assert original_model_cfg.hf.flash.docblock_bias_seq_len is None
        raise SamplingReached

    monkeypatch.setattr(tune_mod, "_parse_args", lambda: args)
    monkeypatch.setattr(tune_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tune_mod.bench, "resolve_out_dir", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(
        tune_mod.bench,
        "load_tool_config_and_loader",
        lambda *args, **kwargs: (cfg, None, object(), object()),
    )
    monkeypatch.setattr(
        tune_mod.bench,
        "build_branch_backbone_config",
        lambda **kwargs: (object(), 64, 768),
    )
    monkeypatch.setattr(tune_mod.bench, "sample_flash_batches", _sample_flash_batches)

    with pytest.raises(SamplingReached):
        tune_mod.main()
