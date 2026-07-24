"""Tests for FlashDeBERTa tooling behavior."""

from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest
import torch

from deberta.config import DataConfig, DataPackingConfig, ModelConfig, ModelHFConfig, ModelHFFlashConfig


def test_parity_cotangent_is_deterministic_nonuniform_and_masks_inactive_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    parity_mod = importlib.import_module("flashdeberta_parity_test")
    loss_mask = torch.tensor([[True, True, False], [True, False, False]])

    first = parity_mod._build_backward_cotangent(
        shape=(2, 3, 4),
        device=torch.device("cpu"),
        loss_mask=loss_mask,
    )
    second = parity_mod._build_backward_cotangent(
        shape=(2, 3, 4),
        device=torch.device("cpu"),
        loss_mask=loss_mask,
    )

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(first[~loss_mask]).item() == 0
    active = first[loss_mask]
    assert torch.unique(active).numel() > 1
    assert bool(active.lt(0).any())
    assert bool(active.gt(0).any())


def test_bias_tuner_samples_ragged_docblock_before_local_dense_replay(
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
        candidate=[],
        warmup=0,
        steps=1,
    )

    class SweepReached(BaseException):
        """Stop after validating the complete dense-replay handoff."""

    doc_ids = torch.tensor([[1, 1, 2, 0]], dtype=torch.long)

    def _sample_flash_batches(**kwargs: object) -> list[object]:
        sampling_model_cfg = kwargs["model_cfg"]
        assert isinstance(sampling_model_cfg, ModelConfig)
        assert sampling_model_cfg is original_model_cfg
        assert sampling_model_cfg.hf.flash.kernel_overrides_path == "overrides.json"
        assert kwargs["route_hints"] == {"docblock"}
        return [
            tune_mod.bench.BatchSample(
                index=0,
                input_ids=torch.tensor([[1, 10, 11, 0]], dtype=torch.long),
                attention_mask=doc_ids.ne(0),
                flash_meta=tune_mod.bench.FlashBatchMeta(
                    doc_ids=doc_ids,
                    active_tokens_scalar=torch.tensor(3, dtype=torch.int32),
                    route_hint="docblock",
                ),
                active_tokens=3,
                slot_tokens=4,
                batch_size=1,
                seq_len=4,
                head_dim=64,
                att_span=768,
                device_capability="sm_120",
            )
        ]

    class _FlashLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flash_kernel_policy_path = str((tmp_path / "original-policy.json").resolve())
            self.flash_kernel_policy_key = "original-policy-key"

    model = torch.nn.Sequential(_FlashLayer())
    original_policy_path = model[0].flash_kernel_policy_path
    original_policy_key = model[0].flash_kernel_policy_key

    def _run_timed_candidate(**kwargs: object) -> None:
        samples = kwargs["samples"]
        assert isinstance(samples, list) and len(samples) == 1
        sample = samples[0]
        expected_mask = tune_mod.build_doc_block_mask(doc_ids)
        assert torch.equal(sample.attention_mask, expected_mask)
        expected_density = float(expected_mask.to(dtype=torch.int32).sum().item()) / 16.0
        assert sample.pair_density == pytest.approx(expected_density)
        metadata = kwargs["meta_fn"](sample)
        assert metadata.route_hint == "docblock_bias"
        assert metadata.kernel_policy_path == ""
        assert metadata.kernel_policy_key == ""
        assert model[0].flash_kernel_policy_path == ""
        assert model[0].flash_kernel_policy_key == ""
        raise SweepReached

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
    monkeypatch.setattr(tune_mod.bench, "build_bf16_backbone", lambda *args, **kwargs: model)
    monkeypatch.setattr(tune_mod.bench, "run_timed_candidate", _run_timed_candidate)

    with pytest.raises(SweepReached):
        tune_mod.main()
    assert model[0].flash_kernel_policy_path == original_policy_path
    assert model[0].flash_kernel_policy_key == original_policy_key
