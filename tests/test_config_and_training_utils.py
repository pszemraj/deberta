from __future__ import annotations

import json
import shlex
import sys
import types
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
    _normalize_sdpa_kernel,
    _normalize_torch_compile_mode,
    normalize_mixed_precision,
    validate_data_config,
    validate_model_config,
    validate_train_config,
    validate_training_workflow_options,
)
from deberta.export_cli import _build_export_parser
from deberta.modeling.builder import build_backbone_configs
from deberta.modeling.rtd import attention_mask_to_active_tokens, compute_generator_loss_term
from deberta.training.pretrain import (
    _build_optimizer,
    _build_training_collator,
    _count_rtd_tokens_for_batch,
    _export_discriminator_hf,
    _finalize_window_metric_loss,
    _find_latest_checkpoint,
    _load_checkpoint_data_progress,
    _persist_or_validate_run_configs,
    _prepare_output_dir,
    _resolve_resume_checkpoint,
    _save_checkpoint_data_progress,
    _save_training_checkpoint,
    _scale_loss_for_backward,
    _should_clip_gradients,
    _should_force_legacy_tf32_for_compile,
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


def test_model_config_defaults_dropouts_to_zero():
    cfg = ModelConfig()
    assert cfg.hidden_dropout_prob == pytest.approx(0.0)
    assert cfg.attention_probs_dropout_prob == pytest.approx(0.0)


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


def test_count_rtd_tokens_for_batch_keeps_masked_positions_active_for_discriminator():
    batch = {
        "input_ids": torch.tensor([[1, 3, 11, 2, 0]], dtype=torch.long),
        "labels": torch.tensor([[-100, 99, -100, -100, -100]], dtype=torch.long),
    }
    gen_count, disc_count = _count_rtd_tokens_for_batch(
        batch,
        special_token_ids=(0, 1, 2, 3),
        pad_token_id=0,
    )
    assert gen_count == pytest.approx(1.0)
    assert disc_count == pytest.approx(2.0)


def test_compute_disc_active_mask_preserves_masked_non_special_tokens():
    from deberta.training.pretrain import _compute_disc_active_mask

    mask = _compute_disc_active_mask(
        input_ids=torch.tensor([[1, 11, 2, 13, 0]], dtype=torch.long),
        labels=torch.tensor([[-100, 99, -100, 77, -100]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
        special_token_ids=(1, 2, 0),
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


def test_validate_model_config_rejects_derived_generator_knobs_with_explicit_generator_source():
    cfg = ModelConfig(
        backbone_type="rope",
        generator_model_name_or_path="microsoft/deberta-v3-small",
        generator_hidden_size=256,
    )
    with pytest.raises(ValueError, match="only used when deriving generator config"):
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


def test_build_backbone_configs_hf_respects_explicit_zero_dropout_overrides(
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
            self.hidden_dropout_prob = 0.1
            self.attention_probs_dropout_prob = 0.2

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = types.SimpleNamespace(from_pretrained=lambda _src: _FakeHFConfig())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    class _Tokenizer:
        pad_token_id = 0
        cls_token_id = 1
        sep_token_id = 2
        mask_token_id = 3

        def __len__(self) -> int:
            return 128

    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    disc_cfg, gen_cfg = build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_Tokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.hidden_dropout_prob == pytest.approx(0.0)
    assert gen_cfg.hidden_dropout_prob == pytest.approx(0.0)
    assert disc_cfg.attention_probs_dropout_prob == pytest.approx(0.0)
    assert gen_cfg.attention_probs_dropout_prob == pytest.approx(0.0)


def test_build_backbone_configs_hf_preserves_dropout_when_set_to_none(
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
            self.hidden_dropout_prob = 0.1
            self.attention_probs_dropout_prob = 0.2

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = types.SimpleNamespace(from_pretrained=lambda _src: _FakeHFConfig())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    class _Tokenizer:
        pad_token_id = 0
        cls_token_id = 1
        sep_token_id = 2
        mask_token_id = 3

        def __len__(self) -> int:
            return 128

    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=True,
        hidden_dropout_prob=None,
        attention_probs_dropout_prob=None,
    )
    disc_cfg, gen_cfg = build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=_Tokenizer(),
        max_position_embeddings=128,
    )

    assert disc_cfg.hidden_dropout_prob == pytest.approx(0.1)
    assert gen_cfg.hidden_dropout_prob == pytest.approx(0.1)
    assert disc_cfg.attention_probs_dropout_prob == pytest.approx(0.2)
    assert gen_cfg.attention_probs_dropout_prob == pytest.approx(0.2)


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
