import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from _config_factories import (
    make_logging_config,
    make_model_config,
    make_optim_config,
    make_train_config,
)
from _fakes import (
    BackboneConfigStub,
    BackboneOutputStub,
    DummyTokenizer,
    EmbeddingsStub,
    FakeAccelerator,
    TinyRTDLikeModel,
    fake_torch_compile,
)

from deberta.config import (
    _normalize_hf_attention_kernel,
    _normalize_sdpa_kernel,
    _normalize_torch_compile_backend,
    _normalize_torch_compile_mode,
    _normalize_torch_compile_scope,
    _normalize_wandb_watch,
    normalize_mixed_precision,
    resolve_effective_mixed_precision,
    validate_logging_config,
    validate_train_config,
)
from deberta.modeling.mask_utils import attention_mask_to_active_tokens, normalize_keep_mask
from deberta.training.checkpointing import _resolve_data_resume_policy
from deberta.training.compile import (
    _compile_backbones_for_scope,
    _resolve_compile_scope,
    _stabilize_compile_attention_mask,
)
from deberta.training.export_helpers import _export_discriminator_hf_subprocess
from deberta.training.loop_utils import (
    _count_input_tokens_for_batch,
    _count_rtd_tokens_for_batch,
    _finalize_window_metric_loss,
    _resolve_window_token_denominators,
    _scale_loss_for_backward,
    _should_clip_gradients,
    _token_weighted_micro_objective,
)
from deberta.training.runtime import (
    _build_optimizer,
    _build_scheduler,
    _build_training_collator,
    _cycle_dataloader,
)
from deberta.training.steps import (
    _any_rank_flag_true,
    _sync_discriminator_embeddings_if_available,
)


class _TinyBackbone(torch.nn.Module):
    def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embeddings = EmbeddingsStub(vocab_size, hidden_size)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> BackboneOutputStub:
        del attention_mask, token_type_ids, return_dict
        hidden = self.embeddings.word_embeddings(input_ids)
        return BackboneOutputStub(last_hidden_state=hidden)


def _tiny_rtd_config() -> BackboneConfigStub:
    return BackboneConfigStub(
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


def test_build_linear_scheduler_reaches_base_lr_before_decay() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = _build_scheduler(
        optimizer,
        train_cfg=make_train_config(max_steps=4),
        optim_cfg=make_optim_config(
            scheduler={"type": "linear", "warmup_steps": 1},
        ),
    )

    assert scheduler.get_last_lr() == pytest.approx([0.0])
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([0.1])
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr()[0] < 0.1


def test_move_batch_to_device_moves_flash_metadata_but_keeps_scalars_on_cpu() -> None:
    from deberta.modeling.mask_utils import FlashBatchMeta
    from deberta.training.steps import _move_batch_to_device

    batch = {
        "_flash_meta": FlashBatchMeta(
            seq_lengths=torch.tensor([2], dtype=torch.int32),
            active_tokens_scalar=torch.tensor(2, dtype=torch.int32),
        ),
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "ordinary_scalar": torch.tensor(7, dtype=torch.int32),
    }

    moved = _move_batch_to_device(batch, torch.device("meta"))

    assert moved["_flash_meta"].seq_lengths.device.type == "meta"
    assert moved["_flash_meta"].active_tokens_scalar.device.type == "cpu"
    assert moved["input_ids"].device.type == "meta"
    assert moved["ordinary_scalar"].device.type == "meta"


@pytest.mark.parametrize(("start_epoch", "count"), [(0, 3), (7, 2)])
def test_cycle_dataloader_advances_dataset_epoch_each_pass(start_epoch: int, count: int):
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
    it = _cycle_dataloader(dl, start_epoch=start_epoch)
    observed = [int(next(it)["epoch"].item()) for _ in range(count)]
    expected = list(range(start_epoch, start_epoch + count))
    assert observed == expected
    assert ds.seen_epochs[:count] == expected


def test_cycle_dataloader_rejects_empty_epochs() -> None:
    loader = torch.utils.data.DataLoader(
        [torch.tensor(1)],
        batch_size=2,
        drop_last=True,
    )

    with pytest.raises(RuntimeError, match="produced zero batches"):
        next(_cycle_dataloader(loader))


def test_resolve_data_resume_policy_auto_replays_when_small():
    cfg = make_train_config(
        checkpoint={"resume_data_strategy": "auto", "resume_replay_max_micro_batches": 100}
    )
    start_epoch, do_replay, reason = _resolve_data_resume_policy(
        train_cfg=cfg,
        consumed_micro_batches=42,
        global_step=9,
    )
    assert start_epoch == 0
    assert do_replay is True
    assert "replay" in reason


def test_resolve_data_resume_policy_auto_restarts_epoch_when_large():
    cfg = make_train_config(
        checkpoint={"resume_data_strategy": "auto", "resume_replay_max_micro_batches": 10}
    )
    start_epoch, do_replay, reason = _resolve_data_resume_policy(
        train_cfg=cfg,
        consumed_micro_batches=42,
        global_step=9,
    )
    assert start_epoch == 9
    assert do_replay is False
    assert "restart_epoch" in reason


def test_resolve_data_resume_policy_respects_explicit_strategy():
    replay_cfg = make_train_config(
        checkpoint={"resume_data_strategy": "replay", "resume_replay_max_micro_batches": 0}
    )
    start_epoch_replay, do_replay_replay, _ = _resolve_data_resume_policy(
        train_cfg=replay_cfg,
        consumed_micro_batches=999,
        global_step=3,
    )
    assert start_epoch_replay == 0
    assert do_replay_replay is True

    restart_cfg = make_train_config(
        checkpoint={"resume_data_strategy": "restart_epoch", "resume_replay_max_micro_batches": 1_000_000}
    )
    start_epoch_restart, do_replay_restart, _ = _resolve_data_resume_policy(
        train_cfg=restart_cfg,
        consumed_micro_batches=12,
        global_step=17,
    )
    assert start_epoch_restart == 17
    assert do_replay_restart is False


def test_build_forbidden_token_mask_rejects_all_forbidden_vocab():
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    with pytest.raises(ValueError, match="excludes all vocabulary ids"):
        DebertaV3RTDPretrainer._build_forbidden_token_mask(
            vocab_size=8,
            forbidden_ids=set(range(8)),
        )


def test_pretrainer_additional_forbidden_token_ids_extend_config_special_set() -> None:
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    cfg = _tiny_rtd_config()
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=_TinyBackbone(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size),
        generator_backbone=_TinyBackbone(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size),
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="none",
        additional_forbidden_token_ids=[7, 15, -1, 99],
    )

    expected = {0, 1, 2, 3, 7, 15}
    assert int(model._forbidden_sample_token_mask.numel()) == 32
    for tid in expected:
        assert bool(model._forbidden_sample_token_mask[tid].item())


def test_pretrainer_skips_discriminator_when_no_masked_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    cfg = _tiny_rtd_config()
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


@pytest.mark.parametrize("publish_fails", [False, True], ids=["replace", "rollback"])
def test_export_discriminator_hf_subprocess_uses_strict_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publish_fails: bool,
) -> None:
    calls: list[list[str]] = []
    checkpoint_dir = tmp_path / "run" / "checkpoint-1"
    checkpoint_dir.mkdir(parents=True)
    output_dir = tmp_path / "run" / "final_hf"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("old", encoding="utf-8")

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
        refresh_dir = Path(cmd[cmd.index("--output-dir") + 1])
        refresh_dir.mkdir()
        (refresh_dir / "new.txt").write_text("new", encoding="utf-8")
        return _Proc()

    import deberta.training.export_helpers as export_mod

    monkeypatch.setattr(export_mod.subprocess, "run", _fake_run)
    if publish_fails:
        original_replace = Path.replace

        def _fail_refresh_publish(path: Path, target: Path) -> Path:
            if path.name.startswith(".final_hf.refresh-"):
                raise OSError("refresh publish failed")
            return original_replace(path, target)

        monkeypatch.setattr(Path, "replace", _fail_refresh_publish)

    if publish_fails:
        with pytest.raises(OSError, match="refresh publish failed"):
            _export_discriminator_hf_subprocess(
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
            )
    else:
        _export_discriminator_hf_subprocess(
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
        )

    assert calls
    cmd = calls[-1]
    assert cmd[0] == sys.executable
    assert cmd[1:5] == ["-m", "deberta", "export", str(checkpoint_dir)]
    assert "--allow-partial-export" not in cmd
    assert Path(cmd[cmd.index("--output-dir") + 1]) != output_dir
    if publish_fails:
        assert (output_dir / "old.txt").read_text(encoding="utf-8") == "old"
        assert not (output_dir / "new.txt").exists()
    else:
        assert (output_dir / "new.txt").read_text(encoding="utf-8") == "new"
        assert not (output_dir / "old.txt").exists()
    assert not list(output_dir.parent.glob(".final_hf.*-*"))


def test_export_discriminator_hf_subprocess_raises_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run" / "final_hf"
    output_dir.mkdir(parents=True)
    (output_dir / "old.txt").write_text("old", encoding="utf-8")

    class _Proc:
        returncode = 7
        stdout = "strict load failed"

    def _fake_run(
        cmd: list[str],
        *,
        stdout: Any,
        stderr: Any,
        text: bool,
        check: bool,
    ) -> _Proc:
        del stdout, stderr, text, check
        refresh_dir = Path(cmd[cmd.index("--output-dir") + 1])
        refresh_dir.mkdir()
        (refresh_dir / "partial.txt").write_text("partial", encoding="utf-8")
        return _Proc()

    import deberta.training.export_helpers as export_mod

    monkeypatch.setattr(export_mod.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="exit=7") as exc_info:
        _export_discriminator_hf_subprocess(
            checkpoint_dir=tmp_path / "run" / "checkpoint-1",
            output_dir=output_dir,
        )

    assert "strict load failed" in str(exc_info.value)
    assert (output_dir / "old.txt").read_text(encoding="utf-8") == "old"
    assert not list(output_dir.parent.glob(".final_hf.*-*"))


@pytest.mark.parametrize(
    ("discriminator_lr", "expected_lrs"),
    [
        (None, {1.0e-3, 5.0e-4}),
        (2.0e-4, {5.0e-4, 2.0e-4}),
    ],
    ids=["generator_override_only", "generator_and_discriminator_overrides"],
)
def test_build_optimizer_supports_branch_specific_lrs(
    discriminator_lr: float | None, expected_lrs: set[float]
):
    model = TinyRTDLikeModel()
    lr = {"base": 1.0e-3, "generator": 5.0e-4}
    if discriminator_lr is not None:
        lr["discriminator"] = float(discriminator_lr)
    cfg = make_optim_config(lr=lr, weight_decay=0.1)
    opt = _build_optimizer(model, cfg)

    lrs = {float(g["lr"]) for g in opt.param_groups}
    assert lrs == expected_lrs

    # We should have both decay and no-decay groups present.
    wds = {float(g["weight_decay"]) for g in opt.param_groups}
    assert wds == {0.0, 0.1}


def test_build_optimizer_keeps_fused_in_bf16_mode(monkeypatch: pytest.MonkeyPatch):
    import deberta.training.runtime as runtime_mod

    model = TinyRTDLikeModel()
    cfg = make_optim_config()

    monkeypatch.setattr(runtime_mod, "_maybe_fused_adamw_kwargs", lambda: {"fused": True})
    opt = runtime_mod._build_optimizer(
        model,
        cfg,
        mixed_precision="bf16",
    )

    assert bool(opt.defaults.get("fused", False)) is True


def test_build_optimizer_raises_adam_epsilon_floor_for_bf16():
    model = TinyRTDLikeModel()
    cfg = make_optim_config(adam={"epsilon": 1e-8})

    opt = _build_optimizer(model, cfg, mixed_precision="bf16")
    assert float(opt.defaults["eps"]) == pytest.approx(1e-6)

    opt_fp32 = _build_optimizer(model, cfg, mixed_precision="no")
    assert float(opt_fp32.defaults["eps"]) == pytest.approx(1e-8)


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
    cfg = make_train_config(mixed_precision=raw)
    validate_train_config(cfg)
    assert cfg.mixed_precision == expected


def test_validate_train_config_rejects_invalid_mixed_precision():
    cfg = make_train_config(mixed_precision="fp16")
    with pytest.raises(ValueError, match="train.mixed_precision must be one of: bf16\\|no"):
        validate_train_config(cfg)


def test_validate_train_config_normalizes_compile_scope_and_backend_aliases():
    cfg = make_train_config(compile={"enabled": True, "scope": "generator_ffn", "backend": "aot-eager"})
    validate_train_config(cfg)
    assert cfg.compile.scope == "gen_ffn"
    assert cfg.compile.backend == "aot_eager"


def test_validate_train_config_normalizes_wandb_watch_aliases():
    cfg = make_logging_config(wandb={"enabled": True, "watch": "weights", "watch_log_freq": 25})
    validate_logging_config(cfg)
    assert cfg.wandb.watch == "parameters"
    assert cfg.wandb.watch_log_freq == 25


def test_validate_train_config_rejects_invalid_wandb_watch_mode():
    cfg = make_logging_config(wandb={"watch": "histogram"})
    with pytest.raises(ValueError, match="logging.wandb.watch must be one of"):
        validate_logging_config(cfg)


def test_validate_train_config_rejects_non_positive_wandb_watch_log_freq():
    cfg = make_logging_config(wandb={"watch_log_freq": 0})
    with pytest.raises(ValueError, match="logging.wandb.watch_log_freq must be >= 1"):
        validate_logging_config(cfg)


def test_validate_train_config_accepts_resume_data_strategy_values():
    cfg = make_train_config(
        checkpoint={"resume_data_strategy": "restart_epoch", "resume_replay_max_micro_batches": 123}
    )
    validate_train_config(cfg)
    assert cfg.checkpoint.resume_data_strategy == "restart_epoch"
    assert cfg.checkpoint.resume_replay_max_micro_batches == 123


def test_validate_train_config_rejects_invalid_resume_data_strategy():
    cfg = make_train_config(checkpoint={"resume_data_strategy": "fast"})
    with pytest.raises(ValueError, match="train.checkpoint.resume_data_strategy must be one of"):
        validate_train_config(cfg)


def test_validate_train_config_rejects_negative_resume_replay_threshold():
    cfg = make_train_config(checkpoint={"resume_replay_max_micro_batches": -1})
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


@pytest.mark.parametrize(
    ("gen_loss_weight", "disc_loss_weight", "expected"),
    [
        (0.0, 2.0, 6.0),
        (3.0, 0.0, 6.0),
    ],
)
def test_token_weighted_micro_objective_ignores_disabled_nonfinite_branch(
    gen_loss_weight: float,
    disc_loss_weight: float,
    expected: float,
) -> None:
    gen_loss = torch.tensor(float("nan") if gen_loss_weight == 0.0 else 2.0)
    disc_loss = torch.tensor(float("nan") if disc_loss_weight == 0.0 else 3.0)

    objective = _token_weighted_micro_objective(
        gen_loss=gen_loss,
        disc_loss=disc_loss,
        gen_count=1.0,
        disc_count=1.0,
        gen_window_tokens_per_rank=1.0,
        disc_window_tokens_per_rank=1.0,
        gen_loss_weight=gen_loss_weight,
        disc_loss_weight=disc_loss_weight,
    )

    torch.testing.assert_close(objective, torch.tensor(expected))


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
    train_cfg = make_train_config(objective={"mlm_probability": 0.2, "mlm_max_ngram": 2})
    collator = _build_training_collator(
        tokenizer=tokenizer,
        train_cfg=train_cfg,
        packed_sequences=True,
        block_cross_document_attention=True,
    )
    assert collator._packed_sequences is True
    assert collator._block_cross_document_attention is True
    assert collator._emit_flash_metadata is False


def test_should_clip_gradients_for_positive_threshold():
    assert _should_clip_gradients(max_grad_norm=None) is False
    assert _should_clip_gradients(max_grad_norm=0.0) is False
    assert _should_clip_gradients(max_grad_norm=-1.0) is False
    assert _should_clip_gradients(max_grad_norm=1.0) is True


def test_clip_gradients_rechecks_finiteness_after_clipping(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.training.entrypoint as entrypoint_mod

    checks = iter((False, True))
    monkeypatch.setattr(
        entrypoint_mod,
        "_any_rank_flag_true",
        lambda **_kwargs: next(checks),
    )
    monkeypatch.setattr(entrypoint_mod, "_global_grad_l2_norm", lambda _model: 2.0)
    accelerator = FakeAccelerator()

    reason, grad_norm = entrypoint_mod._clip_gradients_and_find_nonfinite(
        accelerator=accelerator,
        model=torch.nn.Linear(2, 2),
        max_grad_norm=1.0,
    )

    assert reason == "grad_norm_post_clip"
    assert grad_norm == pytest.approx(2.0)
    assert accelerator.calls["clip_grad_norm_"] == [1.0]


def test_any_rank_flag_true_uses_reduced_flag():
    accel = FakeAccelerator(num_processes=2)

    def _reduce(_self: FakeAccelerator, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        assert reduction == "sum"
        return tensor + 1

    accel.reduce = types.MethodType(_reduce, accel)  # type: ignore[method-assign]
    assert _any_rank_flag_true(accelerator=accel, flag=False) is True
    assert _any_rank_flag_true(accelerator=accel, flag=True) is True


def test_any_rank_flag_true_single_process_uses_local_flag():
    accel = FakeAccelerator(num_processes=1)

    def _reduce(_self: FakeAccelerator, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        del tensor, reduction
        raise AssertionError("reduce() should not be called in single-process mode")

    accel.reduce = types.MethodType(_reduce, accel)  # type: ignore[method-assign]
    assert _any_rank_flag_true(accelerator=accel, flag=False) is False
    assert _any_rank_flag_true(accelerator=accel, flag=True) is True


def test_any_rank_flag_true_propagates_reduce_errors_on_multi_process():
    accel = FakeAccelerator(num_processes=2)

    def _reduce(_self: FakeAccelerator, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        del tensor, reduction
        raise RuntimeError("collective failed")

    accel.reduce = types.MethodType(_reduce, accel)  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="collective failed"):
        _any_rank_flag_true(accelerator=accel, flag=False)


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
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool),
        "labels": torch.tensor([[-100, 99, -100, -100, -100]], dtype=torch.long),
    }
    gen_count, disc_count = _count_rtd_tokens_for_batch(batch)
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


def test_attention_mask_to_active_tokens_treats_pairwise_mask_as_authoritative():
    input_ids = torch.tensor([[11, 12, 13, 0]], dtype=torch.long)
    # Row 2 is a masked non-pad filler. Row 3 is an active pad-valued token.
    # Numeric token identity must not override the pairwise mask diagonal.
    pair_keep = torch.tensor(
        [
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
            ]
        ],
        dtype=torch.bool,
    )
    active = attention_mask_to_active_tokens(
        input_ids=input_ids,
        attention_mask=pair_keep,
    )
    expected = torch.tensor([[True, True, False, True]], dtype=torch.bool)
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
    )
    expected = torch.tensor([[True, True, False, False]], dtype=torch.bool)
    assert torch.equal(active, expected)


def _weight_decay_for_param(opt: torch.optim.Optimizer, param: torch.nn.Parameter) -> float:
    for group in opt.param_groups:
        for grouped_param in group["params"]:
            if grouped_param is param:
                return float(group["weight_decay"])
    raise AssertionError("Parameter missing from optimizer groups")


def test_build_optimizer_marks_scalar_params_as_no_decay():
    optim_cfg = make_optim_config()

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
    opt = _build_optimizer(model, optim_cfg)

    assert _weight_decay_for_param(opt, model.generator.alpha) == pytest.approx(0.0)
    assert _weight_decay_for_param(opt, model.discriminator.alpha) == pytest.approx(0.0)
    assert _weight_decay_for_param(opt, model.discriminator_norm.weight) == pytest.approx(0.0)
    assert _weight_decay_for_param(opt, model.generator.weight) == pytest.approx(optim_cfg.weight_decay)


def test_build_optimizer_applies_decay_to_high_rank_bias_parameters():
    optim_cfg = make_optim_config()

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
    opt = _build_optimizer(model, optim_cfg)

    assert _weight_decay_for_param(opt, model.generator_bias.bias) == pytest.approx(optim_cfg.weight_decay)


def test_normalize_mixed_precision_accepts_bool_and_synonyms():
    assert normalize_mixed_precision("bf16") == "bf16"
    assert normalize_mixed_precision("none") == "no"
    assert normalize_mixed_precision(False) == "no"
    assert normalize_mixed_precision(True) == "bf16"

    with pytest.raises(ValueError, match="train.mixed_precision must be one of: bf16\\|no"):
        normalize_mixed_precision("fp16")


def test_resolve_effective_mixed_precision_errors_for_bf16_preflight_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    import deberta.training.compile as compile_mod

    monkeypatch.setattr(compile_mod, "_bf16_runtime_sanity_check", lambda: False)
    with pytest.raises(RuntimeError, match="Set train.mixed_precision=no explicitly"):
        resolve_effective_mixed_precision("bf16", bf16_sanity_check=compile_mod._bf16_runtime_sanity_check)

    assert (
        resolve_effective_mixed_precision("no", bf16_sanity_check=compile_mod._bf16_runtime_sanity_check)
        == "no"
    )


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


def test_compile_backbones_for_scope_compiles_generic_backbone_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_compile, compile_calls = fake_torch_compile()

    class _Backbone(torch.nn.Module):
        def __init__(self, label: str) -> None:
            super().__init__()
            self.label = str(label)

        def forward(self, **_: Any) -> tuple[str]:
            return (self.label,)

    class _Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = _Backbone("generator")
            self.discriminator = _Backbone("discriminator")

    monkeypatch.setattr(torch, "compile", _fake_compile)

    wrapper = _Wrapper()
    targets = _compile_backbones_for_scope(
        unwrapped_model=wrapper,
        compile_scope="backbones",
        compile_kwargs={"mode": "default", "backend": "inductor", "dynamic": False},
    )

    assert targets == ["generator", "discriminator"]
    assert len(compile_calls) == 2
    for _, kwargs in compile_calls:
        assert kwargs == {"mode": "default", "backend": "inductor", "dynamic": False}

    assert wrapper.generator() == ("generator",)


def test_stable_backbone_compile_dispatch_preserves_flash_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deberta.modeling.mask_utils import FlashBatchMeta
    from deberta.training.compile import _install_stable_backbone_compile_dispatch

    _fake_compile, compile_calls = fake_torch_compile()
    monkeypatch.setattr(torch, "compile", _fake_compile)

    class _PolicyOwner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flash_kernel_policy_path = "/policy-b.json"
            self.flash_kernel_policy_key = "policy-b-key"

    class _StableBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_owner = _PolicyOwner()

        def _resolve_forward_options(self, **kwargs):
            return False, bool(kwargs["output_hidden_states"]), True

        def _forward_dense_hs0(self, **kwargs):
            flash_meta = kwargs.get("flash_meta")
            return (
                "dense_hs0",
                flash_meta.route_hint if flash_meta is not None else None,
                flash_meta.seq_bucket if flash_meta is not None else None,
                flash_meta.kernel_policy_path if flash_meta is not None else None,
                flash_meta.kernel_policy_key if flash_meta is not None else None,
            )

        def _forward_dense_hs1(self, **kwargs):
            flash_meta = kwargs.get("flash_meta")
            return (
                "dense_hs1",
                flash_meta.route_hint if flash_meta is not None else None,
                flash_meta.seq_bucket if flash_meta is not None else None,
                flash_meta.kernel_policy_path if flash_meta is not None else None,
                flash_meta.kernel_policy_key if flash_meta is not None else None,
            )

        def _forward_masked_hs0(self, **kwargs):
            flash_meta = kwargs["flash_meta"]
            return (
                "masked_hs0",
                flash_meta.route_hint,
                flash_meta.seq_bucket,
                flash_meta.kernel_policy_path,
                flash_meta.kernel_policy_key,
            )

        def _forward_masked_hs1(self, **kwargs):
            flash_meta = kwargs["flash_meta"]
            return (
                "masked_hs1",
                flash_meta.route_hint,
                flash_meta.seq_bucket,
                flash_meta.kernel_policy_path,
                flash_meta.kernel_policy_key,
            )

        def _forward_resolved(self, **_kwargs):
            raise AssertionError("standard training options must use a compiled entrypoint")

    backbone = _StableBackbone()
    targets: list[str] = []
    assert _install_stable_backbone_compile_dispatch(
        module=backbone,
        compile_kwargs={"mode": "default", "backend": "inductor", "dynamic": False},
        target="backbone",
        compiled_targets=targets,
    )

    mask = torch.ones((1, 4), dtype=torch.bool)
    for hidden_states in (False, True):
        for route in ("dense", "local_bias"):
            result = backbone(
                output_hidden_states=hidden_states,
                flash_meta=FlashBatchMeta(
                    route_hint=route,
                    seq_bucket="sparse",
                    kernel_policy_path="/policy-b.json",
                    kernel_policy_key="policy-b-key",
                ),
            )
            assert result == (
                f"dense_hs{int(hidden_states)}",
                route,
                "sparse",
                "/policy-b.json",
                "policy-b-key",
            )

    for route in ("fixed", "varlen", "docblock", "docblock_bias"):
        for hidden_states in (False, True):
            result = backbone(
                attention_mask=mask,
                output_hidden_states=hidden_states,
                flash_meta=FlashBatchMeta(
                    route_hint=route,
                    seq_bucket="sparse",
                    kernel_policy_path="/policy-b.json",
                    kernel_policy_key="policy-b-key",
                ),
            )
            assert result == (
                f"masked_hs{int(hidden_states)}",
                route,
                "sparse",
                "/policy-b.json",
                "policy-b-key",
            )

    with pytest.raises(RuntimeError, match="different kernel policy"):
        backbone(
            flash_meta=FlashBatchMeta(
                route_hint="local_bias",
                kernel_policy_path="/policy-a.json",
                kernel_policy_key="policy-a-key",
            )
        )

    assert len(compile_calls) == 16
    assert len(targets) == 16


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
    )
    assert "attention_mask" not in out1

    # RoPE without doc-blocking: no mask injection.
    batch2 = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    out2 = _stabilize_compile_attention_mask(
        batch=dict(batch2),
        compile_enabled=True,
        compile_scope="backbones",
        backbone_type="rope",
    )
    assert "attention_mask" not in out2

    # Compile disabled: no mask injection regardless.
    batch3 = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    out3 = _stabilize_compile_attention_mask(
        batch=dict(batch3),
        compile_enabled=False,
        compile_scope="backbones",
        backbone_type="rope",
    )
    assert "attention_mask" not in out3


def test_resolve_compile_scope_auto_prefers_backbones_except_rope_doc_blocking():
    scope, reason = _resolve_compile_scope(
        requested_scope="auto",
        model_cfg=make_model_config(backbone_type="hf_deberta_v2"),
    )
    assert scope == "backbones"
    assert reason is None

    scope, reason = _resolve_compile_scope(
        requested_scope="auto",
        model_cfg=make_model_config(backbone_type="rope"),
    )
    assert scope == "backbones"
    assert reason is None

    # RoPE + doc-blocking auto-downgrades to FFN to avoid mask shape churn.
    scope, reason = _resolve_compile_scope(
        requested_scope="auto",
        model_cfg=make_model_config(backbone_type="rope"),
        block_cross_document_attention=True,
    )
    assert scope == "ffn"
    assert reason is not None

    scope, reason = _resolve_compile_scope(
        requested_scope="backbones",
        model_cfg=make_model_config(backbone_type="hf_deberta_v2"),
    )
    assert scope == "backbones"
    assert reason is None
