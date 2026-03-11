# ruff: noqa: F403,F405
from _config_and_training_shared_imports import *


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


def test_normalize_resume_consumed_micro_batches_clamps_legacy_partial_window():
    consumed, reason = _normalize_resume_consumed_micro_batches(
        consumed_micro_batches=35,
        global_step=11,
        gradient_accumulation_steps=3,
    )
    assert consumed == 33
    assert reason is not None
    assert "clamped_legacy_partial_accumulation_delta" in reason


def test_normalize_resume_consumed_micro_batches_keeps_non_legacy_mismatch():
    consumed, reason = _normalize_resume_consumed_micro_batches(
        consumed_micro_batches=100,
        global_step=11,
        gradient_accumulation_steps=3,
    )
    assert consumed == 100
    assert reason is None


def test_build_forbidden_token_mask_rejects_all_forbidden_vocab():
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    with pytest.raises(ValueError, match="excludes all vocabulary ids"):
        DebertaV3RTDPretrainer._build_forbidden_token_mask(
            vocab_size=8,
            forbidden_ids=set(range(8)),
        )


def test_pretrainer_additional_forbidden_token_ids_extend_config_special_set() -> None:
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    class _TinyBackbone(torch.nn.Module):
        def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
            super().__init__()
            self.embeddings = types.SimpleNamespace(
                word_embeddings=torch.nn.Embedding(vocab_size, hidden_size),
            )

        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            token_type_ids: torch.Tensor | None = None,
            return_dict: bool = True,
        ) -> Any:
            del attention_mask, token_type_ids, return_dict
            hidden = self.embeddings.word_embeddings(input_ids)
            return types.SimpleNamespace(last_hidden_state=hidden)

    cfg = types.SimpleNamespace(
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
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=_TinyBackbone(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size),
        generator_backbone=_TinyBackbone(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size),
        disc_config=cfg,
        gen_config=cfg,
        embedding_sharing="none",
        additional_forbidden_token_ids=[7, 15, -1, 99],
    )

    expected = {0, 1, 2, 3, 7, 15}
    assert expected.issubset(model._forbidden_sample_token_ids)
    assert int(model._forbidden_sample_token_mask.numel()) == 32
    for tid in expected:
        assert bool(model._forbidden_sample_token_mask[tid].item())


def test_pretrainer_skips_discriminator_when_no_masked_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    class _TinyBackbone(torch.nn.Module):
        def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
            super().__init__()
            self.embeddings = types.SimpleNamespace(
                word_embeddings=torch.nn.Embedding(vocab_size, hidden_size),
            )

        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            token_type_ids: torch.Tensor | None = None,
            return_dict: bool = True,
        ) -> Any:
            del attention_mask, token_type_ids, return_dict
            hidden = self.embeddings.word_embeddings(input_ids)
            return types.SimpleNamespace(last_hidden_state=hidden)

    cfg = types.SimpleNamespace(
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


def test_export_discriminator_hf_subprocess_uses_allow_partial_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

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
        return _Proc()

    import deberta.training.export_helpers as export_mod

    monkeypatch.setattr(export_mod.subprocess, "run", _fake_run)

    _export_discriminator_hf_subprocess(
        checkpoint_dir=Path("runs/demo/checkpoint-1"),
        output_dir=Path("runs/demo/final_hf"),
    )

    assert calls
    cmd = calls[-1]
    assert cmd[0] == sys.executable
    assert cmd[1:5] == ["-m", "deberta", "export", "runs/demo/checkpoint-1"]
    assert "--allow-partial-export" in cmd


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
    cfg_kwargs: dict[str, float] = {
        "learning_rate": 1.0e-3,
        "generator_learning_rate": 5.0e-4,
        "weight_decay": 0.1,
    }
    if discriminator_lr is not None:
        cfg_kwargs["discriminator_learning_rate"] = float(discriminator_lr)
    cfg = TrainConfig(**cfg_kwargs)
    opt = _build_optimizer(model, cfg)

    lrs = {float(g["lr"]) for g in opt.param_groups}
    assert lrs == expected_lrs

    # We should have both decay and no-decay groups present.
    wds = {float(g["weight_decay"]) for g in opt.param_groups}
    assert wds == {0.0, 0.1}


def test_build_decoupled_optimizers_support_discriminator_specific_lr():
    model = TinyRTDLikeModel()
    cfg = TrainConfig(
        learning_rate=1.0e-3,
        generator_learning_rate=5.0e-4,
        discriminator_learning_rate=2.0e-4,
        weight_decay=0.1,
        mixed_precision="no",
    )
    gen_opt, disc_opt = _build_decoupled_optimizers(model, cfg, mixed_precision="no")

    assert gen_opt.param_groups
    assert disc_opt.param_groups
    for group in gen_opt.param_groups:
        assert float(group["lr"]) == pytest.approx(5.0e-4)
    for group in disc_opt.param_groups:
        assert float(group["lr"]) == pytest.approx(2.0e-4)


def test_build_optimizer_keeps_fused_in_bf16_mode(monkeypatch: pytest.MonkeyPatch):
    import deberta.training.runtime as runtime_mod

    model = TinyRTDLikeModel()
    cfg = TrainConfig()

    monkeypatch.setattr(runtime_mod, "_maybe_fused_adamw_kwargs", lambda: {"fused": True})
    opt = runtime_mod._build_optimizer(
        model,
        cfg,
        mixed_precision="bf16",
    )

    assert bool(opt.defaults.get("fused", False)) is True


def test_build_optimizer_raises_adam_epsilon_floor_for_bf16():
    model = TinyRTDLikeModel()
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
    assert model.hidden_dropout_prob == pytest.approx(0.0)
    assert model.attention_probs_dropout_prob == pytest.approx(0.0)
    assert model.hf_model_size == "base"
    assert model.ffn_type == "mlp"
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
    cfg = LoggingConfig(report_to="wandb", wandb_watch="weights", wandb_watch_log_freq=25)
    validate_logging_config(cfg)
    assert cfg.wandb.watch == "parameters"
    assert cfg.wandb.watch_log_freq == 25


def test_validate_train_config_rejects_invalid_wandb_watch_mode():
    cfg = LoggingConfig(wandb_watch="histogram")
    with pytest.raises(ValueError, match="logging.wandb.watch must be one of"):
        validate_logging_config(cfg)


def test_validate_train_config_rejects_non_positive_wandb_watch_log_freq():
    cfg = LoggingConfig(wandb_watch_log_freq=0)
    with pytest.raises(ValueError, match="logging.wandb.watch_log_freq must be >= 1"):
        validate_logging_config(cfg)


def test_validate_train_config_accepts_resume_data_strategy_values():
    cfg = TrainConfig(
        checkpoint={"resume_data_strategy": "restart_epoch", "resume_replay_max_micro_batches": 123}
    )
    validate_train_config(cfg)
    assert cfg.checkpoint.resume_data_strategy == "restart_epoch"
    assert cfg.checkpoint.resume_replay_max_micro_batches == 123


def test_validate_train_config_rejects_invalid_resume_data_strategy():
    cfg = TrainConfig(checkpoint={"resume_data_strategy": "fast"})
    with pytest.raises(ValueError, match="train.checkpoint.resume_data_strategy must be one of"):
        validate_train_config(cfg)


def test_validate_train_config_rejects_negative_resume_replay_threshold():
    cfg = TrainConfig(checkpoint={"resume_replay_max_micro_batches": -1})
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
    train_cfg = TrainConfig(mlm_probability=0.2, mlm_max_ngram=2)
    collator = _build_training_collator(
        tokenizer=tokenizer,
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
    accel = FakeAccelerator(num_processes=2)

    def _reduce(_self: FakeAccelerator, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        assert reduction == "sum"
        return tensor + 1

    accel.reduce = types.MethodType(_reduce, accel)  # type: ignore[method-assign]
    assert _has_nonfinite_grad_norm_any_rank(accelerator=accel, grad_norm=1.0) is True
    assert _has_nonfinite_grad_norm_any_rank(accelerator=accel, grad_norm=float("inf")) is True


def test_has_nonfinite_grad_norm_any_rank_single_process_uses_local_flag():
    accel = FakeAccelerator(num_processes=1)

    def _reduce(_self: FakeAccelerator, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        del tensor, reduction
        raise AssertionError("reduce() should not be called in single-process mode")

    accel.reduce = types.MethodType(_reduce, accel)  # type: ignore[method-assign]
    assert _has_nonfinite_grad_norm_any_rank(accelerator=accel, grad_norm=1.0) is False
    assert _has_nonfinite_grad_norm_any_rank(accelerator=accel, grad_norm=float("inf")) is True


def test_has_nonfinite_grad_norm_any_rank_propagates_reduce_errors_on_multi_process():
    accel = FakeAccelerator(num_processes=2)

    def _reduce(_self: FakeAccelerator, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        del tensor, reduction
        raise RuntimeError("collective failed")

    accel.reduce = types.MethodType(_reduce, accel)  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="collective failed"):
        _has_nonfinite_grad_norm_any_rank(accelerator=accel, grad_norm=1.0)


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
        "labels": torch.tensor([[-100, 99, -100, -100, -100]], dtype=torch.long),
    }
    gen_count, disc_count = _count_rtd_tokens_for_batch(
        batch,
        pad_token_id=0,
    )
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


def test_attention_mask_to_active_tokens_uses_diagonal_activity_with_pad_for_3d_masks():
    input_ids = torch.tensor([[11, 12, 13, 0]], dtype=torch.long)
    # Row 2 has diagonal=False but off-diagonal keep=True. Active-token recovery
    # must respect diagonal activity even when pad_token_id is available.
    # Row 3 is explicit pad and must remain inactive.
    pair_keep = torch.tensor(
        [
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        ],
        dtype=torch.bool,
    )
    active = attention_mask_to_active_tokens(
        input_ids=input_ids,
        attention_mask=pair_keep,
        pad_token_id=0,
    )
    expected = torch.tensor([[True, True, False, False]], dtype=torch.bool)
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
            pad_token_id=None,
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
        pad_token_id=0,
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

    assert _weight_decay_for_param(opt, model.generator.alpha) == pytest.approx(0.0)
    assert _weight_decay_for_param(opt, model.discriminator.alpha) == pytest.approx(0.0)
    assert _weight_decay_for_param(opt, model.discriminator_norm.weight) == pytest.approx(0.0)
    assert _weight_decay_for_param(opt, model.generator.weight) == pytest.approx(train_cfg.weight_decay)


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

    assert _weight_decay_for_param(opt, model.generator_bias.bias) == pytest.approx(train_cfg.weight_decay)


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


def test_resolve_compile_enabled_or_raise_errors_when_torch_compile_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    import deberta.training.compile as compile_mod

    monkeypatch.delattr(compile_mod.torch, "compile", raising=False)
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
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
    )
    assert scope == "backbones"
    assert reason is None

    scope, reason = _resolve_compile_scope(
        requested_scope="auto",
        model_cfg=ModelConfig(backbone_type="rope"),
    )
    assert scope == "backbones"
    assert reason is None

    # RoPE + doc-blocking auto-downgrades to FFN to avoid mask shape churn.
    scope, reason = _resolve_compile_scope(
        requested_scope="auto",
        model_cfg=ModelConfig(backbone_type="rope"),
        block_cross_document_attention=True,
    )
    assert scope == "ffn"
    assert reason is not None

    scope, reason = _resolve_compile_scope(
        requested_scope="auto",
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
    )
    assert scope == "backbones"
    assert reason is None

    scope, reason = _resolve_compile_scope(
        requested_scope="backbones",
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
    )
    assert scope == "backbones"
    assert reason is None


def test_build_compile_kwargs_for_scope_omits_dynamic_for_flash_backbones() -> None:
    class FlashDisentangledSelfAttention(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    FlashDisentangledSelfAttention.__module__ = "deberta.modeling.flashdeberta_attention"

    class _Encoder(torch.nn.Module):
        def __init__(self, *, flash: bool) -> None:
            super().__init__()
            self.attention = FlashDisentangledSelfAttention() if flash else torch.nn.Linear(4, 4)

    class _Backbone(torch.nn.Module):
        def __init__(self, *, flash: bool) -> None:
            super().__init__()
            self.encoder = _Encoder(flash=flash)
            self.proj = torch.nn.Linear(4, 4)

    class _TinyRTD(torch.nn.Module):
        def __init__(self, *, flash: bool) -> None:
            super().__init__()
            self.generator = _Backbone(flash=flash)
            self.discriminator = _Backbone(flash=flash)

    flash_model = _TinyRTD(flash=True)
    flash_kwargs, flash_dynamic = _build_compile_kwargs_for_scope(
        unwrapped_model=flash_model,
        compile_scope="backbones",
        compile_mode="default",
        compile_backend="inductor",
    )
    assert flash_kwargs["mode"] == "default"
    assert flash_kwargs["backend"] == "inductor"
    assert "dynamic" not in flash_kwargs
    assert flash_dynamic == "default"

    ffn_kwargs, ffn_dynamic = _build_compile_kwargs_for_scope(
        unwrapped_model=flash_model,
        compile_scope="ffn",
        compile_mode="default",
        compile_backend="inductor",
    )
    assert ffn_kwargs["dynamic"] is False
    assert ffn_dynamic == "false"

    eager_model = _TinyRTD(flash=False)
    eager_kwargs, eager_dynamic = _build_compile_kwargs_for_scope(
        unwrapped_model=eager_model,
        compile_scope="backbones",
        compile_mode="default",
        compile_backend="inductor",
    )
    assert eager_kwargs["dynamic"] is False
    assert eager_dynamic == "false"


def test_compile_controls_do_not_reference_environment_variables():
    import inspect

    import deberta.training.compile as compile_mod
    import deberta.training.entrypoint as entrypoint_mod

    for source in (inspect.getsource(entrypoint_mod), inspect.getsource(compile_mod)):
        assert "DEBERTA_COMPILE_SCOPE" not in source
        assert "DEBERTA_COMPILE_BACKEND" not in source
        assert "DEBERTA_HF_ATTN_KERNEL" not in source
