# ruff: noqa: F403,F405
from _config_and_training_shared_imports import *


def _write_resume_source_snapshots(run_dir: Path) -> None:
    """Write minimal config snapshots required for strict resume validation."""
    model_cfg = ModelConfig()
    data_cfg = DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy")
    train_cfg = TrainConfig()
    optim_cfg = OptimConfig()
    logging_cfg = LoggingConfig(output_dir=str(run_dir))
    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg, optim_cfg=optim_cfg)
    (run_dir / "model_config.json").write_text(
        json.dumps(asdict(model_cfg), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "data_config.json").write_text(
        json.dumps(asdict(data_cfg), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "train_config.json").write_text(
        json.dumps(asdict(train_cfg), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "optim_config.json").write_text(
        json.dumps(asdict(optim_cfg), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "logging_config.json").write_text(
        json.dumps(asdict(logging_cfg), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"config_schema_version": int(RUN_CONFIG_SCHEMA_VERSION)}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def test_run_pretraining_resume_at_max_steps_skips_data_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoint-2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_resume_source_snapshots(checkpoint_dir.parent)
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "data_state.json").write_text(
        json.dumps({"consumed_micro_batches": 50}),
        encoding="utf-8",
    )
    (checkpoint_dir / ".complete").write_text("ok\n", encoding="utf-8")

    replay_calls = {"next": 0}

    def _fail_cycle(_loader, *, start_epoch: int = 0):
        del start_epoch
        while True:
            replay_calls["next"] += 1
            raise AssertionError("resume replay should be skipped when global_step >= max_steps")
            yield {}  # pragma: no cover

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        cycle_fn=_fail_cycle,
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=2,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        resume_from_checkpoint=str(checkpoint_dir),
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )
    assert replay_calls["next"] == 0


def test_run_pretraining_resume_normalizes_legacy_partial_window_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoint-1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_resume_source_snapshots(checkpoint_dir.parent)
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "data_state.json").write_text(
        json.dumps({"consumed_micro_batches": 3}),
        encoding="utf-8",
    )
    (checkpoint_dir / ".complete").write_text("ok\n", encoding="utf-8")

    captured: dict[str, int] = {}

    def _capture_policy(*, train_cfg: Any, consumed_micro_batches: int, global_step: int):
        del train_cfg
        captured["consumed_micro_batches"] = int(consumed_micro_batches)
        captured["global_step"] = int(global_step)
        return 0, False, "captured"

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
    )
    monkeypatch.setattr(pretrain_mod, "_resolve_data_resume_policy", _capture_policy)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        resume_from_checkpoint=str(checkpoint_dir),
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    assert captured["global_step"] == 1
    assert captured["consumed_micro_batches"] == 2


def test_run_pretraining_resume_normalization_uses_save_time_ga_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoint-1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_resume_source_snapshots(checkpoint_dir.parent)
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "data_state.json").write_text(
        json.dumps(
            {
                "consumed_micro_batches": 3,
                "global_step": 1,
                "gradient_accumulation_steps": 3,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / ".complete").write_text("ok\n", encoding="utf-8")

    captured: dict[str, int] = {}

    def _capture_policy(*, train_cfg: Any, consumed_micro_batches: int, global_step: int):
        del train_cfg
        captured["consumed_micro_batches"] = int(consumed_micro_batches)
        captured["global_step"] = int(global_step)
        return 0, False, "captured"

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
    )
    monkeypatch.setattr(pretrain_mod, "_resolve_data_resume_policy", _capture_policy)

    # Current run uses GA=2, but resume normalization should use save-time GA=3.
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        resume_from_checkpoint=str(checkpoint_dir),
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    assert captured["global_step"] == 1
    assert captured["consumed_micro_batches"] == 3


def test_run_pretraining_resume_requires_data_state_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoint-2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_resume_source_snapshots(checkpoint_dir.parent)
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / ".complete").write_text("ok\n", encoding="utf-8")
    pretrain_mod = setup_pretraining_mocks(monkeypatch)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=3,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        resume_from_checkpoint=str(checkpoint_dir),
    )

    with pytest.raises(ValueError, match="missing/invalid data_state.json"):
        pretrain_mod.run_pretraining(
            model_cfg=ModelConfig(),
            data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
            train_cfg=train_cfg,
        )


def test_run_pretraining_resume_rejects_checkpoint_step_metadata_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoint-2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_resume_source_snapshots(checkpoint_dir.parent)
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "data_state.json").write_text(
        json.dumps({"consumed_micro_batches": 2, "global_step": 1}),
        encoding="utf-8",
    )
    (checkpoint_dir / ".complete").write_text("ok\n", encoding="utf-8")
    pretrain_mod = setup_pretraining_mocks(monkeypatch)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=3,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        resume_from_checkpoint=str(checkpoint_dir),
    )

    with pytest.raises(RuntimeError, match="Checkpoint step mismatch on resume"):
        pretrain_mod.run_pretraining(
            model_cfg=ModelConfig(),
            data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
            train_cfg=train_cfg,
        )


def test_run_pretraining_resume_accepts_legacy_single_digest_in_decoupled_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoint-0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_resume_source_snapshots(checkpoint_dir.parent)
    (checkpoint_dir.parent / "model_config.json").write_text(
        json.dumps(asdict(ModelConfig(backbone_type="hf_deberta_v2", embedding_sharing="gdes")), indent=2)
        + "\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "data_state.json").write_text(
        json.dumps(
            {
                "consumed_micro_batches": 0,
                "global_step": 0,
                "optimizer_param_digest": "legacydeadbeef000",
            }
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / ".complete").write_text("ok\n", encoding="utf-8")

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=SimpleRTD,
    )

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=1,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        decoupled_training=True,
        resume_from_checkpoint=str(checkpoint_dir),
    )
    (checkpoint_dir.parent / "logging_config.json").write_text(
        json.dumps(
            asdict(
                LoggingConfig(
                    output_dir=str(checkpoint_dir.parent),
                    logging_steps=1,
                    report_to="none",
                )
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        pretrain_mod.run_pretraining(
            model_cfg=ModelConfig(backbone_type="hf_deberta_v2", embedding_sharing="gdes"),
            data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
            train_cfg=train_cfg,
        )

    assert any(
        "legacy single optimizer digest while current run uses decoupled mode" in rec.message
        for rec in caplog.records
    )


def test_run_pretraining_logs_window_averaged_rtd_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=lambda **kwargs: SimpleRTD(
            behavior={
                "loss": 1.0,
                "gen_loss": [1.0, 9.0],
                "disc_loss": [10.0, 2.0],
                "disc_accuracy": [0.2, 0.8],
                "gen_token_count": [1.0, 9.0],
                "disc_token_count": [10.0, 2.0],
                "disc_positive_count": [7.0, 1.0],
            },
            **kwargs,
        ),
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=1,
        save_steps=0,
        report_to="tensorboard",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        token_weighted_gradient_accumulation=False,
        decoupled_training=False,
        torch_compile=False,
        export_hf_final=False,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    accel = FakeAccelerator.last_instance
    assert accel is not None
    step_rows = [row for row, step in accel.logged_rows if int(step or -1) == 1]
    assert step_rows
    metrics = step_rows[-1]
    assert metrics["gen_loss"] == pytest.approx(8.2, rel=0.0, abs=1e-6)
    assert metrics["disc_loss"] == pytest.approx(104.0 / 12.0, rel=0.0, abs=1e-6)
    assert metrics["disc_acc"] == pytest.approx(0.3, rel=0.0, abs=1e-6)
    assert "gen_token_count" not in metrics
    assert "disc_token_count" not in metrics
    assert metrics["disc_pos_frac"] == pytest.approx(8.0 / 12.0, rel=0.0, abs=1e-6)


@pytest.mark.parametrize(
    "scenario",
    [
        "steps_and_sync",
        "token_weighted_scaling",
        "branch_loss_weights",
        "skip_generator_step",
        "partial_disc_window_sync",
    ],
)
def test_run_pretraining_decoupled_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    call_order: list[str] = []
    step_counts = {"gen": 0, "disc": 0}
    behavior: dict[str, Any] = {}
    extra_patches: dict[str, Any] | None = None

    if scenario == "steps_and_sync":
        behavior = {
            "generator_phase_loss_scale": 1.0,
            "discriminator_phase_loss_scale": 2.0,
            "on_generator_phase": lambda _m, _i: call_order.append("gen_forward"),
            "on_discriminator_phase": lambda _m, _i: call_order.append("disc_forward"),
            "on_sync_discriminator_embeddings": lambda _m: call_order.append("sync"),
        }
        train_cfg = TrainConfig(
            output_dir=str(tmp_path / "run"),
            max_steps=1,
            logging_steps=1,
            save_steps=0,
            report_to="tensorboard",
            mixed_precision="no",
            tf32=False,
            dataloader_num_workers=0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            token_weighted_gradient_accumulation=False,
            gen_loss_weight=1.0,
            disc_loss_weight=1.0,
            torch_compile=False,
            export_hf_final=False,
            decoupled_training=True,
        )
    elif scenario == "token_weighted_scaling":
        behavior = {
            "generator_phase_loss_scale": [10.0, 20.0],
            "discriminator_phase_loss_scale": [5.0, 7.0],
            "discriminator_phase_accuracy": 0.5,
            "discriminator_phase_positive_count": 0.0,
        }
        micro_counts = iter([(1.0, 4.0), (3.0, 2.0)])

        def _count_tokens_for_microbatch(*_args: Any, **_kwargs: Any) -> tuple[float, float]:
            return next(micro_counts)

        extra_patches = {"_count_rtd_tokens_for_batch": _count_tokens_for_microbatch}
        train_cfg = TrainConfig(
            output_dir=str(tmp_path / "run"),
            max_steps=1,
            logging_steps=1,
            save_steps=0,
            report_to="none",
            mixed_precision="no",
            tf32=False,
            dataloader_num_workers=0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            token_weighted_gradient_accumulation=True,
            gen_loss_weight=1.0,
            disc_loss_weight=1.0,
            torch_compile=False,
            export_hf_final=False,
            decoupled_training=True,
        )
    elif scenario == "branch_loss_weights":
        behavior = {"generator_phase_loss_scale": 2.0, "discriminator_phase_loss_scale": 3.0}
        train_cfg = TrainConfig(
            output_dir=str(tmp_path / "run"),
            max_steps=1,
            logging_steps=0,
            save_steps=0,
            report_to="none",
            mixed_precision="no",
            tf32=False,
            dataloader_num_workers=0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            token_weighted_gradient_accumulation=False,
            gen_loss_weight=0.25,
            disc_loss_weight=4.0,
            torch_compile=False,
            export_hf_final=False,
            decoupled_training=True,
        )
    elif scenario == "skip_generator_step":
        behavior = {"generator_phase_loss_scale": 2.0, "discriminator_phase_loss_scale": 3.0}
        train_cfg = TrainConfig(
            output_dir=str(tmp_path / "run"),
            max_steps=1,
            logging_steps=0,
            save_steps=0,
            report_to="none",
            mixed_precision="no",
            tf32=False,
            dataloader_num_workers=0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            token_weighted_gradient_accumulation=False,
            gen_loss_weight=0.0,
            disc_loss_weight=1.0,
            torch_compile=False,
            export_hf_final=False,
            decoupled_training=True,
        )
    elif scenario == "partial_disc_window_sync":
        behavior = {
            "generator_phase_loss_scale": [2.0, 2.0],
            "generator_phase_token_count": [0.0, 1.0],
            "discriminator_phase_loss_scale": 3.0,
        }
        train_cfg = TrainConfig(
            output_dir=str(tmp_path / "run"),
            max_steps=1,
            logging_steps=0,
            save_steps=0,
            report_to="none",
            mixed_precision="no",
            tf32=False,
            dataloader_num_workers=0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            token_weighted_gradient_accumulation=False,
            gen_loss_weight=1.0,
            disc_loss_weight=1.0,
            torch_compile=False,
            export_hf_final=False,
            decoupled_training=True,
        )
    else:  # pragma: no cover
        raise AssertionError(f"Unsupported scenario: {scenario}")

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=lambda **kwargs: SimpleRTD(behavior=behavior, **kwargs),
        extra_patches=extra_patches,
    )
    original_build_decoupled = pretrain_mod._build_decoupled_optimizers

    def _build_logged_decoupled(model: torch.nn.Module, cfg: TrainConfig, *, mixed_precision: str = "no"):
        gen_opt, disc_opt = original_build_decoupled(model, cfg, mixed_precision=mixed_precision)
        original_gen_step = gen_opt.step
        original_disc_step = disc_opt.step

        def _gen_step(_opt_self: Any, *args: Any, **kwargs: Any):
            step_counts["gen"] += 1
            if scenario == "steps_and_sync":
                call_order.append("gen_step")
            return original_gen_step(*args, **kwargs)

        def _disc_step(_opt_self: Any, *args: Any, **kwargs: Any):
            step_counts["disc"] += 1
            if scenario == "steps_and_sync":
                call_order.append("disc_step")
            return original_disc_step(*args, **kwargs)

        gen_opt.step = types.MethodType(_gen_step, gen_opt)  # type: ignore[method-assign]
        disc_opt.step = types.MethodType(_disc_step, disc_opt)  # type: ignore[method-assign]
        return gen_opt, disc_opt

    monkeypatch.setattr(pretrain_mod, "_build_decoupled_optimizers", _build_logged_decoupled)

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope", embedding_sharing="gdes"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    accel = FakeAccelerator.last_instance
    assert accel is not None

    if scenario == "steps_and_sync":
        assert step_counts == {"gen": 1, "disc": 1}
        gen_forward_idx = call_order.index("gen_forward")
        gen_step_idx = call_order.index("gen_step")
        sync_after_gen_idx = call_order.index("sync", gen_step_idx + 1)
        disc_forward_idx = call_order.index("disc_forward")
        disc_step_idx = call_order.index("disc_step")
        assert gen_forward_idx < gen_step_idx < sync_after_gen_idx < disc_forward_idx < disc_step_idx
        step_rows = [row for row, step in accel.logged_rows if int(step or -1) == 1]
        assert step_rows
        assert "decoupled_training" not in step_rows[-1]
    elif scenario == "token_weighted_scaling":
        assert step_counts == {"gen": 1, "disc": 1}
        assert accel.calls["backward"] == pytest.approx(
            [
                10.0 * (1.0 / 4.0) * 2.0,
                20.0 * (3.0 / 4.0) * 2.0,
                5.0 * (4.0 / 6.0) * 2.0,
                7.0 * (2.0 / 6.0) * 2.0,
            ],
            rel=0.0,
            abs=1e-6,
        )
    elif scenario == "branch_loss_weights":
        assert step_counts == {"gen": 1, "disc": 1}
        assert accel.calls["backward"] == pytest.approx([2.0 * 0.25, 3.0 * 4.0], rel=0.0, abs=1e-6)
    elif scenario == "skip_generator_step":
        assert step_counts == {"gen": 0, "disc": 1}
        assert accel.calls["backward"] == pytest.approx([3.0], rel=0.0, abs=1e-6)
    elif scenario == "partial_disc_window_sync":
        assert step_counts == {"gen": 1, "disc": 1}
        model = SimpleRTD.last_instance
        assert model is not None
        assert len(model.calls["forward_discriminator_phase"]) == 2
        assert accel.calls["backward"] == pytest.approx([2.0, 2.0, 0.0, 3.0], rel=0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("decoupled_training", "expected_nonfinite_checks"),
    [
        (True, 2),
        (False, 1),
    ],
)
def test_run_pretraining_nonfinite_all_reduce_only_on_sync_microstep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decoupled_training: bool,
    expected_nonfinite_checks: int,
) -> None:
    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=SimpleRTD,
    )
    call_count = {"value": 0}
    original_any_rank_flag_true = pretrain_mod._any_rank_flag_true

    def _counted_any_rank_flag_true(*, accelerator: Any, flag: bool) -> bool:
        call_count["value"] += 1
        return bool(original_any_rank_flag_true(accelerator=accelerator, flag=flag))

    monkeypatch.setattr(pretrain_mod, "_any_rank_flag_true", _counted_any_rank_flag_true)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=0,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        decoupled_training=bool(decoupled_training),
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope", embedding_sharing="gdes"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    assert int(call_count["value"]) == int(expected_nonfinite_checks)


def test_run_pretraining_decoupled_nonfinite_disc_does_not_double_step_gen_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler_steps = {"gen": 0, "disc": 0}

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=lambda **kwargs: SimpleRTD(
            behavior={
                "generator_phase_loss_scale": 1.0,
                "discriminator_phase_loss_scale": float("nan"),
            },
            **kwargs,
        ),
    )

    scheduler_build_count = 0

    def _build_counted_scheduler(optimizer: torch.optim.Optimizer, _cfg: TrainConfig):
        nonlocal scheduler_build_count
        phase = "gen" if scheduler_build_count == 0 else "disc"
        scheduler_build_count += 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        original_step = scheduler.step

        def _count_step(*args: Any, **kwargs: Any) -> Any:
            scheduler_steps[phase] += 1
            return original_step(*args, **kwargs)

        scheduler.step = _count_step  # type: ignore[method-assign]
        return scheduler

    monkeypatch.setattr(pretrain_mod, "_build_scheduler", _build_counted_scheduler)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=0,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        decoupled_training=True,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope", embedding_sharing="gdes"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    assert scheduler_steps["gen"] == 1
    assert scheduler_steps["disc"] == 0


def test_run_pretraining_decoupled_skips_discriminator_for_zero_generator_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=lambda **kwargs: SimpleRTD(
            behavior={
                "generator_phase_loss_scale": 0.0,
                "generator_phase_token_count": 0.0,
            },
            **kwargs,
        ),
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=0,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        decoupled_training=True,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope", embedding_sharing="gdes"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    model = SimpleRTD.last_instance
    assert model is not None
    assert len(model.calls.get("forward_discriminator_phase", [])) == 0


def test_run_pretraining_decoupled_routes_phase_calls_through_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=SimpleRTD,
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=0,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=False,
        decoupled_training=True,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope", embedding_sharing="gdes"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    model = SimpleRTD.last_instance
    assert model is not None
    # One generator and one discriminator phase should each route through model(...)
    # so distributed wrappers keep their forward-path bookkeeping/autocast hooks.
    assert len(model.calls.get("forward", [])) == 2
    assert len(model.calls.get("forward_generator_phase", [])) == 1
    assert len(model.calls.get("forward_discriminator_phase", [])) == 1


def test_run_pretraining_final_export_uses_subprocess_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_calls: list[tuple[str, str]] = []

    def _fake_export_subprocess(*, checkpoint_dir: Path, output_dir: Path) -> None:
        export_calls.append((str(checkpoint_dir), str(output_dir)))

    def _fake_save_checkpoint(
        *,
        accelerator: Any,
        checkpoint_dir: Path,
        output_dir: Path,
        consumed_micro_batches: int,
        save_total_limit: int,
        log_label: str,
        **kwargs: Any,
    ) -> None:
        del accelerator, output_dir, consumed_micro_batches, save_total_limit, log_label, kwargs
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        save_checkpoint_fn=_fake_save_checkpoint,
        extra_patches={"_export_discriminator_hf_subprocess": _fake_export_subprocess},
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=0,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=False,
        export_hf_final=True,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    assert export_calls
    ckpt_path, export_path = export_calls[-1]
    assert ckpt_path.endswith("checkpoint-1")
    assert export_path.endswith("final_hf")


def _run_zero_token_weighted_case(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    debug_metrics: bool,
) -> Path:
    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=lambda **kwargs: SimpleRTD(
            behavior={
                "loss": 1.0,
                "gen_loss": 1.0,
                "disc_loss": 1.0,
                "disc_accuracy": 1.0,
                "gen_token_count": 1.0,
                "disc_token_count": 1.0,
                "disc_positive_count": 1.0,
            },
            **kwargs,
        ),
        extra_patches={"_count_rtd_tokens_for_batch": lambda *args, **kwargs: (0.0, 0.0)},
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=1,
        save_steps=0,
        report_to="tensorboard",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        token_weighted_gradient_accumulation=True,
        decoupled_training=False,
        debug_metrics=bool(debug_metrics),
        torch_compile=False,
        export_hf_final=False,
    )
    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )
    return Path(train_cfg.output_dir) / "metrics.jsonl.gz"


def _latest_zero_token_tracker_metrics() -> dict[str, Any]:
    accel = FakeAccelerator.last_instance
    assert accel is not None
    step_rows = [row for row, step in accel.logged_rows if int(step or -1) == 1]
    assert step_rows
    return step_rows[-1]


def _assert_zero_window_metrics_hidden_from_trackers(metrics: dict[str, Any]) -> None:
    assert "zero_gen_window_total" not in metrics
    assert "zero_disc_window_total" not in metrics
    assert "zero_gen_window_since_log" not in metrics
    assert "zero_disc_window_since_log" not in metrics


def _load_last_debug_metrics_row(metrics_path: Path) -> dict[str, Any]:
    with gzip.open(metrics_path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f.read().splitlines()]
    assert rows
    debug_rows = [row for row in rows if bool(row.get("debug_metrics"))]
    assert debug_rows
    return debug_rows[-1]


@pytest.mark.parametrize(
    ("debug_metrics", "expect_metrics_file", "expect_warning"),
    [
        (False, False, True),
        (True, True, False),
    ],
)
def test_run_pretraining_zero_token_weighted_metrics_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    debug_metrics: bool,
    expect_metrics_file: bool,
    expect_warning: bool,
) -> None:
    with caplog.at_level(logging.WARNING):
        metrics_path = _run_zero_token_weighted_case(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            debug_metrics=bool(debug_metrics),
        )
    _assert_zero_window_metrics_hidden_from_trackers(_latest_zero_token_tracker_metrics())

    if expect_warning:
        assert any(
            "Token-weighted GA window has zero effective tokens" in rec.message for rec in caplog.records
        )

    if expect_metrics_file:
        assert metrics_path.exists()
        last = _load_last_debug_metrics_row(metrics_path)
        assert int(last["step"]) == 1
        assert float(last["zero_gen_window_total"]) == pytest.approx(1.0, rel=0.0, abs=1e-6)
        assert float(last["zero_disc_window_total"]) == pytest.approx(1.0, rel=0.0, abs=1e-6)
        assert float(last["zero_gen_window_since_log"]) == pytest.approx(1.0, rel=0.0, abs=1e-6)
        assert float(last["zero_disc_window_since_log"]) == pytest.approx(1.0, rel=0.0, abs=1e-6)
    else:
        assert not metrics_path.exists()


def test_run_pretraining_decoupled_debug_metrics_writes_local_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        rtd_cls=SimpleRTD,
    )
    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=1,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        decoupled_training=True,
        debug_metrics=True,
        torch_compile=False,
        export_hf_final=False,
    )
    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2", embedding_sharing="gdes"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    metrics_path = Path(train_cfg.output_dir) / "metrics.jsonl.gz"
    assert metrics_path.exists()
    last = _load_last_debug_metrics_row(metrics_path)
    assert int(last["step"]) == 1
    assert float(last["zero_gen_window_total"]) == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert float(last["zero_disc_window_total"]) == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert float(last["zero_gen_window_since_log"]) == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert float(last["zero_disc_window_since_log"]) == pytest.approx(0.0, rel=0.0, abs=1e-6)


def test_run_pretraining_compiles_generator_and_discriminator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compile_calls: list[tuple[Any, dict[str, Any]]] = []

    def _fake_compile(
        target: Any, *, mode: str = "default", backend: str = "inductor", dynamic: bool | None = None
    ) -> Any:
        compile_calls.append((target, {"mode": str(mode), "backend": str(backend), "dynamic": dynamic}))
        return target

    pretrain_mod = setup_pretraining_mocks(monkeypatch)
    monkeypatch.setattr(pretrain_mod.torch, "compile", _fake_compile)

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
    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    instance = SimpleRTD.last_instance
    assert instance is not None
    assert len(compile_calls) == 2
    for i in range(2):
        assert callable(compile_calls[i][0])
        assert compile_calls[i][1] == {"mode": "default", "backend": "inductor", "dynamic": False}
    assert getattr(compile_calls[0][0], "__self__", None) is instance.generator
    assert getattr(compile_calls[1][0], "__self__", None) is instance.discriminator


def test_run_pretraining_builds_doc_block_mask_before_compile_stabilizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes import _PRETRAINING_BATCH

    pretrain_mod = setup_pretraining_mocks(monkeypatch)

    batch_with_doc_ids: dict[str, torch.Tensor] = {
        k: v.clone() for k, v in _PRETRAINING_BATCH.items() if isinstance(v, torch.Tensor)
    }
    batch_with_doc_ids["doc_ids"] = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.long)
    expected_mask = pretrain_mod._build_doc_block_mask(batch_with_doc_ids["doc_ids"])

    def _cycle_with_doc_ids(_loader: Any, *, start_epoch: int = 0):
        del _loader, start_epoch
        while True:
            yield {k: v.clone() for k, v in batch_with_doc_ids.items()}

    monkeypatch.setattr(pretrain_mod, "_cycle_dataloader", _cycle_with_doc_ids)
    monkeypatch.setattr(
        pretrain_mod.torch,
        "compile",
        lambda target, *, mode="default", backend="inductor", dynamic=None: target,
    )

    seen_masks: list[torch.Tensor] = []
    original_stabilize = pretrain_mod._stabilize_compile_attention_mask

    def _stabilize_spy(
        *,
        batch: dict[str, Any],
        compile_enabled: bool,
        compile_scope: str,
        backbone_type: str,
    ) -> dict[str, Any]:
        mask = batch.get("attention_mask")
        if isinstance(mask, torch.Tensor):
            seen_masks.append(mask.detach().clone())
        return original_stabilize(
            batch=batch,
            compile_enabled=compile_enabled,
            compile_scope=compile_scope,
            backbone_type=backbone_type,
        )

    monkeypatch.setattr(pretrain_mod, "_stabilize_compile_attention_mask", _stabilize_spy)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        logging_steps=0,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        token_weighted_gradient_accumulation=False,
        torch_compile=True,
        torch_compile_scope="backbones",
        export_hf_final=False,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="rope"),
        data_cfg=DataConfig(
            dataset_name="hf-internal-testing/librispeech_asr_dummy",
            block_cross_document_attention=True,
        ),
        train_cfg=train_cfg,
    )

    assert seen_masks
    assert seen_masks[0].ndim == 3
    assert seen_masks[0].dtype == torch.bool
    assert torch.equal(seen_masks[0], expected_mask)


def test_run_pretraining_hf_deberta_auto_scope_compiles_backbones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = torch.nn.Linear(2, 2)
            self.intermediate = torch.nn.Linear(2, 2)
            self.output = torch.nn.Linear(2, 2)

    class _FakeBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = torch.nn.Linear(2, 2)
            self.encoder = torch.nn.Module()
            self.encoder.layer = torch.nn.ModuleList([_FakeLayer()])

    compile_calls: list[tuple[Any, dict[str, Any]]] = []

    def _fake_compile(
        target: Any, *, mode: str = "default", backend: str = "inductor", dynamic: bool | None = None
    ) -> Any:
        compile_calls.append((target, {"mode": str(mode), "backend": str(backend), "dynamic": dynamic}))
        return target

    created_models: list[SimpleRTD] = []

    def _make_scope_rtd(**kwargs: Any) -> SimpleRTD:
        model = SimpleRTD(**kwargs)
        model.generator = _FakeBackbone()
        model.discriminator = _FakeBackbone()
        created_models.append(model)
        return model

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        rtd_cls=_make_scope_rtd,
        build_backbones_fn=lambda **kwargs: (_FakeBackbone(), _FakeBackbone()),
    )
    monkeypatch.setattr(pretrain_mod.torch, "compile", _fake_compile)

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
    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )

    assert created_models
    instance = created_models[-1]
    assert instance is not None
    assert len(compile_calls) == 2
    for i in range(2):
        assert callable(compile_calls[i][0])
        assert compile_calls[i][1] == {"mode": "default", "backend": "inductor", "dynamic": False}
    assert getattr(compile_calls[0][0], "__self__", None) is instance.generator
    assert getattr(compile_calls[1][0], "__self__", None) is instance.discriminator


def test_run_pretraining_skips_nonfinite_grad_window_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        extra_patches={"_global_grad_l2_norm": lambda _model: float("inf")},
    )
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
        decoupled_training=False,
        torch_compile=False,
        export_hf_final=False,
        max_grad_norm=1.0,
    )

    with caplog.at_level(logging.WARNING):
        pretrain_mod.run_pretraining(
            model_cfg=ModelConfig(),
            data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
            train_cfg=train_cfg,
        )
    assert "nonfinite_window_skipped=1" in caplog.text


def test_run_pretraining_nonfinite_grad_norm_never_steps_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opt_ref: dict[str, Any] = {}

    class _CountingSGD(torch.optim.SGD):
        def __init__(self, params: Any, lr: float) -> None:
            super().__init__(params, lr=lr)
            self.step_calls = 0

        def step(self, closure: Any = None):  # type: ignore[override]
            self.step_calls += 1
            return super().step(closure=closure)

    def _build_optimizer(model: torch.nn.Module, _cfg: Any, **_kwargs: Any) -> torch.optim.Optimizer:
        opt = _CountingSGD(model.parameters(), lr=0.1)
        opt_ref["opt"] = opt
        return opt

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        extra_patches={
            "_global_grad_l2_norm": lambda _model: float("inf"),
            "_build_optimizer": _build_optimizer,
        },
    )
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
        decoupled_training=False,
        torch_compile=False,
        export_hf_final=False,
        max_grad_norm=1.0,
    )

    pretrain_mod.run_pretraining(
        model_cfg=ModelConfig(),
        data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
        train_cfg=train_cfg,
    )
    opt = opt_ref.get("opt")
    assert isinstance(opt, _CountingSGD)
    assert int(opt.step_calls) == 0


def test_apply_nonfinite_recovery_ratchets_lr_mult_and_resets_state_on_interval() -> None:
    # First nonfinite event: lr_mult should halve (0.5 backoff).
    new_mult, did_reset = _apply_nonfinite_recovery(lr_mult=1.0, skip_streak=1)
    assert did_reset is False
    assert abs(new_mult - 0.5) < 1e-9

    # Second consecutive: ratchets further.
    new_mult2, _ = _apply_nonfinite_recovery(lr_mult=new_mult, skip_streak=2)
    assert abs(new_mult2 - 0.25) < 1e-9

    # At streak=4: optimizer state reset triggered.
    new_mult4, did_reset4 = _apply_nonfinite_recovery(lr_mult=new_mult2, skip_streak=4)
    assert did_reset4 is True
    assert new_mult4 >= 0.01  # floor

    # lr_mult cannot go below _NONFINITE_LR_MULT_FLOOR.
    new_mult_floor, _ = _apply_nonfinite_recovery(lr_mult=0.01, skip_streak=5)
    assert abs(new_mult_floor - 0.01) < 1e-9


def test_apply_lr_mult_scales_all_param_groups() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    _apply_lr_mult(optimizer, 0.5)
    assert abs(float(optimizer.param_groups[0]["lr"]) - 5e-4) < 1e-12


def test_apply_lr_mult_uses_unscaled_reference_without_compounding() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    _record_unscaled_lrs(optimizer, scheduler)

    _apply_lr_mult(optimizer, 0.5)
    assert float(optimizer.param_groups[0]["lr"]) == pytest.approx(5e-4, abs=1e-12)

    # Applying a new multiplier must be absolute against scheduler LR (1e-3),
    # not multiplicative against the already-scaled group LR (5e-4).
    _apply_lr_mult(optimizer, 0.25)
    assert float(optimizer.param_groups[0]["lr"]) == pytest.approx(2.5e-4, abs=1e-12)


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


def test_persist_or_validate_run_configs_does_not_backfill_metadata_on_failed_resume(tmp_path: Path):
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

    run_meta_path = out / "run_metadata.json"
    run_meta_path.unlink()
    changed_model = ModelConfig(backbone_type="rope", hidden_size=1024)

    with pytest.raises(ValueError, match="Resume configuration mismatch for model_config.json"):
        _persist_or_validate_run_configs(
            output_dir=out,
            model_cfg=changed_model,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            resume_checkpoint=str(out / "checkpoint-10"),
            is_main_process=True,
        )
    assert not run_meta_path.exists()


def test_persist_or_validate_run_configs_validates_against_resume_source_run_dir(tmp_path: Path):
    source_run = tmp_path / "source-run"
    source_run.mkdir(parents=True, exist_ok=True)
    model_cfg = ModelConfig(backbone_type="rope")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig(max_steps=10)
    _persist_or_validate_run_configs(
        output_dir=source_run,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        is_main_process=True,
    )
    checkpoint_dir = source_run / "checkpoint-10"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    new_output_dir = tmp_path / "new-run"
    new_output_dir.mkdir(parents=True, exist_ok=True)
    mismatched_model = ModelConfig(backbone_type="rope", hidden_size=1024)
    with pytest.raises(ValueError, match="Resume configuration mismatch for model_config.json"):
        _persist_or_validate_run_configs(
            output_dir=new_output_dir,
            model_cfg=mismatched_model,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            resume_checkpoint=str(checkpoint_dir),
            is_main_process=True,
        )


def test_persist_or_validate_run_configs_tracks_resume_source_when_output_dir_differs(tmp_path: Path):
    source_run = tmp_path / "source-run"
    source_run.mkdir(parents=True, exist_ok=True)
    model_cfg = ModelConfig(backbone_type="rope")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig(max_steps=10)
    _persist_or_validate_run_configs(
        output_dir=source_run,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        is_main_process=True,
    )
    checkpoint_dir = source_run / "checkpoint-10"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    new_output_dir = tmp_path / "new-run"
    new_output_dir.mkdir(parents=True, exist_ok=True)
    changed_train_cfg = TrainConfig(max_steps=25)
    _persist_or_validate_run_configs(
        output_dir=new_output_dir,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=changed_train_cfg,
        resume_checkpoint=str(checkpoint_dir),
        is_main_process=True,
    )

    resume_source = json.loads((new_output_dir / "resume_source.json").read_text(encoding="utf-8"))
    assert resume_source["resume_checkpoint"] == str(checkpoint_dir.resolve())
    assert resume_source["resume_run_dir"] == str(source_run.resolve())
    assert (new_output_dir / "model_config.json").read_text(encoding="utf-8") == (
        source_run / "model_config.json"
    ).read_text(encoding="utf-8")
    assert (new_output_dir / "train_config.json").read_text(encoding="utf-8") == (
        source_run / "train_config.json"
    ).read_text(encoding="utf-8")


def test_persist_or_validate_run_configs_allows_resume_when_only_logging_output_dir_differs(
    tmp_path: Path,
) -> None:
    source_run = tmp_path / "source-run"
    source_run.mkdir(parents=True, exist_ok=True)
    source_logging_dir = source_run / "logs"
    source_logging_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig(backbone_type="rope")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig(max_steps=10)
    source_logging_cfg = LoggingConfig(output_dir=str(source_logging_dir))
    _persist_or_validate_run_configs(
        output_dir=source_run,
        logging_output_dir=source_logging_dir,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        logging_cfg=source_logging_cfg,
        resume_checkpoint=None,
        is_main_process=True,
    )
    checkpoint_dir = source_run / "checkpoint-10"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    new_output_dir = tmp_path / "new-run"
    new_output_dir.mkdir(parents=True, exist_ok=True)
    new_logging_dir = new_output_dir / "logs"
    new_logging_dir.mkdir(parents=True, exist_ok=True)
    resumed_logging_cfg = LoggingConfig(output_dir=str(new_logging_dir))
    _persist_or_validate_run_configs(
        output_dir=new_output_dir,
        logging_output_dir=new_logging_dir,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        logging_cfg=resumed_logging_cfg,
        resume_checkpoint=str(checkpoint_dir),
        is_main_process=True,
    )

    resume_source = json.loads((new_output_dir / "resume_source.json").read_text(encoding="utf-8"))
    assert resume_source["resume_checkpoint"] == str(checkpoint_dir.resolve())
    assert resume_source["resume_run_dir"] == str(source_run.resolve())
    assert (new_output_dir / "logging_config.json").read_text(encoding="utf-8") == (
        source_run / "logging_config.json"
    ).read_text(encoding="utf-8")


def test_persist_or_validate_run_configs_preflight_rejects_conflicting_resume_snapshot(tmp_path: Path):
    source_run = tmp_path / "source-run"
    source_run.mkdir(parents=True, exist_ok=True)
    model_cfg = ModelConfig(backbone_type="rope")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig(max_steps=10)
    _persist_or_validate_run_configs(
        output_dir=source_run,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        is_main_process=True,
    )
    checkpoint_dir = source_run / "checkpoint-10"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    new_output_dir = tmp_path / "new-run"
    new_output_dir.mkdir(parents=True, exist_ok=True)
    (new_output_dir / "model_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting run snapshot"):
        _persist_or_validate_run_configs(
            output_dir=new_output_dir,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            resume_checkpoint=str(checkpoint_dir),
            is_main_process=False,
            preflight_only=True,
        )


def test_persist_or_validate_run_configs_rejects_resume_when_source_snapshots_missing(tmp_path: Path):
    source_run = tmp_path / "source-run"
    source_run.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = source_run / "checkpoint-1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="source run directory is missing required config snapshots"):
        _persist_or_validate_run_configs(
            output_dir=out,
            model_cfg=ModelConfig(backbone_type="rope"),
            data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
            train_cfg=TrainConfig(),
            resume_checkpoint=str(checkpoint_dir),
            is_main_process=True,
        )


def test_persist_or_validate_run_configs_allows_resume_when_only_inert_model_fields_change(tmp_path: Path):
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

    inert_changed_model = ModelConfig(
        backbone_type="rope",
        hf_attention_kernel="stable",
        hf_max_position_embeddings=1024,
    )
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=inert_changed_model,
        data_cfg=base_data,
        train_cfg=base_train,
        resume_checkpoint=str(out / "checkpoint-10"),
        is_main_process=True,
    )


def test_persist_or_validate_run_configs_writes_original_and_resolved_yaml(
    tmp_path: Path,
) -> None:
    pytest.importorskip("yaml")

    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    src_cfg = tmp_path / "source.yaml"
    src_text = "model:\n  backbone_type: rope\ntrain:\n  max_steps: 7\n"
    src_cfg.write_text(src_text, encoding="utf-8")

    model_cfg = ModelConfig(backbone_type="rope")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig(max_steps=7)
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        config_path=src_cfg,
        is_main_process=True,
    )

    original_path = out / "config_original.yaml"
    resolved_path = out / "config_resolved.yaml"
    assert original_path.read_text(encoding="utf-8") == src_text
    assert resolved_path.exists()
    loaded_resolved = load_config(resolved_path)
    assert loaded_resolved.model.backbone_type == model_cfg.backbone_type
    assert loaded_resolved.data.dataset_name == data_cfg.dataset_name
    assert loaded_resolved.train.max_steps == train_cfg.max_steps


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


def _checkpoint_saving_accelerator(
    *,
    is_main_process: bool,
    write_weights: bool = True,
) -> FakeAccelerator:
    """Build a fake accelerator whose ``save_state`` writes checkpoint-like files."""

    accel = FakeAccelerator(is_main_process=bool(is_main_process))

    def _save_state(output_dir: str | None) -> None:
        if output_dir is None:
            return
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        if write_weights:
            (p / "model.safetensors").write_bytes(b"weights")
        marker = "main" if accel.is_main_process else "worker"
        (p / f"{marker}.txt").write_text("ok", encoding="utf-8")

    accel.save_state_hook = _save_state
    return accel


def test_save_training_checkpoint_calls_collective_save_on_non_main_rank(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint-1"
    ckpt.mkdir(parents=True, exist_ok=True)

    accel = _checkpoint_saving_accelerator(is_main_process=False)
    _save_training_checkpoint(
        accelerator=accel,
        checkpoint_dir=ckpt,
        output_dir=out,
        consumed_micro_batches=7,
        save_total_limit=3,
        log_label="periodic",
    )

    assert len(accel.calls["save_state"]) == 1
    staged = Path(str(accel.calls["save_state"][0]))
    assert staged.parent == out
    assert staged.name.startswith(f".{ckpt.name}.tmp-")
    assert len(accel.calls["wait_for_everyone"]) >= 3
    assert not (ckpt / "data_state.json").exists()


def test_save_training_checkpoint_writes_data_progress_on_main_rank(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint-3"

    accel = _checkpoint_saving_accelerator(is_main_process=True)
    _save_training_checkpoint(
        accelerator=accel,
        checkpoint_dir=ckpt,
        output_dir=out,
        consumed_micro_batches=42,
        save_total_limit=3,
        log_label="final",
    )

    assert len(accel.calls["save_state"]) == 1
    staged = Path(str(accel.calls["save_state"][0]))
    assert staged.parent == out
    assert staged.name.startswith(f".{ckpt.name}.tmp-")
    consumed, lr_mult, digest = _load_checkpoint_data_progress(ckpt)
    assert consumed == 42
    assert lr_mult == 1.0
    assert digest is None  # no digest passed
    assert (ckpt / ".complete").exists()


def test_save_training_checkpoint_rejects_overwrite_of_nonempty_checkpoint_dir(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint-8"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "stale.bin").write_bytes(b"stale")

    accel = _checkpoint_saving_accelerator(is_main_process=True)
    with pytest.raises(RuntimeError, match="Refusing to overwrite non-empty checkpoint directory"):
        _save_training_checkpoint(
            accelerator=accel,
            checkpoint_dir=ckpt,
            output_dir=out,
            consumed_micro_batches=1,
            save_total_limit=2,
            log_label="periodic",
        )


def test_save_training_checkpoint_rotates_only_after_postsave_validation(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    old_ckpt = out / "checkpoint-1"
    old_ckpt.mkdir(parents=True, exist_ok=True)
    (old_ckpt / "model.safetensors").write_bytes(b"old")
    (old_ckpt / "data_state.json").write_text(
        json.dumps({"consumed_micro_batches": 1}),
        encoding="utf-8",
    )
    (old_ckpt / ".complete").write_text("ok\n", encoding="utf-8")
    new_ckpt = out / "checkpoint-2"

    accel = _checkpoint_saving_accelerator(is_main_process=True)
    _save_training_checkpoint(
        accelerator=accel,
        checkpoint_dir=new_ckpt,
        output_dir=out,
        consumed_micro_batches=2,
        save_total_limit=1,
        log_label="periodic",
    )

    assert not old_ckpt.exists()
    assert new_ckpt.exists()
    assert (new_ckpt / ".complete").exists()


def test_save_training_checkpoint_skips_rotation_when_new_checkpoint_weights_invalid(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    old_ckpt = out / "checkpoint-1"
    old_ckpt.mkdir(parents=True, exist_ok=True)
    (old_ckpt / "model.safetensors").write_bytes(b"old")
    (old_ckpt / "data_state.json").write_text(
        json.dumps({"consumed_micro_batches": 1}),
        encoding="utf-8",
    )
    (old_ckpt / ".complete").write_text("ok\n", encoding="utf-8")
    new_ckpt = out / "checkpoint-2"

    accel = _checkpoint_saving_accelerator(is_main_process=True, write_weights=False)
    with pytest.raises(RuntimeError, match="Post-save structural validation failed"):
        _save_training_checkpoint(
            accelerator=accel,
            checkpoint_dir=new_ckpt,
            output_dir=out,
            consumed_micro_batches=2,
            save_total_limit=1,
            log_label="periodic",
        )

    assert old_ckpt.exists()
    assert not new_ckpt.exists()
