# ruff: noqa: F403,F405
from _config_and_training_shared_imports import *
from test_config_and_training_resume import _checkpoint_saving_accelerator


def test_load_hf_dataset_handles_missing_cache_dir_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def _fake_load_dataset(name: str, **kwargs: Any) -> list[dict[str, str]]:
        calls.append((str(name), dict(kwargs)))
        return [{"text": "ok"}]

    fake_datasets = types.SimpleNamespace(
        load_dataset=_fake_load_dataset,
        load_from_disk=lambda _path: [],
        DatasetDict=dict,
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    cfg = DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy")
    out = load_hf_dataset(cfg=cfg, split="train", streaming=True)

    assert out == [{"text": "ok"}]
    assert calls
    assert calls[0][0] == "hf-internal-testing/librispeech_asr_dummy"
    assert "cache_dir" in calls[0][1]
    assert calls[0][1]["cache_dir"] is None


def test_load_config_returns_frozen_top_level_and_sections(tmp_path: Path):
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 1",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert isinstance(cfg, Config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.train = TrainConfig(max_steps=2)  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.optim.scheduler.warmup_steps = 5  # type: ignore[misc]


def test_load_config_supports_extended_sections_and_projects_to_runtime_train(tmp_path: Path):
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
                "  checkpoint:",
                "    save_steps: 777",
                "optim:",
                "  scheduler:",
                "    warmup_steps: 222",
                "logging:",
                "  backend: none",
                "  wandb:",
                "    enabled: true",
                "    watch: all",
                "  debug:",
                "    metrics: true",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert int(cfg.train.warmup_steps) == 222
    assert int(cfg.train.save_steps) == 777
    assert str(cfg.train.report_to) == "wandb"
    assert str(cfg.train.wandb_watch) == "all"
    assert bool(cfg.train.debug_metrics) is True


def test_load_config_rejects_string_boolean_for_data_streaming(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "bad_bool.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "    streaming: 'false'",
                "train:",
                "  max_steps: 1",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="data.source.streaming must be a boolean"):
        load_config(cfg_path)


def test_load_config_rejects_string_boolean_for_token_weighted_gradient_accumulation(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "bad_bool_train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 1",
                "  token_weighted_gradient_accumulation: 'false'",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train.token_weighted_gradient_accumulation must be a boolean"):
        load_config(cfg_path)


def test_apply_dotted_override_supports_nested_section_paths() -> None:
    cfg = Config(
        data=DataConfig(source={"dataset_name": "HuggingFaceFW/fineweb-edu"}),
        train=TrainConfig(max_steps=1),
    )
    cfg2 = apply_dotted_override(cfg, "logging.wandb.watch=all")
    cfg2 = apply_dotted_override(cfg2, "optim.scheduler.warmup_steps=123")
    assert cfg2.logging.wandb.watch == "all"
    assert int(cfg2.optim.scheduler.warmup_steps) == 123
    assert int(cfg2.train.warmup_steps) == 123


def test_apply_dotted_override_preserves_existing_explicit_fields_per_section() -> None:
    cfg = Config()
    cfg = apply_dotted_override(cfg, "train.objective.mask_token_prob=0.8")
    cfg = apply_dotted_override(cfg, "train.max_steps=20")

    explicit_train_fields = set(getattr(cfg.train, "_explicit_fields", set()))
    assert "objective.mask_token_prob" in explicit_train_fields
    assert "max_steps" in explicit_train_fields

    apply_profile_defaults(model_cfg=cfg.model, train_cfg=cfg.train, optim_cfg=cfg.optim)
    assert cfg.train.mask_token_prob == pytest.approx(0.8)


def test_load_model_config_snapshot_rejects_unknown_legacy_key() -> None:
    with pytest.raises(ValueError, match="Unsupported model_config.json keys"):
        load_model_config_snapshot(
            {"backbone_type": "rope", "legacy_field": 1},
            source="model_config.json",
        )


def test_load_data_config_snapshot_rejects_unknown_legacy_key() -> None:
    with pytest.raises(ValueError, match="Unsupported data_config.json keys"):
        load_data_config_snapshot(
            {"dataset_name": "HuggingFaceFW/fineweb-edu", "legacy_field": 1},
            source="data_config.json",
        )


def test_load_model_config_snapshot_rejects_missing_required_key() -> None:
    model_raw = asdict(ModelConfig())
    model_raw.pop("backbone_type")
    with pytest.raises(ValueError, match="Missing required model_config.json keys"):
        load_model_config_snapshot(model_raw, source="model_config.json")


def test_load_data_config_snapshot_rejects_missing_required_key() -> None:
    data_raw = asdict(DataConfig())
    data_raw.pop("source")
    with pytest.raises(ValueError, match="Missing required data_config.json keys"):
        load_data_config_snapshot(data_raw, source="data_config.json")


def test_prepare_output_dir_respects_overwrite_and_resume(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    keep = out / "existing.txt"
    keep.write_text("x", encoding="utf-8")

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
    assert out.exists() and (not any(out.iterdir()))


@pytest.mark.parametrize("resume_hint", [None, "   "], ids=["unset", "blank"])
def test_prepare_output_dir_rejects_nonempty_without_overwrite_or_resume(
    tmp_path: Path, resume_hint: str | None
):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "existing.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Output directory exists and is not empty"):
        _prepare_output_dir(
            output_dir=out,
            overwrite_output_dir=False,
            resume_from_checkpoint=resume_hint,
            is_main_process=True,
        )


def test_find_latest_checkpoint_picks_highest_step(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoint-2").mkdir()
    (out / "checkpoint-11").mkdir()
    (out / "checkpoint-4").mkdir()

    latest = _find_latest_checkpoint(out)
    assert latest is not None
    assert latest.name == "checkpoint-11"


def test_resolve_output_dir_auto_uses_project_and_config_stem():
    out = _resolve_output_dir(
        output_dir=None,
        project_name="deberta-train",
        config_path="configs/pretrain_rope_fineweb_edu.yaml",
    )
    assert out.parent.name == "deberta-train"
    assert re.fullmatch(r"\d{8}_\d{6}_pretrain_rope_fineweb_edu", out.name) is not None


def test_resolve_output_dir_auto_prefers_run_name():
    out = _resolve_output_dir(
        output_dir=None,
        project_name="deberta-train",
        config_path="configs/pretrain_rope_fineweb_edu.yaml",
        run_name="my run",
    )
    assert out.parent.name == "deberta-train"
    assert re.fullmatch(r"\d{8}_\d{6}_my-run", out.name) is not None


def test_resolve_output_dir_keeps_explicit_path():
    out = _resolve_output_dir(
        output_dir="runs/custom/run-01",
        project_name="ignored",
        config_path="configs/pretrain_rope_fineweb_edu.yaml",
    )
    assert out == Path("runs/custom/run-01")


def _accel_stub(*, is_main_process: bool, num_processes: int) -> Any:
    """Return a minimal accelerator-like object for broadcast helper tests."""
    return types.SimpleNamespace(is_main_process=bool(is_main_process), num_processes=int(num_processes))


def test_resolve_output_dir_for_accelerator_keeps_explicit_path():
    called = {"count": 0}

    def _fake_broadcast(payload: list[str | None], *, from_process: int = 0) -> None:
        del payload, from_process
        called["count"] += 1

    out = _resolve_output_dir_for_accelerator(
        accelerator=_accel_stub(is_main_process=False, num_processes=8),
        output_dir="runs/custom/run-02",
        project_name="ignored",
        config_path="cfg.yaml",
        broadcast_fn=_fake_broadcast,
    )
    assert out == Path("runs/custom/run-02")
    assert called["count"] == 0


def test_resolve_output_dir_for_accelerator_uses_broadcasted_auto_value():
    def _fake_broadcast(payload: list[str | None], *, from_process: int = 0) -> None:
        assert from_process == 0
        payload[0] = "runs/demo/20260101_010101_shared"

    out = _resolve_output_dir_for_accelerator(
        accelerator=_accel_stub(is_main_process=False, num_processes=2),
        output_dir=None,
        project_name="demo",
        config_path="cfg.yaml",
        broadcast_fn=_fake_broadcast,
    )
    assert out == Path("runs/demo/20260101_010101_shared")


def test_resolve_resume_checkpoint_for_accelerator_uses_rank0_broadcast_value(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)

    def _fake_broadcast(payload: list[dict[str, Any]], *, from_process: int = 0) -> None:
        assert from_process == 0
        payload[0] = {
            "ok": True,
            "value": str((out / "checkpoint-9").resolve()),
            "error_type": None,
            "error_message": None,
        }

    resolved = _resolve_resume_checkpoint_for_accelerator(
        accelerator=_accel_stub(is_main_process=False, num_processes=4),
        output_dir=out,
        resume_from_checkpoint="auto",
        broadcast_fn=_fake_broadcast,
    )
    assert resolved == str((out / "checkpoint-9").resolve())


def test_resolve_resume_checkpoint_for_accelerator_propagates_rank0_error(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)

    def _fake_broadcast(payload: list[dict[str, Any]], *, from_process: int = 0) -> None:
        assert from_process == 0
        payload[0] = {
            "ok": False,
            "value": None,
            "error_type": "ValueError",
            "error_message": "rank0 failed",
        }

    with pytest.raises(ValueError, match="rank0 failed"):
        _resolve_resume_checkpoint_for_accelerator(
            accelerator=_accel_stub(is_main_process=False, num_processes=2),
            output_dir=out,
            resume_from_checkpoint="auto",
            broadcast_fn=_fake_broadcast,
        )


def test_resolve_resume_checkpoint_auto_returns_none_when_no_checkpoint(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_resume_checkpoint(
        output_dir=out,
        resume_from_checkpoint="auto",
        is_main_process=True,
    )
    assert resolved is None


def test_resolve_resume_checkpoint_auto_rejects_non_empty_output_dir_without_checkpoints(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "stale.txt").write_text("leftover", encoding="utf-8")

    with pytest.raises(ValueError, match="resume_from_checkpoint=auto was requested"):
        _resolve_resume_checkpoint(
            output_dir=out,
            resume_from_checkpoint="auto",
            is_main_process=True,
        )


@pytest.mark.parametrize(
    ("case", "exc_type", "match"),
    [
        ("missing_path", FileNotFoundError, "checkpoint path does not exist"),
        ("not_directory", ValueError, "must point to a checkpoint directory"),
        ("missing_complete", ValueError, "missing .complete marker"),
        ("missing_data_state", ValueError, "missing/invalid data_state.json"),
        ("missing_weights", ValueError, "model weights appear missing or empty"),
    ],
)
def test_resolve_resume_checkpoint_rejects_invalid_explicit_path_states(
    tmp_path: Path,
    mock_checkpoint: Any,
    case: str,
    exc_type: type[Exception],
    match: str,
):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    if case == "missing_path":
        target = out / "checkpoint-missing"
    elif case == "not_directory":
        target = out / "not-a-dir.txt"
        target.write_text("x", encoding="utf-8")
    elif case == "missing_complete":
        target = mock_checkpoint(
            root=out,
            name="checkpoint-2",
            with_weights=True,
            with_data_state=True,
            with_complete=False,
        )
    elif case == "missing_data_state":
        target = mock_checkpoint(
            root=out,
            name="checkpoint-2",
            with_weights=True,
            with_data_state=False,
            with_complete=True,
        )
    elif case == "missing_weights":
        target = mock_checkpoint(
            root=out,
            name="checkpoint-2",
            with_weights=False,
            with_data_state=True,
            with_complete=True,
        )
    else:  # pragma: no cover
        raise AssertionError(f"Unsupported case: {case}")

    with pytest.raises(exc_type, match=match):
        _resolve_resume_checkpoint(
            output_dir=out,
            resume_from_checkpoint=str(target),
            is_main_process=True,
        )


def test_resolve_resume_checkpoint_returns_resolved_explicit_path(tmp_path: Path, mock_checkpoint: Any):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = mock_checkpoint(root=out, name="checkpoint-2", consumed_micro_batches=2)
    resolved = _resolve_resume_checkpoint(
        output_dir=out,
        resume_from_checkpoint=str(ckpt),
        is_main_process=True,
    )
    assert resolved == str(ckpt.resolve())


def test_resolve_resume_checkpoint_auto_skips_latest_non_resumable_checkpoint(
    tmp_path: Path, mock_checkpoint: Any
):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)

    ckpt1 = mock_checkpoint(root=out, name="checkpoint-1", consumed_micro_batches=10)

    # Simulate interrupted checkpoint write: directory exists but metadata was never written.
    ckpt2 = out / "checkpoint-2"
    ckpt2.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_resume_checkpoint(
        output_dir=out,
        resume_from_checkpoint="auto",
        is_main_process=True,
    )
    assert resolved == str(ckpt1.resolve())


def test_resolve_resume_checkpoint_auto_rejects_when_all_checkpoints_non_resumable(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoint-1").mkdir(parents=True, exist_ok=True)
    (out / "checkpoint-2").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="none are resumable"):
        _resolve_resume_checkpoint(
            output_dir=out,
            resume_from_checkpoint="auto",
            is_main_process=True,
        )


def test_checkpoint_data_progress_roundtrip(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir(parents=True, exist_ok=True)

    consumed, lr_mult, digest = _load_checkpoint_data_progress(ckpt)
    assert consumed is None
    assert lr_mult == 1.0
    assert digest is None

    _save_checkpoint_data_progress(checkpoint_dir=ckpt, consumed_micro_batches=123, lr_mult=0.25)
    consumed, lr_mult, digest = _load_checkpoint_data_progress(ckpt)
    assert consumed == 123
    assert abs(lr_mult - 0.25) < 1e-9
    assert digest is None  # no digest was saved

    # With optimizer param digest.
    _save_checkpoint_data_progress(
        checkpoint_dir=ckpt,
        consumed_micro_batches=200,
        lr_mult=0.5,
        optimizer_param_digest="abc123deadbeef00",
        global_step=17,
        gradient_accumulation_steps=4,
    )
    consumed, lr_mult, digest = _load_checkpoint_data_progress(ckpt)
    assert consumed == 200
    assert abs(lr_mult - 0.5) < 1e-9
    assert digest == "abc123deadbeef00"
    _, _, _, global_step, saved_ga = _load_checkpoint_progress_metadata(ckpt)
    assert global_step == 17
    assert saved_ga == 4

    # Back-compat: old checkpoints without lr_mult or digest default gracefully.
    import json

    (ckpt / "data_state.json").write_text(json.dumps({"consumed_micro_batches": 50}))
    consumed_old, lr_mult_old, digest_old = _load_checkpoint_data_progress(ckpt)
    assert consumed_old == 50
    assert lr_mult_old == 1.0
    assert digest_old is None


def test_checkpoint_data_progress_roundtrip_with_dual_optimizer_digest(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-12"
    ckpt.mkdir(parents=True, exist_ok=True)

    dual_digest = {"generator": "aaaabbbbccccdddd", "discriminator": "1111222233334444"}
    _save_checkpoint_data_progress(
        checkpoint_dir=ckpt,
        consumed_micro_batches=77,
        lr_mult=0.75,
        optimizer_param_digest=dual_digest,
        global_step=9,
        gradient_accumulation_steps=3,
    )

    consumed, lr_mult, digest = _load_checkpoint_data_progress(ckpt)
    assert consumed == 77
    assert lr_mult == pytest.approx(0.75)
    assert isinstance(digest, dict)
    assert digest == dual_digest

    _, _, digest_meta, saved_step, saved_ga = _load_checkpoint_progress_metadata(ckpt)
    assert isinstance(digest_meta, dict)
    assert digest_meta == dual_digest
    assert saved_step == 9
    assert saved_ga == 3


def test_load_checkpoint_data_progress_warns_on_invalid_json(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ckpt = tmp_path / "checkpoint-7"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "data_state.json").write_text('{"consumed_micro_batches": ', encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        consumed, lr_mult, digest = _load_checkpoint_data_progress(ckpt)

    assert consumed is None
    assert lr_mult == 1.0
    assert digest is None
    assert any("invalid data_state.json" in record.message for record in caplog.records)


def test_dump_json_is_atomic_on_serialization_failure(tmp_path: Path) -> None:
    from deberta.utils.io import dump_json

    target = tmp_path / "state.json"
    with pytest.raises(TypeError):
        dump_json({"bad": {1, 2, 3}}, target)

    assert not target.exists()
    tmp_files = list(tmp_path.glob(".*state.json.*.tmp"))
    assert not tmp_files


def test_optimizer_param_order_digest_deterministic() -> None:
    """Same model produces the same digest across calls."""
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    d1 = _optimizer_param_order_digest(m)
    d2 = _optimizer_param_order_digest(m)
    assert d1 == d2
    assert len(d1) == 16  # 16-char hex prefix


def test_optimizer_param_order_digest_changes_on_different_params() -> None:
    """Different parameter names produce a different digest."""
    m1 = torch.nn.ModuleDict({"alpha": torch.nn.Linear(4, 4), "beta": torch.nn.Linear(4, 2)})
    m2 = torch.nn.ModuleDict({"gamma": torch.nn.Linear(4, 4), "beta": torch.nn.Linear(4, 2)})
    d1 = _optimizer_param_order_digest(m1)
    d2 = _optimizer_param_order_digest(m2)
    assert d1 != d2, "Different param names must produce different digest"


def test_optimizer_param_order_digest_ignores_frozen_params() -> None:
    """Frozen parameters are excluded from the digest."""
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    d_all = _optimizer_param_order_digest(m)
    m[0].weight.requires_grad_(False)
    m[0].bias.requires_grad_(False)
    d_partial = _optimizer_param_order_digest(m)
    assert d_all != d_partial, "Freezing params changes the trainable param set and digest"


def test_partition_optimizer_params_deduplicates_shared_parameters() -> None:
    class _SharedRTDLike(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = torch.nn.Module()
            self.discriminator = torch.nn.Module()
            shared = torch.nn.Parameter(torch.randn(3, 3))
            self.generator.shared = shared
            self.discriminator.shared = shared

    model = _SharedRTDLike()
    parts = _partition_optimizer_params(model)
    ordered = ("gen_decay", "gen_no_decay", "disc_decay", "disc_no_decay")
    params = [p for key in ordered for p in parts[key]["params"]]
    names = [n for key in ordered for n in parts[key]["names"]]

    assert len(params) == 1
    assert len(names) == 1
    assert len({id(p) for p in params}) == 1


def test_optimizer_param_order_digest_matches_optimizer_group_insertion_order() -> None:
    model = TinyRTDLikeModel()
    cfg = TrainConfig()
    opt = _build_optimizer(model, cfg)

    param_to_name = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    ordered_names: list[str] = []
    for group in opt.param_groups:
        ordered_names.extend(param_to_name[id(p)] for p in group["params"])
    expected = hashlib.sha256("\n".join(ordered_names).encode()).hexdigest()[:16]

    assert _optimizer_param_order_digest(model) == expected
    assert str(opt._param_order_digest) == expected


def test_build_decoupled_optimizers_uses_branch_lrs_and_tracks_digests() -> None:
    model = TinyRTDLikeModel()
    cfg = TrainConfig(
        learning_rate=5.0e-4,
        generator_learning_rate=2.5e-4,
        weight_decay=0.01,
        mixed_precision="no",
    )
    gen_opt, disc_opt = _build_decoupled_optimizers(model, cfg, mixed_precision="no")

    assert gen_opt.param_groups
    assert disc_opt.param_groups
    for group in gen_opt.param_groups:
        assert float(group["lr"]) == pytest.approx(2.5e-4)
    for group in disc_opt.param_groups:
        assert float(group["lr"]) == pytest.approx(5.0e-4)

    assert isinstance(getattr(gen_opt, "_param_order_digest", ""), str)
    assert isinstance(getattr(disc_opt, "_param_order_digest", ""), str)
    assert len(str(gen_opt._param_order_digest)) == 16
    assert len(str(disc_opt._param_order_digest)) == 16
    assert str(gen_opt._param_order_digest) != str(disc_opt._param_order_digest)


def test_build_decoupled_optimizers_assigns_enhanced_mask_decoder_to_generator_optimizer() -> None:
    class _ModelWithEmd(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = torch.nn.Linear(8, 8)
            self.generator_lm_head = torch.nn.Linear(8, 8)
            self.enhanced_mask_decoder = torch.nn.Linear(8, 8)
            self.discriminator = torch.nn.Linear(8, 8)

    model = _ModelWithEmd()
    cfg = TrainConfig(
        learning_rate=5.0e-4,
        generator_learning_rate=2.5e-4,
        weight_decay=0.01,
        mixed_precision="no",
    )
    gen_opt, disc_opt = _build_decoupled_optimizers(model, cfg, mixed_precision="no")

    gen_param_ids = {id(p) for group in gen_opt.param_groups for p in group["params"]}
    disc_param_ids = {id(p) for group in disc_opt.param_groups for p in group["params"]}
    emd_param_ids = {id(p) for p in model.enhanced_mask_decoder.parameters() if p.requires_grad}

    assert emd_param_ids
    assert emd_param_ids.issubset(gen_param_ids)
    assert emd_param_ids.isdisjoint(disc_param_ids)


def test_save_training_checkpoint_persists_optimizer_digest(tmp_path: Path):
    """_save_training_checkpoint forwards optimizer_param_digest to data_state.json."""
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint-5"

    accel = _checkpoint_saving_accelerator(is_main_process=True)
    _save_training_checkpoint(
        accelerator=accel,
        checkpoint_dir=ckpt,
        output_dir=out,
        consumed_micro_batches=10,
        save_total_limit=3,
        log_label="test",
        optimizer_param_digest="deadbeef12345678",
    )
    _, _, digest = _load_checkpoint_data_progress(ckpt)
    assert digest == "deadbeef12345678"


def test_save_training_checkpoint_persists_dual_optimizer_digest(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint-6"

    dual_digest = {"generator": "feedfacecafebeef", "discriminator": "baadf00d12345678"}
    accel = _checkpoint_saving_accelerator(is_main_process=True)
    _save_training_checkpoint(
        accelerator=accel,
        checkpoint_dir=ckpt,
        output_dir=out,
        consumed_micro_batches=15,
        save_total_limit=3,
        log_label="test",
        optimizer_param_digest=dual_digest,
    )
    _, _, digest = _load_checkpoint_data_progress(ckpt)
    assert isinstance(digest, dict)
    assert digest == dual_digest


def test_canonical_compile_state_key_strips_orig_mod_segments() -> None:
    assert canonical_compile_state_key("generator._orig_mod.encoder.layer.0._orig_mod.weight") == (
        "generator.encoder.layer.0.weight"
    )
    assert canonical_compile_state_key("_orig_mod.generator._orig_mod.encoder.weight") == (
        "generator.encoder.weight"
    )


def test_load_model_state_with_compile_key_remap_matches_checkpoint_with_orig_mod_keys(
    tmp_path: Path,
) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model[0].weight.fill_(1.5)
        model[0].bias.fill_(-0.25)

    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    remapped = {key.replace("0.", "0._orig_mod."): value.detach().clone() for key, value in original.items()}
    torch.save(remapped, checkpoint / "model.bin")

    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()

    stats = load_model_state_with_compile_key_remap(model, checkpoint)
    assert stats == {"matched": 2}
    assert torch.allclose(model[0].weight, original["0.weight"])
    assert torch.allclose(model[0].bias, original["0.bias"])


def test_load_model_state_with_compile_key_remap_matches_top_level_orig_mod_keys(
    tmp_path: Path,
) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model[0].weight.fill_(0.5)
        model[0].bias.fill_(0.125)

    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    remapped = {f"_orig_mod.{key}": value.detach().clone() for key, value in original.items()}
    torch.save(remapped, checkpoint / "model.bin")

    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()

    stats = load_model_state_with_compile_key_remap(model, checkpoint)
    assert stats == {"matched": 2}
    assert torch.allclose(model[0].weight, original["0.weight"])
    assert torch.allclose(model[0].bias, original["0.bias"])


def test_load_model_state_with_compile_key_remap_supports_fsdp_sharded_checkpoint(
    tmp_path: Path,
) -> None:
    dcp = pytest.importorskip("torch.distributed.checkpoint")

    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    checkpoint = tmp_path / "checkpoint-1"
    shard_dir = checkpoint / "pytorch_model_fsdp_0"
    shard_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model[0].weight.fill_(1.25)
        model[0].bias.fill_(-0.75)

    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sharded_saved = {k.replace("0.", "0._orig_mod."): v.clone() for k, v in saved.items()}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to save in a single process.",
            category=UserWarning,
        )
        dcp.save(
            state_dict={"model": sharded_saved},
            checkpoint_id=str(shard_dir),
            no_dist=True,
        )

    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process.",
            category=UserWarning,
        )
        loaded_direct = load_checkpoint_model_state_dict(checkpoint, model_state_template=model.state_dict())
    assert set(loaded_direct.keys()) == set(sharded_saved.keys())

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process.",
            category=UserWarning,
        )
        stats = load_model_state_with_compile_key_remap(model, checkpoint)
    assert stats == {"matched": 2}
    assert torch.allclose(model[0].weight, saved["0.weight"])
    assert torch.allclose(model[0].bias, saved["0.bias"])


def test_load_resume_state_with_compile_fallback_retries_strict_false_and_remaps(
    tmp_path: Path,
) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model[0].weight.fill_(0.75)
        model[0].bias.fill_(0.5)
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(
        {k.replace("0.", "0._orig_mod."): v.clone() for k, v in saved.items()},
        checkpoint / "model.bin",
    )

    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()

    calls: list[tuple[str, dict[str, Any]]] = []

    def _load_state_hook(ckpt: str, kwargs: dict[str, Any]) -> None:
        calls.append((str(ckpt), dict(kwargs)))
        if kwargs.get("strict", True):
            raise RuntimeError("Error(s) in loading state_dict with _orig_mod mismatch")

    accel = FakeAccelerator(load_state_hook=_load_state_hook)
    load_state_with_compile_fallback(
        accelerator=accel,
        model=model,
        checkpoint_dir=str(checkpoint),
        context="resume",
    )

    assert len(calls) == 2
    assert calls[0][1] == {}
    assert calls[1][1] == {"strict": False}
    assert torch.allclose(model[0].weight, saved["0.weight"])
    assert torch.allclose(model[0].bias, saved["0.bias"])


def test_append_metrics_jsonl_row_appends_rows(tmp_path: Path):
    metrics_path = tmp_path / "run" / "metrics.jsonl.gz"
    _append_metrics_jsonl_row(metrics_path, {"step": 1, "loss": 0.123})
    _append_metrics_jsonl_row(metrics_path, {"step": 2, "crash": True, "crash_type": "KeyboardInterrupt"})

    with gzip.open(metrics_path, "rt", encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    row1 = json.loads(lines[1])
    assert row0["step"] == 1
    assert row1["step"] == 2
    assert row1["crash"] is True


def test_flush_loggers_suppresses_handler_flush_errors() -> None:
    class _CountingHandler(logging.Handler):
        def __init__(self, *, raise_on_flush: bool) -> None:
            super().__init__()
            self.raise_on_flush = bool(raise_on_flush)
            self.flush_calls = 0

        def emit(self, record: logging.LogRecord) -> None:
            del record

        def flush(self) -> None:
            self.flush_calls += 1
            if self.raise_on_flush:
                raise RuntimeError("boom")

    test_logger = logging.getLogger("deberta.tests.flush")
    test_logger.setLevel(logging.INFO)
    ok_handler = _CountingHandler(raise_on_flush=False)
    bad_handler = _CountingHandler(raise_on_flush=True)
    test_logger.addHandler(ok_handler)
    test_logger.addHandler(bad_handler)
    try:
        _flush_loggers()
    finally:
        test_logger.removeHandler(ok_handler)
        test_logger.removeHandler(bad_handler)

    assert ok_handler.flush_calls >= 1
    assert bad_handler.flush_calls >= 1


def test_init_trackers_passes_wandb_name_with_wrapped_signature() -> None:
    class _WrappedAccelerator:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def init_trackers(self, *args, **kwargs) -> None:
            del args
            self.calls.append(dict(kwargs))

    accel = _WrappedAccelerator()
    _init_trackers(
        accelerator=accel,
        project_name="demo-project",
        tracker_cfg={"a": 1},
        report_to="wandb",
        run_name="demo-run",
    )

    assert accel.calls
    first = accel.calls[0]
    assert first["project_name"] == "demo-project"
    assert first["init_kwargs"]["wandb"]["name"] == "demo-run"


def test_init_trackers_falls_back_without_init_kwargs(caplog: pytest.LogCaptureFixture) -> None:
    class _LegacyAccelerator:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def init_trackers(self, *, project_name: str, config: dict[str, Any]) -> None:
            self.calls.append({"project_name": project_name, "config": dict(config)})

    accel = _LegacyAccelerator()
    with caplog.at_level(logging.WARNING):
        _init_trackers(
            accelerator=accel,
            project_name="demo-project",
            tracker_cfg={"a": 1},
            report_to="wandb",
            run_name="demo-run",
        )

    assert accel.calls
    assert "rejected init_kwargs" in caplog.text


def test_setup_wandb_watch_calls_watch_with_mode_and_frequency() -> None:
    model = torch.nn.Linear(4, 4)
    run = FakeWandbRun()
    enabled = _setup_wandb_watch(
        accelerator=FakeAccelerator(),
        wandb_run=run,
        model=model,
        watch_mode="gradients",
        watch_log_freq=123,
    )
    assert enabled is True
    assert run.watch_calls
    watched_model, kwargs = run.watch_calls[0]
    assert watched_model is model
    assert kwargs["log"] == "gradients"
    assert kwargs["log_freq"] == 123


def test_setup_wandb_watch_returns_false_when_mode_none() -> None:
    run = FakeWandbRun()
    enabled = _setup_wandb_watch(
        accelerator=FakeAccelerator(),
        wandb_run=run,
        model=torch.nn.Linear(2, 2),
        watch_mode="none",
        watch_log_freq=100,
    )
    assert enabled is False
    assert run.watch_calls == []


def test_upload_wandb_original_config_stages_with_expected_filename(tmp_path: Path) -> None:
    src = tmp_path / "config_original.yaml"
    src.write_text("train:\n  max_steps: 1\n", encoding="utf-8")

    run = FakeWandbRun()
    uploaded = _upload_wandb_original_config(
        accelerator=types.SimpleNamespace(is_main_process=True),
        wandb_run=run,
        config_original_path=src,
        run_name="demo-run",
    )
    assert uploaded is True
    assert run.saved_paths
    assert run.saved_paths[0].name == "config_original.yaml"
    assert run.saved_paths[0] == src


def test_upload_wandb_original_config_uploads_resolved_and_source_files(tmp_path: Path) -> None:
    src_original = tmp_path / "config_original.yaml"
    src_original.write_text("model:\n  backbone_type: hf_deberta_v2\n", encoding="utf-8")
    src_resolved = tmp_path / "config_resolved.yaml"
    src_resolved.write_text(
        "model:\n  backbone_type: hf_deberta_v2\ntrain:\n  warmup_steps: 10000\n", encoding="utf-8"
    )
    src_source = tmp_path / "passed.yaml"
    src_source.write_text("model:\n  profile: deberta_v3_parity\n", encoding="utf-8")

    run = FakeWandbRun()
    uploaded = _upload_wandb_original_config(
        accelerator=types.SimpleNamespace(is_main_process=True),
        wandb_run=run,
        config_original_path=src_original,
        config_resolved_path=src_resolved,
        config_source_path=src_source,
        run_name="demo-run",
    )
    assert uploaded is True
    saved_names = {p.name for p in run.saved_paths}
    assert "config_original.yaml" not in saved_names
    assert "config_resolved.yaml" in saved_names
    assert "passed.yaml" in saved_names


def test_coerce_dataclass_payload_types_accepts_mapping_inputs() -> None:
    payload = {
        "alpha": 1,
        "nested": {"beta": True},
        3: "non-string-key",
    }
    coerced = _coerce_dataclass_payload_types(payload)
    assert coerced == {
        "alpha": 1,
        "nested": {"beta": True},
        "3": "non-string-key",
    }


def test_build_runtime_resolved_tracker_config_populates_effective_values_and_prunes_none() -> None:
    model_cfg = ModelConfig(
        profile="deberta_v3_parity",
        backbone_type="hf_deberta_v2",
        pretrained_discriminator_path="microsoft/deberta-v3-base",
        generator_num_hidden_layers=None,
        hidden_dropout_prob=None,
        attention_probs_dropout_prob=None,
        tokenizer_vocab_target=None,
    )
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig(
        learning_rate=5e-4,
        generator_learning_rate=-1.0,
        discriminator_learning_rate=-1.0,
    )
    optim_cfg = OptimConfig(
        learning_rate=5e-4, generator_learning_rate=-1.0, discriminator_learning_rate=-1.0
    )
    logging_cfg = LoggingConfig()
    disc_cfg = types.SimpleNamespace(
        num_hidden_layers=12,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        to_dict=lambda: {
            "num_hidden_layers": 12,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
        },
    )
    gen_cfg = types.SimpleNamespace(
        num_hidden_layers=6,
        hidden_size=384,
        intermediate_size=1536,
        num_attention_heads=6,
        to_dict=lambda: {
            "num_hidden_layers": 6,
            "hidden_size": 384,
            "intermediate_size": 1536,
            "num_attention_heads": 6,
        },
    )
    tokenizer = DummyTokenizer(vocab_size=32000)

    payload = _build_runtime_resolved_tracker_config(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        optim_cfg=optim_cfg,
        logging_cfg=logging_cfg,
        disc_config=disc_cfg,
        gen_config=gen_cfg,
        tokenizer=tokenizer,
    )

    assert payload["model"]["generator"]["num_hidden_layers"] == 6
    assert payload["model"]["generator"]["hidden_size"] == 384
    assert payload["model"]["generator"]["intermediate_size"] == 1536
    assert payload["model"]["generator"]["num_attention_heads"] == 6
    assert payload["model"]["dropout"]["hidden_prob"] == pytest.approx(0.1)
    assert payload["model"]["dropout"]["attention_probs_prob"] == pytest.approx(0.1)
    assert payload["model"]["tokenizer"]["vocab_target"] == 32000
    assert payload["optim"]["lr"]["base"] == pytest.approx(5e-4)
    assert payload["model"]["pretrained"]["discriminator_path"] == "microsoft/deberta-v3-base"
    assert "generator_path" not in payload["model"]["pretrained"]
    assert "resume_from_checkpoint" not in payload["train"]["checkpoint"]
    assert set(payload.keys()) == {"model", "data", "train", "optim", "logging"}
    assert "effective" not in payload


def test_build_runtime_resolved_tracker_config_omits_effective_backbone_payload() -> None:
    model_cfg = ModelConfig(profile="deberta_v3_parity", backbone_type="hf_deberta_v2")
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu")
    train_cfg = TrainConfig()
    disc_cfg = types.SimpleNamespace(
        to_dict=lambda: {
            "model_type": "deberta-v2",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "vocab_size": 32000,
            "max_position_embeddings": 1024,
            "pad_token_id": 3,
            "max_length": 20,
            "top_k": 50,
            "id2label": {"0": "LABEL_0"},
            "_name_or_path": "",
        },
    )
    gen_cfg = types.SimpleNamespace(
        to_dict=lambda: {
            "model_type": "deberta-v2",
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "vocab_size": 32000,
            "max_position_embeddings": 1024,
            "pad_token_id": 3,
            "max_length": 20,
            "top_k": 50,
            "id2label": {"0": "LABEL_0"},
            "_name_or_path": "",
        },
    )
    tokenizer = DummyTokenizer(vocab_size=32000)

    payload = _build_runtime_resolved_tracker_config(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        disc_config=disc_cfg,
        gen_config=gen_cfg,
        tokenizer=tokenizer,
    )

    assert set(payload.keys()) == {"model", "data", "train", "optim", "logging"}
    assert "effective" not in payload


def test_build_runtime_resolved_tracker_config_coerces_numeric_strings() -> None:
    model_cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        hidden_size="768",  # type: ignore[arg-type]
        num_hidden_layers="12",  # type: ignore[arg-type]
        num_attention_heads="12",  # type: ignore[arg-type]
        intermediate_size="3072",  # type: ignore[arg-type]
        hidden_dropout_prob="0.0",  # type: ignore[arg-type]
        attention_probs_dropout_prob="0.0",  # type: ignore[arg-type]
    )
    data_cfg = DataConfig(dataset_name="HuggingFaceFW/fineweb-edu", max_seq_length="1024")  # type: ignore[arg-type]
    train_cfg = TrainConfig(
        token_weighted_gradient_accumulation="true",  # type: ignore[arg-type]
    )
    optim_cfg = OptimConfig(
        learning_rate="5e-4",  # type: ignore[arg-type]
        adam_epsilon="1e-6",  # type: ignore[arg-type]
        warmup_steps="1000",  # type: ignore[arg-type]
    )
    disc_cfg = types.SimpleNamespace(
        to_dict=lambda: {
            "num_hidden_layers": 12,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 1024,
        },
    )
    gen_cfg = types.SimpleNamespace(
        to_dict=lambda: {
            "num_hidden_layers": 6,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
        },
    )
    tokenizer = DummyTokenizer(vocab_size=32000)

    payload = _build_runtime_resolved_tracker_config(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        optim_cfg=optim_cfg,
        disc_config=disc_cfg,
        gen_config=gen_cfg,
        tokenizer=tokenizer,
    )

    assert payload["model"]["rope"]["hidden_size"] == 768
    assert payload["data"]["packing"]["max_seq_length"] == 1024
    assert payload["optim"]["scheduler"]["warmup_steps"] == 1000
    assert payload["optim"]["lr"]["base"] == pytest.approx(5e-4)
    assert payload["optim"]["adam"]["epsilon"] == pytest.approx(1e-6)
    assert payload["train"]["token_weighted_gradient_accumulation"] is True


def test_run_pretraining_keyboard_interrupt_logs_crash_and_finishes_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes import _PRETRAINING_BATCH

    saved_checkpoints: list[tuple[str, int, str]] = []

    def _fake_save_checkpoint(
        *,
        accelerator,
        checkpoint_dir,
        output_dir,
        consumed_micro_batches,
        save_total_limit,
        log_label,
        **kwargs,
    ):
        del accelerator, output_dir, save_total_limit, kwargs
        saved_checkpoints.append((str(checkpoint_dir), int(consumed_micro_batches), str(log_label)))

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        accelerator_cls=FakeAccelerator,
        save_checkpoint_fn=_fake_save_checkpoint,
    )

    # Override cycle to interrupt after first batch.
    def _interrupt_cycle(_loader, *, start_epoch: int = 0):
        del start_epoch
        yield _PRETRAINING_BATCH
        raise KeyboardInterrupt

    monkeypatch.setattr(pretrain_mod, "_cycle_dataloader", _interrupt_cycle)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=2,
        save_steps=0,
        report_to="wandb",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        torch_compile=False,
        export_hf_final=False,
    )

    with pytest.raises(KeyboardInterrupt):
        pretrain_mod.run_pretraining(
            model_cfg=ModelConfig(),
            data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
            train_cfg=train_cfg,
        )

    metrics_path = Path(train_cfg.output_dir) / "metrics.jsonl.gz"
    with gzip.open(metrics_path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f.read().splitlines()]
    assert rows and rows[-1]["crash"] is True
    assert rows[-1]["crash_type"] == "KeyboardInterrupt"
    assert int(rows[-1]["step"]) == 1

    assert saved_checkpoints
    assert saved_checkpoints[-1][0].endswith("checkpoint-1")
    assert saved_checkpoints[-1][1] == 1
    assert saved_checkpoints[-1][2] == "final"

    accel = FakeAccelerator.last_instance
    assert accel is not None
    assert accel.wandb_run.summary["crashed"] is True
    assert accel.wandb_run.summary["crash_type"] == "KeyboardInterrupt"
    assert accel.wandb_run.finished_exit_code == 130
    assert accel.wandb_run.logged
    assert accel.wandb_run.watch_calls
    assert accel.wandb_run.saved_paths
    saved_names = {p.name for p in accel.wandb_run.saved_paths}
    assert "config_original.yaml" in saved_names
    assert "config_resolved.yaml" in saved_names
    assert accel.wandb_run.watch_calls[0][1]["log"] == "gradients"
    assert accel.tracker_init_calls
    first_tracker_call = accel.tracker_init_calls[0]
    assert first_tracker_call["project_name"] == "deberta-train"
    assert first_tracker_call["init_kwargs"]["wandb"]["name"] == "run"
    assert accel.ended is False


def test_run_pretraining_logs_crash_save_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from _fakes import _PRETRAINING_BATCH

    def _fake_save_checkpoint(
        *,
        accelerator,
        checkpoint_dir,
        output_dir,
        consumed_micro_batches,
        save_total_limit,
        log_label,
        **kwargs,
    ):
        del accelerator, checkpoint_dir, output_dir, consumed_micro_batches, save_total_limit, kwargs
        if str(log_label) == "final":
            raise RuntimeError("disk full")

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        save_checkpoint_fn=_fake_save_checkpoint,
    )

    def _interrupt_cycle(_loader, *, start_epoch: int = 0):
        del start_epoch
        yield _PRETRAINING_BATCH
        raise KeyboardInterrupt

    monkeypatch.setattr(pretrain_mod, "_cycle_dataloader", _interrupt_cycle)

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
        torch_compile=False,
        export_hf_final=False,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyboardInterrupt):
            pretrain_mod.run_pretraining(
                model_cfg=ModelConfig(),
                data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
                train_cfg=train_cfg,
            )

    assert any("Final/crash-time checkpoint save failed" in rec.message for rec in caplog.records)


def test_run_pretraining_crash_checkpoint_saves_committed_microbatch_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes import _PRETRAINING_BATCH

    saved_checkpoints: list[tuple[str, int, str]] = []

    def _fake_save_checkpoint(
        *,
        accelerator,
        checkpoint_dir,
        output_dir,
        consumed_micro_batches,
        save_total_limit,
        log_label,
        **kwargs,
    ):
        del accelerator, output_dir, save_total_limit, kwargs
        saved_checkpoints.append((str(checkpoint_dir), int(consumed_micro_batches), str(log_label)))

    pretrain_mod = setup_pretraining_mocks(
        monkeypatch,
        save_checkpoint_fn=_fake_save_checkpoint,
    )

    # Complete one accumulation window (2 micro-batches), then interrupt in the next
    # window after fetching one more micro-batch. Final checkpoint should persist only
    # committed-step progress for checkpoint-{global_step}.
    def _interrupt_cycle(_loader, *, start_epoch: int = 0):
        del start_epoch
        yield _PRETRAINING_BATCH
        yield _PRETRAINING_BATCH
        yield _PRETRAINING_BATCH
        raise KeyboardInterrupt

    monkeypatch.setattr(pretrain_mod, "_cycle_dataloader", _interrupt_cycle)

    train_cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=3,
        save_steps=0,
        report_to="none",
        mixed_precision="no",
        tf32=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        torch_compile=False,
        export_hf_final=False,
    )

    with pytest.raises(KeyboardInterrupt):
        pretrain_mod.run_pretraining(
            model_cfg=ModelConfig(),
            data_cfg=DataConfig(dataset_name="hf-internal-testing/librispeech_asr_dummy"),
            train_cfg=train_cfg,
        )

    assert saved_checkpoints
    assert saved_checkpoints[-1][0].endswith("checkpoint-1")
    assert saved_checkpoints[-1][1] == 2
    assert saved_checkpoints[-1][2] == "final"
