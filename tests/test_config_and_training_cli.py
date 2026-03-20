# ruff: noqa: F403,F405
from _config_and_training_shared_imports import *


def test_main_cli_train_subcommand_loads_yaml_and_applies_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "  packing:",
                "    max_seq_length: 32",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(["train", str(cfg_path), "--train.max_steps", "7"])

    assert "train_cfg" in seen
    assert seen["data_cfg"].packing.max_seq_length == 32
    assert seen["train_cfg"].max_steps == 7
    assert seen["config_path"] == cfg_path


def test_main_cli_train_honors_explicit_yaml_warmup_value_for_hf_backbone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "model:",
                "  profile: deberta_v3_parity",
                "  backbone_type: hf_deberta_v2",
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
                "optim:",
                "  scheduler:",
                "    warmup_steps: 1000",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(["train", str(cfg_path)])

    assert "train_cfg" in seen
    assert int(seen["train_cfg"].warmup_steps) == 1000
    assert seen["config_path"] == cfg_path
    err = capsys.readouterr().err
    assert "optim.scheduler.warmup_steps" not in err


def test_main_cli_train_reports_when_file_value_is_changed_by_cli_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(["train", str(cfg_path), "--train.max_steps", "7"])

    assert "train_cfg" in seen
    assert int(seen["train_cfg"].max_steps) == 7
    err = capsys.readouterr().err
    assert "train.max_steps: 5 -> 7 (CLI override (--train.max_steps))" in err


def test_main_cli_train_reports_when_runtime_mutation_changes_loaded_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_apply_profile_defaults(*, model_cfg, train_cfg, optim_cfg):
        del model_cfg, optim_cfg
        object.__setattr__(train_cfg, "max_steps", int(train_cfg.max_steps) + 1)

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "apply_profile_defaults", _fake_apply_profile_defaults)
    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(["train", str(cfg_path)])

    assert int(seen["train_cfg"].max_steps) == 6
    err = capsys.readouterr().err
    assert "train.max_steps: 5 -> 6 (runtime normalization/defaulting)" in err


def test_main_cli_train_supports_dotted_overrides_with_type_casting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
                "optim:",
                "  adam:",
                "    epsilon: 1e-6",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(
        [
            "train",
            str(cfg_path),
            "--train.max_steps",
            "11",
            "--optim.adam.epsilon",
            "1e-5",
            "--model.hf.max_position_embeddings",
            "640",
        ]
    )

    assert int(seen["train_cfg"].max_steps) == 11
    assert float(seen["train_cfg"].adam_epsilon) == pytest.approx(1e-5)
    assert int(seen["model_cfg"].hf_max_position_embeddings) == 640
    err = capsys.readouterr().err
    assert "train.max_steps: 5 -> 11 (CLI override (--train.max_steps))" in err


def test_main_cli_train_supports_null_for_optional_numeric_dotted_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "model:",
                "  hf:",
                "    max_position_embeddings: 640",
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(
        [
            "train",
            str(cfg_path),
            "--model.hf.max_position_embeddings",
            "null",
        ]
    )

    assert seen["model_cfg"].hf_max_position_embeddings is None
    err = capsys.readouterr().err
    assert (
        "model.hf.max_position_embeddings: 640 -> None (CLI override (--model.hf.max_position_embeddings))"
    ) in err


def test_main_cli_train_supports_null_for_optional_constrained_dotted_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "model:",
                "  backbone_type: rope",
                "  from_scratch: false",
                "  pretrained:",
                "    discriminator_path: /tmp/rope-disc",
                "  rope:",
                "    pretrained:",
                "      norm_arch: keel",
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(
        [
            "train",
            str(cfg_path),
            "--model.rope.pretrained.norm_arch",
            "none",
        ]
    )

    assert seen["model_cfg"].pretrained_norm_arch is None
    err = capsys.readouterr().err
    assert (
        "model.rope.pretrained.norm_arch: 'keel' -> None (CLI override (--model.rope.pretrained.norm_arch))"
        in err
    )


def test_main_cli_train_supports_dotted_overrides_for_extended_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("yaml")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  source:",
                "    dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(
        [
            "train",
            str(cfg_path),
            "--optim.scheduler.warmup_steps",
            "333",
            "--logging.wandb.watch",
            "all",
        ]
    )

    assert int(seen["train_cfg"].warmup_steps) == 333
    assert seen["train_cfg"].wandb_watch == "all"


def test_main_cli_train_rejects_invalid_dotted_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("yaml")
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_name: HuggingFaceFW/fineweb-edu",
                "train:",
                "  max_steps: 5",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_mod, "run_pretraining", lambda **_: None)
    with pytest.raises(SystemExit):
        cli_mod.main(["train", str(cfg_path), "--train.no_such_field", "1"])


def test_main_cli_train_with_config_and_preset_applies_model_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    cfg_path = tmp_path / "train.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model": {"backbone_type": "rope"},
                "data": {"source": {"dataset_name": "demo-dataset"}, "packing": {"max_seq_length": 128}},
                "train": {"max_steps": 11},
            }
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(["train", str(cfg_path), "--preset", "deberta-v3-base"])

    assert "train_cfg" in seen
    assert seen["model_cfg"].backbone_type == "hf_deberta_v2"
    assert seen["data_cfg"].source.dataset_name == "demo-dataset"
    assert seen["data_cfg"].packing.max_seq_length == 128
    assert seen["train_cfg"].max_steps == 11
    assert seen["config_path"] == cfg_path
    out = capsys.readouterr().out
    assert "model-only overrides (config file provided)" in out


def test_main_cli_train_preset_without_config_applies_defaults_and_cli_overrides(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    cli_mod.main(["train", "--preset", "deberta-v3-base", "--train.max_steps", "123"])

    assert seen["config_path"] is None
    assert seen["model_cfg"].backbone_type == "hf_deberta_v2"
    assert seen["model_cfg"].embedding_sharing == "gdes"
    assert seen["data_cfg"].source.dataset_name == "HuggingFaceFW/fineweb-edu"
    assert seen["data_cfg"].source.dataset_config_name == "default"
    assert seen["train_cfg"].max_steps == 123
    assert seen["train_cfg"].report_to == "none"
    out = capsys.readouterr().out
    assert "model+data+train+optim+logging defaults (no config file)" in out


def test_main_cli_train_dry_run_calls_preflight_and_skips_training(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    def _fake_run_pretraining(*args, **kwargs):
        del args, kwargs
        raise AssertionError("run_pretraining should not be called for --dry-run")

    def _fake_dry_run(*, model_cfg, data_cfg, train_cfg, optim_cfg, logging_cfg, config_path=None):
        seen["model_cfg"] = model_cfg
        seen["data_cfg"] = data_cfg
        seen["train_cfg"] = train_cfg
        seen["optim_cfg"] = optim_cfg
        seen["logging_cfg"] = logging_cfg
        seen["config_path"] = config_path
        return {
            "checkpoint_output_dir": "runs/demo",
            "logging_output_dir": "runs/demo",
            "resume_checkpoint": None,
            "effective_compile_scope": "ffn",
            "sample_batch_shape": (1, 8),
            "sample_active_tokens": 8.0,
            "tokenizer_vocab_size": 32000,
        }

    monkeypatch.setattr(cli_mod, "run_pretraining", _fake_run_pretraining)
    monkeypatch.setattr(cli_mod, "run_pretraining_dry_run", _fake_dry_run)

    cli_mod.main(
        [
            "train",
            "--data.source.dataset_name",
            "HuggingFaceFW/fineweb-edu",
            "--train.max_steps",
            "5",
            "--dry-run",
        ]
    )
    assert "train_cfg" in seen
    assert int(seen["train_cfg"].max_steps) == 5


def test_main_cli_train_explicit_argv_skips_fast_exit(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {"ran": False, "exit_called": False}

    def _fake_run_train(ns: argparse.Namespace, *, dotflag_map: dict[str, str]) -> None:
        del ns, dotflag_map
        seen["ran"] = True

    def _fake_exit(code: int) -> None:
        del code
        seen["exit_called"] = True
        raise AssertionError("os._exit should not be called when main(argv=...) is used.")

    monkeypatch.setattr(cli_mod, "_run_train", _fake_run_train)
    monkeypatch.setattr(cli_mod.os, "_exit", _fake_exit)

    cli_mod.main(["train", "--dry-run"])

    assert seen["ran"] is True
    assert seen["exit_called"] is False


def test_main_cli_train_implicit_argv_fast_exit_enabled(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {"ran": False}

    def _fake_run_train(ns: argparse.Namespace, *, dotflag_map: dict[str, str]) -> None:
        del ns, dotflag_map
        seen["ran"] = True

    def _fake_exit(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(cli_mod, "_run_train", _fake_run_train)
    monkeypatch.setattr(cli_mod.os, "_exit", _fake_exit)
    monkeypatch.setenv("DEBERTA_FAST_EXIT_AFTER_TRAIN", "1")
    monkeypatch.setattr(cli_mod.sys, "argv", ["deberta", "train", "--dry-run"])

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main()

    assert int(exc_info.value.code) == 0
    assert seen["ran"] is True


def test_train_parser_accepts_dry_run_flag():
    parser = cli_mod._build_main_parser()
    ns = parser.parse_args(
        [
            "train",
            "--data.source.dataset_name",
            "HuggingFaceFW/fineweb-edu",
            "--train.max_steps",
            "5",
            "--dry-run",
        ]
    )
    assert ns.command == "train"
    assert ns.dry_run is True


def test_train_parser_accepts_preset_flag():
    parser = cli_mod._build_main_parser()
    ns = parser.parse_args(["train", "--preset", "deberta-v3-base", "--dry-run"])
    assert ns.command == "train"
    assert ns.preset == "deberta-v3-base"
    assert ns.dry_run is True


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


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--model.rope.norm_arch", "invalid"],
        ["train", "--logging.backend", "invalid"],
    ],
)
def test_train_cli_rejects_invalid_constrained_values_at_parse_time(argv: list[str]):
    parser = cli_mod._build_main_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["runs/demo/checkpoint-10", "--safe-serialization", "--no-safe-serialization"],
        ["runs/demo/checkpoint-10", "--offload-to-cpu", "--no-offload-to-cpu"],
        ["runs/demo/checkpoint-10", "--rank0-only", "--no-rank0-only"],
    ],
)
def test_export_parser_rejects_conflicting_boolean_flags(argv: list[str]):
    parser = argparse.ArgumentParser(prog="deberta export")
    add_export_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_validate_data_config_rejects_conflicting_sources():
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_data_config(
            DataConfig(
                load_from_disk="runs/saved_ds",
                dataset_name="HuggingFaceFW/fineweb-edu",
                streaming=False,
            )
        )


def test_validate_train_config_rejects_overwrite_with_resume_conflict():
    with pytest.raises(ValueError, match="cannot be combined with train.checkpoint.resume_from_checkpoint"):
        validate_train_config(
            TrainConfig(
                overwrite_output_dir=True,
                resume_from_checkpoint="auto",
            )
        )


@pytest.mark.parametrize(
    ("resume_hint", "expected"),
    [("   ", None), (" auto ", "auto")],
    ids=["blank_to_none", "trimmed_value"],
)
def test_validate_train_config_normalizes_resume_hints(resume_hint: str, expected: str | None):
    cfg = TrainConfig(resume_from_checkpoint=resume_hint)
    validate_train_config(cfg)
    assert cfg.checkpoint.resume_from_checkpoint == expected


def test_model_config_default_backbone_is_hf_deberta_v2() -> None:
    cfg = ModelConfig()
    assert cfg.backbone_type == "hf_deberta_v2"


def test_apply_profile_defaults_applies_parity_values_when_unset() -> None:
    model_cfg = ModelConfig(profile="deberta_v3_parity")
    train_cfg = TrainConfig()
    optim_cfg = OptimConfig()

    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg, optim_cfg=optim_cfg)

    assert model_cfg.backbone_type == "hf_deberta_v2"
    assert model_cfg.embedding_sharing == "gdes"
    assert model_cfg.hf_attention_kernel == "dynamic"
    assert train_cfg.mask_token_prob == pytest.approx(1.0)
    assert train_cfg.random_token_prob == pytest.approx(0.0)
    assert train_cfg.mlm_max_ngram == 1
    assert train_cfg.disc_loss_weight == pytest.approx(10.0)
    assert optim_cfg.adam.epsilon == pytest.approx(1e-6)
    assert optim_cfg.scheduler.warmup_steps == 10_000
    assert train_cfg.token_weighted_gradient_accumulation is True


def test_apply_profile_defaults_applies_hf_parity_values_without_parity_profile() -> None:
    model_cfg = ModelConfig(profile="modern", backbone_type="hf_deberta_v2")
    train_cfg = TrainConfig()
    optim_cfg = OptimConfig()

    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg, optim_cfg=optim_cfg)

    assert train_cfg.mask_token_prob == pytest.approx(1.0)
    assert train_cfg.random_token_prob == pytest.approx(0.0)
    assert train_cfg.disc_loss_weight == pytest.approx(10.0)
    assert optim_cfg.adam.epsilon == pytest.approx(1e-6)
    assert optim_cfg.scheduler.warmup_steps == 10_000
    assert train_cfg.token_weighted_gradient_accumulation is True


def test_apply_profile_defaults_keeps_rope_modern_defaults() -> None:
    model_cfg = ModelConfig(profile="modern", backbone_type="rope")
    train_cfg = TrainConfig()
    optim_cfg = OptimConfig()

    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg, optim_cfg=optim_cfg)

    assert train_cfg.mask_token_prob == pytest.approx(0.8)
    assert train_cfg.random_token_prob == pytest.approx(0.1)
    assert train_cfg.disc_loss_weight == pytest.approx(50.0)
    assert optim_cfg.adam.epsilon == pytest.approx(1e-8)
    assert optim_cfg.scheduler.warmup_steps == 1_000
    assert train_cfg.token_weighted_gradient_accumulation is True


def test_apply_profile_defaults_keeps_explicit_non_default_values() -> None:
    model_cfg = ModelConfig(
        profile="deberta_v3_parity",
        backbone_type="hf_deberta_v2",
        embedding_sharing="none",
        hf_attention_kernel="stable",
    )
    train_cfg = TrainConfig(
        mask_token_prob=0.95,
        random_token_prob=0.03,
        mlm_max_ngram=2,
        disc_loss_weight=12.5,
        token_weighted_gradient_accumulation=False,
    )
    optim_cfg = OptimConfig(adam_epsilon=5e-6, warmup_steps=2_000)

    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg, optim_cfg=optim_cfg)

    assert model_cfg.backbone_type == "hf_deberta_v2"
    assert model_cfg.embedding_sharing == "none"
    assert model_cfg.hf_attention_kernel == "stable"
    assert train_cfg.mask_token_prob == pytest.approx(0.95)
    assert train_cfg.random_token_prob == pytest.approx(0.03)
    assert train_cfg.mlm_max_ngram == 2
    assert train_cfg.disc_loss_weight == pytest.approx(12.5)
    assert optim_cfg.adam.epsilon == pytest.approx(5e-6)
    assert optim_cfg.scheduler.warmup_steps == 2_000
    assert train_cfg.token_weighted_gradient_accumulation is False


def test_apply_profile_defaults_honors_explicit_fields_even_when_matching_raw_defaults() -> None:
    model_cfg = ModelConfig(
        profile="deberta_v3_parity",
        backbone_type="hf_deberta_v2",
        embedding_sharing="es",
        hf_attention_kernel="auto",
    )
    train_cfg = TrainConfig(
        mask_token_prob=0.8,
        random_token_prob=0.1,
        disc_loss_weight=50.0,
        token_weighted_gradient_accumulation=True,
    )
    optim_cfg = OptimConfig(adam_epsilon=1e-8, warmup_steps=1_000)
    object.__setattr__(
        model_cfg,
        "_explicit_fields",
        {"backbone_type", "embedding_sharing", "hf.attention_kernel"},
    )
    object.__setattr__(
        train_cfg,
        "_explicit_fields",
        {
            "objective.mask_token_prob",
            "objective.random_token_prob",
            "objective.disc_loss_weight",
            "token_weighted_gradient_accumulation",
        },
    )
    object.__setattr__(optim_cfg, "_explicit_fields", {"adam.epsilon", "scheduler.warmup_steps"})

    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg, optim_cfg=optim_cfg)

    assert model_cfg.backbone_type == "hf_deberta_v2"
    assert model_cfg.embedding_sharing == "es"
    assert model_cfg.hf_attention_kernel == "auto"
    assert train_cfg.mask_token_prob == pytest.approx(0.8)
    assert train_cfg.random_token_prob == pytest.approx(0.1)
    assert train_cfg.disc_loss_weight == pytest.approx(50.0)
    assert optim_cfg.adam.epsilon == pytest.approx(1e-8)
    assert optim_cfg.scheduler.warmup_steps == 1_000
    assert train_cfg.token_weighted_gradient_accumulation is True


def test_decoupled_training_defaults_true_and_allows_explicit_disable() -> None:
    assert bool(TrainConfig().decoupled_training) is True
    assert bool(TrainConfig(decoupled_training=False).decoupled_training) is False


def test_validate_train_config_rejects_non_boolean_decoupled_training() -> None:
    cfg = TrainConfig(decoupled_training=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="train.decoupled_training must be a boolean"):
        validate_train_config(cfg)


@pytest.mark.parametrize("value", [-1.0, 1.0e-4, 2.5e-3])
def test_validate_train_config_allows_discriminator_learning_rate_inherit_or_positive(value: float) -> None:
    cfg = OptimConfig(discriminator_learning_rate=value)
    validate_optim_config(cfg)


@pytest.mark.parametrize("value", [0.0, -0.5, -2.0])
def test_validate_train_config_rejects_invalid_discriminator_learning_rate(value: float) -> None:
    cfg = OptimConfig(discriminator_learning_rate=value)
    with pytest.raises(ValueError, match="optim.lr.discriminator"):
        validate_optim_config(cfg)


def test_run_pretraining_dry_run_fails_fast_for_nonempty_output_dir(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "marker.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Output directory exists and is not empty"):
        run_pretraining_dry_run(
            model_cfg=ModelConfig(),
            data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
            train_cfg=TrainConfig(output_dir=str(out_dir), overwrite_output_dir=False, max_steps=5),
            config_path=None,
        )


def test_validate_data_config_rejects_non_streaming_shuffle_buffer_above_one():
    with pytest.raises(ValueError, match="shuffle_buffer_size must be 0 or 1"):
        validate_data_config(
            DataConfig(
                dataset_name="HuggingFaceFW/fineweb-edu",
                streaming=False,
                shuffle_buffer_size=10_000,
            )
        )


def test_validate_data_config_rejects_doc_blocking_when_not_packed():
    with pytest.raises(ValueError, match="requires data.packing.enabled=true"):
        validate_data_config(
            DataConfig(
                dataset_name="HuggingFaceFW/fineweb-edu",
                pack_sequences=False,
                block_cross_document_attention=True,
            )
        )


def test_validate_data_config_warns_on_long_context_dense_doc_blocking():
    with pytest.warns(UserWarning, match="segment-aware attention backends"):
        validate_data_config(
            DataConfig(
                dataset_name="HuggingFaceFW/fineweb-edu",
                pack_sequences=True,
                block_cross_document_attention=True,
                max_seq_length=4096,
            )
        )


def test_validate_training_workflow_options_rejects_flash_with_packing():
    with pytest.raises(
        ValueError, match="train.sdpa_kernel=flash is not supported with data.packing.enabled=true"
    ):
        validate_training_workflow_options(
            data_cfg=DataConfig(
                dataset_name="HuggingFaceFW/fineweb-edu",
                pack_sequences=True,
                block_cross_document_attention=True,
            ),
            train_cfg=TrainConfig(sdpa_kernel="flash"),
        )


def test_validate_training_workflow_options_allows_flash_when_packed_doc_blocking_disabled():
    validate_training_workflow_options(
        data_cfg=DataConfig(
            dataset_name="HuggingFaceFW/fineweb-edu",
            pack_sequences=True,
            block_cross_document_attention=False,
        ),
        train_cfg=TrainConfig(sdpa_kernel="flash"),
    )


def test_validate_training_workflow_options_rejects_sdpa_kernel_override_when_rope_attention_is_eager():
    with pytest.raises(ValueError, match="train.sdpa_kernel only affects rope attention"):
        validate_training_workflow_options(
            data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu", pack_sequences=False),
            train_cfg=TrainConfig(sdpa_kernel="flash"),
            model_cfg=ModelConfig(backbone_type="rope", attention_implementation="eager"),
        )


def test_validate_training_workflow_options_allows_hf_backbone_doc_blocking_in_packed_mode():
    validate_training_workflow_options(
        data_cfg=DataConfig(
            dataset_name="HuggingFaceFW/fineweb-edu",
            pack_sequences=True,
            block_cross_document_attention=True,
        ),
        train_cfg=TrainConfig(),
        model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
    )


def test_validate_training_workflow_options_allows_hf_backbone_doc_blocking_when_sdpa_kernel_is_flash():
    with pytest.warns(UserWarning, match="train.sdpa_kernel has no effect"):
        validate_training_workflow_options(
            data_cfg=DataConfig(
                dataset_name="HuggingFaceFW/fineweb-edu",
                pack_sequences=True,
                block_cross_document_attention=True,
            ),
            train_cfg=TrainConfig(sdpa_kernel="flash"),
            model_cfg=ModelConfig(backbone_type="hf_deberta_v2"),
        )


def test_validate_training_workflow_options_rejects_es_with_divergent_gen_lr():
    with pytest.raises(ValueError, match="embedding_sharing='es'"):
        validate_training_workflow_options(
            data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
            train_cfg=TrainConfig(),
            model_cfg=ModelConfig(embedding_sharing="es"),
            optim_cfg=OptimConfig(learning_rate=5e-4, generator_learning_rate=3e-4),
        )


def test_validate_training_workflow_options_allows_es_with_matching_gen_lr():
    # Explicit gen LR matching disc LR — should pass.
    validate_training_workflow_options(
        data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
        train_cfg=TrainConfig(decoupled_training=False),
        model_cfg=ModelConfig(embedding_sharing="es"),
        optim_cfg=OptimConfig(learning_rate=5e-4, generator_learning_rate=5e-4),
    )
    # Inherited gen LR (-1) — should pass.
    validate_training_workflow_options(
        data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
        train_cfg=TrainConfig(decoupled_training=False),
        model_cfg=ModelConfig(embedding_sharing="es"),
        optim_cfg=OptimConfig(learning_rate=5e-4, generator_learning_rate=-1.0),
    )


def test_validate_training_workflow_options_allows_gdes_with_divergent_gen_lr():
    # GDES handles separate LR correctly — should not raise.
    validate_training_workflow_options(
        data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
        train_cfg=TrainConfig(),
        model_cfg=ModelConfig(embedding_sharing="gdes"),
        optim_cfg=OptimConfig(learning_rate=5e-4, generator_learning_rate=3e-4),
    )


def test_validate_training_workflow_options_rejects_decoupled_with_es_embedding_sharing():
    with pytest.raises(ValueError, match="gradients.*dropped"):
        validate_training_workflow_options(
            data_cfg=DataConfig(dataset_name="HuggingFaceFW/fineweb-edu"),
            train_cfg=TrainConfig(decoupled_training=True),
            model_cfg=ModelConfig(backbone_type="hf_deberta_v2", embedding_sharing="es"),
        )


def test_validate_model_config_rejects_rope_knobs_in_hf_mode():
    cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        rope_theta=50_000.0,
    )
    with pytest.raises(ValueError, match="only valid when model.backbone_type='rope'"):
        validate_model_config(cfg)


def test_validate_model_config_normalizes_hf_attention_kernel_alias():
    cfg = ModelConfig(backbone_type="hf_deberta_v2", hf_attention_kernel="cache")
    validate_model_config(cfg)
    assert cfg.hf_attention_kernel == "cached_bmm"

    cfg = ModelConfig(backbone_type="hf_deberta_v2", hf_attention_kernel="safe")
    validate_model_config(cfg)
    assert cfg.hf_attention_kernel == "stable"


def test_validate_model_config_rejects_non_positive_tokenizer_vocab_multiple():
    cfg = ModelConfig(tokenizer_vocab_multiple=0)
    with pytest.raises(ValueError, match="model.tokenizer.vocab_multiple"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_non_positive_tokenizer_vocab_target():
    cfg = ModelConfig(tokenizer_vocab_target=0)
    with pytest.raises(ValueError, match="model.tokenizer.vocab_target"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_derived_generator_knobs_with_explicit_generator_source():
    cfg = ModelConfig(
        backbone_type="rope",
        pretrained_generator_path="microsoft/deberta-v3-small",
        generator_hidden_size=256,
    )
    with pytest.raises(ValueError, match="only used when deriving generator config"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_hf_max_position_embeddings_for_rope():
    cfg = ModelConfig(
        backbone_type="rope",
        hf_max_position_embeddings=1024,
    )
    with pytest.warns(UserWarning, match="model.hf.max_position_embeddings only applies"):
        validate_model_config(cfg)


def test_validate_model_config_allows_hf_max_position_embeddings_in_hf_scratch_mode():
    cfg = ModelConfig(
        backbone_type="hf_deberta_v2",
        from_scratch=True,
        hf_max_position_embeddings=1024,
    )
    validate_model_config(cfg)


def test_validate_model_config_rejects_pretrained_derived_generator_shape_overrides():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        pretrained_discriminator_path="local-rope-disc",
        generator_hidden_size=256,
    )
    with pytest.raises(ValueError, match="derived generator weights"):
        validate_model_config(cfg)


def test_validate_model_config_allows_pretrained_derived_generator_layer_override():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        pretrained_discriminator_path="local-rope-disc",
        generator_num_hidden_layers=4,
    )
    validate_model_config(cfg)


def test_looks_like_hf_deberta_checkpoint_avoids_local_rope_false_positive():
    assert _looks_like_hf_deberta_checkpoint("microsoft/deberta-v3-base")
    assert _looks_like_hf_deberta_checkpoint("https://huggingface.co/microsoft/deberta-v3-base")
    assert _looks_like_hf_deberta_checkpoint(
        "/home/user/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/abc"
    )
    assert not _looks_like_hf_deberta_checkpoint("runs/deberta-v3-rope/checkpoint-1000")


def test_validate_model_config_allows_local_rope_checkpoint_path_with_deberta_in_name():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        pretrained_discriminator_path="runs/deberta-v3-rope/checkpoint-1000",
    )
    validate_model_config(cfg)


def test_validate_model_config_rejects_scratch_rope_knobs_in_pretrained_mode():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        pretrained_discriminator_path="local-rope-disc",
        rope_theta=50_000.0,
    )
    with pytest.raises(ValueError, match="only affect scratch RoPE initialization"):
        validate_model_config(cfg)


def test_validate_model_config_allows_explicit_pretrained_rope_overrides():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        pretrained_discriminator_path="local-rope-disc",
        pretrained_rope_theta=50_000.0,
        pretrained_norm_arch="keel",
    )
    validate_model_config(cfg)


def test_validate_model_config_rejects_pretrained_rope_overrides_in_scratch_mode():
    cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=True,
        pretrained_rope_theta=50_000.0,
    )
    with pytest.raises(ValueError, match="apply only when model.from_scratch=false"):
        validate_model_config(cfg)


def test_build_backbone_configs_sets_tokenizer_special_ids_for_hf_configs():
    pytest.importorskip("transformers")
    tokenizer = DummyTokenizer(vocab_size=128)

    model_cfg = ModelConfig(backbone_type="hf_deberta_v2", from_scratch=True)
    disc_cfg, gen_cfg = build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
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
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.4,
    )

    monkeypatch.setattr(
        "deberta.modeling.builder.DebertaRoPEConfig.from_pretrained",
        lambda _src: checkpoint_cfg,
    )
    tokenizer = DummyTokenizer(vocab_size=32000)

    model_cfg = ModelConfig(
        backbone_type="rope",
        from_scratch=False,
        pretrained_discriminator_path="local-rope-disc",
        hidden_dropout_prob=None,
        attention_probs_dropout_prob=None,
    )

    disc_cfg, _ = build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
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
    assert disc_cfg.hidden_dropout_prob == pytest.approx(0.3)
    assert disc_cfg.attention_probs_dropout_prob == pytest.approx(0.4)


def test_readme_cli_examples_are_parseable():
    parser = cli_mod._build_main_parser()
    examples = [
        "train configs/pretrain_rope_fineweb_edu.yaml",
        "train configs/pretrain_rope_fineweb_edu_2048.yaml",
        "train configs/pretrain_rope_fineweb_edu_4096.yaml",
        "train --preset deberta-v3-base --dry-run",
        (
            "export runs/deberta_rope_rtd/checkpoint-10000 "
            "--what discriminator "
            "--output-dir runs/deberta_rope_rtd/exported_hf"
        ),
    ]

    for cmd in examples:
        parser.parse_args(shlex.split(cmd))


def test_export_help_does_not_show_misleading_defaults_on_no_flags() -> None:
    parser = argparse.ArgumentParser(
        prog="deberta export",
        formatter_class=ExportArgumentDefaultsHelpFormatter,
    )
    add_export_arguments(parser)
    help_text = parser.format_help()
    assert "--safe-serialization" in help_text and "(default: True)" in help_text
    for opt in ("--no-safe-serialization", "--no-offload-to-cpu", "--no-rank0-only"):
        line = next((ln for ln in help_text.splitlines() if opt in ln), "")
        assert "(default:" not in line


def test_docs_use_current_nested_key_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    docs = [
        repo_root / "docs" / "advanced" / "torch-compile.md",
        repo_root / "docs" / "advanced" / "distributed-training.md",
        repo_root / "docs" / "guides" / "data-pipeline.md",
    ]
    merged = "\n".join(path.read_text(encoding="utf-8") for path in docs)

    stale_tokens = (
        "train.torch_compile",
        "train.torch_compile_mode",
        "train.torch_compile_scope",
        "train.torch_compile_backend",
        "train.report_to",
        "train.resume_data_strategy",
        "train.resume_replay_max_micro_batches",
        "data.block_cross_document_attention",
        "train.mlm_max_ngram",
        "train.mask_token_prob",
        "train.random_token_prob",
    )
    for token in stale_tokens:
        assert token not in merged


@pytest.mark.parametrize(
    "cfg_kwargs,expect_warn",
    [
        # 1A: rope + non-default hf_attention_kernel → warn
        ({"backbone_type": "rope", "hf_attention_kernel": "stable"}, True),
        # 1A: rope + default hf_attention_kernel → no warn
        ({"backbone_type": "rope", "hf_attention_kernel": "dynamic"}, False),
        # 1B: post + non-default keel_alpha_init → warn
        ({"backbone_type": "rope", "norm_arch": "post", "keel_alpha_init": 5.0}, True),
        # 1B: post + non-default keel_alpha_learnable → warn
        ({"backbone_type": "rope", "norm_arch": "post", "keel_alpha_learnable": True}, True),
        # 1B: keel + keel_alpha_init → no warn
        ({"backbone_type": "rope", "norm_arch": "keel", "keel_alpha_init": 5.0}, False),
        # 1B: post + default keel params → no warn
        ({"backbone_type": "rope", "norm_arch": "post"}, False),
        # 1C: mlp + non-default swiglu_adjust_intermediate → warn
        ({"backbone_type": "rope", "ffn_type": "mlp", "swiglu_adjust_intermediate": False}, True),
        # 1C: swiglu + swiglu_adjust_intermediate → no warn
        ({"backbone_type": "rope", "ffn_type": "swiglu", "swiglu_adjust_intermediate": False}, False),
        # 1C: mlp + default swiglu_adjust_intermediate → no warn
        ({"backbone_type": "rope", "ffn_type": "mlp"}, False),
    ],
    ids=[
        "1A-rope-hf_kernel-warn",
        "1A-rope-hf_kernel-default-nowarn",
        "1B-post-keel_alpha_init-warn",
        "1B-post-keel_alpha_learnable-warn",
        "1B-keel-keel_alpha_init-nowarn",
        "1B-post-default-nowarn",
        "1C-mlp-swiglu_adjust-warn",
        "1C-swiglu-swiglu_adjust-nowarn",
        "1C-mlp-default-nowarn",
    ],
)
def test_validate_model_config_inert_param_warnings(cfg_kwargs, expect_warn):
    cfg = ModelConfig(**cfg_kwargs)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_model_config(cfg)
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    if expect_warn:
        assert len(user_warnings) >= 1, f"Expected warning for {cfg_kwargs}, got none"
    else:
        assert len(user_warnings) == 0, (
            f"Unexpected warning for {cfg_kwargs}: {[str(x.message) for x in user_warnings]}"
        )


@pytest.mark.parametrize(
    "cfg_kwargs,expect_warn",
    [
        # 1D: compile=false + non-auto scope → warn
        ({"torch_compile": False, "torch_compile_scope": "backbones"}, True),
        # 1D: compile=false + non-default mode → warn
        ({"torch_compile": False, "torch_compile_mode": "max-autotune"}, True),
        # 1D: compile=false + non-default backend → warn
        ({"torch_compile": False, "torch_compile_backend": "aot_eager"}, True),
        # 1D: report_to!=wandb + non-default watch mode → warn
        ({"report_to": "tensorboard", "wandb_watch": "all"}, True),
        # 1D: report_to!=wandb + non-default watch freq → warn
        ({"report_to": "none", "wandb_watch_log_freq": 7}, True),
        # 1D: compile=false + auto scope → no warn
        ({"torch_compile": False, "torch_compile_scope": "auto"}, False),
        # 1D: compile=false + default mode/backend/watch knobs → no warn
        ({"torch_compile": False, "report_to": "none"}, False),
        # 1D: wandb backend + watch options active → no inert warning
        ({"report_to": "wandb", "wandb_watch": "all", "wandb_watch_log_freq": 7}, False),
        # 1D: compile=true + non-auto scope → no warn
        ({"torch_compile": True, "torch_compile_scope": "backbones"}, False),
    ],
    ids=[
        "1D-nocompile-scope-warn",
        "1D-nocompile-mode-warn",
        "1D-nocompile-backend-warn",
        "1D-nowandb-watchmode-warn",
        "1D-nowandb-watchfreq-warn",
        "1D-nocompile-auto-nowarn",
        "1D-nocompile-defaults-nowarn",
        "1D-wandb-watch-active-nowarn",
        "1D-compile-scope-nowarn",
    ],
)
def test_validate_train_config_inert_param_warnings(cfg_kwargs, expect_warn):
    train_keys = {
        "torch_compile",
        "torch_compile_scope",
        "torch_compile_mode",
        "torch_compile_backend",
    }
    logging_keys = {"report_to", "wandb_watch", "wandb_watch_log_freq"}

    train_kwargs = {k: v for k, v in cfg_kwargs.items() if k in train_keys}
    logging_kwargs = {k: v for k, v in cfg_kwargs.items() if k in logging_keys}

    train_cfg = TrainConfig(**train_kwargs)
    logging_cfg = LoggingConfig(**logging_kwargs) if logging_kwargs else None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_train_config(train_cfg)
        if logging_cfg is not None:
            validate_logging_config(logging_cfg)
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    if expect_warn:
        assert len(user_warnings) >= 1, f"Expected warning for {cfg_kwargs}, got none"
    else:
        assert len(user_warnings) == 0, (
            f"Unexpected warning for {cfg_kwargs}: {[str(x.message) for x in user_warnings]}"
        )


@pytest.mark.parametrize(
    "model_kwargs,train_kwargs,expect_warn",
    [
        # 1E: hf_deberta_v2 + non-auto sdpa_kernel → warn
        ({"backbone_type": "hf_deberta_v2"}, {"sdpa_kernel": "mem_efficient"}, True),
        # 1E: hf_deberta_v2 + auto sdpa_kernel → no warn
        ({"backbone_type": "hf_deberta_v2"}, {"sdpa_kernel": "auto"}, False),
        # 1E: rope + non-auto sdpa_kernel → no warn (rope SDPA is functional)
        ({"backbone_type": "rope"}, {"sdpa_kernel": "mem_efficient"}, False),
    ],
    ids=[
        "1E-hfv2-sdpa-warn",
        "1E-hfv2-auto-nowarn",
        "1E-rope-sdpa-nowarn",
    ],
)
def test_validate_workflow_sdpa_kernel_inert_warning(model_kwargs, train_kwargs, expect_warn):
    model_cfg = ModelConfig(**model_kwargs)
    train_cfg = TrainConfig(**train_kwargs)
    data_cfg = DataConfig(data_files="dummy.txt")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_training_workflow_options(data_cfg=data_cfg, train_cfg=train_cfg, model_cfg=model_cfg)
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    if expect_warn:
        assert len(user_warnings) >= 1, (
            f"Expected warning for model={model_kwargs} train={train_kwargs}, got none"
        )
    else:
        assert len(user_warnings) == 0, (
            f"Unexpected warning for model={model_kwargs} train={train_kwargs}: "
            f"{[str(x.message) for x in user_warnings]}"
        )


@pytest.mark.parametrize(
    ("kwargs", "expected_scope", "expected_reason"),
    [
        (
            {"effective_compile_scope": "backbones", "compile_scope_reason": "auto test"},
            "backbones",
            "auto test",
        ),
        ({}, None, None),
    ],
    ids=["includes_scope", "omits_scope_when_none"],
)
def test_build_run_metadata_scope_fields(
    kwargs: dict[str, str], expected_scope: str | None, expected_reason: str | None
):
    meta = _build_run_metadata(**kwargs)
    if expected_scope is None:
        assert "effective_compile_scope" not in meta
        assert "compile_scope_reason" not in meta
        return
    assert meta["effective_compile_scope"] == expected_scope
    assert meta["compile_scope_reason"] == expected_reason
    assert "config_schema_version" in meta


def test_persist_or_validate_run_configs_preflight_mode_writes_no_snapshots(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=ModelConfig(),
        data_cfg=DataConfig(data_files="dummy.txt"),
        train_cfg=TrainConfig(),
        resume_checkpoint=None,
        is_main_process=True,
        preflight_only=True,
    )
    for filename in ("model_config.json", "data_config.json", "train_config.json", "run_metadata.json"):
        assert not (out / filename).exists()


def test_persist_run_configs_writes_compile_scope_to_metadata(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    model_cfg = ModelConfig()
    data_cfg = DataConfig(data_files="dummy.txt")
    train_cfg = TrainConfig()
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        is_main_process=True,
        effective_compile_scope="ffn",
        compile_scope_reason="auto scope selected FFN-only",
    )
    from deberta.utils.io import load_json_mapping

    meta = load_json_mapping(out / "run_metadata.json")
    assert meta["effective_compile_scope"] == "ffn"
    assert meta["compile_scope_reason"] == "auto scope selected FFN-only"


def test_persist_run_configs_warns_on_compile_scope_drift_on_resume(tmp_path: Path, caplog):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    model_cfg = ModelConfig()
    data_cfg = DataConfig(data_files="dummy.txt")
    train_cfg = TrainConfig()

    # Initial run persists scope=backbones.
    _persist_or_validate_run_configs(
        output_dir=out,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=None,
        is_main_process=True,
        effective_compile_scope="backbones",
    )

    # Simulate a checkpoint directory for resume.
    ckpt = out / "checkpoint-100"
    ckpt.mkdir()

    # Resume with different scope should log a warning.
    with caplog.at_level(logging.WARNING):
        _persist_or_validate_run_configs(
            output_dir=out,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            resume_checkpoint=str(ckpt),
            is_main_process=True,
            effective_compile_scope="ffn",
        )
    assert any("compile scope changed on resume" in r.message.lower() for r in caplog.records)


def test_global_grad_l2_norm_finite_model():
    model = torch.nn.Linear(4, 4, bias=False)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    norm = _global_grad_l2_norm(model)
    assert norm > 0.0
    assert torch.isfinite(torch.tensor(norm))
    # Cross-check against torch.nn.utils.clip_grad_norm_ (which returns the norm).
    ref = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
    assert abs(norm - float(ref)) < 1e-4


@pytest.mark.parametrize(
    "model", [torch.nn.Linear(4, 4, bias=True), torch.nn.Module()], ids=["zero_grads", "no_params"]
)
def test_global_grad_l2_norm_zero_like_states(model: torch.nn.Module):
    model.zero_grad()
    if any(True for _ in model.parameters()):
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
    assert _global_grad_l2_norm(model) == 0.0
