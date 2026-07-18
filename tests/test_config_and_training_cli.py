import argparse
import logging
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from _config_factories import (
    make_data_config,
    make_logging_config,
    make_model_config,
    make_optim_config,
    make_train_config,
)
from _fakes import capture_run_pretraining_kwargs

import deberta.cli as cli_mod
from deberta.config import (
    _looks_like_hf_deberta_checkpoint,
    validate_data_config,
    validate_logging_config,
    validate_model_config,
    validate_optim_config,
    validate_train_config,
    validate_training_workflow_options,
)
from deberta.export_cli import ExportArgumentDefaultsHelpFormatter, add_export_arguments
from deberta.training.entrypoint import run_pretraining_dry_run
from deberta.training.run_config import _build_run_metadata, _persist_or_validate_run_configs
from deberta.training.steps import _global_grad_l2_norm


def test_main_cli_train_subcommand_loads_yaml_and_applies_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):

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

    seen = capture_run_pretraining_kwargs(monkeypatch, cli_mod)
    cli_mod.main(["train", str(cfg_path), "--train.max_steps", "7"])

    assert "train_cfg" in seen
    assert seen["data_cfg"].packing.max_seq_length == 32
    assert seen["train_cfg"].max_steps == 7
    assert seen["config_path"] == cfg_path


def test_main_cli_train_honors_explicit_yaml_warmup_value_for_hf_backbone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "model:",
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

    seen = capture_run_pretraining_kwargs(monkeypatch, cli_mod)
    cli_mod.main(["train", str(cfg_path)])

    assert "train_cfg" in seen
    assert int(seen["optim_cfg"].scheduler.warmup_steps) == 1000
    assert seen["config_path"] == cfg_path
    err = capsys.readouterr().err
    assert "optim.scheduler.warmup_steps" not in err


def test_main_cli_train_reports_when_file_value_is_changed_by_cli_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):

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

    seen = capture_run_pretraining_kwargs(monkeypatch, cli_mod)
    cli_mod.main(["train", str(cfg_path), "--train.max_steps", "7"])

    assert "train_cfg" in seen
    assert int(seen["train_cfg"].max_steps) == 7
    err = capsys.readouterr().err
    assert "train.max_steps: 5 -> 7 (CLI override (--train.max_steps))" in err


def test_main_cli_train_supports_dotted_overrides_with_type_casting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):

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

    seen = capture_run_pretraining_kwargs(monkeypatch, cli_mod)
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
    assert float(seen["optim_cfg"].adam.epsilon) == pytest.approx(1e-5)
    assert int(seen["model_cfg"].hf.max_position_embeddings) == 640
    err = capsys.readouterr().err
    assert "train.max_steps: 5 -> 11 (CLI override (--train.max_steps))" in err


def test_main_cli_train_supports_null_for_optional_numeric_dotted_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):

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

    seen = capture_run_pretraining_kwargs(monkeypatch, cli_mod)
    cli_mod.main(
        [
            "train",
            str(cfg_path),
            "--model.hf.max_position_embeddings",
            "null",
        ]
    )

    assert seen["model_cfg"].hf.max_position_embeddings is None
    err = capsys.readouterr().err
    assert (
        "model.hf.max_position_embeddings: 640 -> None (CLI override (--model.hf.max_position_embeddings))"
    ) in err


def test_main_cli_train_supports_null_for_optional_constrained_dotted_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):

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

    seen = capture_run_pretraining_kwargs(monkeypatch, cli_mod)
    cli_mod.main(
        [
            "train",
            str(cfg_path),
            "--model.rope.pretrained.norm_arch",
            "none",
        ]
    )

    assert seen["model_cfg"].rope.pretrained.norm_arch is None
    err = capsys.readouterr().err
    assert (
        "model.rope.pretrained.norm_arch: 'keel' -> None (CLI override (--model.rope.pretrained.norm_arch))"
        in err
    )


def test_main_cli_train_supports_dotted_overrides_for_extended_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):

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

    seen = capture_run_pretraining_kwargs(monkeypatch, cli_mod)
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

    assert int(seen["optim_cfg"].scheduler.warmup_steps) == 333
    assert seen["logging_cfg"].wandb.watch == "all"


def test_main_cli_train_rejects_invalid_dotted_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


@pytest.mark.parametrize("scope", ["generator-encoder", "gen-encoder", "disc-ffn"])
def test_train_parser_accepts_hyphenated_compile_scope_aliases(scope: str) -> None:
    parser = cli_mod._build_main_parser()
    ns = parser.parse_args(["train", "--train.compile.scope", scope])

    assert vars(ns)["dot__train__compile__scope"] == scope


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
            make_data_config(
                source={
                    "load_from_disk": "runs/saved_ds",
                    "dataset_name": "HuggingFaceFW/fineweb-edu",
                    "streaming": False,
                }
            )
        )


def test_validate_train_config_rejects_overwrite_with_resume_conflict():
    with pytest.raises(ValueError, match="cannot be combined with train.checkpoint.resume_from_checkpoint"):
        validate_train_config(
            make_train_config(checkpoint={"overwrite_output_dir": True, "resume_from_checkpoint": "auto"})
        )


@pytest.mark.parametrize(
    ("resume_hint", "expected"),
    [("   ", None), (" auto ", "auto")],
    ids=["blank_to_none", "trimmed_value"],
)
def test_validate_train_config_normalizes_resume_hints(resume_hint: str, expected: str | None):
    cfg = make_train_config(checkpoint={"resume_from_checkpoint": resume_hint})
    validate_train_config(cfg)
    assert cfg.checkpoint.resume_from_checkpoint == expected


def test_model_config_default_backbone_is_hf_deberta_v2() -> None:
    cfg = make_model_config()
    assert cfg.backbone_type == "hf_deberta_v2"


def test_decoupled_training_defaults_true_and_allows_explicit_disable() -> None:
    assert bool(make_train_config().decoupled_training) is True
    assert bool(make_train_config(decoupled_training=False).decoupled_training) is False


def test_validate_train_config_rejects_non_boolean_decoupled_training() -> None:
    cfg = make_train_config(decoupled_training=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="train.decoupled_training must be a boolean"):
        validate_train_config(cfg)


@pytest.mark.parametrize("value", [-1.0, 1.0e-4, 2.5e-3])
def test_validate_train_config_allows_discriminator_learning_rate_inherit_or_positive(value: float) -> None:
    cfg = make_optim_config(lr={"discriminator": value})
    validate_optim_config(cfg)


@pytest.mark.parametrize("value", [0.0, -0.5, -2.0])
def test_validate_train_config_rejects_invalid_discriminator_learning_rate(value: float) -> None:
    cfg = make_optim_config(lr={"discriminator": value})
    with pytest.raises(ValueError, match="optim.lr.discriminator"):
        validate_optim_config(cfg)


def test_run_pretraining_dry_run_fails_fast_for_nonempty_output_dir(tmp_path: Path):
    out_dir = tmp_path / "run"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "marker.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Output directory exists and is not empty"):
        run_pretraining_dry_run(
            model_cfg=make_model_config(),
            data_cfg=make_data_config(source={"dataset_name": "HuggingFaceFW/fineweb-edu"}),
            train_cfg=make_train_config(
                checkpoint={"output_dir": str(out_dir), "overwrite_output_dir": False}, max_steps=5
            ),
            config_path=None,
        )


def test_validate_data_config_rejects_non_streaming_shuffle_buffer_above_one():
    with pytest.raises(ValueError, match="shuffle_buffer_size must be 0 or 1"):
        validate_data_config(
            make_data_config(
                source={
                    "dataset_name": "HuggingFaceFW/fineweb-edu",
                    "streaming": False,
                    "shuffle_buffer_size": 10_000,
                }
            )
        )


def test_validate_data_config_rejects_doc_blocking_when_not_packed():
    with pytest.raises(ValueError, match="requires data.packing.enabled=true"):
        validate_data_config(
            make_data_config(
                source={"dataset_name": "HuggingFaceFW/fineweb-edu"},
                packing={"enabled": False, "block_cross_document_attention": True},
            )
        )


def test_validate_data_config_warns_on_long_context_dense_doc_blocking():
    with pytest.warns(UserWarning, match="segment-aware attention backends"):
        validate_data_config(
            make_data_config(
                source={"dataset_name": "HuggingFaceFW/fineweb-edu"},
                packing={"enabled": True, "block_cross_document_attention": True, "max_seq_length": 4096},
            )
        )


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_validate_train_config_rejects_seed_outside_accelerate_range(seed: int) -> None:
    with pytest.raises(ValueError, match=r"train.seed must be between 0 and 2\*\*32 - 1"):
        validate_train_config(make_train_config(seed=seed))


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_validate_train_config_accepts_seed_boundaries(seed: int) -> None:
    validate_train_config(make_train_config(seed=seed))


def test_validate_training_workflow_options_rejects_flash_with_packing():
    with pytest.raises(
        ValueError, match="train.sdpa_kernel=flash is not supported with data.packing.enabled=true"
    ):
        validate_training_workflow_options(
            data_cfg=make_data_config(
                source={"dataset_name": "HuggingFaceFW/fineweb-edu"},
                packing={"enabled": True, "block_cross_document_attention": True},
            ),
            train_cfg=make_train_config(sdpa_kernel="flash"),
        )


def test_validate_training_workflow_options_allows_flash_when_packed_doc_blocking_disabled():
    validate_training_workflow_options(
        data_cfg=make_data_config(
            source={"dataset_name": "HuggingFaceFW/fineweb-edu"},
            packing={"enabled": True, "block_cross_document_attention": False},
        ),
        train_cfg=make_train_config(sdpa_kernel="flash"),
    )


def test_validate_training_workflow_options_rejects_sdpa_kernel_override_when_rope_attention_is_eager():
    with pytest.raises(ValueError, match="train.sdpa_kernel only affects rope attention"):
        validate_training_workflow_options(
            data_cfg=make_data_config(
                source={"dataset_name": "HuggingFaceFW/fineweb-edu"}, packing={"enabled": False}
            ),
            train_cfg=make_train_config(sdpa_kernel="flash"),
            model_cfg=make_model_config(backbone_type="rope", rope={"attention_implementation": "eager"}),
        )


def test_validate_training_workflow_options_allows_hf_backbone_doc_blocking_in_packed_mode():
    validate_training_workflow_options(
        data_cfg=make_data_config(
            source={"dataset_name": "HuggingFaceFW/fineweb-edu"},
            packing={"enabled": True, "block_cross_document_attention": True},
        ),
        train_cfg=make_train_config(),
        model_cfg=make_model_config(backbone_type="hf_deberta_v2"),
    )


def test_validate_training_workflow_options_allows_hf_backbone_doc_blocking_when_sdpa_kernel_is_flash():
    with pytest.warns(UserWarning, match="train.sdpa_kernel has no effect"):
        validate_training_workflow_options(
            data_cfg=make_data_config(
                source={"dataset_name": "HuggingFaceFW/fineweb-edu"},
                packing={"enabled": True, "block_cross_document_attention": True},
            ),
            train_cfg=make_train_config(sdpa_kernel="flash"),
            model_cfg=make_model_config(backbone_type="hf_deberta_v2"),
        )


def test_validate_training_workflow_options_rejects_flashdeberta_without_bf16() -> None:
    with pytest.raises(ValueError, match="attention_impl='flash'.*mixed_precision='bf16'"):
        validate_training_workflow_options(
            data_cfg=make_data_config(source={"dataset_name": "HuggingFaceFW/fineweb-edu"}),
            train_cfg=make_train_config(mixed_precision="no"),
            model_cfg=make_model_config(
                backbone_type="hf_deberta_v2",
                hf={"attention_impl": "flash"},
                dropout={"hidden_prob": 0.0, "attention_probs_prob": 0.0},
            ),
        )


def test_validate_training_workflow_options_rejects_es_with_divergent_gen_lr():
    with pytest.raises(ValueError, match="embedding_sharing='es'"):
        validate_training_workflow_options(
            data_cfg=make_data_config(source={"dataset_name": "HuggingFaceFW/fineweb-edu"}),
            train_cfg=make_train_config(),
            model_cfg=make_model_config(embedding_sharing="es"),
            optim_cfg=make_optim_config(lr={"base": 5e-4, "generator": 3e-4}),
        )


def test_validate_training_workflow_options_allows_es_with_matching_gen_lr():
    # Explicit gen LR matching disc LR — should pass.
    validate_training_workflow_options(
        data_cfg=make_data_config(source={"dataset_name": "HuggingFaceFW/fineweb-edu"}),
        train_cfg=make_train_config(decoupled_training=False),
        model_cfg=make_model_config(embedding_sharing="es"),
        optim_cfg=make_optim_config(lr={"base": 5e-4, "generator": 5e-4}),
    )
    # Inherited gen LR (-1) — should pass.
    validate_training_workflow_options(
        data_cfg=make_data_config(source={"dataset_name": "HuggingFaceFW/fineweb-edu"}),
        train_cfg=make_train_config(decoupled_training=False),
        model_cfg=make_model_config(embedding_sharing="es"),
        optim_cfg=make_optim_config(lr={"base": 5e-4, "generator": -1.0}),
    )


def test_validate_training_workflow_options_allows_gdes_with_divergent_gen_lr():
    # GDES handles separate LR correctly — should not raise.
    validate_training_workflow_options(
        data_cfg=make_data_config(source={"dataset_name": "HuggingFaceFW/fineweb-edu"}),
        train_cfg=make_train_config(),
        model_cfg=make_model_config(embedding_sharing="gdes"),
        optim_cfg=make_optim_config(lr={"base": 5e-4, "generator": 3e-4}),
    )


def test_validate_training_workflow_options_rejects_decoupled_with_es_embedding_sharing():
    with pytest.raises(ValueError, match="gradients.*dropped"):
        validate_training_workflow_options(
            data_cfg=make_data_config(source={"dataset_name": "HuggingFaceFW/fineweb-edu"}),
            train_cfg=make_train_config(decoupled_training=True),
            model_cfg=make_model_config(backbone_type="hf_deberta_v2", embedding_sharing="es"),
        )


def test_validate_model_config_rejects_rope_knobs_in_hf_mode():
    cfg = make_model_config(backbone_type="hf_deberta_v2", rope={"rope_theta": 50_000.0})
    with pytest.raises(ValueError, match="only valid when model.backbone_type='rope'"):
        validate_model_config(cfg)


def test_validate_model_config_normalizes_hf_attention_kernel_alias():
    cfg = make_model_config(backbone_type="hf_deberta_v2", hf={"attention_kernel": "cache"})
    validate_model_config(cfg)
    assert cfg.hf.attention_kernel == "cached_bmm"

    cfg = make_model_config(backbone_type="hf_deberta_v2", hf={"attention_kernel": "safe"})
    validate_model_config(cfg)
    assert cfg.hf.attention_kernel == "stable"


def test_validate_model_config_normalizes_hf_flash_config():
    cfg = make_model_config(
        backbone_type="hf_deberta_v2",
        hf={"attention_impl": "flash", "flash": {"docblock_bias_seq_len": "4096"}},
    )
    validate_model_config(cfg)

    assert cfg.hf.attention_impl == "flash"
    assert cfg.hf.flash.docblock_bias_seq_len == 4096


def test_validate_model_config_rejects_flash_attention_for_rope():
    cfg = make_model_config(backbone_type="rope", hf={"attention_impl": "flash"})
    with pytest.raises(ValueError, match="only supported with model.backbone_type='hf_deberta_v2'"):
        validate_model_config(cfg)


@pytest.mark.parametrize(
    "dropout",
    [
        {"hidden_prob": 0.1, "attention_probs_prob": 0.0},
        {"hidden_prob": 0.0, "attention_probs_prob": 0.1},
    ],
)
def test_validate_model_config_rejects_flash_attention_with_dropout(dropout: dict[str, float]):
    cfg = make_model_config(backbone_type="hf_deberta_v2", hf={"attention_impl": "flash"}, dropout=dropout)
    with pytest.raises(ValueError, match="requires dropout disabled"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_non_positive_tokenizer_vocab_multiple():
    cfg = make_model_config(tokenizer={"vocab_multiple": 0})
    with pytest.raises(ValueError, match="model.tokenizer.vocab_multiple"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_non_positive_tokenizer_vocab_target():
    cfg = make_model_config(tokenizer={"vocab_target": 0})
    with pytest.raises(ValueError, match="model.tokenizer.vocab_target"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_derived_generator_knobs_with_explicit_generator_source():
    cfg = make_model_config(
        backbone_type="rope",
        pretrained={"generator_path": "microsoft/deberta-v3-small"},
        generator={"hidden_size": 256},
    )
    with pytest.raises(ValueError, match="only used when deriving generator config"):
        validate_model_config(cfg)


def test_validate_model_config_rejects_hf_max_position_embeddings_for_rope():
    cfg = make_model_config(backbone_type="rope", hf={"max_position_embeddings": 1024})
    with pytest.warns(UserWarning, match="model.hf.max_position_embeddings only applies"):
        validate_model_config(cfg)


def test_validate_model_config_allows_hf_max_position_embeddings_in_hf_scratch_mode():
    cfg = make_model_config(
        backbone_type="hf_deberta_v2", from_scratch=True, hf={"max_position_embeddings": 1024}
    )
    validate_model_config(cfg)


def test_validate_model_config_rejects_pretrained_derived_generator_shape_overrides():
    cfg = make_model_config(
        backbone_type="rope",
        from_scratch=False,
        pretrained={"discriminator_path": "local-rope-disc"},
        generator={"hidden_size": 256},
    )
    with pytest.raises(ValueError, match="derived generator weights"):
        validate_model_config(cfg)


def test_validate_model_config_allows_pretrained_derived_generator_layer_override():
    cfg = make_model_config(
        backbone_type="rope",
        from_scratch=False,
        pretrained={"discriminator_path": "local-rope-disc"},
        generator={"num_hidden_layers": 4},
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
    cfg = make_model_config(
        backbone_type="rope",
        from_scratch=False,
        pretrained={"discriminator_path": "runs/deberta-v3-rope/checkpoint-1000"},
    )
    validate_model_config(cfg)


def test_validate_model_config_rejects_scratch_rope_knobs_in_pretrained_mode():
    cfg = make_model_config(
        backbone_type="rope",
        from_scratch=False,
        pretrained={"discriminator_path": "local-rope-disc"},
        rope={"rope_theta": 50_000.0},
    )
    with pytest.raises(ValueError, match="only affect scratch RoPE initialization"):
        validate_model_config(cfg)


def test_validate_model_config_allows_explicit_pretrained_rope_overrides():
    cfg = make_model_config(
        backbone_type="rope",
        from_scratch=False,
        pretrained={"discriminator_path": "local-rope-disc"},
        rope={"pretrained.rope_theta": 50_000.0, "pretrained.norm_arch": "keel"},
    )
    validate_model_config(cfg)


def test_validate_model_config_rejects_pretrained_rope_overrides_in_scratch_mode():
    cfg = make_model_config(backbone_type="rope", from_scratch=True, rope={"pretrained.rope_theta": 50_000.0})
    with pytest.raises(ValueError, match="apply only when model.from_scratch=false"):
        validate_model_config(cfg)


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


@pytest.mark.parametrize(
    "cfg_kwargs,expect_warn",
    [
        # 1A: rope + non-default hf_attention_kernel → warn
        ({"backbone_type": "rope", "hf": {"attention_kernel": "stable"}}, True),
        # 1A: rope + default hf_attention_kernel → no warn
        ({"backbone_type": "rope", "hf": {"attention_kernel": "dynamic"}}, False),
        # 1B: post + non-default keel_alpha_init → warn
        ({"backbone_type": "rope", "rope": {"norm_arch": "post", "keel_alpha_init": 5.0}}, True),
        # 1B: post + non-default keel_alpha_learnable → warn
        (
            {
                "backbone_type": "rope",
                "rope": {"norm_arch": "post", "keel_alpha_learnable": True},
            },
            True,
        ),
        # 1B: keel + keel_alpha_init → no warn
        ({"backbone_type": "rope", "rope": {"norm_arch": "keel", "keel_alpha_init": 5.0}}, False),
        # 1B: post + default keel params → no warn
        ({"backbone_type": "rope", "rope": {"norm_arch": "post"}}, False),
        # 1C: mlp + non-default swiglu_adjust_intermediate → warn
        (
            {
                "backbone_type": "rope",
                "rope": {"ffn_type": "mlp", "swiglu_adjust_intermediate": False},
            },
            True,
        ),
        # 1C: swiglu + swiglu_adjust_intermediate → no warn
        (
            {
                "backbone_type": "rope",
                "rope": {"ffn_type": "swiglu", "swiglu_adjust_intermediate": False},
            },
            False,
        ),
        # 1C: mlp + default swiglu_adjust_intermediate → no warn
        ({"backbone_type": "rope", "rope": {"ffn_type": "mlp"}}, False),
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
    cfg = make_model_config(**cfg_kwargs)
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
    "train_kwargs,logging_kwargs,expect_warn",
    [
        # 1D: compile=false + non-auto scope → warn
        ({"compile": {"enabled": False, "scope": "backbones"}}, None, True),
        # 1D: compile=false + non-default mode → warn
        ({"compile": {"enabled": False, "mode": "max-autotune"}}, None, True),
        # 1D: compile=false + non-default backend → warn
        ({"compile": {"enabled": False, "backend": "aot_eager"}}, None, True),
        # 1D: report_to!=wandb + non-default watch mode → warn
        ({}, {"wandb": {"enabled": False, "watch": "all"}}, True),
        # 1D: report_to!=wandb + non-default watch freq → warn
        ({}, {"wandb": {"enabled": False, "watch_log_freq": 7}}, True),
        # 1D: compile=false + auto scope → no warn
        ({"compile": {"enabled": False, "scope": "auto"}}, None, False),
        # 1D: compile=false + default mode/backend/watch knobs → no warn
        ({"compile": {"enabled": False}}, {"wandb": {"enabled": False}}, False),
        # 1D: wandb backend + watch options active → no inert warning
        ({}, {"wandb": {"enabled": True, "watch": "all", "watch_log_freq": 7}}, False),
        # 1D: compile=true + non-auto scope → no warn
        ({"compile": {"enabled": True, "scope": "backbones"}}, None, False),
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
def test_validate_train_config_inert_param_warnings(train_kwargs, logging_kwargs, expect_warn):
    train_cfg = make_train_config(**train_kwargs)
    logging_cfg = make_logging_config(**logging_kwargs) if logging_kwargs else None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_train_config(train_cfg)
        if logging_cfg is not None:
            validate_logging_config(logging_cfg)
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    if expect_warn:
        assert len(user_warnings) >= 1, (
            f"Expected warning for train={train_kwargs} logging={logging_kwargs}, got none"
        )
    else:
        assert len(user_warnings) == 0, (
            f"Unexpected warning for train={train_kwargs} logging={logging_kwargs}: "
            f"{[str(x.message) for x in user_warnings]}"
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
    model_cfg = make_model_config(**model_kwargs)
    train_cfg = make_train_config(**train_kwargs)
    data_cfg = make_data_config(source={"data_files": "dummy.txt"})
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
        model_cfg=make_model_config(),
        data_cfg=make_data_config(source={"data_files": "dummy.txt"}),
        train_cfg=make_train_config(),
        resume_checkpoint=None,
        is_main_process=True,
        preflight_only=True,
    )
    for filename in ("model_config.json", "data_config.json", "train_config.json", "run_metadata.json"):
        assert not (out / filename).exists()


def test_persist_run_configs_writes_compile_scope_to_metadata(tmp_path: Path):
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    model_cfg = make_model_config()
    data_cfg = make_data_config(source={"data_files": "dummy.txt"})
    train_cfg = make_train_config()
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
    model_cfg = make_model_config()
    data_cfg = make_data_config(source={"data_files": "dummy.txt"})
    train_cfg = make_train_config()

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
