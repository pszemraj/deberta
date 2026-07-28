import json
import logging
import runpy
import sys
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
from _fakes import capture_run_pretraining_kwargs, setup_pretraining_mocks

import deberta.cli as cli_mod
import deberta.training.runtime as runtime_mod
from deberta.config import (
    _looks_like_hf_deberta_checkpoint,
    validate_data_config,
    validate_logging_config,
    validate_model_config,
    validate_optim_config,
    validate_train_config,
    validate_training_workflow_options,
)
from deberta.training.entrypoint import run_pretraining_dry_run
from deberta.training.run_config import _build_run_metadata, _persist_or_validate_run_configs
from deberta.training.steps import _global_grad_l2_norm


def test_module_entrypoint_invokes_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["deberta"])
    with pytest.raises(SystemExit):
        runpy.run_module("deberta", run_name="__main__")


def test_package_bounds_transformers_to_supported_major() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    dependency = next(
        line.strip().removesuffix(",").strip('"')
        for line in pyproject.splitlines()
        if line.strip().startswith('"transformers')
    )

    assert dependency == "transformers>=4.45.0,<5"


def test_main_cli_train_subcommand_loads_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):

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
                "optim:",
                "  scheduler:",
                "    warmup_steps: 0",
            ]
        ),
        encoding="utf-8",
    )

    seen = capture_run_pretraining_kwargs(monkeypatch, cli_mod)
    cli_mod.main(["train", str(cfg_path)])

    assert "train_cfg" in seen
    assert seen["data_cfg"].packing.max_seq_length == 32
    assert seen["train_cfg"].max_steps == 5
    assert seen["optim_cfg"].scheduler.warmup_steps == 0
    assert seen["config_path"] == cfg_path


def test_main_cli_train_rejects_unknown_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_main_cli_train_dry_run_calls_preflight_and_skips_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    seen: dict[str, Any] = {}

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
                "  scheduler:",
                "    warmup_steps: 0",
            ]
        ),
        encoding="utf-8",
    )

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

    cli_mod.main(["train", str(cfg_path), "--dry-run"])
    assert "train_cfg" in seen
    assert int(seen["train_cfg"].max_steps) == 5
    assert seen["config_path"] == cfg_path


def test_train_parser_requires_config() -> None:
    parser = cli_mod._build_main_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])


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
            optim_cfg=make_optim_config(scheduler={"warmup_steps": 0}),
            config_path=None,
        )


def test_run_pretraining_dry_run_preserves_missing_resume_tokenizer_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    missing_tokenizer = run_dir / "tokenizer"
    missing_artifact_error = FileNotFoundError(
        f"Run is missing required materialized tokenizer directory: {missing_tokenizer}"
    )

    def _missing_materialized_tokenizer(_run_dir: Path) -> Path:
        raise missing_artifact_error

    entrypoint_mod = setup_pretraining_mocks(
        monkeypatch,
        extra_patches={
            "_resolve_resume_checkpoint": lambda **_kwargs: str(checkpoint),
            "_persist_or_validate_run_configs": lambda **_kwargs: None,
            "materialized_tokenizer_path": _missing_materialized_tokenizer,
        },
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        entrypoint_mod.run_pretraining_dry_run(
            model_cfg=make_model_config(),
            data_cfg=make_data_config(source={"dataset_name": "dummy-dataset"}),
            train_cfg=make_train_config(
                checkpoint={
                    "output_dir": str(run_dir),
                    "resume_from_checkpoint": str(checkpoint),
                },
                max_steps=5,
            ),
            optim_cfg=make_optim_config(scheduler={"warmup_steps": 0}),
        )

    assert exc_info.value is missing_artifact_error


def test_run_pretraining_dry_run_rejects_missing_flash_extra_before_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint_mod = setup_pretraining_mocks(monkeypatch)
    import_error = ModuleNotFoundError("No module named 'flashdeberta'")
    dataset_touched = False

    def _unexpected_dataset_load(_cfg):
        nonlocal dataset_touched
        dataset_touched = True
        raise AssertionError("dataset setup must follow FlashDeBERTa dependency validation")

    monkeypatch.setattr(runtime_mod, "_flashdeberta_runtime_import_error", lambda: import_error)
    monkeypatch.setattr(entrypoint_mod, "load_hf_dataset", _unexpected_dataset_load)
    output_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[flash\]'") as exc_info:
        run_pretraining_dry_run(
            model_cfg=make_model_config(hf={"attention_impl": "flash"}),
            data_cfg=make_data_config(source={"dataset_name": "dummy-dataset"}),
            train_cfg=make_train_config(
                checkpoint={"output_dir": str(output_dir)},
                max_steps=5,
            ),
            optim_cfg=make_optim_config(scheduler={"warmup_steps": 0}),
        )

    assert exc_info.value.__cause__ is import_error
    assert dataset_touched is False
    assert not output_dir.exists()


def test_run_pretraining_dry_run_releases_sample_iterator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint_mod = setup_pretraining_mocks(monkeypatch)
    close_calls: list[bool] = []
    collect_calls: list[bool] = []
    shuffle_buffer_sizes: list[int] = []

    class _CloseableDataset:
        def __iter__(self):
            return self

        def __next__(self):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "labels": torch.tensor([[-100, 2, -100]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }

        def close(self) -> None:
            close_calls.append(True)

    dataset = _CloseableDataset()

    def _build_dataset(**kwargs):
        shuffle_buffer_sizes.append(int(kwargs["data_cfg"].source.shuffle_buffer_size))
        return dataset, lambda rows: rows[0]

    monkeypatch.setattr(
        entrypoint_mod,
        "_build_train_dataset_and_collator",
        _build_dataset,
    )
    monkeypatch.setattr(entrypoint_mod.gc, "collect", lambda: collect_calls.append(True) or 0)

    result = entrypoint_mod.run_pretraining_dry_run(
        model_cfg=make_model_config(),
        data_cfg=make_data_config(source={"dataset_name": "dummy-dataset"}),
        train_cfg=make_train_config(
            checkpoint={"output_dir": str(tmp_path / "run")},
            max_steps=5,
        ),
        optim_cfg=make_optim_config(scheduler={"warmup_steps": 0}),
    )

    assert result["status"] == "ok"
    assert close_calls == [True]
    assert collect_calls == [True]
    assert shuffle_buffer_sizes == [0]


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


@pytest.mark.parametrize(
    "scheduler_type",
    ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"],
)
def test_validate_training_workflow_options_rejects_full_run_warmup(
    scheduler_type: str,
) -> None:
    with pytest.raises(ValueError, match="warmup_steps must be less than train.max_steps"):
        validate_training_workflow_options(
            data_cfg=make_data_config(source={"dataset_name": "dummy"}),
            train_cfg=make_train_config(max_steps=10),
            optim_cfg=make_optim_config(
                scheduler={"type": scheduler_type, "warmup_steps": 10},
            ),
        )


def test_validate_training_workflow_options_allows_constant_to_ignore_warmup() -> None:
    validate_training_workflow_options(
        data_cfg=make_data_config(source={"dataset_name": "dummy"}),
        train_cfg=make_train_config(max_steps=10),
        optim_cfg=make_optim_config(
            scheduler={"type": "constant", "warmup_steps": 10_000},
        ),
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
        hf={"attention_impl": "flash", "flash": {"kernel_overrides_path": "  "}},
    )
    validate_model_config(cfg)

    assert cfg.hf.attention_impl == "flash"
    assert cfg.hf.flash.kernel_overrides_path is None


def test_validate_model_config_warns_when_flash_options_are_inactive() -> None:
    cfg = make_model_config(
        backbone_type="hf_deberta_v2",
        hf={"flash": {"kernel_overrides_path": "custom.json"}},
    )

    with pytest.warns(UserWarning, match=r"model\.hf\.flash\.\* has no effect"):
        validate_model_config(cfg)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (None, "Unable to load FlashDeBERTa kernel overrides"),
        ({"kernels": [{}]}, r"kernels\[0\].*missing required field"),
        (
            {"seq_buckets": [{"name": "invalid", "min_seq_len": "soon"}]},
            r"seq_buckets\[0\]\.min_seq_len must be a positive integer",
        ),
        (
            {"seq_buckets": [{"name": "invalid", "min_density": 1.1}]},
            r"seq_buckets\[0\]\.min_density must be a finite number in \[0, 1\]",
        ),
        (
            {
                "route_policies": {
                    "padding": [
                        {
                            "seq_bucket": "default",
                            "choice": "fixed",
                            "max_batch_size": 0,
                        }
                    ]
                }
            },
            r"route_policies\.padding\[0\]\.max_batch_size must be a positive integer",
        ),
        (
            {
                "kernels": [
                    {
                        "route": "fixed",
                        "kind": "fwd",
                        "seq_bucket": "default",
                        "block_m": 16,
                        "block_n": 16,
                        "num_stages": 1,
                        "num_warps": 4,
                        "max_query_len": "many",
                    }
                ]
            },
            r"kernels\[0\]\.max_query_len must be a positive integer",
        ),
        (
            {
                "kernels": [
                    {
                        "route": "fixed",
                        "kind": "fwd",
                        "seq_bucket": "default",
                        "block_m": 3,
                        "block_n": 16,
                        "num_stages": 1,
                        "num_warps": 4,
                    }
                ]
            },
            r"kernels\[0\]\.block_m must be a positive power of two",
        ),
        (
            {
                "kernels": [
                    {
                        "route": "fixed",
                        "kind": "fwd",
                        "seq_bucket": "default",
                        "block_m": 16,
                        "block_n": 24,
                        "num_stages": 1,
                        "num_warps": 4,
                    }
                ]
            },
            r"kernels\[0\]\.block_n must be a positive power of two",
        ),
        (
            {
                "kernels": [
                    {
                        "route": "fixed",
                        "kind": "fwd",
                        "seq_bucket": "default",
                        "block_m": 16,
                        "block_n": 16,
                        "num_stages": 1,
                        "num_warps": 3,
                    }
                ]
            },
            r"kernels\[0\]\.num_warps must be a positive power of two",
        ),
        (
            {
                "kernels": [
                    {
                        "route": "fixed",
                        "kind": "fwd",
                        "seq_bucket": "default",
                        "block_m": 16,
                        "block_n": 16,
                        "num_stages": 1,
                        "num_warps": 4,
                        "num_head": 12,
                    }
                ]
            },
            r"kernels\[0\] has unknown field\(s\): num_head",
        ),
    ],
    ids=[
        "missing_file",
        "malformed_kernel_row",
        "malformed_bucket_bound",
        "out_of_range_density",
        "non_positive_policy_bound",
        "malformed_kernel_bound",
        "non_power_of_two_block",
        "non_power_of_two_block_n",
        "illegal_num_warps",
        "unknown_kernel_constraint",
    ],
)
def test_validate_model_config_rejects_invalid_flash_kernel_overrides(
    tmp_path: Path,
    payload: dict[str, object] | None,
    match: str,
) -> None:
    override_path = tmp_path / "flash-overrides.json"
    if payload is not None:
        override_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = make_model_config(
        backbone_type="hf_deberta_v2",
        hf={
            "attention_impl": "flash",
            "flash": {"kernel_overrides_path": str(override_path)},
        },
    )

    with pytest.raises(ValueError, match=match):
        validate_model_config(cfg)


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


def test_build_run_metadata_records_flash_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata as importlib_metadata

    monkeypatch.setattr(
        importlib_metadata,
        "version",
        lambda name: "0.0.7" if str(name) == "flashdeberta" else "0.0.0",
    )
    model_cfg = make_model_config(
        hf={"attention_impl": "flash"},
    )
    validate_model_config(model_cfg)

    meta = _build_run_metadata(model_cfg=model_cfg)

    assert meta["flash_attention"]["requested_attention_impl"] == "flash"
    assert meta["flash_attention"]["requested_flash_config"]["kernel_overrides_path"] is None
    assert meta["flash_attention"]["flashdeberta_version"] == "0.0.7"


def test_build_run_metadata_omits_flash_attention_for_eager_config() -> None:
    meta = _build_run_metadata(model_cfg=make_model_config())

    assert "flash_attention" not in meta


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


def test_persist_run_configs_allows_runtime_knob_drift_and_warns_on_compile_scope(
    tmp_path: Path, caplog
) -> None:
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    model_cfg = make_model_config()
    data_cfg = make_data_config(source={"data_files": "dummy.txt"})
    train_cfg = make_train_config(compile={"enabled": True, "scope": "backbones"})

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

    resumed_train_cfg = make_train_config(
        dataloader={"pin_memory": False},
        compile={"enabled": True, "scope": "ffn"},
    )

    # Runtime-only pinning and compile scope changes are resume-compatible; scope drift is recorded.
    with caplog.at_level(logging.WARNING):
        _persist_or_validate_run_configs(
            output_dir=out,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            train_cfg=resumed_train_cfg,
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
