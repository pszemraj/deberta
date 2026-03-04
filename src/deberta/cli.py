"""Unified CLI entrypoint for training and export."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

from deberta.config import (
    _ATTN_IMPL_CHOICES,
    _BACKBONE_CHOICES,
    _EMBED_SHARING_CHOICES,
    _FFN_CHOICES,
    _HF_ATTN_KERNEL_ALIASES,
    _HF_ATTN_KERNEL_CHOICES,
    _HF_MODEL_SIZE_CHOICES,
    _LOGGING_BACKEND_CHOICES,
    _LR_SCHEDULER_CHOICES,
    _MODEL_PROFILE_CHOICES,
    _NORM_ARCH_CHOICES,
    _RESUME_DATA_STRATEGY_CHOICES,
    _SDPA_KERNEL_ALIASES,
    _SDPA_KERNEL_CHOICES,
    _TORCH_COMPILE_BACKEND_ALIASES,
    _TORCH_COMPILE_BACKEND_CHOICES,
    _TORCH_COMPILE_MODE_ALIASES,
    _TORCH_COMPILE_MODE_CHOICES,
    _TORCH_COMPILE_SCOPE_ALIASES,
    _TORCH_COMPILE_SCOPE_CHOICES,
    _WANDB_WATCH_ALIASES,
    _WANDB_WATCH_CHOICES,
    Config,
    apply_dotted_override,
    apply_profile_defaults,
    asdict_without_private,
    iter_leaf_paths_for_dataclass,
    load_config,
    validate_data_config,
    validate_logging_config,
    validate_model_config,
    validate_optim_config,
    validate_train_config,
    validate_training_workflow_options,
)
from deberta.export_cli import (
    ExportArgumentDefaultsHelpFormatter,
    add_export_arguments,
    namespace_to_export_config,
    run_export,
)
from deberta.training import run_pretraining, run_pretraining_dry_run
from deberta.utils.mapping import flatten_mapping
from deberta.utils.types import FALSE_STRINGS, coerce_scalar, parse_bool, unwrap_optional_type

# Nested canonical presets. With a config file, presets still only apply model overrides.
_TRAIN_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "deberta-v3-base": {
        "model": {
            "profile": "deberta_v3_parity",
            "backbone_type": "hf_deberta_v2",
            "tokenizer": {
                "name_or_path": "microsoft/deberta-v3-base",
            },
            "from_scratch": True,
            "embedding_sharing": "gdes",
            "hf": {
                "attention_kernel": "dynamic",
            },
        },
        "data": {
            "source": {
                "dataset_name": "HuggingFaceFW/fineweb-edu",
                "dataset_config_name": "default",
                "train_split": "train",
                "streaming": True,
                "text_column_name": "text",
                "shuffle_buffer_size": 10_000,
            },
            "packing": {
                "enabled": True,
                "block_cross_document_attention": False,
                "max_seq_length": 512,
            },
        },
        "train": {
            "max_steps": 500_000,
        },
        "logging": {
            "backend": "none",
            "wandb": {
                "enabled": False,
            },
        },
    }
}

# Parse-time choices for dotted flags.
_DOTFLAG_CHOICES: dict[str, tuple[str, ...]] = {
    "model.profile": tuple(sorted(_MODEL_PROFILE_CHOICES)),
    "model.backbone_type": tuple(sorted(_BACKBONE_CHOICES)),
    "model.embedding_sharing": tuple(sorted(_EMBED_SHARING_CHOICES)),
    "model.hf.model_size": tuple(sorted(_HF_MODEL_SIZE_CHOICES)),
    "model.hf.attention_kernel": tuple(sorted(_HF_ATTN_KERNEL_CHOICES | set(_HF_ATTN_KERNEL_ALIASES.keys()))),
    "model.rope.norm_arch": tuple(sorted(_NORM_ARCH_CHOICES)),
    "model.rope.pretrained.norm_arch": tuple(sorted(_NORM_ARCH_CHOICES)),
    "model.rope.attention_implementation": tuple(sorted(_ATTN_IMPL_CHOICES)),
    "model.rope.ffn_type": tuple(sorted(_FFN_CHOICES)),
    "model.rope.pretrained.ffn_type": tuple(sorted(_FFN_CHOICES)),
    "train.sdpa_kernel": tuple(sorted(_SDPA_KERNEL_CHOICES | set(_SDPA_KERNEL_ALIASES.keys()))),
    "train.compile.mode": tuple(
        sorted(_TORCH_COMPILE_MODE_CHOICES | set(_TORCH_COMPILE_MODE_ALIASES.keys()))
    ),
    "train.compile.scope": tuple(
        sorted(_TORCH_COMPILE_SCOPE_CHOICES | set(_TORCH_COMPILE_SCOPE_ALIASES.keys()))
    ),
    "train.compile.backend": tuple(
        sorted(_TORCH_COMPILE_BACKEND_CHOICES | set(_TORCH_COMPILE_BACKEND_ALIASES.keys()))
    ),
    "train.checkpoint.resume_data_strategy": tuple(sorted(_RESUME_DATA_STRATEGY_CHOICES)),
    "optim.scheduler.type": tuple(sorted(_LR_SCHEDULER_CHOICES)),
    "logging.backend": tuple(sorted(_LOGGING_BACKEND_CHOICES)),
    "logging.wandb.watch": tuple(sorted(_WANDB_WATCH_CHOICES | set(_WANDB_WATCH_ALIASES.keys()))),
}


def _parse_bool(value: str) -> bool:
    """Parse a CLI string value into bool.

    :param str value: Raw value.
    :return bool: Parsed boolean.
    """
    try:
        return parse_bool(value, allow_numeric=False)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}") from exc


def _should_fast_exit_after_train(*, explicit_argv: bool) -> bool:
    """Decide whether train CLI should use os._exit(0) on success.

    :param bool explicit_argv: True when main() was called with argv list.
    :return bool: Whether to fast-exit.
    """
    if explicit_argv:
        return False
    raw = str(os.getenv("DEBERTA_FAST_EXIT_AFTER_TRAIN", "1")).strip().lower()
    return raw not in FALSE_STRINGS


def _argparse_type(field_type: Any) -> Any:
    """Resolve argparse type callable from a dataclass field type.

    :param Any field_type: Dataclass field type.
    :return Any: Callable/type for argparse.
    """
    t, allows_none = unwrap_optional_type(field_type)
    if allows_none:

        def _parse_optional(value: str) -> Any:
            """Parse optional scalar CLI values.

            :param str value: Raw CLI value.
            :raises argparse.ArgumentTypeError: On incompatible scalar conversion.
            :return Any: Parsed value (or ``None`` for ``null``/``none``).
            """
            try:
                return coerce_scalar(value, t, allow_none=True, allow_bool_numeric=False)
            except ValueError as exc:
                if t is bool:
                    raise argparse.ArgumentTypeError(
                        f"Expected a boolean value or null/none, got: {value}"
                    ) from exc
                if t is int:
                    raise argparse.ArgumentTypeError(
                        f"Expected an integer value or null/none, got: {value}"
                    ) from exc
                if t is float:
                    raise argparse.ArgumentTypeError(
                        f"Expected a numeric value or null/none, got: {value}"
                    ) from exc
                raise argparse.ArgumentTypeError(f"Invalid value: {value}") from exc

        return _parse_optional

    if t is bool:
        return _parse_bool
    if t in {str, int, float}:
        return t
    return str


def _dest_for_path(path: str) -> str:
    """Build argparse destination key from a dotted path.

    :param str path: Dotted path.
    :return str: Argparse destination.
    """
    return f"dot__{str(path).replace('.', '__')}"


def _flag_value_to_override_text(value: Any) -> str:
    """Serialize a typed parsed value into apply_dotted_override text form.

    :param Any value: Parsed value.
    :return str: Serialized override value.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if bool(value) else "false"
    return str(value)


def _flatten_cfg(cfg: Config) -> dict[str, Any]:
    """Flatten a Config object into dotted leaf key/value pairs.

    :param Config cfg: Config object.
    :return dict[str, Any]: Flattened mapping.
    """
    return flatten_mapping(asdict_without_private(cfg))


def _add_dotflags(parser: argparse.ArgumentParser) -> dict[str, str]:
    """Register direct dotted flags for all config leaves.

    :param argparse.ArgumentParser parser: Target parser.
    :return dict[str, str]: Mapping of dotted path to argparse destination.
    """

    def _choices_for_path(path: str, field_type: Any) -> tuple[Any, ...] | None:
        """Return argparse choices for one dotflag, including ``None`` when optional.

        Optional constrained fields should accept ``none``/``null`` so users can
        clear values loaded from config files.

        :param str path: Dotted field path.
        :param Any field_type: Dataclass field type annotation.
        :return tuple[Any, ...] | None: Argparse choices or ``None`` when unconstrained.
        """
        base = _DOTFLAG_CHOICES.get(str(path))
        if base is None:
            return None
        _target_t, allows_none = unwrap_optional_type(field_type)
        if not allows_none:
            return base
        if any(choice is None for choice in base):
            return base
        return tuple(base) + (None,)

    dest_by_path: dict[str, str] = {}
    group = parser.add_argument_group("Config Dotflags")

    for path, field_type in iter_leaf_paths_for_dataclass(Config):
        dest = _dest_for_path(path)
        dest_by_path[str(path)] = dest
        group.add_argument(
            f"--{path}",
            dest=dest,
            default=argparse.SUPPRESS,
            type=_argparse_type(field_type),
            choices=_choices_for_path(str(path), field_type),
            help=f"Set {path}",
        )
    return dest_by_path


def _build_train_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> dict[str, str]:
    """Create `deberta train` parser.

    :param argparse._SubParsersAction[argparse.ArgumentParser] subparsers: Parent subparsers action.
    :return dict[str, str]: Dotted path to argparse destination mapping.
    """
    train = subparsers.add_parser(
        "train",
        help="Run pretraining.",
        description=(
            "Run RTD pretraining from a YAML/JSON config file and/or dotted CLI flags. "
            "Example: deberta train cfg.yaml --train.max_steps 1000 --optim.scheduler.warmup_steps 500"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Optional YAML/JSON config path. Dotflags override matching config keys.",
    )
    train.add_argument(
        "--preset",
        choices=tuple(sorted(_TRAIN_PRESETS.keys())),
        default=None,
        help=(
            "Optional training preset. With a config file, presets only override model fields. "
            "Without a config file, presets provide model+data+train+optim+logging defaults."
        ),
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run non-destructive preflight checks (config validation, output/resume checks, "
            "tokenizer+dataset+collator probe, backbone-config build) and exit without "
            "training/checkpoint writes. May still access network and populate tokenizer/dataset caches."
        ),
    )
    return _add_dotflags(train)


def _build_export_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Create `deberta export` parser.

    :param argparse._SubParsersAction[argparse.ArgumentParser] subparsers: Parent subparsers action.
    """
    export = subparsers.add_parser(
        "export",
        help="Consolidate and export checkpoint to standalone HF artifacts.",
        formatter_class=ExportArgumentDefaultsHelpFormatter,
    )
    add_export_arguments(export)


def _build_main_parser() -> argparse.ArgumentParser:
    """Build top-level CLI parser for `deberta`.

    :return argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="deberta",
        description="deberta CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    dotflag_map = _build_train_parser(subparsers)
    _build_export_parser(subparsers)
    parser._dotflag_map = dotflag_map  # type: ignore[attr-defined]
    return parser


def _apply_preset(*, cfg: Config, preset_name: str, with_config_file: bool) -> tuple[Config, str]:
    """Apply a named preset payload to Config.

    :param Config cfg: Current config.
    :param str preset_name: Preset name.
    :param bool with_config_file: Whether a config file was loaded.
    :return tuple[Config, str]: Updated config and mode label.
    """
    name = str(preset_name).strip().lower()
    preset = _TRAIN_PRESETS.get(name)
    if preset is None:
        allowed = ", ".join(sorted(_TRAIN_PRESETS.keys()))
        raise ValueError(f"Unknown train preset: {preset_name!r}. Available presets: {allowed}.")

    sections = ["model"] if with_config_file else ["model", "data", "train", "optim", "logging"]
    for section in sections:
        for path, value in flatten_mapping(dict(preset.get(section, {})), prefix=section).items():
            cfg = apply_dotted_override(cfg, f"{path}={_flag_value_to_override_text(value)}")

    if with_config_file:
        return cfg, "model-only overrides (config file provided)"
    return cfg, "model+data+train+optim+logging defaults (no config file)"


def _apply_dotflags(
    *,
    cfg: Config,
    ns: argparse.Namespace,
    dotflag_map: dict[str, str],
) -> tuple[Config, dict[str, str]]:
    """Apply provided dotted flags onto config.

    :param Config cfg: Current config.
    :param argparse.Namespace ns: Parsed namespace.
    :param dict[str, str] dotflag_map: Path -> argparse destination map.
    :return tuple[Config, dict[str, str]]: Updated config and reason mapping.
    """
    reasons: dict[str, str] = {}
    for path, dest in dotflag_map.items():
        if not hasattr(ns, dest):
            continue
        value = getattr(ns, dest)
        cfg = apply_dotted_override(cfg, f"{path}={_flag_value_to_override_text(value)}")
        reasons[str(path)] = f"CLI override (--{path})"
    return cfg, reasons


def _emit_config_mutation_warnings(
    *,
    cfg_path: Path | None,
    before: Config,
    after: Config,
    reasons: dict[str, str],
) -> None:
    """Emit explicit warnings when runtime config differs from baseline values.

    :param Path | None cfg_path: Optional config file path.
    :param Config before: Baseline config.
    :param Config after: Final config.
    :param dict[str, str] reasons: Optional reason mapping.
    """
    if cfg_path is None:
        return

    flat_before = _flatten_cfg(before)
    flat_after = _flatten_cfg(after)
    changes: list[tuple[str, Any, Any, str]] = []
    for key, old in flat_before.items():
        if key not in flat_after:
            continue
        new = flat_after[key]
        if new != old:
            reason = reasons.get(str(key), "runtime normalization/defaulting")
            changes.append((str(key), old, new, reason))

    if not changes:
        return

    print(
        f"[deberta][config-warning] Loaded config '{cfg_path}' was modified after load:",
        file=sys.stderr,
    )
    for key, old, new, reason in changes:
        print(
            f"[deberta][config-warning] {key}: {old!r} -> {new!r} ({reason})",
            file=sys.stderr,
        )


def _run_train(
    ns: argparse.Namespace,
    *,
    dotflag_map: dict[str, str],
) -> None:
    """Run train flow from parsed namespace.

    :param argparse.Namespace ns: Parsed train args.
    :param dict[str, str] dotflag_map: Dotted path -> argparse destination mapping.
    """
    cfg_path: Path | None = None
    if ns.config is not None:
        cfg_path = Path(ns.config).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(str(cfg_path))
        suffix = cfg_path.suffix.lower()
        if suffix not in {".json", ".yaml", ".yml"}:
            raise ValueError("Config file must end with .json, .yaml, or .yml")
        cfg = load_config(cfg_path)
        baseline_cfg = copy.deepcopy(cfg)
    else:
        cfg = Config()
        baseline_cfg = Config()

    reason_overrides: dict[str, str] = {}

    if ns.preset:
        cfg, preset_mode = _apply_preset(
            cfg=cfg,
            preset_name=str(ns.preset),
            with_config_file=cfg_path is not None,
        )
        print(f"Applying train preset '{ns.preset}': {preset_mode}.")

    cfg, cli_reasons = _apply_dotflags(cfg=cfg, ns=ns, dotflag_map=dotflag_map)
    reason_overrides.update(cli_reasons)

    apply_profile_defaults(model_cfg=cfg.model, train_cfg=cfg.train, optim_cfg=cfg.optim)

    validate_model_config(cfg.model)
    validate_data_config(cfg.data)
    validate_train_config(cfg.train)
    validate_optim_config(cfg.optim)
    validate_logging_config(cfg.logging)
    validate_training_workflow_options(
        data_cfg=cfg.data,
        train_cfg=cfg.train,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        logging_cfg=cfg.logging,
    )

    _emit_config_mutation_warnings(
        cfg_path=cfg_path,
        before=baseline_cfg,
        after=cfg,
        reasons=reason_overrides,
    )

    if bool(getattr(ns, "dry_run", False)):
        report = run_pretraining_dry_run(
            model_cfg=cfg.model,
            data_cfg=cfg.data,
            train_cfg=cfg.train,
            optim_cfg=cfg.optim,
            logging_cfg=cfg.logging,
            config_path=cfg_path,
        )
        summary = (
            "Dry-run preflight OK: "
            f"checkpoint_output_dir={report['checkpoint_output_dir']}, "
            f"logging_output_dir={report['logging_output_dir']}, "
            f"resume_checkpoint={report['resume_checkpoint']}, "
            f"effective_compile_scope={report['effective_compile_scope']}, "
            f"sample_batch_shape={report['sample_batch_shape']}, "
            f"sample_active_tokens={report['sample_active_tokens']}, "
            f"tokenizer_vocab_size={report['tokenizer_vocab_size']}."
        )
        print(summary)
        return

    run_pretraining(
        model_cfg=cfg.model,
        data_cfg=cfg.data,
        train_cfg=cfg.train,
        optim_cfg=cfg.optim,
        logging_cfg=cfg.logging,
        config_path=cfg_path,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the `deberta` command.

    :param list[str] | None argv: Optional CLI argv (excluding program name).
    """
    explicit_argv = argv is not None
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_main_parser()
    dotflag_map = dict(getattr(parser, "_dotflag_map", {}))
    args = parser.parse_args(argv)

    if args.command == "train":
        _run_train(args, dotflag_map=dotflag_map)
        if _should_fast_exit_after_train(explicit_argv=explicit_argv):
            os._exit(0)
        return

    if args.command == "export":
        export_cfg = namespace_to_export_config(args)
        run_export(export_cfg)
        return


if __name__ == "__main__":
    main()
