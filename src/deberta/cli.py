"""Unified CLI entrypoint for training and export."""

from __future__ import annotations

import argparse
import os
import sys
import types
from dataclasses import fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from deberta.config import (
    _ATTN_IMPL_CHOICES,
    _BACKBONE_CHOICES,
    _EMBED_SHARING_CHOICES,
    _FFN_CHOICES,
    _HF_ATTN_KERNEL_ALIASES,
    _HF_ATTN_KERNEL_CHOICES,
    _LR_SCHEDULER_CHOICES,
    _MODEL_PROFILE_CHOICES,
    _NORM_ARCH_CHOICES,
    _REPORT_TO_CHOICES,
    _RESUME_DATA_STRATEGY_CHOICES,
    _SDPA_KERNEL_ALIASES,
    _SDPA_KERNEL_CHOICES,
    _TORCH_COMPILE_BACKEND_ALIASES,
    _TORCH_COMPILE_BACKEND_CHOICES,
    _TORCH_COMPILE_MODE_ALIASES,
    _TORCH_COMPILE_MODE_CHOICES,
    _TORCH_COMPILE_SCOPE_ALIASES,
    _TORCH_COMPILE_SCOPE_CHOICES,
    DataConfig,
    ModelConfig,
    TrainConfig,
    apply_profile_defaults,
    validate_data_config,
    validate_model_config,
    validate_train_config,
    validate_training_workflow_options,
)
from deberta.export_cli import add_export_arguments, namespace_to_export_config, run_export
from deberta.io_utils import load_json_mapping
from deberta.training import run_pretraining, run_pretraining_dry_run

_TRAIN_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "deberta-v3-base": {
        # Paper-faithful architecture path for DeBERTa-v3 experiments.
        "model": {
            "profile": "deberta_v3_parity",
            "backbone_type": "hf_deberta_v2",
            "tokenizer_name_or_path": "microsoft/deberta-v3-base",
            "discriminator_model_name_or_path": "microsoft/deberta-v3-base",
            "generator_model_name_or_path": None,
            "from_scratch": True,
            "embedding_sharing": "gdes",
            "hf_attention_kernel": "dynamic",
        },
        # Applied only when no config path is provided.
        "data": {
            "dataset_name": "HuggingFaceFW/fineweb-edu",
            "dataset_config_name": "default",
            "train_split": "train",
            "streaming": True,
            "pack_sequences": True,
            "block_cross_document_attention": False,
            "text_column_name": "text",
            "max_seq_length": 512,
            "shuffle_buffer_size": 10_000,
        },
        # Applied only when no config path is provided.
        "train": {
            "max_steps": 500_000,
            "report_to": "none",
        },
    }
}


def _split_flat_dict(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split flat config keys into model/data/train sections.

    :param dict[str, Any] raw: Flat config mapping.
    :return tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: Parsed model/data/train dicts.
    """

    model_keys = {f.name for f in fields(ModelConfig)}
    data_keys = {f.name for f in fields(DataConfig)}
    train_keys = {f.name for f in fields(TrainConfig)}

    model_dict: dict[str, Any] = {}
    data_dict: dict[str, Any] = {}
    train_dict: dict[str, Any] = {}

    unknown: dict[str, Any] = {}

    for k, v in raw.items():
        if k in model_keys:
            model_dict[k] = v
        elif k in data_keys:
            data_dict[k] = v
        elif k in train_keys:
            train_dict[k] = v
        else:
            unknown[k] = v

    if unknown:
        keys = ", ".join(sorted(unknown.keys()))
        raise ValueError(f"Unknown keys in config file (not in ModelConfig/DataConfig/TrainConfig): {keys}")

    return model_dict, data_dict, train_dict


def _load_yaml(path: Path) -> tuple[ModelConfig, DataConfig, TrainConfig]:
    """Load model/data/train dataclasses from a YAML file.

    :param Path path: YAML path.
    :return tuple[ModelConfig, DataConfig, TrainConfig]: Parsed config dataclasses.
    """
    model_dict, data_dict, train_dict = _load_yaml_sections(path)
    return ModelConfig(**model_dict), DataConfig(**data_dict), TrainConfig(**train_dict)


def _load_yaml_sections(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load nested model/data/train section mappings from YAML.

    :param Path path: YAML path.
    :return tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: Parsed section mappings.
    """
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pyyaml is required for YAML config files. Install with `pip install pyyaml`."
        ) from e

    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("YAML config must parse to a dict.")

    return _split_nested_or_flat_sections(raw, format_name="YAML")


def _load_json(path: Path) -> tuple[ModelConfig, DataConfig, TrainConfig]:
    """Load model/data/train dataclasses from a JSON file.

    :param Path path: JSON path.
    :return tuple[ModelConfig, DataConfig, TrainConfig]: Parsed config dataclasses.
    """
    model_dict, data_dict, train_dict = _load_json_sections(path)
    return ModelConfig(**model_dict), DataConfig(**data_dict), TrainConfig(**train_dict)


def _load_json_sections(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load nested model/data/train section mappings from JSON.

    :param Path path: JSON path.
    :return tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: Parsed section mappings.
    """
    raw = load_json_mapping(path)
    return _split_nested_or_flat_sections(raw, format_name="JSON")


def _split_nested_or_flat_sections(
    raw: dict[str, Any], *, format_name: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split nested/flat config mappings into model/data/train sections.

    :param dict[str, Any] raw: Parsed config mapping.
    :param str format_name: Human-readable format label (e.g., JSON/YAML).
    :return tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: Section dictionaries.
    """
    if any(k in raw for k in ("model", "data", "train")):
        unknown_top = sorted(k for k in raw.keys() if k not in {"model", "data", "train"})
        if unknown_top:
            raise ValueError(
                f"Unknown top-level keys in nested {format_name} config "
                f"(expected only model/data/train): {', '.join(unknown_top)}"
            )
        model_dict = raw.get("model", {}) or {}
        data_dict = raw.get("data", {}) or {}
        train_dict = raw.get("train", {}) or {}
        if (
            not isinstance(model_dict, dict)
            or not isinstance(data_dict, dict)
            or not isinstance(train_dict, dict)
        ):
            raise ValueError(f"{format_name} config sections model/data/train must be dicts.")
        return model_dict, data_dict, train_dict

    return _split_flat_dict(raw)


def _parse_bool(value: str) -> bool:
    """Parse a CLI string value into bool.

    :param str value: Raw value.
    :return bool: Parsed boolean.
    """
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def _should_fast_exit_after_train(*, explicit_argv: bool) -> bool:
    """Decide whether train CLI should use os._exit(0) on success.

    :param bool explicit_argv: True when main() was called with argv list.
    :return bool: Whether to fast-exit.
    """
    if explicit_argv:
        return False
    raw = str(os.getenv("DEBERTA_FAST_EXIT_AFTER_TRAIN", "1")).strip().lower()
    return raw not in {"0", "false", "f", "no", "n", "off"}


def _unwrap_optional(field_type: Any) -> Any:
    """Unwrap Optional[T] style annotations.

    :param Any field_type: Type annotation.
    :return Any: Base type when optional, otherwise original type.
    """
    origin = get_origin(field_type)
    if origin in {Union, types.UnionType}:
        args = [a for a in get_args(field_type) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return field_type


def _argparse_type(field_type: Any) -> Any:
    """Resolve argparse type callable from a dataclass field type.

    :param Any field_type: Dataclass field type.
    :return Any: Callable/type for argparse.
    """
    t = _unwrap_optional(field_type)
    if t is bool:
        return _parse_bool
    if t in {str, int, float}:
        return t
    return str


def _add_dataclass_flags(parser: argparse.ArgumentParser, cls: Any, *, group_name: str) -> None:
    """Add argparse flags for all dataclass fields.

    :param argparse.ArgumentParser parser: Train subparser.
    :param Any cls: Dataclass type.
    :param str group_name: Group label.
    """
    group = parser.add_argument_group(group_name)
    type_hints = get_type_hints(cls)
    constrained_choices: dict[str, tuple[str, ...]] = {
        "profile": tuple(sorted(_MODEL_PROFILE_CHOICES)),
        "backbone_type": tuple(sorted(_BACKBONE_CHOICES)),
        "norm_arch": tuple(sorted(_NORM_ARCH_CHOICES)),
        "pretrained_norm_arch": tuple(sorted(_NORM_ARCH_CHOICES)),
        "attention_implementation": tuple(sorted(_ATTN_IMPL_CHOICES)),
        "ffn_type": tuple(sorted(_FFN_CHOICES)),
        "pretrained_ffn_type": tuple(sorted(_FFN_CHOICES)),
        "hf_attention_kernel": tuple(sorted(_HF_ATTN_KERNEL_CHOICES | set(_HF_ATTN_KERNEL_ALIASES.keys()))),
        "embedding_sharing": tuple(sorted(_EMBED_SHARING_CHOICES)),
        "report_to": tuple(sorted(_REPORT_TO_CHOICES)),
        "lr_scheduler_type": tuple(sorted(_LR_SCHEDULER_CHOICES)),
        # Keep legacy aliases parseable for UX compatibility.
        "sdpa_kernel": tuple(sorted(_SDPA_KERNEL_CHOICES | set(_SDPA_KERNEL_ALIASES.keys()))),
        "torch_compile_mode": tuple(
            sorted(_TORCH_COMPILE_MODE_CHOICES | set(_TORCH_COMPILE_MODE_ALIASES.keys()))
        ),
        "torch_compile_scope": tuple(
            sorted(_TORCH_COMPILE_SCOPE_CHOICES | set(_TORCH_COMPILE_SCOPE_ALIASES.keys()))
        ),
        "torch_compile_backend": tuple(
            sorted(_TORCH_COMPILE_BACKEND_CHOICES | set(_TORCH_COMPILE_BACKEND_ALIASES.keys()))
        ),
        "resume_data_strategy": tuple(sorted(_RESUME_DATA_STRATEGY_CHOICES)),
    }

    for f in fields(cls):
        help_text = str(f.metadata.get("help", "")) if f.metadata else ""
        default = f.default
        field_type = type_hints.get(f.name, f.type)
        choices = constrained_choices.get(f.name)
        group.add_argument(
            f"--{f.name}",
            f"--{f.name.replace('_', '-')}",
            dest=f.name,
            default=default,
            type=_argparse_type(field_type),
            choices=choices,
            help=help_text,
        )


def _build_train_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Create `deberta train` parser.

    :param argparse._SubParsersAction[argparse.ArgumentParser] subparsers: Parent subparsers action.
    """
    train = subparsers.add_parser(
        "train",
        help="Run pretraining.",
        description=(
            "Run RTD pretraining from a YAML/JSON config file or from explicit CLI flags. "
            "Examples: `deberta train configs/pretrain_hf_deberta_v2_parity_small.yaml` or "
            "`deberta train --dataset_name HuggingFaceFW/fineweb-edu --dataset_config_name default --max_steps 1000`."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Optional YAML/JSON config path. CLI flags override matching config keys.",
    )
    train.add_argument(
        "--preset",
        choices=tuple(sorted(_TRAIN_PRESETS.keys())),
        default=None,
        help=(
            "Optional training preset. With a config file, presets only override model fields. "
            "Without a config file, presets provide model+data+train defaults."
        ),
    )
    _add_dataclass_flags(train, ModelConfig, group_name="Model")
    _add_dataclass_flags(train, DataConfig, group_name="Data")
    _add_dataclass_flags(train, TrainConfig, group_name="Train")
    train.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run non-destructive preflight checks (config validation, output/resume checks, "
            "tokenizer+dataset+collator probe, backbone-config build) and exit without training."
        ),
    )


def _build_export_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Create `deberta export` parser.

    :param argparse._SubParsersAction[argparse.ArgumentParser] subparsers: Parent subparsers action.
    """
    export = subparsers.add_parser(
        "export",
        help="Consolidate and export checkpoint to standalone HF artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    _build_train_parser(subparsers)
    _build_export_parser(subparsers)
    return parser


def _subcommand_argv(argv: list[str], command: str) -> list[str]:
    """Return raw argv slice after a subcommand token.

    :param list[str] argv: Full argv list excluding program name.
    :param str command: Subcommand token.
    :return list[str]: Subcommand argv.
    """
    try:
        idx = argv.index(command)
    except ValueError:
        return []
    return argv[idx + 1 :]


def _extract_flag_names(argv: list[str]) -> set[str]:
    """Extract provided --flag names from an argv list.

    :param list[str] argv: Raw subcommand argv.
    :return set[str]: Canonicalized flag names.
    """
    provided: set[str] = set()
    for tok in argv:
        if not tok.startswith("--") or tok == "--":
            continue
        name = tok[2:].split("=", 1)[0].replace("-", "_")
        provided.add(name)
    return provided


def _apply_overrides(target: Any, source: argparse.Namespace, provided_flags: set[str]) -> set[str]:
    """Apply provided CLI flags onto an existing dataclass instance.

    :param Any target: Dataclass instance to mutate.
    :param argparse.Namespace source: Parsed CLI namespace.
    :param set[str] provided_flags: Flags explicitly provided by the user.
    :return set[str]: Field names applied from explicit CLI flags.
    """
    applied: set[str] = set()
    for f in fields(type(target)):
        if f.name in provided_flags:
            setattr(target, f.name, getattr(source, f.name))
            applied.add(str(f.name))
    return applied


def _apply_mapping_overrides(target: Any, overrides: dict[str, Any]) -> None:
    """Apply mapping values onto a dataclass object by attribute name.

    :param Any target: Dataclass instance to mutate.
    :param dict[str, Any] overrides: Attribute/value mapping.
    :raises ValueError: If an override key does not exist on ``target``.
    """
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise ValueError(f"Preset override key {key!r} is not a valid field for {type(target).__name__}.")
        setattr(target, key, value)


def _apply_train_preset(
    *,
    preset_name: str,
    cfg_path: Path | None,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
) -> str:
    """Apply a named train preset and return a human-readable mode label.

    :param str preset_name: Preset name from CLI.
    :param Path | None cfg_path: Optional config path (None means no config file).
    :param ModelConfig model_cfg: Model config object to mutate.
    :param DataConfig data_cfg: Data config object to mutate.
    :param TrainConfig train_cfg: Train config object to mutate.
    :raises ValueError: If preset name is unknown.
    :return str: Mode label for CLI logging.
    """
    name = str(preset_name).strip().lower()
    preset = _TRAIN_PRESETS.get(name)
    if preset is None:
        allowed = ", ".join(sorted(_TRAIN_PRESETS.keys()))
        raise ValueError(f"Unknown train preset: {preset_name!r}. Available presets: {allowed}.")

    _apply_mapping_overrides(model_cfg, dict(preset.get("model", {})))
    if cfg_path is None:
        _apply_mapping_overrides(data_cfg, dict(preset.get("data", {})))
        _apply_mapping_overrides(train_cfg, dict(preset.get("train", {})))
        return "model+data+train defaults (no config file)"

    return "model-only overrides (config file provided)"


def _build_train_configs_from_namespace(
    ns: argparse.Namespace,
) -> tuple[ModelConfig, DataConfig, TrainConfig]:
    """Construct model/data/train configs from a parsed namespace.

    :param argparse.Namespace ns: Parsed train args.
    :return tuple[ModelConfig, DataConfig, TrainConfig]: Config dataclasses.
    """
    model_kwargs = {f.name: getattr(ns, f.name) for f in fields(ModelConfig)}
    data_kwargs = {f.name: getattr(ns, f.name) for f in fields(DataConfig)}
    train_kwargs = {f.name: getattr(ns, f.name) for f in fields(TrainConfig)}
    return ModelConfig(**model_kwargs), DataConfig(**data_kwargs), TrainConfig(**train_kwargs)


def _mark_explicit_fields(cfg_obj: Any, explicit_fields: set[str]) -> None:
    """Attach explicit-field metadata to a config dataclass.

    :param Any cfg_obj: Config dataclass object.
    :param set[str] explicit_fields: Field names explicitly provided by file/preset/CLI.
    """
    cfg_obj._explicit_fields = set(str(x) for x in explicit_fields)


def _collect_user_config_mutations(
    *,
    section: str,
    original_values: dict[str, Any],
    cfg_obj: Any,
    reason_overrides: dict[str, str],
) -> list[tuple[str, str, Any, Any, str]]:
    """Collect mutations between original config-file values and runtime values.

    :param str section: Section label (`model`/`data`/`train`).
    :param dict[str, Any] original_values: Values loaded from the user config file.
    :param Any cfg_obj: Runtime config object after defaults/validation.
    :param dict[str, str] reason_overrides: Optional reason mapping for known override sources.
    :return list[tuple[str, str, Any, Any, str]]: Changed entries as `(section, key, old, new, reason)`.
    """
    changes: list[tuple[str, str, Any, Any, str]] = []
    for key, old_value in original_values.items():
        if not hasattr(cfg_obj, key):
            continue
        new_value = getattr(cfg_obj, key)
        if new_value != old_value:
            reason = reason_overrides.get(key, "runtime normalization/defaulting")
            changes.append((str(section), str(key), old_value, new_value, str(reason)))
    return changes


def _emit_user_config_mutation_warnings(
    *, cfg_path: Path | None, changes: list[tuple[str, str, Any, Any, str]]
) -> None:
    """Emit explicit warnings when runtime config differs from user-provided file values.

    :param Path | None cfg_path: Original user config file path.
    :param list[tuple[str, str, Any, Any, str]] changes: Mutation rows to report.
    """
    if cfg_path is None or not changes:
        return
    print(
        f"[deberta][config-warning] Loaded config '{cfg_path}' was modified after load:",
        file=sys.stderr,
    )
    for section, key, old_value, new_value, reason in changes:
        print(
            f"[deberta][config-warning] {section}.{key}: {old_value!r} -> {new_value!r} ({reason})",
            file=sys.stderr,
        )


def _run_train(ns: argparse.Namespace, *, raw_train_argv: list[str]) -> None:
    """Run train flow from parsed namespace.

    :param argparse.Namespace ns: Parsed train args.
    :param list[str] raw_train_argv: Raw argv after `train`.
    """
    cfg_path: Path | None = None
    provided_flags = _extract_flag_names(raw_train_argv)
    explicit_model_fields: set[str] = set()
    explicit_data_fields: set[str] = set()
    explicit_train_fields: set[str] = set()
    user_model_values: dict[str, Any] = {}
    user_data_values: dict[str, Any] = {}
    user_train_values: dict[str, Any] = {}
    model_change_reasons: dict[str, str] = {}
    data_change_reasons: dict[str, str] = {}
    train_change_reasons: dict[str, str] = {}
    if ns.config is not None:
        cfg_path = Path(ns.config).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(str(cfg_path))

        suffix = cfg_path.suffix.lower()
        if suffix not in {".json", ".yaml", ".yml"}:
            raise ValueError("Config file must end with .json, .yaml, or .yml")

        if suffix == ".json":
            model_dict, data_dict, train_dict = _load_json_sections(cfg_path)
        else:
            model_dict, data_dict, train_dict = _load_yaml_sections(cfg_path)
        model_cfg = ModelConfig(**model_dict)
        data_cfg = DataConfig(**data_dict)
        train_cfg = TrainConfig(**train_dict)
        user_model_values = dict(model_dict)
        user_data_values = dict(data_dict)
        user_train_values = dict(train_dict)
        explicit_model_fields.update(model_dict.keys())
        explicit_data_fields.update(data_dict.keys())
        explicit_train_fields.update(train_dict.keys())
    else:
        if ns.preset:
            model_cfg, data_cfg, train_cfg = ModelConfig(), DataConfig(), TrainConfig()
        else:
            model_cfg, data_cfg, train_cfg = _build_train_configs_from_namespace(ns)

    if ns.preset:
        preset_name = str(ns.preset).strip().lower()
        preset_payload = _TRAIN_PRESETS.get(preset_name, {})
        preset_model_keys = set(dict(preset_payload.get("model", {})).keys())
        preset_data_keys = set(dict(preset_payload.get("data", {})).keys())
        preset_train_keys = set(dict(preset_payload.get("train", {})).keys())
        preset_mode = _apply_train_preset(
            preset_name=preset_name,
            cfg_path=cfg_path,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
        )
        explicit_model_fields.update(preset_model_keys)
        if cfg_path is None:
            explicit_data_fields.update(preset_data_keys)
            explicit_train_fields.update(preset_train_keys)
        for key in preset_model_keys:
            model_change_reasons[str(key)] = f"preset override (--preset {preset_name})"
        if cfg_path is None:
            for key in preset_data_keys:
                data_change_reasons[str(key)] = f"preset override (--preset {preset_name})"
            for key in preset_train_keys:
                train_change_reasons[str(key)] = f"preset override (--preset {preset_name})"
        if cfg_path is None:
            print(
                f"Applying train preset '{ns.preset}': {preset_mode}. "
                "Explicit CLI flags override preset values."
            )
        else:
            print(
                f"Applying train preset '{ns.preset}': {preset_mode}. "
                "Data/train values remain from config unless explicitly overridden via CLI."
            )

    if ns.config is not None or ns.preset:
        model_cli_overrides = _apply_overrides(model_cfg, ns, provided_flags)
        data_cli_overrides = _apply_overrides(data_cfg, ns, provided_flags)
        train_cli_overrides = _apply_overrides(train_cfg, ns, provided_flags)
        explicit_model_fields.update(model_cli_overrides)
        explicit_data_fields.update(data_cli_overrides)
        explicit_train_fields.update(train_cli_overrides)
        for key in model_cli_overrides:
            model_change_reasons[str(key)] = f"CLI override (--{str(key).replace('_', '-')})"
        for key in data_cli_overrides:
            data_change_reasons[str(key)] = f"CLI override (--{str(key).replace('_', '-')})"
        for key in train_cli_overrides:
            train_change_reasons[str(key)] = f"CLI override (--{str(key).replace('_', '-')})"

    model_flag_fields = {f.name for f in fields(ModelConfig)}
    data_flag_fields = {f.name for f in fields(DataConfig)}
    train_flag_fields = {f.name for f in fields(TrainConfig)}
    for key in provided_flags:
        if key in model_flag_fields:
            explicit_model_fields.add(str(key))
            model_change_reasons.setdefault(str(key), f"CLI override (--{str(key).replace('_', '-')})")
        if key in data_flag_fields:
            explicit_data_fields.add(str(key))
            data_change_reasons.setdefault(str(key), f"CLI override (--{str(key).replace('_', '-')})")
        if key in train_flag_fields:
            explicit_train_fields.add(str(key))
            train_change_reasons.setdefault(str(key), f"CLI override (--{str(key).replace('_', '-')})")

    _mark_explicit_fields(model_cfg, explicit_model_fields)
    _mark_explicit_fields(train_cfg, explicit_train_fields)

    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg)

    # Validate after config load + CLI overrides so failures are immediate and explicit.
    validate_model_config(model_cfg)
    validate_data_config(data_cfg)
    validate_train_config(train_cfg)
    validate_training_workflow_options(data_cfg=data_cfg, train_cfg=train_cfg, model_cfg=model_cfg)

    file_value_changes: list[tuple[str, str, Any, Any, str]] = []
    if cfg_path is not None:
        file_value_changes.extend(
            _collect_user_config_mutations(
                section="model",
                original_values=user_model_values,
                cfg_obj=model_cfg,
                reason_overrides=model_change_reasons,
            )
        )
        file_value_changes.extend(
            _collect_user_config_mutations(
                section="data",
                original_values=user_data_values,
                cfg_obj=data_cfg,
                reason_overrides=data_change_reasons,
            )
        )
        file_value_changes.extend(
            _collect_user_config_mutations(
                section="train",
                original_values=user_train_values,
                cfg_obj=train_cfg,
                reason_overrides=train_change_reasons,
            )
        )
    _emit_user_config_mutation_warnings(cfg_path=cfg_path, changes=file_value_changes)

    if bool(getattr(ns, "dry_run", False)):
        report = run_pretraining_dry_run(
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            config_path=cfg_path,
        )
        summary = (
            "Dry-run preflight OK: "
            f"output_dir={report['output_dir']}, "
            f"resume_checkpoint={report['resume_checkpoint']}, "
            f"effective_compile_scope={report['effective_compile_scope']}, "
            f"sample_batch_shape={report['sample_batch_shape']}, "
            f"sample_active_tokens={report['sample_active_tokens']}, "
            f"tokenizer_vocab_size={report['tokenizer_vocab_size']}."
        )
        print(summary)
        return

    run_pretraining(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        config_path=cfg_path,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the `deberta` command.

    :param list[str] | None argv: Optional CLI argv (excluding program name).
    """
    explicit_argv = argv is not None
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_main_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        _run_train(args, raw_train_argv=_subcommand_argv(argv, "train"))
        if _should_fast_exit_after_train(explicit_argv=explicit_argv):
            os._exit(0)
        return

    if args.command == "export":
        export_cfg = namespace_to_export_config(args)
        run_export(export_cfg)
        return


if __name__ == "__main__":
    main()
