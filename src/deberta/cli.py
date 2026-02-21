"""Unified CLI entrypoint for training and export."""

from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from deberta.config import DataConfig, ModelConfig, TrainConfig
from deberta.export_cli import add_export_arguments, namespace_to_export_config, run_export
from deberta.training import run_pretraining


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

    if any(k in raw for k in ("model", "data", "train")):
        model_dict = raw.get("model", {}) or {}
        data_dict = raw.get("data", {}) or {}
        train_dict = raw.get("train", {}) or {}
        if (
            not isinstance(model_dict, dict)
            or not isinstance(data_dict, dict)
            or not isinstance(train_dict, dict)
        ):
            raise ValueError("YAML config sections model/data/train must be dicts.")
    else:
        model_dict, data_dict, train_dict = _split_flat_dict(raw)

    return ModelConfig(**model_dict), DataConfig(**data_dict), TrainConfig(**train_dict)


def _load_json(path: Path) -> tuple[ModelConfig, DataConfig, TrainConfig]:
    """Load model/data/train dataclasses from a JSON file.

    :param Path path: JSON path.
    :return tuple[ModelConfig, DataConfig, TrainConfig]: Parsed config dataclasses.
    """
    raw = json.loads(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("JSON config must parse to a dict.")

    # Support either nested {model:..., data:..., train:...} or flat.
    if any(k in raw for k in ("model", "data", "train")):
        model_dict = raw.get("model", {}) or {}
        data_dict = raw.get("data", {}) or {}
        train_dict = raw.get("train", {}) or {}
        if (
            not isinstance(model_dict, dict)
            or not isinstance(data_dict, dict)
            or not isinstance(train_dict, dict)
        ):
            raise ValueError("JSON config sections model/data/train must be dicts.")
        return ModelConfig(**model_dict), DataConfig(**data_dict), TrainConfig(**train_dict)

    # Back-compat: try flat json format by splitting keys.
    model_dict, data_dict, train_dict = _split_flat_dict(raw)
    return ModelConfig(**model_dict), DataConfig(**data_dict), TrainConfig(**train_dict)


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
    for f in fields(cls):
        help_text = str(f.metadata.get("help", "")) if f.metadata else ""
        default = f.default
        field_type = type_hints.get(f.name, f.type)
        group.add_argument(
            f"--{f.name}",
            f"--{f.name.replace('_', '-')}",
            dest=f.name,
            default=default,
            type=_argparse_type(field_type),
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
            "Examples: `deberta train configs/pretrain_rope_c4_en.yaml` or "
            "`deberta train --dataset_name c4 --dataset_config_name en --max_steps 1000`."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Optional YAML/JSON config path. CLI flags override matching config keys.",
    )
    _add_dataclass_flags(train, ModelConfig, group_name="Model")
    _add_dataclass_flags(train, DataConfig, group_name="Data")
    _add_dataclass_flags(train, TrainConfig, group_name="Train")


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


def _apply_overrides(target: Any, source: argparse.Namespace, provided_flags: set[str]) -> None:
    """Apply provided CLI flags onto an existing dataclass instance.

    :param Any target: Dataclass instance to mutate.
    :param argparse.Namespace source: Parsed CLI namespace.
    :param set[str] provided_flags: Flags explicitly provided by the user.
    """
    for f in fields(type(target)):
        if f.name in provided_flags:
            setattr(target, f.name, getattr(source, f.name))


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


def _run_train(ns: argparse.Namespace, *, raw_train_argv: list[str]) -> None:
    """Run train flow from parsed namespace.

    :param argparse.Namespace ns: Parsed train args.
    :param list[str] raw_train_argv: Raw argv after `train`.
    """
    if ns.config is not None:
        cfg_path = Path(ns.config).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(str(cfg_path))

        suffix = cfg_path.suffix.lower()
        if suffix not in {".json", ".yaml", ".yml"}:
            raise ValueError("Config file must end with .json, .yaml, or .yml")

        if suffix == ".json":
            model_cfg, data_cfg, train_cfg = _load_json(cfg_path)
        else:
            model_cfg, data_cfg, train_cfg = _load_yaml(cfg_path)

        provided_flags = _extract_flag_names(raw_train_argv)
        _apply_overrides(model_cfg, ns, provided_flags)
        _apply_overrides(data_cfg, ns, provided_flags)
        _apply_overrides(train_cfg, ns, provided_flags)
    else:
        model_cfg, data_cfg, train_cfg = _build_train_configs_from_namespace(ns)

    run_pretraining(model_cfg=model_cfg, data_cfg=data_cfg, train_cfg=train_cfg)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the `deberta` command.

    :param list[str] | None argv: Optional CLI argv (excluding program name).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_main_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        _run_train(args, raw_train_argv=_subcommand_argv(argv, "train"))
        return

    if args.command == "export":
        export_cfg = namespace_to_export_config(args)
        run_export(export_cfg)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
