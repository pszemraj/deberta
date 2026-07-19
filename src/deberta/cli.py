"""Unified CLI entrypoint for training and export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deberta.config import load_config
from deberta.export_cli import (
    ExportArgumentDefaultsHelpFormatter,
    add_export_arguments,
    namespace_to_export_config,
    run_export,
)
from deberta.training import run_pretraining, run_pretraining_dry_run


def _build_train_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Create `deberta train` parser.

    :param argparse._SubParsersAction[argparse.ArgumentParser] subparsers: Parent subparsers action.
    """
    train = subparsers.add_parser(
        "train",
        help="Run pretraining.",
        description="Run RTD pretraining from a YAML or JSON config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train.add_argument(
        "config",
        help="Required YAML or JSON training config path.",
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
    _build_train_parser(subparsers)
    _build_export_parser(subparsers)
    return parser


def _run_train(ns: argparse.Namespace) -> None:
    """Run train flow from parsed namespace.

    :param argparse.Namespace ns: Parsed train args.
    """
    cfg_path = Path(ns.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(str(cfg_path))
    suffix = cfg_path.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise ValueError("Config file must end with .json, .yaml, or .yml")
    cfg = load_config(cfg_path)

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
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_main_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        _run_train(args)
        return

    if args.command == "export":
        export_cfg = namespace_to_export_config(args)
        run_export(export_cfg)
        return


if __name__ == "__main__":
    main()
