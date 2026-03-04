"""Run metadata/config snapshot helpers for training and resume."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from deberta.config import (
    RUN_CONFIG_SCHEMA_VERSION,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    asdict_without_private,
    load_data_config_snapshot,
    load_logging_config_snapshot,
    load_model_config_snapshot,
    load_optim_config_snapshot,
    validate_data_config,
    validate_logging_config,
    validate_model_config,
    validate_optim_config,
)
from deberta.io_utils import dump_json, load_json_mapping
from deberta.run_layout import (
    DATA_CONFIG_FILENAME,
    LOGGING_CONFIG_FILENAME,
    MODEL_CONFIG_FILENAME,
    OPTIM_CONFIG_FILENAME,
    RESUME_SOURCE_FILENAME,
    RUN_METADATA_FILENAME,
    RUN_SNAPSHOT_FILENAMES,
    TRAIN_CONFIG_FILENAME,
    infer_run_dir_from_checkpoint,
    validate_run_metadata_file,
)

logger = logging.getLogger(__name__)


def _build_run_metadata(
    *,
    effective_compile_scope: str | None = None,
    compile_scope_reason: str | None = None,
) -> dict[str, Any]:
    """Build run-metadata payload stored alongside config snapshots.

    :param str | None effective_compile_scope: Resolved compile scope after auto-resolution.
    :param str | None compile_scope_reason: Reason for scope selection when auto-resolved.
    :return dict[str, Any]: Metadata mapping.
    """
    from deberta import __version__

    meta: dict[str, Any] = {
        "config_schema_version": int(RUN_CONFIG_SCHEMA_VERSION),
        "deberta_version": str(__version__),
    }
    if effective_compile_scope is not None:
        meta["effective_compile_scope"] = str(effective_compile_scope)
    if compile_scope_reason is not None:
        meta["compile_scope_reason"] = str(compile_scope_reason)
    return meta


def _dump_yaml_mapping(payload: dict[str, Any], path: Path) -> None:
    """Write a mapping payload to YAML, with JSON fallback if PyYAML is unavailable.

    :param dict[str, Any] payload: Mapping payload.
    :param Path path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore
    except Exception:
        dump_json(payload, path)
        return

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True, default_flow_style=False, allow_unicode=False)


def _resolved_config_payload(
    *,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    optim_cfg: OptimConfig,
    logging_cfg: LoggingConfig,
) -> dict[str, dict[str, Any]]:
    """Build resolved nested config payload for YAML snapshots.

    :param ModelConfig model_cfg: Resolved model config.
    :param DataConfig data_cfg: Resolved data config.
    :param TrainConfig train_cfg: Resolved train config.
    :param OptimConfig optim_cfg: Resolved optim config.
    :param LoggingConfig logging_cfg: Resolved logging config.
    :return dict[str, dict[str, Any]]: Nested resolved payload.
    """
    return {
        "model": asdict_without_private(model_cfg),
        "data": asdict_without_private(data_cfg),
        "train": asdict_without_private(train_cfg),
        "optim": asdict_without_private(optim_cfg),
        "logging": asdict_without_private(logging_cfg),
    }


def _persist_config_yaml_snapshots(
    *,
    logging_output_dir: Path,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    optim_cfg: OptimConfig,
    logging_cfg: LoggingConfig,
    config_path: str | Path | None,
    is_main_process: bool,
) -> None:
    """Persist original/resolved YAML config snapshots in logging output dir.

    :param Path logging_output_dir: Logging output directory.
    :param ModelConfig model_cfg: Resolved model config.
    :param DataConfig data_cfg: Resolved data config.
    :param TrainConfig train_cfg: Resolved train config.
    :param OptimConfig optim_cfg: Resolved optim config.
    :param LoggingConfig logging_cfg: Resolved logging config.
    :param str | Path | None config_path: Optional source config file path.
    :param bool is_main_process: Whether current process owns filesystem writes.
    """
    if not bool(is_main_process):
        return

    resolved_payload = _resolved_config_payload(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        optim_cfg=optim_cfg,
        logging_cfg=logging_cfg,
    )
    resolved_path = logging_output_dir / "config_resolved.yaml"
    _dump_yaml_mapping(resolved_payload, resolved_path)

    original_path = logging_output_dir / "config_original.yaml"
    if config_path is None:
        _dump_yaml_mapping(resolved_payload, original_path)
        return

    source = Path(config_path).expanduser().resolve()
    if source.exists():
        original_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return

    # Backfill with resolved payload when source path is unavailable.
    _dump_yaml_mapping(resolved_payload, original_path)


def _effective_model_config_for_resume_compare(cfg: ModelConfig) -> dict[str, Any]:
    """Build a normalized model config snapshot for resume compatibility checks.

    Fields that are explicitly documented as inert in the active model mode are normalized to
    canonical defaults so no-op config churn does not block resume.

    :param ModelConfig cfg: Model config to canonicalize.
    :return dict[str, Any]: Normalized dict payload suitable for equality checks.
    """
    payload = asdict_without_private(cfg)
    defaults = ModelConfig()

    if cfg.backbone_type == "rope":
        payload["hf"] = asdict_without_private(defaults.hf)

    if cfg.rope.norm_arch == "post":
        payload["rope"]["keel_alpha_init"] = defaults.rope.keel_alpha_init
        payload["rope"]["keel_alpha_learnable"] = defaults.rope.keel_alpha_learnable

    if cfg.rope.ffn_type == "mlp":
        payload["rope"]["swiglu_adjust_intermediate"] = defaults.rope.swiglu_adjust_intermediate

    return payload


def _effective_logging_config_for_resume_compare(cfg: LoggingConfig) -> dict[str, Any]:
    """Build a normalized logging config snapshot for resume compatibility checks.

    ``logging.output_dir`` is run-local plumbing that can legitimately differ when
    resuming into a new output directory, so we intentionally exclude it from strict
    resume-compat validation.

    :param LoggingConfig cfg: Logging config to canonicalize.
    :return dict[str, Any]: Normalized dict payload suitable for equality checks.
    """
    payload = asdict_without_private(cfg)
    payload["output_dir"] = None
    return payload


def _validate_resume_output_snapshot_conflicts(*, source_run_dir: Path, output_dir: Path) -> None:
    """Raise when output_dir contains conflicting copied snapshots for resume provenance.

    :param Path source_run_dir: Source run directory inferred from the resume checkpoint.
    :param Path output_dir: Target output directory for the resumed run.
    :raises ValueError: If any persisted snapshot file exists in both locations with different content.
    """
    for filename in RUN_SNAPSHOT_FILENAMES:
        src = source_run_dir / filename
        if not src.exists():
            continue
        dst = output_dir / filename
        if not dst.exists():
            continue

        src_text = src.read_text(encoding="utf-8")
        dst_text = dst.read_text(encoding="utf-8")
        if dst_text != src_text:
            raise ValueError(
                "Output directory contains conflicting run snapshot while resuming from "
                f"a different source run. Conflicting file: {dst}"
            )


def _persist_or_validate_run_configs(
    *,
    output_dir: Path,
    logging_output_dir: Path | None = None,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    optim_cfg: OptimConfig | None = None,
    logging_cfg: LoggingConfig | None = None,
    resume_checkpoint: str | None,
    resume_run_dir: Path | None = None,
    config_path: str | Path | None = None,
    is_main_process: bool,
    preflight_only: bool = False,
    effective_compile_scope: str | None = None,
    compile_scope_reason: str | None = None,
) -> None:
    """Persist new config snapshots or validate existing snapshots on resume.

    :param Path output_dir: Checkpoint output directory.
    :param Path logging_output_dir: Logging output directory.
    :param ModelConfig model_cfg: Current model config.
    :param DataConfig data_cfg: Current data config.
    :param TrainConfig train_cfg: Current train config.
    :param OptimConfig optim_cfg: Current optim config.
    :param LoggingConfig logging_cfg: Current logging config.
    :param str | None resume_checkpoint: Resolved checkpoint path, if resuming.
    :param Path | None resume_run_dir: Optional explicit source run directory for resume validation.
    :param str | Path | None config_path: Optional original config-file path.
    :param bool is_main_process: Whether this process owns writes.
    :param bool preflight_only: When True, perform full validation without writing/updating files.
    :param str | None effective_compile_scope: Resolved compile scope for metadata.
    :param str | None compile_scope_reason: Reason for compile scope selection.
    :raises ValueError: If resume mode detects incompatible config snapshots.
    """
    resolved_optim_cfg = optim_cfg if optim_cfg is not None else OptimConfig()
    resolved_logging_cfg = logging_cfg if logging_cfg is not None else LoggingConfig()
    resolved_logging_output_dir = logging_output_dir if logging_output_dir is not None else output_dir

    output_dir_abs = output_dir.expanduser().resolve()
    source_run_dir: Path | None = None
    snapshot_dir = output_dir
    if resume_checkpoint is not None:
        source_run_dir = (
            resume_run_dir.expanduser().resolve()
            if resume_run_dir is not None
            else infer_run_dir_from_checkpoint(resume_checkpoint)
        )
        snapshot_dir = source_run_dir

    model_cfg_path = snapshot_dir / MODEL_CONFIG_FILENAME
    data_cfg_path = snapshot_dir / DATA_CONFIG_FILENAME
    optim_cfg_path = snapshot_dir / OPTIM_CONFIG_FILENAME
    logging_cfg_path = snapshot_dir / LOGGING_CONFIG_FILENAME
    run_meta_path = snapshot_dir / RUN_METADATA_FILENAME
    output_model_cfg_path = output_dir / MODEL_CONFIG_FILENAME
    output_data_cfg_path = output_dir / DATA_CONFIG_FILENAME
    output_train_cfg_path = output_dir / TRAIN_CONFIG_FILENAME
    output_optim_cfg_path = output_dir / OPTIM_CONFIG_FILENAME
    output_logging_cfg_path = output_dir / LOGGING_CONFIG_FILENAME
    output_run_meta_path = output_dir / RUN_METADATA_FILENAME

    run_meta = _build_run_metadata(
        effective_compile_scope=effective_compile_scope,
        compile_scope_reason=compile_scope_reason,
    )

    has_saved_required = (
        model_cfg_path.exists()
        and data_cfg_path.exists()
        and optim_cfg_path.exists()
        and logging_cfg_path.exists()
    )
    if resume_checkpoint is not None and not has_saved_required:
        raise ValueError(
            "Resume checkpoint source run directory is missing required config snapshots. "
            "Expected model_config.json, data_config.json, optim_config.json, and logging_config.json under "
            f"{snapshot_dir}."
        )
    if resume_checkpoint is not None and has_saved_required:
        if run_meta_path.exists():
            validate_run_metadata_file(snapshot_dir, required=False)
            if is_main_process and effective_compile_scope is not None:
                saved_meta = load_json_mapping(run_meta_path)
                saved_scope = saved_meta.get("effective_compile_scope")
                if saved_scope is not None and str(saved_scope) != str(effective_compile_scope):
                    logger.warning(
                        "Effective compile scope changed on resume: "
                        f"was {saved_scope!r}, now {effective_compile_scope!r}. "
                        "This may affect compiled graph caching but is recoverable."
                    )

        saved_model_cfg = load_model_config_snapshot(
            load_json_mapping(model_cfg_path), source=str(model_cfg_path)
        )
        saved_data_cfg = load_data_config_snapshot(
            load_json_mapping(data_cfg_path), source=str(data_cfg_path)
        )
        saved_optim_cfg = load_optim_config_snapshot(
            load_json_mapping(optim_cfg_path), source=str(optim_cfg_path)
        )
        saved_logging_cfg = load_logging_config_snapshot(
            load_json_mapping(logging_cfg_path),
            source=str(logging_cfg_path),
        )
        validate_model_config(saved_model_cfg)
        validate_data_config(saved_data_cfg)
        validate_optim_config(saved_optim_cfg)
        validate_logging_config(saved_logging_cfg)

        if _effective_model_config_for_resume_compare(
            saved_model_cfg
        ) != _effective_model_config_for_resume_compare(model_cfg):
            raise ValueError(
                "Resume configuration mismatch for model_config.json. "
                "Refusing to overwrite run metadata with incompatible model settings."
            )
        if asdict_without_private(saved_data_cfg) != asdict_without_private(data_cfg):
            raise ValueError(
                "Resume configuration mismatch for data_config.json. "
                "Refusing to overwrite run metadata with incompatible data settings."
            )
        if asdict_without_private(saved_optim_cfg) != asdict_without_private(resolved_optim_cfg):
            raise ValueError(
                "Resume configuration mismatch for optim_config.json. "
                "Refusing to overwrite run metadata with incompatible optimizer settings."
            )
        if _effective_logging_config_for_resume_compare(
            saved_logging_cfg
        ) != _effective_logging_config_for_resume_compare(resolved_logging_cfg):
            raise ValueError(
                "Resume configuration mismatch for logging_config.json. "
                "Refusing to overwrite run metadata with incompatible logging settings."
            )

        if source_run_dir is not None and source_run_dir != output_dir_abs:
            _validate_resume_output_snapshot_conflicts(
                source_run_dir=source_run_dir,
                output_dir=output_dir,
            )

        if preflight_only:
            return

        if is_main_process and not run_meta_path.exists():
            dump_json(run_meta, run_meta_path)

        if is_main_process and source_run_dir is not None and source_run_dir != output_dir_abs:
            resume_checkpoint_abs = Path(resume_checkpoint).expanduser().resolve()
            for filename in RUN_SNAPSHOT_FILENAMES:
                src = source_run_dir / filename
                if not src.exists():
                    continue
                dst = output_dir / filename
                src_text = src.read_text(encoding="utf-8")
                if dst.exists():
                    dst_text = dst.read_text(encoding="utf-8")
                    if dst_text != src_text:
                        raise RuntimeError(f"Unexpected snapshot conflict after pre-validation: {dst}")
                else:
                    dst.write_text(src_text, encoding="utf-8")

            dump_json(
                {
                    "resume_checkpoint": str(resume_checkpoint_abs),
                    "resume_run_dir": str(source_run_dir),
                },
                output_dir / RESUME_SOURCE_FILENAME,
            )
            logger.info(
                "Resume mode: validated snapshots from source run_dir=%s and synchronized provenance into output_dir.",
                source_run_dir,
            )
        elif is_main_process:
            logger.info("Resume mode: preserving existing config snapshots in output_dir.")
        _persist_config_yaml_snapshots(
            logging_output_dir=resolved_logging_output_dir,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            optim_cfg=resolved_optim_cfg,
            logging_cfg=resolved_logging_cfg,
            config_path=config_path,
            is_main_process=is_main_process,
        )
        return

    if preflight_only:
        return

    if is_main_process:
        dump_json(asdict_without_private(model_cfg), output_model_cfg_path)
        dump_json(asdict_without_private(data_cfg), output_data_cfg_path)
        dump_json(asdict_without_private(train_cfg), output_train_cfg_path)
        dump_json(asdict_without_private(resolved_optim_cfg), output_optim_cfg_path)
        dump_json(asdict_without_private(resolved_logging_cfg), output_logging_cfg_path)
        dump_json(run_meta, output_run_meta_path)
        if resume_checkpoint is not None and source_run_dir is not None and source_run_dir != output_dir_abs:
            dump_json(
                {
                    "resume_checkpoint": str(Path(resume_checkpoint).expanduser().resolve()),
                    "resume_run_dir": str(source_run_dir),
                },
                output_dir / RESUME_SOURCE_FILENAME,
            )
    _persist_config_yaml_snapshots(
        logging_output_dir=resolved_logging_output_dir,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        optim_cfg=resolved_optim_cfg,
        logging_cfg=resolved_logging_cfg,
        config_path=config_path,
        is_main_process=is_main_process,
    )


__all__ = [
    "_build_run_metadata",
    "_dump_yaml_mapping",
    "_effective_model_config_for_resume_compare",
    "_persist_config_yaml_snapshots",
    "_persist_or_validate_run_configs",
    "_resolved_config_payload",
]
