"""Run metadata/config snapshot helpers for training and resume."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from deberta.config import (
    RUN_CONFIG_SCHEMA_VERSION,
    DataConfig,
    ModelConfig,
    TrainConfig,
    load_data_config_snapshot,
    load_model_config_snapshot,
    validate_data_config,
    validate_model_config,
    validate_run_metadata_schema,
)
from deberta.io_utils import dump_json, load_json_mapping

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


def _validate_run_metadata(path: Path) -> None:
    """Validate on-disk run metadata schema compatibility.

    :param Path path: Metadata file path.
    :raises ValueError: If metadata is malformed or schema-incompatible.
    """
    raw = load_json_mapping(path)
    validate_run_metadata_schema(raw, source=str(path))


def _dump_yaml_mapping(payload: dict[str, Any], path: Path) -> None:
    """Write a mapping payload to YAML, with JSON fallback if PyYAML is unavailable.

    :param dict[str, Any] payload: Mapping payload.
    :param Path path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore
    except Exception:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        return

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True, default_flow_style=False, allow_unicode=False)


def _resolved_config_payload(
    *, model_cfg: ModelConfig, data_cfg: DataConfig, train_cfg: TrainConfig
) -> dict[str, dict[str, Any]]:
    """Build resolved nested config payload for YAML snapshots.

    :param ModelConfig model_cfg: Resolved model config.
    :param DataConfig data_cfg: Resolved data config.
    :param TrainConfig train_cfg: Resolved train config.
    :return dict[str, dict[str, Any]]: Nested resolved payload.
    """
    return {
        "model": asdict(model_cfg),
        "data": asdict(data_cfg),
        "train": asdict(train_cfg),
    }


def _persist_config_yaml_snapshots(
    *,
    output_dir: Path,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    config_path: str | Path | None,
    is_main_process: bool,
) -> None:
    """Persist original/resolved YAML config snapshots in ``output_dir``.

    :param Path output_dir: Training output directory.
    :param ModelConfig model_cfg: Resolved model config.
    :param DataConfig data_cfg: Resolved data config.
    :param TrainConfig train_cfg: Resolved train config.
    :param str | Path | None config_path: Optional source config file path.
    :param bool is_main_process: Whether current process owns filesystem writes.
    """
    if not bool(is_main_process):
        return

    resolved_payload = _resolved_config_payload(model_cfg=model_cfg, data_cfg=data_cfg, train_cfg=train_cfg)
    resolved_path = output_dir / "config_resolved.yaml"
    _dump_yaml_mapping(resolved_payload, resolved_path)

    original_path = output_dir / "config_original.yaml"
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
    payload = asdict(cfg)
    defaults = ModelConfig()

    if cfg.backbone_type == "rope":
        payload["hf_attention_kernel"] = defaults.hf_attention_kernel
        payload["hf_max_position_embeddings"] = defaults.hf_max_position_embeddings

    if cfg.norm_arch == "post":
        payload["keel_alpha_init"] = defaults.keel_alpha_init
        payload["keel_alpha_learnable"] = defaults.keel_alpha_learnable

    if cfg.ffn_type == "mlp":
        payload["swiglu_adjust_intermediate"] = defaults.swiglu_adjust_intermediate

    return payload


def _persist_or_validate_run_configs(
    *,
    output_dir: Path,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    resume_checkpoint: str | None,
    config_path: str | Path | None = None,
    is_main_process: bool,
    effective_compile_scope: str | None = None,
    compile_scope_reason: str | None = None,
) -> None:
    """Persist new config snapshots or validate existing snapshots on resume.

    :param Path output_dir: Training output directory.
    :param ModelConfig model_cfg: Current model config.
    :param DataConfig data_cfg: Current data config.
    :param TrainConfig train_cfg: Current train config.
    :param str | None resume_checkpoint: Resolved checkpoint path, if resuming.
    :param str | Path | None config_path: Optional original config-file path.
    :param bool is_main_process: Whether this process owns writes.
    :param str | None effective_compile_scope: Resolved compile scope for metadata.
    :param str | None compile_scope_reason: Reason for compile scope selection.
    :raises ValueError: If resume mode detects incompatible model/data config snapshots.
    """
    model_cfg_path = output_dir / "model_config.json"
    data_cfg_path = output_dir / "data_config.json"
    train_cfg_path = output_dir / "train_config.json"
    run_meta_path = output_dir / "run_metadata.json"

    run_meta = _build_run_metadata(
        effective_compile_scope=effective_compile_scope,
        compile_scope_reason=compile_scope_reason,
    )

    has_saved_model_data = model_cfg_path.exists() and data_cfg_path.exists()
    if resume_checkpoint is not None and has_saved_model_data:
        if run_meta_path.exists():
            _validate_run_metadata(run_meta_path)
            # Check for compile scope drift on resume.
            if is_main_process and effective_compile_scope is not None:
                saved_meta = load_json_mapping(run_meta_path)
                saved_scope = saved_meta.get("effective_compile_scope")
                if saved_scope is not None and str(saved_scope) != str(effective_compile_scope):
                    logger.warning(
                        "Effective compile scope changed on resume: "
                        f"was {saved_scope!r}, now {effective_compile_scope!r}. "
                        "This may affect compiled graph caching but is recoverable."
                    )
        elif is_main_process:
            # Backfill schema metadata for older runs once compatibility has been checked.
            dump_json(run_meta, run_meta_path)

        # Pre-stable policy: do not silently coerce legacy snapshot keys during resume.
        # Snapshots must match current dataclass schemas for correctness/simplicity.
        saved_model_cfg = load_model_config_snapshot(
            load_json_mapping(model_cfg_path), source=str(model_cfg_path)
        )
        saved_data_cfg = load_data_config_snapshot(
            load_json_mapping(data_cfg_path), source=str(data_cfg_path)
        )
        validate_model_config(saved_model_cfg)
        validate_data_config(saved_data_cfg)

        # Match on effective model semantics; inert fields are normalized by mode.
        if _effective_model_config_for_resume_compare(
            saved_model_cfg
        ) != _effective_model_config_for_resume_compare(model_cfg):
            raise ValueError(
                "Resume configuration mismatch for model_config.json. "
                "Refusing to overwrite run metadata with incompatible model settings."
            )
        if asdict(saved_data_cfg) != asdict(data_cfg):
            raise ValueError(
                "Resume configuration mismatch for data_config.json. "
                "Refusing to overwrite run metadata with incompatible data settings."
            )
        if is_main_process:
            logger.info("Resume mode: preserving existing model/data/train config snapshots in output_dir.")
        _persist_config_yaml_snapshots(
            output_dir=output_dir,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            config_path=config_path,
            is_main_process=is_main_process,
        )
        return

    if is_main_process:
        dump_json(asdict(model_cfg), model_cfg_path)
        dump_json(asdict(data_cfg), data_cfg_path)
        dump_json(asdict(train_cfg), train_cfg_path)
        dump_json(run_meta, run_meta_path)
    _persist_config_yaml_snapshots(
        output_dir=output_dir,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
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
    "_validate_run_metadata",
]
