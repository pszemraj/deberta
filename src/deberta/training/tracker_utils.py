"""Tracker initialization and W&B integration helpers."""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

from deberta.training.run_management import _sanitize_run_label
from deberta.utils.checkpoint_utils import unwrap_compiled_model

logger = logging.getLogger(__name__)


def _wandb_save_with_fallback(save_fn: Any, src: Path) -> None:
    """Call ``wandb.save``-style APIs across signature variants.

    :param Any save_fn: Callable save function.
    :param Path src: Source file path.
    :return None: None.
    """
    try:
        save_fn(str(src), base_path=str(src.parent), policy="now")
    except TypeError:
        try:
            save_fn(str(src), base_path=str(src.parent))
        except TypeError:
            save_fn(str(src))


def _init_trackers(
    *,
    accelerator: Any,
    project_name: str,
    tracker_cfg: dict[str, Any],
    report_to: str,
    run_name: str,
) -> None:
    """Initialize Accelerate trackers with tracker-specific kwargs when available.

    :param Any accelerator: Accelerator instance.
    :param str project_name: Tracker project name.
    :param dict[str, Any] tracker_cfg: Tracker configuration payload.
    :param str report_to: Selected tracker backend.
    :param str run_name: Effective run name.
    """
    call_kwargs: dict[str, Any] = {
        "project_name": project_name,
        "config": tracker_cfg,
    }

    init_kwargs: dict[str, Any] = {}
    if str(report_to).strip().lower() == "wandb":
        init_kwargs = {"wandb": {"name": run_name}}

    if not init_kwargs:
        accelerator.init_trackers(**call_kwargs)
        return

    try:
        accelerator.init_trackers(**call_kwargs, init_kwargs=init_kwargs)
    except TypeError as err:
        # Some accelerate builds do not accept init_kwargs; retry without it.
        if "init_kwargs" not in str(err):
            raise
        logger.warning(
            "Accelerate init_trackers() rejected init_kwargs; W&B run name "
            "cannot be set explicitly on this runtime."
        )
        accelerator.init_trackers(**call_kwargs)


def _setup_wandb_watch(
    *,
    accelerator: Any,
    wandb_run: Any | None,
    model: torch.nn.Module,
    watch_mode: str,
    watch_log_freq: int,
) -> bool:
    """Configure W&B gradient/parameter watching for the active run.

    :param Any accelerator: Accelerator runtime.
    :param Any | None wandb_run: W&B tracker object, if available.
    :param torch.nn.Module model: Training model (possibly wrapped).
    :param str watch_mode: W&B watch mode.
    :param int watch_log_freq: Watch logging frequency.
    :return bool: True when watch setup succeeds, else False.
    """
    mode = str(watch_mode).strip().lower()
    if mode == "none":
        return False
    if not bool(getattr(accelerator, "is_main_process", True)):
        return False

    watch_target = unwrap_compiled_model(accelerator, model)
    freq = max(1, int(watch_log_freq))

    watch_owner = wandb_run
    watch_fn = getattr(wandb_run, "watch", None) if wandb_run is not None else None
    if not callable(watch_fn):
        with suppress(Exception):
            tracker_obj = accelerator.get_tracker("wandb", unwrap=True)
            watch_owner = tracker_obj
            watch_fn = getattr(tracker_obj, "watch", None)

    if not callable(watch_fn):
        logger.warning("W&B tracker does not expose watch(); skipping model watch setup.")
        return False

    try:
        watch_fn(watch_target, log=mode, log_freq=freq)
    except TypeError:
        watch_fn(watch_target, log=mode)

    owner_type = type(watch_owner).__name__ if watch_owner is not None else "unknown"
    logger.info("Enabled W&B watch (mode=%s, log_freq=%d, tracker=%s).", mode, freq, owner_type)
    return True


def _upload_wandb_original_config(
    *,
    accelerator: Any,
    wandb_run: Any | None,
    config_original_path: Path,
    run_name: str,
    config_resolved_path: Path | None = None,
    config_source_path: str | Path | None = None,
) -> bool:
    """Upload config snapshots to W&B for reproducibility.

    :param Any accelerator: Accelerator runtime.
    :param Any | None wandb_run: W&B tracker object, if available.
    :param Path config_original_path: Local config-original snapshot path.
    :param str run_name: Effective run name.
    :param Path | None config_resolved_path: Optional resolved-config snapshot path.
    :param str | Path | None config_source_path: Optional original config filepath passed by the user.
    :return bool: True when at least one config file upload succeeds, else False.
    """
    if not bool(getattr(accelerator, "is_main_process", True)):
        return False

    owner = wandb_run
    if owner is None:
        for unwrap in (True, False):
            with suppress(Exception):
                owner = accelerator.get_tracker("wandb", unwrap=unwrap)
            if owner is not None:
                break
    if owner is None:
        return False

    run_owner = owner
    with suppress(Exception):
        run_owner = owner.tracker
    with suppress(Exception):
        run_owner = owner.run

    upload_files: list[Path] = []
    source_uploaded = False
    if config_resolved_path is not None and config_resolved_path.exists():
        upload_files.append(config_resolved_path)
    elif config_resolved_path is not None:
        logger.warning("W&B config upload skipped missing file: %s", config_resolved_path)

    source_path_for_meta: str | None = None
    if config_source_path is not None:
        source_candidate = Path(str(config_source_path)).expanduser()
        with suppress(Exception):
            source_candidate = source_candidate.resolve()
        source_path_for_meta = str(source_candidate)
        if source_candidate.is_file():
            upload_files.append(source_candidate)
            source_uploaded = True
        else:
            logger.warning("W&B config upload skipped missing source config path: %s", source_candidate)

    # Prefer the actual source config path when available; otherwise upload config_original snapshot.
    if config_original_path.exists():
        if not source_uploaded:
            upload_files.append(config_original_path)
    else:
        logger.warning("W&B config upload skipped missing file: %s", config_original_path)

    if not upload_files:
        return False

    # Keep first-seen order while de-duplicating by resolved path.
    deduped_upload_files: list[Path] = []
    seen_paths: set[str] = set()
    for path in upload_files:
        try:
            key = str(path.expanduser().resolve())
        except Exception:
            key = str(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        deduped_upload_files.append(path)

    safe_run_name = _sanitize_run_label(run_name)
    upload_meta = {
        "source_config_path": source_path_for_meta,
        "config_original_snapshot": str(config_original_path),
        "config_resolved_snapshot": str(config_resolved_path) if config_resolved_path is not None else None,
    }
    save_fallback: Any | None = None
    with suppress(Exception):
        import wandb  # type: ignore

        if wandb.run is not None and callable(getattr(wandb, "save", None)):
            save_fallback = wandb.save

    for save_fn, log_template in (
        (getattr(run_owner, "save", None), "Uploaded config snapshot to W&B as run file: %s"),
        (save_fallback, "Uploaded config snapshot via wandb.save: %s"),
    ):
        if not callable(save_fn):
            continue
        uploaded_any = False
        for src in deduped_upload_files:
            _wandb_save_with_fallback(save_fn, src)
            uploaded_any = True
            logger.info(log_template, src)
        if uploaded_any:
            return True

    log_artifact_fn = getattr(run_owner, "log_artifact", None)
    if callable(log_artifact_fn):
        with suppress(Exception):
            import wandb  # type: ignore

            artifact = wandb.Artifact(name=f"config-{safe_run_name}", type="config", metadata=upload_meta)
            for src in deduped_upload_files:
                artifact.add_file(str(src), name=src.name)
            log_artifact_fn(artifact)
            logger.info("Uploaded config snapshot artifact to W&B for run %s", safe_run_name)
            return True

    logger.warning("W&B tracker does not support save()/log_artifact(); skipping config upload.")
    return False


__all__ = [
    "_init_trackers",
    "_setup_wandb_watch",
    "_upload_wandb_original_config",
]
