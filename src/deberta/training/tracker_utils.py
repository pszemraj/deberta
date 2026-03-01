"""Tracker initialization and W&B integration helpers."""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from deberta.checkpoint_utils import unwrap_compiled_model
from deberta.training.run_management import _sanitize_run_label

logger = logging.getLogger(__name__)


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
) -> bool:
    """Upload ``config_original.yaml`` to W&B using a run-name-specific filename.

    :param Any accelerator: Accelerator runtime.
    :param Any | None wandb_run: W&B tracker object, if available.
    :param Path config_original_path: Local config-original snapshot path.
    :param str run_name: Effective run name.
    :return bool: True when upload succeeds, else False.
    """
    if not bool(getattr(accelerator, "is_main_process", True)):
        return False
    if not config_original_path.exists():
        return False

    owner = wandb_run
    if owner is None:
        with suppress(Exception):
            owner = accelerator.get_tracker("wandb", unwrap=True)
    if owner is None:
        return False

    safe_run_name = _sanitize_run_label(run_name)
    upload_name = f"config_deberta_{safe_run_name}"
    with TemporaryDirectory(prefix="deberta-wandb-config-") as tmp_dir:
        staged = Path(tmp_dir) / upload_name
        staged.write_text(config_original_path.read_text(encoding="utf-8"), encoding="utf-8")

        save_fn = getattr(owner, "save", None)
        if callable(save_fn):
            try:
                save_fn(str(staged), base_path=tmp_dir, policy="now")
            except TypeError:
                try:
                    save_fn(str(staged), base_path=tmp_dir)
                except TypeError:
                    save_fn(str(staged))
            logger.info("Uploaded original config to W&B as %s", upload_name)
            return True

        log_artifact_fn = getattr(owner, "log_artifact", None)
        if callable(log_artifact_fn):
            with suppress(Exception):
                import wandb  # type: ignore

                artifact = wandb.Artifact(name=f"config-{safe_run_name}", type="config")
                artifact.add_file(str(staged), name=upload_name)
                log_artifact_fn(artifact)
                logger.info("Uploaded original config artifact to W&B as %s", upload_name)
                return True

    logger.warning("W&B tracker does not support save()/log_artifact(); skipping config upload.")
    return False


__all__ = [
    "_init_trackers",
    "_setup_wandb_watch",
    "_upload_wandb_original_config",
]
