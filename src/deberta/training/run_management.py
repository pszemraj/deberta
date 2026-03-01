"""Run-directory and checkpoint lifecycle helpers for training."""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from deberta.io_utils import dump_json

logger = logging.getLogger(__name__)
_RUN_LABEL_CLEAN_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_run_label(raw: str) -> str:
    """Normalize a free-form run label into a compact safe token.

    :param str raw: Raw label.
    :return str: Sanitized label.
    """
    cleaned = _RUN_LABEL_CLEAN_RE.sub("-", str(raw).strip()).strip("._-")
    return cleaned or "run"


def _resolve_output_dir(
    *,
    output_dir: str | None,
    project_name: str,
    config_path: str | Path | None,
    run_name: str | None = None,
) -> Path:
    """Resolve the concrete training output directory.

    :param str | None output_dir: User-configured output directory.
    :param str project_name: Tracker project namespace.
    :param str | Path | None config_path: Optional config file path for naming hint.
    :param str | None run_name: Optional explicit run-name hint for auto naming.
    :return Path: Concrete output directory path.
    """
    if output_dir is not None and str(output_dir).strip():
        return Path(str(output_dir))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_hint = str(run_name).strip() if run_name is not None else ""
    if not name_hint and config_path is not None:
        name_hint = Path(config_path).stem or "run"
    if not name_hint:
        name_hint = "run"
    run_name = _sanitize_run_label(name_hint)
    project = _sanitize_run_label(project_name)
    return Path("runs") / project / f"{stamp}_{run_name}"


def _resolve_output_dir_for_accelerator(
    *,
    accelerator: Any,
    output_dir: str | None,
    project_name: str,
    config_path: str | Path | None,
    run_name: str | None = None,
    broadcast_fn: Any | None = None,
) -> Path:
    """Resolve output_dir with deterministic auto-naming across distributed ranks.

    :param Any accelerator: Accelerator-like runtime exposing ``is_main_process`` and ``num_processes``.
    :param str | None output_dir: User-configured output directory.
    :param str project_name: Tracker project namespace.
    :param str | Path | None config_path: Optional config path used for auto-naming.
    :param str | None run_name: Optional explicit run-name hint for auto naming.
    :param Any | None broadcast_fn: Optional callable compatible with
        ``accelerate.utils.broadcast_object_list``.
    :raises RuntimeError: If distributed broadcast fails or returns an empty path.
    :return Path: Concrete output directory path.
    """
    explicit = output_dir is not None and str(output_dir).strip()
    if explicit:
        return Path(str(output_dir))

    num_processes = int(getattr(accelerator, "num_processes", 1))
    if num_processes <= 1:
        return _resolve_output_dir(
            output_dir=None,
            project_name=project_name,
            config_path=config_path,
            run_name=run_name,
        )

    if broadcast_fn is None:
        try:
            from accelerate.utils import broadcast_object_list as broadcast_fn  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Distributed auto output_dir resolution requires accelerate.utils.broadcast_object_list."
            ) from exc

    shared: list[str | None] = [None]
    if bool(getattr(accelerator, "is_main_process", False)):
        shared[0] = str(
            _resolve_output_dir(
                output_dir=None,
                project_name=project_name,
                config_path=config_path,
                run_name=run_name,
            )
        )

    try:
        broadcast_fn(shared, from_process=0)
    except Exception as exc:
        raise RuntimeError("Failed to broadcast auto-resolved output_dir across ranks.") from exc

    resolved = shared[0]
    if resolved is None or not str(resolved).strip():
        raise RuntimeError("Broadcasted output_dir is empty in distributed auto-output-dir resolution.")
    return Path(str(resolved))


def _parse_checkpoint_step(path: str) -> int:
    """Parse integer step suffix from checkpoint path.

    :param str path: Checkpoint directory path.
    :return int: Parsed step number, or 0 when missing.
    """
    name = Path(path).name
    m = re.search(r"checkpoint-(\d+)", name)
    if m:
        return int(m.group(1))
    return 0


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return latest ``checkpoint-*`` directory under ``output_dir``.

    :param Path output_dir: Training output directory.
    :return Path | None: Latest checkpoint path, or ``None`` if absent.
    """
    checkpoints: list[tuple[int, Path]] = []
    for p in output_dir.glob("checkpoint-*"):
        if p.is_dir():
            step = _parse_checkpoint_step(str(p))
            checkpoints.append((step, p))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def _resolve_resume_checkpoint(
    *,
    output_dir: Path,
    resume_from_checkpoint: str | None,
    is_main_process: bool,
) -> str | None:
    """Resolve resume checkpoint path, including ``auto`` lookup.

    :param Path output_dir: Training output directory.
    :param str | None resume_from_checkpoint: User resume setting.
    :param bool is_main_process: Whether to emit logs.
    :raises ValueError: If ``resume_from_checkpoint='auto'`` is requested for a non-empty output
        directory that does not contain any ``checkpoint-*`` folders.
    :raises FileNotFoundError: If an explicit resume checkpoint path does not exist.
    :return str | None: Concrete checkpoint path, or ``None``.
    """
    resume_value = str(resume_from_checkpoint).strip() if resume_from_checkpoint is not None else ""
    if not resume_value:
        return None

    if resume_value.lower() != "auto":
        checkpoint_path = Path(resume_value).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "train.resume_from_checkpoint was provided but the checkpoint path does not exist: "
                f"{checkpoint_path}"
            )
        if not checkpoint_path.is_dir():
            raise ValueError(
                "train.resume_from_checkpoint must point to a checkpoint directory. "
                f"Got a non-directory path: {checkpoint_path}"
            )
        return str(checkpoint_path.resolve())

    latest = _find_latest_checkpoint(output_dir)
    if latest is None:
        has_existing_contents = output_dir.exists() and any(output_dir.iterdir())
        if has_existing_contents:
            raise ValueError(
                "resume_from_checkpoint=auto was requested but no checkpoint-* directories were found in "
                f"non-empty output_dir={output_dir}. Clean the directory, enable "
                "train.overwrite_output_dir=true, or provide an explicit checkpoint path."
            )
        if is_main_process:
            logger.info("resume_from_checkpoint=auto but no checkpoint-* dirs found; starting from scratch.")
        return None
    return str(latest)


def _load_checkpoint_data_progress(checkpoint_dir: Path) -> tuple[int | None, float]:
    """Load persisted data progress and LR multiplier for resume alignment.

    :param Path checkpoint_dir: Checkpoint directory.
    :return tuple[int | None, float]: (consumed micro-batch count or None, lr_mult).
    """
    path = checkpoint_dir / "data_state.json"
    if not path.exists():
        return None, 1.0
    try:
        raw = json.loads(path.read_text())
        val = raw.get("consumed_micro_batches", None)
        consumed = max(0, int(val)) if val is not None else None
        lr_mult = float(raw.get("lr_mult", 1.0))
        return consumed, lr_mult
    except Exception:
        return None, 1.0


def _save_checkpoint_data_progress(
    *, checkpoint_dir: Path, consumed_micro_batches: int, lr_mult: float = 1.0
) -> None:
    """Persist data iterator progress and LR multiplier next to a checkpoint.

    :param Path checkpoint_dir: Checkpoint directory.
    :param int consumed_micro_batches: Number of consumed micro-batches.
    :param float lr_mult: Persistent nonfinite recovery LR multiplier.
    """
    dump_json(
        {
            "consumed_micro_batches": int(max(0, consumed_micro_batches)),
            "lr_mult": float(lr_mult),
        },
        checkpoint_dir / "data_state.json",
    )


def _save_training_checkpoint(
    *,
    accelerator: Any,
    checkpoint_dir: Path,
    output_dir: Path,
    consumed_micro_batches: int,
    save_total_limit: int,
    log_label: str,
    lr_mult: float = 1.0,
) -> None:
    """Save one training checkpoint with collective state-dict write.

    :param Any accelerator: Accelerate runtime object.
    :param Path checkpoint_dir: Destination checkpoint directory.
    :param Path output_dir: Parent output directory for checkpoint rotation.
    :param int consumed_micro_batches: Data progress to persist.
    :param int save_total_limit: Number of checkpoints to retain.
    :param str log_label: Logging label for this save.
    :param float lr_mult: Persistent nonfinite recovery LR multiplier.
    """
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # save_state can be collective under FSDP sharded checkpointing; all ranks must participate.
    accelerator.save_state(str(checkpoint_dir))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        _save_checkpoint_data_progress(
            checkpoint_dir=checkpoint_dir,
            consumed_micro_batches=consumed_micro_batches,
            lr_mult=float(lr_mult),
        )
        _rotate_checkpoints(output_dir, save_total_limit=int(save_total_limit))
        logger.info(f"Saved {log_label} checkpoint: {checkpoint_dir}")


def _prepare_output_dir(
    *,
    output_dir: Path,
    overwrite_output_dir: bool,
    resume_from_checkpoint: str | None,
    is_main_process: bool,
) -> None:
    """Prepare output directory with overwrite/resume semantics.

    :param Path output_dir: Target output path.
    :param bool overwrite_output_dir: Whether to delete existing non-empty path.
    :param str | None resume_from_checkpoint: Resume setting from config.
    :param bool is_main_process: Whether current process owns filesystem writes.
    """
    if not is_main_process:
        return

    if output_dir.exists() and any(output_dir.iterdir()):
        if overwrite_output_dir:
            shutil.rmtree(output_dir)
        elif not resume_from_checkpoint:
            raise ValueError(
                f"Output directory exists and is not empty: {output_dir}. "
                "Set train.overwrite_output_dir=true or set train.resume_from_checkpoint."
            )

    output_dir.mkdir(parents=True, exist_ok=True)


def _rotate_checkpoints(output_dir: Path, *, save_total_limit: int) -> None:
    """Delete oldest checkpoints beyond ``save_total_limit``.

    :param Path output_dir: Directory containing checkpoint-* folders.
    :param int save_total_limit: Max number of checkpoints to retain.
    """
    if save_total_limit <= 0:
        return

    checkpoints = []
    for p in output_dir.glob("checkpoint-*"):
        if p.is_dir():
            step = _parse_checkpoint_step(str(p))
            checkpoints.append((step, p))

    checkpoints.sort(key=lambda x: x[0])
    if len(checkpoints) <= save_total_limit:
        return

    to_delete = checkpoints[: max(0, len(checkpoints) - save_total_limit)]
    for step, p in to_delete:
        try:
            shutil.rmtree(p)
            logger.info(f"Deleted old checkpoint: {p} (step={step})")
        except Exception as e:
            logger.warning(f"Failed to delete checkpoint {p}: {e}")


__all__ = [
    "_find_latest_checkpoint",
    "_load_checkpoint_data_progress",
    "_parse_checkpoint_step",
    "_prepare_output_dir",
    "_resolve_output_dir",
    "_resolve_output_dir_for_accelerator",
    "_resolve_resume_checkpoint",
    "_rotate_checkpoints",
    "_sanitize_run_label",
    "_save_checkpoint_data_progress",
    "_save_training_checkpoint",
]
