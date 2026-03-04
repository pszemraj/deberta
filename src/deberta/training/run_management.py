"""Run-directory and checkpoint lifecycle helpers for training."""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import uuid
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from deberta.utils.io_utils import dump_json, load_json_mapping

logger = logging.getLogger(__name__)
_RUN_LABEL_CLEAN_RE = re.compile(r"[^A-Za-z0-9._-]+")
_CHECKPOINT_DATA_STATE_FILENAME = "data_state.json"
_CHECKPOINT_COMPLETE_MARKER = ".complete"


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


def _broadcast_rank0_payload(
    *,
    broadcast_fn: Any | None,
    payload: list[Any],
    purpose: str,
    error_message: str,
) -> Any:
    """Broadcast one payload list from rank 0 and return the broadcasted object.

    :param Any | None broadcast_fn: Optional broadcast callable.
    :param list[Any] payload: Mutable single-item payload list.
    :param str purpose: Human-readable operation purpose.
    :param str error_message: RuntimeError message for broadcast failures.
    :raises RuntimeError: If broadcast fails or returns an empty payload list.
    :return Any: Broadcast payload item from rank 0.
    """
    fn = broadcast_fn
    if fn is None:
        try:
            from accelerate.utils import broadcast_object_list as fn  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"{purpose} requires accelerate.utils.broadcast_object_list.") from exc
    try:
        fn(payload, from_process=0)
    except Exception as exc:
        raise RuntimeError(error_message) from exc
    if not payload:
        raise RuntimeError(f"{error_message.rstrip('.')} (empty broadcast payload).")
    return payload[0]


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

    resolved = _broadcast_rank0_payload(
        broadcast_fn=broadcast_fn,
        purpose="Distributed auto output_dir resolution",
        payload=shared,
        error_message="Failed to broadcast auto-resolved output_dir across ranks.",
    )
    if resolved is None or not str(resolved).strip():
        raise RuntimeError("Broadcasted output_dir is empty in distributed auto-output-dir resolution.")
    return Path(str(resolved))


def _resolve_resume_checkpoint_for_accelerator(
    *,
    accelerator: Any,
    output_dir: Path,
    resume_from_checkpoint: str | None,
    broadcast_fn: Any | None = None,
) -> str | None:
    """Resolve resume checkpoint on rank0 and broadcast to all ranks.

    :param Any accelerator: Accelerator-like runtime exposing ``is_main_process`` and ``num_processes``.
    :param Path output_dir: Training output directory.
    :param str | None resume_from_checkpoint: User resume setting.
    :param Any | None broadcast_fn: Optional callable compatible with
        ``accelerate.utils.broadcast_object_list``.
    :raises RuntimeError: If distributed broadcast fails or returns invalid payload.
    :raises ValueError: If rank0 resume resolution fails with a ``ValueError``.
    :raises FileNotFoundError: If rank0 resume resolution fails with a ``FileNotFoundError``.
    :return str | None: Concrete checkpoint path, or ``None``.
    """
    resume_value = str(resume_from_checkpoint).strip() if resume_from_checkpoint is not None else ""
    if not resume_value:
        return None

    num_processes = int(getattr(accelerator, "num_processes", 1))
    is_main = bool(getattr(accelerator, "is_main_process", False))
    if num_processes <= 1:
        return _resolve_resume_checkpoint(
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            is_main_process=is_main,
        )

    shared: list[dict[str, Any]] = [
        {
            "ok": True,
            "value": None,
            "error_type": None,
            "error_message": None,
        }
    ]
    if is_main:
        try:
            shared[0]["value"] = _resolve_resume_checkpoint(
                output_dir=output_dir,
                resume_from_checkpoint=resume_from_checkpoint,
                is_main_process=True,
            )
        except Exception as exc:
            shared[0] = {
                "ok": False,
                "value": None,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

    payload = _broadcast_rank0_payload(
        broadcast_fn=broadcast_fn,
        purpose="Distributed resume resolution",
        payload=shared,
        error_message="Failed to broadcast resolved resume checkpoint across ranks.",
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Broadcasted resume checkpoint payload is malformed.")
    if not bool(payload.get("ok", False)):
        error_type = str(payload.get("error_type") or "RuntimeError")
        error_message = str(payload.get("error_message") or "Unknown resume resolution error on rank0.")
        if error_type == "ValueError":
            raise ValueError(error_message)
        if error_type == "FileNotFoundError":
            raise FileNotFoundError(error_message)
        raise RuntimeError(error_message)

    resolved = payload.get("value")
    if resolved is None:
        return None
    resolved_str = str(resolved).strip()
    if not resolved_str:
        raise RuntimeError("Broadcasted resume checkpoint path is empty.")
    return resolved_str


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
    checkpoints = _list_checkpoints(output_dir)
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def _checkpoint_weights_appear_valid(checkpoint_dir: Path) -> bool:
    """Return whether a checkpoint has non-empty model-weight payloads.

    This is a structural check intended to catch common crash artifacts
    (missing/zero-byte model files), not a full deserialization validation.

    :param Path checkpoint_dir: Candidate checkpoint directory.
    :return bool: ``True`` when model-weight files appear present and non-empty.
    """
    root_patterns = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
        "*model*.safetensors",
        "*model*.bin",
    )
    for pattern in root_patterns:
        for candidate in checkpoint_dir.glob(pattern):
            if not candidate.is_file():
                continue
            with suppress(OSError):
                if int(candidate.stat().st_size) > 0:
                    return True

    # FSDP sharded model-state directories written by accelerate/torch DCP.
    for subdir in checkpoint_dir.glob("pytorch_model_fsdp*"):
        if not subdir.is_dir():
            continue
        for shard in subdir.rglob("*"):
            if not shard.is_file():
                continue
            with suppress(OSError):
                if int(shard.stat().st_size) > 0:
                    return True
    return False


def _checkpoint_complete_marker_path(checkpoint_dir: Path) -> Path:
    """Return the completion-marker path for a checkpoint directory.

    :param Path checkpoint_dir: Checkpoint directory.
    :return Path: ``.complete`` marker path.
    """
    return checkpoint_dir / _CHECKPOINT_COMPLETE_MARKER


def _is_checkpoint_committed(checkpoint_dir: Path) -> bool:
    """Return whether a checkpoint carries an explicit completion marker.

    :param Path checkpoint_dir: Checkpoint directory.
    :return bool: ``True`` when the checkpoint has a ``.complete`` marker.
    """
    return _checkpoint_complete_marker_path(checkpoint_dir).is_file()


def _is_checkpoint_resumable(checkpoint_dir: Path) -> bool:
    """Return whether a checkpoint satisfies strict resumability invariants.

    :param Path checkpoint_dir: Checkpoint directory.
    :return bool: ``True`` when marker, metadata, and weights are all present.
    """
    if not _is_checkpoint_committed(checkpoint_dir):
        return False
    consumed, _, _ = _load_checkpoint_data_progress(checkpoint_dir)
    if consumed is None:
        return False
    if not _checkpoint_weights_appear_valid(checkpoint_dir):
        return False
    return True


def _find_latest_resumable_checkpoint(output_dir: Path) -> Path | None:
    """Return the latest checkpoint directory that can be resumed safely.

    Transactional checkpoints are preferred and identified by a ``.complete``
    marker written only after checkpoint metadata persistence.

    :param Path output_dir: Training output directory.
    :return Path | None: Latest resumable checkpoint path, or ``None`` if absent.
    """
    checkpoints = _list_checkpoints(output_dir)
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    for _, checkpoint_dir in checkpoints:
        resumable = _is_checkpoint_resumable(checkpoint_dir)
        if not resumable:
            consumed, _, _ = _load_checkpoint_data_progress(checkpoint_dir)
            if consumed is not None and not _checkpoint_weights_appear_valid(checkpoint_dir):
                logger.warning(
                    "Checkpoint %s has resume metadata but model weights appear missing/empty; "
                    "skipping as unresumable.",
                    checkpoint_dir,
                )
            continue
        return checkpoint_dir
    return None


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
        consumed, _, _ = _load_checkpoint_data_progress(checkpoint_path)
        weights_ok = _checkpoint_weights_appear_valid(checkpoint_path)
        committed = _is_checkpoint_committed(checkpoint_path)
        if not committed:
            raise ValueError(
                f"Explicit resume checkpoint '{checkpoint_path}' is missing .complete marker. "
                "Only transactionally committed checkpoints are resumable."
            )
        if consumed is None:
            raise ValueError(
                f"Explicit resume checkpoint '{checkpoint_path}' has .complete marker but failed "
                "resume integrity checks (missing/invalid data_state.json with consumed_micro_batches). "
                "The checkpoint may be incomplete due to a crashed save."
            )
        if not weights_ok:
            raise ValueError(
                f"Explicit resume checkpoint '{checkpoint_path}' has .complete marker but model weights "
                "appear missing or empty. The checkpoint may be incomplete due to a crashed save."
            )
        return str(checkpoint_path.resolve())

    latest_any = _find_latest_checkpoint(output_dir)
    if latest_any is None:
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

    latest_resumable = _find_latest_resumable_checkpoint(output_dir)
    if latest_resumable is None:
        raise ValueError(
            "resume_from_checkpoint=auto found checkpoint-* directories but none are resumable "
            "(missing .complete marker and/or missing valid data_state.json/model weights). "
            "Provide an explicit checkpoint path or clean stale/incomplete checkpoints."
        )

    if is_main_process and latest_resumable != latest_any:
        logger.warning(
            "resume_from_checkpoint=auto skipped newest checkpoint %s because resume metadata "
            "is missing/invalid; falling back to %s.",
            latest_any,
            latest_resumable,
        )
    return str(latest_resumable.resolve())


def _load_checkpoint_progress_metadata(
    checkpoint_dir: Path,
) -> tuple[int | None, float, str | dict[str, str] | None, int | None, int | None]:
    """Load persisted resume metadata from ``data_state.json``.

    :param Path checkpoint_dir: Checkpoint directory.
    :return tuple[int | None, float, str | dict[str, str] | None, int | None, int | None]:
        ``(consumed_micro_batches, lr_mult, optimizer_param_digest, global_step, gradient_accumulation_steps)``.
    """
    path = checkpoint_dir / _CHECKPOINT_DATA_STATE_FILENAME
    if not path.exists():
        return None, 1.0, None, None, None
    try:
        raw = load_json_mapping(path)
        val = raw.get("consumed_micro_batches", None)
        consumed = max(0, int(val)) if val is not None else None
        lr_mult = float(raw.get("lr_mult", 1.0))
        digest_raw = raw.get("optimizer_param_digest", None)
        digest: str | dict[str, str] | None
        if isinstance(digest_raw, dict):
            digest = {str(k): str(v) for k, v in digest_raw.items()}
        elif digest_raw is not None:
            digest = str(digest_raw)
        else:
            digest = None
        global_step_raw = raw.get("global_step", None)
        global_step = max(0, int(global_step_raw)) if global_step_raw is not None else None
        ga_steps_raw = raw.get("gradient_accumulation_steps", None)
        ga_steps = max(1, int(ga_steps_raw)) if ga_steps_raw is not None else None
        return consumed, lr_mult, digest, global_step, ga_steps
    except (TypeError, ValueError) as exc:
        logger.warning(
            "Checkpoint %s has invalid data_state.json (%s: %s); treating as unresumable.",
            checkpoint_dir,
            type(exc).__name__,
            exc,
        )
        return None, 1.0, None, None, None
    except Exception as exc:
        logger.warning(
            "Unexpected error reading data_state.json for checkpoint %s (%s); treating as unresumable.",
            checkpoint_dir,
            exc,
        )
        return None, 1.0, None, None, None


def _load_checkpoint_data_progress(
    checkpoint_dir: Path,
) -> tuple[int | None, float, str | dict[str, str] | None]:
    """Load persisted data progress, LR multiplier, and optimizer param digest.

    :param Path checkpoint_dir: Checkpoint directory.
    :return tuple[int | None, float, str | dict[str, str] | None]:
        ``(consumed_micro_batches, lr_mult, optimizer_param_digest)``.
    """
    consumed, lr_mult, digest, _, _ = _load_checkpoint_progress_metadata(checkpoint_dir)
    return consumed, lr_mult, digest


def _save_checkpoint_data_progress(
    *,
    checkpoint_dir: Path,
    consumed_micro_batches: int,
    lr_mult: float = 1.0,
    optimizer_param_digest: str | dict[str, str] | None = None,
    global_step: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> None:
    """Persist data iterator progress, LR multiplier, and optimizer param digest.

    :param Path checkpoint_dir: Checkpoint directory.
    :param int consumed_micro_batches: Number of consumed micro-batches.
    :param float lr_mult: Persistent nonfinite recovery LR multiplier.
    :param str | dict[str, str] | None optimizer_param_digest: SHA-256 prefix digest(s) of trainable param names.
    :param int | None global_step: Committed optimizer step at checkpoint save time.
    :param int | None gradient_accumulation_steps: Gradient accumulation steps at save time.
    """
    payload: dict[str, Any] = {
        "consumed_micro_batches": int(max(0, consumed_micro_batches)),
        "lr_mult": float(lr_mult),
    }
    if optimizer_param_digest is not None:
        if isinstance(optimizer_param_digest, dict):
            payload["optimizer_param_digest"] = {str(k): str(v) for k, v in optimizer_param_digest.items()}
        else:
            payload["optimizer_param_digest"] = str(optimizer_param_digest)
    if global_step is not None:
        payload["global_step"] = int(max(0, global_step))
    if gradient_accumulation_steps is not None:
        payload["gradient_accumulation_steps"] = int(max(1, gradient_accumulation_steps))
    dump_json(payload, checkpoint_dir / _CHECKPOINT_DATA_STATE_FILENAME)


def _save_training_checkpoint(
    *,
    accelerator: Any,
    checkpoint_dir: Path,
    output_dir: Path,
    consumed_micro_batches: int,
    save_total_limit: int,
    log_label: str,
    lr_mult: float = 1.0,
    optimizer_param_digest: str | dict[str, str] | None = None,
    global_step: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> None:
    """Save one training checkpoint using a transactional temp-dir commit.

    :param Any accelerator: Accelerate runtime object.
    :param Path checkpoint_dir: Destination checkpoint directory.
    :param Path output_dir: Parent output directory for checkpoint rotation.
    :param int consumed_micro_batches: Data progress to persist.
    :param int save_total_limit: Number of checkpoints to retain.
    :param str log_label: Logging label for this save.
    :param float lr_mult: Persistent nonfinite recovery LR multiplier.
    :param str | dict[str, str] | None optimizer_param_digest: SHA-256 prefix digest(s) of trainable param names.
    :param int | None global_step: Committed optimizer step at save time.
    :param int | None gradient_accumulation_steps: Gradient accumulation steps at save time.
    """
    staging_dir = checkpoint_dir.parent / f".{checkpoint_dir.name}.tmp-{uuid.uuid4().hex}"

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if checkpoint_dir.exists():
            if any(checkpoint_dir.iterdir()):
                raise RuntimeError(
                    f"Refusing to overwrite non-empty checkpoint directory: {checkpoint_dir}. "
                    "Checkpoint directories are immutable; use a new step directory."
                )
            checkpoint_dir.rmdir()
        staging_dir.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()

    # save_state can be collective under FSDP sharded checkpointing; all ranks must participate.
    accelerator.save_state(str(staging_dir))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        try:
            _save_checkpoint_data_progress(
                checkpoint_dir=staging_dir,
                consumed_micro_batches=consumed_micro_batches,
                lr_mult=float(lr_mult),
                optimizer_param_digest=optimizer_param_digest,
                global_step=global_step,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            progress_ok = _load_checkpoint_data_progress(staging_dir)[0] is not None
            weights_ok = _checkpoint_weights_appear_valid(staging_dir)
            if not (progress_ok and weights_ok):
                raise RuntimeError(
                    "Post-save structural validation failed for staged checkpoint "
                    f"{staging_dir} (data_state_ok={bool(progress_ok)}, weights_ok={bool(weights_ok)})."
                )

            marker_path = _checkpoint_complete_marker_path(staging_dir)
            fd, tmp_name = tempfile.mkstemp(
                dir=staging_dir,
                prefix=f".{marker_path.name}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write("ok\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_name, marker_path)
            except BaseException:
                with suppress(OSError):
                    os.unlink(tmp_name)
                raise

            staging_dir.replace(checkpoint_dir)
            _rotate_checkpoints(output_dir, save_total_limit=int(save_total_limit))
            logger.info(f"Saved {log_label} checkpoint: {checkpoint_dir}")
        except Exception:
            with suppress(Exception):
                if staging_dir.exists():
                    shutil.rmtree(staging_dir, ignore_errors=True)
            raise


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

    resume_value = str(resume_from_checkpoint).strip() if resume_from_checkpoint is not None else ""
    if output_dir.exists() and any(output_dir.iterdir()):
        if overwrite_output_dir:
            shutil.rmtree(output_dir)
        elif not resume_value:
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

    checkpoints: list[tuple[int, Path]] = []
    for step, checkpoint_dir in _list_checkpoints(output_dir):
        if _is_checkpoint_committed(checkpoint_dir):
            checkpoints.append((step, checkpoint_dir))
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


def _list_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    """Collect ``checkpoint-*`` directories with parsed step ids.

    :param Path output_dir: Directory to scan.
    :return list[tuple[int, Path]]: ``(step, checkpoint_dir)`` pairs.
    """
    checkpoints: list[tuple[int, Path]] = []
    for p in output_dir.glob("checkpoint-*"):
        if p.is_dir():
            checkpoints.append((_parse_checkpoint_step(str(p)), p))
    return checkpoints


__all__ = [
    "_find_latest_checkpoint",
    "_load_checkpoint_data_progress",
    "_load_checkpoint_progress_metadata",
    "_parse_checkpoint_step",
    "_prepare_output_dir",
    "_resolve_output_dir",
    "_resolve_output_dir_for_accelerator",
    "_resolve_resume_checkpoint",
    "_resolve_resume_checkpoint_for_accelerator",
    "_rotate_checkpoints",
    "_sanitize_run_label",
    "_save_checkpoint_data_progress",
    "_save_training_checkpoint",
]
