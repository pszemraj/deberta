"""Checkpoint progress helpers for pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deberta.config import TrainConfig
from deberta.training.run_management import _save_training_checkpoint


def _resolve_data_resume_policy(
    *,
    train_cfg: Any,
    consumed_micro_batches: int,
    global_step: int,
) -> tuple[int, bool, str]:
    """Resolve data-iterator resume strategy.

    Returns ``(start_epoch, do_replay, reason)``:

    - ``start_epoch``: Dataset epoch offset used when starting the cyclical dataloader.
    - ``do_replay``: Whether to replay consumed microbatches.
    - ``reason``: Human-readable policy decision.

    :param Any train_cfg: Train config object.
    :param int consumed_micro_batches: Number of consumed microbatches loaded from checkpoint.
    :param int global_step: Resumed global optimizer step.
    :return tuple[int, bool, str]: Resolved policy triple.
    """
    consumed = max(0, int(consumed_micro_batches))
    if consumed <= 0:
        return 0, False, "fresh-start"

    strategy = str(train_cfg.checkpoint.resume_data_strategy or "auto").strip().lower()
    max_replay = max(0, int(train_cfg.checkpoint.resume_replay_max_micro_batches))

    if strategy == "replay":
        return 0, True, "resume_data_strategy=replay"

    if strategy == "restart_epoch":
        return int(max(0, global_step)), False, "resume_data_strategy=restart_epoch"

    if consumed <= max_replay:
        return 0, True, f"resume_data_strategy=auto (replay <= {max_replay})"
    return (
        int(max(0, global_step)),
        False,
        f"resume_data_strategy=auto (restart_epoch; replay > {max_replay})",
    )


def _save_periodic_checkpoint_if_due(
    *,
    accelerator: Any,
    train_cfg: TrainConfig,
    output_dir: Path,
    global_step: int,
    consumed_micro_batches_committed: int,
    lr_mult: float,
    optimizer_param_digest: str | dict[str, str],
    gradient_accumulation_steps: int,
    last_saved_step: int,
) -> int:
    """Persist a periodic checkpoint when ``global_step`` hits ``train.checkpoint.save_steps``.

    :param Any accelerator: Accelerator runtime.
    :param TrainConfig train_cfg: Training config.
    :param Path output_dir: Output directory containing checkpoints.
    :param int global_step: Current global step.
    :param int consumed_micro_batches_committed: Committed micro-batch progress.
    :param float lr_mult: Persistent recovery LR multiplier.
    :param str | dict[str, str] optimizer_param_digest: Trainable-parameter digest payload.
    :param int gradient_accumulation_steps: Active accumulation steps.
    :param int last_saved_step: Last checkpoint step already saved.
    :return int: Updated ``last_saved_step`` value.
    """
    if not train_cfg.checkpoint.save_steps or (global_step % int(train_cfg.checkpoint.save_steps) != 0):
        return int(last_saved_step)

    ckpt_dir = output_dir / f"checkpoint-{int(global_step)}"
    _save_training_checkpoint(
        accelerator=accelerator,
        checkpoint_dir=ckpt_dir,
        output_dir=output_dir,
        consumed_micro_batches=consumed_micro_batches_committed,
        save_total_limit=int(train_cfg.checkpoint.save_total_limit),
        log_label="periodic",
        lr_mult=float(lr_mult),
        optimizer_param_digest=optimizer_param_digest,
        global_step=int(global_step),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
    )
    return int(global_step)
