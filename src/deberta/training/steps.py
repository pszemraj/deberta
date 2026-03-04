"""Per-step optimization helpers for pretraining loops."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from contextlib import suppress
from typing import Any

import torch

from deberta.training.loop_utils import (
    _count_input_tokens_for_batch,
    _count_rtd_tokens_for_batch,
    _resolve_window_token_denominators,
)
from deberta.utils.checkpoint import unwrap_compiled_model

logger = logging.getLogger(__name__)
_NONFINITE_LR_BACKOFF = 0.5
_NONFINITE_LR_MULT_FLOOR = 0.01  # persistent multiplier floor (1% of scheduled)
_NONFINITE_LR_MULT_RECOVERY = 1.1  # gradual recovery factor per successful step
_NONFINITE_OPT_STATE_RESET_EVERY = 4


def _global_grad_l2_norm(model: torch.nn.Module) -> float:
    """Compute global gradient L2 norm over model parameters.

    Accumulates squared norms on-device and issues a single ``.item()`` sync
    instead of one per parameter.

    :param torch.nn.Module model: Model whose gradients are inspected.
    :return float: Global gradient norm (can be non-finite).
    """
    sq_norms: list[torch.Tensor] = []
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        g = grad.detach()
        if g.is_sparse:
            g = g.coalesce().values()
        sq_norms.append(g.float().pow(2).sum())
    if not sq_norms:
        return 0.0
    total = torch.stack(sq_norms).sum()
    return float(total.sqrt().item())


def _has_nonfinite_grad_norm_any_rank(*, accelerator: Any, grad_norm: float) -> bool:
    """Return whether any rank observed a non-finite gradient norm.

    :param Any accelerator: Accelerator-like runtime object.
    :param float grad_norm: Local gradient L2 norm.
    :return bool: True when at least one rank reports non-finite norm.
    """
    local_flag = 0 if math.isfinite(float(grad_norm)) else 1
    device = getattr(accelerator, "device", torch.device("cpu"))
    local = torch.tensor([local_flag], device=device, dtype=torch.int32)
    if int(getattr(accelerator, "num_processes", 1)) <= 1:
        return local_flag > 0
    reduced = accelerator.reduce(local, reduction="sum")
    count = int(reduced.reshape(-1)[0].item())
    return count > 0


def _any_rank_flag_true(*, accelerator: Any, flag: bool) -> bool:
    """Return whether any rank set ``flag=True``.

    :param Any accelerator: Accelerator-like runtime object.
    :param bool flag: Local boolean flag.
    :return bool: True when at least one rank set the flag.
    """
    device = getattr(accelerator, "device", torch.device("cpu"))
    local = torch.tensor([1 if bool(flag) else 0], device=device, dtype=torch.int32)
    if int(getattr(accelerator, "num_processes", 1)) <= 1:
        return bool(flag)
    reduced = accelerator.reduce(local, reduction="sum")
    count = int(reduced.reshape(-1)[0].item())
    return count > 0


def _record_unscaled_lrs(optimizer: torch.optim.Optimizer, scheduler: Any | None) -> None:
    """Record unscaled scheduler LRs into optimizer param groups.

    This stores the scheduler-computed LR (before non-finite recovery scaling)
    in ``group["_lr_unscaled"]`` so recovery scaling can be applied absolutely.

    :param torch.optim.Optimizer optimizer: Runtime optimizer.
    :param Any | None scheduler: Scheduler-like object.
    """
    scheduler_lrs: list[float] | None = None
    if scheduler is not None and hasattr(scheduler, "get_last_lr"):
        with suppress(Exception):
            raw = scheduler.get_last_lr()
            if isinstance(raw, (list, tuple)) and len(raw) == len(optimizer.param_groups):
                scheduler_lrs = [float(x) for x in raw]
    if scheduler_lrs is None:
        scheduler_lrs = [float(group.get("_lr_unscaled", group["lr"])) for group in optimizer.param_groups]

    for group, lr in zip(optimizer.param_groups, scheduler_lrs, strict=True):
        group["_lr_unscaled"] = float(lr)


def _apply_lr_mult(optimizer: torch.optim.Optimizer, lr_mult: float) -> None:
    """Apply persistent LR multiplier against recorded unscaled scheduler LRs.

    :param torch.optim.Optimizer optimizer: Runtime optimizer.
    :param float lr_mult: Multiplier to apply (typically in (0, 1]).
    """
    mult = float(lr_mult)
    for group in optimizer.param_groups:
        base_lr = float(group.get("_lr_unscaled", group["lr"]))
        group["lr"] = base_lr * mult


def _apply_nonfinite_recovery(
    *,
    lr_mult: float,
    skip_streak: int,
) -> tuple[float, bool]:
    """Compute updated persistent LR multiplier after a non-finite window.

    The multiplier ratchets down on each nonfinite event and is applied after
    every ``lr_scheduler.step()`` so the scheduler cannot overwrite it.

    :param float lr_mult: Current persistent LR multiplier.
    :param int skip_streak: Current consecutive non-finite streak.
    :return tuple[float, bool]: (new lr_mult, optimizer_state_reset flag).
    """
    new_lr_mult = max(float(lr_mult) * float(_NONFINITE_LR_BACKOFF), float(_NONFINITE_LR_MULT_FLOOR))

    reset_state = False
    if int(skip_streak) % int(_NONFINITE_OPT_STATE_RESET_EVERY) == 0:
        reset_state = True

    return new_lr_mult, reset_state


def _scheduler_current_lr(scheduler: Any) -> float | None:
    """Read current LR from a scheduler when supported.

    :param Any scheduler: Scheduler-like object.
    :return float | None: Current LR for the first param group, if available.
    """
    if not hasattr(scheduler, "get_last_lr"):
        return None
    with suppress(Exception):
        values = scheduler.get_last_lr()
        if isinstance(values, (list, tuple)) and len(values) > 0:
            return float(values[0])
    return None


def _optimizer_has_stepped(optimizer: torch.optim.Optimizer) -> bool:
    """Best-effort check whether optimizer state has been initialized by a step.

    :param torch.optim.Optimizer optimizer: Optimizer to inspect.
    :return bool: True when optimizer has non-empty state.
    """
    with suppress(Exception):
        state = getattr(optimizer, "state", None)
        if state is not None:
            return bool(len(state) > 0)
    return False


def _sync_discriminator_embeddings_if_available(
    model: torch.nn.Module, *, accelerator: Any | None = None
) -> None:
    """Sync discriminator embedding buffers if the model exposes the hook.

    :param torch.nn.Module model: Runtime model (wrapped or unwrapped).
    :param Any | None accelerator: Optional Accelerator runtime for unwrapping.
    """
    wrapped_model = model
    target_model = model
    if accelerator is not None:
        with suppress(Exception):
            target_model = accelerator.unwrap_model(model)
        with suppress(Exception):
            target_model = unwrap_compiled_model(accelerator, target_model)

    fn = getattr(target_model, "sync_discriminator_embeddings_from_generator", None)
    if not callable(fn):
        return

    embedding_sharing = getattr(target_model, "embedding_sharing", None)
    if embedding_sharing is not None:
        if str(embedding_sharing).strip().lower() != "gdes":
            return
        if not bool(getattr(target_model, "_gdes_synced_embeddings", None)):
            return

    fsdp_cls = None
    fsdp2_module_cls = None
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as fsdp_cls  # type: ignore
    except ImportError:
        fsdp_cls = None
    try:
        from torch.distributed.fsdp import FSDPModule as fsdp2_module_cls  # type: ignore
    except ImportError:
        fsdp2_module_cls = None

    if fsdp_cls is not None and isinstance(wrapped_model, fsdp_cls):
        # FSDP1 wrapper path: summon only root-level params to avoid recursive
        # all-gathering of nested transformer shards. Nested FSDP1 wrapping is
        # not a supported path here; if nested params stay sharded, the sync hook
        # fails loudly on DTensor checks instead of silently desynchronizing.
        with fsdp_cls.summon_full_params(wrapped_model, recurse=False, writeback=True):
            with torch.no_grad():
                fn()
        return

    if fsdp2_module_cls is not None and isinstance(wrapped_model, fsdp2_module_cls):
        # FSDP2 (fully_shard) path: unshard/reshard around sync to avoid calling
        # sync_from() on sharded DTensor parameters.
        handle = wrapped_model.unshard(async_op=False)
        if handle is not None:
            handle.wait()
        try:
            with torch.no_grad():
                fn()
        finally:
            wrapped_model.reshard()
        return

    # Non-FSDP path.
    with torch.no_grad():
        fn()


def _collect_ga_window(
    *,
    train_iter: Iterator[dict[str, torch.Tensor]],
    ga_steps: int,
    token_weighted_ga: bool,
    disc_pad_token_id: int | None,
    include_has_gen_targets: bool,
    default_unweighted_token_count: float,
) -> tuple[list[Any], int, float, float, float]:
    """Collect one accumulation window and per-window token counts.

    :param Iterator[dict[str, torch.Tensor]] train_iter: Batch iterator.
    :param int ga_steps: Accumulation steps per window.
    :param bool token_weighted_ga: Token-weighted GA toggle.
    :param int | None disc_pad_token_id: Optional discriminator pad token id.
    :param bool include_has_gen_targets: Whether to append per-batch generator-target flags.
    :param float default_unweighted_token_count: Fallback token count when token weighting is disabled.
    :return tuple[list[Any], int, float, float, float]: Window payload and local token counters.
    """
    window: list[Any] = []
    consumed_in_window = 0
    local_window_input_tokens = 0.0
    local_gen_tokens = 0.0
    local_disc_tokens = 0.0

    for _ in range(max(1, int(ga_steps))):
        batch = next(train_iter)
        consumed_in_window += 1
        local_window_input_tokens += _count_input_tokens_for_batch(batch)

        if token_weighted_ga:
            gen_count, disc_count = _count_rtd_tokens_for_batch(
                batch,
                pad_token_id=disc_pad_token_id,
            )
            local_gen_tokens += gen_count
            local_disc_tokens += disc_count
        else:
            gen_count = float(default_unweighted_token_count)
            disc_count = float(default_unweighted_token_count)

        if include_has_gen_targets:
            has_gen_targets = bool(batch["labels"].ne(-100).any().item())
            window.append((batch, gen_count, disc_count, has_gen_targets))
        else:
            window.append((batch, gen_count, disc_count))

    return (
        window,
        int(consumed_in_window),
        float(local_window_input_tokens),
        float(local_gen_tokens),
        float(local_disc_tokens),
    )


def _resolve_window_token_weights(
    *,
    accelerator: Any,
    token_weighted_ga: bool,
    local_gen_tokens: float,
    local_disc_tokens: float,
    next_step: int,
) -> tuple[float, float, bool, bool]:
    """Resolve per-window token denominators and zero-token flags.

    :param Any accelerator: Accelerator runtime.
    :param bool token_weighted_ga: Token-weighted GA toggle.
    :param float local_gen_tokens: Local generator-token count for the window.
    :param float local_disc_tokens: Local discriminator-token count for the window.
    :param int next_step: Step number used in zero-token warnings.
    :return tuple[float, float, bool, bool]: Generator/discriminator denominators and zero-token flags.
    """
    if not token_weighted_ga:
        return 1.0, 1.0, False, False

    local_totals = torch.tensor(
        [local_gen_tokens, local_disc_tokens],
        device=accelerator.device,
        dtype=torch.float32,
    )
    mean_totals = accelerator.reduce(local_totals, reduction="mean")
    raw_gen_window_tokens_per_rank = float(mean_totals[0].item())
    raw_disc_window_tokens_per_rank = float(mean_totals[1].item())
    (
        gen_window_tokens_per_rank,
        disc_window_tokens_per_rank,
        gen_window_zero_tokens,
        disc_window_zero_tokens,
    ) = _resolve_window_token_denominators(
        gen_window_tokens_per_rank_raw=raw_gen_window_tokens_per_rank,
        disc_window_tokens_per_rank_raw=raw_disc_window_tokens_per_rank,
    )
    if bool(getattr(accelerator, "is_main_process", False)) and (
        gen_window_zero_tokens or disc_window_zero_tokens
    ):
        logger.warning(
            "Token-weighted GA window has zero effective tokens "
            "(next_step=%d, gen_zero=%s, disc_zero=%s, gen_raw=%.1f, disc_raw=%.1f); "
            "corresponding loss term is zero-weighted for this window.",
            int(next_step),
            bool(gen_window_zero_tokens),
            bool(disc_window_zero_tokens),
            float(raw_gen_window_tokens_per_rank),
            float(raw_disc_window_tokens_per_rank),
        )
    return (
        float(gen_window_tokens_per_rank),
        float(disc_window_tokens_per_rank),
        bool(gen_window_zero_tokens),
        bool(disc_window_zero_tokens),
    )


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move all batch tensors onto a device.

    :param dict[str, torch.Tensor] batch: Tensor batch mapping.
    :param torch.device device: Destination device.
    :return dict[str, torch.Tensor]: Batch placed on ``device``.
    """
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
