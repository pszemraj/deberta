"""Loop math and token-count utilities for RTD pretraining."""

from __future__ import annotations

from typing import Any

import torch

from deberta.modeling.rtd import attention_mask_to_active_tokens


def _count_rtd_tokens_for_batch(
    batch: dict[str, torch.Tensor],
    *,
    pad_token_id: int | None,
) -> tuple[float, float]:
    """Return generator/discriminator active-token counts for one microbatch.

    :param dict[str, torch.Tensor] batch: Microbatch tensors.
    :param int | None pad_token_id: Padding token id.
    :return tuple[float, float]: (generator_count, discriminator_count).
    """
    labels = batch["labels"]
    gen_count = float(labels.ne(-100).sum().item())
    disc_active = attention_mask_to_active_tokens(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        pad_token_id=pad_token_id,
    )
    disc_count = float(disc_active.sum().item())
    return gen_count, disc_count


def _count_input_tokens_for_batch(batch: dict[str, torch.Tensor]) -> float:
    """Return non-padding input-token count for one microbatch.

    :param dict[str, torch.Tensor] batch: Microbatch mapping.
    :return float: Count of active input tokens.
    """
    attention_mask = batch.get("attention_mask")
    input_ids = batch.get("input_ids")
    if isinstance(attention_mask, torch.Tensor):
        if isinstance(input_ids, torch.Tensor):
            active = attention_mask_to_active_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=None,
            )
            return float(active.detach().sum().item())

        mask = attention_mask.detach().to(dtype=torch.bool)
        if mask.ndim == 4:
            mask = mask[:, 0] if mask.shape[1] == 1 else mask.any(dim=1)
        if mask.ndim == 3:
            if mask.shape[-2] == 1:
                mask = mask[:, 0, :]
            else:
                mask = torch.diagonal(mask, dim1=-2, dim2=-1)
        if mask.ndim == 2:
            return float(mask.sum().item())
        return float(mask.reshape(-1).sum().item())

    if isinstance(input_ids, torch.Tensor):
        return float(input_ids.numel())

    return 0.0


def _token_weighted_micro_objective(
    *,
    gen_loss: torch.Tensor,
    disc_loss: torch.Tensor,
    gen_count: float,
    disc_count: float,
    gen_window_tokens_per_rank: float,
    disc_window_tokens_per_rank: float,
    gen_loss_weight: float,
    disc_loss_weight: float,
) -> torch.Tensor:
    """Build token-weighted microbatch objective for one accumulation window.

    :param torch.Tensor gen_loss: Generator loss mean for the microbatch.
    :param torch.Tensor disc_loss: Discriminator loss mean for the microbatch.
    :param float gen_count: Generator token count for the microbatch.
    :param float disc_count: Discriminator token count for the microbatch.
    :param float gen_window_tokens_per_rank: Mean generator-token total per rank in the accumulation window.
    :param float disc_window_tokens_per_rank: Mean discriminator-token total per rank in the accumulation window.
    :param float gen_loss_weight: Generator loss weight.
    :param float disc_loss_weight: Discriminator loss weight.
    :return torch.Tensor: Unscaled microbatch objective contribution.
    """
    gen_scale = float(gen_count) / max(float(gen_window_tokens_per_rank), 1.0)
    disc_scale = float(disc_count) / max(float(disc_window_tokens_per_rank), 1.0)
    return float(gen_loss_weight) * gen_scale * gen_loss + float(disc_loss_weight) * disc_scale * disc_loss


def _resolve_window_token_denominators(
    *, gen_window_tokens_per_rank_raw: float, disc_window_tokens_per_rank_raw: float
) -> tuple[float, float, bool, bool]:
    """Resolve safe per-window token denominators for token-weighted GA.

    :param float gen_window_tokens_per_rank_raw: Raw mean generator-token count per rank for the window.
    :param float disc_window_tokens_per_rank_raw: Raw mean discriminator-token count per rank for the window.
    :return tuple[float, float, bool, bool]: ``(gen_denom, disc_denom, gen_is_zero, disc_is_zero)``.
    """
    gen_zero = float(gen_window_tokens_per_rank_raw) <= 0.0
    disc_zero = float(disc_window_tokens_per_rank_raw) <= 0.0
    gen_denom = max(float(gen_window_tokens_per_rank_raw), 1.0)
    disc_denom = max(float(disc_window_tokens_per_rank_raw), 1.0)
    return float(gen_denom), float(disc_denom), bool(gen_zero), bool(disc_zero)


def _finalize_window_metric_loss(
    *, accumulated_loss: torch.Tensor, ga_steps: int, token_weighted_ga: bool
) -> torch.Tensor:
    """Finalize per-window loss metric for logging.

    :param torch.Tensor accumulated_loss: Sum of microbatch metric contributions.
    :param int ga_steps: Gradient accumulation steps in the window.
    :param bool token_weighted_ga: Whether token-weighted GA is enabled.
    :return torch.Tensor: Window-level scalar loss metric.
    """
    if token_weighted_ga:
        return accumulated_loss
    denom = max(1, int(ga_steps))
    return accumulated_loss / float(denom)


def _scale_loss_for_backward(*, loss: torch.Tensor, ga_steps: int, token_weighted_ga: bool) -> torch.Tensor:
    """Prepare microbatch loss for ``Accelerator.backward``.

    Accelerate scales all backward losses by ``1 / gradient_accumulation_steps``.
    Token-weighted micro objectives are already normalized over the accumulation
    window, so we cancel that scaling for the token-weighted path only.

    :param torch.Tensor loss: Raw microbatch loss/objective.
    :param int ga_steps: Gradient accumulation steps.
    :param bool token_weighted_ga: Whether token-weighted GA is enabled.
    :return torch.Tensor: Loss to pass into ``accelerator.backward``.
    """
    return loss if not token_weighted_ga else (loss * float(max(1, int(ga_steps))))


def _should_clip_gradients(*, sync_gradients: bool, max_grad_norm: float | int | None) -> bool:
    """Return whether gradient clipping should run for this micro-step.

    :param bool sync_gradients: Whether gradients are synchronized this step.
    :param float | int | None max_grad_norm: Configured clipping norm.
    :return bool: ``True`` when clipping should be applied.
    """
    return bool(sync_gradients) and max_grad_norm is not None and float(max_grad_norm) > 0.0


def _sum_local_scalar(*, accelerator: Any, x: float) -> float:
    """Sum a local float across all ranks.

    :param Any accelerator: Accelerate-style runtime object.
    :param float x: Local scalar.
    :return float: Reduced global sum.
    """
    local = torch.tensor([x], device=accelerator.device, dtype=torch.float64)
    return float(accelerator.reduce(local, reduction="sum")[0].item())
