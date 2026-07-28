#!/usr/bin/env python3
"""Evaluate RTD checkpoints on a deterministic alternate shuffle of their training source.

Example:
    python tools/evaluate_rtd_checkpoint.py CONFIG CHECKPOINT \
        --output local-scratch/run/eval-checkpoint.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from deberta.config import load_config
from deberta.data.loading import load_hf_dataset
from deberta.modeling import DebertaV3RTDPretrainer, build_backbones
from deberta.modeling.mask_utils import attention_mask_to_active_tokens
from deberta.run_artifacts import load_materialized_backbone_configs, materialized_tokenizer_path
from deberta.run_layout import infer_run_dir_from_checkpoint
from deberta.training.compile import prepare_flash_attention_batch_metadata
from deberta.training.runtime import _build_train_dataset_and_collator
from deberta.utils.checkpoint import load_model_state_with_compile_key_remap


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def _checkpoint_step(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[-1])


def _autocast_context(precision: str) -> Any:
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def _forward_discriminator_with_diagnostics(
    model: DebertaV3RTDPretrainer,
    *,
    input_ids: torch.Tensor,
    corrupted_input_ids: torch.Tensor,
    disc_labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    token_type_ids: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    doc_context_index: torch.Tensor | None,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the shared discriminator phase and capture its hidden states and logits."""
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _capture_head_output(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured.append((inputs[0], output))

    handle = model.discriminator_head.register_forward_hook(_capture_head_output)
    try:
        with _autocast_context(precision):
            model.forward_discriminator_phase(
                input_ids=input_ids,
                corrupted_input_ids=corrupted_input_ids,
                disc_labels=disc_labels,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                doc_context_index=doc_context_index,
            )
    finally:
        handle.remove()

    (diagnostics,) = captured
    return diagnostics


def _ranking_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float | None, float | None]:
    """Return tie-aware binary ranking metrics, using ``None`` when a class is absent."""
    scores = logits.detach().float().cpu().flatten()
    targets = labels.detach().bool().cpu().flatten()
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order]

    group_end = torch.ones_like(sorted_targets, dtype=torch.bool)
    if group_end.numel() > 1:
        group_end[:-1] = sorted_scores[:-1].ne(sorted_scores[1:])
    group_indices = torch.nonzero(group_end, as_tuple=False).squeeze(-1)

    true_positives = sorted_targets.to(torch.float64).cumsum(0)[group_indices]
    false_positives = (~sorted_targets).to(torch.float64).cumsum(0)[group_indices]
    positive_count = sorted_targets.sum().to(torch.float64)
    negative_count = (~sorted_targets).sum().to(torch.float64)
    if int(positive_count.item()) == 0:
        return None, None

    recall = true_positives / positive_count
    precision = true_positives / (true_positives + false_positives)
    previous_recall = torch.cat([torch.zeros(1, dtype=torch.float64), recall[:-1]])
    average_precision = ((recall - previous_recall) * precision).sum()

    if int(negative_count.item()) == 0:
        return None, float(average_precision.item())

    tpr = torch.cat([torch.zeros(1, dtype=torch.float64), recall])
    fpr = torch.cat([torch.zeros(1, dtype=torch.float64), false_positives / negative_count])
    roc_auc = torch.trapezoid(tpr, fpr)
    return float(roc_auc.item()), float(average_precision.item())


def _discriminator_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float | int | None]:
    scores = logits.detach().float().cpu().flatten()
    targets = labels.detach().float().cpu().flatten()
    positive_count = int(targets.sum().item())
    positive_rate = float(targets.mean().item())
    prior_bce = -sum(
        probability * math.log(probability)
        for probability in (positive_rate, 1.0 - positive_rate)
        if probability > 0.0
    )
    bce = float(F.binary_cross_entropy_with_logits(scores, targets).item())
    roc_auc, average_precision = _ranking_metrics(scores, targets)
    predictions = scores.gt(0)
    true_positive_count = int((predictions & targets.bool()).sum().item())
    return {
        "tokens": int(targets.numel()),
        "positives": positive_count,
        "positive_rate": positive_rate,
        "bce": bce,
        "prior_bce": prior_bce,
        "loss_gain": prior_bce - bce,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "accuracy_at_zero": float(predictions.eq(targets.bool()).float().mean().item()),
        "recall_at_zero": float(true_positive_count / positive_count) if positive_count else 0.0,
        "predicted_positive_rate_at_zero": float(predictions.float().mean().item()),
        "logit_mean": float(scores.mean().item()),
        "logit_std": float(scores.std(unbiased=False).item()),
    }


def _load_checkpoint_artifacts(cfg: Any, checkpoint: Path) -> tuple[Any, Any, Any]:
    run_dir = infer_run_dir_from_checkpoint(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(materialized_tokenizer_path(run_dir), use_fast=True)
    disc_cfg, gen_cfg = load_materialized_backbone_configs(
        run_dir=run_dir,
        model_cfg=cfg.model,
    )
    return tokenizer, disc_cfg, gen_cfg


def _shared_run_dir(checkpoints: list[Path]) -> Path:
    """Return the common run directory for a checkpoint trajectory."""
    run_dir = infer_run_dir_from_checkpoint(checkpoints[0])
    for checkpoint in checkpoints[1:]:
        checkpoint_run_dir = infer_run_dir_from_checkpoint(checkpoint)
        if checkpoint_run_dir != run_dir:
            raise ValueError(
                "All evaluated checkpoints must belong to the same run; "
                f"got '{run_dir}' and '{checkpoint_run_dir}'."
            )
    return run_dir


def _build_model(
    cfg: Any,
    tokenizer: Any,
    *,
    disc_cfg: Any,
    gen_cfg: Any,
) -> DebertaV3RTDPretrainer:
    model_cfg = replace(cfg.model, hf=replace(cfg.model.hf, attention_impl="eager"))
    if str(model_cfg.backbone_type).strip().lower() == "hf_deberta_v2":
        for component_config in (disc_cfg, gen_cfg):
            component_config.hf_attention_impl = "eager"
            component_config.hf_flash = {"kernel_overrides_path": None}
    disc, gen = build_backbones(
        model_cfg=model_cfg,
        disc_config=disc_cfg,
        gen_config=gen_cfg,
        load_pretrained_weights=False,
    )
    return DebertaV3RTDPretrainer(
        discriminator_backbone=disc,
        generator_backbone=gen,
        disc_config=disc_cfg,
        gen_config=gen_cfg,
        embedding_sharing=model_cfg.embedding_sharing,
        additional_forbidden_token_ids=tokenizer.all_special_ids,
    )


def _build_eval_batch(cfg: Any, tokenizer: Any, *, batches: int, seed: int) -> dict[str, torch.Tensor]:
    eval_train_cfg = replace(
        cfg.train,
        seed=int(seed),
        dataloader=replace(cfg.train.dataloader, num_workers=0),
    )
    raw_train = load_hf_dataset(cfg.data)
    dataset, collator = _build_train_dataset_and_collator(
        raw_train=raw_train,
        tokenizer=tokenizer,
        data_cfg=cfg.data,
        train_cfg=eval_train_cfg,
        process_index=0,
        num_processes=1,
        flash_enabled=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.per_device_train_batch_size),
        collate_fn=collator,
        num_workers=0,
        drop_last=True,
    )

    torch.manual_seed(int(seed))
    iterator = iter(loader)
    rows = [next(iterator) for _ in range(int(batches))]
    batch = _assemble_eval_batch(rows, backbone_type=str(cfg.model.backbone_type))
    # Release streaming/PyArrow iterator state before model evaluation so its
    # background callbacks cannot survive until interpreter shutdown.
    del rows, iterator, loader, dataset, raw_train
    gc.collect()
    return batch


def _assemble_eval_batch(rows: list[dict[str, Any]], *, backbone_type: str) -> dict[str, torch.Tensor]:
    """Concatenate collated rows and apply training-path attention routing.

    :param list[dict[str, Any]] rows: Collator output batches.
    :param str backbone_type: Backbone type string for mask routing.
    :return dict[str, torch.Tensor]: Evaluation batch with ``doc_ids`` converted
        to a pairwise doc-block attention mask, mirroring the training loop.
    """
    # The collator drops all-ones masks per collate call, so restore them
    # before concatenation when any row carries real padding.
    rows = [dict(row) for row in rows]
    if any(isinstance(row.get("attention_mask"), torch.Tensor) for row in rows):
        for row in rows:
            if not isinstance(row.get("attention_mask"), torch.Tensor):
                row["attention_mask"] = torch.ones_like(row["input_ids"])
    tensor_keys = (
        "input_ids",
        "labels",
        "attention_mask",
        "token_type_ids",
        "position_ids",
        "doc_context_index",
        "doc_ids",
    )
    # attention_mask is exempt: the restoration block above already normalizes
    # its presence across rows, so mixed presence there is expected, not a bug.
    partial_presence_exempt = {"attention_mask"}
    batch: dict[str, torch.Tensor] = {}
    for key in tensor_keys:
        present = [isinstance(row.get(key), torch.Tensor) for row in rows]
        if all(present):
            batch[key] = torch.cat([row[key] for row in rows])
        elif any(present) and key not in partial_presence_exempt:
            # Silently keeping only the rows that happen to carry this key
            # would drop it from the batch without warning, the same failure
            # class fixed for doc_ids in commit 2319fe1.
            raise ValueError(
                f"Key {key!r} is present on some collated rows but not others; "
                "refusing to silently drop it from the evaluation batch."
            )
    # Doc-block batches carry compact doc_ids that must become a pairwise
    # attention mask before any forward pass; otherwise checkpoints trained
    # with blocked cross-document attention are scored without it.
    batch, _ = prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type=backbone_type,
        flash_enabled=False,
    )
    return batch


def _state_stats(model: torch.nn.Module) -> dict[str, int]:
    state = model.state_dict()
    elements = 0
    nonfinite = 0
    for tensor in state.values():
        elements += int(tensor.numel())
        if tensor.is_floating_point() or tensor.is_complex():
            nonfinite += int((~torch.isfinite(tensor)).sum().item())
    return {"tensors": len(state), "elements": elements, "nonfinite_elements": nonfinite}


def _replacement_rate(disc_labels: torch.Tensor, masked_tokens: float) -> float:
    """Fraction of masked positions where the generator sampled a replacement.

    ``disc_labels`` marks replaced positions across the full ``(N, S)`` grid,
    including padding and unmasked positions that were never eligible for
    replacement. Dividing by ``masked_tokens`` (rather than ``disc_labels``'s
    own element count, i.e. ``disc_labels.mean()``) is what makes this a rate
    over masked positions instead of a rate over the whole padded batch.

    :param torch.Tensor disc_labels: ``(N, S)`` 0/1 tensor marking replaced positions.
    :param float masked_tokens: Count of MLM-masked positions across the batch.
    :return float: ``replacements / max(masked_tokens, 1)``.
    """
    replacements = float(disc_labels.sum().item())
    return replacements / max(float(masked_tokens), 1.0)


def _precision_mismatch_warning(eval_precision: str, configured_mixed_precision: str) -> str | None:
    """Flag when the requested eval precision disagrees with the run's training precision.

    :param str eval_precision: CLI ``--precision`` choice (``bf16`` or ``fp32``).
    :param str configured_mixed_precision: Normalized ``cfg.train.mixed_precision``
        value (``bf16`` or ``no``).
    :return str | None: A one-line warning message, or ``None`` when they agree.
    """
    expected_eval_precision = "bf16" if configured_mixed_precision == "bf16" else "fp32"
    if eval_precision == expected_eval_precision:
        return None
    return (
        f"--precision={eval_precision} does not match the run's "
        f"train.mixed_precision={configured_mixed_precision!r} "
        f"(expected --precision={expected_eval_precision})."
    )


def _evaluation_provenance(
    *,
    precision: str,
    sampling_temperature: float,
    configured_mixed_precision: str,
) -> dict[str, Any]:
    """Build the provenance block recording how a checkpoint was evaluated.

    The tool always evaluates with eager attention regardless of how the run
    trained, so that fact is recorded explicitly alongside the precision and
    sampling temperature actually used.

    :param str precision: Autocast precision used for evaluation (``bf16``/``fp32``).
    :param float sampling_temperature: Generator sampling temperature used for evaluation.
    :param str configured_mixed_precision: The run's configured ``train.mixed_precision``.
    :return dict[str, Any]: Provenance block for the top-level JSON payload.
    """
    return {
        "attention_impl": "eager",
        "flash_enabled": False,
        "precision": precision,
        "sampling_temperature": float(sampling_temperature),
        "configured_mixed_precision": configured_mixed_precision,
    }


@torch.no_grad()
def _evaluate_checkpoint(
    model: DebertaV3RTDPretrainer,
    *,
    batch: dict[str, torch.Tensor],
    checkpoint: Path,
    batch_size: int,
    precision: str,
    sampling_temperature: float,
    seed: int,
) -> dict[str, Any]:
    load_stats = load_model_state_with_compile_key_remap(model, checkpoint)
    model.sync_discriminator_embeddings_from_generator()
    model.eval()
    state_stats = _state_stats(model)

    device = next(model.parameters()).device
    payload = {key: value.to(device) for key, value in batch.items()}
    masked = payload["input_ids"]
    labels = payload["labels"]
    attention_mask = payload.get("attention_mask")
    token_type_ids = payload.get("token_type_ids")
    position_ids = payload.get("position_ids")
    doc_context_index = payload.get("doc_context_index")
    original = masked.clone()
    selected = labels.ne(-100)
    original[selected] = labels[selected]

    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    corrupted_batches: list[torch.Tensor] = []
    disc_label_batches: list[torch.Tensor] = []
    gen_loss_numerator = 0.0
    masked_tokens = 0.0
    for start in range(0, int(masked.shape[0]), int(batch_size)):
        stop = start + int(batch_size)
        with _autocast_context(precision):
            output = model.forward_generator_phase(
                input_ids=masked[start:stop],
                attention_mask=attention_mask[start:stop] if attention_mask is not None else None,
                labels=labels[start:stop],
                token_type_ids=token_type_ids[start:stop] if token_type_ids is not None else None,
                position_ids=position_ids[start:stop] if position_ids is not None else None,
                sampling_temperature=sampling_temperature,
            )
        count = float(output.gen_token_count.item())
        gen_loss_numerator += float(output.gen_loss_raw.float().item()) * count
        masked_tokens += count
        corrupted_batches.append(output.corrupted_input_ids)
        disc_label_batches.append(output.disc_labels)

    corrupted = torch.cat(corrupted_batches)
    disc_labels = torch.cat(disc_label_batches)
    replacement_rate = _replacement_rate(disc_labels, masked_tokens)

    logits_batches: list[torch.Tensor] = []
    active_label_batches: list[torch.Tensor] = []
    centered_square_sum = 0.0
    centered_element_count = 0
    for start in range(0, int(original.shape[0]), int(batch_size)):
        stop = start + int(batch_size)
        batch_attention = attention_mask[start:stop] if attention_mask is not None else None
        hidden, disc_logits = _forward_discriminator_with_diagnostics(
            model,
            input_ids=original[start:stop],
            corrupted_input_ids=corrupted[start:stop],
            disc_labels=disc_labels[start:stop],
            attention_mask=batch_attention,
            token_type_ids=token_type_ids[start:stop] if token_type_ids is not None else None,
            position_ids=position_ids[start:stop] if position_ids is not None else None,
            doc_context_index=doc_context_index[start:stop] if doc_context_index is not None else None,
            precision=precision,
        )

        active = attention_mask_to_active_tokens(
            input_ids=original[start:stop],
            attention_mask=batch_attention,
        )
        hidden = hidden.float()
        active_f = active.unsqueeze(-1).to(hidden)
        token_count = active_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        sequence_mean = (hidden * active_f).sum(dim=1, keepdim=True) / token_count
        centered_square_sum += float(((hidden - sequence_mean) * active_f).square().sum().item())
        centered_element_count += int(active.sum().item()) * int(hidden.shape[-1])
        logits_batches.append(disc_logits[active].float().cpu())
        active_label_batches.append(disc_labels[start:stop][active].float().cpu())

    discriminator = _discriminator_metrics(torch.cat(logits_batches), torch.cat(active_label_batches))
    discriminator["token_centered_rms"] = math.sqrt(centered_square_sum / centered_element_count)
    generator_ce = gen_loss_numerator / masked_tokens
    return {
        "path": str(checkpoint.resolve()),
        "load": load_stats,
        "state": state_stats,
        "generator": {
            "cross_entropy": generator_ce,
            "perplexity": math.exp(generator_ce),
            "masked_tokens": int(masked_tokens),
            "replacement_rate": replacement_rate,
            "replacements": int(disc_labels.sum().item()),
        },
        "discriminator": discriminator,
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint evaluation")

    cfg = load_config(args.config)
    sampling_temperature = float(cfg.train.objective.sampling_temperature)
    configured_mixed_precision = str(cfg.train.mixed_precision)
    mismatch_warning = _precision_mismatch_warning(args.precision, configured_mixed_precision)
    if mismatch_warning is not None:
        print(f"WARNING: {mismatch_warning}", file=sys.stderr)

    checkpoints = sorted(args.checkpoints, key=_checkpoint_step)
    _shared_run_dir(checkpoints)
    tokenizer, disc_cfg, gen_cfg = _load_checkpoint_artifacts(cfg, checkpoints[0])
    batch = _build_eval_batch(cfg, tokenizer, batches=args.batches, seed=args.seed)
    model = (
        _build_model(
            cfg,
            tokenizer,
            disc_cfg=disc_cfg,
            gen_cfg=gen_cfg,
        )
        .to(torch.device("cuda"))
        .eval()
    )

    results: dict[str, Any] = {}
    for checkpoint in checkpoints:
        step = str(_checkpoint_step(checkpoint))
        result = _evaluate_checkpoint(
            model,
            batch=batch,
            checkpoint=checkpoint,
            batch_size=int(cfg.train.per_device_train_batch_size),
            precision=args.precision,
            sampling_temperature=sampling_temperature,
            seed=args.seed,
        )
        results[step] = result
        summary = result["discriminator"]
        print(
            json.dumps(
                {
                    "step": int(step),
                    "generator_ce": result["generator"]["cross_entropy"],
                    "replacement_rate": result["generator"]["replacement_rate"],
                    "discriminator_bce": summary["bce"],
                    "loss_gain": summary["loss_gain"],
                    "roc_auc": summary["roc_auc"],
                    "average_precision": summary["average_precision"],
                    "token_centered_rms": summary["token_centered_rms"],
                    "nonfinite_elements": result["state"]["nonfinite_elements"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    attention_mask = batch.get("attention_mask")
    active = attention_mask_to_active_tokens(
        input_ids=batch["input_ids"],
        attention_mask=attention_mask,
    )
    output = {
        "config": str(args.config.resolve()),
        "precision": args.precision,
        "evaluation": _evaluation_provenance(
            precision=args.precision,
            sampling_temperature=sampling_temperature,
            configured_mixed_precision=configured_mixed_precision,
        ),
        "evaluation_data": {
            "dataset": cfg.data.source.dataset_name,
            "dataset_config": cfg.data.source.dataset_config_name,
            "split": cfg.data.source.train_split,
            "shuffle_seed": int(args.seed),
            "batches": int(args.batches),
            "sequences": int(batch["input_ids"].shape[0]),
            "sequence_length": int(batch["input_ids"].shape[1]),
            "active_tokens": int(active.sum().item()),
            "masked_tokens": int(batch["labels"].ne(-100).sum().item()),
        },
        "checkpoints": results,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
