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
from deberta.modeling import DebertaV3RTDPretrainer, build_backbone_configs, build_backbones
from deberta.modeling.rtd import attention_mask_to_active_tokens
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


def _ranking_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Return tie-aware ROC-AUC and average precision for binary targets."""
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

    recall = true_positives / positive_count
    precision = true_positives / (true_positives + false_positives)
    previous_recall = torch.cat([torch.zeros(1, dtype=torch.float64), recall[:-1]])
    average_precision = ((recall - previous_recall) * precision).sum()

    tpr = torch.cat([torch.zeros(1, dtype=torch.float64), recall])
    fpr = torch.cat([torch.zeros(1, dtype=torch.float64), false_positives / negative_count])
    roc_auc = torch.trapezoid(tpr, fpr)
    return float(roc_auc.item()), float(average_precision.item())


def _discriminator_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int]:
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


def _build_model(cfg: Any, tokenizer: Any) -> DebertaV3RTDPretrainer:
    model_cfg = replace(cfg.model, hf=replace(cfg.model.hf, attention_impl="eager"))
    disc_cfg, gen_cfg = build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=int(cfg.data.packing.max_seq_length),
    )
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
    tensor_keys = (
        "input_ids",
        "labels",
        "attention_mask",
        "token_type_ids",
        "position_ids",
        "doc_context_index",
    )
    batch = {
        key: torch.cat([row[key] for row in rows])
        for key in tensor_keys
        if all(isinstance(row.get(key), torch.Tensor) for row in rows)
    }
    del rows, iterator, loader, dataset, raw_train
    gc.collect()
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


@torch.no_grad()
def _evaluate_checkpoint(
    model: DebertaV3RTDPretrainer,
    *,
    batch: dict[str, torch.Tensor],
    checkpoint: Path,
    batch_size: int,
    precision: str,
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
                sampling_temperature=1.0,
            )
        count = float(output.gen_token_count.item())
        gen_loss_numerator += float(output.gen_loss_raw.float().item()) * count
        masked_tokens += count
        corrupted_batches.append(output.corrupted_input_ids)
        disc_label_batches.append(output.disc_labels)

    corrupted = torch.cat(corrupted_batches)
    disc_labels = torch.cat(disc_label_batches)
    replacement_rate = float(disc_labels.mean().item())

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
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer.name_or_path, use_fast=True)
    batch = _build_eval_batch(cfg, tokenizer, batches=args.batches, seed=args.seed)
    model = _build_model(cfg, tokenizer).to(torch.device("cuda")).eval()

    results: dict[str, Any] = {}
    for checkpoint in sorted(args.checkpoints, key=_checkpoint_step):
        step = str(_checkpoint_step(checkpoint))
        result = _evaluate_checkpoint(
            model,
            batch=batch,
            checkpoint=checkpoint,
            batch_size=int(cfg.train.per_device_train_batch_size),
            precision=args.precision,
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
