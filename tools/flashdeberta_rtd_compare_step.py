#!/usr/bin/env python3
"""Compare one packed RTD step between eager and FlashDeBERTa.

This is a diagnostic for training collapse: both models start from identical
weights, consume the same packed batch, and run the same RTD phase logic. The
tool reports where eager and flash first diverge: generator hidden/logits,
sampling artifacts, discriminator logits/loss, or gradients.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import _bench_common as bench  # noqa: E402  (inserts src/ on sys.path at import)
import torch
import torch.nn.functional as F

from deberta.config import load_config  # noqa: E402
from deberta.modeling import DebertaV3RTDPretrainer  # noqa: E402
from deberta.modeling.rtd import attention_mask_to_active_tokens  # noqa: E402
from deberta.training.compile import (  # noqa: E402
    _stabilize_compile_attention_mask,
    prepare_flash_attention_batch_metadata,
)
from deberta.training.steps import (  # noqa: E402
    _move_batch_to_device,
    _sync_discriminator_embeddings_if_available,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/flashdeberta/pretrain_rtd_hf_deberta_v3pos_smol2stage4_1024_wp32k_v2_docblock.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument(
        "--docblock-bias-seq-len",
        type=int,
        default=1024,
        help="Route flash doc-block batches through docblock_bias at this sequence length; 0 disables it.",
    )
    parser.add_argument(
        "--flash-route",
        choices=("docblock", "docblock_bias"),
        default="docblock_bias",
        help="Expected flash route for the comparison batch.",
    )
    parser.add_argument(
        "--max-grad-report",
        type=int,
        default=25,
        help="Maximum number of highest-difference gradient rows to include per phase.",
    )
    return parser.parse_args()


_autocast_context = bench.autocast_context


def _tensor_batch_clone(batch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return out


def _model_overrides(*, attention_impl: str, docblock_bias_seq_len: int) -> list[str]:
    overrides = [
        "logging.wandb.enabled=false",
        "logging.backend=none",
        "train.checkpoint.export_hf_final=false",
        f"model.hf.attention_impl={attention_impl}",
        "train.compile.enabled=false",
    ]
    if str(attention_impl).strip().lower() == "flash":
        value = "null" if int(docblock_bias_seq_len) <= 0 else str(int(docblock_bias_seq_len))
        overrides.append(f"model.hf.flash.docblock_bias_seq_len={value}")
    return overrides


def _build_model(
    *,
    cfg: Any,
    tokenizer: Any,
    device: torch.device,
) -> DebertaV3RTDPretrainer:
    return bench.build_rtd_pretrainer(cfg=cfg, tokenizer=tokenizer, device=device, train_mode=True)


def _prepare_batch(
    *,
    batch_cpu: dict[str, Any],
    device: torch.device,
    cfg: Any,
    flash_enabled: bool,
) -> tuple[dict[str, Any], Any | None]:
    batch = _move_batch_to_device(_tensor_batch_clone(batch_cpu), device)
    backbone_type = str(cfg.model.backbone_type)
    batch = _stabilize_compile_attention_mask(
        batch=batch,
        compile_enabled=False,
        compile_scope="disabled",
        backbone_type=backbone_type,
    )
    batch, flash_meta = prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type=backbone_type,
        flash_enabled=flash_enabled,
        flash_cfg=getattr(cfg.model.hf, "flash", None),
    )
    return batch, flash_meta


def _masked_generator_logits(
    *,
    model: DebertaV3RTDPretrainer,
    batch: dict[str, Any],
    flash_meta: Any | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = batch["labels"]
    masked_positions = labels.ne(-100)
    masked_idx = torch.nonzero(masked_positions.view(-1), as_tuple=False).squeeze(-1)

    gen_forward_kwargs: dict[str, Any] = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch.get("attention_mask"),
        "token_type_ids": batch.get("token_type_ids"),
        "return_dict": True,
    }
    pos_biased = bool(getattr(model.gen_config, "position_biased_input", True))
    z_steps = int(getattr(model.generator, "z_steps", getattr(model.gen_config, "z_steps", 0)) or 0)
    use_emd = (not pos_biased) and z_steps <= 1
    if use_emd:
        gen_forward_kwargs["output_hidden_states"] = True
    if flash_meta is not None and getattr(model, "_generator_accepts_flash_kwargs", False):
        gen_forward_kwargs["flash_meta"] = flash_meta

    gen_out = model.generator(**gen_forward_kwargs)
    hidden = gen_out.last_hidden_state

    if masked_idx.numel() == 0:
        empty_logits = hidden.new_zeros((0, model.generator_lm_head.bias.numel()))
        empty_labels = labels.new_zeros((0,))
        return hidden, empty_logits, empty_labels, masked_positions

    if use_emd:
        gen_masked_hidden = model.enhanced_mask_decoder(
            encoder_hidden_states=gen_out.hidden_states,
            masked_positions=masked_positions,
            attention_mask=batch.get("attention_mask"),
            embeddings=model.generator.embeddings,
            encoder=model.generator.encoder,
            flash_meta=flash_meta if getattr(model, "_generator_accepts_flash_kwargs", False) else None,
        )
    else:
        hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        gen_masked_hidden = hidden_flat.index_select(0, masked_idx)

    masked_labels = labels.view(-1).index_select(0, masked_idx)
    word_w = model._get_generator_word_embedding_weight()
    logits = model.generator_lm_head(gen_masked_hidden, word_embedding_weight=word_w)
    return hidden, logits, masked_labels, masked_positions


def _discriminator_logits_and_loss(
    *,
    model: DebertaV3RTDPretrainer,
    batch: dict[str, Any],
    flash_meta: Any | None,
    corrupted_input_ids: torch.Tensor,
    disc_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    disc_forward_kwargs: dict[str, Any] = {
        "input_ids": corrupted_input_ids,
        "attention_mask": batch.get("attention_mask"),
        "token_type_ids": batch.get("token_type_ids"),
        "return_dict": True,
    }
    if flash_meta is not None and getattr(model, "_discriminator_accepts_flash_kwargs", False):
        disc_forward_kwargs["flash_meta"] = flash_meta
    disc_out = model.discriminator(**disc_forward_kwargs)
    logits = model.discriminator_head(
        disc_out.last_hidden_state,
        attention_mask=batch.get("attention_mask"),
        flash_meta=flash_meta,
    )
    active = attention_mask_to_active_tokens(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
    )
    active_f = active.to(dtype=torch.float32)
    loss = (
        F.binary_cross_entropy_with_logits(logits.float(), disc_labels.float(), reduction="none") * active_f
    ).sum() / active_f.sum().clamp_min(1.0)
    return logits, loss, active


def _masked_tensor_values(tensor: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    values = tensor.detach().float()
    if mask is None:
        return values.reshape(-1)
    bool_mask = mask.to(device=values.device, dtype=torch.bool)
    while bool_mask.ndim < values.ndim:
        bool_mask = bool_mask.unsqueeze(-1)
    bool_mask = bool_mask.expand_as(values)
    return values[bool_mask]


def _compare_tensors(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    a = _masked_tensor_values(left, mask)
    b = _masked_tensor_values(right, mask)
    if int(a.numel()) == 0 or int(b.numel()) == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "rms_abs": 0.0, "max_rel": 0.0}
    diff = (a - b).abs()
    denom = torch.maximum(a.abs(), b.abs()).clamp_min(1e-8)
    rel = diff / denom
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rms_abs": float(diff.pow(2).mean().sqrt().item()),
        "max_rel": float(rel.max().item()),
    }


def _compare_scalar(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    return _compare_tensors(left.reshape(1), right.reshape(1))


def _named_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().float().cpu()
    return grads


def _grad_report(
    eager: dict[str, torch.Tensor],
    flash: dict[str, torch.Tensor],
    *,
    max_rows: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name in sorted(set(eager) & set(flash)):
        e = eager[name]
        f = flash[name]
        if tuple(e.shape) != tuple(f.shape):
            continue
        diff = (e - f).abs()
        e_norm = float(e.norm().item())
        f_norm = float(f.norm().item())
        denom = max(e_norm, f_norm, 1e-12)
        rows.append(
            {
                "name": name,
                "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
                "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
                "eager_norm": e_norm,
                "flash_norm": f_norm,
                "rel_l2": float((e - f).norm().item() / denom),
            }
        )
    rows.sort(key=lambda row: (float(row["rel_l2"]), float(row["max_abs"])), reverse=True)
    return rows[: int(max_rows)]


def _metadata_summary(meta: Any | None) -> dict[str, Any]:
    if meta is None:
        return {"route_hint": None}
    return {
        "route_hint": meta.normalized_route_hint() if hasattr(meta, "normalized_route_hint") else None,
        "seq_lengths_shape": list(meta.seq_lengths.shape)
        if isinstance(meta.seq_lengths, torch.Tensor)
        else None,
        "active_tokens_host": meta.active_tokens_host,
        "doc_num_segments_host": meta.doc_num_segments_host,
        "doc_max_segment_length_host": meta.doc_max_segment_length_host,
        "doc_segment_offsets_shape": list(meta.doc_segment_offsets.shape)
        if isinstance(meta.doc_segment_offsets, torch.Tensor)
        else None,
        "doc_segment_lengths_shape": list(meta.doc_segment_lengths.shape)
        if isinstance(meta.doc_segment_lengths, torch.Tensor)
        else None,
    }


def _first_batch(loader: Any, *, batch_index: int) -> dict[str, Any]:
    iterator = iter(loader)
    batch: dict[str, Any] | None = None
    for _ in range(int(batch_index) + 1):
        batch = next(iterator)
    if batch is None:
        raise RuntimeError("No batch produced.")
    return batch


def _max_doc_segments(batch_cpu: dict[str, Any]) -> int | None:
    doc_ids = batch_cpu.get("doc_ids")
    if not isinstance(doc_ids, torch.Tensor):
        return None
    counts: list[int] = []
    for row in doc_ids:
        active = row[row.ne(0)]
        counts.append(int(torch.unique_consecutive(active).numel()) if int(active.numel()) else 0)
    return max(counts) if counts else 0


def _zero_grad(*models: torch.nn.Module) -> None:
    for model in models:
        model.zero_grad(set_to_none=True)


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for flashdeberta_rtd_compare_step.py")
    device = torch.device("cuda")

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    eager_cfg, mixed_precision, tokenizer, loader = bench.load_tool_config_and_loader(
        str(args.config),
        _model_overrides(attention_impl="eager", docblock_bias_seq_len=int(args.docblock_bias_seq_len)),
        tool_name="flashdeberta_rtd_compare_step.py",
        deterministic_loader=True,
    )
    flash_cfg = load_config(
        args.config,
        overrides=_model_overrides(
            attention_impl="flash", docblock_bias_seq_len=int(args.docblock_bias_seq_len)
        ),
    )
    batch_cpu = _first_batch(loader, batch_index=int(args.batch_index))

    eager_model = _build_model(cfg=eager_cfg, tokenizer=tokenizer, device=device)
    flash_model = _build_model(cfg=flash_cfg, tokenizer=tokenizer, device=device)
    incompatible = flash_model.load_state_dict(eager_model.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"State dict mismatch: {incompatible}")
    _sync_discriminator_embeddings_if_available(flash_model)

    eager_batch, eager_meta = _prepare_batch(
        batch_cpu=batch_cpu,
        device=device,
        cfg=eager_cfg,
        flash_enabled=False,
    )
    flash_batch, flash_meta = _prepare_batch(
        batch_cpu=batch_cpu,
        device=device,
        cfg=flash_cfg,
        flash_enabled=True,
    )
    route_hint = flash_meta.normalized_route_hint() if flash_meta is not None else None
    if route_hint != str(args.flash_route):
        raise RuntimeError(f"Expected flash route {args.flash_route!r}, got {route_hint!r}.")

    active = attention_mask_to_active_tokens(
        input_ids=eager_batch["input_ids"],
        attention_mask=eager_batch.get("attention_mask"),
    )
    report: dict[str, Any] = {
        "config": str(args.config),
        "seed": int(args.seed),
        "sample_seed": int(args.sample_seed),
        "batch_index": int(args.batch_index),
        "mixed_precision": str(mixed_precision),
        "seq_len": int(eager_batch["input_ids"].shape[-1]),
        "batch_size": int(eager_batch["input_ids"].shape[0]),
        "max_doc_segments_per_row": _max_doc_segments(batch_cpu),
        "active_tokens": int(active.sum().item()),
        "masked_tokens": int(eager_batch["labels"].ne(-100).sum().item()),
        "eager_meta": _metadata_summary(eager_meta),
        "flash_meta": _metadata_summary(flash_meta),
    }

    with _autocast_context(str(mixed_precision)):
        eager_hidden, eager_gen_logits, eager_masked_labels, masked_positions = _masked_generator_logits(
            model=eager_model,
            batch=eager_batch,
            flash_meta=eager_meta,
        )
        flash_hidden, flash_gen_logits, flash_masked_labels, _ = _masked_generator_logits(
            model=flash_model,
            batch=flash_batch,
            flash_meta=flash_meta,
        )
        eager_gen_loss = F.cross_entropy(eager_gen_logits.float(), eager_masked_labels)
        flash_gen_loss = F.cross_entropy(flash_gen_logits.float(), flash_masked_labels)

    report["generator_forward"] = {
        "hidden_active": _compare_tensors(eager_hidden, flash_hidden, mask=active),
        "masked_logits": _compare_tensors(eager_gen_logits, flash_gen_logits),
        "loss": _compare_scalar(eager_gen_loss.detach(), flash_gen_loss.detach()),
        "eager_loss": float(eager_gen_loss.detach().float().item()),
        "flash_loss": float(flash_gen_loss.detach().float().item()),
    }
    if not torch.equal(eager_masked_labels, flash_masked_labels):
        raise RuntimeError("Masked labels differ between eager and flash batches.")

    _zero_grad(eager_model, flash_model)
    eager_gen_loss.backward()
    flash_gen_loss.backward()
    report["generator_gradients"] = _grad_report(
        _named_grads(eager_model),
        _named_grads(flash_model),
        max_rows=int(args.max_grad_report),
    )

    _zero_grad(eager_model, flash_model)
    torch.manual_seed(int(args.sample_seed))
    torch.cuda.manual_seed_all(int(args.sample_seed))
    with _autocast_context(str(mixed_precision)):
        eager_gen_phase = eager_model(
            input_ids=eager_batch["input_ids"],
            attention_mask=eager_batch.get("attention_mask"),
            labels=eager_batch["labels"],
            token_type_ids=eager_batch.get("token_type_ids"),
            sampling_temperature=float(eager_cfg.train.sampling_temperature),
            phase="generator",
            flash_meta=eager_meta,
        )
    torch.manual_seed(int(args.sample_seed))
    torch.cuda.manual_seed_all(int(args.sample_seed))
    with _autocast_context(str(mixed_precision)):
        flash_gen_phase = flash_model(
            input_ids=flash_batch["input_ids"],
            attention_mask=flash_batch.get("attention_mask"),
            labels=flash_batch["labels"],
            token_type_ids=flash_batch.get("token_type_ids"),
            sampling_temperature=float(flash_cfg.train.sampling_temperature),
            phase="generator",
            flash_meta=flash_meta,
        )
    sample_mismatch = eager_gen_phase.corrupted_input_ids.ne(flash_gen_phase.corrupted_input_ids)
    label_mismatch = eager_gen_phase.disc_labels.ne(flash_gen_phase.disc_labels)
    report["generator_phase_sampling"] = {
        "loss": _compare_scalar(eager_gen_phase.gen_loss_raw.detach(), flash_gen_phase.gen_loss_raw.detach()),
        "corrupted_mismatch_count": int(sample_mismatch.sum().item()),
        "corrupted_mismatch_frac_active": float(
            (sample_mismatch & active).sum().float().item() / max(float(active.sum().item()), 1.0)
        ),
        "disc_label_mismatch_count": int(label_mismatch.sum().item()),
        "disc_label_mismatch_frac_active": float(
            (label_mismatch & active).sum().float().item() / max(float(active.sum().item()), 1.0)
        ),
        "eager_replaced_frac_active": float(
            (eager_gen_phase.disc_labels.gt(0.5) & active).sum().float().item()
            / max(float(active.sum().item()), 1.0)
        ),
        "flash_replaced_frac_active": float(
            (flash_gen_phase.disc_labels.gt(0.5) & active).sum().float().item()
            / max(float(active.sum().item()), 1.0)
        ),
    }

    # Feed identical discriminator targets into both models so discriminator
    # divergence is not confounded by generator sampling differences.
    corrupted = eager_gen_phase.corrupted_input_ids.detach()
    disc_labels = eager_gen_phase.disc_labels.detach()
    _zero_grad(eager_model, flash_model)
    with _autocast_context(str(mixed_precision)):
        eager_disc_logits, eager_disc_loss, eager_disc_active = _discriminator_logits_and_loss(
            model=eager_model,
            batch=eager_batch,
            flash_meta=eager_meta,
            corrupted_input_ids=corrupted,
            disc_labels=disc_labels,
        )
        flash_disc_logits, flash_disc_loss, flash_disc_active = _discriminator_logits_and_loss(
            model=flash_model,
            batch=flash_batch,
            flash_meta=flash_meta,
            corrupted_input_ids=corrupted,
            disc_labels=disc_labels,
        )
    if not torch.equal(eager_disc_active, flash_disc_active):
        raise RuntimeError("Discriminator active masks differ between eager and flash.")
    report["discriminator_forward_same_targets"] = {
        "logits_active": _compare_tensors(eager_disc_logits, flash_disc_logits, mask=eager_disc_active),
        "loss": _compare_scalar(eager_disc_loss.detach(), flash_disc_loss.detach()),
        "eager_loss": float(eager_disc_loss.detach().float().item()),
        "flash_loss": float(flash_disc_loss.detach().float().item()),
    }
    eager_disc_loss.backward()
    flash_disc_loss.backward()
    report["discriminator_gradients_same_targets"] = _grad_report(
        _named_grads(eager_model),
        _named_grads(flash_model),
        max_rows=int(args.max_grad_report),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
