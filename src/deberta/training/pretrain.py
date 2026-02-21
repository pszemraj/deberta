from __future__ import annotations

import json
import logging
import re
import shutil
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from deberta.config import DataConfig, ModelConfig, TrainConfig
from deberta.data import DebertaV3ElectraCollator, PackedStreamingDataset
from deberta.data.collator import MLMConfig
from deberta.data.loading import load_hf_dataset
from deberta.data.streaming import PackedStreamingConfig
from deberta.modeling import DebertaV3RTDPretrainer, build_backbone_configs, build_backbones

logger = logging.getLogger(__name__)


def _setup_logging(is_main: bool) -> None:
    level = logging.INFO if is_main else logging.WARN
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _maybe_enable_tf32(enabled: bool) -> None:
    if not enabled:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _maybe_fused_adamw_kwargs() -> dict[str, Any]:
    # torch.optim.AdamW supports fused=True on CUDA builds.
    try:
        import inspect

        sig = inspect.signature(torch.optim.AdamW)
        if "fused" in sig.parameters and torch.cuda.is_available():
            return {"fused": True}
    except Exception:
        pass
    return {}


def _build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Create AdamW with sensible parameter grouping.

    - Splits decay vs no_decay (bias + norm weights)
    - Optionally uses a separate generator LR (cfg.generator_learning_rate)
    - Enables fused AdamW if supported on the current CUDA build
    """

    gen_lr = (
        float(cfg.generator_learning_rate)
        if float(cfg.generator_learning_rate) > 0
        else float(cfg.learning_rate)
    )
    disc_lr = float(cfg.learning_rate)

    gen_decay: list[torch.nn.Parameter] = []
    gen_no_decay: list[torch.nn.Parameter] = []
    disc_decay: list[torch.nn.Parameter] = []
    disc_no_decay: list[torch.nn.Parameter] = []

    def _is_no_decay(name: str, p: torch.Tensor) -> bool:
        lname = name.lower()
        if lname.endswith(".bias"):
            return True
        if "layernorm" in lname or "layer_norm" in lname or "rmsnorm" in lname or "rms_norm" in lname:
            return True
        # 1D params are almost always norm scales / biases
        if p.dim() == 1:
            return True
        return False

    def _is_generator(name: str) -> bool:
        return name.startswith("generator.") or name.startswith("generator_lm_head.")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        no_decay = _is_no_decay(name, param)
        is_gen = _is_generator(name)

        if is_gen:
            (gen_no_decay if no_decay else gen_decay).append(param)
        else:
            (disc_no_decay if no_decay else disc_decay).append(param)

    groups: list[dict[str, Any]] = []
    if gen_decay:
        groups.append({"params": gen_decay, "weight_decay": float(cfg.weight_decay), "lr": gen_lr})
    if gen_no_decay:
        groups.append({"params": gen_no_decay, "weight_decay": 0.0, "lr": gen_lr})
    if disc_decay:
        groups.append({"params": disc_decay, "weight_decay": float(cfg.weight_decay), "lr": disc_lr})
    if disc_no_decay:
        groups.append({"params": disc_no_decay, "weight_decay": 0.0, "lr": disc_lr})

    fused_kwargs = _maybe_fused_adamw_kwargs()
    opt = torch.optim.AdamW(
        groups,
        lr=disc_lr,
        betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
        eps=float(cfg.adam_epsilon),
        **fused_kwargs,
    )
    return opt


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    try:
        from transformers import get_scheduler
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required for schedulers.") from e

    return get_scheduler(
        name=cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(cfg.warmup_steps),
        num_training_steps=int(cfg.max_steps),
    )


def _cycle_dataloader(dl: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in dl:
            yield batch


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _parse_checkpoint_step(path: str) -> int:
    name = Path(path).name
    m = re.search(r"checkpoint-(\d+)", name)
    if m:
        return int(m.group(1))
    return 0


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the latest checkpoint-* directory in output_dir, or None.

    This is used to support resume_from_checkpoint="auto".
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


def _rotate_checkpoints(output_dir: Path, *, save_total_limit: int) -> None:
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


def _maybe_tokenize_non_streaming(
    *,
    raw_ds: Any,
    tokenizer: Any,
    data_cfg: DataConfig,
    is_train: bool,
) -> Any:
    """Tokenize + pack a map-style dataset into fixed-length examples."""

    try:
        import datasets
    except Exception as e:  # pragma: no cover
        raise RuntimeError("datasets is required.") from e

    if not isinstance(raw_ds, datasets.Dataset):
        # Already iterable or dict; caller should handle.
        return raw_ds

    text_key = data_cfg.text_column_name
    if text_key not in raw_ds.column_names:
        raise KeyError(f"Text column '{text_key}' not found. Available: {raw_ds.column_names}")

    remove_columns = [c for c in raw_ds.column_names if c != text_key]

    cls_id = int(tokenizer.cls_token_id)
    sep_id = int(tokenizer.sep_token_id)

    def tokenize_fn(examples: dict[str, list[str]]) -> dict[str, Any]:
        texts = examples[text_key]
        # No special tokens: we will add [CLS]/[SEP] after packing.
        return tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

    tokenized = raw_ds.map(
        tokenize_fn,
        batched=True,
        num_proc=int(data_cfg.preprocessing_num_workers),
        remove_columns=remove_columns,
        desc="Tokenizing",
    )

    block_len = int(data_cfg.max_seq_length) - 2
    if block_len <= 0:
        raise ValueError("max_seq_length must be >= 3")

    def group_texts(examples: dict[str, list[list[int]]]) -> dict[str, Any]:
        # Concatenate all texts.
        concatenated: list[int] = []
        for seq in examples["input_ids"]:
            concatenated.extend(seq)
            concatenated.append(sep_id)

        total_length = len(concatenated)
        total_length = (total_length // block_len) * block_len
        if total_length == 0:
            return {"input_ids": [], "attention_mask": [], "special_tokens_mask": []}

        input_ids = []
        attention_mask = []
        special_tokens_mask = []

        for i in range(0, total_length, block_len):
            chunk = concatenated[i : i + block_len]
            ids = [cls_id] + chunk + [sep_id]
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
            chunk_special = [1 if (t == sep_id or t == cls_id) else 0 for t in chunk]
            special_tokens_mask.append([1] + chunk_special + [1])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
        }

    packed = tokenized.map(
        group_texts,
        batched=True,
        num_proc=int(data_cfg.preprocessing_num_workers),
        remove_columns=tokenized.column_names,
        desc="Packing",
    )

    # Drop empty rows (can happen for small datasets)
    packed = packed.filter(lambda x: len(x["input_ids"]) == data_cfg.max_seq_length)

    return packed


def _export_discriminator_hf(
    *,
    accelerator: Any,
    model: DebertaV3RTDPretrainer,
    tokenizer: Any,
    output_dir: Path,
    embedding_sharing: str,
) -> None:
    """Best-effort export of a standalone discriminator model.

    Why "best-effort"?
      - Under FSDP2 + SHARDED_STATE_DICT, gathering a full state dict inside the training process
        can fail depending on accelerate/FSDP state-dict configuration.
      - For a *guaranteed* export from sharded checkpoints, use the `deberta-export` CLI which
        loads the checkpoint and consolidates weights with FULL_STATE_DICT on rank0.

    This function is intentionally lightweight and is safe to keep enabled by default.
    """

    if not accelerator.is_main_process:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoModel
    except Exception as e:
        logger.warning(f"Skipping HF export: transformers import failed: {e}")
        return

    try:
        from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel
    except Exception:
        DebertaRoPEConfig = None  # type: ignore
        DebertaRoPEModel = None  # type: ignore

    try:
        unwrapped = accelerator.unwrap_model(model)

        # Try to gather state dicts via accelerator (preferred for distributed).
        disc_sd = accelerator.get_state_dict(model.discriminator)
        gen_sd = accelerator.get_state_dict(model.generator)

        # Build a fresh model from config.
        if DebertaRoPEConfig is not None and isinstance(
            getattr(unwrapped, "disc_config", None), DebertaRoPEConfig
        ):
            export_disc = DebertaRoPEModel(unwrapped.disc_config)  # type: ignore[arg-type]
        else:
            export_disc = AutoModel.from_config(unwrapped.disc_config)

        # Load as much as possible.
        export_keys = set(export_disc.state_dict().keys())
        filtered_disc_sd = {k: v for k, v in disc_sd.items() if k in export_keys}
        missing = export_disc.load_state_dict(filtered_disc_sd, strict=False)
        if missing.missing_keys:
            logger.info(
                f"HF export: missing keys (often expected with tied embeddings): {missing.missing_keys[:5]}..."
            )

        mode = (embedding_sharing or "none").lower()
        if mode in {"es", "gdes"}:

            def merge_embedding(attr: str) -> None:
                if not hasattr(export_disc, "embeddings") or not hasattr(export_disc.embeddings, attr):
                    return
                gen_key = f"embeddings.{attr}.weight"
                gen_w = gen_sd.get(gen_key)
                if gen_w is None:
                    return

                if mode == "es":
                    merged = gen_w
                else:
                    bias = disc_sd.get(f"embeddings.{attr}.bias")
                    if bias is None:
                        raise RuntimeError(f"discriminator embedding bias not found for '{attr}' (gdes)")
                    merged = gen_w.to(dtype=bias.dtype) + bias

                emb_mod = getattr(export_disc.embeddings, attr)
                if hasattr(emb_mod, "weight") and emb_mod.weight is not None:
                    emb_mod.weight.data.copy_(merged.to(emb_mod.weight.dtype))

            merge_embedding("word_embeddings")
            merge_embedding("position_embeddings")
            merge_embedding("token_type_embeddings")

        tokenizer.save_pretrained(str(output_dir))
        export_disc.save_pretrained(str(output_dir / "discriminator"), safe_serialization=True)
        _json_dump({"embedding_sharing": embedding_sharing}, output_dir / "export_meta.json")
        logger.info(f"Exported discriminator to: {output_dir}")

    except Exception as e:
        logger.warning(
            "HF export failed (common under FSDP2 + SHARDED_STATE_DICT). "
            "Use `deberta-export` after training for a guaranteed consolidation+export. "
            f"Error: {e}"
        )


def run_pretraining(*, model_cfg: ModelConfig, data_cfg: DataConfig, train_cfg: TrainConfig) -> None:
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    # Accelerator first so we know ranks.
    log_with = None if train_cfg.report_to == "none" else train_cfg.report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps, log_with=log_with
    )

    _setup_logging(accelerator.is_main_process)
    _maybe_enable_tf32(train_cfg.tf32)

    logger.info(f"Accelerate state: {accelerator.state}")

    # Make output dir on main.
    output_dir = Path(train_cfg.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        _json_dump(asdict(model_cfg), output_dir / "model_config.json")
        _json_dump(asdict(data_cfg), output_dir / "data_config.json")
        _json_dump(asdict(train_cfg), output_dir / "train_config.json")

    accelerator.wait_for_everyone()

    set_seed(train_cfg.seed, device_specific=True)

    # Tokenizer
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required.") from e

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name_or_path, use_fast=True)

    # Sanity
    if tokenizer.pad_token_id is None:
        # Many DeBERTa tokenizers define PAD.
        raise ValueError("Tokenizer must have pad_token_id.")

    # Data
    raw_train = load_hf_dataset(cfg=data_cfg, split=data_cfg.train_split, streaming=data_cfg.streaming)

    if data_cfg.streaming:
        train_dataset = PackedStreamingDataset(
            hf_dataset=raw_train,
            tokenizer=tokenizer,
            cfg=PackedStreamingConfig(
                text_column_name=data_cfg.text_column_name,
                max_seq_length=data_cfg.max_seq_length,
                seed=train_cfg.seed,
                shuffle_buffer_size=data_cfg.shuffle_buffer_size,
            ),
            process_index=accelerator.process_index,
            num_processes=accelerator.num_processes,
        )
    else:
        train_dataset = _maybe_tokenize_non_streaming(
            raw_ds=raw_train, tokenizer=tokenizer, data_cfg=data_cfg, is_train=True
        )

    collator = DebertaV3ElectraCollator(
        tokenizer=tokenizer,
        cfg=MLMConfig(
            mlm_probability=train_cfg.mlm_probability,
            mask_token_prob=train_cfg.mask_token_prob,
            random_token_prob=train_cfg.random_token_prob,
            max_ngram=train_cfg.mlm_max_ngram,
        ),
    )

    # Dataloader
    num_workers = int(train_cfg.dataloader_num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.per_device_train_batch_size),
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=bool(train_cfg.dataloader_pin_memory),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    # Model
    disc_config, gen_config = build_backbone_configs(
        model_cfg=model_cfg, tokenizer=tokenizer, max_position_embeddings=int(data_cfg.max_seq_length)
    )

    # Instantiate backbones
    disc_backbone, gen_backbone = build_backbones(
        model_cfg=model_cfg,
        disc_config=disc_config,
        gen_config=gen_config,
    )

    # Optional grad checkpointing
    if model_cfg.gradient_checkpointing:
        if hasattr(disc_backbone, "gradient_checkpointing_enable"):
            disc_backbone.gradient_checkpointing_enable()
        if hasattr(gen_backbone, "gradient_checkpointing_enable"):
            gen_backbone.gradient_checkpointing_enable()

    model = DebertaV3RTDPretrainer(
        discriminator_backbone=disc_backbone,
        generator_backbone=gen_backbone,
        disc_config=disc_config,
        gen_config=gen_config,
        embedding_sharing=model_cfg.embedding_sharing,
        tie_generator_word_embeddings=True,
    )

    # Optimizer + scheduler
    optimizer = _build_optimizer(model, train_cfg)
    lr_scheduler = _build_scheduler(optimizer, train_cfg)

    # Prepare (wrap for DDP/FSDP etc)
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # Torch compile (best-effort)
    if train_cfg.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=train_cfg.torch_compile_mode)  # type: ignore[attr-defined]
            logger.info(f"Enabled torch.compile(mode={train_cfg.torch_compile_mode})")
        except Exception as e:
            logger.warning(f"torch.compile failed, continuing without: {e}")

    # Trackers
    if train_cfg.report_to != "none":
        tracker_cfg = {
            "model": asdict(model_cfg),
            "data": asdict(data_cfg),
            "train": asdict(train_cfg),
        }
        accelerator.init_trackers(project_name=train_cfg.run_name or "deberta-pretrain", config=tracker_cfg)

    # Resume
    global_step = 0
    ckpt = train_cfg.resume_from_checkpoint
    if ckpt:
        if str(ckpt).lower() == "auto":
            latest = _find_latest_checkpoint(output_dir)
            if latest is None:
                logger.info(
                    "resume_from_checkpoint=auto but no checkpoint-* dirs found; starting from scratch."
                )
                ckpt = None
            else:
                ckpt = str(latest)
        if ckpt:
            logger.info(f"Resuming from checkpoint: {ckpt}")
            accelerator.load_state(ckpt)
            global_step = _parse_checkpoint_step(ckpt)

    # Training loop
    model.train()

    train_iter = _cycle_dataloader(train_loader)

    # Simple rolling stats
    last_log_time = time.time()

    while global_step < int(train_cfg.max_steps):
        batch = next(train_iter)
        batch = _move_batch_to_device(batch, accelerator.device)

        with accelerator.accumulate(model):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                token_type_ids=batch.get("token_type_ids"),
                sampling_temperature=train_cfg.sampling_temperature,
                gen_loss_weight=train_cfg.gen_loss_weight,
                disc_loss_weight=train_cfg.disc_loss_weight,
                decoupled_loss_scaling=train_cfg.decoupled_loss_scaling,
            )
            loss = out.loss
            accelerator.backward(loss)

            if train_cfg.max_grad_norm and float(train_cfg.max_grad_norm) > 0:
                accelerator.clip_grad_norm_(model.parameters(), float(train_cfg.max_grad_norm))

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # We count *optimizer* steps (not micro-steps).
        if accelerator.sync_gradients:
            global_step += 1

            if train_cfg.logging_steps and (global_step % int(train_cfg.logging_steps) == 0):
                # Reduce scalar metrics across processes.
                def _mean(x: torch.Tensor) -> float:
                    x = x.detach().float().reshape(1)
                    return accelerator.gather(x).mean().item()

                lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else float("nan")
                metrics = {
                    "step": global_step,
                    "lr": lr,
                    "loss": _mean(loss),
                    "gen_loss": _mean(out.gen_loss),
                    "disc_loss": _mean(out.disc_loss),
                    "disc_acc": _mean(out.disc_accuracy),
                }

                now = time.time()
                dt = max(1e-6, now - last_log_time)
                last_log_time = now

                if accelerator.is_main_process:
                    logger.info(
                        " | ".join(
                            [
                                f"step={metrics['step']}",
                                f"lr={metrics['lr']:.3e}",
                                f"loss={metrics['loss']:.4f}",
                                f"gen={metrics['gen_loss']:.4f}",
                                f"disc={metrics['disc_loss']:.4f}",
                                f"acc={metrics['disc_acc']:.4f}",
                            ]
                        )
                    )

                if train_cfg.report_to != "none":
                    accelerator.log({k: v for k, v in metrics.items() if k != "step"}, step=global_step)

            # Checkpoint
            if train_cfg.save_steps and (global_step % int(train_cfg.save_steps) == 0):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    accelerator.save_state(str(ckpt_dir))
                    _rotate_checkpoints(output_dir, save_total_limit=int(train_cfg.save_total_limit))
                    logger.info(f"Saved checkpoint: {ckpt_dir}")

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_ckpt = output_dir / f"checkpoint-{global_step}"
        final_ckpt.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(str(final_ckpt))
        _rotate_checkpoints(output_dir, save_total_limit=int(train_cfg.save_total_limit))
        logger.info(f"Saved final checkpoint: {final_ckpt}")

    # Best-effort HF export
    if train_cfg.export_hf_final:
        accelerator.wait_for_everyone()
        _export_discriminator_hf(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir / "final_hf",
            embedding_sharing=model_cfg.embedding_sharing,
        )

    if train_cfg.report_to != "none":
        accelerator.end_training()
