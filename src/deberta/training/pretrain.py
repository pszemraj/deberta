"""Training loop and export utilities for DeBERTa v3 RTD pretraining."""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from deberta.config import (
    RUN_CONFIG_SCHEMA_VERSION,
    DataConfig,
    ModelConfig,
    TrainConfig,
    _normalize_sdpa_kernel,
    _normalize_torch_compile_mode,
    normalize_mixed_precision,
    validate_data_config,
    validate_model_config,
    validate_train_config,
    validate_training_workflow_options,
)
from deberta.data import DebertaV3ElectraCollator, PackedStreamingDataset, SequentialStreamingDataset
from deberta.data.collator import MLMConfig
from deberta.data.loading import load_hf_dataset
from deberta.data.streaming import PackedStreamingConfig
from deberta.log_utils import setup_process_logging
from deberta.modeling import DebertaV3RTDPretrainer, build_backbone_configs, build_backbones
from deberta.modeling.export_utils import load_intersection_state_dict, merge_embeddings_into_export_backbone
from deberta.modeling.rtd import attention_mask_to_active_tokens, compute_generator_loss_term

logger = logging.getLogger(__name__)


def _json_dump(obj: Any, path: Path) -> None:
    """Write JSON to disk with stable formatting.

    :param Any obj: Serializable object.
    :param Path path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _json_load(path: Path) -> dict[str, Any]:
    """Read JSON mapping from disk.

    :param Path path: Source path.
    :return dict[str, Any]: Parsed mapping.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(raw).__name__}.")
    return raw


def _build_run_metadata() -> dict[str, Any]:
    """Build run-metadata payload stored alongside config snapshots.

    :return dict[str, Any]: Metadata mapping.
    """
    from deberta import __version__

    return {
        "config_schema_version": int(RUN_CONFIG_SCHEMA_VERSION),
        "deberta_version": str(__version__),
    }


def _validate_run_metadata(path: Path) -> None:
    """Validate on-disk run metadata schema compatibility.

    :param Path path: Metadata file path.
    :raises ValueError: If metadata is malformed or schema-incompatible.
    """
    raw = _json_load(path)
    if "config_schema_version" not in raw:
        raise ValueError(
            f"run metadata missing `config_schema_version` at {path}. "
            "Refusing resume/export with ambiguous config schema."
        )

    try:
        schema_version = int(raw["config_schema_version"])
    except Exception as e:
        raise ValueError(f"Invalid config_schema_version in {path}: {raw['config_schema_version']!r}") from e

    if schema_version != int(RUN_CONFIG_SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported run metadata schema at {path}: {schema_version}. "
            f"Expected {int(RUN_CONFIG_SCHEMA_VERSION)}."
        )


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
    :return str | None: Concrete checkpoint path, or ``None``.
    """
    if not resume_from_checkpoint:
        return None

    if str(resume_from_checkpoint).lower() != "auto":
        return str(resume_from_checkpoint)

    latest = _find_latest_checkpoint(output_dir)
    if latest is None:
        if is_main_process:
            logger.info("resume_from_checkpoint=auto but no checkpoint-* dirs found; starting from scratch.")
        return None
    return str(latest)


def _persist_or_validate_run_configs(
    *,
    output_dir: Path,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    resume_checkpoint: str | None,
    is_main_process: bool,
) -> None:
    """Persist new config snapshots or validate existing snapshots on resume.

    :param Path output_dir: Training output directory.
    :param ModelConfig model_cfg: Current model config.
    :param DataConfig data_cfg: Current data config.
    :param TrainConfig train_cfg: Current train config.
    :param str | None resume_checkpoint: Resolved checkpoint path, if resuming.
    :param bool is_main_process: Whether this process owns writes.
    :raises ValueError: If resume mode detects incompatible model/data config snapshots.
    """
    model_cfg_path = output_dir / "model_config.json"
    data_cfg_path = output_dir / "data_config.json"
    train_cfg_path = output_dir / "train_config.json"
    run_meta_path = output_dir / "run_metadata.json"

    has_saved_model_data = model_cfg_path.exists() and data_cfg_path.exists()
    if resume_checkpoint is not None and has_saved_model_data:
        if run_meta_path.exists():
            _validate_run_metadata(run_meta_path)
        elif is_main_process:
            # Backfill schema metadata for older runs once compatibility has been checked.
            _json_dump(_build_run_metadata(), run_meta_path)

        saved_model_cfg = ModelConfig(**_json_load(model_cfg_path))
        saved_data_cfg = DataConfig(**_json_load(data_cfg_path))
        validate_model_config(saved_model_cfg)
        validate_data_config(saved_data_cfg)

        if asdict(saved_model_cfg) != asdict(model_cfg):
            raise ValueError(
                "Resume configuration mismatch for model_config.json. "
                "Refusing to overwrite run metadata with incompatible model settings."
            )
        if asdict(saved_data_cfg) != asdict(data_cfg):
            raise ValueError(
                "Resume configuration mismatch for data_config.json. "
                "Refusing to overwrite run metadata with incompatible data settings."
            )
        if is_main_process:
            logger.info("Resume mode: preserving existing model/data/train config snapshots in output_dir.")
        return

    if is_main_process:
        _json_dump(asdict(model_cfg), model_cfg_path)
        _json_dump(asdict(data_cfg), data_cfg_path)
        _json_dump(asdict(train_cfg), train_cfg_path)
        _json_dump(_build_run_metadata(), run_meta_path)


def _maybe_enable_tf32(enabled: bool, *, force_legacy: bool = False) -> None:
    """Configure TF32 compute policy for CUDA matmul/cudnn.

    :param bool enabled: Whether to enable TF32.
    :param bool force_legacy: Whether to force legacy ``allow_tf32`` flags.
    """
    if force_legacy:
        torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
        torch.backends.cudnn.allow_tf32 = bool(enabled)
        return

    # Prefer the modern fp32_precision API when available (PyTorch 2.9+).
    # Fallback to allow_tf32 flags on older builds.
    target = "tf32" if enabled else "ieee"
    configured = False

    try:
        if hasattr(torch.backends, "fp32_precision"):
            torch.backends.fp32_precision = target
            configured = True
    except Exception:
        pass

    try:
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = target
            configured = True
    except Exception:
        pass

    try:
        if hasattr(torch.backends.cudnn, "fp32_precision"):
            torch.backends.cudnn.fp32_precision = target
            configured = True
    except Exception:
        pass

    # Granular cudnn knobs on newer builds.
    try:
        if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = target
            configured = True
        if hasattr(torch.backends.cudnn, "rnn") and hasattr(torch.backends.cudnn.rnn, "fp32_precision"):
            torch.backends.cudnn.rnn.fp32_precision = target
            configured = True
    except Exception:
        pass

    if configured:
        return

    # Legacy fallback.
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)


def _maybe_configure_sdpa_kernels(policy: str, *, is_main: bool) -> None:
    """Configure PyTorch SDPA backend toggles on CUDA.

    :param str policy: SDPA kernel policy.
    :param bool is_main: Whether current process should emit logs.
    """
    if not torch.cuda.is_available():
        return

    policy = _normalize_sdpa_kernel(policy)

    enable_flash = True
    enable_mem_efficient = True
    enable_math = True

    if policy == "flash_only":
        enable_mem_efficient = False
        enable_math = False
    elif policy == "mem_efficient":
        enable_flash = False
    elif policy == "math":
        enable_flash = False
        enable_mem_efficient = False

    try:
        cuda_backend = getattr(torch.backends, "cuda", None)
        if cuda_backend is not None:
            for name, enabled in (
                ("enable_flash_sdp", enable_flash),
                ("enable_mem_efficient_sdp", enable_mem_efficient),
                ("enable_math_sdp", enable_math),
            ):
                fn = getattr(cuda_backend, name, None)
                if callable(fn):
                    fn(bool(enabled))
    except Exception:
        # Best-effort only; do not fail training if this backend API changes.
        pass

    if is_main:
        logger.info(
            "SDPA kernel policy=%s (requested flash=%s, mem_efficient=%s, math=%s).",
            policy,
            enable_flash,
            enable_mem_efficient,
            enable_math,
        )


def _bf16_runtime_sanity_check() -> bool:
    """Check whether bf16 autocast executes a tiny CUDA matmul.

    :return bool: True when a tiny bf16 autocast path succeeds.
    """
    if not torch.cuda.is_available():
        logger.warning(
            "bf16 mixed precision requested but CUDA is not available; falling back to full precision."
        )
        return False
    if not torch.cuda.is_bf16_supported():
        logger.warning(
            "bf16 mixed precision requested but this CUDA device reports no bf16 support; "
            "falling back to full precision."
        )
        return False

    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            a = torch.randn((64, 64), device="cuda", dtype=torch.float32)
            b = torch.randn((64, 64), device="cuda", dtype=torch.float32)
            c = a @ b
            _ = c.sum().item()
        return True
    except Exception as e:
        logger.warning(f"bf16 autocast preflight failed; falling back to full precision. Error: {e}")
        return False


def _maybe_cudagraph_mark_step_begin() -> None:
    """Mark cudagraph step boundaries when the API is available.

    This is a no-op on PyTorch builds that do not expose ``torch.compiler``
    or ``cudagraph_mark_step_begin``.
    """
    try:
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
    except Exception:
        # Best-effort: keep training running if backend API shape changes.
        pass


def _should_force_legacy_tf32_for_compile(*, torch_compile: bool, compile_mode: str) -> bool:
    """Return whether TF32 should use legacy flags for compile compatibility.

    :param bool torch_compile: Whether ``torch.compile`` is enabled.
    :param str compile_mode: Canonical compile mode.
    :return bool: True when legacy TF32 flags are preferred.
    """
    if not torch_compile:
        return False
    # On some PyTorch builds, max-autotune paths still query legacy allow_tf32
    # and can error if only the new fp32_precision API has been configured.
    return compile_mode.startswith("max-autotune")


def _maybe_fused_adamw_kwargs() -> dict[str, Any]:
    """Return optimizer kwargs enabling fused AdamW when available.

    :return dict[str, Any]: Optional kwargs for ``torch.optim.AdamW``.
    """
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
    """Create AdamW with parameter grouping for RTD training.

    :param torch.nn.Module model: RTD model.
    :param TrainConfig cfg: Training configuration.
    :return torch.optim.Optimizer: Configured AdamW optimizer.
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
        """Check whether parameter should skip weight decay.

        :param str name: Parameter name.
        :param torch.Tensor p: Parameter tensor.
        :return bool: True when parameter belongs to no-decay group.
        """
        lname = name.lower()
        # Keep vector/scalar biases in no-decay; high-rank "bias" tensors (for example
        # GDES embedding deltas) should follow standard weight-decay behavior.
        if lname.endswith(".bias") and p.dim() <= 1:
            return True
        if "layernorm" in lname or "layer_norm" in lname or "rmsnorm" in lname or "rms_norm" in lname:
            return True
        # Scalars and 1D params are typically excluded from decay.
        if p.dim() <= 1:
            return True
        return False

    def _is_generator(name: str) -> bool:
        """Check whether parameter belongs to the generator branch.

        :param str name: Parameter name.
        :return bool: True when parameter is generator-owned.
        """
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


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> Any:
    """Build a Hugging Face learning-rate scheduler.

    :param torch.optim.Optimizer optimizer: Optimizer instance.
    :param TrainConfig cfg: Training configuration.
    :return Any: Scheduler object from ``transformers.get_scheduler``.
    """
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
    """Yield batches forever by cycling through a dataloader.

    :param DataLoader dl: Source dataloader.
    :return Iterator[dict[str, torch.Tensor]]: Infinite batch iterator.
    """
    epoch = 0
    dataset = getattr(dl, "dataset", None)
    while True:
        set_epoch = getattr(dataset, "set_epoch", None)
        if callable(set_epoch):
            try:
                set_epoch(epoch)
            except Exception:
                pass
        yield from dl
        epoch += 1


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move all batch tensors onto a device.

    :param dict[str, torch.Tensor] batch: Tensor batch mapping.
    :param torch.device device: Destination device.
    :return dict[str, torch.Tensor]: Batch placed on ``device``.
    """
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _build_training_collator(
    *,
    tokenizer: Any,
    train_cfg: TrainConfig,
    packed_sequences: bool,
    block_cross_document_attention: bool,
) -> DebertaV3ElectraCollator:
    """Build the RTD masking collator from train/data config.

    :param Any tokenizer: Tokenizer used for dynamic masking.
    :param TrainConfig train_cfg: Training configuration.
    :param bool packed_sequences: Whether dataset packing is enabled.
    :param bool block_cross_document_attention: Whether packed batches should block cross-document attention.
    :return DebertaV3ElectraCollator: Configured collator.
    """
    return DebertaV3ElectraCollator(
        tokenizer=tokenizer,
        cfg=MLMConfig(
            mlm_probability=train_cfg.mlm_probability,
            mask_token_prob=train_cfg.mask_token_prob,
            random_token_prob=train_cfg.random_token_prob,
            max_ngram=train_cfg.mlm_max_ngram,
        ),
        packed_sequences=bool(packed_sequences),
        block_cross_document_attention=bool(block_cross_document_attention),
    )


def _compute_disc_active_mask(
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    special_token_ids: tuple[int, ...],
    pad_token_id: int | None,
) -> torch.Tensor:
    """Compute discriminator-active token mask before model forward.

    This mirrors the RTD discriminator masking semantics while accounting for
    masked positions that will be replaced before discriminator scoring.

    :param torch.Tensor input_ids: Input token ids.
    :param torch.Tensor labels: MLM labels (-100 for non-masked positions).
    :param torch.Tensor | None attention_mask: Optional attention mask.
    :param tuple[int, ...] special_token_ids: Token ids excluded from discriminator loss.
    :param int | None pad_token_id: Padding token id.
    :return torch.Tensor: Boolean active-token mask.
    """
    active = attention_mask_to_active_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
    )

    if not special_token_ids:
        return active

    special = torch.isin(
        input_ids,
        torch.tensor(
            sorted(int(sid) for sid in special_token_ids), device=input_ids.device, dtype=input_ids.dtype
        ),
    )

    # Masked positions are replaced before discriminator scoring and should not
    # be excluded solely because their pre-corruption token may be special.
    masked_positions = labels.ne(-100)
    special = special & (~masked_positions)
    return active & (~special)


def _count_rtd_tokens_for_batch(
    batch: dict[str, torch.Tensor],
    *,
    special_token_ids: tuple[int, ...],
    pad_token_id: int | None,
) -> tuple[float, float]:
    """Return generator/discriminator active-token counts for one microbatch.

    :param dict[str, torch.Tensor] batch: Microbatch tensors.
    :param tuple[int, ...] special_token_ids: Token ids excluded from discriminator loss.
    :param int | None pad_token_id: Padding token id.
    :return tuple[float, float]: (generator_count, discriminator_count).
    """
    labels = batch["labels"]
    gen_count = float(labels.ne(-100).sum().item())
    disc_active = _compute_disc_active_mask(
        input_ids=batch["input_ids"],
        labels=labels,
        attention_mask=batch.get("attention_mask"),
        special_token_ids=special_token_ids,
        pad_token_id=pad_token_id,
    )
    disc_count = float(disc_active.sum().item())
    return gen_count, disc_count


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
    decoupled_loss_scaling: bool,
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
    :param bool decoupled_loss_scaling: Whether to use DeBERTa-style decoupled scaling.
    :return torch.Tensor: Unscaled microbatch objective contribution.
    """
    gen_scale = float(gen_count) / max(float(gen_window_tokens_per_rank), 1.0)
    disc_scale = float(disc_count) / max(float(disc_window_tokens_per_rank), 1.0)

    gen_term = compute_generator_loss_term(
        gen_loss=gen_loss,
        disc_loss=disc_loss,
        decoupled_loss_scaling=bool(decoupled_loss_scaling),
    )

    return float(gen_loss_weight) * gen_scale * gen_term + float(disc_loss_weight) * disc_scale * disc_loss


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
        # Token-weighted path already accumulates normalized micro objectives.
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
    if not token_weighted_ga:
        return loss
    return loss * float(max(1, int(ga_steps)))


def _should_clip_gradients(*, sync_gradients: bool, max_grad_norm: float | int | None) -> bool:
    """Return whether gradient clipping should run for this micro-step.

    :param bool sync_gradients: Whether gradients are synchronized this step.
    :param float | int | None max_grad_norm: Configured clipping norm.
    :return bool: ``True`` when clipping should be applied.
    """
    if not bool(sync_gradients):
        return False
    if max_grad_norm is None:
        return False
    return float(max_grad_norm) > 0.0


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


def _load_checkpoint_data_progress(checkpoint_dir: Path) -> int | None:
    """Load persisted data progress for resume alignment.

    :param Path checkpoint_dir: Checkpoint directory.
    :return int | None: Consumed micro-batch count, or ``None`` if unavailable.
    """
    path = checkpoint_dir / "data_state.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        val = raw.get("consumed_micro_batches", None)
        if val is None:
            return None
        return max(0, int(val))
    except Exception:
        return None


def _save_checkpoint_data_progress(*, checkpoint_dir: Path, consumed_micro_batches: int) -> None:
    """Persist data iterator progress next to a checkpoint.

    :param Path checkpoint_dir: Checkpoint directory.
    :param int consumed_micro_batches: Number of consumed micro-batches.
    """
    _json_dump(
        {"consumed_micro_batches": int(max(0, consumed_micro_batches))},
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
) -> None:
    """Save one training checkpoint with collective state-dict write.

    :param Any accelerator: Accelerate runtime object.
    :param Path checkpoint_dir: Destination checkpoint directory.
    :param Path output_dir: Parent output directory for checkpoint rotation.
    :param int consumed_micro_batches: Data progress to persist.
    :param int save_total_limit: Number of checkpoints to retain.
    :param str log_label: Logging label for this save.
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
      - For a *guaranteed* export from sharded checkpoints, use `deberta export` which
        loads the checkpoint and consolidates weights with FULL_STATE_DICT on rank0.

    This function is intentionally lightweight and is safe to keep enabled by default.

    :param Any accelerator: Accelerate runtime object.
    :param DebertaV3RTDPretrainer model: Wrapped RTD pretrainer.
    :param Any tokenizer: Tokenizer to export.
    :param Path output_dir: Export destination directory.
    :param str embedding_sharing: Sharing mode used during training.
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
        disc_mod = getattr(unwrapped, "discriminator", None)
        gen_mod = getattr(unwrapped, "generator", None)
        if disc_mod is None or gen_mod is None:
            raise RuntimeError(
                "Unwrapped RTD model must expose discriminator and generator modules for export."
            )
        disc_sd = accelerator.get_state_dict(disc_mod)
        gen_sd = accelerator.get_state_dict(gen_mod)

        # Build a fresh model from config.
        if DebertaRoPEConfig is not None and isinstance(
            getattr(unwrapped, "disc_config", None), DebertaRoPEConfig
        ):
            export_disc = DebertaRoPEModel(unwrapped.disc_config)  # type: ignore[arg-type]
        else:
            export_disc = AutoModel.from_config(unwrapped.disc_config)

        # Load overlap keys only to tolerate training/export module-shape differences.
        missing = load_intersection_state_dict(export_disc, disc_sd)
        if missing.missing_keys:
            logger.info(
                f"HF export: missing keys (often expected with tied embeddings): {missing.missing_keys[:5]}..."
            )

        mode = (embedding_sharing or "none").lower()
        merge_embeddings_into_export_backbone(
            export_model=export_disc,
            disc_sd=disc_sd,
            gen_sd=gen_sd,
            mode=mode,
            fp32_accumulate=False,
        )

        tokenizer.save_pretrained(str(output_dir))
        export_disc.save_pretrained(str(output_dir / "discriminator"), safe_serialization=True)
        _json_dump({"embedding_sharing": embedding_sharing}, output_dir / "export_meta.json")
        logger.info(f"Exported discriminator to: {output_dir}")

    except Exception as e:
        logger.warning(
            "HF export failed (common under FSDP2 + SHARDED_STATE_DICT). "
            "Use `deberta export` after training for a guaranteed consolidation+export. "
            f"Error: {e}"
        )


def run_pretraining(*, model_cfg: ModelConfig, data_cfg: DataConfig, train_cfg: TrainConfig) -> None:
    """Run RTD pretraining with Accelerate/FSDP2-compatible plumbing.

    :param ModelConfig model_cfg: Model configuration.
    :param DataConfig data_cfg: Data configuration.
    :param TrainConfig train_cfg: Training configuration.
    """
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    # Accelerator first so we know ranks.
    log_with = None if train_cfg.report_to == "none" else train_cfg.report_to
    mixed_precision = normalize_mixed_precision(train_cfg.mixed_precision)
    compile_mode = _normalize_torch_compile_mode(train_cfg.torch_compile_mode)
    if mixed_precision == "bf16" and not _bf16_runtime_sanity_check():
        mixed_precision = "no"
    # Keep persisted config/tracker snapshots aligned with the effective runtime mode.
    train_cfg.mixed_precision = mixed_precision
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        log_with=log_with,
        mixed_precision=mixed_precision,
    )

    setup_process_logging(accelerator.is_main_process)
    # Validate config contract up-front before side effects (filesystem/network/model loading).
    validate_model_config(model_cfg)
    validate_data_config(data_cfg)
    validate_train_config(train_cfg)
    validate_training_workflow_options(data_cfg=data_cfg, train_cfg=train_cfg, model_cfg=model_cfg)

    force_legacy_tf32 = _should_force_legacy_tf32_for_compile(
        torch_compile=bool(train_cfg.torch_compile),
        compile_mode=compile_mode,
    )
    if force_legacy_tf32 and accelerator.is_main_process:
        logger.info(
            "Using legacy TF32 backend flags for torch.compile max-autotune mode "
            "to avoid known TF32 API conflicts on some PyTorch builds."
        )
    _maybe_enable_tf32(train_cfg.tf32, force_legacy=force_legacy_tf32)
    _maybe_configure_sdpa_kernels(str(train_cfg.sdpa_kernel), is_main=accelerator.is_main_process)

    logger.info(f"Accelerate state: {accelerator.state}")

    # Make/validate output dir on main.
    output_dir = Path(train_cfg.output_dir)
    _prepare_output_dir(
        output_dir=output_dir,
        overwrite_output_dir=bool(train_cfg.overwrite_output_dir),
        resume_from_checkpoint=train_cfg.resume_from_checkpoint,
        is_main_process=accelerator.is_main_process,
    )
    ckpt = _resolve_resume_checkpoint(
        output_dir=output_dir,
        resume_from_checkpoint=train_cfg.resume_from_checkpoint,
        is_main_process=accelerator.is_main_process,
    )
    _persist_or_validate_run_configs(
        output_dir=output_dir,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        resume_checkpoint=ckpt,
        is_main_process=accelerator.is_main_process,
    )

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

    dataset_cls = PackedStreamingDataset if bool(data_cfg.pack_sequences) else SequentialStreamingDataset
    train_dataset = dataset_cls(
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

    collator = _build_training_collator(
        tokenizer=tokenizer,
        train_cfg=train_cfg,
        packed_sequences=bool(data_cfg.pack_sequences),
        block_cross_document_attention=bool(data_cfg.block_cross_document_attention),
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
    compile_enabled = bool(train_cfg.torch_compile and hasattr(torch, "compile"))
    if compile_enabled:
        try:
            model = torch.compile(model, mode=compile_mode)  # type: ignore[attr-defined]
            logger.info(f"Enabled torch.compile(mode={compile_mode})")
        except Exception as e:
            logger.warning(f"torch.compile failed, continuing without: {e}")
            compile_enabled = False
    elif train_cfg.torch_compile:
        logger.warning(
            "torch.compile requested but this torch build does not expose torch.compile; continuing."
        )

    # Trackers
    if train_cfg.report_to != "none":
        tracker_cfg = {
            "model": asdict(model_cfg),
            "data": asdict(data_cfg),
            "train": asdict(train_cfg),
        }
        accelerator.init_trackers(project_name=train_cfg.run_name or "deberta-train", config=tracker_cfg)

    # Resume
    global_step = 0
    consumed_micro_batches = 0
    if ckpt:
        logger.info(f"Resuming from checkpoint: {ckpt}")
        accelerator.load_state(ckpt)
        global_step = _parse_checkpoint_step(ckpt)
        restored = _load_checkpoint_data_progress(Path(ckpt))
        if restored is None:
            # Back-compat fallback for older checkpoints without data_state.json.
            restored = global_step * int(train_cfg.gradient_accumulation_steps)
            logger.warning(
                "Checkpoint missing data_state.json; approximating data replay offset "
                f"as global_step * grad_accum = {restored} micro-batches."
            )
        consumed_micro_batches = int(restored)

    # Training loop
    model.train()

    train_iter = _cycle_dataloader(train_loader)
    ga_steps = int(train_cfg.gradient_accumulation_steps)
    token_weighted_ga = bool(train_cfg.token_weighted_gradient_accumulation)
    unwrapped_model = accelerator.unwrap_model(model)
    special_token_ids = tuple(
        sorted(int(sid) for sid in getattr(unwrapped_model, "_forbidden_sample_token_ids", set()))
    )
    disc_pad_token_id = getattr(getattr(unwrapped_model, "disc_config", None), "pad_token_id", None)
    if disc_pad_token_id is not None:
        disc_pad_token_id = int(disc_pad_token_id)

    if consumed_micro_batches > 0:
        logger.info(
            "Replaying data iterator by %d micro-batches to align resume data position.",
            consumed_micro_batches,
        )
        for _ in range(consumed_micro_batches):
            _ = next(train_iter)

    while global_step < int(train_cfg.max_steps):
        window: list[tuple[dict[str, torch.Tensor], float, float]] = []
        local_gen_tokens = 0.0
        local_disc_tokens = 0.0

        for _ in range(ga_steps):
            batch = next(train_iter)
            consumed_micro_batches += 1
            if token_weighted_ga:
                gen_count, disc_count = _count_rtd_tokens_for_batch(
                    batch,
                    special_token_ids=special_token_ids,
                    pad_token_id=disc_pad_token_id,
                )
                local_gen_tokens += gen_count
                local_disc_tokens += disc_count
            else:
                gen_count, disc_count = 0.0, 0.0
            window.append((batch, gen_count, disc_count))

        if token_weighted_ga:
            local_totals = torch.tensor(
                [local_gen_tokens, local_disc_tokens],
                device=accelerator.device,
                dtype=torch.float32,
            )
            # DDP/FSDP gradients are averaged across ranks. Mean token totals per rank
            # preserve global token-normalized scaling when used with local micro counts.
            mean_totals = accelerator.reduce(local_totals, reduction="mean")
            gen_window_tokens_per_rank = max(float(mean_totals[0].item()), 1.0)
            disc_window_tokens_per_rank = max(float(mean_totals[1].item()), 1.0)
        else:
            gen_window_tokens_per_rank = 1.0
            disc_window_tokens_per_rank = 1.0

        out = None
        loss_for_metrics = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        did_sync_step = False

        for step_idx, (batch, gen_count, disc_count) in enumerate(window):
            batch = _move_batch_to_device(batch, accelerator.device)
            if compile_enabled:
                _maybe_cudagraph_mark_step_begin()

            is_sync_step = step_idx == (ga_steps - 1)
            sync_ctx = nullcontext() if is_sync_step else accelerator.no_sync(model)

            with sync_ctx:
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                    token_type_ids=batch.get("token_type_ids"),
                    sampling_temperature=train_cfg.sampling_temperature,
                    gen_loss_weight=train_cfg.gen_loss_weight,
                    disc_loss_weight=train_cfg.disc_loss_weight,
                    decoupled_loss_scaling=train_cfg.decoupled_loss_scaling,
                )

                if token_weighted_ga:
                    micro_obj = _token_weighted_micro_objective(
                        gen_loss=out.gen_loss_raw,
                        disc_loss=out.disc_loss_raw,
                        gen_count=gen_count,
                        disc_count=disc_count,
                        gen_window_tokens_per_rank=gen_window_tokens_per_rank,
                        disc_window_tokens_per_rank=disc_window_tokens_per_rank,
                        gen_loss_weight=float(train_cfg.gen_loss_weight),
                        disc_loss_weight=float(train_cfg.disc_loss_weight),
                        decoupled_loss_scaling=bool(train_cfg.decoupled_loss_scaling),
                    )
                    loss = micro_obj
                    loss_for_metrics = loss_for_metrics + micro_obj.detach()
                else:
                    loss = out.loss
                    loss_for_metrics = loss_for_metrics + out.loss.detach()

                backward_loss = _scale_loss_for_backward(
                    loss=loss,
                    ga_steps=ga_steps,
                    token_weighted_ga=token_weighted_ga,
                )
                accelerator.backward(backward_loss)

            if is_sync_step:
                did_sync_step = True
                if _should_clip_gradients(
                    sync_gradients=True,
                    max_grad_norm=train_cfg.max_grad_norm,
                ):
                    accelerator.clip_grad_norm_(model.parameters(), float(train_cfg.max_grad_norm))

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if not did_sync_step:
            raise RuntimeError("Accumulation window produced no synchronized optimization step.")

        loss_for_metrics = _finalize_window_metric_loss(
            accumulated_loss=loss_for_metrics,
            ga_steps=ga_steps,
            token_weighted_ga=token_weighted_ga,
        )

        # We count *optimizer* steps (not micro-steps).
        if did_sync_step:
            if out is None:
                raise RuntimeError("Accumulation window produced no forward pass outputs.")
            global_step += 1

            if train_cfg.logging_steps and (global_step % int(train_cfg.logging_steps) == 0):
                # Reduce scalar metrics across processes.
                def _mean(x: torch.Tensor) -> float:
                    """Compute global mean scalar across processes.

                    :param torch.Tensor x: Scalar-like tensor.
                    :return float: Process-aggregated mean value.
                    """
                    x = x.detach().float().reshape(1)
                    return accelerator.gather(x).mean().item()

                lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else float("nan")
                metrics = {
                    "step": global_step,
                    "lr": lr,
                    "loss": _mean(loss_for_metrics),
                    "gen_loss": _mean(out.gen_loss),
                    "disc_loss": _mean(out.disc_loss),
                    "disc_acc": _mean(out.disc_accuracy),
                    "gen_tokens": _mean(out.gen_token_count),
                    "disc_tokens": _mean(out.disc_token_count),
                }

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
                                f"gen_tok={metrics['gen_tokens']:.1f}",
                                f"disc_tok={metrics['disc_tokens']:.1f}",
                            ]
                        )
                    )

                if train_cfg.report_to != "none":
                    accelerator.log({k: v for k, v in metrics.items() if k != "step"}, step=global_step)

            # Checkpoint
            if train_cfg.save_steps and (global_step % int(train_cfg.save_steps) == 0):
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                _save_training_checkpoint(
                    accelerator=accelerator,
                    checkpoint_dir=ckpt_dir,
                    output_dir=output_dir,
                    consumed_micro_batches=consumed_micro_batches,
                    save_total_limit=int(train_cfg.save_total_limit),
                    log_label="periodic",
                )

    # Final save
    final_ckpt = output_dir / f"checkpoint-{global_step}"
    _save_training_checkpoint(
        accelerator=accelerator,
        checkpoint_dir=final_ckpt,
        output_dir=output_dir,
        consumed_micro_batches=consumed_micro_batches,
        save_total_limit=int(train_cfg.save_total_limit),
        log_label="final",
    )

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
