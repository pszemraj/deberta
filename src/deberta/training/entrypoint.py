"""Training loop entrypoints for DeBERTa v3 RTD pretraining."""

from __future__ import annotations

import gc
import inspect
import logging
import math
import time
from collections.abc import Iterable, Iterator
from contextlib import nullcontext, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deberta.config import (
    Config,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    _normalize_torch_compile_mode,
    resolve_effective_mixed_precision,
)
from deberta.data.loading import load_hf_dataset
from deberta.modeling import DebertaV3RTDPretrainer, build_backbone_configs, build_backbones
from deberta.modeling.flashdeberta_op_utils import is_flash_attention_impl
from deberta.run_artifacts import (
    load_materialized_backbone_configs,
    materialized_tokenizer_path,
    persist_materialized_run_artifacts,
)
from deberta.run_layout import infer_run_dir_from_checkpoint
from deberta.training.checkpointing import _resolve_data_resume_policy, _save_periodic_checkpoint_if_due
from deberta.training.compile import (
    _bf16_runtime_sanity_check,
    _compile_backbones_for_scope,
    _dtype_for_mixed_precision,
    _maybe_configure_sdpa_kernels,
    _maybe_cudagraph_mark_step_begin,
    _maybe_enable_tf32,
    _prefill_rotary_caches_for_compile,
    _resolve_effective_compile_scope,
    _stabilize_compile_attention_mask,
    prepare_flash_attention_batch_metadata,
)
from deberta.training.export_helpers import _export_discriminator_hf_subprocess
from deberta.training.loop_utils import (
    _count_input_tokens_for_batch,
    _finalize_window_metric_loss,
    _scale_loss_for_backward,
    _should_clip_gradients,
    _sum_local_scalar,
    _token_weighted_micro_objective,
)
from deberta.training.metrics import (
    _append_metrics_jsonl_row,
    _build_runtime_resolved_tracker_config,
    _flush_loggers,
    _write_nonfinite_debug_artifact,
)
from deberta.training.run_config import _persist_or_validate_run_configs
from deberta.training.run_management import (
    _load_checkpoint_progress_metadata,
    _parse_checkpoint_step,
    _prepare_output_dir,
    _resolve_output_dir,
    _resolve_output_dir_for_accelerator,
    _resolve_resume_checkpoint,
    _resolve_resume_checkpoint_for_accelerator,
    _save_training_checkpoint,
)
from deberta.training.runtime import (
    _build_decoupled_optimizers,
    _build_optimizer,
    _build_scheduler,
    _build_train_dataset_and_collator,
    _cycle_dataloader,
    _validate_training_configs,
)
from deberta.training.steps import (
    _NONFINITE_LR_MULT_FLOOR,
    _NONFINITE_LR_MULT_RECOVERY,
    _any_rank_flag_true,
    _apply_lr_mult,
    _apply_nonfinite_recovery,
    _collect_ga_window,
    _global_grad_l2_norm,
    _move_batch_to_device,
    _record_unscaled_lrs,
    _resolve_window_token_weights,
    _scheduler_current_lr,
    _sync_discriminator_embeddings_if_available,
)
from deberta.training.tracker_utils import _init_trackers, _setup_wandb_watch, _upload_wandb_original_config
from deberta.utils.checkpoint import load_state_with_compile_fallback, unwrap_compiled_model
from deberta.utils.log import setup_process_logging
from deberta.utils.paths import validate_existing_output_dir

logger = logging.getLogger(__name__)


def _replay_data_iterator_preserving_rng(
    *,
    train_iter: Iterator[dict[str, Any]],
    replay_steps: Iterable[object],
) -> None:
    """Advance a data iterator without changing checkpoint-restored process RNG.

    :param Iterator[dict[str, Any]] train_iter: Training iterator to advance.
    :param Iterable[object] replay_steps: One item per discarded historical batch.
    :return None: None.
    """

    with torch.random.fork_rng(devices=[]):
        for _ in replay_steps:
            _ = next(train_iter)


def _resolve_entrypoint_profile_sections(
    *,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig | None,
    optim_cfg: OptimConfig | None,
) -> tuple[TrainConfig, OptimConfig]:
    """Resolve omitted training sections against the selected backbone profile.

    Supplied sections are returned unchanged because the section-only entrypoint API cannot
    distinguish a field omitted by its caller from an explicitly supplied value equal to the
    dataclass default. Passing None is the unambiguous request for profile defaults.

    :param ModelConfig model_cfg: Model configuration selecting the backbone profile.
    :param TrainConfig | None train_cfg: Optional explicit training configuration.
    :param OptimConfig | None optim_cfg: Optional explicit optimizer configuration.
    :return tuple[TrainConfig, OptimConfig]: Explicit sections or profile-resolved defaults.
    """

    defaults = Config(model=model_cfg)
    return (
        train_cfg if train_cfg is not None else defaults.train,
        optim_cfg if optim_cfg is not None else defaults.optim,
    )


def _flash_attention_enabled_for_runtime(model_cfg: ModelConfig) -> bool:
    """Return whether current training can consume FlashDeBERTa metadata by config.

    :param ModelConfig model_cfg: Resolved model config.
    :return bool: True when flash attention is enabled by config.
    """

    hf_cfg = getattr(model_cfg, "hf", None)
    return is_flash_attention_impl(getattr(hf_cfg, "attention_impl", "eager"))


def _clip_gradients_and_find_nonfinite(
    *,
    accelerator: Any,
    model: torch.nn.Module,
    max_grad_norm: float,
) -> tuple[str | None, float]:
    """Validate gradients before and after optional clipping.

    :param Any accelerator: Accelerator used for rank-wide checks and clipping.
    :param torch.nn.Module model: Model whose gradients are inspected.
    :param float max_grad_norm: Positive clipping threshold, or a disabled value.
    :return tuple[str | None, float]: Failure reason and the inspected norm.
    """

    grad_norm = _global_grad_l2_norm(model)
    if _any_rank_flag_true(accelerator=accelerator, flag=not math.isfinite(float(grad_norm))):
        return "grad_norm", float(grad_norm)
    if not _should_clip_gradients(max_grad_norm):
        return None, float(grad_norm)
    accelerator.clip_grad_norm_(model.parameters(), float(max_grad_norm))
    post_clip_grad_norm = _global_grad_l2_norm(model)
    if _any_rank_flag_true(
        accelerator=accelerator,
        flag=not math.isfinite(float(post_clip_grad_norm)),
    ):
        return "grad_norm_post_clip", float(post_clip_grad_norm)
    return None, float(post_clip_grad_norm)


@dataclass
class _NonfiniteWindowObservation:
    """First local nonfinite observation for one accumulation phase."""

    detected: bool = False
    reason: str | None = None
    micro_step: int | None = None

    def record(self, reason: str | None, *, micro_step: int) -> None:
        """Record the first local nonfinite reason in the phase.

        :param str | None reason: Current micro-step failure reason.
        :param int micro_step: Current micro-step index.
        :return None: None.
        """

        if reason is None:
            return
        self.detected = True
        if self.reason is None:
            self.reason = str(reason)
            self.micro_step = int(micro_step)

    def synchronized_failure(
        self,
        *,
        accelerator: Any,
        is_sync_step: bool,
        current_micro_step: int,
    ) -> tuple[str, int] | None:
        """Resolve the rank-wide failure at the accumulation sync point.

        :param Any accelerator: Accelerator used for rank-wide flag reduction.
        :param bool is_sync_step: Whether this is the phase's synchronization step.
        :param int current_micro_step: Current micro-step fallback index.
        :return tuple[str, int] | None: Effective reason and first failing step, if any.
        """

        if not is_sync_step or not _any_rank_flag_true(accelerator=accelerator, flag=self.detected):
            return None
        return (
            self.reason if self.reason is not None else "other_rank_nonfinite",
            self.micro_step if self.micro_step is not None else int(current_micro_step),
        )


def run_pretraining_dry_run(
    *,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig | None = None,
    optim_cfg: OptimConfig | None = None,
    logging_cfg: LoggingConfig | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run preflight checks for `deberta train`.

    This validates configuration contracts and probes core runtime dependencies
    (tokenizer, dataset access, collator output, model config construction)
    without starting optimization/training loops. It may access network sources and populate
    dependency caches; see ``configs/config_reference.yaml``.

    :param ModelConfig model_cfg: Model configuration.
    :param DataConfig data_cfg: Data configuration.
    :param TrainConfig | None train_cfg: Explicit training configuration, or None to use the
        selected backbone profile defaults.
    :param OptimConfig | None optim_cfg: Explicit optimizer configuration, or None to use the
        selected backbone profile defaults.
    :param LoggingConfig | None logging_cfg: Optional logging configuration.
    :param str | Path | None config_path: Optional source config path.
    :raises RuntimeError: If a preflight stage fails.
    :return dict[str, Any]: Summary of resolved dry-run checks.
    """
    train_cfg, resolved_optim_cfg = _resolve_entrypoint_profile_sections(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        optim_cfg=optim_cfg,
    )
    resolved_logging_cfg = logging_cfg if logging_cfg is not None else LoggingConfig()

    _validate_training_configs(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        optim_cfg=resolved_optim_cfg,
        logging_cfg=resolved_logging_cfg,
    )

    checkpoint_output_dir = _resolve_output_dir(
        output_dir=train_cfg.checkpoint.output_dir,
        project_name=resolved_logging_cfg.project_name,
        config_path=config_path,
        run_name=resolved_logging_cfg.run_name,
    )
    logging_output_dir = (
        Path(str(resolved_logging_cfg.output_dir)).expanduser().resolve()
        if resolved_logging_cfg.output_dir is not None and str(resolved_logging_cfg.output_dir).strip()
        else checkpoint_output_dir
    )
    resume_hint = (
        str(train_cfg.checkpoint.resume_from_checkpoint).strip()
        if train_cfg.checkpoint.resume_from_checkpoint is not None
        else ""
    )
    validate_existing_output_dir(
        output_dir=checkpoint_output_dir,
        allow_nonempty=bool(train_cfg.checkpoint.overwrite_output_dir) or bool(resume_hint),
        nonempty_error=(
            f"Output directory exists and is not empty: {checkpoint_output_dir}. "
            "Set train.checkpoint.overwrite_output_dir=true or set "
            "train.checkpoint.resume_from_checkpoint."
        ),
        nondir_error=f"Output directory exists and is not a directory: {checkpoint_output_dir}",
    )
    ckpt = _resolve_resume_checkpoint(
        output_dir=checkpoint_output_dir,
        resume_from_checkpoint=train_cfg.checkpoint.resume_from_checkpoint,
        is_main_process=True,
    )

    mixed_precision = resolve_effective_mixed_precision(
        train_cfg.mixed_precision,
        bf16_sanity_check=_bf16_runtime_sanity_check,
    )
    compile_enabled = bool(train_cfg.compile.enabled)
    _, compile_scope, compile_scope_reason = _resolve_effective_compile_scope(
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        compile_enabled=compile_enabled,
    )

    _persist_or_validate_run_configs(
        output_dir=checkpoint_output_dir,
        logging_output_dir=logging_output_dir,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        optim_cfg=resolved_optim_cfg,
        logging_cfg=resolved_logging_cfg,
        resume_checkpoint=ckpt,
        config_path=config_path,
        is_main_process=False,
        preflight_only=True,
        effective_compile_scope=compile_scope if compile_enabled else None,
        compile_scope_reason=compile_scope_reason,
    )

    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for dry-run preflight.") from exc

    try:
        tokenizer_source: str | Path = (
            materialized_tokenizer_path(infer_run_dir_from_checkpoint(ckpt))
            if ckpt is not None
            else model_cfg.tokenizer.name_or_path
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load tokenizer from {str(tokenizer_source)!r}.") from exc
    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "Tokenizer preflight failed: tokenizer.pad_token_id is unset. "
            "Use a tokenizer with a defined PAD token."
        )

    try:
        raw_train = load_hf_dataset(data_cfg)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load dataset for dry-run preflight. Check data.source.dataset_name, "
            "data.source.data_files, data.source.load_from_disk, data.source.train_split, and "
            "network/auth settings."
        ) from exc

    preflight_data_cfg = replace(
        data_cfg,
        source=replace(data_cfg.source, shuffle_buffer_size=0),
    )
    train_dataset, collator = _build_train_dataset_and_collator(
        raw_train=raw_train,
        tokenizer=tokenizer,
        data_cfg=preflight_data_cfg,
        train_cfg=train_cfg,
        process_index=0,
        num_processes=1,
        flash_enabled=_flash_attention_enabled_for_runtime(model_cfg),
    )

    example_iter = iter(train_dataset)
    try:
        sample = next(example_iter)
    except StopIteration as exc:
        raise RuntimeError(
            "Dataset preflight produced zero examples. Check text column, split, and tokenization inputs."
        ) from exc

    try:
        batch = collator([sample])
    except Exception as exc:
        raise RuntimeError("Collator preflight failed while building a sample batch.") from exc

    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise RuntimeError(
            "Collator preflight failed: expected tensor fields 'input_ids' and 'labels' in sample batch."
        )
    if input_ids.ndim != 2 or labels.ndim != 2:
        raise RuntimeError(
            "Collator preflight failed: expected 2D [B,S] tensors for input_ids/labels, "
            f"got shapes input_ids={tuple(input_ids.shape)}, labels={tuple(labels.shape)}."
        )
    active_tokens = _count_input_tokens_for_batch(batch)
    if active_tokens <= 0:
        raise RuntimeError("Sample batch preflight failed: active token count is zero.")

    try:
        if ckpt is not None:
            disc_config, gen_config = load_materialized_backbone_configs(
                run_dir=infer_run_dir_from_checkpoint(ckpt),
                model_cfg=model_cfg,
            )
        else:
            disc_config, gen_config = build_backbone_configs(
                model_cfg=model_cfg,
                tokenizer=tokenizer,
                max_position_embeddings=int(data_cfg.packing.max_seq_length),
            )
    except Exception as exc:
        raise RuntimeError(
            "Backbone config preflight failed. Check model/backbone/tokenizer settings and vocab alignment options."
        ) from exc

    summary = {
        "status": "ok",
        "output_dir": str(checkpoint_output_dir),
        "checkpoint_output_dir": str(checkpoint_output_dir),
        "logging_output_dir": str(logging_output_dir),
        "resume_checkpoint": str(ckpt) if ckpt is not None else None,
        "effective_compile_scope": str(compile_scope) if compile_enabled else None,
        "mixed_precision": str(mixed_precision),
        "sample_batch_shape": tuple(int(x) for x in input_ids.shape),
        "sample_active_tokens": float(active_tokens),
        "tokenizer_vocab_size": int(len(tokenizer)),
        "discriminator_vocab_size": int(disc_config.vocab_size),
        "generator_vocab_size": int(gen_config.vocab_size),
    }
    close_example_iter = getattr(example_iter, "close", None)
    if callable(close_example_iter):
        close_example_iter()
    # Streaming PyArrow iterators can retain shutdown-time callbacks after a short
    # preflight. Release the iterator and its dataset graph before returning.
    del batch, sample, close_example_iter, example_iter, train_dataset, raw_train
    gc.collect()
    return summary


def run_pretraining(
    *,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig | None = None,
    optim_cfg: OptimConfig | None = None,
    logging_cfg: LoggingConfig | None = None,
    config_path: str | Path | None = None,
) -> None:
    """Run RTD pretraining.

    :param ModelConfig model_cfg: Model configuration.
    :param DataConfig data_cfg: Data configuration.
    :param TrainConfig | None train_cfg: Explicit training configuration, or None to use the
        selected backbone profile defaults.
    :param OptimConfig | None optim_cfg: Explicit optimizer configuration, or None to use the
        selected backbone profile defaults.
    :param LoggingConfig | None logging_cfg: Optional logging configuration.
    :param str | Path | None config_path: Optional source config path for auto output-dir naming.
    """
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import set_seed

    train_cfg, resolved_optim_cfg = _resolve_entrypoint_profile_sections(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        optim_cfg=optim_cfg,
    )
    resolved_logging_cfg = logging_cfg if logging_cfg is not None else LoggingConfig()
    report_to = "wandb" if bool(resolved_logging_cfg.wandb.enabled) else "none"
    if report_to == "wandb":
        try:
            __import__("wandb")
        except Exception as exc:
            raise RuntimeError(
                "logging.wandb.enabled=true requires the W&B dependency. "
                "Install it with `pip install -e '.[wandb]'`."
            ) from exc

    log_with = None if report_to == "none" else report_to
    mixed_precision = resolve_effective_mixed_precision(
        train_cfg.mixed_precision,
        bf16_sanity_check=_bf16_runtime_sanity_check,
    )
    compile_mode = _normalize_torch_compile_mode(train_cfg.compile.mode)
    compile_enabled = bool(train_cfg.compile.enabled)
    object.__setattr__(train_cfg, "mixed_precision", mixed_precision)
    accelerator_kwargs: dict[str, Any] = {
        "gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,
        "log_with": log_with,
        "mixed_precision": mixed_precision,
    }
    if bool(train_cfg.decoupled_training):
        # Generator/discriminator phases each touch only a subset of parameters.
        # DDP must track unused params to avoid cross-rank reducer stalls.
        accelerator_kwargs["kwargs_handlers"] = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(
        **accelerator_kwargs,
    )

    setup_process_logging(accelerator.is_main_process)
    _validate_training_configs(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        optim_cfg=resolved_optim_cfg,
        logging_cfg=resolved_logging_cfg,
    )
    compile_backend = str(train_cfg.compile.backend).strip().lower()
    compile_scope_requested, compile_scope, compile_scope_reason = _resolve_effective_compile_scope(
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        compile_enabled=compile_enabled,
    )
    _maybe_enable_tf32(train_cfg.tf32)
    _maybe_configure_sdpa_kernels(str(train_cfg.sdpa_kernel), is_main=accelerator.is_main_process)

    logger.info(f"Accelerate state: {accelerator.state}")

    # Resolve output dir before side effects and persist the concrete path in train_cfg.
    configured_output_dir = train_cfg.checkpoint.output_dir
    output_dir = _resolve_output_dir_for_accelerator(
        accelerator=accelerator,
        output_dir=train_cfg.checkpoint.output_dir,
        project_name=resolved_logging_cfg.project_name,
        config_path=config_path,
        run_name=resolved_logging_cfg.run_name,
    )
    object.__setattr__(train_cfg.checkpoint, "output_dir", str(output_dir))
    if resolved_logging_cfg.output_dir is None or not str(resolved_logging_cfg.output_dir).strip():
        logging_output_dir = output_dir
    else:
        logging_output_dir = Path(str(resolved_logging_cfg.output_dir)).expanduser().resolve()
    object.__setattr__(resolved_logging_cfg, "output_dir", str(logging_output_dir))
    if accelerator.is_main_process and (
        configured_output_dir is None or not str(configured_output_dir).strip()
    ):
        logger.info("train.checkpoint.output_dir unset; auto-selected output_dir=%s", output_dir)
    if accelerator.is_main_process and logging_output_dir != output_dir:
        logger.info(
            "logging.output_dir explicitly set to %s (checkpoint output_dir=%s)",
            logging_output_dir,
            output_dir,
        )

    # Make/validate output dir on main.
    _prepare_output_dir(
        output_dir=output_dir,
        overwrite_output_dir=bool(train_cfg.checkpoint.overwrite_output_dir),
        resume_from_checkpoint=train_cfg.checkpoint.resume_from_checkpoint,
        is_main_process=accelerator.is_main_process,
    )
    if accelerator.is_main_process:
        logging_output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    ckpt = _resolve_resume_checkpoint_for_accelerator(
        accelerator=accelerator,
        output_dir=output_dir,
        resume_from_checkpoint=train_cfg.checkpoint.resume_from_checkpoint,
    )
    _persist_or_validate_run_configs(
        output_dir=output_dir,
        logging_output_dir=logging_output_dir,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        optim_cfg=resolved_optim_cfg,
        logging_cfg=resolved_logging_cfg,
        resume_checkpoint=ckpt,
        config_path=config_path,
        is_main_process=accelerator.is_main_process,
        effective_compile_scope=compile_scope if compile_enabled else None,
        compile_scope_reason=compile_scope_reason,
    )

    accelerator.wait_for_everyone()

    set_seed(train_cfg.seed, device_specific=True)

    # Tokenizer
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required.") from e

    # Suppress repeated fast-tokenizer advisory logs that add noise in multi-worker runs.
    with suppress(Exception):
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    materialized_source_run = infer_run_dir_from_checkpoint(ckpt) if ckpt is not None else None
    tokenizer_source: str | Path = (
        materialized_tokenizer_path(materialized_source_run)
        if materialized_source_run is not None
        else model_cfg.tokenizer.name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    # Sanity
    if tokenizer.pad_token_id is None:
        # Many DeBERTa tokenizers define PAD.
        raise ValueError("Tokenizer must have pad_token_id.")

    if materialized_source_run is not None:
        disc_config, gen_config = load_materialized_backbone_configs(
            run_dir=materialized_source_run,
            model_cfg=model_cfg,
        )
    else:
        disc_config, gen_config = build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            max_position_embeddings=int(data_cfg.packing.max_seq_length),
        )

    if accelerator.is_main_process and (
        materialized_source_run is None or materialized_source_run != output_dir
    ):
        persist_materialized_run_artifacts(
            run_dir=output_dir,
            tokenizer=tokenizer,
            discriminator_config=disc_config,
            generator_config=gen_config,
        )
    accelerator.wait_for_everyone()

    # Data
    raw_train = load_hf_dataset(data_cfg)

    train_dataset, collator = _build_train_dataset_and_collator(
        raw_train=raw_train,
        tokenizer=tokenizer,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        process_index=int(accelerator.process_index),
        num_processes=int(accelerator.num_processes),
        flash_enabled=_flash_attention_enabled_for_runtime(model_cfg),
    )

    # Dataloader
    num_workers = int(train_cfg.dataloader.num_workers)
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(int(train_cfg.seed) + int(accelerator.process_index))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.per_device_train_batch_size),
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=bool(train_cfg.dataloader.pin_memory),
        # drop_last keeps the batch shape invariant across steps; flash
        # doc-block route hints depend on batch size, so a smaller final
        # batch would flip routes (and recompile) mid-epoch.
        drop_last=True,
        persistent_workers=(num_workers > 0),
        generator=data_loader_generator,
    )

    # Instantiate backbones
    # Resume paths should instantiate from config and rely on accelerate checkpoint state
    # instead of fetching original pretrained model sources again.
    load_pretrained_backbones = ckpt is None
    disc_backbone, gen_backbone = build_backbones(
        model_cfg=model_cfg,
        disc_config=disc_config,
        gen_config=gen_config,
        load_pretrained_weights=load_pretrained_backbones,
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
        additional_forbidden_token_ids=getattr(tokenizer, "all_special_ids", []),
    )
    flash_policy_keys = {
        str(module.flash_kernel_policy_key)
        for module in model.modules()
        if hasattr(module, "flash_kernel_policy_key")
    }
    if len(flash_policy_keys) > 1:
        raise RuntimeError(
            f"Model contains conflicting FlashDeBERTa kernel policies: {sorted(flash_policy_keys)}."
        )
    flash_kernel_policy_key = next(iter(flash_policy_keys), None)

    effective_decoupled_training = bool(train_cfg.decoupled_training)

    optimizer: torch.optim.Optimizer | None = None
    lr_scheduler: Any | None = None
    gen_optimizer: torch.optim.Optimizer | None = None
    disc_optimizer: torch.optim.Optimizer | None = None
    gen_lr_scheduler: Any | None = None
    disc_lr_scheduler: Any | None = None
    param_digest: str | dict[str, str]

    # Optimizer + scheduler
    if effective_decoupled_training:
        gen_optimizer, disc_optimizer = _build_decoupled_optimizers(
            model,
            resolved_optim_cfg,
            mixed_precision=mixed_precision,
        )
        gen_lr_scheduler = _build_scheduler(gen_optimizer, train_cfg=train_cfg, optim_cfg=resolved_optim_cfg)
        disc_lr_scheduler = _build_scheduler(
            disc_optimizer, train_cfg=train_cfg, optim_cfg=resolved_optim_cfg
        )
        param_digest = {
            "generator": str(gen_optimizer._param_order_digest),
            "discriminator": str(disc_optimizer._param_order_digest),
        }
        model, gen_optimizer, disc_optimizer, gen_lr_scheduler, disc_lr_scheduler = accelerator.prepare(
            model, gen_optimizer, disc_optimizer, gen_lr_scheduler, disc_lr_scheduler
        )
        _record_unscaled_lrs(gen_optimizer, gen_lr_scheduler)
        _record_unscaled_lrs(disc_optimizer, disc_lr_scheduler)
    else:
        optimizer = _build_optimizer(
            model,
            resolved_optim_cfg,
            mixed_precision=mixed_precision,
        )
        param_digest = str(optimizer._param_order_digest)
        lr_scheduler = _build_scheduler(optimizer, train_cfg=train_cfg, optim_cfg=resolved_optim_cfg)
        model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        _record_unscaled_lrs(optimizer, lr_scheduler)

    # GDES embedding sharing uses synced non-trainable base weights inside discriminator embeddings.
    # Those tensors must be initialized/synced from generator weights before any compiled forward runs.
    with suppress(Exception):
        _sync_discriminator_embeddings_if_available(model, accelerator=accelerator)

    if compile_enabled:
        try:
            if compile_scope_reason:
                logger.warning(compile_scope_reason)

            unwrapped = unwrap_compiled_model(accelerator, model)
            compile_kwargs: dict[str, Any] = {"mode": compile_mode, "backend": compile_backend}
            try:
                compile_params = inspect.signature(torch.compile).parameters  # type: ignore[attr-defined]
                if "dynamic" in compile_params:
                    compile_kwargs["dynamic"] = False
            except Exception:
                pass
            compile_scope_key = str(compile_scope).strip().lower()
            if compile_scope_key in {"backbones", "encoder", "gen_encoder", "disc_encoder"}:
                prefilled_rotary = _prefill_rotary_caches_for_compile(
                    model=unwrapped,
                    seq_len=int(data_cfg.packing.max_seq_length),
                    device=torch.device(getattr(accelerator, "device", torch.device("cpu"))),
                    dtype=_dtype_for_mixed_precision(mixed_precision),
                )
                if prefilled_rotary > 0:
                    logger.info(
                        "Prefilled rotary caches for %d module(s) at seq_len=%d before torch.compile.",
                        int(prefilled_rotary),
                        int(data_cfg.packing.max_seq_length),
                    )
            compiled_targets = _compile_backbones_for_scope(
                unwrapped_model=unwrapped,
                compile_scope=compile_scope,
                compile_kwargs=compile_kwargs,
            )

            logger.info(
                "Enabled torch.compile for %s (scope=%s, requested_scope=%s, %s)",
                ",".join(compiled_targets),
                compile_scope,
                compile_scope_requested,
                ", ".join(f"{k}={v}" for k, v in compile_kwargs.items()),
            )
        except Exception as e:
            raise RuntimeError(
                f"torch.compile failed for scope={compile_scope}, mode={compile_mode}, backend={compile_backend}."
            ) from e

    tracker_cfg_runtime = _build_runtime_resolved_tracker_config(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        optim_cfg=resolved_optim_cfg,
        logging_cfg=resolved_logging_cfg,
        disc_config=disc_config,
        gen_config=gen_config,
        tokenizer=tokenizer,
    )
    global_step = 0
    consumed_micro_batches = 0
    consumed_micro_batches_committed = 0
    ga_steps = max(1, int(train_cfg.gradient_accumulation_steps))
    last_saved_step = 0
    lr_mult = 1.0
    nonfinite_skip_total = 0
    nonfinite_skip_streak = 0
    uncommitted_training_window = False
    crash_type: str | None = None
    crash_reason: str | None = None
    crash_step: int | None = None
    exit_code = 0
    train_started_at = time.perf_counter()
    metrics_path = logging_output_dir / "metrics.jsonl.gz"
    debug_metrics_enabled = bool(resolved_logging_cfg.debug.metrics)
    wandb_run: Any | None = None
    max_tracker_step_logged = 0
    train_progress: Any | None = None

    def _log_tracker_metrics(row: dict[str, Any], *, step: int) -> int:
        """Log tracker metrics with monotonic global step.

        :param dict[str, Any] row: Metrics row.
        :param int step: Requested step.
        :return int: Effective step used for logging.
        """
        nonlocal max_tracker_step_logged
        if report_to == "none":
            return int(step)

        effective_step = int(step)
        if effective_step < int(max_tracker_step_logged):
            effective_step = int(max_tracker_step_logged)
        else:
            max_tracker_step_logged = int(effective_step)
        accelerator.log(row, step=effective_step)
        return int(effective_step)

    def _save_checkpoint_if_due(
        *,
        global_step: int,
        consumed_micro_batches_committed: int,
        lr_mult: float,
        last_saved_step: int,
    ) -> int:
        """Save a periodic checkpoint when due, stamping run-invariant kwargs.

        :param int global_step: Committed global step.
        :param int consumed_micro_batches_committed: Committed micro-batch count.
        :param float lr_mult: Current LR multiplier.
        :param int last_saved_step: Step of the last saved checkpoint.
        :return int: Updated last-saved step.
        """
        return _save_periodic_checkpoint_if_due(
            accelerator=accelerator,
            train_cfg=train_cfg,
            output_dir=output_dir,
            global_step=int(global_step),
            consumed_micro_batches_committed=int(consumed_micro_batches_committed),
            lr_mult=float(lr_mult),
            optimizer_param_digest=param_digest,
            gradient_accumulation_steps=int(ga_steps),
            last_saved_step=int(last_saved_step),
        )

    def _write_nonfinite_artifact(
        *,
        micro_step_idx: int,
        offending: str,
        gen_loss_raw: torch.Tensor | None,
        disc_loss_raw: torch.Tensor | None,
        forward_loss: torch.Tensor | None,
        backward_loss: torch.Tensor | None,
        grad_norm: float | None,
        lr: float | None,
    ) -> Path:
        """Write a non-finite debug artifact, stamping run-invariant fields.

        The step is stamped as ``global_step + 1``: every skip path reports the
        in-flight step being attempted, not the last committed one.

        :param int micro_step_idx: Micro-step index within the window.
        :param str offending: Name of the first offending tensor/stat.
        :param torch.Tensor | None gen_loss_raw: Generator raw loss.
        :param torch.Tensor | None disc_loss_raw: Discriminator raw loss.
        :param torch.Tensor | None forward_loss: Forward scalar objective.
        :param torch.Tensor | None backward_loss: Backward scalar objective.
        :param float | None grad_norm: Global gradient norm.
        :param float | None lr: Scheduler LR snapshot.
        :return Path: Written artifact path.
        """
        return _write_nonfinite_debug_artifact(
            output_dir=logging_output_dir,
            step=int(global_step + 1),
            micro_step_idx=int(micro_step_idx),
            offending=str(offending),
            gen_loss_raw=gen_loss_raw,
            disc_loss_raw=disc_loss_raw,
            forward_loss=forward_loss,
            backward_loss=backward_loss,
            grad_norm=grad_norm,
            lr=lr,
            compile_enabled=compile_enabled,
            compile_mode=compile_mode,
            embedding_sharing=str(model_cfg.embedding_sharing),
        )

    try:
        # Trackers
        if report_to != "none":
            tracker_cfg = dict(tracker_cfg_runtime)
            if resolved_logging_cfg.run_name is not None and str(resolved_logging_cfg.run_name).strip():
                tracker_run_name = str(resolved_logging_cfg.run_name).strip()
            else:
                tracker_run_name = logging_output_dir.name
            _init_trackers(
                accelerator=accelerator,
                project_name=str(resolved_logging_cfg.project_name).strip(),
                tracker_cfg=tracker_cfg,
                report_to=str(report_to),
                run_name=tracker_run_name,
                logging_dir=logging_output_dir,
            )
            if str(report_to).lower() == "wandb":
                try:
                    wandb_run = accelerator.get_tracker("wandb", unwrap=True)
                except Exception:
                    wandb_run = None
                    logger.exception("Failed to resolve W&B tracker from Accelerator.")

                try:
                    uploaded_config = _upload_wandb_original_config(
                        accelerator=accelerator,
                        wandb_run=wandb_run,
                        config_original_path=logging_output_dir / "config_original.yaml",
                        run_name=tracker_run_name,
                        config_resolved_path=logging_output_dir / "config_resolved.yaml",
                        config_source_path=config_path,
                    )
                    if not uploaded_config:
                        logger.warning(
                            "W&B config snapshot upload skipped; expected paths: %s, %s (source=%s).",
                            logging_output_dir / "config_original.yaml",
                            logging_output_dir / "config_resolved.yaml",
                            config_path,
                        )
                except Exception:
                    logger.exception("Failed to upload config snapshots to W&B.")

                try:
                    _setup_wandb_watch(
                        accelerator=accelerator,
                        wandb_run=wandb_run,
                        model=model,
                        watch_mode=resolved_logging_cfg.wandb.watch,
                        watch_log_freq=int(resolved_logging_cfg.wandb.watch_log_freq),
                    )
                except Exception:
                    logger.exception("Failed to initialize W&B model watch.")

        # Resume
        if ckpt:
            logger.info(f"Resuming from checkpoint: {ckpt}")
            load_state_with_compile_fallback(
                accelerator=accelerator,
                model=model,
                checkpoint_dir=ckpt,
                context="resume",
            )
            if effective_decoupled_training:
                if gen_optimizer is not None and gen_lr_scheduler is not None:
                    _record_unscaled_lrs(gen_optimizer, gen_lr_scheduler)
                if disc_optimizer is not None and disc_lr_scheduler is not None:
                    _record_unscaled_lrs(disc_optimizer, disc_lr_scheduler)
            else:
                if optimizer is not None and lr_scheduler is not None:
                    _record_unscaled_lrs(optimizer, lr_scheduler)

            # GDES base weights are non-persistent and must be refreshed after loading a checkpoint.
            with suppress(Exception):
                _sync_discriminator_embeddings_if_available(model, accelerator=accelerator)

            (
                restored,
                restored_lr_mult,
                saved_digest,
                saved_global_step,
                saved_ga_steps,
            ) = _load_checkpoint_progress_metadata(Path(ckpt))
            if restored is None:
                raise RuntimeError(
                    "Checkpoint resume requires data_state.json with consumed_micro_batches metadata. "
                    f"File is missing or invalid for checkpoint '{ckpt}'. "
                    "The checkpoint may be incomplete due to a crashed save. "
                    "Resume from a different checkpoint created by this code version or start a new run."
                )
            parsed_checkpoint_step = _parse_checkpoint_step(ckpt)
            if saved_global_step is None or saved_ga_steps is None or saved_digest is None:
                raise RuntimeError(
                    "Checkpoint resume requires current data_state.json metadata: global_step, "
                    "gradient_accumulation_steps, and optimizer_param_digest. "
                    f"Checkpoint '{ckpt}' was created by an unsupported code revision or is incomplete."
                )
            global_step = int(max(0, saved_global_step))
            if int(parsed_checkpoint_step) > 0 and int(parsed_checkpoint_step) != int(global_step):
                raise RuntimeError(
                    "Checkpoint step mismatch on resume: "
                    f"path step={int(parsed_checkpoint_step)} but data_state.json global_step={int(global_step)} "
                    f"for checkpoint '{ckpt}'."
                )
            last_saved_step = int(global_step)
            if int(saved_ga_steps) != int(ga_steps) and accelerator.is_main_process:
                logger.warning(
                    "Resume checkpoint '%s' was saved with gradient_accumulation_steps=%d but current run uses %d.",
                    ckpt,
                    int(saved_ga_steps),
                    int(ga_steps),
                )
            consumed_micro_batches = int(restored)
            consumed_micro_batches_committed = int(consumed_micro_batches)
            lr_mult = float(restored_lr_mult)

            if isinstance(saved_digest, dict):
                if isinstance(param_digest, dict):
                    mismatch = {
                        key: (saved_digest.get(key), param_digest.get(key))
                        for key in ("generator", "discriminator")
                        if saved_digest.get(key) != param_digest.get(key)
                    }
                    if mismatch:
                        raise RuntimeError(
                            f"Optimizer parameter-order digest mismatch on resume from '{ckpt}'. "
                            f"Mismatched keys: {mismatch}. Start a new run or restore with matching code."
                        )
                else:
                    raise RuntimeError(
                        f"Optimizer topology mismatch on resume from '{ckpt}': checkpoint stores "
                        "decoupled optimizer digests but the current run uses one optimizer."
                    )
            elif isinstance(param_digest, dict):
                raise RuntimeError(
                    f"Optimizer topology mismatch on resume from '{ckpt}': checkpoint stores one "
                    "optimizer digest but the current run uses decoupled optimizers."
                )
            elif saved_digest != param_digest:
                raise RuntimeError(
                    f"Optimizer parameter-order digest mismatch on resume from '{ckpt}'. "
                    f"Saved digest: {saved_digest}, current digest: {param_digest}. "
                    "This means optimizer group insertion order changed between the "
                    "checkpoint code version and the current code version. Resuming would "
                    "silently map optimizer momentum/variance to wrong parameters. "
                    "Start a new run or restore with matching code."
                )

            max_tracker_step_logged = max(int(max_tracker_step_logged), int(global_step))

        # Training loop
        model.train()
        if accelerator.is_main_process:
            train_progress = tqdm(
                total=int(train_cfg.max_steps),
                initial=int(global_step),
                desc="train",
                unit="step",
                mininterval=1.0,
                dynamic_ncols=True,
                leave=True,
            )

        start_epoch, do_replay, resume_policy_reason = _resolve_data_resume_policy(
            train_cfg=train_cfg,
            consumed_micro_batches=consumed_micro_batches,
            global_step=global_step,
        )
        train_iter = _cycle_dataloader(train_loader, start_epoch=start_epoch)
        token_weighted_ga = bool(train_cfg.token_weighted_gradient_accumulation)
        if consumed_micro_batches > 0:
            if int(global_step) >= int(train_cfg.max_steps):
                logger.info(
                    "Resume global_step=%d already reached max_steps=%d; skipping data replay.",
                    int(global_step),
                    int(train_cfg.max_steps),
                )
            elif bool(do_replay):
                logger.info(
                    "Replaying data iterator by %d micro-batches to align resume data position (%s).",
                    int(consumed_micro_batches),
                    resume_policy_reason,
                )
                replay_iter = range(consumed_micro_batches)
                if accelerator.is_main_process:
                    replay_iter = tqdm(
                        replay_iter,
                        total=consumed_micro_batches,
                        desc="resume-replay",
                        unit="microbatch",
                        mininterval=1.0,
                        dynamic_ncols=True,
                        leave=False,
                    )
                _replay_data_iterator_preserving_rng(
                    train_iter=train_iter,
                    replay_steps=replay_iter,
                )
            else:
                logger.warning(
                    "Skipping data replay on resume (%s). "
                    "Resume becomes O(1) but data order may differ from the pre-crash run.",
                    resume_policy_reason,
                )

        local_input_tokens_seen = 0.0
        local_input_tokens_since_log = 0.0
        last_log_started_at = time.perf_counter()
        zero_gen_window_total = 0
        zero_disc_window_total = 0
        zero_gen_window_since_log = 0
        zero_disc_window_since_log = 0

        def _maybe_log_training_metrics(
            *,
            lr_scheduler: Any,
            gen_loss_num: torch.Tensor,
            gen_token_count_window: torch.Tensor,
            disc_loss_num: torch.Tensor,
            disc_acc_num: torch.Tensor,
            disc_token_count_window: torch.Tensor,
            disc_positive_count_window: torch.Tensor,
            loss_override: float | None = None,
        ) -> None:
            """Emit per-window metrics when ``logging_steps`` interval is reached.

            :param Any lr_scheduler: Scheduler used for LR reporting.
            :param torch.Tensor gen_loss_num: Window numerator for generator loss.
            :param torch.Tensor gen_token_count_window: Window denominator for generator loss.
            :param torch.Tensor disc_loss_num: Window numerator for discriminator loss.
            :param torch.Tensor disc_acc_num: Window numerator for discriminator accuracy.
            :param torch.Tensor disc_token_count_window: Window denominator for discriminator metrics.
            :param torch.Tensor disc_positive_count_window: Window numerator for discriminator positive fraction.
            :param float | None loss_override: Optional explicit loss scalar for metrics.
            :return None: None.
            """
            nonlocal local_input_tokens_since_log
            nonlocal last_log_started_at
            nonlocal zero_gen_window_since_log
            nonlocal zero_disc_window_since_log
            if not resolved_logging_cfg.logging_steps or (
                global_step % int(resolved_logging_cfg.logging_steps) != 0
            ):
                return

            log_now = time.perf_counter()
            elapsed_since_log = max(log_now - last_log_started_at, 1e-9)
            global_input_tokens_interval = _sum_local_scalar(
                accelerator=accelerator,
                x=local_input_tokens_since_log,
            )
            global_input_tokens_seen = _sum_local_scalar(
                accelerator=accelerator,
                x=local_input_tokens_seen,
            )
            input_tokens_per_sec = global_input_tokens_interval / elapsed_since_log
            local_input_tokens_since_log = 0.0
            last_log_started_at = log_now

            lr_raw = _scheduler_current_lr(lr_scheduler)
            lr = float(lr_raw) if lr_raw is not None else float("nan")
            weighted_metrics = accelerator.reduce(
                torch.stack(
                    [
                        gen_loss_num,
                        gen_token_count_window,
                        disc_loss_num,
                        disc_acc_num,
                        disc_token_count_window,
                        disc_positive_count_window,
                    ]
                ),
                reduction="sum",
            )
            global_gen_loss_num = float(weighted_metrics[0].item())
            global_gen_tokens = float(weighted_metrics[1].item())
            global_disc_loss_num = float(weighted_metrics[2].item())
            global_disc_acc_num = float(weighted_metrics[3].item())
            global_disc_tokens = float(weighted_metrics[4].item())
            global_disc_positive = float(weighted_metrics[5].item())
            gen_loss_window = (
                global_gen_loss_num / global_gen_tokens if global_gen_tokens > 0.0 else float("nan")
            )
            disc_loss_window = (
                global_disc_loss_num / global_disc_tokens if global_disc_tokens > 0.0 else float("nan")
            )
            disc_acc_window = (
                global_disc_acc_num / global_disc_tokens if global_disc_tokens > 0.0 else float("nan")
            )
            disc_pos_frac = (
                global_disc_positive / global_disc_tokens if global_disc_tokens > 0.0 else float("nan")
            )
            disc_prior_loss = -sum(
                probability * math.log(probability)
                for probability in (disc_pos_frac, 1.0 - disc_pos_frac)
                if probability > 0.0
            )
            disc_loss_gain = disc_prior_loss - disc_loss_window
            loss = float(train_cfg.objective.gen_loss_weight) * float(gen_loss_window) + float(
                train_cfg.objective.disc_loss_weight
            ) * float(disc_loss_window)
            if loss_override is not None:
                loss = float(loss_override)

            zero_metrics = {
                "zero_gen_window_total": float(zero_gen_window_total),
                "zero_disc_window_total": float(zero_disc_window_total),
                "zero_gen_window_since_log": float(zero_gen_window_since_log),
                "zero_disc_window_since_log": float(zero_disc_window_since_log),
            }
            metrics = {
                "step": int(global_step),
                "lr": float(lr),
                "loss": float(loss),
                "gen_loss": float(gen_loss_window),
                "disc_loss": float(disc_loss_window),
                "disc_loss_gain": float(disc_loss_gain),
                "disc_acc": float(disc_acc_window),
                "disc_pos_frac": float(disc_pos_frac),
                "input_tokens_per_sec": float(input_tokens_per_sec),
                "input_tokens_seen": float(global_input_tokens_seen),
            }
            zero_gen_window_since_log = 0
            zero_disc_window_since_log = 0
            if accelerator.is_main_process:
                logger.info(
                    " | ".join(
                        [
                            f"step={int(metrics['step'])}",
                            f"lr={metrics['lr']:.3e}",
                            f"loss={metrics['loss']:.4f}",
                            f"gen={metrics['gen_loss']:.4f}",
                            f"disc={metrics['disc_loss']:.4f}",
                            f"gain={metrics['disc_loss_gain']:.4f}",
                            f"acc={metrics['disc_acc']:.4f}",
                            f"pos={metrics['disc_pos_frac']:.4f}",
                            f"tok/s={metrics['input_tokens_per_sec']:.1f}",
                            f"tok_seen={metrics['input_tokens_seen']:.0f}",
                        ]
                    )
                )
            if report_to != "none":
                _log_tracker_metrics({k: v for k, v in metrics.items() if k != "step"}, step=global_step)
            if debug_metrics_enabled and accelerator.is_main_process:
                _append_metrics_jsonl_row(
                    metrics_path,
                    {
                        "step": int(global_step),
                        "debug_metrics": True,
                        **zero_metrics,
                    },
                )

        def _next_training_window(
            *,
            default_unweighted_token_count: float,
        ) -> tuple[list[tuple[dict[str, torch.Tensor], float, float]], float, float]:
            """Collect and account for one shared accumulation window.

            :param float default_unweighted_token_count: Phase fallback count when token weighting is off.
            :return tuple[list[tuple[dict[str, torch.Tensor], float, float]], float, float]:
                Window batches plus generator/discriminator per-rank denominators.
            """

            nonlocal consumed_micro_batches
            nonlocal local_input_tokens_seen, local_input_tokens_since_log
            nonlocal zero_gen_window_total, zero_gen_window_since_log
            nonlocal zero_disc_window_total, zero_disc_window_since_log

            (
                window,
                consumed_in_window,
                local_window_input_tokens,
                local_gen_tokens,
                local_disc_tokens,
            ) = _collect_ga_window(
                train_iter=train_iter,
                ga_steps=ga_steps,
                token_weighted_ga=token_weighted_ga,
                default_unweighted_token_count=default_unweighted_token_count,
            )
            consumed_micro_batches += int(consumed_in_window)
            local_input_tokens_seen += local_window_input_tokens
            local_input_tokens_since_log += local_window_input_tokens
            (
                gen_window_tokens_per_rank,
                disc_window_tokens_per_rank,
                gen_window_zero_tokens,
                disc_window_zero_tokens,
            ) = _resolve_window_token_weights(
                accelerator=accelerator,
                token_weighted_ga=token_weighted_ga,
                local_gen_tokens=local_gen_tokens,
                local_disc_tokens=local_disc_tokens,
                next_step=int(global_step + 1),
            )
            if gen_window_zero_tokens:
                zero_gen_window_total += 1
                zero_gen_window_since_log += 1
            if disc_window_zero_tokens:
                zero_disc_window_total += 1
                zero_disc_window_since_log += 1
            return window, gen_window_tokens_per_rank, disc_window_tokens_per_rank

        def _prepare_training_batch(
            batch: dict[str, Any],
        ) -> tuple[dict[str, Any], Any | None]:
            """Move one batch and attach compile-stable flash metadata.

            :param dict[str, Any] batch: Host-side collated training batch.
            :return tuple[dict[str, Any], Any | None]: Device batch and flash metadata.
            """

            prepared = _move_batch_to_device(batch, accelerator.device)
            prepared = _stabilize_compile_attention_mask(
                batch=prepared,
                compile_enabled=compile_enabled,
                compile_scope=compile_scope,
                backbone_type=str(model_cfg.backbone_type),
            )
            prepared, flash_meta = prepare_flash_attention_batch_metadata(
                batch=prepared,
                backbone_type=str(model_cfg.backbone_type),
                flash_enabled=_flash_attention_enabled_for_runtime(model_cfg),
                flash_cfg=getattr(model_cfg.hf, "flash", None),
                kernel_policy_key=flash_kernel_policy_key,
            )
            if compile_enabled:
                _maybe_cudagraph_mark_step_begin()
            return prepared, flash_meta

        def _increment_nonfinite_skip() -> int:
            """Increment shared nonfinite skip counters and return the total.

            :return int: Updated run-wide skip total.
            """

            nonlocal nonfinite_skip_total, nonfinite_skip_streak
            nonfinite_skip_total += 1
            nonfinite_skip_streak += 1
            return nonfinite_skip_total

        def _commit_training_window() -> None:
            """Commit one consumed accumulation window to visible progress.

            :return None: None.
            """

            nonlocal global_step, consumed_micro_batches_committed
            nonlocal uncommitted_training_window
            global_step += 1
            consumed_micro_batches_committed = int(consumed_micro_batches)
            uncommitted_training_window = False
            if train_progress is not None:
                train_progress.update(1)

        def _record_successful_optimizer_window() -> None:
            """Reset skip state and recover the learning-rate multiplier.

            :return None: None.
            """

            nonlocal nonfinite_skip_streak, lr_mult
            nonfinite_skip_streak = 0
            if lr_mult < 1.0:
                lr_mult = min(lr_mult * float(_NONFINITE_LR_MULT_RECOVERY), 1.0)

        def _finalize_nonfinite_skip(
            *,
            optimizer_phases: list[tuple[Any, Any, bool]],
            reason: str | None,
            debug_path: Path | None,
        ) -> None:
            """Recover from one pre-step non-finite accumulation window.

            :param list[tuple[Any, Any, bool]] optimizer_phases: Optimizer, scheduler,
                and whether that phase already stepped in this window.
            :param str | None reason: Recorded nonfinite failure reason.
            :param Path | None debug_path: Optional written debug artifact.
            :raises RuntimeError: If a phase already stepped or recovery is exhausted.
            :return None: None.
            """

            nonlocal lr_mult
            stepped_phases = [
                index for index, (_, _, did_step) in enumerate(optimizer_phases) if bool(did_step)
            ]
            if stepped_phases:
                raise RuntimeError(
                    "A decoupled RTD phase became non-finite after an earlier phase optimizer "
                    "step. The accumulation window is not atomic and will not be checkpointed; "
                    "resume from the last completed checkpoint. "
                    f"reason={reason or 'unknown'}, stepped_phase_indices={stepped_phases}."
                )

            previous_lr_mult = float(lr_mult)
            for phase_optimizer, phase_scheduler, _ in optimizer_phases:
                _record_unscaled_lrs(phase_optimizer, phase_scheduler)

            lr_mult, reset_state = _apply_nonfinite_recovery(
                lr_mult=lr_mult,
                skip_streak=int(nonfinite_skip_streak),
            )
            for phase_optimizer, _, _ in optimizer_phases:
                _apply_lr_mult(phase_optimizer, lr_mult)
                if reset_state:
                    with suppress(Exception):
                        phase_optimizer.state.clear()

            if report_to != "none":
                _log_tracker_metrics(
                    {
                        "nonfinite_window_skipped": 1.0,
                        "nonfinite_skip_total": float(nonfinite_skip_total),
                        "nonfinite_skip_streak": float(nonfinite_skip_streak),
                        "nonfinite_recovery_lr_mult": float(lr_mult),
                        "nonfinite_recovery_optimizer_state_reset": 1.0 if reset_state else 0.0,
                    },
                    step=int(global_step),
                )
            if accelerator.is_main_process:
                logger.warning(
                    "attempted_step=%d | nonfinite_window_skipped=1 | reason=%s | "
                    "streak=%d | total_skips=%d | "
                    "lr_mult=%.4f | opt_state_reset=%s | debug=%s",
                    int(global_step + 1),
                    str(reason or "unknown"),
                    int(nonfinite_skip_streak),
                    int(nonfinite_skip_total),
                    float(lr_mult),
                    bool(reset_state),
                    str(debug_path) if debug_path is not None else "n/a",
                )
            if previous_lr_mult <= float(_NONFINITE_LR_MULT_FLOOR) and not reset_state:
                raise RuntimeError(
                    "Non-finite recovery is exhausted: the learning-rate multiplier is already "
                    f"at its floor ({float(_NONFINITE_LR_MULT_FLOOR):.4f}) and the latest retry "
                    "did not trigger an optimizer-state reset. Resume from the last completed "
                    "checkpoint after correcting the numerical failure."
                )

        if effective_decoupled_training:
            if gen_optimizer is None or disc_optimizer is None:
                raise RuntimeError("Decoupled training requires generator/discriminator optimizers.")
            if gen_lr_scheduler is None or disc_lr_scheduler is None:
                raise RuntimeError("Decoupled training requires generator/discriminator schedulers.")

            while global_step < int(train_cfg.max_steps):
                uncommitted_training_window = True
                (
                    window,
                    gen_window_tokens_per_rank,
                    disc_window_tokens_per_rank,
                ) = _next_training_window(
                    default_unweighted_token_count=1.0,
                )

                disc_phase_inputs: list[dict[str, torch.Tensor | float | None]] = []
                gen_loss_num = torch.zeros((), device=accelerator.device, dtype=torch.float32)
                disc_loss_num = torch.zeros((), device=accelerator.device, dtype=torch.float32)
                disc_acc_num = torch.zeros((), device=accelerator.device, dtype=torch.float32)
                gen_token_count_window = torch.zeros((), device=accelerator.device, dtype=torch.float32)
                disc_token_count_window = torch.zeros((), device=accelerator.device, dtype=torch.float32)
                disc_positive_count_window = torch.zeros((), device=accelerator.device, dtype=torch.float32)
                skipped_window_due_nonfinite = False
                nonfinite_reason: str | None = None
                nonfinite_debug_path: Path | None = None
                did_gen_optimizer_step = False
                did_disc_optimizer_step = False
                gen_phase_out: Any | None = None
                disc_phase_out: Any | None = None
                gen_loss_weight = float(train_cfg.objective.gen_loss_weight)
                disc_loss_weight = float(train_cfg.objective.disc_loss_weight)
                gen_phase_enabled = gen_loss_weight != 0.0
                disc_phase_enabled = disc_loss_weight != 0.0

                gen_optimizer.zero_grad(set_to_none=True)
                disc_optimizer.zero_grad(set_to_none=True)

                # Phase 1: generator update + corruption target construction.
                gen_nonfinite = _NonfiniteWindowObservation()
                for step_idx, (batch, gen_count, disc_count) in enumerate(window):
                    batch, flash_meta = _prepare_training_batch(batch)

                    is_sync_step = step_idx == (ga_steps - 1)
                    sync_ctx = nullcontext() if is_sync_step else accelerator.no_sync(model)
                    with sync_ctx:
                        gen_phase_out = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            labels=batch["labels"],
                            token_type_ids=batch.get("token_type_ids"),
                            position_ids=batch.get("position_ids"),
                            sampling_temperature=train_cfg.objective.sampling_temperature,
                            phase="generator",
                            flash_meta=flash_meta,
                        )
                        gen_loss = gen_phase_out.gen_loss_raw
                        if token_weighted_ga:
                            gen_obj = gen_loss * (
                                float(gen_count) / max(float(gen_window_tokens_per_rank), 1.0)
                            )
                        else:
                            gen_obj = gen_loss

                        offending: str | None = None
                        if (
                            gen_phase_enabled
                            and not torch.isfinite(gen_phase_out.gen_loss_raw.detach()).all()
                        ):
                            offending = "gen_loss_raw"

                        backward_loss: torch.Tensor | None = None
                        if offending is None and gen_phase_enabled:
                            weighted_gen_obj = gen_obj * gen_loss_weight
                            backward_loss = _scale_loss_for_backward(
                                loss=weighted_gen_obj,
                                ga_steps=ga_steps,
                                token_weighted_ga=token_weighted_ga,
                            )
                            if not torch.isfinite(backward_loss.detach()).all():
                                offending = "gen_backward_loss"

                        gen_nonfinite.record(offending, micro_step=step_idx)

                        # Keep non-finite coordination out of non-sync microsteps to preserve
                        # no_sync accumulation performance characteristics.
                        synchronized_failure = gen_nonfinite.synchronized_failure(
                            accelerator=accelerator,
                            is_sync_step=is_sync_step,
                            current_micro_step=step_idx,
                        )
                        if synchronized_failure is not None:
                            offending_effective, offending_micro_step = synchronized_failure
                            skipped_window_due_nonfinite = True
                            _increment_nonfinite_skip()
                            nonfinite_reason = str(offending_effective)
                            lr_now = _scheduler_current_lr(gen_lr_scheduler)
                            nonfinite_debug_path = _write_nonfinite_artifact(
                                micro_step_idx=int(offending_micro_step),
                                offending=str(offending_effective),
                                gen_loss_raw=gen_phase_out.gen_loss_raw,
                                disc_loss_raw=None,
                                forward_loss=gen_phase_out.gen_loss_raw,
                                backward_loss=backward_loss,
                                grad_norm=None,
                                lr=lr_now,
                            )
                            gen_optimizer.zero_grad(set_to_none=True)
                            disc_optimizer.zero_grad(set_to_none=True)
                            break
                        if offending is not None:
                            continue

                        if gen_phase_enabled:
                            micro_gen_token_count = gen_phase_out.gen_token_count.detach().float()
                            gen_token_count_window = gen_token_count_window + micro_gen_token_count
                            gen_loss_num = gen_loss_num + (
                                gen_phase_out.gen_loss_raw.detach().float() * micro_gen_token_count
                            )
                        # Keep discriminator micro-step counts aligned across ranks:
                        # when local generator targets are absent we still run the
                        # corresponding discriminator pass with zero objective weight.
                        disc_objective_weight = 1.0 if bool(gen_phase_out.has_masked_targets) else 0.0
                        disc_phase_inputs.append(
                            {
                                "input_ids": batch["input_ids"],
                                "attention_mask": batch.get("attention_mask"),
                                "token_type_ids": batch.get("token_type_ids"),
                                "position_ids": batch.get("position_ids"),
                                "doc_context_index": batch.get("doc_context_index"),
                                "flash_meta": flash_meta,
                                "corrupted_input_ids": gen_phase_out.corrupted_input_ids,
                                "disc_labels": gen_phase_out.disc_labels,
                                "disc_count": float(disc_count),
                                "disc_objective_weight": float(disc_objective_weight),
                            }
                        )
                        if backward_loss is not None:
                            accelerator.backward(backward_loss)

                    if is_sync_step and not skipped_window_due_nonfinite:
                        if not gen_phase_enabled:
                            continue
                        grad_norm_reason, _ = _clip_gradients_and_find_nonfinite(
                            accelerator=accelerator,
                            model=model,
                            max_grad_norm=float(resolved_optim_cfg.max_grad_norm),
                        )
                        if grad_norm_reason is not None:
                            skipped_window_due_nonfinite = True
                            _increment_nonfinite_skip()
                            nonfinite_reason = f"gen_{grad_norm_reason}_nonfinite"
                            gen_optimizer.zero_grad(set_to_none=True)
                            disc_optimizer.zero_grad(set_to_none=True)
                            break
                        gen_optimizer.step()
                        gen_lr_scheduler.step()
                        _record_unscaled_lrs(gen_optimizer, gen_lr_scheduler)
                        if lr_mult < 1.0:
                            _apply_lr_mult(gen_optimizer, lr_mult)
                        gen_optimizer.zero_grad(set_to_none=True)
                        did_gen_optimizer_step = True
                        _sync_discriminator_embeddings_if_available(model, accelerator=accelerator)

                window_has_global_disc_targets = False
                if disc_phase_enabled and not skipped_window_due_nonfinite and disc_phase_inputs:
                    window_has_local_disc_targets = any(
                        float(payload.get("disc_objective_weight", 0.0)) > 0.0
                        for payload in disc_phase_inputs
                    )
                    # Skip all-negative discriminator windows only when every rank has
                    # zero generator-supervised tokens; otherwise keep per-rank step
                    # counts aligned and use zero-weight local contributions.
                    window_has_global_disc_targets = bool(
                        _sum_local_scalar(
                            accelerator=accelerator,
                            x=1.0 if window_has_local_disc_targets else 0.0,
                        )
                        > 0.0
                    )

                # Phase 2: discriminator update from cached corruption targets.
                if (
                    disc_phase_enabled
                    and not skipped_window_due_nonfinite
                    and disc_phase_inputs
                    and bool(window_has_global_disc_targets)
                ):
                    disc_phase_steps = len(disc_phase_inputs)
                    disc_nonfinite = _NonfiniteWindowObservation()
                    for step_idx, payload in enumerate(disc_phase_inputs):
                        if compile_enabled:
                            _maybe_cudagraph_mark_step_begin()
                        is_sync_step = step_idx == (disc_phase_steps - 1)
                        sync_ctx = nullcontext() if is_sync_step else accelerator.no_sync(model)
                        with sync_ctx:
                            disc_phase_out = model(
                                input_ids=payload["input_ids"],  # type: ignore[arg-type]
                                corrupted_input_ids=payload["corrupted_input_ids"],  # type: ignore[arg-type]
                                disc_labels=payload["disc_labels"],  # type: ignore[arg-type]
                                attention_mask=payload["attention_mask"],  # type: ignore[arg-type]
                                token_type_ids=payload["token_type_ids"],  # type: ignore[arg-type]
                                position_ids=payload["position_ids"],  # type: ignore[arg-type]
                                doc_context_index=payload["doc_context_index"],  # type: ignore[arg-type]
                                phase="discriminator",
                                flash_meta=payload["flash_meta"],  # type: ignore[arg-type]
                            )
                            disc_loss = disc_phase_out.disc_loss_raw
                            disc_objective_weight = float(payload.get("disc_objective_weight", 1.0))
                            if token_weighted_ga:
                                disc_obj = disc_loss * (
                                    float(payload["disc_count"])
                                    / max(float(disc_window_tokens_per_rank), 1.0)
                                )
                            else:
                                disc_obj = disc_loss
                            disc_obj = disc_obj * float(disc_objective_weight)
                            offending = None
                            if not torch.isfinite(disc_phase_out.disc_loss_raw.detach()).all():
                                offending = "disc_loss_raw"

                            backward_loss: torch.Tensor | None = None
                            if offending is None:
                                weighted_disc_obj = disc_obj * disc_loss_weight
                                backward_loss = _scale_loss_for_backward(
                                    loss=weighted_disc_obj,
                                    ga_steps=ga_steps,
                                    token_weighted_ga=token_weighted_ga,
                                )
                                if not torch.isfinite(backward_loss.detach()).all():
                                    offending = "disc_backward_loss"

                            disc_nonfinite.record(offending, micro_step=step_idx)

                            # Keep non-finite coordination out of non-sync microsteps to preserve
                            # no_sync accumulation performance characteristics.
                            synchronized_failure = disc_nonfinite.synchronized_failure(
                                accelerator=accelerator,
                                is_sync_step=is_sync_step,
                                current_micro_step=step_idx,
                            )
                            if synchronized_failure is not None:
                                offending_effective, offending_micro_step = synchronized_failure
                                skipped_window_due_nonfinite = True
                                _increment_nonfinite_skip()
                                nonfinite_reason = str(offending_effective)
                                lr_now = _scheduler_current_lr(disc_lr_scheduler)
                                nonfinite_debug_path = _write_nonfinite_artifact(
                                    micro_step_idx=int(offending_micro_step),
                                    offending=str(offending_effective),
                                    gen_loss_raw=gen_phase_out.gen_loss_raw
                                    if gen_phase_out is not None
                                    else None,
                                    disc_loss_raw=disc_phase_out.disc_loss_raw,
                                    forward_loss=disc_phase_out.disc_loss_raw,
                                    backward_loss=backward_loss,
                                    grad_norm=None,
                                    lr=lr_now,
                                )
                                gen_optimizer.zero_grad(set_to_none=True)
                                disc_optimizer.zero_grad(set_to_none=True)
                                break
                            if offending is not None:
                                continue

                            micro_disc_token_count = disc_phase_out.disc_token_count.detach().float() * float(
                                disc_objective_weight
                            )
                            disc_token_count_window = disc_token_count_window + micro_disc_token_count
                            disc_positive_count_window = (
                                disc_positive_count_window
                                + disc_phase_out.disc_positive_count.detach().float()
                                * float(disc_objective_weight)
                            )
                            disc_loss_num = disc_loss_num + (
                                disc_phase_out.disc_loss_raw.detach().float() * micro_disc_token_count
                            )
                            disc_acc_num = disc_acc_num + (
                                disc_phase_out.disc_accuracy.detach().float() * micro_disc_token_count
                            )
                            if backward_loss is not None:
                                accelerator.backward(backward_loss)

                        if is_sync_step and not skipped_window_due_nonfinite:
                            grad_norm_reason, _ = _clip_gradients_and_find_nonfinite(
                                accelerator=accelerator,
                                model=model,
                                max_grad_norm=float(resolved_optim_cfg.max_grad_norm),
                            )
                            if grad_norm_reason is not None:
                                skipped_window_due_nonfinite = True
                                _increment_nonfinite_skip()
                                nonfinite_reason = f"disc_{grad_norm_reason}_nonfinite"
                                gen_optimizer.zero_grad(set_to_none=True)
                                disc_optimizer.zero_grad(set_to_none=True)
                                break
                            disc_optimizer.step()
                            disc_lr_scheduler.step()
                            _record_unscaled_lrs(disc_optimizer, disc_lr_scheduler)
                            if lr_mult < 1.0:
                                _apply_lr_mult(disc_optimizer, lr_mult)
                            disc_optimizer.zero_grad(set_to_none=True)
                            did_disc_optimizer_step = True

                gen_phase_complete = bool(not gen_phase_enabled or did_gen_optimizer_step)
                disc_phase_complete = bool(
                    not disc_phase_enabled
                    or did_disc_optimizer_step
                    or not disc_phase_inputs
                    or not window_has_global_disc_targets
                )
                if not (gen_phase_complete and disc_phase_complete):
                    if skipped_window_due_nonfinite:
                        _finalize_nonfinite_skip(
                            optimizer_phases=[
                                (gen_optimizer, gen_lr_scheduler, did_gen_optimizer_step),
                                (disc_optimizer, disc_lr_scheduler, did_disc_optimizer_step),
                            ],
                            reason=nonfinite_reason,
                            debug_path=nonfinite_debug_path,
                        )
                        continue
                    raise RuntimeError(
                        "Decoupled accumulation window produced no synchronized optimization step."
                    )
                if not (did_gen_optimizer_step or did_disc_optimizer_step):
                    raise RuntimeError(
                        "Decoupled accumulation window had no active optimizer step. The enabled "
                        "objective produced no trainable targets."
                    )

                _record_successful_optimizer_window()
                _commit_training_window()

                _maybe_log_training_metrics(
                    lr_scheduler=disc_lr_scheduler,
                    gen_loss_num=gen_loss_num,
                    gen_token_count_window=gen_token_count_window,
                    disc_loss_num=disc_loss_num,
                    disc_acc_num=disc_acc_num,
                    disc_token_count_window=disc_token_count_window,
                    disc_positive_count_window=disc_positive_count_window,
                )

                last_saved_step = _save_checkpoint_if_due(
                    global_step=global_step,
                    consumed_micro_batches_committed=consumed_micro_batches_committed,
                    lr_mult=lr_mult,
                    last_saved_step=last_saved_step,
                )

        while not effective_decoupled_training and global_step < int(train_cfg.max_steps):
            uncommitted_training_window = True
            (
                window,
                gen_window_tokens_per_rank,
                disc_window_tokens_per_rank,
            ) = _next_training_window(
                default_unweighted_token_count=0.0,
            )

            out = None
            loss_for_metrics = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            gen_loss_num = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            disc_loss_num = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            disc_acc_num = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            gen_token_count_window = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            disc_token_count_window = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            disc_positive_count_window = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            did_optimizer_step = False
            skipped_window_due_nonfinite = False
            nonfinite_reason: str | None = None
            nonfinite_debug_path: Path | None = None
            window_nonfinite = _NonfiniteWindowObservation()
            gen_phase_enabled = float(train_cfg.objective.gen_loss_weight) != 0.0
            disc_phase_enabled = float(train_cfg.objective.disc_loss_weight) != 0.0

            for step_idx, (batch, gen_count, disc_count) in enumerate(window):
                batch, flash_meta = _prepare_training_batch(batch)

                is_sync_step = step_idx == (ga_steps - 1)
                sync_ctx = nullcontext() if is_sync_step else accelerator.no_sync(model)

                with sync_ctx:
                    out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                        token_type_ids=batch.get("token_type_ids"),
                        position_ids=batch.get("position_ids"),
                        doc_context_index=batch.get("doc_context_index"),
                        sampling_temperature=train_cfg.objective.sampling_temperature,
                        gen_loss_weight=train_cfg.objective.gen_loss_weight,
                        disc_loss_weight=train_cfg.objective.disc_loss_weight,
                        flash_meta=flash_meta,
                    )

                    if token_weighted_ga:
                        micro_obj = _token_weighted_micro_objective(
                            gen_loss=out.gen_loss_raw,
                            disc_loss=out.disc_loss_raw,
                            gen_count=gen_count,
                            disc_count=disc_count,
                            gen_window_tokens_per_rank=gen_window_tokens_per_rank,
                            disc_window_tokens_per_rank=disc_window_tokens_per_rank,
                            gen_loss_weight=float(train_cfg.objective.gen_loss_weight),
                            disc_loss_weight=float(train_cfg.objective.disc_loss_weight),
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
                    offending: str | None = None
                    if gen_phase_enabled and not torch.isfinite(out.gen_loss_raw.detach()).all():
                        offending = "gen_loss_raw"
                    elif disc_phase_enabled and not torch.isfinite(out.disc_loss_raw.detach()).all():
                        offending = "disc_loss_raw"
                    elif not torch.isfinite(out.loss.detach()).all():
                        offending = "forward_loss"
                    elif not torch.isfinite(backward_loss.detach()).all():
                        offending = "backward_loss"

                    window_nonfinite.record(offending, micro_step=step_idx)

                    # Keep non-finite coordination out of non-sync microsteps to preserve
                    # no_sync accumulation performance characteristics.
                    synchronized_failure = window_nonfinite.synchronized_failure(
                        accelerator=accelerator,
                        is_sync_step=is_sync_step,
                        current_micro_step=step_idx,
                    )
                    if synchronized_failure is not None:
                        offending_effective, offending_micro_step = synchronized_failure
                        skipped_window_due_nonfinite = True
                        _increment_nonfinite_skip()
                        nonfinite_reason = str(offending_effective)
                        lr_now = _scheduler_current_lr(lr_scheduler)
                        nonfinite_debug_path = _write_nonfinite_artifact(
                            micro_step_idx=int(offending_micro_step),
                            offending=str(offending_effective),
                            gen_loss_raw=out.gen_loss_raw,
                            disc_loss_raw=out.disc_loss_raw,
                            forward_loss=out.loss,
                            backward_loss=backward_loss,
                            grad_norm=None,
                            lr=lr_now,
                        )
                        logger.warning(
                            "Skipping accumulation window due non-finite %s "
                            "(step=%d, micro_step=%d, streak=%d, total_skips=%d). Debug artifact: %s",
                            offending_effective,
                            int(global_step + 1),
                            int(step_idx),
                            int(nonfinite_skip_streak),
                            int(nonfinite_skip_total),
                            nonfinite_debug_path,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        break
                    if offending is not None:
                        continue
                    if gen_phase_enabled:
                        micro_gen_tokens = out.gen_token_count.detach().float()
                        gen_token_count_window = gen_token_count_window + micro_gen_tokens
                        gen_loss_num = gen_loss_num + out.gen_loss.detach().float() * micro_gen_tokens
                    if disc_phase_enabled:
                        micro_disc_tokens = out.disc_token_count.detach().float()
                        disc_token_count_window = disc_token_count_window + micro_disc_tokens
                        disc_positive_count_window = (
                            disc_positive_count_window + out.disc_positive_count.detach().float()
                        )
                        disc_loss_num = disc_loss_num + out.disc_loss.detach().float() * micro_disc_tokens
                        disc_acc_num = disc_acc_num + out.disc_accuracy.detach().float() * micro_disc_tokens
                    accelerator.backward(backward_loss)

                if is_sync_step:
                    grad_norm_reason, grad_norm_for_check = _clip_gradients_and_find_nonfinite(
                        accelerator=accelerator,
                        model=model,
                        max_grad_norm=float(resolved_optim_cfg.max_grad_norm),
                    )
                    if grad_norm_reason is not None:
                        skipped_window_due_nonfinite = True
                        _increment_nonfinite_skip()
                        nonfinite_reason = f"{grad_norm_reason}_skip_{int(nonfinite_skip_total)}"
                        lr_now = _scheduler_current_lr(lr_scheduler)
                        nonfinite_debug_path = _write_nonfinite_artifact(
                            micro_step_idx=int(step_idx),
                            offending=str(nonfinite_reason),
                            gen_loss_raw=out.gen_loss_raw if out is not None else None,
                            disc_loss_raw=out.disc_loss_raw if out is not None else None,
                            forward_loss=out.loss if out is not None else None,
                            backward_loss=None,
                            grad_norm=float(grad_norm_for_check),
                            lr=lr_now,
                        )
                        logger.warning(
                            "Skipping optimizer step due non-finite %s "
                            "(step=%d, streak=%d, total_skips=%d). Debug artifact: %s",
                            str(grad_norm_reason).replace("_", " "),
                            int(global_step + 1),
                            int(nonfinite_skip_streak),
                            int(nonfinite_skip_total),
                            nonfinite_debug_path,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        break

                    optimizer.step()
                    lr_scheduler.step()
                    _record_unscaled_lrs(optimizer, lr_scheduler)
                    if lr_mult < 1.0:
                        _apply_lr_mult(optimizer, lr_mult)
                    optimizer.zero_grad(set_to_none=True)
                    did_optimizer_step = True
                    _record_successful_optimizer_window()

                    _sync_discriminator_embeddings_if_available(model, accelerator=accelerator)

            if not did_optimizer_step:
                if skipped_window_due_nonfinite:
                    _finalize_nonfinite_skip(
                        optimizer_phases=[(optimizer, lr_scheduler, False)],
                        reason=nonfinite_reason,
                        debug_path=nonfinite_debug_path,
                    )
                    continue
                raise RuntimeError("Accumulation window produced no synchronized optimization step.")

            loss_for_metrics = _finalize_window_metric_loss(
                accumulated_loss=loss_for_metrics,
                ga_steps=ga_steps,
                token_weighted_ga=token_weighted_ga,
            )

            _commit_training_window()

            _maybe_log_training_metrics(
                lr_scheduler=lr_scheduler,
                gen_loss_num=gen_loss_num,
                gen_token_count_window=gen_token_count_window,
                disc_loss_num=disc_loss_num,
                disc_acc_num=disc_acc_num,
                disc_token_count_window=disc_token_count_window,
                disc_positive_count_window=disc_positive_count_window,
                loss_override=float(
                    accelerator.gather(loss_for_metrics.detach().float().reshape(1)).mean().item()
                ),
            )

            last_saved_step = _save_checkpoint_if_due(
                global_step=global_step,
                consumed_micro_batches_committed=consumed_micro_batches_committed,
                lr_mult=lr_mult,
                last_saved_step=last_saved_step,
            )

    except KeyboardInterrupt as exc:
        exit_code = 130
        crash_type = type(exc).__name__
        crash_reason = str(exc) or "Interrupted by user (CTRL+C)"
        crash_step = int(global_step)
        logger.warning("Training interrupted at step %s", crash_step)
        raise
    except Exception as exc:
        exit_code = 1
        crash_type = type(exc).__name__
        crash_reason = str(exc)
        crash_step = int(global_step)
        logger.exception("Training crashed at step %s", crash_step)
        raise
    finally:
        if train_progress is not None:
            with suppress(Exception):
                train_progress.close()

        # Attempt a final checkpoint if we progressed and this exact step wasn't already saved.
        final_checkpoint_error: Exception | None = None
        final_step = int(global_step)
        should_try_crash_save = (crash_reason is None) or int(getattr(accelerator, "num_processes", 1)) == 1
        if (
            final_step > 0
            and final_step != int(last_saved_step)
            and should_try_crash_save
            and not uncommitted_training_window
        ):
            try:
                final_ckpt = output_dir / f"checkpoint-{final_step}"
                _save_training_checkpoint(
                    accelerator=accelerator,
                    checkpoint_dir=final_ckpt,
                    output_dir=output_dir,
                    consumed_micro_batches=consumed_micro_batches_committed,
                    save_total_limit=int(train_cfg.checkpoint.save_total_limit),
                    log_label="final",
                    lr_mult=lr_mult,
                    optimizer_param_digest=param_digest,
                    global_step=int(final_step),
                    gradient_accumulation_steps=int(ga_steps),
                )
                last_saved_step = final_step
            except Exception as save_exc:
                if crash_reason is None:
                    exit_code = 1
                    final_checkpoint_error = save_exc
                logger.error(
                    "Final/crash-time checkpoint save failed at step %d: %s. "
                    "Progress since checkpoint-%d may be lost.",
                    int(final_step),
                    str(save_exc),
                    int(last_saved_step),
                    exc_info=True,
                )
        elif crash_reason is not None and accelerator.is_main_process:
            if uncommitted_training_window:
                logger.warning(
                    "Skipping crash-time final checkpoint save because the current accumulation "
                    "window was not fully committed. The last completed checkpoint is the "
                    "recovery boundary."
                )
            elif not should_try_crash_save:
                logger.warning(
                    "Skipping crash-time final checkpoint save on distributed run "
                    "(num_processes=%s) to avoid potential collective deadlocks after failure.",
                    getattr(accelerator, "num_processes", "unknown"),
                )

        final_export_error: Exception | None = None
        if (
            bool(train_cfg.checkpoint.export_hf_final)
            and crash_reason is None
            and final_checkpoint_error is None
            and bool(getattr(accelerator, "is_main_process", True))
        ):
            export_step = int(global_step)
            try:
                if export_step > 0:
                    checkpoint_dir = output_dir / f"checkpoint-{export_step}"
                    if not checkpoint_dir.exists():
                        raise FileNotFoundError(
                            "Cannot export the final training step because its checkpoint directory "
                            f"does not exist: {checkpoint_dir}"
                        )
                    _export_discriminator_hf_subprocess(
                        checkpoint_dir=checkpoint_dir,
                        output_dir=output_dir / "final_hf",
                    )
            except Exception as export_exc:
                exit_code = 1
                final_export_error = export_exc

        if crash_reason is not None:
            crash_log_step = int(
                max(crash_step if crash_step is not None else global_step, max_tracker_step_logged)
            )
            crash_row = {
                "step": crash_log_step,
                "crash": True,
                "crash_type": crash_type,
                "crash_reason": crash_reason,
                "wall_time_s": float(time.perf_counter() - train_started_at),
            }
            if accelerator.is_main_process:
                with suppress(Exception):
                    _append_metrics_jsonl_row(metrics_path, crash_row)
                if report_to != "none":
                    with suppress(Exception):
                        _log_tracker_metrics(
                            {k: v for k, v in crash_row.items() if k != "step"},
                            step=int(crash_row["step"]),
                        )
                if wandb_run is not None:
                    with suppress(Exception):
                        wandb_run.log(
                            {
                                "crash": True,
                                "crash_type": crash_type,
                                "crash_reason": crash_reason,
                            },
                            step=int(crash_row["step"]),
                        )

        if report_to != "none":
            if wandb_run is not None:
                with suppress(Exception):
                    if crash_reason is not None:
                        wandb_run.summary["crashed"] = True
                        wandb_run.summary["crash_type"] = crash_type
                        wandb_run.summary["crash_reason"] = crash_reason
                    wandb_run.finish(exit_code=exit_code)
            else:
                with suppress(Exception):
                    accelerator.end_training()

        _flush_loggers()
        if final_checkpoint_error is not None:
            raise final_checkpoint_error
        if final_export_error is not None:
            raise final_export_error
