"""Runtime construction helpers for pretraining."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader

from deberta.config import (
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    _sync_legacy_train_aliases,
    apply_profile_defaults,
    validate_data_config,
    validate_logging_config,
    validate_model_config,
    validate_optim_config,
    validate_train_config,
    validate_training_workflow_options,
)
from deberta.data import DebertaV3ElectraCollator, PackedStreamingDataset, SequentialStreamingDataset
from deberta.data.collator import MLMConfig
from deberta.data.streaming import PackedStreamingConfig

logger = logging.getLogger(__name__)


def _is_no_decay_param(*, name: str, param: torch.Tensor) -> bool:
    """Check whether a parameter should skip weight decay.

    :param str name: Parameter name.
    :param torch.Tensor param: Parameter tensor.
    :return bool: True when parameter belongs to no-decay groups.
    """
    lname = str(name).lower()
    # Keep vector/scalar biases in no-decay; high-rank "bias" tensors (for example
    # GDES embedding deltas) should follow standard weight-decay behavior.
    if lname.endswith(".bias") and param.dim() <= 1:
        return True
    if "layernorm" in lname or "layer_norm" in lname or "rmsnorm" in lname or "rms_norm" in lname:
        return True
    # Scalars and 1D params are typically excluded from decay.
    if param.dim() <= 1:
        return True
    return False


def _is_generator_param(name: str) -> bool:
    """Check whether a parameter belongs to the generator branch.

    :param str name: Parameter name.
    :return bool: True when parameter is generator-owned.
    """
    # Generator-owned modules:
    # - generator backbone
    # - generator MLM head
    # - enhanced mask decoder (used only on the generator path)
    return (
        name.startswith("generator.")
        or name.startswith("generator_lm_head.")
        or name.startswith("enhanced_mask_decoder.")
    )


def _partition_optimizer_params(model: torch.nn.Module) -> dict[str, dict[str, list[Any]]]:
    """Partition trainable parameters into optimizer groups with ordered names.

    :param torch.nn.Module model: Model whose trainable parameters should be partitioned.
    :return dict[str, dict[str, list[Any]]]: Group map keyed by
        ``gen_decay|gen_no_decay|disc_decay|disc_no_decay`` with ``params`` and ``names`` lists.
    """
    groups: dict[str, dict[str, list[Any]]] = {
        "gen_decay": {"params": [], "names": []},
        "gen_no_decay": {"params": [], "names": []},
        "disc_decay": {"params": [], "names": []},
        "disc_no_decay": {"params": [], "names": []},
    }
    seen_param_ids: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Guard against shared-parameter aliasing (for example ES embedding sharing)
        # to keep each Parameter in exactly one optimizer group.
        pid = id(param)
        if pid in seen_param_ids:
            continue
        seen_param_ids.add(pid)

        no_decay = _is_no_decay_param(name=name, param=param)
        is_gen = _is_generator_param(name)
        if is_gen and no_decay:
            key = "gen_no_decay"
        elif is_gen:
            key = "gen_decay"
        elif no_decay:
            key = "disc_no_decay"
        else:
            key = "disc_decay"

        groups[key]["params"].append(param)
        groups[key]["names"].append(str(name))

    return groups


def _digest_param_name_order(names: list[str]) -> str:
    """Compute a short digest for an ordered list of parameter names.

    :param list[str] names: Ordered parameter names.
    :return str: 16-char SHA-256 hex prefix.
    """
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


def _optimizer_param_order_digest(model: torch.nn.Module) -> str:
    """Compute digest of trainable parameter names in optimizer insertion order.

    This mirrors `_build_optimizer` ordering (grouped as gen-decay, gen-no-decay,
    disc-decay, disc-no-decay), not raw ``named_parameters()`` registration order.

    :param torch.nn.Module model: Model whose optimizer ordering to digest.
    :return str: 16-char hex digest.
    """
    partitions = _partition_optimizer_params(model)
    ordered_names: list[str] = []
    for key in ("gen_decay", "gen_no_decay", "disc_decay", "disc_no_decay"):
        ordered_names.extend(partitions[key]["names"])
    return _digest_param_name_order(ordered_names)


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


def _resolve_optimizer_hyperparams(
    *,
    cfg: TrainConfig,
    mixed_precision: str,
) -> tuple[float, tuple[float, float], float, float, dict[str, Any]]:
    """Resolve shared AdamW hyperparameters for optimizer construction.

    :param TrainConfig cfg: Training configuration.
    :param str mixed_precision: Effective mixed-precision mode.
    :return tuple[float, tuple[float, float], float, float, dict[str, Any]]:
        ``(eps, betas, gen_lr, disc_lr, fused_kwargs)``.
    """
    eps = float(cfg.adam_epsilon)
    if str(mixed_precision).strip().lower() == "bf16" and eps < 1e-6:
        eps = 1e-6
        logger.warning("Raised Adam epsilon to 1e-6 for bf16 stability.")

    base_lr = float(cfg.learning_rate)
    gen_lr_raw = float(cfg.generator_learning_rate)
    disc_lr_raw = float(getattr(cfg, "discriminator_learning_rate", -1.0))
    gen_lr = gen_lr_raw if gen_lr_raw > 0 else base_lr
    disc_lr = disc_lr_raw if disc_lr_raw > 0 else base_lr
    betas = (float(cfg.adam_beta1), float(cfg.adam_beta2))
    fused_kwargs = _maybe_fused_adamw_kwargs()
    return eps, betas, gen_lr, disc_lr, fused_kwargs


def _build_branch_param_groups(
    *,
    partitions: dict[str, dict[str, list[Any]]],
    branch_key: str,
    lr: float,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build ordered AdamW groups for one model branch.

    :param dict[str, dict[str, list[Any]]] partitions: Partitioned model parameters.
    :param str branch_key: Branch prefix (``gen`` or ``disc``).
    :param float lr: Branch learning rate.
    :param float weight_decay: Weight decay for decay groups.
    :return tuple[list[dict[str, Any]], list[str]]: ``(groups, ordered_names)``.
    """
    groups: list[dict[str, Any]] = []
    ordered_names: list[str] = []
    for key, decay in ((f"{branch_key}_decay", weight_decay), (f"{branch_key}_no_decay", 0.0)):
        params = partitions[key]["params"]
        if not params:
            continue
        groups.append({"params": params, "weight_decay": decay, "lr": lr})
        ordered_names.extend(partitions[key]["names"])
    return groups, ordered_names


def _build_optimizer(
    model: torch.nn.Module,
    cfg: TrainConfig,
    *,
    mixed_precision: str = "no",
) -> torch.optim.Optimizer:
    """Create AdamW with parameter grouping for RTD training.

    :param torch.nn.Module model: RTD model.
    :param TrainConfig cfg: Training configuration.
    :param str mixed_precision: Effective mixed-precision mode.
    :return torch.optim.Optimizer: Configured AdamW optimizer.
    """
    eps, betas, gen_lr, disc_lr, fused_kwargs = _resolve_optimizer_hyperparams(
        cfg=cfg,
        mixed_precision=mixed_precision,
    )
    partitions = _partition_optimizer_params(model)
    gen_groups, gen_names = _build_branch_param_groups(
        partitions=partitions,
        branch_key="gen",
        lr=gen_lr,
        weight_decay=float(cfg.weight_decay),
    )
    disc_groups, disc_names = _build_branch_param_groups(
        partitions=partitions,
        branch_key="disc",
        lr=disc_lr,
        weight_decay=float(cfg.weight_decay),
    )
    optimizer = torch.optim.AdamW(
        [*gen_groups, *disc_groups],
        lr=disc_lr,
        betas=betas,
        eps=eps,
        **fused_kwargs,
    )
    optimizer._param_order_digest = _digest_param_name_order([*gen_names, *disc_names])
    return optimizer


def _build_decoupled_optimizers(
    model: torch.nn.Module,
    cfg: TrainConfig,
    *,
    mixed_precision: str = "no",
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Create separate generator/discriminator AdamW optimizers.

    :param torch.nn.Module model: RTD model.
    :param TrainConfig cfg: Training configuration.
    :param str mixed_precision: Effective mixed-precision mode.
    :return tuple[torch.optim.Optimizer, torch.optim.Optimizer]: (generator_optimizer, discriminator_optimizer).
    """
    eps, betas, gen_lr, disc_lr, fused_kwargs = _resolve_optimizer_hyperparams(
        cfg=cfg,
        mixed_precision=mixed_precision,
    )
    partitions = _partition_optimizer_params(model)
    gen_groups, gen_names = _build_branch_param_groups(
        partitions=partitions,
        branch_key="gen",
        lr=gen_lr,
        weight_decay=float(cfg.weight_decay),
    )
    disc_groups, disc_names = _build_branch_param_groups(
        partitions=partitions,
        branch_key="disc",
        lr=disc_lr,
        weight_decay=float(cfg.weight_decay),
    )
    gen_opt = torch.optim.AdamW(
        gen_groups,
        lr=gen_lr,
        betas=betas,
        eps=eps,
        **fused_kwargs,
    )
    disc_opt = torch.optim.AdamW(
        disc_groups,
        lr=disc_lr,
        betas=betas,
        eps=eps,
        **fused_kwargs,
    )
    gen_opt._param_order_digest = _digest_param_name_order(gen_names)
    disc_opt._param_order_digest = _digest_param_name_order(disc_names)
    return gen_opt, disc_opt


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


def _cycle_dataloader(
    dl: DataLoader,
    *,
    start_epoch: int = 0,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield batches forever by cycling through a dataloader.

    :param DataLoader dl: Source dataloader.
    :param int start_epoch: Initial dataset epoch passed into ``set_epoch`` when available.
    :return Iterator[dict[str, torch.Tensor]]: Infinite batch iterator.
    """
    epoch = int(start_epoch)
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


def _resolve_section_cfg_compat(
    *,
    train_cfg: TrainConfig,
    optim_cfg: OptimConfig | None,
    logging_cfg: LoggingConfig | None,
) -> tuple[OptimConfig, LoggingConfig]:
    """Resolve optional optim/logging configs with train-legacy compatibility.

    :param TrainConfig train_cfg: Train config object.
    :param OptimConfig | None optim_cfg: Optional explicit optim config.
    :param LoggingConfig | None logging_cfg: Optional explicit logging config.
    :return tuple[OptimConfig, LoggingConfig]: Effective optim/logging configs.
    """
    if optim_cfg is None:
        resolved_optim_cfg = OptimConfig(
            learning_rate=getattr(train_cfg, "learning_rate", 5e-4),
            generator_learning_rate=getattr(train_cfg, "generator_learning_rate", -1.0),
            discriminator_learning_rate=getattr(train_cfg, "discriminator_learning_rate", -1.0),
            weight_decay=getattr(train_cfg, "weight_decay", 0.01),
            adam_beta1=getattr(train_cfg, "adam_beta1", 0.9),
            adam_beta2=getattr(train_cfg, "adam_beta2", 0.999),
            adam_epsilon=getattr(train_cfg, "adam_epsilon", 1e-8),
            lr_scheduler_type=getattr(train_cfg, "lr_scheduler_type", "linear"),
            warmup_steps=getattr(train_cfg, "warmup_steps", 1_000),
            max_grad_norm=getattr(train_cfg, "max_grad_norm", 1.0),
        )
    else:
        resolved_optim_cfg = optim_cfg

    if logging_cfg is None:
        resolved_logging_cfg = LoggingConfig(
            project_name=getattr(train_cfg, "project_name", "deberta-train"),
            run_name=getattr(train_cfg, "run_name", None),
            output_dir=getattr(train_cfg, "logging_output_dir", None),
            logging_steps=getattr(train_cfg, "logging_steps", 50),
            report_to=getattr(train_cfg, "report_to", "none"),
            wandb_watch=getattr(train_cfg, "wandb_watch", "gradients"),
            wandb_watch_log_freq=getattr(train_cfg, "wandb_watch_log_freq", 100),
            debug_metrics=getattr(train_cfg, "debug_metrics", False),
        )
    else:
        resolved_logging_cfg = logging_cfg

    return resolved_optim_cfg, resolved_logging_cfg


def _apply_profile_and_validate_training_configs(
    *,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    optim_cfg: OptimConfig,
    logging_cfg: LoggingConfig,
) -> None:
    """Apply profile defaults and validate full training config contract.

    :param ModelConfig model_cfg: Model config.
    :param DataConfig data_cfg: Data config.
    :param TrainConfig train_cfg: Train config.
    :param OptimConfig optim_cfg: Effective optimizer config.
    :param LoggingConfig logging_cfg: Effective logging config.
    :return None: None.
    """
    apply_profile_defaults(model_cfg=model_cfg, train_cfg=train_cfg, optim_cfg=optim_cfg)
    _sync_legacy_train_aliases(
        train_cfg=train_cfg,
        optim_cfg=optim_cfg,
        logging_cfg=logging_cfg,
    )
    validate_model_config(model_cfg)
    validate_data_config(data_cfg)
    validate_train_config(train_cfg)
    validate_optim_config(optim_cfg)
    validate_logging_config(logging_cfg)
    validate_training_workflow_options(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        logging_cfg=logging_cfg,
    )


def _build_train_dataset_and_collator(
    *,
    raw_train: Any,
    tokenizer: Any,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    process_index: int,
    num_processes: int,
) -> tuple[Any, Any]:
    """Build streaming train dataset and collator.

    :param Any raw_train: Loaded HF dataset split.
    :param Any tokenizer: Runtime tokenizer.
    :param DataConfig data_cfg: Data config.
    :param TrainConfig train_cfg: Train config.
    :param int process_index: Current process index.
    :param int num_processes: Total process count.
    :return tuple[Any, Any]: ``(train_dataset, collator)``.
    """
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
        process_index=process_index,
        num_processes=num_processes,
    )
    collator = _build_training_collator(
        tokenizer=tokenizer,
        train_cfg=train_cfg,
        packed_sequences=bool(data_cfg.pack_sequences),
        block_cross_document_attention=bool(data_cfg.block_cross_document_attention),
    )
    return train_dataset, collator
