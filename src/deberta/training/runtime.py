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
    if "layernorm" in lname or "layer_norm" in lname or "rmsnorm" in lname or "rms_norm" in lname:
        return True
    # Scalars and 1D params, including conventional biases, skip decay. High-rank
    # "bias" tensors (for example GDES embedding deltas) retain standard decay.
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
    cfg: OptimConfig,
    mixed_precision: str,
) -> tuple[float, tuple[float, float], float, float, dict[str, Any]]:
    """Resolve shared AdamW hyperparameters for optimizer construction.

    :param OptimConfig cfg: Optimizer configuration.
    :param str mixed_precision: Effective mixed-precision mode.
    :return tuple[float, tuple[float, float], float, float, dict[str, Any]]:
        ``(eps, betas, gen_lr, disc_lr, fused_kwargs)``.
    """
    eps = float(cfg.adam.epsilon)
    if str(mixed_precision).strip().lower() == "bf16" and eps < 1e-6:
        eps = 1e-6
        logger.warning("Raised Adam epsilon to 1e-6 for bf16 stability.")

    base_lr = float(cfg.lr.base)
    gen_lr_raw = float(cfg.lr.generator)
    disc_lr_raw = float(cfg.lr.discriminator)
    gen_lr = gen_lr_raw if gen_lr_raw > 0 else base_lr
    disc_lr = disc_lr_raw if disc_lr_raw > 0 else base_lr
    betas = (float(cfg.adam.beta1), float(cfg.adam.beta2))
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


def _prepare_optimizer_groups(
    model: torch.nn.Module,
    cfg: OptimConfig,
    *,
    mixed_precision: str,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[str], float, float, dict[str, Any]]:
    """Resolve ordered branch groups and shared AdamW arguments.

    :param torch.nn.Module model: Model whose parameters are partitioned.
    :param OptimConfig cfg: Optimizer configuration.
    :param str mixed_precision: Effective precision mode.
    :return tuple: Generator/discriminator groups, names, learning rates, and shared kwargs.
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
    return (
        gen_groups,
        gen_names,
        disc_groups,
        disc_names,
        gen_lr,
        disc_lr,
        {
            "betas": betas,
            "eps": eps,
            **fused_kwargs,
        },
    )


def _build_optimizer(
    model: torch.nn.Module,
    cfg: OptimConfig,
    *,
    mixed_precision: str = "no",
) -> torch.optim.Optimizer:
    """Create AdamW with parameter grouping for RTD training.

    :param torch.nn.Module model: RTD model.
    :param OptimConfig cfg: Optimizer configuration.
    :param str mixed_precision: Effective mixed-precision mode.
    :return torch.optim.Optimizer: Configured AdamW optimizer.
    """
    gen_groups, gen_names, disc_groups, disc_names, _, disc_lr, adamw_kwargs = _prepare_optimizer_groups(
        model,
        cfg,
        mixed_precision=mixed_precision,
    )
    optimizer = torch.optim.AdamW(
        [*gen_groups, *disc_groups],
        lr=disc_lr,
        **adamw_kwargs,
    )
    optimizer._param_order_digest = _digest_param_name_order([*gen_names, *disc_names])
    return optimizer


def _build_decoupled_optimizers(
    model: torch.nn.Module,
    cfg: OptimConfig,
    *,
    mixed_precision: str = "no",
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Create separate generator/discriminator AdamW optimizers.

    :param torch.nn.Module model: RTD model.
    :param OptimConfig cfg: Optimizer configuration.
    :param str mixed_precision: Effective mixed-precision mode.
    :return tuple[torch.optim.Optimizer, torch.optim.Optimizer]: (generator_optimizer, discriminator_optimizer).
    """
    gen_groups, gen_names, disc_groups, disc_names, gen_lr, disc_lr, adamw_kwargs = _prepare_optimizer_groups(
        model, cfg, mixed_precision=mixed_precision
    )
    gen_opt = torch.optim.AdamW(
        gen_groups,
        lr=gen_lr,
        **adamw_kwargs,
    )
    disc_opt = torch.optim.AdamW(
        disc_groups,
        lr=disc_lr,
        **adamw_kwargs,
    )
    gen_opt._param_order_digest = _digest_param_name_order(gen_names)
    disc_opt._param_order_digest = _digest_param_name_order(disc_names)
    return gen_opt, disc_opt


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    train_cfg: TrainConfig,
    optim_cfg: OptimConfig,
) -> Any:
    """Build a Hugging Face learning-rate scheduler.

    :param torch.optim.Optimizer optimizer: Optimizer instance.
    :param TrainConfig train_cfg: Training step budget.
    :param OptimConfig optim_cfg: Scheduler configuration.
    :return Any: Scheduler object from ``transformers.get_scheduler``.
    """
    try:
        from transformers import get_scheduler
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required for schedulers.") from e

    return get_scheduler(
        name=optim_cfg.scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=int(optim_cfg.scheduler.warmup_steps),
        num_training_steps=int(train_cfg.max_steps),
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
            set_epoch(epoch)
        yield from dl
        epoch += 1


def _build_training_collator(
    *,
    tokenizer: Any,
    train_cfg: TrainConfig,
    packed_sequences: bool,
    block_cross_document_attention: bool,
    emit_flash_metadata: bool = False,
) -> DebertaV3ElectraCollator:
    """Build the RTD masking collator from train/data config.

    :param Any tokenizer: Tokenizer used for dynamic masking.
    :param TrainConfig train_cfg: Training configuration.
    :param bool packed_sequences: Whether dataset packing is enabled.
    :param bool block_cross_document_attention: Whether packed batches should block cross-document attention.
    :param bool emit_flash_metadata: Whether to build Flash routing metadata.
    :return DebertaV3ElectraCollator: Configured collator.
    """
    return DebertaV3ElectraCollator(
        tokenizer=tokenizer,
        cfg=MLMConfig(
            mlm_probability=train_cfg.objective.mlm_probability,
            mask_token_prob=train_cfg.objective.mask_token_prob,
            random_token_prob=train_cfg.objective.random_token_prob,
            max_ngram=train_cfg.objective.mlm_max_ngram,
        ),
        packed_sequences=bool(packed_sequences),
        block_cross_document_attention=bool(block_cross_document_attention),
        emit_flash_metadata=bool(emit_flash_metadata),
    )


def _flashdeberta_runtime_import_error() -> Exception | None:
    """Return the fixed-kernel import failure for the optional Flash runtime.

    :return Exception | None: Import failure, or None when the core runtime is available.
    """

    try:
        from deberta.modeling.flashdeberta_fixed_op import flashdeberta_fixed_import_error
    except Exception as exc:  # pragma: no cover - optional import boundary
        return exc
    return flashdeberta_fixed_import_error()


def _validate_flashdeberta_runtime(model_cfg: ModelConfig) -> None:
    """Fail early when configured FlashDeBERTa dependencies cannot be imported.

    :param ModelConfig model_cfg: Validated model configuration.
    :raises RuntimeError: If Flash attention is configured without its usable runtime.
    """

    if str(model_cfg.hf.attention_impl).strip().lower() != "flash":
        return
    detail = _flashdeberta_runtime_import_error()
    if detail is not None:
        raise RuntimeError(
            "model.hf.attention_impl='flash' requires the FlashDeBERTa runtime. Install the "
            "project with the flash extra (`pip install -e '.[flash]'`) and ensure "
            "FlashDeBERTa and Triton are compatible with the installed PyTorch build."
        ) from detail


def _validate_training_configs(
    *,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    optim_cfg: OptimConfig,
    logging_cfg: LoggingConfig,
) -> None:
    """Validate the full training config contract.

    :param ModelConfig model_cfg: Model config.
    :param DataConfig data_cfg: Data config.
    :param TrainConfig train_cfg: Train config.
    :param OptimConfig optim_cfg: Effective optimizer config.
    :param LoggingConfig logging_cfg: Effective logging config.
    :return None: None.
    """
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
    )
    _validate_flashdeberta_runtime(model_cfg)


def _build_train_dataset_and_collator(
    *,
    raw_train: Any,
    tokenizer: Any,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    process_index: int,
    num_processes: int,
    flash_enabled: bool = False,
) -> tuple[Any, Any]:
    """Build the training dataset wrapper and collator.

    :param Any raw_train: Loaded HF dataset split.
    :param Any tokenizer: Runtime tokenizer.
    :param DataConfig data_cfg: Data config.
    :param TrainConfig train_cfg: Train config.
    :param int process_index: Current process index.
    :param int num_processes: Total process count.
    :param bool flash_enabled: Whether Flash metadata should be emitted.
    :return tuple[Any, Any]: ``(train_dataset, collator)``.
    """
    dataset_cls = PackedStreamingDataset if bool(data_cfg.packing.enabled) else SequentialStreamingDataset
    train_dataset = dataset_cls(
        hf_dataset=raw_train,
        tokenizer=tokenizer,
        cfg=PackedStreamingConfig(
            text_column_name=data_cfg.source.text_column_name,
            max_seq_length=data_cfg.packing.max_seq_length,
            seed=train_cfg.seed,
            shuffle_buffer_size=data_cfg.source.shuffle_buffer_size,
            block_cross_document_attention=bool(data_cfg.packing.block_cross_document_attention),
        ),
        process_index=process_index,
        num_processes=num_processes,
    )
    collator = _build_training_collator(
        tokenizer=tokenizer,
        train_cfg=train_cfg,
        packed_sequences=bool(data_cfg.packing.enabled),
        block_cross_document_attention=bool(data_cfg.packing.block_cross_document_attention),
        emit_flash_metadata=bool(flash_enabled),
    )
    return train_dataset, collator
