"""Training loop and export utilities for DeBERTa v3 RTD pretraining."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import inspect
import json
import logging
import math
import time
from collections.abc import Iterator
from contextlib import nullcontext, suppress
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, get_type_hints

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deberta.config import (
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    _normalize_sdpa_kernel,
    _normalize_torch_compile_mode,
    _sync_legacy_train_aliases,
    apply_profile_defaults,
    resolve_effective_mixed_precision,
    validate_data_config,
    validate_logging_config,
    validate_model_config,
    validate_optim_config,
    validate_train_config,
    validate_training_workflow_options,
)
from deberta.data import DebertaV3ElectraCollator, PackedStreamingDataset, SequentialStreamingDataset
from deberta.data.collator import MLMConfig
from deberta.data.loading import load_hf_dataset
from deberta.data.streaming import PackedStreamingConfig
from deberta.modeling import DebertaV3RTDPretrainer, build_backbone_configs, build_backbones
from deberta.training.export_helpers import _export_discriminator_hf_subprocess
from deberta.training.loop_utils import (
    _count_input_tokens_for_batch,
    _count_rtd_tokens_for_batch,
    _finalize_window_metric_loss,
    _resolve_window_token_denominators,
    _scale_loss_for_backward,
    _should_clip_gradients,
    _sum_local_scalar,
    _token_weighted_micro_objective,
)
from deberta.training.run_config import _dump_yaml_mapping, _persist_or_validate_run_configs
from deberta.training.run_management import (
    _load_checkpoint_progress_metadata,
    _parse_checkpoint_step,
    _prepare_output_dir,
    _resolve_output_dir,
    _resolve_output_dir_for_accelerator,
    _resolve_resume_checkpoint,
    _resolve_resume_checkpoint_for_accelerator,
    _sanitize_run_label,
    _save_training_checkpoint,
)
from deberta.training.tracker_utils import _init_trackers, _setup_wandb_watch, _upload_wandb_original_config
from deberta.utils.checkpoint import (
    load_state_with_compile_fallback,
    unwrap_compiled_model,
)
from deberta.utils.io import dump_json
from deberta.utils.log import setup_process_logging
from deberta.utils.types import coerce_scalar, unwrap_optional_type

logger = logging.getLogger(__name__)
_NONFINITE_LR_BACKOFF = 0.5
_NONFINITE_LR_MULT_FLOOR = 0.01  # persistent multiplier floor (1% of scheduled)
_NONFINITE_LR_MULT_RECOVERY = 1.1  # gradual recovery factor per successful step
_NONFINITE_OPT_STATE_RESET_EVERY = 4
_DOC_BLOCK_EYE_CACHE: dict[tuple[int, str, int | None], torch.Tensor] = {}
_DOC_BLOCK_CLS_KEY_CACHE: dict[tuple[int, str, int | None], torch.Tensor] = {}


def _append_metrics_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    """Append one metrics row to JSONL(.gz) with immediate flush.

    Opens the file per-write rather than keeping a persistent handle.  This is
    intentional: the function is called only every ``logging_steps`` (not per
    micro-batch), so the overhead is negligible, and the immediate close
    guarantees crash-safe writes — important for long pretraining runs.

    :param Path path: Metrics path, typically ``*.jsonl`` or ``*.jsonl.gz``.
    :param dict[str, Any] row: Serializable metrics row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    opener = (
        gzip.open(path, "at", encoding="utf-8") if path.suffix == ".gz" else path.open("a", encoding="utf-8")
    )
    with opener as f:
        f.write(line)
        f.flush()


def _flush_loggers() -> None:
    """Flush all configured logger handlers best-effort.

    :return None: None.
    """
    root = logging.getLogger()
    logger_objs: list[logging.Logger] = [root]
    for obj in logging.Logger.manager.loggerDict.values():
        if isinstance(obj, logging.Logger):
            logger_objs.append(obj)

    seen_handlers: set[int] = set()
    for log_obj in logger_objs:
        for handler in log_obj.handlers:
            hid = id(handler)
            if hid in seen_handlers:
                continue
            seen_handlers.add(hid)
            with suppress(Exception):
                handler.flush()


def _drop_none_recursive(value: Any) -> Any:
    """Recursively drop ``None`` entries from mappings/lists.

    :param Any value: Arbitrary nested value.
    :return Any: Value with ``None`` keys/items removed.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            cleaned = _drop_none_recursive(item)
            if cleaned is None:
                continue
            out[str(key)] = cleaned
        return out
    if isinstance(value, list):
        out_list: list[Any] = []
        for item in value:
            cleaned = _drop_none_recursive(item)
            if cleaned is None:
                continue
            out_list.append(cleaned)
        return out_list
    if isinstance(value, tuple):
        out_tuple: list[Any] = []
        for item in value:
            cleaned = _drop_none_recursive(item)
            if cleaned is None:
                continue
            out_tuple.append(cleaned)
        return out_tuple
    return value


def _config_obj_to_mapping(config_obj: Any) -> dict[str, Any]:
    """Convert config-like objects to serializable dictionaries.

    :param Any config_obj: Config object exposing ``to_dict``/``__dict__``.
    :return dict[str, Any]: Best-effort plain mapping.
    """
    if config_obj is None:
        return {}
    if isinstance(config_obj, dict):
        return dict(config_obj)
    to_dict_fn = getattr(config_obj, "to_dict", None)
    if callable(to_dict_fn):
        raw = to_dict_fn()
        if isinstance(raw, dict):
            return dict(raw)
    raw_dict = getattr(config_obj, "__dict__", None)
    if isinstance(raw_dict, dict):
        return dict(raw_dict)
    return {}


def _coerce_dataclass_payload_types(cfg_obj: Any) -> dict[str, Any]:
    """Serialize a dataclass config and coerce scalar fields to declared types.

    This keeps tracker payloads deterministic when YAML parsed numeric-like strings
    (for example ``1e-6``) bypass dataclass runtime typing.

    :param Any cfg_obj: Dataclass config object.
    :return dict[str, Any]: Serialized mapping with best-effort scalar coercion.
    """

    def _coerce_dataclass_instance(obj: Any) -> dict[str, Any]:
        """Recursively coerce nested dataclass payload fields.

        :param Any obj: Dataclass instance.
        :return dict[str, Any]: Serialized/coerced mapping.
        """
        payload: dict[str, Any] = {}
        type_hints = get_type_hints(type(obj))
        for f in fields(type(obj)):
            name = str(f.name)
            value = getattr(obj, name)
            target_t, _allows_none = unwrap_optional_type(type_hints.get(name, f.type))
            if dataclasses.is_dataclass(value):
                payload[name] = _coerce_dataclass_instance(value)
            else:
                try:
                    payload[name] = coerce_scalar(
                        value,
                        target_t,
                        allow_none=True,
                        allow_bool_numeric=True,
                    )
                except Exception:
                    payload[name] = value
        return payload

    if not dataclasses.is_dataclass(cfg_obj):
        if isinstance(cfg_obj, dict):
            return {str(key): value for key, value in cfg_obj.items()}
        return {}
    return _coerce_dataclass_instance(cfg_obj)


def _build_runtime_resolved_tracker_config(
    *,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    optim_cfg: OptimConfig | None = None,
    logging_cfg: LoggingConfig | None = None,
    disc_config: Any,
    gen_config: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    """Build a runtime-resolved tracker payload with effective model/train values.

    :param ModelConfig model_cfg: Effective model config used by training.
    :param DataConfig data_cfg: Effective data config used by training.
    :param TrainConfig train_cfg: Effective train config used by training.
    :param OptimConfig optim_cfg: Effective optimizer config used by training.
    :param LoggingConfig logging_cfg: Effective logging config used by training.
    :param Any disc_config: Runtime discriminator backbone config.
    :param Any gen_config: Runtime generator backbone config.
    :param Any tokenizer: Runtime tokenizer.
    :return dict[str, Any]: Null-pruned resolved config payload.
    """
    resolved_optim_cfg = optim_cfg if optim_cfg is not None else OptimConfig()
    resolved_logging_cfg = logging_cfg if logging_cfg is not None else LoggingConfig()
    _sync_legacy_train_aliases(
        train_cfg=train_cfg,
        optim_cfg=resolved_optim_cfg,
        logging_cfg=resolved_logging_cfg,
    )

    payload: dict[str, Any] = {
        "model": _coerce_dataclass_payload_types(model_cfg),
        "data": _coerce_dataclass_payload_types(data_cfg),
        "train": _coerce_dataclass_payload_types(train_cfg),
        "optim": _coerce_dataclass_payload_types(resolved_optim_cfg),
        "logging": _coerce_dataclass_payload_types(resolved_logging_cfg),
    }
    model_payload = payload["model"]

    disc_cfg_map = _config_obj_to_mapping(disc_config)
    gen_cfg_map = _config_obj_to_mapping(gen_config)

    generator_payload = dict(model_payload.get("generator", {}) or {})
    if generator_payload.get("num_hidden_layers") is None and "num_hidden_layers" in gen_cfg_map:
        generator_payload["num_hidden_layers"] = int(gen_cfg_map["num_hidden_layers"])
    if generator_payload.get("hidden_size") is None and "hidden_size" in gen_cfg_map:
        generator_payload["hidden_size"] = int(gen_cfg_map["hidden_size"])
    if generator_payload.get("intermediate_size") is None and "intermediate_size" in gen_cfg_map:
        generator_payload["intermediate_size"] = int(gen_cfg_map["intermediate_size"])
    if generator_payload.get("num_attention_heads") is None and "num_attention_heads" in gen_cfg_map:
        generator_payload["num_attention_heads"] = int(gen_cfg_map["num_attention_heads"])
    model_payload["generator"] = generator_payload

    dropout_payload = dict(model_payload.get("dropout", {}) or {})
    if dropout_payload.get("hidden_prob") is None and "hidden_dropout_prob" in disc_cfg_map:
        dropout_payload["hidden_prob"] = float(disc_cfg_map["hidden_dropout_prob"])
    if dropout_payload.get("attention_probs_prob") is None and "attention_probs_dropout_prob" in disc_cfg_map:
        dropout_payload["attention_probs_prob"] = float(disc_cfg_map["attention_probs_dropout_prob"])
    model_payload["dropout"] = dropout_payload

    rope_payload = dict(model_payload.get("rope", {}) or {})
    if rope_payload.get("max_position_embeddings") is None and "max_position_embeddings" in disc_cfg_map:
        rope_payload["max_position_embeddings"] = int(disc_cfg_map["max_position_embeddings"])
    model_payload["rope"] = rope_payload

    tokenizer_payload = dict(model_payload.get("tokenizer", {}) or {})
    if tokenizer_payload.get("vocab_target") is None:
        with suppress(Exception):
            tokenizer_payload["vocab_target"] = int(len(tokenizer))
    model_payload["tokenizer"] = tokenizer_payload

    pretrained_payload = dict(model_payload.get("pretrained", {}) or {})
    if not str(pretrained_payload.get("discriminator_path", "")).strip():
        pretrained_payload["discriminator_path"] = None
    if not str(pretrained_payload.get("generator_path", "")).strip():
        pretrained_payload["generator_path"] = None
    model_payload["pretrained"] = pretrained_payload

    return _drop_none_recursive(payload)


def _maybe_enable_tf32(enabled: bool, *, force_legacy: bool = False) -> None:
    """Configure TF32 compute policy for CUDA matmul/cudnn.

    :param bool enabled: Whether to enable TF32.
    :param bool force_legacy: Whether to force legacy ``allow_tf32`` flags.
    """
    del force_legacy
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

    if policy == "flash":
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
        logger.error("bf16 mixed precision requested but CUDA is not available.")
        return False
    if not torch.cuda.is_bf16_supported():
        logger.error("bf16 mixed precision requested but this CUDA device reports no bf16 support.")
        return False

    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            a = torch.randn((64, 64), device="cuda", dtype=torch.float32)
            b = torch.randn((64, 64), device="cuda", dtype=torch.float32)
            c = a @ b
            _ = c.sum().item()
        return True
    except Exception:
        logger.error("bf16 autocast preflight failed.", exc_info=True)
        return False


def _resolve_compile_enabled_or_raise(requested: bool) -> bool:
    """Return compile-enabled flag, raising when torch.compile is unavailable.

    :param bool requested: Whether compile was requested by config.
    :raises RuntimeError: If compile was requested but torch.compile is unavailable.
    :return bool: True when compile should be enabled.
    """
    if not bool(requested):
        return False
    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "train.torch_compile=true requested but this PyTorch build does not expose torch.compile."
        )
    return True


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


def _safe_float_for_json(value: float | int | None) -> float | str | None:
    """Convert numeric values into JSON-stable scalars.

    :param float | int | None value: Numeric value.
    :return float | str | None: Finite float, string marker for non-finite, or None.
    """
    if value is None:
        return None
    out = float(value)
    if math.isfinite(out):
        return out
    if math.isnan(out):
        return "nan"
    if out > 0:
        return "inf"
    return "-inf"


def _tensor_scalar_for_debug(tensor: torch.Tensor | None) -> float | str | None:
    """Return a compact scalar summary for debug artifacts.

    :param torch.Tensor | None tensor: Tensor to summarize.
    :return float | str | None: Scalar summary.
    """
    if tensor is None:
        return None
    with suppress(Exception):
        return _safe_float_for_json(float(tensor.detach().float().mean().item()))
    return None


def _compact_rng_state_snapshot() -> dict[str, Any]:
    """Capture compact CPU/CUDA RNG state heads for diagnostics.

    :return dict[str, Any]: Compact RNG metadata.
    """
    cpu_state = torch.get_rng_state()
    payload: dict[str, Any] = {
        "cpu_len": int(cpu_state.numel()),
        "cpu_head": cpu_state[:16].tolist(),
    }
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
        payload["cuda"] = [
            {
                "device_index": idx,
                "len": int(state.numel()),
                "head": state[:16].tolist(),
            }
            for idx, state in enumerate(cuda_states)
        ]
    return payload


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


def _write_nonfinite_debug_artifact(
    *,
    output_dir: Path,
    step: int,
    micro_step_idx: int,
    offending: str,
    gen_loss_raw: torch.Tensor | None,
    disc_loss_raw: torch.Tensor | None,
    forward_loss: torch.Tensor | None,
    backward_loss: torch.Tensor | None,
    grad_norm: float | None,
    lr: float | None,
    compile_enabled: bool,
    compile_mode: str,
    embedding_sharing: str,
) -> Path:
    """Write a compact non-finite diagnostics artifact and return its path.

    :param Path output_dir: Run output directory.
    :param int step: Optimizer step index (1-based intent).
    :param int micro_step_idx: Micro-step index within accumulation window.
    :param str offending: Name of first offending tensor/stat.
    :param torch.Tensor | None gen_loss_raw: Generator raw loss.
    :param torch.Tensor | None disc_loss_raw: Discriminator raw loss.
    :param torch.Tensor | None forward_loss: Forward scalar objective.
    :param torch.Tensor | None backward_loss: Backward scalar objective.
    :param float | None grad_norm: Global gradient norm.
    :param float | None lr: Scheduler LR snapshot.
    :param bool compile_enabled: Whether torch.compile is active.
    :param str compile_mode: Compile mode string.
    :param str embedding_sharing: Embedding sharing mode string.
    :return Path: Written artifact path.
    """
    safe_offending = _sanitize_run_label(str(offending)).replace("-", "_")
    if safe_offending == "run":
        safe_offending = "nonfinite"
    path = output_dir / "debug" / f"nonfinite_step_{int(step)}_{safe_offending}.json"
    payload = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "step": int(step),
        "micro_step_idx": int(micro_step_idx),
        "offending": str(offending),
        "compile_enabled": bool(compile_enabled),
        "compile_mode": str(compile_mode),
        "embedding_sharing": str(embedding_sharing),
        "lr": _safe_float_for_json(lr),
        "gen_loss_raw": _tensor_scalar_for_debug(gen_loss_raw),
        "disc_loss_raw": _tensor_scalar_for_debug(disc_loss_raw),
        "forward_loss": _tensor_scalar_for_debug(forward_loss),
        "backward_loss": _tensor_scalar_for_debug(backward_loss),
        "grad_norm": _safe_float_for_json(grad_norm),
        "rng_state": _compact_rng_state_snapshot(),
    }
    dump_json(payload, path)
    return path


def _should_force_legacy_tf32_for_compile(*, torch_compile: bool, compile_mode: str) -> bool:
    """Return whether TF32 should use legacy flags for compile compatibility.

    :param bool torch_compile: Whether ``torch.compile`` is enabled.
    :param str compile_mode: Canonical compile mode.
    :return bool: True when legacy TF32 flags are preferred.
    """
    if not torch_compile:
        return False
    return compile_mode.startswith("max-autotune")


def _resolve_compile_scope(
    *,
    requested_scope: str,
    model_cfg: ModelConfig,
    compile_mode: str,
    compile_backend: str,
    block_cross_document_attention: bool = False,
) -> tuple[str, str | None]:
    """Resolve effective compile scope for auto compile mode.

    :param str requested_scope: Requested canonical compile scope.
    :param ModelConfig model_cfg: Model configuration.
    :param str compile_mode: Canonical torch.compile mode.
    :param str compile_backend: Compile backend name.
    :param bool block_cross_document_attention: Whether doc-blocking is enabled.
    :return tuple[str, str | None]: Effective scope and optional reason message.
    """
    if requested_scope != "auto":
        return requested_scope, None

    del compile_mode
    del compile_backend
    backbone_type = str(getattr(model_cfg, "backbone_type", "")).strip().lower()
    if bool(block_cross_document_attention) and backbone_type == "rope":
        return (
            "ffn",
            "auto scope selected FFN-only for rope + doc-blocking (mask shape churn under compile)",
        )
    return "backbones", None


def _full_backbone_hf_inductor_warning(
    *,
    model_cfg: ModelConfig,
    compile_enabled: bool,
    compile_scope: str,
    compile_backend: str,
) -> str | None:
    """Return warning text for unsupported full-backbone compile combinations.

    :param ModelConfig model_cfg: Model configuration.
    :param bool compile_enabled: Whether compile is active.
    :param str compile_scope: Effective compile scope.
    :param str compile_backend: Compile backend.
    :return str | None: Warning message for unstable configuration, else ``None``.
    """
    del model_cfg
    del compile_enabled
    del compile_scope
    del compile_backend
    return None


def _compile_backbones_for_scope(
    *,
    unwrapped_model: torch.nn.Module,
    compile_scope: str,
    compile_kwargs: dict[str, Any],
) -> list[str]:
    """Compile selected generator/discriminator submodules for a requested scope.

    :param torch.nn.Module unwrapped_model: Unwrapped RTD pretrainer model.
    :param str compile_scope: Effective compile scope.
    :param dict[str, Any] compile_kwargs: Keyword arguments passed to ``torch.compile``.
    :raises RuntimeError: If required backbone submodules are missing.
    :return list[str]: Human-readable list of compiled targets.
    """
    generator = getattr(unwrapped_model, "generator", None)
    discriminator = getattr(unwrapped_model, "discriminator", None)
    if not isinstance(generator, torch.nn.Module) or not isinstance(discriminator, torch.nn.Module):
        raise RuntimeError("RTD model must expose generator and discriminator modules for compilation.")

    compiled_targets: list[str] = []

    def _compile_module_forward(*, module: torch.nn.Module, target: str) -> None:
        """Compile one module's ``forward`` method in-place.

        Keeping module identity stable avoids ``._orig_mod`` wrapper keys in saved checkpoints.

        :param torch.nn.Module module: Module whose forward should be compiled.
        :param str target: Human-readable target name for logs.
        """
        forward = getattr(module, "forward", None)
        if not callable(forward):
            raise RuntimeError(f"{target}.forward is required for compile scope.")
        module.forward = torch.compile(forward, **compile_kwargs)  # type: ignore[assignment]
        compiled_targets.append(target)

    def _compile_encoder_ffn(*, backbone: torch.nn.Module, branch: str) -> None:
        """Compile FFN blocks in all encoder layers.

        Supports both HF DeBERTa-v2 (``encoder.layer[i].intermediate``/``.output``)
        and RoPE (``encoder.layers[i].mlp``) module layouts.

        :param torch.nn.Module backbone: Generator/discriminator backbone.
        :param str branch: Branch label used in compiled target names.
        :raises RuntimeError: If encoder/layer/FFN modules are missing.
        """
        encoder = getattr(backbone, "encoder", None)
        if not isinstance(encoder, torch.nn.Module):
            raise RuntimeError(f"{branch}.encoder is required for ffn-only compile scope.")

        # Try HF DeBERTa-v2 convention (encoder.layer) then RoPE (encoder.layers).
        layers = getattr(encoder, "layer", None)
        layers_attr = "layer"
        if not isinstance(layers, (torch.nn.ModuleList, list, tuple)):
            layers = getattr(encoder, "layers", None)
            layers_attr = "layers"
        if not isinstance(layers, (torch.nn.ModuleList, list, tuple)):
            raise RuntimeError(f"{branch}.encoder.layer/layers is required for ffn-only compile scope.")
        if len(layers) == 0:
            raise RuntimeError(
                f"{branch}.encoder.{layers_attr} is empty; cannot apply ffn-only compile scope."
            )

        for idx, layer in enumerate(layers):
            if not isinstance(layer, torch.nn.Module):
                raise RuntimeError(f"{branch}.encoder.{layers_attr}[{idx}] is not a torch.nn.Module.")

            # HF DeBERTa-v2: intermediate + output modules.
            intermediate = getattr(layer, "intermediate", None)
            output = getattr(layer, "output", None)
            if isinstance(intermediate, torch.nn.Module) and isinstance(output, torch.nn.Module):
                pfx = f"{branch}.encoder.{layers_attr}[{idx}]"
                _compile_module_forward(module=intermediate, target=f"{pfx}.intermediate")
                _compile_module_forward(module=output, target=f"{pfx}.output")
                continue

            # RoPE: single mlp module.
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, torch.nn.Module):
                _compile_module_forward(module=mlp, target=f"{branch}.encoder.{layers_attr}[{idx}].mlp")
                continue

            raise RuntimeError(
                f"{branch}.encoder.{layers_attr}[{idx}] has no recognized FFN module "
                "(expected intermediate+output or mlp)."
            )

    if compile_scope == "backbones":
        _compile_module_forward(module=generator, target="generator")
        _compile_module_forward(module=discriminator, target="discriminator")
        return ["generator", "discriminator"]

    if compile_scope in {"encoder", "gen_encoder"}:
        gen_encoder = getattr(generator, "encoder", None)
        if not isinstance(gen_encoder, torch.nn.Module):
            raise RuntimeError("generator.encoder is required for encoder-only compile scope.")
        _compile_module_forward(module=gen_encoder, target="generator.encoder")

    if compile_scope in {"encoder", "disc_encoder"}:
        disc_encoder = getattr(discriminator, "encoder", None)
        if not isinstance(disc_encoder, torch.nn.Module):
            raise RuntimeError("discriminator.encoder is required for encoder-only compile scope.")
        _compile_module_forward(module=disc_encoder, target="discriminator.encoder")

    if compile_scope in {"ffn", "gen_ffn"}:
        _compile_encoder_ffn(backbone=generator, branch="generator")

    if compile_scope in {"ffn", "disc_ffn"}:
        _compile_encoder_ffn(backbone=discriminator, branch="discriminator")

    if not compiled_targets:
        raise ValueError(f"Unsupported compile scope: {compile_scope}")
    return compiled_targets


def _dtype_for_mixed_precision(mode: str) -> torch.dtype:
    """Map configured mixed-precision mode to runtime compute dtype.

    :param str mode: Effective mixed-precision mode.
    :return torch.dtype: Expected activation dtype used in forward passes.
    """
    normalized = str(mode).strip().lower()
    if normalized == "bf16":
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _prefill_rotary_caches_for_compile(
    *,
    model: torch.nn.Module,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> int:
    """Prefill rotary caches on all RoPE attention modules before compile.

    :param torch.nn.Module model: Unwrapped RTD model.
    :param int seq_len: Maximum runtime sequence length.
    :param torch.device device: Runtime device.
    :param torch.dtype dtype: Runtime compute dtype.
    :return int: Number of rotary modules prefilled.
    """
    if int(seq_len) <= 0:
        return 0

    prefilled = 0
    for module in model.modules():
        rope = getattr(module, "rope", None)
        prefill = getattr(rope, "prefill_cache", None)
        if callable(prefill):
            prefill(int(seq_len), device=device, dtype=dtype)
            prefilled += 1
    return prefilled


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


def _build_optimizer(
    model: torch.nn.Module,
    cfg: TrainConfig,
    *,
    backbone_type: str = "",
    compile_enabled: bool = False,
    compile_scope: str = "backbones",
    mixed_precision: str = "no",
) -> torch.optim.Optimizer:
    """Create AdamW with parameter grouping for RTD training.

    :param torch.nn.Module model: RTD model.
    :param TrainConfig cfg: Training configuration.
    :param str backbone_type: Backbone type used by the runtime model.
    :param bool compile_enabled: Whether torch.compile is enabled at runtime.
    :param str compile_scope: Effective torch.compile scope.
    :param str mixed_precision: Effective mixed-precision mode.
    :return torch.optim.Optimizer: Configured AdamW optimizer.
    """

    eps = float(cfg.adam_epsilon)
    if str(mixed_precision).strip().lower() == "bf16" and eps < 1e-6:
        eps = 1e-6
        logger.warning("Raised Adam epsilon to 1e-6 for bf16 stability.")

    gen_lr = (
        float(cfg.generator_learning_rate)
        if float(cfg.generator_learning_rate) > 0
        else float(cfg.learning_rate)
    )
    disc_lr = (
        float(cfg.discriminator_learning_rate)
        if float(getattr(cfg, "discriminator_learning_rate", -1.0)) > 0
        else float(cfg.learning_rate)
    )
    partitions = _partition_optimizer_params(model)
    gen_decay = partitions["gen_decay"]["params"]
    gen_no_decay = partitions["gen_no_decay"]["params"]
    disc_decay = partitions["disc_decay"]["params"]
    disc_no_decay = partitions["disc_no_decay"]["params"]

    groups: list[dict[str, Any]] = []
    ordered_names: list[str] = []
    if gen_decay:
        groups.append({"params": gen_decay, "weight_decay": float(cfg.weight_decay), "lr": gen_lr})
        ordered_names.extend(partitions["gen_decay"]["names"])
    if gen_no_decay:
        groups.append({"params": gen_no_decay, "weight_decay": 0.0, "lr": gen_lr})
        ordered_names.extend(partitions["gen_no_decay"]["names"])
    if disc_decay:
        groups.append({"params": disc_decay, "weight_decay": float(cfg.weight_decay), "lr": disc_lr})
        ordered_names.extend(partitions["disc_decay"]["names"])
    if disc_no_decay:
        groups.append({"params": disc_no_decay, "weight_decay": 0.0, "lr": disc_lr})
        ordered_names.extend(partitions["disc_no_decay"]["names"])

    fused_kwargs = _maybe_fused_adamw_kwargs()

    opt = torch.optim.AdamW(
        groups,
        lr=disc_lr,
        betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
        eps=eps,
        **fused_kwargs,
    )
    opt._param_order_digest = _digest_param_name_order(ordered_names)
    return opt


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
    eps = float(cfg.adam_epsilon)
    if str(mixed_precision).strip().lower() == "bf16" and eps < 1e-6:
        eps = 1e-6
        logger.warning("Raised Adam epsilon to 1e-6 for bf16 stability.")

    fused_kwargs = _maybe_fused_adamw_kwargs()
    gen_lr = (
        float(cfg.generator_learning_rate)
        if float(cfg.generator_learning_rate) > 0
        else float(cfg.learning_rate)
    )
    disc_lr = (
        float(cfg.discriminator_learning_rate)
        if float(getattr(cfg, "discriminator_learning_rate", -1.0)) > 0
        else float(cfg.learning_rate)
    )
    partitions = _partition_optimizer_params(model)

    gen_groups: list[dict[str, Any]] = []
    gen_names: list[str] = []
    if partitions["gen_decay"]["params"]:
        gen_groups.append(
            {
                "params": partitions["gen_decay"]["params"],
                "weight_decay": float(cfg.weight_decay),
                "lr": gen_lr,
            }
        )
        gen_names.extend(partitions["gen_decay"]["names"])
    if partitions["gen_no_decay"]["params"]:
        gen_groups.append({"params": partitions["gen_no_decay"]["params"], "weight_decay": 0.0, "lr": gen_lr})
        gen_names.extend(partitions["gen_no_decay"]["names"])

    disc_groups: list[dict[str, Any]] = []
    disc_names: list[str] = []
    if partitions["disc_decay"]["params"]:
        disc_groups.append(
            {
                "params": partitions["disc_decay"]["params"],
                "weight_decay": float(cfg.weight_decay),
                "lr": disc_lr,
            }
        )
        disc_names.extend(partitions["disc_decay"]["names"])
    if partitions["disc_no_decay"]["params"]:
        disc_groups.append(
            {"params": partitions["disc_no_decay"]["params"], "weight_decay": 0.0, "lr": disc_lr}
        )
        disc_names.extend(partitions["disc_no_decay"]["names"])

    gen_opt = torch.optim.AdamW(
        gen_groups,
        lr=gen_lr,
        betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
        eps=eps,
        **fused_kwargs,
    )
    disc_opt = torch.optim.AdamW(
        disc_groups,
        lr=disc_lr,
        betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
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


def _resolve_data_resume_policy(
    *,
    train_cfg: Any,
    consumed_micro_batches: int,
    global_step: int,
) -> tuple[int, bool, str]:
    """Resolve data-iterator resume strategy.

    Returns ``(start_epoch, do_replay, reason)``:

    - ``start_epoch``: Dataset epoch offset used when starting the cyclical dataloader.
    - ``do_replay``: Whether to replay consumed microbatches.
    - ``reason``: Human-readable policy decision.

    :param Any train_cfg: Train config object.
    :param int consumed_micro_batches: Number of consumed microbatches loaded from checkpoint.
    :param int global_step: Resumed global optimizer step.
    :return tuple[int, bool, str]: Resolved policy triple.
    """
    consumed = max(0, int(consumed_micro_batches))
    if consumed <= 0:
        return 0, False, "fresh-start"

    strategy = str(getattr(train_cfg, "resume_data_strategy", "auto") or "auto").strip().lower()
    max_replay = max(0, int(getattr(train_cfg, "resume_replay_max_micro_batches", 10_000) or 10_000))

    if strategy == "replay":
        return 0, True, "resume_data_strategy=replay"

    if strategy == "restart_epoch":
        return int(max(0, global_step)), False, "resume_data_strategy=restart_epoch"

    if consumed <= max_replay:
        return 0, True, f"resume_data_strategy=auto (replay <= {max_replay})"
    return (
        int(max(0, global_step)),
        False,
        f"resume_data_strategy=auto (restart_epoch; replay > {max_replay})",
    )


def _normalize_resume_consumed_micro_batches(
    *,
    consumed_micro_batches: int,
    global_step: int,
    gradient_accumulation_steps: int,
) -> tuple[int, str | None]:
    """Normalize legacy resume data progress to committed optimizer-step boundaries.

    Legacy checkpoints may contain micro-batch progress ahead of ``global_step`` when
    a crash happened mid-accumulation window. Detect this pattern and clamp to the
    last committed window boundary.

    :param int consumed_micro_batches: Restored consumed micro-batch count.
    :param int global_step: Resumed optimizer step from checkpoint path.
    :param int gradient_accumulation_steps: Accumulation steps used to interpret saved progress.
    :return tuple[int, str | None]: ``(normalized_consumed, reason_or_none)``.
    """
    consumed = max(0, int(consumed_micro_batches))
    step = max(0, int(global_step))
    ga_steps = max(1, int(gradient_accumulation_steps))

    # Non-standard checkpoint names can parse as step=0; avoid clamping in that case.
    if step <= 0:
        return int(consumed), None

    expected_committed = int(step * ga_steps)
    if consumed > expected_committed:
        delta = int(consumed - expected_committed)
        if 0 < delta < ga_steps:
            return int(expected_committed), f"clamped_legacy_partial_accumulation_delta={delta}"
    return int(consumed), None


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


def _build_doc_block_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Build a pairwise ``(B, S, S)`` keep-mask from document ids on-device.

    Contract:

    - Active tokens (``doc_id != 0``) attend only within the same document.
    - The diagonal encodes query activity (active ``True``, pad/inactive ``False``).
    - Inactive/pad queries get a single keep-edge to the CLS key (position 0) so SDPA
      never sees all-False rows.

    :param torch.Tensor doc_ids: Document id tensor ``(B, S)`` with 0 for padding.
    :return torch.Tensor: Bool keep-mask ``(B, S, S)``.
    """
    if doc_ids.ndim != 2:
        raise ValueError(f"doc_ids must be rank-2 (B,S); got shape={tuple(doc_ids.shape)}")

    bsz, seq_len = int(doc_ids.shape[0]), int(doc_ids.shape[1])
    device = doc_ids.device

    key = (seq_len, str(device.type), int(device.index) if device.index is not None else None)
    eye = _DOC_BLOCK_EYE_CACHE.get(key)
    if eye is None or eye.device != device or eye.shape != (seq_len, seq_len):
        eye = torch.eye(seq_len, dtype=torch.bool, device=device)
        _DOC_BLOCK_EYE_CACHE[key] = eye

    cls_key = _DOC_BLOCK_CLS_KEY_CACHE.get(key)
    if cls_key is None or cls_key.device != device or cls_key.shape != (seq_len,):
        cls_key = torch.zeros(seq_len, dtype=torch.bool, device=device)
        cls_key[0] = True
        _DOC_BLOCK_CLS_KEY_CACHE[key] = cls_key

    active = doc_ids.ne(0)  # (B,S)
    same_doc = doc_ids[:, :, None].eq(doc_ids[:, None, :])  # (B,S,S)
    keep = same_doc & active[:, :, None] & active[:, None, :]
    keep = keep | ((~active)[:, :, None] & cls_key[None, None, :])
    keep = (keep & ~eye[None, :, :]) | (eye[None, :, :] & active[:, :, None])

    if int(keep.shape[0]) != bsz:
        raise RuntimeError("doc-block mask batch dimension mismatch.")
    return keep


def _stabilize_compile_attention_mask(
    *,
    batch: dict[str, torch.Tensor],
    compile_enabled: bool,
    compile_scope: str,
    backbone_type: str,
    block_cross_document_attention: bool = False,
) -> dict[str, torch.Tensor]:
    """Canonicalize attention-mask dtype for compiled attention paths.

    For HF DeBERTa-v2: normalizes existing masks to bool but does **not**
    materialize a mask when absent — the backbone's no-mask fast path handles
    ``None`` directly.

    RoPE + doc-blocking mask shape churn is handled by auto-downgrading compile
    scope to FFN in ``_resolve_compile_scope`` instead of materializing S² masks.

    :param dict[str, torch.Tensor] batch: Device-local batch mapping.
    :param bool compile_enabled: Whether torch.compile is active.
    :param str compile_scope: Effective compile scope.
    :param str backbone_type: Model backbone type.
    :param bool block_cross_document_attention: Whether doc-blocking is enabled.
    :return dict[str, torch.Tensor]: Possibly updated batch mapping.
    """
    if not bool(compile_enabled):
        return batch

    scope = str(compile_scope).strip().lower()
    if scope not in {"backbones", "encoder", "gen_encoder", "disc_encoder"}:
        return batch

    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        return batch

    btype = str(backbone_type).strip().lower()

    if btype == "hf_deberta_v2":
        attn = batch.get("attention_mask")
        if isinstance(attn, torch.Tensor) and attn.dtype != torch.bool:
            batch["attention_mask"] = attn.to(dtype=torch.bool)
        return batch

    return batch


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


def _validate_output_dir_preflight(
    *,
    output_dir: Path,
    overwrite_output_dir: bool,
    resume_from_checkpoint: str | None,
) -> None:
    """Validate output-dir preconditions without mutating filesystem state.

    :param Path output_dir: Resolved output directory.
    :param bool overwrite_output_dir: Whether training is configured to delete non-empty output_dir.
    :param str | None resume_from_checkpoint: Resume setting.
    :raises ValueError: If output-dir settings are contradictory or unsafe.
    """
    if bool(overwrite_output_dir) and bool(resume_from_checkpoint):
        raise ValueError(
            "train.overwrite_output_dir=true cannot be combined with train.resume_from_checkpoint. "
            "Overwrite would delete checkpoints before resume. Disable overwrite or unset resume."
        )

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and (not overwrite_output_dir)
        and (not resume_from_checkpoint)
    ):
        raise ValueError(
            f"Output directory exists and is not empty: {output_dir}. "
            "Set train.overwrite_output_dir=true or set train.resume_from_checkpoint."
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


def _save_periodic_checkpoint_if_due(
    *,
    accelerator: Any,
    train_cfg: TrainConfig,
    output_dir: Path,
    global_step: int,
    consumed_micro_batches_committed: int,
    lr_mult: float,
    optimizer_param_digest: str | dict[str, str],
    gradient_accumulation_steps: int,
    last_saved_step: int,
) -> int:
    """Persist a periodic checkpoint when ``global_step`` hits ``train.save_steps``.

    :param Any accelerator: Accelerator runtime.
    :param TrainConfig train_cfg: Training config.
    :param Path output_dir: Output directory containing checkpoints.
    :param int global_step: Current global step.
    :param int consumed_micro_batches_committed: Committed micro-batch progress.
    :param float lr_mult: Persistent recovery LR multiplier.
    :param str | dict[str, str] optimizer_param_digest: Trainable-parameter digest payload.
    :param int gradient_accumulation_steps: Active accumulation steps.
    :param int last_saved_step: Last checkpoint step already saved.
    :return int: Updated ``last_saved_step`` value.
    """
    if not train_cfg.save_steps or (global_step % int(train_cfg.save_steps) != 0):
        return int(last_saved_step)

    ckpt_dir = output_dir / f"checkpoint-{int(global_step)}"
    _save_training_checkpoint(
        accelerator=accelerator,
        checkpoint_dir=ckpt_dir,
        output_dir=output_dir,
        consumed_micro_batches=consumed_micro_batches_committed,
        save_total_limit=int(train_cfg.save_total_limit),
        log_label="periodic",
        lr_mult=float(lr_mult),
        optimizer_param_digest=optimizer_param_digest,
        global_step=int(global_step),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
    )
    return int(global_step)


def run_pretraining_dry_run(
    *,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    optim_cfg: OptimConfig | None = None,
    logging_cfg: LoggingConfig | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run non-destructive preflight checks for `deberta train`.

    This validates configuration contracts and probes core runtime dependencies
    (tokenizer, dataset access, collator output, model config construction)
    without starting optimization/training loops.

    :param ModelConfig model_cfg: Model configuration.
    :param DataConfig data_cfg: Data configuration.
    :param TrainConfig train_cfg: Training configuration.
    :param OptimConfig | None optim_cfg: Optional optimizer configuration.
    :param LoggingConfig | None logging_cfg: Optional logging configuration.
    :param str | Path | None config_path: Optional source config path.
    :raises RuntimeError: If a preflight stage fails.
    :return dict[str, Any]: Summary of resolved dry-run checks.
    """
    resolved_optim_cfg, resolved_logging_cfg = _resolve_section_cfg_compat(
        train_cfg=train_cfg,
        optim_cfg=optim_cfg,
        logging_cfg=logging_cfg,
    )
    _sync_legacy_train_aliases(
        train_cfg=train_cfg,
        optim_cfg=resolved_optim_cfg,
        logging_cfg=resolved_logging_cfg,
    )

    _apply_profile_and_validate_training_configs(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        optim_cfg=resolved_optim_cfg,
        logging_cfg=resolved_logging_cfg,
    )

    checkpoint_output_dir = _resolve_output_dir(
        output_dir=train_cfg.output_dir,
        project_name=train_cfg.project_name,
        config_path=config_path,
        run_name=train_cfg.run_name,
    )
    logging_output_dir = (
        Path(str(resolved_logging_cfg.output_dir))
        if resolved_logging_cfg.output_dir is not None and str(resolved_logging_cfg.output_dir).strip()
        else checkpoint_output_dir
    )
    _validate_output_dir_preflight(
        output_dir=checkpoint_output_dir,
        overwrite_output_dir=bool(train_cfg.overwrite_output_dir),
        resume_from_checkpoint=train_cfg.resume_from_checkpoint,
    )
    ckpt = _resolve_resume_checkpoint(
        output_dir=checkpoint_output_dir,
        resume_from_checkpoint=train_cfg.resume_from_checkpoint,
        is_main_process=True,
    )

    mixed_precision = resolve_effective_mixed_precision(
        train_cfg.mixed_precision,
        bf16_sanity_check=_bf16_runtime_sanity_check,
    )
    compile_mode = _normalize_torch_compile_mode(train_cfg.torch_compile_mode)
    compile_enabled = _resolve_compile_enabled_or_raise(train_cfg.torch_compile)
    compile_scope_requested = str(train_cfg.torch_compile_scope).strip().lower()
    compile_backend = str(train_cfg.torch_compile_backend).strip().lower()
    compile_scope = compile_scope_requested
    compile_scope_reason: str | None = None
    if compile_enabled:
        compile_scope, compile_scope_reason = _resolve_compile_scope(
            requested_scope=compile_scope_requested,
            model_cfg=model_cfg,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
            block_cross_document_attention=bool(data_cfg.block_cross_document_attention),
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
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name_or_path, use_fast=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load tokenizer from model.tokenizer_name_or_path="
            f"{model_cfg.tokenizer_name_or_path!r}."
        ) from exc
    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "Tokenizer preflight failed: tokenizer.pad_token_id is unset. "
            "Use a tokenizer with a defined PAD token."
        )

    try:
        raw_train = load_hf_dataset(cfg=data_cfg, split=data_cfg.train_split, streaming=data_cfg.streaming)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load dataset for dry-run preflight. Check data.dataset_name/data_files/load_from_disk, "
            "split, and network/auth settings."
        ) from exc

    train_dataset, collator = _build_train_dataset_and_collator(
        raw_train=raw_train,
        tokenizer=tokenizer,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        process_index=0,
        num_processes=1,
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
        disc_config, gen_config = build_backbone_configs(
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            max_position_embeddings=int(data_cfg.max_seq_length),
        )
    except Exception as exc:
        raise RuntimeError(
            "Backbone config preflight failed. Check model/backbone/tokenizer settings and vocab alignment options."
        ) from exc

    return {
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


def run_pretraining(
    *,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    optim_cfg: OptimConfig | None = None,
    logging_cfg: LoggingConfig | None = None,
    config_path: str | Path | None = None,
) -> None:
    """Run RTD pretraining with Accelerate/FSDP2-compatible plumbing.

    :param ModelConfig model_cfg: Model configuration.
    :param DataConfig data_cfg: Data configuration.
    :param TrainConfig train_cfg: Training configuration.
    :param OptimConfig | None optim_cfg: Optional optimizer configuration.
    :param LoggingConfig | None logging_cfg: Optional logging configuration.
    :param str | Path | None config_path: Optional source config path for auto output-dir naming.
    """
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    try:
        from accelerate import DistributedDataParallelKwargs
    except Exception:  # pragma: no cover
        DistributedDataParallelKwargs = None  # type: ignore[assignment]

    resolved_optim_cfg, resolved_logging_cfg = _resolve_section_cfg_compat(
        train_cfg=train_cfg,
        optim_cfg=optim_cfg,
        logging_cfg=logging_cfg,
    )
    _sync_legacy_train_aliases(
        train_cfg=train_cfg,
        optim_cfg=resolved_optim_cfg,
        logging_cfg=resolved_logging_cfg,
    )

    log_with = None if train_cfg.report_to == "none" else train_cfg.report_to
    mixed_precision = resolve_effective_mixed_precision(
        train_cfg.mixed_precision,
        bf16_sanity_check=_bf16_runtime_sanity_check,
    )
    compile_mode = _normalize_torch_compile_mode(train_cfg.torch_compile_mode)
    compile_enabled = _resolve_compile_enabled_or_raise(train_cfg.torch_compile)
    object.__setattr__(train_cfg, "mixed_precision", mixed_precision)
    accelerator_kwargs: dict[str, Any] = {
        "gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,
        "log_with": log_with,
        "mixed_precision": mixed_precision,
    }
    if bool(train_cfg.decoupled_training) and DistributedDataParallelKwargs is not None:
        # Generator/discriminator phases each touch only a subset of parameters.
        # DDP must track unused params to avoid cross-rank reducer stalls.
        accelerator_kwargs["kwargs_handlers"] = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(
        **accelerator_kwargs,
    )

    setup_process_logging(accelerator.is_main_process)
    _apply_profile_and_validate_training_configs(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        optim_cfg=resolved_optim_cfg,
        logging_cfg=resolved_logging_cfg,
    )
    compile_scope_requested = str(train_cfg.torch_compile_scope).strip().lower()
    compile_backend = str(train_cfg.torch_compile_backend).strip().lower()
    compile_scope = compile_scope_requested
    compile_scope_reason: str | None = None
    if compile_enabled:
        compile_scope, compile_scope_reason = _resolve_compile_scope(
            requested_scope=compile_scope_requested,
            model_cfg=model_cfg,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
            block_cross_document_attention=bool(data_cfg.block_cross_document_attention),
        )
        full_backbone_warn = _full_backbone_hf_inductor_warning(
            model_cfg=model_cfg,
            compile_enabled=compile_enabled,
            compile_scope=compile_scope,
            compile_backend=compile_backend,
        )
        if full_backbone_warn is not None and accelerator.is_main_process:
            logger.warning(full_backbone_warn)

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

    # Resolve output dir before side effects and persist the concrete path in train_cfg.
    configured_output_dir = train_cfg.output_dir
    output_dir = _resolve_output_dir_for_accelerator(
        accelerator=accelerator,
        output_dir=train_cfg.output_dir,
        project_name=train_cfg.project_name,
        config_path=config_path,
        run_name=train_cfg.run_name,
    )
    object.__setattr__(train_cfg.checkpoint, "output_dir", str(output_dir))
    if resolved_logging_cfg.output_dir is None or not str(resolved_logging_cfg.output_dir).strip():
        logging_output_dir = output_dir
    else:
        logging_output_dir = Path(str(resolved_logging_cfg.output_dir)).expanduser().resolve()
    object.__setattr__(resolved_logging_cfg, "output_dir", str(logging_output_dir))
    _sync_legacy_train_aliases(
        train_cfg=train_cfg,
        optim_cfg=resolved_optim_cfg,
        logging_cfg=resolved_logging_cfg,
    )
    if accelerator.is_main_process and (
        configured_output_dir is None or not str(configured_output_dir).strip()
    ):
        logger.info("train.output_dir unset; auto-selected output_dir=%s", output_dir)
    if accelerator.is_main_process and logging_output_dir != output_dir:
        logger.info(
            "logging.output_dir explicitly set to %s (checkpoint output_dir=%s)",
            logging_output_dir,
            output_dir,
        )

    # Make/validate output dir on main.
    _prepare_output_dir(
        output_dir=output_dir,
        overwrite_output_dir=bool(train_cfg.overwrite_output_dir),
        resume_from_checkpoint=train_cfg.resume_from_checkpoint,
        is_main_process=accelerator.is_main_process,
    )
    if accelerator.is_main_process:
        logging_output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    ckpt = _resolve_resume_checkpoint_for_accelerator(
        accelerator=accelerator,
        output_dir=output_dir,
        resume_from_checkpoint=train_cfg.resume_from_checkpoint,
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
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name_or_path, use_fast=True)

    # Sanity
    if tokenizer.pad_token_id is None:
        # Many DeBERTa tokenizers define PAD.
        raise ValueError("Tokenizer must have pad_token_id.")

    # Data
    raw_train = load_hf_dataset(cfg=data_cfg, split=data_cfg.train_split, streaming=data_cfg.streaming)

    train_dataset, collator = _build_train_dataset_and_collator(
        raw_train=raw_train,
        tokenizer=tokenizer,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        process_index=int(accelerator.process_index),
        num_processes=int(accelerator.num_processes),
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
        tie_generator_word_embeddings=True,
        additional_forbidden_token_ids=getattr(tokenizer, "all_special_ids", []),
    )

    effective_decoupled_training = bool(train_cfg.decoupled_training)
    if effective_decoupled_training and (
        not hasattr(model, "forward_generator_phase") or not hasattr(model, "forward_discriminator_phase")
    ):
        raise RuntimeError(
            "train.decoupled_training=true requires runtime model methods "
            "forward_generator_phase and forward_discriminator_phase."
        )

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
            train_cfg,
            mixed_precision=mixed_precision,
        )
        gen_lr_scheduler = _build_scheduler(gen_optimizer, train_cfg)
        disc_lr_scheduler = _build_scheduler(disc_optimizer, train_cfg)
        param_digest = {
            "generator": str(getattr(gen_optimizer, "_param_order_digest", "")),
            "discriminator": str(getattr(disc_optimizer, "_param_order_digest", "")),
        }
        model, gen_optimizer, disc_optimizer, gen_lr_scheduler, disc_lr_scheduler = accelerator.prepare(
            model, gen_optimizer, disc_optimizer, gen_lr_scheduler, disc_lr_scheduler
        )
        _record_unscaled_lrs(gen_optimizer, gen_lr_scheduler)
        _record_unscaled_lrs(disc_optimizer, disc_lr_scheduler)
    else:
        optimizer = _build_optimizer(
            model,
            train_cfg,
            backbone_type=str(model_cfg.backbone_type),
            compile_enabled=compile_enabled,
            compile_scope=compile_scope,
            mixed_precision=mixed_precision,
        )
        param_digest = str(getattr(optimizer, "_param_order_digest", _optimizer_param_order_digest(model)))
        lr_scheduler = _build_scheduler(optimizer, train_cfg)
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

            compile_kwargs: dict[str, Any] = {"mode": compile_mode, "backend": compile_backend}
            try:
                compile_params = inspect.signature(torch.compile).parameters  # type: ignore[attr-defined]
                if "dynamic" in compile_params:
                    compile_kwargs["dynamic"] = False
            except Exception:
                pass

            unwrapped = unwrap_compiled_model(accelerator, model)
            compile_scope_key = str(compile_scope).strip().lower()
            if compile_scope_key in {"backbones", "encoder", "gen_encoder", "disc_encoder"}:
                prefilled_rotary = _prefill_rotary_caches_for_compile(
                    model=unwrapped,
                    seq_len=int(data_cfg.max_seq_length),
                    device=torch.device(getattr(accelerator, "device", torch.device("cpu"))),
                    dtype=_dtype_for_mixed_precision(mixed_precision),
                )
                if prefilled_rotary > 0:
                    logger.info(
                        "Prefilled rotary caches for %d module(s) at seq_len=%d before torch.compile.",
                        int(prefilled_rotary),
                        int(data_cfg.max_seq_length),
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
    if accelerator.is_main_process:
        try:
            _dump_yaml_mapping(tracker_cfg_runtime, logging_output_dir / "config_resolved.yaml")
        except Exception:
            logger.exception("Failed to write runtime-resolved config snapshot.")

    global_step = 0
    consumed_micro_batches = 0
    consumed_micro_batches_committed = 0
    ga_steps = max(1, int(train_cfg.gradient_accumulation_steps))
    last_saved_step = 0
    lr_mult = 1.0
    nonfinite_skip_total = 0
    nonfinite_skip_streak = 0
    crash_type: str | None = None
    crash_reason: str | None = None
    crash_step: int | None = None
    exit_code = 0
    train_started_at = time.perf_counter()
    metrics_path = logging_output_dir / "metrics.jsonl.gz"
    debug_metrics_enabled = bool(train_cfg.debug_metrics)
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
        if train_cfg.report_to == "none":
            return int(step)

        effective_step = int(step)
        if effective_step < int(max_tracker_step_logged):
            effective_step = int(max_tracker_step_logged)
        else:
            max_tracker_step_logged = int(effective_step)
        accelerator.log(row, step=effective_step)
        return int(effective_step)

    try:
        # Trackers
        if train_cfg.report_to != "none":
            tracker_cfg = dict(tracker_cfg_runtime)
            if train_cfg.run_name is not None and str(train_cfg.run_name).strip():
                tracker_run_name = str(train_cfg.run_name).strip()
            else:
                tracker_run_name = logging_output_dir.name
            _init_trackers(
                accelerator=accelerator,
                project_name=str(train_cfg.project_name).strip(),
                tracker_cfg=tracker_cfg,
                report_to=str(train_cfg.report_to),
                run_name=tracker_run_name,
            )
            if str(train_cfg.report_to).lower() == "wandb":
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
                        watch_mode=train_cfg.wandb_watch,
                        watch_log_freq=int(train_cfg.wandb_watch_log_freq),
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
            if saved_global_step is None:
                global_step = int(parsed_checkpoint_step)
            else:
                global_step = int(max(0, saved_global_step))
                if int(parsed_checkpoint_step) > 0 and int(parsed_checkpoint_step) != int(global_step):
                    raise RuntimeError(
                        "Checkpoint step mismatch on resume: "
                        f"path step={int(parsed_checkpoint_step)} but data_state.json global_step={int(global_step)} "
                        f"for checkpoint '{ckpt}'."
                    )
            last_saved_step = int(global_step)
            normalization_ga_steps = (
                int(saved_ga_steps) if saved_ga_steps is not None else int(max(1, int(ga_steps)))
            )
            if (
                saved_ga_steps is not None
                and int(saved_ga_steps) != int(ga_steps)
                and accelerator.is_main_process
            ):
                logger.warning(
                    "Resume checkpoint '%s' was saved with gradient_accumulation_steps=%d but current run uses %d; "
                    "using save-time GA only for resume-progress normalization.",
                    ckpt,
                    int(saved_ga_steps),
                    int(ga_steps),
                )
            (
                consumed_micro_batches,
                consumed_normalize_reason,
            ) = _normalize_resume_consumed_micro_batches(
                consumed_micro_batches=int(restored),
                global_step=int(global_step),
                gradient_accumulation_steps=int(normalization_ga_steps),
            )
            if consumed_normalize_reason is not None and accelerator.is_main_process:
                logger.warning(
                    "Resume checkpoint '%s' had consumed_micro_batches=%d ahead of committed step boundary; "
                    "clamped to %d (%s).",
                    ckpt,
                    int(restored),
                    int(consumed_micro_batches),
                    consumed_normalize_reason,
                )
            consumed_micro_batches_committed = int(consumed_micro_batches)
            lr_mult = float(restored_lr_mult)

            if saved_digest is None:
                logger.warning(
                    "Checkpoint '%s' has no optimizer_param_digest; skipping param-order "
                    "validation. Future checkpoints will include the digest.",
                    ckpt,
                )
            elif isinstance(saved_digest, dict):
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
                    # Back-compat for transitioning from single optimizer to decoupled mode.
                    logger.warning(
                        "Checkpoint '%s' stores decoupled optimizer digests, but current run uses a single optimizer. "
                        "Skipping strict digest validation.",
                        ckpt,
                    )
            elif isinstance(param_digest, dict):
                logger.warning(
                    "Checkpoint '%s' stores legacy single optimizer digest while current run uses decoupled mode. "
                    "Skipping strict digest validation for this resume.",
                    ckpt,
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
        unwrapped_model = unwrap_compiled_model(accelerator, model)
        disc_pad_token_id = getattr(getattr(unwrapped_model, "disc_config", None), "pad_token_id", None)
        if disc_pad_token_id is not None:
            disc_pad_token_id = int(disc_pad_token_id)

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
                for _ in replay_iter:
                    _ = next(train_iter)
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
            if not train_cfg.logging_steps or (global_step % int(train_cfg.logging_steps) != 0):
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
            loss = float(train_cfg.gen_loss_weight) * float(gen_loss_window) + float(
                train_cfg.disc_loss_weight
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
                            f"acc={metrics['disc_acc']:.4f}",
                            f"tok/s={metrics['input_tokens_per_sec']:.1f}",
                            f"tok_seen={metrics['input_tokens_seen']:.0f}",
                        ]
                    )
                )
            if train_cfg.report_to != "none":
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

        if effective_decoupled_training:
            if gen_optimizer is None or disc_optimizer is None:
                raise RuntimeError("Decoupled training requires generator/discriminator optimizers.")
            if gen_lr_scheduler is None or disc_lr_scheduler is None:
                raise RuntimeError("Decoupled training requires generator/discriminator schedulers.")

            while global_step < int(train_cfg.max_steps):
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
                    disc_pad_token_id=disc_pad_token_id,
                    include_has_gen_targets=True,
                    default_unweighted_token_count=1.0,
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

                disc_phase_inputs: list[dict[str, torch.Tensor | float | None]] = []
                loss_for_metrics = torch.zeros((), device=accelerator.device, dtype=torch.float32)
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
                gen_loss_weight = float(train_cfg.gen_loss_weight)
                disc_loss_weight = float(train_cfg.disc_loss_weight)
                gen_phase_enabled = gen_loss_weight != 0.0
                disc_phase_enabled = disc_loss_weight != 0.0

                gen_optimizer.zero_grad(set_to_none=True)
                disc_optimizer.zero_grad(set_to_none=True)

                # Phase 1: generator update + corruption target construction.
                gen_window_nonfinite_local = False
                gen_first_nonfinite_reason_local: str | None = None
                gen_first_nonfinite_micro_step: int | None = None
                for step_idx, (batch, gen_count, disc_count, has_gen_targets) in enumerate(window):
                    batch = _move_batch_to_device(batch, accelerator.device)
                    doc_ids = batch.pop("doc_ids", None)
                    if doc_ids is not None:
                        batch["attention_mask"] = _build_doc_block_mask(doc_ids)
                    batch = _stabilize_compile_attention_mask(
                        batch=batch,
                        compile_enabled=compile_enabled,
                        compile_scope=compile_scope,
                        backbone_type=str(model_cfg.backbone_type),
                        block_cross_document_attention=bool(data_cfg.block_cross_document_attention),
                    )
                    if compile_enabled:
                        _maybe_cudagraph_mark_step_begin()

                    is_sync_step = step_idx == (ga_steps - 1)
                    sync_ctx = nullcontext() if is_sync_step else accelerator.no_sync(model)
                    with sync_ctx:
                        gen_phase_out = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            labels=batch["labels"],
                            token_type_ids=batch.get("token_type_ids"),
                            sampling_temperature=train_cfg.sampling_temperature,
                            phase="generator",
                        )
                        gen_loss = gen_phase_out.gen_loss_raw
                        if token_weighted_ga:
                            gen_obj = gen_loss * (
                                float(gen_count) / max(float(gen_window_tokens_per_rank), 1.0)
                            )
                        else:
                            gen_obj = gen_loss
                        loss_for_metrics = loss_for_metrics + (gen_loss_weight * gen_obj.detach())

                        offending: str | None = None
                        if not torch.isfinite(gen_phase_out.gen_loss_raw.detach()).all():
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

                        if offending is not None:
                            gen_window_nonfinite_local = True
                            if gen_first_nonfinite_reason_local is None:
                                gen_first_nonfinite_reason_local = str(offending)
                                gen_first_nonfinite_micro_step = int(step_idx)

                        # Keep non-finite coordination out of non-sync microsteps to preserve
                        # no_sync accumulation performance characteristics.
                        if is_sync_step and _any_rank_flag_true(
                            accelerator=accelerator,
                            flag=gen_window_nonfinite_local,
                        ):
                            offending_effective = (
                                str(gen_first_nonfinite_reason_local)
                                if gen_first_nonfinite_reason_local is not None
                                else "other_rank_nonfinite"
                            )
                            offending_micro_step = (
                                int(gen_first_nonfinite_micro_step)
                                if gen_first_nonfinite_micro_step is not None
                                else int(step_idx)
                            )
                            skipped_window_due_nonfinite = True
                            nonfinite_skip_total += 1
                            nonfinite_skip_streak += 1
                            nonfinite_reason = str(offending_effective)
                            lr_now = _scheduler_current_lr(gen_lr_scheduler)
                            nonfinite_debug_path = _write_nonfinite_debug_artifact(
                                output_dir=logging_output_dir,
                                step=int(global_step + 1),
                                micro_step_idx=int(offending_micro_step),
                                offending=str(offending_effective),
                                gen_loss_raw=gen_phase_out.gen_loss_raw,
                                disc_loss_raw=None,
                                forward_loss=gen_phase_out.gen_loss_raw,
                                backward_loss=backward_loss,
                                grad_norm=None,
                                lr=lr_now,
                                compile_enabled=compile_enabled,
                                compile_mode=compile_mode,
                                embedding_sharing=str(model_cfg.embedding_sharing),
                            )
                            gen_optimizer.zero_grad(set_to_none=True)
                            disc_optimizer.zero_grad(set_to_none=True)
                            break
                        if offending is not None:
                            continue

                        micro_gen_token_count = gen_phase_out.gen_token_count.detach().float()
                        gen_token_count_window = gen_token_count_window + micro_gen_token_count
                        gen_loss_num = gen_loss_num + (
                            gen_phase_out.gen_loss_raw.detach().float() * micro_gen_token_count
                        )
                        # Keep discriminator micro-step counts aligned across ranks:
                        # when local generator targets are absent we still run the
                        # corresponding discriminator pass with zero objective weight.
                        # Prefer explicit phase metadata when available; otherwise
                        # fall back to CPU labels metadata gathered pre-device transfer.
                        phase_has_targets = getattr(gen_phase_out, "has_masked_targets", None)
                        if phase_has_targets is None:
                            phase_has_targets = bool(has_gen_targets)
                        disc_objective_weight = 1.0 if bool(phase_has_targets) else 0.0
                        disc_phase_inputs.append(
                            {
                                "input_ids": batch["input_ids"],
                                "attention_mask": batch.get("attention_mask"),
                                "token_type_ids": batch.get("token_type_ids"),
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
                            did_gen_optimizer_step = True
                            continue
                        grad_norm_for_check = _global_grad_l2_norm(model)
                        if _has_nonfinite_grad_norm_any_rank(
                            accelerator=accelerator,
                            grad_norm=float(grad_norm_for_check),
                        ):
                            skipped_window_due_nonfinite = True
                            nonfinite_skip_total += 1
                            nonfinite_skip_streak += 1
                            nonfinite_reason = "gen_grad_norm_nonfinite"
                            gen_optimizer.zero_grad(set_to_none=True)
                            disc_optimizer.zero_grad(set_to_none=True)
                            break
                        if _should_clip_gradients(sync_gradients=True, max_grad_norm=train_cfg.max_grad_norm):
                            accelerator.clip_grad_norm_(model.parameters(), float(train_cfg.max_grad_norm))
                        gen_optimizer.step()
                        gen_lr_scheduler.step()
                        _record_unscaled_lrs(gen_optimizer, gen_lr_scheduler)
                        if lr_mult < 1.0:
                            _apply_lr_mult(gen_optimizer, lr_mult)
                        gen_optimizer.zero_grad(set_to_none=True)
                        did_gen_optimizer_step = True
                        _sync_discriminator_embeddings_if_available(model, accelerator=accelerator)

                window_has_global_disc_targets = False
                if not skipped_window_due_nonfinite and disc_phase_inputs:
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
                    not skipped_window_due_nonfinite
                    and disc_phase_inputs
                    and bool(window_has_global_disc_targets)
                ):
                    disc_phase_steps = len(disc_phase_inputs)
                    disc_window_nonfinite_local = False
                    disc_first_nonfinite_reason_local: str | None = None
                    disc_first_nonfinite_micro_step: int | None = None
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
                                phase="discriminator",
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
                            loss_for_metrics = loss_for_metrics + (disc_loss_weight * disc_obj.detach())
                            offending = None
                            if not torch.isfinite(disc_phase_out.disc_loss_raw.detach()).all():
                                offending = "disc_loss_raw"

                            backward_loss: torch.Tensor | None = None
                            if offending is None and disc_phase_enabled:
                                weighted_disc_obj = disc_obj * disc_loss_weight
                                backward_loss = _scale_loss_for_backward(
                                    loss=weighted_disc_obj,
                                    ga_steps=ga_steps,
                                    token_weighted_ga=token_weighted_ga,
                                )
                                if not torch.isfinite(backward_loss.detach()).all():
                                    offending = "disc_backward_loss"

                            if offending is not None:
                                disc_window_nonfinite_local = True
                                if disc_first_nonfinite_reason_local is None:
                                    disc_first_nonfinite_reason_local = str(offending)
                                    disc_first_nonfinite_micro_step = int(step_idx)

                            # Keep non-finite coordination out of non-sync microsteps to preserve
                            # no_sync accumulation performance characteristics.
                            if is_sync_step and _any_rank_flag_true(
                                accelerator=accelerator,
                                flag=disc_window_nonfinite_local,
                            ):
                                offending_effective = (
                                    str(disc_first_nonfinite_reason_local)
                                    if disc_first_nonfinite_reason_local is not None
                                    else "other_rank_nonfinite"
                                )
                                offending_micro_step = (
                                    int(disc_first_nonfinite_micro_step)
                                    if disc_first_nonfinite_micro_step is not None
                                    else int(step_idx)
                                )
                                skipped_window_due_nonfinite = True
                                nonfinite_skip_total += 1
                                nonfinite_skip_streak += 1
                                nonfinite_reason = str(offending_effective)
                                lr_now = _scheduler_current_lr(disc_lr_scheduler)
                                nonfinite_debug_path = _write_nonfinite_debug_artifact(
                                    output_dir=logging_output_dir,
                                    step=int(global_step + 1),
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
                                    compile_enabled=compile_enabled,
                                    compile_mode=compile_mode,
                                    embedding_sharing=str(model_cfg.embedding_sharing),
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
                            if not disc_phase_enabled:
                                did_disc_optimizer_step = True
                                continue
                            grad_norm_for_check = _global_grad_l2_norm(model)
                            if _has_nonfinite_grad_norm_any_rank(
                                accelerator=accelerator,
                                grad_norm=float(grad_norm_for_check),
                            ):
                                skipped_window_due_nonfinite = True
                                nonfinite_skip_total += 1
                                nonfinite_skip_streak += 1
                                nonfinite_reason = "disc_grad_norm_nonfinite"
                                gen_optimizer.zero_grad(set_to_none=True)
                                disc_optimizer.zero_grad(set_to_none=True)
                                break
                            if _should_clip_gradients(
                                sync_gradients=True, max_grad_norm=train_cfg.max_grad_norm
                            ):
                                accelerator.clip_grad_norm_(
                                    model.parameters(), float(train_cfg.max_grad_norm)
                                )
                            disc_optimizer.step()
                            disc_lr_scheduler.step()
                            _record_unscaled_lrs(disc_optimizer, disc_lr_scheduler)
                            if lr_mult < 1.0:
                                _apply_lr_mult(disc_optimizer, lr_mult)
                            disc_optimizer.zero_grad(set_to_none=True)
                            did_disc_optimizer_step = True

                if not skipped_window_due_nonfinite and (
                    not disc_phase_inputs or not bool(window_has_global_disc_targets)
                ):
                    # No generator-supervised tokens were produced in this window, so there are
                    # no discriminator targets to train on.
                    did_disc_optimizer_step = True

                did_optimizer_step = bool(did_gen_optimizer_step and did_disc_optimizer_step)
                if not did_optimizer_step:
                    if skipped_window_due_nonfinite:
                        # Keep scheduler state moving on skipped windows only for phases that
                        # did not already step in this accumulation window.
                        if (not did_gen_optimizer_step) and _optimizer_has_stepped(gen_optimizer):
                            with suppress(Exception):
                                gen_lr_scheduler.step()
                        if (not did_disc_optimizer_step) and _optimizer_has_stepped(disc_optimizer):
                            with suppress(Exception):
                                disc_lr_scheduler.step()
                        _record_unscaled_lrs(gen_optimizer, gen_lr_scheduler)
                        _record_unscaled_lrs(disc_optimizer, disc_lr_scheduler)
                        lr_mult, reset_state = _apply_nonfinite_recovery(
                            lr_mult=lr_mult,
                            skip_streak=int(nonfinite_skip_streak),
                        )
                        _apply_lr_mult(gen_optimizer, lr_mult)
                        _apply_lr_mult(disc_optimizer, lr_mult)
                        if reset_state:
                            with suppress(Exception):
                                gen_optimizer.state.clear()
                            with suppress(Exception):
                                disc_optimizer.state.clear()
                        global_step += 1
                        consumed_micro_batches_committed = int(consumed_micro_batches)
                        if train_progress is not None:
                            train_progress.update(1)
                        last_saved_step = _save_periodic_checkpoint_if_due(
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
                        continue
                    raise RuntimeError(
                        "Decoupled accumulation window produced no synchronized optimization step."
                    )

                nonfinite_skip_streak = 0
                if lr_mult < 1.0:
                    lr_mult = min(lr_mult * float(_NONFINITE_LR_MULT_RECOVERY), 1.0)

                global_step += 1
                consumed_micro_batches_committed = int(consumed_micro_batches)
                if train_progress is not None:
                    train_progress.update(1)

                _maybe_log_training_metrics(
                    lr_scheduler=disc_lr_scheduler,
                    gen_loss_num=gen_loss_num,
                    gen_token_count_window=gen_token_count_window,
                    disc_loss_num=disc_loss_num,
                    disc_acc_num=disc_acc_num,
                    disc_token_count_window=disc_token_count_window,
                    disc_positive_count_window=disc_positive_count_window,
                )

                last_saved_step = _save_periodic_checkpoint_if_due(
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

        while global_step < int(train_cfg.max_steps):
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
                disc_pad_token_id=disc_pad_token_id,
                include_has_gen_targets=False,
                default_unweighted_token_count=0.0,
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
            window_nonfinite_local = False
            first_nonfinite_reason_local: str | None = None
            first_nonfinite_micro_step: int | None = None

            for step_idx, (batch, gen_count, disc_count) in enumerate(window):
                batch = _move_batch_to_device(batch, accelerator.device)
                doc_ids = batch.pop("doc_ids", None)
                if doc_ids is not None:
                    batch["attention_mask"] = _build_doc_block_mask(doc_ids)
                batch = _stabilize_compile_attention_mask(
                    batch=batch,
                    compile_enabled=compile_enabled,
                    compile_scope=compile_scope,
                    backbone_type=str(model_cfg.backbone_type),
                    block_cross_document_attention=bool(data_cfg.block_cross_document_attention),
                )
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
                    if not torch.isfinite(out.gen_loss_raw.detach()).all():
                        offending = "gen_loss_raw"
                    elif not torch.isfinite(out.disc_loss_raw.detach()).all():
                        offending = "disc_loss_raw"
                    elif not torch.isfinite(out.loss.detach()).all():
                        offending = "forward_loss"
                    elif not torch.isfinite(backward_loss.detach()).all():
                        offending = "backward_loss"

                    if offending is not None:
                        window_nonfinite_local = True
                        if first_nonfinite_reason_local is None:
                            first_nonfinite_reason_local = str(offending)
                            first_nonfinite_micro_step = int(step_idx)

                    # Keep non-finite coordination out of non-sync microsteps to preserve
                    # no_sync accumulation performance characteristics.
                    if is_sync_step and _any_rank_flag_true(
                        accelerator=accelerator,
                        flag=window_nonfinite_local,
                    ):
                        offending_effective = (
                            str(first_nonfinite_reason_local)
                            if first_nonfinite_reason_local is not None
                            else "other_rank_nonfinite"
                        )
                        offending_micro_step = (
                            int(first_nonfinite_micro_step)
                            if first_nonfinite_micro_step is not None
                            else int(step_idx)
                        )
                        skipped_window_due_nonfinite = True
                        nonfinite_skip_total += 1
                        nonfinite_skip_streak += 1
                        nonfinite_reason = str(offending_effective)
                        lr_now = _scheduler_current_lr(lr_scheduler)
                        nonfinite_debug_path = _write_nonfinite_debug_artifact(
                            output_dir=logging_output_dir,
                            step=int(global_step + 1),
                            micro_step_idx=int(offending_micro_step),
                            offending=str(offending_effective),
                            gen_loss_raw=out.gen_loss_raw,
                            disc_loss_raw=out.disc_loss_raw,
                            forward_loss=out.loss,
                            backward_loss=backward_loss,
                            grad_norm=None,
                            lr=lr_now,
                            compile_enabled=compile_enabled,
                            compile_mode=compile_mode,
                            embedding_sharing=str(model_cfg.embedding_sharing),
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
                    micro_gen_tokens = out.gen_token_count.detach().float()
                    micro_disc_tokens = out.disc_token_count.detach().float()
                    gen_token_count_window = gen_token_count_window + micro_gen_tokens
                    disc_token_count_window = disc_token_count_window + micro_disc_tokens
                    disc_positive_count_window = (
                        disc_positive_count_window + out.disc_positive_count.detach().float()
                    )
                    gen_loss_num = gen_loss_num + out.gen_loss.detach().float() * micro_gen_tokens
                    disc_loss_num = disc_loss_num + out.disc_loss.detach().float() * micro_disc_tokens
                    disc_acc_num = disc_acc_num + out.disc_accuracy.detach().float() * micro_disc_tokens
                    accelerator.backward(backward_loss)

                if is_sync_step:
                    grad_norm_for_check = _global_grad_l2_norm(model)
                    if _has_nonfinite_grad_norm_any_rank(
                        accelerator=accelerator,
                        grad_norm=float(grad_norm_for_check),
                    ):
                        skipped_window_due_nonfinite = True
                        nonfinite_skip_total += 1
                        nonfinite_skip_streak += 1
                        nonfinite_reason = f"grad_norm_skip_{int(nonfinite_skip_total)}"
                        lr_now = _scheduler_current_lr(lr_scheduler)
                        nonfinite_debug_path = _write_nonfinite_debug_artifact(
                            output_dir=logging_output_dir,
                            step=int(global_step + 1),
                            micro_step_idx=int(step_idx),
                            offending=str(nonfinite_reason),
                            gen_loss_raw=out.gen_loss_raw if out is not None else None,
                            disc_loss_raw=out.disc_loss_raw if out is not None else None,
                            forward_loss=out.loss if out is not None else None,
                            backward_loss=None,
                            grad_norm=float(grad_norm_for_check),
                            lr=lr_now,
                            compile_enabled=compile_enabled,
                            compile_mode=compile_mode,
                            embedding_sharing=str(model_cfg.embedding_sharing),
                        )
                        logger.warning(
                            "Skipping optimizer step due non-finite gradient norm "
                            "(step=%d, streak=%d, total_skips=%d). Debug artifact: %s",
                            int(global_step + 1),
                            int(nonfinite_skip_streak),
                            int(nonfinite_skip_total),
                            nonfinite_debug_path,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        break

                    if _should_clip_gradients(
                        sync_gradients=True,
                        max_grad_norm=train_cfg.max_grad_norm,
                    ):
                        accelerator.clip_grad_norm_(model.parameters(), float(train_cfg.max_grad_norm))
                        post_clip_grad_norm = _global_grad_l2_norm(model)
                        if _has_nonfinite_grad_norm_any_rank(
                            accelerator=accelerator,
                            grad_norm=float(post_clip_grad_norm),
                        ):
                            skipped_window_due_nonfinite = True
                            nonfinite_skip_total += 1
                            nonfinite_skip_streak += 1
                            nonfinite_reason = f"grad_norm_post_clip_skip_{int(nonfinite_skip_total)}"
                            lr_now = _scheduler_current_lr(lr_scheduler)
                            nonfinite_debug_path = _write_nonfinite_debug_artifact(
                                output_dir=logging_output_dir,
                                step=int(global_step + 1),
                                micro_step_idx=int(step_idx),
                                offending=str(nonfinite_reason),
                                gen_loss_raw=out.gen_loss_raw if out is not None else None,
                                disc_loss_raw=out.disc_loss_raw if out is not None else None,
                                forward_loss=out.loss if out is not None else None,
                                backward_loss=None,
                                grad_norm=float(post_clip_grad_norm),
                                lr=lr_now,
                                compile_enabled=compile_enabled,
                                compile_mode=compile_mode,
                                embedding_sharing=str(model_cfg.embedding_sharing),
                            )
                            logger.warning(
                                "Skipping optimizer step due non-finite post-clip gradient norm "
                                "(step=%d, streak=%d, total_skips=%d). Debug artifact: %s",
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
                    nonfinite_skip_streak = 0
                    if lr_mult < 1.0:
                        lr_mult = min(lr_mult * float(_NONFINITE_LR_MULT_RECOVERY), 1.0)

                    _sync_discriminator_embeddings_if_available(model, accelerator=accelerator)

            if not did_optimizer_step:
                if skipped_window_due_nonfinite:
                    if _optimizer_has_stepped(optimizer):
                        with suppress(Exception):
                            lr_scheduler.step()
                    _record_unscaled_lrs(optimizer, lr_scheduler)
                    lr_mult, reset_state = _apply_nonfinite_recovery(
                        lr_mult=lr_mult,
                        skip_streak=int(nonfinite_skip_streak),
                    )
                    _apply_lr_mult(optimizer, lr_mult)
                    if reset_state:
                        with suppress(Exception):
                            optimizer.state.clear()
                    global_step += 1
                    consumed_micro_batches_committed = int(consumed_micro_batches)
                    if train_progress is not None:
                        train_progress.update(1)
                    if train_cfg.report_to != "none":
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
                            "step=%d | nonfinite_window_skipped=1 | reason=%s | streak=%d | total_skips=%d | "
                            "lr_mult=%.4f | opt_state_reset=%s | debug=%s",
                            int(global_step),
                            str(nonfinite_reason or "unknown"),
                            int(nonfinite_skip_streak),
                            int(nonfinite_skip_total),
                            float(lr_mult),
                            bool(reset_state),
                            str(nonfinite_debug_path) if nonfinite_debug_path is not None else "n/a",
                        )

                    last_saved_step = _save_periodic_checkpoint_if_due(
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
                    continue
                raise RuntimeError("Accumulation window produced no synchronized optimization step.")

            loss_for_metrics = _finalize_window_metric_loss(
                accumulated_loss=loss_for_metrics,
                ga_steps=ga_steps,
                token_weighted_ga=token_weighted_ga,
            )

            if did_optimizer_step:
                if out is None:
                    raise RuntimeError("Accumulation window produced no forward pass outputs.")
                global_step += 1
                consumed_micro_batches_committed = int(consumed_micro_batches)
                if train_progress is not None:
                    train_progress.update(1)

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

                last_saved_step = _save_periodic_checkpoint_if_due(
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
        final_step = int(global_step)
        should_try_crash_save = (crash_reason is None) or int(getattr(accelerator, "num_processes", 1)) == 1
        if final_step > 0 and final_step != int(last_saved_step) and should_try_crash_save:
            try:
                final_ckpt = output_dir / f"checkpoint-{final_step}"
                _save_training_checkpoint(
                    accelerator=accelerator,
                    checkpoint_dir=final_ckpt,
                    output_dir=output_dir,
                    consumed_micro_batches=consumed_micro_batches_committed,
                    save_total_limit=int(train_cfg.save_total_limit),
                    log_label="final",
                    lr_mult=lr_mult,
                    optimizer_param_digest=param_digest,
                    global_step=int(final_step),
                    gradient_accumulation_steps=int(ga_steps),
                )
                last_saved_step = final_step
            except Exception as save_exc:
                logger.error(
                    "Final/crash-time checkpoint save failed at step %d: %s. "
                    "Progress since checkpoint-%d may be lost.",
                    int(final_step),
                    str(save_exc),
                    int(last_saved_step),
                    exc_info=True,
                )
        elif crash_reason is not None and not should_try_crash_save and accelerator.is_main_process:
            logger.warning(
                "Skipping crash-time final checkpoint save on distributed run "
                "(num_processes=%s) to avoid potential collective deadlocks after failure.",
                getattr(accelerator, "num_processes", "unknown"),
            )

        if (
            bool(train_cfg.export_hf_final)
            and crash_reason is None
            and bool(getattr(accelerator, "is_main_process", True))
        ):
            export_step = int(last_saved_step)
            if export_step <= 0 and int(global_step) > 0:
                export_step = int(global_step)
            if export_step > 0:
                checkpoint_dir = output_dir / f"checkpoint-{export_step}"
                if checkpoint_dir.exists():
                    with suppress(Exception):
                        _export_discriminator_hf_subprocess(
                            checkpoint_dir=checkpoint_dir,
                            output_dir=output_dir / "final_hf",
                        )
                else:
                    logger.warning(
                        "Skipping final export: checkpoint directory does not exist: %s",
                        checkpoint_dir,
                    )

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
                if train_cfg.report_to != "none":
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

        if train_cfg.report_to != "none":
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
