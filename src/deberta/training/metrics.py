"""Pretraining metrics and debug-artifact helpers."""

from __future__ import annotations

import gzip
import json
import logging
import math
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from deberta.config import (
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    _sync_legacy_train_aliases,
)
from deberta.training.run_management import _sanitize_run_label
from deberta.utils.io import dump_json
from deberta.utils.serialize import (
    coerce_dataclass_payload_types as _coerce_dataclass_payload_types,
)
from deberta.utils.serialize import (
    drop_none_recursive as _drop_none_recursive,
)
from deberta.utils.serialize import (
    mapping_from_config_obj as _config_obj_to_mapping,
)

logger = logging.getLogger(__name__)


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
