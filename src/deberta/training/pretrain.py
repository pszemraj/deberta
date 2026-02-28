"""Training loop and export utilities for DeBERTa v3 RTD pretraining."""

from __future__ import annotations

import gzip
import inspect
import json
import logging
import math
import re
import shutil
import time
from collections.abc import Iterator
from contextlib import nullcontext, suppress
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deberta.checkpoint_utils import (
    canonicalize_state_dict_keys,
    load_model_state_with_compile_key_remap,
    unwrap_compiled_model,
)
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
    validate_run_metadata_schema,
    validate_train_config,
    validate_training_workflow_options,
)
from deberta.data import DebertaV3ElectraCollator, PackedStreamingDataset, SequentialStreamingDataset
from deberta.data.collator import MLMConfig
from deberta.data.loading import load_hf_dataset
from deberta.data.streaming import PackedStreamingConfig
from deberta.io_utils import dump_json, load_json_mapping
from deberta.log_utils import setup_process_logging
from deberta.modeling import DebertaV3RTDPretrainer, build_backbone_configs, build_backbones
from deberta.modeling.export_utils import load_intersection_state_dict, merge_embeddings_into_export_backbone
from deberta.modeling.rtd import attention_mask_to_active_tokens, compute_generator_loss_term

logger = logging.getLogger(__name__)
_RUN_LABEL_CLEAN_RE = re.compile(r"[^A-Za-z0-9._-]+")
_NONFINITE_LR_BACKOFF = 0.5
_NONFINITE_LR_MULT_FLOOR = 0.01  # persistent multiplier floor (1% of scheduled)
_NONFINITE_LR_MULT_RECOVERY = 1.1  # gradual recovery factor per successful step
_NONFINITE_OPT_STATE_RESET_EVERY = 4


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
    if path.suffix == ".gz":
        with gzip.open(path, "at", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        return
    with path.open("a", encoding="utf-8", buffering=1) as f:
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


def _build_run_metadata(
    *,
    effective_compile_scope: str | None = None,
    compile_scope_reason: str | None = None,
) -> dict[str, Any]:
    """Build run-metadata payload stored alongside config snapshots.

    :param str | None effective_compile_scope: Resolved compile scope after auto-resolution.
    :param str | None compile_scope_reason: Reason for scope selection when auto-resolved.
    :return dict[str, Any]: Metadata mapping.
    """
    from deberta import __version__

    meta: dict[str, Any] = {
        "config_schema_version": int(RUN_CONFIG_SCHEMA_VERSION),
        "deberta_version": str(__version__),
    }
    if effective_compile_scope is not None:
        meta["effective_compile_scope"] = str(effective_compile_scope)
    if compile_scope_reason is not None:
        meta["compile_scope_reason"] = str(compile_scope_reason)
    return meta


def _sanitize_run_label(raw: str) -> str:
    """Normalize a free-form run label into a compact safe token.

    :param str raw: Raw label.
    :return str: Sanitized label.
    """
    cleaned = _RUN_LABEL_CLEAN_RE.sub("-", str(raw).strip()).strip("._-")
    return cleaned or "run"


def _resolve_output_dir(
    *,
    output_dir: str | None,
    project_name: str,
    config_path: str | Path | None,
) -> Path:
    """Resolve the concrete training output directory.

    :param str | None output_dir: User-configured output directory.
    :param str project_name: Tracker project namespace.
    :param str | Path | None config_path: Optional config file path for naming hint.
    :return Path: Concrete output directory path.
    """
    if output_dir is not None and str(output_dir).strip():
        return Path(str(output_dir))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_hint = "run"
    if config_path is not None:
        name_hint = Path(config_path).stem or "run"
    run_name = _sanitize_run_label(name_hint)
    project = _sanitize_run_label(project_name)
    return Path("runs") / project / f"{stamp}_{run_name}"


def _resolve_output_dir_for_accelerator(
    *,
    accelerator: Any,
    output_dir: str | None,
    project_name: str,
    config_path: str | Path | None,
    broadcast_fn: Any | None = None,
) -> Path:
    """Resolve output_dir with deterministic auto-naming across distributed ranks.

    :param Any accelerator: Accelerator-like runtime exposing ``is_main_process`` and ``num_processes``.
    :param str | None output_dir: User-configured output directory.
    :param str project_name: Tracker project namespace.
    :param str | Path | None config_path: Optional config path used for auto-naming.
    :param Any | None broadcast_fn: Optional callable compatible with
        ``accelerate.utils.broadcast_object_list``.
    :raises RuntimeError: If distributed broadcast fails or returns an empty path.
    :return Path: Concrete output directory path.
    """
    explicit = output_dir is not None and str(output_dir).strip()
    if explicit:
        return Path(str(output_dir))

    num_processes = int(getattr(accelerator, "num_processes", 1))
    if num_processes <= 1:
        return _resolve_output_dir(
            output_dir=None,
            project_name=project_name,
            config_path=config_path,
        )

    if broadcast_fn is None:
        try:
            from accelerate.utils import broadcast_object_list as broadcast_fn  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Distributed auto output_dir resolution requires accelerate.utils.broadcast_object_list."
            ) from exc

    shared: list[str | None] = [None]
    if bool(getattr(accelerator, "is_main_process", False)):
        shared[0] = str(
            _resolve_output_dir(
                output_dir=None,
                project_name=project_name,
                config_path=config_path,
            )
        )

    try:
        broadcast_fn(shared, from_process=0)
    except Exception as exc:
        raise RuntimeError("Failed to broadcast auto-resolved output_dir across ranks.") from exc

    resolved = shared[0]
    if resolved is None or not str(resolved).strip():
        raise RuntimeError("Broadcasted output_dir is empty in distributed auto-output-dir resolution.")
    return Path(str(resolved))


def _init_trackers(
    *,
    accelerator: Any,
    project_name: str,
    tracker_cfg: dict[str, Any],
    report_to: str,
    run_name: str,
) -> None:
    """Initialize Accelerate trackers with tracker-specific kwargs when available.

    :param Any accelerator: Accelerator instance.
    :param str project_name: Tracker project name.
    :param dict[str, Any] tracker_cfg: Tracker configuration payload.
    :param str report_to: Selected tracker backend.
    :param str run_name: Effective run name.
    """
    call_kwargs: dict[str, Any] = {
        "project_name": project_name,
        "config": tracker_cfg,
    }

    init_kwargs: dict[str, Any] = {}
    if str(report_to).strip().lower() == "wandb":
        init_kwargs = {"wandb": {"name": run_name}}

    if not init_kwargs:
        accelerator.init_trackers(**call_kwargs)
        return

    try:
        accelerator.init_trackers(**call_kwargs, init_kwargs=init_kwargs)
    except TypeError as err:
        # Some accelerate builds do not accept init_kwargs; retry without it.
        if "init_kwargs" not in str(err):
            raise
        logger.warning(
            "Accelerate init_trackers() rejected init_kwargs; W&B run name "
            "cannot be set explicitly on this runtime."
        )
        accelerator.init_trackers(**call_kwargs)


def _setup_wandb_watch(
    *,
    accelerator: Any,
    wandb_run: Any | None,
    model: torch.nn.Module,
    watch_mode: str,
    watch_log_freq: int,
) -> bool:
    """Configure W&B gradient/parameter watching for the active run.

    :param Any accelerator: Accelerator runtime.
    :param Any | None wandb_run: W&B tracker object, if available.
    :param torch.nn.Module model: Training model (possibly wrapped).
    :param str watch_mode: W&B watch mode.
    :param int watch_log_freq: Watch logging frequency.
    :return bool: True when watch setup succeeds, else False.
    """
    mode = str(watch_mode).strip().lower()
    if mode == "none":
        return False
    if not bool(getattr(accelerator, "is_main_process", True)):
        return False

    watch_target = unwrap_compiled_model(accelerator, model)
    freq = max(1, int(watch_log_freq))

    watch_owner = wandb_run
    watch_fn = getattr(wandb_run, "watch", None) if wandb_run is not None else None
    if not callable(watch_fn):
        with suppress(Exception):
            tracker_obj = accelerator.get_tracker("wandb", unwrap=True)
            watch_owner = tracker_obj
            watch_fn = getattr(tracker_obj, "watch", None)

    if not callable(watch_fn):
        logger.warning("W&B tracker does not expose watch(); skipping model watch setup.")
        return False

    try:
        watch_fn(watch_target, log=mode, log_freq=freq)
    except TypeError:
        watch_fn(watch_target, log=mode)

    owner_type = type(watch_owner).__name__ if watch_owner is not None else "unknown"
    logger.info("Enabled W&B watch (mode=%s, log_freq=%d, tracker=%s).", mode, freq, owner_type)
    return True


def _validate_run_metadata(path: Path) -> None:
    """Validate on-disk run metadata schema compatibility.

    :param Path path: Metadata file path.
    :raises ValueError: If metadata is malformed or schema-incompatible.
    """
    raw = load_json_mapping(path)
    validate_run_metadata_schema(raw, source=str(path))


def _dump_yaml_mapping(payload: dict[str, Any], path: Path) -> None:
    """Write a mapping payload to YAML, with JSON fallback if PyYAML is unavailable.

    :param dict[str, Any] payload: Mapping payload.
    :param Path path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore
    except Exception:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        return

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True, default_flow_style=False, allow_unicode=False)


def _resolved_config_payload(
    *, model_cfg: ModelConfig, data_cfg: DataConfig, train_cfg: TrainConfig
) -> dict[str, dict[str, Any]]:
    """Build resolved nested config payload for YAML snapshots.

    :param ModelConfig model_cfg: Resolved model config.
    :param DataConfig data_cfg: Resolved data config.
    :param TrainConfig train_cfg: Resolved train config.
    :return dict[str, dict[str, Any]]: Nested resolved payload.
    """
    return {
        "model": asdict(model_cfg),
        "data": asdict(data_cfg),
        "train": asdict(train_cfg),
    }


def _persist_config_yaml_snapshots(
    *,
    output_dir: Path,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    config_path: str | Path | None,
    is_main_process: bool,
) -> None:
    """Persist original/resolved YAML config snapshots in ``output_dir``.

    :param Path output_dir: Training output directory.
    :param ModelConfig model_cfg: Resolved model config.
    :param DataConfig data_cfg: Resolved data config.
    :param TrainConfig train_cfg: Resolved train config.
    :param str | Path | None config_path: Optional source config file path.
    :param bool is_main_process: Whether current process owns filesystem writes.
    """
    if not bool(is_main_process):
        return

    resolved_payload = _resolved_config_payload(model_cfg=model_cfg, data_cfg=data_cfg, train_cfg=train_cfg)
    resolved_path = output_dir / "config_resolved.yaml"
    _dump_yaml_mapping(resolved_payload, resolved_path)

    original_path = output_dir / "config_original.yaml"
    if config_path is None:
        _dump_yaml_mapping(resolved_payload, original_path)
        return

    source = Path(config_path).expanduser().resolve()
    if source.exists():
        original_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return

    # Backfill with resolved payload when source path is unavailable.
    _dump_yaml_mapping(resolved_payload, original_path)


def _upload_wandb_original_config(
    *,
    accelerator: Any,
    wandb_run: Any | None,
    config_original_path: Path,
    run_name: str,
) -> bool:
    """Upload ``config_original.yaml`` to W&B using a run-name-specific filename.

    :param Any accelerator: Accelerator runtime.
    :param Any | None wandb_run: W&B tracker object, if available.
    :param Path config_original_path: Local config-original snapshot path.
    :param str run_name: Effective run name.
    :return bool: True when upload succeeds, else False.
    """
    if not bool(getattr(accelerator, "is_main_process", True)):
        return False
    if not config_original_path.exists():
        return False

    owner = wandb_run
    if owner is None:
        with suppress(Exception):
            owner = accelerator.get_tracker("wandb", unwrap=True)
    if owner is None:
        return False

    safe_run_name = _sanitize_run_label(run_name)
    upload_name = f"config_deberta_{safe_run_name}"
    with TemporaryDirectory(prefix="deberta-wandb-config-") as tmp_dir:
        staged = Path(tmp_dir) / upload_name
        staged.write_text(config_original_path.read_text(encoding="utf-8"), encoding="utf-8")

        save_fn = getattr(owner, "save", None)
        if callable(save_fn):
            try:
                save_fn(str(staged), base_path=tmp_dir, policy="now")
            except TypeError:
                try:
                    save_fn(str(staged), base_path=tmp_dir)
                except TypeError:
                    save_fn(str(staged))
            logger.info("Uploaded original config to W&B as %s", upload_name)
            return True

        log_artifact_fn = getattr(owner, "log_artifact", None)
        if callable(log_artifact_fn):
            with suppress(Exception):
                import wandb  # type: ignore

                artifact = wandb.Artifact(name=f"config-{safe_run_name}", type="config")
                artifact.add_file(str(staged), name=upload_name)
                log_artifact_fn(artifact)
                logger.info("Uploaded original config artifact to W&B as %s", upload_name)
                return True

    logger.warning("W&B tracker does not support save()/log_artifact(); skipping config upload.")
    return False


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
    :raises ValueError: If ``resume_from_checkpoint='auto'`` is requested for a non-empty output
        directory that does not contain any ``checkpoint-*`` folders.
    :return str | None: Concrete checkpoint path, or ``None``.
    """
    if not resume_from_checkpoint:
        return None

    if str(resume_from_checkpoint).lower() != "auto":
        return str(resume_from_checkpoint)

    latest = _find_latest_checkpoint(output_dir)
    if latest is None:
        has_existing_contents = output_dir.exists() and any(output_dir.iterdir())
        if has_existing_contents:
            raise ValueError(
                "resume_from_checkpoint=auto was requested but no checkpoint-* directories were found in "
                f"non-empty output_dir={output_dir}. Clean the directory, enable "
                "train.overwrite_output_dir=true, or provide an explicit checkpoint path."
            )
        if is_main_process:
            logger.info("resume_from_checkpoint=auto but no checkpoint-* dirs found; starting from scratch.")
        return None
    return str(latest)


def _load_resume_state_with_compile_fallback(
    accelerator: Any, model: torch.nn.Module, checkpoint_dir: str
) -> None:
    """Load resume state with fallback for ``torch.compile`` wrapper key mismatches.

    :param Any accelerator: Accelerator instance.
    :param torch.nn.Module model: Potentially wrapped training model.
    :param str checkpoint_dir: Resume checkpoint directory.
    :raises RuntimeError: If both normal and fallback load paths fail.
    :return None: None.
    """
    try:
        accelerator.load_state(checkpoint_dir)
        return
    except RuntimeError as err:
        if "_orig_mod" not in str(err):
            raise

    logger.warning(
        "Checkpoint model key mismatch due compile wrappers detected; retrying resume with "
        "strict=False and canonical key remap."
    )
    accelerator.load_state(checkpoint_dir, strict=False)
    unwrapped = unwrap_compiled_model(accelerator, model)
    stats = load_model_state_with_compile_key_remap(unwrapped, Path(checkpoint_dir))
    logger.info(
        "Resume model remap loaded %d tensors from %s.",
        int(stats["matched"]),
        checkpoint_dir,
    )


def _persist_or_validate_run_configs(
    *,
    output_dir: Path,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    resume_checkpoint: str | None,
    config_path: str | Path | None = None,
    is_main_process: bool,
    effective_compile_scope: str | None = None,
    compile_scope_reason: str | None = None,
) -> None:
    """Persist new config snapshots or validate existing snapshots on resume.

    :param Path output_dir: Training output directory.
    :param ModelConfig model_cfg: Current model config.
    :param DataConfig data_cfg: Current data config.
    :param TrainConfig train_cfg: Current train config.
    :param str | None resume_checkpoint: Resolved checkpoint path, if resuming.
    :param str | Path | None config_path: Optional original config-file path.
    :param bool is_main_process: Whether this process owns writes.
    :param str | None effective_compile_scope: Resolved compile scope for metadata.
    :param str | None compile_scope_reason: Reason for compile scope selection.
    :raises ValueError: If resume mode detects incompatible model/data config snapshots.
    """
    model_cfg_path = output_dir / "model_config.json"
    data_cfg_path = output_dir / "data_config.json"
    train_cfg_path = output_dir / "train_config.json"
    run_meta_path = output_dir / "run_metadata.json"

    run_meta = _build_run_metadata(
        effective_compile_scope=effective_compile_scope,
        compile_scope_reason=compile_scope_reason,
    )

    has_saved_model_data = model_cfg_path.exists() and data_cfg_path.exists()
    if resume_checkpoint is not None and has_saved_model_data:
        if run_meta_path.exists():
            _validate_run_metadata(run_meta_path)
            # Check for compile scope drift on resume.
            if is_main_process and effective_compile_scope is not None:
                saved_meta = load_json_mapping(run_meta_path)
                saved_scope = saved_meta.get("effective_compile_scope")
                if saved_scope is not None and str(saved_scope) != str(effective_compile_scope):
                    logger.warning(
                        "Effective compile scope changed on resume: "
                        f"was {saved_scope!r}, now {effective_compile_scope!r}. "
                        "This may affect compiled graph caching but is recoverable."
                    )
        elif is_main_process:
            # Backfill schema metadata for older runs once compatibility has been checked.
            dump_json(run_meta, run_meta_path)

        saved_model_cfg = ModelConfig(**load_json_mapping(model_cfg_path))
        saved_data_cfg = DataConfig(**load_json_mapping(data_cfg_path))
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
        _persist_config_yaml_snapshots(
            output_dir=output_dir,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            config_path=config_path,
            is_main_process=is_main_process,
        )
        return

    if is_main_process:
        dump_json(asdict(model_cfg), model_cfg_path)
        dump_json(asdict(data_cfg), data_cfg_path)
        dump_json(asdict(train_cfg), train_cfg_path)
        dump_json(run_meta, run_meta_path)
    _persist_config_yaml_snapshots(
        output_dir=output_dir,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        config_path=config_path,
        is_main_process=is_main_process,
    )


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


def _resolve_effective_mixed_precision_or_raise(requested: str) -> str:
    """Normalize mixed precision and fail fast when bf16 is unavailable.

    :param str requested: User-requested mixed precision mode.
    :raises RuntimeError: If bf16 is requested but runtime preflight fails.
    :return str: Effective mixed precision mode.
    """
    mixed_precision = normalize_mixed_precision(requested)
    if mixed_precision == "bf16" and not _bf16_runtime_sanity_check():
        raise RuntimeError(
            "train.mixed_precision=bf16 requested but bf16 preflight failed. "
            "Set train.mixed_precision=no explicitly if you want to continue in full precision."
        )
    return mixed_precision


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
    with suppress(Exception):
        reduced = accelerator.reduce(local, reduction="sum")
        count = int(reduced.reshape(-1)[0].item())
        return count > 0
    return local_flag > 0


@torch.no_grad()
def _sanitize_nonfinite_gradients_(model: torch.nn.Module) -> int:
    """Replace non-finite gradient entries in-place with zeros.

    :param torch.nn.Module model: Model whose gradients are sanitized.
    :return int: Number of non-finite gradient elements replaced.
    """
    replaced = 0
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        if grad.is_sparse:
            grad = grad.coalesce()
            values = grad.values()
            bad = ~torch.isfinite(values)
            count = int(bad.sum().item())
            if count > 0:
                replaced += count
                values.masked_fill_(bad, 0.0)
                param.grad = grad
            continue

        bad = ~torch.isfinite(grad)
        count = int(bad.sum().item())
        if count <= 0:
            continue
        replaced += count
        grad.masked_fill_(bad, 0.0)
    return replaced


def _apply_lr_mult(optimizer: torch.optim.Optimizer, lr_mult: float) -> None:
    """Scale all optimizer param-group LRs by a persistent multiplier.

    :param torch.optim.Optimizer optimizer: Runtime optimizer.
    :param float lr_mult: Multiplier to apply (typically in (0, 1]).
    """
    for group in optimizer.param_groups:
        group["lr"] = float(group["lr"]) * float(lr_mult)


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


def _sync_discriminator_embeddings_if_available(model: torch.nn.Module) -> None:
    """Sync discriminator embedding buffers if the model exposes the hook.

    :param torch.nn.Module model: Unwrapped runtime model.
    """
    fn = getattr(model, "sync_discriminator_embeddings_from_generator", None)
    if callable(fn):
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
    safe_offending = _RUN_LABEL_CLEAN_RE.sub("_", str(offending)).strip("._-") or "nonfinite"
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
    # On some PyTorch builds, max-autotune paths still query legacy allow_tf32
    # and can error if only the new fp32_precision API has been configured.
    return compile_mode.startswith("max-autotune")


def _resolve_compile_scope(
    *,
    requested_scope: str,
    model_cfg: ModelConfig,
    compile_mode: str,
    compile_backend: str,
    block_cross_document_attention: bool = False,
) -> tuple[str, str | None]:
    """Resolve effective compile scope with default-mode stability fallback.

    :param str requested_scope: Requested canonical compile scope.
    :param ModelConfig model_cfg: Model configuration.
    :param str compile_mode: Canonical torch.compile mode.
    :param str compile_backend: Compile backend name.
    :param bool block_cross_document_attention: Whether doc-blocking is enabled.
    :return tuple[str, str | None]: Effective scope and optional reason message.
    """
    if requested_scope != "auto":
        return requested_scope, None

    backbone_type = str(getattr(model_cfg, "backbone_type", "")).strip().lower()
    # Empirically, full-backbone inductor compile on HF DeBERTa v2 defaults can drift
    # during train-mode updates. Auto scope keeps compile on the dominant FFN FLOPs
    # while leaving attention + embeddings eager.
    if backbone_type == "hf_deberta_v2" and compile_mode == "default" and compile_backend == "inductor":
        return (
            "ffn",
            "auto scope selected FFN-only fallback for hf_deberta_v2 (default+inductor)",
        )
    # Doc-blocking batches alternate between None and 3D masks (single-doc vs multi-doc),
    # causing mask shape churn under compile. Downgrade to FFN-only to avoid recompilation.
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
    """Return warning text for empirically unstable HFv2 full-backbone compile requests.

    :param ModelConfig model_cfg: Model configuration.
    :param bool compile_enabled: Whether compile is active.
    :param str compile_scope: Effective compile scope.
    :param str compile_backend: Compile backend.
    :return str | None: Warning message for unstable configuration, else ``None``.
    """
    if not bool(compile_enabled):
        return None
    if str(getattr(model_cfg, "backbone_type", "")).strip().lower() != "hf_deberta_v2":
        return None
    if str(compile_backend).strip().lower() != "inductor":
        return None
    if str(compile_scope).strip().lower() != "backbones":
        return None
    return (
        "Requested full-backbone torch.compile for hf_deberta_v2 + inductor. "
        "This path is empirically unstable and not recommended for production training. "
        "Preferred stable path: train.torch_compile_scope=ffn, "
        "model.hf_attention_kernel=stable, train.torch_compile_mode=default."
    )


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

    eps = float(cfg.adam_epsilon)
    if str(mixed_precision).strip().lower() == "bf16" and eps < 1e-6:
        eps = 1e-6
        logger.warning("Raised Adam epsilon to 1e-6 for bf16 stability.")

    opt = torch.optim.AdamW(
        groups,
        lr=disc_lr,
        betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
        eps=eps,
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


def _build_doc_block_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Build a pairwise ``(B, S, S)`` attention keep-mask from document ids on-device.

    Contract: the diagonal is unconditionally True for **all** positions (including
    pad/dead rows).  This eliminates all-False query rows that cause NaN in SDPA
    backends, so attention layers do not need per-layer dead-row patching.

    :param torch.Tensor doc_ids: Document id tensor ``(B, S)`` with 0 for padding.
    :return torch.Tensor: Bool keep-mask ``(B, S, S)``.
    """
    active = doc_ids.ne(0)
    same_doc = doc_ids[:, :, None].eq(doc_ids[:, None, :])
    keep = same_doc & active[:, :, None] & active[:, None, :]
    # Unconditional diagonal: every position self-attends. Dead/pad row outputs are
    # zeroed by query_keep_tokens in the attention layer.
    eye = torch.eye(doc_ids.shape[1], dtype=torch.bool, device=doc_ids.device).unsqueeze(0)
    return keep | eye


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


def _compute_disc_active_mask(
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    special_token_mask: torch.Tensor | None,
    pad_token_id: int | None,
) -> torch.Tensor:
    """Compute discriminator-active token mask before model forward.

    This mirrors the RTD discriminator masking semantics while accounting for
    masked positions that will be replaced before discriminator scoring.

    :param torch.Tensor input_ids: Input token ids.
    :param torch.Tensor labels: MLM labels (-100 for non-masked positions).
    :param torch.Tensor | None attention_mask: Optional attention mask.
    :param torch.Tensor | None special_token_mask: Boolean vocab mask (V,) with True for special token ids.
    :param int | None pad_token_id: Padding token id.
    :return torch.Tensor: Boolean active-token mask.
    """
    active = attention_mask_to_active_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
    )

    if (
        special_token_mask is None
        or not isinstance(special_token_mask, torch.Tensor)
        or special_token_mask.numel() == 0
    ):
        return active

    # Indexing a boolean vocab mask is substantially cheaper (and compile-friendly)
    # than per-step torch.isin + device tensor construction.
    if special_token_mask.device != input_ids.device:
        special_token_mask = special_token_mask.to(device=input_ids.device)
    special = special_token_mask[input_ids]

    # Masked positions are replaced before discriminator scoring and should not
    # be excluded solely because their pre-corruption token may be special.
    masked_positions = labels.ne(-100)
    special = special & (~masked_positions)
    return active & (~special)


def _count_rtd_tokens_for_batch(
    batch: dict[str, torch.Tensor],
    *,
    special_token_mask: torch.Tensor | None,
    pad_token_id: int | None,
) -> tuple[float, float]:
    """Return generator/discriminator active-token counts for one microbatch.

    :param dict[str, torch.Tensor] batch: Microbatch tensors.
    :param torch.Tensor | None special_token_mask: Boolean vocab mask (V,) with True for special token ids.
    :param int | None pad_token_id: Padding token id.
    :return tuple[float, float]: (generator_count, discriminator_count).
    """
    labels = batch["labels"]
    gen_count = float(labels.ne(-100).sum().item())
    disc_active = _compute_disc_active_mask(
        input_ids=batch["input_ids"],
        labels=labels,
        attention_mask=batch.get("attention_mask"),
        special_token_mask=special_token_mask,
        pad_token_id=pad_token_id,
    )
    disc_count = float(disc_active.sum().item())
    return gen_count, disc_count


def _count_input_tokens_for_batch(batch: dict[str, torch.Tensor]) -> float:
    """Return non-padding input-token count for one microbatch.

    :param dict[str, torch.Tensor] batch: Microbatch mapping.
    :return float: Count of active input tokens.
    """
    attention_mask = batch.get("attention_mask")
    input_ids = batch.get("input_ids")
    if isinstance(attention_mask, torch.Tensor):
        if isinstance(input_ids, torch.Tensor):
            active = attention_mask_to_active_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=None,
            )
            return float(active.detach().sum().item())

        mask = attention_mask.detach().to(dtype=torch.bool)
        if mask.ndim == 4:
            mask = mask[:, 0] if mask.shape[1] == 1 else mask.any(dim=1)
        if mask.ndim == 3:
            mask = torch.diagonal(mask, dim1=-2, dim2=-1)
        return float(mask.sum().item())

    if not isinstance(input_ids, torch.Tensor):
        return 0.0
    return float(input_ids.numel())


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


def _load_checkpoint_data_progress(checkpoint_dir: Path) -> tuple[int | None, float]:
    """Load persisted data progress and LR multiplier for resume alignment.

    :param Path checkpoint_dir: Checkpoint directory.
    :return tuple[int | None, float]: (consumed micro-batch count or None, lr_mult).
    """
    path = checkpoint_dir / "data_state.json"
    if not path.exists():
        return None, 1.0
    try:
        raw = json.loads(path.read_text())
        val = raw.get("consumed_micro_batches", None)
        consumed = max(0, int(val)) if val is not None else None
        lr_mult = float(raw.get("lr_mult", 1.0))
        return consumed, lr_mult
    except Exception:
        return None, 1.0


def _save_checkpoint_data_progress(
    *, checkpoint_dir: Path, consumed_micro_batches: int, lr_mult: float = 1.0
) -> None:
    """Persist data iterator progress and LR multiplier next to a checkpoint.

    :param Path checkpoint_dir: Checkpoint directory.
    :param int consumed_micro_batches: Number of consumed micro-batches.
    :param float lr_mult: Persistent nonfinite recovery LR multiplier.
    """
    dump_json(
        {
            "consumed_micro_batches": int(max(0, consumed_micro_batches)),
            "lr_mult": float(lr_mult),
        },
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
    lr_mult: float = 1.0,
) -> None:
    """Save one training checkpoint with collective state-dict write.

    :param Any accelerator: Accelerate runtime object.
    :param Path checkpoint_dir: Destination checkpoint directory.
    :param Path output_dir: Parent output directory for checkpoint rotation.
    :param int consumed_micro_batches: Data progress to persist.
    :param int save_total_limit: Number of checkpoints to retain.
    :param str log_label: Logging label for this save.
    :param float lr_mult: Persistent nonfinite recovery LR multiplier.
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
            lr_mult=float(lr_mult),
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


_REPO_URL = "https://github.com/pszemraj/deberta"

# Keys injected by this repo's training configs that should not leak into
# exported HF model configs (they are not part of the upstream schema).
_EXPORT_CONFIG_STRIP_KEYS = frozenset(
    {
        "hf_attention_kernel",
        "use_rmsnorm_heads",
        "cls_token_id",
        "mask_token_id",
        "sep_token_id",
        "bos_token_id",
        "eos_token_id",
        "legacy",
    }
)


def _clean_exported_config(config_path: Path) -> None:
    """Remove training-internal keys from an exported config.json in-place.

    :param Path config_path: Path to config.json.
    """
    if not config_path.exists():
        return
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        cleaned = {k: v for k, v in raw.items() if k not in _EXPORT_CONFIG_STRIP_KEYS}
        config_path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def _write_export_readme(
    output_dir: Path,
    *,
    model_cfg: Any,
    train_cfg: Any,
    embedding_sharing: str,
) -> None:
    """Write a basic README.md and LICENSE to the export directory.

    :param Path output_dir: Export destination directory.
    :param Any model_cfg: Model configuration dataclass.
    :param Any train_cfg: Training configuration dataclass.
    :param str embedding_sharing: Embedding sharing mode.
    """
    backbone = str(getattr(model_cfg, "backbone_type", "unknown"))
    hidden = int(getattr(model_cfg, "hidden_size", 0))
    layers = int(getattr(model_cfg, "num_hidden_layers", 0) or 0)
    heads = int(getattr(model_cfg, "num_attention_heads", 0) or 0)
    seq_len = int(getattr(train_cfg, "max_seq_length", 0) or 0)
    if seq_len == 0:
        seq_len = int(getattr(model_cfg, "max_position_embeddings", 0) or 0)
    steps = int(getattr(train_cfg, "max_steps", 0) or 0)

    if backbone == "rope":
        arch_desc = "RoPE encoder (RMSNorm, SwiGLU, rotary embeddings)"
        usage_snippet = """from transformers import AutoTokenizer
from deberta.modeling.rope_encoder import DebertaRoPEModel

model = DebertaRoPEModel.from_pretrained("path/to/this/dir")
tokenizer = AutoTokenizer.from_pretrained("path/to/this/dir")
"""
        compatibility_note = (
            "Note: RoPE exports use a custom `model_type` (`deberta-rope`) and are not currently "
            "loadable via `transformers.AutoModel.from_pretrained(...)` without custom auto-class registration."
        )
    else:
        arch_desc = "DeBERTa-v2 (disentangled attention, LayerNorm)"
        usage_snippet = """from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("path/to/this/dir")
tokenizer = AutoTokenizer.from_pretrained("path/to/this/dir")
"""
        compatibility_note = ""

    readme = f"""---
library_name: transformers
tags:
- deberta
- encoder
- rtd
- fill-mask
license: mit
---

# {backbone}-{hidden}h-{layers}L-{heads}H

RTD-pretrained encoder ({arch_desc}).

| Parameter | Value |
|---|---|
| Backbone | `{backbone}` |
| Hidden size | {hidden} |
| Layers | {layers} |
| Attention heads | {heads} |
| Max sequence length | {seq_len} |
| Embedding sharing | `{embedding_sharing}` |
| Training steps | {steps} |

## Training

Pretrained with replaced-token detection (RTD / ELECTRA-style) using
[pszemraj/deberta]({_REPO_URL}).

## Usage

```python
{usage_snippet}
```
{compatibility_note}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    mit_license = (
        "MIT License\n\n"
        "Copyright (c) 2025-2026 Peter Szemraj\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        'of this software and associated documentation files (the "Software"), to deal\n'
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n\n"
        "The above copyright notice and this permission notice shall be included in all\n"
        "copies or substantial portions of the Software.\n\n"
        'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n'
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
        "SOFTWARE.\n"
    )
    (output_dir / "LICENSE").write_text(mit_license, encoding="utf-8")


def _export_discriminator_hf(
    *,
    accelerator: Any,
    model: DebertaV3RTDPretrainer,
    tokenizer: Any,
    output_dir: Path,
    embedding_sharing: str,
    model_cfg: Any = None,
    train_cfg: Any = None,
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
    :param Any model_cfg: Optional model config for README generation.
    :param Any train_cfg: Optional training config for README generation.
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
        unwrapped = unwrap_compiled_model(accelerator, model)

        # Try to gather state dicts via accelerator (preferred for distributed).
        disc_mod = getattr(unwrapped, "discriminator", None)
        gen_mod = getattr(unwrapped, "generator", None)
        if disc_mod is None or gen_mod is None:
            raise RuntimeError(
                "Unwrapped RTD model must expose discriminator and generator modules for export."
            )
        disc_sd = canonicalize_state_dict_keys(dict(accelerator.get_state_dict(disc_mod)))
        gen_sd = canonicalize_state_dict_keys(dict(accelerator.get_state_dict(gen_mod)))

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
            fp32_accumulate=True,
        )

        tokenizer.save_pretrained(str(output_dir))
        export_disc.save_pretrained(str(output_dir), safe_serialization=True)

        # Strip training-internal keys from the exported config.json.
        _clean_exported_config(output_dir / "config.json")

        dump_json({"embedding_sharing": embedding_sharing}, output_dir / "export_meta.json")

        # Write README.md and LICENSE.
        if model_cfg is not None and train_cfg is not None:
            _write_export_readme(
                output_dir,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                embedding_sharing=embedding_sharing,
            )

        logger.info(f"Exported discriminator to: {output_dir}")

    except Exception as e:
        logger.warning(
            "HF export failed (common under FSDP2 + SHARDED_STATE_DICT). "
            "Use `deberta export` after training for a guaranteed consolidation+export. "
            f"Error: {e}"
        )


def run_pretraining(
    *,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    config_path: str | Path | None = None,
) -> None:
    """Run RTD pretraining with Accelerate/FSDP2-compatible plumbing.

    :param ModelConfig model_cfg: Model configuration.
    :param DataConfig data_cfg: Data configuration.
    :param TrainConfig train_cfg: Training configuration.
    :param str | Path | None config_path: Optional source config path for auto output-dir naming.
    """
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    # Accelerator first so we know ranks.
    log_with = None if train_cfg.report_to == "none" else train_cfg.report_to
    mixed_precision = _resolve_effective_mixed_precision_or_raise(train_cfg.mixed_precision)
    compile_mode = _normalize_torch_compile_mode(train_cfg.torch_compile_mode)
    compile_enabled = _resolve_compile_enabled_or_raise(train_cfg.torch_compile)
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
    )
    train_cfg.output_dir = str(output_dir)
    if accelerator.is_main_process and (
        configured_output_dir is None or not str(configured_output_dir).strip()
    ):
        logger.info("train.output_dir unset; auto-selected output_dir=%s", output_dir)

    # Make/validate output dir on main.
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
    )

    # Optimizer + scheduler
    optimizer = _build_optimizer(
        model,
        train_cfg,
        backbone_type=str(model_cfg.backbone_type),
        compile_enabled=compile_enabled,
        compile_scope=compile_scope,
        mixed_precision=mixed_precision,
    )
    lr_scheduler = _build_scheduler(optimizer, train_cfg)

    # Prepare (wrap for DDP/FSDP etc)
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # GDES embedding sharing uses synced non-trainable base weights inside discriminator embeddings.
    # Those tensors must be initialized/synced from generator weights before any compiled forward runs.
    with suppress(Exception):
        _sync_discriminator_embeddings_if_available(unwrap_compiled_model(accelerator, model))

    # torch.compile
    #
    # Compiling the *entire* RTD wrapper (generator sampling + corruption + discriminator)
    # is high-risk: it contains dynamic indexing, RNG sampling, and Python-side caching.
    # Those patterns are frequent sources of graph breaks, cudagraph partitioning, and
    # (in some PyTorch builds) silent numerical miscompiles.
    #
    # The stable + fast approach is to compile only the heavy transformer backbones.
    if compile_enabled:
        try:
            if compile_scope_reason:
                logger.warning(compile_scope_reason)

            compile_kwargs: dict[str, Any] = {"mode": compile_mode, "backend": compile_backend}
            # Backbones run with fixed sequence length (packing) and stable shapes.
            # Prefer static compilation for better perf and fewer guards.
            try:
                compile_params = inspect.signature(torch.compile).parameters  # type: ignore[attr-defined]
                if "dynamic" in compile_params:
                    compile_kwargs["dynamic"] = False
            except Exception:
                pass

            unwrapped = unwrap_compiled_model(accelerator, model)
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

    global_step = 0
    consumed_micro_batches = 0
    last_saved_step = 0
    lr_mult = 1.0
    nonfinite_skip_total = 0
    nonfinite_skip_streak = 0
    crash_type: str | None = None
    crash_reason: str | None = None
    crash_step: int | None = None
    exit_code = 0
    train_started_at = time.perf_counter()
    metrics_path = output_dir / "metrics.jsonl.gz"
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
            tracker_cfg = {
                "model": asdict(model_cfg),
                "data": asdict(data_cfg),
                "train": asdict(train_cfg),
            }
            if train_cfg.run_name is not None and str(train_cfg.run_name).strip():
                tracker_run_name = str(train_cfg.run_name).strip()
            else:
                tracker_run_name = output_dir.name
            _init_trackers(
                accelerator=accelerator,
                project_name=str(train_cfg.project_name).strip(),
                tracker_cfg=tracker_cfg,
                report_to=str(train_cfg.report_to),
                run_name=tracker_run_name,
            )
            if str(train_cfg.report_to).lower() == "wandb":
                with suppress(Exception):
                    wandb_run = accelerator.get_tracker("wandb", unwrap=True)
                with suppress(Exception):
                    _upload_wandb_original_config(
                        accelerator=accelerator,
                        wandb_run=wandb_run,
                        config_original_path=output_dir / "config_original.yaml",
                        run_name=tracker_run_name,
                    )
                with suppress(Exception):
                    _setup_wandb_watch(
                        accelerator=accelerator,
                        wandb_run=wandb_run,
                        model=model,
                        watch_mode=train_cfg.wandb_watch,
                        watch_log_freq=int(train_cfg.wandb_watch_log_freq),
                    )

        # Resume
        if ckpt:
            logger.info(f"Resuming from checkpoint: {ckpt}")
            _load_resume_state_with_compile_fallback(
                accelerator=accelerator,
                model=model,
                checkpoint_dir=ckpt,
            )

            # GDES base weights are non-persistent and must be refreshed after loading a checkpoint.
            with suppress(Exception):
                _sync_discriminator_embeddings_if_available(unwrap_compiled_model(accelerator, model))

            global_step = _parse_checkpoint_step(ckpt)
            last_saved_step = global_step
            restored, restored_lr_mult = _load_checkpoint_data_progress(Path(ckpt))
            if restored is None:
                # Back-compat fallback for older checkpoints without data_state.json.
                restored = global_step * int(train_cfg.gradient_accumulation_steps)
                logger.warning(
                    "Checkpoint missing data_state.json; approximating data replay offset "
                    f"as global_step * grad_accum = {restored} micro-batches."
                )
            consumed_micro_batches = int(restored)
            lr_mult = float(restored_lr_mult)
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

        train_iter = _cycle_dataloader(train_loader)
        ga_steps = int(train_cfg.gradient_accumulation_steps)
        token_weighted_ga = bool(train_cfg.token_weighted_gradient_accumulation)
        unwrapped_model = unwrap_compiled_model(accelerator, model)
        # Boolean vocab mask used both by the RTD module and token-weighted GA.
        # Keep a CPU copy for token counting to avoid per-microbatch GPU->CPU transfers
        # before `_move_batch_to_device` runs.
        special_token_mask = getattr(unwrapped_model, "_forbidden_sample_token_mask", None)
        special_token_mask_cpu: torch.Tensor | None
        if isinstance(special_token_mask, torch.Tensor):
            special_token_mask_cpu = special_token_mask.detach()
            if special_token_mask_cpu.device.type != "cpu":
                special_token_mask_cpu = special_token_mask_cpu.to(device="cpu")
        else:
            special_token_mask_cpu = None
        disc_pad_token_id = getattr(getattr(unwrapped_model, "disc_config", None), "pad_token_id", None)
        if disc_pad_token_id is not None:
            disc_pad_token_id = int(disc_pad_token_id)

        # TODO(data-resume): persist dataset iterator state (packing buffer + RNG) for O(1) resume
        # instead of replaying O(consumed_micro_batches) samples.  Requires serializable snapshot
        # support in PackedStreamingDataset / SequentialStreamingDataset.
        if consumed_micro_batches > 0:
            if int(global_step) >= int(train_cfg.max_steps):
                logger.info(
                    "Resume global_step=%d already reached max_steps=%d; skipping data replay.",
                    int(global_step),
                    int(train_cfg.max_steps),
                )
            else:
                logger.info(
                    "Replaying data iterator by %d micro-batches to align resume data position.",
                    consumed_micro_batches,
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

        local_input_tokens_seen = 0.0
        local_input_tokens_since_log = 0.0
        last_log_started_at = time.perf_counter()

        while global_step < int(train_cfg.max_steps):
            window: list[tuple[dict[str, torch.Tensor], float, float]] = []
            local_window_input_tokens = 0.0
            local_gen_tokens = 0.0
            local_disc_tokens = 0.0

            for _ in range(ga_steps):
                batch = next(train_iter)
                consumed_micro_batches += 1
                local_window_input_tokens += _count_input_tokens_for_batch(batch)
                if token_weighted_ga:
                    gen_count, disc_count = _count_rtd_tokens_for_batch(
                        batch,
                        special_token_mask=special_token_mask_cpu,
                        pad_token_id=disc_pad_token_id,
                    )
                    local_gen_tokens += gen_count
                    local_disc_tokens += disc_count
                else:
                    gen_count, disc_count = 0.0, 0.0
                window.append((batch, gen_count, disc_count))

            local_input_tokens_seen += local_window_input_tokens
            local_input_tokens_since_log += local_window_input_tokens

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
            did_optimizer_step = False
            skipped_window_due_nonfinite = False
            nonfinite_reason: str | None = None
            nonfinite_debug_path: Path | None = None
            nonfinite_sanitized_count = 0

            for step_idx, (batch, gen_count, disc_count) in enumerate(window):
                batch = _move_batch_to_device(batch, accelerator.device)
                batch = _stabilize_compile_attention_mask(
                    batch=batch,
                    compile_enabled=compile_enabled,
                    compile_scope=compile_scope,
                    backbone_type=str(model_cfg.backbone_type),
                    block_cross_document_attention=bool(data_cfg.block_cross_document_attention),
                )
                doc_ids = batch.pop("doc_ids", None)
                if doc_ids is not None:
                    batch["attention_mask"] = _build_doc_block_mask(doc_ids)
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
                        skipped_window_due_nonfinite = True
                        nonfinite_skip_total += 1
                        nonfinite_skip_streak += 1
                        nonfinite_reason = str(offending)
                        lr_now = _scheduler_current_lr(lr_scheduler)
                        nonfinite_debug_path = _write_nonfinite_debug_artifact(
                            output_dir=output_dir,
                            step=int(global_step + 1),
                            micro_step_idx=int(step_idx),
                            offending=offending,
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
                            offending,
                            int(global_step + 1),
                            int(step_idx),
                            int(nonfinite_skip_streak),
                            int(nonfinite_skip_total),
                            nonfinite_debug_path,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        break
                    accelerator.backward(backward_loss)

                if is_sync_step:
                    grad_norm_for_check = _global_grad_l2_norm(model)
                    if _has_nonfinite_grad_norm_any_rank(
                        accelerator=accelerator,
                        grad_norm=float(grad_norm_for_check),
                    ):
                        sanitized = _sanitize_nonfinite_gradients_(model)
                        nonfinite_sanitized_count += int(sanitized)
                        grad_norm_for_check = _global_grad_l2_norm(model)
                        if not _has_nonfinite_grad_norm_any_rank(
                            accelerator=accelerator,
                            grad_norm=float(grad_norm_for_check),
                        ):
                            logger.warning(
                                "Sanitized %d non-finite gradient elements before optimizer step "
                                "(step=%d, micro_step=%d); continuing.",
                                int(sanitized),
                                int(global_step + 1),
                                int(step_idx),
                            )
                        else:
                            skipped_window_due_nonfinite = True
                            nonfinite_skip_total += 1
                            nonfinite_skip_streak += 1
                            nonfinite_reason = f"grad_norm_skip_{int(nonfinite_skip_total)}"
                            lr_now = _scheduler_current_lr(lr_scheduler)
                            nonfinite_debug_path = _write_nonfinite_debug_artifact(
                                output_dir=output_dir,
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
                                "Skipping optimizer step due persistent non-finite gradient norm "
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
                            sanitized = _sanitize_nonfinite_gradients_(model)
                            nonfinite_sanitized_count += int(sanitized)
                            post_clip_grad_norm = _global_grad_l2_norm(model)
                            if not _has_nonfinite_grad_norm_any_rank(
                                accelerator=accelerator,
                                grad_norm=float(post_clip_grad_norm),
                            ):
                                logger.warning(
                                    "Sanitized %d non-finite post-clip gradient elements "
                                    "(step=%d, micro_step=%d); continuing.",
                                    int(sanitized),
                                    int(global_step + 1),
                                    int(step_idx),
                                )
                            else:
                                skipped_window_due_nonfinite = True
                                nonfinite_skip_total += 1
                                nonfinite_skip_streak += 1
                                nonfinite_reason = f"grad_norm_post_clip_skip_{int(nonfinite_skip_total)}"
                                lr_now = _scheduler_current_lr(lr_scheduler)
                                nonfinite_debug_path = _write_nonfinite_debug_artifact(
                                    output_dir=output_dir,
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
                                    "Skipping optimizer step due persistent non-finite post-clip gradient norm "
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
                    if lr_mult < 1.0:
                        _apply_lr_mult(optimizer, lr_mult)
                    optimizer.zero_grad(set_to_none=True)
                    did_optimizer_step = True
                    nonfinite_skip_streak = 0
                    # Gradually recover lr_mult toward 1.0 on successful steps.
                    if lr_mult < 1.0:
                        lr_mult = min(lr_mult * float(_NONFINITE_LR_MULT_RECOVERY), 1.0)

                    # If embedding_sharing=gdes, the discriminator embedding base weights must track the
                    # generator embedding weights. Generator embeddings update only on optimizer steps,
                    # so syncing here is sufficient (and avoids per-microbatch copies).
                    _sync_discriminator_embeddings_if_available(unwrapped_model)

            if not did_optimizer_step:
                if skipped_window_due_nonfinite:
                    # Recovery path: ratchet down persistent lr_mult and optionally reset
                    # optimizer state.  The scheduler advances normally; lr_mult is applied
                    # after every lr_scheduler.step() so it cannot be overwritten.
                    if _optimizer_has_stepped(optimizer):
                        with suppress(Exception):
                            lr_scheduler.step()
                    lr_mult, reset_state = _apply_nonfinite_recovery(
                        lr_mult=lr_mult,
                        skip_streak=int(nonfinite_skip_streak),
                    )
                    # Apply lr_mult after scheduler step so recovery is persistent.
                    _apply_lr_mult(optimizer, lr_mult)
                    if reset_state:
                        with suppress(Exception):
                            optimizer.state.clear()
                    global_step += 1
                    if train_progress is not None:
                        train_progress.update(1)
                    if train_cfg.report_to != "none":
                        _log_tracker_metrics(
                            {
                                "nonfinite_window_skipped": 1.0,
                                "nonfinite_skip_total": float(nonfinite_skip_total),
                                "nonfinite_skip_streak": float(nonfinite_skip_streak),
                                "nonfinite_sanitized_grad_elems": float(nonfinite_sanitized_count),
                                "nonfinite_recovery_lr_mult": float(lr_mult),
                                "nonfinite_recovery_optimizer_state_reset": 1.0 if reset_state else 0.0,
                            },
                            step=int(global_step),
                        )

                    if accelerator.is_main_process:
                        logger.warning(
                            "step=%d | nonfinite_window_skipped=1 | reason=%s | streak=%d | total_skips=%d | "
                            "sanitized_grad_elems=%d | lr_mult=%.4f | opt_state_reset=%s | debug=%s",
                            int(global_step),
                            str(nonfinite_reason or "unknown"),
                            int(nonfinite_skip_streak),
                            int(nonfinite_skip_total),
                            int(nonfinite_sanitized_count),
                            float(lr_mult),
                            bool(reset_state),
                            str(nonfinite_debug_path) if nonfinite_debug_path is not None else "n/a",
                        )

                    if train_cfg.save_steps and (global_step % int(train_cfg.save_steps) == 0):
                        ckpt_dir = output_dir / f"checkpoint-{global_step}"
                        _save_training_checkpoint(
                            accelerator=accelerator,
                            checkpoint_dir=ckpt_dir,
                            output_dir=output_dir,
                            consumed_micro_batches=consumed_micro_batches,
                            save_total_limit=int(train_cfg.save_total_limit),
                            log_label="periodic",
                            lr_mult=lr_mult,
                        )
                        last_saved_step = global_step
                    continue
                raise RuntimeError("Accumulation window produced no synchronized optimization step.")

            loss_for_metrics = _finalize_window_metric_loss(
                accumulated_loss=loss_for_metrics,
                ga_steps=ga_steps,
                token_weighted_ga=token_weighted_ga,
            )

            # We count accumulation windows as global steps (successful optimizer step or
            # recovered non-finite skip window).
            if did_optimizer_step:
                if out is None:
                    raise RuntimeError("Accumulation window produced no forward pass outputs.")
                global_step += 1
                if train_progress is not None:
                    train_progress.update(1)

                if train_cfg.logging_steps and (global_step % int(train_cfg.logging_steps) == 0):
                    # Reduce scalar metrics across processes.
                    def _mean(x: torch.Tensor) -> float:
                        """Compute global mean scalar across processes.

                        :param torch.Tensor x: Scalar-like tensor.
                        :return float: Process-aggregated mean value.
                        """
                        x = x.detach().float().reshape(1)
                        return accelerator.gather(x).mean().item()

                    def _sum_local_scalar(x: float) -> float:
                        """Compute global sum for a local scalar across processes.

                        :param float x: Local scalar value.
                        :return float: Summed value across all ranks.
                        """
                        local = torch.tensor([x], device=accelerator.device, dtype=torch.float64)
                        return float(accelerator.reduce(local, reduction="sum")[0].item())

                    log_now = time.perf_counter()
                    elapsed_since_log = max(log_now - last_log_started_at, 1e-9)
                    global_input_tokens_interval = _sum_local_scalar(local_input_tokens_since_log)
                    global_input_tokens_seen = _sum_local_scalar(local_input_tokens_seen)
                    input_tokens_per_sec = global_input_tokens_interval / elapsed_since_log
                    local_input_tokens_since_log = 0.0
                    last_log_started_at = log_now

                    lr = (
                        lr_scheduler.get_last_lr()[0]
                        if hasattr(lr_scheduler, "get_last_lr")
                        else float("nan")
                    )
                    metrics = {
                        "step": global_step,
                        "lr": lr,
                        "loss": _mean(loss_for_metrics),
                        "gen_loss": _mean(out.gen_loss),
                        "disc_loss": _mean(out.disc_loss),
                        "disc_acc": _mean(out.disc_accuracy),
                        "input_tokens_per_sec": float(input_tokens_per_sec),
                        "input_tokens_seen": float(global_input_tokens_seen),
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
                                    f"tok/s={metrics['input_tokens_per_sec']:.1f}",
                                    f"tok_seen={metrics['input_tokens_seen']:.0f}",
                                ]
                            )
                        )

                    if train_cfg.report_to != "none":
                        _log_tracker_metrics(
                            {k: v for k, v in metrics.items() if k != "step"}, step=global_step
                        )

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
                        lr_mult=lr_mult,
                    )
                    last_saved_step = global_step

        # Best-effort HF export
        if train_cfg.export_hf_final:
            accelerator.wait_for_everyone()
            _export_discriminator_hf(
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir / "final_hf",
                embedding_sharing=model_cfg.embedding_sharing,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
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
            with suppress(Exception):
                final_ckpt = output_dir / f"checkpoint-{final_step}"
                _save_training_checkpoint(
                    accelerator=accelerator,
                    checkpoint_dir=final_ckpt,
                    output_dir=output_dir,
                    consumed_micro_batches=consumed_micro_batches,
                    save_total_limit=int(train_cfg.save_total_limit),
                    log_label="final",
                    lr_mult=lr_mult,
                )
                last_saved_step = final_step
        elif crash_reason is not None and not should_try_crash_save and accelerator.is_main_process:
            logger.warning(
                "Skipping crash-time final checkpoint save on distributed run "
                "(num_processes=%s) to avoid potential collective deadlocks after failure.",
                getattr(accelerator, "num_processes", "unknown"),
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
