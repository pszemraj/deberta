# ruff: noqa: F401
import argparse
import dataclasses
import gzip
import hashlib
import json
import logging
import re
import shlex
import sys
import types
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from _fakes import (
    DummyTokenizer,
    FakeAccelerator,
    FakeWandbRun,
    SimpleRTD,
    TinyRTDLikeModel,
    setup_pretraining_mocks,
)

import deberta.cli as cli_mod
from deberta.config import (
    RUN_CONFIG_SCHEMA_VERSION,
    Config,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    _looks_like_hf_deberta_checkpoint,
    _normalize_hf_attention_kernel,
    _normalize_sdpa_kernel,
    _normalize_torch_compile_backend,
    _normalize_torch_compile_mode,
    _normalize_torch_compile_scope,
    _normalize_wandb_watch,
    apply_dotted_override,
    apply_profile_defaults,
    load_config,
    load_data_config_snapshot,
    load_model_config_snapshot,
    normalize_mixed_precision,
    resolve_effective_mixed_precision,
    validate_data_config,
    validate_logging_config,
    validate_model_config,
    validate_optim_config,
    validate_train_config,
    validate_training_workflow_options,
)
from deberta.data.loading import load_hf_dataset
from deberta.export_cli import ExportArgumentDefaultsHelpFormatter, add_export_arguments
from deberta.modeling.builder import build_backbone_configs
from deberta.modeling.mask_utils import normalize_keep_mask
from deberta.modeling.rtd import attention_mask_to_active_tokens
from deberta.training.checkpointing import (
    _normalize_resume_consumed_micro_batches,
    _resolve_data_resume_policy,
)
from deberta.training.compile import (
    _compile_backbones_for_scope,
    _resolve_compile_enabled_or_raise,
    _resolve_compile_scope,
    _stabilize_compile_attention_mask,
)
from deberta.training.entrypoint import run_pretraining_dry_run
from deberta.training.export_helpers import _export_discriminator_hf_subprocess
from deberta.training.loop_utils import (
    _count_input_tokens_for_batch,
    _count_rtd_tokens_for_batch,
    _finalize_window_metric_loss,
    _resolve_window_token_denominators,
    _scale_loss_for_backward,
    _should_clip_gradients,
    _token_weighted_micro_objective,
)
from deberta.training.metrics import (
    _append_metrics_jsonl_row,
    _build_runtime_resolved_tracker_config,
    _coerce_dataclass_payload_types,
    _flush_loggers,
)
from deberta.training.run_config import (
    _build_run_metadata,
    _persist_or_validate_run_configs,
)
from deberta.training.run_management import (
    _find_latest_checkpoint,
    _load_checkpoint_data_progress,
    _load_checkpoint_progress_metadata,
    _prepare_output_dir,
    _resolve_output_dir,
    _resolve_output_dir_for_accelerator,
    _resolve_resume_checkpoint,
    _resolve_resume_checkpoint_for_accelerator,
    _save_checkpoint_data_progress,
    _save_training_checkpoint,
)
from deberta.training.runtime import (
    _build_decoupled_optimizers,
    _build_optimizer,
    _build_training_collator,
    _cycle_dataloader,
    _optimizer_param_order_digest,
    _partition_optimizer_params,
)
from deberta.training.steps import (
    _any_rank_flag_true,
    _apply_lr_mult,
    _apply_nonfinite_recovery,
    _global_grad_l2_norm,
    _has_nonfinite_grad_norm_any_rank,
    _record_unscaled_lrs,
    _sync_discriminator_embeddings_if_available,
)
from deberta.training.tracker_utils import (
    _init_trackers,
    _setup_wandb_watch,
    _upload_wandb_original_config,
)
from deberta.utils.checkpoint import (
    canonical_compile_state_key,
    load_checkpoint_model_state_dict,
    load_model_state_with_compile_key_remap,
    load_state_with_compile_fallback,
)

__all__ = [name for name in globals() if (not name.startswith("__")) and name != "__all__"]
