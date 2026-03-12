#!/usr/bin/env python3
"""Profile real RTD training steps for eager vs FlashDeBERTa runs.

This tool mirrors the repo's actual RTD pretraining stack closely enough to
answer "where is flash losing or winning?" without the extra noise from long
benchmark runs or checkpointing.

It is intentionally single-process and profiler-first:
- builds the real tokenizer, streaming dataset, collator, model, optimizer
- optionally enables FlashDeBERTa and backbone-only ``torch.compile``
- runs warmup steps, then captures a short profile window
- exports a Chrome trace, profiler tables, and a small phase-timing summary
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_src_on_path()

from deberta.config import load_config, resolve_effective_mixed_precision  # noqa: E402
from deberta.data.loading import load_hf_dataset  # noqa: E402
from deberta.modeling import (  # noqa: E402
    DebertaV3RTDPretrainer,
    build_backbone_configs,
    build_backbones,
)
from deberta.modeling.flashdeberta_patch import (  # noqa: E402
    disable_flashdeberta_attention,
    enable_flashdeberta_attention,
)
from deberta.training.compile import (  # noqa: E402
    _bf16_runtime_sanity_check,
    _compile_backbones_for_scope,
    _dtype_for_mixed_precision,
    _maybe_cudagraph_mark_step_begin,
    _maybe_enable_tf32,
    _prefill_rotary_caches_for_compile,
    _resolve_compile_enabled_or_raise,
    _resolve_compile_scope,
    _stabilize_compile_attention_mask,
)
from deberta.training.loop_utils import _resolve_window_token_denominators  # noqa: E402
from deberta.training.runtime import (  # noqa: E402
    _build_decoupled_optimizers,
    _build_optimizer,
    _build_scheduler,
    _build_train_dataset_and_collator,
)
from deberta.training.steps import (  # noqa: E402
    _collect_ga_window,
    _global_grad_l2_norm,
    _move_batch_to_device,
    _record_unscaled_lrs,
    _scheduler_current_lr,
    _sync_discriminator_embeddings_if_available,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/custom/pretrain_rtd_hf_deberta_v3pos_smol2stage4_1024_wp32k_v2.yaml",
    )
    parser.add_argument("--mode", choices=("eager", "flash"), required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--profile-steps", type=int, default=4)
    parser.add_argument(
        "--packing-enabled",
        choices=("true", "false"),
        default=None,
        help="Optional override for data.packing.enabled.",
    )
    parser.add_argument(
        "--ga-steps",
        type=int,
        default=None,
        help="Optional override for gradient accumulation steps used in the profiler loop.",
    )
    parser.add_argument("--dense-policy", type=int, default=None)
    parser.add_argument("--profile-dir", type=Path, required=True)
    return parser.parse_args()


def _bool_text(value: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"Expected true/false, got: {value}")


def _autocast_context(mixed_precision: str):
    normalized = str(mixed_precision).strip().lower()
    if normalized == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if normalized in {"fp16", "float16"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _write_profiler_outputs(profile_dir: Path, profiler: torch.profiler.profile) -> None:
    """Persist profiler summaries and a Chrome trace."""

    profile_dir.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(profile_dir / "trace.json"))
    cuda_table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=120)
    cpu_table = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=120)
    (profile_dir / "key_averages_cuda.txt").write_text(cuda_table + "\n", encoding="utf-8")
    (profile_dir / "key_averages_cpu.txt").write_text(cpu_table + "\n", encoding="utf-8")


class _TimedPhase:
    """Context manager that records wall time and profiler ranges for one phase."""

    def __init__(self, name: str, phase_times_ms: dict[str, list[float]]) -> None:
        self.name = str(name)
        self.phase_times_ms = phase_times_ms
        self._start = 0.0
        self._record = torch.profiler.record_function(self.name)

    def __enter__(self) -> None:
        self._record.__enter__()
        self._start = time.perf_counter()
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self.phase_times_ms[self.name].append(elapsed_ms)
        return self._record.__exit__(exc_type, exc, tb)


def _maybe_override_config(args: argparse.Namespace) -> list[str]:
    overrides: list[str] = [
        "logging.wandb.enabled=false",
        "logging.backend=none",
        "train.checkpoint.export_hf_final=false",
    ]
    if args.packing_enabled is not None:
        overrides.append(f"data.packing.enabled={str(_bool_text(args.packing_enabled)).lower()}")
    if args.ga_steps is not None:
        overrides.append(f"train.gradient_accumulation_steps={int(args.ga_steps)}")
    return overrides


def _resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for flashdeberta_rtd_profile.py")
    return torch.device("cuda")


def _compile_model_if_enabled(
    *,
    model: torch.nn.Module,
    compile_enabled: bool,
    compile_scope_requested: str,
    train_cfg: Any,
    model_cfg: Any,
    data_cfg: Any,
    mixed_precision: str,
) -> str:
    """Compile backbones when enabled and return the effective compile scope."""

    if not compile_enabled:
        return "disabled"

    compile_scope, compile_scope_reason = _resolve_compile_scope(
        requested_scope=compile_scope_requested,
        model_cfg=model_cfg,
        block_cross_document_attention=bool(data_cfg.block_cross_document_attention),
    )
    if compile_scope_reason:
        print(f"[profile] compile_scope_reason={compile_scope_reason}")

    compile_kwargs: dict[str, Any] = {
        "mode": str(train_cfg.torch_compile_mode),
        "backend": str(train_cfg.torch_compile_backend),
    }
    try:
        compile_params = torch.compile.__wrapped__.__signature__.parameters  # type: ignore[attr-defined]
    except Exception:
        try:
            import inspect

            compile_params = inspect.signature(torch.compile).parameters  # type: ignore[attr-defined]
        except Exception:
            compile_params = {}
    if "dynamic" in compile_params:
        compile_kwargs["dynamic"] = False

    compile_scope_key = str(compile_scope).strip().lower()
    if compile_scope_key in {"backbones", "encoder", "gen_encoder", "disc_encoder"}:
        prefilled_rotary = _prefill_rotary_caches_for_compile(
            model=model,
            seq_len=int(data_cfg.max_seq_length),
            device=_resolve_device(),
            dtype=_dtype_for_mixed_precision(mixed_precision),
        )
        if prefilled_rotary > 0:
            print(f"[profile] prefilled_rotary_modules={int(prefilled_rotary)}")

    compiled_targets = _compile_backbones_for_scope(
        unwrapped_model=model,
        compile_scope=compile_scope,
        compile_kwargs=compile_kwargs,
    )
    print(
        "[profile] compiled_targets="
        + ",".join(compiled_targets)
        + f" scope={compile_scope} kwargs={compile_kwargs}"
    )
    return str(compile_scope)


def _run_decoupled_window(
    *,
    model: DebertaV3RTDPretrainer,
    train_iter: Any,
    ga_steps: int,
    token_weighted_ga: bool,
    disc_pad_token_id: int | None,
    device: torch.device,
    compile_enabled: bool,
    compile_scope: str,
    backbone_type: str,
    gen_optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    gen_lr_scheduler: Any,
    disc_lr_scheduler: Any,
    phase_times_ms: dict[str, list[float]],
    mixed_precision: str,
    sampling_temperature: float,
    max_grad_norm: float,
) -> dict[str, float]:
    """Run one optimizer step for decoupled RTD training."""

    with _TimedPhase("step_total", phase_times_ms):
        with _TimedPhase("window_collect", phase_times_ms):
            window, _consumed, local_window_input_tokens, local_gen_tokens, local_disc_tokens = (
                _collect_ga_window(
                    train_iter=train_iter,
                    ga_steps=ga_steps,
                    token_weighted_ga=token_weighted_ga,
                    disc_pad_token_id=disc_pad_token_id,
                    include_has_gen_targets=True,
                    default_unweighted_token_count=1.0,
                )
            )

        gen_window_tokens_per_rank, disc_window_tokens_per_rank, _gen_zero, _disc_zero = (
            _resolve_window_token_denominators(
                gen_window_tokens_per_rank_raw=float(local_gen_tokens),
                disc_window_tokens_per_rank_raw=float(local_disc_tokens),
            )
        )

        gen_optimizer.zero_grad(set_to_none=True)
        disc_optimizer.zero_grad(set_to_none=True)

        disc_phase_inputs: list[dict[str, torch.Tensor | float | None]] = []
        gen_loss_num = 0.0
        disc_loss_num = 0.0
        disc_acc_num = 0.0
        gen_token_count_window = 0.0
        disc_token_count_window = 0.0

        for batch, gen_count, disc_count, has_gen_targets in window:
            with _TimedPhase("batch_to_device", phase_times_ms):
                batch = _move_batch_to_device(batch, device)
                doc_ids = batch.pop("doc_ids", None)
                if doc_ids is not None:
                    raise RuntimeError("Doc-block profiling is not supported in this tool.")
                batch = _stabilize_compile_attention_mask(
                    batch=batch,
                    compile_enabled=compile_enabled,
                    compile_scope=compile_scope,
                    backbone_type=backbone_type,
                )
            if compile_enabled:
                _maybe_cudagraph_mark_step_begin()

            with _TimedPhase("generator_forward", phase_times_ms):
                with _autocast_context(mixed_precision):
                    gen_phase_out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                        token_type_ids=batch.get("token_type_ids"),
                        sampling_temperature=float(sampling_temperature),
                        phase="generator",
                    )

            gen_loss = gen_phase_out.gen_loss_raw
            if token_weighted_ga:
                gen_obj = gen_loss * (float(gen_count) / max(float(gen_window_tokens_per_rank), 1.0))
                backward_loss = gen_obj
            else:
                backward_loss = gen_loss / float(max(1, ga_steps))

            with _TimedPhase("generator_backward", phase_times_ms):
                backward_loss.backward()

            micro_gen_token_count = float(gen_phase_out.gen_token_count.detach().float().item())
            gen_token_count_window += micro_gen_token_count
            gen_loss_num += float(gen_phase_out.gen_loss_raw.detach().float().item()) * micro_gen_token_count

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

        with _TimedPhase("generator_grad_norm_clip", phase_times_ms):
            _ = _global_grad_l2_norm(model)
            if float(max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        with _TimedPhase("generator_optimizer_step", phase_times_ms):
            gen_optimizer.step()
            gen_lr_scheduler.step()
            _record_unscaled_lrs(gen_optimizer, gen_lr_scheduler)
            gen_optimizer.zero_grad(set_to_none=True)
            _sync_discriminator_embeddings_if_available(model)

        for payload in disc_phase_inputs:
            if compile_enabled:
                _maybe_cudagraph_mark_step_begin()
            with _TimedPhase("discriminator_forward", phase_times_ms):
                with _autocast_context(mixed_precision):
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
                    float(payload["disc_count"]) / max(float(disc_window_tokens_per_rank), 1.0)
                )
                backward_loss = disc_obj * float(disc_objective_weight)
            else:
                backward_loss = (disc_loss * float(disc_objective_weight)) / float(max(1, ga_steps))

            with _TimedPhase("discriminator_backward", phase_times_ms):
                backward_loss.backward()

            micro_disc_token_count = float(disc_phase_out.disc_token_count.detach().float().item()) * float(
                disc_objective_weight
            )
            disc_token_count_window += micro_disc_token_count
            disc_loss_num += (
                float(disc_phase_out.disc_loss_raw.detach().float().item()) * micro_disc_token_count
            )
            disc_acc_num += (
                float(disc_phase_out.disc_accuracy.detach().float().item()) * micro_disc_token_count
            )

        with _TimedPhase("discriminator_grad_norm_clip", phase_times_ms):
            _ = _global_grad_l2_norm(model)
            if float(max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        with _TimedPhase("discriminator_optimizer_step", phase_times_ms):
            disc_optimizer.step()
            disc_lr_scheduler.step()
            _record_unscaled_lrs(disc_optimizer, disc_lr_scheduler)
            disc_optimizer.zero_grad(set_to_none=True)

    return {
        "input_tokens": float(local_window_input_tokens),
        "gen_tokens": float(gen_token_count_window),
        "disc_tokens": float(disc_token_count_window),
        "gen_loss": float(gen_loss_num / gen_token_count_window)
        if gen_token_count_window > 0
        else float("nan"),
        "disc_loss": float(disc_loss_num / disc_token_count_window)
        if disc_token_count_window > 0
        else float("nan"),
        "disc_acc": float(disc_acc_num / disc_token_count_window)
        if disc_token_count_window > 0
        else float("nan"),
        "gen_lr": float(_scheduler_current_lr(gen_lr_scheduler) or 0.0),
        "disc_lr": float(_scheduler_current_lr(disc_lr_scheduler) or 0.0),
    }


def _run_coupled_window(
    *,
    model: DebertaV3RTDPretrainer,
    train_iter: Any,
    ga_steps: int,
    token_weighted_ga: bool,
    disc_pad_token_id: int | None,
    device: torch.device,
    compile_enabled: bool,
    compile_scope: str,
    backbone_type: str,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    phase_times_ms: dict[str, list[float]],
    mixed_precision: str,
    gen_loss_weight: float,
    disc_loss_weight: float,
    sampling_temperature: float,
    max_grad_norm: float,
) -> dict[str, float]:
    """Run one optimizer step for coupled RTD training."""

    with _TimedPhase("step_total", phase_times_ms):
        with _TimedPhase("window_collect", phase_times_ms):
            window, _consumed, local_window_input_tokens, local_gen_tokens, local_disc_tokens = (
                _collect_ga_window(
                    train_iter=train_iter,
                    ga_steps=ga_steps,
                    token_weighted_ga=token_weighted_ga,
                    disc_pad_token_id=disc_pad_token_id,
                    include_has_gen_targets=False,
                    default_unweighted_token_count=0.0,
                )
            )

        gen_window_tokens_per_rank, disc_window_tokens_per_rank, _gen_zero, _disc_zero = (
            _resolve_window_token_denominators(
                gen_window_tokens_per_rank_raw=float(local_gen_tokens),
                disc_window_tokens_per_rank_raw=float(local_disc_tokens),
            )
        )

        optimizer.zero_grad(set_to_none=True)
        gen_loss_num = 0.0
        disc_loss_num = 0.0
        disc_acc_num = 0.0
        gen_token_count_window = 0.0
        disc_token_count_window = 0.0

        for batch, gen_count, disc_count in window:
            with _TimedPhase("batch_to_device", phase_times_ms):
                batch = _move_batch_to_device(batch, device)
                doc_ids = batch.pop("doc_ids", None)
                if doc_ids is not None:
                    raise RuntimeError("Doc-block profiling is not supported in this tool.")
                batch = _stabilize_compile_attention_mask(
                    batch=batch,
                    compile_enabled=compile_enabled,
                    compile_scope=compile_scope,
                    backbone_type=backbone_type,
                )
            if compile_enabled:
                _maybe_cudagraph_mark_step_begin()

            with _TimedPhase("coupled_forward", phase_times_ms):
                with _autocast_context(mixed_precision):
                    out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                        token_type_ids=batch.get("token_type_ids"),
                        sampling_temperature=float(sampling_temperature),
                        gen_loss_weight=gen_loss_weight,
                        disc_loss_weight=disc_loss_weight,
                    )

            if token_weighted_ga:
                loss = (
                    float(gen_loss_weight)
                    * (float(gen_count) / max(float(gen_window_tokens_per_rank), 1.0))
                    * out.gen_loss_raw
                    + float(disc_loss_weight)
                    * (float(disc_count) / max(float(disc_window_tokens_per_rank), 1.0))
                    * out.disc_loss_raw
                )
            else:
                loss = out.loss / float(max(1, ga_steps))

            with _TimedPhase("coupled_backward", phase_times_ms):
                loss.backward()

            micro_gen_tokens = float(out.gen_token_count.detach().float().item())
            micro_disc_tokens = float(out.disc_token_count.detach().float().item())
            gen_token_count_window += micro_gen_tokens
            disc_token_count_window += micro_disc_tokens
            gen_loss_num += float(out.gen_loss.detach().float().item()) * micro_gen_tokens
            disc_loss_num += float(out.disc_loss.detach().float().item()) * micro_disc_tokens
            disc_acc_num += float(out.disc_accuracy.detach().float().item()) * micro_disc_tokens

        with _TimedPhase("coupled_grad_norm_clip", phase_times_ms):
            _ = _global_grad_l2_norm(model)
            if float(max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        with _TimedPhase("coupled_optimizer_step", phase_times_ms):
            optimizer.step()
            lr_scheduler.step()
            _record_unscaled_lrs(optimizer, lr_scheduler)
            optimizer.zero_grad(set_to_none=True)
            _sync_discriminator_embeddings_if_available(model)

    return {
        "input_tokens": float(local_window_input_tokens),
        "gen_tokens": float(gen_token_count_window),
        "disc_tokens": float(disc_token_count_window),
        "gen_loss": float(gen_loss_num / gen_token_count_window)
        if gen_token_count_window > 0
        else float("nan"),
        "disc_loss": float(disc_loss_num / disc_token_count_window)
        if disc_token_count_window > 0
        else float("nan"),
        "disc_acc": float(disc_acc_num / disc_token_count_window)
        if disc_token_count_window > 0
        else float("nan"),
        "lr": float(_scheduler_current_lr(lr_scheduler) or 0.0),
    }


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.profile_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats(device)

    overrides = _maybe_override_config(args)
    cfg = load_config(args.config, overrides=overrides)
    model_cfg, data_cfg, train_cfg = cfg.model, cfg.data, cfg.train

    if str(args.mode) == "flash":
        if args.dense_policy is not None:
            os.environ["FLASHDEBERTA_EAGER_DENSE_MAX_SEQ_LEN"] = str(int(args.dense_policy))
        enable_flashdeberta_attention(strict=True)
    else:
        disable_flashdeberta_attention()

    mixed_precision = resolve_effective_mixed_precision(
        train_cfg.mixed_precision,
        bf16_sanity_check=_bf16_runtime_sanity_check,
    )
    _maybe_enable_tf32(bool(train_cfg.tf32))

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name_or_path, use_fast=True)
    raw_train = load_hf_dataset(cfg=data_cfg, split=data_cfg.train_split, streaming=data_cfg.streaming)
    train_dataset, collator = _build_train_dataset_and_collator(
        raw_train=raw_train,
        tokenizer=tokenizer,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        process_index=0,
        num_processes=1,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.per_device_train_batch_size),
        collate_fn=collator,
        num_workers=int(train_cfg.dataloader_num_workers),
        pin_memory=bool(train_cfg.dataloader_pin_memory),
        drop_last=True,
        persistent_workers=int(train_cfg.dataloader_num_workers) > 0,
    )
    train_iter = iter(loader)

    disc_config, gen_config = build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=int(data_cfg.max_seq_length),
    )
    disc_backbone, gen_backbone = build_backbones(
        model_cfg=model_cfg,
        disc_config=disc_config,
        gen_config=gen_config,
        load_pretrained_weights=True,
    )
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=disc_backbone,
        generator_backbone=gen_backbone,
        disc_config=disc_config,
        gen_config=gen_config,
        embedding_sharing=model_cfg.embedding_sharing,
        tie_generator_word_embeddings=True,
        additional_forbidden_token_ids=getattr(tokenizer, "all_special_ids", []),
    ).to(device=device)

    effective_decoupled_training = bool(train_cfg.decoupled_training)
    if effective_decoupled_training:
        gen_optimizer, disc_optimizer = _build_decoupled_optimizers(
            model,
            train_cfg,
            mixed_precision=mixed_precision,
        )
        gen_lr_scheduler = _build_scheduler(gen_optimizer, train_cfg)
        disc_lr_scheduler = _build_scheduler(disc_optimizer, train_cfg)
    else:
        optimizer = _build_optimizer(model, train_cfg, mixed_precision=mixed_precision)
        lr_scheduler = _build_scheduler(optimizer, train_cfg)

    _sync_discriminator_embeddings_if_available(model)

    compile_enabled = _resolve_compile_enabled_or_raise(train_cfg.torch_compile)
    compile_scope = _compile_model_if_enabled(
        model=model,
        compile_enabled=compile_enabled,
        compile_scope_requested=str(train_cfg.torch_compile_scope).strip().lower(),
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        mixed_precision=mixed_precision,
    )
    model.train()

    ga_steps = int(args.ga_steps) if args.ga_steps is not None else int(train_cfg.gradient_accumulation_steps)
    token_weighted_ga = bool(train_cfg.token_weighted_gradient_accumulation)
    disc_pad_token_id = getattr(tokenizer, "pad_token_id", None)
    phase_times_ms: dict[str, list[float]] = defaultdict(list)

    for _ in range(int(args.warmup_steps)):
        if effective_decoupled_training:
            _ = _run_decoupled_window(
                model=model,
                train_iter=train_iter,
                ga_steps=ga_steps,
                token_weighted_ga=token_weighted_ga,
                disc_pad_token_id=disc_pad_token_id,
                device=device,
                compile_enabled=compile_enabled,
                compile_scope=compile_scope,
                backbone_type=str(model_cfg.backbone_type),
                gen_optimizer=gen_optimizer,
                disc_optimizer=disc_optimizer,
                gen_lr_scheduler=gen_lr_scheduler,
                disc_lr_scheduler=disc_lr_scheduler,
                phase_times_ms=defaultdict(list),
                mixed_precision=mixed_precision,
                sampling_temperature=float(train_cfg.sampling_temperature),
                max_grad_norm=float(train_cfg.max_grad_norm),
            )
        else:
            _ = _run_coupled_window(
                model=model,
                train_iter=train_iter,
                ga_steps=ga_steps,
                token_weighted_ga=token_weighted_ga,
                disc_pad_token_id=disc_pad_token_id,
                device=device,
                compile_enabled=compile_enabled,
                compile_scope=compile_scope,
                backbone_type=str(model_cfg.backbone_type),
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                phase_times_ms=defaultdict(list),
                mixed_precision=mixed_precision,
                gen_loss_weight=float(train_cfg.gen_loss_weight),
                disc_loss_weight=float(train_cfg.disc_loss_weight),
                sampling_temperature=float(train_cfg.sampling_temperature),
                max_grad_norm=float(train_cfg.max_grad_norm),
            )

    metrics: list[dict[str, float]] = []
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        for _ in range(int(args.profile_steps)):
            if effective_decoupled_training:
                step_metrics = _run_decoupled_window(
                    model=model,
                    train_iter=train_iter,
                    ga_steps=ga_steps,
                    token_weighted_ga=token_weighted_ga,
                    disc_pad_token_id=disc_pad_token_id,
                    device=device,
                    compile_enabled=compile_enabled,
                    compile_scope=compile_scope,
                    backbone_type=str(model_cfg.backbone_type),
                    gen_optimizer=gen_optimizer,
                    disc_optimizer=disc_optimizer,
                    gen_lr_scheduler=gen_lr_scheduler,
                    disc_lr_scheduler=disc_lr_scheduler,
                    phase_times_ms=phase_times_ms,
                    mixed_precision=mixed_precision,
                    sampling_temperature=float(train_cfg.sampling_temperature),
                    max_grad_norm=float(train_cfg.max_grad_norm),
                )
            else:
                step_metrics = _run_coupled_window(
                    model=model,
                    train_iter=train_iter,
                    ga_steps=ga_steps,
                    token_weighted_ga=token_weighted_ga,
                    disc_pad_token_id=disc_pad_token_id,
                    device=device,
                    compile_enabled=compile_enabled,
                    compile_scope=compile_scope,
                    backbone_type=str(model_cfg.backbone_type),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    phase_times_ms=phase_times_ms,
                    mixed_precision=mixed_precision,
                    gen_loss_weight=float(train_cfg.gen_loss_weight),
                    disc_loss_weight=float(train_cfg.disc_loss_weight),
                    sampling_temperature=float(train_cfg.sampling_temperature),
                    max_grad_norm=float(train_cfg.max_grad_norm),
                )
            metrics.append(step_metrics)
            profiler.step()

    _write_profiler_outputs(args.profile_dir, profiler)

    phase_summary: dict[str, dict[str, float]] = {}
    for name, values in sorted(phase_times_ms.items()):
        if not values:
            continue
        phase_summary[name] = {
            "mean_ms": float(statistics.mean(values)),
            "median_ms": float(statistics.median(values)),
            "count": float(len(values)),
        }

    summary = {
        "mode": str(args.mode),
        "config": str(args.config),
        "packing_enabled": bool(data_cfg.pack_sequences),
        "compile_enabled": bool(compile_enabled),
        "compile_scope": str(compile_scope),
        "gradient_accumulation_steps": int(ga_steps),
        "token_weighted_ga": bool(token_weighted_ga),
        "warmup_steps": int(args.warmup_steps),
        "profile_steps": int(args.profile_steps),
        "max_memory_gib": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "step_metrics_mean": {
            key: float(statistics.mean([float(m[key]) for m in metrics])) for key in sorted(metrics[0].keys())
        }
        if metrics
        else {},
        "phase_summary": phase_summary,
    }
    (args.profile_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
