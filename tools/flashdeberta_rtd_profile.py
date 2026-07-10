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
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import _bench_common as bench  # noqa: E402  (inserts src/ on sys.path at import)
import torch
from torch.profiler import ProfilerActivity

from deberta.modeling import DebertaV3RTDPretrainer  # noqa: E402
from deberta.training.compile import (  # noqa: E402
    _compile_backbones_for_scope,
    _dtype_for_mixed_precision,
    _maybe_cudagraph_mark_step_begin,
    _prefill_rotary_caches_for_compile,
    _resolve_compile_scope,
    _stabilize_compile_attention_mask,
    prepare_flash_attention_batch_metadata,
)
from deberta.training.loop_utils import _resolve_window_token_denominators  # noqa: E402
from deberta.training.runtime import (  # noqa: E402
    _build_decoupled_optimizers,
    _build_optimizer,
    _build_scheduler,
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
        default="configs/flashdeberta/pretrain_rtd_hf_deberta_v3pos_smol2stage4_1024_wp32k_v2.yaml",
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
        "--block-cross-document-attention",
        choices=("true", "false"),
        default=None,
        help="Optional override for data.packing.block_cross_document_attention.",
    )
    parser.add_argument(
        "--ga-steps",
        type=int,
        default=None,
        help="Optional override for gradient accumulation steps used in the profiler loop.",
    )
    parser.add_argument("--dense-policy", type=int, default=None)
    parser.add_argument(
        "--docblock-bias-seq-len",
        type=int,
        default=None,
        help="Optional override for model.hf.flash.docblock_bias_seq_len.",
    )
    parser.add_argument(
        "--kernel-overrides-path",
        type=Path,
        default=None,
        help="Optional override for model.hf.flash.kernel_overrides_path.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Measure wall-clock phases without torch.profiler traces or profiler overhead.",
    )
    parser.add_argument("--profile-dir", type=Path, required=True)
    return parser.parse_args()


def _bool_text(value: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"Expected true/false, got: {value}")


_autocast_context = bench.autocast_context


def _write_profiler_outputs(profile_dir: Path, profiler: torch.profiler.profile) -> None:
    """Persist profiler summaries and a Chrome trace."""

    profile_dir.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(profile_dir / "trace.json"))
    cuda_table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=120)
    cpu_table = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=120)
    (profile_dir / "key_averages_cuda.txt").write_text(cuda_table + "\n", encoding="utf-8")
    (profile_dir / "key_averages_cpu.txt").write_text(cpu_table + "\n", encoding="utf-8")


def _package_version(name: str) -> str | None:
    """Return an installed package version when available."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_driver_version() -> str | None:
    """Return the NVIDIA driver version from ``nvidia-smi`` when available."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    line = result.stdout.strip().splitlines()
    return line[0].strip() if line else None


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest for a local file."""

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_config_hash(*, config: str, config_sha256: str, overrides: list[str]) -> str:
    """Return a stable hash for the config file plus runtime overrides."""

    payload = json.dumps(
        {
            "config": str(config),
            "config_sha256": str(config_sha256),
            "overrides": list(overrides),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        f"model.hf.attention_impl={str(args.mode)}",
    ]
    if args.dense_policy is not None:
        overrides.append(f"model.hf.flash.eager_dense_max_seq_len={int(args.dense_policy)}")
    if args.docblock_bias_seq_len is not None:
        overrides.append(f"model.hf.flash.docblock_bias_seq_len={int(args.docblock_bias_seq_len)}")
    if args.kernel_overrides_path is not None:
        overrides.append(f"model.hf.flash.kernel_overrides_path={args.kernel_overrides_path}")
    if args.packing_enabled is not None:
        overrides.append(f"data.packing.enabled={str(_bool_text(args.packing_enabled)).lower()}")
    if args.block_cross_document_attention is not None:
        overrides.append(
            "data.packing.block_cross_document_attention="
            f"{str(_bool_text(args.block_cross_document_attention)).lower()}"
        )
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
    device: torch.device,
    compile_enabled: bool,
    compile_scope: str,
    backbone_type: str,
    flash_enabled: bool,
    flash_cfg: Any | None,
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

        disc_phase_inputs: list[dict[str, Any]] = []
        gen_loss_num = 0.0
        disc_loss_num = 0.0
        disc_acc_num = 0.0
        gen_token_count_window = 0.0
        disc_token_count_window = 0.0

        for batch, gen_count, disc_count in window:
            with _TimedPhase("batch_to_device", phase_times_ms):
                batch = _move_batch_to_device(batch, device)
                batch = _stabilize_compile_attention_mask(
                    batch=batch,
                    compile_enabled=compile_enabled,
                    compile_scope=compile_scope,
                    backbone_type=backbone_type,
                )
                batch, flash_meta = prepare_flash_attention_batch_metadata(
                    batch=batch,
                    backbone_type=backbone_type,
                    flash_enabled=flash_enabled,
                    flash_cfg=flash_cfg,
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
                        flash_meta=flash_meta,
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

            disc_objective_weight = 1.0 if bool(gen_phase_out.has_masked_targets) else 0.0
            disc_phase_inputs.append(
                {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch.get("attention_mask"),
                    "token_type_ids": batch.get("token_type_ids"),
                    "flash_meta": flash_meta,
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
                        flash_meta=payload["flash_meta"],  # type: ignore[arg-type]
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
    device: torch.device,
    compile_enabled: bool,
    compile_scope: str,
    backbone_type: str,
    flash_enabled: bool,
    flash_cfg: Any | None,
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
                batch = _stabilize_compile_attention_mask(
                    batch=batch,
                    compile_enabled=compile_enabled,
                    compile_scope=compile_scope,
                    backbone_type=backbone_type,
                )
                batch, flash_meta = prepare_flash_attention_batch_metadata(
                    batch=batch,
                    backbone_type=backbone_type,
                    flash_enabled=flash_enabled,
                    flash_cfg=flash_cfg,
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
                        flash_meta=flash_meta,
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
    cfg, mixed_precision, tokenizer, loader = bench.load_tool_config_and_loader(
        str(args.config),
        overrides,
        tool_name="flashdeberta_rtd_profile.py",
    )
    model_cfg, data_cfg, train_cfg = cfg.model, cfg.data, cfg.train
    train_iter = iter(loader)

    model = bench.build_rtd_pretrainer(cfg=cfg, tokenizer=tokenizer, device=device, train_mode=False)

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

    compile_enabled = bool(train_cfg.torch_compile)
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
    flash_enabled = str(model_cfg.hf.attention_impl).strip().lower() == "flash"
    phase_times_ms: dict[str, list[float]] = defaultdict(list)

    for _ in range(int(args.warmup_steps)):
        if effective_decoupled_training:
            _ = _run_decoupled_window(
                model=model,
                train_iter=train_iter,
                ga_steps=ga_steps,
                token_weighted_ga=token_weighted_ga,
                device=device,
                compile_enabled=compile_enabled,
                compile_scope=compile_scope,
                backbone_type=str(model_cfg.backbone_type),
                flash_enabled=flash_enabled,
                flash_cfg=getattr(model_cfg.hf, "flash", None),
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
                device=device,
                compile_enabled=compile_enabled,
                compile_scope=compile_scope,
                backbone_type=str(model_cfg.backbone_type),
                flash_enabled=flash_enabled,
                flash_cfg=getattr(model_cfg.hf, "flash", None),
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                phase_times_ms=defaultdict(list),
                mixed_precision=mixed_precision,
                gen_loss_weight=float(train_cfg.gen_loss_weight),
                disc_loss_weight=float(train_cfg.disc_loss_weight),
                sampling_temperature=float(train_cfg.sampling_temperature),
                max_grad_norm=float(train_cfg.max_grad_norm),
            )

    def _run_measured_window() -> dict[str, float]:
        """Run one measured optimizer window."""

        if effective_decoupled_training:
            return _run_decoupled_window(
                model=model,
                train_iter=train_iter,
                ga_steps=ga_steps,
                token_weighted_ga=token_weighted_ga,
                device=device,
                compile_enabled=compile_enabled,
                compile_scope=compile_scope,
                backbone_type=str(model_cfg.backbone_type),
                flash_enabled=flash_enabled,
                flash_cfg=getattr(model_cfg.hf, "flash", None),
                gen_optimizer=gen_optimizer,
                disc_optimizer=disc_optimizer,
                gen_lr_scheduler=gen_lr_scheduler,
                disc_lr_scheduler=disc_lr_scheduler,
                phase_times_ms=phase_times_ms,
                mixed_precision=mixed_precision,
                sampling_temperature=float(train_cfg.sampling_temperature),
                max_grad_norm=float(train_cfg.max_grad_norm),
            )
        return _run_coupled_window(
            model=model,
            train_iter=train_iter,
            ga_steps=ga_steps,
            token_weighted_ga=token_weighted_ga,
            device=device,
            compile_enabled=compile_enabled,
            compile_scope=compile_scope,
            backbone_type=str(model_cfg.backbone_type),
            flash_enabled=flash_enabled,
            flash_cfg=getattr(model_cfg.hf, "flash", None),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            phase_times_ms=phase_times_ms,
            mixed_precision=mixed_precision,
            gen_loss_weight=float(train_cfg.gen_loss_weight),
            disc_loss_weight=float(train_cfg.disc_loss_weight),
            sampling_temperature=float(train_cfg.sampling_temperature),
            max_grad_norm=float(train_cfg.max_grad_norm),
        )

    metrics: list[dict[str, float]] = []
    if bool(args.summary_only):
        for _ in range(int(args.profile_steps)):
            metrics.append(_run_measured_window())
    else:
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            for _ in range(int(args.profile_steps)):
                metrics.append(_run_measured_window())
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

    config_path = Path(args.config)
    config_sha256 = _sha256_file(config_path)
    step_total_ms = [float(value) for value in phase_times_ms.get("step_total", [])]
    input_tokens_per_sec = [
        float(metric["input_tokens"]) / (float(step_ms) / 1000.0)
        for metric, step_ms in zip(metrics, step_total_ms, strict=False)
        if float(step_ms) > 0.0
    ]
    summary = {
        "mode": str(args.mode),
        "config": str(args.config),
        "config_file_sha256": config_sha256,
        "run_config_hash": _run_config_hash(
            config=str(args.config),
            config_sha256=config_sha256,
            overrides=overrides,
        ),
        "overrides": list(overrides),
        "environment": {
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
            "nvidia_driver_version": _nvidia_driver_version(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "triton_version": _package_version("triton"),
            "flashdeberta_version": _package_version("flashdeberta"),
        },
        "packing_enabled": bool(data_cfg.pack_sequences),
        "compile_enabled": bool(compile_enabled),
        "compile_scope": str(compile_scope),
        "gradient_accumulation_steps": int(ga_steps),
        "token_weighted_ga": bool(token_weighted_ga),
        "warmup_steps": int(args.warmup_steps),
        "profile_steps": int(args.profile_steps),
        "summary_only": bool(args.summary_only),
        "max_memory_gib": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "step_metrics_mean": {
            key: float(statistics.mean([float(m[key]) for m in metrics])) for key in sorted(metrics[0].keys())
        }
        if metrics
        else {},
        "throughput": {
            "input_tokens_per_sec_mean": float(statistics.mean(input_tokens_per_sec))
            if input_tokens_per_sec
            else float("nan"),
            "input_tokens_per_sec_median": float(statistics.median(input_tokens_per_sec))
            if input_tokens_per_sec
            else float("nan"),
        },
        "phase_summary": phase_summary,
    }
    (args.profile_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
