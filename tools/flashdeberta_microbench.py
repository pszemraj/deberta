#!/usr/bin/env python3
"""Synthetic forward+backward microbenchmark for eager vs FlashDeBERTa attention.

This script is intentionally small and self-contained. It isolates the backbone
attention regime without involving the full dataset / RTD loop.

Examples
--------
Dense packed-like 1024 (mask dropped):
    python tools/flashdeberta_microbench.py --mode eager --seq-len 1024 --batch-size 8 --pad-ratio 0.0
    python tools/flashdeberta_microbench.py --mode flash --seq-len 1024 --batch-size 8 --pad-ratio 0.0

Padding-heavy 1024 (defaults to the fixed flash path with per-example seq_lengths):
    python tools/flashdeberta_microbench.py --mode flash --seq-len 1024 --batch-size 8 --pad-ratio 0.35

Longer padded regime with varlen enabled:
    python tools/flashdeberta_microbench.py \
        --mode flash --seq-len 2048 --batch-size 4 --pad-ratio 0.35
"""

from __future__ import annotations

import argparse
import statistics
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import _bench_common  # noqa: E402,F401  (inserts src/ on sys.path at import)
import torch

from deberta.config import ModelHFFlashConfig  # noqa: E402
from deberta.modeling.deberta_v2_native import DebertaV2Config, DebertaV2Model  # noqa: E402
from deberta.modeling.mask_utils import FlashBatchMeta  # noqa: E402
from deberta.training.compile import prepare_flash_attention_batch_metadata  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "flash"), required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--pad-ratio", type=float, default=0.0)
    parser.add_argument("--vocab-size", type=int, default=128_100)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--drop-all-ones-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-dir", type=Path, default=None)
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> DebertaV2Config:
    return _bench_common.build_synthetic_backbone_config(
        mode=str(args.mode),
        seq_len=int(args.seq_len),
        vocab_size=int(args.vocab_size),
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        intermediate_size=int(args.intermediate_size),
    )


def _build_batch(args: argparse.Namespace, *, device: torch.device, pad_token_id: int) -> dict[str, Any]:
    batch = int(args.batch_size)
    seq_len = int(args.seq_len)
    pad_ratio = max(0.0, min(0.95, float(args.pad_ratio)))
    active_len = max(1, int(round(seq_len * (1.0 - pad_ratio))))

    input_ids = torch.randint(5, int(args.vocab_size), (batch, seq_len), device=device, dtype=torch.long)

    attention_mask: torch.Tensor | None
    if active_len >= seq_len:
        attention_mask = (
            None
            if bool(args.drop_all_ones_mask)
            else torch.ones((batch, seq_len), device=device, dtype=torch.bool)
        )
    else:
        attention_mask = torch.zeros((batch, seq_len), device=device, dtype=torch.bool)
        attention_mask[:, :active_len] = True
        input_ids[:, active_len:] = int(pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "active_tokens_per_batch": int(batch * active_len),
        "slot_tokens_per_batch": int(batch * seq_len),
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for flashdeberta_microbench.py")

    device = torch.device("cuda")
    # bf16 only: the repo's precision contract excludes fp16.
    dtype = torch.bfloat16
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats(device)

    cfg = _build_config(args)
    model = DebertaV2Model(cfg).to(device=device, dtype=dtype)
    model.train()

    batch = _build_batch(args, device=torch.device("cpu"), pad_token_id=int(cfg.pad_token_id))
    active_tokens_per_batch = int(batch["active_tokens_per_batch"])
    slot_tokens_per_batch = int(batch["slot_tokens_per_batch"])
    attention_mask = batch.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        batch["_flash_meta"] = FlashBatchMeta(
            seq_lengths=attention_mask.sum(dim=-1, dtype=torch.int32),
            active_tokens_scalar=torch.tensor(active_tokens_per_batch, dtype=torch.int32),
        )
    batch, flash_meta = prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type="hf_deberta_v2",
        flash_enabled=str(args.mode) == "flash",
        flash_cfg=ModelHFFlashConfig(**cfg.hf_flash),
        route_device=device,
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = (
        batch["attention_mask"].to(device) if isinstance(batch.get("attention_mask"), torch.Tensor) else None
    )
    if flash_meta is not None:
        flash_meta = flash_meta.to(device)

    times_ms: list[float] = []
    total_warmup = int(args.warmup)
    measured_steps = int(args.steps)

    for _ in range(total_warmup):
        _bench_common.run_timed_backbone_step(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            flash_meta=flash_meta,
        )

    profiler_ctx: Any
    if args.profile_dir is not None:
        profiler_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
    else:
        profiler_ctx = nullcontext()

    with profiler_ctx as profiler:
        for _ in range(measured_steps):
            times_ms.append(
                _bench_common.run_timed_backbone_step(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    flash_meta=flash_meta,
                )
            )

            if profiler is not None:
                profiler.step()

    if args.profile_dir is not None and profiler is not None:
        _bench_common.write_profiler_outputs(Path(args.profile_dir), profiler, row_limit=80)

    elapsed_s = sum(times_ms) / 1000.0
    active_tok_s = (active_tokens_per_batch * measured_steps) / elapsed_s
    slot_tok_s = (slot_tokens_per_batch * measured_steps) / elapsed_s
    mean_ms = statistics.mean(times_ms)
    p50_ms = statistics.median(times_ms)
    p90_ms = statistics.quantiles(times_ms, n=10)[8] if len(times_ms) >= 10 else max(times_ms)
    max_mem_gib = torch.cuda.max_memory_allocated(device) / (1024**3)

    print(f"mode={args.mode}")
    print(f"seq_len={args.seq_len} batch_size={args.batch_size} pad_ratio={args.pad_ratio:.3f} dtype=bf16")
    print(f"mean_ms={mean_ms:.2f} p50_ms={p50_ms:.2f} p90_ms={p90_ms:.2f}")
    print(f"active_tok_per_s={active_tok_s:.2f}")
    print(f"slot_tok_per_s={slot_tok_s:.2f}")
    print(f"max_memory_gib={max_mem_gib:.3f}")


if __name__ == "__main__":
    main()
