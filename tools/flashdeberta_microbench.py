#!/usr/bin/env python3
"""Synthetic forward+backward microbenchmark for eager vs FlashDeBERTa attention.

This script is intentionally small and self-contained. It isolates the backbone
attention regime without involving the full dataset / RTD loop.

Examples
--------
Dense packed-like 1024 (mask dropped):
    python tools/flashdeberta_microbench.py --mode eager --seq-len 1024 --batch-size 8 --pad-ratio 0.0
    python tools/flashdeberta_microbench.py --mode flash --seq-len 1024 --batch-size 8 --pad-ratio 0.0

Padding-heavy 1024:
    python tools/flashdeberta_microbench.py --mode flash --seq-len 1024 --batch-size 8 --pad-ratio 0.35

Longer padded regime with varlen enabled:
    FLASHDEBERTA_VARLEN_MIN_SEQ_LEN=1024 python tools/flashdeberta_microbench.py \
        --mode flash --seq-len 2048 --batch-size 4 --pad-ratio 0.35
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_src_on_path()

from deberta.modeling.deberta_v2_native import DebertaV2Config, DebertaV2Model  # noqa: E402
from deberta.modeling.flashdeberta_patch import (  # noqa: E402
    disable_flashdeberta_attention,
    enable_flashdeberta_attention,
)

try:  # noqa: E402
    from deberta.modeling.flashdeberta_attention import (  # type: ignore
        flashdeberta_stats_snapshot,
        reset_flashdeberta_stats,
    )
except Exception:  # pragma: no cover
    flashdeberta_stats_snapshot = None
    reset_flashdeberta_stats = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "flash"), required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--pad-ratio", type=float, default=0.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--vocab-size", type=int, default=128_100)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--drop-all-ones-mask", action="store_true", default=True)
    parser.add_argument("--no-drop-all-ones-mask", dest="drop_all_ones_mask", action="store_false")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> DebertaV2Config:
    return DebertaV2Config(
        vocab_size=int(args.vocab_size),
        hidden_size=int(args.hidden_size),
        num_hidden_layers=int(args.num_layers),
        num_attention_heads=int(args.num_heads),
        intermediate_size=int(args.intermediate_size),
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=int(args.seq_len),
        type_vocab_size=0,
        layer_norm_eps=1e-7,
        relative_attention=True,
        position_buckets=256,
        max_relative_positions=int(args.seq_len),
        pos_att_type=["c2p", "p2c"],
        pad_token_id=0,
        position_biased_input=False,
    )


def _dtype_from_arg(name: str) -> torch.dtype:
    return torch.bfloat16 if str(name) == "bf16" else torch.float16


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


def _format_stats(stats: dict[str, int] | None) -> str:
    if not stats:
        return "{}"
    items = ", ".join(f"{k}={v}" for k, v in sorted(stats.items()))
    return "{" + items + "}"


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for flashdeberta_microbench.py")

    device = torch.device("cuda")
    dtype = _dtype_from_arg(str(args.dtype))
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats(device)

    if str(args.mode) == "flash":
        enable_flashdeberta_attention(strict=True)
    else:
        disable_flashdeberta_attention()

    cfg = _build_config(args)
    model = DebertaV2Model(cfg).to(device=device, dtype=dtype)
    model.train()

    batch = _build_batch(args, device=device, pad_token_id=int(cfg.pad_token_id))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    active_tokens_per_batch = int(batch["active_tokens_per_batch"])
    slot_tokens_per_batch = int(batch["slot_tokens_per_batch"])

    if callable(reset_flashdeberta_stats):
        reset_flashdeberta_stats()

    times_ms: list[float] = []
    total_steps = int(args.warmup) + int(args.steps)

    for step in range(total_steps):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        start = time.perf_counter()

        out = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        loss = out.float().pow(2).mean()
        loss.backward()

        torch.cuda.synchronize(device)
        end = time.perf_counter()

        if step >= int(args.warmup):
            times_ms.append((end - start) * 1000.0)

    elapsed_s = sum(times_ms) / 1000.0
    active_tok_s = (active_tokens_per_batch * int(args.steps)) / elapsed_s
    slot_tok_s = (slot_tokens_per_batch * int(args.steps)) / elapsed_s
    mean_ms = statistics.mean(times_ms)
    p50_ms = statistics.median(times_ms)
    p90_ms = statistics.quantiles(times_ms, n=10)[8] if len(times_ms) >= 10 else max(times_ms)
    max_mem_gib = torch.cuda.max_memory_allocated(device) / (1024**3)

    stats = flashdeberta_stats_snapshot() if callable(flashdeberta_stats_snapshot) else None

    print(f"mode={args.mode}")
    print(
        f"seq_len={args.seq_len} batch_size={args.batch_size} pad_ratio={args.pad_ratio:.3f} dtype={args.dtype}"
    )
    print(f"mean_ms={mean_ms:.2f} p50_ms={p50_ms:.2f} p90_ms={p90_ms:.2f}")
    print(f"active_tok_per_s={active_tok_s:.2f}")
    print(f"slot_tok_per_s={slot_tok_s:.2f}")
    print(f"max_memory_gib={max_mem_gib:.3f}")
    print(f"flash_stats={_format_stats(stats)}")


if __name__ == "__main__":
    main()
