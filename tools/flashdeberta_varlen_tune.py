#!/usr/bin/env python3
"""Tune FlashDeBERTa fixed/varlen routes against sampled real loader batches.

This tool bridges the gap between the synthetic microbench and full RTD
training. It samples real unpacked batches from the configured dataloader,
records their padding structure, then replays those batches through a native
HF DeBERTa backbone under explicit fixed or varlen flash routing.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
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
from deberta.modeling.builder import build_backbone_configs  # noqa: E402
from deberta.modeling.deberta_v2_native import DebertaV2Model  # noqa: E402
from deberta.modeling.flashdeberta_patch import enable_flashdeberta_attention  # noqa: E402
from deberta.training.compile import (  # noqa: E402
    _bf16_runtime_sanity_check,
    _maybe_enable_tf32,
    _stabilize_compile_attention_mask,
    prepare_flash_attention_batch_metadata,
)
from deberta.training.runtime import _build_train_dataset_and_collator  # noqa: E402
from deberta.training.steps import _move_batch_to_device  # noqa: E402


@dataclass(frozen=True)
class BatchSample:
    """One sampled batch plus its flash-routing metadata."""

    index: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    flash_seq_lengths: torch.Tensor | None
    active_tokens: int
    slot_tokens: int
    batch_size: int
    seq_len: int
    head_dim: int
    att_span: int
    device_capability: str
    density_bucket: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/custom/pretrain_rtd_hf_deberta_v3pos_smol2stage4_2048_wp32k_v2.yaml",
    )
    parser.add_argument("--branch", choices=("discriminator", "generator"), default="discriminator")
    parser.add_argument("--route", choices=("fixed", "varlen", "both"), default="both")
    parser.add_argument("--sample-batches", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate env bundle as 'name:key=value,key=value'. Use 'default' for no overrides.",
    )
    parser.add_argument("--packing-enabled", choices=("true", "false"), default="false")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def _default_out_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("local-scratch/benchmarks/flashdeberta") / f"varlen_tuning_{stamp}"


def _parse_candidate_specs(values: list[str]) -> list[tuple[str, dict[str, str]]]:
    if not values:
        return [("default", {})]
    out: list[tuple[str, dict[str, str]]] = []
    for raw in values:
        text = str(raw).strip()
        if not text or text == "default":
            out.append(("default", {}))
            continue
        if ":" not in text:
            raise ValueError(f"Candidate must be 'name:key=value,...'; got {text!r}")
        name, env_text = text.split(":", 1)
        env_map: dict[str, str] = {}
        for item in env_text.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                raise ValueError(f"Candidate env override must be KEY=VALUE; got {item!r}")
            key, value = item.split("=", 1)
            env_map[key.strip()] = value.strip()
        out.append((name.strip(), env_map))
    return out


def _density_bucket(*, seq_len: int, active_tokens: int, batch_size: int) -> str:
    capacity = max(1, int(seq_len) * max(1, int(batch_size)))
    density = float(active_tokens) / float(capacity)
    if int(seq_len) >= 4096:
        return "4096_plus"
    if int(seq_len) >= 2048:
        return "2048_medium" if density >= 0.60 else "2048_sparse"
    return "1024_dense_or_medium"


def _device_capability_text(device: torch.device) -> str:
    index = device.index
    if index is None:
        major, minor = torch.cuda.get_device_capability()
    else:
        major, minor = torch.cuda.get_device_capability(index)
    return f"sm_{major}{minor}"


def _resolve_out_dir(path: Path | None) -> Path:
    out_dir = path if path is not None else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir.resolve()


def _sample_batches(
    *,
    config_path: str,
    branch: str,
    sample_batches: int,
    packing_enabled: str,
    device: torch.device,
) -> tuple[list[BatchSample], Any]:
    overrides = [
        "logging.wandb.enabled=false",
        "logging.backend=none",
        "train.checkpoint.export_hf_final=false",
        f"data.packing.enabled={packing_enabled}",
    ]
    cfg = load_config(config_path, overrides=overrides)
    model_cfg = cfg.model
    data_cfg = cfg.data
    train_cfg = cfg.train

    mixed_precision = resolve_effective_mixed_precision(
        train_cfg.mixed_precision,
        bf16_sanity_check=_bf16_runtime_sanity_check,
    )
    _maybe_enable_tf32(bool(train_cfg.tf32))
    if str(mixed_precision).strip().lower() != "bf16":
        raise RuntimeError("flashdeberta_varlen_tune.py currently expects bf16 mixed precision.")

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

    disc_config, gen_config = build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=int(data_cfg.max_seq_length),
    )
    backbone_config = disc_config if str(branch) == "discriminator" else gen_config
    head_dim = int(backbone_config.hidden_size) // int(backbone_config.num_attention_heads)
    att_span = (
        int(backbone_config.position_buckets)
        if int(getattr(backbone_config, "position_buckets", 0)) > 0
        else int(backbone_config.max_relative_positions)
    )
    capability_text = _device_capability_text(device)

    samples: list[BatchSample] = []
    for batch_idx, batch in enumerate(loader):
        if len(samples) >= int(sample_batches):
            break
        batch = _move_batch_to_device(batch, device)
        doc_ids = batch.pop("doc_ids", None)
        if doc_ids is not None:
            continue
        batch = _stabilize_compile_attention_mask(
            batch=batch,
            compile_enabled=True,
            compile_scope="backbones",
            backbone_type=str(model_cfg.backbone_type),
        )
        batch, flash_route_hint = prepare_flash_attention_batch_metadata(
            batch=batch,
            backbone_type=str(model_cfg.backbone_type),
        )
        if flash_route_hint not in {"fixed", "varlen"}:
            continue
        input_ids = batch["input_ids"].detach().clone()
        attention_mask = batch.get("attention_mask")
        flash_seq_lengths = batch.get("flash_seq_lengths")
        seq_len = int(input_ids.shape[-1])
        batch_size = int(input_ids.shape[0])
        active_tokens_value = batch.get("flash_active_tokens", 0)
        if isinstance(active_tokens_value, torch.Tensor):
            active_tokens = int(active_tokens_value.item())
        else:
            active_tokens = int(active_tokens_value)
        samples.append(
            BatchSample(
                index=int(batch_idx),
                input_ids=input_ids,
                attention_mask=attention_mask.detach().clone()
                if isinstance(attention_mask, torch.Tensor)
                else None,
                flash_seq_lengths=flash_seq_lengths.detach().clone()
                if isinstance(flash_seq_lengths, torch.Tensor)
                else None,
                active_tokens=active_tokens,
                slot_tokens=int(batch_size * seq_len),
                batch_size=batch_size,
                seq_len=seq_len,
                head_dim=head_dim,
                att_span=att_span,
                device_capability=capability_text,
                density_bucket=_density_bucket(
                    seq_len=seq_len,
                    active_tokens=active_tokens,
                    batch_size=batch_size,
                ),
            )
        )
    if not samples:
        raise RuntimeError("Failed to sample any fixed/varlen-capable unpacked batches.")
    return samples, backbone_config


def _build_model(backbone_config: Any, *, device: torch.device) -> DebertaV2Model:
    model = DebertaV2Model(backbone_config).to(device=device, dtype=torch.bfloat16)
    model.train()
    return model


def _apply_candidate_env(env_map: dict[str, str]) -> dict[str, str | None]:
    saved: dict[str, str | None] = {}
    for key, value in env_map.items():
        saved[key] = os.environ.get(key)
        os.environ[key] = str(value)
    return saved


def _restore_candidate_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _run_candidate(
    *,
    model: DebertaV2Model,
    samples: list[BatchSample],
    route: str,
    warmup: int,
    steps: int,
) -> tuple[float, float, float, float, dict[int, float]]:
    times_ms: list[float] = []
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats(device)
    per_sample_times: dict[int, list[float]] = {int(sample.index): [] for sample in samples}

    def _run_one(sample: BatchSample) -> float:
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = model(
            input_ids=sample.input_ids,
            attention_mask=sample.attention_mask,
            flash_seq_lengths=sample.flash_seq_lengths,
            flash_route_hint=str(route),
        ).last_hidden_state
        loss = out.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0

    for _ in range(int(warmup)):
        for sample in samples:
            _run_one(sample)

    for _ in range(int(steps)):
        for sample in samples:
            elapsed_ms = _run_one(sample)
            times_ms.append(elapsed_ms)
            per_sample_times[int(sample.index)].append(elapsed_ms)

    elapsed_s = sum(times_ms) / 1000.0
    active_tokens = sum(sample.active_tokens for sample in samples) * int(steps)
    slot_tokens = sum(sample.slot_tokens for sample in samples) * int(steps)
    active_tok_s = float(active_tokens) / max(elapsed_s, 1e-9)
    slot_tok_s = float(slot_tokens) / max(elapsed_s, 1e-9)
    max_mem_gib = torch.cuda.max_memory_allocated() / (1024**3)
    per_sample_mean = {
        sample_idx: statistics.mean(values) for sample_idx, values in per_sample_times.items() if values
    }
    return statistics.mean(times_ms), active_tok_s, slot_tok_s, max_mem_gib, per_sample_mean


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for flashdeberta_varlen_tune.py")

    out_dir = _resolve_out_dir(args.out_dir)
    device = torch.device("cuda")
    torch.manual_seed(0)
    enable_flashdeberta_attention(strict=True)

    samples, backbone_config = _sample_batches(
        config_path=str(args.config),
        branch=str(args.branch),
        sample_batches=int(args.sample_batches),
        packing_enabled=str(args.packing_enabled),
        device=device,
    )
    model = _build_model(backbone_config, device=device)

    routes = ["fixed", "varlen"] if str(args.route) == "both" else [str(args.route)]
    candidates = _parse_candidate_specs(list(args.candidate))

    (out_dir / "batches.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "index": sample.index,
                    "seq_len": sample.seq_len,
                    "batch_size": sample.batch_size,
                    "active_tokens": sample.active_tokens,
                    "slot_tokens": sample.slot_tokens,
                    "density_bucket": sample.density_bucket,
                    "head_dim": sample.head_dim,
                    "att_span": sample.att_span,
                    "device_capability": sample.device_capability,
                }
            )
            for sample in samples
        )
        + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "candidate\troute\tmean_ms\tactive_tok_per_s\tslot_tok_per_s\tmax_memory_gib\tseq_len\ttotal_tokens\thead_dim\tatt_span\tdevice_capability\tdensity_bucket"
    ]
    best_by_key: dict[str, dict[str, Any]] = {}

    for candidate_name, env_map in candidates:
        saved_env = _apply_candidate_env(env_map)
        try:
            for route in routes:
                mean_ms, active_tok_s, slot_tok_s, max_mem_gib, per_sample_mean = _run_candidate(
                    model=model,
                    samples=samples,
                    route=route,
                    warmup=int(args.warmup),
                    steps=int(args.steps),
                )
                for sample in samples:
                    key = json.dumps(
                        {
                            "seq_len": sample.seq_len,
                            "total_tokens": sample.active_tokens,
                            "head_dim": sample.head_dim,
                            "att_span": sample.att_span,
                            "device_capability": sample.device_capability,
                            "route": route,
                        },
                        sort_keys=True,
                    )
                    existing = best_by_key.get(key)
                    sample_mean_ms = float(per_sample_mean.get(int(sample.index), mean_ms))
                    if existing is None or sample_mean_ms < float(existing["mean_ms"]):
                        best_by_key[key] = {
                            "candidate": candidate_name,
                            "route": route,
                            "mean_ms": sample_mean_ms,
                            "env": env_map,
                            "density_bucket": sample.density_bucket,
                        }
                sample0 = samples[0]
                density_buckets = {sample.density_bucket for sample in samples}
                summary_lines.append(
                    "\t".join(
                        [
                            candidate_name,
                            route,
                            f"{mean_ms:.4f}",
                            f"{active_tok_s:.2f}",
                            f"{slot_tok_s:.2f}",
                            f"{max_mem_gib:.3f}",
                            str(sample0.seq_len),
                            str(sum(sample.active_tokens for sample in samples)),
                            str(sample0.head_dim),
                            str(sample0.att_span),
                            sample0.device_capability,
                            next(iter(density_buckets)) if len(density_buckets) == 1 else "mixed",
                        ]
                    )
                )
        finally:
            _restore_candidate_env(saved_env)

    (out_dir / "summary.tsv").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (out_dir / "best_configs.json").write_text(
        json.dumps(best_by_key, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote_summary={out_dir / 'summary.tsv'}")
    print(f"wrote_batches={out_dir / 'batches.jsonl'}")
    print(f"wrote_best={out_dir / 'best_configs.json'}")


if __name__ == "__main__":
    main()
