#!/usr/bin/env python3
"""Tune FlashDeBERTa dense local-bias kernels against real packed doc-block batches.

This tool complements the existing varlen tuner. It samples actual packed
doc-block batches from the repo dataloader, forces the short-sequence dense
flash-with-bias route, and sweeps repo-local dense-bias kernel overrides.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
from pathlib import Path
from typing import Any

import _bench_common as bench
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/flashdeberta/pretrain_rtd_hf_deberta_v3pos_smol2stage4_1024_wp32k_v2_docblock.yaml",
    )
    parser.add_argument("--branch", choices=("discriminator", "generator"), default="discriminator")
    parser.add_argument("--sample-batches", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=JSON_PATH",
        help="Named model.hf.flash.kernel_overrides_path candidate; use 'default' for the shipped table.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for flashdeberta_bias_tune.py")

    out_dir = bench.resolve_out_dir(args.out_dir, default_prefix="bias_tuning")
    device = torch.device("cuda")
    torch.manual_seed(0)

    cfg, _, tokenizer, loader = bench.load_tool_config_and_loader(
        str(args.config),
        [
            "logging.wandb.enabled=false",
            "logging.backend=none",
            "train.checkpoint.export_hf_final=false",
            "model.hf.attention_impl=flash",
            "data.packing.enabled=true",
            "data.packing.block_cross_document_attention=true",
        ],
        tool_name="flashdeberta_bias_tune.py",
        require_bf16=True,
    )
    backbone_config, head_dim, att_span = bench.build_branch_backbone_config(
        model_cfg=cfg.model,
        data_cfg=cfg.data,
        tokenizer=tokenizer,
        branch=str(args.branch),
    )
    samples = bench.sample_flash_batches(
        loader=loader,
        model_cfg=cfg.model,
        sample_batches=int(args.sample_batches),
        device=device,
        head_dim=head_dim,
        att_span=att_span,
        route_hints={"docblock_bias"},
        require_attention_mask=True,
        with_pair_density=True,
        empty_error="Failed to sample any dense doc-block flash batches.",
    )
    # Build once and reuse across candidates: override tables only change kernel
    # selection, and per-candidate rebuilds gave each sweep a different random
    # init (a drift from the varlen tuner) for pure GPU alloc/cast overhead.
    model = bench.build_bf16_backbone(backbone_config, device=device)
    candidates = bench.parse_candidate_specs(list(args.candidate))

    batch_rows = [
        {
            "index": sample.index,
            "seq_len": sample.seq_len,
            "batch_size": sample.batch_size,
            "active_tokens": sample.active_tokens,
            "slot_tokens": sample.slot_tokens,
            "head_dim": sample.head_dim,
            "att_span": sample.att_span,
            "device_capability": sample.device_capability,
            "pair_density": sample.pair_density,
        }
        for sample in samples
    ]

    summary_lines = [
        "candidate\tstatus\tmean_ms\tslot_tok_per_s\tmax_memory_gib\tseq_len\tbatch_size\ttotal_tokens\thead_dim\tatt_span\tdevice_capability\tpair_density\terror"
    ]
    best_by_key: dict[str, dict[str, Any]] = {}

    results = bench.run_candidate_sweep(
        model=model,
        samples=samples,
        candidates=candidates,
        routes=["docblock_bias"],
        restore_path=cfg.model.hf.flash.kernel_overrides_path,
        meta_fn=lambda sample, route: (
            dataclasses.replace(sample.flash_meta, route_hint=route)
            if sample.flash_meta is not None
            else None
        ),
        warmup=int(args.warmup),
        steps=int(args.steps),
    )
    sample0 = samples[0]
    mean_pair_density = statistics.mean(sample.pair_density for sample in samples)
    total_slot_tokens = sum(sample.slot_tokens for sample in samples)
    for result in results:
        if result.timing is None:
            summary_lines.append(
                "\t".join(
                    [
                        result.candidate_name,
                        "failed",
                        "",
                        "",
                        "",
                        str(sample0.seq_len),
                        str(sample0.batch_size),
                        str(total_slot_tokens),
                        str(sample0.head_dim),
                        str(sample0.att_span),
                        sample0.device_capability,
                        f"{mean_pair_density:.6f}",
                        result.error or "",
                    ]
                )
            )
            print(f"candidate_failed name={result.candidate_name} error={result.error}")
            continue

        timing = result.timing
        for sample in samples:
            bench.update_best_candidate(
                best_by_key,
                key_fields={
                    "seq_len": sample.seq_len,
                    "batch_size": sample.batch_size,
                    "head_dim": sample.head_dim,
                    "att_span": sample.att_span,
                    "device_capability": sample.device_capability,
                },
                candidate_name=result.candidate_name,
                candidate_path=result.candidate_path,
                mean_ms=timing.per_sample_mean_ms.get(int(sample.index), timing.mean_ms),
                details={"pair_density": sample.pair_density},
            )
        summary_lines.append(
            "\t".join(
                [
                    result.candidate_name,
                    "ok",
                    f"{timing.mean_ms:.4f}",
                    f"{timing.slot_tok_per_s:.2f}",
                    f"{timing.max_memory_gib:.3f}",
                    str(sample0.seq_len),
                    str(sample0.batch_size),
                    str(total_slot_tokens),
                    str(sample0.head_dim),
                    str(sample0.att_span),
                    sample0.device_capability,
                    f"{mean_pair_density:.6f}",
                    "",
                ]
            )
        )

    bench.write_tuning_outputs(
        out_dir=out_dir,
        batch_rows=batch_rows,
        summary_lines=summary_lines,
        best_by_key=best_by_key,
    )


if __name__ == "__main__":
    main()
