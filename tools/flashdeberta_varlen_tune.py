#!/usr/bin/env python3
"""Tune FlashDeBERTa fixed/varlen routes against sampled real loader batches.

This tool bridges the gap between the synthetic microbench and full RTD
training. It samples real unpacked batches from the configured dataloader,
records their padding structure, then replays those batches through a native
HF DeBERTa backbone under explicit fixed or varlen flash routing.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import _bench_common as bench
import torch

from deberta.modeling.mask_utils import FlashBatchMeta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/flashdeberta/pretrain_rtd_hf_deberta_v3pos_smol2stage4_2048_wp32k_v2.yaml",
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
        metavar="NAME=JSON_PATH",
        help="Named model.hf.flash.kernel_overrides_path candidate; use 'default' for the shipped table.",
    )
    parser.add_argument("--packing-enabled", choices=("true", "false"), default="false")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def _route_meta(sample: bench.BatchSample, route: str) -> FlashBatchMeta:
    """Rebuild sampled flash metadata with an explicit route hint.

    :param bench.BatchSample sample: Sampled batch.
    :param str route: Forced route hint.
    :return FlashBatchMeta: Metadata with ``route`` stamped.
    """

    if sample.flash_meta is None:
        return FlashBatchMeta(route_hint=str(route))
    return dataclasses.replace(sample.flash_meta, route_hint=str(route))


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for flashdeberta_varlen_tune.py")

    out_dir = bench.resolve_out_dir(args.out_dir, default_prefix="varlen_tuning")
    device = torch.device("cuda")
    torch.manual_seed(0)

    cfg, _, tokenizer, loader = bench.load_tool_config_and_loader(
        str(args.config),
        [
            "logging.wandb.enabled=false",
            "logging.backend=none",
            "train.checkpoint.export_hf_final=false",
            "model.hf.attention_impl=flash",
            f"data.packing.enabled={args.packing_enabled}",
        ],
        tool_name="flashdeberta_varlen_tune.py",
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
        route_hints={"fixed", "varlen"},
        skip_doc_batches=True,
        with_density_bucket=True,
        empty_error="Failed to sample any fixed/varlen-capable unpacked batches.",
    )
    model = bench.build_bf16_backbone(backbone_config, device=device)

    routes = ["fixed", "varlen"] if str(args.route) == "both" else [str(args.route)]
    candidates = bench.parse_candidate_specs(list(args.candidate))

    batch_rows = [
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
        for sample in samples
    ]

    summary_lines = [
        "candidate\troute\tstatus\tmean_ms\tactive_tok_per_s\tslot_tok_per_s\tmax_memory_gib\tseq_len\ttotal_tokens\thead_dim\tatt_span\tdevice_capability\tdensity_bucket\terror"
    ]
    best_by_key: dict[str, dict[str, Any]] = {}

    results = bench.run_candidate_sweep(
        model=model,
        samples=samples,
        candidates=candidates,
        routes=routes,
        restore_path=cfg.model.hf.flash.kernel_overrides_path,
        meta_fn=_route_meta,
        warmup=int(args.warmup),
        steps=int(args.steps),
    )
    sample0 = samples[0]
    density_buckets = {sample.density_bucket for sample in samples}
    density_bucket = next(iter(density_buckets)) if len(density_buckets) == 1 else "mixed"
    total_active_tokens = sum(sample.active_tokens for sample in samples)
    for result in results:
        if result.timing is None:
            summary_lines.append(
                "\t".join(
                    [
                        result.candidate_name,
                        result.route,
                        "failed",
                        "",
                        "",
                        "",
                        "",
                        str(sample0.seq_len),
                        str(total_active_tokens),
                        str(sample0.head_dim),
                        str(sample0.att_span),
                        sample0.device_capability,
                        density_bucket,
                        result.error or "",
                    ]
                )
            )
            print(f"candidate_failed name={result.candidate_name} route={result.route} error={result.error}")
            continue

        timing = result.timing
        for sample in samples:
            bench.update_best_candidate(
                best_by_key,
                key_fields={
                    "seq_len": sample.seq_len,
                    "total_tokens": sample.active_tokens,
                    "head_dim": sample.head_dim,
                    "att_span": sample.att_span,
                    "device_capability": sample.device_capability,
                    "route": result.route,
                },
                candidate_name=result.candidate_name,
                candidate_path=result.candidate_path,
                mean_ms=timing.per_sample_mean_ms.get(int(sample.index), timing.mean_ms),
                details={"route": result.route, "density_bucket": sample.density_bucket},
            )
        summary_lines.append(
            "\t".join(
                [
                    result.candidate_name,
                    result.route,
                    "ok",
                    f"{timing.mean_ms:.4f}",
                    f"{timing.active_tok_per_s:.2f}",
                    f"{timing.slot_tok_per_s:.2f}",
                    f"{timing.max_memory_gib:.3f}",
                    str(sample0.seq_len),
                    str(total_active_tokens),
                    str(sample0.head_dim),
                    str(sample0.att_span),
                    sample0.device_capability,
                    density_bucket,
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
