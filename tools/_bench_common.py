#!/usr/bin/env python3
"""Shared harness helpers for FlashDeBERTa benchmark and diagnostic tools.

Every real-batch tool in ``tools/`` needs the same scaffolding: put ``src/``
on ``sys.path``, build the repo tokenizer/dataset/collator/loader from a
config, resolve bf16/tf32 policy, construct backbones, and (for the kernel
tuners) sample flash-routed batches and time candidate env bundles. This
module owns that scaffolding once; the tools keep only their CLI surface and
report schemas.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def ensure_src_on_path() -> None:
    """Insert the repo ``src/`` directory into ``sys.path`` once.

    :return None: None.
    """

    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


ensure_src_on_path()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from deberta.config import load_config, resolve_effective_mixed_precision  # noqa: E402
from deberta.data.loading import load_hf_dataset  # noqa: E402
from deberta.modeling import DebertaV3RTDPretrainer, build_backbone_configs, build_backbones  # noqa: E402
from deberta.modeling.deberta_v2_native import DebertaV2Model  # noqa: E402
from deberta.modeling.flashdeberta_kernel_tuning import (  # noqa: E402
    compute_capability_key,
    flash_seq_bucket,
)
from deberta.modeling.flashdeberta_op_utils import device_compute_capability  # noqa: E402
from deberta.modeling.mask_utils import FlashBatchMeta  # noqa: E402
from deberta.training.compile import (  # noqa: E402
    _bf16_runtime_sanity_check,
    _maybe_enable_tf32,
    _stabilize_compile_attention_mask,
    prepare_flash_attention_batch_metadata,
)
from deberta.training.runtime import _build_train_dataset_and_collator  # noqa: E402
from deberta.training.steps import (  # noqa: E402
    _move_batch_to_device,
    _sync_discriminator_embeddings_if_available,
)


def autocast_context(mixed_precision: str) -> Any:
    """Return the autocast context matching a mixed-precision mode.

    Mixed precision in this repo is ``bf16`` or ``no``; fp16 is deliberately
    unsupported, matching the training config contract.

    :param str mixed_precision: Effective mixed-precision mode.
    :return Any: Autocast context manager (or nullcontext for full precision).
    """

    normalized = str(mixed_precision).strip().lower()
    if normalized == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def device_capability_text(device: torch.device) -> str:
    """Return the tuning-table capability key for one device.

    :param torch.device device: CUDA device.
    :return str: Capability key such as ``"sm_120"``.
    """

    return compute_capability_key(device_compute_capability(device))


def resolve_out_dir(path: Path | None, *, default_prefix: str) -> Path:
    """Resolve (and create) a benchmark output directory.

    :param Path | None path: Explicit output directory, or None for a default.
    :param str default_prefix: Prefix for the timestamped default directory.
    :return Path: Resolved output directory.
    """

    if path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = Path("local-scratch/benchmarks/flashdeberta") / f"{default_prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def parse_candidate_specs(values: list[str]) -> list[tuple[str, dict[str, str]]]:
    """Parse ``name:key=value,...`` candidate env-bundle specs.

    :param list[str] values: Raw candidate specs (``default`` means no overrides).
    :raises ValueError: If a spec is malformed.
    :return list[tuple[str, dict[str, str]]]: ``(name, env_map)`` pairs.
    """

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


@contextmanager
def candidate_env(env_map: dict[str, str]) -> Iterator[None]:
    """Apply candidate env overrides for the duration of one sweep.

    :param dict[str, str] env_map: Environment overrides to apply.
    :return Iterator[None]: Context that restores prior values on exit.
    """

    saved: dict[str, str | None] = {}
    for key, value in env_map.items():
        saved[key] = os.environ.get(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_tool_config_and_loader(
    config_path: str,
    overrides: list[str],
    *,
    tool_name: str,
    require_bf16: bool = False,
    deterministic_loader: bool = False,
) -> tuple[Any, str, Any, DataLoader]:
    """Load a training config and build the real tokenizer/dataset loader.

    :param str config_path: Training config path.
    :param list[str] overrides: Dotted config overrides.
    :param str tool_name: Tool name used in error messages.
    :param bool require_bf16: Whether to require bf16 effective mixed precision.
    :param bool deterministic_loader: Use single-process, unpinned loading.
    :raises RuntimeError: If bf16 is required but not effective.
    :return tuple[Any, str, Any, DataLoader]: Config, effective mixed
        precision, tokenizer, and train dataloader.
    """

    cfg = load_config(config_path, overrides=overrides)
    mixed_precision = resolve_effective_mixed_precision(
        cfg.train.mixed_precision,
        bf16_sanity_check=_bf16_runtime_sanity_check,
    )
    _maybe_enable_tf32(bool(cfg.train.tf32))
    if require_bf16 and str(mixed_precision).strip().lower() != "bf16":
        raise RuntimeError(f"{tool_name} currently expects bf16 mixed precision.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer.name_or_path, use_fast=True)
    raw_train = load_hf_dataset(cfg.data)
    train_dataset, collator = _build_train_dataset_and_collator(
        raw_train=raw_train,
        tokenizer=tokenizer,
        data_cfg=cfg.data,
        train_cfg=cfg.train,
        process_index=0,
        num_processes=1,
    )
    num_workers = 0 if deterministic_loader else int(cfg.train.dataloader_num_workers)
    loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.train.per_device_train_batch_size),
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False if deterministic_loader else bool(cfg.train.dataloader_pin_memory),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return cfg, str(mixed_precision), tokenizer, loader


def build_branch_backbone_config(
    *,
    model_cfg: Any,
    data_cfg: Any,
    tokenizer: Any,
    branch: str,
) -> tuple[Any, int, int]:
    """Build the backbone config for one RTD branch plus derived kernel shape.

    :param Any model_cfg: Model config.
    :param Any data_cfg: Data config.
    :param Any tokenizer: Tokenizer instance.
    :param str branch: ``discriminator`` or ``generator``.
    :return tuple[Any, int, int]: Backbone config, head dim, and attention span.
    """

    disc_config, gen_config = build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=int(data_cfg.packing.max_seq_length),
    )
    backbone_config = disc_config if str(branch) == "discriminator" else gen_config
    head_dim = int(backbone_config.hidden_size) // int(backbone_config.num_attention_heads)
    att_span = (
        int(backbone_config.position_buckets)
        if int(getattr(backbone_config, "position_buckets", 0)) > 0
        else int(backbone_config.max_relative_positions)
    )
    return backbone_config, head_dim, att_span


def build_bf16_backbone(backbone_config: Any, *, device: torch.device) -> DebertaV2Model:
    """Build one bf16 native HF DeBERTa backbone in train mode.

    :param Any backbone_config: Backbone config.
    :param torch.device device: Target device.
    :return DebertaV2Model: Backbone model.
    """

    model = DebertaV2Model(backbone_config).to(device=device, dtype=torch.bfloat16)
    model.train()
    return model


def build_rtd_pretrainer(
    *,
    cfg: Any,
    tokenizer: Any,
    device: torch.device,
    train_mode: bool = True,
) -> DebertaV3RTDPretrainer:
    """Build the full RTD pretrainer with pretrained backbones on one device.

    :param Any cfg: Full training config.
    :param Any tokenizer: Tokenizer instance.
    :param torch.device device: Target device.
    :param bool train_mode: Whether to switch the model into train mode.
    :return DebertaV3RTDPretrainer: RTD pretrainer model.
    """

    disc_config, gen_config = build_backbone_configs(
        model_cfg=cfg.model,
        tokenizer=tokenizer,
        max_position_embeddings=int(cfg.data.packing.max_seq_length),
    )
    disc_backbone, gen_backbone = build_backbones(
        model_cfg=cfg.model,
        disc_config=disc_config,
        gen_config=gen_config,
        load_pretrained_weights=True,
    )
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=disc_backbone,
        generator_backbone=gen_backbone,
        disc_config=disc_config,
        gen_config=gen_config,
        embedding_sharing=cfg.model.embedding_sharing,
        additional_forbidden_token_ids=getattr(tokenizer, "all_special_ids", []),
    ).to(device=device)
    _sync_discriminator_embeddings_if_available(model)
    if train_mode:
        model.train()
    return model


@dataclass(frozen=True)
class BatchSample:
    """One sampled real batch plus its flash-routing metadata."""

    index: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    flash_meta: FlashBatchMeta | None
    active_tokens: int
    slot_tokens: int
    batch_size: int
    seq_len: int
    head_dim: int
    att_span: int
    device_capability: str
    density_bucket: str | None = None
    pair_density: float | None = None


def sample_flash_batches(
    *,
    loader: DataLoader,
    model_cfg: Any,
    sample_batches: int,
    device: torch.device,
    head_dim: int,
    att_span: int,
    route_hints: set[str],
    skip_doc_batches: bool = False,
    require_attention_mask: bool = False,
    with_density_bucket: bool = False,
    with_pair_density: bool = False,
    empty_error: str = "Failed to sample any flash-routed batches.",
) -> list[BatchSample]:
    """Sample real batches whose flash route hint matches ``route_hints``.

    :param DataLoader loader: Train dataloader.
    :param Any model_cfg: Model config (for backbone type and flash config).
    :param int sample_batches: Number of batches to collect.
    :param torch.device device: Target device.
    :param int head_dim: Backbone head dimension (recorded per sample).
    :param int att_span: Backbone attention span (recorded per sample).
    :param set[str] route_hints: Accepted normalized route hints.
    :param bool skip_doc_batches: Drop packed doc-block batches (unpacked tuning).
    :param bool require_attention_mask: Drop batches without a tensor mask.
    :param bool with_density_bucket: Record the tuning-table density bucket.
    :param bool with_pair_density: Record the pairwise keep-mask density.
    :param str empty_error: Error message when nothing matched.
    :raises RuntimeError: If no matching batches were sampled.
    :return list[BatchSample]: Sampled batches.
    """

    capability_text = device_capability_text(device)
    samples: list[BatchSample] = []
    for batch_idx, batch in enumerate(loader):
        if len(samples) >= int(sample_batches):
            break
        batch = _move_batch_to_device(batch, device)
        if skip_doc_batches and batch.pop("doc_ids", None) is not None:
            continue
        batch = _stabilize_compile_attention_mask(
            batch=batch,
            compile_enabled=True,
            compile_scope="backbones",
            backbone_type=str(model_cfg.backbone_type),
        )
        batch, flash_meta = prepare_flash_attention_batch_metadata(
            batch=batch,
            backbone_type=str(model_cfg.backbone_type),
            flash_enabled=True,
            flash_cfg=getattr(model_cfg.hf, "flash", None),
        )
        flash_route_hint = flash_meta.normalized_route_hint() if flash_meta is not None else None
        if flash_route_hint not in route_hints:
            continue
        attention_mask = batch.get("attention_mask")
        if require_attention_mask and not isinstance(attention_mask, torch.Tensor):
            continue

        input_ids = batch["input_ids"].detach().clone()
        seq_len = int(input_ids.shape[-1])
        batch_size = int(input_ids.shape[0])
        active_tokens = (
            int(flash_meta.active_tokens_host)
            if flash_meta is not None and flash_meta.active_tokens_host is not None
            else batch_size * seq_len
        )

        density_bucket: str | None = None
        if with_density_bucket:
            density_bucket = flash_seq_bucket(
                seq_len=seq_len,
                total_tokens=active_tokens,
                batch_size=batch_size,
            )
        pair_density: float | None = None
        if with_pair_density and isinstance(attention_mask, torch.Tensor):
            keep_pairs = int(attention_mask.to(dtype=torch.int32).sum().item())
            total_pairs = max(1, batch_size * seq_len * seq_len)
            pair_density = float(keep_pairs) / float(total_pairs)

        samples.append(
            BatchSample(
                index=int(batch_idx),
                input_ids=input_ids,
                attention_mask=attention_mask.detach().clone()
                if isinstance(attention_mask, torch.Tensor)
                else None,
                flash_meta=flash_meta,
                active_tokens=active_tokens,
                slot_tokens=int(batch_size * seq_len),
                batch_size=batch_size,
                seq_len=seq_len,
                head_dim=int(head_dim),
                att_span=int(att_span),
                device_capability=capability_text,
                density_bucket=density_bucket,
                pair_density=pair_density,
            )
        )
    if not samples:
        raise RuntimeError(empty_error)
    return samples


@dataclass(frozen=True)
class CandidateTiming:
    """Timing summary for one candidate sweep over sampled batches."""

    mean_ms: float
    active_tok_per_s: float
    slot_tok_per_s: float
    max_memory_gib: float
    per_sample_mean_ms: dict[int, float]


def run_timed_candidate(
    *,
    model: DebertaV2Model,
    samples: list[BatchSample],
    meta_fn: Callable[[BatchSample], FlashBatchMeta | None],
    warmup: int,
    steps: int,
) -> CandidateTiming:
    """Time forward+backward over sampled batches for one candidate.

    :param DebertaV2Model model: Backbone model under test.
    :param list[BatchSample] samples: Sampled batches.
    :param Callable[[BatchSample], FlashBatchMeta | None] meta_fn: Per-sample
        flash-metadata builder (stamps the route under test).
    :param int warmup: Warmup sweeps over all samples.
    :param int steps: Timed sweeps over all samples.
    :return CandidateTiming: Aggregated timing and throughput stats.
    """

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
            flash_meta=meta_fn(sample),
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
    return CandidateTiming(
        mean_ms=statistics.mean(times_ms),
        active_tok_per_s=float(active_tokens) / max(elapsed_s, 1e-9),
        slot_tok_per_s=float(slot_tokens) / max(elapsed_s, 1e-9),
        max_memory_gib=torch.cuda.max_memory_allocated(device) / (1024**3),
        per_sample_mean_ms={
            sample_idx: statistics.mean(values) for sample_idx, values in per_sample_times.items() if values
        },
    )
