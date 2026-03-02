#!/usr/bin/env python3
"""Train-step drift probe for eager vs torch.compile on native HFv2 RTD.

This utility runs eager and compiled models side-by-side from identical initial
weights on synthetic fixed-shape batches. It is intended to answer two questions:
1) does train-mode loss/parameter drift grow under a given compile scope/kernel?
2) does Dynamo show excessive recompiles/graph churn for that setup?
"""

from __future__ import annotations

import argparse
import copy
import inspect
import math
import time
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from typing import Any

import torch

from deberta.modeling.deberta_v2_native import DebertaV2Model
from deberta.modeling.rtd import DebertaV3RTDPretrainer

try:
    from transformers import DebertaV2Config, get_scheduler
except Exception as exc:  # pragma: no cover
    raise RuntimeError("transformers is required to run tools/compile_drift_probe.py") from exc


@dataclass
class ProbeConfig:
    """Runtime settings for compile drift probing."""

    device: str
    seed: int
    steps: int
    batch_size: int
    seq_len: int
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    learning_rate: float
    weight_decay: float
    adam_epsilon: float
    warmup_steps: int
    total_steps: int
    max_grad_norm: float
    mlm_probability: float
    sampling_temperature: float
    gen_loss_weight: float
    disc_loss_weight: float
    embedding_sharing: str
    hf_attention_kernel: str
    compile_backend: str
    compile_mode: str
    compile_scope: str
    compile_dynamic: bool
    bf16: bool
    print_every: int


def _parse_args() -> ProbeConfig:
    """Parse CLI flags into :class:`ProbeConfig`.

    :return ProbeConfig: Parsed probe settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-epsilon", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--total-steps", type=int, default=300)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--gen-loss-weight", type=float, default=1.0)
    parser.add_argument("--disc-loss-weight", type=float, default=50.0)
    parser.add_argument("--embedding-sharing", choices=["none", "es", "gdes"], default="gdes")
    parser.add_argument(
        "--hf-attention-kernel",
        choices=["dynamic", "cached_bmm", "stable"],
        default="stable",
    )
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--compile-scope",
        choices=["none", "backbones", "encoder", "ffn"],
        default="ffn",
    )
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    return ProbeConfig(
        device=str(args.device),
        seed=int(args.seed),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        vocab_size=int(args.vocab_size),
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        intermediate_size=int(args.intermediate_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        adam_epsilon=float(args.adam_epsilon),
        warmup_steps=int(args.warmup_steps),
        total_steps=int(args.total_steps),
        max_grad_norm=float(args.max_grad_norm),
        mlm_probability=float(args.mlm_probability),
        sampling_temperature=float(args.sampling_temperature),
        gen_loss_weight=float(args.gen_loss_weight),
        disc_loss_weight=float(args.disc_loss_weight),
        embedding_sharing=str(args.embedding_sharing),
        hf_attention_kernel=str(args.hf_attention_kernel),
        compile_backend=str(args.compile_backend),
        compile_mode=str(args.compile_mode),
        compile_scope=str(args.compile_scope),
        compile_dynamic=bool(args.compile_dynamic),
        bf16=bool(args.bf16),
        print_every=max(1, int(args.print_every)),
    )


def _build_backbone_config(cfg: ProbeConfig) -> DebertaV2Config:
    """Construct a minimal DeBERTa-v2 config for synthetic probing.

    :param ProbeConfig cfg: Probe settings.
    :return DebertaV2Config: Backbone configuration.
    """
    config = DebertaV2Config(
        vocab_size=int(cfg.vocab_size),
        hidden_size=int(cfg.hidden_size),
        num_hidden_layers=int(cfg.num_layers),
        num_attention_heads=int(cfg.num_heads),
        intermediate_size=int(cfg.intermediate_size),
        max_position_embeddings=int(cfg.seq_len),
        type_vocab_size=2,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-7,
        relative_attention=True,
        pos_att_type="c2p|p2c",
        max_relative_positions=int(cfg.seq_len),
    )
    config.pad_token_id = 0
    config.cls_token_id = 1
    config.sep_token_id = 2
    config.mask_token_id = 3
    config.hf_attention_kernel = str(cfg.hf_attention_kernel)
    return config


def _build_rtd_model(cfg: ProbeConfig) -> DebertaV3RTDPretrainer:
    """Build RTD pretrainer with native HFv2 generator/discriminator backbones.

    :param ProbeConfig cfg: Probe settings.
    :return DebertaV3RTDPretrainer: Initialized RTD model.
    """
    disc_cfg = _build_backbone_config(cfg)
    gen_cfg = _build_backbone_config(cfg)
    discriminator = DebertaV2Model(disc_cfg)
    generator = DebertaV2Model(gen_cfg)
    return DebertaV3RTDPretrainer(
        discriminator_backbone=discriminator,
        generator_backbone=generator,
        disc_config=disc_cfg,
        gen_config=gen_cfg,
        embedding_sharing=str(cfg.embedding_sharing),
        tie_generator_word_embeddings=True,
    )


def _apply_compile_scope(model: DebertaV3RTDPretrainer, cfg: ProbeConfig) -> None:
    """Apply torch.compile to the requested module scope.

    :param DebertaV3RTDPretrainer model: Target model.
    :param ProbeConfig cfg: Probe settings.
    """
    if cfg.compile_scope == "none":
        return

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch build.")

    kwargs: dict[str, Any] = {
        "backend": str(cfg.compile_backend),
        "mode": str(cfg.compile_mode),
    }
    with suppress(Exception):
        sig = inspect.signature(torch.compile)
        if "dynamic" in sig.parameters:
            kwargs["dynamic"] = bool(cfg.compile_dynamic)

    if cfg.compile_scope == "backbones":
        model.generator = torch.compile(model.generator, **kwargs)  # type: ignore[assignment]
        model.discriminator = torch.compile(model.discriminator, **kwargs)  # type: ignore[assignment]
        return

    if cfg.compile_scope == "encoder":
        model.generator.encoder = torch.compile(model.generator.encoder, **kwargs)  # type: ignore[attr-defined]
        model.discriminator.encoder = torch.compile(model.discriminator.encoder, **kwargs)  # type: ignore[attr-defined]
        return

    if cfg.compile_scope == "ffn":
        for branch_name in ("generator", "discriminator"):
            backbone = getattr(model, branch_name)
            layers = getattr(getattr(backbone, "encoder", None), "layer", None)
            if not isinstance(layers, (torch.nn.ModuleList, list, tuple)):
                raise RuntimeError(f"{branch_name}.encoder.layer missing; cannot apply ffn scope.")
            for layer in layers:
                layer.intermediate = torch.compile(layer.intermediate, **kwargs)  # type: ignore[assignment]
                layer.output = torch.compile(layer.output, **kwargs)  # type: ignore[assignment]
        return

    raise ValueError(f"Unsupported compile scope: {cfg.compile_scope}")


def _build_optimizer_scheduler(model: torch.nn.Module, cfg: ProbeConfig) -> tuple[Any, Any]:
    """Create optimizer and linear warmup scheduler.

    :param torch.nn.Module model: Target model.
    :param ProbeConfig cfg: Probe settings.
    :return tuple[Any, Any]: Optimizer and scheduler.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        betas=(0.9, 0.999),
        eps=float(cfg.adam_epsilon),
        weight_decay=float(cfg.weight_decay),
    )
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(cfg.warmup_steps),
        num_training_steps=int(cfg.total_steps),
    )
    return optimizer, scheduler


def _make_batch(cfg: ProbeConfig, *, rng: torch.Generator, device: torch.device) -> dict[str, torch.Tensor]:
    """Create one synthetic fixed-shape RTD batch.

    :param ProbeConfig cfg: Probe settings.
    :param torch.Generator rng: RNG used for deterministic generation.
    :param torch.device device: Target device.
    :return dict[str, torch.Tensor]: Input tensors.
    """
    batch = int(cfg.batch_size)
    seq_len = int(cfg.seq_len)
    vocab_size = int(cfg.vocab_size)
    cls_id, sep_id, mask_id = 1, 2, 3

    ids = torch.randint(4, vocab_size, (batch, seq_len), generator=rng, device=device, dtype=torch.long)
    ids[:, 0] = cls_id
    ids[:, -1] = sep_id

    mask_pos = torch.rand((batch, seq_len), generator=rng, device=device) < float(cfg.mlm_probability)
    mask_pos[:, 0] = False
    mask_pos[:, -1] = False
    for row in range(batch):
        if not bool(mask_pos[row].any()):
            idx = int(torch.randint(1, seq_len - 1, (1,), generator=rng, device=device).item())
            mask_pos[row, idx] = True

    labels = torch.full((batch, seq_len), -100, device=device, dtype=torch.long)
    labels[mask_pos] = ids[mask_pos]

    input_ids = ids.clone()
    input_ids[mask_pos] = mask_id

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones((batch, seq_len), device=device, dtype=torch.bool),
    }


def _param_probe_vector(model: torch.nn.Module) -> torch.Tensor:
    """Extract a compact parameter vector for cheap drift tracking.

    :param torch.nn.Module model: Source model.
    :return torch.Tensor: Flattened probe vector.
    """
    for name, param in model.named_parameters():
        if "embeddings.word_embeddings.weight" in name:
            return param.detach().float().reshape(-1)[:4096].cpu()
    for _, param in model.named_parameters():
        return param.detach().float().reshape(-1)[:4096].cpu()
    return torch.zeros(1, dtype=torch.float32)


def _run_step(
    *,
    model: DebertaV3RTDPretrainer,
    optimizer: Any,
    scheduler: Any,
    batch: dict[str, torch.Tensor],
    cfg: ProbeConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run one optimizer step and return key scalars + elapsed seconds.

    :param DebertaV3RTDPretrainer model: RTD model.
    :param Any optimizer: Optimizer.
    :param Any scheduler: LR scheduler.
    :param dict[str, torch.Tensor] batch: Step batch.
    :param ProbeConfig cfg: Probe settings.
    :param torch.device device: Runtime device.
    :return tuple[float, float, float]: (loss, gen_loss, elapsed_sec).
    """
    model.train()
    started = time.perf_counter()

    autocast_ctx: Any
    if bool(cfg.bf16) and device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    with autocast_ctx:
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            sampling_temperature=float(cfg.sampling_temperature),
            gen_loss_weight=float(cfg.gen_loss_weight),
            disc_loss_weight=float(cfg.disc_loss_weight),
        )
        loss = out.loss

    loss.backward()
    if float(cfg.max_grad_norm) > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.max_grad_norm))

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    sync_fn = getattr(model, "sync_discriminator_embeddings_from_generator", None)
    if callable(sync_fn):
        sync_fn()

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elapsed = float(time.perf_counter() - started)
    return float(loss.detach().item()), float(out.gen_loss.detach().item()), elapsed


def _format_lr(scheduler: Any) -> float:
    """Read current scheduler LR.

    :param Any scheduler: Scheduler object.
    :return float: Current LR.
    """
    with suppress(Exception):
        values = scheduler.get_last_lr()
        if values:
            return float(values[0])
    return float("nan")


def main() -> None:
    """Run eager-vs-compiled drift probe."""
    cfg = _parse_args()
    device = torch.device(str(cfg.device))

    torch.manual_seed(int(cfg.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.seed))

    eager_model = _build_rtd_model(cfg).to(device)
    compiled_model = copy.deepcopy(eager_model).to(device)

    sync_fn = getattr(compiled_model, "sync_discriminator_embeddings_from_generator", None)
    if callable(sync_fn):
        sync_fn()

    with suppress(Exception):
        import torch._dynamo as dynamo  # type: ignore

        dynamo.reset()

    _apply_compile_scope(compiled_model, cfg)

    eager_opt, eager_sched = _build_optimizer_scheduler(eager_model, cfg)
    comp_opt, comp_sched = _build_optimizer_scheduler(compiled_model, cfg)

    eager_rng = torch.Generator(device=device).manual_seed(int(cfg.seed) + 1)
    comp_rng = torch.Generator(device=device).manual_seed(int(cfg.seed) + 1)

    print(
        "probe:",
        f"device={device}",
        f"steps={cfg.steps}",
        f"kernel={cfg.hf_attention_kernel}",
        f"scope={cfg.compile_scope}",
        f"mode={cfg.compile_mode}",
        f"backend={cfg.compile_backend}",
        f"bf16={cfg.bf16}",
    )
    print("")

    for step in range(1, int(cfg.steps) + 1):
        eager_batch = _make_batch(cfg, rng=eager_rng, device=device)
        comp_batch = _make_batch(cfg, rng=comp_rng, device=device)

        eager_loss, eager_gen, eager_time = _run_step(
            model=eager_model,
            optimizer=eager_opt,
            scheduler=eager_sched,
            batch=eager_batch,
            cfg=cfg,
            device=device,
        )
        comp_loss, comp_gen, comp_time = _run_step(
            model=compiled_model,
            optimizer=comp_opt,
            scheduler=comp_sched,
            batch=comp_batch,
            cfg=cfg,
            device=device,
        )

        probe_e = _param_probe_vector(eager_model)
        probe_c = _param_probe_vector(compiled_model)
        drift = float((probe_e - probe_c).abs().mean().item())
        loss_diff = float(abs(eager_loss - comp_loss))

        if step == 1 or step % int(cfg.print_every) == 0:
            print(
                f"step={step:4d} "
                f"lr(e/c)={_format_lr(eager_sched):.3e}/{_format_lr(comp_sched):.3e} "
                f"loss(e/c)={eager_loss:.4f}/{comp_loss:.4f} "
                f"gen(e/c)={eager_gen:.4f}/{comp_gen:.4f} "
                f"|loss_diff|={loss_diff:.4e} "
                f"param_drift_meanabs={drift:.4e} "
                f"time_ms(e/c)={eager_time * 1000.0:.1f}/{comp_time * 1000.0:.1f}"
            )

        if not math.isfinite(eager_loss) or not math.isfinite(comp_loss):
            print("non-finite loss detected; stopping early.")
            break

    with suppress(Exception):
        from torch._dynamo.utils import counters  # type: ignore

        stats = dict(counters.get("stats", {}))
        recompiles = dict(counters.get("recompiles", {}))
        print("\n--- torch._dynamo counters ---")
        if stats:
            print(f"stats={stats}")
        if recompiles:
            print(f"recompiles={recompiles}")


if __name__ == "__main__":
    main()
