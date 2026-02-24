#!/usr/bin/env python
"""Deterministic torch.compile parity check for DeBERTa RTD pretraining.

This script compares eager vs compiled runs on the same model state, same batch,
and restored RNG state. It supports selective compile scopes to bisect issues.
"""

from __future__ import annotations

import argparse
import inspect
import math
from pathlib import Path
from typing import Any

import torch


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file into a dictionary.

    :param Path path: YAML path.
    :return dict[str, Any]: Parsed YAML dictionary.
    """
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("pyyaml is required: pip install pyyaml") from e

    raw = yaml.safe_load(path.read_text())
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("YAML must parse to a dict")
    return raw


def _split_sections(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split config into model/data/train sections.

    :param dict[str, Any] raw: Parsed config dictionary.
    :return tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: model/data/train mappings.
    """
    if any(k in raw for k in ("model", "data", "train")):
        return raw.get("model", {}) or {}, raw.get("data", {}) or {}, raw.get("train", {}) or {}

    from deberta.config import DataConfig, ModelConfig, TrainConfig

    model_keys = {f.name for f in ModelConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    data_keys = {f.name for f in DataConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    train_keys = {f.name for f in TrainConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]

    m: dict[str, Any] = {}
    d: dict[str, Any] = {}
    t: dict[str, Any] = {}
    for key, value in raw.items():
        if key in model_keys:
            m[key] = value
        elif key in data_keys:
            d[key] = value
        elif key in train_keys:
            t[key] = value
        else:
            raise ValueError(f"Unknown top-level key: {key}")
    return m, d, t


def _global_grad_norm(model: torch.nn.Module) -> float:
    """Compute global L2 norm over all gradients.

    :param torch.nn.Module model: Model to inspect.
    :return float: Global gradient L2 norm.
    """
    sq_sum = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        sq_sum += float(grad.float().pow(2).sum().item())
    return float(sq_sum) ** 0.5


def _capture_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Capture gradients into a CPU dictionary.

    :param torch.nn.Module model: Model to inspect.
    :return dict[str, torch.Tensor]: Mapping from parameter name to gradient tensor.
    """
    out: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        key = name.replace("._orig_mod.", ".")
        if key in out:
            raise RuntimeError(f"Duplicate gradient key after canonicalization: {key}")
        out[key] = grad.float().cpu().clone()
    return out


def _gradient_vector_delta(
    *,
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> tuple[float, float]:
    """Return relative L2 and max-abs deltas between two gradient vectors.

    :param dict[str, torch.Tensor] reference: Baseline gradients.
    :param dict[str, torch.Tensor] candidate: Candidate gradients.
    :return tuple[float, float]: (relative_l2_delta, max_abs_delta).
    """
    if set(reference.keys()) != set(candidate.keys()):
        ref_only = sorted(set(reference.keys()) - set(candidate.keys()))
        cand_only = sorted(set(candidate.keys()) - set(reference.keys()))
        raise RuntimeError(
            "Gradient parameter mismatch between runs. "
            f"reference_only={ref_only[:5]}, candidate_only={cand_only[:5]}"
        )

    diff_sq = 0.0
    ref_sq = 0.0
    max_abs = 0.0
    for name in sorted(reference.keys()):
        grad_ref = reference[name]
        grad_cand = candidate[name]
        if grad_ref.shape != grad_cand.shape:
            raise RuntimeError(f"Gradient shape mismatch for {name}: {tuple(grad_ref.shape)} vs {tuple(grad_cand.shape)}")
        diff = (grad_cand - grad_ref).float()
        diff_sq += float(diff.pow(2).sum().item())
        ref_sq += float(grad_ref.float().pow(2).sum().item())
        if diff.numel() != 0:
            max_abs = max(max_abs, float(diff.abs().max().item()))

    rel_l2 = math.sqrt(diff_sq) / max(math.sqrt(ref_sq), 1e-12)
    return rel_l2, max_abs


@torch.no_grad()
def _make_synth_batch(
    *,
    vocab_size: int,
    mask_token_id: int,
    forbidden_ids: set[int],
    batch_size: int,
    seq_len: int,
    mlm_probability: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build a synthetic masked batch.

    :param int vocab_size: Vocabulary size.
    :param int mask_token_id: Mask token id.
    :param set[int] forbidden_ids: Forbidden token ids for base sequence creation.
    :param int batch_size: Batch size.
    :param int seq_len: Sequence length.
    :param float mlm_probability: Masking probability.
    :param torch.device device: Target device.
    :return dict[str, torch.Tensor]: Batch dictionary.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    for token_id in forbidden_ids:
        if 0 <= token_id < vocab_size:
            input_ids = torch.where(input_ids == token_id, (input_ids + 1) % vocab_size, input_ids)

    labels = torch.full((batch_size, seq_len), -100, device=device, dtype=torch.long)
    mask = torch.rand((batch_size, seq_len), device=device) < float(mlm_probability)
    mask[0, 0] = True
    labels[mask] = input_ids[mask]

    input_ids_masked = input_ids.clone()
    input_ids_masked[mask] = int(mask_token_id)
    attention_mask = torch.ones_like(input_ids_masked)
    return {"input_ids": input_ids_masked, "labels": labels, "attention_mask": attention_mask}


def _snapshot_rng_state(*, device: torch.device) -> dict[str, Any]:
    """Capture RNG state for CPU and CUDA.

    :param torch.device device: Active device.
    :return dict[str, Any]: RNG state payload.
    """
    out: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if device.type == "cuda" and torch.cuda.is_available():
        out["cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return out


def _restore_rng_state(*, device: torch.device, state: dict[str, Any]) -> None:
    """Restore RNG state for CPU and CUDA.

    :param torch.device device: Active device.
    :param dict[str, Any] state: RNG state payload from ``_snapshot_rng_state``.
    """
    cpu_state = state.get("cpu", None)
    if isinstance(cpu_state, torch.Tensor):
        torch.set_rng_state(cpu_state)
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_state = state.get("cuda", None)
        if isinstance(cuda_state, list):
            torch.cuda.set_rng_state_all(cuda_state)


def _set_seed(*, seed: int, device: torch.device) -> None:
    """Set global RNG seed.

    :param int seed: Seed value.
    :param torch.device device: Active device.
    """
    torch.manual_seed(int(seed))
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _rel_delta(a: float, b: float) -> float:
    """Compute scale-invariant relative delta.

    :param float a: First value.
    :param float b: Second value.
    :return float: Relative absolute difference.
    """
    denom = max(abs(a), abs(b), 1e-8)
    return abs(a - b) / denom


def main() -> None:
    """Run eager vs compiled parity checks and exit non-zero on eval-gate failures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="YAML config path")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--compile-mode", type=str, default="default")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--fail-threshold", type=float, default=1e-2)
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic algorithm mode (warn-only) for reproducibility.",
    )
    ap.add_argument(
        "--scope",
        type=str,
        default="matrix",
        choices=("gen", "disc", "both", "matrix"),
        help="Compile scope to test. matrix=gen+disc+both.",
    )
    ap.add_argument(
        "--backend",
        type=str,
        default="inductor",
        choices=("inductor", "aot_eager"),
        help="Compile backend selection.",
    )
    ap.add_argument(
        "--eval",
        dest="eval_mode",
        action="store_true",
        default=True,
        help="Run in eval() mode (default; pass/fail gate).",
    )
    ap.add_argument(
        "--train",
        dest="eval_mode",
        action="store_false",
        help="Run in train() mode (informational only; no threshold-based failure).",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    raw = _load_yaml(cfg_path)
    model_dict, data_dict, train_dict = _split_sections(raw)

    from deberta.config import DataConfig, ModelConfig, TrainConfig
    from deberta.modeling.builder import build_backbone_configs, build_backbones
    from deberta.modeling.rtd import DebertaV3RTDPretrainer

    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers is required") from e

    model_cfg = ModelConfig(**model_dict)
    data_cfg = DataConfig(**data_dict)
    train_cfg = TrainConfig(**train_dict)

    device = torch.device(args.device)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    _set_seed(seed=int(args.seed), device=device)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name_or_path, use_fast=True)
    required_max_position_embeddings = max(int(data_cfg.max_seq_length), int(args.seq_len))

    disc_cfg, gen_cfg = build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=required_max_position_embeddings,
    )

    def build_model() -> torch.nn.Module:
        disc_model, gen_model = build_backbones(
            model_cfg=model_cfg,
            disc_config=disc_cfg,
            gen_config=gen_cfg,
        )
        rtd = DebertaV3RTDPretrainer(
            discriminator_backbone=disc_model,
            generator_backbone=gen_model,
            disc_config=disc_cfg,
            gen_config=gen_cfg,
            embedding_sharing=model_cfg.embedding_sharing,
            tie_generator_word_embeddings=True,
        ).to(device)
        if args.eval_mode:
            rtd.eval()
        else:
            rtd.train()
        return rtd

    mask_token_id = getattr(gen_cfg, "mask_token_id", None)
    if mask_token_id is None:
        raise RuntimeError("mask_token_id is not set on config; tokenizer binding may have failed")

    base_model = build_model()
    forbidden: set[int] = set(getattr(base_model, "_forbidden_sample_token_ids", set()))
    batch = _make_synth_batch(
        vocab_size=int(gen_cfg.vocab_size),
        mask_token_id=int(mask_token_id),
        forbidden_ids=forbidden,
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        mlm_probability=float(args.mlm_prob),
        device=device,
    )

    sampling_temp = float(getattr(train_cfg, "sampling_temperature", 1.0))
    gen_weight = float(getattr(train_cfg, "gen_loss_weight", 1.0))
    disc_weight = float(getattr(train_cfg, "disc_loss_weight", 50.0))
    decoupled = bool(getattr(train_cfg, "decoupled_loss_scaling", False))
    autocast_enabled = bool(args.bf16 and device.type == "cuda")

    base_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
    rng_state = _snapshot_rng_state(device=device)

    if not hasattr(torch, "compile"):
        raise RuntimeError("This torch build does not expose torch.compile")

    compile_kwargs: dict[str, Any] = {"mode": str(args.compile_mode)}
    if str(args.backend) == "aot_eager":
        compile_kwargs["backend"] = "aot_eager"
    try:
        if "dynamic" in inspect.signature(torch.compile).parameters:  # type: ignore[attr-defined]
            compile_kwargs["dynamic"] = False
    except Exception:
        pass

    def run_one(
        *,
        label: str,
        compile_gen: bool,
        compile_disc: bool,
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        model = build_model()
        model.load_state_dict(base_state, strict=True)

        sync_fn = getattr(model, "sync_discriminator_embeddings_from_generator", None)
        if callable(sync_fn):
            sync_fn()

        if compile_gen:
            model.generator = torch.compile(model.generator, **compile_kwargs)  # type: ignore[attr-defined]
        if compile_disc:
            model.discriminator = torch.compile(model.discriminator, **compile_kwargs)  # type: ignore[attr-defined]

        _restore_rng_state(device=device, state=rng_state)

        for param in model.parameters():
            param.grad = None

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            out = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
                token_type_ids=None,
                sampling_temperature=sampling_temp,
                gen_loss_weight=gen_weight,
                disc_loss_weight=disc_weight,
                decoupled_loss_scaling=decoupled,
            )
            loss = out.loss

        loss.backward()

        scalars = {
            "total_loss": float(out.loss.detach()),
            "gen_loss": float(out.gen_loss.detach()),
            "disc_loss": float(out.disc_loss.detach()),
            "grad_norm": _global_grad_norm(model),
        }
        grads = _capture_gradients(model)
        print(
            f"{label}: loss={scalars['total_loss']:.6f} "
            f"gen={scalars['gen_loss']:.6f} disc={scalars['disc_loss']:.6f} "
            f"grad_norm={scalars['grad_norm']:.6f} grads={len(grads)}"
        )
        return scalars, grads

    eager_scalars, eager_grads = run_one(label="eager", compile_gen=False, compile_disc=False)

    if args.scope == "matrix":
        scopes = [
            ("compile(gen)", True, False),
            ("compile(disc)", False, True),
            ("compile(both)", True, True),
        ]
    elif args.scope == "gen":
        scopes = [("compile(gen)", True, False)]
    elif args.scope == "disc":
        scopes = [("compile(disc)", False, True)]
    else:
        scopes = [("compile(both)", True, True)]

    failures: list[str] = []
    print("\nRelative deltas vs eager:")

    for label, compile_gen, compile_disc in scopes:
        cand_scalars, cand_grads = run_one(
            label=label,
            compile_gen=compile_gen,
            compile_disc=compile_disc,
        )

        scalar_rel = {
            key: _rel_delta(cand_scalars[key], eager_scalars[key])
            for key in ("total_loss", "gen_loss", "disc_loss", "grad_norm")
        }
        grad_rel_l2, grad_max_abs = _gradient_vector_delta(reference=eager_grads, candidate=cand_grads)

        print(
            f"  {label:<14} total={scalar_rel['total_loss']:.3e} "
            f"gen={scalar_rel['gen_loss']:.3e} "
            f"disc={scalar_rel['disc_loss']:.3e} "
            f"grad_norm={scalar_rel['grad_norm']:.3e} "
            f"grad_rel_l2={grad_rel_l2:.3e} grad_max_abs={grad_max_abs:.3e}"
        )

        scalar_allclose = all(
            math.isclose(
                cand_scalars[key],
                eager_scalars[key],
                rel_tol=float(args.rtol),
                abs_tol=float(args.atol),
            )
            for key in ("total_loss", "gen_loss", "disc_loss", "grad_norm")
        )

        if args.eval_mode and ((not scalar_allclose) or (grad_rel_l2 > float(args.fail_threshold))):
            failures.append(
                f"{label}: scalar_allclose={scalar_allclose}, grad_rel_l2={grad_rel_l2:.3e}, "
                f"fail_threshold={float(args.fail_threshold):.3e}"
            )

    if args.eval_mode:
        if failures:
            print("\nPARITY_CHECK: FAIL")
            for row in failures:
                print(f"  - {row}")
            raise SystemExit(1)
        print(
            "\nPARITY_CHECK: PASS "
            f"(scope={args.scope}, backend={args.backend}, mode={args.compile_mode}, "
            f"rtol={float(args.rtol):.3e}, atol={float(args.atol):.3e}, "
            f"fail_threshold={float(args.fail_threshold):.3e})"
        )
        return

    print(
        "\nPARITY_CHECK: INFO_ONLY "
        "(train mode selected; deltas reported but not used for pass/fail)."
    )


if __name__ == "__main__":
    main()
