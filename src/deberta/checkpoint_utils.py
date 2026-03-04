"""Checkpoint state-dict utilities shared by training and export paths."""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

_FSDP_SHARDED_MODEL_PREFIX = "pytorch_model_fsdp"
logger = logging.getLogger(__name__)


def _clone_tensor_state_dict(template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Clone tensor values from a flat state dict.

    :param dict[str, torch.Tensor] template: Source tensor mapping.
    :return dict[str, torch.Tensor]: Detached/cloned tensor mapping.
    """
    cloned: dict[str, torch.Tensor] = {}
    for key, value in template.items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"Expected tensor value in model state template for key {key!r}; got {type(value).__name__}."
            )
        cloned[key] = value.detach().clone()
    return cloned


def _candidate_fsdp_sharded_dirs(checkpoint_dir: Path) -> list[Path]:
    """Return candidate FSDP sharded-model directories under a checkpoint path.

    :param Path checkpoint_dir: Checkpoint root or candidate sharded directory.
    :return list[Path]: Candidate directories in probe order.
    """
    candidates: list[Path] = []
    if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith(_FSDP_SHARDED_MODEL_PREFIX):
        candidates.append(checkpoint_dir)
    if checkpoint_dir.is_dir():
        matches = sorted(
            (
                p
                for p in checkpoint_dir.iterdir()
                if p.is_dir() and p.name.startswith(_FSDP_SHARDED_MODEL_PREFIX)
            ),
            key=lambda p: p.name,
        )
        for match in matches:
            if match not in candidates:
                candidates.append(match)
    return candidates


def _load_sharded_fsdp_model_state_dict(
    *, checkpoint_dir: Path, model_state_template: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor] | None:
    """Load model state from an Accelerate FSDP SHARDED_STATE_DICT checkpoint.

    :param Path checkpoint_dir: Checkpoint root or sharded model directory.
    :param dict[str, torch.Tensor] model_state_template: Template state dict used to materialize sharded tensors.
    :return dict[str, torch.Tensor] | None: Loaded model state dict, or ``None`` when no sharded payload is found.
    :raises RuntimeError: If sharded payload exists but cannot be loaded.
    """
    candidates = _candidate_fsdp_sharded_dirs(checkpoint_dir)
    if not candidates:
        return None

    try:
        import torch.distributed as dist
        import torch.distributed.checkpoint as dist_cp
    except Exception as exc:
        raise RuntimeError(
            "FSDP sharded checkpoint detected but torch.distributed.checkpoint is unavailable."
        ) from exc

    no_dist = not (dist.is_available() and dist.is_initialized())
    last_error: BaseException | None = None
    for candidate in candidates:
        reader = dist_cp.FileSystemReader(str(candidate))
        template_variants: list[dict[str, torch.Tensor]] = [_clone_tensor_state_dict(model_state_template)]
        with suppress(Exception):
            metadata = reader.read_metadata()
            state_meta = getattr(metadata, "state_dict_metadata", None)
            if isinstance(state_meta, dict):
                inner_keys = [
                    str(key)[len("model.") :] for key in state_meta.keys() if str(key).startswith("model.")
                ]
                if inner_keys:
                    key_matched: dict[str, torch.Tensor] = {}
                    for inner_key in inner_keys:
                        source = model_state_template.get(canonical_compile_state_key(inner_key))
                        if source is None:
                            raise RuntimeError(
                                "Missing canonical model key for sharded checkpoint entry "
                                f"{inner_key!r} in {candidate}."
                            )
                        if not isinstance(source, torch.Tensor):
                            raise RuntimeError(
                                f"Expected tensor state for key {inner_key!r}; got {type(source).__name__}."
                            )
                        key_matched[inner_key] = source.detach().clone()
                    if key_matched and set(key_matched) != set(template_variants[0]):
                        template_variants.append(key_matched)

        for template in template_variants:
            try:
                wrapped_state: dict[str, dict[str, torch.Tensor]] = {"model": template}
                dist_cp.load(
                    state_dict=wrapped_state,
                    storage_reader=reader,
                    no_dist=bool(no_dist),
                )
                loaded = wrapped_state.get("model")
                if not isinstance(loaded, dict):
                    raise RuntimeError(
                        f"Sharded checkpoint payload at {candidate} does not contain 'model' state."
                    )
                return loaded
            except BaseException as exc:  # pragma: no cover - exercised via failure path in integration tests
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                last_error = exc

    assert last_error is not None
    raise RuntimeError(
        "Failed to load FSDP sharded model state from candidates: "
        + ", ".join(str(path) for path in candidates)
    ) from last_error


def canonical_compile_state_key(key: str) -> str:
    """Return a canonical state-dict key with compile wrappers removed.

    :param str key: Raw state-dict key.
    :return str: Canonical key where ``_orig_mod`` path segments are removed.
    """
    parts = [part for part in str(key).split(".") if part and part != "_orig_mod"]
    return ".".join(parts)


def load_checkpoint_model_state_dict(
    checkpoint_dir: Path,
    *,
    model_state_template: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Load model state dict from an Accelerate checkpoint directory.

    :param Path checkpoint_dir: Checkpoint directory containing model weights.
    :param dict[str, torch.Tensor] | None model_state_template: Optional model state template used to load
        FSDP sharded checkpoints.
    :raises RuntimeError: If no supported model state file exists.
    :return dict[str, torch.Tensor]: Loaded state dictionary.
    """
    safe_path = checkpoint_dir / "model.safetensors"
    if safe_path.exists():
        try:
            from safetensors.torch import load_file as safetensors_load_file
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to import safetensors loader for {safe_path}") from exc
        return dict(safetensors_load_file(str(safe_path), device="cpu"))

    for bin_name in ("model.bin", "pytorch_model.bin"):
        bin_path = checkpoint_dir / bin_name
        if not bin_path.exists():
            continue
        loaded = torch.load(bin_path, map_location="cpu")
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Checkpoint model file does not contain a state dict: {bin_path}")
        return loaded

    if model_state_template is not None:
        sharded_loaded = _load_sharded_fsdp_model_state_dict(
            checkpoint_dir=checkpoint_dir,
            model_state_template=model_state_template,
        )
        if sharded_loaded is not None:
            return sharded_loaded

    raise RuntimeError(f"No model state file found in checkpoint directory: {checkpoint_dir}")


def load_model_state_with_compile_key_remap(model: torch.nn.Module, checkpoint_dir: Path) -> dict[str, int]:
    """Load checkpoint weights into ``model`` with compile-wrapper key remapping.

    :param torch.nn.Module model: Target model instance.
    :param Path checkpoint_dir: Checkpoint directory path.
    :raises RuntimeError: If canonicalized key sets are incompatible.
    :return dict[str, int]: Matched/missing/unexpected key counts.
    """
    target_state = model.state_dict()
    checkpoint_state = load_checkpoint_model_state_dict(
        checkpoint_dir,
        model_state_template=target_state,
    )

    target_by_canonical: dict[str, str] = {}
    for key in target_state.keys():
        canonical = canonical_compile_state_key(key)
        prev = target_by_canonical.get(canonical)
        if prev is not None and prev != key:
            raise RuntimeError(
                "Ambiguous target model keys after compile canonicalization: "
                f"{prev!r} and {key!r} -> {canonical!r}"
            )
        target_by_canonical[canonical] = key

    checkpoint_by_canonical: dict[str, tuple[str, torch.Tensor]] = {}
    for key, value in checkpoint_state.items():
        canonical = canonical_compile_state_key(key)
        prev = checkpoint_by_canonical.get(canonical)
        if prev is not None and prev[0] != key:
            raise RuntimeError(
                "Ambiguous checkpoint keys after compile canonicalization: "
                f"{prev[0]!r} and {key!r} -> {canonical!r}"
            )
        checkpoint_by_canonical[canonical] = (key, value)

    remapped_state: dict[str, torch.Tensor] = {}
    missing_keys: list[str] = []
    for canonical, target_key in target_by_canonical.items():
        source = checkpoint_by_canonical.get(canonical)
        if source is None:
            missing_keys.append(target_key)
            continue
        remapped_state[target_key] = source[1]

    unexpected_keys = [
        source_key
        for canonical, (source_key, _value) in checkpoint_by_canonical.items()
        if canonical not in target_by_canonical
    ]

    if missing_keys or unexpected_keys:
        missing_preview = ", ".join(missing_keys[:8])
        unexpected_preview = ", ".join(unexpected_keys[:8])
        raise RuntimeError(
            "Canonical checkpoint/model key mismatch after compile remap. "
            f"missing={len(missing_keys)} ({missing_preview}), "
            f"unexpected={len(unexpected_keys)} ({unexpected_preview})"
        )

    incompatible = model.load_state_dict(remapped_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Model load_state_dict reported incompatibilities after compile remap: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )

    return {"matched": len(remapped_state)}


def unwrap_compiled_model(accelerator: Any, model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap accelerate/compile wrappers, tolerating torch.compile internals.

    :param Any accelerator: Accelerator runtime.
    :param torch.nn.Module model: Possibly wrapped model.
    :return torch.nn.Module: Best-effort unwrapped model.
    """
    unwrap = accelerator.unwrap_model
    with suppress(Exception):
        return unwrap(model, keep_torch_compile=False)
    with suppress(Exception):
        return unwrap(model)

    candidate = model
    while hasattr(candidate, "module"):
        candidate = candidate.module
    return candidate


def load_state_with_compile_fallback(
    *,
    accelerator: Any,
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    context: str,
    remap_loader: Any | None = None,
) -> None:
    """Load accelerate state with fallback for ``torch.compile`` wrapper mismatches.

    :param Any accelerator: Accelerator runtime exposing ``load_state`` and ``unwrap_model``.
    :param torch.nn.Module model: Potentially wrapped target model.
    :param str | Path checkpoint_dir: Checkpoint directory path.
    :param str context: Human-readable context label (for example ``resume`` or ``export``).
    :param Any | None remap_loader: Optional loader callable for remapping state dict keys.
        Signature must match ``load_model_state_with_compile_key_remap(model, checkpoint_dir)``.
    :raises RuntimeError: If normal load fails for non-compile reasons or remap fallback fails.
    """
    ckpt = str(checkpoint_dir)
    try:
        accelerator.load_state(ckpt)
        return
    except RuntimeError as err:
        if "_orig_mod" not in str(err):
            raise

    ctx = str(context).strip() or "checkpoint"
    logger.warning(
        "Checkpoint model key mismatch due compile wrappers detected; retrying %s load "
        "with strict=False and canonical key remap.",
        ctx,
    )
    accelerator.load_state(ckpt, strict=False)
    unwrapped = unwrap_compiled_model(accelerator, model)
    remap_fn = load_model_state_with_compile_key_remap if remap_loader is None else remap_loader
    stats = remap_fn(unwrapped, Path(ckpt))
    logger.info(
        "%s model remap loaded %d tensors from %s.",
        ctx.capitalize(),
        int(stats["matched"]),
        ckpt,
    )
