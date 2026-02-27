"""Checkpoint state-dict utilities shared by training and export paths."""

from __future__ import annotations

from pathlib import Path

import torch


def canonical_compile_state_key(key: str) -> str:
    """Return a canonical state-dict key with compile wrappers removed.

    :param str key: Raw state-dict key.
    :return str: Canonical key where ``._orig_mod`` segments are removed.
    """
    return str(key).replace("._orig_mod", "")


def canonicalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a state dict with compile-wrapper key segments normalized.

    :param dict[str, torch.Tensor] state_dict: Raw state dictionary.
    :raises RuntimeError: If two distinct source keys collide after canonicalization.
    :return dict[str, torch.Tensor]: Canonicalized state dictionary.
    """
    canonical: dict[str, tuple[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        key_norm = canonical_compile_state_key(key)
        prev = canonical.get(key_norm)
        if prev is not None and prev[0] != key:
            raise RuntimeError(
                "Ambiguous state-dict keys after compile canonicalization: "
                f"{prev[0]!r} and {key!r} -> {key_norm!r}"
            )
        canonical[key_norm] = (key, value)
    return {key_norm: entry[1] for key_norm, entry in canonical.items()}


def load_checkpoint_model_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load model state dict from an Accelerate checkpoint directory.

    :param Path checkpoint_dir: Checkpoint directory containing model weights.
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

    raise RuntimeError(f"No model state file found in checkpoint directory: {checkpoint_dir}")


def load_model_state_with_compile_key_remap(model: torch.nn.Module, checkpoint_dir: Path) -> dict[str, int]:
    """Load checkpoint weights into ``model`` with compile-wrapper key remapping.

    :param torch.nn.Module model: Target model instance.
    :param Path checkpoint_dir: Checkpoint directory path.
    :raises RuntimeError: If canonicalized key sets are incompatible.
    :return dict[str, int]: Matched/missing/unexpected key counts.
    """
    target_state = model.state_dict()
    checkpoint_state = load_checkpoint_model_state_dict(checkpoint_dir)

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
