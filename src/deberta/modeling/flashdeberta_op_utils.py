"""Shared torch/CUDA infrastructure helpers for FlashDeBERTa custom-op modules."""

from __future__ import annotations

from typing import Any

import torch

try:  # pragma: no cover - optional Triton dependency
    import triton as _triton
except Exception:  # pragma: no cover - optional Triton dependency
    _triton = None


def lookup_registered_op(namespace: str, name: str) -> Any | None:
    """Return a previously registered custom op overload, if one exists.

    :param str namespace: Operator namespace.
    :param str name: Operator name.
    :return Any | None: Registered overload or ``None``.
    """

    ns = getattr(torch.ops, namespace, None)
    if ns is None or not hasattr(ns, name):
        return None
    op = getattr(ns, name)
    return getattr(op, "default", op)


def lookup_existing_op_pair(namespace: str, fwd_name: str, bwd_name: str) -> tuple[Any, Any] | None:
    """Return an already-registered forward/backward custom-op pair, if complete.

    :param str namespace: Operator namespace.
    :param str fwd_name: Forward operator name.
    :param str bwd_name: Backward operator name.
    :return tuple[Any, Any] | None: ``(forward, backward)`` overloads, or ``None``
        unless both are registered.
    """

    existing_forward = lookup_registered_op(namespace, fwd_name)
    existing_backward = lookup_registered_op(namespace, bwd_name)
    if existing_forward is not None and existing_backward is not None:
        return existing_forward, existing_backward
    return None


def kernel_dtype_name(dtype: torch.dtype) -> str:
    """Return a compact dtype name for tuning-table matching.

    :param torch.dtype dtype: Torch dtype.
    :return str: Dtype name without the ``torch.`` prefix.
    """

    return str(dtype).removeprefix("torch.")


def optional_triton_jit(fn: object) -> object:
    """Apply ``triton.jit`` only when Triton imported successfully.

    :param object fn: Kernel function.
    :return object: JIT kernel or unchanged function in no-Triton environments.
    """

    if _triton is None:
        return fn
    return _triton.jit(fn)


def traceable_triton_kernel(kernel: object) -> object:
    """Return a traceable Triton kernel wrapper when PyTorch exposes one.

    :param object kernel: Raw Triton kernel or autotuned wrapper.
    :return object: Traceable wrapper when available, otherwise ``kernel`` unchanged.
    """

    if not hasattr(torch, "library") or not hasattr(torch.library, "wrap_triton"):
        return kernel
    try:
        return torch.library.wrap_triton(kernel)
    except Exception:
        return kernel


def flatten_padded_rows(
    tensor: torch.Tensor, *, context: str
) -> tuple[torch.Tensor, tuple[int, ...], int, int]:
    """Flatten a contiguous ``(B, S, ...)`` tensor into row-major ``(B*S, F)`` form.

    :param torch.Tensor tensor: Tensor with at least two leading dims ``(B, S)``.
    :param str context: Caller name used in validation errors (e.g. ``"Prefix-pack"``).
    :raises ValueError: If the tensor rank is less than two or non-contiguous.
    :return tuple[torch.Tensor, tuple[int, ...], int, int]:
        Flattened tensor, trailing shape, batch size, and padded sequence length.
    """

    if tensor.ndim < 2:
        raise ValueError(f"Expected tensor with leading (B,S) dims; got shape={tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{context} Triton path requires contiguous tensors.")
    batch_size = int(tensor.shape[0])
    seq_len = int(tensor.shape[1])
    trailing_shape = tuple(int(dim) for dim in tensor.shape[2:])
    flat = tensor.view(batch_size * seq_len, -1)
    return flat, trailing_shape, batch_size, seq_len


def can_use_triton_pack(
    *,
    available: bool,
    tensor: torch.Tensor,
    metadata_tensors: tuple[torch.Tensor, ...],
) -> bool:
    """Return whether a Triton pack/unpack path is usable for one call.

    :param bool available: Whether the caller's Triton kernels imported.
    :param torch.Tensor tensor: Input or output tensor.
    :param tuple[torch.Tensor, ...] metadata_tensors: Metadata tensors that must
        live on the same device as ``tensor``.
    :return bool: True when the Triton path is usable.
    """

    if not available:
        return False
    if tensor.device.type != "cuda":
        return False
    if not tensor.is_contiguous():
        return False
    return all(meta.device == tensor.device for meta in metadata_tensors)


def device_compute_capability(device: torch.device) -> tuple[int, int]:
    """Return CUDA compute capability for one device.

    :param torch.device device: Device to query.
    :return tuple[int, int]: ``(major, minor)`` capability, or ``(0, 0)`` off CUDA.
    """

    if device.type != "cuda":
        return (0, 0)
    index = device.index
    if index is None:
        return torch.cuda.get_device_capability()
    return torch.cuda.get_device_capability(index)
