# ruff: noqa: F821,UP037
"""Compile-safe dense DeBERTa bias assembly for FlashDeBERTa local-bias routes.

The short-sequence local-bias flash path still needs a dense additive bias
tensor in ``(B,H,S,S)`` layout. Building that tensor with standard PyTorch ops
works, but on current GPUs the compiled gather/mask/scaling chain is still a
visible steady-state hotspot for packed doc-block training.

This module exposes a repo-local fused forward builder as an opaque custom op.
It gathers c2p/p2c positional terms and applies the optional keep mask plus
softmax scale inside one Triton launch, while the backward path keeps the
existing semantics via explicit scatter-add rules.
"""

from __future__ import annotations

import os
from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional import
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc

_DENSE_BIAS_NAMESPACE = "deberta"
_DENSE_BIAS_OP_NAME = "flashdeberta_dense_bias"


def flashdeberta_compiled_dense_bias_available() -> bool:
    """Return whether the opaque dense-bias CUDA op is available.

    :return bool: True when the custom-op based CUDA path is registered.
    """

    return _FLASHDEBERTA_DENSE_BIAS_CUSTOM_OP is not None


def flashdeberta_dense_bias_import_error() -> Exception | None:
    """Return the import/runtime error for the fused dense-bias builder, if any.

    :return Exception | None: Import failure or ``None`` when the fused builder is available.
    """

    return _TRITON_IMPORT_ERROR


def _lookup_registered_op(namespace: str, name: str) -> Any | None:
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


def _dense_bias_kernel_override_from_env() -> tuple[int, int, int, int] | None:
    """Return dense-bias builder launch overrides when fully specified.

    Supported env vars:
    - ``FLASHDEBERTA_DENSE_BIAS_BLOCK_M``
    - ``FLASHDEBERTA_DENSE_BIAS_BLOCK_N``
    - ``FLASHDEBERTA_DENSE_BIAS_NUM_STAGES``
    - ``FLASHDEBERTA_DENSE_BIAS_NUM_WARPS``

    :return tuple[int, int, int, int] | None: Override ``(BLOCK_M, BLOCK_N, stages, warps)``
        or ``None`` when unset / invalid / incomplete.
    """

    names = (
        "FLASHDEBERTA_DENSE_BIAS_BLOCK_M",
        "FLASHDEBERTA_DENSE_BIAS_BLOCK_N",
        "FLASHDEBERTA_DENSE_BIAS_NUM_STAGES",
        "FLASHDEBERTA_DENSE_BIAS_NUM_WARPS",
    )
    raw = [os.environ.get(name) for name in names]
    if any(value is None or not str(value).strip() for value in raw):
        return None
    try:
        block_m, block_n, num_stages, num_warps = (int(str(value).strip()) for value in raw)
    except Exception:
        return None
    return int(block_m), int(block_n), int(num_stages), int(num_warps)


def _dense_bias_repo_tuned_config(
    *,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    has_mask: bool,
) -> tuple[int, int, int, int] | None:
    """Return repo-local dense-bias builder configs for measured hot paths.

    This helper stays conservative. Promote a tuned config only after it wins
    in the real packed-docblock RTD loop, not just in isolated sweeps.

    :param int batch_size: Batch size.
    :param int num_heads: Number of attention heads.
    :param int seq_len: Query/key sequence length.
    :param torch.dtype dtype: Input/output dtype.
    :param torch.device device: CUDA device hosting the builder tensors.
    :param bool has_mask: Whether the keep mask is active.
    :return tuple[int, int, int, int] | None: Tuned ``(BLOCK_M, BLOCK_N, stages, warps)``
        or ``None`` when no measured config is promoted.
    """

    if device.type != "cuda":
        return None
    if dtype not in {torch.float16, torch.bfloat16}:
        return None
    if int(seq_len) != 1024:
        return None
    if int(batch_size) != 4 or int(num_heads) != 12:
        return None
    capability = (
        torch.cuda.get_device_capability(device.index)
        if device.index is not None
        else torch.cuda.get_device_capability()
    )
    if int(capability[0]) < 12:
        return None
    del has_mask
    return (64, 128, 2, 4)


def _dense_bias_kernel_config(
    *,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    has_mask: bool,
) -> tuple[int, int, int, int]:
    """Resolve the fused dense-bias builder launch config.

    :param int batch_size: Batch size.
    :param int num_heads: Number of attention heads.
    :param int seq_len: Query/key sequence length.
    :param torch.dtype dtype: Input/output dtype.
    :param torch.device device: CUDA device hosting the builder tensors.
    :param bool has_mask: Whether the keep mask is active.
    :return tuple[int, int, int, int]: ``(BLOCK_M, BLOCK_N, stages, warps)``.
    """

    override = _dense_bias_kernel_override_from_env()
    if override is not None:
        return override
    tuned = _dense_bias_repo_tuned_config(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        dtype=dtype,
        device=device,
        has_mask=has_mask,
    )
    if tuned is not None:
        return tuned
    return (64, 64, 2, 4)


def _dense_bias_forward_fallback(
    *,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    bucket_index: torch.Tensor,
    keep_mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Build dense flash bias with eager gather/mask ops as a fallback.

    :param torch.Tensor | None pos_key: Optional c2p term in ``(B,H,S,P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c term in ``(B,H,S,P)`` layout.
    :param torch.Tensor bucket_index: Dense bucket map in ``(S,S)`` layout.
    :param torch.Tensor | None keep_mask: Optional keep mask in ``(B,1,S,S)`` layout.
    :param float scale: Scale applied to the final additive bias.
    :raises RuntimeError: If both positional terms are missing.
    :return torch.Tensor: Scaled dense additive bias in ``(B,H,S,S)`` layout.
    """

    seq_len = int(bucket_index.shape[0])
    gather_index = bucket_index.view(1, 1, seq_len, seq_len)
    bias: torch.Tensor | None = None

    if pos_key is not None:
        bias = torch.take_along_dim(pos_key, gather_index, dim=-1)

    if pos_query is not None:
        reverse_index = bucket_index.transpose(0, 1).view(1, 1, seq_len, seq_len)
        p2c_bias = torch.take_along_dim(pos_query, reverse_index, dim=-1).transpose(-1, -2)
        if bias is None:
            bias = p2c_bias
        else:
            bias.add_(p2c_bias)

    if bias is None:  # pragma: no cover - guarded by callers
        raise RuntimeError("Dense flash bias construction requires at least one positional term.")

    bias.mul_(float(scale))
    if keep_mask is not None:
        bias.masked_fill_(~keep_mask, -1.0e4 * float(scale))
    return bias


if triton is not None:

    @triton.jit
    def _dense_bias_fwd_kernel(
        pos_key_ptr: "ptr",
        pos_query_ptr: "ptr",
        bucket_ptr: "ptr",
        keep_mask_ptr: "ptr",
        out_ptr: "ptr",
        stride_pk_b: "int",
        stride_pk_h: "int",
        stride_pk_s: "int",
        stride_pk_p: "int",
        stride_pq_b: "int",
        stride_pq_h: "int",
        stride_pq_s: "int",
        stride_pq_p: "int",
        stride_bucket_s: "int",
        stride_bucket_n: "int",
        stride_mask_b: "int",
        stride_mask_h: "int",
        stride_mask_s: "int",
        stride_mask_n: "int",
        stride_out_b: "int",
        stride_out_h: "int",
        stride_out_s: "int",
        stride_out_n: "int",
        H: "int",
        S: "int",
        scale: "float",
        neg_bias: "float",
        HAS_POS_KEY: tl.constexpr,
        HAS_POS_QUERY: tl.constexpr,
        HAS_MASK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ) -> "None":
        """Assemble scaled dense local-bias tiles inside one Triton kernel.

        :param pos_key_ptr: Base pointer for the optional c2p tensor.
        :param pos_query_ptr: Base pointer for the optional p2c tensor.
        :param bucket_ptr: Base pointer for the dense bucket map.
        :param keep_mask_ptr: Base pointer for the optional keep mask.
        :param out_ptr: Base pointer for the output bias tensor.
        :param stride_pk_b: Batch stride for ``pos_key_ptr``.
        :param stride_pk_h: Head stride for ``pos_key_ptr``.
        :param stride_pk_s: Sequence stride for ``pos_key_ptr``.
        :param stride_pk_p: Bucket stride for ``pos_key_ptr``.
        :param stride_pq_b: Batch stride for ``pos_query_ptr``.
        :param stride_pq_h: Head stride for ``pos_query_ptr``.
        :param stride_pq_s: Sequence stride for ``pos_query_ptr``.
        :param stride_pq_p: Bucket stride for ``pos_query_ptr``.
        :param stride_bucket_s: Row stride for ``bucket_ptr``.
        :param stride_bucket_n: Column stride for ``bucket_ptr``.
        :param stride_mask_b: Batch stride for ``keep_mask_ptr``.
        :param stride_mask_h: Head stride for ``keep_mask_ptr``.
        :param stride_mask_s: Row stride for ``keep_mask_ptr``.
        :param stride_mask_n: Column stride for ``keep_mask_ptr``.
        :param stride_out_b: Batch stride for ``out_ptr``.
        :param stride_out_h: Head stride for ``out_ptr``.
        :param stride_out_s: Row stride for ``out_ptr``.
        :param stride_out_n: Column stride for ``out_ptr``.
        :param H: Number of attention heads.
        :param S: Sequence length.
        :param scale: Softmax scale applied to positional terms.
        :param neg_bias: Value written for masked entries.
        :param HAS_POS_KEY: Whether the c2p term is active.
        :param HAS_POS_QUERY: Whether the p2c term is active.
        :param HAS_MASK: Whether the keep mask is active.
        :param BLOCK_M: Tile height.
        :param BLOCK_N: Tile width.
        :return: Triton kernels write results in-place.
        """

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_bh = tl.program_id(2)

        off_h = pid_bh % H
        off_b = pid_bh // H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = (offs_m[:, None] < S) & (offs_n[None, :] < S)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        if HAS_POS_KEY:
            bucket = tl.load(
                bucket_ptr + offs_m[:, None] * stride_bucket_s + offs_n[None, :] * stride_bucket_n,
                mask=valid,
                other=0,
            ).to(tl.int32)
            pk_ptrs = (
                pos_key_ptr
                + off_b * stride_pk_b
                + off_h * stride_pk_h
                + offs_m[:, None] * stride_pk_s
                + bucket * stride_pk_p
            )
            acc += tl.load(pk_ptrs, mask=valid, other=0.0).to(tl.float32)

        if HAS_POS_QUERY:
            bucket_t = tl.load(
                bucket_ptr + offs_n[None, :] * stride_bucket_s + offs_m[:, None] * stride_bucket_n,
                mask=valid,
                other=0,
            ).to(tl.int32)
            pq_ptrs = (
                pos_query_ptr
                + off_b * stride_pq_b
                + off_h * stride_pq_h
                + offs_n[None, :] * stride_pq_s
                + bucket_t * stride_pq_p
            )
            acc += tl.load(pq_ptrs, mask=valid, other=0.0).to(tl.float32)

        acc *= scale
        if HAS_MASK:
            keep = tl.load(
                keep_mask_ptr
                + off_b * stride_mask_b
                + offs_m[:, None] * stride_mask_s
                + offs_n[None, :] * stride_mask_n,
                mask=valid,
                other=0,
            )
            acc = tl.where(keep.to(tl.int1), acc, neg_bias)

        out_ptrs = (
            out_ptr
            + off_b * stride_out_b
            + off_h * stride_out_h
            + offs_m[:, None] * stride_out_s
            + offs_n[None, :] * stride_out_n
        )
        tl.store(out_ptrs, acc, mask=valid)


def _dense_bias_forward_cuda(
    *,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    bucket_index: torch.Tensor,
    keep_mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Build scaled dense flash bias with the fused Triton forward kernel.

    :param torch.Tensor | None pos_key: Optional c2p term in ``(B,H,S,P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c term in ``(B,H,S,P)`` layout.
    :param torch.Tensor bucket_index: Dense bucket map in ``(S,S)`` layout.
    :param torch.Tensor | None keep_mask: Optional keep mask in ``(B,1,S,S)`` layout.
    :param float scale: Scale applied to the final additive bias.
    :raises RuntimeError: If both positional terms are missing.
    :return torch.Tensor: Scaled dense additive bias in ``(B,H,S,S)`` layout.
    """

    reference = pos_key if pos_key is not None else pos_query
    if reference is None:
        raise RuntimeError("Dense flash bias construction requires at least one positional term.")

    batch_size, num_heads, seq_len = int(reference.shape[0]), int(reference.shape[1]), int(reference.shape[2])
    output = torch.empty(
        (batch_size, num_heads, seq_len, seq_len),
        device=reference.device,
        dtype=reference.dtype,
    )

    pos_key_tensor = (
        pos_key if pos_key is not None else torch.empty((0,), device=reference.device, dtype=reference.dtype)
    )
    pos_query_tensor = (
        pos_query
        if pos_query is not None
        else torch.empty((0,), device=reference.device, dtype=reference.dtype)
    )
    keep_mask_tensor = (
        keep_mask if keep_mask is not None else torch.empty((0,), device=reference.device, dtype=torch.bool)
    )
    block_m, block_n, num_stages, num_warps = _dense_bias_kernel_config(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        dtype=reference.dtype,
        device=reference.device,
        has_mask=keep_mask is not None,
    )

    grid = (triton.cdiv(seq_len, block_m), triton.cdiv(seq_len, block_n), batch_size * num_heads)
    with torch.cuda.device(reference.device.index):
        _dense_bias_fwd_kernel[grid](
            pos_key_tensor,
            pos_query_tensor,
            bucket_index,
            keep_mask_tensor,
            output,
            pos_key_tensor.stride(0) if pos_key is not None else 0,
            pos_key_tensor.stride(1) if pos_key is not None else 0,
            pos_key_tensor.stride(2) if pos_key is not None else 0,
            pos_key_tensor.stride(3) if pos_key is not None else 0,
            pos_query_tensor.stride(0) if pos_query is not None else 0,
            pos_query_tensor.stride(1) if pos_query is not None else 0,
            pos_query_tensor.stride(2) if pos_query is not None else 0,
            pos_query_tensor.stride(3) if pos_query is not None else 0,
            bucket_index.stride(0),
            bucket_index.stride(1),
            keep_mask_tensor.stride(0) if keep_mask is not None else 0,
            keep_mask_tensor.stride(1) if keep_mask is not None else 0,
            keep_mask_tensor.stride(2) if keep_mask is not None else 0,
            keep_mask_tensor.stride(3) if keep_mask is not None else 0,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            num_heads,
            seq_len,
            float(scale),
            float(-1.0e4 * float(scale)),
            HAS_POS_KEY=pos_key is not None,
            HAS_POS_QUERY=pos_query is not None,
            HAS_MASK=keep_mask is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return output


def _build_dense_bias_custom_op() -> Any | None:
    """Register or retrieve the opaque fused dense-bias custom op.

    :return Any | None: Forward custom-op handle or ``None`` when unavailable.
    """

    existing = _lookup_registered_op(_DENSE_BIAS_NAMESPACE, _DENSE_BIAS_OP_NAME)
    if existing is not None:
        return existing
    if triton is None or not hasattr(torch, "library") or not hasattr(torch.library, "custom_op"):
        return None

    @torch.library.custom_op(
        f"{_DENSE_BIAS_NAMESPACE}::{_DENSE_BIAS_OP_NAME}",
        mutates_args=(),
        device_types="cuda",
        schema=(
            "(Tensor pos_key, Tensor pos_query, Tensor bucket_index, Tensor keep_mask, float scale, "
            "bool has_pos_key, bool has_pos_query, bool has_keep_mask) -> Tensor"
        ),
    )
    def _forward_op(
        pos_key: torch.Tensor,
        pos_query: torch.Tensor,
        bucket_index: torch.Tensor,
        keep_mask: torch.Tensor,
        scale: float,
        has_pos_key: bool,
        has_pos_query: bool,
        has_keep_mask: bool,
    ) -> torch.Tensor:
        """Run dense bias assembly as one opaque CUDA op.

        :param torch.Tensor pos_key: c2p tensor or empty sentinel.
        :param torch.Tensor pos_query: p2c tensor or empty sentinel.
        :param torch.Tensor bucket_index: Dense bucket map in ``(S,S)`` layout.
        :param torch.Tensor keep_mask: Keep mask tensor or empty sentinel.
        :param float scale: Bias scale factor.
        :param bool has_pos_key: Whether ``pos_key`` is active.
        :param bool has_pos_query: Whether ``pos_query`` is active.
        :param bool has_keep_mask: Whether ``keep_mask`` is active.
        :return torch.Tensor: Scaled dense bias tensor.
        """

        return _dense_bias_forward_cuda(
            pos_key=pos_key if bool(has_pos_key) else None,
            pos_query=pos_query if bool(has_pos_query) else None,
            bucket_index=bucket_index,
            keep_mask=keep_mask if bool(has_keep_mask) else None,
            scale=float(scale),
        )

    @torch.library.register_fake(_forward_op)
    def _forward_op_fake(
        pos_key: torch.Tensor,
        pos_query: torch.Tensor,
        bucket_index: torch.Tensor,
        keep_mask: torch.Tensor,
        scale: float,
        has_pos_key: bool,
        has_pos_query: bool,
        has_keep_mask: bool,
    ) -> torch.Tensor:
        """Return fake dense-bias output with the correct static shape.

        :param torch.Tensor pos_key: Fake c2p tensor or empty sentinel.
        :param torch.Tensor pos_query: Fake p2c tensor or empty sentinel.
        :param torch.Tensor bucket_index: Fake dense bucket map.
        :param torch.Tensor keep_mask: Fake keep mask tensor or empty sentinel.
        :param float scale: Fake scale factor.
        :param bool has_pos_key: Whether ``pos_key`` is active.
        :param bool has_pos_query: Whether ``pos_query`` is active.
        :param bool has_keep_mask: Whether ``keep_mask`` is active.
        :raises RuntimeError: If neither positional term is active.
        :return torch.Tensor: Fake dense bias tensor.
        """

        del bucket_index, keep_mask, scale, has_keep_mask
        reference = pos_key if bool(has_pos_key) else pos_query
        if not bool(has_pos_key) and not bool(has_pos_query):
            raise RuntimeError("Dense flash bias construction requires at least one positional term.")
        seq_len = int(reference.shape[2])
        return torch.empty(
            (reference.shape[0], reference.shape[1], seq_len, seq_len),
            device=reference.device,
            dtype=reference.dtype,
        )

    def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        """Save tensors needed for the explicit dense-bias backward rule.

        :param Any ctx: Autograd context.
        :param tuple[Any, ...] inputs: Forward inputs.
        :param torch.Tensor output: Forward output tensor.
        """

        del output
        pos_key, pos_query, bucket_index, keep_mask, scale, has_pos_key, has_pos_query, has_keep_mask = inputs
        ctx.save_for_backward(pos_key, pos_query, bucket_index, keep_mask)
        ctx.scale = float(scale)
        ctx.has_pos_key = bool(has_pos_key)
        ctx.has_pos_query = bool(has_pos_query)
        ctx.has_keep_mask = bool(has_keep_mask)

    def _backward(ctx: Any, grad_out: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        """Propagate dense-bias gradients back to c2p/p2c tensors.

        :param Any ctx: Autograd context populated by ``_setup_context``.
        :param torch.Tensor | None grad_out: Gradient of the dense bias output.
        :return tuple[torch.Tensor | None, ...]: Gradients for the forward inputs.
        """

        pos_key, pos_query, bucket_index, keep_mask = ctx.saved_tensors
        if grad_out is None:
            return None, None, None, None, None, None, None, None

        grad = grad_out
        if ctx.has_keep_mask:
            grad = grad.masked_fill(~keep_mask, 0)
        grad = grad * float(ctx.scale)

        dpos_key: torch.Tensor | None = None
        if ctx.has_pos_key:
            gather_index = bucket_index.view(1, 1, bucket_index.shape[0], bucket_index.shape[1]).expand(
                grad.shape[0], grad.shape[1], -1, -1
            )
            dpos_key = torch.zeros_like(pos_key).scatter_add_(-1, gather_index, grad)

        dpos_query: torch.Tensor | None = None
        if ctx.has_pos_query:
            reverse_index = (
                bucket_index.transpose(0, 1)
                .view(1, 1, bucket_index.shape[0], bucket_index.shape[1])
                .expand(grad.shape[0], grad.shape[1], -1, -1)
            )
            dpos_query = torch.zeros_like(pos_query).scatter_add_(-1, reverse_index, grad.transpose(-1, -2))

        return dpos_key, dpos_query, None, None, None, None, None, None

    torch.library.register_autograd(_forward_op, _backward, setup_context=_setup_context)
    return _forward_op


_FLASHDEBERTA_DENSE_BIAS_CUSTOM_OP = _build_dense_bias_custom_op()


def flashdeberta_dense_bias(
    *,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    bucket_index: torch.Tensor,
    keep_mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Build a scaled dense DeBERTa bias tensor for FlashDeBERTa bias routes.

    :param torch.Tensor | None pos_key: Optional c2p term in ``(B,H,S,P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c term in ``(B,H,S,P)`` layout.
    :param torch.Tensor bucket_index: Dense bucket map in ``(S,S)`` layout.
    :param torch.Tensor | None keep_mask: Optional keep mask in ``(B,1,S,S)`` layout.
    :param float scale: Scale applied to the final additive bias.
    :return torch.Tensor: Scaled dense bias tensor in ``(B,H,S,S)`` layout.
    """

    reference = pos_key if pos_key is not None else pos_query
    if reference is None:
        raise RuntimeError("Dense flash bias construction requires at least one positional term.")
    if _FLASHDEBERTA_DENSE_BIAS_CUSTOM_OP is not None and reference.device.type == "cuda":
        sentinel = torch.empty((0,), device=reference.device, dtype=reference.dtype)
        keep_sentinel = (
            keep_mask
            if keep_mask is not None
            else torch.empty((0,), device=reference.device, dtype=torch.bool)
        )
        return _FLASHDEBERTA_DENSE_BIAS_CUSTOM_OP(
            pos_key if pos_key is not None else sentinel,
            pos_query if pos_query is not None else sentinel,
            bucket_index,
            keep_sentinel,
            float(scale),
            bool(pos_key is not None),
            bool(pos_query is not None),
            bool(keep_mask is not None),
        )
    return _dense_bias_forward_fallback(
        pos_key=pos_key,
        pos_query=pos_query,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
        scale=float(scale),
    )


__all__ = [
    "flashdeberta_compiled_dense_bias_available",
    "flashdeberta_dense_bias",
    "flashdeberta_dense_bias_import_error",
]
