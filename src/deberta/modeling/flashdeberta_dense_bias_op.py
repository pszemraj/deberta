# ruff: noqa: F821,UP037
"""Compile-safe dense DeBERTa bias assembly for FlashDeBERTa local-bias routes.

The short-sequence local-bias flash path still needs a dense additive bias
tensor in ``(B,H,S,S)`` layout. Building that tensor with standard PyTorch ops
works, but on current GPUs the compiled gather/mask/scaling chain is still a
visible steady-state hotspot for packed doc-block training.

This module owns the eager and Triton builders used by the production
position-bias custom op. It gathers c2p/p2c positional terms and applies the
optional keep mask plus softmax scale; backward helpers reduce dense bias
gradients with cached row-range metadata derived from the bucket map.
"""

from __future__ import annotations

import weakref

import torch

from deberta.modeling.flashdeberta_kernel_tuning import (
    FlashKernelContext,
    resolve_flash_kernel_config,
)
from deberta.modeling.flashdeberta_op_utils import (
    device_compute_capability,
    keep_mask_head_stride,
)
from deberta.modeling.flashdeberta_op_utils import (
    kernel_dtype_name as _kernel_dtype_name,
)
from deberta.modeling.mask_utils import is_torch_compiling

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional import
    triton = None
    tl = None
_DENSE_BUCKET_RANGE_CACHE: dict[
    tuple[int, int, int, tuple[int, ...], tuple[int, ...], str, int],
    tuple[weakref.ReferenceType[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
] = {}


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

    if device.type != "cuda" or dtype not in {torch.float16, torch.bfloat16}:
        return None
    capability = device_compute_capability(device)
    return resolve_flash_kernel_config(
        FlashKernelContext(
            compute_capability=capability,
            route="dense_bias",
            kind="fwd",
            seq_len=int(seq_len),
            batch_size=int(batch_size),
            num_heads=int(num_heads),
            head_dim=0,
            dtype=_kernel_dtype_name(dtype),
            has_mask=bool(has_mask),
        )
    )


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
    :param torch.Tensor | None keep_mask: Optional keep mask in ``(B,1,S,S)`` or per-head ``(B,H,S,S)`` layout.
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
        # p2c at (query m, key n) reads pos_query[n, bucket_index[n, m]]: gather
        # each key row with its own signed bucket map, then transpose to (m, n).
        # Gathering the transposed map instead would flip the signed bucket and
        # diverge from the Triton builder and eager attention.
        p2c_bias = torch.take_along_dim(pos_query, gather_index, dim=-1).transpose(-1, -2)
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


def _dense_bucket_range_cache_key(
    bucket_index: torch.Tensor,
    *,
    num_buckets: int,
) -> tuple[int, int, int, tuple[int, ...], tuple[int, ...], str, int]:
    """Return a tensor-stable cache key for one dense bucket map.

    :param torch.Tensor bucket_index: Dense bucket map in ``(S,S)`` layout.
    :param int num_buckets: Positional-bias width.
    :return tuple[int, int, int, tuple[int, ...], tuple[int, ...], str, int]:
        Tensor id, version, storage offset, shape, stride, device text, and bucket count.
    """

    return (
        id(bucket_index),
        int(bucket_index._version),
        int(bucket_index.storage_offset()),
        tuple(int(dim) for dim in bucket_index.shape),
        tuple(int(dim) for dim in bucket_index.stride()),
        str(bucket_index.device),
        int(num_buckets),
    )


def _dense_bucket_ranges(
    bucket_index: torch.Tensor,
    *,
    num_buckets: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached contiguous bucket ranges for one dense bucket map.

    Each DeBERTa row maps columns into bucket ids without re-entering the same
    bucket later in the row. That lets backward replace random scatter-add
    traffic with contiguous segment reductions.

    :param torch.Tensor bucket_index: Dense bucket map in ``(S,S)`` layout.
    :param int num_buckets: Positional-bias width.
    :return tuple[torch.Tensor, torch.Tensor]: Inclusive start/end indices in ``(S,P)`` layout.
    """

    cache_key: tuple[int, int, int, tuple[int, ...], tuple[int, ...], str, int] | None = None
    if not is_torch_compiling():
        try:
            cache_key = _dense_bucket_range_cache_key(bucket_index, num_buckets=num_buckets)
        except Exception:
            cache_key = None
        if cache_key is not None:
            cached = _DENSE_BUCKET_RANGE_CACHE.get(cache_key)
            if cached is not None:
                bucket_ref, ranges = cached
                if bucket_ref() is bucket_index:
                    return ranges
                _DENSE_BUCKET_RANGE_CACHE.pop(cache_key, None)

    seq_len = int(bucket_index.shape[0])
    column_ids = torch.arange(seq_len, device=bucket_index.device, dtype=torch.int64)
    column_ids = column_ids.view(1, seq_len).expand(seq_len, seq_len)
    bucket_ids = bucket_index.to(dtype=torch.int64)

    missing = torch.full(
        (seq_len, int(num_buckets)),
        int(seq_len),
        device=bucket_index.device,
        dtype=torch.int64,
    )
    start = missing.scatter_reduce(
        -1,
        bucket_ids,
        column_ids,
        reduce="amin",
        include_self=True,
    )
    start = torch.where(start == int(seq_len), torch.full_like(start, -1), start)

    end = torch.full(
        (seq_len, int(num_buckets)),
        -1,
        device=bucket_index.device,
        dtype=torch.int64,
    )
    end = end.scatter_reduce(
        -1,
        bucket_ids,
        column_ids,
        reduce="amax",
        include_self=True,
    )

    if cache_key is not None:
        try:
            bucket_ref = weakref.ref(
                bucket_index,
                lambda _ref, key=cache_key: _DENSE_BUCKET_RANGE_CACHE.pop(key, None),
            )
        except TypeError:
            bucket_ref = None
        if bucket_ref is not None:
            _DENSE_BUCKET_RANGE_CACHE[cache_key] = (bucket_ref, (start, end))
        if len(_DENSE_BUCKET_RANGE_CACHE) > 32:
            stale_keys = list(_DENSE_BUCKET_RANGE_CACHE.keys())[:16]
            for stale_key in stale_keys:
                _DENSE_BUCKET_RANGE_CACHE.pop(stale_key, None)
    return start, end


def _dense_bucket_reduce_from_ranges(
    *,
    grad: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Reduce dense bias gradients back into bucket space from cached ranges.

    :param torch.Tensor grad: Dense bias gradient in ``(B,H,S,S)`` layout.
    :param torch.Tensor start: Inclusive bucket start indices in ``(S,P)`` layout.
    :param torch.Tensor end: Inclusive bucket end indices in ``(S,P)`` layout.
    :param torch.dtype output_dtype: Output dtype for the reduced bucket tensor.
    :return torch.Tensor: Reduced gradient in ``(B,H,S,P)`` layout.
    """

    seq_len = int(start.shape[0])
    num_buckets = int(start.shape[1])
    if int(num_buckets) <= 0:
        return grad.new_zeros((*grad.shape[:3], 0), dtype=output_dtype)
    prefix = grad.to(dtype=torch.float32).cumsum(dim=-1)

    start_view = start.view(1, 1, seq_len, int(num_buckets))
    end_view = end.view(1, 1, seq_len, int(num_buckets))
    valid = end_view >= 0

    end_idx = end_view.clamp_min(0).expand(int(grad.shape[0]), int(grad.shape[1]), -1, -1)
    end_vals = torch.take_along_dim(prefix, end_idx, dim=-1)

    start_prev = start_view - 1
    start_prev_idx = start_prev.clamp_min(0).expand(int(grad.shape[0]), int(grad.shape[1]), -1, -1)
    start_vals = torch.take_along_dim(prefix, start_prev_idx, dim=-1)
    start_mask = start_prev >= 0

    reduced = end_vals - torch.where(start_mask.expand_as(end_vals), start_vals, torch.zeros_like(start_vals))
    reduced = torch.where(valid.expand_as(reduced), reduced, torch.zeros_like(reduced))
    return reduced.to(dtype=output_dtype)


def _dense_bucket_reduce(
    *,
    grad: torch.Tensor,
    bucket_index: torch.Tensor,
    num_buckets: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Reduce dense bias gradients back into bucket space.

    :param torch.Tensor grad: Dense bias gradient in ``(B,H,S,S)`` layout.
    :param torch.Tensor bucket_index: Dense bucket map in ``(S,S)`` layout.
    :param int num_buckets: Positional-bias width.
    :param torch.dtype output_dtype: Output dtype for the reduced bucket tensor.
    :return torch.Tensor: Reduced gradient in ``(B,H,S,P)`` layout.
    """

    start, end = _dense_bucket_ranges(bucket_index, num_buckets=int(num_buckets))
    return _dense_bucket_reduce_from_ranges(
        grad=grad,
        start=start,
        end=end,
        output_dtype=output_dtype,
    )


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
            # stride_mask_h is normalized at launch: 0 for broadcast (B,1,S,S)
            # masks, the real head stride for per-head (B,H,S,S) masks.
            keep = tl.load(
                keep_mask_ptr
                + off_b * stride_mask_b
                + off_h * stride_mask_h
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
    :param torch.Tensor | None keep_mask: Optional keep mask in ``(B,1,S,S)`` or per-head ``(B,H,S,S)`` layout.
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
            keep_mask_head_stride(keep_mask, num_heads=num_heads),
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
