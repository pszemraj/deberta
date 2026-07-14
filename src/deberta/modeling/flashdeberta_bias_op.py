"""Compile-safe wrappers for dense FlashDeBERTa local-bias attention.

Dense short-sequence DeBERTa can be faster when the disentangled relative
position terms are materialized as an additive bias matrix and dispatched
through FlashDeBERTa's standard flash-with-bias kernels. This module exposes
that path as an opaque custom op so ``torch.compile`` does not trace through
the upstream Python autograd wrapper.
"""

from __future__ import annotations

from typing import Any

import torch

from deberta.modeling.flashdeberta_dense_bias_op import (
    _dense_bias_forward_cuda,
    _dense_bias_forward_fallback,
    _dense_bucket_reduce,
)
from deberta.modeling.flashdeberta_kernel_tuning import (
    CONSERVATIVE_FLASH_KERNEL_CONFIG,
    FlashKernelContext,
    resolve_flash_kernel_config,
    resolve_repo_tuned_config,
)
from deberta.modeling.flashdeberta_op_utils import (
    device_compute_capability,
    keep_mask_head_stride,
    lookup_existing_op_pair,
    strides_or_zeros,
)
from deberta.modeling.flashdeberta_op_utils import (
    kernel_dtype_name as _kernel_dtype_name,
)

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional import
    triton = None
    tl = None

try:
    from flashdeberta.ops.flash_attention_bias import (
        flash_attention_with_bias as _flash_attention_with_bias_highlevel,
    )

    _FLASH_BIAS_HIGHLEVEL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional import
    _flash_attention_with_bias_highlevel = None
    _FLASH_BIAS_HIGHLEVEL_IMPORT_ERROR = exc

try:
    from flashdeberta.ops.flash_attention_bias import (
        _bwd_kv_kernel as _bwd_kv_kernel_bias_raw,
    )
    from flashdeberta.ops.flash_attention_bias import (
        _bwd_preprocess as _bwd_preprocess_bias_raw,
    )
    from flashdeberta.ops.flash_attention_bias import (
        _bwd_q_kernel as _bwd_q_kernel_bias_raw,
    )
    from flashdeberta.ops.flash_attention_bias import (
        flash_attn_v2_bwd as _flash_attn_v2_bwd_bias_lowlevel,
    )
    from flashdeberta.ops.flash_attention_bias import (
        flash_attn_v2_fwd as _flash_attn_v2_fwd_bias_lowlevel,
    )

    _FLASH_BIAS_LOWLEVEL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional import
    _bwd_kv_kernel_bias_raw = None
    _bwd_preprocess_bias_raw = None
    _bwd_q_kernel_bias_raw = None
    _flash_attn_v2_bwd_bias_lowlevel = None
    _flash_attn_v2_fwd_bias_lowlevel = None
    _FLASH_BIAS_LOWLEVEL_IMPORT_ERROR = exc

_BIAS_OP_NAMESPACE = "deberta"
_POSITION_BIAS_FWD_OP_NAME = "flashdeberta_position_bias_attention_v2"
_POSITION_BIAS_BWD_OP_NAME = "flashdeberta_position_bias_attention_backward_v2"


def flashdeberta_bias_import_error() -> Exception | None:
    """Return the most relevant import failure for local-bias support.

    :return Exception | None: Import failure or ``None`` when some bias path is available.
    """

    if _flash_attention_with_bias_highlevel is not None:
        return None
    if _flash_attn_v2_fwd_bias_lowlevel is not None and _flash_attn_v2_bwd_bias_lowlevel is not None:
        return None
    if _FLASH_BIAS_HIGHLEVEL_IMPORT_ERROR is not None:
        return _FLASH_BIAS_HIGHLEVEL_IMPORT_ERROR
    return _FLASH_BIAS_LOWLEVEL_IMPORT_ERROR


def flashdeberta_compiled_position_bias_available() -> bool:
    """Return whether the fused position-bias CUDA op is available.

    :return bool: True when the custom-op based CUDA path is registered.
    """

    return (
        _FLASHDEBERTA_POSITION_BIAS_CUSTOM_OP is not None
        and _FLASHDEBERTA_POSITION_BIAS_BWD_CUSTOM_OP is not None
    )


def _bias_repo_tuned_config(
    *,
    kind: str,
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[int, int, int, int] | None:
    """Return repo-local tuned dense-bias configs for measured hot paths.

    Shape ownership lives in the shared JSON tuning table. This helper only
    applies cheap safety gates before resolving a table row for the current
    dense flash-with-bias kernel launch.

    :param str kind: One of ``"fwd"``, ``"bwd"``, ``"bwd_kv"``, or ``"bwd_q"``.
    :param int batch_size: Batch size.
    :param int num_heads: Number of attention heads.
    :param int query_len: Query sequence length.
    :param int key_len: Key sequence length.
    :param int head_dim: Per-head hidden size.
    :param bool causal: Whether causal masking is enabled.
    :param torch.dtype dtype: Activation dtype.
    :param torch.device device: CUDA device.
    :return tuple[int, int, int, int] | None: Tuned ``(BLOCK_M, BLOCK_N, stages, warps)``
        or ``None`` when no repo-local override applies.
    """

    normalized_kind = str(kind).strip().lower()
    return resolve_repo_tuned_config(
        guard=lambda: (
            normalized_kind in {"fwd", "bwd", "bwd_kv", "bwd_q"}
            and not causal
            and dtype in {torch.float16, torch.bfloat16}
            and head_dim <= 64
        ),
        compute_capability=lambda: device_compute_capability(device),
        route="bias",
        kind=normalized_kind,
        seq_len=max(query_len, key_len),
        batch_size=batch_size,
        query_len=query_len,
        key_len=key_len,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=_kernel_dtype_name(dtype),
        causal=causal,
    )


def _bias_config(
    *,
    kind: str,
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[int, int, int, int]:
    """Resolve a dense-bias Triton tile config.

    :param str kind: Tuning kind name.
    :param int batch_size: Batch size.
    :param int num_heads: Number of attention heads.
    :param int query_len: Query sequence length.
    :param int key_len: Key sequence length.
    :param int head_dim: Per-head hidden size.
    :param bool causal: Whether causal masking is enabled.
    :param torch.dtype dtype: Activation dtype.
    :param torch.device device: CUDA device.
    :return tuple[int, int, int, int]: ``(BLOCK_M, BLOCK_N, stages, warps)``.
    """

    tuned = _bias_repo_tuned_config(
        kind=kind,
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=bool(causal),
        dtype=dtype,
        device=device,
    )
    if tuned is not None:
        return tuned
    return CONSERVATIVE_FLASH_KERNEL_CONFIG


def _resolve_bias_bwd_kernel_config(
    *,
    kind: str,
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[int, int, int, int]:
    """Resolve one dense-bias backward kernel config.

    Specific ``KV`` / ``Q`` overrides take precedence over the generic
    backward override so the repo can specialize the two backward kernels
    independently for measured hot paths.

    :param str kind: Either ``"kv"`` or ``"q"``.
    :param int batch_size: Batch size.
    :param int num_heads: Number of attention heads.
    :param int query_len: Query sequence length.
    :param int key_len: Key sequence length.
    :param int head_dim: Per-head hidden size.
    :param bool causal: Whether causal masking is enabled.
    :param torch.dtype dtype: Activation dtype.
    :param torch.device device: CUDA device.
    :raises ValueError: If ``kind`` is unsupported.
    :return tuple[int, int, int, int]: ``(BLOCK_M, BLOCK_N, stages, warps)``.
    """

    normalized_kind = str(kind).strip().lower()
    if normalized_kind not in {"kv", "q"}:
        raise ValueError(f"Unsupported dense-bias backward kernel kind: {kind!r}")

    repo_tuned = _bias_repo_tuned_config(
        kind=f"bwd_{normalized_kind}",
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=bool(causal),
        dtype=dtype,
        device=device,
    )
    if repo_tuned is not None:
        return repo_tuned

    return _bias_config(
        kind="bwd",
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=bool(causal),
        dtype=dtype,
        device=device,
    )


def _should_use_specialized_docblock_bias_backward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    causal: bool,
) -> bool:
    """Return whether the repo-local short doc-block bias backward should run.

    This path is intentionally narrow: it targets measured packed doc-block
    RTD hot paths where the dense-bias route is viable but the generic
    local-bias backward still spends significant time in dense positional
    gradient reduction. The tuning table is the single source of truth for
    which shapes are enabled (``bias_docblock_specialized`` rows exact-match
    the measured lengths); kernel shape constraints are enforced separately by
    the launch-config divisibility checks, which fall back to the generic path.

    :param torch.Tensor q: Forward query tensor in ``(B,H,S,D)`` layout.
    :param torch.Tensor k: Forward key tensor in ``(B,H,S,D)`` layout.
    :param torch.Tensor v: Forward value tensor in ``(B,H,S,D)`` layout.
    :param torch.Tensor bias: Dense additive bias tensor in ``(B,H,S,S)`` layout.
    :param bool causal: Whether causal masking is enabled.
    :return bool: True when the exact-match specialized backward should run.
    """

    if triton is None:
        return False
    if bool(causal):
        return False
    if q.device.type != "cuda" or bias.device.type != "cuda":
        return False
    if q.dtype not in {torch.float16, torch.bfloat16}:
        return False
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype != bias.dtype:
        return False
    if int(q.shape[-1]) != 64:
        return False
    policy = resolve_flash_kernel_config(
        FlashKernelContext(
            compute_capability=device_compute_capability(q.device),
            route="bias_docblock_specialized",
            kind="bwd",
            seq_len=max(int(q.shape[-2]), int(k.shape[-2])),
            batch_size=int(q.shape[0]),
            query_len=int(q.shape[-2]),
            key_len=int(k.shape[-2]),
            num_heads=int(q.shape[1]),
            head_dim=int(q.shape[-1]),
            dtype=_kernel_dtype_name(q.dtype),
            causal=bool(causal),
            has_mask=True,
        )
    )
    if policy is None:
        return False
    query_len = int(q.shape[-2])
    key_len = int(k.shape[-2])
    if query_len != key_len or int(v.shape[-2]) != key_len:
        return False
    if tuple(q.shape[:2]) != tuple(k.shape[:2]) or tuple(q.shape[:2]) != tuple(v.shape[:2]):
        return False
    if tuple(bias.shape) != (int(q.shape[0]), int(q.shape[1]), query_len, key_len):
        return False
    if int(bias.stride(0)) == 0 or int(bias.stride(1)) == 0:
        return False
    return True


def _resolve_docblock_specialized_bwd_kernel_config(
    *,
    kind: str,
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[int, int, int, int]:
    """Resolve the short doc-block dense-bias backward launch tuple.

    The specialized doc-block kernels are not the generic dense-bias kernels:
    their best tile can differ because the hot path always writes a full
    dense ``d_bias`` for later positional-bucket reduction. Prefer rows in the
    ``bias_docblock_specialized`` tuning namespace, with a generic ``bwd`` row
    shared by the KV and Q launches when no per-launch row is present.

    :param str kind: Either ``"kv"`` or ``"q"``.
    :param int batch_size: Batch size.
    :param int num_heads: Number of heads.
    :param int query_len: Query sequence length.
    :param int key_len: Key sequence length.
    :param int head_dim: Per-head hidden size.
    :param bool causal: Whether causal masking is enabled.
    :param torch.dtype dtype: Activation dtype.
    :param torch.device device: CUDA device.
    :raises ValueError: If ``kind`` is unsupported.
    :return tuple[int, int, int, int]: ``(BLOCK_M, BLOCK_N, stages, warps)``.
    """

    normalized_kind = str(kind).strip().lower()
    if normalized_kind not in {"kv", "q"}:
        raise ValueError(f"Unsupported doc-block bias backward kernel kind: {kind!r}")

    capability = device_compute_capability(device)
    context_kwargs = {
        "compute_capability": capability,
        "route": "bias_docblock_specialized",
        "seq_len": max(int(query_len), int(key_len)),
        "batch_size": int(batch_size),
        "query_len": int(query_len),
        "key_len": int(key_len),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "dtype": _kernel_dtype_name(dtype),
        "causal": bool(causal),
        "has_mask": True,
    }
    for table_kind in (f"bwd_{normalized_kind}", "bwd"):
        table_config = resolve_flash_kernel_config(
            FlashKernelContext(
                kind=table_kind,
                **context_kwargs,
            )
        )
        if table_config is not None:
            return table_config

    return _resolve_bias_bwd_kernel_config(
        kind=normalized_kind,
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=causal,
        dtype=dtype,
        device=device,
    )


if triton is not None:

    @triton.jit
    def _bwd_kv_kernel_docblock1024(
        Q: None,
        K: None,
        V: None,
        B: None,
        sm_scale: float,
        DO: None,
        DK: None,
        DV: None,
        DS: None,
        DPOS_KEY: None,
        BUCKET: None,
        KEEP_MASK: None,
        L: None,
        D: None,
        stride_qz: int,
        stride_qh: int,
        stride_qm: int,
        stride_qk: int,
        stride_kz: int,
        stride_kh: int,
        stride_kn: int,
        stride_kk: int,
        stride_vz: int,
        stride_vh: int,
        stride_vn: int,
        stride_vk: int,
        stride_bz: int,
        stride_bh: int,
        stride_bm: int,
        stride_bn: int,
        stride_doz: int,
        stride_doh: int,
        stride_dom: int,
        stride_dok: int,
        stride_dkz: int,
        stride_dkh: int,
        stride_dkn: int,
        stride_dkk: int,
        stride_dvz: int,
        stride_dvh: int,
        stride_dvn: int,
        stride_dvk: int,
        stride_dpkz: int,
        stride_dpkh: int,
        stride_dpkm: int,
        stride_dpkp: int,
        stride_bucket_m: int,
        stride_bucket_n: int,
        stride_mask_b: int,
        stride_mask_h: int,
        stride_mask_m: int,
        stride_mask_n: int,
        H: int,
        M: int,
        N: int,
        P: int,
        bias_scale: float,
        HAS_POS_KEY: tl.constexpr,
        HAS_KEEP_MASK: tl.constexpr,
        WRITE_DBIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ) -> None:
        """Specialized dense-bias KV backward for measured non-causal full-bias squares.

        :param Any Q: Triton pointer to the query tensor.
        :param Any K: Triton pointer to the key tensor.
        :param Any V: Triton pointer to the value tensor.
        :param Any B: Triton pointer to the dense additive bias tensor.
        :param float sm_scale: Softmax scale factor.
        :param Any DO: Triton pointer to the output gradient tensor.
        :param Any DK: Triton pointer to the key gradient tensor.
        :param Any DV: Triton pointer to the value gradient tensor.
        :param Any DS: Triton pointer to the dense bias gradient tensor.
        :param Any DPOS_KEY: Triton pointer to the optional c2p positional-gradient tensor.
        :param Any BUCKET: Triton pointer to the dense bucket map.
        :param Any KEEP_MASK: Triton pointer to the optional document keep mask.
        :param Any L: Triton pointer to the per-row log-sum-exp tensor.
        :param Any D: Triton pointer to the per-row delta tensor.
        :param int stride_qz: Query batch stride.
        :param int stride_qh: Query head stride.
        :param int stride_qm: Query row stride.
        :param int stride_qk: Query column stride.
        :param int stride_kz: Key batch stride.
        :param int stride_kh: Key head stride.
        :param int stride_kn: Key row stride.
        :param int stride_kk: Key column stride.
        :param int stride_vz: Value batch stride.
        :param int stride_vh: Value head stride.
        :param int stride_vn: Value row stride.
        :param int stride_vk: Value column stride.
        :param int stride_bz: Bias batch stride.
        :param int stride_bh: Bias head stride.
        :param int stride_bm: Bias row stride.
        :param int stride_bn: Bias column stride.
        :param int stride_doz: Output-gradient batch stride.
        :param int stride_doh: Output-gradient head stride.
        :param int stride_dom: Output-gradient row stride.
        :param int stride_dok: Output-gradient column stride.
        :param int stride_dkz: Key-gradient batch stride.
        :param int stride_dkh: Key-gradient head stride.
        :param int stride_dkn: Key-gradient row stride.
        :param int stride_dkk: Key-gradient column stride.
        :param int stride_dvz: Value-gradient batch stride.
        :param int stride_dvh: Value-gradient head stride.
        :param int stride_dvn: Value-gradient row stride.
        :param int stride_dvk: Value-gradient column stride.
        :param int stride_dpkz: Pos-key gradient batch stride.
        :param int stride_dpkh: Pos-key gradient head stride.
        :param int stride_dpkm: Pos-key gradient sequence stride.
        :param int stride_dpkp: Pos-key gradient bucket stride.
        :param int stride_bucket_m: Bucket-map row stride.
        :param int stride_bucket_n: Bucket-map column stride.
        :param int stride_mask_b: Keep-mask batch stride.
        :param int stride_mask_h: Keep-mask head stride.
        :param int stride_mask_m: Keep-mask row stride.
        :param int stride_mask_n: Keep-mask column stride.
        :param int H: Number of heads.
        :param int M: Query sequence length.
        :param int N: Key sequence length.
        :param int P: Positional bucket width.
        :param float bias_scale: Scale applied to positional-bias gradients.
        :param Any HAS_POS_KEY: Whether to accumulate c2p positional gradients.
        :param Any HAS_KEEP_MASK: Whether the keep mask is active.
        :param Any WRITE_DBIAS: Whether to materialize dense ``d_bias``.
        :param Any BLOCK_M: Query tile height.
        :param Any BLOCK_DMODEL: Head dimension tile width.
        :param Any BLOCK_N: Key tile width.
        :return None: This Triton kernel writes gradients directly to its outputs.
        """

        input_dtype = Q.dtype.element_ty
        start_n = tl.program_id(0)
        off_h = tl.program_id(1)
        off_z = tl.program_id(2)
        log2e: tl.constexpr = 1.4426950408889634

        Q += off_z * stride_qz + off_h * stride_qh
        K += off_z * stride_kz + off_h * stride_kh
        V += off_z * stride_vz + off_h * stride_vh
        B += off_z * stride_bz + off_h * stride_bh
        DO += off_z * stride_doz + off_h * stride_doh
        DK += off_z * stride_dkz + off_h * stride_dkh
        DV += off_z * stride_dvz + off_h * stride_dvh
        DS += off_z * stride_bz + off_h * stride_bh
        D += (off_z * H + off_h) * M
        L += (off_z * H + off_h) * M

        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m_base = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_DMODEL)

        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk)
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk)

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

        for start_m in range(0, M, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m = start_m + offs_m_base
            q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
            do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok)
            bias_ptrs = B + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
            ds_ptrs = DS + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)

            q = tl.load(q_ptrs)
            do = tl.load(do_ptrs)
            lse_row = tl.load(L + offs_m)
            delta = tl.load(D + offs_m)
            b = tl.load(bias_ptrs)

            s = tl.dot(q, tl.trans(k)) * sm_scale
            s += b
            p = tl.math.exp2((s - lse_row[:, None]) * log2e)
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)
            dp = tl.dot(do, tl.trans(v))
            ds = (p * (dp - delta[:, None])).to(input_dtype)
            dk += tl.dot(tl.trans(ds), q)
            if WRITE_DBIAS:
                tl.store(ds_ptrs, ds)
            if HAS_POS_KEY:
                bucket = tl.load(
                    BUCKET + offs_m[:, None] * stride_bucket_m + offs_n[None, :] * stride_bucket_n
                ).to(tl.int32)
                dpos_key_ptrs = (
                    DPOS_KEY
                    + off_z * stride_dpkz
                    + off_h * stride_dpkh
                    + offs_m[:, None] * stride_dpkm
                    + bucket * stride_dpkp
                )
                dpos_mask = bucket < P
                if HAS_KEEP_MASK:
                    # stride_mask_h is normalized at launch: 0 for broadcast
                    # (B,1,S,S) masks, the real stride for per-head masks.
                    keep = tl.load(
                        KEEP_MASK
                        + off_z * stride_mask_b
                        + off_h * stride_mask_h
                        + offs_m[:, None] * stride_mask_m
                        + offs_n[None, :] * stride_mask_n
                    )
                    dpos_mask = dpos_mask & keep.to(tl.int1)
                tl.atomic_add(
                    dpos_key_ptrs,
                    ds.to(tl.float32) * bias_scale,
                    mask=dpos_mask,
                    sem="relaxed",
                    scope="gpu",
                )

        dk *= sm_scale
        tl.store(dk_ptrs, dk.to(input_dtype))
        tl.store(dv_ptrs, dv.to(input_dtype))

    @triton.jit
    def _bwd_q_kernel_docblock1024(
        Q: None,
        K: None,
        V: None,
        B: None,
        sm_scale: float,
        DO: None,
        DQ: None,
        DPOS_QUERY: None,
        BUCKET: None,
        KEEP_MASK: None,
        L: None,
        D: None,
        stride_qz: int,
        stride_qh: int,
        stride_qm: int,
        stride_qk: int,
        stride_kz: int,
        stride_kh: int,
        stride_kn: int,
        stride_kk: int,
        stride_vz: int,
        stride_vh: int,
        stride_vn: int,
        stride_vk: int,
        stride_bz: int,
        stride_bh: int,
        stride_bm: int,
        stride_bn: int,
        stride_doz: int,
        stride_doh: int,
        stride_dom: int,
        stride_dok: int,
        stride_dqz: int,
        stride_dqh: int,
        stride_dqm: int,
        stride_dqk: int,
        stride_dpqz: int,
        stride_dpqh: int,
        stride_dpqm: int,
        stride_dpqp: int,
        stride_bucket_m: int,
        stride_bucket_n: int,
        stride_mask_b: int,
        stride_mask_h: int,
        stride_mask_m: int,
        stride_mask_n: int,
        H: int,
        M: int,
        N: int,
        P: int,
        bias_scale: float,
        HAS_POS_QUERY: tl.constexpr,
        HAS_KEEP_MASK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ) -> None:
        """Specialized dense-bias Q backward for measured non-causal full-bias squares.

        :param Any Q: Triton pointer to the query tensor.
        :param Any K: Triton pointer to the key tensor.
        :param Any V: Triton pointer to the value tensor.
        :param Any B: Triton pointer to the dense additive bias tensor.
        :param float sm_scale: Softmax scale factor.
        :param Any DO: Triton pointer to the output gradient tensor.
        :param Any DQ: Triton pointer to the query gradient tensor.
        :param Any DPOS_QUERY: Triton pointer to the optional p2c positional-gradient tensor.
        :param Any BUCKET: Triton pointer to the dense bucket map.
        :param Any KEEP_MASK: Triton pointer to the optional document keep mask.
        :param Any L: Triton pointer to the per-row log-sum-exp tensor.
        :param Any D: Triton pointer to the per-row delta tensor.
        :param int stride_qz: Query batch stride.
        :param int stride_qh: Query head stride.
        :param int stride_qm: Query row stride.
        :param int stride_qk: Query column stride.
        :param int stride_kz: Key batch stride.
        :param int stride_kh: Key head stride.
        :param int stride_kn: Key row stride.
        :param int stride_kk: Key column stride.
        :param int stride_vz: Value batch stride.
        :param int stride_vh: Value head stride.
        :param int stride_vn: Value row stride.
        :param int stride_vk: Value column stride.
        :param int stride_bz: Bias batch stride.
        :param int stride_bh: Bias head stride.
        :param int stride_bm: Bias row stride.
        :param int stride_bn: Bias column stride.
        :param int stride_doz: Output-gradient batch stride.
        :param int stride_doh: Output-gradient head stride.
        :param int stride_dom: Output-gradient row stride.
        :param int stride_dok: Output-gradient column stride.
        :param int stride_dqz: Query-gradient batch stride.
        :param int stride_dqh: Query-gradient head stride.
        :param int stride_dqm: Query-gradient row stride.
        :param int stride_dqk: Query-gradient column stride.
        :param int stride_dpqz: Pos-query gradient batch stride.
        :param int stride_dpqh: Pos-query gradient head stride.
        :param int stride_dpqm: Pos-query gradient sequence stride.
        :param int stride_dpqp: Pos-query gradient bucket stride.
        :param int stride_bucket_m: Bucket-map row stride.
        :param int stride_bucket_n: Bucket-map column stride.
        :param int stride_mask_b: Keep-mask batch stride.
        :param int stride_mask_h: Keep-mask head stride.
        :param int stride_mask_m: Keep-mask row stride.
        :param int stride_mask_n: Keep-mask column stride.
        :param int H: Number of heads.
        :param int M: Query sequence length.
        :param int N: Key sequence length.
        :param int P: Positional bucket width.
        :param float bias_scale: Scale applied to positional-bias gradients.
        :param Any HAS_POS_QUERY: Whether to accumulate p2c positional gradients.
        :param Any HAS_KEEP_MASK: Whether the keep mask is active.
        :param Any BLOCK_M: Query tile height.
        :param Any BLOCK_DMODEL: Head dimension tile width.
        :param Any BLOCK_N: Key tile width.
        :return None: This Triton kernel writes gradients directly to ``DQ``.
        """

        input_dtype = Q.dtype.element_ty
        start_m = tl.program_id(0)
        off_h = tl.program_id(1)
        off_z = tl.program_id(2)
        log2e: tl.constexpr = 1.4426950408889634

        Q += off_z * stride_qz + off_h * stride_qh
        K += off_z * stride_kz + off_h * stride_kh
        V += off_z * stride_vz + off_h * stride_vh
        B += off_z * stride_bz + off_h * stride_bh
        DO += off_z * stride_doz + off_h * stride_doh
        DQ += off_z * stride_dqz + off_h * stride_dqh
        D += (off_z * H + off_h) * M
        L += (off_z * H + off_h) * M

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n_base = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)

        q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk)
        do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok)

        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        delta = tl.load(D + offs_m)
        lse_row = tl.load(L + offs_m)
        dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + offs_n_base
            k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
            v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
            bias_ptrs = B + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)

            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            b = tl.load(bias_ptrs)

            s = tl.dot(q, tl.trans(k)) * sm_scale
            s += b
            p = tl.math.exp2((s - lse_row[:, None]) * log2e)
            dp = tl.dot(do.to(input_dtype), tl.trans(v))
            ds = (p * (dp - delta[:, None])).to(input_dtype)
            dq += tl.dot(ds, k)
            if HAS_POS_QUERY:
                bucket = tl.load(
                    BUCKET + offs_m[:, None] * stride_bucket_m + offs_n[None, :] * stride_bucket_n
                ).to(tl.int32)
                dpos_query_ptrs = (
                    DPOS_QUERY
                    + off_z * stride_dpqz
                    + off_h * stride_dpqh
                    + offs_n[None, :] * stride_dpqm
                    + bucket * stride_dpqp
                )
                dpos_mask = bucket < P
                if HAS_KEEP_MASK:
                    # stride_mask_h is normalized at launch: 0 for broadcast
                    # (B,1,S,S) masks, the real stride for per-head masks.
                    keep = tl.load(
                        KEEP_MASK
                        + off_z * stride_mask_b
                        + off_h * stride_mask_h
                        + offs_m[:, None] * stride_mask_m
                        + offs_n[None, :] * stride_mask_n
                    )
                    dpos_mask = dpos_mask & keep.to(tl.int1)
                tl.atomic_add(
                    dpos_query_ptrs,
                    ds.to(tl.float32) * bias_scale,
                    mask=dpos_mask,
                    sem="relaxed",
                    scope="gpu",
                )

        dq *= sm_scale
        tl.store(dq_ptrs, dq.to(input_dtype))

else:  # pragma: no cover - optional dependency fallback
    _bwd_kv_kernel_docblock1024 = None
    _bwd_q_kernel_docblock1024 = None


def _resolve_docblock1024_bwd_configs(
    *,
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """Resolve KV/Q launch configs for the specialized docblock1024 backward.

    :param int batch_size: Batch size.
    :param int num_heads: Number of attention heads.
    :param int query_len: Query length.
    :param int key_len: Key length.
    :param int head_dim: Head dimension.
    :param torch.dtype dtype: Input dtype.
    :param torch.device device: CUDA device.
    :return tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
        ``(kv_config, q_config)`` when the specialized kernels can run for this
        shape, otherwise ``None`` (caller falls back to the generic path).
    """

    kv_config = _resolve_docblock_specialized_bwd_kernel_config(
        kind="kv",
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=False,
        dtype=dtype,
        device=device,
    )
    q_config = _resolve_docblock_specialized_bwd_kernel_config(
        kind="q",
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=False,
        dtype=dtype,
        device=device,
    )
    kv_block_m, kv_block_n, _, _ = kv_config
    q_block_m, q_block_n, _, _ = q_config
    if (
        _bwd_kv_kernel_docblock1024 is None
        or _bwd_q_kernel_docblock1024 is None
        or int(head_dim) != 64
        or int(query_len) % int(kv_block_m) != 0
        or int(key_len) % int(kv_block_n) != 0
        or int(query_len) % int(q_block_m) != 0
        or int(key_len) % int(q_block_n) != 0
    ):
        return None
    return kv_config, q_config


def _launch_bias_preprocess(
    *,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    lse: torch.Tensor,
    block_m: int,
) -> torch.Tensor:
    """Launch the shared bias-backward delta preprocessing kernel.

    :param torch.Tensor out: Forward attention output.
    :param torch.Tensor grad_out: Output gradient.
    :param torch.Tensor lse: Forward log-sum-exp tensor used for output shape.
    :param int block_m: Query tile size.
    :return torch.Tensor: Preprocessed delta tensor.
    """

    batch_size, num_heads, query_len, head_dim = out.shape
    grid = (-(-int(query_len) // int(block_m)), int(num_heads), int(batch_size))
    delta = torch.empty_like(lse)
    with torch.cuda.device(out.device.index):
        _bwd_preprocess_bias_raw[grid](
            out,
            grad_out,
            delta,
            *out.stride(),
            *grad_out.stride(),
            *delta.stride(),
            int(query_len),
            BLOCK_M=int(block_m),
            D_HEAD=int(head_dim),
            DIVISIBLE_M=bool(int(query_len) % int(block_m) == 0),
        )
    return delta


def _launch_docblock1024_backward(
    *,
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float,
    kv_config: tuple[int, int, int, int],
    q_config: tuple[int, int, int, int],
    d_bias: torch.Tensor | None,
    dpos_key_accum: torch.Tensor | None,
    dpos_query_accum: torch.Tensor | None,
    bucket_index: torch.Tensor | None,
    keep_mask: torch.Tensor | None,
    bias_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the specialized docblock1024 preprocess/KV/Q backward kernels.

    The two Triton kernels take strictly positional arguments; this launcher
    owns that ordering once. Optional outputs are selected via sentinels:
    ``d_bias`` (when not ``None``) enables the dense-bias write
    (``WRITE_DBIAS``), while ``dpos_key_accum``/``dpos_query_accum`` (float32,
    zero-initialized by the caller) enable fused positional accumulation with
    ``bucket_index``/``keep_mask``/``bias_scale``.

    :param torch.Tensor grad_out: Gradient of the output tensor.
    :param torch.Tensor q: Forward queries.
    :param torch.Tensor k: Forward keys.
    :param torch.Tensor v: Forward values.
    :param torch.Tensor bias: Forward dense additive bias tensor.
    :param torch.Tensor out: Forward output tensor.
    :param torch.Tensor lse: Forward LSE tensor.
    :param float sm_scale: Attention score scale.
    :param tuple[int, int, int, int] kv_config: ``(BLOCK_M, BLOCK_N, stages, warps)`` for KV.
    :param tuple[int, int, int, int] q_config: ``(BLOCK_M, BLOCK_N, stages, warps)`` for Q.
    :param torch.Tensor | None d_bias: Optional dense-bias gradient output.
    :param torch.Tensor | None dpos_key_accum: Optional c2p gradient accumulator.
    :param torch.Tensor | None dpos_query_accum: Optional p2c gradient accumulator.
    :param torch.Tensor | None bucket_index: Bucket map for positional accumulation.
    :param torch.Tensor | None keep_mask: Optional keep mask.
    :param float bias_scale: Scale applied to positional-bias gradients.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Gradients for q/k/v
        (``d_bias``/accumulators are written in place).
    """

    batch_size, num_heads, query_len, head_dim = q.shape
    key_len = int(k.shape[2])
    kv_block_m, kv_block_n, kv_num_stages, kv_num_warps = kv_config
    q_block_m, q_block_n, q_num_stages, q_num_warps = q_config

    delta = _launch_bias_preprocess(
        out=out,
        grad_out=grad_out,
        lse=lse,
        block_m=max(int(kv_block_m), int(q_block_m)),
    )

    empty_float = torch.empty((0,), device=q.device, dtype=torch.float32)
    empty_int = torch.empty((0,), device=q.device, dtype=torch.int32)
    empty_bool = torch.empty((0,), device=q.device, dtype=torch.bool)
    d_bias_tensor = d_bias if d_bias is not None else empty_float
    dpos_key_tensor = dpos_key_accum if dpos_key_accum is not None else empty_float
    dpos_query_tensor = dpos_query_accum if dpos_query_accum is not None else empty_float
    bucket_tensor = bucket_index if bucket_index is not None else empty_int
    keep_mask_tensor = keep_mask if keep_mask is not None else empty_bool
    raw_mask_strides = strides_or_zeros(keep_mask, 4)
    keep_mask_strides = (
        raw_mask_strides[0],
        keep_mask_head_stride(keep_mask, num_heads=int(num_heads)),
        raw_mask_strides[2],
        raw_mask_strides[3],
    )

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    kv_grid = (
        -(-int(key_len) // int(kv_block_n)),
        int(num_heads),
        int(batch_size),
    )
    with torch.cuda.device(q.device.index):
        _bwd_kv_kernel_docblock1024[kv_grid](
            q,
            k,
            v,
            bias,
            float(sm_scale),
            grad_out,
            dk,
            dv,
            d_bias_tensor,
            dpos_key_tensor,
            bucket_tensor,
            keep_mask_tensor,
            lse,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            bias.stride(3),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            grad_out.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            *strides_or_zeros(dpos_key_accum, 4),
            *strides_or_zeros(bucket_index, 2),
            *keep_mask_strides,
            int(num_heads),
            int(query_len),
            int(key_len),
            int(dpos_key_tensor.shape[-1]) if dpos_key_accum is not None else 0,
            float(bias_scale),
            HAS_POS_KEY=dpos_key_accum is not None,
            HAS_KEEP_MASK=keep_mask is not None,
            WRITE_DBIAS=d_bias is not None,
            BLOCK_M=int(kv_block_m),
            BLOCK_DMODEL=int(head_dim),
            BLOCK_N=int(kv_block_n),
            num_stages=int(kv_num_stages),
            num_warps=int(kv_num_warps),
        )

    dq = torch.empty_like(q)
    q_grid = (
        -(-int(query_len) // int(q_block_m)),
        int(num_heads),
        int(batch_size),
    )
    with torch.cuda.device(q.device.index):
        _bwd_q_kernel_docblock1024[q_grid](
            q,
            k,
            v,
            bias,
            float(sm_scale),
            grad_out,
            dq,
            dpos_query_tensor,
            bucket_tensor,
            keep_mask_tensor,
            lse,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            bias.stride(3),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            grad_out.stride(3),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            *strides_or_zeros(dpos_query_accum, 4),
            *strides_or_zeros(bucket_index, 2),
            *keep_mask_strides,
            int(num_heads),
            int(query_len),
            int(key_len),
            int(dpos_query_tensor.shape[-1]) if dpos_query_accum is not None else 0,
            float(bias_scale),
            HAS_POS_QUERY=dpos_query_accum is not None,
            HAS_KEEP_MASK=keep_mask is not None,
            BLOCK_M=int(q_block_m),
            BLOCK_DMODEL=int(head_dim),
            BLOCK_N=int(q_block_n),
            num_stages=int(q_num_stages),
            num_warps=int(q_num_warps),
        )

    return dq, dk, dv


def _bias_specialized_docblock_backward_impl(
    *,
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the repo-local specialized dense-bias backward for short doc-block batches.

    :param torch.Tensor grad_out: Gradient of the output tensor.
    :param torch.Tensor q: Forward queries.
    :param torch.Tensor k: Forward keys.
    :param torch.Tensor v: Forward values.
    :param torch.Tensor bias: Dense additive bias tensor.
    :param torch.Tensor out: Forward output tensor.
    :param torch.Tensor lse: Forward LSE tensor.
    :param float sm_scale: Softmax scale.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Gradients for q/k/v and bias.
    """

    batch_size, num_heads, query_len, head_dim = q.shape
    key_len = int(k.shape[2])
    configs = _resolve_docblock1024_bwd_configs(
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        dtype=q.dtype,
        device=q.device,
    )
    if configs is None:
        return _bias_generic_backward_impl(
            grad_out=grad_out,
            q=q,
            k=k,
            v=v,
            bias=bias,
            out=out,
            lse=lse,
            sm_scale=sm_scale,
            causal=False,
        )

    kv_config, q_config = configs
    d_bias = torch.empty_like(bias)
    dq, dk, dv = _launch_docblock1024_backward(
        grad_out=grad_out,
        q=q,
        k=k,
        v=v,
        bias=bias,
        out=out,
        lse=lse,
        sm_scale=sm_scale,
        kv_config=kv_config,
        q_config=q_config,
        d_bias=d_bias,
        dpos_key_accum=None,
        dpos_query_accum=None,
        bucket_index=None,
        keep_mask=None,
        bias_scale=0.0,
    )
    return dq, dk, dv, d_bias


def _should_use_specialized_docblock_position_bias_backward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    bucket_index: torch.Tensor,
    pos_key_num_buckets: int,
    pos_query_num_buckets: int,
    causal: bool,
) -> bool:
    """Return whether dense doc-block bias backward can accumulate positional grads directly.

    :param torch.Tensor q: Forward query tensor in ``(B,H,S,D)`` layout.
    :param torch.Tensor k: Forward key tensor in ``(B,H,S,D)`` layout.
    :param torch.Tensor v: Forward value tensor in ``(B,H,S,D)`` layout.
    :param torch.Tensor bias: Dense additive bias tensor in ``(B,H,S,S)`` layout.
    :param torch.Tensor bucket_index: Bucket map in ``(S,S)`` layout.
    :param int pos_key_num_buckets: c2p bucket count, or zero when absent.
    :param int pos_query_num_buckets: p2c bucket count, or zero when absent.
    :param bool causal: Whether causal masking is active.
    :return bool: True when the direct positional-gradient specialization should run.
    """

    if not _should_use_specialized_docblock_bias_backward(q=q, k=k, v=v, bias=bias, causal=causal):
        return False
    seq_len = int(q.shape[-2])
    if bucket_index.device.type != "cuda" or tuple(bucket_index.shape) != (seq_len, seq_len):
        return False
    return int(pos_key_num_buckets) > 0 or int(pos_query_num_buckets) > 0


def _position_bias_specialized_docblock_backward_impl(
    *,
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    pos_key_num_buckets: int,
    pos_query_num_buckets: int,
    bucket_index: torch.Tensor,
    keep_mask: torch.Tensor | None,
    bias_scale: float,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run short doc-block dense-bias backward with fused positional-gradient accumulation.

    The generic dense route materializes ``d_bias`` and then reduces it into
    c2p/p2c bucket tensors with PyTorch scatter operations. This narrow
    specialization keeps the same attention backward math but atomically
    accumulates positional gradients from the Triton backward tiles, avoiding
    the full dense ``d_bias`` write plus the separate reduction pass.

    :param torch.Tensor grad_out: Gradient of the output tensor.
    :param torch.Tensor q: Forward queries.
    :param torch.Tensor k: Forward keys.
    :param torch.Tensor v: Forward values.
    :param torch.Tensor bias: Forward dense additive bias tensor.
    :param torch.Tensor out: Forward output tensor.
    :param torch.Tensor lse: Forward LSE tensor.
    :param int pos_key_num_buckets: c2p bucket count, or zero when absent.
    :param int pos_query_num_buckets: p2c bucket count, or zero when absent.
    :param torch.Tensor bucket_index: Dense bucket map.
    :param torch.Tensor | None keep_mask: Optional keep mask.
    :param float bias_scale: Scale applied to positional-bias gradients.
    :param float sm_scale: Attention score scale.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        Gradients for q/k/v and optional c2p/p2c positional tensors.
    """

    batch_size, num_heads, query_len, head_dim = q.shape
    key_len = int(k.shape[2])
    configs = _resolve_docblock1024_bwd_configs(
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        dtype=q.dtype,
        device=q.device,
    )
    if configs is None:
        bias_dq, bias_dk, bias_dv, d_bias = _bias_generic_backward_impl(
            grad_out=grad_out,
            q=q,
            k=k,
            v=v,
            bias=bias,
            out=out,
            lse=lse,
            sm_scale=sm_scale,
            causal=False,
        )
        dpos_key, dpos_query = _position_bias_backward_from_dense_grad(
            d_bias=d_bias,
            pos_key_num_buckets=pos_key_num_buckets,
            pos_query_num_buckets=pos_query_num_buckets,
            bucket_index=bucket_index,
            keep_mask=keep_mask,
            scale=bias_scale,
            output_dtype=q.dtype,
        )
        return bias_dq, bias_dk, bias_dv, dpos_key, dpos_query

    kv_config, q_config = configs
    pos_shape = tuple(int(dim) for dim in q.shape[:3])
    dpos_key_accum = (
        torch.zeros((*pos_shape, int(pos_key_num_buckets)), device=q.device, dtype=torch.float32)
        if int(pos_key_num_buckets) > 0
        else None
    )
    dpos_query_accum = (
        torch.zeros((*pos_shape, int(pos_query_num_buckets)), device=q.device, dtype=torch.float32)
        if int(pos_query_num_buckets) > 0
        else None
    )
    dq, dk, dv = _launch_docblock1024_backward(
        grad_out=grad_out,
        q=q,
        k=k,
        v=v,
        bias=bias,
        out=out,
        lse=lse,
        sm_scale=sm_scale,
        kv_config=kv_config,
        q_config=q_config,
        d_bias=None,
        dpos_key_accum=dpos_key_accum,
        dpos_query_accum=dpos_query_accum,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
        bias_scale=bias_scale,
    )

    return (
        dq,
        dk,
        dv,
        dpos_key_accum.to(dtype=q.dtype) if dpos_key_accum is not None else None,
        dpos_query_accum.to(dtype=q.dtype) if dpos_query_accum is not None else None,
    )


def _bias_eager_forward_impl(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    sm_scale: float,
    causal: bool,
    require_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run dense bias attention through the low-level CUDA launcher.

    :param torch.Tensor q: Queries in ``(B, H, S, D)`` layout.
    :param torch.Tensor k: Keys in ``(B, H, S, D)`` layout.
    :param torch.Tensor v: Values in ``(B, H, S, D)`` layout.
    :param torch.Tensor bias: Additive bias in ``(B, H, S, S)`` layout.
    :param float sm_scale: Softmax scale.
    :param bool causal: Whether causal masking is enabled.
    :param bool require_lse: Whether to return LSE for backward.
    :return tuple[torch.Tensor, torch.Tensor | None]: Output and optional LSE tensor.
    """

    if _flash_attn_v2_fwd_bias_lowlevel is None:
        if _flash_attention_with_bias_highlevel is None:
            raise RuntimeError("FlashDeBERTa local-bias attention is unavailable.")
        out = _flash_attention_with_bias_highlevel(
            q, k, v, bias, causal=bool(causal), sm_scale=float(sm_scale)
        )
        return out, None

    batch_size, num_heads, query_len, head_dim = q.shape
    key_len = int(k.shape[2])
    block_m, block_n, num_stages, num_warps = _bias_config(
        kind="fwd",
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=bool(causal),
        dtype=q.dtype,
        device=q.device,
    )
    out, lse = _flash_attn_v2_fwd_bias_lowlevel(
        q,
        k,
        v,
        bias,
        bool(causal),
        float(sm_scale),
        int(block_m),
        int(block_n),
        int(num_warps),
        int(num_stages),
    )
    if not require_lse:
        return out, None
    return out, lse


def _bias_generic_backward_impl(
    *,
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run dense bias backward through the generic low-level CUDA launcher.

    :param torch.Tensor grad_out: Gradient of the output tensor.
    :param torch.Tensor q: Forward queries.
    :param torch.Tensor k: Forward keys.
    :param torch.Tensor v: Forward values.
    :param torch.Tensor bias: Forward additive bias tensor.
    :param torch.Tensor out: Forward output tensor.
    :param torch.Tensor lse: Forward LSE tensor.
    :param float sm_scale: Softmax scale.
    :param bool causal: Whether causal masking is enabled.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Gradients for q/k/v and the additive bias tensor.
    """

    batch_size, num_heads, query_len, head_dim = q.shape
    key_len = int(k.shape[2])
    if _bwd_preprocess_bias_raw is None or _bwd_kv_kernel_bias_raw is None or _bwd_q_kernel_bias_raw is None:
        block_m, block_n, num_stages, num_warps = _bias_config(
            kind="bwd",
            batch_size=batch_size,
            num_heads=num_heads,
            query_len=query_len,
            key_len=key_len,
            head_dim=head_dim,
            causal=bool(causal),
            dtype=q.dtype,
            device=q.device,
        )
        dq, dk, dv, d_bias = _flash_attn_v2_bwd_bias_lowlevel(
            out,
            grad_out,
            q,
            k,
            v,
            bias,
            lse,
            bool(causal),
            float(sm_scale),
            int(block_m),
            int(block_n),
            int(num_warps),
            int(num_stages),
        )
        return dq, dk, dv, d_bias

    kv_block_m, kv_block_n, kv_num_stages, kv_num_warps = _resolve_bias_bwd_kernel_config(
        kind="kv",
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=bool(causal),
        dtype=q.dtype,
        device=q.device,
    )
    q_block_m, q_block_n, q_num_stages, q_num_warps = _resolve_bias_bwd_kernel_config(
        kind="q",
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=bool(causal),
        dtype=q.dtype,
        device=q.device,
    )

    bias_batch_stride = bias.stride(0)
    bias_heads_stride = bias.stride(1)
    if int(bias.shape[0]) != int(q.shape[0]) and int(bias.shape[0]) == 1:
        bias_batch_stride = 0
    if int(bias.shape[1]) != int(q.shape[1]) and int(bias.shape[1]) == 1:
        bias_heads_stride = 0

    delta = _launch_bias_preprocess(
        out=out,
        grad_out=grad_out,
        lse=lse,
        block_m=max(int(kv_block_m), int(q_block_m)),
    )

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    is_batch_reduced = int(bias_batch_stride) == 0
    group_size_bias = int(batch_size)
    if is_batch_reduced:
        if bool(causal):
            d_bias = torch.zeros((group_size_bias, *bias.shape[1:]), dtype=bias.dtype, device=bias.device)
        else:
            d_bias = torch.empty((group_size_bias, *bias.shape[1:]), dtype=bias.dtype, device=bias.device)
        locks = torch.zeros(2 * group_size_bias, dtype=torch.int32, device=q.device)
    else:
        if bool(causal):
            d_bias = torch.zeros_like(bias)
        else:
            d_bias = torch.empty_like(bias)
        locks = None

    kv_grid = (
        -(-int(key_len) // int(kv_block_n)),
        int(num_heads),
        int(batch_size),
    )
    with torch.cuda.device(q.device.index):
        _bwd_kv_kernel_bias_raw[kv_grid](
            q,
            k,
            v,
            bias,
            float(sm_scale),
            grad_out,
            dk,
            dv,
            d_bias,
            lse,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            bias.stride(0),
            bias_heads_stride,
            bias.stride(2),
            bias.stride(3),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            grad_out.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            int(batch_size),
            int(num_heads),
            int(query_len),
            int(key_len),
            int(key_len - query_len),
            locks,
            BLOCK_M=int(kv_block_m),
            BLOCK_DMODEL=int(head_dim),
            BLOCK_N=int(kv_block_n),
            CAUSAL=bool(causal),
            DIVISIBLE_M=bool(int(query_len) % int(kv_block_m) == 0),
            DIVISIBLE_N=bool(int(key_len) % int(kv_block_n) == 0),
            HAS_BIAS=True,
            RETURN_DS=True,
            IS_BATCH_REDUCED=bool(is_batch_reduced),
            GROUP_SIZE_BIAS=int(group_size_bias),
            num_stages=int(kv_num_stages),
            num_warps=int(kv_num_warps),
        )

    if is_batch_reduced and group_size_bias > 1:
        d_bias = d_bias.sum(0, keepdim=True)

    dq = torch.empty_like(q)
    q_grid = (
        -(-int(query_len) // int(q_block_m)),
        int(num_heads),
        int(batch_size),
    )
    with torch.cuda.device(q.device.index):
        _bwd_q_kernel_bias_raw[q_grid](
            q,
            k,
            v,
            bias,
            float(sm_scale),
            grad_out,
            dq,
            lse,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            bias_batch_stride,
            bias_heads_stride,
            bias.stride(2),
            bias.stride(3),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            grad_out.stride(3),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            int(batch_size),
            int(num_heads),
            int(query_len),
            int(key_len),
            int(key_len - query_len),
            BLOCK_M=int(q_block_m),
            BLOCK_DMODEL=int(head_dim),
            BLOCK_N=int(q_block_n),
            CAUSAL=bool(causal),
            LARGER_M=bool(int(query_len) > int(key_len)),
            DIVISIBLE_M=bool(int(query_len) % int(q_block_m) == 0),
            DIVISIBLE_N=bool(int(key_len) % int(q_block_n) == 0),
            HAS_BIAS=True,
            num_stages=int(q_num_stages),
            num_warps=int(q_num_warps),
        )
    return dq, dk, dv, d_bias


def _bias_eager_backward_impl(
    *,
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run dense bias backward, dispatching to repo-local special cases when applicable.

    :param torch.Tensor grad_out: Gradient of the output tensor.
    :param torch.Tensor q: Forward queries.
    :param torch.Tensor k: Forward keys.
    :param torch.Tensor v: Forward values.
    :param torch.Tensor bias: Forward additive bias tensor.
    :param torch.Tensor out: Forward output tensor.
    :param torch.Tensor lse: Forward LSE tensor.
    :param float sm_scale: Softmax scale.
    :param bool causal: Whether causal masking is enabled.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Gradients for q/k/v and the additive bias tensor.
    """

    if _flash_attn_v2_bwd_bias_lowlevel is None:
        raise RuntimeError("FlashDeBERTa local-bias backward is unavailable.")

    if _should_use_specialized_docblock_bias_backward(q=q, k=k, v=v, bias=bias, causal=causal):
        return _bias_specialized_docblock_backward_impl(
            grad_out=grad_out,
            q=q,
            k=k,
            v=v,
            bias=bias,
            out=out,
            lse=lse,
            sm_scale=sm_scale,
        )

    return _bias_generic_backward_impl(
        grad_out=grad_out,
        q=q,
        k=k,
        v=v,
        bias=bias,
        out=out,
        lse=lse,
        sm_scale=sm_scale,
        causal=causal,
    )


def _position_bias_forward_impl(
    *,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    bucket_index: torch.Tensor,
    keep_mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Materialize dense positional bias for one attention launch.

    :param torch.Tensor | None pos_key: Optional c2p term in ``(B,H,S,P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c term in ``(B,H,S,P)`` layout.
    :param torch.Tensor bucket_index: Dense bucket map in ``(S,S)`` layout.
    :param torch.Tensor | None keep_mask: Optional keep mask in ``(B,1,S,S)`` or per-head ``(B,H,S,S)`` layout.
    :param float scale: Scale applied to the additive position bias.
    :return torch.Tensor: Dense additive bias in ``(B,H,S,S)`` layout.
    """

    reference = pos_key if pos_key is not None else pos_query
    if reference is None:
        raise RuntimeError("FlashDeBERTa position-bias attention requires at least one positional term.")
    if reference.device.type == "cuda" and triton is not None:
        return _dense_bias_forward_cuda(
            pos_key=pos_key,
            pos_query=pos_query,
            bucket_index=bucket_index,
            keep_mask=keep_mask,
            scale=float(scale),
        )
    return _dense_bias_forward_fallback(
        pos_key=pos_key,
        pos_query=pos_query,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
        scale=float(scale),
    )


def _position_bias_backward_from_dense_grad(
    *,
    d_bias: torch.Tensor,
    pos_key_num_buckets: int,
    pos_query_num_buckets: int,
    bucket_index: torch.Tensor,
    keep_mask: torch.Tensor | None,
    scale: float,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Reduce dense additive-bias gradients into positional score tensors.

    :param torch.Tensor d_bias: Dense additive-bias gradient in ``(B,H,S,S)`` layout.
    :param int pos_key_num_buckets: c2p bucket count, or zero when absent.
    :param int pos_query_num_buckets: p2c bucket count, or zero when absent.
    :param torch.Tensor bucket_index: Dense bucket map in ``(S,S)`` layout.
    :param torch.Tensor | None keep_mask: Optional keep mask used by the forward pass.
    :param float scale: Scale applied to the additive position bias in forward.
    :param torch.dtype output_dtype: Positional-gradient output dtype.
    :return tuple[torch.Tensor | None, torch.Tensor | None]: Gradients for c2p and p2c terms.
    """

    grad = d_bias.to(dtype=torch.float32)
    if keep_mask is not None:
        grad = grad.masked_fill(~keep_mask, 0.0)
    grad = grad * float(scale)

    dpos_key: torch.Tensor | None = None
    if int(pos_key_num_buckets) > 0:
        dpos_key = _dense_bucket_reduce(
            grad=grad,
            bucket_index=bucket_index,
            num_buckets=int(pos_key_num_buckets),
            output_dtype=output_dtype,
        )

    dpos_query: torch.Tensor | None = None
    if int(pos_query_num_buckets) > 0:
        # P2C forward reads pos_query[n, bucket_index[m,n]]. In key-major
        # layout both the dense gradient and canonical bucket map transpose.
        dpos_query = _dense_bucket_reduce(
            grad=grad.transpose(-1, -2).contiguous(),
            bucket_index=bucket_index.transpose(0, 1),
            num_buckets=int(pos_query_num_buckets),
            output_dtype=output_dtype,
        )

    return dpos_key, dpos_query


def _build_position_bias_custom_ops() -> tuple[Any | None, Any | None]:
    """Register or retrieve the fused position-bias attention custom ops.

    :return tuple[Any | None, Any | None]: Forward and backward custom-op handles.
    """

    existing = lookup_existing_op_pair(
        _BIAS_OP_NAMESPACE, _POSITION_BIAS_FWD_OP_NAME, _POSITION_BIAS_BWD_OP_NAME
    )
    if existing is not None:
        return existing

    if (
        triton is None
        or _flash_attn_v2_fwd_bias_lowlevel is None
        or _flash_attn_v2_bwd_bias_lowlevel is None
        or not hasattr(torch, "library")
        or not hasattr(torch.library, "custom_op")
    ):
        return None, None

    @torch.library.custom_op(
        f"{_BIAS_OP_NAMESPACE}::{_POSITION_BIAS_FWD_OP_NAME}",
        mutates_args=(),
        device_types="cuda",
        schema=(
            "(Tensor q, Tensor k, Tensor v, Tensor pos_key, Tensor pos_query, Tensor bucket_index, "
            "Tensor keep_mask, float bias_scale, float sm_scale, bool causal, bool has_pos_key, "
            "bool has_pos_query, bool has_keep_mask) -> (Tensor, Tensor, Tensor)"
        ),
    )
    def _forward_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_key: torch.Tensor,
        pos_query: torch.Tensor,
        bucket_index: torch.Tensor,
        keep_mask: torch.Tensor,
        bias_scale: float,
        sm_scale: float,
        causal: bool,
        has_pos_key: bool,
        has_pos_query: bool,
        has_keep_mask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run position-bias attention and return dense bias as autograd-owned aux.

        :param torch.Tensor q: Queries in ``(B,H,S,D)`` layout.
        :param torch.Tensor k: Keys in ``(B,H,S,D)`` layout.
        :param torch.Tensor v: Values in ``(B,H,S,D)`` layout.
        :param torch.Tensor pos_key: c2p tensor or empty sentinel.
        :param torch.Tensor pos_query: p2c tensor or empty sentinel.
        :param torch.Tensor bucket_index: Dense bucket map in ``(S,S)`` layout.
        :param torch.Tensor keep_mask: Keep mask tensor or empty sentinel.
        :param float bias_scale: Scale applied to the additive position bias.
        :param float sm_scale: Softmax scale applied to content scores.
        :param bool causal: Whether causal masking is enabled.
        :param bool has_pos_key: Whether ``pos_key`` is active.
        :param bool has_pos_query: Whether ``pos_query`` is active.
        :param bool has_keep_mask: Whether ``keep_mask`` is active.
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output, LSE, and dense bias aux.
        """

        bias = _position_bias_forward_impl(
            pos_key=pos_key if bool(has_pos_key) else None,
            pos_query=pos_query if bool(has_pos_query) else None,
            bucket_index=bucket_index,
            keep_mask=keep_mask if bool(has_keep_mask) else None,
            scale=float(bias_scale),
        )
        out, lse = _bias_eager_forward_impl(
            q=q,
            k=k,
            v=v,
            bias=bias,
            sm_scale=float(sm_scale),
            causal=bool(causal),
            require_lse=True,
        )
        return out, lse, bias

    @torch.library.register_fake(_forward_op)
    def _forward_op_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_key: torch.Tensor,
        pos_query: torch.Tensor,
        bucket_index: torch.Tensor,
        keep_mask: torch.Tensor,
        bias_scale: float,
        sm_scale: float,
        causal: bool,
        has_pos_key: bool,
        has_pos_query: bool,
        has_keep_mask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fake position-bias attention outputs with static shapes.

        :param torch.Tensor q: Fake query tensor.
        :param torch.Tensor k: Fake key tensor.
        :param torch.Tensor v: Fake value tensor.
        :param torch.Tensor pos_key: Fake c2p tensor or empty sentinel.
        :param torch.Tensor pos_query: Fake p2c tensor or empty sentinel.
        :param torch.Tensor bucket_index: Fake dense bucket map.
        :param torch.Tensor keep_mask: Fake keep mask or empty sentinel.
        :param float bias_scale: Fake position-bias scale.
        :param float sm_scale: Fake softmax scale.
        :param bool causal: Fake causal flag.
        :param bool has_pos_key: Fake c2p presence flag.
        :param bool has_pos_query: Fake p2c presence flag.
        :param bool has_keep_mask: Fake keep-mask presence flag.
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Fake output, LSE, and dense bias aux.
        """

        del v, pos_key, pos_query, bucket_index, keep_mask, bias_scale
        del sm_scale, causal, has_pos_key, has_pos_query, has_keep_mask
        lse = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        bias = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2], k.shape[2]),
            device=q.device,
            dtype=q.dtype,
        )
        return torch.empty(q.shape, device=q.device, dtype=q.dtype), lse, bias

    @torch.library.custom_op(
        f"{_BIAS_OP_NAMESPACE}::{_POSITION_BIAS_BWD_OP_NAME}",
        mutates_args=(),
        device_types="cuda",
        schema=(
            "(Tensor grad_out, Tensor q, Tensor k, Tensor v, Tensor bucket_index, Tensor keep_mask, "
            "Tensor bias, Tensor out, Tensor lse, float bias_scale, float sm_scale, bool causal, "
            "int pos_key_num_buckets, int pos_query_num_buckets, bool has_keep_mask) "
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
        ),
    )
    def _backward_op(
        grad_out: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bucket_index: torch.Tensor,
        keep_mask: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        bias_scale: float,
        sm_scale: float,
        causal: bool,
        pos_key_num_buckets: int,
        pos_query_num_buckets: int,
        has_keep_mask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run position-bias attention backward with saved dense bias.

        :param torch.Tensor grad_out: Gradient of the output tensor.
        :param torch.Tensor q: Forward query tensor.
        :param torch.Tensor k: Forward key tensor.
        :param torch.Tensor v: Forward value tensor.
        :param torch.Tensor bucket_index: Dense bucket map in ``(S,S)`` layout.
        :param torch.Tensor keep_mask: Forward keep mask or empty sentinel.
        :param torch.Tensor bias: Forward dense additive bias tensor.
        :param torch.Tensor out: Forward output tensor.
        :param torch.Tensor lse: Forward LSE tensor.
        :param float bias_scale: Scale applied to the additive position bias.
        :param float sm_scale: Softmax scale applied to content scores.
        :param bool causal: Whether causal masking is enabled.
        :param int pos_key_num_buckets: c2p bucket count, or zero when absent.
        :param int pos_query_num_buckets: p2c bucket count, or zero when absent.
        :param bool has_keep_mask: Whether ``keep_mask`` is active.
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Gradients for q/k/v/pos_key/pos_query.
        """

        keep_mask_tensor = keep_mask if bool(has_keep_mask) else None
        if _should_use_specialized_docblock_position_bias_backward(
            q=q,
            k=k,
            v=v,
            bias=bias,
            bucket_index=bucket_index,
            pos_key_num_buckets=int(pos_key_num_buckets),
            pos_query_num_buckets=int(pos_query_num_buckets),
            causal=bool(causal),
        ):
            dq, dk, dv, dpos_key, dpos_query = _position_bias_specialized_docblock_backward_impl(
                grad_out=grad_out,
                q=q,
                k=k,
                v=v,
                bias=bias,
                out=out,
                lse=lse,
                pos_key_num_buckets=int(pos_key_num_buckets),
                pos_query_num_buckets=int(pos_query_num_buckets),
                bucket_index=bucket_index,
                keep_mask=keep_mask_tensor,
                bias_scale=float(bias_scale),
                sm_scale=float(sm_scale),
            )
        else:
            dq, dk, dv, d_bias = _bias_eager_backward_impl(
                grad_out=grad_out,
                q=q,
                k=k,
                v=v,
                bias=bias,
                out=out,
                lse=lse,
                sm_scale=float(sm_scale),
                causal=bool(causal),
            )
            dpos_key, dpos_query = _position_bias_backward_from_dense_grad(
                d_bias=d_bias,
                pos_key_num_buckets=int(pos_key_num_buckets),
                pos_query_num_buckets=int(pos_query_num_buckets),
                bucket_index=bucket_index,
                keep_mask=keep_mask_tensor,
                scale=float(bias_scale),
                output_dtype=q.dtype,
            )
        return (
            dq,
            dk,
            dv,
            dpos_key if dpos_key is not None else q.new_empty((0,)),
            dpos_query if dpos_query is not None else q.new_empty((0,)),
        )

    @torch.library.register_fake(_backward_op)
    def _backward_op_fake(
        grad_out: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bucket_index: torch.Tensor,
        keep_mask: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        bias_scale: float,
        sm_scale: float,
        causal: bool,
        pos_key_num_buckets: int,
        pos_query_num_buckets: int,
        has_keep_mask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fake position-bias backward outputs with static shapes.

        :param torch.Tensor grad_out: Fake output gradient.
        :param torch.Tensor q: Fake query tensor.
        :param torch.Tensor k: Fake key tensor.
        :param torch.Tensor v: Fake value tensor.
        :param torch.Tensor bucket_index: Fake dense bucket map.
        :param torch.Tensor keep_mask: Fake keep mask or empty sentinel.
        :param torch.Tensor bias: Fake dense additive bias tensor.
        :param torch.Tensor out: Fake output tensor.
        :param torch.Tensor lse: Fake LSE tensor.
        :param float bias_scale: Fake position-bias scale.
        :param float sm_scale: Fake softmax scale.
        :param bool causal: Fake causal flag.
        :param int pos_key_num_buckets: Fake c2p bucket count.
        :param int pos_query_num_buckets: Fake p2c bucket count.
        :param bool has_keep_mask: Fake keep-mask presence flag.
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Fake gradients for q/k/v/pos_key/pos_query.
        """

        del grad_out, bucket_index, keep_mask, bias, out, lse, bias_scale, sm_scale
        del causal, has_keep_mask
        pos_shape = (q.shape[0], q.shape[1], q.shape[2])
        return (
            torch.empty(q.shape, device=q.device, dtype=q.dtype),
            torch.empty(k.shape, device=k.device, dtype=k.dtype),
            torch.empty(v.shape, device=v.device, dtype=v.dtype),
            torch.empty((*pos_shape, pos_key_num_buckets), device=q.device, dtype=q.dtype),
            torch.empty((*pos_shape, pos_query_num_buckets), device=q.device, dtype=q.dtype),
        )

    def _setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Save inputs and forward dense bias needed by backward.

        :param Any ctx: Autograd context object.
        :param tuple[Any, ...] inputs: Forward custom-op inputs.
        :param tuple[torch.Tensor, torch.Tensor, torch.Tensor] output: Forward outputs.
        """

        (
            q,
            k,
            v,
            pos_key,
            pos_query,
            bucket_index,
            keep_mask,
            bias_scale,
            sm_scale,
            causal,
            has_pos_key,
            has_pos_query,
            has_keep_mask,
        ) = inputs
        out, lse, bias = output
        ctx.mark_non_differentiable(lse, bias)
        ctx.save_for_backward(q, k, v, bucket_index, keep_mask, bias, out, lse)
        ctx.bias_scale = float(bias_scale)
        ctx.sm_scale = float(sm_scale)
        ctx.causal = bool(causal)
        ctx.has_pos_key = bool(has_pos_key)
        ctx.has_pos_query = bool(has_pos_query)
        ctx.pos_key_num_buckets = int(pos_key.shape[-1]) if ctx.has_pos_key else 0
        ctx.pos_query_num_buckets = int(pos_query.shape[-1]) if ctx.has_pos_query else 0
        ctx.has_keep_mask = bool(has_keep_mask)

    def _backward(
        ctx: Any,
        grad_out: torch.Tensor | None,
        grad_lse: torch.Tensor | None,
        grad_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Dispatch fused position-bias backward through an opaque CUDA op.

        :param Any ctx: Autograd context populated by ``_setup_context``.
        :param torch.Tensor | None grad_out: Gradient of the output tensor.
        :param torch.Tensor | None grad_lse: Gradient of the LSE output.
        :param torch.Tensor | None grad_bias: Ignored gradient for dense-bias aux output.
        :return tuple[torch.Tensor | None, ...]: Gradients for the forward inputs.
        """

        del grad_lse, grad_bias
        q, k, v, bucket_index, keep_mask, bias, out, lse = ctx.saved_tensors
        grad = grad_out if grad_out is not None else torch.zeros_like(out)
        dq, dk, dv, dpos_key, dpos_query = _backward_op(
            grad,
            q,
            k,
            v,
            bucket_index,
            keep_mask,
            bias,
            out,
            lse,
            ctx.bias_scale,
            ctx.sm_scale,
            ctx.causal,
            ctx.pos_key_num_buckets,
            ctx.pos_query_num_buckets,
            ctx.has_keep_mask,
        )
        return (
            dq,
            dk,
            dv,
            dpos_key if ctx.has_pos_key else None,
            dpos_query if ctx.has_pos_query else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(_forward_op, _backward, setup_context=_setup_context)
    return _forward_op, _backward_op


_FLASHDEBERTA_POSITION_BIAS_CUSTOM_OP, _FLASHDEBERTA_POSITION_BIAS_BWD_CUSTOM_OP = (
    _build_position_bias_custom_ops()
)


def flashdeberta_bias_from_positions(
    *,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    bucket_index: torch.Tensor,
    keep_mask: torch.Tensor | None,
    bias_scale: float,
    sm_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Run local-bias attention from compact position-bias ingredients.

    :param torch.Tensor query_layer: Queries in ``(B,H,S,D)`` layout.
    :param torch.Tensor key_layer: Keys in ``(B,H,S,D)`` layout.
    :param torch.Tensor value_layer: Values in ``(B,H,S,D)`` layout.
    :param torch.Tensor | None pos_key: Optional c2p term in ``(B,H,S,P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c term in ``(B,H,S,P)`` layout.
    :param torch.Tensor bucket_index: Dense bucket map in ``(S,S)`` layout.
    :param torch.Tensor | None keep_mask: Optional keep mask in ``(B,1,S,S)`` or per-head ``(B,H,S,S)`` layout.
    :param float bias_scale: Scale applied to the additive position bias.
    :param float sm_scale: Softmax scale applied to content scores.
    :param bool causal: Whether causal masking is enabled.
    :raises RuntimeError: If both positional terms are missing.
    :return torch.Tensor: Attention output in ``(B,H,S,D)`` layout.
    """

    reference = pos_key if pos_key is not None else pos_query
    if reference is None:
        raise RuntimeError("FlashDeBERTa position-bias attention requires at least one positional term.")
    expected_prefix = tuple(int(dim) for dim in query_layer.shape[:3])
    for name, tensor in (("pos_key", pos_key), ("pos_query", pos_query)):
        if tensor is None:
            continue
        if tensor.ndim != 4 or tuple(int(dim) for dim in tensor.shape[:3]) != expected_prefix:
            raise ValueError(
                f"{name} must have shape (B,H,S,P) matching query_layer; "
                f"got {tuple(tensor.shape)} versus prefix {expected_prefix}."
            )
        if int(tensor.shape[-1]) <= 0:
            raise ValueError(f"{name} must contain at least one position bucket.")
        if tensor.device != query_layer.device or tensor.dtype != query_layer.dtype:
            raise ValueError(
                f"{name} must share query_layer device/dtype; "
                f"got {tensor.device}/{tensor.dtype} versus {query_layer.device}/{query_layer.dtype}."
            )

    if _FLASHDEBERTA_POSITION_BIAS_CUSTOM_OP is not None and query_layer.device.type == "cuda":
        pos_key_tensor = (
            pos_key
            if pos_key is not None
            else torch.empty((0,), device=reference.device, dtype=reference.dtype)
        )
        pos_query_tensor = (
            pos_query
            if pos_query is not None
            else torch.empty((0,), device=reference.device, dtype=reference.dtype)
        )
        keep_mask_tensor = (
            keep_mask
            if keep_mask is not None
            else torch.empty((0,), device=reference.device, dtype=torch.bool)
        )
        output, _, _ = _FLASHDEBERTA_POSITION_BIAS_CUSTOM_OP(
            query_layer,
            key_layer,
            value_layer,
            pos_key_tensor,
            pos_query_tensor,
            bucket_index,
            keep_mask_tensor,
            float(bias_scale),
            float(sm_scale),
            bool(causal),
            pos_key is not None,
            pos_query is not None,
            keep_mask is not None,
        )
        return output

    output, _ = _bias_eager_forward_impl(
        q=query_layer,
        k=key_layer,
        v=value_layer,
        bias=_position_bias_forward_impl(
            pos_key=pos_key,
            pos_query=pos_query,
            bucket_index=bucket_index,
            keep_mask=keep_mask,
            scale=float(bias_scale),
        ),
        sm_scale=sm_scale,
        causal=causal,
        require_lse=False,
    )
    return output


__all__ = [
    "flashdeberta_bias_from_positions",
    "flashdeberta_bias_import_error",
    "flashdeberta_compiled_position_bias_available",
]
