"""Compile-safe wrappers for dense FlashDeBERTa local-bias attention.

Dense short-sequence DeBERTa can be faster when the disentangled relative
position terms are materialized as an additive bias matrix and dispatched
through FlashDeBERTa's standard flash-with-bias kernels. This module exposes
that path as an opaque custom op so ``torch.compile`` does not trace through
the upstream Python autograd wrapper.
"""

from __future__ import annotations

import os
from typing import Any

import torch

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
    from flashdeberta.ops.flash_attention_bias import (
        get_bwd_config as _get_bwd_config_bias_lowlevel,
    )
    from flashdeberta.ops.flash_attention_bias import (
        get_fwd_config as _get_fwd_config_bias_lowlevel,
    )

    _FLASH_BIAS_LOWLEVEL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional import
    _bwd_kv_kernel_bias_raw = None
    _bwd_preprocess_bias_raw = None
    _bwd_q_kernel_bias_raw = None
    _flash_attn_v2_bwd_bias_lowlevel = None
    _flash_attn_v2_fwd_bias_lowlevel = None
    _get_bwd_config_bias_lowlevel = None
    _get_fwd_config_bias_lowlevel = None
    _FLASH_BIAS_LOWLEVEL_IMPORT_ERROR = exc

_BIAS_OP_NAMESPACE = "deberta"
_BIAS_FWD_OP_NAME = "flashdeberta_bias"
_BIAS_BWD_OP_NAME = "flashdeberta_bias_backward"


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


def flashdeberta_compiled_bias_available() -> bool:
    """Return whether the opaque CUDA local-bias op is available.

    :return bool: True when the custom-op based CUDA path is registered.
    """

    return _FLASHDEBERTA_BIAS_CUSTOM_OP is not None and _FLASHDEBERTA_BIAS_BWD_CUSTOM_OP is not None


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


def _bias_device_capability(device: torch.device) -> tuple[int, int]:
    """Return CUDA device capability for the given tensor device.

    :param torch.device device: CUDA device to query.
    :return tuple[int, int]: ``(major, minor)`` compute capability.
    """

    if device.type != "cuda":
        return (0, 0)
    index = device.index
    if index is None:
        return torch.cuda.get_device_capability()
    return torch.cuda.get_device_capability(index)


def _bias_kernel_override_from_env(*, kind: str) -> tuple[int, int, int, int] | None:
    """Return repo-side dense-bias kernel overrides when fully specified.

    These overrides intentionally apply only to the repo's dense flash-with-bias
    wrapper so doc-block tuning does not perturb the fixed or varlen paths.

    Supported env vars:
    - ``FLASHDEBERTA_BIAS_FWD_BLOCK_M``
    - ``FLASHDEBERTA_BIAS_FWD_BLOCK_N``
    - ``FLASHDEBERTA_BIAS_FWD_NUM_STAGES``
    - ``FLASHDEBERTA_BIAS_FWD_NUM_WARPS``
    - ``FLASHDEBERTA_BIAS_BWD_BLOCK_M``
    - ``FLASHDEBERTA_BIAS_BWD_BLOCK_N``
    - ``FLASHDEBERTA_BIAS_BWD_NUM_STAGES``
    - ``FLASHDEBERTA_BIAS_BWD_NUM_WARPS``
    - ``FLASHDEBERTA_BIAS_BWD_KV_BLOCK_M``
    - ``FLASHDEBERTA_BIAS_BWD_KV_BLOCK_N``
    - ``FLASHDEBERTA_BIAS_BWD_KV_NUM_STAGES``
    - ``FLASHDEBERTA_BIAS_BWD_KV_NUM_WARPS``
    - ``FLASHDEBERTA_BIAS_BWD_Q_BLOCK_M``
    - ``FLASHDEBERTA_BIAS_BWD_Q_BLOCK_N``
    - ``FLASHDEBERTA_BIAS_BWD_Q_NUM_STAGES``
    - ``FLASHDEBERTA_BIAS_BWD_Q_NUM_WARPS``

    :param str kind: One of ``"fwd"``, ``"bwd"``, ``"bwd_kv"``, or ``"bwd_q"``.
    :return tuple[int, int, int, int] | None: Override ``(BLOCK_M, BLOCK_N, stages, warps)``
        or ``None`` when unset / invalid / incomplete.
    """

    normalized = str(kind).strip().lower()
    if normalized not in {"fwd", "bwd", "bwd_kv", "bwd_q"}:
        raise ValueError(f"Unsupported bias kernel override kind: {kind!r}")

    prefix = f"FLASHDEBERTA_BIAS_{normalized.upper()}"
    names = (
        f"{prefix}_BLOCK_M",
        f"{prefix}_BLOCK_N",
        f"{prefix}_NUM_STAGES",
        f"{prefix}_NUM_WARPS",
    )
    raw = [os.environ.get(name) for name in names]
    if any(value is None or not str(value).strip() for value in raw):
        return None
    try:
        block_m, block_n, num_stages, num_warps = (int(str(value).strip()) for value in raw)
    except Exception:
        return None
    return int(block_m), int(block_n), int(num_stages), int(num_warps)


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

    This helper stays intentionally narrow. Only promote configs that have been
    measured on real packed doc-block RTD batches for the repo's current
    DeBERTa ``1024 x 1024`` non-causal bf16 regime on ``sm_120`` hardware.

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

    del batch_size, num_heads
    normalized_kind = str(kind).strip().lower()
    if normalized_kind not in {"fwd", "bwd", "bwd_kv", "bwd_q"}:
        return None
    if bool(causal):
        return None
    if dtype not in {torch.float16, torch.bfloat16}:
        return None
    if int(head_dim) > 64:
        return None
    if int(query_len) != 1024 or int(key_len) != 1024:
        return None
    capability = _bias_device_capability(device)
    if int(capability[0]) < 12:
        return None
    return None


def _bias_forward_config(
    *,
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[int, int, int, int]:
    """Resolve the dense-bias forward Triton tile config.

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

    override = _bias_kernel_override_from_env(kind="fwd")
    tuned = _bias_repo_tuned_config(
        kind="fwd",
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=bool(causal),
        dtype=dtype,
        device=device,
    )
    if override is not None:
        return override
    if tuned is not None:
        return tuned
    if _get_fwd_config_bias_lowlevel is None:
        raise RuntimeError("FlashDeBERTa local-bias config helper is unavailable.")
    return _get_fwd_config_bias_lowlevel(
        batch_size,
        num_heads,
        query_len,
        key_len,
        head_dim,
        bool(causal),
    )


def _bias_backward_config(
    *,
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[int, int, int, int]:
    """Resolve the dense-bias backward Triton tile config.

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

    override = _bias_kernel_override_from_env(kind="bwd")
    tuned = _bias_repo_tuned_config(
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
    if override is not None:
        return override
    if tuned is not None:
        return tuned
    if _get_bwd_config_bias_lowlevel is None:
        raise RuntimeError("FlashDeBERTa local-bias backward is unavailable.")
    return _get_bwd_config_bias_lowlevel(
        batch_size,
        num_heads,
        query_len,
        key_len,
        head_dim,
        bool(causal),
    )


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

    specific_override = _bias_kernel_override_from_env(kind=f"bwd_{normalized_kind}")
    if specific_override is not None:
        return specific_override

    generic_override = _bias_kernel_override_from_env(kind="bwd")
    if generic_override is not None:
        return generic_override

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

    return _bias_backward_config(
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

    This path is intentionally narrow: it targets the measured packed doc-block
    RTD hot path where FlashDeBERTa already wins overall but still spends most
    of its remaining attention time in the generic local-bias backward kernels.

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
    if int(q.shape[-2]) != 1024 or int(k.shape[-2]) != 1024 or int(v.shape[-2]) != 1024:
        return False
    if tuple(q.shape[:2]) != tuple(k.shape[:2]) or tuple(q.shape[:2]) != tuple(v.shape[:2]):
        return False
    if tuple(bias.shape) != (int(q.shape[0]), int(q.shape[1]), 1024, 1024):
        return False
    if int(bias.stride(0)) == 0 or int(bias.stride(1)) == 0:
        return False
    return True


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
        H: int,
        M: int,
        N: int,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ) -> None:
        """Specialized dense-bias KV backward for non-causal full-bias ``1024 x 1024``.

        :param Any Q: Triton pointer to the query tensor.
        :param Any K: Triton pointer to the key tensor.
        :param Any V: Triton pointer to the value tensor.
        :param Any B: Triton pointer to the dense additive bias tensor.
        :param float sm_scale: Softmax scale factor.
        :param Any DO: Triton pointer to the output gradient tensor.
        :param Any DK: Triton pointer to the key gradient tensor.
        :param Any DV: Triton pointer to the value gradient tensor.
        :param Any DS: Triton pointer to the dense bias gradient tensor.
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
        :param int H: Number of heads.
        :param int M: Query sequence length.
        :param int N: Key sequence length.
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
            tl.store(ds_ptrs, ds)

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
        H: int,
        M: int,
        N: int,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ) -> None:
        """Specialized dense-bias Q backward for non-causal full-bias ``1024 x 1024``.

        :param Any Q: Triton pointer to the query tensor.
        :param Any K: Triton pointer to the key tensor.
        :param Any V: Triton pointer to the value tensor.
        :param Any B: Triton pointer to the dense additive bias tensor.
        :param float sm_scale: Softmax scale factor.
        :param Any DO: Triton pointer to the output gradient tensor.
        :param Any DQ: Triton pointer to the query gradient tensor.
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
        :param int H: Number of heads.
        :param int M: Query sequence length.
        :param int N: Key sequence length.
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

        dq *= sm_scale
        tl.store(dq_ptrs, dq.to(input_dtype))

else:  # pragma: no cover - optional dependency fallback
    _bwd_kv_kernel_docblock1024 = None
    _bwd_q_kernel_docblock1024 = None


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
    kv_block_m, kv_block_n, kv_num_stages, kv_num_warps = _resolve_bias_bwd_kernel_config(
        kind="kv",
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=False,
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
        causal=False,
        dtype=q.dtype,
        device=q.device,
    )
    if (
        _bwd_kv_kernel_docblock1024 is None
        or _bwd_q_kernel_docblock1024 is None
        or int(query_len) != 1024
        or int(key_len) != 1024
        or int(head_dim) != 64
        or int(query_len) % int(kv_block_m) != 0
        or int(key_len) % int(kv_block_n) != 0
        or int(query_len) % int(q_block_m) != 0
        or int(key_len) % int(q_block_n) != 0
    ):
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

    preprocess_block_m = max(int(kv_block_m), int(q_block_m))
    preprocess_grid = (
        -(-int(query_len) // int(preprocess_block_m)),
        int(num_heads),
        int(batch_size),
    )
    delta = torch.empty_like(lse)
    with torch.cuda.device(q.device.index):
        _bwd_preprocess_bias_raw[preprocess_grid](
            out,
            grad_out,
            delta,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            grad_out.stride(3),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            int(query_len),
            BLOCK_M=int(preprocess_block_m),
            D_HEAD=int(head_dim),
            DIVISIBLE_M=bool(int(query_len) % int(preprocess_block_m) == 0),
        )

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    d_bias = torch.empty_like(bias)
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
            int(num_heads),
            int(query_len),
            int(key_len),
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
            int(num_heads),
            int(query_len),
            int(key_len),
            BLOCK_M=int(q_block_m),
            BLOCK_DMODEL=int(head_dim),
            BLOCK_N=int(q_block_n),
            num_stages=int(q_num_stages),
            num_warps=int(q_num_warps),
        )
    return dq, dk, dv, d_bias


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
    block_m, block_n, num_stages, num_warps = _bias_forward_config(
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
        block_m, block_n, num_stages, num_warps = _bias_backward_config(
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

    preprocess_block_m = max(int(kv_block_m), int(q_block_m))
    preprocess_grid = (
        -(-int(query_len) // int(preprocess_block_m)),
        int(num_heads),
        int(batch_size),
    )
    delta = torch.empty_like(lse)
    with torch.cuda.device(q.device.index):
        _bwd_preprocess_bias_raw[preprocess_grid](
            out,
            grad_out,
            delta,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            grad_out.stride(3),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            int(query_len),
            BLOCK_M=int(preprocess_block_m),
            D_HEAD=int(head_dim),
            DIVISIBLE_M=bool(int(query_len) % int(preprocess_block_m) == 0),
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


def _build_bias_custom_ops() -> tuple[Any | None, Any | None]:
    """Register or retrieve the opaque local-bias custom ops.

    :return tuple[Any | None, Any | None]: Forward and backward custom-op handles.
    """

    existing_forward = _lookup_registered_op(_BIAS_OP_NAMESPACE, _BIAS_FWD_OP_NAME)
    existing_backward = _lookup_registered_op(_BIAS_OP_NAMESPACE, _BIAS_BWD_OP_NAME)
    if existing_forward is not None and existing_backward is not None:
        return existing_forward, existing_backward

    if (
        _flash_attn_v2_fwd_bias_lowlevel is None
        or _flash_attn_v2_bwd_bias_lowlevel is None
        or not hasattr(torch, "library")
        or not hasattr(torch.library, "custom_op")
    ):
        return None, None

    @torch.library.custom_op(
        f"{_BIAS_OP_NAMESPACE}::{_BIAS_FWD_OP_NAME}",
        mutates_args=(),
        device_types="cuda",
        schema="(Tensor q, Tensor k, Tensor v, Tensor bias, float sm_scale, bool causal) -> (Tensor, Tensor)",
    )
    def _forward_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        sm_scale: float,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run dense bias forward as one opaque CUDA op.

        :param torch.Tensor q: Queries in ``(B, H, S, D)`` layout.
        :param torch.Tensor k: Keys in ``(B, H, S, D)`` layout.
        :param torch.Tensor v: Values in ``(B, H, S, D)`` layout.
        :param torch.Tensor bias: Additive bias in ``(B, H, S, S)`` layout.
        :param float sm_scale: Softmax scale.
        :param bool causal: Whether causal masking is enabled.
        :return tuple[torch.Tensor, torch.Tensor]: Output and LSE tensors.
        """

        return _bias_eager_forward_impl(
            q=q,
            k=k,
            v=v,
            bias=bias,
            sm_scale=sm_scale,
            causal=causal,
            require_lse=True,
        )

    @torch.library.register_fake(_forward_op)
    def _forward_op_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        sm_scale: float,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fake dense bias outputs with static shapes.

        :param torch.Tensor q: Fake query tensor.
        :param torch.Tensor k: Fake key tensor.
        :param torch.Tensor v: Fake value tensor.
        :param torch.Tensor bias: Fake bias tensor.
        :param float sm_scale: Fake softmax scale.
        :param bool causal: Fake causal flag.
        :return tuple[torch.Tensor, torch.Tensor]: Fake output and LSE tensors.
        """

        del k, v, bias, sm_scale, causal
        lse = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        return torch.empty(q.shape, device=q.device, dtype=q.dtype), lse

    @torch.library.custom_op(
        f"{_BIAS_OP_NAMESPACE}::{_BIAS_BWD_OP_NAME}",
        mutates_args=(),
        device_types="cuda",
        schema=(
            "(Tensor grad_out, Tensor q, Tensor k, Tensor v, Tensor bias, Tensor out, Tensor lse, "
            "float sm_scale, bool causal) -> (Tensor, Tensor, Tensor, Tensor)"
        ),
    )
    def _backward_op(
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
        """Run dense bias backward as one opaque CUDA op.

        :param torch.Tensor grad_out: Gradient of the dense output.
        :param torch.Tensor q: Forward query tensor.
        :param torch.Tensor k: Forward key tensor.
        :param torch.Tensor v: Forward value tensor.
        :param torch.Tensor bias: Forward additive bias tensor.
        :param torch.Tensor out: Forward output tensor.
        :param torch.Tensor lse: Forward LSE tensor.
        :param float sm_scale: Softmax scale.
        :param bool causal: Whether causal masking is enabled.
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Gradients for q/k/v and bias.
        """

        return _bias_eager_backward_impl(
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

    @torch.library.register_fake(_backward_op)
    def _backward_op_fake(
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
        """Return fake dense bias backward outputs with static shapes.

        :param torch.Tensor grad_out: Fake gradient tensor.
        :param torch.Tensor q: Fake query tensor.
        :param torch.Tensor k: Fake key tensor.
        :param torch.Tensor v: Fake value tensor.
        :param torch.Tensor bias: Fake bias tensor.
        :param torch.Tensor out: Fake output tensor.
        :param torch.Tensor lse: Fake LSE tensor.
        :param float sm_scale: Fake softmax scale.
        :param bool causal: Fake causal flag.
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Fake gradients for q/k/v and bias.
        """

        del grad_out, out, lse, sm_scale, causal
        return (
            torch.empty(q.shape, device=q.device, dtype=q.dtype),
            torch.empty(k.shape, device=k.device, dtype=k.dtype),
            torch.empty(v.shape, device=v.device, dtype=v.dtype),
            torch.empty(bias.shape, device=bias.device, dtype=bias.dtype),
        )

    def _setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Save tensors needed by the dense bias backward helper.

        :param Any ctx: Autograd context object.
        :param tuple[Any, ...] inputs: Forward custom-op inputs.
        :param tuple[torch.Tensor, torch.Tensor] output: Forward outputs.
        """

        q, k, v, bias, sm_scale, causal = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, bias, out, lse)
        ctx.sm_scale = float(sm_scale)
        ctx.causal = bool(causal)

    def _backward(
        ctx: Any,
        grad_out: torch.Tensor | None,
        grad_lse: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Dispatch backward through the opaque dense bias helper.

        :param Any ctx: Autograd context populated by ``_setup_context``.
        :param torch.Tensor | None grad_out: Gradient of the output tensor.
        :param torch.Tensor | None grad_lse: Gradient of the LSE output.
        :return tuple[torch.Tensor | None, ...]: Gradients for the forward inputs.
        """

        del grad_lse
        q, k, v, bias, out, lse = ctx.saved_tensors
        grad = grad_out if grad_out is not None else torch.zeros_like(out)
        dq, dk, dv, d_bias = _backward_op(
            grad,
            q,
            k,
            v,
            bias,
            out,
            lse,
            ctx.sm_scale,
            ctx.causal,
        )
        return dq, dk, dv, d_bias, None, None

    torch.library.register_autograd(_forward_op, _backward, setup_context=_setup_context)
    return _forward_op, _backward_op


_FLASHDEBERTA_BIAS_CUSTOM_OP, _FLASHDEBERTA_BIAS_BWD_CUSTOM_OP = _build_bias_custom_ops()


def flashdeberta_bias(
    *,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    bias: torch.Tensor,
    sm_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Run FlashDeBERTa local-bias attention.

    :param torch.Tensor query_layer: Queries in ``(B, H, S, D)`` layout.
    :param torch.Tensor key_layer: Keys in ``(B, H, S, D)`` layout.
    :param torch.Tensor value_layer: Values in ``(B, H, S, D)`` layout.
    :param torch.Tensor bias: Additive bias in ``(B, H, S, S)`` layout.
    :param float sm_scale: Softmax scale.
    :param bool causal: Whether causal masking is enabled.
    :return torch.Tensor: Attention output in ``(B, H, S, D)`` layout.
    """

    if _FLASHDEBERTA_BIAS_CUSTOM_OP is not None and query_layer.device.type == "cuda":
        output, _ = _FLASHDEBERTA_BIAS_CUSTOM_OP(
            query_layer,
            key_layer,
            value_layer,
            bias,
            float(sm_scale),
            bool(causal),
        )
        return output

    out, _ = _bias_eager_forward_impl(
        q=query_layer,
        k=key_layer,
        v=value_layer,
        bias=bias,
        sm_scale=sm_scale,
        causal=causal,
        require_lse=False,
    )
    return out


__all__ = [
    "flashdeberta_bias",
    "flashdeberta_bias_import_error",
    "flashdeberta_compiled_bias_available",
]
