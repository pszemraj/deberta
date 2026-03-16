"""Compile-safe wrappers for fixed-length FlashDeBERTa attention.

FlashDeBERTa's fixed-length path is exposed upstream as a Python autograd
wrapper with cached configuration lookup and Triton launch setup. That works
correctly in eager mode, but it also leaves TorchDynamo tracing through Python
helpers and guarding on the upstream wrapper object.

This module mirrors the varlen integration strategy: it exposes fixed-length
attention through an opaque CUDA custom op backed by the low-level Triton
primitives. CPU and test-only environments fall back to the upstream eager
wrapper so semantics stay unchanged when the low-level pieces are unavailable.
"""

from __future__ import annotations

import os
from typing import Any

import torch

try:
    from flashdeberta.ops.flash_attention import (
        flash_attention_with_disentangled as _flash_attention_with_disentangled_highlevel,
    )

    _FLASH_FIXED_HIGHLEVEL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional import
    _flash_attention_with_disentangled_highlevel = None
    _FLASH_FIXED_HIGHLEVEL_IMPORT_ERROR = exc

try:
    from flashdeberta.ops.flash_attention import (
        _bwd_kv_dise_kernel as _bwd_kv_dise_kernel_raw,
    )
    from flashdeberta.ops.flash_attention import (
        _bwd_preprocess as _bwd_preprocess_raw,
    )
    from flashdeberta.ops.flash_attention import (
        _bwd_q_dise_kernel as _bwd_q_dise_kernel_raw,
    )
    from flashdeberta.ops.flash_attention import (
        _fwd_kernel_deberta_disentangled_attention as _fwd_kernel_dise_raw,
    )
    from flashdeberta.ops.flash_attention import (
        flash_attn_v2_bwd_dise as _flash_attn_v2_bwd_dise_lowlevel,
    )
    from flashdeberta.ops.flash_attention import (
        flash_attn_v2_fwd_dise as _flash_attn_v2_fwd_dise_lowlevel,
    )
    from flashdeberta.ops.flash_attention import (
        get_bwd_config as _get_bwd_config_lowlevel,
    )
    from flashdeberta.ops.flash_attention import (
        get_fwd_config as _get_fwd_config_lowlevel,
    )

    _FLASH_FIXED_LOWLEVEL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional import
    _bwd_kv_dise_kernel_raw = None
    _bwd_preprocess_raw = None
    _bwd_q_dise_kernel_raw = None
    _fwd_kernel_dise_raw = None
    _flash_attn_v2_bwd_dise_lowlevel = None
    _flash_attn_v2_fwd_dise_lowlevel = None
    _get_bwd_config_lowlevel = None
    _get_fwd_config_lowlevel = None
    _FLASH_FIXED_LOWLEVEL_IMPORT_ERROR = exc

_FIXED_OP_NAMESPACE = "deberta"
_FIXED_FWD_OP_NAME = "flashdeberta_fixed"
_FIXED_BWD_OP_NAME = "flashdeberta_fixed_backward"


def flashdeberta_fixed_import_error() -> Exception | None:
    """Return the most relevant import failure for fixed-length support.

    :return Exception | None: Import failure or ``None`` when some fixed path is available.
    """

    if _flash_attention_with_disentangled_highlevel is not None:
        return None
    if _flash_attn_v2_fwd_dise_lowlevel is not None and _flash_attn_v2_bwd_dise_lowlevel is not None:
        return None
    if _FLASH_FIXED_HIGHLEVEL_IMPORT_ERROR is not None:
        return _FLASH_FIXED_HIGHLEVEL_IMPORT_ERROR
    return _FLASH_FIXED_LOWLEVEL_IMPORT_ERROR


def flashdeberta_compiled_fixed_available() -> bool:
    """Return whether the opaque compiled fixed-length CUDA op is available.

    :return bool: True when the custom-op based CUDA path is registered.
    """

    return _FLASHDEBERTA_FIXED_CUSTOM_OP is not None and _FLASHDEBERTA_FIXED_BWD_CUSTOM_OP is not None


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


def _cdiv(a: int, b: int) -> int:
    """Return ceil-division for positive integers.

    :param int a: Dividend.
    :param int b: Divisor.
    :return int: ``ceil(a / b)``.
    """

    return (int(a) + int(b) - 1) // int(b)


def _fixed_attention_span(position_buckets: int, max_relative_distance: int) -> int:
    """Return the relative-position span used by the fixed kernels.

    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :return int: Effective relative-position span.
    """

    return int(position_buckets) if int(position_buckets) > 0 else int(max_relative_distance)


def _fixed_device_capability(device: torch.device) -> tuple[int, int]:
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


def _fixed_repo_tuned_config(
    *,
    kind: str,
    query_len: int,
    key_len: int,
    head_dim: int,
    causal: bool,
    disentangled: bool,
    att_span: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[int, int, int, int] | None:
    """Return repo-local tuned kernel configs for measured hot paths.

    Upstream FlashDeBERTa does not currently special-case ``sm_120``. On the
    repo's dense DeBERTa ``1024 x 1024`` backward regime, the stock
    ``64 x 64`` backward tile materially underperforms a smaller ``16 x 16``
    tile on measured ``sm_120`` hardware. Keep this override narrow and fall
    back to upstream selection everywhere else.

    :param str kind: Either ``"fwd"`` or ``"bwd"``.
    :param int query_len: Query sequence length.
    :param int key_len: Key sequence length.
    :param int head_dim: Attention head dimension.
    :param bool causal: Whether causal masking is enabled.
    :param bool disentangled: Whether c2p/p2c position terms are active.
    :param int att_span: Effective relative-position span.
    :param torch.dtype dtype: Kernel dtype.
    :param torch.device device: CUDA device for the kernel launch.
    :return tuple[int, int, int, int] | None: Tuned ``(BLOCK_M, BLOCK_N, stages, warps)``
        or ``None`` when no repo-local override applies.
    """

    del dtype
    normalized_kind = str(kind).strip().lower()
    if normalized_kind not in {"fwd", "bwd"}:
        return None
    if bool(causal) or not bool(disentangled):
        return None
    if int(head_dim) > 64:
        return None
    if int(query_len) != 1024 or int(key_len) != 1024:
        return None
    if int(att_span) < 128:
        return None
    capability = _fixed_device_capability(device)
    if int(capability[0]) < 12:
        return None
    if normalized_kind == "fwd":
        return (64, 64, 2, 4)
    return (16, 16, 1, 2)


def _fixed_kernel_override_from_env(
    *,
    kind: str,
) -> tuple[int, int, int, int] | None:
    """Return repo-side fixed-kernel overrides when fully specified.

    These overrides intentionally apply only to the repo's fixed-length wrapper.
    They exist so dense fixed-kernel tuning can be done without also perturbing
    the padded-varlen path.

    Supported env vars:
    - ``FLASHDEBERTA_FIXED_FWD_BLOCK_M``
    - ``FLASHDEBERTA_FIXED_FWD_BLOCK_N``
    - ``FLASHDEBERTA_FIXED_FWD_NUM_STAGES``
    - ``FLASHDEBERTA_FIXED_FWD_NUM_WARPS``
    - ``FLASHDEBERTA_FIXED_BWD_BLOCK_M``
    - ``FLASHDEBERTA_FIXED_BWD_BLOCK_N``
    - ``FLASHDEBERTA_FIXED_BWD_NUM_STAGES``
    - ``FLASHDEBERTA_FIXED_BWD_NUM_WARPS``

    :param str kind: Either ``"fwd"`` or ``"bwd"``.
    :return tuple[int, int, int, int] | None: Override ``(BLOCK_M, BLOCK_N, stages, warps)``
        or ``None`` when unset / invalid / incomplete.
    """

    normalized = str(kind).strip().lower()
    if normalized not in {"fwd", "bwd"}:
        raise ValueError(f"Unsupported fixed kernel override kind: {kind!r}")

    prefix = f"FLASHDEBERTA_FIXED_{normalized.upper()}"
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


def _fixed_use_triton_op() -> bool:
    """Return whether the compile-visible Triton fixed op can be registered.

    :return bool: True when raw fixed kernels and ``torch.library.triton_op`` are available.
    """

    return (
        _fwd_kernel_dise_raw is not None
        and _bwd_preprocess_raw is not None
        and _bwd_kv_dise_kernel_raw is not None
        and _bwd_q_dise_kernel_raw is not None
        and hasattr(torch, "library")
        and hasattr(torch.library, "triton_op")
        and hasattr(torch.library, "wrap_triton")
    )


def _materialize_fixed_seq_lengths(
    *,
    seq_lengths: torch.Tensor | None,
    batch_size: int,
    query_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a concrete sequence-length tensor for Triton kernel launches.

    :param torch.Tensor | None seq_lengths: Optional caller-provided sequence lengths.
    :param int batch_size: Batch size.
    :param int query_len: Query sequence length.
    :param torch.device device: Runtime CUDA device.
    :return torch.Tensor: Sequence lengths with dtype ``int32`` on ``device``.
    """

    if seq_lengths is not None:
        return seq_lengths
    return torch.full((batch_size,), int(query_len), dtype=torch.int32, device=device)


def _fixed_forward_config(
    *,
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
    head_dim: int,
    causal: bool,
    position_buckets: int,
    max_relative_distance: int,
    dtype: torch.dtype,
    device: torch.device,
    has_pos: bool,
) -> tuple[int, int, int, int]:
    """Resolve the fixed forward Triton tile config.

    :param int batch_size: Batch size.
    :param int num_heads: Number of attention heads.
    :param int query_len: Query sequence length.
    :param int key_len: Key sequence length.
    :param int head_dim: Per-head hidden size.
    :param bool causal: Whether causal masking is enabled.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param torch.dtype dtype: Activation dtype.
    :param torch.device device: CUDA device.
    :param bool has_pos: Whether any disentangled positional term is active.
    :return tuple[int, int, int, int]: ``(BLOCK_M, BLOCK_N, stages, warps)``.
    """

    att_span = _fixed_attention_span(position_buckets, max_relative_distance)
    override = _fixed_kernel_override_from_env(kind="fwd")
    tuned = _fixed_repo_tuned_config(
        kind="fwd",
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=bool(causal),
        disentangled=bool(has_pos),
        att_span=att_span,
        dtype=dtype,
        device=device,
    )
    if override is not None:
        return override
    if tuned is not None:
        return tuned
    if _get_fwd_config_lowlevel is None:
        raise RuntimeError("FlashDeBERTa fixed forward config helper is unavailable.")
    return _get_fwd_config_lowlevel(
        batch_size,
        num_heads,
        query_len,
        key_len,
        head_dim,
        bool(causal),
        disentangled=bool(has_pos),
        att_span=att_span,
    )


def _fixed_backward_config(
    *,
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
    head_dim: int,
    causal: bool,
    position_buckets: int,
    max_relative_distance: int,
    dtype: torch.dtype,
    device: torch.device,
    has_pos: bool,
) -> tuple[int, int, int, int]:
    """Resolve the fixed backward Triton tile config.

    :param int batch_size: Batch size.
    :param int num_heads: Number of attention heads.
    :param int query_len: Query sequence length.
    :param int key_len: Key sequence length.
    :param int head_dim: Per-head hidden size.
    :param bool causal: Whether causal masking is enabled.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param torch.dtype dtype: Activation dtype.
    :param torch.device device: CUDA device.
    :param bool has_pos: Whether any disentangled positional term is active.
    :return tuple[int, int, int, int]: ``(BLOCK_M, BLOCK_N, stages, warps)``.
    """

    att_span = _fixed_attention_span(position_buckets, max_relative_distance)
    override = _fixed_kernel_override_from_env(kind="bwd")
    tuned = _fixed_repo_tuned_config(
        kind="bwd",
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=bool(causal),
        disentangled=bool(has_pos),
        att_span=att_span,
        dtype=dtype,
        device=device,
    )
    if override is not None:
        return override
    if tuned is not None:
        return tuned
    if _get_bwd_config_lowlevel is None:
        raise RuntimeError("FlashDeBERTa fixed backward config helper is unavailable.")
    return _get_bwd_config_lowlevel(
        batch_size,
        num_heads,
        query_len,
        key_len,
        head_dim,
        bool(causal),
        disentangled=bool(has_pos),
        att_span=att_span,
        dtype=dtype,
    )


def _fixed_eager_forward_impl(
    *,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    seq_lengths: torch.Tensor | None,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
    require_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run fixed-length FlashDeBERTa attention eagerly.

    :param torch.Tensor query_layer: Queries in ``(B, H, S, D)`` layout.
    :param torch.Tensor key_layer: Keys in ``(B, H, S, D)`` layout.
    :param torch.Tensor value_layer: Values in ``(B, H, S, D)`` layout.
    :param torch.Tensor | None seq_lengths: Optional per-example active lengths.
    :param torch.Tensor | None pos_key: Optional c2p tensor.
    :param torch.Tensor | None pos_query: Optional p2c tensor.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :param bool require_lse: Whether padded LSE output is required.
    :raises RuntimeError: If no fixed-length implementation is importable.
    :return tuple[torch.Tensor, torch.Tensor | None]: Output tensor and optional LSE.
    """

    if _flash_attn_v2_fwd_dise_lowlevel is None and _flash_attention_with_disentangled_highlevel is None:
        detail = flashdeberta_fixed_import_error()
        raise RuntimeError(
            "FlashDeBERTa fixed-length attention is unavailable."
            if detail is None
            else f"FlashDeBERTa fixed-length attention is unavailable ({detail})."
        )

    batch_size, num_heads, query_len, head_dim = query_layer.shape
    key_len = int(key_layer.shape[-2])
    if _flash_attn_v2_fwd_dise_lowlevel is not None and _get_fwd_config_lowlevel is not None:
        att_span = _fixed_attention_span(position_buckets, max_relative_distance)
        block_m, block_n, num_stages, num_warps = _fixed_forward_config(
            batch_size=batch_size,
            num_heads=num_heads,
            query_len=query_len,
            key_len=key_len,
            head_dim=head_dim,
            causal=bool(causal),
            position_buckets=position_buckets,
            max_relative_distance=max_relative_distance,
            dtype=query_layer.dtype,
            device=query_layer.device,
            has_pos=(pos_key is not None or pos_query is not None),
        )
        output, lse = _flash_attn_v2_fwd_dise_lowlevel(
            query_layer,
            key_layer,
            value_layer,
            seq_lengths,
            pos_key,
            pos_query,
            bool(causal),
            float(sm_scale),
            block_m,
            block_n,
            int(position_buckets),
            int(max_relative_distance),
            num_warps,
            num_stages,
            att_span,
        )
        return output, lse

    if require_lse:
        raise RuntimeError(
            "Compiled FlashDeBERTa fixed-length attention requires low-level forward primitives with LSE output."
        )

    output = _flash_attention_with_disentangled_highlevel(
        query_layer,
        key_layer,
        value_layer,
        seq_lengths,
        pos_key,
        pos_query,
        bool(causal),
        float(sm_scale),
        int(position_buckets),
        int(max_relative_distance),
    )
    return output, None


def _fixed_eager_backward_impl(
    *,
    grad_output: torch.Tensor,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    seq_lengths: torch.Tensor | None,
    output: torch.Tensor,
    lse: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run fixed-length FlashDeBERTa backward eagerly.

    :param torch.Tensor grad_output: Gradient of the fixed attention output.
    :param torch.Tensor query_layer: Forward queries.
    :param torch.Tensor key_layer: Forward keys.
    :param torch.Tensor value_layer: Forward values.
    :param torch.Tensor | None seq_lengths: Optional per-example active lengths.
    :param torch.Tensor output: Forward output tensor.
    :param torch.Tensor lse: Forward LSE tensor.
    :param torch.Tensor | None pos_key: Optional c2p tensor.
    :param torch.Tensor | None pos_query: Optional p2c tensor.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :raises RuntimeError: If low-level backward primitives are unavailable.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        Gradients for q/k/v and optional positional tensors.
    """

    if _flash_attn_v2_bwd_dise_lowlevel is None or _get_bwd_config_lowlevel is None:
        detail = _FLASH_FIXED_LOWLEVEL_IMPORT_ERROR
        raise RuntimeError(
            "Compiled FlashDeBERTa fixed-length backward is unavailable."
            if detail is None
            else f"Compiled FlashDeBERTa fixed-length backward is unavailable ({detail})."
        )

    batch_size, num_heads, query_len, head_dim = query_layer.shape
    key_len = int(key_layer.shape[-2])
    att_span = _fixed_attention_span(position_buckets, max_relative_distance)
    block_m, block_n, num_stages, num_warps = _fixed_backward_config(
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=bool(causal),
        position_buckets=position_buckets,
        max_relative_distance=max_relative_distance,
        dtype=query_layer.dtype,
        device=query_layer.device,
        has_pos=(pos_key is not None or pos_query is not None),
    )
    return _flash_attn_v2_bwd_dise_lowlevel(
        output,
        grad_output,
        query_layer,
        key_layer,
        value_layer,
        seq_lengths,
        pos_key,
        pos_query,
        lse,
        bool(causal),
        float(sm_scale),
        block_m,
        block_n,
        int(position_buckets),
        int(max_relative_distance),
        num_warps,
        num_stages,
        att_span,
    )


def _fixed_triton_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lengths: torch.Tensor | None,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the raw fixed forward Triton kernel through ``wrap_triton``.

    :param torch.Tensor q: Queries in ``(B, H, S, D)`` layout.
    :param torch.Tensor k: Keys in ``(B, H, S, D)`` layout.
    :param torch.Tensor v: Values in ``(B, H, S, D)`` layout.
    :param torch.Tensor | None seq_lengths: Optional per-example active lengths.
    :param torch.Tensor | None pos_key: Optional c2p tensor.
    :param torch.Tensor | None pos_query: Optional p2c tensor.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :return tuple[torch.Tensor, torch.Tensor]: Output tensor and padded LSE tensor.
    """

    if not _fixed_use_triton_op():
        raise RuntimeError("Fixed FlashDeBERTa Triton op support is unavailable.")

    batch_size = int(q.shape[0])
    num_heads = int(q.shape[1])
    query_len = int(q.shape[2])
    key_len = int(k.shape[2])
    head_dim = int(q.shape[3])
    att_span = _fixed_attention_span(position_buckets, max_relative_distance)
    block_m, block_n, num_stages, num_warps = _fixed_forward_config(
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=bool(causal),
        position_buckets=position_buckets,
        max_relative_distance=max_relative_distance,
        dtype=q.dtype,
        device=q.device,
        has_pos=(pos_key is not None or pos_query is not None),
    )

    seq_lengths_full = _materialize_fixed_seq_lengths(
        seq_lengths=seq_lengths,
        batch_size=batch_size,
        query_len=query_len,
        device=q.device,
    )
    full_length = seq_lengths is None
    if full_length:
        output = torch.empty_like(q)
        lse = torch.empty((batch_size, num_heads, query_len), device=q.device, dtype=torch.float32)
    else:
        output = torch.zeros_like(q)
        lse = torch.zeros((batch_size, num_heads, query_len), device=q.device, dtype=torch.float32)

    if pos_key is not None:
        stride_pk0, stride_pk1, stride_pk2, stride_pk3 = pos_key.stride()
    else:
        stride_pk0 = stride_pk1 = stride_pk2 = stride_pk3 = 0
    if pos_query is not None:
        stride_pq0, stride_pq1, stride_pq2, stride_pq3 = pos_query.stride()
    else:
        stride_pq0 = stride_pq1 = stride_pq2 = stride_pq3 = 0

    grid = (_cdiv(query_len, block_m), num_heads, batch_size)
    torch.library.wrap_triton(_fwd_kernel_dise_raw)[grid](
        q,
        k,
        v,
        pos_key,
        pos_query,
        lse,
        output,
        seq_lengths_full,
        float(sm_scale),
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
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        stride_pk0,
        stride_pk1,
        stride_pk2,
        stride_pk3,
        stride_pq0,
        stride_pq1,
        stride_pq2,
        stride_pq3,
        batch_size,
        num_heads,
        query_len,
        key_len,
        key_len - query_len,
        BLOCK_M=block_m,
        BLOCK_DMODEL=head_dim,
        BLOCK_N=block_n,
        IS_CAUSAL=bool(causal),
        LARGER_M=bool(query_len > key_len),
        DIVISIBLE_M=bool(query_len % block_m == 0),
        DIVISIBLE_N=bool(key_len % block_n == 0),
        HAS_C2P=bool(pos_key is not None),
        HAS_P2C=bool(pos_query is not None),
        ATT_SPAN=att_span,
        NUM_BUCKETS=int(position_buckets),
        MAX_DISTANCE=int(max_relative_distance),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output, lse


def _fixed_triton_backward_impl(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lengths: torch.Tensor | None,
    out: torch.Tensor,
    lse: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Launch the raw fixed backward Triton kernels through ``wrap_triton``.

    :param torch.Tensor grad_out: Gradient of the attention output.
    :param torch.Tensor q: Forward queries.
    :param torch.Tensor k: Forward keys.
    :param torch.Tensor v: Forward values.
    :param torch.Tensor | None seq_lengths: Optional per-example active lengths.
    :param torch.Tensor out: Forward output tensor.
    :param torch.Tensor lse: Forward LSE tensor.
    :param torch.Tensor | None pos_key: Optional c2p tensor.
    :param torch.Tensor | None pos_query: Optional p2c tensor.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        Gradients for q/k/v and optional positional tensors.
    """

    if not _fixed_use_triton_op():
        raise RuntimeError("Fixed FlashDeBERTa Triton op support is unavailable.")

    batch_size = int(q.shape[0])
    num_heads = int(q.shape[1])
    query_len = int(q.shape[2])
    key_len = int(k.shape[2])
    head_dim = int(q.shape[3])
    att_span = _fixed_attention_span(position_buckets, max_relative_distance)
    block_m, block_n, num_stages, num_warps = _fixed_backward_config(
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=bool(causal),
        position_buckets=position_buckets,
        max_relative_distance=max_relative_distance,
        dtype=q.dtype,
        device=q.device,
        has_pos=(pos_key is not None or pos_query is not None),
    )

    seq_lengths_full = _materialize_fixed_seq_lengths(
        seq_lengths=seq_lengths,
        batch_size=batch_size,
        query_len=query_len,
        device=q.device,
    )
    full_length = seq_lengths is None
    if full_length:
        delta = torch.empty_like(lse)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
    else:
        delta = torch.zeros_like(lse)
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
    dk_pos = torch.zeros_like(pos_key) if pos_key is not None else None
    dq_pos = torch.zeros_like(pos_query) if pos_query is not None else None

    if pos_key is not None:
        stride_pk0, stride_pk1, stride_pk2, stride_pk3 = pos_key.stride()
    else:
        stride_pk0 = stride_pk1 = stride_pk2 = stride_pk3 = 0
    if pos_query is not None:
        stride_pq0, stride_pq1, stride_pq2, stride_pq3 = pos_query.stride()
    else:
        stride_pq0 = stride_pq1 = stride_pq2 = stride_pq3 = 0

    grid_delta = (_cdiv(query_len, block_m), num_heads, batch_size)
    torch.library.wrap_triton(_bwd_preprocess_raw)[grid_delta](
        out,
        grad_out,
        delta,
        seq_lengths_full,
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
        query_len,
        BLOCK_M=block_m,
        D_HEAD=head_dim,
        DIVISIBLE_M=bool(query_len % block_m == 0),
    )

    grid_kv = (_cdiv(key_len, block_n), num_heads, batch_size)
    torch.library.wrap_triton(_bwd_kv_dise_kernel_raw)[grid_kv](
        q,
        k,
        v,
        seq_lengths_full,
        pos_key,
        pos_query,
        float(sm_scale),
        grad_out,
        dk,
        dv,
        dq_pos,
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
        stride_pk0,
        stride_pk1,
        stride_pk2,
        stride_pk3,
        stride_pq0,
        stride_pq1,
        stride_pq2,
        stride_pq3,
        batch_size,
        num_heads,
        query_len,
        key_len,
        key_len - query_len,
        BLOCK_M=block_m,
        BLOCK_DMODEL=head_dim,
        BLOCK_N=block_n,
        CAUSAL=bool(causal),
        HAS_C2P=bool(pos_key is not None),
        HAS_P2C=bool(pos_query is not None),
        DIVISIBLE_M=bool(query_len % block_m == 0),
        DIVISIBLE_N=bool(key_len % block_n == 0),
        ATT_SPAN=att_span,
        NUM_BUCKETS=int(position_buckets),
        MAX_DISTANCE=int(max_relative_distance),
        num_warps=num_warps,
        num_stages=num_stages,
    )

    grid_q = (_cdiv(query_len, block_m), num_heads, batch_size)
    torch.library.wrap_triton(_bwd_q_dise_kernel_raw)[grid_q](
        q,
        k,
        v,
        seq_lengths_full,
        pos_key,
        pos_query,
        float(sm_scale),
        grad_out,
        dq,
        dk_pos,
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
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        grad_out.stride(3),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        stride_pk0,
        stride_pk1,
        stride_pk2,
        stride_pk3,
        stride_pq0,
        stride_pq1,
        stride_pq2,
        stride_pq3,
        batch_size,
        num_heads,
        query_len,
        key_len,
        key_len - query_len,
        BLOCK_M=block_m,
        BLOCK_DMODEL=head_dim,
        BLOCK_N=block_n,
        CAUSAL=bool(causal),
        HAS_C2P=bool(pos_key is not None),
        HAS_P2C=bool(pos_query is not None),
        LARGER_M=bool(query_len > key_len),
        DIVISIBLE_M=bool(query_len % block_m == 0),
        DIVISIBLE_N=bool(key_len % block_n == 0),
        ATT_SPAN=att_span,
        NUM_BUCKETS=int(position_buckets),
        MAX_DISTANCE=int(max_relative_distance),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dq, dk, dv, dk_pos, dq_pos


def _build_fixed_triton_ops() -> tuple[Any | None, Any | None]:
    """Register or retrieve the compile-visible fixed-length Triton ops.

    :return tuple[Any | None, Any | None]: Forward and backward custom-op handles.
    """

    existing_forward = _lookup_registered_op(_FIXED_OP_NAMESPACE, _FIXED_FWD_OP_NAME)
    existing_backward = _lookup_registered_op(_FIXED_OP_NAMESPACE, _FIXED_BWD_OP_NAME)
    if existing_forward is not None and existing_backward is not None:
        return existing_forward, existing_backward

    if not _fixed_use_triton_op():
        return None, None

    @torch.library.triton_op(
        f"{_FIXED_OP_NAMESPACE}::{_FIXED_FWD_OP_NAME}",
        mutates_args=(),
        schema=(
            "(Tensor q, Tensor k, Tensor v, Tensor? seq_lengths, Tensor? pos_key, Tensor? pos_query, "
            "float sm_scale, int position_buckets, int max_relative_distance, bool causal) -> (Tensor, Tensor)"
        ),
    )
    def _forward_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_lengths: torch.Tensor | None,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run fixed-length attention as one opaque CUDA op.

        :param torch.Tensor q: Padded queries in ``(B, H, S, D)`` layout.
        :param torch.Tensor k: Padded keys in ``(B, H, S, D)`` layout.
        :param torch.Tensor v: Padded values in ``(B, H, S, D)`` layout.
        :param torch.Tensor | None seq_lengths: Optional per-example active lengths.
        :param torch.Tensor | None pos_key: Optional c2p tensor.
        :param torch.Tensor | None pos_query: Optional p2c tensor.
        :param float sm_scale: Softmax scale.
        :param int position_buckets: Relative-position bucket count.
        :param int max_relative_distance: Maximum relative distance.
        :param bool causal: Whether causal masking is enabled.
        :return tuple[torch.Tensor, torch.Tensor]: Fixed-length output and LSE tensors.
        """

        return _fixed_triton_forward_impl(
            q=q,
            k=k,
            v=v,
            seq_lengths=seq_lengths,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            position_buckets=position_buckets,
            max_relative_distance=max_relative_distance,
            causal=causal,
        )

    @torch.library.triton_op(
        f"{_FIXED_OP_NAMESPACE}::{_FIXED_BWD_OP_NAME}",
        mutates_args=(),
        schema=(
            "(Tensor grad_out, Tensor q, Tensor k, Tensor v, Tensor? seq_lengths, Tensor out, Tensor lse, "
            "Tensor? pos_key, Tensor? pos_query, float sm_scale, int position_buckets, "
            "int max_relative_distance, bool causal) -> (Tensor, Tensor, Tensor, Tensor?, Tensor?)"
        ),
    )
    def _backward_op(
        grad_out: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_lengths: torch.Tensor | None,
        out: torch.Tensor,
        lse: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run fixed-length backward as one compile-visible Triton op.

        :param torch.Tensor grad_out: Gradient of the fixed output tensor.
        :param torch.Tensor q: Forward queries.
        :param torch.Tensor k: Forward keys.
        :param torch.Tensor v: Forward values.
        :param torch.Tensor | None seq_lengths: Optional per-example active lengths.
        :param torch.Tensor out: Forward output tensor.
        :param torch.Tensor lse: Forward LSE tensor.
        :param torch.Tensor | None pos_key: Optional c2p tensor.
        :param torch.Tensor | None pos_query: Optional p2c tensor.
        :param float sm_scale: Softmax scale.
        :param int position_buckets: Relative-position bucket count.
        :param int max_relative_distance: Maximum relative distance.
        :param bool causal: Whether causal masking is enabled.
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Gradients for q/k/v and positional tensors, with empty tensor sentinels
            when the corresponding positional input is absent.
        """

        dq, dk, dv, dpos_key, dpos_query = _fixed_triton_backward_impl(
            grad_out=grad_out,
            q=q,
            k=k,
            v=v,
            seq_lengths=seq_lengths,
            out=out,
            lse=lse,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            position_buckets=position_buckets,
            max_relative_distance=max_relative_distance,
            causal=causal,
        )
        if dpos_key is None:
            dpos_key = q.new_empty((0,))
        if dpos_query is None:
            dpos_query = q.new_empty((0,))
        return dq, dk, dv, dpos_key, dpos_query

    def _setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Save tensors needed by the fixed-length backward helper.

        :param Any ctx: Autograd context object.
        :param tuple[Any, ...] inputs: Forward custom-op inputs.
        :param tuple[torch.Tensor, torch.Tensor] output: Forward custom-op outputs.
        """

        (
            q,
            k,
            v,
            seq_lengths,
            pos_key,
            pos_query,
            sm_scale,
            position_buckets,
            max_relative_distance,
            causal,
        ) = inputs
        out, lse = output
        saved: list[torch.Tensor] = [q, k, v, out, lse]
        if seq_lengths is not None:
            saved.append(seq_lengths)
        if pos_key is not None:
            saved.append(pos_key)
        if pos_query is not None:
            saved.append(pos_query)
        ctx.has_seq_lengths = seq_lengths is not None
        ctx.has_pos_key = pos_key is not None
        ctx.has_pos_query = pos_query is not None
        ctx.save_for_backward(*saved)
        ctx.sm_scale = float(sm_scale)
        ctx.position_buckets = int(position_buckets)
        ctx.max_relative_distance = int(max_relative_distance)
        ctx.causal = bool(causal)

    def _backward(
        ctx: Any,
        grad_out: torch.Tensor | None,
        grad_lse: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Dispatch backward through the opaque fixed-length helper.

        :param Any ctx: Autograd context populated by ``_setup_context``.
        :param torch.Tensor | None grad_out: Gradient of the fixed output tensor.
        :param torch.Tensor | None grad_lse: Gradient of the fixed LSE tensor.
        :return tuple[torch.Tensor | None, ...]: Gradients for the custom-op inputs.
        """

        del grad_lse
        saved = list(ctx.saved_tensors)
        q, k, v, out, lse = saved[:5]
        next_idx = 5
        seq_lengths = saved[next_idx] if bool(ctx.has_seq_lengths) else None
        if bool(ctx.has_seq_lengths):
            next_idx += 1
        pos_key = saved[next_idx] if bool(ctx.has_pos_key) else None
        if bool(ctx.has_pos_key):
            next_idx += 1
        pos_query = saved[next_idx] if bool(ctx.has_pos_query) else None
        grad = grad_out if grad_out is not None else torch.zeros_like(out)
        dq, dk, dv, dpos_key, dpos_query = _backward_op(
            grad,
            q,
            k,
            v,
            seq_lengths,
            out,
            lse,
            pos_key,
            pos_query,
            ctx.sm_scale,
            ctx.position_buckets,
            ctx.max_relative_distance,
            ctx.causal,
        )
        if not bool(ctx.has_pos_key):
            dpos_key = None
        if not bool(ctx.has_pos_query):
            dpos_query = None
        return dq, dk, dv, None, dpos_key, dpos_query, None, None, None, None

    torch.library.register_autograd(_forward_op, _backward, setup_context=_setup_context)
    return _forward_op, _backward_op


_FLASHDEBERTA_FIXED_CUSTOM_OP, _FLASHDEBERTA_FIXED_BWD_CUSTOM_OP = _build_fixed_triton_ops()


def flashdeberta_fixed(
    *,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    seq_lengths: torch.Tensor | None,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
) -> torch.Tensor:
    """Run fixed-length FlashDeBERTa attention.

    On CUDA with the low-level FlashDeBERTa primitives available, this uses an
    opaque custom-op path so ``torch.compile`` does not trace through the
    upstream Python autograd wrapper. Otherwise it falls back to the upstream
    eager implementation.

    :param torch.Tensor query_layer: Queries in ``(B, H, S, D)`` layout.
    :param torch.Tensor key_layer: Keys in ``(B, H, S, D)`` layout.
    :param torch.Tensor value_layer: Values in ``(B, H, S, D)`` layout.
    :param torch.Tensor | None seq_lengths: Optional per-example active lengths.
    :param torch.Tensor | None pos_key: Optional c2p tensor.
    :param torch.Tensor | None pos_query: Optional p2c tensor.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :return torch.Tensor: Attention output in ``(B, H, S, D)`` layout.
    """

    if _FLASHDEBERTA_FIXED_CUSTOM_OP is not None and query_layer.device.type == "cuda":
        output, _ = _FLASHDEBERTA_FIXED_CUSTOM_OP(
            query_layer,
            key_layer,
            value_layer,
            seq_lengths,
            pos_key,
            pos_query,
            float(sm_scale),
            int(position_buckets),
            int(max_relative_distance),
            bool(causal),
        )
        return output

    output, _ = _fixed_eager_forward_impl(
        query_layer=query_layer,
        key_layer=key_layer,
        value_layer=value_layer,
        seq_lengths=seq_lengths,
        pos_key=pos_key,
        pos_query=pos_query,
        sm_scale=sm_scale,
        position_buckets=position_buckets,
        max_relative_distance=max_relative_distance,
        causal=causal,
        require_lse=False,
    )
    return output


__all__ = [
    "flashdeberta_compiled_fixed_available",
    "flashdeberta_fixed",
    "flashdeberta_fixed_import_error",
    "_fixed_kernel_override_from_env",
    "_fixed_repo_tuned_config",
]
