"""Compile-safe wrappers for padded FlashDeBERTa varlen attention.

FlashDeBERTa's upstream varlen path is implemented as a Python/Triton wrapper
that TorchDynamo tries to trace through. That wrapper contains CPU-side cache
construction and Triton launch setup which is fine in eager mode, but it is a
poor match for ``torch.compile``.

This module wraps the padded varlen path as an opaque custom operator on CUDA.
The operator boundary hides the Python/Triton launcher details from Dynamo
while preserving real varlen execution and gradients. CPU and test-only paths
fall back to an eager Python implementation.
"""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

try:  # pragma: no cover - optional Triton dependency
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional Triton dependency
    triton = None
    tl = None

from deberta.modeling.flashdeberta_prefix_pack import (
    prefix_pack_padded_rows,
    prefix_pack_padded_rows_pair,
    prefix_pack_padded_rows_triple,
    prefix_unpack_padded_rows,
    prefix_unpack_padded_rows_pair,
    prefix_unpack_padded_rows_triple,
)

try:
    from flashdeberta.ops.flash_attention_varlen import (
        flash_attention_with_disentangled_varlen as _flash_attention_with_disentangled_varlen_highlevel,
    )

    _FLASH_VARLEN_HIGHLEVEL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional import
    _flash_attention_with_disentangled_varlen_highlevel = None
    _FLASH_VARLEN_HIGHLEVEL_IMPORT_ERROR = exc

try:
    import flashdeberta.ops.flash_attention_varlen as _flash_attention_varlen_module
    from flashdeberta.ops.flash_attention_varlen import (
        _bwd_kv_dise_kernel_varlen as _bwd_kv_dise_kernel_varlen_raw,
    )
    from flashdeberta.ops.flash_attention_varlen import (
        _bwd_preprocess_varlen as _bwd_preprocess_varlen_raw,
    )
    from flashdeberta.ops.flash_attention_varlen import (
        _bwd_q_dise_kernel_varlen as _bwd_q_dise_kernel_varlen_raw,
    )
    from flashdeberta.ops.flash_attention_varlen import (
        _fwd_kernel_deberta_disentangled_attention as _fwd_kernel_varlen_raw,
    )
    from flashdeberta.ops.flash_attention_varlen import (
        flash_attn_v2_bwd_dise_varlen as _flash_attn_v2_bwd_dise_varlen_lowlevel,
    )
    from flashdeberta.ops.flash_attention_varlen import (
        flash_attn_v2_fwd_dise as _flash_attn_v2_fwd_dise_lowlevel,
    )
    from flashdeberta.ops.flash_attention_varlen import (
        get_bwd_config_varlen as _get_bwd_config_varlen_lowlevel,
    )
    from flashdeberta.ops.flash_attention_varlen import (
        get_fwd_config as _get_fwd_config_lowlevel,
    )

    _FLASH_VARLEN_LOWLEVEL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional import
    _flash_attention_varlen_module = None
    _bwd_kv_dise_kernel_varlen_raw = None
    _bwd_preprocess_varlen_raw = None
    _bwd_q_dise_kernel_varlen_raw = None
    _fwd_kernel_varlen_raw = None
    _flash_attn_v2_bwd_dise_varlen_lowlevel = None
    _flash_attn_v2_fwd_dise_lowlevel = None
    _get_bwd_config_varlen_lowlevel = None
    _get_fwd_config_lowlevel = None
    _FLASH_VARLEN_LOWLEVEL_IMPORT_ERROR = exc

_VARLEN_OP_NAMESPACE = "deberta"
_VARLEN_FWD_OP_NAME = "flashdeberta_varlen_padded"
_VARLEN_BWD_OP_NAME = "flashdeberta_varlen_padded_backward"
_GRAD_PACK_DELTA_BLOCK_ROWS = 32


@dataclass
class _MaskMetadataCacheEntry:
    """Cached unpadding metadata for one padding-mask tensor."""

    mask_ref: weakref.ReferenceType[torch.Tensor] | None
    version: int
    seqlens: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    total_tokens: int
    cu_seqlens_host: tuple[int, ...]


@dataclass
class _CuSeqlensHostCacheEntry:
    """Cached host tuple for one cumulative-seqlens tensor."""

    cu_ref: weakref.ReferenceType[torch.Tensor] | None
    cu_seqlens_host: tuple[int, ...]


@dataclass
class _MidTensorCacheEntry:
    """Cached varlen tile-metadata tensors for one cumulative-seqlens tensor."""

    cu_ref: weakref.ReferenceType[torch.Tensor] | None
    block_m: int
    device_type: str
    device_index: int | None
    mid_batch: torch.Tensor
    mid_start: torch.Tensor
    mn: int


@dataclass
class _ForwardAuxCacheEntry:
    """Forward-side varlen tensors reused by the padded backward helper."""

    output_ref: weakref.ReferenceType[torch.Tensor] | None
    seqlens: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    total_tokens: int
    q_unpad: torch.Tensor
    k_unpad: torch.Tensor
    v_unpad: torch.Tensor
    out_unpad: torch.Tensor
    lse_unpad: torch.Tensor
    pos_key_unpad: torch.Tensor | None
    pos_query_unpad: torch.Tensor | None


_MASK_METADATA_CACHE: dict[
    tuple[int, int, tuple[int, ...], tuple[int, ...], str, int], _MaskMetadataCacheEntry
] = {}
_CU_SEQLENS_HOST_CACHE: dict[int, _CuSeqlensHostCacheEntry] = {}
_MID_TENSOR_CACHE: dict[tuple[int, int, str, int | None], _MidTensorCacheEntry] = {}
_FORWARD_AUX_CACHE: dict[int, _ForwardAuxCacheEntry] = {}


def flashdeberta_varlen_import_error() -> Exception | None:
    """Return the most relevant import failure for varlen support.

    :return Exception | None: Import failure or ``None`` when some varlen path is available.
    """

    if _flash_attention_with_disentangled_varlen_highlevel is not None:
        return None
    if _FLASH_VARLEN_HIGHLEVEL_IMPORT_ERROR is not None:
        return _FLASH_VARLEN_HIGHLEVEL_IMPORT_ERROR
    return _FLASH_VARLEN_LOWLEVEL_IMPORT_ERROR


def flashdeberta_compiled_varlen_available() -> bool:
    """Return whether the compile-visible padded-varlen Triton op is available.

    :return bool: True when the Triton-op based CUDA path is registered.
    """

    return _FLASHDEBERTA_VARLEN_TRITON_OP is not None and _FLASHDEBERTA_VARLEN_TRITON_BWD_OP is not None


def _is_torch_compiling() -> bool:
    """Return whether execution is happening inside ``torch.compile``.

    :return bool: True when a compiled/traced graph is active.
    """

    if not hasattr(torch, "compiler") or not hasattr(torch.compiler, "is_compiling"):
        return False
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


def _varlen_use_triton_op() -> bool:
    """Return whether the compile-visible Triton varlen path is available.

    :return bool: True when raw varlen kernels and ``torch.library.triton_op`` are available.
    """

    return (
        _fwd_kernel_varlen_raw is not None
        and _bwd_preprocess_varlen_raw is not None
        and _bwd_kv_dise_kernel_varlen_raw is not None
        and _bwd_q_dise_kernel_varlen_raw is not None
        and hasattr(torch, "library")
        and hasattr(torch.library, "triton_op")
        and hasattr(torch.library, "wrap_triton")
    )


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


def _varlen_kernel_override_from_env(
    *,
    kind: str,
) -> tuple[int, int, int, int] | None:
    """Return repo-side varlen-only kernel overrides when fully specified.

    These overrides intentionally apply only to the custom padded-varlen path.
    They exist because upstream ``FLASHDEBERTA_{FWD,BWD}_*`` knobs affect both
    fixed and varlen kernels, which makes tuning padded workloads awkward in
    mixed dense+masked runs.

    Supported env vars:
    - ``FLASHDEBERTA_VARLEN_FWD_BLOCK_M``
    - ``FLASHDEBERTA_VARLEN_FWD_BLOCK_N``
    - ``FLASHDEBERTA_VARLEN_FWD_NUM_STAGES``
    - ``FLASHDEBERTA_VARLEN_FWD_NUM_WARPS``
    - ``FLASHDEBERTA_VARLEN_BWD_BLOCK_M``
    - ``FLASHDEBERTA_VARLEN_BWD_BLOCK_N``
    - ``FLASHDEBERTA_VARLEN_BWD_NUM_STAGES``
    - ``FLASHDEBERTA_VARLEN_BWD_NUM_WARPS``
    - ``FLASHDEBERTA_VARLEN_BWD_KV_BLOCK_M``
    - ``FLASHDEBERTA_VARLEN_BWD_KV_BLOCK_N``
    - ``FLASHDEBERTA_VARLEN_BWD_KV_NUM_STAGES``
    - ``FLASHDEBERTA_VARLEN_BWD_KV_NUM_WARPS``
    - ``FLASHDEBERTA_VARLEN_BWD_Q_BLOCK_M``
    - ``FLASHDEBERTA_VARLEN_BWD_Q_BLOCK_N``
    - ``FLASHDEBERTA_VARLEN_BWD_Q_NUM_STAGES``
    - ``FLASHDEBERTA_VARLEN_BWD_Q_NUM_WARPS``

    :param str kind: One of ``"fwd"``, ``"bwd"``, ``"bwd_kv"``, or ``"bwd_q"``.
    :return tuple[int, int, int, int] | None: Override ``(BLOCK_M, BLOCK_N, stages, warps)``
        or ``None`` when unset / invalid / incomplete.
    """

    normalized = str(kind).strip().lower()
    if normalized not in {"fwd", "bwd", "bwd_kv", "bwd_q"}:
        raise ValueError(f"Unsupported varlen kernel override kind: {kind!r}")

    prefix = f"FLASHDEBERTA_VARLEN_{normalized.upper()}"
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


def _varlen_device_capability(device: torch.device) -> tuple[int, int]:
    """Return CUDA device capability for one device.

    :param torch.device device: CUDA device to query.
    :return tuple[int, int]: ``(major, minor)`` compute capability.
    """

    if device.type != "cuda":
        return (0, 0)
    index = device.index
    if index is None:
        return torch.cuda.get_device_capability()
    return torch.cuda.get_device_capability(index)


def _varlen_density_bucket(*, seq_len: int, total_tokens: int, batch_size: int) -> str:
    """Return the repo-local density bucket for one padded-varlen batch.

    :param int seq_len: Padded sequence length.
    :param int total_tokens: Total active tokens across the batch.
    :param int batch_size: Batch size.
    :return str: Density bucket label used by repo-local heuristics.
    """

    capacity = max(1, int(seq_len) * max(1, int(batch_size)))
    density = float(total_tokens) / float(capacity)
    if int(seq_len) >= 4096:
        return "4096_plus"
    if int(seq_len) >= 2048:
        return "2048_medium" if density >= 0.60 else "2048_sparse"
    return "1024_dense_or_medium"


def _varlen_repo_tuned_bwd_config(
    *,
    kind: str,
    seq_len: int,
    total_tokens: int,
    batch_size: int,
    head_dim: int,
    causal: bool,
    disentangled: bool,
    att_span: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[int, int, int, int] | None:
    """Return repo-local backward-kernel heuristics for measured hot paths.

    :param str kind: Either ``"kv"`` or ``"q"``.
    :param int seq_len: Padded sequence length.
    :param int total_tokens: Total active tokens in the packed batch.
    :param int batch_size: Batch size.
    :param int head_dim: Attention head dimension.
    :param bool causal: Whether causal masking is enabled.
    :param bool disentangled: Whether c2p/p2c position terms are active.
    :param int att_span: Effective relative-position span.
    :param torch.dtype dtype: Kernel dtype.
    :param torch.device device: Launch device.
    :return tuple[int, int, int, int] | None: Tuned ``(BLOCK_M, BLOCK_N, stages, warps)``
        or ``None`` when no repo-local override applies.
    """

    del dtype
    normalized_kind = str(kind).strip().lower()
    if normalized_kind not in {"kv", "q"}:
        return None
    if bool(causal) or not bool(disentangled):
        return None
    if int(head_dim) != 64:
        return None
    if int(att_span) < 128:
        return None
    capability = _varlen_device_capability(device)
    if int(capability[0]) < 12:
        return None
    bucket = _varlen_density_bucket(
        seq_len=int(seq_len), total_tokens=int(total_tokens), batch_size=int(batch_size)
    )
    tuned: dict[tuple[str, str], tuple[int, int, int, int]] = {
        # Measured on sampled unpacked HF DeBERTa RTD batches on sm_120.
        ("kv", "2048_medium"): (64, 32, 2, 4),
        ("kv", "2048_sparse"): (64, 32, 2, 4),
        ("kv", "4096_plus"): (32, 64, 2, 4),
        ("q", "2048_medium"): (32, 64, 2, 4),
        ("q", "2048_sparse"): (32, 64, 2, 4),
        ("q", "4096_plus"): (64, 64, 3, 8),
    }
    return tuned.get((normalized_kind, bucket))


def _resolve_varlen_bwd_kernel_config(
    *,
    kind: str,
    total_tokens_q: int,
    total_tokens_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    batch_size: int,
    head_dim: int,
    causal: bool,
    disentangled: bool,
    att_span: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[int, int, int, int]:
    """Resolve one repo-local varlen backward kernel config.

    :param str kind: Either ``"kv"`` or ``"q"``.
    :param int total_tokens_q: Total query tokens.
    :param int total_tokens_k: Total key tokens.
    :param int max_seqlen_q: Maximum query sequence length.
    :param int max_seqlen_k: Maximum key sequence length.
    :param int batch_size: Batch size.
    :param int head_dim: Attention head dimension.
    :param bool causal: Whether causal masking is enabled.
    :param bool disentangled: Whether c2p/p2c position terms are active.
    :param int att_span: Effective relative-position span.
    :param torch.dtype dtype: Kernel dtype.
    :param torch.device device: Launch device.
    :return tuple[int, int, int, int]: Resolved ``(BLOCK_M, BLOCK_N, stages, warps)``.
    """

    specific_override = _varlen_kernel_override_from_env(kind=f"bwd_{kind}")
    if specific_override is not None:
        return specific_override

    generic_override = _varlen_kernel_override_from_env(kind="bwd")
    if generic_override is not None:
        return generic_override

    repo_tuned = _varlen_repo_tuned_bwd_config(
        kind=kind,
        seq_len=max(max_seqlen_q, max_seqlen_k),
        total_tokens=max(total_tokens_q, total_tokens_k),
        batch_size=batch_size,
        head_dim=head_dim,
        causal=causal,
        disentangled=disentangled,
        att_span=att_span,
        dtype=dtype,
        device=device,
    )
    if repo_tuned is not None:
        return repo_tuned

    if _get_bwd_config_varlen_lowlevel is None:
        raise RuntimeError("FlashDeBERTa varlen backward config helper is unavailable.")
    return _get_bwd_config_varlen_lowlevel(
        total_tokens_q=int(total_tokens_q),
        total_tokens_k=int(total_tokens_k),
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        D=int(head_dim),
        causal=bool(causal),
        disentangled=bool(disentangled),
        att_span=int(att_span),
        dtype=dtype,
    )


def _build_unpad_metadata(
    mask_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int, tuple[int, ...]]:
    """Build prefix-padding metadata for a 2D keep mask.

    :param torch.Tensor mask_2d: Boolean mask with shape ``(B, S)``.
    :return tuple[torch.Tensor, torch.Tensor, int, int, tuple[int, ...]]:
        Per-example sequence lengths, cumulative lengths, max sequence length,
        total active tokens, and host cumulative lengths.
    """

    seqlens = mask_2d.sum(dim=-1, dtype=torch.int32)
    seqlens_host = tuple(int(value) for value in seqlens.detach().cpu().tolist())
    max_seqlen = max(seqlens_host, default=0)
    total_tokens = sum(seqlens_host)
    cu_seqlens_host_list = [0]
    for seqlen in seqlens_host:
        cu_seqlens_host_list.append(int(cu_seqlens_host_list[-1]) + int(seqlen))
    cu_seqlens_host = tuple(int(value) for value in cu_seqlens_host_list)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return seqlens, cu_seqlens, max_seqlen, total_tokens, cu_seqlens_host


def _mask_version(mask_2d: torch.Tensor) -> int:
    """Return a best-effort mutation version for one mask tensor.

    :param torch.Tensor mask_2d: Boolean padding mask.
    :return int: Tensor version counter when available.
    """

    try:
        return int(getattr(mask_2d, "_version", 0))
    except Exception:
        return 0


def _mask_metadata_cache_key(
    mask_2d: torch.Tensor,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...], str, int]:
    """Return a storage-stable cache key for one padding-mask tensor.

    The encoder often passes fresh view objects of the same broadcast padding
    mask to many attention layers. Keying only by ``id(mask)`` misses those
    cases; keying by storage pointer + offset + layout + version preserves cache
    reuse across equivalent views without synchronizing on tensor contents.

    :param torch.Tensor mask_2d: Boolean keep mask with shape ``(B, S)``.
    :return tuple[int, int, tuple[int, ...], tuple[int, ...], str, int]:
        Storage pointer, storage offset, shape, stride, device text, and version.
    """

    storage = mask_2d.untyped_storage()
    return (
        int(storage.data_ptr()),
        int(mask_2d.storage_offset()),
        tuple(int(dim) for dim in mask_2d.shape),
        tuple(int(dim) for dim in mask_2d.stride()),
        str(mask_2d.device),
        _mask_version(mask_2d),
    )


def _clear_unpad_metadata_cache() -> None:
    """Clear cached unpadding metadata.

    This exists primarily for tests.
    """

    _MASK_METADATA_CACHE.clear()
    _CU_SEQLENS_HOST_CACHE.clear()
    _MID_TENSOR_CACHE.clear()


def _clear_forward_aux_cache() -> None:
    """Clear cached forward tensors reused by the padded backward helper.

    This exists primarily for tests.
    """

    _FORWARD_AUX_CACHE.clear()


def _clear_mid_tensor_cache() -> None:
    """Clear cached varlen mid tensors.

    This exists primarily for tests.
    """

    _MID_TENSOR_CACHE.clear()


def _register_cu_seqlens_host_tuple(
    *,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: tuple[int, ...],
) -> None:
    """Register a host tuple for one cumulative-seqlens tensor.

    :param torch.Tensor cu_seqlens: Device cumulative-seqlens tensor.
    :param tuple[int, ...] cu_seqlens_host: Host cumulative lengths.
    """

    cache_key = id(cu_seqlens)
    cu_ref: weakref.ReferenceType[torch.Tensor] | None = None
    try:

        def _cleanup(_ref: object, key: int = cache_key) -> None:
            """Remove cached host and mid tensors when ``cu_seqlens`` is released.

            :param object _ref: Weakref callback payload from the released tensor.
            :param int key: Cache key associated with the released tensor.
            :return None: This callback mutates the module-local caches in place.
            """

            _CU_SEQLENS_HOST_CACHE.pop(key, None)
            stale_keys = [mid_key for mid_key in _MID_TENSOR_CACHE if mid_key[0] == key]
            for stale_key in stale_keys:
                _MID_TENSOR_CACHE.pop(stale_key, None)

        cu_ref = weakref.ref(cu_seqlens, _cleanup)
    except TypeError:
        cu_ref = None

    _CU_SEQLENS_HOST_CACHE[cache_key] = _CuSeqlensHostCacheEntry(
        cu_ref=cu_ref,
        cu_seqlens_host=tuple(int(value) for value in cu_seqlens_host),
    )


def _cu_seqlens_host_tuple(cu_seqlens: torch.Tensor) -> tuple[int, ...]:
    """Return a cached host copy of one cumulative-seqlens tensor.

    :param torch.Tensor cu_seqlens: Device cumulative-seqlens tensor.
    :return tuple[int, ...]: Host cumulative lengths.
    """

    cache_key = id(cu_seqlens)
    cached = _CU_SEQLENS_HOST_CACHE.get(cache_key)
    if cached is not None:
        cached_tensor = cached.cu_ref() if cached.cu_ref is not None else None
        if cached_tensor is cu_seqlens:
            return cached.cu_seqlens_host
        _CU_SEQLENS_HOST_CACHE.pop(cache_key, None)
        stale_keys = [mid_key for mid_key in _MID_TENSOR_CACHE if mid_key[0] == cache_key]
        for stale_key in stale_keys:
            _MID_TENSOR_CACHE.pop(stale_key, None)

    cu_seqlens_host = tuple(int(value) for value in cu_seqlens.detach().cpu().tolist())
    _register_cu_seqlens_host_tuple(cu_seqlens=cu_seqlens, cu_seqlens_host=cu_seqlens_host)
    return cu_seqlens_host


def _build_mid_host_tuples(
    *,
    cu_seqlens_host: tuple[int, ...],
    block_m: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Build host-side varlen tile metadata for one cumulative-seqlens tuple.

    :param tuple[int, ...] cu_seqlens_host: Host cumulative lengths with shape ``(B+1,)``.
    :param int block_m: Query tile height.
    :return tuple[tuple[int, ...], tuple[int, ...], int]:
        Batch ids, start offsets, and total tile count.
    """

    mid_batch: list[int] = []
    mid_start: list[int] = []
    mn = 0
    for batch_idx in range(max(0, len(cu_seqlens_host) - 1)):
        q_start = int(cu_seqlens_host[batch_idx])
        q_end = int(cu_seqlens_host[batch_idx + 1])
        n_batch_blocks = max(0, (q_end - q_start + int(block_m) - 1) // int(block_m))
        mn += int(n_batch_blocks)
        for block_idx in range(int(n_batch_blocks)):
            mid_batch.append(int(batch_idx))
            mid_start.append(int(q_start + block_idx * int(block_m)))
    return tuple(mid_batch), tuple(mid_start), int(mn)


def _get_mid_tensors_cached(
    *,
    cu_seqlens: torch.Tensor,
    block_m: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return cached varlen mid tensors for one cumulative-seqlens tensor.

    :param torch.Tensor cu_seqlens: Device cumulative-seqlens tensor.
    :param int block_m: Query tile height.
    :param torch.device device: Device where the returned tensors should live.
    :return tuple[torch.Tensor, torch.Tensor, int]: ``(mid_batch, mid_start, mn)``.
    """

    cache_key = (id(cu_seqlens), int(block_m), str(device.type), device.index)
    cached = _MID_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        cached_tensor = cached.cu_ref() if cached.cu_ref is not None else None
        if cached_tensor is cu_seqlens:
            return cached.mid_batch, cached.mid_start, cached.mn
        _MID_TENSOR_CACHE.pop(cache_key, None)

    cu_seqlens_host = _cu_seqlens_host_tuple(cu_seqlens)
    mid_batch_host, mid_start_host, mn = _build_mid_host_tuples(
        cu_seqlens_host=cu_seqlens_host,
        block_m=int(block_m),
    )
    mid_batch = torch.tensor(mid_batch_host, dtype=torch.long, device=device)
    mid_start = torch.tensor(mid_start_host, dtype=torch.long, device=device)

    cu_ref: weakref.ReferenceType[torch.Tensor] | None = None
    try:
        cu_ref = weakref.ref(cu_seqlens)
    except TypeError:
        cu_ref = None

    _MID_TENSOR_CACHE[cache_key] = _MidTensorCacheEntry(
        cu_ref=cu_ref,
        block_m=int(block_m),
        device_type=str(device.type),
        device_index=device.index,
        mid_batch=mid_batch,
        mid_start=mid_start,
        mn=int(mn),
    )
    return mid_batch, mid_start, int(mn)


def _build_dense_mid_tensors(
    *,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build compile-friendly tile metadata without Python-side caches.

    The compile-visible varlen path uses fixed-capacity packed buffers of size
    ``B*S``. That means the number of tiles per batch is also fixed by the
    padded sequence length, even though the actual token counts still come from
    ``cu_seqlens``.

    :param torch.Tensor cu_seqlens: Cumulative active lengths tensor ``(B+1,)``.
    :param int batch_size: Batch size.
    :param int seq_len: Padded sequence length.
    :param int block_size: Tile size for the corresponding kernel axis.
    :return tuple[torch.Tensor, torch.Tensor, int]: Device ``(mid_batch, mid_start, tile_count)``.
    """

    tiles_per_batch = max(1, (int(seq_len) + int(block_size) - 1) // int(block_size))
    tile_offsets = torch.arange(tiles_per_batch, device=cu_seqlens.device, dtype=torch.long) * int(block_size)
    starts = cu_seqlens[:-1].to(dtype=torch.long)
    mid_start = (starts[:, None] + tile_offsets[None, :]).reshape(-1)
    mid_batch = torch.arange(batch_size, device=cu_seqlens.device, dtype=torch.long).repeat_interleave(
        tiles_per_batch
    )
    return mid_batch, mid_start, int(batch_size * tiles_per_batch)


@triton.jit
def _pack_grad_and_delta_prefix_kernel(
    grad_padded_ptr: None,
    out_unpad_ptr: None,
    grad_unpad_ptr: None,
    delta_ptr: None,
    seqlens_ptr: None,
    cu_seqlens_ptr: None,
    stride_gb: int,
    stride_gs: int,
    stride_gh: int,
    stride_gd: int,
    stride_oz: int,
    stride_oh: int,
    stride_od: int,
    stride_guz: int,
    stride_guh: int,
    stride_gud: int,
    stride_dz: int,
    stride_dh: int,
    head_dim: int,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
) -> None:
    """Pack padded output gradients and compute delta in one Triton pass.

    :param Any grad_padded_ptr: Pointer to padded output gradients.
    :param Any out_unpad_ptr: Pointer to packed forward outputs.
    :param Any grad_unpad_ptr: Pointer to packed output gradients.
    :param Any delta_ptr: Pointer to packed delta tensor.
    :param Any seqlens_ptr: Pointer to per-example active lengths.
    :param Any cu_seqlens_ptr: Pointer to cumulative sequence lengths.
    :param Any stride_gb: Batch stride for padded gradients.
    :param Any stride_gs: Sequence stride for padded gradients.
    :param Any stride_gh: Head stride for padded gradients.
    :param Any stride_gd: Feature stride for padded gradients.
    :param Any stride_oz: Token stride for packed forward outputs.
    :param Any stride_oh: Head stride for packed forward outputs.
    :param Any stride_od: Feature stride for packed forward outputs.
    :param Any stride_guz: Token stride for packed output gradients.
    :param Any stride_guh: Head stride for packed output gradients.
    :param Any stride_gud: Feature stride for packed output gradients.
    :param Any stride_dz: Token stride for packed delta.
    :param Any stride_dh: Head stride for packed delta.
    :param Any head_dim: Runtime head dimension.
    :param Any BLOCK_ROWS: Row tile size.
    :param Any BLOCK_DMODEL: Feature tile size.
    :return None: This Triton kernel writes in place.
    """

    batch_idx = tl.program_id(0)
    tile_row = tl.program_id(1)
    head_idx = tl.program_id(2)

    row_offsets = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    d_offsets = tl.arange(0, BLOCK_DMODEL)

    seqlen = tl.load(seqlens_ptr + batch_idx)
    packed_base = tl.load(cu_seqlens_ptr + batch_idx)
    packed_rows = packed_base + row_offsets
    mask_rows = row_offsets < seqlen
    mask_d = d_offsets < head_dim
    mask = mask_rows[:, None] & mask_d[None, :]

    grad_ptrs = grad_padded_ptr + (
        batch_idx * stride_gb
        + row_offsets[:, None] * stride_gs
        + head_idx * stride_gh
        + d_offsets[None, :] * stride_gd
    )
    out_ptrs = out_unpad_ptr + (
        packed_rows[:, None] * stride_oz + head_idx * stride_oh + d_offsets[None, :] * stride_od
    )
    grad_unpad_ptrs = grad_unpad_ptr + (
        packed_rows[:, None] * stride_guz + head_idx * stride_guh + d_offsets[None, :] * stride_gud
    )

    grad = tl.load(grad_ptrs, mask=mask, other=0.0)
    out = tl.load(out_ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(grad_unpad_ptrs, grad, mask=mask)

    delta = tl.sum(out * grad.to(tl.float32), axis=1)
    delta_ptrs = delta_ptr + packed_rows * stride_dz + head_idx * stride_dh
    tl.store(delta_ptrs, delta, mask=mask_rows)


def _packed_delta_fallback(
    *,
    out_unpad: torch.Tensor,
    grad_unpad: torch.Tensor,
    total_tokens: int,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Return ``delta = sum(out * grad, dim=-1)`` with safe inactive-tail handling.

    :param torch.Tensor out_unpad: Packed forward output tensor.
    :param torch.Tensor grad_unpad: Packed output gradient tensor.
    :param int total_tokens: Packed-token capacity.
    :param torch.Tensor cu_seqlens: Cumulative sequence lengths.
    :return torch.Tensor: Packed delta tensor in ``(NNZ, H)`` layout.
    """

    delta = torch.zeros(
        (int(total_tokens), int(out_unpad.shape[1])),
        device=out_unpad.device,
        dtype=torch.float32,
    )
    if int(total_tokens) == 0 or int(cu_seqlens.numel()) == 0:
        return delta
    active_tokens = int(cu_seqlens[-1].detach().cpu().item())
    if active_tokens > 0:
        delta[:active_tokens] = (
            out_unpad[:active_tokens].to(dtype=torch.float32)
            * grad_unpad[:active_tokens].to(dtype=torch.float32)
        ).sum(dim=-1)
    return delta


def _pack_grad_and_delta_from_padded(
    *,
    grad_output: torch.Tensor,
    output_padded: torch.Tensor,
    out_unpad: torch.Tensor | None,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack padded output gradients and build packed delta.

    :param torch.Tensor grad_output: Padded output gradient in ``(B, S, H, D)`` layout.
    :param torch.Tensor output_padded: Padded forward output in ``(B, S, H, D)`` layout.
    :param torch.Tensor | None out_unpad: Optional cached packed forward output.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative sequence lengths.
    :param int max_seqlen: Sequence bound used for prefix-pack grids.
    :param int total_tokens: Packed-token count or fixed-capacity packed size.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Packed forward output, packed output gradient, and packed delta tensor.
    """

    total_tokens = int(total_tokens)
    if total_tokens == 0:
        empty_grad = grad_output.new_empty((0, *grad_output.shape[2:]))
        empty_delta = grad_output.new_empty((0, int(grad_output.shape[2])), dtype=torch.float32)
        empty_out = empty_grad if out_unpad is None else out_unpad
        return empty_out, empty_grad, empty_delta

    if out_unpad is None:
        out_unpad, grad_unpad = prefix_pack_padded_rows_pair(
            output_padded,
            grad_output,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            total_tokens=total_tokens,
        )
        delta = _packed_delta_fallback(
            out_unpad=out_unpad,
            grad_unpad=grad_unpad,
            total_tokens=total_tokens,
            cu_seqlens=cu_seqlens,
        )
        return out_unpad, grad_unpad, delta

    can_use_triton = (
        triton is not None
        and tl is not None
        and hasattr(torch, "library")
        and hasattr(torch.library, "wrap_triton")
        and grad_output.device.type == "cuda"
        and out_unpad.device.type == "cuda"
        and seqlens.device == grad_output.device
        and cu_seqlens.device == grad_output.device
    )
    if not can_use_triton:
        grad_unpad = prefix_pack_padded_rows(
            grad_output,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            total_tokens=total_tokens,
        )
        delta = _packed_delta_fallback(
            out_unpad=out_unpad,
            grad_unpad=grad_unpad,
            total_tokens=total_tokens,
            cu_seqlens=cu_seqlens,
        )
        return out_unpad, grad_unpad, delta

    head_dim = int(grad_output.shape[-1])
    block_dmodel = min(256, max(16, triton.next_power_of_2(head_dim)))
    grad_unpad = torch.empty(
        (total_tokens, int(grad_output.shape[2]), head_dim),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    delta = torch.zeros(
        (total_tokens, int(grad_output.shape[2])),
        device=grad_output.device,
        dtype=torch.float32,
    )
    grid = (
        int(grad_output.shape[0]),
        triton.cdiv(int(max_seqlen), _GRAD_PACK_DELTA_BLOCK_ROWS),
        int(grad_output.shape[2]),
    )
    torch.library.wrap_triton(_pack_grad_and_delta_prefix_kernel)[grid](
        grad_output,
        out_unpad,
        grad_unpad,
        delta,
        seqlens,
        cu_seqlens,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        out_unpad.stride(0),
        out_unpad.stride(1),
        out_unpad.stride(2),
        grad_unpad.stride(0),
        grad_unpad.stride(1),
        grad_unpad.stride(2),
        delta.stride(0),
        delta.stride(1),
        head_dim,
        BLOCK_ROWS=_GRAD_PACK_DELTA_BLOCK_ROWS,
        BLOCK_DMODEL=block_dmodel,
        num_warps=4,
        num_stages=2,
    )
    return out_unpad, grad_unpad, delta


def _varlen_backward_raw_impl(
    *,
    q_unpad: torch.Tensor,
    k_unpad: torch.Tensor,
    v_unpad: torch.Tensor,
    out_unpad: torch.Tensor,
    grad_unpad: torch.Tensor,
    lse_unpad: torch.Tensor,
    delta: torch.Tensor,
    pos_key_unpad: torch.Tensor | None,
    pos_query_unpad: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_bound: int,
    token_capacity: int,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
    dense_mid_tensors: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Launch raw varlen backward Triton kernels over already packed tensors.

    :param torch.Tensor q_unpad: Packed queries.
    :param torch.Tensor k_unpad: Packed keys.
    :param torch.Tensor v_unpad: Packed values.
    :param torch.Tensor out_unpad: Packed forward output.
    :param torch.Tensor grad_unpad: Packed output gradient.
    :param torch.Tensor lse_unpad: Packed forward log-sum-exp tensor.
    :param torch.Tensor delta: Packed delta tensor.
    :param torch.Tensor | None pos_key_unpad: Optional packed c2p tensor.
    :param torch.Tensor | None pos_query_unpad: Optional packed p2c tensor.
    :param torch.Tensor cu_seqlens: Cumulative sequence lengths.
    :param int batch_size: Batch size.
    :param int seq_bound: Sequence bound used for config and tile construction.
    :param int token_capacity: Packed-token count or fixed-capacity size.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :param bool dense_mid_tensors: Whether to build fixed-capacity dense mid tensors.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        Packed gradients for q/k/v and optional positional tensors.
    """

    att_span = int(position_buckets) if int(position_buckets) > 0 else int(max_relative_distance)
    kv_block_m, kv_block_n, kv_num_stages, kv_num_warps = _resolve_varlen_bwd_kernel_config(
        kind="kv",
        total_tokens_q=int(token_capacity),
        total_tokens_k=int(token_capacity),
        max_seqlen_q=int(seq_bound),
        max_seqlen_k=int(seq_bound),
        batch_size=int(batch_size),
        head_dim=int(q_unpad.shape[-1]),
        causal=bool(causal),
        disentangled=True,
        att_span=att_span,
        dtype=q_unpad.dtype,
        device=q_unpad.device,
    )
    q_block_m, q_block_n, q_num_stages, q_num_warps = _resolve_varlen_bwd_kernel_config(
        kind="q",
        total_tokens_q=int(token_capacity),
        total_tokens_k=int(token_capacity),
        max_seqlen_q=int(seq_bound),
        max_seqlen_k=int(seq_bound),
        batch_size=int(batch_size),
        head_dim=int(q_unpad.shape[-1]),
        causal=bool(causal),
        disentangled=True,
        att_span=att_span,
        dtype=q_unpad.dtype,
        device=q_unpad.device,
    )

    if dense_mid_tensors:
        mid_m_batch, mid_m_start, m_tile_count = _build_dense_mid_tensors(
            cu_seqlens=cu_seqlens,
            batch_size=batch_size,
            seq_len=int(seq_bound),
            block_size=q_block_m,
        )
        mid_n_batch, mid_n_start, n_tile_count = _build_dense_mid_tensors(
            cu_seqlens=cu_seqlens,
            batch_size=batch_size,
            seq_len=int(seq_bound),
            block_size=kv_block_n,
        )
    else:
        mid_m_batch, mid_m_start, m_tile_count = _get_mid_tensors_cached(
            cu_seqlens=cu_seqlens,
            block_m=q_block_m,
            device=q_unpad.device,
        )
        mid_n_batch, mid_n_start, n_tile_count = _get_mid_tensors_cached(
            cu_seqlens=cu_seqlens,
            block_m=kv_block_n,
            device=q_unpad.device,
        )

    dq_unpad = torch.empty_like(q_unpad)
    dk_unpad = torch.empty_like(k_unpad)
    dv_unpad = torch.empty_like(v_unpad)
    dpos_key_unpad = torch.zeros_like(pos_key_unpad) if pos_key_unpad is not None else None
    dpos_query_unpad = torch.zeros_like(pos_query_unpad) if pos_query_unpad is not None else None

    if pos_key_unpad is not None:
        stride_pk0, stride_pk1, stride_pk2 = pos_key_unpad.stride()
    else:
        stride_pk0 = stride_pk1 = stride_pk2 = 0
    if pos_query_unpad is not None:
        stride_pq0, stride_pq1, stride_pq2 = pos_query_unpad.stride()
    else:
        stride_pq0 = stride_pq1 = stride_pq2 = 0

    grid_kv = (n_tile_count, int(q_unpad.shape[1]))
    torch.library.wrap_triton(_bwd_kv_dise_kernel_varlen_raw)[grid_kv](
        q_unpad,
        k_unpad,
        v_unpad,
        pos_key_unpad if pos_key_unpad is not None else k_unpad,
        pos_query_unpad if pos_query_unpad is not None else v_unpad,
        float(sm_scale),
        grad_unpad,
        dk_unpad,
        dv_unpad,
        dpos_key_unpad if dpos_key_unpad is not None else k_unpad,
        dpos_query_unpad if dpos_query_unpad is not None else v_unpad,
        lse_unpad,
        delta,
        cu_seqlens,
        cu_seqlens,
        mid_n_batch,
        mid_n_start,
        q_unpad.stride(0),
        q_unpad.stride(1),
        q_unpad.stride(2),
        k_unpad.stride(0),
        k_unpad.stride(1),
        k_unpad.stride(2),
        v_unpad.stride(0),
        v_unpad.stride(1),
        v_unpad.stride(2),
        grad_unpad.stride(0),
        grad_unpad.stride(1),
        grad_unpad.stride(2),
        dk_unpad.stride(0),
        dk_unpad.stride(1),
        dk_unpad.stride(2),
        dv_unpad.stride(0),
        dv_unpad.stride(1),
        dv_unpad.stride(2),
        stride_pk0,
        stride_pk1,
        stride_pk2,
        stride_pq0,
        stride_pq1,
        stride_pq2,
        batch_size,
        int(q_unpad.shape[1]),
        BLOCK_M=kv_block_m,
        BLOCK_DMODEL=int(q_unpad.shape[-1]),
        BLOCK_N=kv_block_n,
        CAUSAL=bool(causal),
        HAS_C2P=bool(pos_key_unpad is not None),
        HAS_P2C=bool(pos_query_unpad is not None),
        ATT_SPAN=att_span,
        NUM_BUCKETS=int(position_buckets),
        MAX_DISTANCE=int(max_relative_distance),
        num_warps=kv_num_warps,
        num_stages=kv_num_stages,
    )

    grid_q = (m_tile_count, int(q_unpad.shape[1]))
    torch.library.wrap_triton(_bwd_q_dise_kernel_varlen_raw)[grid_q](
        q_unpad,
        k_unpad,
        v_unpad,
        pos_key_unpad if pos_key_unpad is not None else q_unpad,
        pos_query_unpad if pos_query_unpad is not None else k_unpad,
        float(sm_scale),
        grad_unpad,
        dq_unpad,
        lse_unpad,
        delta,
        cu_seqlens,
        cu_seqlens,
        mid_m_batch,
        mid_m_start,
        q_unpad.stride(0),
        q_unpad.stride(1),
        q_unpad.stride(2),
        k_unpad.stride(0),
        k_unpad.stride(1),
        k_unpad.stride(2),
        v_unpad.stride(0),
        v_unpad.stride(1),
        v_unpad.stride(2),
        grad_unpad.stride(0),
        grad_unpad.stride(1),
        grad_unpad.stride(2),
        dq_unpad.stride(0),
        dq_unpad.stride(1),
        dq_unpad.stride(2),
        stride_pk0,
        stride_pk1,
        stride_pk2,
        stride_pq0,
        stride_pq1,
        stride_pq2,
        batch_size,
        int(q_unpad.shape[1]),
        BLOCK_M=q_block_m,
        BLOCK_DMODEL=int(q_unpad.shape[-1]),
        BLOCK_N=q_block_n,
        CAUSAL=bool(causal),
        HAS_C2P=bool(pos_key_unpad is not None),
        HAS_P2C=bool(pos_query_unpad is not None),
        ATT_SPAN=att_span,
        NUM_BUCKETS=int(position_buckets),
        MAX_DISTANCE=int(max_relative_distance),
        num_warps=q_num_warps,
        num_stages=q_num_stages,
    )
    return dq_unpad, dk_unpad, dv_unpad, dpos_key_unpad, dpos_query_unpad


def _patch_upstream_varlen_mid_cache() -> None:
    """Replace upstream varlen mid-cache helpers with repo-local caching."""

    if _flash_attention_varlen_module is None:
        return

    def _repo_get_mid_cached(
        cu_seqlens: torch.Tensor,
        B: int,
        BLOCK_M: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return repo-cached mid tensors with the upstream helper signature.

        :param torch.Tensor cu_seqlens: Cumulative seqlens tensor for one batch.
        :param int B: Upstream batch-size argument, unused by the repo-local cache.
        :param int BLOCK_M: Query tile height.
        :param torch.device device: Device where the returned tensors should live.
        :return tuple[torch.Tensor, torch.Tensor, int]:
            Cached ``(mid_batch, mid_start, mn)`` tensors and tile count.
        """

        del B
        return _get_mid_tensors_cached(
            cu_seqlens=cu_seqlens,
            block_m=int(BLOCK_M),
            device=device,
        )

    _flash_attention_varlen_module.get_mid_cached = _repo_get_mid_cached


def _get_unpad_metadata_entry(mask_2d: torch.Tensor) -> _MaskMetadataCacheEntry:
    """Return cached unpadding metadata entry for one mask tensor.

    :param torch.Tensor mask_2d: Boolean keep mask with shape ``(B, S)``.
    :return _MaskMetadataCacheEntry: Cached or newly built metadata entry.
    """

    cache_key = _mask_metadata_cache_key(mask_2d)
    cached = _MASK_METADATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    seqlens, cu_seqlens, max_seqlen, total_tokens, cu_seqlens_host = _build_unpad_metadata(mask_2d)

    entry = _MaskMetadataCacheEntry(
        mask_ref=None,
        version=cache_key[-1],
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        total_tokens=total_tokens,
        cu_seqlens_host=cu_seqlens_host,
    )
    _register_cu_seqlens_host_tuple(cu_seqlens=cu_seqlens, cu_seqlens_host=cu_seqlens_host)
    _MASK_METADATA_CACHE[cache_key] = entry
    if len(_MASK_METADATA_CACHE) > 512:
        stale_keys = list(_MASK_METADATA_CACHE.keys())[:256]
        for stale_key in stale_keys:
            _MASK_METADATA_CACHE.pop(stale_key, None)
    return entry


def _get_unpad_metadata_cached(mask_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return cached unpadding metadata for one mask tensor when possible.

    Real training reuses the same padding-mask tensor across many attention
    layers in one generator/discriminator phase and then again in backward.
    Recomputing ``nonzero`` and cumulative lengths for every call adds a large
    amount of host overhead, especially for varlen-heavy compiled runs.

    :param torch.Tensor mask_2d: Boolean keep mask with shape ``(B, S)``.
    :return tuple[torch.Tensor, torch.Tensor, int]: Cached or newly built metadata.
    """

    entry = _get_unpad_metadata_entry(mask_2d)
    return entry.seqlens, entry.cu_seqlens, entry.max_seqlen


def _store_forward_aux_cache(
    *,
    output_padded: torch.Tensor,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_tokens: int,
    q_unpad: torch.Tensor,
    k_unpad: torch.Tensor,
    v_unpad: torch.Tensor,
    out_unpad: torch.Tensor,
    lse_unpad: torch.Tensor,
    pos_key_unpad: torch.Tensor | None,
    pos_query_unpad: torch.Tensor | None,
) -> None:
    """Stash forward unpadded tensors so backward does not rebuild them.

    :param torch.Tensor output_padded: Returned padded attention output tensor.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative sequence lengths.
    :param int max_seqlen: Maximum active length in batch.
    :param int total_tokens: Total active tokens in batch.
    :param torch.Tensor q_unpad: Unpadded query tensor.
    :param torch.Tensor k_unpad: Unpadded key tensor.
    :param torch.Tensor v_unpad: Unpadded value tensor.
    :param torch.Tensor out_unpad: Unpadded forward output tensor.
    :param torch.Tensor lse_unpad: Unpadded forward LSE tensor.
    :param torch.Tensor | None pos_key_unpad: Optional unpadded c2p tensor.
    :param torch.Tensor | None pos_query_unpad: Optional unpadded p2c tensor.
    """

    cache_key = id(output_padded)
    output_ref: weakref.ReferenceType[torch.Tensor] | None = None
    try:
        output_ref = weakref.ref(output_padded, lambda _ref, key=cache_key: _FORWARD_AUX_CACHE.pop(key, None))
    except TypeError:
        output_ref = None

    _FORWARD_AUX_CACHE[cache_key] = _ForwardAuxCacheEntry(
        output_ref=output_ref,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        total_tokens=int(total_tokens),
        q_unpad=q_unpad,
        k_unpad=k_unpad,
        v_unpad=v_unpad,
        out_unpad=out_unpad,
        lse_unpad=lse_unpad,
        pos_key_unpad=pos_key_unpad,
        pos_query_unpad=pos_query_unpad,
    )


def _pop_forward_aux_cache(output_padded: torch.Tensor) -> _ForwardAuxCacheEntry | None:
    """Return and remove cached forward aux tensors for one output tensor.

    :param torch.Tensor output_padded: Padded output tensor returned by the custom op.
    :return _ForwardAuxCacheEntry | None: Cached aux entry or ``None``.
    """

    cache_key = id(output_padded)
    cached = _FORWARD_AUX_CACHE.pop(cache_key, None)
    if cached is None:
        return None
    cached_output = cached.output_ref() if cached.output_ref is not None else None
    if cached.output_ref is not None and cached_output is not output_padded:
        return None
    return cached


def _varlen_eager_forward_impl(
    *,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
    require_lse: bool,
    stash_backward_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run padded varlen attention eagerly, returning padded outputs.

    :param torch.Tensor query_layer: Queries in ``(B, S, H, D)`` layout.
    :param torch.Tensor key_layer: Keys in ``(B, S, H, D)`` layout.
    :param torch.Tensor value_layer: Values in ``(B, S, H, D)`` layout.
    :param torch.Tensor attention_mask_2d: Boolean keep mask in ``(B, S)`` layout.
    :param torch.Tensor | None pos_key: Optional c2p tensor in ``(B, S, H, P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c tensor in ``(B, S, H, P)`` layout.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :param bool require_lse: Whether the caller also needs padded log-sum-exp values.
    :param bool stash_backward_cache: Whether to cache unpadded forward tensors for reuse in backward.
    :raises RuntimeError: If no eager varlen implementation is importable.
    :return tuple[torch.Tensor, torch.Tensor | None]: Padded output in ``(B, S, H, D)``
        layout and optional padded LSE in ``(B, S, H)`` layout.
    """

    if (
        _flash_attn_v2_fwd_dise_lowlevel is None
        and _flash_attention_with_disentangled_varlen_highlevel is None
    ):
        detail = flashdeberta_varlen_import_error()
        raise RuntimeError(
            "FlashDeBERTa varlen attention is unavailable."
            if detail is None
            else f"FlashDeBERTa varlen attention is unavailable ({detail})."
        )

    batch_size = int(query_layer.shape[0])
    seq_len = int(query_layer.shape[1])
    att_span = int(position_buckets) if int(position_buckets) > 0 else int(max_relative_distance)

    metadata = _get_unpad_metadata_entry(attention_mask_2d)
    seqlens = metadata.seqlens
    cu_seqlens = metadata.cu_seqlens
    max_seqlen = metadata.max_seqlen
    total_tokens = metadata.total_tokens
    if total_tokens == 0:
        output = torch.zeros_like(query_layer)
        if require_lse:
            lse = torch.zeros(
                (batch_size, seq_len, int(query_layer.shape[2])),
                device=query_layer.device,
                dtype=torch.float32,
            )
            return output, lse
        return output, None

    q_unpad, k_unpad, v_unpad = prefix_pack_padded_rows_triple(
        query_layer,
        key_layer,
        value_layer,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        total_tokens=total_tokens,
    )
    if pos_key is not None and pos_query is not None:
        pos_key_unpad, pos_query_unpad = prefix_pack_padded_rows_pair(
            pos_key,
            pos_query,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            total_tokens=total_tokens,
        )
    else:
        pos_key_unpad = (
            prefix_pack_padded_rows(
                pos_key,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                total_tokens=total_tokens,
            )
            if pos_key is not None
            else None
        )
        pos_query_unpad = (
            prefix_pack_padded_rows(
                pos_query,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                total_tokens=total_tokens,
            )
            if pos_query is not None
            else None
        )

    if _flash_attn_v2_fwd_dise_lowlevel is not None and _get_fwd_config_lowlevel is not None:
        override = _varlen_kernel_override_from_env(kind="fwd")
        if override is not None:
            block_m, block_n, num_stages, num_warps = override
        else:
            block_m, block_n, num_stages, num_warps = _get_fwd_config_lowlevel(
                total_tokens=int(q_unpad.shape[0]),
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                D=int(query_layer.shape[-1]),
                causal=bool(causal),
                disentangled=True,
                att_span=att_span,
            )
        out_unpad, lse_unpad = _flash_attn_v2_fwd_dise_lowlevel(
            q_unpad,
            k_unpad,
            v_unpad,
            pos_key_unpad,
            pos_query_unpad,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
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
    else:
        out_unpad = _flash_attention_with_disentangled_varlen_highlevel(
            q_unpad,
            k_unpad,
            v_unpad,
            pos_key_unpad,
            pos_query_unpad,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            bool(causal),
            float(sm_scale),
            int(position_buckets),
            int(max_relative_distance),
        )
        lse_unpad = None

    out_padded = prefix_unpack_padded_rows(
        out_unpad,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    if not require_lse:
        return out_padded, None
    if lse_unpad is None:
        raise RuntimeError(
            "Compiled FlashDeBERTa varlen requires low-level forward primitives with padded LSE support."
        )

    lse_padded = prefix_unpack_padded_rows(
        lse_unpad,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
    ).contiguous()
    if stash_backward_cache:
        _store_forward_aux_cache(
            output_padded=out_padded,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            total_tokens=total_tokens,
            q_unpad=q_unpad,
            k_unpad=k_unpad,
            v_unpad=v_unpad,
            out_unpad=out_unpad,
            lse_unpad=lse_unpad,
            pos_key_unpad=pos_key_unpad,
            pos_query_unpad=pos_query_unpad,
        )
    return out_padded, lse_padded


def _varlen_eager_backward_impl(
    *,
    grad_output: torch.Tensor,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    output_padded: torch.Tensor,
    lse_padded: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run padded varlen backward eagerly and repad gradients.

    :param torch.Tensor grad_output: Gradient of padded output in ``(B, S, H, D)`` layout.
    :param torch.Tensor query_layer: Forward queries in ``(B, S, H, D)`` layout.
    :param torch.Tensor key_layer: Forward keys in ``(B, S, H, D)`` layout.
    :param torch.Tensor value_layer: Forward values in ``(B, S, H, D)`` layout.
    :param torch.Tensor attention_mask_2d: Boolean keep mask in ``(B, S)`` layout.
    :param torch.Tensor output_padded: Forward output in ``(B, S, H, D)`` layout.
    :param torch.Tensor lse_padded: Forward padded LSE in ``(B, S, H)`` layout.
    :param torch.Tensor | None pos_key: Optional c2p tensor in ``(B, S, H, P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c tensor in ``(B, S, H, P)`` layout.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :raises RuntimeError: If low-level backward primitives are unavailable.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        Gradients in the same padded layouts as the forward inputs.
    """

    if _flash_attn_v2_bwd_dise_varlen_lowlevel is None or _get_bwd_config_varlen_lowlevel is None:
        detail = _FLASH_VARLEN_LOWLEVEL_IMPORT_ERROR
        raise RuntimeError(
            "Compiled FlashDeBERTa varlen backward is unavailable."
            if detail is None
            else f"Compiled FlashDeBERTa varlen backward is unavailable ({detail})."
        )

    metadata = _get_unpad_metadata_entry(attention_mask_2d)
    return _varlen_eager_backward_cached_impl(
        grad_output=grad_output,
        query_layer=query_layer,
        key_layer=key_layer,
        value_layer=value_layer,
        output_padded=output_padded,
        lse_padded=lse_padded,
        pos_key=pos_key,
        pos_query=pos_query,
        sm_scale=sm_scale,
        position_buckets=position_buckets,
        max_relative_distance=max_relative_distance,
        causal=causal,
        seqlens=metadata.seqlens,
        cu_seqlens=metadata.cu_seqlens,
        max_seqlen=metadata.max_seqlen,
        total_tokens=metadata.total_tokens,
        q_unpad=None,
        k_unpad=None,
        v_unpad=None,
        out_unpad=None,
        lse_unpad=None,
        pos_key_unpad=None,
        pos_query_unpad=None,
    )


def _varlen_eager_backward_cached_impl(
    *,
    grad_output: torch.Tensor,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    output_padded: torch.Tensor,
    lse_padded: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_tokens: int,
    q_unpad: torch.Tensor | None,
    k_unpad: torch.Tensor | None,
    v_unpad: torch.Tensor | None,
    out_unpad: torch.Tensor | None,
    lse_unpad: torch.Tensor | None,
    pos_key_unpad: torch.Tensor | None,
    pos_query_unpad: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run padded varlen backward, optionally reusing forward-side unpadded tensors.

    :param torch.Tensor grad_output: Gradient of padded output in ``(B, S, H, D)`` layout.
    :param torch.Tensor query_layer: Forward queries in ``(B, S, H, D)`` layout.
    :param torch.Tensor key_layer: Forward keys in ``(B, S, H, D)`` layout.
    :param torch.Tensor value_layer: Forward values in ``(B, S, H, D)`` layout.
    :param torch.Tensor output_padded: Forward output in ``(B, S, H, D)`` layout.
    :param torch.Tensor lse_padded: Forward padded LSE in ``(B, S, H)`` layout.
    :param torch.Tensor | None pos_key: Optional c2p tensor in ``(B, S, H, P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c tensor in ``(B, S, H, P)`` layout.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :param torch.Tensor seqlens: Per-example active lengths.
    :param torch.Tensor cu_seqlens: Cumulative sequence lengths.
    :param int max_seqlen: Maximum active length in batch.
    :param int total_tokens: Total active tokens in batch.
    :param torch.Tensor | None q_unpad: Optional cached unpadded query tensor.
    :param torch.Tensor | None k_unpad: Optional cached unpadded key tensor.
    :param torch.Tensor | None v_unpad: Optional cached unpadded value tensor.
    :param torch.Tensor | None out_unpad: Optional cached unpadded output tensor.
    :param torch.Tensor | None lse_unpad: Optional cached unpadded LSE tensor.
    :param torch.Tensor | None pos_key_unpad: Optional cached unpadded c2p tensor.
    :param torch.Tensor | None pos_query_unpad: Optional cached unpadded p2c tensor.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        Gradients in the same padded layouts as the forward inputs.
    """

    batch_size = int(query_layer.shape[0])
    seq_len = int(query_layer.shape[1])

    if total_tokens == 0:
        dq = torch.zeros_like(query_layer)
        dk = torch.zeros_like(key_layer)
        dv = torch.zeros_like(value_layer)
        dpos_key = torch.zeros_like(pos_key) if pos_key is not None else None
        dpos_query = torch.zeros_like(pos_query) if pos_query is not None else None
        return dq, dk, dv, dpos_key, dpos_query

    if q_unpad is None or k_unpad is None or v_unpad is None:
        q_unpad, k_unpad, v_unpad = prefix_pack_padded_rows_triple(
            query_layer,
            key_layer,
            value_layer,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            total_tokens=total_tokens,
        )
    if out_unpad is None:
        out_unpad = prefix_pack_padded_rows(
            output_padded,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            total_tokens=total_tokens,
        )
    out_unpad, grad_unpad, delta = _pack_grad_and_delta_from_padded(
        grad_output=grad_output,
        output_padded=output_padded,
        out_unpad=out_unpad,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        total_tokens=total_tokens,
    )
    if lse_unpad is None:
        lse_unpad = prefix_pack_padded_rows(
            lse_padded,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            total_tokens=total_tokens,
        )
    if pos_key is not None and pos_query is not None and (pos_key_unpad is None or pos_query_unpad is None):
        pos_key_unpad, pos_query_unpad = prefix_pack_padded_rows_pair(
            pos_key,
            pos_query,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            total_tokens=total_tokens,
        )
    else:
        if pos_key is not None and pos_key_unpad is None:
            pos_key_unpad = prefix_pack_padded_rows(
                pos_key,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                total_tokens=total_tokens,
            )
        if pos_query is not None and pos_query_unpad is None:
            pos_query_unpad = prefix_pack_padded_rows(
                pos_query,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                total_tokens=total_tokens,
            )

    dq_unpad, dk_unpad, dv_unpad, dpos_key_unpad, dpos_query_unpad = _varlen_backward_raw_impl(
        q_unpad=q_unpad,
        k_unpad=k_unpad,
        v_unpad=v_unpad,
        out_unpad=out_unpad,
        grad_unpad=grad_unpad,
        lse_unpad=lse_unpad,
        delta=delta,
        pos_key_unpad=pos_key_unpad,
        pos_query_unpad=pos_query_unpad,
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_bound=max_seqlen,
        token_capacity=total_tokens,
        sm_scale=sm_scale,
        position_buckets=position_buckets,
        max_relative_distance=max_relative_distance,
        causal=causal,
        dense_mid_tensors=False,
    )

    dq, dk, dv = prefix_unpack_padded_rows_triple(
        dq_unpad,
        dk_unpad,
        dv_unpad,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    if dpos_key_unpad is not None and dpos_query_unpad is not None:
        dpos_key, dpos_query = prefix_unpack_padded_rows_pair(
            dpos_key_unpad,
            dpos_query_unpad,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            batch_size=batch_size,
            seq_len=seq_len,
        )
    else:
        dpos_key = (
            prefix_unpack_padded_rows(
                dpos_key_unpad,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            if dpos_key_unpad is not None
            else None
        )
        dpos_query = (
            prefix_unpack_padded_rows(
                dpos_query_unpad,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            if dpos_query_unpad is not None
            else None
        )
    return dq, dk, dv, dpos_key, dpos_query


def _build_varlen_custom_ops() -> tuple[Any | None, Any | None]:
    """Register or retrieve the opaque padded-varlen custom ops.

    :return tuple[Any | None, Any | None]: Forward and backward custom-op handles.
    """

    existing_forward = _lookup_registered_op(_VARLEN_OP_NAMESPACE, _VARLEN_FWD_OP_NAME)
    existing_backward = _lookup_registered_op(_VARLEN_OP_NAMESPACE, _VARLEN_BWD_OP_NAME)
    if existing_forward is not None and existing_backward is not None:
        return existing_forward, existing_backward

    if (
        _flash_attn_v2_fwd_dise_lowlevel is None
        or _flash_attn_v2_bwd_dise_varlen_lowlevel is None
        or not hasattr(torch, "library")
        or not hasattr(torch.library, "custom_op")
    ):
        return None, None

    @torch.library.custom_op(
        f"{_VARLEN_OP_NAMESPACE}::{_VARLEN_FWD_OP_NAME}",
        mutates_args=(),
        device_types="cuda",
        schema=(
            "(Tensor q, Tensor k, Tensor v, Tensor mask, Tensor? pos_key, Tensor? pos_query, "
            "float sm_scale, int position_buckets, int max_relative_distance, bool causal) -> (Tensor, Tensor)"
        ),
    )
    def _forward_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run padded varlen forward as one opaque CUDA op.

        :param torch.Tensor q: Padded queries in ``(B, S, H, D)`` layout.
        :param torch.Tensor k: Padded keys in ``(B, S, H, D)`` layout.
        :param torch.Tensor v: Padded values in ``(B, S, H, D)`` layout.
        :param torch.Tensor mask: Boolean keep mask in ``(B, S)`` layout.
        :param torch.Tensor | None pos_key: Optional c2p tensor.
        :param torch.Tensor | None pos_query: Optional p2c tensor.
        :param float sm_scale: Softmax scale.
        :param int position_buckets: Relative-position bucket count.
        :param int max_relative_distance: Maximum relative distance.
        :param bool causal: Whether causal masking is enabled.
        :return tuple[torch.Tensor, torch.Tensor]: Padded output and padded LSE tensors.
        """

        return _varlen_eager_forward_impl(
            query_layer=q,
            key_layer=k,
            value_layer=v,
            attention_mask_2d=mask,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            position_buckets=position_buckets,
            max_relative_distance=max_relative_distance,
            causal=causal,
            require_lse=True,
            stash_backward_cache=True,
        )

    @torch.library.register_fake(_forward_op)
    def _forward_op_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fake forward outputs with static padded shapes.

        :param torch.Tensor q: Fake query tensor.
        :param torch.Tensor k: Fake key tensor.
        :param torch.Tensor v: Fake value tensor.
        :param torch.Tensor mask: Fake mask tensor.
        :param torch.Tensor | None pos_key: Fake optional c2p tensor.
        :param torch.Tensor | None pos_query: Fake optional p2c tensor.
        :param float sm_scale: Fake softmax scale.
        :param int position_buckets: Fake bucket count.
        :param int max_relative_distance: Fake maximum relative distance.
        :param bool causal: Fake causal flag.
        :return tuple[torch.Tensor, torch.Tensor]: Fake padded output and padded LSE tensors.
        """

        del k, v, mask, pos_key, pos_query, sm_scale, position_buckets, max_relative_distance, causal
        lse = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]),
            device=q.device,
            dtype=torch.float32,
        )
        return torch.empty(q.shape, device=q.device, dtype=q.dtype), lse

    @torch.library.custom_op(
        f"{_VARLEN_OP_NAMESPACE}::{_VARLEN_BWD_OP_NAME}",
        mutates_args=(),
        device_types="cuda",
        schema=(
            "(Tensor grad_out, Tensor q, Tensor k, Tensor v, Tensor mask, Tensor out, Tensor lse, "
            "Tensor? pos_key, Tensor? pos_query, float sm_scale, int position_buckets, "
            "int max_relative_distance, bool causal) -> (Tensor, Tensor, Tensor, Tensor?, Tensor?)"
        ),
    )
    def _backward_op(
        grad_out: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Run padded varlen backward as one opaque CUDA op.

        :param torch.Tensor grad_out: Gradient of padded output.
        :param torch.Tensor q: Forward padded queries.
        :param torch.Tensor k: Forward padded keys.
        :param torch.Tensor v: Forward padded values.
        :param torch.Tensor mask: Boolean keep mask in ``(B, S)`` layout.
        :param torch.Tensor out: Forward padded output.
        :param torch.Tensor lse: Forward padded LSE tensor.
        :param torch.Tensor | None pos_key: Optional c2p tensor.
        :param torch.Tensor | None pos_query: Optional p2c tensor.
        :param float sm_scale: Softmax scale.
        :param int position_buckets: Relative-position bucket count.
        :param int max_relative_distance: Maximum relative distance.
        :param bool causal: Whether causal masking is enabled.
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            Padded gradients for q/k/v and optional positional tensors.
        """

        return _varlen_eager_backward_impl(
            grad_output=grad_out,
            query_layer=q,
            key_layer=k,
            value_layer=v,
            attention_mask_2d=mask,
            output_padded=out,
            lse_padded=lse,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            position_buckets=position_buckets,
            max_relative_distance=max_relative_distance,
            causal=causal,
        )

    @torch.library.register_fake(_backward_op)
    def _backward_op_fake(
        grad_out: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return fake backward outputs with static padded shapes.

        :param torch.Tensor grad_out: Fake gradient tensor.
        :param torch.Tensor q: Fake query tensor.
        :param torch.Tensor k: Fake key tensor.
        :param torch.Tensor v: Fake value tensor.
        :param torch.Tensor mask: Fake mask tensor.
        :param torch.Tensor out: Fake output tensor.
        :param torch.Tensor lse: Fake LSE tensor.
        :param torch.Tensor | None pos_key: Fake optional c2p tensor.
        :param torch.Tensor | None pos_query: Fake optional p2c tensor.
        :param float sm_scale: Fake softmax scale.
        :param int position_buckets: Fake bucket count.
        :param int max_relative_distance: Fake maximum relative distance.
        :param bool causal: Fake causal flag.
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            Fake padded gradients for q/k/v and optional positional tensors.
        """

        del grad_out, mask, out, lse, sm_scale, position_buckets, max_relative_distance, causal
        dpos_key = (
            torch.empty(pos_key.shape, device=pos_key.device, dtype=pos_key.dtype)
            if pos_key is not None
            else None
        )
        dpos_query = (
            torch.empty(pos_query.shape, device=pos_query.device, dtype=pos_query.dtype)
            if pos_query is not None
            else None
        )
        return (
            torch.empty(q.shape, device=q.device, dtype=q.dtype),
            torch.empty(k.shape, device=k.device, dtype=k.dtype),
            torch.empty(v.shape, device=v.device, dtype=v.dtype),
            dpos_key,
            dpos_query,
        )

    def _setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Save forward inputs and outputs needed by the padded backward helper.

        :param Any ctx: Autograd context object.
        :param tuple[Any, ...] inputs: Forward custom-op inputs.
        :param tuple[torch.Tensor, torch.Tensor] output: Forward custom-op outputs.
        """

        q, k, v, mask, pos_key, pos_query, sm_scale, position_buckets, max_relative_distance, causal = inputs
        out, lse = output
        saved: list[torch.Tensor] = [q, k, v, mask, out, lse]
        if pos_key is not None:
            saved.append(pos_key)
        if pos_query is not None:
            saved.append(pos_query)
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
        """Dispatch backward through the opaque padded varlen backward helper.

        :param Any ctx: Autograd context populated by ``_setup_context``.
        :param torch.Tensor | None grad_out: Gradient of padded output.
        :param torch.Tensor | None grad_lse: Gradient of padded LSE output.
        :return tuple[torch.Tensor | None, ...]: Gradients for the forward custom-op inputs.
        """

        del grad_lse
        saved = list(ctx.saved_tensors)
        q, k, v, mask, out, lse = saved[:6]
        next_idx = 6
        pos_key = saved[next_idx] if bool(ctx.has_pos_key) else None
        if bool(ctx.has_pos_key):
            next_idx += 1
        pos_query = saved[next_idx] if bool(ctx.has_pos_query) else None
        grad = grad_out if grad_out is not None else torch.zeros_like(out)
        cached = _pop_forward_aux_cache(out)
        if cached is not None:
            dq, dk, dv, dpos_key, dpos_query = _varlen_eager_backward_cached_impl(
                grad_output=grad,
                query_layer=q,
                key_layer=k,
                value_layer=v,
                output_padded=out,
                lse_padded=lse,
                pos_key=pos_key,
                pos_query=pos_query,
                sm_scale=ctx.sm_scale,
                position_buckets=ctx.position_buckets,
                max_relative_distance=ctx.max_relative_distance,
                causal=ctx.causal,
                seqlens=cached.seqlens,
                cu_seqlens=cached.cu_seqlens,
                max_seqlen=cached.max_seqlen,
                total_tokens=cached.total_tokens,
                q_unpad=cached.q_unpad,
                k_unpad=cached.k_unpad,
                v_unpad=cached.v_unpad,
                out_unpad=cached.out_unpad,
                lse_unpad=cached.lse_unpad,
                pos_key_unpad=cached.pos_key_unpad,
                pos_query_unpad=cached.pos_query_unpad,
            )
        else:
            dq, dk, dv, dpos_key, dpos_query = _backward_op(
                grad,
                q,
                k,
                v,
                mask,
                out,
                lse,
                pos_key,
                pos_query,
                ctx.sm_scale,
                ctx.position_buckets,
                ctx.max_relative_distance,
                ctx.causal,
            )
        return dq, dk, dv, None, dpos_key, dpos_query, None, None, None, None

    torch.library.register_autograd(_forward_op, _backward, setup_context=_setup_context)
    return _forward_op, _backward_op


_FLASHDEBERTA_VARLEN_CUSTOM_OP, _FLASHDEBERTA_VARLEN_BWD_CUSTOM_OP = _build_varlen_custom_ops()
_patch_upstream_varlen_mid_cache()


def _varlen_triton_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run compile-visible padded varlen forward with fixed-capacity packed buffers.

    :param torch.Tensor q: Queries in ``(B, S, H, D)`` layout.
    :param torch.Tensor k: Keys in ``(B, S, H, D)`` layout.
    :param torch.Tensor v: Values in ``(B, S, H, D)`` layout.
    :param torch.Tensor mask: Boolean keep mask in ``(B, S)`` layout.
    :param torch.Tensor | None pos_key: Optional c2p tensor.
    :param torch.Tensor | None pos_query: Optional p2c tensor.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :return tuple[torch.Tensor, ...]:
        Padded output, padded LSE, cumulative sequence lengths, packed q/k/v,
        packed output/LSE, and optional packed positional tensors encoded as
        empty sentinels when absent.
    """

    if not _varlen_use_triton_op():
        raise RuntimeError("Compile-visible FlashDeBERTa varlen Triton support is unavailable.")

    batch_size = int(q.shape[0])
    seq_len = int(q.shape[1])
    num_heads = int(q.shape[2])
    head_dim = int(q.shape[3])
    capacity_tokens = int(batch_size * seq_len)
    att_span = int(position_buckets) if int(position_buckets) > 0 else int(max_relative_distance)

    seqlens = mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

    q_unpad, k_unpad, v_unpad = prefix_pack_padded_rows_triple(
        q,
        k,
        v,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=seq_len,
        total_tokens=capacity_tokens,
    )
    if pos_key is not None and pos_query is not None:
        pos_key_unpad, pos_query_unpad = prefix_pack_padded_rows_pair(
            pos_key,
            pos_query,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=seq_len,
            total_tokens=capacity_tokens,
        )
    else:
        pos_key_unpad = (
            prefix_pack_padded_rows(
                pos_key,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                max_seqlen=seq_len,
                total_tokens=capacity_tokens,
            )
            if pos_key is not None
            else None
        )
        pos_query_unpad = (
            prefix_pack_padded_rows(
                pos_query,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                max_seqlen=seq_len,
                total_tokens=capacity_tokens,
            )
            if pos_query is not None
            else None
        )

    override = _varlen_kernel_override_from_env(kind="fwd")
    if override is not None:
        block_m, block_n, num_stages, num_warps = override
    else:
        if _get_fwd_config_lowlevel is None:
            raise RuntimeError("FlashDeBERTa varlen forward config helper is unavailable.")
        block_m, block_n, num_stages, num_warps = _get_fwd_config_lowlevel(
            total_tokens=capacity_tokens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            D=head_dim,
            causal=bool(causal),
            disentangled=True,
            att_span=att_span,
        )

    mid_batch, mid_start, tile_count = _build_dense_mid_tensors(
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
        block_size=block_m,
    )
    out_unpad = torch.empty_like(q_unpad)
    lse_unpad = torch.empty((capacity_tokens, num_heads), device=q.device, dtype=torch.float32)

    if pos_key_unpad is not None:
        stride_pk0, stride_pk1, stride_pk2 = pos_key_unpad.stride()
    else:
        stride_pk0 = stride_pk1 = stride_pk2 = 0
    if pos_query_unpad is not None:
        stride_pq0, stride_pq1, stride_pq2 = pos_query_unpad.stride()
    else:
        stride_pq0 = stride_pq1 = stride_pq2 = 0

    grid = (tile_count, num_heads)
    torch.library.wrap_triton(_fwd_kernel_varlen_raw)[grid](
        q_unpad,
        k_unpad,
        v_unpad,
        pos_key_unpad,
        pos_query_unpad,
        lse_unpad,
        out_unpad,
        float(sm_scale),
        cu_seqlens,
        cu_seqlens,
        mid_batch,
        mid_start,
        q_unpad.stride(0),
        q_unpad.stride(1),
        q_unpad.stride(2),
        k_unpad.stride(0),
        k_unpad.stride(1),
        k_unpad.stride(2),
        v_unpad.stride(0),
        v_unpad.stride(1),
        v_unpad.stride(2),
        out_unpad.stride(0),
        out_unpad.stride(1),
        out_unpad.stride(2),
        stride_pk0,
        stride_pk1,
        stride_pk2,
        stride_pq0,
        stride_pq1,
        stride_pq2,
        batch_size,
        num_heads,
        seq_len,
        seq_len,
        BLOCK_M=block_m,
        BLOCK_DMODEL=head_dim,
        BLOCK_N=block_n,
        IS_CAUSAL=bool(causal),
        HAS_C2P=bool(pos_key_unpad is not None),
        HAS_P2C=bool(pos_query_unpad is not None),
        ATT_SPAN=att_span,
        NUM_BUCKETS=int(position_buckets),
        MAX_DISTANCE=int(max_relative_distance),
        num_warps=num_warps,
        num_stages=num_stages,
    )

    out_padded = prefix_unpack_padded_rows(
        out_unpad,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    lse_padded = prefix_unpack_padded_rows(
        lse_unpad,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
    ).contiguous()
    pos_key_saved = pos_key_unpad if pos_key_unpad is not None else q.new_empty((0,))
    pos_query_saved = pos_query_unpad if pos_query_unpad is not None else q.new_empty((0,))
    return (
        out_padded,
        lse_padded,
        cu_seqlens,
        q_unpad,
        k_unpad,
        v_unpad,
        out_unpad,
        lse_unpad,
        pos_key_saved,
        pos_query_saved,
    )


def _varlen_triton_backward_impl(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
    cu_seqlens: torch.Tensor | None = None,
    q_unpad: torch.Tensor | None = None,
    k_unpad: torch.Tensor | None = None,
    v_unpad: torch.Tensor | None = None,
    out_unpad: torch.Tensor | None = None,
    lse_unpad: torch.Tensor | None = None,
    pos_key_unpad: torch.Tensor | None = None,
    pos_query_unpad: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run compile-visible padded varlen backward with fixed-capacity packed buffers.

    :param torch.Tensor grad_out: Gradient of padded output.
    :param torch.Tensor q: Forward padded queries.
    :param torch.Tensor k: Forward padded keys.
    :param torch.Tensor v: Forward padded values.
    :param torch.Tensor mask: Boolean keep mask in ``(B, S)`` layout.
    :param torch.Tensor out: Forward padded output.
    :param torch.Tensor lse: Forward padded LSE tensor.
    :param torch.Tensor | None pos_key: Optional c2p tensor.
    :param torch.Tensor | None pos_query: Optional p2c tensor.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :param torch.Tensor | None cu_seqlens: Optional cached cumulative sequence lengths.
    :param torch.Tensor | None q_unpad: Optional cached packed queries.
    :param torch.Tensor | None k_unpad: Optional cached packed keys.
    :param torch.Tensor | None v_unpad: Optional cached packed values.
    :param torch.Tensor | None out_unpad: Optional cached packed forward output.
    :param torch.Tensor | None lse_unpad: Optional cached packed forward LSE.
    :param torch.Tensor | None pos_key_unpad: Optional cached packed c2p tensor.
    :param torch.Tensor | None pos_query_unpad: Optional cached packed p2c tensor.
    :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        Padded gradients for q/k/v and optional positional tensors.
    """

    if not _varlen_use_triton_op():
        raise RuntimeError("Compile-visible FlashDeBERTa varlen Triton support is unavailable.")

    batch_size = int(q.shape[0])
    seq_len = int(q.shape[1])
    capacity_tokens = int(batch_size * seq_len)

    if cu_seqlens is None:
        seqlens = mask.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    else:
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

    if q_unpad is None or k_unpad is None or v_unpad is None:
        q_unpad, k_unpad, v_unpad = prefix_pack_padded_rows_triple(
            q,
            k,
            v,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=seq_len,
            total_tokens=capacity_tokens,
        )
    if out_unpad is None:
        out_unpad = prefix_pack_padded_rows(
            out,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=seq_len,
            total_tokens=capacity_tokens,
        )
    out_unpad, grad_unpad, delta = _pack_grad_and_delta_from_padded(
        grad_output=grad_out,
        output_padded=out,
        out_unpad=out_unpad,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=seq_len,
        total_tokens=capacity_tokens,
    )
    if lse_unpad is None:
        lse_unpad = prefix_pack_padded_rows(
            lse,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=seq_len,
            total_tokens=capacity_tokens,
        )
    if pos_key is not None and pos_query is not None and (pos_key_unpad is None or pos_query_unpad is None):
        pos_key_unpad, pos_query_unpad = prefix_pack_padded_rows_pair(
            pos_key,
            pos_query,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=seq_len,
            total_tokens=capacity_tokens,
        )
    else:
        if pos_key is not None and pos_key_unpad is None:
            pos_key_unpad = prefix_pack_padded_rows(
                pos_key,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                max_seqlen=seq_len,
                total_tokens=capacity_tokens,
            )
        if pos_query is not None and pos_query_unpad is None:
            pos_query_unpad = prefix_pack_padded_rows(
                pos_query,
                seqlens=seqlens,
                cu_seqlens=cu_seqlens,
                max_seqlen=seq_len,
                total_tokens=capacity_tokens,
            )

    dq_unpad, dk_unpad, dv_unpad, dpos_key_unpad, dpos_query_unpad = _varlen_backward_raw_impl(
        q_unpad=q_unpad,
        k_unpad=k_unpad,
        v_unpad=v_unpad,
        out_unpad=out_unpad,
        grad_unpad=grad_unpad,
        lse_unpad=lse_unpad,
        delta=delta,
        pos_key_unpad=pos_key_unpad,
        pos_query_unpad=pos_query_unpad,
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_bound=seq_len,
        token_capacity=capacity_tokens,
        sm_scale=sm_scale,
        position_buckets=position_buckets,
        max_relative_distance=max_relative_distance,
        causal=causal,
        dense_mid_tensors=True,
    )

    dq, dk, dv = prefix_unpack_padded_rows_triple(
        dq_unpad,
        dk_unpad,
        dv_unpad,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    if dpos_key_unpad is not None and dpos_query_unpad is not None:
        dpos_key, dpos_query = prefix_unpack_padded_rows_pair(
            dpos_key_unpad,
            dpos_query_unpad,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            batch_size=batch_size,
            seq_len=seq_len,
        )
    else:
        dpos_key = None
        dpos_query = None
    return dq, dk, dv, dpos_key, dpos_query


def _build_varlen_triton_ops() -> tuple[Any | None, Any | None]:
    """Register or retrieve compile-visible padded-varlen Triton ops.

    :return tuple[Any | None, Any | None]: Forward and backward Triton-op handles.
    """

    if not _varlen_use_triton_op():
        return None, None

    existing_forward = _lookup_registered_op(_VARLEN_OP_NAMESPACE, f"{_VARLEN_FWD_OP_NAME}_triton")
    existing_backward = _lookup_registered_op(_VARLEN_OP_NAMESPACE, f"{_VARLEN_BWD_OP_NAME}_triton")
    if existing_forward is not None and existing_backward is not None:
        return existing_forward, existing_backward

    @torch.library.triton_op(
        f"{_VARLEN_OP_NAMESPACE}::{_VARLEN_FWD_OP_NAME}_triton",
        mutates_args=(),
        schema=(
            "(Tensor q, Tensor k, Tensor v, Tensor mask, Tensor? pos_key, Tensor? pos_query, "
            "float sm_scale, int position_buckets, int max_relative_distance, bool causal) -> "
            "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
        ),
    )
    def _forward_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        causal: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run compile-visible padded varlen forward.

        :param torch.Tensor q: Padded queries in ``(B, S, H, D)`` layout.
        :param torch.Tensor k: Padded keys in ``(B, S, H, D)`` layout.
        :param torch.Tensor v: Padded values in ``(B, S, H, D)`` layout.
        :param torch.Tensor mask: Boolean keep mask in ``(B, S)`` layout.
        :param torch.Tensor | None pos_key: Optional c2p tensor.
        :param torch.Tensor | None pos_query: Optional p2c tensor.
        :param float sm_scale: Softmax scale.
        :param int position_buckets: Relative-position bucket count.
        :param int max_relative_distance: Maximum relative distance.
        :param bool causal: Whether causal masking is enabled.
        :return tuple[torch.Tensor, ...]:
            Padded output, padded LSE, cumulative sequence lengths, packed q/k/v,
            packed output/LSE, and optional packed positional tensors.
        """

        return _varlen_triton_forward_impl(
            q=q,
            k=k,
            v=v,
            mask=mask,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            position_buckets=position_buckets,
            max_relative_distance=max_relative_distance,
            causal=causal,
        )

    @torch.library.triton_op(
        f"{_VARLEN_OP_NAMESPACE}::{_VARLEN_BWD_OP_NAME}_triton",
        mutates_args=(),
        schema=(
            "(Tensor grad_out, Tensor q, Tensor k, Tensor v, Tensor mask, Tensor out, Tensor lse, "
            "Tensor? pos_key, Tensor? pos_query, float sm_scale, int position_buckets, "
            "int max_relative_distance, bool causal, Tensor cu_seqlens, Tensor q_unpad, Tensor k_unpad, "
            "Tensor v_unpad, Tensor out_unpad, Tensor lse_unpad, Tensor? pos_key_unpad, "
            "Tensor? pos_query_unpad) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
        ),
    )
    def _backward_op(
        grad_out: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        causal: bool,
        cu_seqlens: torch.Tensor,
        q_unpad: torch.Tensor,
        k_unpad: torch.Tensor,
        v_unpad: torch.Tensor,
        out_unpad: torch.Tensor,
        lse_unpad: torch.Tensor,
        pos_key_unpad: torch.Tensor | None,
        pos_query_unpad: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run compile-visible padded varlen backward.

        :param torch.Tensor grad_out: Gradient of padded output.
        :param torch.Tensor q: Forward padded queries.
        :param torch.Tensor k: Forward padded keys.
        :param torch.Tensor v: Forward padded values.
        :param torch.Tensor mask: Boolean keep mask in ``(B, S)`` layout.
        :param torch.Tensor out: Forward padded output.
        :param torch.Tensor lse: Forward padded LSE tensor.
        :param torch.Tensor | None pos_key: Optional c2p tensor.
        :param torch.Tensor | None pos_query: Optional p2c tensor.
        :param float sm_scale: Softmax scale.
        :param int position_buckets: Relative-position bucket count.
        :param int max_relative_distance: Maximum relative distance.
        :param bool causal: Whether causal masking is enabled.
        :param torch.Tensor cu_seqlens: Cached cumulative sequence lengths.
        :param torch.Tensor q_unpad: Cached packed queries.
        :param torch.Tensor k_unpad: Cached packed keys.
        :param torch.Tensor v_unpad: Cached packed values.
        :param torch.Tensor out_unpad: Cached packed forward output.
        :param torch.Tensor lse_unpad: Cached packed forward LSE.
        :param torch.Tensor | None pos_key_unpad: Cached packed c2p tensor.
        :param torch.Tensor | None pos_query_unpad: Cached packed p2c tensor.
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Padded q/k/v gradients plus positional gradients with empty sentinels
            for absent positional inputs.
        """

        dq, dk, dv, dpos_key, dpos_query = _varlen_triton_backward_impl(
            grad_out=grad_out,
            q=q,
            k=k,
            v=v,
            mask=mask,
            out=out,
            lse=lse,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            position_buckets=position_buckets,
            max_relative_distance=max_relative_distance,
            causal=causal,
            cu_seqlens=cu_seqlens,
            q_unpad=q_unpad,
            k_unpad=k_unpad,
            v_unpad=v_unpad,
            out_unpad=out_unpad,
            lse_unpad=lse_unpad,
            pos_key_unpad=pos_key_unpad,
            pos_query_unpad=pos_query_unpad,
        )
        if dpos_key is None:
            dpos_key = q.new_empty((0,))
        if dpos_query is None:
            dpos_query = q.new_empty((0,))
        return dq, dk, dv, dpos_key, dpos_query

    def _setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        output: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        """Save forward inputs and outputs needed by compile-visible varlen backward.

        :param Any ctx: Autograd context object.
        :param tuple[Any, ...] inputs: Forward Triton-op inputs.
        :param tuple[torch.Tensor, ...] output: Forward Triton-op outputs.
        """

        q, k, v, mask, pos_key, pos_query, sm_scale, position_buckets, max_relative_distance, causal = inputs
        (
            out,
            lse,
            cu_seqlens,
            q_unpad,
            k_unpad,
            v_unpad,
            out_unpad,
            lse_unpad,
            pos_key_unpad,
            pos_query_unpad,
        ) = output
        saved: list[torch.Tensor] = [
            q,
            k,
            v,
            mask,
            out,
            lse,
            cu_seqlens,
            q_unpad,
            k_unpad,
            v_unpad,
            out_unpad,
            lse_unpad,
        ]
        ctx.has_pos_key = pos_key is not None
        ctx.has_pos_query = pos_query is not None
        if bool(ctx.has_pos_key):
            saved.append(pos_key_unpad)
        if bool(ctx.has_pos_query):
            saved.append(pos_query_unpad)
        if hasattr(ctx, "mark_non_differentiable"):
            non_diff = [lse, cu_seqlens, q_unpad, k_unpad, v_unpad, out_unpad, lse_unpad]
            if bool(ctx.has_pos_key):
                non_diff.append(pos_key_unpad)
            if bool(ctx.has_pos_query):
                non_diff.append(pos_query_unpad)
            ctx.mark_non_differentiable(*non_diff)
        ctx.save_for_backward(*saved)
        ctx.sm_scale = float(sm_scale)
        ctx.position_buckets = int(position_buckets)
        ctx.max_relative_distance = int(max_relative_distance)
        ctx.causal = bool(causal)

    def _backward(
        ctx: Any,
        grad_out: torch.Tensor | None,
        grad_lse: torch.Tensor | None,
        grad_cu_seqlens: torch.Tensor | None,
        grad_q_unpad: torch.Tensor | None,
        grad_k_unpad: torch.Tensor | None,
        grad_v_unpad: torch.Tensor | None,
        grad_out_unpad: torch.Tensor | None,
        grad_lse_unpad: torch.Tensor | None,
        grad_pos_key_unpad: torch.Tensor | None,
        grad_pos_query_unpad: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Dispatch backward through compile-visible padded varlen backward.

        :param Any ctx: Autograd context populated by ``_setup_context``.
        :param torch.Tensor | None grad_out: Gradient of padded output.
        :param torch.Tensor | None grad_lse: Gradient of padded LSE tensor.
        :param torch.Tensor | None grad_cu_seqlens: Ignored gradient of cached cumulative lengths.
        :param torch.Tensor | None grad_q_unpad: Ignored gradient of cached packed queries.
        :param torch.Tensor | None grad_k_unpad: Ignored gradient of cached packed keys.
        :param torch.Tensor | None grad_v_unpad: Ignored gradient of cached packed values.
        :param torch.Tensor | None grad_out_unpad: Ignored gradient of cached packed outputs.
        :param torch.Tensor | None grad_lse_unpad: Ignored gradient of cached packed LSE values.
        :param torch.Tensor | None grad_pos_key_unpad: Ignored gradient of cached packed c2p tensor.
        :param torch.Tensor | None grad_pos_query_unpad: Ignored gradient of cached packed p2c tensor.
        :return tuple[torch.Tensor | None, ...]: Gradients for the forward inputs.
        """

        del (
            grad_lse,
            grad_cu_seqlens,
            grad_q_unpad,
            grad_k_unpad,
            grad_v_unpad,
            grad_out_unpad,
            grad_lse_unpad,
            grad_pos_key_unpad,
            grad_pos_query_unpad,
        )
        saved = list(ctx.saved_tensors)
        q, k, v, mask, out, lse, cu_seqlens, q_unpad, k_unpad, v_unpad, out_unpad, lse_unpad = saved[:12]
        next_idx = 12
        pos_key_unpad = saved[next_idx] if bool(ctx.has_pos_key) else None
        if bool(ctx.has_pos_key):
            next_idx += 1
        pos_query_unpad = saved[next_idx] if bool(ctx.has_pos_query) else None
        grad = grad_out if grad_out is not None else torch.zeros_like(out)
        dq, dk, dv, dpos_key, dpos_query = _backward_op(
            grad,
            q,
            k,
            v,
            mask,
            out,
            lse,
            None,
            None,
            ctx.sm_scale,
            ctx.position_buckets,
            ctx.max_relative_distance,
            ctx.causal,
            cu_seqlens,
            q_unpad,
            k_unpad,
            v_unpad,
            out_unpad,
            lse_unpad,
            pos_key_unpad if bool(ctx.has_pos_key) else None,
            pos_query_unpad if bool(ctx.has_pos_query) else None,
        )
        if not bool(ctx.has_pos_key):
            dpos_key = None
        if not bool(ctx.has_pos_query):
            dpos_query = None
        return dq, dk, dv, None, dpos_key, dpos_query, None, None, None, None

    torch.library.register_autograd(_forward_op, _backward, setup_context=_setup_context)
    return _forward_op, _backward_op


_FLASHDEBERTA_VARLEN_TRITON_OP, _FLASHDEBERTA_VARLEN_TRITON_BWD_OP = _build_varlen_triton_ops()


def flashdeberta_varlen_padded(
    *,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
    causal: bool,
) -> torch.Tensor:
    """Run padded varlen FlashDeBERTa attention.

    On CUDA, eager execution prefers the cached custom-op path because it can
    reuse forward-side unpadding metadata across backward. Compiled execution
    prefers the compile-visible Triton-op path and carries packed forward aux
    tensors through autograd so backward does not repack the full forward state.

    :param torch.Tensor query_layer: Queries in ``(B, S, H, D)`` layout.
    :param torch.Tensor key_layer: Keys in ``(B, S, H, D)`` layout.
    :param torch.Tensor value_layer: Values in ``(B, S, H, D)`` layout.
    :param torch.Tensor attention_mask_2d: Boolean keep mask in ``(B, S)`` layout.
    :param torch.Tensor | None pos_key: Optional c2p tensor in ``(B, S, H, P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c tensor in ``(B, S, H, P)`` layout.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :return torch.Tensor: Attention output in ``(B, S, H, D)`` layout.
    """

    if (
        _is_torch_compiling()
        and _FLASHDEBERTA_VARLEN_TRITON_OP is not None
        and query_layer.device.type == "cuda"
    ):
        output, *_ = _FLASHDEBERTA_VARLEN_TRITON_OP(
            query_layer,
            key_layer,
            value_layer,
            attention_mask_2d,
            pos_key,
            pos_query,
            float(sm_scale),
            int(position_buckets),
            int(max_relative_distance),
            bool(causal),
        )
        return output

    if _FLASHDEBERTA_VARLEN_CUSTOM_OP is not None and query_layer.device.type == "cuda":
        output, _ = _FLASHDEBERTA_VARLEN_CUSTOM_OP(
            query_layer,
            key_layer,
            value_layer,
            attention_mask_2d,
            pos_key,
            pos_query,
            float(sm_scale),
            int(position_buckets),
            int(max_relative_distance),
            bool(causal),
        )
        return output

    output, _ = _varlen_eager_forward_impl(
        query_layer=query_layer,
        key_layer=key_layer,
        value_layer=value_layer,
        attention_mask_2d=attention_mask_2d,
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
    "flashdeberta_compiled_varlen_available",
    "flashdeberta_varlen_import_error",
    "flashdeberta_varlen_padded",
]
