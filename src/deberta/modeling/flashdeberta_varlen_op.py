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

try:
    from flashdeberta.ops.flash_attention_varlen import (
        flash_attention_with_disentangled_varlen as _flash_attention_with_disentangled_varlen_highlevel,
    )

    _FLASH_VARLEN_HIGHLEVEL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional import
    _flash_attention_with_disentangled_varlen_highlevel = None
    _FLASH_VARLEN_HIGHLEVEL_IMPORT_ERROR = exc

try:
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
    _flash_attn_v2_bwd_dise_varlen_lowlevel = None
    _flash_attn_v2_fwd_dise_lowlevel = None
    _get_bwd_config_varlen_lowlevel = None
    _get_fwd_config_lowlevel = None
    _FLASH_VARLEN_LOWLEVEL_IMPORT_ERROR = exc

_VARLEN_OP_NAMESPACE = "deberta"
_VARLEN_FWD_OP_NAME = "flashdeberta_varlen_padded"
_VARLEN_BWD_OP_NAME = "flashdeberta_varlen_padded_backward"


@dataclass
class _MaskMetadataCacheEntry:
    """Cached unpadding metadata for one padding-mask tensor."""

    mask_ref: weakref.ReferenceType[torch.Tensor] | None
    version: int
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    batch_indices: torch.Tensor
    seq_indices: torch.Tensor


@dataclass
class _ForwardAuxCacheEntry:
    """Forward-side varlen tensors reused by the padded backward helper."""

    output_ref: weakref.ReferenceType[torch.Tensor] | None
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    batch_indices: torch.Tensor
    seq_indices: torch.Tensor
    q_unpad: torch.Tensor
    k_unpad: torch.Tensor
    v_unpad: torch.Tensor
    out_unpad: torch.Tensor
    lse_unpad: torch.Tensor
    pos_key_unpad: torch.Tensor | None
    pos_query_unpad: torch.Tensor | None


_MASK_METADATA_CACHE: dict[int, _MaskMetadataCacheEntry] = {}
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
    """Return whether the opaque compiled varlen CUDA op is available.

    :return bool: True when the custom-op based CUDA path is registered.
    """

    return _FLASHDEBERTA_VARLEN_CUSTOM_OP is not None and _FLASHDEBERTA_VARLEN_BWD_CUSTOM_OP is not None


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

    :param str kind: Either ``"fwd"`` or ``"bwd"``.
    :return tuple[int, int, int, int] | None: Override ``(BLOCK_M, BLOCK_N, stages, warps)``
        or ``None`` when unset / invalid / incomplete.
    """

    normalized = str(kind).strip().lower()
    if normalized not in {"fwd", "bwd"}:
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


def _build_unpad_metadata(
    mask_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Build flattened-token indices and cumulative lengths for a 2D keep mask.

    :param torch.Tensor mask_2d: Boolean mask with shape ``(B, S)``.
    :return tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        Valid-token indices, cumulative lengths, max sequence length, and cached
        ``(batch_idx, seq_idx)`` selectors for ``(B, H, S, D)`` tensors.
    """

    seqlens = mask_2d.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(mask_2d.reshape(-1), as_tuple=False).squeeze(-1)
    max_seqlen = int(seqlens.max().item()) if int(seqlens.numel()) > 0 else 0
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    seq_len = int(mask_2d.shape[-1])
    batch_indices = torch.div(indices, seq_len, rounding_mode="floor")
    seq_indices = torch.remainder(indices, seq_len)
    return indices, cu_seqlens, max_seqlen, batch_indices, seq_indices


def _mask_version(mask_2d: torch.Tensor) -> int:
    """Return a best-effort mutation version for one mask tensor.

    :param torch.Tensor mask_2d: Boolean padding mask.
    :return int: Tensor version counter when available.
    """

    try:
        return int(getattr(mask_2d, "_version", 0))
    except Exception:
        return 0


def _clear_unpad_metadata_cache() -> None:
    """Clear cached unpadding metadata.

    This exists primarily for tests.
    """

    _MASK_METADATA_CACHE.clear()


def _clear_forward_aux_cache() -> None:
    """Clear cached forward tensors reused by the padded backward helper.

    This exists primarily for tests.
    """

    _FORWARD_AUX_CACHE.clear()


def _get_unpad_metadata_entry(mask_2d: torch.Tensor) -> _MaskMetadataCacheEntry:
    """Return cached unpadding metadata entry for one mask tensor.

    :param torch.Tensor mask_2d: Boolean keep mask with shape ``(B, S)``.
    :return _MaskMetadataCacheEntry: Cached or newly built metadata entry.
    """

    cache_key = id(mask_2d)
    current_version = _mask_version(mask_2d)
    cached = _MASK_METADATA_CACHE.get(cache_key)
    if cached is not None:
        cached_mask = cached.mask_ref() if cached.mask_ref is not None else None
        if cached_mask is mask_2d and cached.version == current_version:
            return cached
        _MASK_METADATA_CACHE.pop(cache_key, None)

    indices, cu_seqlens, max_seqlen, batch_indices, seq_indices = _build_unpad_metadata(mask_2d)

    mask_ref: weakref.ReferenceType[torch.Tensor] | None = None
    try:
        mask_ref = weakref.ref(mask_2d, lambda _ref, key=cache_key: _MASK_METADATA_CACHE.pop(key, None))
    except TypeError:
        mask_ref = None

    entry = _MaskMetadataCacheEntry(
        mask_ref=mask_ref,
        version=current_version,
        indices=indices,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        batch_indices=batch_indices,
        seq_indices=seq_indices,
    )
    if mask_ref is not None:
        _MASK_METADATA_CACHE[cache_key] = entry
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
    return entry.indices, entry.cu_seqlens, entry.max_seqlen


def _flatten_valid_tokens(
    tensor: torch.Tensor,
    *,
    batch_indices: torch.Tensor,
    seq_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather valid tokens from ``(B, H, S, D)`` padded layout without a full transpose copy.

    :param torch.Tensor tensor: Padded tensor with leading shape ``(B, H, S, D)``.
    :param torch.Tensor batch_indices: Batch selector for each valid token.
    :param torch.Tensor seq_indices: Sequence-position selector for each valid token.
    :return torch.Tensor: Gathered tensor with leading shape ``(NNZ, ...)``.
    """

    return tensor.transpose(1, 2)[batch_indices, seq_indices]


def _flatten_valid_lse(
    tensor: torch.Tensor,
    *,
    batch_indices: torch.Tensor,
    seq_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather valid padded LSE values from ``(B, S, H)`` layout.

    :param torch.Tensor tensor: Padded LSE tensor with leading shape ``(B, S, H)``.
    :param torch.Tensor batch_indices: Batch selector for each valid token.
    :param torch.Tensor seq_indices: Sequence-position selector for each valid token.
    :return torch.Tensor: Gathered tensor with leading shape ``(NNZ, H)``.
    """

    return tensor[batch_indices, seq_indices]


def _pad_valid_tokens(
    values: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Scatter unpadded values back into a padded batch layout.

    :param torch.Tensor values: Unpadded values with leading shape ``(NNZ, ...)``.
    :param torch.Tensor indices: Flattened valid-token indices.
    :param int batch_size: Batch size.
    :param int seq_len: Padded sequence length.
    :return torch.Tensor: Padded tensor with leading shape ``(B, S, ...)``.
    """

    flat_out = values.new_zeros((batch_size * seq_len, *values.shape[1:]))
    flat_out = flat_out.index_copy(0, indices, values)
    return flat_out.view(batch_size, seq_len, *values.shape[1:])


def _store_forward_aux_cache(
    *,
    output_padded: torch.Tensor,
    indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    batch_indices: torch.Tensor,
    seq_indices: torch.Tensor,
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
    :param torch.Tensor indices: Flattened valid-token indices.
    :param torch.Tensor cu_seqlens: Cumulative sequence lengths.
    :param int max_seqlen: Maximum active length in batch.
    :param torch.Tensor batch_indices: Batch selector for each valid token.
    :param torch.Tensor seq_indices: Sequence-position selector for each valid token.
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
        indices=indices,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        batch_indices=batch_indices,
        seq_indices=seq_indices,
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

    :param torch.Tensor query_layer: Queries in ``(B, H, S, D)`` layout.
    :param torch.Tensor key_layer: Keys in ``(B, H, S, D)`` layout.
    :param torch.Tensor value_layer: Values in ``(B, H, S, D)`` layout.
    :param torch.Tensor attention_mask_2d: Boolean keep mask in ``(B, S)`` layout.
    :param torch.Tensor | None pos_key: Optional c2p tensor in ``(B, H, S, P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c tensor in ``(B, H, S, P)`` layout.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :param bool require_lse: Whether the caller also needs padded log-sum-exp values.
    :param bool stash_backward_cache: Whether to cache unpadded forward tensors for reuse in backward.
    :raises RuntimeError: If no eager varlen implementation is importable.
    :return tuple[torch.Tensor, torch.Tensor | None]: Padded output in ``(B, H, S, D)``
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
    seq_len = int(query_layer.shape[-2])
    att_span = int(position_buckets) if int(position_buckets) > 0 else int(max_relative_distance)

    metadata = _get_unpad_metadata_entry(attention_mask_2d)
    indices = metadata.indices
    cu_seqlens = metadata.cu_seqlens
    max_seqlen = metadata.max_seqlen
    if int(indices.numel()) == 0:
        output = torch.zeros_like(query_layer)
        if require_lse:
            lse = torch.zeros(
                (batch_size, seq_len, int(query_layer.shape[1])),
                device=query_layer.device,
                dtype=torch.float32,
            )
            return output, lse
        return output, None

    q_unpad = _flatten_valid_tokens(
        query_layer,
        batch_indices=metadata.batch_indices,
        seq_indices=metadata.seq_indices,
    )
    k_unpad = _flatten_valid_tokens(
        key_layer,
        batch_indices=metadata.batch_indices,
        seq_indices=metadata.seq_indices,
    )
    v_unpad = _flatten_valid_tokens(
        value_layer,
        batch_indices=metadata.batch_indices,
        seq_indices=metadata.seq_indices,
    )
    pos_key_unpad = (
        _flatten_valid_tokens(
            pos_key,
            batch_indices=metadata.batch_indices,
            seq_indices=metadata.seq_indices,
        )
        if pos_key is not None
        else None
    )
    pos_query_unpad = (
        _flatten_valid_tokens(
            pos_query,
            batch_indices=metadata.batch_indices,
            seq_indices=metadata.seq_indices,
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

    out_padded = _pad_valid_tokens(out_unpad, indices, batch_size, seq_len)
    out_padded = out_padded.permute(0, 2, 1, 3).contiguous()

    if not require_lse:
        return out_padded, None
    if lse_unpad is None:
        raise RuntimeError(
            "Compiled FlashDeBERTa varlen requires low-level forward primitives with padded LSE support."
        )

    lse_padded = _pad_valid_tokens(lse_unpad, indices, batch_size, seq_len).contiguous()
    if stash_backward_cache:
        _store_forward_aux_cache(
            output_padded=out_padded,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_indices=metadata.batch_indices,
            seq_indices=metadata.seq_indices,
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

    :param torch.Tensor grad_output: Gradient of padded output in ``(B, H, S, D)`` layout.
    :param torch.Tensor query_layer: Forward queries in ``(B, H, S, D)`` layout.
    :param torch.Tensor key_layer: Forward keys in ``(B, H, S, D)`` layout.
    :param torch.Tensor value_layer: Forward values in ``(B, H, S, D)`` layout.
    :param torch.Tensor attention_mask_2d: Boolean keep mask in ``(B, S)`` layout.
    :param torch.Tensor output_padded: Forward output in ``(B, H, S, D)`` layout.
    :param torch.Tensor lse_padded: Forward padded LSE in ``(B, S, H)`` layout.
    :param torch.Tensor | None pos_key: Optional c2p tensor in ``(B, H, S, P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c tensor in ``(B, H, S, P)`` layout.
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
        indices=metadata.indices,
        cu_seqlens=metadata.cu_seqlens,
        max_seqlen=metadata.max_seqlen,
        batch_indices=metadata.batch_indices,
        seq_indices=metadata.seq_indices,
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
    indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    batch_indices: torch.Tensor,
    seq_indices: torch.Tensor,
    q_unpad: torch.Tensor | None,
    k_unpad: torch.Tensor | None,
    v_unpad: torch.Tensor | None,
    out_unpad: torch.Tensor | None,
    lse_unpad: torch.Tensor | None,
    pos_key_unpad: torch.Tensor | None,
    pos_query_unpad: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run padded varlen backward, optionally reusing forward-side unpadded tensors.

    :param torch.Tensor grad_output: Gradient of padded output in ``(B, H, S, D)`` layout.
    :param torch.Tensor query_layer: Forward queries in ``(B, H, S, D)`` layout.
    :param torch.Tensor key_layer: Forward keys in ``(B, H, S, D)`` layout.
    :param torch.Tensor value_layer: Forward values in ``(B, H, S, D)`` layout.
    :param torch.Tensor output_padded: Forward output in ``(B, H, S, D)`` layout.
    :param torch.Tensor lse_padded: Forward padded LSE in ``(B, S, H)`` layout.
    :param torch.Tensor | None pos_key: Optional c2p tensor in ``(B, H, S, P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c tensor in ``(B, H, S, P)`` layout.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :param torch.Tensor indices: Flattened valid-token indices.
    :param torch.Tensor cu_seqlens: Cumulative sequence lengths.
    :param int max_seqlen: Maximum active length in batch.
    :param torch.Tensor batch_indices: Batch selector for each valid token.
    :param torch.Tensor seq_indices: Sequence-position selector for each valid token.
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
    seq_len = int(query_layer.shape[-2])
    att_span = int(position_buckets) if int(position_buckets) > 0 else int(max_relative_distance)

    if int(indices.numel()) == 0:
        dq = torch.zeros_like(query_layer)
        dk = torch.zeros_like(key_layer)
        dv = torch.zeros_like(value_layer)
        dpos_key = torch.zeros_like(pos_key) if pos_key is not None else None
        dpos_query = torch.zeros_like(pos_query) if pos_query is not None else None
        return dq, dk, dv, dpos_key, dpos_query

    if q_unpad is None:
        q_unpad = _flatten_valid_tokens(
            query_layer,
            batch_indices=batch_indices,
            seq_indices=seq_indices,
        )
    if k_unpad is None:
        k_unpad = _flatten_valid_tokens(
            key_layer,
            batch_indices=batch_indices,
            seq_indices=seq_indices,
        )
    if v_unpad is None:
        v_unpad = _flatten_valid_tokens(
            value_layer,
            batch_indices=batch_indices,
            seq_indices=seq_indices,
        )
    if out_unpad is None:
        out_unpad = _flatten_valid_tokens(
            output_padded,
            batch_indices=batch_indices,
            seq_indices=seq_indices,
        )
    grad_unpad = _flatten_valid_tokens(
        grad_output,
        batch_indices=batch_indices,
        seq_indices=seq_indices,
    )
    if lse_unpad is None:
        lse_unpad = _flatten_valid_lse(
            lse_padded,
            batch_indices=batch_indices,
            seq_indices=seq_indices,
        )
    if pos_key is not None and pos_key_unpad is None:
        pos_key_unpad = _flatten_valid_tokens(
            pos_key,
            batch_indices=batch_indices,
            seq_indices=seq_indices,
        )
    if pos_query is not None and pos_query_unpad is None:
        pos_query_unpad = _flatten_valid_tokens(
            pos_query,
            batch_indices=batch_indices,
            seq_indices=seq_indices,
        )

    override = _varlen_kernel_override_from_env(kind="bwd")
    if override is not None:
        block_m, block_n, num_stages, num_warps = override
    else:
        block_m, block_n, num_stages, num_warps = _get_bwd_config_varlen_lowlevel(
            total_tokens_q=int(q_unpad.shape[0]),
            total_tokens_k=int(k_unpad.shape[0]),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            D=int(query_layer.shape[-1]),
            causal=bool(causal),
            disentangled=True,
            att_span=att_span,
            dtype=query_layer.dtype,
        )
    dq_unpad, dk_unpad, dv_unpad, dpos_key_unpad, dpos_query_unpad = _flash_attn_v2_bwd_dise_varlen_lowlevel(
        out_unpad,
        grad_unpad,
        q_unpad,
        k_unpad,
        v_unpad,
        pos_key_unpad,
        pos_query_unpad,
        lse_unpad,
        cu_seqlens,
        cu_seqlens,
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

    dq = _pad_valid_tokens(dq_unpad, indices, batch_size, seq_len).permute(0, 2, 1, 3).contiguous()
    dk = _pad_valid_tokens(dk_unpad, indices, batch_size, seq_len).permute(0, 2, 1, 3).contiguous()
    dv = _pad_valid_tokens(dv_unpad, indices, batch_size, seq_len).permute(0, 2, 1, 3).contiguous()
    dpos_key = (
        _pad_valid_tokens(dpos_key_unpad, indices, batch_size, seq_len).permute(0, 2, 1, 3).contiguous()
        if dpos_key_unpad is not None
        else None
    )
    dpos_query = (
        _pad_valid_tokens(dpos_query_unpad, indices, batch_size, seq_len).permute(0, 2, 1, 3).contiguous()
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

        :param torch.Tensor q: Padded queries in ``(B, H, S, D)`` layout.
        :param torch.Tensor k: Padded keys in ``(B, H, S, D)`` layout.
        :param torch.Tensor v: Padded values in ``(B, H, S, D)`` layout.
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
            (q.shape[0], q.shape[2], q.shape[1]),
            device=q.device,
            dtype=torch.float32,
        )
        return torch.empty_like(q), lse

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
        dpos_key = torch.empty_like(pos_key) if pos_key is not None else None
        dpos_query = torch.empty_like(pos_query) if pos_query is not None else None
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), dpos_key, dpos_query

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
                indices=cached.indices,
                cu_seqlens=cached.cu_seqlens,
                max_seqlen=cached.max_seqlen,
                batch_indices=cached.batch_indices,
                seq_indices=cached.seq_indices,
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

    On CUDA with the low-level FlashDeBERTa primitives available, this uses the
    opaque custom-op path so ``torch.compile`` does not trace through the
    upstream Python/Triton launcher. Otherwise it falls back to the eager Python
    implementation.

    :param torch.Tensor query_layer: Queries in ``(B, H, S, D)`` layout.
    :param torch.Tensor key_layer: Keys in ``(B, H, S, D)`` layout.
    :param torch.Tensor value_layer: Values in ``(B, H, S, D)`` layout.
    :param torch.Tensor attention_mask_2d: Boolean keep mask in ``(B, S)`` layout.
    :param torch.Tensor | None pos_key: Optional c2p tensor in ``(B, H, S, P)`` layout.
    :param torch.Tensor | None pos_query: Optional p2c tensor in ``(B, H, S, P)`` layout.
    :param float sm_scale: Softmax scale.
    :param int position_buckets: Relative-position bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param bool causal: Whether causal masking is enabled.
    :return torch.Tensor: Attention output in ``(B, H, S, D)`` layout.
    """

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
