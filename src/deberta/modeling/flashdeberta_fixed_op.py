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
    att_span = _fixed_attention_span(position_buckets, max_relative_distance)

    if _flash_attn_v2_fwd_dise_lowlevel is not None and _get_fwd_config_lowlevel is not None:
        override = _fixed_kernel_override_from_env(kind="fwd")
        tuned = _fixed_repo_tuned_config(
            kind="fwd",
            query_len=query_len,
            key_len=key_len,
            head_dim=head_dim,
            causal=bool(causal),
            disentangled=(pos_key is not None or pos_query is not None),
            att_span=att_span,
            dtype=query_layer.dtype,
            device=query_layer.device,
        )
        if override is not None:
            block_m, block_n, num_stages, num_warps = override
        elif tuned is not None:
            block_m, block_n, num_stages, num_warps = tuned
        else:
            block_m, block_n, num_stages, num_warps = _get_fwd_config_lowlevel(
                batch_size,
                num_heads,
                query_len,
                key_len,
                head_dim,
                bool(causal),
                disentangled=True,
                att_span=att_span,
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

    override = _fixed_kernel_override_from_env(kind="bwd")
    tuned = _fixed_repo_tuned_config(
        kind="bwd",
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
        causal=bool(causal),
        disentangled=(pos_key is not None or pos_query is not None),
        att_span=att_span,
        dtype=query_layer.dtype,
        device=query_layer.device,
    )
    if override is not None:
        block_m, block_n, num_stages, num_warps = override
    elif tuned is not None:
        block_m, block_n, num_stages, num_warps = tuned
    else:
        block_m, block_n, num_stages, num_warps = _get_bwd_config_lowlevel(
            batch_size,
            num_heads,
            query_len,
            key_len,
            head_dim,
            bool(causal),
            disentangled=(pos_key is not None or pos_query is not None),
            att_span=att_span,
            dtype=query_layer.dtype,
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


def _build_fixed_custom_ops() -> tuple[Any | None, Any | None]:
    """Register or retrieve the opaque fixed-length custom ops.

    :return tuple[Any | None, Any | None]: Forward and backward custom-op handles.
    """

    existing_forward = _lookup_registered_op(_FIXED_OP_NAMESPACE, _FIXED_FWD_OP_NAME)
    existing_backward = _lookup_registered_op(_FIXED_OP_NAMESPACE, _FIXED_BWD_OP_NAME)
    if existing_forward is not None and existing_backward is not None:
        return existing_forward, existing_backward

    if (
        _flash_attn_v2_fwd_dise_lowlevel is None
        or _flash_attn_v2_bwd_dise_lowlevel is None
        or not hasattr(torch, "library")
        or not hasattr(torch.library, "custom_op")
    ):
        return None, None

    @torch.library.custom_op(
        f"{_FIXED_OP_NAMESPACE}::{_FIXED_FWD_OP_NAME}",
        mutates_args=(),
        device_types="cuda",
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

        output, lse = _fixed_eager_forward_impl(
            query_layer=q,
            key_layer=k,
            value_layer=v,
            seq_lengths=seq_lengths,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            position_buckets=position_buckets,
            max_relative_distance=max_relative_distance,
            causal=causal,
            require_lse=True,
        )
        if lse is None:  # pragma: no cover - guarded by require_lse
            raise RuntimeError("Fixed FlashDeBERTa custom op requires forward LSE output.")
        return output, lse

    @torch.library.register_fake(_forward_op)
    def _forward_op_fake(
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
        """Return fake fixed-length outputs with static shapes.

        :param torch.Tensor q: Fake query tensor.
        :param torch.Tensor k: Fake key tensor.
        :param torch.Tensor v: Fake value tensor.
        :param torch.Tensor | None seq_lengths: Fake optional sequence lengths.
        :param torch.Tensor | None pos_key: Fake optional c2p tensor.
        :param torch.Tensor | None pos_query: Fake optional p2c tensor.
        :param float sm_scale: Fake softmax scale.
        :param int position_buckets: Fake relative-position bucket count.
        :param int max_relative_distance: Fake maximum relative distance.
        :param bool causal: Fake causal flag.
        :return tuple[torch.Tensor, torch.Tensor]: Fake output and LSE tensors.
        """

        del (
            k,
            v,
            seq_lengths,
            pos_key,
            pos_query,
            sm_scale,
            position_buckets,
            max_relative_distance,
            causal,
        )
        lse = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        return torch.empty_like(q), lse

    @torch.library.custom_op(
        f"{_FIXED_OP_NAMESPACE}::{_FIXED_BWD_OP_NAME}",
        mutates_args=(),
        device_types="cuda",
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Run fixed-length backward as one opaque CUDA op.

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
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            Gradients for q/k/v and optional positional tensors.
        """

        return _fixed_eager_backward_impl(
            grad_output=grad_out,
            query_layer=q,
            key_layer=k,
            value_layer=v,
            seq_lengths=seq_lengths,
            output=out,
            lse=lse,
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
        """Return fake fixed-length backward outputs with static shapes.

        :param torch.Tensor grad_out: Fake gradient tensor.
        :param torch.Tensor q: Fake query tensor.
        :param torch.Tensor k: Fake key tensor.
        :param torch.Tensor v: Fake value tensor.
        :param torch.Tensor | None seq_lengths: Fake optional sequence lengths.
        :param torch.Tensor out: Fake output tensor.
        :param torch.Tensor lse: Fake LSE tensor.
        :param torch.Tensor | None pos_key: Fake optional c2p tensor.
        :param torch.Tensor | None pos_query: Fake optional p2c tensor.
        :param float sm_scale: Fake softmax scale.
        :param int position_buckets: Fake relative-position bucket count.
        :param int max_relative_distance: Fake maximum relative distance.
        :param bool causal: Fake causal flag.
        :return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            Fake gradients for q/k/v and optional positional tensors.
        """

        del grad_out, seq_lengths, out, lse, sm_scale, position_buckets, max_relative_distance, causal
        dpos_key = torch.empty_like(pos_key) if pos_key is not None else None
        dpos_query = torch.empty_like(pos_query) if pos_query is not None else None
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), dpos_key, dpos_query

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
        return dq, dk, dv, None, dpos_key, dpos_query, None, None, None, None

    torch.library.register_autograd(_forward_op, _backward, setup_context=_setup_context)
    return _forward_op, _backward_op


_FLASHDEBERTA_FIXED_CUSTOM_OP, _FLASHDEBERTA_FIXED_BWD_CUSTOM_OP = _build_fixed_custom_ops()


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
