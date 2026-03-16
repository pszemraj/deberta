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
    if _get_fwd_config_bias_lowlevel is None:
        raise RuntimeError("FlashDeBERTa local-bias config helper is unavailable.")
    block_m, block_n, num_stages, num_warps = _get_fwd_config_bias_lowlevel(
        batch_size,
        num_heads,
        query_len,
        key_len,
        head_dim,
        bool(causal),
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
    """Run dense bias backward through the low-level CUDA launcher.

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

    if _flash_attn_v2_bwd_bias_lowlevel is None or _get_bwd_config_bias_lowlevel is None:
        raise RuntimeError("FlashDeBERTa local-bias backward is unavailable.")

    batch_size, num_heads, query_len, head_dim = q.shape
    key_len = int(k.shape[2])
    block_m, block_n, num_stages, num_warps = _get_bwd_config_bias_lowlevel(
        batch_size,
        num_heads,
        query_len,
        key_len,
        head_dim,
        bool(causal),
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
