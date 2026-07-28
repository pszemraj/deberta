"""Shared pack, launch, and unpack orchestration for FlashDeBERTa backward paths."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

PackedGradients = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]


@dataclass(frozen=True)
class PackedBackwardInputs:
    """Packed tensors consumed by a FlashDeBERTa backward kernel."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor
    grad_output: torch.Tensor
    lse: torch.Tensor
    delta: torch.Tensor
    pos_key: torch.Tensor | None
    pos_query: torch.Tensor | None


def run_packed_backward(
    *,
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    query_packed: torch.Tensor | None,
    key_packed: torch.Tensor | None,
    value_packed: torch.Tensor | None,
    output_packed: torch.Tensor | None,
    lse_packed: torch.Tensor | None,
    pos_key_packed: torch.Tensor | None,
    pos_query_packed: torch.Tensor | None,
    pack_rows: Callable[[torch.Tensor], torch.Tensor],
    pack_triple: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    pack_optional_pair: Callable[
        [torch.Tensor | None, torch.Tensor | None],
        tuple[torch.Tensor | None, torch.Tensor | None],
    ],
    pack_grad_and_delta: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    launch_backward: Callable[[PackedBackwardInputs], PackedGradients],
    unpack_triple: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    unpack_optional_pair: Callable[
        [torch.Tensor | None, torch.Tensor | None],
        tuple[torch.Tensor | None, torch.Tensor | None],
    ],
    normalize_packed: Callable[[torch.Tensor | None], torch.Tensor | None] | None = None,
) -> PackedGradients:
    """Run the invariant pack, backward-launch, and unpack sequence.

    Callers close the pack/unpack callbacks over prefix or segment metadata,
    keeping packed-capacity and addressing policy explicit at each route.

    :param torch.Tensor grad_output: Padded output gradient.
    :param torch.Tensor query: Padded query tensor.
    :param torch.Tensor key: Padded key tensor.
    :param torch.Tensor value: Padded value tensor.
    :param torch.Tensor output: Padded forward output.
    :param torch.Tensor lse: Padded forward log-sum-exp tensor.
    :param torch.Tensor | None pos_key: Optional padded c2p tensor.
    :param torch.Tensor | None pos_query: Optional padded p2c tensor.
    :param torch.Tensor | None query_packed: Optional cached packed query.
    :param torch.Tensor | None key_packed: Optional cached packed key.
    :param torch.Tensor | None value_packed: Optional cached packed value.
    :param torch.Tensor | None output_packed: Optional cached packed output.
    :param torch.Tensor | None lse_packed: Optional cached packed LSE.
    :param torch.Tensor | None pos_key_packed: Optional cached packed c2p tensor.
    :param torch.Tensor | None pos_query_packed: Optional cached packed p2c tensor.
    :param Callable pack_rows: Pack one padded tensor.
    :param Callable pack_triple: Pack q/k/v together.
    :param Callable pack_optional_pair: Pack whichever positional tensors are missing.
    :param Callable pack_grad_and_delta: Pack the output gradient and compute delta.
    :param Callable launch_backward: Launch the route-specific packed backward kernel.
    :param Callable unpack_triple: Unpack q/k/v gradients together.
    :param Callable unpack_optional_pair: Unpack optional positional gradients.
    :param Callable | None normalize_packed: Normalize cached fixed-capacity auxiliaries.
    :return PackedGradients: Padded q/k/v and optional positional gradients.
    """

    normalize = normalize_packed or (lambda tensor: tensor)
    if query_packed is None or key_packed is None or value_packed is None:
        query_packed, key_packed, value_packed = pack_triple(query, key, value)
    query_packed = normalize(query_packed)
    key_packed = normalize(key_packed)
    value_packed = normalize(value_packed)
    if query_packed is None or key_packed is None or value_packed is None:
        raise RuntimeError("Packed backward requires query, key, and value auxiliaries.")

    if output_packed is None:
        output_packed = pack_rows(output)
    output_packed = normalize(output_packed)
    if output_packed is None:
        raise RuntimeError("Packed backward requires a forward-output auxiliary.")
    grad_packed, delta = pack_grad_and_delta(grad_output, output_packed)

    if lse_packed is None:
        lse_packed = pack_rows(lse)
    lse_packed = normalize(lse_packed)
    if lse_packed is None:
        raise RuntimeError("Packed backward requires an LSE auxiliary.")

    missing_pos_key = pos_key if pos_key is not None and pos_key_packed is None else None
    missing_pos_query = pos_query if pos_query is not None and pos_query_packed is None else None
    if missing_pos_key is not None or missing_pos_query is not None:
        packed_key, packed_query = pack_optional_pair(missing_pos_key, missing_pos_query)
        if pos_key_packed is None:
            pos_key_packed = packed_key
        if pos_query_packed is None:
            pos_query_packed = packed_query

    packed = PackedBackwardInputs(
        query=query_packed,
        key=key_packed,
        value=value_packed,
        output=output_packed,
        grad_output=grad_packed,
        lse=lse_packed,
        delta=delta,
        pos_key=normalize(pos_key_packed),
        pos_query=normalize(pos_query_packed),
    )
    dq_packed, dk_packed, dv_packed, dpos_key_packed, dpos_query_packed = launch_backward(packed)
    dq, dk, dv = unpack_triple(dq_packed, dk_packed, dv_packed)
    dpos_key, dpos_query = unpack_optional_pair(dpos_key_packed, dpos_query_packed)
    return dq, dk, dv, dpos_key, dpos_query


__all__ = ["PackedBackwardInputs", "PackedGradients", "run_packed_backward"]
