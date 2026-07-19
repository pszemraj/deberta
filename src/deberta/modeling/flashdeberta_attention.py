"""FlashDeBERTa attention adapter for the native DeBERTa-v2/v3 backbone.

Dense-vs-varlen routing does not inspect attention-mask contents inside the
compiled forward path. In this repository's training loop, dense batches already
arrive as ``attention_mask=None`` because the collator drops all-ones masks.

Important behavior
------------------
- Fixed-length flash remains the default path for dense / maskless batches.
- Varlen flash is used for padded batches when enabled by runtime policy.
- Compiled padded-varlen attention uses a repo-owned Triton op so
  ``torch.compile`` does not trace into FlashDeBERTa's Python launcher.
- Pairwise masks still fall back to eager attention for correctness.

Use ``model.hf.attention_impl=flash`` and ``model.hf.flash.*`` for routing and
kernel policy.
"""

from __future__ import annotations

import math
from functools import lru_cache, partial
from typing import Any

import torch

from deberta.modeling.deberta_v2_native import (
    DisentangledSelfAttention as _EagerDisentangledSelfAttention,
)
from deberta.modeling.deberta_v2_native import (
    build_relative_position as _build_relative_position,
)
from deberta.modeling.flashdeberta_bias_op import (
    flashdeberta_bias_from_positions,
    flashdeberta_bias_import_error,
    flashdeberta_compiled_position_bias_available,
)
from deberta.modeling.flashdeberta_docblock_op import (
    flashdeberta_compiled_docblock_available,
    flashdeberta_docblock,
    flashdeberta_docblock_import_error,
)
from deberta.modeling.flashdeberta_fixed_op import (
    flashdeberta_fixed,
    flashdeberta_fixed_import_error,
)
from deberta.modeling.flashdeberta_kernel_tuning import (
    configure_flashdeberta_kernel_overrides,
    flash_padding_route,
)
from deberta.modeling.flashdeberta_op_utils import device_compute_capability
from deberta.modeling.flashdeberta_varlen_op import (
    flashdeberta_compiled_varlen_available,
    flashdeberta_varlen_import_error,
    flashdeberta_varlen_padded,
)
from deberta.modeling.mask_utils import (
    FlashBatchMeta,
    build_doc_block_mask,
    expand_keep_mask_to_4d,
    is_pairwise_mask,
    is_torch_compiling,
    mask_to_2d_keep_mask,
)

_FLASH_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


def _should_use_varlen(
    *,
    attention_mask: torch.Tensor | None,
    seq_len: int,
) -> bool:
    """Return whether the varlen kernel should be used for this call.

    This deliberately does not inspect tensor contents. In this repository's
    training path, the collator already drops all-ones masks, so
    ``attention_mask is None`` is the dense signal and ``attention_mask is not None``
    is the padded signal. When ``torch.compile`` is active we require the
    compile-visible Triton varlen op so Dynamo does not trace into Python
    launch setup.

    Route policy is shared with the training loop through
    :func:`flash_padding_route`. This fallback only runs when no
    ``flash_meta`` route hint was provided, and it cannot know batch density
    without a host sync, so density-gated table buckets resolve to their
    density-agnostic fallback rows here; the training path passes the
    density-aware hint through ``FlashBatchMeta`` instead.

    :param torch.Tensor | None attention_mask: Optional attention mask.
    :param int seq_len: Sequence length for the current call.
    :return bool: True when the varlen kernel should run.
    """

    if attention_mask is None:
        return False

    if is_torch_compiling() and not flashdeberta_compiled_varlen_available():
        return False

    return (
        flash_padding_route(
            seq_len=int(seq_len),
            compute_capability=device_compute_capability(attention_mask.device),
        )
        == "varlen"
    )


def _resolve_docblock_scalars(
    flash_meta: FlashBatchMeta | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Resolve doc-block active-token/segment scalars from flash metadata.

    :param FlashBatchMeta | None flash_meta: Optional flash metadata bundle.
    :return tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        ``(active_tokens, num_segments, max_segment_length)`` as scalar tensors.
    """

    if flash_meta is None:
        return None, None, None
    return (
        flash_meta.active_tokens_scalar,
        flash_meta.doc_num_segments_scalar,
        flash_meta.doc_max_segment_length_scalar,
    )


@lru_cache(maxsize=8)
def _dense_bucket_index_tensor(
    *,
    seq_len: int,
    position_buckets: int,
    max_relative_distance: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a cached dense DeBERTa bucket-index matrix ``(S, S)``.

    The log-bucket math is shared with the eager backbone through
    :func:`deberta.modeling.deberta_v2_native.build_relative_position`; this
    helper only adds the ``+position_buckets`` embedding-row offset expected
    by the flash bias kernels. Sharing one implementation keeps eager-vs-flash
    relative-position parity by construction.

    :param int seq_len: Sequence length.
    :param int position_buckets: Relative bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param torch.device device: Target device.
    :return torch.Tensor: Dense int64 bucket-index tensor.
    """

    max_rel = int(max_relative_distance)
    if max_rel <= 0:
        max_rel = int(seq_len)
    return (
        _build_relative_position(
            int(seq_len),
            int(seq_len),
            bucket_size=int(position_buckets),
            max_position=max_rel,
            device=device,
        )
        .add_(int(position_buckets))
        .clamp_(0, 2 * int(position_buckets) - 1)
    )


class FlashDisentangledSelfAttention(_EagerDisentangledSelfAttention):
    """FlashDeBERTa-backed variant of native disentangled self-attention."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the FlashDeBERTa runtime policy.

        :param Any args: Positional constructor arguments.
        :param Any kwargs: Keyword constructor arguments.
        """

        config = args[0] if args else kwargs.get("config")
        flash_config = getattr(config, "hf_flash", {}) if config is not None else {}
        configure_flashdeberta_kernel_overrides(flash_config.get("kernel_overrides_path"))
        super().__init__(*args, **kwargs)

    def _requires_eager_fallback(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        query_states: torch.Tensor,
        rel_embeddings: torch.Tensor | None,
        flash_meta: FlashBatchMeta | None = None,
    ) -> bool:
        """Return whether this call requires eager attention.

        :param torch.Tensor hidden_states: Key/value hidden states.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param torch.Tensor query_states: Query hidden states.
        :param torch.Tensor | None rel_embeddings: Relative embedding table.
        :param FlashBatchMeta | None flash_meta: Optional out-of-graph routing metadata.
        :return bool: True when the call requires eager attention.
        """

        query_len = int(query_states.shape[-2])
        key_len = int(hidden_states.shape[-2])
        dropout_p = float(getattr(self.dropout, "p", 0.0))
        route = flash_meta.route_hint if flash_meta is not None else None

        if (
            attention_mask is not None
            and is_pairwise_mask(attention_mask, query_len=query_len, key_len=key_len)
            and route != "docblock_bias"
        ):
            return True
        if attention_mask is not None and (
            flash_meta is None or (not flash_meta.is_cross_document() and flash_meta.seq_lengths is None)
        ):
            return True
        if "p2p" in self.pos_att_type:
            return True
        if self.training and dropout_p > 0.0:
            # Validated repo configs reject positive dropout with flash. Keep
            # this fallback for direct/custom module construction.
            return True
        if not self.relative_attention:
            return True
        if rel_embeddings is None:
            return True
        if int(self.position_buckets) <= 0:
            return True
        if flashdeberta_fixed_import_error() is not None:
            return True
        if hidden_states.device.type != "cuda":
            return True
        if query_len != key_len:
            return True
        return False

    def _projected_qkv_requires_eager_fallback(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
    ) -> bool:
        """Return whether projected QKV tensors are unsupported by the flash kernels.

        :param torch.Tensor query_layer: Projected query tensor.
        :param torch.Tensor key_layer: Projected key tensor.
        :param torch.Tensor value_layer: Projected value tensor.
        :return bool: True when the projected tensors require eager attention.
        """

        qkv_dtypes = {query_layer.dtype, key_layer.dtype, value_layer.dtype}
        if len(qkv_dtypes) != 1:
            return True

        qkv_dtype = query_layer.dtype
        if qkv_dtype not in _FLASH_SUPPORTED_DTYPES:
            return True
        return False

    def _shape_varlen(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape projection output into varlen-friendly multi-head layout.

        The padded-varlen path operates more naturally in ``(B, S, H, D)``
        layout, which lets the custom op flatten ``(B*S)`` directly and use one
        flat gather/scatter instead of per-token advanced indexing over a
        ``(B, H, S, D)`` transpose.

        :param torch.Tensor x: Projected tensor with trailing dim ``all_head_size``.
        :return torch.Tensor: Tensor with shape ``(B, S, H, D)``.
        """

        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size)

    def _flash_fixed(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        flash_meta: FlashBatchMeta | None,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
    ) -> torch.Tensor:
        """Run the fixed-length FlashDeBERTa kernel.

        :param torch.Tensor query_layer: Projected queries in ``(B, H, S, D)`` layout.
        :param torch.Tensor key_layer: Projected keys in ``(B, H, S, D)`` layout.
        :param torch.Tensor value_layer: Projected values in ``(B, H, S, D)`` layout.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :param torch.Tensor | None pos_key: Optional c2p term.
        :param torch.Tensor | None pos_query: Optional p2c term.
        :param float sm_scale: Softmax scale.
        :return torch.Tensor: Flash output in ``(B, H, S, D)`` layout.
        """

        seq_lengths = flash_meta.seq_lengths if flash_meta is not None else None
        return flashdeberta_fixed(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            seq_lengths=seq_lengths,
            pos_key=pos_key,
            pos_query=pos_query,
            causal=False,
            sm_scale=sm_scale,
            position_buckets=int(self.position_buckets),
            max_relative_distance=int(self.max_relative_positions),
        )

    def _should_use_local_bias(
        self,
        *,
        attention_mask: torch.Tensor | None,
        route_hint: str | None,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
    ) -> bool:
        """Return whether the dense local-bias flash path should run.

        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param str | None route_hint: Batch-prepared flash route.
        :param torch.Tensor | None pos_key: Optional c2p term.
        :param torch.Tensor | None pos_query: Optional p2c term.
        :return bool: True when dense local-bias flash should run.
        """

        if attention_mask is not None:
            return False
        if not self.training:
            return False
        if route_hint != "local_bias":
            return False
        if pos_key is None and pos_query is None:
            return False
        if flashdeberta_bias_import_error() is not None:
            return False
        if is_torch_compiling() and not flashdeberta_compiled_position_bias_available():
            return False
        return True

    def _flash_dense_bias(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        keep_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run dense position-bias attention with optional pairwise masking.

        :param torch.Tensor query_layer: Projected queries in ``(B, H, S, D)`` layout.
        :param torch.Tensor key_layer: Projected keys in ``(B, H, S, D)`` layout.
        :param torch.Tensor value_layer: Projected values in ``(B, H, S, D)`` layout.
        :param torch.Tensor | None pos_key: Optional c2p term.
        :param torch.Tensor | None pos_query: Optional p2c term.
        :param float sm_scale: Softmax scale.
        :param torch.Tensor | None keep_mask: Optional pairwise keep mask.
        :return torch.Tensor: Flash output in ``(B, H, S, D)`` layout.
        """

        seq_len = int(query_layer.shape[-2])
        bucket_index = _dense_bucket_index_tensor(
            seq_len=seq_len,
            position_buckets=int(self.position_buckets),
            max_relative_distance=int(self.max_relative_positions),
            device=query_layer.device,
        )
        output = flashdeberta_bias_from_positions(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            pos_key=pos_key,
            pos_query=pos_query,
            bucket_index=bucket_index,
            keep_mask=keep_mask,
            bias_scale=float(sm_scale),
            sm_scale=sm_scale,
            causal=False,
        )
        if keep_mask is None:
            return output
        query_live = torch.diagonal(keep_mask, dim1=-2, dim2=-1).unsqueeze(-1)
        return output * query_live.to(dtype=output.dtype)

    def _flash_local_bias(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
    ) -> torch.Tensor:
        """Run unmasked dense local-bias attention.

        :param torch.Tensor query_layer: Projected queries.
        :param torch.Tensor key_layer: Projected keys.
        :param torch.Tensor value_layer: Projected values.
        :param torch.Tensor | None pos_key: Optional c2p term.
        :param torch.Tensor | None pos_query: Optional p2c term.
        :param float sm_scale: Softmax scale.
        :return torch.Tensor: Dense attention output.
        """
        return self._flash_dense_bias(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            keep_mask=None,
        )

    def _flash_varlen(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        flash_meta: FlashBatchMeta | None,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
    ) -> torch.Tensor:
        """Run padded varlen FlashDeBERTa attention.

        :param torch.Tensor query_layer: Projected queries in ``(B, S, H, D)`` layout.
        :param torch.Tensor key_layer: Projected keys in ``(B, S, H, D)`` layout.
        :param torch.Tensor value_layer: Projected values in ``(B, S, H, D)`` layout.
        :param torch.Tensor attention_mask: Padding-style keep mask.
        :param FlashBatchMeta | None flash_meta: Optional precomputed padding metadata.
        :param torch.Tensor | None pos_key: Optional c2p term.
        :param torch.Tensor | None pos_query: Optional p2c term.
        :param float sm_scale: Softmax scale.
        :return torch.Tensor: Flash output in ``(B, S, H, D)`` layout.
        """

        seq_len = int(query_layer.shape[1])
        mask_2d = mask_to_2d_keep_mask(attention_mask, seq_len=seq_len)
        out = flashdeberta_varlen_padded(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            attention_mask_2d=mask_2d,
            seq_lengths=flash_meta.seq_lengths if flash_meta is not None else None,
            active_tokens=(flash_meta.active_tokens_scalar if flash_meta is not None else None),
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            position_buckets=int(self.position_buckets),
            max_relative_distance=int(self.max_relative_positions),
            causal=False,
        )
        return out

    def _flash_docblock(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        flash_meta: FlashBatchMeta,
        active_tokens: torch.Tensor,
        doc_num_segments: torch.Tensor,
        doc_max_segment_length: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
    ) -> torch.Tensor:
        """Run doc-block-aware FlashDeBERTa attention over packed document segments.

        :param torch.Tensor query_layer: Projected queries in ``(B, S, H, D)`` layout.
        :param torch.Tensor key_layer: Projected keys in ``(B, S, H, D)`` layout.
        :param torch.Tensor value_layer: Projected values in ``(B, S, H, D)`` layout.
        :param FlashBatchMeta flash_meta: FlashDeBERTa doc-block metadata bundle.
        :param torch.Tensor active_tokens: Resolved active-token scalar.
        :param torch.Tensor doc_num_segments: Resolved segment-count scalar.
        :param torch.Tensor doc_max_segment_length: Resolved max-segment scalar.
        :param torch.Tensor | None pos_key: Optional c2p term.
        :param torch.Tensor | None pos_query: Optional p2c term.
        :param float sm_scale: Softmax scale.
        :return torch.Tensor: Flash output in ``(B, S, H, D)`` layout.
        """

        out = flashdeberta_docblock(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            segment_offsets=flash_meta.doc_segment_offsets,
            segment_lengths=flash_meta.doc_segment_lengths,
            cu_seqlens=flash_meta.doc_cu_seqlens,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            position_buckets=int(self.position_buckets),
            max_relative_distance=int(self.max_relative_positions),
            num_segments=doc_num_segments,
            max_seqlen=doc_max_segment_length,
            total_tokens=active_tokens,
            causal=False,
        )
        return out

    def _flash_docblock_bias(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
    ) -> torch.Tensor:
        """Run dense flash-with-bias for short packed doc-block batches.

        :param torch.Tensor query_layer: Queries in ``(B, H, S, D)`` layout.
        :param torch.Tensor key_layer: Keys in ``(B, H, S, D)`` layout.
        :param torch.Tensor value_layer: Values in ``(B, H, S, D)`` layout.
        :param torch.Tensor attention_mask: Pairwise keep-mask tensor.
        :param torch.Tensor | None pos_key: Optional c2p term in ``(B, H, S, P)`` layout.
        :param torch.Tensor | None pos_query: Optional p2c term in ``(B, H, S, P)`` layout.
        :param float sm_scale: Softmax scale.
        :return torch.Tensor: Flash output in ``(B, H, S, D)`` layout with
            inactive query rows zeroed to match eager attention.
        """

        keep_mask = expand_keep_mask_to_4d(attention_mask, collapse_heads=False)
        return self._flash_dense_bias(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            pos_key=pos_key,
            pos_query=pos_query,
            keep_mask=keep_mask,
            sm_scale=sm_scale,
        )

    def _eager_fallback_attention_mask(
        self,
        *,
        attention_mask: torch.Tensor | None,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        flash_meta: FlashBatchMeta | None,
    ) -> torch.Tensor | None:
        """Return an eager-safe attention mask, rebuilding doc-block masks on demand.

        Doc-block batches may carry a compact 2D mask for the flash kernels.
        Eager attention needs the pairwise mask rebuilt from their ``doc_ids``.

        :param torch.Tensor | None attention_mask: Original attention mask.
        :param torch.Tensor hidden_states: Key/value hidden states.
        :param torch.Tensor query_states: Query hidden states.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :raises RuntimeError: If a doc-block fallback lacks its source ``doc_ids``.
        :return torch.Tensor | None: Eager-compatible attention mask.
        """

        if flash_meta is None or not flash_meta.is_cross_document():
            return attention_mask
        key_len = int(hidden_states.shape[-2])
        query_len = int(query_states.shape[-2])
        if attention_mask is not None and is_pairwise_mask(
            attention_mask,
            query_len=query_len,
            key_len=key_len,
        ):
            return attention_mask
        if query_len != key_len:
            raise RuntimeError(
                "FlashDeBERTa doc-block eager fallback cannot rebuild a pairwise mask for "
                f"query_len={query_len} != key_len={key_len}; provide an explicit pairwise "
                "attention_mask."
            )
        if flash_meta.doc_ids is None:
            raise RuntimeError(
                "FlashDeBERTa doc-block eager fallback requires doc_ids or an explicit "
                "pairwise doc-block attention_mask."
            )
        return build_doc_block_mask(flash_meta.doc_ids).unsqueeze(1)

    def _eager_forward_fallback(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_attentions: bool,
        query_states: torch.Tensor,
        relative_pos: torch.Tensor | None,
        rel_embeddings: torch.Tensor | None,
        flash_meta: FlashBatchMeta | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run eager attention with doc-block fallback masks rebuilt only on demand.

        :param torch.Tensor hidden_states: Key/value hidden states.
        :param torch.Tensor | None attention_mask: Original attention mask.
        :param bool output_attentions: Whether to return attention probabilities.
        :param torch.Tensor query_states: Query hidden states.
        :param torch.Tensor | None relative_pos: Caller-provided relative-position ids, if any.
        :param torch.Tensor | None rel_embeddings: Relative embedding table.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :return tuple[torch.Tensor, torch.Tensor | None]: Eager attention output and optional probs.
        """

        eager_attention_mask = self._eager_fallback_attention_mask(
            attention_mask=attention_mask,
            hidden_states=hidden_states,
            query_states=query_states,
            flash_meta=flash_meta,
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=eager_attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
        flash_meta: FlashBatchMeta | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run flash-backed attention when the runtime contract is compatible.

        :param torch.Tensor hidden_states: Input hidden states.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param bool output_attentions: Whether to return attention probabilities.
        :param torch.Tensor | None query_states: Optional query states.
        :param torch.Tensor | None relative_pos: Optional relative-position ids; a
            non-``None`` value always routes to eager attention with the tensor
            preserved, because the flash kernels only compute the default map.
        :param torch.Tensor | None rel_embeddings: Optional relative embedding table.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :return tuple[torch.Tensor, torch.Tensor | None]: Attention output and optional probs.
        """

        if query_states is None:
            query_states = hidden_states
        fallback_to_eager = partial(
            self._eager_forward_fallback,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            flash_meta=flash_meta,
        )

        if relative_pos is not None:
            # The flash kernels compute default relative positions on device and
            # cannot honor an arbitrary caller-provided tensor; only eager
            # attention preserves the native relative_pos contract.
            return fallback_to_eager()

        if output_attentions:
            return fallback_to_eager()

        if self._requires_eager_fallback(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            query_states=query_states,
            rel_embeddings=rel_embeddings,
            flash_meta=flash_meta,
        ):
            # The encoder-level get_rel_pos patch suppresses the shared (S,S)
            # allocation globally. Unsupported correctness fallbacks rebuild
            # relative-position bias inside eager attention instead.
            return fallback_to_eager()

        model_dtype = hidden_states.dtype
        bsz, query_len, _ = query_states.shape
        route = flash_meta.route_hint if flash_meta is not None else None
        use_docblock_bias = route == "docblock_bias"
        use_docblock = route == "docblock"
        if use_docblock:
            use_varlen = True
        elif use_docblock_bias:
            use_varlen = False
        elif route == "varlen":
            use_varlen = True
        elif route in {"fixed", "dense", "local_bias"}:
            use_varlen = False
        else:
            use_varlen = _should_use_varlen(
                attention_mask=attention_mask,
                seq_len=int(hidden_states.shape[-2]),
            )

        if use_docblock_bias:
            if attention_mask is None or not is_pairwise_mask(
                attention_mask,
                query_len=query_len,
                key_len=int(hidden_states.shape[-2]),
            ):
                return fallback_to_eager()
            bias_import_error = flashdeberta_bias_import_error()
            if bias_import_error is not None:
                return fallback_to_eager()
            if is_torch_compiling() and not flashdeberta_compiled_position_bias_available():
                return fallback_to_eager()

        if use_docblock:
            docblock_active_tokens, docblock_num_segments, docblock_max_segment = _resolve_docblock_scalars(
                flash_meta
            )
            if (
                flash_meta is None
                or flash_meta.doc_segment_offsets is None
                or flash_meta.doc_segment_lengths is None
                or flash_meta.doc_cu_seqlens is None
                or docblock_active_tokens is None
                or docblock_num_segments is None
                or docblock_max_segment is None
            ):
                return fallback_to_eager()
            docblock_import_error = flashdeberta_docblock_import_error()
            if docblock_import_error is not None:
                return fallback_to_eager()
            if is_torch_compiling() and not flashdeberta_compiled_docblock_available():
                return fallback_to_eager()

        if use_varlen and not use_docblock:
            varlen_import_error = flashdeberta_varlen_import_error()
            if varlen_import_error is not None:
                return fallback_to_eager()
            if is_torch_compiling() and not flashdeberta_compiled_varlen_available():
                return fallback_to_eager()

        if use_varlen:
            query_layer = self._shape_varlen(self.query_proj(query_states))
            key_layer = self._shape_varlen(self.key_proj(hidden_states))
            value_layer = self._shape_varlen(self.value_proj(hidden_states))
        else:
            query_layer = self._shape(self.query_proj(query_states)).contiguous()
            key_layer = self._shape(self.key_proj(hidden_states)).contiguous()
            value_layer = self._shape(self.value_proj(hidden_states)).contiguous()

        if self._projected_qkv_requires_eager_fallback(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
        ):
            # Keep the same eager fallback contract here for dtype/layout
            # mismatches instead of reviving the encoder-wide relative_pos tensor.
            return fallback_to_eager()

        pos_key: torch.Tensor | None = None
        pos_query: torch.Tensor | None = None
        rel_embeddings = self.pos_dropout(rel_embeddings)

        if "c2p" in self.pos_att_type:
            pos_key_layer = self._project_rel(rel_embeddings, use_query=False).to(dtype=query_layer.dtype)
            if use_varlen:
                pos_key = torch.einsum("bshd,hpd->bshp", query_layer, pos_key_layer)
            else:
                pos_key = torch.matmul(query_layer, pos_key_layer.unsqueeze(0).transpose(-1, -2))

        if "p2c" in self.pos_att_type:
            pos_query_layer = self._project_rel(rel_embeddings, use_query=True).to(dtype=key_layer.dtype)
            if use_varlen:
                pos_query = torch.einsum("bshd,hpd->bshp", key_layer, pos_query_layer)
            else:
                pos_query = torch.matmul(key_layer, pos_query_layer.unsqueeze(0).transpose(-1, -2))

        scale_factor = 1
        if pos_key is not None:
            scale_factor += 1
        if pos_query is not None:
            scale_factor += 1
        sm_scale = 1.0 / math.sqrt(float(self.attention_head_size * scale_factor))

        if use_docblock_bias:
            output = self._flash_docblock_bias(
                query_layer=query_layer,
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask,
                pos_key=pos_key,
                pos_query=pos_query,
                sm_scale=sm_scale,
            )
            output = output.transpose(1, 2).contiguous().view(bsz, query_len, self.all_head_size)
        elif use_docblock:
            output = self._flash_docblock(
                query_layer=query_layer,
                key_layer=key_layer,
                value_layer=value_layer,
                flash_meta=flash_meta,
                active_tokens=docblock_active_tokens,
                doc_num_segments=docblock_num_segments,
                doc_max_segment_length=docblock_max_segment,
                pos_key=pos_key,
                pos_query=pos_query,
                sm_scale=sm_scale,
            )
            output = output.contiguous().view(bsz, query_len, self.all_head_size)
        elif use_varlen:
            output = self._flash_varlen(
                query_layer=query_layer,
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask,
                flash_meta=flash_meta,
                pos_key=pos_key,
                pos_query=pos_query,
                sm_scale=sm_scale,
            )
            output = output.contiguous().view(bsz, query_len, self.all_head_size)
        else:
            if self._should_use_local_bias(
                attention_mask=attention_mask,
                route_hint=route,
                pos_key=pos_key,
                pos_query=pos_query,
            ):
                output = self._flash_local_bias(
                    query_layer=query_layer,
                    key_layer=key_layer,
                    value_layer=value_layer,
                    pos_key=pos_key,
                    pos_query=pos_query,
                    sm_scale=sm_scale,
                )
            else:
                output = self._flash_fixed(
                    query_layer=query_layer,
                    key_layer=key_layer,
                    value_layer=value_layer,
                    flash_meta=flash_meta,
                    pos_key=pos_key,
                    pos_query=pos_query,
                    sm_scale=sm_scale,
                )
            output = output.transpose(1, 2).contiguous().view(bsz, query_len, self.all_head_size)
        output = output.to(dtype=model_dtype)
        # output_attentions=True returned via the eager fallback above; flash
        # routes never materialize attention probabilities.
        return output, None


__all__ = ["FlashDisentangledSelfAttention"]
