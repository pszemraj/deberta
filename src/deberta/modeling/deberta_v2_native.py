"""Native torch implementation of the DeBERTa-v2 encoder backbone.

This module intentionally avoids importing/instantiating ``transformers.DebertaV2Model``
for runtime training. It preserves the DeBERTa-v2/v3 architectural contract (disentangled
relative-position attention + LayerNorm stack) while keeping the implementation local to
this repository.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from transformers import DebertaV2Config, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from deberta.config import _normalize_hf_attention_kernel
from deberta.modeling.activations import get_act_fn
from deberta.modeling.flashdeberta_op_utils import is_flash_attention_impl
from deberta.modeling.mask_utils import (
    FlashBatchMeta,
    expand_keep_mask_to_4d,
    is_torch_compiling,
    normalize_keep_mask,
    reduce_keep_mask_to_2d,
)


def _normalize_pos_att_type(raw: Any) -> list[str]:
    """Normalize positional-attention type config into a canonical string list.

    :param Any raw: Raw ``config.pos_att_type`` value.
    :return list[str]: Normalized positional-attention tags.
    """

    if raw is None:
        return []
    if isinstance(raw, str):
        chunks = raw.replace(",", "|").split("|")
    elif isinstance(raw, (list, tuple, set)):
        chunks = [str(x) for x in raw]
    else:
        chunks = [str(raw)]
    out: list[str] = []
    for chunk in chunks:
        value = str(chunk).strip().lower()
        if value:
            out.append(value)
    return out


def _make_log_bucket_position(
    relative_pos: torch.Tensor, bucket_size: int, max_position: int
) -> torch.Tensor:
    """Apply log-bucket compression to relative-position deltas.

    :param torch.Tensor relative_pos: Relative position deltas.
    :param int bucket_size: Number of buckets.
    :param int max_position: Maximum represented absolute distance.
    :return torch.Tensor: Bucketized relative-position ids.
    """

    # Match original DeBERTa behavior: clamp representable distances before
    # bucketing so extreme relative offsets saturate instead of drifting.
    rel = relative_pos
    if int(max_position) > 0:
        rel = rel.clamp(min=-int(max_position) + 1, max=int(max_position) - 1)

    sign = torch.sign(rel)
    mid = int(bucket_size // 2)
    if mid <= 1:
        return rel

    abs_pos = rel.abs().clamp_min(1)
    near = abs_pos < mid

    # Match HF semantics while avoiding scripted helper branches in forward.
    log_base = math.log(max(float(max_position - 1), float(mid + 1)) / float(mid))
    if log_base <= 0.0:
        return rel

    log_pos = torch.ceil(torch.log(abs_pos.float() / float(mid)) / log_base * float(mid - 1)) + float(mid)
    bucket = torch.where(near, abs_pos.to(log_pos.dtype), log_pos)
    bucket = bucket * sign.to(bucket.dtype)
    return bucket.to(torch.long)


def build_relative_position(
    query_len: int,
    key_len: int,
    *,
    bucket_size: int,
    max_position: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a relative-position id matrix for query/key lengths.

    :param int query_len: Query sequence length.
    :param int key_len: Key sequence length.
    :param int bucket_size: Optional bucket count (``<=0`` disables bucketing).
    :param int max_position: Maximum absolute distance used for bucketing.
    :param torch.device device: Target tensor device.
    :return torch.Tensor: Relative-position ids with shape ``(query_len, key_len)``.
    """

    q_ids = torch.arange(int(query_len), dtype=torch.long, device=device)
    k_ids = torch.arange(int(key_len), dtype=torch.long, device=device)
    rel_pos = q_ids[:, None] - k_ids[None, :]

    if int(bucket_size) > 0 and int(max_position) > 0:
        rel_pos = _make_log_bucket_position(rel_pos, int(bucket_size), int(max_position))

    return rel_pos.to(torch.long)


class DebertaV2SelfOutput(nn.Module):
    """Self-attention output projection block."""

    def __init__(self, config: DebertaV2Config) -> None:
        """Create post-attention projection, dropout, and LayerNorm.

        :param DebertaV2Config config: Backbone configuration.
        """
        super().__init__()
        self.dense = nn.Linear(int(config.hidden_size), int(config.hidden_size))
        self.LayerNorm = nn.LayerNorm(int(config.hidden_size), eps=float(config.layer_norm_eps))
        self.dropout = nn.Dropout(float(config.hidden_dropout_prob))

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Project and residual-normalize self-attention outputs.

        :param torch.Tensor hidden_states: Attention output states.
        :param torch.Tensor input_tensor: Residual input tensor.
        :return torch.Tensor: Layer output.
        """

        return _apply_output_residual_norm(
            hidden_states=hidden_states,
            input_tensor=input_tensor,
            dense=self.dense,
            dropout=self.dropout,
            layer_norm=self.LayerNorm,
        )


def _apply_output_residual_norm(
    *,
    hidden_states: torch.Tensor,
    input_tensor: torch.Tensor,
    dense: nn.Module,
    dropout: nn.Module,
    layer_norm: nn.Module,
) -> torch.Tensor:
    """Apply shared output projection + dropout + residual LayerNorm pattern.

    :param torch.Tensor hidden_states: Projected branch input tensor.
    :param torch.Tensor input_tensor: Residual branch tensor.
    :param nn.Module dense: Output projection module.
    :param nn.Module dropout: Dropout module.
    :param nn.Module layer_norm: LayerNorm module.
    :return torch.Tensor: Residual-normalized output.
    """
    x = dense(hidden_states)
    x = dropout(x)
    x = layer_norm(x + input_tensor)
    return x


class DisentangledSelfAttention(nn.Module):
    """DeBERTa-v2 disentangled self-attention implementation."""

    def __init__(self, config: DebertaV2Config) -> None:
        """Create attention projections and relative-position helpers.

        :param DebertaV2Config config: Backbone configuration.
        :raises ValueError: If hidden-size/head-count are inconsistent.
        """
        super().__init__()
        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_heads})."
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(getattr(config, "attention_head_size", hidden_size // num_heads))
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_proj = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(hidden_size, self.all_head_size, bias=True)

        self.share_att_key = bool(getattr(config, "share_att_key", False))
        self.pos_att_type = _normalize_pos_att_type(getattr(config, "pos_att_type", None))
        self.relative_attention = bool(getattr(config, "relative_attention", False))

        self.position_buckets = int(getattr(config, "position_buckets", -1))
        self.max_relative_positions = int(getattr(config, "max_relative_positions", -1))
        if self.max_relative_positions < 1:
            self.max_relative_positions = int(config.max_position_embeddings)

        self.pos_ebd_size = self.max_relative_positions
        if self.position_buckets > 0:
            self.pos_ebd_size = self.position_buckets

        self.pos_dropout = nn.Dropout(float(config.hidden_dropout_prob))
        self.attn_kernel = _normalize_hf_attention_kernel(getattr(config, "hf_attention_kernel", "dynamic"))

        if self.relative_attention and (not self.share_att_key):
            if ("c2p" in self.pos_att_type) or ("p2p" in self.pos_att_type):
                self.pos_key_proj = nn.Linear(hidden_size, self.all_head_size, bias=True)
            else:
                self.pos_key_proj = None
            if ("p2c" in self.pos_att_type) or ("p2p" in self.pos_att_type):
                self.pos_query_proj = nn.Linear(hidden_size, self.all_head_size, bias=True)
            else:
                self.pos_query_proj = None
        else:
            self.pos_key_proj = None
            self.pos_query_proj = None

        self.dropout = nn.Dropout(float(config.attention_probs_dropout_prob))

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape projection output into multi-head format.

        :param torch.Tensor x: Projected tensor with trailing dim ``all_head_size``.
        :return torch.Tensor: Tensor with shape ``(B, H, S, D)``.
        """

        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3).contiguous()

    def _normalize_relative_pos(
        self,
        relative_pos: torch.Tensor | None,
        *,
        query_len: int,
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Resolve a canonical 2D relative-position matrix for attention bias.

        :param torch.Tensor | None relative_pos: Optional user-provided relative-position ids.
        :param int query_len: Query length.
        :param int key_len: Key length.
        :param torch.device device: Runtime device.
        :return torch.Tensor: Relative-position ids shaped ``(query_len, key_len)``.
        """

        if relative_pos is None:
            return build_relative_position(
                query_len,
                key_len,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=device,
            )

        rp = relative_pos.to(device=device, dtype=torch.long)
        if rp.ndim == 4:
            rp = rp[0, 0]
        elif rp.ndim == 3:
            rp = rp[0]
        elif rp.ndim != 2:
            raise ValueError(f"relative_pos must be rank-2/3/4, got rank={rp.ndim}")

        if rp.shape[0] != query_len or rp.shape[1] != key_len:
            rp = rp[:query_len, :key_len]
        return rp

    def _project_rel(
        self,
        rel_embeddings: torch.Tensor,
        *,
        use_query: bool,
    ) -> torch.Tensor:
        """Project relative embedding table into per-head key/query vectors.

        :param torch.Tensor rel_embeddings: Relative embedding table.
        :param bool use_query: Whether to use query projection path.
        :return torch.Tensor: Projected table with shape ``(H, 2A, D)``.
        """

        att_span = int(self.pos_ebd_size)
        rel = rel_embeddings[: 2 * att_span, :]
        if use_query:
            proj_layer = self.query_proj if self.share_att_key else self.pos_query_proj
        else:
            proj_layer = self.key_proj if self.share_att_key else self.pos_key_proj
        if proj_layer is None:
            raise RuntimeError("Requested relative projection is not configured for this attention module.")
        projected = proj_layer(rel)
        projected = projected.view(2 * att_span, self.num_attention_heads, self.attention_head_size)
        return projected.permute(1, 0, 2).contiguous()

    def _p2p_bias(
        self,
        *,
        rel_pos: torch.Tensor,
        pos_query_layer: torch.Tensor,
        pos_key_layer: torch.Tensor,
        bsz: int,
        nheads: int,
        query_len: int,
        key_len: int,
        att_span: int,
        scale_factor: int,
    ) -> torch.Tensor:
        """Compute position-to-position (p2p) bias.

        :param torch.Tensor rel_pos: Relative ids with shape ``(Q,K)``.
        :param torch.Tensor pos_query_layer: Relative query projections ``(H,2A,D)``.
        :param torch.Tensor pos_key_layer: Relative key projections ``(H,2A,D)``.
        :param int bsz: Batch size.
        :param int nheads: Number of attention heads.
        :param int query_len: Query length.
        :param int key_len: Key length.
        :param int att_span: Relative-attention span ``A``.
        :param int scale_factor: Attention scale factor.
        :return torch.Tensor: p2p bias tensor with shape ``(B,H,Q,K)``.
        """

        # Mirror reference behavior: use positive-half relative query table.
        pos_query = pos_query_layer[:, att_span:, :]  # (H, A, D)
        if pos_query.shape[1] == 0:
            return torch.zeros(
                (bsz, nheads, query_len, key_len),
                device=rel_pos.device,
                dtype=pos_query_layer.dtype,
            )

        # (H, A, 2A)
        p2p_table = torch.einsum("hqd,hkd->hqk", pos_query, pos_key_layer)

        # Map runtime query positions to available p2p rows.
        q_index = torch.arange(query_len, device=rel_pos.device, dtype=torch.long)
        q_index = q_index.clamp(max=int(p2p_table.shape[1]) - 1)
        p2p_query = p2p_table.index_select(1, q_index)  # (H, Q, 2A)

        p2p_idx = (rel_pos + att_span).clamp(min=0, max=(2 * att_span) - 1)  # (Q, K)
        p2p_idx = p2p_idx.unsqueeze(0).expand(nheads, query_len, key_len)
        p2p_bias = p2p_query.gather(-1, p2p_idx)  # (H, Q, K)
        p2p_scale = math.sqrt(float(self.attention_head_size * scale_factor))
        p2p_bias = p2p_bias / p2p_scale
        return p2p_bias.unsqueeze(0).expand(bsz, nheads, query_len, key_len)

    def disentangled_attention_bias(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        relative_pos: torch.Tensor | None,
        rel_embeddings: torch.Tensor,
        scale_factor: int,
    ) -> torch.Tensor:
        """Compute DeBERTa disentangled positional attention bias.

        The ``dynamic`` kernel scores c2p/p2c terms with einsum; the
        ``cached_bmm``/``stable`` kernels use explicit permute + batched matmul.
        Index construction, scaling, p2p handling, and the zero fallback are
        shared. Cross-call score caching is intentionally avoided: c2p/p2c
        terms depend on runtime query/key activations and must be recomputed
        every forward.

        :param torch.Tensor query_layer: Query tensor shaped ``(B,H,Q,D)``.
        :param torch.Tensor key_layer: Key tensor shaped ``(B,H,K,D)``.
        :param torch.Tensor | None relative_pos: Optional relative-position ids.
        :param torch.Tensor rel_embeddings: Relative embedding table.
        :param int scale_factor: Attention scale factor.
        :return torch.Tensor: Relative bias tensor shaped ``(B,H,Q,K)``.
        """

        use_bmm = self.attn_kernel in {"cached_bmm", "stable"}
        bsz, nheads, query_len, _ = query_layer.shape
        _, _, key_len, _ = key_layer.shape
        rel_pos = self._normalize_relative_pos(
            relative_pos,
            query_len=query_len,
            key_len=key_len,
            device=query_layer.device,
        )

        att_span = int(self.pos_ebd_size)
        rel_pos = rel_pos.clamp(min=-att_span, max=att_span)

        score: torch.Tensor | None = None
        pos_key_layer: torch.Tensor | None = None
        pos_query_layer: torch.Tensor | None = None

        if ("c2p" in self.pos_att_type) or ("p2p" in self.pos_att_type):
            # Keep both kernels on the same explicit dtype contract as the
            # caller's query/key score path (fp32 in stabilized attention).
            pos_key_layer = self._project_rel(rel_embeddings, use_query=False).to(dtype=query_layer.dtype)
        if ("p2c" in self.pos_att_type) or ("p2p" in self.pos_att_type):
            pos_query_layer = self._project_rel(rel_embeddings, use_query=True).to(dtype=query_layer.dtype)

        if "c2p" in self.pos_att_type:
            if pos_key_layer is None:
                raise RuntimeError("p2p/c2p path requires pos_key projection.")
            c2p_scale = math.sqrt(float(self.attention_head_size * scale_factor))
            if use_bmm:
                q_flat = query_layer.permute(1, 0, 2, 3).reshape(
                    nheads, bsz * query_len, self.attention_head_size
                )
                pos_key_t = pos_key_layer.transpose(1, 2).contiguous()  # (H,D,2A)
                c2p_att = torch.bmm(q_flat, pos_key_t).reshape(nheads, bsz, query_len, 2 * att_span)
                c2p_att = c2p_att.permute(1, 0, 2, 3).contiguous()  # (B,H,Q,2A)
            else:
                c2p_att = torch.einsum("bhqd,hkd->bhqk", query_layer, pos_key_layer)

            c2p_idx = (rel_pos + att_span).clamp(min=0, max=(2 * att_span) - 1)
            c2p_idx = c2p_idx.unsqueeze(0).unsqueeze(0).expand(bsz, nheads, query_len, key_len)
            c2p_bias = c2p_att.gather(-1, c2p_idx) / c2p_scale
            score = c2p_bias if score is None else score + c2p_bias

        if "p2c" in self.pos_att_type:
            if pos_query_layer is None:
                raise RuntimeError("p2p/p2c path requires pos_query projection.")
            p2c_scale = math.sqrt(float(self.attention_head_size * scale_factor))
            if use_bmm:
                k_flat = key_layer.permute(1, 0, 2, 3).reshape(
                    nheads, bsz * key_len, self.attention_head_size
                )
                pos_query_t = pos_query_layer.transpose(1, 2).contiguous()  # (H,D,2A)
                p2c_att = torch.bmm(k_flat, pos_query_t).reshape(nheads, bsz, key_len, 2 * att_span)
                p2c_att = p2c_att.permute(1, 0, 2, 3).contiguous()  # (B,H,K,2A)
            else:
                p2c_att = torch.einsum("bhkd,hqd->bhkq", key_layer, pos_query_layer)

            # Convert canonical signed buckets from query-major [Q,K] to the
            # key-major [K,Q] layout of p2c_att without reversing their sign.
            # Entry [k,q] must still select bucket(q-k).
            p2c_idx = (rel_pos.transpose(0, 1) + att_span).clamp(min=0, max=(2 * att_span) - 1)
            p2c_idx = p2c_idx.unsqueeze(0).unsqueeze(0).expand(bsz, nheads, key_len, query_len)
            p2c_bias = p2c_att.gather(-1, p2c_idx).transpose(-1, -2) / p2c_scale
            score = p2c_bias if score is None else score + p2c_bias

        if "p2p" in self.pos_att_type:
            if pos_key_layer is None or pos_query_layer is None:
                raise RuntimeError("p2p path requires both pos_key and pos_query projections.")
            p2p_bias = self._p2p_bias(
                rel_pos=rel_pos,
                pos_query_layer=pos_query_layer,
                pos_key_layer=pos_key_layer,
                bsz=bsz,
                nheads=nheads,
                query_len=query_len,
                key_len=key_len,
                att_span=att_span,
                scale_factor=scale_factor,
            )
            score = p2p_bias if score is None else score + p2p_bias

        if score is None:
            score = torch.zeros(
                (bsz, nheads, query_len, key_len), device=query_layer.device, dtype=query_layer.dtype
            )
        return score

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
        """Run disentangled self-attention.

        :param torch.Tensor hidden_states: Input hidden states.
        :param torch.Tensor | None attention_mask: Boolean keep mask ``(B,1,Q,K)`` or ``None`` for unpadded batches.
        :param bool output_attentions: Whether to return attention probabilities.
        :param torch.Tensor | None query_states: Optional query states (for iterative decoding).
        :param torch.Tensor | None relative_pos: Optional relative-position ids.
        :param torch.Tensor | None rel_embeddings: Optional relative-position embedding table.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle, ignored by eager attention.
        :return tuple[torch.Tensor, torch.Tensor | None]: Attention output and optional probs.
        """
        del flash_meta

        if query_states is None:
            query_states = hidden_states

        query_layer = self._shape(self.query_proj(query_states))
        key_layer = self._shape(self.key_proj(hidden_states))
        value_layer = self._shape(self.value_proj(hidden_states))
        model_dtype = hidden_states.dtype

        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        if "p2p" in self.pos_att_type:
            scale_factor += 1

        scale = math.sqrt(float(self.attention_head_size * scale_factor))
        # Keep score-path numerics in fp32 under compile for improved stability.
        query_layer_f = query_layer.float()
        key_layer_f = key_layer.float()
        attention_scores = torch.matmul(query_layer_f, key_layer_f.transpose(-1, -2)) / scale

        if self.relative_attention and rel_embeddings is not None:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(
                query_layer_f,
                key_layer_f,
                relative_pos,
                rel_embeddings,
                scale_factor,
            )
            attention_scores = attention_scores + rel_att.float()

        if attention_mask is not None:
            keep_mask = normalize_keep_mask(attention_mask)
            if keep_mask.ndim != 4:
                raise ValueError(
                    f"attention_mask must be rank-4 [B,1,Q,K]; got shape={tuple(keep_mask.shape)}"
                )
            mask_fill_value = torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores.masked_fill(~keep_mask, mask_fill_value)
            # For broadcast padding masks (B,1,1,S), query activity equals key activity —
            # transpose the key dim to get per-query (B,1,S,1). For pairwise masks
            # (B,1,S,S), query activity is encoded on the diagonal (inactive queries
            # may still keep a CLS fallback edge to avoid all-False rows).
            if keep_mask.shape[-2] == 1:
                live_queries = keep_mask.transpose(-2, -1)  # (B,1,S,1)
            else:
                live_queries = torch.diagonal(keep_mask, dim1=-2, dim2=-1).unsqueeze(-1)  # (B,1,S,1)
            attention_scores = torch.where(live_queries, attention_scores, torch.zeros_like(attention_scores))

        probs = torch.softmax(attention_scores, dim=-1)
        probs = self.dropout(probs)

        if attention_mask is not None:
            probs = probs * live_queries.to(dtype=probs.dtype)

        # Keep value/context path at model dtype to avoid unnecessary fp32 activation
        # expansion; only score-path numerics require fp32 stabilization.
        probs = probs.to(dtype=value_layer.dtype)
        context_layer = torch.matmul(probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        bsz, seq_len, _, _ = context_layer.shape
        context_layer = context_layer.view(bsz, seq_len, self.all_head_size).to(dtype=model_dtype)
        probs_out = probs.to(dtype=model_dtype)

        if not output_attentions:
            return context_layer, None
        return context_layer, probs_out


class DebertaV2Attention(nn.Module):
    """Attention wrapper containing disentangled attention and output projection."""

    def __init__(self, config: DebertaV2Config) -> None:
        """Create one DeBERTa attention block.

        :param DebertaV2Config config: Backbone configuration.
        """
        super().__init__()
        if is_flash_attention_impl(getattr(config, "hf_attention_impl", "eager")):
            from deberta.modeling.flashdeberta_attention import FlashDisentangledSelfAttention

            self.self = FlashDisentangledSelfAttention(config)
        else:
            self.self = DisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config)

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
        """Run attention and post-attention projection.

        :param torch.Tensor hidden_states: Input hidden states.
        :param torch.Tensor | None attention_mask: Boolean keep mask or ``None`` for unpadded batches.
        :param bool output_attentions: Whether to return attention probabilities.
        :param torch.Tensor | None query_states: Optional query states.
        :param torch.Tensor | None relative_pos: Optional relative-position ids.
        :param torch.Tensor | None rel_embeddings: Optional relative embedding table.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :return tuple[torch.Tensor, torch.Tensor | None]: Layer outputs.
        """

        self_output, attn_probs = self.self(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            flash_meta=flash_meta,
        )
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)
        if output_attentions:
            return attention_output, attn_probs
        return attention_output, None


class DebertaV2Intermediate(nn.Module):
    """Intermediate FFN projection + activation."""

    def __init__(self, config: DebertaV2Config) -> None:
        """Create intermediate MLP projection.

        :param DebertaV2Config config: Backbone configuration.
        """
        super().__init__()
        self.dense = nn.Linear(int(config.hidden_size), int(config.intermediate_size))
        self.intermediate_act_fn = get_act_fn(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project then activate hidden states.

        :param torch.Tensor hidden_states: Input tensor.
        :return torch.Tensor: Activated intermediate tensor.
        """

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class DebertaV2Output(nn.Module):
    """Output FFN projection with residual LayerNorm."""

    def __init__(self, config: DebertaV2Config) -> None:
        """Create output FFN projection stack.

        :param DebertaV2Config config: Backbone configuration.
        """
        super().__init__()
        self.dense = nn.Linear(int(config.intermediate_size), int(config.hidden_size))
        self.LayerNorm = nn.LayerNorm(int(config.hidden_size), eps=float(config.layer_norm_eps))
        self.dropout = nn.Dropout(float(config.hidden_dropout_prob))

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Project intermediate states back to hidden-size with residual norm.

        :param torch.Tensor hidden_states: Intermediate FFN tensor.
        :param torch.Tensor input_tensor: Residual input.
        :return torch.Tensor: Layer output tensor.
        """

        return _apply_output_residual_norm(
            hidden_states=hidden_states,
            input_tensor=input_tensor,
            dense=self.dense,
            dropout=self.dropout,
            layer_norm=self.LayerNorm,
        )


class DebertaV2Layer(nn.Module):
    """Single DeBERTa-v2 transformer layer."""

    def __init__(self, config: DebertaV2Config) -> None:
        """Create attention + FFN sublayers.

        :param DebertaV2Config config: Backbone configuration.
        """
        super().__init__()
        self.attention = DebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
        output_attentions: bool = False,
        flash_meta: FlashBatchMeta | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one transformer layer.

        :param torch.Tensor hidden_states: Layer input states.
        :param torch.Tensor | None attention_mask: Boolean keep mask or ``None`` for unpadded batches.
        :param torch.Tensor | None query_states: Optional query states.
        :param torch.Tensor | None relative_pos: Optional relative-position ids.
        :param torch.Tensor | None rel_embeddings: Optional relative embedding table.
        :param bool output_attentions: Whether to emit attention probs.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :return tuple[torch.Tensor, torch.Tensor | None]: Layer output and optional attentions.
        """

        attention_output, attn_probs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            flash_meta=flash_meta,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        if output_attentions:
            return layer_output, attn_probs
        return layer_output, None


class DebertaV2Embeddings(nn.Module):
    """Token/position/type embedding stack for DeBERTa-v2."""

    def __init__(self, config: DebertaV2Config) -> None:
        """Create embedding layers and projection.

        :param DebertaV2Config config: Backbone configuration.
        """
        super().__init__()
        pad_token_id = int(getattr(config, "pad_token_id", 0))
        self.embedding_size = int(getattr(config, "embedding_size", config.hidden_size))

        self.word_embeddings = nn.Embedding(
            int(config.vocab_size),
            self.embedding_size,
            padding_idx=pad_token_id,
        )

        # NOTE: DeBERTa-v2/v3 often set position_biased_input=False (no learned absolute
        # positions added to the input). The original Microsoft implementation STILL
        # instantiates position_embeddings because they are consumed by the Enhanced
        # Mask Decoder (EMD) during RTD pretraining.
        #
        # We therefore ALWAYS create position_embeddings, but only ADD them in forward
        # when position_biased_input=True.
        self.position_biased_input = bool(getattr(config, "position_biased_input", True))
        self.position_embeddings = nn.Embedding(int(config.max_position_embeddings), self.embedding_size)

        if int(config.type_vocab_size) > 0:
            self.token_type_embeddings = nn.Embedding(int(config.type_vocab_size), self.embedding_size)
        else:
            self.token_type_embeddings = None

        if self.embedding_size != int(config.hidden_size):
            self.embed_proj = nn.Linear(self.embedding_size, int(config.hidden_size), bias=False)
        else:
            self.embed_proj = None

        self.LayerNorm = nn.LayerNorm(int(config.hidden_size), eps=float(config.layer_norm_eps))
        self.dropout = nn.Dropout(float(config.hidden_dropout_prob))

        self.register_buffer(
            "position_ids",
            torch.arange(int(config.max_position_embeddings), dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute embedding outputs.

        :param torch.Tensor | None input_ids: Optional token ids.
        :param torch.Tensor | None token_type_ids: Optional type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None mask: Optional token mask.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :raises ValueError: If neither ``input_ids`` nor ``inputs_embeds`` is provided.
        :return torch.Tensor: Embedded hidden states.
        """

        if input_ids is not None:
            input_shape = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        bsz, seq_len = int(input_shape[0]), int(input_shape[1])

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len].to(device=device)
        else:
            position_ids = position_ids.to(device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros((bsz, seq_len), dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        if self.position_biased_input:
            embeddings = embeddings + self.position_embeddings(position_ids.long())

        if self.token_type_embeddings is not None:
            embeddings = embeddings + self.token_type_embeddings(token_type_ids.long())

        if self.embed_proj is not None:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            keep = reduce_keep_mask_to_2d(mask, seq_len=seq_len).to(dtype=embeddings.dtype)
            embeddings = embeddings * keep.unsqueeze(-1)

        embeddings = self.dropout(embeddings)
        return embeddings


class DebertaV2Encoder(nn.Module):
    """Encoder stack with optional relative-position table and convolution block."""

    def __init__(self, config: DebertaV2Config) -> None:
        """Create encoder layers and relative-position parameters.

        :param DebertaV2Config config: Backbone configuration.
        """
        super().__init__()
        self.layer = nn.ModuleList([DebertaV2Layer(config) for _ in range(int(config.num_hidden_layers))])
        self.relative_attention = bool(getattr(config, "relative_attention", False))

        self.max_relative_positions = int(getattr(config, "max_relative_positions", -1))
        if self.max_relative_positions < 1:
            self.max_relative_positions = int(config.max_position_embeddings)

        self.position_buckets = int(getattr(config, "position_buckets", -1))
        if self.relative_attention:
            rel_size = self.max_relative_positions * 2
            if self.position_buckets > 0:
                rel_size = self.position_buckets * 2
            self.rel_embeddings = nn.Embedding(int(rel_size), int(config.hidden_size))
        else:
            self.rel_embeddings = None

        norm_rel_ebd = [x.strip() for x in str(getattr(config, "norm_rel_ebd", "none")).lower().split("|")]
        if "layer_norm" in norm_rel_ebd:
            self.LayerNorm = nn.LayerNorm(int(config.hidden_size), eps=float(config.layer_norm_eps))
        else:
            self.LayerNorm = None

        conv_kernel_size = int(getattr(config, "conv_kernel_size", 0) or 0)
        if conv_kernel_size > 0:
            raise ValueError(
                "conv_kernel_size > 0 is not supported by the native DeBERTa-v2 backbone; "
                "DeBERTa-v2 conv-refinement checkpoints (e.g. v2-xlarge/xxlarge) are out of scope."
            )
        self.gradient_checkpointing = False
        self.flash_attention_enabled = is_flash_attention_impl(getattr(config, "hf_attention_impl", "eager"))

    def get_rel_embedding(self) -> torch.Tensor | None:
        """Return optionally normalized relative embedding table.

        :return torch.Tensor | None: Relative embedding weights.
        """

        if self.rel_embeddings is None:
            return None
        rel_embeddings = self.rel_embeddings.weight
        if self.LayerNorm is not None:
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Normalize input attention mask to a 4D boolean keep mask.

        For 2D key-padding masks ``(B, S)`` returns ``(B, 1, 1, S)`` — a broadcast-
        friendly shape that avoids the O(S²) outer-product expansion.  For 3D
        pairwise masks ``(B, S, S)`` returns ``(B, 1, S, S)`` as before.

        :param torch.Tensor attention_mask: Input mask tensor.
        :return torch.Tensor: Keep mask ``(B, 1, 1, S)`` or ``(B, 1, S, S)``.
        """
        return expand_keep_mask_to_4d(attention_mask)

    def get_rel_pos(
        self,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Resolve relative-position ids for the current attention call.

        :param torch.Tensor hidden_states: Key/value states.
        :param torch.Tensor | None query_states: Optional query states.
        :param torch.Tensor | None relative_pos: Optional precomputed relative positions.
        :return torch.Tensor | None: Relative-position ids or ``None``.
        """
        if not self.relative_attention:
            return None
        if relative_pos is not None:
            return relative_pos
        if self.flash_attention_enabled:
            return None

        key_len = int(hidden_states.shape[-2])
        query_len = int(query_states.shape[-2]) if query_states is not None else key_len
        return build_relative_position(
            query_len,
            key_len,
            bucket_size=self.position_buckets,
            max_position=self.max_relative_positions,
            device=hidden_states.device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        return_dict: bool = True,
        flash_meta: FlashBatchMeta | None = None,
    ) -> (
        BaseModelOutput
        | tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, tuple[torch.Tensor, ...] | None]
    ):
        """Run the encoder stack.

        :param torch.Tensor hidden_states: Input hidden states.
        :param torch.Tensor | None attention_mask: Input attention mask or ``None`` for unpadded batches.
        :param bool output_hidden_states: Whether to return hidden states.
        :param bool output_attentions: Whether to return attentions.
        :param torch.Tensor | None query_states: Optional query states.
        :param torch.Tensor | None relative_pos: Optional relative-position ids.
        :param bool return_dict: Whether to return HF output dataclass.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :return BaseModelOutput | tuple: Encoder outputs.
        """

        attn_mask = self.get_attention_mask(attention_mask) if attention_mask is not None else None
        rel_pos = self.get_rel_pos(hidden_states, query_states=query_states, relative_pos=relative_pos)
        rel_embeddings = self.get_rel_embedding()

        capture_hidden_states = bool(output_hidden_states)
        compile_snapshot_hidden_states = bool(capture_hidden_states and is_torch_compiling())
        initial_hidden_state = hidden_states.clone() if compile_snapshot_hidden_states else hidden_states
        all_hidden_states: tuple[torch.Tensor, ...] | None = (
            (initial_hidden_state,) if capture_hidden_states else None
        )
        all_attentions: tuple[torch.Tensor, ...] | None = () if output_attentions else None

        next_kv = hidden_states
        output_states = hidden_states

        for layer_module in self.layer:
            if self.gradient_checkpointing and self.training and query_states is None:

                def _custom_forward(
                    hs: torch.Tensor,
                    am: torch.Tensor | None,
                    rp: torch.Tensor | None,
                    re: torch.Tensor | None,
                    layer: DebertaV2Layer = layer_module,
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
                    """Checkpoint wrapper for one encoder layer.

                    :param torch.Tensor hs: Hidden states.
                    :param torch.Tensor | None am: Attention mask.
                    :param torch.Tensor | None rp: Relative-position ids.
                    :param torch.Tensor | None re: Relative embedding table.
                    :param DebertaV2Layer layer: Layer instance to execute.
                    :return tuple[torch.Tensor, torch.Tensor | None]: Layer output and optional attention tensor.
                    """
                    return layer(
                        hs,
                        am,
                        query_states=None,
                        relative_pos=rp,
                        rel_embeddings=re,
                        output_attentions=output_attentions,
                        flash_meta=flash_meta,
                    )

                output_states, attn_weights = torch.utils.checkpoint.checkpoint(
                    _custom_forward,
                    next_kv,
                    attn_mask,
                    rel_pos,
                    rel_embeddings,
                    use_reentrant=False,
                )
            else:
                output_states, attn_weights = layer_module(
                    next_kv,
                    attn_mask,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                    flash_meta=flash_meta,
                )

            if output_attentions and all_attentions is not None:
                if attn_weights is None:
                    attn_weights = torch.empty(0, device=output_states.device)
                all_attentions = all_attentions + (attn_weights,)

            if capture_hidden_states and all_hidden_states is not None:
                hidden_state_snapshot = (
                    output_states.clone() if compile_snapshot_hidden_states else output_states
                )
                all_hidden_states = all_hidden_states + (hidden_state_snapshot,)

            if query_states is not None:
                query_states = output_states
            next_kv = output_states

        if not return_dict:
            return tuple(v for v in (output_states, all_hidden_states, all_attentions) if v is not None)

        last_hidden_state = output_states.clone() if compile_snapshot_hidden_states else output_states
        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class DebertaV2PreTrainedModel(PreTrainedModel):
    """PreTrainedModel base for native DeBERTa-v2 modules."""

    config_class = DebertaV2Config
    base_model_prefix = "deberta"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:  # pragma: no cover (HF init contract)
        """Initialize module parameters from config defaults.

        :param nn.Module module: Module to initialize.
        """

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=float(self.config.initializer_range))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=float(self.config.initializer_range))
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        """Toggle gradient checkpointing on encoder modules.

        :param nn.Module module: Candidate module.
        :param bool value: Whether checkpointing should be enabled.
        """

        if isinstance(module, DebertaV2Encoder):
            module.gradient_checkpointing = bool(value)


class DebertaV2Model(DebertaV2PreTrainedModel):
    """Native encoder-only DeBERTa-v2 model returning ``BaseModelOutput``."""

    def __init__(self, config: DebertaV2Config) -> None:
        """Create native DeBERTa-v2 backbone.

        :param DebertaV2Config config: Backbone configuration.
        """
        super().__init__(config)
        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)
        self.z_steps = int(getattr(config, "z_steps", 0))
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Return word embedding module.

        :return nn.Module: Input embedding module.
        """

        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Module) -> None:
        """Replace input word embedding module.

        :param nn.Module new_embeddings: Replacement embedding module.
        """

        self.embeddings.word_embeddings = new_embeddings  # type: ignore[assignment]

    def _resolve_forward_options(
        self,
        *,
        output_attentions: bool | None,
        output_hidden_states: bool | None,
        return_dict: bool | None,
    ) -> tuple[bool, bool, bool]:
        """Resolve optional public forward flags into stable booleans.

        Keeping this logic in a dedicated helper lets compile-time dispatchers
        normalize Python optionals before entering compiled code.

        :param bool | None output_attentions: Optional attention-output flag.
        :param bool | None output_hidden_states: Optional hidden-state-output flag.
        :param bool | None return_dict: Optional return-format flag.
        :return tuple[bool, bool, bool]: Resolved ``(output_attentions, output_hidden_states, return_dict)``.
        """

        resolved_attentions = bool(
            getattr(self.config, "output_attentions", False)
            if output_attentions is None
            else output_attentions
        )
        resolved_hidden_states = (
            bool(getattr(self.config, "output_hidden_states", False))
            if output_hidden_states is None
            else bool(output_hidden_states)
        )
        resolved_return_dict = bool(
            getattr(self.config, "use_return_dict", True) if return_dict is None else return_dict
        )
        return resolved_attentions, resolved_hidden_states, resolved_return_dict

    def _resolve_forward_inputs(
        self,
        *,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Validate mutually exclusive inputs and materialize token-type ids.

        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :raises ValueError: If both/neither ``input_ids`` and ``inputs_embeds`` are set.
        :return torch.Tensor: Supplied or materialized token-type ids.
        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds.")

        if input_ids is not None:
            input_shape = input_ids.shape
            device = input_ids.device
        else:
            input_shape = inputs_embeds.shape[:-1]
            device = inputs_embeds.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        return token_type_ids

    def _forward_resolved(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        flash_meta: FlashBatchMeta | None = None,
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]:
        """Run forward with already-resolved boolean output flags.

        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param bool output_attentions: Resolved attention-output flag.
        :param bool output_hidden_states: Resolved hidden-state-output flag.
        :param bool return_dict: Resolved return-format flag.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :raises ValueError: If both/neither ``input_ids`` and ``inputs_embeds`` are set.
        :return BaseModelOutput | tuple[torch.Tensor, ...]: Model outputs.
        """

        output_attentions = bool(output_attentions)
        output_hidden_states = bool(output_hidden_states)
        return_dict = bool(return_dict)

        token_type_ids = self._resolve_forward_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
        )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        need_hidden_states_for_z = int(self.z_steps) > 1
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=(output_hidden_states or need_hidden_states_for_z),
            output_attentions=output_attentions,
            return_dict=True,
            flash_meta=flash_meta,
        )

        sequence_output = encoder_outputs.last_hidden_state
        hidden_states = encoder_outputs.hidden_states

        if int(self.z_steps) > 1:
            if hidden_states is None or len(hidden_states) < 2:
                raise RuntimeError("z_steps>1 requires encoder hidden states.")
            z_base_states = hidden_states[-2]
            z_query_states = hidden_states[-1]
            last_layer = self.encoder.layer[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attn_mask = (
                self.encoder.get_attention_mask(attention_mask) if attention_mask is not None else None
            )
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            z_extras: list[torch.Tensor] = []
            for _ in range(int(self.z_steps) - 1):
                z_query_states, _ = last_layer(
                    z_base_states,
                    attn_mask,
                    output_attentions=False,
                    query_states=z_query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                    flash_meta=flash_meta,
                )
                z_extras.append(z_query_states)
            sequence_output = z_query_states
            if output_hidden_states and z_extras:
                hidden_states = tuple(hidden_states) + tuple(z_extras)

        if not return_dict:
            out: list[torch.Tensor | tuple[torch.Tensor, ...] | None] = [sequence_output]
            if output_hidden_states:
                out.append(hidden_states)
            if output_attentions:
                out.append(encoder_outputs.attentions)
            return tuple(x for x in out if x is not None)

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )

    def _forward_dense_resolved(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        flash_meta: FlashBatchMeta | None = None,
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]:
        """Run forward on the dense no-mask path with resolved flags.

        Ordinary dense batches carry no Flash metadata. The optional bundle is
        reserved for the route-bound ``local_bias`` compile specialization.

        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param bool output_attentions: Resolved attention-output flag.
        :param bool output_hidden_states: Resolved hidden-state-output flag.
        :param bool return_dict: Resolved return-format flag.
        :param FlashBatchMeta | None flash_meta: Optional route-bound FlashDeBERTa metadata.
        :return BaseModelOutput | tuple[torch.Tensor, ...]: Model outputs.
        """

        return self._forward_resolved(
            input_ids=input_ids,
            attention_mask=None,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            flash_meta=flash_meta,
        )

    def _forward_masked_resolved(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        flash_meta: FlashBatchMeta | None = None,
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]:
        """Run forward on the masked path with resolved flags.

        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor attention_mask: Attention mask tensor.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param bool output_attentions: Resolved attention-output flag.
        :param bool output_hidden_states: Resolved hidden-state-output flag.
        :param bool return_dict: Resolved return-format flag.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :return BaseModelOutput | tuple[torch.Tensor, ...]: Model outputs.
        """

        return self._forward_resolved(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            flash_meta=flash_meta,
        )

    def _forward_dense_hs0(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        flash_meta: FlashBatchMeta | None = None,
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]:
        """Run the dense training fast path with hidden-state outputs disabled.

        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param FlashBatchMeta | None flash_meta: Optional route-bound FlashDeBERTa metadata.
        :return BaseModelOutput | tuple[torch.Tensor, ...]: Model outputs.
        """

        return self._forward_dense_resolved(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            flash_meta=flash_meta,
        )

    def _forward_dense_hs1(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        flash_meta: FlashBatchMeta | None = None,
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]:
        """Run the dense training fast path with hidden-state outputs enabled.

        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param FlashBatchMeta | None flash_meta: Optional route-bound FlashDeBERTa metadata.
        :return BaseModelOutput | tuple[torch.Tensor, ...]: Model outputs.
        """

        return self._forward_dense_resolved(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            flash_meta=flash_meta,
        )

    def _forward_masked_hs0(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        flash_meta: FlashBatchMeta | None = None,
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]:
        """Run the masked training fast path with hidden-state outputs disabled.

        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor attention_mask: Attention mask tensor.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :return BaseModelOutput | tuple[torch.Tensor, ...]: Model outputs.
        """

        return self._forward_masked_resolved(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            flash_meta=flash_meta,
        )

    def _forward_masked_hs1(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        flash_meta: FlashBatchMeta | None = None,
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]:
        """Run the masked training fast path with hidden-state outputs enabled.

        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor attention_mask: Attention mask tensor.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :return BaseModelOutput | tuple[torch.Tensor, ...]: Model outputs.
        """

        return self._forward_masked_resolved(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            flash_meta=flash_meta,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        flash_meta: FlashBatchMeta | None = None,
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]:
        """Run DeBERTa-v2 encoder forward pass.

        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param bool | None output_attentions: Optional attention-output flag.
        :param bool | None output_hidden_states: Optional hidden-state-output flag.
        :param bool | None return_dict: Optional return-format flag.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :raises ValueError: If both/neither ``input_ids`` and ``inputs_embeds`` are set.
        :return BaseModelOutput | tuple[torch.Tensor, ...]: Model outputs.
        """

        (
            resolved_output_attentions,
            resolved_output_hidden_states,
            resolved_return_dict,
        ) = self._resolve_forward_options(
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return self._forward_resolved(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=resolved_output_attentions,
            output_hidden_states=resolved_output_hidden_states,
            return_dict=resolved_return_dict,
            flash_meta=flash_meta,
        )


__all__ = [
    "DebertaV2Config",
    "DebertaV2Model",
    "DebertaV2Embeddings",
    "DebertaV2Encoder",
    "DebertaV2Layer",
    "DisentangledSelfAttention",
    "build_relative_position",
]
