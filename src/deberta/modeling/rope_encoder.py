"""Modernized RoPE-based encoder backbone for DeBERTa-style models."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from deberta.modeling.norm import RMSNorm
from deberta.modeling.rope import RotaryEmbedding

try:
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import BaseModelOutput
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "transformers is required for the RoPE backbone (PreTrainedModel/PretrainedConfig)."
    ) from e


class DebertaRoPEConfig(PretrainedConfig):
    """Config for the modernized RoPE encoder backbone."""

    model_type = "deberta-rope"

    def __init__(
        self,
        *,
        vocab_size: int = 50265,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        ffn_type: str = "swiglu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        pad_token_id: int = 0,
        rope_theta: float = 10000.0,
        rotary_pct: float = 1.0,
        use_absolute_position_embeddings: bool = False,
        norm_eps: float = 1e-6,
        norm_arch: str = "post",
        keel_alpha_init: float | None = None,
        keel_alpha_learnable: bool = False,
        attention_implementation: str = "sdpa",
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        """Initialize encoder config fields and validate constraints.

        :param int vocab_size: Vocabulary size.
        :param int hidden_size: Hidden width.
        :param int num_hidden_layers: Number of encoder layers.
        :param int num_attention_heads: Number of attention heads.
        :param int intermediate_size: FFN intermediate width.
        :param str hidden_act: FFN activation name.
        :param str ffn_type: FFN block type (``swiglu`` or ``mlp``).
        :param float hidden_dropout_prob: Hidden dropout probability.
        :param float attention_probs_dropout_prob: Attention dropout probability.
        :param int max_position_embeddings: Maximum supported positions.
        :param int type_vocab_size: Token type vocabulary size.
        :param int pad_token_id: Padding token id.
        :param float rope_theta: RoPE base theta.
        :param float rotary_pct: Fraction of head dim receiving RoPE.
        :param bool use_absolute_position_embeddings: Whether to add learned absolute positions.
        :param float norm_eps: RMSNorm epsilon.
        :param str norm_arch: Residual topology name.
        :param float | None keel_alpha_init: Optional KEEL alpha init.
        :param bool keel_alpha_learnable: Whether KEEL alpha is trainable.
        :param str attention_implementation: Attention backend.
        :param float initializer_range: Weight initialization std.
        :param Any kwargs: Extra ``PretrainedConfig`` kwargs.
        """
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size)
        self.hidden_act = str(hidden_act)
        self.ffn_type = str(ffn_type)
        self.hidden_dropout_prob = float(hidden_dropout_prob)
        self.attention_probs_dropout_prob = float(attention_probs_dropout_prob)
        self.max_position_embeddings = int(max_position_embeddings)
        self.type_vocab_size = int(type_vocab_size)
        self.rope_theta = float(rope_theta)
        self.rotary_pct = float(rotary_pct)
        self.use_absolute_position_embeddings = bool(use_absolute_position_embeddings)
        self.norm_eps = float(norm_eps)
        self.norm_arch = str(norm_arch)
        self.keel_alpha_init = float(keel_alpha_init) if keel_alpha_init is not None else None
        self.keel_alpha_learnable = bool(keel_alpha_learnable)
        self.attention_implementation = str(attention_implementation)
        self.initializer_range = float(initializer_range)

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})."
            )
        if self.rotary_pct <= 0.0 or self.rotary_pct > 1.0:
            raise ValueError("rotary_pct must be in (0, 1].")
        if self.ffn_type not in {"swiglu", "mlp"}:
            raise ValueError("ffn_type must be one of: swiglu|mlp")
        if self.norm_arch not in {"post", "keel"}:
            raise ValueError("norm_arch must be one of: post|keel")
        if self.attention_implementation not in {"sdpa", "eager"}:
            raise ValueError("attention_implementation must be one of: sdpa|eager")


def _get_act_fn(name: str) -> Any:
    """Resolve activation callable from transformers registry.

    :param str name: Activation function key.
    :return Any: Activation callable.
    """
    from transformers.activations import ACT2FN

    return ACT2FN[name]


class DebertaRoPEEmbeddings(nn.Module):
    """Embedding stack combining token, optional type/position, norm, and dropout."""

    def __init__(self, config: DebertaRoPEConfig) -> None:
        """Create embedding layers.

        :param DebertaRoPEConfig config: Backbone configuration.
        """
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        if config.type_vocab_size and config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        else:
            self.token_type_embeddings = None

        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        else:
            self.position_embeddings = None

        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Embed token ids and apply normalization/dropout.

        :param torch.Tensor input_ids: Input token ids.
        :param torch.Tensor | None token_type_ids: Optional segment ids.
        :return torch.Tensor: Embedded hidden states.
        """
        bsz, seq_len = input_ids.shape
        x = self.word_embeddings(input_ids)

        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros((bsz, seq_len), dtype=torch.long, device=input_ids.device)
            x = x + self.token_type_embeddings(token_type_ids)

        if self.position_embeddings is not None:
            position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
            )
            x = x + self.position_embeddings(position_ids)

        x = self.norm(x)
        x = self.dropout(x)
        return x


class DebertaRoPESelfAttention(nn.Module):
    """Multi-head self-attention block with optional SDPA backend and RoPE."""

    def __init__(self, config: DebertaRoPEConfig) -> None:
        """Create attention projections and rotary embedding helper.

        :param DebertaRoPEConfig config: Backbone configuration.
        """
        super().__init__()
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_heads

        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.attn_dropout = float(config.attention_probs_dropout_prob)
        self.resid_dropout = float(config.hidden_dropout_prob)
        self.attn_impl = str(config.attention_implementation)

        rotary_dim = int(self.head_dim * float(config.rotary_pct))
        rotary_dim = rotary_dim - (rotary_dim % 2)  # ensure even
        self.rotary_dim = rotary_dim
        self.rope = RotaryEmbedding(rotary_dim, base=float(config.rope_theta)) if rotary_dim > 0 else None

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Run self-attention.

        :param torch.Tensor x: Input hidden states.
        :param torch.Tensor | None attention_mask: Binary attention mask.
        :return torch.Tensor: Attention output states.
        """
        bsz, seq_len, _ = x.shape

        qkv = self.qkv(x)  # (B,S,3H)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, nh, S, hd)

        if self.rope is not None:
            q, k = self.rope.apply(q, k)

        attn_mask = None
        if attention_mask is not None:
            # attention_mask is 1 for tokens, 0 for pad. SDPA expects True = mask.
            attn_mask = attention_mask.eq(0)[:, None, None, :]  # (B,1,1,S) bool

        if self.attn_impl == "sdpa" and hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            # Eager attention (debug/fallback)
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, nh, S, S)
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        out = self.out_proj(out)
        out = F.dropout(out, p=self.resid_dropout, training=self.training)
        return out


class DebertaRoPEMLP(nn.Module):
    """Feed-forward block used inside each encoder layer.

    Supported modes:
      - ``mlp``: Linear -> activation -> Linear
      - ``swiglu``: fused gate+up projection, SiLU gate, then down projection
    """

    def __init__(self, config: DebertaRoPEConfig) -> None:
        """Create FFN projections.

        :param DebertaRoPEConfig config: Backbone configuration.
        """
        super().__init__()
        self.ffn_type = str(getattr(config, "ffn_type", "swiglu")).lower()
        if self.ffn_type not in {"swiglu", "mlp"}:
            raise ValueError(f"Unsupported ffn_type: {self.ffn_type}")

        if self.ffn_type == "swiglu":
            hidden_size = int(config.hidden_size)
            intermediate = int(config.intermediate_size)
            # Fused projection: one matmul for gate+up, one for down projection.
            self.w12 = nn.Linear(hidden_size, 2 * intermediate, bias=True)
            self.w3 = nn.Linear(intermediate, hidden_size, bias=True)
        else:
            self.dense_in = nn.Linear(int(config.hidden_size), int(config.intermediate_size))
            self.act = _get_act_fn(config.hidden_act)
            self.dense_out = nn.Linear(int(config.intermediate_size), int(config.hidden_size))

        self.dropout = nn.Dropout(float(config.hidden_dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN transform.

        :param torch.Tensor x: Input hidden states.
        :return torch.Tensor: Output hidden states.
        """
        if self.ffn_type == "swiglu":
            gate, up = self.w12(x).chunk(2, dim=-1)
            x = self.w3(F.silu(gate) * up)
            x = self.dropout(x)
            return x

        x = self.dense_in(x)
        x = self.act(x)
        x = self.dense_out(x)
        x = self.dropout(x)
        return x


class _KEELAlpha(nn.Module):
    """Optional learnable scalar alpha for KEEL."""

    def __init__(self, init: float, learnable: bool) -> None:
        """Create KEEL alpha scaling module.

        :param float init: Initial alpha value.
        :param bool learnable: Whether alpha is a parameter.
        """
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(float(init), dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(float(init), dtype=torch.float32), persistent=False)
        self.learnable = bool(learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return alpha cast to the runtime dtype.

        :param torch.Tensor x: Reference tensor for dtype.
        :return torch.Tensor: Alpha scalar tensor.
        """
        return self.alpha.to(dtype=x.dtype)


class DebertaRoPELayer(nn.Module):
    """One encoder layer with attention + MLP.

    Norm placement is controlled by config.norm_arch:
      - 'post' : classic Post-Norm (RMSNorm after each residual addition)
      - 'keel' : KEEL-style: RMSNorm(alpha*x + F(RMSNorm(x))) per sub-layer
    """

    def __init__(self, config: DebertaRoPEConfig, *, alpha_init: float) -> None:
        """Create one encoder layer.

        :param DebertaRoPEConfig config: Backbone configuration.
        :param float alpha_init: KEEL alpha initial value.
        """
        super().__init__()
        self.config = config
        self.norm_arch = str(config.norm_arch)

        self.attn = DebertaRoPESelfAttention(config)
        self.mlp = DebertaRoPEMLP(config)

        if self.norm_arch == "post":
            self.norm1 = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.norm2 = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        else:
            # KEEL: inner norm + outer norm per sublayer
            self.inner_norm1 = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.outer_norm1 = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.inner_norm2 = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.outer_norm2 = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            self.alpha1 = _KEELAlpha(alpha_init, learnable=config.keel_alpha_learnable)
            self.alpha2 = _KEELAlpha(alpha_init, learnable=config.keel_alpha_learnable)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Run one encoder layer pass.

        :param torch.Tensor x: Input hidden states.
        :param torch.Tensor | None attention_mask: Binary attention mask.
        :return torch.Tensor: Output hidden states.
        """
        if self.norm_arch == "post":
            h = self.attn(x, attention_mask)
            x = self.norm1(x + self.dropout(h))
            h = self.mlp(x)
            x = self.norm2(x + self.dropout(h))
            return x

        # KEEL
        h = self.inner_norm1(x)
        h = self.attn(h, attention_mask)
        x = self.outer_norm1(self.alpha1(x) * x + self.dropout(h))

        h = self.inner_norm2(x)
        h = self.mlp(h)
        x = self.outer_norm2(self.alpha2(x) * x + self.dropout(h))
        return x


class DebertaRoPEEncoder(nn.Module):
    """Stack of RoPE encoder layers with optional gradient checkpointing."""

    def __init__(self, config: DebertaRoPEConfig) -> None:
        """Create encoder layer stack.

        :param DebertaRoPEConfig config: Backbone configuration.
        """
        super().__init__()
        self.config = config

        if config.keel_alpha_init is not None:
            alpha_init = float(config.keel_alpha_init)
        else:
            # Per user guidance: alpha defaults to 2 * num_layers (number of sublayers).
            alpha_init = float(2 * int(config.num_hidden_layers))

        self.layers = nn.ModuleList(
            [DebertaRoPELayer(config, alpha_init=alpha_init) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Run all encoder layers.

        :param torch.Tensor x: Input hidden states.
        :param torch.Tensor | None attention_mask: Binary attention mask.
        :return torch.Tensor: Output hidden states.
        """
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)
        return x


class DebertaRoPEPreTrainedModel(PreTrainedModel):
    """HF ``PreTrainedModel`` base for RoPE encoder models."""

    config_class = DebertaRoPEConfig
    base_model_prefix = "deberta_rope"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:  # pragma: no cover (HF init contracts)
        """Initialize module weights using config-defined initializer range.

        :param nn.Module module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            if getattr(module, "weight", None) is not None:
                module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        """Enable or disable checkpointing on compatible encoder modules.

        :param nn.Module module: Candidate module.
        :param bool value: Desired checkpointing state.
        """
        if isinstance(module, DebertaRoPEEncoder):
            module.gradient_checkpointing = value


class DebertaRoPEModel(DebertaRoPEPreTrainedModel):
    """Encoder-only model returning last_hidden_state."""

    def __init__(self, config: DebertaRoPEConfig) -> None:
        """Create encoder-only model.

        :param DebertaRoPEConfig config: Backbone configuration.
        """
        super().__init__(config)
        self.embeddings = DebertaRoPEEmbeddings(config)
        self.encoder = DebertaRoPEEncoder(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Return input word embedding module.

        :return nn.Module: Word embedding module.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Replace input word embedding module.

        :param nn.Module value: New word embedding module.
        """
        self.embeddings.word_embeddings = value  # type: ignore[assignment]

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> BaseModelOutput:
        """Run encoder forward pass.

        :param torch.Tensor input_ids: Input token ids.
        :param torch.Tensor | None attention_mask: Optional attention mask. When omitted, we
            reconstruct a pad mask from ``input_ids`` only if pad tokens are present; otherwise we
            keep ``None`` to preserve SDPA flash-friendly unpadded execution.
        :param torch.Tensor | None token_type_ids: Optional segment ids.
        :param bool return_dict: Whether to return HF output dataclass.
        :param Any kwargs: Additional compatibility kwargs forwarded by callers.
        :return BaseModelOutput: Last hidden states container.
        """
        if attention_mask is None:
            pad_id = getattr(self.config, "pad_token_id", None)
            if pad_id is not None:
                pad_positions = input_ids.eq(int(pad_id))
                # Keep the fast path for packed/unpadded batches (mask stays None), but make
                # omitted-mask calls with padded input_ids semantically correct.
                if bool(pad_positions.any().item()):
                    attention_mask = (~pad_positions).to(dtype=torch.long)

        if attention_mask is not None and attention_mask.dtype != torch.long:
            attention_mask = attention_mask.to(dtype=torch.long)

        x = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        x = self.encoder(x, attention_mask)

        if not return_dict:
            return (x,)

        return BaseModelOutput(last_hidden_state=x)
