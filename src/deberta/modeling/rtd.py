"""RTD (ELECTRA/DeBERTa-v3 style) pretraining heads and wrapper module.

This file intentionally keeps *all* ELECTRA/RTD-specific logic in one place.
The rest of the codebase can treat generator/discriminator backbones as plain
``transformers``-style encoder models.

Key parity targets vs the original Microsoft DeBERTa RTD codebase:

1) **Enhanced Mask Decoder (EMD)** for generator MLM when
   ``position_biased_input=False``.
   - DeBERTa-v2/v3 do not add absolute position embeddings to the input.
   - EMD re-runs the *last* encoder layer twice, using a position-only query
     (z_states) attending over the penultimate layer states.

2) **Discriminator head** matches DeBERTa RTD:
   ``LayerNorm(CLS + token) -> Dense -> Act -> Linear(1)``
   - Note the CLS conditioning happens *before* the dense+activation.
   - The original implementation does **not** apply dropout here.

3) **Masking / label contracts**:
   - The rest of this repo uses HF-style ``labels=-100`` ignore index.
   - The original DeBERTa code uses ``labels=0`` as ignore.
   - This module is written to work with ``-100`` labels (recommended).

Modernization goals preserved:
- torch.compile friendly (we keep RTD glue eager; compile backbones/FFNs).
- FSDP2 friendly (no parameter re-creation in forward, stable param names).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from deberta.modeling.activations import get_act_fn
from deberta.modeling.mask_utils import normalize_keep_mask
from deberta.modeling.norm import RMSNorm

# -----------------------------------------------------------------------------
# Mask utilities
# -----------------------------------------------------------------------------


def attention_mask_to_active_tokens(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pad_token_id: int | None,
) -> torch.Tensor:
    """Convert optional attention mask variants into a 2D active-token mask.

    We support the mask layouts used throughout this codebase:
    - (B,S) key-padding mask
    - (B,S,S) pairwise keep mask
    - (B,H,S,S) head-specific keep mask

    :param torch.Tensor input_ids: Input ids with shape ``(B,S)``.
    :param torch.Tensor | None attention_mask: Optional keep mask in rank-2/3/4 layout.
    :param int | None pad_token_id: Optional padding id used when inferring activity.
    :return torch.Tensor: Boolean active-token mask with shape ``(B,S)``.
    """

    if attention_mask is None:
        if pad_token_id is None:
            return torch.ones_like(input_ids, dtype=torch.bool)
        return input_ids.ne(int(pad_token_id))

    mask = normalize_keep_mask(attention_mask)
    if mask.ndim == 2:
        return mask

    if mask.ndim == 3:
        # Diagonal encodes per-token query activity.
        active = torch.diagonal(mask, dim1=-2, dim2=-1)
        if pad_token_id is not None:
            active = active & input_ids.ne(int(pad_token_id))
        return active

    if mask.ndim == 4:
        # Reduce head dimension if present.
        squeezed = mask[:, 0] if mask.shape[1] == 1 else mask.any(dim=1)
        if squeezed.shape[-2] == 1:
            # Broadcast path: (B,1,1,S) -> (B,S)
            active = squeezed[:, 0, :]
        else:
            active = torch.diagonal(squeezed, dim1=-2, dim2=-1)
        if pad_token_id is not None:
            active = active & input_ids.ne(int(pad_token_id))
        return active

    raise ValueError("attention_mask must have shape (B,S), (B,S,S), or (B,H,S,S).")


def _ensure_emd_pairwise_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Convert mask to a DeBERTa-style pairwise keep mask (B,1,S,S).

    The original DeBERTa EMD code expands 2D input masks to a full pairwise mask
    via an outer product. This is *not* strictly necessary with our attention
    implementation (which supports broadcast masks), but keeping this conversion
    improves parity.

    :param torch.Tensor attention_mask: Input keep mask in rank-2/3/4 layout.
    :return torch.Tensor: Pairwise keep mask with shape ``(B,1,S,S)``.
    """

    m = normalize_keep_mask(attention_mask)

    # 2D: (B,S) -> (B,1,S,S) using outer product.
    if m.ndim == 2:
        # (B,1,1,S)
        ext = m[:, None, None, :]
        # Outer product: key mask * query mask.
        # (B,1,1,S) * (B,1,S,1) -> (B,1,S,S)
        return ext & ext.transpose(-1, -2)

    # 3D: (B,S,S) -> (B,1,S,S)
    if m.ndim == 3:
        return m[:, None, :, :]

    # 4D: (B,H,S,S) -> (B,1,S,S)
    if m.ndim == 4:
        if m.shape[1] == 1:
            return m
        return m.any(dim=1, keepdim=True)

    raise ValueError(f"Unsupported attention_mask rank for EMD: {m.ndim}")


# -----------------------------------------------------------------------------
# Embedding sharing (ES / GDES)
# -----------------------------------------------------------------------------


class _SyncedBufferEmbedding(nn.Module):
    """Compile-safe embedding used for GDES.

    Effective output is:
        E(ids) = base_weight[ids] + bias[ids]

    - ``base_weight``: non-trainable Parameter (requires_grad=False)
      synced from generator weights.
    - ``bias``: trainable parameter (same shape as base_weight).

    This mirrors the original DeBERTa "gdes" sharing, but implemented in a way
    that remains compatible with torch.compile + FSDP2 (stable parameter objects).
    """

    def __init__(
        self,
        *,
        init_weight: torch.Tensor,
        padding_idx: int | None,
        add_bias: bool,
    ) -> None:
        """Initialize synced embedding buffers.

        :param torch.Tensor init_weight: Source embedding matrix.
        :param int | None padding_idx: Optional padding index.
        :param bool add_bias: Whether to create a trainable additive bias table.
        """
        super().__init__()
        if not isinstance(init_weight, torch.Tensor) or init_weight.ndim != 2:
            raise ValueError("init_weight must be a rank-2 tensor")

        # base_weight must be a Parameter so version counters update on sync copy_.
        self.base_weight = nn.Parameter(init_weight.detach().clone(), requires_grad=False)
        self.padding_idx = int(padding_idx) if padding_idx is not None else None

        self.bias: torch.nn.Parameter | None
        if add_bias:
            self.bias = nn.Parameter(torch.zeros_like(init_weight))
        else:
            self.bias = None

    @torch.no_grad()
    def sync_from(self, weight: torch.Tensor) -> None:
        """Copy generator weights into the local base buffer."""
        if weight.shape != self.base_weight.shape:
            raise ValueError(
                f"sync_from shape mismatch: src={tuple(weight.shape)} dst={tuple(self.base_weight.shape)}"
            )
        src = weight.detach()
        if src.device != self.base_weight.device:
            src = src.to(device=self.base_weight.device)
        if src.dtype != self.base_weight.dtype:
            src = src.to(dtype=self.base_weight.dtype)
        self.base_weight.copy_(src)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids with frozen base weights plus optional trainable bias.

        :param torch.Tensor input_ids: Token ids.
        :return torch.Tensor: Embedded states.
        """
        out = F.embedding(input_ids, self.base_weight, padding_idx=self.padding_idx)
        if self.bias is not None:
            out = out + F.embedding(input_ids, self.bias, padding_idx=self.padding_idx)
        return out


# -----------------------------------------------------------------------------
# Generator MLM head (+ EMD)
# -----------------------------------------------------------------------------


class MLMTransform(nn.Module):
    """DeBERTa-style MLM transform: Dense -> Act -> Norm.

    **Important**: This transform projects from hidden_size -> embedding_size.

    The original DeBERTa implementation uses ``embedding_size`` here so that the
    output projection can be tied directly to the input word embeddings (which
    live in embedding space even when hidden_size differs).
    """

    def __init__(self, config: Any) -> None:
        """Build MLM transform layers.

        :param Any config: Backbone config with hidden/embedding dimensions and activation settings.
        """
        super().__init__()
        hidden_size = int(config.hidden_size)
        embedding_size = int(getattr(config, "embedding_size", hidden_size))

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.dense = nn.Linear(hidden_size, embedding_size)
        self.act = get_act_fn(getattr(config, "hidden_act", "gelu"))

        eps = float(getattr(config, "norm_eps", getattr(config, "layer_norm_eps", 1e-6)))
        if bool(getattr(config, "use_rmsnorm_heads", False)):
            # RMSNorm is a modernization option. For strict DeBERTa parity, keep this False.
            self.norm = RMSNorm(embedding_size, eps=eps)
        else:
            self.norm = nn.LayerNorm(embedding_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Transform hidden states into embedding space.

        :param torch.Tensor hidden_states: Hidden states in ``hidden_size`` space.
        :return torch.Tensor: Transformed states in ``embedding_size`` space.
        """
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.norm(x)
        return x


class MaskedLMHead(nn.Module):
    """Masked LM head with optional weight tying.

    Tied mode:
        logits = (transform(h) @ word_embedding_weight.T) + bias

    Untied mode:
        logits = decoder(transform(h))

    Notes:
        - We keep the bias as a dedicated Parameter in tied mode.
        - We avoid casting the full embedding matrix under mixed precision; we cast
          activations when needed.
    """

    def __init__(self, config: Any, *, tie_word_embeddings: bool = True) -> None:
        """Initialize MLM head.

        :param Any config: Backbone config with vocab and hidden sizes.
        :param bool tie_word_embeddings: Whether to project with tied input embeddings.
        """
        super().__init__()
        self.transform = MLMTransform(config)
        self.vocab_size = int(config.vocab_size)

        self.tie_word_embeddings = bool(tie_word_embeddings)
        if self.tie_word_embeddings:
            self.decoder = None
            self.bias = nn.Parameter(torch.zeros(self.vocab_size))
        else:
            self.decoder = nn.Linear(self.transform.embedding_size, self.vocab_size, bias=True)
            self.bias = self.decoder.bias

    def forward(
        self, hidden_states: torch.Tensor, *, word_embedding_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Project masked hidden states to vocabulary logits.

        :param torch.Tensor hidden_states: Hidden states for prediction positions.
        :param torch.Tensor | None word_embedding_weight: Optional tied embedding matrix.
        :return torch.Tensor: Vocabulary logits.
        """
        x = self.transform(hidden_states)

        if self.tie_word_embeddings:
            if word_embedding_weight is None:
                raise RuntimeError(
                    "MaskedLMHead requires `word_embedding_weight` when tie_word_embeddings=True."
                )
            if word_embedding_weight.shape[1] != x.shape[-1]:
                raise RuntimeError(
                    "Tied word_embedding_weight hidden size mismatch: "
                    f"got {word_embedding_weight.shape[1]}, expected {x.shape[-1]}."
                )

            w = word_embedding_weight
            b = self.bias
            # Never cast full embedding matrix; cast activations if needed.
            if not torch.is_autocast_enabled() and x.dtype != w.dtype:
                x = x.to(dtype=w.dtype)
            if b.dtype != w.dtype:
                b = b.to(dtype=w.dtype)
            return F.linear(x, w, b)

        if self.decoder is None:
            raise RuntimeError("MaskedLMHead decoder is not initialized.")
        return self.decoder(x)


class EnhancedMaskDecoder(nn.Module):
    """Enhanced Mask Decoder (EMD) for DeBERTa-v2/v3.

    The original DeBERTa implementation applies EMD only when
    ``position_biased_input=False``.

    Intuition:
      - When the model does *not* add absolute position embeddings to the input,
        masked token prediction can be improved by feeding a position-only query
        through the last layer.
      - EMD reuses the last encoder layer twice with shared weights.

    This implementation is intentionally written to be:
      - compatible with this repo's DebertaV2Encoder/DebertaV2Layer API
      - safe under torch.compile (EMD itself stays eager; the layer modules can
        still be compiled individually if desired)
    """

    def __init__(self, config: Any, *, num_last_layer_passes: int = 2) -> None:
        """Initialize Enhanced Mask Decoder.

        :param Any config: Generator backbone config.
        :param int num_last_layer_passes: Number of last-layer EMD reapplication passes.
        """
        super().__init__()
        self.position_biased_input = bool(getattr(config, "position_biased_input", True))
        self.num_passes = int(num_last_layer_passes)
        if self.num_passes < 1:
            raise ValueError("num_last_layer_passes must be >= 1")

    def _position_states(
        self,
        *,
        embeddings: nn.Module,
        position_ids: torch.Tensor,
        hidden_size: int,
    ) -> torch.Tensor:
        """Compute z_states (position embeddings) in hidden space.

        In the original DeBERTa code, z_states come from ``embeddings.position_embeddings``.
        If embedding_size != hidden_size, we project them with embeddings.embed_proj.

        :param nn.Module embeddings: Embeddings module exposing position embeddings.
        :param torch.Tensor position_ids: Position ids.
        :param int hidden_size: Expected hidden size.
        :return torch.Tensor: Position states projected to hidden space.
        """

        pos_mod = getattr(embeddings, "position_embeddings", None)
        if pos_mod is None:
            raise RuntimeError(
                "EnhancedMaskDecoder requires embeddings.position_embeddings. "
                "Your DeBERTa-v2 embeddings must instantiate it even when position_biased_input=False."
            )

        z = pos_mod(position_ids.long())

        # If the backbone uses a smaller embedding_size, project z to hidden_size.
        proj = getattr(embeddings, "embed_proj", None)
        if isinstance(proj, nn.Linear):
            z = proj(z)

        if z.shape[-1] != int(hidden_size):
            raise RuntimeError(
                "Position embedding hidden size mismatch after optional projection: "
                f"got {z.shape[-1]}, expected {hidden_size}."
            )
        return z

    def forward(
        self,
        *,
        encoder_hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
        masked_positions: torch.Tensor,
        attention_mask: torch.Tensor | None,
        embeddings: nn.Module,
        encoder: nn.Module,
        position_ids: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return hidden states for masked positions, using EMD when applicable.

        Args:
            encoder_hidden_states:
                Full hidden-state stack returned by the generator backbone.
                We expect at least 2 states so we can use penultimate as KV.
            masked_positions:
                Bool mask (B,S) marking MLM-supervised positions.
            attention_mask:
                Any supported attention-mask variant, or None.
            embeddings:
                Backbone embeddings module (must expose position_embeddings and optional embed_proj).
            encoder:
                Backbone encoder module (must expose .layer, last layer must accept query_states).
            position_ids:
                Optional position ids (B,S). If None, we create a 0..S-1 range.
            relative_pos:
                Optional precomputed relative-position ids (for disentangled attention).

        Returns:
            Tensor (N,H) of contextual states for masked positions.
        """

        if isinstance(encoder_hidden_states, tuple):
            hs = list(encoder_hidden_states)
        else:
            hs = list(encoder_hidden_states)

        if len(hs) < 2:
            raise RuntimeError(
                "EnhancedMaskDecoder requires output_hidden_states=True on the generator "
                "so we can access penultimate-layer states."
            )

        # Masked indices (flattened) used to gather token states.
        masked_flat = masked_positions.reshape(-1)
        masked_idx = torch.nonzero(masked_flat, as_tuple=False).squeeze(-1)
        if masked_idx.numel() == 0:
            # Degenerate: caller should handle, but return an empty (0,H) tensor.
            last = hs[-1]
            return last.reshape(-1, last.shape[-1]).index_select(0, masked_idx)

        # Parity with the original implementation:
        # - KV states come from the penultimate layer.
        # - For position_biased_input=True, just use the last layer.
        if self.position_biased_input:
            last = hs[-1]
            flat = last.reshape(-1, last.shape[-1])
            return flat.index_select(0, masked_idx)

        # --- EMD path (position_biased_input=False) ---
        # KV from penultimate layer.
        kv_states = hs[-2]
        bsz, seq_len, hidden_size = kv_states.shape

        # Build/normalize attention mask.
        if attention_mask is None:
            # If the caller didn't provide a mask, assume unpadded input.
            # We still build an all-True keep mask for simplicity.
            keep_2d = torch.ones((bsz, seq_len), dtype=torch.bool, device=kv_states.device)
            attn = _ensure_emd_pairwise_attention_mask(keep_2d)
        else:
            attn = _ensure_emd_pairwise_attention_mask(attention_mask)

        # Position ids default: 0..S-1.
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=kv_states.device).unsqueeze(0).expand(bsz, -1)
        else:
            position_ids = position_ids.to(device=kv_states.device)

        # z_states in hidden space.
        z_states = self._position_states(
            embeddings=embeddings, position_ids=position_ids, hidden_size=hidden_size
        )

        # Initial query = z + KV (matches original `z_states += hidden_states`).
        query_states = z_states + kv_states

        # Resolve relative-position ids and relative embedding table if available.
        rel_embeddings = None
        get_rel_embedding = getattr(encoder, "get_rel_embedding", None)
        if callable(get_rel_embedding):
            rel_embeddings = get_rel_embedding()

        rel_pos = relative_pos
        get_rel_pos = getattr(encoder, "get_rel_pos", None)
        if callable(get_rel_pos):
            # Note: query_len==key_len==seq_len here.
            rel_pos = get_rel_pos(kv_states, query_states=query_states, relative_pos=relative_pos)

        # The original runs the *last layer* twice, sharing weights.
        layers = getattr(encoder, "layer", None)
        if not isinstance(layers, (nn.ModuleList, list, tuple)) or len(layers) == 0:
            raise RuntimeError("EnhancedMaskDecoder expects encoder.layer to be a non-empty sequence")
        last_layer = layers[-1]

        outputs: list[torch.Tensor] = []
        for _ in range(self.num_passes):
            # DebertaV2Layer signature: (hidden_states, attention_mask, ..., query_states=...)
            out, _att = last_layer(
                kv_states,
                attn,
                output_attentions=False,
                query_states=query_states,
                relative_pos=rel_pos,
                rel_embeddings=rel_embeddings,
            )
            query_states = out
            outputs.append(out)

        # Gather masked positions from the final pass.
        final = outputs[-1]
        flat = final.reshape(-1, final.shape[-1])
        return flat.index_select(0, masked_idx)


# -----------------------------------------------------------------------------
# Discriminator RTD head
# -----------------------------------------------------------------------------


class RTDHead(nn.Module):
    """DeBERTa RTD discriminator head.

    Matches the original ordering:
        ctx = hidden[:,0]
        x = LayerNorm(ctx + hidden)
        x = Dense(x)
        x = Act(x)
        logits = Linear(x)

    Note:
        The original code does not apply dropout in this head.
        If you want dropout for regularization, do it in the backbone.
    """

    def __init__(self, config: Any) -> None:
        """Initialize discriminator token classifier head.

        :param Any config: Discriminator backbone config.
        """
        super().__init__()
        hidden_size = int(config.hidden_size)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = get_act_fn(getattr(config, "hidden_act", "gelu"))

        eps = float(getattr(config, "norm_eps", getattr(config, "layer_norm_eps", 1e-6)))
        if bool(getattr(config, "use_rmsnorm_heads", False)):
            self.norm = RMSNorm(hidden_size, eps=eps)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=eps)

        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute per-token replacement logits.

        :param torch.Tensor hidden_states: Discriminator hidden states ``(B,S,H)``.
        :return torch.Tensor: Per-token logits ``(B,S)``.
        """
        # hidden_states: (B,S,H)
        ctx = hidden_states[:, 0:1, :]  # (B,1,H)
        x = self.norm(hidden_states + ctx)
        x = self.dense(x)
        x = self.act(x)
        return self.classifier(x).squeeze(-1)  # (B,S)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------


@dataclass
class RTDOutput:
    """Structured outputs from one RTD forward pass."""

    loss: torch.Tensor
    gen_loss: torch.Tensor
    disc_loss: torch.Tensor
    disc_accuracy: torch.Tensor
    gen_token_count: torch.Tensor
    disc_token_count: torch.Tensor
    disc_positive_count: torch.Tensor
    gen_loss_raw: torch.Tensor
    disc_loss_raw: torch.Tensor


@dataclass
class RTDGeneratorPhaseOutput:
    """Outputs from the generator-only phase in decoupled RTD training."""

    gen_loss_raw: torch.Tensor
    gen_token_count: torch.Tensor
    corrupted_input_ids: torch.Tensor
    disc_labels: torch.Tensor


@dataclass
class RTDDiscriminatorPhaseOutput:
    """Outputs from the discriminator-only phase in decoupled RTD training."""

    disc_loss_raw: torch.Tensor
    disc_accuracy: torch.Tensor
    disc_token_count: torch.Tensor
    disc_positive_count: torch.Tensor


# -----------------------------------------------------------------------------
# Main wrapper: Generator + Discriminator
# -----------------------------------------------------------------------------


class DebertaV3RTDPretrainer(nn.Module):
    """Generator + discriminator pretraining module (DeBERTaV3 / ELECTRA objective)."""

    def __init__(
        self,
        *,
        discriminator_backbone: nn.Module,
        generator_backbone: nn.Module,
        disc_config: Any,
        gen_config: Any,
        embedding_sharing: str = "gdes",
        tie_generator_word_embeddings: bool = True,
        use_enhanced_mask_decoder: bool = True,
        additional_forbidden_token_ids: Iterable[int] | None = None,
    ) -> None:
        """Initialize RTD pretrainer wrapper.

        :param nn.Module discriminator_backbone: Discriminator encoder backbone.
        :param nn.Module generator_backbone: Generator encoder backbone.
        :param Any disc_config: Discriminator config.
        :param Any gen_config: Generator config.
        :param str embedding_sharing: Embedding-sharing policy (none|es|gdes).
        :param bool tie_generator_word_embeddings: Whether MLM head ties word embeddings.
        :param bool use_enhanced_mask_decoder: Whether EMD is enabled when applicable.
        :param Iterable[int] | None additional_forbidden_token_ids: Extra ids excluded from sampling.
        """
        super().__init__()
        self.disc_config = disc_config
        self.gen_config = gen_config

        self.generator = generator_backbone
        self.discriminator = discriminator_backbone

        # Generator heads
        self.generator_lm_head = MaskedLMHead(gen_config, tie_word_embeddings=tie_generator_word_embeddings)

        # EMD module (only active when gen_config.position_biased_input=False)
        self.use_enhanced_mask_decoder = bool(use_enhanced_mask_decoder)
        self.enhanced_mask_decoder = EnhancedMaskDecoder(gen_config, num_last_layer_passes=2)

        # Discriminator head
        self.discriminator_head = RTDHead(disc_config)

        self.embedding_sharing = str(embedding_sharing or "none")

        # Special ids excluded from generator sampling.
        self._forbidden_sample_token_ids = self._collect_forbidden_sample_token_ids(
            additional_forbidden_token_ids=additional_forbidden_token_ids
        )

        vocab_size = int(getattr(self.gen_config, "vocab_size", 0) or 0)
        self.register_buffer(
            "_forbidden_sample_token_mask",
            self._build_forbidden_token_mask(
                vocab_size=vocab_size, forbidden_ids=self._forbidden_sample_token_ids
            ),
            persistent=False,
        )

        # For GDES: list of (attr, disc_emb_module, gen_emb_module)
        self._gdes_synced_embeddings: list[tuple[str, _SyncedBufferEmbedding, nn.Module]] = []

        self._patch_discriminator_embeddings_for_sharing()

    # ------------------------------
    # Forbidden token mask helpers
    # ------------------------------

    @staticmethod
    def _build_forbidden_token_mask(*, vocab_size: int, forbidden_ids: set[int]) -> torch.Tensor:
        """Build boolean vocabulary mask for ids that must never be sampled.

        :param int vocab_size: Vocabulary size.
        :param set[int] forbidden_ids: Ids to exclude from sampling.
        :return torch.Tensor: Boolean vocabulary mask.
        """
        if vocab_size <= 0:
            return torch.empty(0, dtype=torch.bool)
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        valid = [int(tid) for tid in forbidden_ids if 0 <= int(tid) < vocab_size]
        if valid:
            mask[valid] = True
        return mask

    _SPECIAL_ID_ATTRS = (
        "pad_token_id",
        "cls_token_id",
        "sep_token_id",
        "mask_token_id",
        "unk_token_id",
        "bos_token_id",
        "eos_token_id",
    )

    def _collect_forbidden_sample_token_ids(
        self, *, additional_forbidden_token_ids: Iterable[int] | None = None
    ) -> set[int]:
        """Collect special token ids to exclude from generator sampling.

        This is an intentional divergence from strict DeBERTa parity to prevent
        sampled replacements from landing on control/special tokens.

        :param Iterable[int] | None additional_forbidden_token_ids: Optional extra forbidden ids.
        :return set[int]: Forbidden vocabulary ids.
        """
        vocab_size = int(getattr(self.gen_config, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            return set()

        candidates: list[Any] = []
        for cfg in (self.gen_config, self.disc_config):
            candidates.extend(getattr(cfg, attr, None) for attr in self._SPECIAL_ID_ATTRS)
        if additional_forbidden_token_ids is not None:
            candidates.extend(additional_forbidden_token_ids)

        out: set[int] = set()
        for sid in candidates:
            if sid is None:
                continue
            try:
                sid_i = int(sid)
            except Exception:
                continue
            if 0 <= sid_i < vocab_size:
                out.add(sid_i)
        return out

    # ------------------------------
    # Embedding sharing
    # ------------------------------

    def _patch_discriminator_embeddings_for_sharing(self) -> None:
        """Apply configured embedding sharing policy to discriminator embeddings."""

        mode = (self.embedding_sharing or "none").lower()
        if mode not in {"none", "es", "gdes"}:
            raise ValueError("embedding_sharing must be one of: none|es|gdes")
        if mode == "none":
            return

        try:
            gen_embeddings = self.generator.embeddings
            disc_embeddings = self.discriminator.embeddings
        except Exception as e:
            raise RuntimeError(
                "Expected backbones with an `.embeddings` module containing word_embeddings."
            ) from e

        attrs = ("word_embeddings", "position_embeddings", "token_type_embeddings")

        def _padding_idx(mod: nn.Module) -> int | None:
            """Extract optional padding index from an embedding module.

            :param nn.Module mod: Embedding-like module.
            :return int | None: Padding index when present.
            """
            val = getattr(mod, "padding_idx", None)
            return int(val) if val is not None else None

        def _weight(mod: nn.Module) -> torch.Tensor:
            """Extract embedding weight tensor.

            :param nn.Module mod: Embedding-like module.
            :return torch.Tensor: Embedding weight tensor.
            """
            w = getattr(mod, "weight", None)
            if not isinstance(w, torch.Tensor):
                raise RuntimeError("Embedding module must expose .weight tensor")
            return w

        def _validate(attr: str, gen_mod: nn.Module, disc_mod: nn.Module) -> torch.Tensor:
            """Validate embedding shape compatibility for sharing.

            :param str attr: Embedding attribute name.
            :param nn.Module gen_mod: Generator embedding module.
            :param nn.Module disc_mod: Discriminator embedding module.
            :return torch.Tensor: Generator embedding weights.
            """
            gw = _weight(gen_mod)
            dw = _weight(disc_mod)
            if gw.shape != dw.shape:
                raise ValueError(
                    f"Cannot share embeddings for '{attr}': generator shape {tuple(gw.shape)} != discriminator shape {tuple(dw.shape)}."
                )
            return gw

        if mode == "es":
            # Share Parameters directly.
            for attr in attrs:
                gen_mod = getattr(gen_embeddings, attr, None)
                disc_mod = getattr(disc_embeddings, attr, None)
                if gen_mod is None or disc_mod is None:
                    continue
                gw = _validate(attr, gen_mod, disc_mod)
                disc_mod.weight = gw  # type: ignore[assignment]
            return

        # GDES: discriminator uses frozen base weights + trainable bias.
        for attr in attrs:
            gen_mod = getattr(gen_embeddings, attr, None)
            disc_mod = getattr(disc_embeddings, attr, None)
            if gen_mod is None or disc_mod is None:
                continue
            gw = _validate(attr, gen_mod, disc_mod)

            synced = _SyncedBufferEmbedding(init_weight=gw, padding_idx=_padding_idx(disc_mod), add_bias=True)
            setattr(disc_embeddings, attr, synced)
            self._gdes_synced_embeddings.append((attr, synced, gen_mod))

    @torch.no_grad()
    def sync_discriminator_embeddings_from_generator(self) -> None:
        """Sync GDES base weights from generator embedding weights.

        Must be called after each optimizer step and after checkpoint load.
        """

        mode = (self.embedding_sharing or "none").lower()
        if mode != "gdes" or not self._gdes_synced_embeddings:
            return

        for attr, disc_mod, gen_mod in self._gdes_synced_embeddings:
            w = getattr(gen_mod, "weight", None)
            if not isinstance(w, torch.Tensor):
                raise RuntimeError(f"Generator embedding '{attr}' no longer exposes .weight")
            disc_mod.sync_from(w)

    # ------------------------------
    # Generator helpers
    # ------------------------------

    def _get_generator_word_embedding_weight(self) -> torch.Tensor:
        """Fetch generator word embedding weights used for MLM head tying.

        :return torch.Tensor: Generator word embedding matrix.
        """

        embeddings = getattr(self.generator, "embeddings", None)
        if embeddings is None:
            raise RuntimeError("Generator backbone must expose an `.embeddings` module.")

        word_emb = getattr(embeddings, "word_embeddings", None)
        if word_emb is None or not hasattr(word_emb, "weight"):
            raise RuntimeError("Generator word_embeddings must expose a `.weight` tensor for LM-head tying.")

        weight = word_emb.weight
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError("Generator word_embeddings.weight must be a torch.Tensor.")
        return weight

    # ------------------------------
    # Sampling
    # ------------------------------

    @staticmethod
    def _gumbel_sample(
        logits: torch.Tensor,
        *,
        temperature: float,
        forbidden_vocab_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Sample token ids from categorical logits via Gumbel-max.

        :param torch.Tensor logits: Vocabulary logits.
        :param float temperature: Sampling temperature.
        :param torch.Tensor | None forbidden_vocab_mask: Optional boolean mask of unsampleable ids.
        :return torch.Tensor: Sampled token ids.
        """

        temp = float(temperature)
        if temp <= 0:
            raise ValueError("sampling_temperature must be > 0")

        x = logits.float() / temp
        if forbidden_vocab_mask is not None and forbidden_vocab_mask.numel() != 0:
            if forbidden_vocab_mask.device != x.device:
                forbidden_vocab_mask = forbidden_vocab_mask.to(device=x.device)
            mask = forbidden_vocab_mask.to(dtype=torch.bool)
            if mask.ndim == 1:
                if int(mask.shape[0]) != int(x.shape[-1]):
                    raise ValueError(
                        "forbidden_vocab_mask length must match logits vocabulary dimension: "
                        f"{int(mask.shape[0])} vs {int(x.shape[-1])}."
                    )
                if bool(mask.all().item()):
                    raise ValueError(
                        "forbidden_vocab_mask excludes all vocabulary ids; at least one token must remain sampleable."
                    )
            else:
                if int(mask.shape[-1]) != int(x.shape[-1]):
                    raise ValueError(
                        "forbidden_vocab_mask last dimension must match logits vocabulary dimension: "
                        f"{int(mask.shape[-1])} vs {int(x.shape[-1])}."
                    )
                try:
                    expanded = mask.expand_as(x)
                except RuntimeError as exc:
                    raise ValueError(
                        "forbidden_vocab_mask with rank > 1 must be broadcastable to logits."
                    ) from exc
                if bool(expanded.all(dim=-1).any().item()):
                    raise ValueError(
                        "forbidden_vocab_mask excludes all vocabulary ids for at least one row; "
                        "at least one token must remain sampleable."
                    )
            x = x.masked_fill(mask, -1e9)

        # Gumbel noise
        u = torch.rand_like(x).clamp_(min=1e-6, max=1.0 - 1e-6)
        g = -torch.log(-torch.log(u))
        return torch.argmax(x + g, dim=-1)

    # ------------------------------
    # Forward
    # ------------------------------

    def forward_generator_phase(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        sampling_temperature: float = 1.0,
    ) -> RTDGeneratorPhaseOutput:
        """Run generator forward/corruption only, returning discriminator targets.

        :param torch.Tensor input_ids: Masked input ids.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param torch.Tensor labels: MLM labels with ``-100`` ignore index.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param float sampling_temperature: Generator sampling temperature.
        :return RTDGeneratorPhaseOutput: Generator loss and corruption artifacts.
        """

        if labels is None:
            raise ValueError("labels must be provided (MLM labels with -100 for unmasked positions)")

        ignore_index = -100
        masked_positions = labels.ne(ignore_index)  # (B,S)

        masked_flat = masked_positions.view(-1)
        masked_idx = torch.nonzero(masked_flat, as_tuple=False).squeeze(-1)  # (N,)
        gen_token_count = masked_flat.sum().to(dtype=torch.float32)

        # EMD is only applicable for DeBERTa-v2/v3 when position_biased_input=False.
        #
        # When the generator backbone already applies iterative last-layer passes
        # via z_steps>1, its output is already in the EMD-style regime; running the
        # standalone EMD module again would double-apply that path.
        pos_biased = bool(getattr(self.gen_config, "position_biased_input", True))
        z_steps = int(getattr(self.generator, "z_steps", getattr(self.gen_config, "z_steps", 0)) or 0)
        use_emd = bool(self.use_enhanced_mask_decoder) and (not pos_biased) and z_steps <= 1

        gen_forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "return_dict": True,
        }
        if use_emd:
            gen_forward_kwargs["output_hidden_states"] = True

        gen_out = self.generator(**gen_forward_kwargs)

        corrupted_input_ids = input_ids
        disc_labels = torch.zeros_like(input_ids, dtype=torch.float32)

        if masked_idx.numel() == 0:
            hidden = gen_out.last_hidden_state
            gen_loss = hidden.sum() * 0.0
            return RTDGeneratorPhaseOutput(
                gen_loss_raw=gen_loss,
                gen_token_count=gen_token_count.detach(),
                corrupted_input_ids=corrupted_input_ids.detach(),
                disc_labels=disc_labels.detach(),
            )

        if use_emd:
            gen_masked_hidden = self.enhanced_mask_decoder(
                encoder_hidden_states=gen_out.hidden_states,
                masked_positions=masked_positions,
                attention_mask=attention_mask,
                embeddings=self.generator.embeddings,
                encoder=self.generator.encoder,
            )
        else:
            hidden = gen_out.last_hidden_state
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            gen_masked_hidden = hidden_flat.index_select(0, masked_idx)

        labels_flat = labels.view(-1)
        masked_labels = labels_flat.index_select(0, masked_idx)

        word_w = self._get_generator_word_embedding_weight()
        gen_logits = self.generator_lm_head(gen_masked_hidden, word_embedding_weight=word_w)
        gen_loss = F.cross_entropy(gen_logits.float(), masked_labels)

        with torch.no_grad():
            forbidden_mask = getattr(self, "_forbidden_sample_token_mask", None)
            sampled = self._gumbel_sample(
                gen_logits,
                temperature=sampling_temperature,
                forbidden_vocab_mask=forbidden_mask,
            ).to(dtype=input_ids.dtype)

            corrupted_flat = input_ids.view(-1).scatter(0, masked_idx, sampled)
            corrupted_input_ids = corrupted_flat.view_as(input_ids)

            replaced = sampled.ne(masked_labels.to(dtype=sampled.dtype)).to(dtype=torch.float32)
            disc_labels_flat = torch.zeros(input_ids.numel(), dtype=torch.float32, device=input_ids.device)
            disc_labels_flat.scatter_(0, masked_idx, replaced)
            disc_labels = disc_labels_flat.view_as(input_ids)

        return RTDGeneratorPhaseOutput(
            gen_loss_raw=gen_loss,
            gen_token_count=gen_token_count.detach(),
            corrupted_input_ids=corrupted_input_ids.detach(),
            disc_labels=disc_labels.detach(),
        )

    def forward_discriminator_phase(
        self,
        *,
        input_ids: torch.Tensor,
        corrupted_input_ids: torch.Tensor,
        disc_labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> RTDDiscriminatorPhaseOutput:
        """Run discriminator scoring only, given prebuilt corrupted ids/labels.

        :param torch.Tensor input_ids: Input ids used only for active-token masking.
        :param torch.Tensor corrupted_input_ids: Corrupted ids sampled from generator logits.
        :param torch.Tensor disc_labels: Binary RTD labels.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :return RTDDiscriminatorPhaseOutput: Discriminator loss and metrics.
        """

        disc_out = self.discriminator(
            input_ids=corrupted_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        disc_hidden = disc_out.last_hidden_state
        disc_logits = self.discriminator_head(disc_hidden)

        pad_token_id = getattr(self.disc_config, "pad_token_id", None)
        active = attention_mask_to_active_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=int(pad_token_id) if pad_token_id is not None else None,
        )

        disc_active_f = active.to(dtype=torch.float32)
        disc_token_count = disc_active_f.sum()
        disc_denom = disc_token_count.clamp(min=1.0)

        disc_loss_per_token = F.binary_cross_entropy_with_logits(
            disc_logits.float(),
            disc_labels.float(),
            reduction="none",
        )
        disc_loss = (disc_loss_per_token * disc_active_f).sum() / disc_denom

        disc_pred = disc_logits.gt(0)
        disc_true = disc_labels.gt(0.5)
        disc_positive_count = (disc_true.to(dtype=torch.float32) * disc_active_f).sum()
        correct = (disc_pred == disc_true).to(dtype=torch.float32) * disc_active_f
        disc_acc = correct.sum() / disc_denom

        return RTDDiscriminatorPhaseOutput(
            disc_loss_raw=disc_loss,
            disc_accuracy=disc_acc.detach(),
            disc_token_count=disc_token_count.detach(),
            disc_positive_count=disc_positive_count.detach(),
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        sampling_temperature: float = 1.0,
        gen_loss_weight: float = 1.0,
        disc_loss_weight: float = 50.0,
    ) -> RTDOutput:
        """Run one RTD forward pass with generator corruption and discriminator scoring.

        :param torch.Tensor input_ids: Masked input ids.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param torch.Tensor labels: MLM labels with ``-100`` ignore index.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param float sampling_temperature: Generator sampling temperature.
        :param float gen_loss_weight: Generator loss weight.
        :param float disc_loss_weight: Discriminator loss weight.
        :return RTDOutput: Combined loss and detached metrics.
        """

        gen_phase = self.forward_generator_phase(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids,
            sampling_temperature=sampling_temperature,
        )
        if float(gen_phase.gen_token_count.item()) <= 0.0:
            disc_zero = torch.zeros((), device=input_ids.device, dtype=torch.float32)
            total = float(gen_loss_weight) * gen_phase.gen_loss_raw
            return RTDOutput(
                loss=total,
                gen_loss=gen_phase.gen_loss_raw.detach(),
                disc_loss=disc_zero.detach(),
                disc_accuracy=disc_zero.detach(),
                gen_token_count=gen_phase.gen_token_count.detach(),
                disc_token_count=disc_zero.detach(),
                disc_positive_count=disc_zero.detach(),
                gen_loss_raw=gen_phase.gen_loss_raw,
                disc_loss_raw=disc_zero,
            )

        disc_phase = self.forward_discriminator_phase(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            input_ids=input_ids,
            corrupted_input_ids=gen_phase.corrupted_input_ids,
            disc_labels=gen_phase.disc_labels,
        )

        total = (
            float(gen_loss_weight) * gen_phase.gen_loss_raw
            + float(disc_loss_weight) * disc_phase.disc_loss_raw
        )

        return RTDOutput(
            loss=total,
            gen_loss=gen_phase.gen_loss_raw.detach(),
            disc_loss=disc_phase.disc_loss_raw.detach(),
            disc_accuracy=disc_phase.disc_accuracy.detach(),
            gen_token_count=gen_phase.gen_token_count.detach(),
            disc_token_count=disc_phase.disc_token_count.detach(),
            disc_positive_count=disc_phase.disc_positive_count.detach(),
            gen_loss_raw=gen_phase.gen_loss_raw,
            disc_loss_raw=disc_phase.disc_loss_raw,
        )


__all__ = [
    "RTDOutput",
    "RTDGeneratorPhaseOutput",
    "RTDDiscriminatorPhaseOutput",
    "DebertaV3RTDPretrainer",
    "MaskedLMHead",
    "EnhancedMaskDecoder",
    "RTDHead",
]
