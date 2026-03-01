"""RTD (ELECTRA/DeBERTa-v3 style) pretraining heads and wrapper module.

Objective:
- Generator MLM cross entropy on masked positions.
- Discriminator replaced-token detection BCE on active tokens.

Compile boundary:
- RTD glue stays eager (sampling/corruption/label assembly).
- Training compiles backbone modules only (configured in training runtime).

Embedding sharing:
- ``none``: no sharing.
- ``es``: direct parameter sharing.
- ``gdes``: discriminator uses synced non-trainable base weights + trainable bias tensors.
  The training loop must call ``sync_discriminator_embeddings_from_generator()``
  after checkpoint load and after optimizer steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from deberta.modeling.activations import get_act_fn
from deberta.modeling.mask_utils import normalize_keep_mask
from deberta.modeling.norm import RMSNorm


def compute_generator_loss_term(
    *, gen_loss: torch.Tensor, disc_loss: torch.Tensor, decoupled_loss_scaling: bool
) -> torch.Tensor:
    """Compute the generator contribution to total RTD loss.

    :param torch.Tensor gen_loss: Raw generator MLM loss.
    :param torch.Tensor disc_loss: Raw discriminator RTD loss.
    :param bool decoupled_loss_scaling: Whether to scale generator loss by detached discriminator/gen ratio.
    :return torch.Tensor: Generator term used in total loss.
    """

    if not decoupled_loss_scaling:
        return gen_loss

    eps = 1e-6
    alpha = (disc_loss.detach() / (gen_loss.detach() + eps)).clamp(min=0.0, max=1e4)
    return alpha * gen_loss


def attention_mask_to_active_tokens(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pad_token_id: int | None,
) -> torch.Tensor:
    """Convert optional attention mask variants into a 2D active-token mask.

    :param torch.Tensor input_ids: Input token ids with shape ``(B, S)``.
    :param torch.Tensor | None attention_mask: Optional attention mask in ``(B,S)``, ``(B,S,S)``, or ``(B,H,S,S)`` form.
    :param int | None pad_token_id: Optional pad token id used when mask does not encode token activity directly.
    :raises ValueError: If ``attention_mask`` rank is unsupported.
    :return torch.Tensor: Boolean active-token mask with shape ``(B, S)``.
    """

    if attention_mask is None:
        if pad_token_id is None:
            return torch.ones_like(input_ids, dtype=torch.bool)
        return input_ids.ne(int(pad_token_id))

    mask = normalize_keep_mask(attention_mask)
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3:
        # Diagonal encodes per-token query activity for pairwise masks.
        active = torch.diagonal(mask, dim1=-2, dim2=-1)
        if pad_token_id is not None:
            active = active & input_ids.ne(int(pad_token_id))
        return active
    if mask.ndim == 4:
        squeezed = mask[:, 0] if mask.shape[1] == 1 else mask.any(dim=1)
        if squeezed.shape[-2] == 1:
            # Broadcast path: (B,1,1,S) -> (B,1,S), keep full per-token activity mask.
            active = squeezed[:, 0, :]
        else:
            active = torch.diagonal(squeezed, dim1=-2, dim2=-1)
        if pad_token_id is not None:
            active = active & input_ids.ne(int(pad_token_id))
        return active
    raise ValueError("attention_mask must have shape (B,S), (B,S,S), or (B,H,S,S).")


class _SyncedBufferEmbedding(nn.Module):
    """Compile-safe embedding used for GDES.

    Effective output is:
        E(ids) = base_weight[ids] + bias[ids]

    base_weight is a non-trainable Parameter synced from generator weights.
    bias is a trainable parameter (same shape).
    """

    def __init__(
        self,
        *,
        init_weight: torch.Tensor,
        padding_idx: int | None,
        add_bias: bool,
    ) -> None:
        """Initialize synced-buffer embedding used for GDES.

        :param torch.Tensor init_weight: Initial embedding matrix used to seed ``base_weight``.
        :param int | None padding_idx: Optional embedding padding index.
        :param bool add_bias: Whether to create a trainable per-token bias matrix.
        :raises ValueError: If ``init_weight`` is not rank-2.
        :return None: This constructor does not return a value.
        """
        super().__init__()
        if not isinstance(init_weight, torch.Tensor) or init_weight.ndim != 2:
            raise ValueError("init_weight must be a rank-2 tensor")

        # Keep base_weight as a Parameter (requires_grad=False) so torch.compile
        # observes version-counter bumps from sync copy_() updates.
        self.base_weight = nn.Parameter(
            init_weight.detach().clone(),
            requires_grad=False,
        )
        self.padding_idx = int(padding_idx) if padding_idx is not None else None

        self.bias: torch.nn.Parameter | None
        if add_bias:
            self.bias = nn.Parameter(torch.zeros_like(init_weight))
        else:
            self.bias = None

    @torch.no_grad()
    def sync_from(self, weight: torch.Tensor) -> None:
        """Copy generator weights into the local base buffer.

        :param torch.Tensor weight: Source embedding matrix from the generator.
        :raises ValueError: If shapes differ from the local ``base_weight`` shape.
        :return None: This method updates the local buffer in-place.
        """
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
        """Embed token ids using synced base weights plus optional bias.

        :param torch.Tensor input_ids: Token ids to embed.
        :return torch.Tensor: Embedded representations.
        """
        out = F.embedding(input_ids, self.base_weight, padding_idx=self.padding_idx)
        if self.bias is not None:
            out = out + F.embedding(input_ids, self.bias, padding_idx=self.padding_idx)
        return out


class MLMTransform(nn.Module):
    """Projection-activation-norm transform used by MLM head."""

    def __init__(self, config: Any) -> None:
        """Initialize the MLM transform stack.

        :param Any config: Backbone config containing hidden size, activation, and norm settings.
        :return None: This constructor does not return a value.
        """
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = get_act_fn(getattr(config, "hidden_act", "gelu"))
        eps = float(getattr(config, "norm_eps", getattr(config, "layer_norm_eps", 1e-6)))
        if bool(getattr(config, "use_rmsnorm_heads", True)):
            self.norm = RMSNorm(hidden_size, eps=eps)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project, activate, and normalize hidden states for MLM prediction.

        :param torch.Tensor hidden_states: Input hidden states.
        :return torch.Tensor: Transformed hidden states.
        """
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.norm(x)
        return x


class MaskedLMHead(nn.Module):
    """Masked LM head.

    In tied mode we keep only the output bias and project using external embedding weights.
    """

    def __init__(self, config: Any, *, tie_word_embeddings: bool = True) -> None:
        """Initialize the masked language modeling head.

        :param Any config: Generator config containing ``hidden_size`` and ``vocab_size``.
        :param bool tie_word_embeddings: Whether to use external tied embedding weights for output projection.
        :return None: This constructor does not return a value.
        """
        super().__init__()
        self.transform = MLMTransform(config)
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.hidden_size)

        self.tie_word_embeddings = bool(tie_word_embeddings)
        if self.tie_word_embeddings:
            self.decoder = None
            self.bias = nn.Parameter(torch.zeros(self.vocab_size))
        else:
            self.decoder = nn.Linear(self.hidden_size, self.vocab_size, bias=True)
            self.bias = self.decoder.bias

    def forward(
        self, hidden_states: torch.Tensor, *, word_embedding_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute MLM logits from hidden states.

        :param torch.Tensor hidden_states: Hidden states for masked-token positions.
        :param torch.Tensor | None word_embedding_weight: Tied embedding matrix when ``tie_word_embeddings`` is enabled.
        :raises RuntimeError: If required tied embedding weights are missing or incompatible.
        :return torch.Tensor: Token logits over the vocabulary.
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
            # Never cast the full embedding matrix. Outside autocast, cast the activation instead.
            if not torch.is_autocast_enabled() and x.dtype != w.dtype:
                x = x.to(dtype=w.dtype)
            if b.dtype != w.dtype:
                b = b.to(dtype=w.dtype)
            return F.linear(x, w, b)

        if self.decoder is None:
            raise RuntimeError("MaskedLMHead decoder is not initialized.")
        return self.decoder(x)


class RTDHead(nn.Module):
    """Discriminator head for replaced-token detection."""

    def __init__(self, config: Any) -> None:
        """Initialize discriminator token-classification head.

        :param Any config: Discriminator config with hidden size, dropout, activation, and norm settings.
        :return None: This constructor does not return a value.
        """
        super().__init__()
        hidden_size = int(config.hidden_size)
        drop_out = getattr(config, "hidden_dropout_prob", 0.0)
        self.dropout = nn.Dropout(float(drop_out))

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = get_act_fn(getattr(config, "hidden_act", "gelu"))
        eps = float(getattr(config, "norm_eps", getattr(config, "layer_norm_eps", 1e-6)))
        if bool(getattr(config, "use_rmsnorm_heads", True)):
            self.norm = RMSNorm(hidden_size, eps=eps)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=eps)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute replaced-token logits per token.

        :param torch.Tensor hidden_states: Discriminator hidden states.
        :return torch.Tensor: Scalar logit per token.
        """
        x = self.dropout(hidden_states)
        x = self.dense(x)
        x = self.act(x)
        x = self.norm(x)
        return self.classifier(x).squeeze(-1)


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
    ) -> None:
        """Initialize RTD pretrainer with generator/discriminator backbones.

        :param nn.Module discriminator_backbone: Discriminator backbone model.
        :param nn.Module generator_backbone: Generator backbone model.
        :param Any disc_config: Discriminator configuration object.
        :param Any gen_config: Generator configuration object.
        :param str embedding_sharing: Embedding sharing mode (``none``, ``es``, or ``gdes``).
        :param bool tie_generator_word_embeddings: Whether MLM output head is tied to generator word embeddings.
        :return None: This constructor does not return a value.
        """
        super().__init__()
        self.disc_config = disc_config
        self.gen_config = gen_config

        self.generator = generator_backbone
        self.discriminator = discriminator_backbone

        self.generator_lm_head = MaskedLMHead(gen_config, tie_word_embeddings=tie_generator_word_embeddings)
        self.discriminator_head = RTDHead(disc_config)

        self.embedding_sharing = str(embedding_sharing or "none")

        # Special ids excluded from generator sampling and (for non-masked positions) from discriminator loss.
        self._forbidden_sample_token_ids = self._collect_forbidden_sample_token_ids()

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

        :param int vocab_size: Generator vocabulary size.
        :param set[int] forbidden_ids: Token ids disallowed for generator sampling.
        :return torch.Tensor: Boolean mask of shape ``(vocab_size,)``.
        """
        if vocab_size <= 0:
            return torch.empty(0, dtype=torch.bool)
        if not forbidden_ids:
            return torch.zeros(vocab_size, dtype=torch.bool)

        mask = torch.zeros(vocab_size, dtype=torch.bool)
        for tid in forbidden_ids:
            if 0 <= int(tid) < vocab_size:
                mask[int(tid)] = True
        return mask

    def _collect_forbidden_sample_token_ids(self) -> set[int]:
        """Collect special token ids to exclude from generator sampling.

        :return set[int]: Valid special token ids present in generator/discriminator configs.
        """
        out: set[int] = set()
        vocab_size = int(getattr(self.gen_config, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            return out

        for cfg in (self.gen_config, self.disc_config):
            for attr in (
                "pad_token_id",
                "cls_token_id",
                "sep_token_id",
                "mask_token_id",
                "bos_token_id",
                "eos_token_id",
            ):
                sid = getattr(cfg, attr, None)
                if sid is None:
                    continue
                try:
                    sid_i = int(sid)
                except Exception:
                    continue
                if 0 <= sid_i < vocab_size:
                    out.add(sid_i)
        return out

    def _special_position_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map input ids to a boolean mask of special-token positions.

        :param torch.Tensor input_ids: Input token ids with shape ``(B, S)``.
        :return torch.Tensor: Boolean mask marking positions whose ids are forbidden special tokens.
        """
        mask = getattr(self, "_forbidden_sample_token_mask", None)
        if not isinstance(mask, torch.Tensor) or mask.numel() == 0:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        return mask[input_ids]

    # ------------------------------
    # Embedding sharing
    # ------------------------------

    def _patch_discriminator_embeddings_for_sharing(self) -> None:
        """Apply configured embedding sharing policy to discriminator embeddings.

        :raises ValueError: If ``embedding_sharing`` mode is unsupported or embedding shapes mismatch.
        :raises RuntimeError: If expected embedding modules/weights are unavailable.
        :return None: This method mutates discriminator embedding modules in-place.
        """
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
            """Extract optional padding index from embedding-like module.

            :param nn.Module mod: Embedding-like module.
            :return int | None: Padding index when available.
            """
            val = getattr(mod, "padding_idx", None)
            return int(val) if val is not None else None

        def _weight(mod: nn.Module) -> torch.Tensor:
            """Read required ``weight`` tensor from an embedding-like module.

            :param nn.Module mod: Embedding-like module.
            :raises RuntimeError: If module has no tensor ``weight`` attribute.
            :return torch.Tensor: Embedding weight tensor.
            """
            w = getattr(mod, "weight", None)
            if not isinstance(w, torch.Tensor):
                raise RuntimeError("Embedding module must expose .weight tensor")
            return w

        def _validate(attr: str, gen_mod: nn.Module, disc_mod: nn.Module) -> torch.Tensor:
            """Validate shareable shapes between generator and discriminator embedding modules.

            :param str attr: Embedding attribute name.
            :param nn.Module gen_mod: Generator embedding module.
            :param nn.Module disc_mod: Discriminator embedding module.
            :raises ValueError: If generator/discriminator weight shapes differ.
            :return torch.Tensor: Generator embedding weight tensor.
            """
            gw = _weight(gen_mod)
            dw = _weight(disc_mod)
            if gw.shape != dw.shape:
                raise ValueError(
                    f"Cannot share embeddings for '{attr}': generator shape {tuple(gw.shape)} != discriminator shape {tuple(dw.shape)}."
                )
            return gw

        if mode == "es":
            # Share Parameters directly (compile-safe).
            for attr in attrs:
                gen_mod = getattr(gen_embeddings, attr, None)
                disc_mod = getattr(disc_embeddings, attr, None)
                if gen_mod is None or disc_mod is None:
                    continue
                gw = _validate(attr, gen_mod, disc_mod)
                disc_mod.weight = gw  # type: ignore[assignment]
            return

        # GDES
        for attr in attrs:
            gen_mod = getattr(gen_embeddings, attr, None)
            disc_mod = getattr(disc_embeddings, attr, None)
            if gen_mod is None or disc_mod is None:
                continue
            gw = _validate(attr, gen_mod, disc_mod)

            synced = _SyncedBufferEmbedding(
                init_weight=gw,
                padding_idx=_padding_idx(disc_mod),
                add_bias=True,
            )
            setattr(disc_embeddings, attr, synced)
            self._gdes_synced_embeddings.append((attr, synced, gen_mod))

    @torch.no_grad()
    def sync_discriminator_embeddings_from_generator(self) -> None:
        """Sync GDES base weights from generator embedding weights.

        Must be called after each optimizer step and after checkpoint load.
        No-op unless embedding_sharing == 'gdes'.
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

        :raises RuntimeError: If generator embedding modules or weight tensor are unavailable.
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

        :param torch.Tensor logits: Unnormalized logits with vocabulary dimension on the last axis.
        :param float temperature: Sampling temperature (> 0).
        :param torch.Tensor | None forbidden_vocab_mask: Optional boolean mask over vocabulary ids to suppress.
        :raises ValueError: If ``temperature`` is non-positive.
        :return torch.Tensor: Sampled token ids for each row of ``logits``.
        """

        temp = float(temperature)
        if temp <= 0:
            raise ValueError("sampling_temperature must be > 0")

        x = logits.float() / temp
        if forbidden_vocab_mask is not None and forbidden_vocab_mask.numel() != 0:
            if forbidden_vocab_mask.device != x.device:
                forbidden_vocab_mask = forbidden_vocab_mask.to(device=x.device)
            x = x.masked_fill(forbidden_vocab_mask, -1e9)

        u = torch.rand_like(x).clamp_(min=1e-6, max=1.0 - 1e-6)
        g = -torch.log(-torch.log(u))
        return torch.argmax(x + g, dim=-1)

    # ------------------------------
    # Forward
    # ------------------------------

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
        decoupled_loss_scaling: bool = False,
    ) -> RTDOutput:
        """Run one RTD forward pass with generator corruption and discriminator scoring.

        :param torch.Tensor input_ids: Input token ids with shape ``(B, S)``.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param torch.Tensor labels: MLM labels with ``-100`` for non-masked positions.
        :param torch.Tensor | None token_type_ids: Optional segment ids.
        :param float sampling_temperature: Generator sampling temperature.
        :param float gen_loss_weight: Scalar multiplier for generator term.
        :param float disc_loss_weight: Scalar multiplier for discriminator term.
        :param bool decoupled_loss_scaling: Whether to apply detached discriminator/gen scaling to generator term.
        :raises ValueError: If labels are missing.
        :return RTDOutput: Total loss plus detached logging metrics and raw differentiable loss terms.
        """

        if labels is None:
            raise ValueError("labels must be provided (MLM labels with -100 for unmasked positions)")

        masked_positions = labels.ne(-100)  # (B,S)
        masked_flat = masked_positions.view(-1)
        masked_idx = torch.nonzero(masked_flat, as_tuple=False).squeeze(-1)  # (N,)
        gen_token_count = masked_flat.sum().to(dtype=torch.float32)

        # Generator
        gen_out = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        hidden = gen_out.last_hidden_state

        corrupted_input_ids = input_ids
        disc_labels = torch.zeros_like(input_ids, dtype=torch.float32)

        if masked_idx.numel() == 0:
            gen_loss = hidden.sum() * 0.0
        else:
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            masked_hidden = hidden_flat.index_select(0, masked_idx)

            labels_flat = labels.view(-1)
            masked_labels = labels_flat.index_select(0, masked_idx)

            word_w = self._get_generator_word_embedding_weight()
            gen_logits = self.generator_lm_head(masked_hidden, word_embedding_weight=word_w)
            gen_loss = F.cross_entropy(gen_logits.float(), masked_labels)

            with torch.no_grad():
                forbidden_mask = getattr(self, "_forbidden_sample_token_mask", None)
                sampled = self._gumbel_sample(
                    gen_logits,
                    temperature=sampling_temperature,
                    forbidden_vocab_mask=forbidden_mask,
                ).to(dtype=input_ids.dtype)

                # Corrupt ids — functional scatter returns a new tensor (no clone needed).
                corrupted_flat = input_ids.view(-1).scatter(0, masked_idx, sampled)
                corrupted_input_ids = corrupted_flat.view_as(input_ids)

                replaced = sampled.ne(masked_labels.to(dtype=sampled.dtype)).to(dtype=torch.float32)
                disc_labels_flat = torch.zeros(
                    input_ids.numel(), dtype=torch.float32, device=input_ids.device
                )
                disc_labels_flat.scatter_(0, masked_idx, replaced)
                disc_labels = disc_labels_flat.view_as(input_ids)

        # Discriminator
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

        special = self._special_position_mask(input_ids)
        special = special & (~masked_positions)
        disc_active = active & (~special)

        disc_active_f = disc_active.to(dtype=torch.float32)
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

        gen_term = compute_generator_loss_term(
            gen_loss=gen_loss,
            disc_loss=disc_loss,
            decoupled_loss_scaling=decoupled_loss_scaling,
        )
        total = float(gen_loss_weight) * gen_term + float(disc_loss_weight) * disc_loss

        return RTDOutput(
            loss=total,
            gen_loss=gen_loss.detach(),
            disc_loss=disc_loss.detach(),
            disc_accuracy=disc_acc.detach(),
            gen_token_count=gen_token_count.detach(),
            disc_token_count=disc_token_count.detach(),
            disc_positive_count=disc_positive_count.detach(),
            gen_loss_raw=gen_loss,
            disc_loss_raw=disc_loss,
        )
