"""RTD (ELECTRA-style) pretraining heads and wrapper module."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from deberta.modeling.norm import RMSNorm


def _get_act_fn(name_or_fn: str | Any) -> Any:
    """Resolve activation callable from name or passthrough callable.

    :param str | Any name_or_fn: Activation name or callable.
    :return Any: Activation callable.
    """
    try:
        from transformers.activations import ACT2FN

        if isinstance(name_or_fn, str):
            return ACT2FN[name_or_fn]
        return name_or_fn
    except Exception:
        return name_or_fn


class _TiedEmbedding(nn.Module):
    """Embedding that reads weights from a different module via weakref.

    This avoids sharing *module instances* (which can break FSDP2 wrapping), while still allowing:

      - vanilla sharing (grad flows to base weight) via detach_base=False
      - gdes sharing (grad does NOT flow to base weight) via detach_base=True + trainable bias

    NOTE: We intentionally do not register the base weight as a parameter or buffer.
    """

    def __init__(
        self,
        *,
        base_weight: torch.nn.Parameter,
        padding_idx: int | None,
        detach_base: bool,
        add_bias: bool,
    ) -> None:
        """Create tied-embedding adapter.

        :param torch.nn.Parameter base_weight: Source embedding weight.
        :param int | None padding_idx: Padding index for embedding lookup.
        :param bool detach_base: Whether to detach base weights.
        :param bool add_bias: Whether to add trainable bias matrix.
        """
        super().__init__()
        self._base_weight_ref = weakref.ref(base_weight)
        self.padding_idx = int(padding_idx) if padding_idx is not None else None
        self.detach_base = bool(detach_base)

        self.bias: torch.nn.Parameter | None
        if add_bias:
            # Bias is a full embedding matrix (same shape as base_weight). This matches DeBERTaV3 gdes behavior.
            self.bias = nn.Parameter(torch.zeros_like(base_weight, dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings from tied source weight.

        :param torch.Tensor input_ids: Input token ids.
        :return torch.Tensor: Embedded representations.
        """
        w = self._base_weight_ref()
        if w is None:
            raise RuntimeError("Base embedding weight reference is dead.")

        base_w = w.detach() if self.detach_base else w
        out = F.embedding(input_ids, base_w, padding_idx=self.padding_idx)

        if self.bias is not None:
            # Efficient: lookup bias rows only (avoid constructing full base_w + bias).
            out = out + F.embedding(input_ids, self.bias, padding_idx=self.padding_idx)

        return out


class MLMTransform(nn.Module):
    """Projection-activation-norm transform used by MLM head."""

    def __init__(self, config: Any) -> None:
        """Create MLM transform layers.

        :param Any config: Backbone config.
        """
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = _get_act_fn(getattr(config, "hidden_act", "gelu"))
        self.norm = RMSNorm(
            hidden_size, eps=float(getattr(config, "norm_eps", getattr(config, "layer_norm_eps", 1e-6)))
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Transform hidden states before vocab projection.

        :param torch.Tensor hidden_states: Input hidden states.
        :return torch.Tensor: Transformed hidden states.
        """
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.norm(x)
        return x


class MaskedLMHead(nn.Module):
    """Generic masked LM head.

    We optionally tie the output projection to the input embeddings if dimensions match.
    """

    def __init__(self, config: Any, *, tie_word_embeddings: bool = True) -> None:
        """Create masked language modeling head.

        :param Any config: Backbone config.
        :param bool tie_word_embeddings: Whether to tie decoder to embeddings.
        """
        super().__init__()
        self.transform = MLMTransform(config)
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.hidden_size)

        self.tie_word_embeddings = bool(tie_word_embeddings)
        if self.tie_word_embeddings:
            # Tied mode projects with external embedding weights; keep only output bias.
            self.decoder = None
            self.bias = nn.Parameter(torch.zeros(self.vocab_size))
        else:
            self.decoder = nn.Linear(self.hidden_size, self.vocab_size, bias=True)
            self.bias = self.decoder.bias

    def forward(
        self, hidden_states: torch.Tensor, *, word_embedding_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Project hidden states to vocabulary logits.

        :param torch.Tensor hidden_states: Input hidden states.
        :param torch.Tensor | None word_embedding_weight: Optional tied embedding weight.
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
            if w.dtype != x.dtype:
                w = w.to(dtype=x.dtype)
            b = self.bias
            if b.dtype != x.dtype:
                b = b.to(dtype=x.dtype)
            return F.linear(x, w, b)

        if self.decoder is None:
            raise RuntimeError("MaskedLMHead decoder is not initialized.")
        return self.decoder(x)


class RTDHead(nn.Module):
    """Discriminator head for replaced-token detection."""

    def __init__(self, config: Any) -> None:
        """Create RTD discriminator head.

        :param Any config: Backbone config.
        """
        super().__init__()
        hidden_size = int(config.hidden_size)
        drop_out = getattr(config, "hidden_dropout_prob", 0.0)
        self.dropout = nn.Dropout(float(drop_out))

        # A small transform helps stability.
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = _get_act_fn(getattr(config, "hidden_act", "gelu"))
        self.norm = RMSNorm(
            hidden_size, eps=float(getattr(config, "norm_eps", getattr(config, "layer_norm_eps", 1e-6)))
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute RTD logits per token position.

        :param torch.Tensor hidden_states: Input hidden states.
        :return torch.Tensor: RTD logits.
        """
        x = self.dropout(hidden_states)
        x = self.dense(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.dropout(x)
        logits = self.classifier(x).squeeze(-1)
        return logits


@dataclass
class RTDOutput:
    """Structured outputs from one RTD forward pass."""

    loss: torch.Tensor
    gen_loss: torch.Tensor
    disc_loss: torch.Tensor
    disc_accuracy: torch.Tensor
    gen_token_count: torch.Tensor
    disc_token_count: torch.Tensor
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
        """Create combined generator+discriminator RTD module.

        :param nn.Module discriminator_backbone: Discriminator encoder backbone.
        :param nn.Module generator_backbone: Generator encoder backbone.
        :param Any disc_config: Discriminator config.
        :param Any gen_config: Generator config.
        :param str embedding_sharing: Embedding sharing mode.
        :param bool tie_generator_word_embeddings: Whether to tie generator LM head to embeddings.
        """
        super().__init__()
        self.disc_config = disc_config
        self.gen_config = gen_config

        self.generator = generator_backbone
        self.discriminator = discriminator_backbone

        self.generator_lm_head = MaskedLMHead(gen_config, tie_word_embeddings=tie_generator_word_embeddings)
        self.discriminator_head = RTDHead(disc_config)

        self.embedding_sharing = embedding_sharing
        self._forbidden_sample_token_ids = self._collect_forbidden_sample_token_ids()
        self._allowed_sample_token_ids_cpu = self._build_allowed_sample_token_ids()
        self._allowed_sample_token_ids_by_device: dict[str, torch.Tensor] = {}
        self._maybe_patch_discriminator_embeddings()

    def _collect_forbidden_sample_token_ids(self) -> set[int]:
        """Collect special token ids to exclude from generator replacement sampling.

        :return set[int]: Special token ids bounded to generator vocab range.
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

    def _build_allowed_sample_token_ids(self) -> torch.Tensor | None:
        """Build allowed generator replacement token ids.

        :return torch.Tensor | None: CPU tensor of sampleable token ids, or None.
        """
        if not self._forbidden_sample_token_ids:
            return None

        vocab_size = int(getattr(self.gen_config, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            return None

        allowed = torch.ones(vocab_size, dtype=torch.bool)
        for sid in self._forbidden_sample_token_ids:
            if 0 <= int(sid) < vocab_size:
                allowed[int(sid)] = False

        if not bool(allowed.any().item()):
            return None
        return torch.arange(vocab_size, dtype=torch.long)[allowed]

    def _get_allowed_sample_token_ids(self, device: torch.device) -> torch.Tensor | None:
        """Get cached allowed token ids on a device.

        :param torch.device device: Device for returned tensor.
        :return torch.Tensor | None: Device-local allowed ids, or None.
        """
        if self._allowed_sample_token_ids_cpu is None:
            return None

        key = f"{device.type}:{device.index if device.index is not None else -1}"
        ids = self._allowed_sample_token_ids_by_device.get(key)
        if ids is None:
            ids = self._allowed_sample_token_ids_cpu.to(device=device)
            self._allowed_sample_token_ids_by_device[key] = ids
        return ids

    def _sample_generator_tokens(self, logits: torch.Tensor, *, sampling_temperature: float) -> torch.Tensor:
        """Sample generator replacements from logits with optional special-id filtering.

        :param torch.Tensor logits: Generator logits of shape (N, vocab_size).
        :param float sampling_temperature: Positive sampling temperature.
        :return torch.Tensor: Sampled token ids of shape (N,).
        """
        temp = float(sampling_temperature)
        if temp <= 0:
            raise ValueError("sampling_temperature must be > 0")

        logits_f = logits.float()
        allowed = self._get_allowed_sample_token_ids(logits.device)
        if allowed is not None:
            filtered_logits = logits_f.index_select(dim=-1, index=allowed)
            probs = torch.softmax(filtered_logits / temp, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            return allowed[sampled_idx]

        probs = torch.softmax(logits_f / temp, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _get_generator_word_embedding_weight(self) -> torch.Tensor:
        """Return generator token embedding matrix for LM-head tying.

        :return torch.Tensor: Generator word embedding weights.
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

    def _special_position_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build a boolean mask over configured special token ids.

        :param torch.Tensor input_ids: Token ids shaped (B, S).
        :return torch.Tensor: True at special-token positions.
        """
        if not self._forbidden_sample_token_ids:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        out = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in self._forbidden_sample_token_ids:
            out = out | input_ids.eq(int(sid))
        return out

    def _maybe_patch_discriminator_embeddings(self) -> None:
        """Patch discriminator embeddings to follow configured sharing mode."""
        mode = (self.embedding_sharing or "none").lower()
        if mode not in {"none", "es", "gdes"}:
            raise ValueError("embedding_sharing must be one of: none|es|gdes")
        if mode == "none":
            return

        # We avoid sharing module instances (FSDP2 hazard). We only reference the generator weight via weakref.
        try:
            gen_embeddings = self.generator.embeddings
            disc_embeddings = self.discriminator.embeddings
        except Exception as e:
            raise RuntimeError(
                "Expected backbones with an `.embeddings` module containing `word_embeddings`. "
                "Use backbone_type='rope' or HF DebertaV2Model-like backbones."
            ) from e

        detach_base = mode == "gdes"
        add_bias = mode == "gdes"

        # Helper: tie attribute if present on both
        def tie_attr(attr: str) -> None:
            """Tie one embedding attribute between generator and discriminator.

            :param str attr: Embedding attribute name.
            """
            gen_mod = getattr(gen_embeddings, attr, None)
            disc_mod = getattr(disc_embeddings, attr, None)
            if gen_mod is None or disc_mod is None:
                return
            if not hasattr(gen_mod, "weight") or not hasattr(disc_mod, "weight"):
                return
            if gen_mod.weight.shape != disc_mod.weight.shape:
                raise ValueError(
                    f"Cannot share embeddings for '{attr}': generator shape {tuple(gen_mod.weight.shape)} != discriminator shape {tuple(disc_mod.weight.shape)}. "
                    "Set embedding_sharing=none or make hidden_size match."
                )
            setattr(
                disc_embeddings,
                attr,
                _TiedEmbedding(
                    base_weight=gen_mod.weight,
                    padding_idx=getattr(disc_mod, "padding_idx", None),
                    detach_base=detach_base,
                    add_bias=add_bias,
                ),
            )

        tie_attr("word_embeddings")
        tie_attr("position_embeddings")
        tie_attr("token_type_embeddings")

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
        """Run one ELECTRA-style forward pass.

        :param torch.Tensor input_ids: Masked input ids.
        :param torch.Tensor | None attention_mask: Binary mask (1 real token, 0 pad).
        :param torch.Tensor labels: Original token ids at masked positions, else -100.
        :param torch.Tensor | None token_type_ids: Optional segment ids.
        :param float sampling_temperature: Sampling temperature for generator tokens.
        :param float gen_loss_weight: Generator loss weight.
        :param float disc_loss_weight: Discriminator loss weight.
        :param bool decoupled_loss_scaling: Whether to apply decoupled loss scaling.
        :return RTDOutput: Total loss plus generator/discriminator diagnostics.
        """

        if labels is None:
            raise ValueError("labels must be provided (MLM labels with -100 for unmasked positions)")

        # -------------------
        # Generator
        # -------------------
        gen_out = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        hidden = gen_out.last_hidden_state

        masked = labels.ne(-100)
        gen_token_count = masked.to(torch.float32).sum()
        if masked.any():
            masked_hidden = hidden[masked]
            # Compute logits only on masked positions.
            word_w = self._get_generator_word_embedding_weight()
            gen_logits = self.generator_lm_head(masked_hidden, word_embedding_weight=word_w)
            gen_labels = labels[masked]
            gen_loss = F.cross_entropy(gen_logits.float(), gen_labels)

            # Sample replacements from detached logits.
            with torch.no_grad():
                sampled = self._sample_generator_tokens(gen_logits, sampling_temperature=sampling_temperature)

            corrupted_input_ids = input_ids.clone()
            corrupted_input_ids[masked] = sampled

            # discriminator labels: 1 if replaced, 0 otherwise
            disc_labels = torch.zeros_like(input_ids, dtype=torch.float32)
            disc_labels[masked] = (sampled != gen_labels).float()
        else:
            # Degenerate case: no masked tokens (very small sequences or mlm_probability=0)
            gen_loss = hidden.sum() * 0.0
            corrupted_input_ids = input_ids
            disc_labels = torch.zeros_like(input_ids, dtype=torch.float32)

        # -------------------
        # Discriminator
        # -------------------
        disc_out = self.discriminator(
            input_ids=corrupted_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        disc_hidden = disc_out.last_hidden_state
        disc_logits = self.discriminator_head(disc_hidden)

        if attention_mask is None:
            pad_id = getattr(self.disc_config, "pad_token_id", None)
            if pad_id is None:
                active = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                active = input_ids.ne(int(pad_id))
        else:
            mask = attention_mask.to(torch.bool)
            if mask.ndim == 2:
                active = mask
            elif mask.ndim == 3:
                active = mask.any(dim=-1)
            elif mask.ndim == 4:
                active = mask.any(dim=-1)
                if active.ndim == 3:
                    active = active.any(dim=1)
            else:
                raise ValueError("attention_mask must have shape (B,S), (B,S,S), or (B,H,S,S).")

        # Discriminator should not learn on special tokens.
        active = active & (~self._special_position_mask(corrupted_input_ids))
        disc_token_count = active.to(torch.float32).sum()

        if bool(active.any().item()):
            disc_loss = F.binary_cross_entropy_with_logits(disc_logits[active].float(), disc_labels[active])
            # Accuracy for monitoring: threshold at 0
            with torch.no_grad():
                pred = (disc_logits[active] > 0).to(torch.float32)
                disc_acc = (pred == disc_labels[active]).to(torch.float32).mean()
        else:
            disc_loss = disc_logits.sum() * 0.0
            disc_acc = disc_logits.sum() * 0.0

        # -------------------
        # Total
        # -------------------
        gen_loss_scaled = gen_loss
        if decoupled_loss_scaling:
            # Match original implementations: scale gen_loss by disc/gen magnitudes.
            eps = 1e-6
            alpha = (disc_loss.detach() / (gen_loss.detach() + eps)).clamp(min=0.0, max=1e4)
            gen_loss_scaled = alpha * gen_loss

        total = float(gen_loss_weight) * gen_loss_scaled + float(disc_loss_weight) * disc_loss
        return RTDOutput(
            loss=total,
            gen_loss=gen_loss.detach(),
            disc_loss=disc_loss.detach(),
            disc_accuracy=disc_acc,
            gen_token_count=gen_token_count.detach(),
            disc_token_count=disc_token_count.detach(),
            gen_loss_raw=gen_loss,
            disc_loss_raw=disc_loss,
        )
