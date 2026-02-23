"""RTD (ELECTRA-style) pretraining heads and wrapper module.

This file intentionally keeps the RTD objective logic *numerically stable* and
*torch.compile-friendly*.

Key design constraints:
  - Generator/discriminator backbones are the hot path; everything else must be
    lightweight and avoid CPU<->GPU transfers.
  - Replacement sampling must avoid device copies and should be implemented in
    pure tensor ops (GPU RNG) when possible.
  - Special-token filtering should be done via a boolean vocab mask buffer rather
    than index_select/mapping (which causes graph breaks and cudagraph partitions).

The training loop may choose to compile only the generator/discriminator
backbones for stability.
"""

from __future__ import annotations

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


def compute_generator_loss_term(
    *, gen_loss: torch.Tensor, disc_loss: torch.Tensor, decoupled_loss_scaling: bool
) -> torch.Tensor:
    """Return generator loss term with optional decoupled RTD scaling.

    DeBERTa-v3 uses a *decoupled* generator loss scaling that roughly matches the
    discriminator loss magnitude.

    :param torch.Tensor gen_loss: Raw generator loss.
    :param torch.Tensor disc_loss: Raw discriminator loss.
    :param bool decoupled_loss_scaling: Whether to rescale generator loss.
    :return torch.Tensor: Generator loss term used in total objective.
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

    Supported variants:
      - None: infer from pad_token_id or assume fully active.
      - (B, S): standard attention mask.
      - (B, S, S): packed pairwise mask (document blocks).
      - (B, H, S, S): attention mask with heads.

    :param torch.Tensor input_ids: Input token ids shaped ``(B, S)``.
    :param torch.Tensor | None attention_mask: Optional 2D/3D/4D attention mask.
    :param int | None pad_token_id: Optional pad token id when ``attention_mask`` is omitted.
    :return torch.Tensor: Active-token mask shaped ``(B, S)``.
    """

    if attention_mask is None:
        if pad_token_id is None:
            return torch.ones_like(input_ids, dtype=torch.bool)
        return input_ids.ne(int(pad_token_id))

    mask = attention_mask.to(torch.bool)
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3:
        # Packed pairwise masks encode document constraints, not padding identity.
        # Recover token activity from input ids when padding metadata is available.
        if pad_token_id is not None:
            return input_ids.ne(int(pad_token_id))
        return mask.any(dim=-1)
    if mask.ndim == 4:
        if pad_token_id is not None:
            return input_ids.ne(int(pad_token_id))
        active = mask.any(dim=-1)
        if active.ndim == 3:
            active = active.any(dim=1)
        return active
    raise ValueError("attention_mask must have shape (B,S), (B,S,S), or (B,H,S,S).")


class _TiedEmbedding(nn.Module):
    """Embedding that reads weights from a different module.

    This avoids sharing *module instances* (which can break FSDP2 wrapping), while still allowing:

      - vanilla sharing (grad flows to base weight) via detach_base=False
      - gdes sharing (grad does NOT flow to base weight) via detach_base=True + trainable bias

    NOTE: We intentionally keep an unregistered *strong* reference to the base module
    and do not register the base weight as a parameter or buffer.
    """

    def __init__(
        self,
        *,
        base_embedding_module: nn.Module,
        padding_idx: int | None,
        detach_base: bool,
        add_bias: bool,
    ) -> None:
        super().__init__()
        base_weight = getattr(base_embedding_module, "weight", None)
        if not isinstance(base_weight, torch.Tensor):
            raise RuntimeError("base_embedding_module must expose a `.weight` tensor.")
        # Keep an unregistered strong ref to survive FSDP parameter replacement.
        object.__setattr__(self, "_base_embedding_module", base_embedding_module)
        self.padding_idx = int(padding_idx) if padding_idx is not None else None
        self.detach_base = bool(detach_base)

        self.bias: torch.nn.Parameter | None
        if add_bias:
            # Bias is a full embedding matrix (same shape as base_weight). This matches DeBERTaV3 gdes behavior.
            self.bias = nn.Parameter(torch.zeros_like(base_weight))
        else:
            self.bias = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base_mod = getattr(self, "_base_embedding_module", None)
        w = getattr(base_mod, "weight", None)
        if not isinstance(w, torch.Tensor):
            raise RuntimeError("Base embedding module no longer exposes a `.weight` tensor.")

        base_w = w.detach() if self.detach_base else w
        out = F.embedding(input_ids, base_w, padding_idx=self.padding_idx)

        if self.bias is not None:
            # Efficient: lookup bias rows only (avoid constructing full base_w + bias).
            out = out + F.embedding(input_ids, self.bias, padding_idx=self.padding_idx)

        return out


class MLMTransform(nn.Module):
    """Projection-activation-norm transform used by MLM head."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = _get_act_fn(getattr(config, "hidden_act", "gelu"))
        eps = float(getattr(config, "norm_eps", getattr(config, "layer_norm_eps", 1e-6)))
        if bool(getattr(config, "use_rmsnorm_heads", True)):
            self.norm = RMSNorm(hidden_size, eps=eps)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.norm(x)
        return x


class MaskedLMHead(nn.Module):
    """Masked LM head.

    In tied mode we keep only the output bias and project using external embedding weights.
    """

    def __init__(self, config: Any, *, tie_word_embeddings: bool = True) -> None:
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
            # Never cast the full embedding matrix in the hot path. Outside autocast,
            # align dtypes by casting the much smaller activation tensor instead.
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
        super().__init__()
        hidden_size = int(config.hidden_size)
        drop_out = getattr(config, "hidden_dropout_prob", 0.0)
        self.dropout = nn.Dropout(float(drop_out))

        # A small transform helps stability.
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = _get_act_fn(getattr(config, "hidden_act", "gelu"))
        eps = float(getattr(config, "norm_eps", getattr(config, "layer_norm_eps", 1e-6)))
        if bool(getattr(config, "use_rmsnorm_heads", True)):
            self.norm = RMSNorm(hidden_size, eps=eps)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=eps)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dropout(hidden_states)
        x = self.dense(x)
        x = self.act(x)
        x = self.norm(x)
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
        super().__init__()
        self.disc_config = disc_config
        self.gen_config = gen_config

        self.generator = generator_backbone
        self.discriminator = discriminator_backbone

        self.generator_lm_head = MaskedLMHead(gen_config, tie_word_embeddings=tie_generator_word_embeddings)
        self.discriminator_head = RTDHead(disc_config)

        self.embedding_sharing = embedding_sharing

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

        self._tied_embedding_attrs: tuple[str, ...] = ()
        self._tied_embedding_bindings_validated = False
        self._maybe_patch_discriminator_embeddings()

    @staticmethod
    def _build_forbidden_token_mask(*, vocab_size: int, forbidden_ids: set[int]) -> torch.Tensor:
        """Create a 1D boolean vocab mask for forbidden ids.

        Returned tensor is always length vocab_size (or empty if vocab_size<=0).
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
        """Collect special token ids to exclude from generator replacement sampling."""

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

    def _get_generator_word_embedding_weight(self) -> torch.Tensor:
        """Return generator token embedding matrix for LM-head tying."""

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
        """Return True for positions whose token id is in the forbidden vocab mask."""

        mask = getattr(self, "_forbidden_sample_token_mask", None)
        if not isinstance(mask, torch.Tensor) or mask.numel() == 0:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        # mask[input_ids] -> (B,S) bool
        return mask[input_ids]

    @staticmethod
    def _gumbel_sample(
        logits: torch.Tensor,
        *,
        temperature: float,
        forbidden_vocab_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Sample categorical ids from logits using Gumbel-max.

        This is mathematically equivalent to sampling from softmax(logits / T).

        :param torch.Tensor logits: Logits shaped (N, V).
        :param float temperature: Positive temperature.
        :param torch.Tensor | None forbidden_vocab_mask: Bool mask shaped (V,) marking forbidden ids.
        :return torch.Tensor: Sampled ids shaped (N,).
        """

        temp = float(temperature)
        if temp <= 0:
            raise ValueError("sampling_temperature must be > 0")

        x = logits.float() / temp
        if forbidden_vocab_mask is not None and forbidden_vocab_mask.numel() != 0:
            # Ensure mask is on correct device.
            if forbidden_vocab_mask.device != x.device:
                forbidden_vocab_mask = forbidden_vocab_mask.to(device=x.device)
            # Using finfo.min is risky in bf16/float16; prefer a large negative constant.
            x = x.masked_fill(forbidden_vocab_mask, -1e9)

        # Clamp to avoid log(0) or log(1).
        u = torch.rand_like(x).clamp_(min=1e-6, max=1.0 - 1e-6)
        g = -torch.log(-torch.log(u))
        return torch.argmax(x + g, dim=-1)

    def _maybe_patch_discriminator_embeddings(self) -> None:
        """Patch discriminator embeddings to follow configured sharing mode."""

        mode = (self.embedding_sharing or "none").lower()
        if mode not in {"none", "es", "gdes"}:
            raise ValueError("embedding_sharing must be one of: none|es|gdes")
        if mode == "none":
            self._tied_embedding_attrs = ()
            return

        # We avoid sharing module instances (FSDP2 hazard). Tied adapters keep an
        # unregistered strong ref to the generator embedding module.
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
        tied_attrs: list[str] = []

        def tie_attr(attr: str) -> None:
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
                    base_embedding_module=gen_mod,
                    padding_idx=getattr(disc_mod, "padding_idx", None),
                    detach_base=detach_base,
                    add_bias=add_bias,
                ),
            )
            tied_attrs.append(attr)

        tie_attr("word_embeddings")
        tie_attr("position_embeddings")
        tie_attr("token_type_embeddings")
        self._tied_embedding_attrs = tuple(tied_attrs)

    def _validate_tied_embedding_bindings(self) -> None:
        """Ensure tied embedding adapters still point to live generator modules."""

        if not self._tied_embedding_attrs:
            return

        try:
            gen_embeddings = self.generator.embeddings
            disc_embeddings = self.discriminator.embeddings
        except Exception as e:
            raise RuntimeError(
                "Expected generator/discriminator backbones with `.embeddings` modules while validating "
                "tied embedding adapters."
            ) from e

        stale: list[str] = []
        for attr in self._tied_embedding_attrs:
            gen_mod = getattr(gen_embeddings, attr, None)
            disc_mod = getattr(disc_embeddings, attr, None)
            if not isinstance(disc_mod, _TiedEmbedding):
                stale.append(attr)
                continue
            base_mod = getattr(disc_mod, "_base_embedding_module", None)
            if gen_mod is None or base_mod is None:
                stale.append(attr)
                continue
            gen_weight = getattr(gen_mod, "weight", None)
            base_weight = getattr(base_mod, "weight", None)
            if not isinstance(gen_weight, torch.Tensor) or not isinstance(base_weight, torch.Tensor):
                stale.append(attr)
                continue
            if gen_weight.data_ptr() != base_weight.data_ptr():
                stale.append(attr)

        if stale:
            names = ", ".join(sorted(stale))
            raise RuntimeError(
                "Detected stale tied embeddings after module replacement for: "
                f"{names}. Recreate DebertaV3RTDPretrainer to rebind embedding sharing."
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
        decoupled_loss_scaling: bool = False,
    ) -> RTDOutput:
        """Run one ELECTRA-style forward pass."""

        if labels is None:
            raise ValueError("labels must be provided (MLM labels with -100 for unmasked positions)")
        if not self._tied_embedding_bindings_validated:
            self._validate_tied_embedding_bindings()
            self._tied_embedding_bindings_validated = True

        # -------------------
        # Mask bookkeeping (static-shape friendly)
        # -------------------
        masked_positions = labels.ne(-100)  # (B,S) bool
        masked_flat = masked_positions.view(-1)
        masked_idx = torch.nonzero(masked_flat, as_tuple=False).squeeze(-1)  # (N,)

        gen_token_count = masked_flat.sum().to(dtype=torch.float32)

        # -------------------
        # Generator forward
        # -------------------
        gen_out = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        hidden = gen_out.last_hidden_state  # (B,S,H)

        # Default corruption is identity (no replacements) for edge cases.
        corrupted_input_ids = input_ids
        disc_labels = torch.zeros_like(input_ids, dtype=torch.float32)

        if masked_idx.numel() == 0:
            # No MLM targets. Generator loss is exactly zero; discriminator still trains on "all original".
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
                    gen_logits.detach(),
                    temperature=sampling_temperature,
                    forbidden_vocab_mask=forbidden_mask,
                ).to(dtype=input_ids.dtype)

                # Corrupt input ids (flat scatter avoids boolean advanced-index assignment).
                input_flat = input_ids.view(-1)
                corrupted_flat = input_flat.clone()
                corrupted_flat.scatter_(0, masked_idx, sampled)
                corrupted_input_ids = corrupted_flat.view_as(input_ids)

                # Discriminator labels: 1 at positions where sampled token != original label.
                replaced = sampled.ne(masked_labels.to(dtype=sampled.dtype)).to(dtype=torch.float32)
                disc_labels_flat = torch.zeros_like(input_flat, dtype=torch.float32)
                disc_labels_flat.scatter_(0, masked_idx, replaced)
                disc_labels = disc_labels_flat.view_as(input_ids)

        # -------------------
        # Discriminator forward
        # -------------------
        disc_out = self.discriminator(
            input_ids=corrupted_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        disc_hidden = disc_out.last_hidden_state
        disc_logits = self.discriminator_head(disc_hidden)  # (B,S)

        # Active tokens: based on attention/padding, excluding special tokens (except masked positions).
        pad_token_id = getattr(self.disc_config, "pad_token_id", None)
        active = attention_mask_to_active_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=int(pad_token_id) if pad_token_id is not None else None,
        )

        special = self._special_position_mask(input_ids)
        # Masked positions should remain active even though they contain [MASK].
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

        # Accuracy on active tokens only.
        disc_pred = disc_logits.gt(0)
        disc_true = disc_labels.gt(0.5)
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
            gen_loss_raw=gen_loss,
            disc_loss_raw=disc_loss,
        )
