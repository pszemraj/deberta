"""Backbone config/model builders for RoPE and HF DeBERTa variants."""

from __future__ import annotations

import copy
from typing import Any

from deberta.config import ModelConfig
from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel


def _derive_generator_config(base_cfg: Any, model_cfg: ModelConfig) -> Any:
    """Derive generator config from discriminator config.

    :param Any base_cfg: Base discriminator config.
    :param ModelConfig model_cfg: User model configuration.
    :return Any: Derived generator config.
    """
    gen_cfg = copy.deepcopy(base_cfg)

    # Default ELECTRA heuristic: fewer layers.
    if getattr(gen_cfg, "num_hidden_layers", None) is not None:
        disc_layers = int(gen_cfg.num_hidden_layers)
        default_gen_layers = max(1, disc_layers // 3)
        gen_cfg.num_hidden_layers = int(model_cfg.generator_num_hidden_layers or default_gen_layers)

    if model_cfg.generator_hidden_size is not None:
        gen_cfg.hidden_size = int(model_cfg.generator_hidden_size)
    if model_cfg.generator_intermediate_size is not None:
        gen_cfg.intermediate_size = int(model_cfg.generator_intermediate_size)
    if model_cfg.generator_num_attention_heads is not None:
        gen_cfg.num_attention_heads = int(model_cfg.generator_num_attention_heads)

    # Sanity
    if getattr(gen_cfg, "hidden_size", None) and getattr(gen_cfg, "num_attention_heads", None):
        if int(gen_cfg.hidden_size) % int(gen_cfg.num_attention_heads) != 0:
            raise ValueError(
                "generator_hidden_size must be divisible by generator_num_attention_heads. "
                f"Got hidden_size={gen_cfg.hidden_size}, heads={gen_cfg.num_attention_heads}."
            )

    return gen_cfg


def build_backbone_configs(
    *,
    model_cfg: ModelConfig,
    tokenizer: Any,
    max_position_embeddings: int,
) -> tuple[Any, Any]:
    """Build discriminator + generator configs.

    - For backbone_type='hf_deberta_v2': returns HF configs (AutoConfig instances).
    - For backbone_type='rope': returns DebertaRoPEConfig instances.

    Generator config is loaded if specified, otherwise derived from discriminator config.

    :param ModelConfig model_cfg: User model configuration.
    :param Any tokenizer: Tokenizer used for vocab/pad metadata.
    :param int max_position_embeddings: Sequence length budget.
    :return tuple[Any, Any]: Discriminator and generator configs.
    """
    bt = (model_cfg.backbone_type or "rope").lower()

    if bt == "hf_deberta_v2":
        from transformers import AutoConfig

        disc_src = model_cfg.discriminator_config_name_or_path or model_cfg.discriminator_model_name_or_path
        disc_cfg = AutoConfig.from_pretrained(disc_src)

        # Generator config
        if model_cfg.generator_config_name_or_path:
            gen_cfg = AutoConfig.from_pretrained(model_cfg.generator_config_name_or_path)
        elif model_cfg.generator_model_name_or_path:
            gen_cfg = AutoConfig.from_pretrained(model_cfg.generator_model_name_or_path)
        else:
            gen_cfg = _derive_generator_config(disc_cfg, model_cfg)

        # Optional dropout overrides
        if model_cfg.hidden_dropout_prob is not None:
            disc_cfg.hidden_dropout_prob = float(model_cfg.hidden_dropout_prob)
            gen_cfg.hidden_dropout_prob = float(model_cfg.hidden_dropout_prob)
        if model_cfg.attention_probs_dropout_prob is not None:
            disc_cfg.attention_probs_dropout_prob = float(model_cfg.attention_probs_dropout_prob)
            gen_cfg.attention_probs_dropout_prob = float(model_cfg.attention_probs_dropout_prob)

        return disc_cfg, gen_cfg

    if bt != "rope":
        raise ValueError(f"Unsupported backbone_type: {model_cfg.backbone_type}")

    # RoPE backbone
    disc_src = model_cfg.discriminator_config_name_or_path or model_cfg.discriminator_model_name_or_path
    disc_cfg = DebertaRoPEConfig.from_pretrained(disc_src)

    # Override/force critical fields for RoPE mode
    disc_cfg.vocab_size = int(len(tokenizer))
    disc_cfg.pad_token_id = int(tokenizer.pad_token_id or 0)
    disc_cfg.max_position_embeddings = int(model_cfg.max_position_embeddings or max_position_embeddings)

    disc_cfg.rope_theta = float(model_cfg.rope_theta)
    disc_cfg.rotary_pct = float(model_cfg.rotary_pct)
    disc_cfg.use_absolute_position_embeddings = bool(model_cfg.use_absolute_position_embeddings)
    disc_cfg.type_vocab_size = int(model_cfg.type_vocab_size)

    disc_cfg.norm_eps = float(model_cfg.norm_eps)
    disc_cfg.norm_arch = str(model_cfg.norm_arch)
    disc_cfg.keel_alpha_init = (
        float(model_cfg.keel_alpha_init) if model_cfg.keel_alpha_init is not None else None
    )
    disc_cfg.keel_alpha_learnable = bool(model_cfg.keel_alpha_learnable)
    disc_cfg.attention_implementation = str(model_cfg.attention_implementation)
    disc_cfg.initializer_range = float(model_cfg.initializer_range)

    # Optional dropout overrides
    if model_cfg.hidden_dropout_prob is not None:
        disc_cfg.hidden_dropout_prob = float(model_cfg.hidden_dropout_prob)
    if model_cfg.attention_probs_dropout_prob is not None:
        disc_cfg.attention_probs_dropout_prob = float(model_cfg.attention_probs_dropout_prob)

    # Generator config
    if model_cfg.generator_config_name_or_path:
        gen_cfg = DebertaRoPEConfig.from_pretrained(model_cfg.generator_config_name_or_path)
    elif model_cfg.generator_model_name_or_path:
        gen_cfg = DebertaRoPEConfig.from_pretrained(model_cfg.generator_model_name_or_path)
    else:
        gen_cfg = _derive_generator_config(disc_cfg, model_cfg)

    # Make sure generator uses the same RoPE/norm settings unless explicitly overridden
    gen_cfg.vocab_size = int(len(tokenizer))
    gen_cfg.pad_token_id = int(tokenizer.pad_token_id or 0)
    gen_cfg.max_position_embeddings = int(model_cfg.max_position_embeddings or max_position_embeddings)

    gen_cfg.rope_theta = float(model_cfg.rope_theta)
    gen_cfg.rotary_pct = float(model_cfg.rotary_pct)
    gen_cfg.use_absolute_position_embeddings = bool(model_cfg.use_absolute_position_embeddings)
    gen_cfg.type_vocab_size = int(model_cfg.type_vocab_size)

    gen_cfg.norm_eps = float(model_cfg.norm_eps)
    gen_cfg.norm_arch = str(model_cfg.norm_arch)
    gen_cfg.keel_alpha_init = (
        float(model_cfg.keel_alpha_init) if model_cfg.keel_alpha_init is not None else None
    )
    gen_cfg.keel_alpha_learnable = bool(model_cfg.keel_alpha_learnable)
    gen_cfg.attention_implementation = str(model_cfg.attention_implementation)
    gen_cfg.initializer_range = float(model_cfg.initializer_range)

    if model_cfg.hidden_dropout_prob is not None:
        gen_cfg.hidden_dropout_prob = float(model_cfg.hidden_dropout_prob)
    if model_cfg.attention_probs_dropout_prob is not None:
        gen_cfg.attention_probs_dropout_prob = float(model_cfg.attention_probs_dropout_prob)

    return disc_cfg, gen_cfg


def build_backbones(
    *,
    model_cfg: ModelConfig,
    disc_config: Any,
    gen_config: Any,
) -> tuple[Any, Any]:
    """Instantiate discriminator + generator backbones.

    :param ModelConfig model_cfg: User model configuration.
    :param Any disc_config: Discriminator config object.
    :param Any gen_config: Generator config object.
    :return tuple[Any, Any]: Instantiated discriminator and generator modules.
    """
    bt = (model_cfg.backbone_type or "rope").lower()

    if bt == "hf_deberta_v2":
        from transformers import AutoModel

        if model_cfg.from_scratch:
            disc = AutoModel.from_config(disc_config)
            gen = AutoModel.from_config(gen_config)
        else:
            disc = AutoModel.from_pretrained(model_cfg.discriminator_model_name_or_path, config=disc_config)
            gen_src = model_cfg.generator_model_name_or_path or model_cfg.discriminator_model_name_or_path
            gen = AutoModel.from_pretrained(gen_src, config=gen_config)
        return disc, gen

    # RoPE backbone
    if model_cfg.from_scratch:
        disc = DebertaRoPEModel(disc_config)
        gen = DebertaRoPEModel(gen_config)
        return disc, gen

    # from_pretrained path for rope (must point to a DebertaRoPEModel checkpoint)
    disc = DebertaRoPEModel.from_pretrained(model_cfg.discriminator_model_name_or_path, config=disc_config)
    gen_src = model_cfg.generator_model_name_or_path or model_cfg.discriminator_model_name_or_path
    gen = DebertaRoPEModel.from_pretrained(gen_src, config=gen_config)
    return disc, gen
