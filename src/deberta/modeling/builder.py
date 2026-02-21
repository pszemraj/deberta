"""Backbone config/model builders for RoPE and HF DeBERTa variants."""

from __future__ import annotations

import copy
from typing import Any

from deberta.config import ModelConfig, validate_model_config
from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel


def _apply_tokenizer_special_ids(cfg: Any, tokenizer: Any) -> None:
    """Copy tokenizer special-token ids onto a config object when available.

    :param Any cfg: Target config object.
    :param Any tokenizer: Tokenizer source.
    """
    for attr in (
        "pad_token_id",
        "cls_token_id",
        "sep_token_id",
        "mask_token_id",
        "bos_token_id",
        "eos_token_id",
    ):
        tok_id = getattr(tokenizer, attr, None)
        if tok_id is None:
            continue
        try:
            setattr(cfg, attr, int(tok_id))
        except Exception:
            continue


def _scaled_swiglu_intermediate_size(value: int) -> int:
    """Return a SwiGLU intermediate size scaled to MLP-equivalent parameter budget.

    :param int value: Baseline MLP-oriented intermediate size.
    :return int: Scaled intermediate size for SwiGLU.
    """
    multiple_of = 128
    raw = max(1, int(int(value) * (2.0 / 3.0)))
    remainder = raw % multiple_of
    if remainder == 0:
        return raw
    return raw + (multiple_of - remainder)


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


def _apply_rope_config_overrides(
    cfg: Any,
    *,
    model_cfg: ModelConfig,
    tokenizer: Any,
    max_position_embeddings: int,
    adjust_swiglu_intermediate: bool = False,
    include_arch_from_model_cfg: bool = True,
) -> None:
    """Apply shared RoPE config overrides for generator/discriminator configs.

    :param Any cfg: Target config object.
    :param ModelConfig model_cfg: User model configuration.
    :param Any tokenizer: Tokenizer for vocab/pad metadata.
    :param int max_position_embeddings: Sequence length budget.
    :param bool adjust_swiglu_intermediate: Whether to apply 2/3 intermediate-size scaling for SwiGLU.
    :param bool include_arch_from_model_cfg: Whether to copy architecture dimensions from `model_cfg`.
    """
    defaults = ModelConfig()
    cfg.vocab_size = int(len(tokenizer))
    if model_cfg.from_scratch and include_arch_from_model_cfg:
        cfg.hidden_size = int(model_cfg.hidden_size)
        cfg.num_hidden_layers = int(model_cfg.num_hidden_layers)
        cfg.num_attention_heads = int(model_cfg.num_attention_heads)
        cfg.intermediate_size = int(model_cfg.intermediate_size)
        cfg.hidden_act = str(model_cfg.hidden_act)
    _apply_tokenizer_special_ids(cfg, tokenizer)
    if model_cfg.from_scratch:
        cfg.max_position_embeddings = int(model_cfg.max_position_embeddings or max_position_embeddings)
    elif model_cfg.max_position_embeddings is not None:
        # Keep checkpoint-native max positions by default; only override when explicit.
        cfg.max_position_embeddings = int(model_cfg.max_position_embeddings)

    if model_cfg.from_scratch:
        cfg.rope_theta = float(model_cfg.rope_theta)
        cfg.rotary_pct = float(model_cfg.rotary_pct)
        cfg.use_absolute_position_embeddings = bool(model_cfg.use_absolute_position_embeddings)
        cfg.type_vocab_size = int(model_cfg.type_vocab_size)
        cfg.norm_eps = float(model_cfg.norm_eps)
        cfg.norm_arch = str(model_cfg.norm_arch)
        cfg.keel_alpha_init = (
            float(model_cfg.keel_alpha_init) if model_cfg.keel_alpha_init is not None else None
        )
        cfg.keel_alpha_learnable = bool(model_cfg.keel_alpha_learnable)
    else:
        # For pretrained RoPE checkpoints, preserve checkpoint architecture unless
        # user explicitly deviates from ModelConfig defaults.
        if float(model_cfg.rope_theta) != float(defaults.rope_theta):
            cfg.rope_theta = float(model_cfg.rope_theta)
        if float(model_cfg.rotary_pct) != float(defaults.rotary_pct):
            cfg.rotary_pct = float(model_cfg.rotary_pct)
        if bool(model_cfg.use_absolute_position_embeddings) != bool(
            defaults.use_absolute_position_embeddings
        ):
            cfg.use_absolute_position_embeddings = bool(model_cfg.use_absolute_position_embeddings)
        if int(model_cfg.type_vocab_size) != int(defaults.type_vocab_size):
            cfg.type_vocab_size = int(model_cfg.type_vocab_size)
        if float(model_cfg.norm_eps) != float(defaults.norm_eps):
            cfg.norm_eps = float(model_cfg.norm_eps)
        if str(model_cfg.norm_arch) != str(defaults.norm_arch):
            cfg.norm_arch = str(model_cfg.norm_arch)
        if model_cfg.keel_alpha_init != defaults.keel_alpha_init:
            cfg.keel_alpha_init = (
                float(model_cfg.keel_alpha_init) if model_cfg.keel_alpha_init is not None else None
            )
        if bool(model_cfg.keel_alpha_learnable) != bool(defaults.keel_alpha_learnable):
            cfg.keel_alpha_learnable = bool(model_cfg.keel_alpha_learnable)

    cfg.attention_implementation = str(model_cfg.attention_implementation)
    cfg.use_rmsnorm_heads = True
    if model_cfg.from_scratch:
        # Scratch RoPE builds fully follow model_cfg architecture knobs.
        cfg.ffn_type = str(model_cfg.ffn_type)
        cfg.use_bias = bool(model_cfg.use_bias)
        if bool(adjust_swiglu_intermediate):
            curr_intermediate = int(cfg.intermediate_size)
            cfg.intermediate_size = _scaled_swiglu_intermediate_size(curr_intermediate)
        cfg.initializer_range = float(model_cfg.initializer_range)
    elif float(model_cfg.initializer_range) != float(defaults.initializer_range):
        # No effect on loaded parameters, but keep explicit metadata override consistent.
        cfg.initializer_range = float(model_cfg.initializer_range)

    if model_cfg.hidden_dropout_prob is not None:
        cfg.hidden_dropout_prob = float(model_cfg.hidden_dropout_prob)

    if model_cfg.attention_probs_dropout_prob is not None:
        cfg.attention_probs_dropout_prob = float(model_cfg.attention_probs_dropout_prob)


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
    validate_model_config(model_cfg)
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

        # Optional dropout overrides. ``None`` preserves checkpoint-native values;
        # any numeric value (including 0.0) is an explicit override.
        if model_cfg.hidden_dropout_prob is not None:
            disc_cfg.hidden_dropout_prob = float(model_cfg.hidden_dropout_prob)
            gen_cfg.hidden_dropout_prob = float(model_cfg.hidden_dropout_prob)

        if model_cfg.attention_probs_dropout_prob is not None:
            disc_cfg.attention_probs_dropout_prob = float(model_cfg.attention_probs_dropout_prob)
            gen_cfg.attention_probs_dropout_prob = float(model_cfg.attention_probs_dropout_prob)

        disc_cfg.use_rmsnorm_heads = False
        gen_cfg.use_rmsnorm_heads = False

        _apply_tokenizer_special_ids(disc_cfg, tokenizer)
        _apply_tokenizer_special_ids(gen_cfg, tokenizer)

        return disc_cfg, gen_cfg

    # RoPE backbone
    if model_cfg.from_scratch:
        disc_cfg = DebertaRoPEConfig(
            hidden_size=model_cfg.hidden_size,
            num_hidden_layers=model_cfg.num_hidden_layers,
            num_attention_heads=model_cfg.num_attention_heads,
            intermediate_size=model_cfg.intermediate_size,
            hidden_act=model_cfg.hidden_act,
        )
    else:
        disc_src = model_cfg.discriminator_config_name_or_path or model_cfg.discriminator_model_name_or_path
        disc_cfg = DebertaRoPEConfig.from_pretrained(disc_src)
    should_adjust_swiglu = bool(
        model_cfg.from_scratch
        and str(model_cfg.ffn_type).strip().lower() == "swiglu"
        and bool(model_cfg.swiglu_adjust_intermediate)
    )

    _apply_rope_config_overrides(
        disc_cfg,
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=max_position_embeddings,
        adjust_swiglu_intermediate=should_adjust_swiglu,
    )

    # Generator config
    has_explicit_generator_src = bool(
        model_cfg.generator_config_name_or_path or model_cfg.generator_model_name_or_path
    )
    if model_cfg.generator_config_name_or_path:
        gen_cfg = DebertaRoPEConfig.from_pretrained(model_cfg.generator_config_name_or_path)
    elif model_cfg.generator_model_name_or_path:
        gen_cfg = DebertaRoPEConfig.from_pretrained(model_cfg.generator_model_name_or_path)
    else:
        gen_cfg = _derive_generator_config(disc_cfg, model_cfg)

    _apply_rope_config_overrides(
        gen_cfg,
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=max_position_embeddings,
        # Explicit generator configs should honor model_cfg overrides, while derived configs should
        # preserve discriminator-derived architecture settings.
        adjust_swiglu_intermediate=(should_adjust_swiglu and has_explicit_generator_src),
        include_arch_from_model_cfg=has_explicit_generator_src,
    )

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
    try:
        disc = DebertaRoPEModel.from_pretrained(
            model_cfg.discriminator_model_name_or_path, config=disc_config
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load discriminator RoPE checkpoint with model.from_scratch=false. "
            "Ensure model.discriminator_model_name_or_path points to a DebertaRoPE checkpoint "
            "(not an HF DeBERTa v2/v3 checkpoint)."
        ) from e
    gen_src = model_cfg.generator_model_name_or_path or model_cfg.discriminator_model_name_or_path
    try:
        gen = DebertaRoPEModel.from_pretrained(gen_src, config=gen_config)
    except Exception as e:
        raise RuntimeError(
            "Failed to load generator RoPE checkpoint with model.from_scratch=false. "
            "Ensure model.generator_model_name_or_path (or discriminator source) points to "
            "a DebertaRoPE checkpoint (not an HF DeBERTa v2/v3 checkpoint)."
        ) from e
    return disc, gen
