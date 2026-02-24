"""Backbone config/model builders for RoPE and HF DeBERTa variants."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

from deberta.config import ModelConfig, validate_model_config
from deberta.modeling.deberta_v2_native import DebertaV2Model
from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel

_SPECIAL_ID_ATTRS = (
    "pad_token_id",
    "cls_token_id",
    "sep_token_id",
    "mask_token_id",
    "bos_token_id",
    "eos_token_id",
)
_COMPONENT_KIND = Literal["discriminator", "generator"]


@dataclass(frozen=True)
class _ResolvedComponentSources:
    """Resolved config/weight sources for one backbone component.

    :param str component: Component name (discriminator/generator).
    :param str | None config_source: Source used to load the component config, or None for synthetic/derived.
    :param str config_origin: Human-readable config-origin label.
    :param str | None weight_source: Source used to load pretrained weights, or None for scratch init.
    :param str weight_origin: Human-readable weight-origin label.
    :param bool derived_from_discriminator: Whether this component derives from discriminator config/weights.
    """

    component: _COMPONENT_KIND
    config_source: str | None
    config_origin: str
    weight_source: str | None
    weight_origin: str
    derived_from_discriminator: bool = False


@dataclass(frozen=True)
class _ResolvedBackboneSources:
    """Resolved config/weight source bundle for discriminator and generator."""

    discriminator: _ResolvedComponentSources
    generator: _ResolvedComponentSources


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    """Return tokenizer vocabulary size as an int.

    :param Any tokenizer: Tokenizer instance with ``__len__``.
    :return int: Tokenizer vocabulary size.
    """
    try:
        return int(len(tokenizer))
    except Exception as e:
        raise ValueError("Tokenizer must support len(tokenizer) for vocab-size alignment.") from e


def _resolve_backbone_sources(model_cfg: ModelConfig) -> _ResolvedBackboneSources:
    """Resolve deterministic config/weight sources for both components.

    :param ModelConfig model_cfg: User model configuration.
    :return _ResolvedBackboneSources: Resolved sources for discriminator/generator.
    """
    bt = (model_cfg.backbone_type or "rope").lower()
    from_scratch = bool(model_cfg.from_scratch)

    has_gen_cfg_src = bool(model_cfg.generator_config_name_or_path)
    has_gen_model_src = bool(model_cfg.generator_model_name_or_path)

    if not from_scratch and has_gen_cfg_src and not has_gen_model_src:
        raise ValueError(
            "model.generator_config_name_or_path requires model.generator_model_name_or_path when "
            "model.from_scratch=false. Explicit generator configs must pair with explicit generator "
            "weights; leave both unset to use derived-generator fallback from discriminator weights."
        )

    disc_cfg_source = (
        model_cfg.discriminator_config_name_or_path or model_cfg.discriminator_model_name_or_path
    )
    if bt == "rope" and from_scratch:
        discriminator = _ResolvedComponentSources(
            component="discriminator",
            config_source=None,
            config_origin="synthetic_from_model_cfg",
            weight_source=None,
            weight_origin="scratch",
            derived_from_discriminator=False,
        )
    else:
        discriminator = _ResolvedComponentSources(
            component="discriminator",
            config_source=disc_cfg_source,
            config_origin=(
                "discriminator_config_name_or_path"
                if model_cfg.discriminator_config_name_or_path
                else "discriminator_model_name_or_path"
            ),
            weight_source=(None if from_scratch else model_cfg.discriminator_model_name_or_path),
            weight_origin=("scratch" if from_scratch else "discriminator_model_name_or_path"),
            derived_from_discriminator=False,
        )

    if from_scratch:
        if model_cfg.generator_config_name_or_path:
            generator = _ResolvedComponentSources(
                component="generator",
                config_source=model_cfg.generator_config_name_or_path,
                config_origin="generator_config_name_or_path",
                weight_source=None,
                weight_origin="scratch",
                derived_from_discriminator=False,
            )
        elif model_cfg.generator_model_name_or_path:
            generator = _ResolvedComponentSources(
                component="generator",
                config_source=model_cfg.generator_model_name_or_path,
                config_origin="generator_model_name_or_path",
                weight_source=None,
                weight_origin="scratch",
                derived_from_discriminator=False,
            )
        else:
            generator = _ResolvedComponentSources(
                component="generator",
                config_source=None,
                config_origin="derived_from_discriminator_config",
                weight_source=None,
                weight_origin="scratch",
                derived_from_discriminator=True,
            )
    else:
        if model_cfg.generator_config_name_or_path and model_cfg.generator_model_name_or_path:
            generator = _ResolvedComponentSources(
                component="generator",
                config_source=model_cfg.generator_config_name_or_path,
                config_origin="generator_config_name_or_path",
                weight_source=model_cfg.generator_model_name_or_path,
                weight_origin="generator_model_name_or_path",
                derived_from_discriminator=False,
            )
        elif model_cfg.generator_model_name_or_path:
            generator = _ResolvedComponentSources(
                component="generator",
                config_source=model_cfg.generator_model_name_or_path,
                config_origin="generator_model_name_or_path",
                weight_source=model_cfg.generator_model_name_or_path,
                weight_origin="generator_model_name_or_path",
                derived_from_discriminator=False,
            )
        else:
            generator = _ResolvedComponentSources(
                component="generator",
                config_source=None,
                config_origin="derived_from_discriminator_config",
                weight_source=model_cfg.discriminator_model_name_or_path,
                weight_origin="derived_from_discriminator_model_name_or_path",
                derived_from_discriminator=True,
            )

    return _ResolvedBackboneSources(discriminator=discriminator, generator=generator)


def _apply_tokenizer_special_ids(cfg: Any, tokenizer: Any) -> None:
    """Copy tokenizer special-token ids onto a config object when available.

    :param Any cfg: Target config object.
    :param Any tokenizer: Tokenizer source.
    """
    for attr in _SPECIAL_ID_ATTRS:
        tok_id = getattr(tokenizer, attr, None)
        if tok_id is None:
            continue
        try:
            setattr(cfg, attr, int(tok_id))
        except Exception:
            continue


def _validate_or_fill_special_ids_from_tokenizer(
    cfg: Any,
    tokenizer: Any,
    *,
    component: _COMPONENT_KIND,
) -> None:
    """Validate special-token compatibility for pretrained configs.

    Missing config ids are backfilled from tokenizer; conflicting ids raise.

    :param Any cfg: Target config object.
    :param Any tokenizer: Tokenizer source.
    :param str component: Component name for error messaging.
    """
    for attr in _SPECIAL_ID_ATTRS:
        tok_id = getattr(tokenizer, attr, None)
        if tok_id is None:
            continue
        tok_int = int(tok_id)
        cfg_val = getattr(cfg, attr, None)
        if cfg_val is None:
            setattr(cfg, attr, tok_int)
            continue
        try:
            cfg_int = int(cfg_val)
        except Exception as e:
            raise ValueError(
                f"Invalid {component} config field {attr}={cfg_val!r}; expected int-like token id."
            ) from e
        if cfg_int != tok_int:
            raise ValueError(
                f"Tokenizer/config special-token mismatch for {component}.{attr}: "
                f"tokenizer={tok_int}, config={cfg_int}. Use a tokenizer compatible with the checkpoint."
            )


def _align_or_validate_tokenizer_contract(
    cfg: Any,
    tokenizer: Any,
    *,
    component: _COMPONENT_KIND,
    from_scratch: bool,
) -> None:
    """Apply tokenizer/config contract policy.

    Scratch mode aligns vocab/special ids from tokenizer. Pretrained mode validates vocab
    compatibility and validates/fills special ids.

    :param Any cfg: Target config object.
    :param Any tokenizer: Tokenizer source.
    :param str component: Component name for error messaging.
    :param bool from_scratch: Whether this run uses random initialization.
    """
    tok_vocab = _tokenizer_vocab_size(tokenizer)
    if from_scratch:
        cfg.vocab_size = tok_vocab
        _apply_tokenizer_special_ids(cfg, tokenizer)
        return

    cfg_vocab = getattr(cfg, "vocab_size", None)
    if cfg_vocab is None:
        raise ValueError(
            f"{component} config is missing vocab_size; cannot validate tokenizer/checkpoint compatibility."
        )
    if int(cfg_vocab) != tok_vocab:
        raise ValueError(
            f"Tokenizer/checkpoint vocab mismatch for {component}: tokenizer={tok_vocab}, "
            f"config={int(cfg_vocab)}. Use a matching tokenizer or checkpoint."
        )
    _validate_or_fill_special_ids_from_tokenizer(cfg, tokenizer, component=component)


def _validate_required_max_positions(
    cfg: Any,
    *,
    required_max_position_embeddings: int,
    component: _COMPONENT_KIND,
) -> None:
    """Validate component max_position_embeddings against required sequence length.

    :param Any cfg: Target config object.
    :param int required_max_position_embeddings: Required max sequence length.
    :param str component: Component name for error messaging.
    """
    cfg_max = getattr(cfg, "max_position_embeddings", None)
    if cfg_max is None:
        return
    if int(cfg_max) < int(required_max_position_embeddings):
        raise ValueError(
            f"{component} max_position_embeddings={int(cfg_max)} is smaller than required "
            f"sequence length {int(required_max_position_embeddings)}."
        )


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


def _apply_dropout_overrides(cfg: Any, model_cfg: ModelConfig) -> None:
    """Apply optional dropout overrides shared across backbones.

    :param Any cfg: Target config object.
    :param ModelConfig model_cfg: User model configuration.
    """
    if model_cfg.hidden_dropout_prob is not None:
        cfg.hidden_dropout_prob = float(model_cfg.hidden_dropout_prob)
    if model_cfg.attention_probs_dropout_prob is not None:
        cfg.attention_probs_dropout_prob = float(model_cfg.attention_probs_dropout_prob)


def _apply_rope_runtime_overrides(cfg: Any, model_cfg: ModelConfig) -> None:
    """Apply runtime-safe RoPE overrides common to scratch and pretrained flows.

    :param Any cfg: Target config object.
    :param ModelConfig model_cfg: User model configuration.
    """
    cfg.attention_implementation = str(model_cfg.attention_implementation)
    cfg.use_rmsnorm_heads = True
    _apply_dropout_overrides(cfg, model_cfg)


def _apply_rope_scratch_arch_overrides(
    cfg: Any,
    *,
    model_cfg: ModelConfig,
    max_position_embeddings: int,
    include_arch_from_model_cfg: bool,
    adjust_swiglu_intermediate: bool,
) -> None:
    """Apply scratch-only RoPE architecture overrides.

    :param Any cfg: Target config object.
    :param ModelConfig model_cfg: User model configuration.
    :param int max_position_embeddings: Sequence length budget.
    :param bool include_arch_from_model_cfg: Whether to apply hidden/layer/head/intermediate overrides.
    :param bool adjust_swiglu_intermediate: Whether to apply 2/3 intermediate-size scaling for SwiGLU.
    """
    if include_arch_from_model_cfg:
        cfg.hidden_size = int(model_cfg.hidden_size)
        cfg.num_hidden_layers = int(model_cfg.num_hidden_layers)
        cfg.num_attention_heads = int(model_cfg.num_attention_heads)
        cfg.intermediate_size = int(model_cfg.intermediate_size)
        cfg.hidden_act = str(model_cfg.hidden_act)

    cfg.max_position_embeddings = int(model_cfg.max_position_embeddings or max_position_embeddings)
    cfg.rope_theta = float(model_cfg.rope_theta)
    cfg.rotary_pct = float(model_cfg.rotary_pct)
    cfg.use_absolute_position_embeddings = bool(model_cfg.use_absolute_position_embeddings)
    cfg.type_vocab_size = int(model_cfg.type_vocab_size)
    cfg.norm_eps = float(model_cfg.norm_eps)
    cfg.norm_arch = str(model_cfg.norm_arch)
    cfg.keel_alpha_init = float(model_cfg.keel_alpha_init) if model_cfg.keel_alpha_init is not None else None
    cfg.keel_alpha_learnable = bool(model_cfg.keel_alpha_learnable)
    cfg.ffn_type = str(model_cfg.ffn_type)
    cfg.use_bias = bool(model_cfg.use_bias)
    cfg.initializer_range = float(model_cfg.initializer_range)

    if adjust_swiglu_intermediate:
        curr_intermediate = int(cfg.intermediate_size)
        cfg.intermediate_size = _scaled_swiglu_intermediate_size(curr_intermediate)


def _apply_rope_pretrained_explicit_overrides(cfg: Any, model_cfg: ModelConfig) -> None:
    """Apply explicit pretrained RoPE overrides.

    :param Any cfg: Target config object.
    :param ModelConfig model_cfg: User model configuration.
    """
    if model_cfg.pretrained_max_position_embeddings is not None:
        cfg.max_position_embeddings = int(model_cfg.pretrained_max_position_embeddings)
    if model_cfg.pretrained_rope_theta is not None:
        cfg.rope_theta = float(model_cfg.pretrained_rope_theta)
    if model_cfg.pretrained_rotary_pct is not None:
        cfg.rotary_pct = float(model_cfg.pretrained_rotary_pct)
    if model_cfg.pretrained_use_absolute_position_embeddings is not None:
        cfg.use_absolute_position_embeddings = bool(model_cfg.pretrained_use_absolute_position_embeddings)
    if model_cfg.pretrained_type_vocab_size is not None:
        cfg.type_vocab_size = int(model_cfg.pretrained_type_vocab_size)
    if model_cfg.pretrained_norm_arch is not None:
        cfg.norm_arch = str(model_cfg.pretrained_norm_arch)
    if model_cfg.pretrained_norm_eps is not None:
        cfg.norm_eps = float(model_cfg.pretrained_norm_eps)
    if model_cfg.pretrained_keel_alpha_init is not None:
        cfg.keel_alpha_init = float(model_cfg.pretrained_keel_alpha_init)
    if model_cfg.pretrained_keel_alpha_learnable is not None:
        cfg.keel_alpha_learnable = bool(model_cfg.pretrained_keel_alpha_learnable)
    if model_cfg.pretrained_ffn_type is not None:
        cfg.ffn_type = str(model_cfg.pretrained_ffn_type)
    if model_cfg.pretrained_use_bias is not None:
        cfg.use_bias = bool(model_cfg.pretrained_use_bias)
    if model_cfg.pretrained_initializer_range is not None:
        cfg.initializer_range = float(model_cfg.pretrained_initializer_range)


def _apply_hf_config_normalization(
    cfg: Any,
    *,
    model_cfg: ModelConfig,
    tokenizer: Any,
    component: _COMPONENT_KIND,
) -> None:
    """Apply normalized HF config policy.

    :param Any cfg: Target config object.
    :param ModelConfig model_cfg: User model configuration.
    :param Any tokenizer: Tokenizer source.
    :param str component: Component name for diagnostics.
    """
    _align_or_validate_tokenizer_contract(
        cfg,
        tokenizer,
        component=component,
        from_scratch=bool(model_cfg.from_scratch),
    )
    _apply_dropout_overrides(cfg, model_cfg)
    cfg.use_rmsnorm_heads = False


def _apply_rope_config_normalization(
    cfg: Any,
    *,
    model_cfg: ModelConfig,
    tokenizer: Any,
    max_position_embeddings: int,
    component: _COMPONENT_KIND,
    explicit_source: bool,
    derived_from_discriminator: bool,
) -> None:
    """Apply normalized RoPE config policy.

    :param Any cfg: Target config object.
    :param ModelConfig model_cfg: User model configuration.
    :param Any tokenizer: Tokenizer source.
    :param int max_position_embeddings: Sequence length budget.
    :param str component: Component name for diagnostics.
    :param bool explicit_source: Whether this component config came from explicit source.
    :param bool derived_from_discriminator: Whether this config was derived from discriminator config.
    """
    if model_cfg.from_scratch:
        if explicit_source:
            if model_cfg.max_position_embeddings is not None:
                cfg.max_position_embeddings = int(model_cfg.max_position_embeddings)
            else:
                cfg_max = getattr(cfg, "max_position_embeddings", None)
                if cfg_max is None:
                    cfg.max_position_embeddings = int(max_position_embeddings)
        elif derived_from_discriminator:
            # Derived configs already inherit discriminator architecture and position budget.
            pass
        else:
            should_adjust_swiglu = bool(
                str(model_cfg.ffn_type).strip().lower() == "swiglu"
                and bool(model_cfg.swiglu_adjust_intermediate)
            )
            _apply_rope_scratch_arch_overrides(
                cfg,
                model_cfg=model_cfg,
                max_position_embeddings=max_position_embeddings,
                include_arch_from_model_cfg=True,
                adjust_swiglu_intermediate=should_adjust_swiglu,
            )
    else:
        _apply_rope_pretrained_explicit_overrides(cfg, model_cfg)

    _apply_rope_runtime_overrides(cfg, model_cfg)
    _align_or_validate_tokenizer_contract(
        cfg,
        tokenizer,
        component=component,
        from_scratch=bool(model_cfg.from_scratch),
    )
    _validate_required_max_positions(
        cfg,
        required_max_position_embeddings=int(max_position_embeddings),
        component=component,
    )


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
    resolved = _resolve_backbone_sources(model_cfg)

    if bt == "hf_deberta_v2":
        from transformers import AutoConfig

        if resolved.discriminator.config_source is None:
            raise RuntimeError("Resolved discriminator config source is missing for hf_deberta_v2 backbone.")
        disc_cfg = AutoConfig.from_pretrained(resolved.discriminator.config_source)

        if resolved.generator.config_source is None:
            gen_cfg = _derive_generator_config(disc_cfg, model_cfg)
        else:
            gen_cfg = AutoConfig.from_pretrained(resolved.generator.config_source)

        _apply_hf_config_normalization(
            disc_cfg,
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            component="discriminator",
        )
        _apply_hf_config_normalization(
            gen_cfg,
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            component="generator",
        )
        _validate_required_max_positions(
            disc_cfg,
            required_max_position_embeddings=int(max_position_embeddings),
            component="discriminator",
        )
        _validate_required_max_positions(
            gen_cfg,
            required_max_position_embeddings=int(max_position_embeddings),
            component="generator",
        )

        return disc_cfg, gen_cfg

    # RoPE backbone
    if resolved.discriminator.config_source is None:
        disc_cfg = DebertaRoPEConfig(
            hidden_size=model_cfg.hidden_size,
            num_hidden_layers=model_cfg.num_hidden_layers,
            num_attention_heads=model_cfg.num_attention_heads,
            intermediate_size=model_cfg.intermediate_size,
            hidden_act=model_cfg.hidden_act,
        )
    else:
        disc_cfg = DebertaRoPEConfig.from_pretrained(resolved.discriminator.config_source)

    _apply_rope_config_normalization(
        disc_cfg,
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=max_position_embeddings,
        component="discriminator",
        explicit_source=resolved.discriminator.config_source is not None,
        derived_from_discriminator=False,
    )

    if resolved.generator.config_source is None:
        gen_cfg = _derive_generator_config(disc_cfg, model_cfg)
    else:
        gen_cfg = DebertaRoPEConfig.from_pretrained(resolved.generator.config_source)

    _apply_rope_config_normalization(
        gen_cfg,
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=max_position_embeddings,
        component="generator",
        explicit_source=resolved.generator.config_source is not None,
        derived_from_discriminator=resolved.generator.derived_from_discriminator,
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
    resolved = _resolve_backbone_sources(model_cfg)

    if bt == "hf_deberta_v2":
        if model_cfg.from_scratch:
            disc = DebertaV2Model(disc_config)
            gen = DebertaV2Model(gen_config)
            return disc, gen

        disc_src = resolved.discriminator.weight_source
        gen_src = resolved.generator.weight_source
        if disc_src is None or gen_src is None:
            raise RuntimeError("Resolved pretrained HF weight source is missing.")

        try:
            disc = DebertaV2Model.from_pretrained(disc_src, config=disc_config)
        except Exception as e:
            raise RuntimeError(
                "Failed to load discriminator HF backbone from "
                f"source '{disc_src}' (resolved from {resolved.discriminator.weight_origin})."
            ) from e

        try:
            gen = DebertaV2Model.from_pretrained(gen_src, config=gen_config)
        except Exception as e:
            raise RuntimeError(
                "Failed to load generator HF backbone from "
                f"source '{gen_src}' (resolved from {resolved.generator.weight_origin})."
            ) from e

        return disc, gen

    # RoPE backbone
    if model_cfg.from_scratch:
        disc = DebertaRoPEModel(disc_config)
        gen = DebertaRoPEModel(gen_config)
        return disc, gen

    disc_src = resolved.discriminator.weight_source
    gen_src = resolved.generator.weight_source
    if disc_src is None or gen_src is None:
        raise RuntimeError("Resolved pretrained RoPE weight source is missing.")

    try:
        disc = DebertaRoPEModel.from_pretrained(disc_src, config=disc_config)
    except Exception as e:
        raise RuntimeError(
            "Failed to load discriminator RoPE checkpoint with model.from_scratch=false. "
            f"Resolved source: '{disc_src}' ({resolved.discriminator.weight_origin})."
        ) from e

    try:
        gen = DebertaRoPEModel.from_pretrained(gen_src, config=gen_config)
    except Exception as e:
        raise RuntimeError(
            "Failed to load generator RoPE checkpoint with model.from_scratch=false. "
            f"Resolved source: '{gen_src}' ({resolved.generator.weight_origin})."
        ) from e

    return disc, gen
