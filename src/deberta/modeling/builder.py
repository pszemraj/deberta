"""Backbone config/model builders for RoPE and HF DeBERTa variants."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

from deberta.config import ModelConfig, validate_model_config
from deberta.modeling.deberta_v2_native import DebertaV2Config, DebertaV2Model
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
_EXTRA_TOKEN_TEMPLATE = "<|deberta_extra_token_{idx}|>"


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


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round ``value`` up to ``multiple``.

    :param int value: Input value.
    :param int multiple: Positive multiple.
    :return int: Rounded value.
    """
    if multiple <= 1:
        return int(value)
    remainder = int(value) % int(multiple)
    if remainder == 0:
        return int(value)
    return int(value) + (int(multiple) - remainder)


def _resize_tokenizer_to_vocab_size(
    *,
    tokenizer: Any,
    target_size: int,
    component: _COMPONENT_KIND,
    allow_resize: bool,
) -> None:
    """Grow tokenizer vocabulary to ``target_size`` by adding inert placeholder tokens.

    :param Any tokenizer: Tokenizer instance.
    :param int target_size: Desired tokenizer vocabulary size.
    :param str component: Component name for diagnostics.
    :param bool allow_resize: Whether tokenizer growth is permitted.
    :raises ValueError: If growth is required but not permitted/available.
    :raises RuntimeError: If tokenizer growth does not reach the requested size.
    """
    current = _tokenizer_vocab_size(tokenizer)
    if int(target_size) <= int(current):
        return
    if not bool(allow_resize):
        raise ValueError(
            f"{component} tokenizer vocab ({current}) is smaller than required size ({target_size}), "
            "but model.tokenizer_allow_vocab_resize=false. Enable tokenizer resize or align vocab settings."
        )

    add_tokens = getattr(tokenizer, "add_tokens", None)
    if not callable(add_tokens):
        raise ValueError(
            f"{component} tokenizer does not support add_tokens(...), cannot grow vocab from "
            f"{current} to {target_size}."
        )

    needed = int(target_size) - int(current)
    extra_tokens = [_EXTRA_TOKEN_TEMPLATE.format(idx=idx) for idx in range(int(current), int(target_size))]
    added = int(add_tokens(extra_tokens, special_tokens=False))
    final_size = _tokenizer_vocab_size(tokenizer)
    if added != needed or final_size != int(target_size):
        raise RuntimeError(
            f"Failed to grow {component} tokenizer vocab: needed={needed}, added={added}, "
            f"final_size={final_size}, target={int(target_size)}."
        )


def _resolve_required_tokenizer_vocab_size(
    *,
    current_size: int,
    model_cfg: ModelConfig,
    from_scratch: bool,
    component: _COMPONENT_KIND,
    config_vocab_size: int | None = None,
) -> int:
    """Resolve required tokenizer size from config controls.

    :param int current_size: Current tokenizer size.
    :param ModelConfig model_cfg: Runtime model config.
    :param bool from_scratch: Whether run builds backbones from scratch.
    :param str component: Component name for diagnostics.
    :param int | None config_vocab_size: Existing config vocab size for pretrained validation.
    :raises ValueError: If requested size constraints are invalid.
    :return int: Required tokenizer size.
    """
    requested_target = (
        int(model_cfg.tokenizer_vocab_target) if model_cfg.tokenizer_vocab_target is not None else None
    )
    multiple = int(model_cfg.tokenizer_vocab_multiple)
    desired = int(current_size)

    if requested_target is not None:
        if requested_target < int(current_size):
            raise ValueError(
                f"model.tokenizer_vocab_target ({requested_target}) is smaller than tokenizer size "
                f"({current_size}) for {component}; shrinking tokenizer vocab is not supported."
            )
        desired = max(desired, int(requested_target))

    if not bool(from_scratch):
        if config_vocab_size is None:
            raise ValueError(f"{component} config vocab_size is required in pretrained mode.")
        cfg_vocab = int(config_vocab_size)
        if int(current_size) > cfg_vocab:
            raise ValueError(
                f"Tokenizer/checkpoint vocab mismatch for {component}: tokenizer={int(current_size)}, "
                f"config={cfg_vocab}. Use a matching tokenizer or checkpoint."
            )
        # Optional convenience path: when resize is enabled and no explicit target is provided,
        # align tokenizer length to checkpoint vocab_size.
        if requested_target is None and bool(model_cfg.tokenizer_allow_vocab_resize):
            desired = max(desired, cfg_vocab)
        if desired > cfg_vocab:
            raise ValueError(
                f"Requested tokenizer vocab for {component} ({desired}) exceeds checkpoint config "
                f"vocab_size ({cfg_vocab})."
            )

    desired = _round_up_to_multiple(desired, multiple)
    if not bool(from_scratch):
        cfg_vocab = int(config_vocab_size)
        if desired > cfg_vocab:
            raise ValueError(
                f"model.tokenizer_vocab_multiple={multiple} rounds {component} tokenizer size to {desired}, "
                f"which exceeds checkpoint config vocab_size ({cfg_vocab})."
            )
    return int(desired)


def _resolve_backbone_sources(model_cfg: ModelConfig) -> _ResolvedBackboneSources:
    """Resolve deterministic config/weight sources for both components.

    :param ModelConfig model_cfg: User model configuration.
    :return _ResolvedBackboneSources: Resolved sources for discriminator/generator.
    """
    bt = (model_cfg.backbone_type or "hf_deberta_v2").lower()
    from_scratch = bool(model_cfg.from_scratch)

    if bt == "hf_deberta_v2":
        discriminator = _ResolvedComponentSources(
            component="discriminator",
            config_source=None,
            config_origin="repo_hf_defaults",
            weight_source=(None if from_scratch else model_cfg.pretrained_discriminator_path),
            weight_origin=("scratch" if from_scratch else "pretrained_discriminator_path"),
            derived_from_discriminator=False,
        )
        if from_scratch:
            generator = _ResolvedComponentSources(
                component="generator",
                config_source=None,
                config_origin="derived_from_discriminator_config",
                weight_source=None,
                weight_origin="scratch",
                derived_from_discriminator=not bool(model_cfg.pretrained_generator_path),
            )
        else:
            gen_weight_src = model_cfg.pretrained_generator_path or model_cfg.pretrained_discriminator_path
            gen_weight_origin = (
                "pretrained_generator_path"
                if model_cfg.pretrained_generator_path
                else "derived_from_pretrained_discriminator_path"
            )
            generator = _ResolvedComponentSources(
                component="generator",
                config_source=None,
                config_origin="derived_from_discriminator_config",
                weight_source=gen_weight_src,
                weight_origin=gen_weight_origin,
                derived_from_discriminator=not bool(model_cfg.pretrained_generator_path),
            )
        return _ResolvedBackboneSources(discriminator=discriminator, generator=generator)

    disc_cfg_source = model_cfg.pretrained_discriminator_path
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
            config_origin="pretrained_discriminator_path",
            weight_source=(None if from_scratch else model_cfg.pretrained_discriminator_path),
            weight_origin=("scratch" if from_scratch else "pretrained_discriminator_path"),
            derived_from_discriminator=False,
        )

    if from_scratch:
        if model_cfg.pretrained_generator_path:
            generator = _ResolvedComponentSources(
                component="generator",
                config_source=model_cfg.pretrained_generator_path,
                config_origin="pretrained_generator_path",
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
        if model_cfg.pretrained_generator_path:
            generator = _ResolvedComponentSources(
                component="generator",
                config_source=model_cfg.pretrained_generator_path,
                config_origin="pretrained_generator_path",
                weight_source=model_cfg.pretrained_generator_path,
                weight_origin="pretrained_generator_path",
                derived_from_discriminator=False,
            )
        else:
            generator = _ResolvedComponentSources(
                component="generator",
                config_source=None,
                config_origin="derived_from_discriminator_config",
                weight_source=model_cfg.pretrained_discriminator_path,
                weight_origin="derived_from_pretrained_discriminator_path",
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
    model_cfg: ModelConfig,
) -> None:
    """Apply tokenizer/config contract policy.

    Scratch mode aligns vocab/special ids from tokenizer. Pretrained mode validates vocab
    compatibility and validates/fills special ids.

    :param Any cfg: Target config object.
    :param Any tokenizer: Tokenizer source.
    :param str component: Component name for error messaging.
    :param ModelConfig model_cfg: Runtime model configuration.
    """
    from_scratch = bool(model_cfg.from_scratch)
    tok_vocab = _tokenizer_vocab_size(tokenizer)

    if from_scratch:
        required_vocab = _resolve_required_tokenizer_vocab_size(
            current_size=tok_vocab,
            model_cfg=model_cfg,
            from_scratch=True,
            component=component,
        )
        _resize_tokenizer_to_vocab_size(
            tokenizer=tokenizer,
            target_size=required_vocab,
            component=component,
            allow_resize=bool(model_cfg.tokenizer_allow_vocab_resize),
        )
        tok_vocab = _tokenizer_vocab_size(tokenizer)
        cfg.vocab_size = tok_vocab
        _apply_tokenizer_special_ids(cfg, tokenizer)
        return

    cfg_vocab = getattr(cfg, "vocab_size", None)
    if cfg_vocab is None:
        raise ValueError(
            f"{component} config is missing vocab_size; cannot validate tokenizer/checkpoint compatibility."
        )
    required_vocab = _resolve_required_tokenizer_vocab_size(
        current_size=tok_vocab,
        model_cfg=model_cfg,
        from_scratch=False,
        component=component,
        config_vocab_size=int(cfg_vocab),
    )
    _resize_tokenizer_to_vocab_size(
        tokenizer=tokenizer,
        target_size=required_vocab,
        component=component,
        allow_resize=bool(model_cfg.tokenizer_allow_vocab_resize),
    )
    tok_vocab = _tokenizer_vocab_size(tokenizer)

    if int(cfg_vocab) != int(tok_vocab):
        raise ValueError(
            f"Tokenizer/checkpoint vocab mismatch for {component}: tokenizer={int(tok_vocab)}, "
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

    # Default heuristic:
    # - hf_deberta_v2 parity path uses DeBERTa's half-depth generator.
    # - other backbones keep the lighter ELECTRA-style 1/3 depth.
    if getattr(gen_cfg, "num_hidden_layers", None) is not None:
        disc_layers = int(gen_cfg.num_hidden_layers)
        if str(model_cfg.backbone_type).strip().lower() == "hf_deberta_v2":
            default_gen_layers = max(1, disc_layers // 2)
        else:
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


_PRETRAINED_OVERRIDE_MAP: tuple[tuple[str, str, type], ...] = (
    ("pretrained_max_position_embeddings", "max_position_embeddings", int),
    ("pretrained_rope_theta", "rope_theta", float),
    ("pretrained_rotary_pct", "rotary_pct", float),
    ("pretrained_use_absolute_position_embeddings", "use_absolute_position_embeddings", bool),
    ("pretrained_type_vocab_size", "type_vocab_size", int),
    ("pretrained_norm_arch", "norm_arch", str),
    ("pretrained_norm_eps", "norm_eps", float),
    ("pretrained_keel_alpha_init", "keel_alpha_init", float),
    ("pretrained_keel_alpha_learnable", "keel_alpha_learnable", bool),
    ("pretrained_ffn_type", "ffn_type", str),
    ("pretrained_use_bias", "use_bias", bool),
    ("pretrained_initializer_range", "initializer_range", float),
)


def _apply_rope_pretrained_explicit_overrides(cfg: Any, model_cfg: ModelConfig) -> None:
    """Apply explicit pretrained RoPE overrides.

    :param Any cfg: Target config object.
    :param ModelConfig model_cfg: User model configuration.
    """
    for src_attr, dst_attr, cast in _PRETRAINED_OVERRIDE_MAP:
        val = getattr(model_cfg, src_attr, None)
        if val is not None:
            setattr(cfg, dst_attr, cast(val))


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
        model_cfg=model_cfg,
    )
    if model_cfg.hf_max_position_embeddings is not None:
        cfg.max_position_embeddings = int(model_cfg.hf_max_position_embeddings)
    _apply_dropout_overrides(cfg, model_cfg)
    cfg.hf_attention_kernel = str(model_cfg.hf_attention_kernel)
    cfg.use_rmsnorm_heads = False


def _build_repo_hf_deberta_v2_config(*, model_cfg: ModelConfig) -> DebertaV2Config:
    """Build a repo-owned HF-compatible DeBERTa-v2/v3 architecture config.

    :param ModelConfig model_cfg: User model configuration.
    :return DebertaV2Config: Fresh config object populated from repo numeric presets.
    """
    presets: dict[str, dict[str, int]] = {
        "xsmall": {
            "hidden_size": 384,
            "num_hidden_layers": 12,
            "num_attention_heads": 6,
            "intermediate_size": 1536,
        },
        "small": {
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
        },
        "base": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
        },
        "large": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
        },
    }
    size_key = str(model_cfg.hf_model_size).strip().lower()
    dims = presets[size_key]

    return DebertaV2Config(
        vocab_size=128_100,
        hidden_size=int(dims["hidden_size"]),
        num_hidden_layers=int(dims["num_hidden_layers"]),
        num_attention_heads=int(dims["num_attention_heads"]),
        intermediate_size=int(dims["intermediate_size"]),
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        legacy=True,
        pooler_hidden_size=int(dims["hidden_size"]),
        pooler_hidden_act="gelu",
        pooler_dropout=0.0,
        z_steps=0,
        max_position_embeddings=512,
        max_relative_positions=-1,
        position_buckets=256,
        relative_attention=True,
        position_biased_input=False,
        share_att_key=True,
        norm_rel_ebd="layer_norm",
        pos_att_type=["p2c", "c2p"],
        type_vocab_size=0,
    )


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
        model_cfg=model_cfg,
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

    - For backbone_type='hf_deberta_v2': returns DebertaV2Config instances from repo defaults.
    - For backbone_type='rope': returns DebertaRoPEConfig instances.

    Generator config is loaded if specified, otherwise derived from discriminator config.

    :param ModelConfig model_cfg: User model configuration.
    :param Any tokenizer: Tokenizer used for vocab/pad metadata.
    :param int max_position_embeddings: Sequence length budget.
    :return tuple[Any, Any]: Discriminator and generator configs.
    """
    validate_model_config(model_cfg)
    bt = (model_cfg.backbone_type or "hf_deberta_v2").lower()
    resolved = _resolve_backbone_sources(model_cfg)

    if bt == "hf_deberta_v2":
        disc_cfg = _build_repo_hf_deberta_v2_config(model_cfg=model_cfg)
        if model_cfg.pretrained_generator_path:
            # Explicit generator model sources affect weight loading, not config synthesis.
            gen_cfg = copy.deepcopy(disc_cfg)
        else:
            gen_cfg = _derive_generator_config(disc_cfg, model_cfg)

        # Match released DeBERTa-v3 xsmall generator behavior.
        if str(model_cfg.hf_model_size).strip().lower() == "xsmall":
            gen_cfg.z_steps = 2
        else:
            gen_cfg.z_steps = 0

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
    load_pretrained_weights: bool = True,
) -> tuple[Any, Any]:
    """Instantiate discriminator + generator backbones.

    :param ModelConfig model_cfg: User model configuration.
    :param Any disc_config: Discriminator config object.
    :param Any gen_config: Generator config object.
    :param bool load_pretrained_weights: Whether to load pretrained weights from resolved model sources
        when ``model.from_scratch=false``. Set ``False`` for resume/export flows that will load from an
        accelerate checkpoint immediately after instantiation.
    :return tuple[Any, Any]: Instantiated discriminator and generator modules.
    """
    validate_model_config(model_cfg)
    bt = (model_cfg.backbone_type or "hf_deberta_v2").lower()
    resolved = _resolve_backbone_sources(model_cfg)

    if bt == "hf_deberta_v2":
        if model_cfg.from_scratch or not bool(load_pretrained_weights):
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
    if model_cfg.from_scratch or not bool(load_pretrained_weights):
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
