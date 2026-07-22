"""Dataclass-based model, data, and training configuration definitions."""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Callable
from dataclasses import InitVar, asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

from deberta.utils.io import load_json_mapping
from deberta.utils.serialize import asdict_without_private as _asdict_without_private
from deberta.utils.types import coerce_scalar, unwrap_optional_type

_BACKBONE_CHOICES = {"rope", "hf_deberta_v2"}
_NORM_ARCH_CHOICES = {"post", "keel"}
_ATTN_IMPL_CHOICES = {"sdpa", "eager"}
_HF_ATTN_IMPL_CHOICES = {"eager", "flash"}
_FFN_CHOICES = {"swiglu", "mlp"}
_EMBED_SHARING_CHOICES = {"none", "es", "gdes"}
_WANDB_WATCH_CHOICES = {"none", "gradients", "parameters", "all"}
_WANDB_WATCH_ALIASES = {
    "off": "none",
    "disabled": "none",
    "false": "none",
    "0": "none",
    "grad": "gradients",
    "gradient": "gradients",
    "weights": "parameters",
    "params": "parameters",
    "param": "parameters",
}
_LR_SCHEDULER_CHOICES = {
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
}
_RESUME_DATA_STRATEGY_CHOICES = {"auto", "replay", "restart_epoch"}
_SDPA_KERNEL_CHOICES = {"auto", "flash", "mem_efficient", "math"}
_SDPA_KERNEL_ALIASES = {
    "mem": "mem_efficient",
    "mem-efficient": "mem_efficient",
    "efficient": "mem_efficient",
    "flashattention": "flash",
    "flash_attention": "flash",
}
_TORCH_COMPILE_MODE_CHOICES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}
_TORCH_COMPILE_MODE_ALIASES = {
    "reduce_overhead": "reduce-overhead",
    "max_autotune": "max-autotune",
    "max_autotune_no_cudagraphs": "max-autotune-no-cudagraphs",
}
_TORCH_COMPILE_SCOPE_CHOICES = {
    "auto",
    "backbones",
    "encoder",
    "gen_encoder",
    "disc_encoder",
    "ffn",
    "gen_ffn",
    "disc_ffn",
}
_TORCH_COMPILE_SCOPE_ALIASES = {
    "both": "backbones",
    "backbone": "backbones",
    "full": "backbones",
    "encoder": "encoder",
    "encoders": "encoder",
    "gen_encoder": "gen_encoder",
    "generator_encoder": "gen_encoder",
    "disc_encoder": "disc_encoder",
    "discriminator_encoder": "disc_encoder",
    "ffn": "ffn",
    "ffns": "ffn",
    "gen_ffn": "gen_ffn",
    "generator_ffn": "gen_ffn",
    "disc_ffn": "disc_ffn",
    "discriminator_ffn": "disc_ffn",
}

# The literal dataclass defaults below describe the default hf_deberta_v2
# profile. Config-file omissions still need to resolve against the selected
# backbone without overwriting values explicitly supplied in YAML/JSON or by a
# programmatic dotted override.
_BACKBONE_PROFILE_DEFAULTS: dict[str, dict[str, float | int]] = {
    "hf_deberta_v2": {
        "train.objective.mask_token_prob": 1.0,
        "train.objective.random_token_prob": 0.0,
        "train.objective.disc_loss_weight": 10.0,
        "optim.lr.base": 1e-4,
        "optim.adam.epsilon": 1e-6,
    },
    "rope": {
        "train.objective.mask_token_prob": 0.8,
        "train.objective.random_token_prob": 0.1,
        "train.objective.disc_loss_weight": 50.0,
        "optim.lr.base": 5e-4,
        "optim.adam.epsilon": 1e-8,
    },
}
_TORCH_COMPILE_BACKEND_CHOICES = {"inductor", "aot_eager"}
_TORCH_COMPILE_BACKEND_ALIASES = {
    "aot-eager": "aot_eager",
}
_HF_ATTN_KERNEL_CHOICES = {"dynamic", "cached_bmm", "stable"}
_HF_MODEL_SIZE_CHOICES = {"xsmall", "small", "base", "large"}
_HF_ATTN_KERNEL_ALIASES = {
    "default": "dynamic",
    "cache": "cached_bmm",
    "cached": "cached_bmm",
    "safe": "stable",
    "compile_safe": "stable",
}
_MIXED_PRECISION_CANONICAL = {"bf16", "no"}
_MIXED_PRECISION_ALIASES = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "true": "bf16",
    "1": "bf16",
    "yes": "bf16",
    "y": "bf16",
    "on": "bf16",
    "no": "no",
    "none": "no",
    "false": "no",
    "0": "no",
    "off": "no",
    "n": "no",
}
_HF_DEBERTA_PRETRAINED_PREFIXES = (
    "microsoft/deberta-v2",
    "microsoft/deberta-v3",
    "microsoft/mdeberta-v3",
)
_DENSE_DOC_BLOCK_WARN_SEQ_LEN = 2048
# Pre-stable policy: persisted run schemas may change when needed for correctness/simplicity.
# Backward checkpoint/resume compatibility is intentionally not guaranteed until a stable release.
RUN_CONFIG_SCHEMA_VERSION = 8


@dataclass(frozen=True)
class ModelTokenizerConfig:
    """Tokenizer sub-configuration."""

    name_or_path: str = field(default="microsoft/deberta-v3-base")
    allow_vocab_resize: bool = field(default=False)
    vocab_target: int | None = field(default=None)
    vocab_multiple: int = field(default=1)


@dataclass(frozen=True)
class ModelHFFlashConfig:
    """FlashDeBERTa runtime policy for native HF DeBERTa-v2/v3 attention.

    ``docblock_bias_seq_len`` forces the dense doc-block route at exactly that
    sequence length on any GPU, bypassing the tuning table's
    ``max_batch_size``/``max_seq_len`` safety bounds entirely - the dense
    route saves a ``(B,H,S,S)`` bias for backward, so a forgotten knob plus a
    larger batch can OOM.
    """

    docblock_bias_seq_len: int | None = field(default=None)
    kernel_overrides_path: str | None = field(default=None)


@dataclass(frozen=True)
class ModelHFConfig:
    """HF DeBERTa-v2/v3 backbone synthesis options."""

    model_size: str = field(default="base")
    attention_kernel: str = field(default="dynamic")
    attention_impl: str = field(default="eager")
    flash: ModelHFFlashConfig = field(default_factory=ModelHFFlashConfig)
    max_position_embeddings: int | None = field(default=None)


@dataclass(frozen=True)
class ModelPretrainedConfig:
    """Pretrained source paths for discriminator/generator."""

    discriminator_path: str = field(default="")
    generator_path: str | None = field(default=None)


@dataclass(frozen=True)
class ModelGeneratorConfig:
    """Generator shape overrides when deriving generator configs."""

    num_hidden_layers: int | None = field(default=None)
    hidden_size: int | None = field(default=None)
    intermediate_size: int | None = field(default=None)
    num_attention_heads: int | None = field(default=None)


@dataclass(frozen=True)
class ModelRopePretrainedConfig:
    """Pretrained-only overrides for rope checkpoints."""

    max_position_embeddings: int | None = field(default=None)
    rope_theta: float | None = field(default=None)
    rotary_pct: float | None = field(default=None)
    use_absolute_position_embeddings: bool | None = field(default=None)
    type_vocab_size: int | None = field(default=None)
    norm_arch: str | None = field(default=None)
    norm_eps: float | None = field(default=None)
    keel_alpha_init: float | None = field(default=None)
    keel_alpha_learnable: bool | None = field(default=None)
    ffn_type: str | None = field(default=None)
    use_bias: bool | None = field(default=None)
    initializer_range: float | None = field(default=None)


@dataclass(frozen=True)
class ModelRopeConfig:
    """RoPE backbone architecture options."""

    hidden_size: int = field(default=768)
    num_hidden_layers: int = field(default=12)
    num_attention_heads: int = field(default=12)
    intermediate_size: int = field(default=3072)
    hidden_act: str = field(default="gelu")
    rope_theta: float = field(default=10000.0)
    rotary_pct: float = field(default=1.0)
    use_absolute_position_embeddings: bool = field(default=False)
    max_position_embeddings: int | None = field(default=None)
    type_vocab_size: int = field(default=2)
    norm_arch: str = field(default="post")
    norm_eps: float = field(default=1e-6)
    keel_alpha_init: float | None = field(default=None)
    keel_alpha_learnable: bool = field(default=False)
    attention_implementation: str = field(default="sdpa")
    ffn_type: str = field(default="mlp")
    use_bias: bool = field(default=False)
    swiglu_adjust_intermediate: bool = field(default=True)
    initializer_range: float = field(default=0.02)
    pretrained: ModelRopePretrainedConfig = field(default_factory=ModelRopePretrainedConfig)


@dataclass(frozen=True)
class ModelDropoutConfig:
    """Dropout overrides shared across backbone families."""

    hidden_prob: float | None = field(default=0.0)
    attention_probs_prob: float | None = field(default=0.0)


def _nested_get(obj: Any, path: str) -> Any:
    """Resolve dotted nested attributes.

    :param Any obj: Source object.
    :param str path: Dotted attribute path.
    :return Any: Resolved value.
    """
    cur = obj
    for part in str(path).split("."):
        cur = getattr(cur, part)
    return cur


def _replace_path(obj: Any, parts: list[str], value: Any) -> Any:
    """Replace one dotted path on a dataclass object.

    :param Any obj: Source dataclass instance.
    :param list[str] parts: Dotted path parts.
    :param Any value: Replacement value.
    :return Any: Updated dataclass object.
    """
    key = str(parts[0])
    if not hasattr(obj, key):
        raise TypeError(f"Unknown field {key!r} for {type(obj).__name__}.")
    if len(parts) == 1:
        return replace(obj, **{key: value})
    child = getattr(obj, key)
    new_child = _replace_path(child, parts[1:], value)
    return replace(obj, **{key: new_child})


@dataclass(frozen=True)
class ModelConfig:
    """Model-related arguments."""

    backbone_type: str = field(default="hf_deberta_v2")
    from_scratch: bool = field(default=True)
    embedding_sharing: str = field(default="gdes")
    gradient_checkpointing: bool = field(default=False)
    tokenizer: ModelTokenizerConfig = field(default_factory=ModelTokenizerConfig)
    hf: ModelHFConfig = field(default_factory=ModelHFConfig)
    pretrained: ModelPretrainedConfig = field(default_factory=ModelPretrainedConfig)
    generator: ModelGeneratorConfig = field(default_factory=ModelGeneratorConfig)
    rope: ModelRopeConfig = field(default_factory=ModelRopeConfig)
    dropout: ModelDropoutConfig = field(default_factory=ModelDropoutConfig)


@dataclass(frozen=True)
class DataSourceConfig:
    """Dataset source configuration."""

    dataset_name: str | None = field(default=None)
    dataset_config_name: str | None = field(default=None)
    data_files: str | None = field(default=None)
    load_from_disk: str | None = field(default=None)
    train_split: str = field(default="train")
    text_column_name: str = field(default="text")
    streaming: bool = field(default=True)
    shuffle_buffer_size: int = field(default=10_000)


@dataclass(frozen=True)
class DataPackingConfig:
    """Packing and sequence-shape configuration."""

    enabled: bool = field(default=True)
    max_seq_length: int = field(default=512)
    block_cross_document_attention: bool = field(default=False)


@dataclass(frozen=True)
class DataConfig:
    """Data-related arguments."""

    source: DataSourceConfig = field(default_factory=DataSourceConfig)
    packing: DataPackingConfig = field(default_factory=DataPackingConfig)


@dataclass(frozen=True)
class TrainDataloaderConfig:
    """DataLoader settings."""

    num_workers: int = field(default=2)
    pin_memory: bool = field(default=True)


@dataclass(frozen=True)
class TrainCompileConfig:
    """torch.compile runtime controls."""

    enabled: bool = field(default=False)
    mode: str = field(default="default")
    scope: str = field(default="auto")
    backend: str = field(default="inductor")


@dataclass(frozen=True)
class TrainObjectiveConfig:
    """RTD/MLM objective controls."""

    mlm_probability: float = field(default=0.15)
    mask_token_prob: float = field(default=1.0)
    random_token_prob: float = field(default=0.0)
    mlm_max_ngram: int = field(default=1)
    sampling_temperature: float = field(default=1.0)
    gen_loss_weight: float = field(default=1.0)
    disc_loss_weight: float = field(default=10.0)


@dataclass(frozen=True)
class TrainCheckpointConfig:
    """Checkpoint and resume settings."""

    output_dir: str | None = field(default=None)
    overwrite_output_dir: bool = field(default=False)
    save_steps: int = field(default=1_000)
    save_total_limit: int = field(default=3)
    resume_from_checkpoint: str | None = field(default=None)
    resume_data_strategy: str = field(default="auto")
    resume_replay_max_micro_batches: int = field(default=10_000)
    export_hf_final: bool = field(default=True)


@dataclass(frozen=True)
class TrainConfig:
    """Training-related arguments."""

    seed: int = field(default=42)
    max_steps: int = field(default=10_000)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    token_weighted_gradient_accumulation: bool = field(default=True)
    mixed_precision: str = field(default="bf16")
    tf32: bool = field(default=True)
    sdpa_kernel: str = field(default="auto")
    decoupled_training: bool = field(default=True)
    dataloader: TrainDataloaderConfig = field(default_factory=TrainDataloaderConfig)
    compile: TrainCompileConfig = field(default_factory=TrainCompileConfig)
    objective: TrainObjectiveConfig = field(default_factory=TrainObjectiveConfig)
    checkpoint: TrainCheckpointConfig = field(default_factory=TrainCheckpointConfig)


@dataclass(frozen=True)
class OptimLRConfig:
    """Learning-rate values for optimizer setup."""

    base: float = field(default=1e-4)
    generator: float = field(default=-1.0)
    discriminator: float = field(default=-1.0)


@dataclass(frozen=True)
class OptimAdamConfig:
    """Adam/AdamW parameter group settings."""

    beta1: float = field(default=0.9)
    beta2: float = field(default=0.999)
    epsilon: float = field(default=1e-6)


@dataclass(frozen=True)
class OptimSchedulerConfig:
    """Scheduler controls."""

    type: str = field(default="linear")
    warmup_steps: int = field(default=1_000)


@dataclass(frozen=True)
class OptimConfig:
    """Optimization section configuration."""

    lr: OptimLRConfig = field(default_factory=OptimLRConfig)
    adam: OptimAdamConfig = field(default_factory=OptimAdamConfig)
    scheduler: OptimSchedulerConfig = field(default_factory=OptimSchedulerConfig)
    weight_decay: float = field(default=0.01)
    max_grad_norm: float = field(default=1.0)


@dataclass(frozen=True)
class LoggingWandbConfig:
    """W&B logging controls."""

    enabled: bool = field(default=False)
    watch: str = field(default="gradients")
    watch_log_freq: int = field(default=100)


@dataclass(frozen=True)
class LoggingDebugConfig:
    """Debug metrics logging controls."""

    metrics: bool = field(default=False)


@dataclass(frozen=True)
class LoggingConfig:
    """Logging/tracker section."""

    project_name: str = field(default="deberta-train")
    run_name: str | None = field(default=None)
    output_dir: str | None = field(default=None)
    logging_steps: int = field(default=50)
    wandb: LoggingWandbConfig = field(default_factory=LoggingWandbConfig)
    debug: LoggingDebugConfig = field(default_factory=LoggingDebugConfig)


@dataclass(frozen=True, init=False)
class Config:
    """Top-level training config bundle."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    _explicit_fields: InitVar[frozenset[str] | None] = None

    def __init__(
        self,
        model: ModelConfig | None = None,
        data: DataConfig | None = None,
        train: TrainConfig | None = None,
        optim: OptimConfig | None = None,
        logging: LoggingConfig | None = None,
        _explicit_fields: frozenset[str] | None = None,
    ) -> None:
        """Initialize a config bundle while preserving supplied section values.

        A programmatically supplied ``train`` or ``optim`` section is explicit,
        including values equal to the literal schema defaults. Omitted sections
        remain eligible for backbone-profile defaults. File loaders pass exact
        dotted-field provenance through ``_explicit_fields``.

        :param ModelConfig | None model: Model configuration, defaults to the schema value.
        :param DataConfig | None data: Data configuration, defaults to the schema value.
        :param TrainConfig | None train: Explicit training configuration, defaults to omission.
        :param OptimConfig | None optim: Explicit optimizer configuration, defaults to omission.
        :param LoggingConfig | None logging: Logging configuration, defaults to the schema value.
        :param frozenset[str] | None _explicit_fields: Exact loader-owned dotted-field provenance.
        """
        object.__setattr__(self, "model", model if model is not None else ModelConfig())
        object.__setattr__(self, "data", data if data is not None else DataConfig())
        object.__setattr__(self, "train", train if train is not None else TrainConfig())
        object.__setattr__(self, "optim", optim if optim is not None else OptimConfig())
        object.__setattr__(self, "logging", logging if logging is not None else LoggingConfig())

        explicit = set(_explicit_fields or ())
        if _explicit_fields is None:
            schema_profile = _BACKBONE_PROFILE_DEFAULTS["hf_deberta_v2"]
            if train is not None:
                explicit.update(path for path in schema_profile if path.startswith("train."))
            if optim is not None:
                explicit.update(path for path in schema_profile if path.startswith("optim."))
        self.__post_init__(frozenset(explicit))

    def __post_init__(self, _explicit_fields: frozenset[str] | None) -> None:
        """Resolve omitted fields against the selected backbone profile.

        :param frozenset[str] | None _explicit_fields: Dotted paths treated as explicit.
        """
        explicit = frozenset(_explicit_fields or ())
        object.__setattr__(self, "_explicit_field_paths", explicit)

        profile = _BACKBONE_PROFILE_DEFAULTS.get(str(self.model.backbone_type).strip().lower())
        if profile is None:
            return

        objective_updates = {
            path.rsplit(".", 1)[-1]: value
            for path, value in profile.items()
            if path.startswith("train.objective.") and path not in explicit
        }
        adam_updates = {
            path.rsplit(".", 1)[-1]: value
            for path, value in profile.items()
            if path.startswith("optim.adam.") and path not in explicit
        }
        lr_updates = {
            path.rsplit(".", 1)[-1]: value
            for path, value in profile.items()
            if path.startswith("optim.lr.") and path not in explicit
        }
        scheduler_updates = {
            path.rsplit(".", 1)[-1]: value
            for path, value in profile.items()
            if path.startswith("optim.scheduler.") and path not in explicit
        }

        object.__setattr__(
            self,
            "train",
            replace(
                self.train,
                objective=replace(self.train.objective, **objective_updates),
            ),
        )
        object.__setattr__(
            self,
            "optim",
            replace(
                self.optim,
                lr=replace(self.optim.lr, **lr_updates),
                adam=replace(self.optim.adam, **adam_updates),
                scheduler=replace(self.optim.scheduler, **scheduler_updates),
            ),
        )


def _explicit_config_fields(cfg: Config) -> frozenset[str]:
    """Return non-serialized explicit-field provenance for a config bundle.

    :param Config cfg: Config bundle.
    :return frozenset[str]: Explicit dotted field paths.
    """
    return frozenset(getattr(cfg, "_explicit_field_paths", frozenset()))


def _ensure_choice(name: str, value: str, choices: set[str]) -> str:
    """Normalize and validate a string option against allowed choices.

    :param str name: Option name.
    :param str value: Raw option value.
    :param set[str] choices: Allowed lower-case values.
    :return str: Canonical lower-case value.
    """
    v = str(value).strip().lower()
    if v not in choices:
        allowed = "|".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {allowed}. Got: {value}")
    return v


def _normalize_choice_with_aliases(
    *,
    name: str,
    value: str,
    aliases: dict[str, str],
    choices: set[str],
    replace_hyphen: bool = False,
) -> str:
    """Normalize a string option with alias mapping and choice validation.

    :param str name: Option name.
    :param str value: Raw option value.
    :param dict[str, str] aliases: Alias-to-canonical mapping.
    :param set[str] choices: Allowed canonical values.
    :param bool replace_hyphen: Whether to normalize '-' to '_' before alias lookup.
    :return str: Canonical normalized value.
    """
    v = str(value).strip().lower()
    if bool(replace_hyphen):
        v = v.replace("-", "_")
    v = aliases.get(v, v)
    return _ensure_choice(name, v, choices)


def _normalize_sdpa_kernel(value: str) -> str:
    """Normalize and validate SDPA kernel policy values.

    :param str value: Raw SDPA kernel value.
    :return str: Canonical lower-case SDPA kernel policy.
    """
    return _normalize_choice_with_aliases(
        name="train.sdpa_kernel",
        value=value,
        aliases=_SDPA_KERNEL_ALIASES,
        choices=_SDPA_KERNEL_CHOICES,
        replace_hyphen=False,
    )


def _normalize_torch_compile_mode(value: str) -> str:
    """Normalize and validate torch.compile mode values.

    :param str value: Raw compile mode value.
    :return str: Canonical compile mode.
    """
    return _normalize_choice_with_aliases(
        name="train.compile.mode",
        value=value,
        aliases=_TORCH_COMPILE_MODE_ALIASES,
        choices=_TORCH_COMPILE_MODE_CHOICES,
        replace_hyphen=False,
    )


def _normalize_torch_compile_scope(value: str) -> str:
    """Normalize and validate torch.compile scope values.

    :param str value: Raw compile scope value.
    :return str: Canonical compile scope.
    """
    return _normalize_choice_with_aliases(
        name="train.compile.scope",
        value=value,
        aliases=_TORCH_COMPILE_SCOPE_ALIASES,
        choices=_TORCH_COMPILE_SCOPE_CHOICES,
        replace_hyphen=True,
    )


def _normalize_torch_compile_backend(value: str) -> str:
    """Normalize and validate torch.compile backend values.

    :param str value: Raw compile backend value.
    :return str: Canonical compile backend.
    """
    return _normalize_choice_with_aliases(
        name="train.compile.backend",
        value=value,
        aliases=_TORCH_COMPILE_BACKEND_ALIASES,
        choices=_TORCH_COMPILE_BACKEND_CHOICES,
        replace_hyphen=True,
    )


def _normalize_wandb_watch(value: str) -> str:
    """Normalize and validate W&B watch mode values.

    :param str value: Raw W&B watch mode.
    :return str: Canonical W&B watch mode.
    """
    return _normalize_choice_with_aliases(
        name="logging.wandb.watch",
        value=value,
        aliases=_WANDB_WATCH_ALIASES,
        choices=_WANDB_WATCH_CHOICES,
        replace_hyphen=True,
    )


def _normalize_hf_attention_kernel(value: str) -> str:
    """Normalize and validate native hf_deberta_v2 attention-kernel values.

    :param str value: Raw attention-kernel value.
    :return str: Canonical attention-kernel name.
    """
    return _normalize_choice_with_aliases(
        name="model.hf.attention_kernel",
        value=value,
        aliases=_HF_ATTN_KERNEL_ALIASES,
        choices=_HF_ATTN_KERNEL_CHOICES,
        replace_hyphen=True,
    )


def _normalize_hf_attention_impl(value: str) -> str:
    """Normalize and validate native hf_deberta_v2 attention implementation values.

    :param str value: Raw attention implementation value.
    :return str: Canonical attention implementation name.
    """

    return _ensure_choice("model.hf.attention_impl", value, _HF_ATTN_IMPL_CHOICES)


def normalize_mixed_precision(value: object) -> str:
    """Normalize and validate mixed precision values.

    :param object value: Raw mixed precision value.
    :return str: Canonical mixed precision mode ('bf16' or 'no').
    """
    if isinstance(value, bool):
        mp = "bf16" if bool(value) else "no"
    else:
        mp = str(value).strip().lower()
        mp = _MIXED_PRECISION_ALIASES.get(mp, mp)

    if mp not in _MIXED_PRECISION_CANONICAL:
        raise ValueError(
            "train.mixed_precision must be one of: bf16|no (or aliases true|false|yes|no|on|off|1|0)."
        )
    return mp


def resolve_effective_mixed_precision(
    value: object,
    *,
    bf16_sanity_check: Callable[[], bool] | None = None,
) -> str:
    """Normalize mixed precision and optionally enforce bf16 runtime sanity.

    :param object value: Raw mixed-precision value/alias.
    :param Callable[[], bool] | None bf16_sanity_check: Optional bf16 runtime probe callback.
    :raises RuntimeError: If ``bf16`` is requested and the sanity check fails.
    :return str: Effective mixed precision mode.
    """
    mixed_precision = normalize_mixed_precision(value)
    if mixed_precision == "bf16" and bf16_sanity_check is not None and not bf16_sanity_check():
        raise RuntimeError(
            "train.mixed_precision=bf16 requested but bf16 preflight failed. "
            "Set train.mixed_precision=no explicitly if you want to continue in full precision."
        )
    return mixed_precision


def _cfg_set(cfg_obj: Any, field_name: str, value: Any) -> None:
    """Set a dataclass field via object.__setattr__.

    :param Any cfg_obj: Dataclass instance.
    :param str field_name: Field name.
    :param Any value: New field value.
    """
    object.__setattr__(cfg_obj, str(field_name), value)


def _looks_like_hf_deberta_checkpoint(value: str) -> bool:
    """Return whether a model source appears to be an HF DeBERTa v2/v3 checkpoint id.

    :param str value: Model source string.
    :return bool: True when source matches known HF DeBERTa hub-id prefixes.
    """
    raw = str(value).strip().lower()
    if not raw:
        return False

    v = raw.replace("\\", "/")
    if v.startswith("hf://"):
        v = v[len("hf://") :]
    if v.startswith("https://huggingface.co/") or v.startswith("http://huggingface.co/"):
        v = v.split("huggingface.co/", 1)[1]
    v = v.lstrip("/")

    def _matches_repo_id(candidate: str) -> bool:
        """Return whether candidate is a DeBERTa repo id or repo-scoped path.

        :param str candidate: Candidate repository id or scoped path.
        :return bool: True when candidate matches known DeBERTa HF prefixes.
        """
        return any(
            candidate == prefix or candidate.startswith(f"{prefix}-") or candidate.startswith(f"{prefix}/")
            for prefix in _HF_DEBERTA_PRETRAINED_PREFIXES
        )

    if _matches_repo_id(v):
        return True

    # Common local Hugging Face cache layout:
    # .../models--microsoft--deberta-v3-base/snapshots/<rev>/...
    cache_path = f"/{v}/"
    return any(
        marker in cache_path
        for marker in (
            "/models--microsoft--deberta-v2",
            "/models--microsoft--deberta-v3",
            "/models--microsoft--mdeberta-v3",
        )
    )


def _apply_backbone_option_severity_policy(cfg: ModelConfig, defaults: ModelConfig) -> None:
    """Apply the explicit error/warning policy for inactive backbone options.

    :param ModelConfig cfg: Model configuration under validation.
    :param ModelConfig defaults: Default model configuration for change detection.
    :raises ValueError: If an inactive option group has error severity.
    :return None: None.
    """

    if cfg.backbone_type == "hf_deberta_v2":
        rules = [
            (
                asdict(cfg.rope) != asdict(defaults.rope),
                "error",
                "These options are only valid when model.backbone_type='rope': model.rope.*",
            )
        ]
    else:
        rules = [
            (
                cfg.hf.attention_kernel != defaults.hf.attention_kernel,
                "warning",
                "model.hf.attention_kernel only applies when model.backbone_type='hf_deberta_v2'. "
                f"Current value ({cfg.hf.attention_kernel!r}) has no effect on the rope backbone.",
            ),
            (
                cfg.hf.max_position_embeddings is not None,
                "warning",
                "model.hf.max_position_embeddings only applies when model.backbone_type='hf_deberta_v2'. "
                f"Current value ({cfg.hf.max_position_embeddings!r}) has no effect on the rope backbone.",
            ),
            (
                cfg.hf.model_size != defaults.hf.model_size,
                "warning",
                "model.hf.model_size only applies when model.backbone_type='hf_deberta_v2'. "
                f"Current value ({cfg.hf.model_size!r}) has no effect on the rope backbone.",
            ),
        ]
    rules.append(
        (
            cfg.hf.attention_impl != "flash" and asdict(cfg.hf.flash) != asdict(defaults.hf.flash),
            "warning",
            "model.hf.flash.* has no effect unless model.hf.attention_impl='flash'.",
        )
    )

    for active, severity, message in rules:
        if not active:
            continue
        if severity == "error":
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=3)


def validate_model_config(cfg: ModelConfig) -> None:
    """Validate model config semantics and normalize constrained values.

    :param ModelConfig cfg: Model configuration.
    """
    _cfg_set(
        cfg, "backbone_type", _ensure_choice("model.backbone_type", cfg.backbone_type, _BACKBONE_CHOICES)
    )
    _cfg_set(
        cfg,
        "embedding_sharing",
        _ensure_choice("model.embedding_sharing", cfg.embedding_sharing, _EMBED_SHARING_CHOICES),
    )

    _cfg_set(cfg.hf, "attention_kernel", _normalize_hf_attention_kernel(cfg.hf.attention_kernel))
    _cfg_set(cfg.hf, "attention_impl", _normalize_hf_attention_impl(cfg.hf.attention_impl))
    _cfg_set(
        cfg.hf, "model_size", _ensure_choice("model.hf.model_size", cfg.hf.model_size, _HF_MODEL_SIZE_CHOICES)
    )
    if cfg.hf.flash.docblock_bias_seq_len is not None:
        _cfg_set(cfg.hf.flash, "docblock_bias_seq_len", int(cfg.hf.flash.docblock_bias_seq_len))
    if cfg.hf.flash.kernel_overrides_path is not None:
        _cfg_set(
            cfg.hf.flash, "kernel_overrides_path", str(cfg.hf.flash.kernel_overrides_path).strip() or None
        )
    if cfg.hf.attention_impl == "flash" and cfg.hf.flash.kernel_overrides_path is not None:
        from deberta.modeling.flashdeberta_kernel_tuning import (
            validate_flashdeberta_kernel_overrides,
        )

        try:
            validate_flashdeberta_kernel_overrides(cfg.hf.flash.kernel_overrides_path)
        except ValueError as exc:
            raise ValueError(f"Invalid model.hf.flash.kernel_overrides_path: {exc}") from exc

    _cfg_set(
        cfg.rope, "norm_arch", _ensure_choice("model.rope.norm_arch", cfg.rope.norm_arch, _NORM_ARCH_CHOICES)
    )
    _cfg_set(
        cfg.rope,
        "attention_implementation",
        _ensure_choice(
            "model.rope.attention_implementation",
            cfg.rope.attention_implementation,
            _ATTN_IMPL_CHOICES,
        ),
    )
    _cfg_set(cfg.rope, "ffn_type", _ensure_choice("model.rope.ffn_type", cfg.rope.ffn_type, _FFN_CHOICES))
    hidden_act = str(cfg.rope.hidden_act).strip().lower()
    try:
        from transformers.activations import ACT2FN

        if hidden_act not in ACT2FN:
            allowed = "|".join(sorted(ACT2FN))
            raise ValueError(f"model.rope.hidden_act must be one of: {allowed}. Got: {cfg.rope.hidden_act}")
    except ImportError:  # pragma: no cover - transformers is a required runtime dependency
        pass
    _cfg_set(cfg.rope, "hidden_act", hidden_act)

    _cfg_set(cfg.tokenizer, "name_or_path", str(cfg.tokenizer.name_or_path).strip())
    _cfg_set(cfg.pretrained, "discriminator_path", str(cfg.pretrained.discriminator_path or "").strip())
    if cfg.pretrained.generator_path is not None:
        _cfg_set(cfg.pretrained, "generator_path", str(cfg.pretrained.generator_path).strip() or None)

    if not cfg.tokenizer.name_or_path:
        raise ValueError("model.tokenizer.name_or_path must be a non-empty tokenizer source.")
    if not bool(cfg.from_scratch) and not cfg.pretrained.discriminator_path:
        raise ValueError(
            "model.pretrained.discriminator_path must be set when model.from_scratch=false "
            "(weights are loaded from this source)."
        )

    if cfg.rope.max_position_embeddings is not None and int(cfg.rope.max_position_embeddings) <= 0:
        raise ValueError("model.rope.max_position_embeddings must be > 0 when provided.")
    if not math.isfinite(float(cfg.rope.rope_theta)) or float(cfg.rope.rope_theta) <= 0.0:
        raise ValueError("model.rope.rope_theta must be finite and > 0.")
    if (
        not math.isfinite(float(cfg.rope.rotary_pct))
        or float(cfg.rope.rotary_pct) <= 0.0
        or float(cfg.rope.rotary_pct) > 1.0
    ):
        raise ValueError("model.rope.rotary_pct must be finite and in (0, 1].")
    if int(cfg.rope.type_vocab_size) < 0:
        raise ValueError("model.rope.type_vocab_size must be >= 0.")
    if not math.isfinite(float(cfg.rope.norm_eps)) or float(cfg.rope.norm_eps) <= 0.0:
        raise ValueError("model.rope.norm_eps must be finite and > 0.")
    if not math.isfinite(float(cfg.rope.initializer_range)) or float(cfg.rope.initializer_range) < 0.0:
        raise ValueError("model.rope.initializer_range must be finite and >= 0.")
    if cfg.rope.keel_alpha_init is not None and not math.isfinite(float(cfg.rope.keel_alpha_init)):
        raise ValueError("model.rope.keel_alpha_init must be finite when provided.")
    for field_name in (
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
    ):
        value = getattr(cfg.generator, field_name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"model.generator.{field_name} must be > 0 when provided.")
    for field_name in ("hidden_prob", "attention_probs_prob"):
        value = getattr(cfg.dropout, field_name)
        if value is None:
            continue
        probability = float(value)
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError(f"model.dropout.{field_name} must be finite and in [0, 1] or null.")
    if cfg.hf.flash.docblock_bias_seq_len is not None and int(cfg.hf.flash.docblock_bias_seq_len) < 0:
        raise ValueError("model.hf.flash.docblock_bias_seq_len must be >= 0.")
    if cfg.backbone_type != "hf_deberta_v2" and cfg.hf.attention_impl == "flash":
        raise ValueError(
            "model.hf.attention_impl='flash' is only supported with model.backbone_type='hf_deberta_v2'."
        )
    if cfg.hf.attention_impl == "flash":
        dropout_values = {
            "model.dropout.hidden_prob": cfg.dropout.hidden_prob,
            "model.dropout.attention_probs_prob": cfg.dropout.attention_probs_prob,
        }
        enabled_dropout = [
            f"{name}={value!r}"
            for name, value in dropout_values.items()
            if value is None or float(value) != 0.0
        ]
        if enabled_dropout:
            raise ValueError(
                "model.hf.attention_impl='flash' requires dropout disabled: both fields must resolve "
                "to 0.0; "
                "null preserves backbone/checkpoint dropout and is not accepted. Invalid values: "
                + ", ".join(enabled_dropout)
            )
    if int(cfg.tokenizer.vocab_multiple) <= 0:
        raise ValueError("model.tokenizer.vocab_multiple must be >= 1.")
    if cfg.tokenizer.vocab_target is not None and int(cfg.tokenizer.vocab_target) <= 0:
        raise ValueError("model.tokenizer.vocab_target must be > 0 when provided.")

    defaults = ModelConfig()

    if cfg.backbone_type == "hf_deberta_v2":
        if cfg.hf.max_position_embeddings is not None and int(cfg.hf.max_position_embeddings) <= 0:
            raise ValueError("model.hf.max_position_embeddings must be > 0 when provided.")
        if cfg.hf.max_position_embeddings is not None and not bool(cfg.from_scratch):
            raise ValueError(
                "model.hf.max_position_embeddings is only supported when model.from_scratch=true "
                "for hf_deberta_v2 runs."
            )
    _apply_backbone_option_severity_policy(cfg, defaults)

    if cfg.backbone_type == "rope" and bool(cfg.from_scratch):
        pretrained_changed = asdict(cfg.rope.pretrained) != asdict(defaults.rope.pretrained)
        if pretrained_changed:
            raise ValueError(
                "These options apply only when model.from_scratch=false: model.rope.pretrained.*"
            )

    if cfg.backbone_type == "rope" and not bool(cfg.from_scratch):
        invalid_sources: list[str] = []
        for field_name in ("discriminator_path", "generator_path"):
            src = getattr(cfg.pretrained, field_name, None)
            if src and _looks_like_hf_deberta_checkpoint(str(src)):
                invalid_sources.append(f"{field_name}={src}")
        if invalid_sources:
            raise ValueError(
                "model.from_scratch=false with model.backbone_type='rope' requires DebertaRoPE checkpoints, "
                "not HF DeBERTa v2/v3 checkpoints. "
                "Use model.backbone_type='hf_deberta_v2' for HF DeBERTa weights, or keep model.from_scratch=true "
                "for RoPE model initialization. Invalid sources: " + ", ".join(sorted(invalid_sources))
            )

        scratch_cfg = replace(cfg.rope, pretrained=defaults.rope.pretrained)
        scratch_defaults = replace(defaults.rope, pretrained=defaults.rope.pretrained)
        if asdict(scratch_cfg) != asdict(scratch_defaults):
            raise ValueError(
                "These options only affect scratch RoPE initialization and are not applied when "
                "model.from_scratch=false. Use explicit model.rope.pretrained.* overrides instead."
            )

        pre = cfg.rope.pretrained
        if pre.max_position_embeddings is not None and int(pre.max_position_embeddings) <= 0:
            raise ValueError("model.rope.pretrained.max_position_embeddings must be > 0 when provided.")
        if pre.rotary_pct is not None:
            pct = float(pre.rotary_pct)
            if not math.isfinite(pct) or pct <= 0.0 or pct > 1.0:
                raise ValueError(
                    "model.rope.pretrained.rotary_pct must be finite and in (0, 1] when provided."
                )
        if pre.rope_theta is not None and (
            not math.isfinite(float(pre.rope_theta)) or float(pre.rope_theta) <= 0.0
        ):
            raise ValueError("model.rope.pretrained.rope_theta must be finite and > 0 when provided.")
        if pre.type_vocab_size is not None and int(pre.type_vocab_size) < 0:
            raise ValueError("model.rope.pretrained.type_vocab_size must be >= 0 when provided.")
        if pre.norm_eps is not None and (
            not math.isfinite(float(pre.norm_eps)) or float(pre.norm_eps) <= 0.0
        ):
            raise ValueError("model.rope.pretrained.norm_eps must be finite and > 0 when provided.")
        if pre.keel_alpha_init is not None and not math.isfinite(float(pre.keel_alpha_init)):
            raise ValueError("model.rope.pretrained.keel_alpha_init must be finite when provided.")
        if pre.initializer_range is not None and (
            not math.isfinite(float(pre.initializer_range)) or float(pre.initializer_range) < 0.0
        ):
            raise ValueError("model.rope.pretrained.initializer_range must be finite and >= 0 when provided.")
        if pre.norm_arch is not None:
            _cfg_set(
                pre,
                "norm_arch",
                _ensure_choice("model.rope.pretrained.norm_arch", pre.norm_arch, _NORM_ARCH_CHOICES),
            )
        if pre.ffn_type is not None:
            _cfg_set(
                pre, "ffn_type", _ensure_choice("model.rope.pretrained.ffn_type", pre.ffn_type, _FFN_CHOICES)
            )

    if cfg.pretrained.generator_path:
        if asdict(cfg.generator) != asdict(defaults.generator):
            raise ValueError(
                "model.generator.* overrides are only used when deriving generator config and must be unset "
                "when model.pretrained.generator_path is provided."
            )

    if not bool(cfg.from_scratch) and not cfg.pretrained.generator_path:
        if (
            cfg.generator.hidden_size is not None
            or cfg.generator.intermediate_size is not None
            or cfg.generator.num_attention_heads is not None
        ):
            raise ValueError(
                "model.from_scratch=false with derived generator weights (pretrained.generator_path unset) "
                "cannot use model.generator shape overrides."
            )

    if cfg.backbone_type == "rope":
        if int(cfg.rope.hidden_size) <= 0:
            raise ValueError("model.rope.hidden_size must be > 0.")
        if int(cfg.rope.num_hidden_layers) <= 0:
            raise ValueError("model.rope.num_hidden_layers must be > 0.")
        if int(cfg.rope.num_attention_heads) <= 0:
            raise ValueError("model.rope.num_attention_heads must be > 0.")
        if int(cfg.rope.intermediate_size) <= 0:
            raise ValueError("model.rope.intermediate_size must be > 0.")
        if int(cfg.rope.hidden_size) % int(cfg.rope.num_attention_heads) != 0:
            raise ValueError("model.rope.hidden_size must be divisible by model.rope.num_attention_heads.")

    if cfg.rope.norm_arch == "post":
        if cfg.rope.keel_alpha_init is not None and cfg.rope.keel_alpha_init != defaults.rope.keel_alpha_init:
            warnings.warn(
                "model.rope.keel_alpha_init has no effect when model.rope.norm_arch='post'. "
                "Set model.rope.norm_arch='keel' to use KEEL alpha scaling.",
                UserWarning,
                stacklevel=2,
            )
        if cfg.rope.keel_alpha_learnable != defaults.rope.keel_alpha_learnable:
            warnings.warn(
                "model.rope.keel_alpha_learnable has no effect when model.rope.norm_arch='post'. "
                "Set model.rope.norm_arch='keel' to use learnable KEEL alpha.",
                UserWarning,
                stacklevel=2,
            )

    if (
        cfg.rope.ffn_type == "mlp"
        and cfg.rope.swiglu_adjust_intermediate != defaults.rope.swiglu_adjust_intermediate
    ):
        warnings.warn(
            "model.rope.swiglu_adjust_intermediate has no effect when model.rope.ffn_type='mlp'. "
            "The intermediate size scaling is only applied for ffn_type='swiglu'.",
            UserWarning,
            stacklevel=2,
        )


def validate_data_config(cfg: DataConfig) -> None:
    """Validate data-source and preprocessing option combinations.

    :param DataConfig cfg: Data configuration.
    """
    src = cfg.source
    pack = cfg.packing

    for field_name in ("dataset_name", "dataset_config_name", "data_files", "load_from_disk"):
        value = getattr(src, field_name)
        if value is not None:
            _cfg_set(src, field_name, str(value).strip() or None)
    for field_name in ("train_split", "text_column_name"):
        value = str(getattr(src, field_name)).strip()
        if not value:
            raise ValueError(f"data.source.{field_name} must be a non-empty string.")
        _cfg_set(src, field_name, value)

    if src.load_from_disk:
        if src.streaming:
            raise ValueError(
                "data.source.streaming=true is not compatible with data.source.load_from_disk. "
                "Set data.source.streaming=false."
            )
        conflicting = []
        if src.dataset_name:
            conflicting.append("data.source.dataset_name")
        if src.data_files:
            conflicting.append("data.source.data_files")
        if src.dataset_config_name:
            conflicting.append("data.source.dataset_config_name")
        if conflicting:
            raise ValueError(
                "data.source.load_from_disk cannot be combined with: " + ", ".join(sorted(conflicting))
            )

    if src.dataset_config_name and not src.dataset_name:
        raise ValueError("data.source.dataset_config_name requires data.source.dataset_name.")

    if not src.load_from_disk and not src.dataset_name and not src.data_files:
        raise ValueError(
            "No dataset source configured. Provide one of: data.source.load_from_disk, "
            "data.source.dataset_name, or data.source.data_files."
        )

    if int(pack.max_seq_length) < 8:
        raise ValueError("data.packing.max_seq_length must be >= 8 for pretraining.")
    if int(src.shuffle_buffer_size) < 0:
        raise ValueError("data.source.shuffle_buffer_size must be >= 0.")
    if not bool(src.streaming) and int(src.shuffle_buffer_size) not in {0, 1}:
        raise ValueError(
            "data.source.shuffle_buffer_size must be 0 or 1 when data.source.streaming=false "
            "(non-streaming datasets only support shuffle off/on)."
        )
    if not bool(pack.enabled) and bool(pack.block_cross_document_attention):
        raise ValueError(
            "data.packing.block_cross_document_attention=true requires data.packing.enabled=true."
        )
    if (
        bool(pack.enabled)
        and bool(pack.block_cross_document_attention)
        and int(pack.max_seq_length) > int(_DENSE_DOC_BLOCK_WARN_SEQ_LEN)
    ):
        warnings.warn(
            "data.packing.block_cross_document_attention may build dense O(S^2) pairwise masks on "
            "non-segment-aware attention backends. "
            f"Configured data.packing.max_seq_length={int(pack.max_seq_length)} may be expensive; "
            "consider reducing sequence length or disabling data.packing.block_cross_document_attention "
            f"until a segment-aware backend is enabled (warning threshold: {int(_DENSE_DOC_BLOCK_WARN_SEQ_LEN)}).",
            UserWarning,
            stacklevel=2,
        )


def validate_train_config(cfg: TrainConfig) -> None:
    """Validate train config scalar ranges and constrained options.

    :param TrainConfig cfg: Training configuration.
    """
    _cfg_set(cfg, "sdpa_kernel", _normalize_sdpa_kernel(cfg.sdpa_kernel))
    _cfg_set(cfg, "mixed_precision", normalize_mixed_precision(cfg.mixed_precision))
    _cfg_set(cfg.compile, "mode", _normalize_torch_compile_mode(cfg.compile.mode))
    _cfg_set(cfg.compile, "scope", _normalize_torch_compile_scope(cfg.compile.scope))
    _cfg_set(cfg.compile, "backend", _normalize_torch_compile_backend(cfg.compile.backend))
    _cfg_set(
        cfg.checkpoint,
        "resume_data_strategy",
        _ensure_choice(
            "train.checkpoint.resume_data_strategy",
            cfg.checkpoint.resume_data_strategy,
            _RESUME_DATA_STRATEGY_CHOICES,
        ),
    )

    if cfg.checkpoint.output_dir is not None:
        checkpoint_output_dir = str(cfg.checkpoint.output_dir).strip()
        _cfg_set(cfg.checkpoint, "output_dir", checkpoint_output_dir or None)
    if cfg.checkpoint.resume_from_checkpoint is not None:
        resume_from_checkpoint = str(cfg.checkpoint.resume_from_checkpoint).strip()
        _cfg_set(
            cfg.checkpoint,
            "resume_from_checkpoint",
            resume_from_checkpoint if resume_from_checkpoint else None,
        )
    if bool(cfg.checkpoint.overwrite_output_dir) and bool(cfg.checkpoint.resume_from_checkpoint):
        raise ValueError(
            "train.checkpoint.overwrite_output_dir=true cannot be combined with "
            "train.checkpoint.resume_from_checkpoint. Overwrite would delete checkpoints before resume."
        )

    for _name, _min in (
        ("max_steps", 1),
        ("per_device_train_batch_size", 1),
        ("gradient_accumulation_steps", 1),
        ("dataloader.num_workers", 0),
        ("checkpoint.save_steps", 0),
        ("checkpoint.save_total_limit", 0),
        ("checkpoint.resume_replay_max_micro_batches", 0),
        ("objective.mlm_max_ngram", 1),
    ):
        val = _nested_get(cfg, _name)
        if int(val) < int(_min):
            raise ValueError(f"train.{_name} must be >= {_min}.")

    seed = int(cfg.seed)
    if seed < 0 or seed > (2**32 - 1):
        raise ValueError("train.seed must be between 0 and 2**32 - 1 (inclusive).")

    mlm = float(cfg.objective.mlm_probability)
    if not math.isfinite(mlm) or mlm <= 0.0 or mlm >= 1.0:
        raise ValueError("train.objective.mlm_probability must be finite and in (0, 1).")
    mask_p = float(cfg.objective.mask_token_prob)
    rand_p = float(cfg.objective.random_token_prob)
    if (
        not math.isfinite(mask_p)
        or not math.isfinite(rand_p)
        or mask_p < 0.0
        or rand_p < 0.0
        or (mask_p + rand_p) > 1.0
    ):
        raise ValueError(
            "Invalid masking probabilities: train.objective.mask_token_prob + "
            "train.objective.random_token_prob must be finite, nonnegative, and <= 1."
        )
    temperature = float(cfg.objective.sampling_temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("train.objective.sampling_temperature must be finite and > 0.")
    gen_weight = float(cfg.objective.gen_loss_weight)
    disc_weight = float(cfg.objective.disc_loss_weight)
    if (
        not math.isfinite(gen_weight)
        or not math.isfinite(disc_weight)
        or gen_weight < 0.0
        or disc_weight < 0.0
    ):
        raise ValueError("train objective loss weights must be finite and >= 0.")
    if gen_weight == 0.0 and disc_weight == 0.0:
        raise ValueError("At least one train objective loss weight must be > 0.")
    if not isinstance(cfg.decoupled_training, bool):
        raise ValueError(
            "train.decoupled_training must be a boolean (true/false). "
            f"Got {type(cfg.decoupled_training).__name__}."
        )

    defaults = TrainConfig()

    if not bool(cfg.compile.enabled):
        for _knob in ("mode", "scope", "backend"):
            if getattr(cfg.compile, _knob) != getattr(defaults.compile, _knob):
                warnings.warn(
                    f"train.compile.{_knob} has no effect when train.compile.enabled=false. "
                    f"Current value ({getattr(cfg.compile, _knob)!r}) will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )


def validate_optim_config(cfg: OptimConfig) -> None:
    """Validate optimizer configuration.

    :param OptimConfig cfg: Optimizer config.
    """
    _cfg_set(
        cfg.scheduler,
        "type",
        _ensure_choice("optim.scheduler.type", cfg.scheduler.type, _LR_SCHEDULER_CHOICES),
    )

    if not math.isfinite(float(cfg.lr.base)) or float(cfg.lr.base) <= 0.0:
        raise ValueError("optim.lr.base must be finite and > 0.")
    if float(cfg.lr.generator) != -1.0 and (
        not math.isfinite(float(cfg.lr.generator)) or float(cfg.lr.generator) <= 0.0
    ):
        raise ValueError("optim.lr.generator must be -1 (inherit) or finite and > 0.")
    if float(cfg.lr.discriminator) != -1.0 and (
        not math.isfinite(float(cfg.lr.discriminator)) or float(cfg.lr.discriminator) <= 0.0
    ):
        raise ValueError("optim.lr.discriminator must be -1 (inherit) or finite and > 0.")
    for field_name in ("beta1", "beta2"):
        value = float(getattr(cfg.adam, field_name))
        if not math.isfinite(value) or value < 0.0 or value >= 1.0:
            raise ValueError(f"optim.adam.{field_name} must be finite and in [0, 1).")
    if not math.isfinite(float(cfg.adam.epsilon)) or float(cfg.adam.epsilon) <= 0.0:
        raise ValueError("optim.adam.epsilon must be finite and > 0.")
    if not math.isfinite(float(cfg.weight_decay)) or float(cfg.weight_decay) < 0.0:
        raise ValueError("optim.weight_decay must be finite and >= 0.")
    if not math.isfinite(float(cfg.max_grad_norm)) or float(cfg.max_grad_norm) < 0.0:
        raise ValueError("optim.max_grad_norm must be finite and >= 0.")
    if int(cfg.scheduler.warmup_steps) < 0:
        raise ValueError("optim.scheduler.warmup_steps must be >= 0.")


def validate_logging_config(cfg: LoggingConfig) -> None:
    """Validate logging and tracker configuration.

    :param LoggingConfig cfg: Logging config.
    """
    _cfg_set(cfg.wandb, "watch", _normalize_wandb_watch(cfg.wandb.watch))

    project_name = str(cfg.project_name).strip()
    if not project_name:
        raise ValueError("logging.project_name must be non-empty.")
    _cfg_set(cfg, "project_name", project_name)
    if cfg.run_name is not None:
        run_name = str(cfg.run_name).strip()
        _cfg_set(cfg, "run_name", run_name or None)
    if cfg.output_dir is not None:
        output_dir = str(cfg.output_dir).strip()
        _cfg_set(cfg, "output_dir", output_dir or None)
    if int(cfg.logging_steps) < 0:
        raise ValueError("logging.logging_steps must be >= 0.")
    if int(cfg.wandb.watch_log_freq) < 1:
        raise ValueError("logging.wandb.watch_log_freq must be >= 1.")

    defaults = LoggingConfig()
    if not bool(cfg.wandb.enabled):
        if cfg.wandb.watch != defaults.wandb.watch:
            warnings.warn(
                "logging.wandb.watch only applies when logging.wandb.enabled=true. "
                f"Current value ({cfg.wandb.watch!r}) will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        if int(cfg.wandb.watch_log_freq) != int(defaults.wandb.watch_log_freq):
            warnings.warn(
                "logging.wandb.watch_log_freq only applies when logging.wandb.enabled=true. "
                f"Current value ({int(cfg.wandb.watch_log_freq)}) will be ignored.",
                UserWarning,
                stacklevel=2,
            )


def validate_training_workflow_options(
    *,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig | None = None,
    optim_cfg: OptimConfig | None = None,
) -> None:
    """Validate options tied to workflow support (for example, eval mode availability).

    :param DataConfig data_cfg: Data configuration.
    :param TrainConfig train_cfg: Training configuration.
    :param ModelConfig | None model_cfg: Optional model configuration.
    :param OptimConfig | None optim_cfg: Optional optimizer configuration.
    """
    sdpa_policy = str(train_cfg.sdpa_kernel).strip().lower()
    reject_flash_sdpa_for_doc_block = (
        bool(data_cfg.packing.enabled)
        and bool(data_cfg.packing.block_cross_document_attention)
        and sdpa_policy == "flash"
        and (model_cfg is None or str(model_cfg.backbone_type).strip().lower() == "rope")
    )
    if reject_flash_sdpa_for_doc_block:
        raise ValueError(
            "train.sdpa_kernel=flash is not supported with data.packing.enabled=true. "
            "Packed batches may require 3D document-blocking attention masks that are incompatible "
            "with strict flash SDPA kernels. Use train.sdpa_kernel=auto|mem_efficient|math instead."
        )

    if optim_cfg is not None:
        scheduler_type = str(optim_cfg.scheduler.type).strip().lower()
        warmup_steps = int(optim_cfg.scheduler.warmup_steps)
        max_steps = int(train_cfg.max_steps)
        if scheduler_type != "constant" and warmup_steps >= max_steps:
            raise ValueError(
                "optim.scheduler.warmup_steps must be less than train.max_steps for "
                f"scheduler type {scheduler_type!r}; got warmup_steps={warmup_steps}, "
                f"max_steps={max_steps}. Only scheduler type 'constant' ignores warmup_steps."
            )

    if model_cfg is not None:
        backbone_type = str(model_cfg.backbone_type).strip().lower()
        attn_impl = str(model_cfg.rope.attention_implementation).strip().lower()
        hf_attention_impl = str(model_cfg.hf.attention_impl).strip().lower()
        if (
            backbone_type == "hf_deberta_v2"
            and hf_attention_impl == "flash"
            and str(train_cfg.mixed_precision).strip().lower() != "bf16"
        ):
            raise ValueError(
                "model.hf.attention_impl='flash' requires train.mixed_precision='bf16'. "
                "Full-precision tensors are unsupported by the configured FlashDeBERTa kernels."
            )
        if backbone_type != "rope" and sdpa_policy != "auto":
            warnings.warn(
                "train.sdpa_kernel has no effect when model.backbone_type='hf_deberta_v2'. "
                "The HF DeBERTa-v2 disentangled attention uses explicit matmuls, not F.scaled_dot_product_attention. "
                f"Current value ({train_cfg.sdpa_kernel!r}) will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        if backbone_type == "rope" and attn_impl != "sdpa" and sdpa_policy != "auto":
            raise ValueError(
                "train.sdpa_kernel only affects rope attention when model.rope.attention_implementation='sdpa'. "
                "Set train.sdpa_kernel=auto or switch model.rope.attention_implementation=sdpa."
            )
        embed_sharing = str(model_cfg.embedding_sharing).strip().lower()
        local_optim = optim_cfg or OptimConfig()
        gen_lr = float(local_optim.lr.generator)
        if embed_sharing == "es" and gen_lr > 0 and gen_lr != float(local_optim.lr.base):
            raise ValueError(
                f"model.embedding_sharing='es' shares embedding parameters between generator and discriminator, "
                f"but optim.lr.generator ({gen_lr}) differs from optim.lr.base ({local_optim.lr.base}). "
                "Set optim.lr.generator=-1 (inherit) or match it to optim.lr.base, "
                "or switch to embedding_sharing='gdes'/'none'."
            )
        if bool(train_cfg.decoupled_training) and embed_sharing == "es":
            raise ValueError(
                "train.decoupled_training is incompatible with model.embedding_sharing='es' because "
                "shared embedding parameters are assigned to the generator optimizer only. "
                "Discriminator-phase gradients on shared embeddings would be dropped at discriminator "
                "optimizer zero_grad() because discriminator optimizer param groups do not own those shared params. "
                "Use embedding_sharing='gdes' or 'none' for decoupled training."
            )


def validate_run_metadata_schema(raw: dict[str, object], *, source: str) -> None:
    """Validate run-metadata schema compatibility.

    :param dict[str, object] raw: Parsed run metadata payload.
    :param str source: Human-readable source location for errors.
    :raises ValueError: If schema metadata is missing or incompatible.
    """
    if "config_schema_version" not in raw:
        raise ValueError(
            f"run metadata missing `config_schema_version` at {source}. "
            "Refusing resume/export with ambiguous config schema."
        )

    try:
        schema_version = int(raw["config_schema_version"])
    except Exception as e:
        raise ValueError(
            f"Invalid config_schema_version in {source}: {raw['config_schema_version']!r}"
        ) from e

    if schema_version != int(RUN_CONFIG_SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported run metadata schema at {source}: {schema_version}. "
            f"Expected {int(RUN_CONFIG_SCHEMA_VERSION)}. "
            "Backward resume/export compatibility is not guaranteed before stable release."
        )


_SnapshotConfigT = TypeVar(
    "_SnapshotConfigT",
    ModelConfig,
    DataConfig,
    TrainConfig,
    OptimConfig,
    LoggingConfig,
)


def _load_snapshot_dataclass(
    raw: dict[str, object], *, cls: type[_SnapshotConfigT], source: str, config_name: str
) -> _SnapshotConfigT:
    """Construct a config dataclass from a persisted snapshot with strict key checks.

    :param dict[str, object] raw: Raw persisted JSON mapping.
    :param type[_SnapshotConfigT] cls: Target dataclass type.
    :param str source: Snapshot source path for errors.
    :param str config_name: Human label used in error messages.
    :raises ValueError: If unknown keys are present or dataclass construction fails.
    :return _SnapshotConfigT: Parsed dataclass instance.
    """
    expected = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - expected)
    if unknown:
        raise ValueError(
            f"Unsupported {config_name} keys in {source}: {', '.join(unknown)}. "
            "This snapshot was produced by an older pre-release schema; "
            "backward resume/export compatibility is not guaranteed before stable release."
        )
    missing = sorted(expected - set(raw))
    if missing:
        raise ValueError(
            f"Missing required {config_name} keys in {source}: {', '.join(missing)}. "
            "This snapshot does not match the current config schema."
        )

    try:
        return _replace_from_mapping_recursive(cls(), dict(raw), section_name=config_name)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Failed to parse {config_name} at {source}. "
            "The persisted config schema does not match this code version."
        ) from e


def load_model_config_snapshot(raw: dict[str, object], *, source: str) -> ModelConfig:
    """Parse persisted `model_config.json` into ModelConfig.

    :param dict[str, object] raw: Raw model config mapping.
    :param str source: Source path for error messages.
    :return ModelConfig: Parsed model configuration.
    """
    return _load_snapshot_dataclass(raw, cls=ModelConfig, source=source, config_name="model_config.json")


def load_data_config_snapshot(raw: dict[str, object], *, source: str) -> DataConfig:
    """Parse persisted `data_config.json` into DataConfig.

    :param dict[str, object] raw: Raw data config mapping.
    :param str source: Source path for error messages.
    :return DataConfig: Parsed data configuration.
    """
    return _load_snapshot_dataclass(raw, cls=DataConfig, source=source, config_name="data_config.json")


def load_train_config_snapshot(raw: dict[str, object], *, source: str) -> TrainConfig:
    """Parse persisted `train_config.json` into TrainConfig.

    :param dict[str, object] raw: Raw train config mapping.
    :param str source: Source path for error messages.
    :return TrainConfig: Parsed training configuration.
    """
    return _load_snapshot_dataclass(raw, cls=TrainConfig, source=source, config_name="train_config.json")


def load_optim_config_snapshot(raw: dict[str, object], *, source: str) -> OptimConfig:
    """Parse persisted `optim_config.json` into OptimConfig.

    :param dict[str, object] raw: Raw optim config mapping.
    :param str source: Source path for error messages.
    :return OptimConfig: Parsed optimizer configuration.
    """
    return _load_snapshot_dataclass(raw, cls=OptimConfig, source=source, config_name="optim_config.json")


def load_logging_config_snapshot(raw: dict[str, object], *, source: str) -> LoggingConfig:
    """Parse persisted `logging_config.json` into LoggingConfig.

    :param dict[str, object] raw: Raw logging config mapping.
    :param str source: Source path for error messages.
    :return LoggingConfig: Parsed logging configuration.
    """
    return _load_snapshot_dataclass(raw, cls=LoggingConfig, source=source, config_name="logging_config.json")


# Old train-section keys that moved to other sections. Same-section migrations
# derive from each config class's moved-key table.
_TRAIN_CROSS_SECTION_SUGGESTIONS: dict[str, str] = {
    "project_name": "logging.project_name",
    "run_name": "logging.run_name",
    "report_to": "logging.wandb.enabled",
    "logging_steps": "logging.logging_steps",
    "wandb_watch": "logging.wandb.watch",
    "wandb_watch_log_freq": "logging.wandb.watch_log_freq",
    "debug_metrics": "logging.debug.metrics",
    "learning_rate": "optim.lr.base",
    "generator_learning_rate": "optim.lr.generator",
    "discriminator_learning_rate": "optim.lr.discriminator",
    "weight_decay": "optim.weight_decay",
    "adam_beta1": "optim.adam.beta1",
    "adam_beta2": "optim.adam.beta2",
    "adam_epsilon": "optim.adam.epsilon",
    "warmup_steps": "optim.scheduler.warmup_steps",
    "lr_scheduler_type": "optim.scheduler.type",
    "max_grad_norm": "optim.max_grad_norm",
}


def _legacy_key_suggestion(section_name: str, key: str) -> str | None:
    """Return an actionable migration suggestion for an unknown key.

    Only cross-section train migrations and former root groups retain targeted hints.

    :param str section_name: Section path.
    :param str key: Unknown key.
    :return str | None: Suggested replacement path.
    """
    section = str(section_name)
    k = str(key)
    if section == "train":
        return _TRAIN_CROSS_SECTION_SUGGESTIONS.get(k)
    if section == "root":
        if k == "checkpoint":
            return "train.checkpoint"
        if k == "debug":
            return "logging.debug"
    return None


def _load_raw_config_mapping(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load raw config mapping from YAML/JSON and return format label.

    :param str | Path path: Config path.
    :raises ValueError: If path extension/content is invalid.
    :return tuple[dict[str, Any], str]: Parsed mapping and format name.
    """
    cfg_path = Path(path).expanduser().resolve()
    suffix = cfg_path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError("Config file must end with .json, .yaml, or .yml")

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "pyyaml is required for YAML config files. Install with `pip install pyyaml`."
            ) from e

        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    else:
        raw = load_json_mapping(cfg_path)

    if not isinstance(raw, dict):
        raise ValueError("Config file must parse to a dict.")
    format_name = "YAML" if suffix in {".yaml", ".yml"} else "JSON"
    return raw, format_name


def _split_full_sections(raw: dict[str, Any], *, format_name: str) -> dict[str, dict[str, Any]]:
    """Split config mappings into strict top-level sections.

    :param dict[str, Any] raw: Parsed config mapping.
    :param str format_name: Format label for error messages.
    :raises ValueError: If config shape is invalid.
    :return dict[str, dict[str, Any]]: Section dictionaries.
    """
    known = {"model", "data", "train", "optim", "logging"}
    unknown_top = sorted(key for key in raw.keys() if key not in known)
    if unknown_top:
        details = []
        for key in unknown_top:
            sug = _legacy_key_suggestion("root", key)
            if sug is not None:
                details.append(f"{key} (use {sug})")
            else:
                details.append(str(key))
        raise ValueError(
            f"Unknown top-level keys in nested {format_name} config "
            f"(expected only {', '.join(sorted(known))}): {', '.join(details)}"
        )

    sections = {key: (raw.get(key, {}) or {}) for key in known}
    for key, section in sections.items():
        if not isinstance(section, dict):
            raise ValueError(f"{format_name} config section {key!r} must be a dict.")
    return sections


def _collect_mapping_leaf_paths(value: Any, *, prefix: str = "") -> set[str]:
    """Collect dotted leaf paths from a nested config mapping.

    :param Any value: Nested config value.
    :param str prefix: Current dotted path.
    :return set[str]: Dotted leaf paths explicitly present in the mapping.
    """
    if not isinstance(value, dict):
        return {prefix} if prefix else set()

    paths: set[str] = set()
    for key, item in value.items():
        child = f"{prefix}.{key}" if prefix else str(key)
        paths.update(_collect_mapping_leaf_paths(item, prefix=child))
    return paths


def _replace_from_mapping_recursive(cfg_obj: Any, mapping: dict[str, Any], *, section_name: str) -> Any:
    """Apply mapping values onto a frozen dataclass recursively.

    :param Any cfg_obj: Dataclass instance.
    :param dict[str, Any] mapping: Field/value mapping.
    :param str section_name: Section label for error messages.
    :raises ValueError: If unknown keys are provided.
    :return Any: Replaced dataclass object.
    """
    if not mapping:
        return cfg_obj

    allowed = {f.name for f in fields(type(cfg_obj))}
    unknown = sorted(set(mapping.keys()) - allowed)
    if unknown:
        rendered: list[str] = []
        for key in unknown:
            sug = _legacy_key_suggestion(section_name, key)
            if sug is None:
                rendered.append(str(key))
            else:
                rendered.append(f"{key} (use {sug})")
        raise ValueError(f"Unknown keys in section {section_name!r}: {', '.join(rendered)}")

    updates: dict[str, Any] = {}
    type_hints = get_type_hints(type(cfg_obj))
    for key, value in mapping.items():
        cur = getattr(cfg_obj, key)
        if dataclasses.is_dataclass(cur):
            if not isinstance(value, dict):
                raise ValueError(f"Section {section_name}.{key!s} must be a mapping.")
            updates[str(key)] = _replace_from_mapping_recursive(
                cur,
                value,
                section_name=f"{section_name}.{key}",
            )
        else:
            field_type = type_hints.get(str(key), Any)
            updates[str(key)] = _coerce_config_mapping_scalar_value(
                raw_value=value,
                field_type=field_type,
                field_path=f"{section_name}.{key}",
            )

    return replace(cfg_obj, **updates)


def _coerce_config_mapping_scalar_value(*, raw_value: Any, field_type: Any, field_path: str) -> Any:
    """Validate config-file scalar values against dataclass field annotations.

    :param Any raw_value: Parsed YAML/JSON scalar value.
    :param Any field_type: Dataclass field type annotation.
    :param str field_path: Dotted field path for error context.
    :raises ValueError: If a scalar value does not match the declared field type.
    :return Any: Type-checked (and numeric-normalized) scalar value.
    """
    target_t, allows_none = unwrap_optional_type(field_type)
    value = raw_value
    path = str(field_path)

    if value is None:
        if allows_none:
            return None
        raise ValueError(f"Config field {path} cannot be null.")

    if target_t is bool:
        if isinstance(value, bool):
            return value
        raise ValueError(
            f"Config field {path} must be a boolean true/false, got {type(value).__name__}: {value!r}."
        )

    if target_t is int:
        if isinstance(value, bool):
            raise ValueError(f"Config field {path} must be an integer, got bool: {value!r}.")
        if isinstance(value, int):
            return int(value)
        if isinstance(value, str):
            text = str(value).strip()
            try:
                return int(text)
            except ValueError as exc:
                raise ValueError(f"Config field {path} must be an integer, got string: {value!r}.") from exc
        raise ValueError(f"Config field {path} must be an integer, got {type(value).__name__}: {value!r}.")

    if target_t is float:
        if isinstance(value, bool):
            raise ValueError(f"Config field {path} must be a number, got bool: {value!r}.")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = str(value).strip()
            try:
                return float(text)
            except ValueError as exc:
                raise ValueError(f"Config field {path} must be a number, got string: {value!r}.") from exc
        raise ValueError(f"Config field {path} must be a number, got {type(value).__name__}: {value!r}.")

    if target_t is str:
        if isinstance(value, bool):
            raise ValueError(
                f"Config field {path} must be a string, got bool: {value!r}. "
                "YAML parses unquoted yes/no/true/false as booleans; quote the value "
                f'(e.g. {path}: "no").'
            )
        if not isinstance(value, str):
            raise ValueError(f"Config field {path} must be a string, got {type(value).__name__}: {value!r}.")
        return str(value)

    return value


def _build_config_from_section_mappings(section_maps: dict[str, dict[str, Any]]) -> Config:
    """Construct top-level Config from parsed section mappings.

    :param dict[str, dict[str, Any]] section_maps: Parsed section mappings.
    :return Config: Top-level config object.
    """
    model_cfg = _replace_from_mapping_recursive(
        ModelConfig(), section_maps.get("model", {}), section_name="model"
    )
    data_cfg = _replace_from_mapping_recursive(
        DataConfig(), section_maps.get("data", {}), section_name="data"
    )
    train_cfg = _replace_from_mapping_recursive(
        TrainConfig(), section_maps.get("train", {}), section_name="train"
    )
    optim_cfg = _replace_from_mapping_recursive(
        OptimConfig(), section_maps.get("optim", {}), section_name="optim"
    )
    logging_cfg = _replace_from_mapping_recursive(
        LoggingConfig(), section_maps.get("logging", {}), section_name="logging"
    )

    explicit_fields: set[str] = set()
    for section_name, section_mapping in section_maps.items():
        explicit_fields.update(_collect_mapping_leaf_paths(section_mapping, prefix=section_name))

    cfg = Config(
        model=model_cfg,
        data=data_cfg,
        train=train_cfg,
        optim=optim_cfg,
        logging=logging_cfg,
        _explicit_fields=frozenset(explicit_fields),
    )
    return cfg


def _coerce_override_value(raw: str, field_type: Any) -> Any:
    """Cast a dotted-override raw string to the target dataclass field type.

    :param str raw: Raw override value string.
    :param Any field_type: Dataclass field type annotation.
    :raises ValueError: If value cannot be parsed as the target type.
    :return Any: Typed override value.
    """
    return coerce_scalar(raw, field_type, allow_none=False, allow_bool_numeric=False)


def apply_dotted_override(cfg: Config, override: str) -> Config:
    """Apply one dotted override expression to a Config object.

    :param Config cfg: Existing config bundle.
    :param str override: Override expression.
    :raises ValueError: If expression/section/field/value is invalid.
    :return Config: New config with one updated field.
    """
    text = str(override).strip()
    if "=" not in text:
        raise ValueError(f"Invalid override {override!r}. Expected format like model.rope.hidden_size=768.")
    path, raw_value = text.split("=", 1)
    path = path.strip()
    raw_value = raw_value.strip()
    parts = [str(p).strip() for p in path.split(".") if str(p).strip()]
    if len(parts) < 2:
        raise ValueError(
            f"Invalid override path {path!r}. Expected format section.field (for example train.max_steps)."
        )

    root = parts[0]
    public_roots = {f.name for f in fields(Config) if not f.name.startswith("_")}
    if root not in public_roots:
        raise ValueError(
            f"Unknown override section {root!r}; expected one of {', '.join(sorted(public_roots))}."
        )

    def _resolve_override_leaf_type(obj: Any, remaining: list[str], full_path: str) -> Any:
        """Resolve and validate the leaf field type for one override path.

        :param Any obj: Root section dataclass.
        :param list[str] remaining: Path parts under the section.
        :param str full_path: Full dotted path for error reporting.
        :raises ValueError: If any path segment is unknown.
        :return Any: Leaf field type hint.
        """
        node = obj
        for key in remaining[:-1]:
            if not hasattr(node, key):
                raise ValueError(f"Unknown override field {full_path!r}.")
            node = getattr(node, key)
        leaf = remaining[-1]
        if not hasattr(node, leaf):
            raise ValueError(f"Unknown override field {full_path!r}.")
        return get_type_hints(type(node)).get(leaf, Any)

    root_obj = getattr(cfg, root)
    leaf_type = _resolve_override_leaf_type(root_obj, parts[1:], path)
    coerced_value = _coerce_override_value(raw_value, leaf_type)
    new_root = _replace_path(root_obj, parts[1:], coerced_value)
    new_cfg = replace(
        cfg,
        **{root: new_root},
        _explicit_fields=_explicit_config_fields(cfg) | {path},
    )

    return new_cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load, resolve, override, and validate config into a top-level Config object.

    :param str | Path path: YAML/JSON config path.
    :param list[str] | None overrides: Optional dotted overrides.
    :return Config: Validated immutable config object.
    """
    raw, format_name = _load_raw_config_mapping(path)
    section_maps = _split_full_sections(raw, format_name=format_name)
    cfg = _build_config_from_section_mappings(section_maps)
    if overrides:
        for expr in overrides:
            cfg = apply_dotted_override(cfg, expr)
    validate_model_config(cfg.model)
    validate_data_config(cfg.data)
    validate_train_config(cfg.train)
    validate_optim_config(cfg.optim)
    validate_logging_config(cfg.logging)
    validate_training_workflow_options(
        data_cfg=cfg.data,
        train_cfg=cfg.train,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
    )
    return cfg


def iter_leaf_paths_for_dataclass(cls: type[Any], *, prefix: str = "") -> list[tuple[str, Any]]:
    """List dotted leaf paths and field types for a dataclass type.

    :param type[Any] cls: Dataclass type.
    :param str prefix: Optional prefix.
    :return list[tuple[str, Any]]: Leaf path + type tuples.
    """
    out: list[tuple[str, Any]] = []
    type_hints = get_type_hints(cls)
    for f in fields(cls):
        if f.name.startswith("_"):
            continue
        path = f"{prefix}.{f.name}" if prefix else str(f.name)
        field_type = type_hints.get(f.name, f.type)
        target_t, _allows_none = unwrap_optional_type(field_type)
        if dataclasses.is_dataclass(target_t):
            out.extend(iter_leaf_paths_for_dataclass(target_t, prefix=path))
        else:
            out.append((path, field_type))
    return out


asdict_without_private = _asdict_without_private


__all__ = [
    "RUN_CONFIG_SCHEMA_VERSION",
    "Config",
    "ModelHFFlashConfig",
    "ModelHFConfig",
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "OptimConfig",
    "LoggingConfig",
    "apply_dotted_override",
    "asdict_without_private",
    "iter_leaf_paths_for_dataclass",
    "load_config",
    "load_data_config_snapshot",
    "load_logging_config_snapshot",
    "load_model_config_snapshot",
    "load_optim_config_snapshot",
    "normalize_mixed_precision",
    "resolve_effective_mixed_precision",
    "validate_data_config",
    "validate_logging_config",
    "validate_model_config",
    "validate_optim_config",
    "validate_run_metadata_schema",
    "validate_train_config",
    "validate_training_workflow_options",
]
