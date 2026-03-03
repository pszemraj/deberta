"""Dataclass-based model, data, and training configuration definitions."""

from __future__ import annotations

import dataclasses
import re
import types
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from deberta.io_utils import load_json_mapping

_BACKBONE_CHOICES = {"rope", "hf_deberta_v2"}
_MODEL_PROFILE_CHOICES = {"modern", "deberta_v3_parity"}
_NORM_ARCH_CHOICES = {"post", "keel"}
_ATTN_IMPL_CHOICES = {"sdpa", "eager"}
_FFN_CHOICES = {"swiglu", "mlp"}
_EMBED_SHARING_CHOICES = {"none", "es", "gdes"}
_LOGGING_BACKEND_CHOICES = {"none", "tensorboard"}
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
RUN_CONFIG_SCHEMA_VERSION = 4
_VAR_FULL_RE = re.compile(r"^\$variables\.([A-Za-z0-9_.-]+)$")
_VAR_INLINE_RE = re.compile(r"\{\$variables\.([A-Za-z0-9_.-]+)\}")
_VAR_BRACE_RE = re.compile(r"\$\{variables\.([A-Za-z0-9_.-]+)\}")
_VAR_SUSPICIOUS_RE = re.compile(r"\$variables\.[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ModelTokenizerConfig:
    """Tokenizer sub-configuration."""

    name_or_path: str = field(default="microsoft/deberta-v3-base")
    allow_vocab_resize: bool = field(default=False)
    vocab_target: int | None = field(default=None)
    vocab_multiple: int = field(default=1)


@dataclass(frozen=True)
class ModelHFConfig:
    """HF DeBERTa-v2/v3 backbone synthesis options."""

    model_size: str = field(default="base")
    attention_kernel: str = field(default="dynamic")
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


def _flatten_mapping_for_init(mapping: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten nested mapping keys for constructor update helpers.

    :param dict[str, Any] mapping: Nested mapping payload.
    :param str prefix: Optional prefix path.
    :return dict[str, Any]: Flattened dotted mapping.
    """
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_mapping_for_init(value, prefix=path))
        else:
            out[str(path)] = value
    return out


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


def _apply_dotted_updates(obj: Any, updates: dict[str, Any]) -> Any:
    """Apply dotted update mapping to a dataclass object.

    :param Any obj: Source dataclass instance.
    :param dict[str, Any] updates: Dotted updates.
    :return Any: Updated dataclass object.
    """
    out = obj
    for path, value in updates.items():
        parts = [str(p).strip() for p in str(path).split(".") if str(p).strip()]
        if not parts:
            continue
        out = _replace_path(out, parts, value)
    return out


def _coerce_subconfig(value: Any, cls: type[Any], *, field_name: str) -> Any:
    """Coerce constructor subconfig values to dataclass instances.

    :param Any value: Candidate subconfig value.
    :param type[Any] cls: Subconfig dataclass type.
    :param str field_name: Field name for error reporting.
    :return Any: Coerced subconfig dataclass instance.
    """
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return _apply_dotted_updates(cls(), _flatten_mapping_for_init(value))
    raise TypeError(f"{field_name} must be a {cls.__name__} or mapping, got {type(value).__name__}.")


@dataclass(frozen=True)
class ModelConfig:
    """Model-related arguments."""

    profile: str = field(default="modern")
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

    _LEGACY_MAP = {
        "tokenizer_name_or_path": "tokenizer.name_or_path",
        "tokenizer_allow_vocab_resize": "tokenizer.allow_vocab_resize",
        "tokenizer_vocab_target": "tokenizer.vocab_target",
        "tokenizer_vocab_multiple": "tokenizer.vocab_multiple",
        "hf_attention_kernel": "hf.attention_kernel",
        "hf_model_size": "hf.model_size",
        "hf_max_position_embeddings": "hf.max_position_embeddings",
        "pretrained_discriminator_path": "pretrained.discriminator_path",
        "pretrained_generator_path": "pretrained.generator_path",
        "generator_num_hidden_layers": "generator.num_hidden_layers",
        "generator_hidden_size": "generator.hidden_size",
        "generator_intermediate_size": "generator.intermediate_size",
        "generator_num_attention_heads": "generator.num_attention_heads",
        "hidden_size": "rope.hidden_size",
        "num_hidden_layers": "rope.num_hidden_layers",
        "num_attention_heads": "rope.num_attention_heads",
        "intermediate_size": "rope.intermediate_size",
        "hidden_act": "rope.hidden_act",
        "rope_theta": "rope.rope_theta",
        "rotary_pct": "rope.rotary_pct",
        "use_absolute_position_embeddings": "rope.use_absolute_position_embeddings",
        "max_position_embeddings": "rope.max_position_embeddings",
        "type_vocab_size": "rope.type_vocab_size",
        "norm_arch": "rope.norm_arch",
        "norm_eps": "rope.norm_eps",
        "keel_alpha_init": "rope.keel_alpha_init",
        "keel_alpha_learnable": "rope.keel_alpha_learnable",
        "attention_implementation": "rope.attention_implementation",
        "ffn_type": "rope.ffn_type",
        "use_bias": "rope.use_bias",
        "swiglu_adjust_intermediate": "rope.swiglu_adjust_intermediate",
        "initializer_range": "rope.initializer_range",
        "pretrained_max_position_embeddings": "rope.pretrained.max_position_embeddings",
        "pretrained_rope_theta": "rope.pretrained.rope_theta",
        "pretrained_rotary_pct": "rope.pretrained.rotary_pct",
        "pretrained_use_absolute_position_embeddings": "rope.pretrained.use_absolute_position_embeddings",
        "pretrained_type_vocab_size": "rope.pretrained.type_vocab_size",
        "pretrained_norm_arch": "rope.pretrained.norm_arch",
        "pretrained_norm_eps": "rope.pretrained.norm_eps",
        "pretrained_keel_alpha_init": "rope.pretrained.keel_alpha_init",
        "pretrained_keel_alpha_learnable": "rope.pretrained.keel_alpha_learnable",
        "pretrained_ffn_type": "rope.pretrained.ffn_type",
        "pretrained_use_bias": "rope.pretrained.use_bias",
        "pretrained_initializer_range": "rope.pretrained.initializer_range",
        "hidden_dropout_prob": "dropout.hidden_prob",
        "attention_probs_dropout_prob": "dropout.attention_probs_prob",
    }

    def __init__(
        self,
        profile: str = "modern",
        backbone_type: str = "hf_deberta_v2",
        from_scratch: bool = True,
        embedding_sharing: str = "gdes",
        gradient_checkpointing: bool = False,
        tokenizer: ModelTokenizerConfig | dict[str, Any] | None = None,
        hf: ModelHFConfig | dict[str, Any] | None = None,
        pretrained: ModelPretrainedConfig | dict[str, Any] | None = None,
        generator: ModelGeneratorConfig | dict[str, Any] | None = None,
        rope: ModelRopeConfig | dict[str, Any] | None = None,
        dropout: ModelDropoutConfig | dict[str, Any] | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        """Initialize model config while accepting legacy flat kwargs.

        :param str profile: Model profile.
        :param str backbone_type: Backbone type.
        :param bool from_scratch: Scratch/pretrained mode.
        :param str embedding_sharing: Embedding sharing policy.
        :param bool gradient_checkpointing: Gradient checkpointing toggle.
        :param ModelTokenizerConfig | dict[str, Any] | None tokenizer: Tokenizer config.
        :param ModelHFConfig | dict[str, Any] | None hf: HF backbone config.
        :param ModelPretrainedConfig | dict[str, Any] | None pretrained: Pretrained source config.
        :param ModelGeneratorConfig | dict[str, Any] | None generator: Generator shape overrides.
        :param ModelRopeConfig | dict[str, Any] | None rope: RoPE config.
        :param ModelDropoutConfig | dict[str, Any] | None dropout: Dropout config.
        :param Any legacy_kwargs: Optional legacy flat kwargs.
        :raises TypeError: If unknown kwargs are provided.
        """
        tokenizer_cfg = _coerce_subconfig(tokenizer, ModelTokenizerConfig, field_name="tokenizer")
        hf_cfg = _coerce_subconfig(hf, ModelHFConfig, field_name="hf")
        pretrained_cfg = _coerce_subconfig(pretrained, ModelPretrainedConfig, field_name="pretrained")
        generator_cfg = _coerce_subconfig(generator, ModelGeneratorConfig, field_name="generator")
        rope_cfg = _coerce_subconfig(rope, ModelRopeConfig, field_name="rope")
        dropout_cfg = _coerce_subconfig(dropout, ModelDropoutConfig, field_name="dropout")

        sub_updates: dict[str, dict[str, Any]] = {
            "tokenizer": {},
            "hf": {},
            "pretrained": {},
            "generator": {},
            "rope": {},
            "dropout": {},
        }
        unknown: list[str] = []
        for key, value in legacy_kwargs.items():
            mapped = self._LEGACY_MAP.get(str(key))
            if mapped is None and "." in str(key):
                mapped = str(key)
            if mapped is None:
                unknown.append(str(key))
                continue
            if "." in mapped:
                root, child = mapped.split(".", 1)
                if root in sub_updates:
                    sub_updates[root][child] = value
                else:
                    unknown.append(str(key))
            else:
                if mapped == "profile":
                    profile = value
                elif mapped == "backbone_type":
                    backbone_type = value
                elif mapped == "from_scratch":
                    from_scratch = value
                elif mapped == "embedding_sharing":
                    embedding_sharing = value
                elif mapped == "gradient_checkpointing":
                    gradient_checkpointing = value
                else:
                    unknown.append(str(key))

        if unknown:
            unknown_rendered = ", ".join(sorted(unknown))
            raise TypeError(f"ModelConfig.__init__ got unexpected keyword argument(s): {unknown_rendered}")

        if sub_updates["tokenizer"]:
            tokenizer_cfg = _apply_dotted_updates(tokenizer_cfg, sub_updates["tokenizer"])
        if sub_updates["hf"]:
            hf_cfg = _apply_dotted_updates(hf_cfg, sub_updates["hf"])
        if sub_updates["pretrained"]:
            pretrained_cfg = _apply_dotted_updates(pretrained_cfg, sub_updates["pretrained"])
        if sub_updates["generator"]:
            generator_cfg = _apply_dotted_updates(generator_cfg, sub_updates["generator"])
        if sub_updates["rope"]:
            rope_cfg = _apply_dotted_updates(rope_cfg, sub_updates["rope"])
        if sub_updates["dropout"]:
            dropout_cfg = _apply_dotted_updates(dropout_cfg, sub_updates["dropout"])

        object.__setattr__(self, "profile", str(profile))
        object.__setattr__(self, "backbone_type", str(backbone_type))
        object.__setattr__(self, "from_scratch", bool(from_scratch))
        object.__setattr__(self, "embedding_sharing", str(embedding_sharing))
        object.__setattr__(self, "gradient_checkpointing", bool(gradient_checkpointing))
        object.__setattr__(self, "tokenizer", tokenizer_cfg)
        object.__setattr__(self, "hf", hf_cfg)
        object.__setattr__(self, "pretrained", pretrained_cfg)
        object.__setattr__(self, "generator", generator_cfg)
        object.__setattr__(self, "rope", rope_cfg)
        object.__setattr__(self, "dropout", dropout_cfg)

    def __getattr__(self, name: str) -> Any:
        """Provide runtime read compatibility for legacy flat attributes.

        :param str name: Attribute name.
        :return Any: Legacy flat value when mapped.
        """
        path = self._LEGACY_MAP.get(str(name))
        if path is None:
            raise AttributeError(name)
        return _nested_get(self, path)


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

    _LEGACY_MAP = {
        "dataset_name": "source.dataset_name",
        "dataset_config_name": "source.dataset_config_name",
        "data_files": "source.data_files",
        "load_from_disk": "source.load_from_disk",
        "train_split": "source.train_split",
        "text_column_name": "source.text_column_name",
        "streaming": "source.streaming",
        "shuffle_buffer_size": "source.shuffle_buffer_size",
        "pack_sequences": "packing.enabled",
        "max_seq_length": "packing.max_seq_length",
        "block_cross_document_attention": "packing.block_cross_document_attention",
    }

    def __init__(
        self,
        source: DataSourceConfig | dict[str, Any] | None = None,
        packing: DataPackingConfig | dict[str, Any] | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        """Initialize data config while accepting legacy flat kwargs.

        :param DataSourceConfig | dict[str, Any] | None source: Data source config.
        :param DataPackingConfig | dict[str, Any] | None packing: Packing config.
        :param Any legacy_kwargs: Optional legacy flat kwargs.
        :raises TypeError: If unknown kwargs are provided.
        """
        source_cfg = _coerce_subconfig(source, DataSourceConfig, field_name="source")
        packing_cfg = _coerce_subconfig(packing, DataPackingConfig, field_name="packing")

        source_updates: dict[str, Any] = {}
        packing_updates: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in legacy_kwargs.items():
            mapped = self._LEGACY_MAP.get(str(key))
            if mapped is None and "." in str(key):
                mapped = str(key)
            if mapped is None:
                unknown.append(str(key))
                continue
            if mapped.startswith("source."):
                source_updates[mapped.split(".", 1)[1]] = value
            elif mapped.startswith("packing."):
                packing_updates[mapped.split(".", 1)[1]] = value
            else:
                unknown.append(str(key))

        if unknown:
            unknown_rendered = ", ".join(sorted(unknown))
            raise TypeError(f"DataConfig.__init__ got unexpected keyword argument(s): {unknown_rendered}")

        if source_updates:
            source_cfg = _apply_dotted_updates(source_cfg, source_updates)
        if packing_updates:
            packing_cfg = _apply_dotted_updates(packing_cfg, packing_updates)

        object.__setattr__(self, "source", source_cfg)
        object.__setattr__(self, "packing", packing_cfg)

    def __getattr__(self, name: str) -> Any:
        """Provide runtime read compatibility for legacy flat attributes.

        :param str name: Attribute name.
        :return Any: Legacy flat value when mapped.
        """
        path = self._LEGACY_MAP.get(str(name))
        if path is None:
            raise AttributeError(name)
        return _nested_get(self, path)


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
    mask_token_prob: float = field(default=0.8)
    random_token_prob: float = field(default=0.1)
    mlm_max_ngram: int = field(default=1)
    sampling_temperature: float = field(default=1.0)
    gen_loss_weight: float = field(default=1.0)
    disc_loss_weight: float = field(default=50.0)


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

    _LEGACY_MAP = {
        "dataloader_num_workers": "dataloader.num_workers",
        "dataloader_pin_memory": "dataloader.pin_memory",
        "torch_compile": "compile.enabled",
        "torch_compile_mode": "compile.mode",
        "torch_compile_scope": "compile.scope",
        "torch_compile_backend": "compile.backend",
        "mlm_probability": "objective.mlm_probability",
        "mask_token_prob": "objective.mask_token_prob",
        "random_token_prob": "objective.random_token_prob",
        "mlm_max_ngram": "objective.mlm_max_ngram",
        "sampling_temperature": "objective.sampling_temperature",
        "gen_loss_weight": "objective.gen_loss_weight",
        "disc_loss_weight": "objective.disc_loss_weight",
        "output_dir": "checkpoint.output_dir",
        "overwrite_output_dir": "checkpoint.overwrite_output_dir",
        "save_steps": "checkpoint.save_steps",
        "save_total_limit": "checkpoint.save_total_limit",
        "resume_from_checkpoint": "checkpoint.resume_from_checkpoint",
        "resume_data_strategy": "checkpoint.resume_data_strategy",
        "resume_replay_max_micro_batches": "checkpoint.resume_replay_max_micro_batches",
        "export_hf_final": "checkpoint.export_hf_final",
    }

    _LEGACY_DYNAMIC_DEFAULTS = {
        "learning_rate": 5e-4,
        "generator_learning_rate": -1.0,
        "discriminator_learning_rate": -1.0,
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "warmup_steps": 1_000,
        "lr_scheduler_type": "linear",
        "max_grad_norm": 1.0,
        "project_name": "deberta-train",
        "run_name": None,
        "logging_steps": 50,
        "report_to": "none",
        "wandb_watch": "gradients",
        "wandb_watch_log_freq": 100,
        "debug_metrics": False,
        "logging_output_dir": None,
    }

    def __init__(
        self,
        seed: int = 42,
        max_steps: int = 10_000,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        token_weighted_gradient_accumulation: bool = True,
        mixed_precision: str = "bf16",
        tf32: bool = True,
        sdpa_kernel: str = "auto",
        decoupled_training: bool = True,
        dataloader: TrainDataloaderConfig | dict[str, Any] | None = None,
        compile: TrainCompileConfig | dict[str, Any] | None = None,
        objective: TrainObjectiveConfig | dict[str, Any] | None = None,
        checkpoint: TrainCheckpointConfig | dict[str, Any] | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        """Initialize train config while accepting legacy flat kwargs.

        :param int seed: Random seed.
        :param int max_steps: Max training steps.
        :param int per_device_train_batch_size: Micro-batch size.
        :param int gradient_accumulation_steps: Gradient accumulation steps.
        :param bool token_weighted_gradient_accumulation: Token-weighted GA toggle.
        :param str mixed_precision: Mixed precision mode.
        :param bool tf32: TF32 toggle.
        :param str sdpa_kernel: SDPA kernel policy.
        :param bool decoupled_training: Decoupled training toggle.
        :param TrainDataloaderConfig | dict[str, Any] | None dataloader: Dataloader config.
        :param TrainCompileConfig | dict[str, Any] | None compile: Compile config.
        :param TrainObjectiveConfig | dict[str, Any] | None objective: Objective config.
        :param TrainCheckpointConfig | dict[str, Any] | None checkpoint: Checkpoint config.
        :param Any legacy_kwargs: Optional legacy flat kwargs.
        :raises TypeError: If unknown kwargs are provided.
        """
        dataloader_cfg = _coerce_subconfig(dataloader, TrainDataloaderConfig, field_name="dataloader")
        compile_cfg = _coerce_subconfig(compile, TrainCompileConfig, field_name="compile")
        objective_cfg = _coerce_subconfig(objective, TrainObjectiveConfig, field_name="objective")
        checkpoint_cfg = _coerce_subconfig(checkpoint, TrainCheckpointConfig, field_name="checkpoint")

        dataloader_updates: dict[str, Any] = {}
        compile_updates: dict[str, Any] = {}
        objective_updates: dict[str, Any] = {}
        checkpoint_updates: dict[str, Any] = {}
        dynamic_overrides: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in legacy_kwargs.items():
            mapped = self._LEGACY_MAP.get(str(key))
            if mapped is None and "." in str(key):
                mapped = str(key)

            if str(key) in self._LEGACY_DYNAMIC_DEFAULTS:
                dynamic_overrides[str(key)] = value
                continue

            if mapped is None:
                unknown.append(str(key))
                continue
            if mapped.startswith("dataloader."):
                dataloader_updates[mapped.split(".", 1)[1]] = value
            elif mapped.startswith("compile."):
                compile_updates[mapped.split(".", 1)[1]] = value
            elif mapped.startswith("objective."):
                objective_updates[mapped.split(".", 1)[1]] = value
            elif mapped.startswith("checkpoint."):
                checkpoint_updates[mapped.split(".", 1)[1]] = value
            elif mapped == "seed":
                seed = value
            elif mapped == "max_steps":
                max_steps = value
            elif mapped == "per_device_train_batch_size":
                per_device_train_batch_size = value
            elif mapped == "gradient_accumulation_steps":
                gradient_accumulation_steps = value
            elif mapped == "token_weighted_gradient_accumulation":
                token_weighted_gradient_accumulation = value
            elif mapped == "mixed_precision":
                mixed_precision = value
            elif mapped == "tf32":
                tf32 = value
            elif mapped == "sdpa_kernel":
                sdpa_kernel = value
            elif mapped == "decoupled_training":
                decoupled_training = value
            else:
                unknown.append(str(key))

        if unknown:
            unknown_rendered = ", ".join(sorted(unknown))
            raise TypeError(f"TrainConfig.__init__ got unexpected keyword argument(s): {unknown_rendered}")

        if dataloader_updates:
            dataloader_cfg = _apply_dotted_updates(dataloader_cfg, dataloader_updates)
        if compile_updates:
            compile_cfg = _apply_dotted_updates(compile_cfg, compile_updates)
        if objective_updates:
            objective_cfg = _apply_dotted_updates(objective_cfg, objective_updates)
        if checkpoint_updates:
            checkpoint_cfg = _apply_dotted_updates(checkpoint_cfg, checkpoint_updates)

        object.__setattr__(self, "seed", int(seed))
        object.__setattr__(self, "max_steps", int(max_steps))
        object.__setattr__(self, "per_device_train_batch_size", int(per_device_train_batch_size))
        object.__setattr__(self, "gradient_accumulation_steps", int(gradient_accumulation_steps))
        object.__setattr__(self, "token_weighted_gradient_accumulation", token_weighted_gradient_accumulation)
        object.__setattr__(self, "mixed_precision", str(mixed_precision))
        object.__setattr__(self, "tf32", tf32)
        object.__setattr__(self, "sdpa_kernel", str(sdpa_kernel))
        object.__setattr__(self, "decoupled_training", decoupled_training)
        object.__setattr__(self, "dataloader", dataloader_cfg)
        object.__setattr__(self, "compile", compile_cfg)
        object.__setattr__(self, "objective", objective_cfg)
        object.__setattr__(self, "checkpoint", checkpoint_cfg)

        for key, value in dynamic_overrides.items():
            object.__setattr__(self, str(key), value)

    def __getattr__(self, name: str) -> Any:
        """Provide runtime read compatibility for legacy flat attributes.

        :param str name: Attribute name.
        :return Any: Legacy flat value when mapped.
        """
        path = self._LEGACY_MAP.get(str(name))
        if path is not None:
            return _nested_get(self, path)
        if str(name) in self._LEGACY_DYNAMIC_DEFAULTS:
            return self._LEGACY_DYNAMIC_DEFAULTS[str(name)]
        raise AttributeError(name)


@dataclass(frozen=True)
class OptimLRConfig:
    """Learning-rate values for optimizer setup."""

    base: float = field(default=5e-4)
    generator: float = field(default=-1.0)
    discriminator: float = field(default=-1.0)


@dataclass(frozen=True)
class OptimAdamConfig:
    """Adam/AdamW parameter group settings."""

    beta1: float = field(default=0.9)
    beta2: float = field(default=0.999)
    epsilon: float = field(default=1e-8)


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

    def __init__(
        self,
        lr: OptimLRConfig | dict[str, Any] | None = None,
        adam: OptimAdamConfig | dict[str, Any] | None = None,
        scheduler: OptimSchedulerConfig | dict[str, Any] | None = None,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        **legacy_kwargs: Any,
    ) -> None:
        """Initialize optimizer config with nested-dict coercion.

        :param OptimLRConfig | dict[str, Any] | None lr: LR config.
        :param OptimAdamConfig | dict[str, Any] | None adam: Adam config.
        :param OptimSchedulerConfig | dict[str, Any] | None scheduler: Scheduler config.
        :param float weight_decay: Weight decay.
        :param float max_grad_norm: Gradient clipping norm.
        :param Any legacy_kwargs: Optional legacy aliases.
        :raises TypeError: If unknown kwargs are provided.
        """
        lr_cfg = _coerce_subconfig(lr, OptimLRConfig, field_name="lr")
        adam_cfg = _coerce_subconfig(adam, OptimAdamConfig, field_name="adam")
        scheduler_cfg = _coerce_subconfig(scheduler, OptimSchedulerConfig, field_name="scheduler")

        scheduler_updates: dict[str, Any] = {}
        lr_updates: dict[str, Any] = {}
        adam_updates: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in legacy_kwargs.items():
            k = str(key)
            if k == "learning_rate":
                lr_updates["base"] = value
            elif k == "generator_learning_rate":
                lr_updates["generator"] = value
            elif k == "discriminator_learning_rate":
                lr_updates["discriminator"] = value
            elif k == "adam_beta1":
                adam_updates["beta1"] = value
            elif k == "adam_beta2":
                adam_updates["beta2"] = value
            elif k == "adam_epsilon":
                adam_updates["epsilon"] = value
            elif k == "lr_scheduler_type":
                scheduler_updates["type"] = value
            elif k == "warmup_steps":
                scheduler_updates["warmup_steps"] = value
            else:
                unknown.append(k)

        if unknown:
            unknown_rendered = ", ".join(sorted(unknown))
            raise TypeError(f"OptimConfig.__init__ got unexpected keyword argument(s): {unknown_rendered}")

        if lr_updates:
            lr_cfg = _apply_dotted_updates(lr_cfg, lr_updates)
        if adam_updates:
            adam_cfg = _apply_dotted_updates(adam_cfg, adam_updates)
        if scheduler_updates:
            scheduler_cfg = _apply_dotted_updates(scheduler_cfg, scheduler_updates)

        object.__setattr__(self, "lr", lr_cfg)
        object.__setattr__(self, "adam", adam_cfg)
        object.__setattr__(self, "scheduler", scheduler_cfg)
        object.__setattr__(self, "weight_decay", float(weight_decay))
        object.__setattr__(self, "max_grad_norm", float(max_grad_norm))


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
    backend: str = field(default="none")
    wandb: LoggingWandbConfig = field(default_factory=LoggingWandbConfig)
    debug: LoggingDebugConfig = field(default_factory=LoggingDebugConfig)

    def __init__(
        self,
        project_name: str = "deberta-train",
        run_name: str | None = None,
        output_dir: str | None = None,
        logging_steps: int = 50,
        backend: str = "none",
        wandb: LoggingWandbConfig | dict[str, Any] | None = None,
        debug: LoggingDebugConfig | dict[str, Any] | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        """Initialize logging config with nested-dict coercion.

        :param str project_name: Project name.
        :param str | None run_name: Run name.
        :param str | None output_dir: Logging output directory.
        :param int logging_steps: Logging step interval.
        :param str backend: Logging backend when W&B disabled.
        :param LoggingWandbConfig | dict[str, Any] | None wandb: W&B config.
        :param LoggingDebugConfig | dict[str, Any] | None debug: Debug logging config.
        :param Any legacy_kwargs: Optional legacy aliases.
        :raises TypeError: If unknown kwargs are provided.
        """
        wandb_cfg = _coerce_subconfig(wandb, LoggingWandbConfig, field_name="wandb")
        debug_cfg = _coerce_subconfig(debug, LoggingDebugConfig, field_name="debug")

        wandb_updates: dict[str, Any] = {}
        debug_updates: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in legacy_kwargs.items():
            k = str(key)
            if k == "report_to":
                report_to = str(value).strip().lower()
                if report_to == "wandb":
                    wandb_updates["enabled"] = True
                else:
                    wandb_updates["enabled"] = False
                    backend = report_to
            elif k == "wandb_watch":
                wandb_updates["watch"] = value
            elif k == "wandb_watch_log_freq":
                wandb_updates["watch_log_freq"] = value
            elif k == "debug_metrics":
                debug_updates["metrics"] = value
            else:
                unknown.append(k)

        if unknown:
            unknown_rendered = ", ".join(sorted(unknown))
            raise TypeError(f"LoggingConfig.__init__ got unexpected keyword argument(s): {unknown_rendered}")

        if wandb_updates:
            wandb_cfg = _apply_dotted_updates(wandb_cfg, wandb_updates)
        if debug_updates:
            debug_cfg = _apply_dotted_updates(debug_cfg, debug_updates)

        object.__setattr__(self, "project_name", str(project_name))
        object.__setattr__(self, "run_name", run_name if run_name is None else str(run_name))
        object.__setattr__(self, "output_dir", output_dir if output_dir is None else str(output_dir))
        object.__setattr__(self, "logging_steps", int(logging_steps))
        object.__setattr__(self, "backend", str(backend))
        object.__setattr__(self, "wandb", wandb_cfg)
        object.__setattr__(self, "debug", debug_cfg)


@dataclass(frozen=True)
class Config:
    """Top-level training config bundle."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _sync_legacy_train_aliases(
    *, train_cfg: TrainConfig, optim_cfg: OptimConfig, logging_cfg: LoggingConfig
) -> None:
    """Attach dynamic legacy aliases on TrainConfig for runtime compatibility.

    :param TrainConfig train_cfg: Train config instance.
    :param OptimConfig optim_cfg: Optim config instance.
    :param LoggingConfig logging_cfg: Logging config instance.
    """
    object.__setattr__(train_cfg, "learning_rate", float(optim_cfg.lr.base))
    object.__setattr__(train_cfg, "generator_learning_rate", float(optim_cfg.lr.generator))
    object.__setattr__(train_cfg, "discriminator_learning_rate", float(optim_cfg.lr.discriminator))
    object.__setattr__(train_cfg, "weight_decay", float(optim_cfg.weight_decay))
    object.__setattr__(train_cfg, "adam_beta1", float(optim_cfg.adam.beta1))
    object.__setattr__(train_cfg, "adam_beta2", float(optim_cfg.adam.beta2))
    object.__setattr__(train_cfg, "adam_epsilon", float(optim_cfg.adam.epsilon))
    object.__setattr__(train_cfg, "warmup_steps", int(optim_cfg.scheduler.warmup_steps))
    object.__setattr__(train_cfg, "lr_scheduler_type", str(optim_cfg.scheduler.type))
    object.__setattr__(train_cfg, "max_grad_norm", float(optim_cfg.max_grad_norm))

    report_to = "wandb" if bool(logging_cfg.wandb.enabled) else str(logging_cfg.backend).strip().lower()
    object.__setattr__(train_cfg, "project_name", str(logging_cfg.project_name))
    object.__setattr__(train_cfg, "run_name", logging_cfg.run_name)
    object.__setattr__(train_cfg, "logging_output_dir", logging_cfg.output_dir)
    object.__setattr__(train_cfg, "logging_steps", int(logging_cfg.logging_steps))
    object.__setattr__(train_cfg, "report_to", str(report_to))
    object.__setattr__(train_cfg, "wandb_watch", str(logging_cfg.wandb.watch))
    object.__setattr__(train_cfg, "wandb_watch_log_freq", int(logging_cfg.wandb.watch_log_freq))
    object.__setattr__(train_cfg, "debug_metrics", bool(logging_cfg.debug.metrics))


def _explicit_fields(cfg_obj: Any) -> set[str]:
    """Return explicitly provided field names attached to a config object.

    :param Any cfg_obj: Config dataclass object.
    :return set[str]: Explicitly provided field names.
    """
    raw = getattr(cfg_obj, "_explicit_fields", None)
    if raw is None:
        return set()
    if isinstance(raw, set):
        return {str(x) for x in raw}
    if isinstance(raw, (list, tuple, frozenset)):
        return {str(x) for x in raw}
    return set()


def _mark_explicit_fields(cfg_obj: Any, explicit_fields: set[str]) -> None:
    """Attach explicit-field metadata to a config dataclass.

    :param Any cfg_obj: Config dataclass object.
    :param set[str] explicit_fields: Explicit field names.
    """
    object.__setattr__(cfg_obj, "_explicit_fields", set(str(x) for x in explicit_fields))


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


def validate_model_config(cfg: ModelConfig) -> None:
    """Validate model config semantics and normalize constrained values.

    :param ModelConfig cfg: Model configuration.
    """
    _cfg_set(
        cfg, "backbone_type", _ensure_choice("model.backbone_type", cfg.backbone_type, _BACKBONE_CHOICES)
    )
    _cfg_set(cfg, "profile", _ensure_choice("model.profile", cfg.profile, _MODEL_PROFILE_CHOICES))
    _cfg_set(
        cfg,
        "embedding_sharing",
        _ensure_choice("model.embedding_sharing", cfg.embedding_sharing, _EMBED_SHARING_CHOICES),
    )

    _cfg_set(cfg.hf, "attention_kernel", _normalize_hf_attention_kernel(cfg.hf.attention_kernel))
    _cfg_set(
        cfg.hf, "model_size", _ensure_choice("model.hf.model_size", cfg.hf.model_size, _HF_MODEL_SIZE_CHOICES)
    )

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
    if float(cfg.rope.rotary_pct) <= 0.0 or float(cfg.rope.rotary_pct) > 1.0:
        raise ValueError("model.rope.rotary_pct must be in (0, 1].")
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

        rope_changed = asdict(cfg.rope) != asdict(defaults.rope)
        if rope_changed:
            raise ValueError("These options are only valid when model.backbone_type='rope': model.rope.*")
    else:
        if cfg.hf.attention_kernel != defaults.hf.attention_kernel:
            warnings.warn(
                "model.hf.attention_kernel only applies when model.backbone_type='hf_deberta_v2'. "
                f"Current value ({cfg.hf.attention_kernel!r}) has no effect on the rope backbone.",
                UserWarning,
                stacklevel=2,
            )
        if cfg.hf.max_position_embeddings is not None:
            warnings.warn(
                "model.hf.max_position_embeddings only applies when model.backbone_type='hf_deberta_v2'. "
                f"Current value ({cfg.hf.max_position_embeddings!r}) has no effect on the rope backbone.",
                UserWarning,
                stacklevel=2,
            )
        if cfg.hf.model_size != defaults.hf.model_size:
            warnings.warn(
                "model.hf.model_size only applies when model.backbone_type='hf_deberta_v2'. "
                f"Current value ({cfg.hf.model_size!r}) has no effect on the rope backbone.",
                UserWarning,
                stacklevel=2,
            )

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
            if pct <= 0.0 or pct > 1.0:
                raise ValueError("model.rope.pretrained.rotary_pct must be in (0, 1] when provided.")
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
            "data.packing.block_cross_document_attention builds dense O(S^2) pairwise masks. "
            f"Configured data.packing.max_seq_length={int(pack.max_seq_length)} may be expensive; "
            "consider reducing sequence length or disabling data.packing.block_cross_document_attention "
            f"until sparse/segment-aware attention support lands (warning threshold: {int(_DENSE_DOC_BLOCK_WARN_SEQ_LEN)}).",
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

    if cfg.checkpoint.output_dir is not None and not str(cfg.checkpoint.output_dir).strip():
        _cfg_set(cfg.checkpoint, "output_dir", None)
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

    mlm = float(cfg.objective.mlm_probability)
    if mlm <= 0.0 or mlm >= 1.0:
        raise ValueError("train.objective.mlm_probability must be in (0, 1).")
    mask_p = float(cfg.objective.mask_token_prob)
    rand_p = float(cfg.objective.random_token_prob)
    if mask_p < 0.0 or rand_p < 0.0 or (mask_p + rand_p) > 1.0:
        raise ValueError(
            "Invalid masking probabilities: train.objective.mask_token_prob + "
            "train.objective.random_token_prob must be <= 1."
        )
    if float(cfg.objective.sampling_temperature) <= 0.0:
        raise ValueError("train.objective.sampling_temperature must be > 0.")
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

    if float(cfg.lr.base) <= 0.0:
        raise ValueError("optim.lr.base must be > 0.")
    if float(cfg.lr.generator) != -1.0 and float(cfg.lr.generator) <= 0.0:
        raise ValueError("optim.lr.generator must be -1 (inherit) or > 0.")
    if float(cfg.lr.discriminator) != -1.0 and float(cfg.lr.discriminator) <= 0.0:
        raise ValueError("optim.lr.discriminator must be -1 (inherit) or > 0.")
    if float(cfg.weight_decay) < 0.0:
        raise ValueError("optim.weight_decay must be >= 0.")
    if float(cfg.max_grad_norm) < 0.0:
        raise ValueError("optim.max_grad_norm must be >= 0.")
    if int(cfg.scheduler.warmup_steps) < 0:
        raise ValueError("optim.scheduler.warmup_steps must be >= 0.")


def validate_logging_config(cfg: LoggingConfig) -> None:
    """Validate logging and tracker configuration.

    :param LoggingConfig cfg: Logging config.
    """
    _cfg_set(cfg, "backend", _ensure_choice("logging.backend", cfg.backend, _LOGGING_BACKEND_CHOICES))
    _cfg_set(cfg.wandb, "watch", _normalize_wandb_watch(cfg.wandb.watch))

    if not str(cfg.project_name).strip():
        raise ValueError("logging.project_name must be non-empty.")
    if cfg.output_dir is not None and not str(cfg.output_dir).strip():
        _cfg_set(cfg, "output_dir", None)
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
    logging_cfg: LoggingConfig | None = None,
) -> None:
    """Validate options tied to workflow support (for example, eval mode availability).

    :param DataConfig data_cfg: Data configuration.
    :param TrainConfig train_cfg: Training configuration.
    :param ModelConfig | None model_cfg: Optional model configuration.
    :param OptimConfig | None optim_cfg: Optional optimizer configuration.
    :param LoggingConfig | None logging_cfg: Optional logging configuration.
    """
    sdpa_policy = str(train_cfg.sdpa_kernel).strip().lower()
    if (
        bool(data_cfg.packing.enabled)
        and bool(data_cfg.packing.block_cross_document_attention)
        and sdpa_policy == "flash"
    ):
        raise ValueError(
            "train.sdpa_kernel=flash is not supported with data.packing.enabled=true. "
            "Packed batches may require 3D document-blocking attention masks that are incompatible "
            "with strict flash SDPA kernels. Use train.sdpa_kernel=auto|mem_efficient|math instead."
        )

    if model_cfg is not None:
        backbone_type = str(model_cfg.backbone_type).strip().lower()
        attn_impl = str(model_cfg.rope.attention_implementation).strip().lower()
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
        if (
            backbone_type != "rope"
            and bool(data_cfg.packing.enabled)
            and bool(data_cfg.packing.block_cross_document_attention)
        ):
            raise ValueError(
                "data.packing.block_cross_document_attention=true is only supported with model.backbone_type='rope'. "
                "Use data.packing.block_cross_document_attention=false or switch to model.backbone_type='rope'."
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
                "shared embedding parameters would be stepped in both generator and discriminator phases. "
                "Use embedding_sharing='gdes' or 'none' for decoupled training."
            )

    if logging_cfg is not None:
        # backend already validated as none|tensorboard.
        # Effective backend is wandb when enabled; otherwise backend.
        _ = "wandb" if bool(logging_cfg.wandb.enabled) else str(logging_cfg.backend).strip().lower()


def apply_profile_defaults(
    *,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    optim_cfg: OptimConfig | None = None,
) -> None:
    """Apply profile/backbone-specific defaults while preserving explicit values.

    :param ModelConfig model_cfg: Model config to update in-place.
    :param TrainConfig train_cfg: Train config to update in-place.
    :param OptimConfig | None optim_cfg: Optim config to update in-place.
    """
    # Explicit-field metadata is populated from YAML + dotted CLI flags and is
    # checked before equality comparisons so explicit values are preserved even
    # when they match raw dataclass defaults.
    explicit_model_fields = _explicit_fields(model_cfg)
    explicit_train_fields = _explicit_fields(train_cfg)
    explicit_optim_fields = _explicit_fields(optim_cfg) if optim_cfg is not None else set()

    profile = str(model_cfg.profile).strip().lower()
    model_defaults = ModelConfig()
    train_defaults = TrainConfig()
    optim_defaults = OptimConfig()

    if profile == "deberta_v3_parity":
        if "backbone_type" not in explicit_model_fields and str(model_cfg.backbone_type) == str(
            model_defaults.backbone_type
        ):
            _cfg_set(model_cfg, "backbone_type", "hf_deberta_v2")

        if "embedding_sharing" not in explicit_model_fields and str(model_cfg.embedding_sharing) == str(
            model_defaults.embedding_sharing
        ):
            _cfg_set(model_cfg, "embedding_sharing", "gdes")

        if "hf.attention_kernel" not in explicit_model_fields and str(model_cfg.hf.attention_kernel) == str(
            model_defaults.hf.attention_kernel
        ):
            _cfg_set(model_cfg.hf, "attention_kernel", "dynamic")

    if str(model_cfg.backbone_type).strip().lower() == "hf_deberta_v2":
        if "objective.mask_token_prob" not in explicit_train_fields and float(
            train_cfg.objective.mask_token_prob
        ) == float(train_defaults.objective.mask_token_prob):
            _cfg_set(train_cfg.objective, "mask_token_prob", 1.0)
        if "objective.random_token_prob" not in explicit_train_fields and float(
            train_cfg.objective.random_token_prob
        ) == float(train_defaults.objective.random_token_prob):
            _cfg_set(train_cfg.objective, "random_token_prob", 0.0)
        if "objective.disc_loss_weight" not in explicit_train_fields and float(
            train_cfg.objective.disc_loss_weight
        ) == float(train_defaults.objective.disc_loss_weight):
            _cfg_set(train_cfg.objective, "disc_loss_weight", 10.0)
        if optim_cfg is not None:
            if "adam.epsilon" not in explicit_optim_fields and float(optim_cfg.adam.epsilon) == float(
                optim_defaults.adam.epsilon
            ):
                _cfg_set(optim_cfg.adam, "epsilon", 1e-6)
            if "scheduler.warmup_steps" not in explicit_optim_fields and int(
                optim_cfg.scheduler.warmup_steps
            ) == int(optim_defaults.scheduler.warmup_steps):
                _cfg_set(optim_cfg.scheduler, "warmup_steps", 10_000)
        if "token_weighted_gradient_accumulation" not in explicit_train_fields and bool(
            train_cfg.token_weighted_gradient_accumulation
        ) == bool(train_defaults.token_weighted_gradient_accumulation):
            _cfg_set(train_cfg, "token_weighted_gradient_accumulation", True)


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
    expected_keys = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - expected_keys)
    if unknown:
        unknown_str = ", ".join(unknown)
        raise ValueError(
            f"Unsupported {config_name} keys in {source}: {unknown_str}. "
            "This snapshot was produced by an older pre-release schema; "
            "backward resume/export compatibility is not guaranteed before stable release."
        )
    missing = sorted(expected_keys - set(raw))
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Missing required {config_name} keys in {source}: {missing_str}. "
            "This snapshot does not match the current config schema."
        )

    try:
        return cls(**raw)
    except TypeError as e:
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


def _collect_leaf_paths(value: Any, *, prefix: str = "") -> set[str]:
    """Collect dotted leaf paths from a nested mapping.

    :param Any value: Nested mapping value.
    :param str prefix: Prefix path.
    :return set[str]: Dotted leaf paths.
    """
    if isinstance(value, dict):
        out: set[str] = set()
        for key, item in value.items():
            key_s = str(key)
            child = f"{prefix}.{key_s}" if prefix else key_s
            out.update(_collect_leaf_paths(item, prefix=child))
        return out
    if prefix:
        return {prefix}
    return set()


def _legacy_key_suggestion(section_name: str, key: str) -> str | None:
    """Return actionable migration suggestion for an unknown key.

    :param str section_name: Section path.
    :param str key: Unknown key.
    :return str | None: Suggested replacement path.
    """
    section = str(section_name)
    k = str(key)
    model_map = {
        "tokenizer_name_or_path": "model.tokenizer.name_or_path",
        "tokenizer_allow_vocab_resize": "model.tokenizer.allow_vocab_resize",
        "tokenizer_vocab_target": "model.tokenizer.vocab_target",
        "tokenizer_vocab_multiple": "model.tokenizer.vocab_multiple",
        "hf_model_size": "model.hf.model_size",
        "hf_attention_kernel": "model.hf.attention_kernel",
        "hf_max_position_embeddings": "model.hf.max_position_embeddings",
        "pretrained_discriminator_path": "model.pretrained.discriminator_path",
        "pretrained_generator_path": "model.pretrained.generator_path",
        "hidden_dropout_prob": "model.dropout.hidden_prob",
        "attention_probs_dropout_prob": "model.dropout.attention_probs_prob",
        "generator_num_hidden_layers": "model.generator.num_hidden_layers",
        "generator_hidden_size": "model.generator.hidden_size",
        "generator_intermediate_size": "model.generator.intermediate_size",
        "generator_num_attention_heads": "model.generator.num_attention_heads",
    }
    data_map = {
        "dataset_name": "data.source.dataset_name",
        "dataset_config_name": "data.source.dataset_config_name",
        "data_files": "data.source.data_files",
        "load_from_disk": "data.source.load_from_disk",
        "train_split": "data.source.train_split",
        "text_column_name": "data.source.text_column_name",
        "streaming": "data.source.streaming",
        "shuffle_buffer_size": "data.source.shuffle_buffer_size",
        "pack_sequences": "data.packing.enabled",
        "max_seq_length": "data.packing.max_seq_length",
        "block_cross_document_attention": "data.packing.block_cross_document_attention",
    }
    train_map = {
        "output_dir": "train.checkpoint.output_dir",
        "overwrite_output_dir": "train.checkpoint.overwrite_output_dir",
        "save_steps": "train.checkpoint.save_steps",
        "save_total_limit": "train.checkpoint.save_total_limit",
        "resume_from_checkpoint": "train.checkpoint.resume_from_checkpoint",
        "resume_data_strategy": "train.checkpoint.resume_data_strategy",
        "resume_replay_max_micro_batches": "train.checkpoint.resume_replay_max_micro_batches",
        "export_hf_final": "train.checkpoint.export_hf_final",
        "dataloader_num_workers": "train.dataloader.num_workers",
        "dataloader_pin_memory": "train.dataloader.pin_memory",
        "torch_compile": "train.compile.enabled",
        "torch_compile_mode": "train.compile.mode",
        "torch_compile_scope": "train.compile.scope",
        "torch_compile_backend": "train.compile.backend",
        "mlm_probability": "train.objective.mlm_probability",
        "mask_token_prob": "train.objective.mask_token_prob",
        "random_token_prob": "train.objective.random_token_prob",
        "mlm_max_ngram": "train.objective.mlm_max_ngram",
        "sampling_temperature": "train.objective.sampling_temperature",
        "gen_loss_weight": "train.objective.gen_loss_weight",
        "disc_loss_weight": "train.objective.disc_loss_weight",
        "project_name": "logging.project_name",
        "run_name": "logging.run_name",
        "report_to": "logging.wandb.enabled + logging.backend",
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
    if section == "model":
        if k in model_map:
            return model_map[k]
        rope_like = {
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "hidden_act",
            "rope_theta",
            "rotary_pct",
            "use_absolute_position_embeddings",
            "max_position_embeddings",
            "type_vocab_size",
            "norm_arch",
            "norm_eps",
            "keel_alpha_init",
            "keel_alpha_learnable",
            "attention_implementation",
            "ffn_type",
            "use_bias",
            "swiglu_adjust_intermediate",
            "initializer_range",
        }
        if k in rope_like:
            return f"model.rope.{k}"
        if k.startswith("pretrained_"):
            return "model.rope.pretrained.<field>"
    if section == "data" and k in data_map:
        return data_map[k]
    if section == "train" and k in train_map:
        return train_map[k]
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


def _resolve_variables(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve `$variables.*` references in a config mapping.

    :param dict[str, Any] data: Raw config mapping.
    :raises ValueError: If a variable reference is missing or circular.
    :return dict[str, Any]: Mapping with variables expanded and removed.
    """
    raw_vars = data.get("variables") or {}
    if not isinstance(raw_vars, dict):
        raise ValueError("variables must be a mapping if provided.")

    resolved: dict[str, Any] = {}
    resolving: set[str] = set()

    def _lookup_var(path: str) -> Any:
        """Resolve one variable path with cycle detection.

        :param str path: Variable path under ``variables``.
        :return Any: Resolved variable value.
        """
        if path in resolved:
            return resolved[path]
        if path in resolving:
            cycle = " -> ".join(list(resolving) + [path])
            raise ValueError(f"Circular variable reference: {cycle}")

        cur: Any = raw_vars
        for part in str(path).split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise ValueError(f"Unknown variable reference: variables.{path}")
            cur = cur[part]

        resolving.add(path)
        value = _resolve_value(cur)
        resolving.remove(path)
        resolved[path] = value
        return value

    def _sub_var(match: re.Match[str]) -> str:
        """Render one regex variable match as string.

        :param re.Match[str] match: Regex match object.
        :return str: String replacement value.
        """
        return str(_lookup_var(match.group(1)))

    def _resolve_value(value: Any) -> Any:
        """Recursively resolve variable references inside nested values.

        :param Any value: Raw nested value.
        :return Any: Resolved nested value.
        """
        if isinstance(value, dict):
            return {k: _resolve_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve_value(v) for v in value]
        if isinstance(value, str):
            full = _VAR_FULL_RE.fullmatch(value)
            if full:
                return _lookup_var(full.group(1))
            out = _VAR_INLINE_RE.sub(_sub_var, value)
            out = _VAR_BRACE_RE.sub(_sub_var, out)
            remaining = _VAR_SUSPICIOUS_RE.findall(out)
            if remaining:
                warnings.warn(
                    f"String contains unresolved variable-like patterns: {remaining}. "
                    "Use {$variables.name} or ${variables.name} for inline substitution.",
                    stacklevel=2,
                )
            return out
        return value

    def _collect_var_leaf_paths(prefix: str, value: Any) -> list[str]:
        """Collect dotted variable leaf paths from a nested variable mapping.

        :param str prefix: Current path prefix.
        :param Any value: Nested mapping value.
        :return list[str]: Dotted leaf paths.
        """
        if isinstance(value, dict):
            out: list[str] = []
            for key, item in value.items():
                part = str(key).strip()
                child = f"{prefix}.{part}" if prefix else part
                out.extend(_collect_var_leaf_paths(child, item))
            return out
        return [prefix]

    for var_path in _collect_var_leaf_paths("", raw_vars):
        _lookup_var(var_path)

    return {k: _resolve_value(v) for k, v in data.items() if k != "variables"}


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
    target_t, allows_none = _unwrap_optional_type(field_type)
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

    _mark_explicit_fields(model_cfg, _collect_leaf_paths(section_maps.get("model", {})))
    _mark_explicit_fields(data_cfg, _collect_leaf_paths(section_maps.get("data", {})))
    _mark_explicit_fields(train_cfg, _collect_leaf_paths(section_maps.get("train", {})))
    _mark_explicit_fields(optim_cfg, _collect_leaf_paths(section_maps.get("optim", {})))
    _mark_explicit_fields(logging_cfg, _collect_leaf_paths(section_maps.get("logging", {})))

    cfg = Config(
        model=model_cfg,
        data=data_cfg,
        train=train_cfg,
        optim=optim_cfg,
        logging=logging_cfg,
    )
    _sync_legacy_train_aliases(train_cfg=cfg.train, optim_cfg=cfg.optim, logging_cfg=cfg.logging)
    return cfg


def _unwrap_optional_type(field_type: Any) -> tuple[Any, bool]:
    """Unwrap Optional[T] annotations.

    :param Any field_type: Raw type annotation.
    :return tuple[Any, bool]: Unwrapped type and whether None is allowed.
    """
    origin = get_origin(field_type)
    if origin in {Union, types.UnionType}:
        args = get_args(field_type)
        allows_none = any(a is type(None) for a in args)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], allows_none
    return field_type, False


def _coerce_override_value(raw: str, field_type: Any) -> Any:
    """Cast a dotted-override raw string to the target dataclass field type.

    :param str raw: Raw override value string.
    :param Any field_type: Dataclass field type annotation.
    :raises ValueError: If value cannot be parsed as the target type.
    :return Any: Typed override value.
    """
    target_t, allows_none = _unwrap_optional_type(field_type)
    text = str(raw).strip()
    if allows_none and text.lower() in {"none", "null"}:
        return None
    if target_t is bool:
        v = text.lower()
        if v in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Expected a boolean override value, got: {raw!r}")
    if target_t is int:
        return int(text)
    if target_t is float:
        return float(text)
    if target_t is str:
        return text
    return raw


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
    if root not in {f.name for f in fields(Config)}:
        raise ValueError(
            "Unknown override section "
            f"{root!r}; expected one of {', '.join(sorted(f.name for f in fields(Config)))}."
        )

    def _apply_to_obj(obj: Any, remaining: list[str], value_text: str, full_path: str) -> Any:
        """Recursively apply one override path to a dataclass instance.

        :param Any obj: Current dataclass object.
        :param list[str] remaining: Remaining path parts.
        :param str value_text: Raw override value text.
        :param str full_path: Full dotted path for error reporting.
        :return Any: Updated dataclass object.
        """
        key = remaining[0]
        if not hasattr(obj, key):
            raise ValueError(f"Unknown override field {full_path!r}.")
        if len(remaining) == 1:
            type_hints = get_type_hints(type(obj))
            field_type = type_hints.get(key, Any)
            coerced = _coerce_override_value(value_text, field_type)
            return replace(obj, **{key: coerced})
        child = getattr(obj, key)
        new_child = _apply_to_obj(child, remaining[1:], value_text, full_path)
        return replace(obj, **{key: new_child})

    root_obj = getattr(cfg, root)
    new_root = _apply_to_obj(root_obj, parts[1:], raw_value, path)
    explicit_leaf_path = ".".join(parts[1:])
    _mark_explicit_fields(new_root, _explicit_fields(root_obj) | {explicit_leaf_path})
    new_cfg = replace(cfg, **{root: new_root})

    # keep runtime train alias mirror coherent.
    _sync_legacy_train_aliases(
        train_cfg=new_cfg.train,
        optim_cfg=new_cfg.optim,
        logging_cfg=new_cfg.logging,
    )

    return new_cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load, resolve, override, and validate config into a top-level Config object.

    :param str | Path path: YAML/JSON config path.
    :param list[str] | None overrides: Optional dotted overrides.
    :return Config: Validated immutable config object.
    """
    raw, format_name = _load_raw_config_mapping(path)
    resolved_raw = _resolve_variables(raw)
    section_maps = _split_full_sections(resolved_raw, format_name=format_name)
    cfg = _build_config_from_section_mappings(section_maps)
    if overrides:
        for expr in overrides:
            cfg = apply_dotted_override(cfg, expr)
    apply_profile_defaults(model_cfg=cfg.model, train_cfg=cfg.train, optim_cfg=cfg.optim)
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
        logging_cfg=cfg.logging,
    )
    _sync_legacy_train_aliases(train_cfg=cfg.train, optim_cfg=cfg.optim, logging_cfg=cfg.logging)
    return cfg


def load_config_sections(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load strict model/data/train mappings from a YAML or JSON file.

    :param str | Path path: Config path.
    :raises ValueError: If file content is invalid for this config schema.
    :return tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: Parsed section mappings.
    """
    raw, format_name = _load_raw_config_mapping(path)
    resolved_raw = _resolve_variables(raw)
    section_maps = _split_full_sections(resolved_raw, format_name=format_name)

    # Validate section keys/types early while preserving explicit-key semantics.
    _build_config_from_section_mappings(section_maps)

    model_dict = dict(section_maps.get("model", {}) or {})
    data_dict = dict(section_maps.get("data", {}) or {})
    train_dict = dict(section_maps.get("train", {}) or {})
    return model_dict, data_dict, train_dict


def iter_leaf_paths_for_dataclass(cls: type[Any], *, prefix: str = "") -> list[tuple[str, Any]]:
    """List dotted leaf paths and field types for a dataclass type.

    :param type[Any] cls: Dataclass type.
    :param str prefix: Optional prefix.
    :return list[tuple[str, Any]]: Leaf path + type tuples.
    """
    out: list[tuple[str, Any]] = []
    type_hints = get_type_hints(cls)
    for f in fields(cls):
        path = f"{prefix}.{f.name}" if prefix else str(f.name)
        field_type = type_hints.get(f.name, f.type)
        target_t, _allows_none = _unwrap_optional_type(field_type)
        if dataclasses.is_dataclass(target_t):
            out.extend(iter_leaf_paths_for_dataclass(target_t, prefix=path))
        else:
            out.append((path, field_type))
    return out


def resolve_effective_report_to(logging_cfg: LoggingConfig) -> str:
    """Resolve effective tracker backend from logging config.

    :param LoggingConfig logging_cfg: Logging configuration.
    :return str: Effective report_to backend.
    """
    if bool(logging_cfg.wandb.enabled):
        return "wandb"
    return str(logging_cfg.backend).strip().lower()


def asdict_without_private(value: Any) -> Any:
    """Convert nested dataclasses to dicts while skipping private fields.

    :param Any value: Dataclass or nested value.
    :return Any: Mapping/list/scalar payload.
    """
    if dataclasses.is_dataclass(value):
        out: dict[str, Any] = {}
        for f in fields(value):
            if str(f.name).startswith("_"):
                continue
            out[str(f.name)] = asdict_without_private(getattr(value, f.name))
        return out
    if isinstance(value, dict):
        return {k: asdict_without_private(v) for k, v in value.items()}
    if isinstance(value, list):
        return [asdict_without_private(v) for v in value]
    if isinstance(value, tuple):
        return [asdict_without_private(v) for v in value]
    return value


__all__ = [
    "RUN_CONFIG_SCHEMA_VERSION",
    "Config",
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "OptimConfig",
    "LoggingConfig",
    "apply_dotted_override",
    "apply_profile_defaults",
    "asdict_without_private",
    "iter_leaf_paths_for_dataclass",
    "load_config",
    "load_config_sections",
    "load_data_config_snapshot",
    "load_logging_config_snapshot",
    "load_model_config_snapshot",
    "load_optim_config_snapshot",
    "normalize_mixed_precision",
    "resolve_effective_report_to",
    "validate_data_config",
    "validate_logging_config",
    "validate_model_config",
    "validate_optim_config",
    "validate_run_metadata_schema",
    "validate_train_config",
    "validate_training_workflow_options",
]
