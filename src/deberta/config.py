"""Dataclass-based model, data, and training configuration definitions."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

_BACKBONE_CHOICES = {"rope", "hf_deberta_v2"}
_NORM_ARCH_CHOICES = {"post", "keel"}
_ATTN_IMPL_CHOICES = {"sdpa", "eager"}
_FFN_CHOICES = {"swiglu", "mlp"}
_EMBED_SHARING_CHOICES = {"none", "es", "gdes"}
_REPORT_TO_CHOICES = {"none", "wandb", "tensorboard"}
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
RUN_CONFIG_SCHEMA_VERSION = 1


@dataclass
class ModelConfig:
    """Model-related arguments.

    This codebase supports two backbone families:

      1) backbone_type="rope" (default): a modern encoder stack that replaces DeBERTa's
         disentangled position bias with RoPE (rotary embeddings) and uses RMSNorm with
         a Post-Norm or KEEL residual topology.

      2) backbone_type="hf_deberta_v2": compatibility path using this repo's native
         DeBERTa-v2/v3-style encoder implementation (disentangled attention + LayerNorm),
         with configs/weights sourced from HF checkpoints when requested.

    For RTD/ELECTRA pretraining, the discriminator config is the primary source; the
    generator config is either provided explicitly or derived from the discriminator.
    """

    tokenizer_name_or_path: str = field(
        default="microsoft/deberta-v3-base",
        metadata={"help": "Tokenizer name or local path."},
    )

    backbone_type: str = field(
        default="rope",
        metadata={
            "help": (
                "Backbone implementation: 'rope' (recommended) or 'hf_deberta_v2' (HF DeBERTa compatibility mode). "
                "RoPE/RMSNorm/KEEL only apply to 'rope'; hf_deberta_v2 uses the native DeBERTa-v2 stack."
            )
        },
    )

    hf_attention_kernel: str = field(
        default="dynamic",
        metadata={
            "help": (
                "Native hf_deberta_v2 attention kernel variant (dynamic|cached_bmm|stable). "
                "Only applies when model.backbone_type=hf_deberta_v2."
            )
        },
    )

    # These are used as CONFIG SOURCES (and optionally weight sources if from_scratch=False).
    discriminator_model_name_or_path: str = field(
        default="microsoft/deberta-v3-base",
        metadata={
            "help": (
                "HF model name or path for the discriminator. Used to fetch a config (and weights if from_scratch=false) "
                "unless discriminator_config_name_or_path is provided."
            )
        },
    )

    generator_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Optional HF model name/path for generator (config and weights)."},
    )

    discriminator_config_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Optional config name/path (JSON or directory) for discriminator."},
    )

    generator_config_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Optional config name/path (JSON or directory) for generator."},
    )

    from_scratch: bool = field(
        default=True,
        metadata={
            "help": "If true, initialize models from config (random init). If false, load pretrained weights."
        },
    )

    # -------------------------
    # Modernized 'rope' backbone knobs
    # -------------------------
    hidden_size: int = field(
        default=768,
        metadata={"help": "RoPE model hidden size."},
    )

    num_hidden_layers: int = field(
        default=12,
        metadata={"help": "RoPE model layer depth."},
    )

    num_attention_heads: int = field(
        default=12,
        metadata={"help": "RoPE model attention head count."},
    )

    intermediate_size: int = field(
        default=3072,
        metadata={"help": "RoPE model intermediate FFN size before optional SwigLU adjustment."},
    )

    hidden_act: str = field(
        default="gelu",
        metadata={"help": "RoPE model hidden activation function name."},
    )

    rope_theta: float = field(
        default=10000.0,
        metadata={"help": "RoPE base theta."},
    )

    rotary_pct: float = field(
        default=1.0,
        metadata={"help": "Fraction of head_dim to apply RoPE to (1.0 = full)."},
    )

    use_absolute_position_embeddings: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, adds learned absolute position embeddings in addition to RoPE. "
                "For length generalization, keep this false."
            )
        },
    )

    max_position_embeddings: int | None = field(
        default=None,
        metadata={"help": "Initial max positions for RoPE cache / optional absolute position embeddings."},
    )

    type_vocab_size: int = field(
        default=2,
        metadata={"help": "Token type vocab size (segment embeddings)."},
    )

    norm_arch: str = field(
        default="post",
        metadata={"help": "Residual+norm topology for the rope backbone: 'post' or 'keel'."},
    )

    norm_eps: float = field(
        default=1e-6,
        metadata={"help": "RMSNorm epsilon."},
    )

    keel_alpha_init: float | None = field(
        default=None,
        metadata={
            "help": (
                "KEEL alpha initial value (skip scaling). If unset, defaults to "
                "2*num_hidden_layers (two residual sublayers per transformer block)."
            )
        },
    )

    keel_alpha_learnable: bool = field(
        default=False,
        metadata={"help": "If true, make KEEL alpha a learned scalar per sub-layer."},
    )

    attention_implementation: str = field(
        default="sdpa",
        metadata={"help": "Attention backend for rope backbone: 'sdpa' (recommended) or 'eager'."},
    )

    ffn_type: str = field(
        default="swiglu",
        metadata={
            "help": (
                "FFN block type for rope backbone: 'swiglu' (default) or 'mlp'. "
                "Applied when model.from_scratch=true; pretrained rope loads preserve checkpoint ffn_type."
            )
        },
    )

    use_bias: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether rope attention/FFN linear projections include bias terms. "
                "Applied when model.from_scratch=true; pretrained rope loads preserve checkpoint setting."
            )
        },
    )

    swiglu_adjust_intermediate: bool = field(
        default=True,
        metadata={
            "help": (
                "When model.from_scratch=true and ffn_type=swiglu, scale intermediate_size by 2/3 "
                "to keep FFN parameter budget comparable to MLP."
            )
        },
    )

    initializer_range: float = field(
        default=0.02,
        metadata={"help": "Weight init std for rope backbone."},
    )

    # -------------------------
    # Explicit overrides for pretrained RoPE loads (model.from_scratch=false).
    # These are ignored for scratch runs.
    # -------------------------
    pretrained_max_position_embeddings: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional max_position_embeddings override applied only when loading pretrained "
                "RoPE configs (model.from_scratch=false)."
            )
        },
    )

    pretrained_rope_theta: float | None = field(
        default=None,
        metadata={
            "help": ("Optional rope_theta override applied only when loading pretrained RoPE configs.")
        },
    )

    pretrained_rotary_pct: float | None = field(
        default=None,
        metadata={
            "help": ("Optional rotary_pct override applied only when loading pretrained RoPE configs.")
        },
    )

    pretrained_use_absolute_position_embeddings: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Optional absolute-position toggle override applied only when loading pretrained "
                "RoPE configs."
            )
        },
    )

    pretrained_type_vocab_size: int | None = field(
        default=None,
        metadata={
            "help": ("Optional type_vocab_size override applied only when loading pretrained RoPE configs.")
        },
    )

    pretrained_norm_arch: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional norm topology override ('post' or 'keel') applied only when loading "
                "pretrained RoPE configs."
            )
        },
    )

    pretrained_norm_eps: float | None = field(
        default=None,
        metadata={
            "help": ("Optional norm epsilon override applied only when loading pretrained RoPE configs.")
        },
    )

    pretrained_keel_alpha_init: float | None = field(
        default=None,
        metadata={
            "help": (
                "Optional KEEL alpha initialization override applied only when loading pretrained "
                "RoPE configs."
            )
        },
    )

    pretrained_keel_alpha_learnable: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Optional KEEL alpha learnable-flag override applied only when loading pretrained "
                "RoPE configs."
            )
        },
    )

    pretrained_ffn_type: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional FFN type override ('swiglu' or 'mlp') applied only when loading pretrained "
                "RoPE configs."
            )
        },
    )

    pretrained_use_bias: bool | None = field(
        default=None,
        metadata={
            "help": ("Optional projection-bias override applied only when loading pretrained RoPE configs.")
        },
    )

    pretrained_initializer_range: float | None = field(
        default=None,
        metadata={
            "help": (
                "Optional initializer_range metadata override applied only when loading pretrained "
                "RoPE configs."
            )
        },
    )

    hidden_dropout_prob: float | None = field(
        default=None,
        metadata={
            "help": (
                "Hidden dropout probability for discriminator/generator configs. "
                "Default is null (no override). Set a numeric value (including 0.0) to "
                "explicitly override discriminator/generator dropout."
            )
        },
    )

    attention_probs_dropout_prob: float | None = field(
        default=None,
        metadata={
            "help": (
                "Attention dropout probability for discriminator/generator configs. "
                "Default is null (no override). Set a numeric value (including 0.0) to "
                "explicitly override discriminator/generator dropout."
            )
        },
    )

    # -------------------------
    # ELECTRA-style generator sizing knobs (used only if generator config is derived)
    # -------------------------
    generator_num_hidden_layers: int | None = field(
        default=None,
        metadata={"help": "Override generator num_hidden_layers (if deriving generator config)."},
    )

    generator_hidden_size: int | None = field(
        default=None,
        metadata={"help": "Override generator hidden_size (if deriving generator config)."},
    )

    generator_intermediate_size: int | None = field(
        default=None,
        metadata={"help": "Override generator intermediate_size (if deriving generator config)."},
    )

    generator_num_attention_heads: int | None = field(
        default=None,
        metadata={"help": "Override generator num_attention_heads (if deriving generator config)."},
    )

    embedding_sharing: str = field(
        default="gdes",
        metadata={
            "help": (
                "Embedding sharing mode between generator and discriminator. "
                "Choices: none | es (vanilla sharing) | gdes (gradient-disentangled sharing, recommended). "
                "If enabled, generator/discriminator embedding dims must match."
            )
        },
    )

    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Enable activation/gradient checkpointing on both generator and discriminator backbones."
        },
    )


@dataclass
class DataConfig:
    """Data-related arguments."""

    # Mutually exclusive-ish inputs; precedence: load_from_disk > dataset_name > data_files
    dataset_name: str | None = field(
        default=None,
        metadata={"help": "Hugging Face dataset name (e.g., HuggingFaceFW/fineweb-edu, oscar, openwebtext)."},
    )

    dataset_config_name: str | None = field(
        default=None,
        metadata={"help": "Dataset config name (for example, 'default' for fineweb-edu)."},
    )

    data_files: str | None = field(
        default=None,
        metadata={
            "help": (
                "Local data files glob (e.g., /data/*.txt). Uses datasets 'text' builder. "
                "For multiple globs, pass a comma-separated list."
            )
        },
    )

    load_from_disk: str | None = field(
        default=None,
        metadata={"help": "Load a dataset saved via datasets.save_to_disk()."},
    )

    train_split: str = field(default="train", metadata={"help": "Train split name."})

    text_column_name: str = field(
        default="text",
        metadata={"help": "Text column to read from the dataset (for streaming packing)."},
    )

    streaming: bool = field(
        default=True,
        metadata={"help": "Use datasets streaming (IterableDataset). Recommended for pretraining."},
    )

    pack_sequences: bool = field(
        default=True,
        metadata={
            "help": (
                "If true, concatenate multiple documents into fixed-length packed sequences. "
                "If false, emit one-document chunks only (reference/no-pack mode)."
            )
        },
    )
    block_cross_document_attention: bool = field(
        default=False,
        metadata={
            "help": (
                "When data.pack_sequences=true, optionally emit 3D pairwise masks that block cross-document "
                "attention inside packed sequences. Default false to avoid quadratic mask overhead; "
                "enable only when strict doc-blocking is required."
            )
        },
    )

    cache_dir: str | None = field(default=None, metadata={"help": "Optional datasets cache dir."})

    max_seq_length: int = field(
        default=512,
        metadata={"help": "Sequence length after packing (includes special tokens)."},
    )

    shuffle_buffer_size: int = field(
        default=10_000,
        metadata={
            "help": (
                "Streaming shuffle buffer size. 0 disables shuffle. "
                "When data.streaming=false, this must be 0 or 1 "
                "(non-streaming datasets only support shuffle off/on)."
            )
        },
    )


@dataclass
class TrainConfig:
    """Training-related arguments."""

    output_dir: str | None = field(
        default=None,
        metadata={
            "help": (
                "Output directory (checkpoints, logs). If null/empty, auto-create "
                "runs/<project_name>/<timestamp>_<config_stem_or_run>."
            )
        },
    )

    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, delete an existing non-empty output_dir before training. "
                "If false, a non-empty output_dir requires resume_from_checkpoint."
            )
        },
    )

    project_name: str = field(
        default="deberta-train",
        metadata={
            "help": (
                "Tracker project name (for example W&B project) and auto output-dir namespace "
                "when train.output_dir is null."
            )
        },
    )

    run_name: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional run name for experiment trackers. When null, defaults to the "
                "resolved output directory basename."
            )
        },
    )

    seed: int = field(default=42, metadata={"help": "Random seed."})

    max_steps: int = field(default=10_000, metadata={"help": "Total optimizer steps (streaming-friendly)."})

    per_device_train_batch_size: int = field(default=4, metadata={"help": "Train batch size per device."})

    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Accumulate gradients this many steps before optimizer step."}
    )

    token_weighted_gradient_accumulation: bool = field(
        default=True,
        metadata={
            "help": (
                "If true, weight microbatch losses by active-token counts across each accumulation window "
                "instead of equal microbatch averaging."
            )
        },
    )

    learning_rate: float = field(default=5e-4, metadata={"help": "Peak learning rate."})

    generator_learning_rate: float = field(
        default=-1.0,
        metadata={"help": "Optional override learning rate for generator params. -1 uses learning_rate."},
    )

    weight_decay: float = field(default=0.01, metadata={"help": "AdamW weight decay."})

    adam_beta1: float = field(default=0.9, metadata={"help": "Adam beta1."})

    adam_beta2: float = field(default=0.999, metadata={"help": "Adam beta2."})

    adam_epsilon: float = field(default=1e-8, metadata={"help": "Adam epsilon."})

    warmup_steps: int = field(default=1_000, metadata={"help": "LR warmup steps."})

    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "Scheduler: linear|cosine|cosine_with_restarts|polynomial|constant|constant_with_warmup"
        },
    )

    max_grad_norm: float = field(default=1.0, metadata={"help": "Gradient clipping norm. 0 disables."})

    # Objective knobs
    mlm_probability: float = field(default=0.15, metadata={"help": "Masking probability."})

    mask_token_prob: float = field(
        default=0.8,
        metadata={"help": "For masked tokens: probability to replace with [MASK]."},
    )

    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "For masked tokens: probability to replace with a random token."},
    )

    mlm_max_ngram: int = field(
        default=1,
        metadata={"help": "Max whole-word n-gram size for masking. 1 = token-level masking."},
    )

    sampling_temperature: float = field(default=1.0, metadata={"help": "Generator sampling temperature."})

    gen_loss_weight: float = field(default=1.0, metadata={"help": "Generator loss weight."})

    disc_loss_weight: float = field(default=50.0, metadata={"help": "Discriminator loss weight."})

    decoupled_loss_scaling: bool = field(
        default=False,
        metadata={
            "help": "If true, rescales generator loss to match discriminator loss magnitude (DeBERTa RTD style)."
        },
    )

    # Logging / eval / save
    logging_steps: int = field(default=50, metadata={"help": "Log every N optimizer steps."})

    save_steps: int = field(default=1_000, metadata={"help": "Save checkpoint every N steps."})

    save_total_limit: int = field(
        default=3, metadata={"help": "Keep last N checkpoints. 0 disables deletion."}
    )

    report_to: str = field(
        default="none",
        metadata={"help": "Experiment tracker: none|wandb|tensorboard (accelerate loggers)."},
    )

    wandb_watch: str = field(
        default="gradients",
        metadata={
            "help": (
                "W&B model watch mode when train.report_to=wandb: "
                "none|gradients|parameters|all. Default gradients."
            )
        },
    )

    wandb_watch_log_freq: int = field(
        default=100,
        metadata={"help": "W&B watch logging frequency in optimizer steps (>=1)."},
    )

    mixed_precision: str = field(
        default="bf16",
        metadata={
            "help": (
                "Accelerate mixed precision mode (bf16|no). Default is bf16 autocast "
                "(not full-parameter bf16 casting)."
            )
        },
    )

    # Performance knobs
    dataloader_num_workers: int = field(default=2, metadata={"help": "DataLoader workers."})

    dataloader_pin_memory: bool = field(default=True, metadata={"help": "Pin memory in DataLoader."})

    torch_compile: bool = field(
        default=False, metadata={"help": "Enable torch.compile for the pretraining model."}
    )

    torch_compile_mode: str = field(
        default="default",
        metadata={
            "help": "torch.compile mode (default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs)."
        },
    )

    torch_compile_scope: str = field(
        default="auto",
        metadata={
            "help": (
                "torch.compile scope (auto|backbones|encoder|gen_encoder|disc_encoder|ffn|gen_ffn|disc_ffn)."
            )
        },
    )

    torch_compile_backend: str = field(
        default="inductor",
        metadata={"help": "torch.compile backend (inductor|aot_eager)."},
    )

    tf32: bool = field(default=True, metadata={"help": "Enable TF32 matmul on Ampere+ GPUs."})

    sdpa_kernel: str = field(
        default="auto",
        metadata={
            "help": (
                "SDPA kernel policy: auto|flash|mem_efficient|math. "
                "On CUDA, flash is strict (no mem_efficient/math fallback)."
            )
        },
    )

    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={
            "help": "Resume from an accelerate checkpoint directory. Use 'auto' to resume the latest checkpoint-* in output_dir."
        },
    )

    export_hf_final: bool = field(
        default=True,
        metadata={
            "help": (
                "Attempt to export a final HF discriminator model into output_dir/final_hf. "
                "For FSDP2+sharded checkpoints, prefer running `deberta export` after training."
            )
        },
    )


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


def _normalize_sdpa_kernel(value: str) -> str:
    """Normalize and validate SDPA kernel policy values.

    :param str value: Raw SDPA kernel value.
    :return str: Canonical lower-case SDPA kernel policy.
    """
    v = str(value).strip().lower()
    v = _SDPA_KERNEL_ALIASES.get(v, v)
    return _ensure_choice("train.sdpa_kernel", v, _SDPA_KERNEL_CHOICES)


def _normalize_torch_compile_mode(value: str) -> str:
    """Normalize and validate torch.compile mode values.

    :param str value: Raw compile mode value.
    :return str: Canonical compile mode.
    """
    v = str(value).strip().lower()
    v = _TORCH_COMPILE_MODE_ALIASES.get(v, v)
    return _ensure_choice("train.torch_compile_mode", v, _TORCH_COMPILE_MODE_CHOICES)


def _normalize_torch_compile_scope(value: str) -> str:
    """Normalize and validate torch.compile scope values.

    :param str value: Raw compile scope value.
    :return str: Canonical compile scope.
    """
    v = str(value).strip().lower().replace("-", "_")
    v = _TORCH_COMPILE_SCOPE_ALIASES.get(v, v)
    return _ensure_choice("train.torch_compile_scope", v, _TORCH_COMPILE_SCOPE_CHOICES)


def _normalize_torch_compile_backend(value: str) -> str:
    """Normalize and validate torch.compile backend values.

    :param str value: Raw compile backend value.
    :return str: Canonical compile backend.
    """
    v = str(value).strip().lower().replace("-", "_")
    v = _TORCH_COMPILE_BACKEND_ALIASES.get(v, v)
    return _ensure_choice("train.torch_compile_backend", v, _TORCH_COMPILE_BACKEND_CHOICES)


def _normalize_wandb_watch(value: str) -> str:
    """Normalize and validate W&B watch mode values.

    :param str value: Raw W&B watch mode.
    :return str: Canonical W&B watch mode.
    """
    v = str(value).strip().lower().replace("-", "_")
    v = _WANDB_WATCH_ALIASES.get(v, v)
    return _ensure_choice("train.wandb_watch", v, _WANDB_WATCH_CHOICES)


def _normalize_hf_attention_kernel(value: str) -> str:
    """Normalize and validate native hf_deberta_v2 attention-kernel values.

    :param str value: Raw attention-kernel value.
    :return str: Canonical attention-kernel name.
    """
    v = str(value).strip().lower().replace("-", "_")
    v = _HF_ATTN_KERNEL_ALIASES.get(v, v)
    return _ensure_choice("model.hf_attention_kernel", v, _HF_ATTN_KERNEL_CHOICES)


def normalize_mixed_precision(value: object) -> str:
    """Normalize and validate mixed precision values.

    :param object value: Raw mixed precision value.
    :return str: Canonical mixed precision mode (``bf16`` or ``no``).
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
            f"Expected {int(RUN_CONFIG_SCHEMA_VERSION)}."
        )


def _looks_like_hf_deberta_checkpoint(value: str) -> bool:
    """Return whether a model source appears to be an HF DeBERTa v2/v3 checkpoint id.

    :param str value: Model source string.
    :return bool: True when source matches known HF DeBERTa hub-id prefixes.
    """
    v = str(value).strip().lower()
    return any(v.startswith(prefix) for prefix in _HF_DEBERTA_PRETRAINED_PREFIXES)


def validate_model_config(cfg: ModelConfig) -> None:
    """Validate model config semantics and normalize constrained values.

    :param ModelConfig cfg: Model configuration.
    """
    cfg.backbone_type = _ensure_choice("model.backbone_type", cfg.backbone_type, _BACKBONE_CHOICES)
    cfg.hf_attention_kernel = _normalize_hf_attention_kernel(cfg.hf_attention_kernel)
    cfg.norm_arch = _ensure_choice("model.norm_arch", cfg.norm_arch, _NORM_ARCH_CHOICES)
    cfg.attention_implementation = _ensure_choice(
        "model.attention_implementation", cfg.attention_implementation, _ATTN_IMPL_CHOICES
    )
    cfg.ffn_type = _ensure_choice("model.ffn_type", cfg.ffn_type, _FFN_CHOICES)
    cfg.embedding_sharing = _ensure_choice(
        "model.embedding_sharing", cfg.embedding_sharing, _EMBED_SHARING_CHOICES
    )

    if cfg.max_position_embeddings is not None and int(cfg.max_position_embeddings) <= 0:
        raise ValueError("model.max_position_embeddings must be > 0 when provided.")
    if float(cfg.rotary_pct) <= 0.0 or float(cfg.rotary_pct) > 1.0:
        raise ValueError("model.rotary_pct must be in (0, 1].")

    # Explicit dependency check: rope-only knobs are invalid in HF-compat mode.
    if cfg.backbone_type == "hf_deberta_v2":
        defaults = ModelConfig()
        rope_knobs = (
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
            "pretrained_max_position_embeddings",
            "pretrained_rope_theta",
            "pretrained_rotary_pct",
            "pretrained_use_absolute_position_embeddings",
            "pretrained_type_vocab_size",
            "pretrained_norm_arch",
            "pretrained_norm_eps",
            "pretrained_keel_alpha_init",
            "pretrained_keel_alpha_learnable",
            "pretrained_ffn_type",
            "pretrained_use_bias",
            "pretrained_initializer_range",
        )
        changed = [name for name in rope_knobs if getattr(cfg, name) != getattr(defaults, name)]
        if changed:
            raise ValueError(
                "These options are only valid when model.backbone_type='rope': " + ", ".join(sorted(changed))
            )
    else:
        default_hf_kernel = ModelConfig().hf_attention_kernel
        if cfg.hf_attention_kernel != default_hf_kernel:
            warnings.warn(
                "model.hf_attention_kernel only applies when model.backbone_type='hf_deberta_v2'. "
                f"Current value ({cfg.hf_attention_kernel!r}) has no effect on the rope backbone.",
                UserWarning,
                stacklevel=2,
            )

    if (
        not bool(cfg.from_scratch)
        and cfg.generator_config_name_or_path
        and not cfg.generator_model_name_or_path
    ):
        raise ValueError(
            "model.generator_config_name_or_path requires model.generator_model_name_or_path when "
            "model.from_scratch=false. Explicit generator configs must pair with explicit generator "
            "weights; leave both unset to use derived-generator fallback from discriminator weights."
        )

    pretrained_override_fields = (
        "pretrained_max_position_embeddings",
        "pretrained_rope_theta",
        "pretrained_rotary_pct",
        "pretrained_use_absolute_position_embeddings",
        "pretrained_type_vocab_size",
        "pretrained_norm_arch",
        "pretrained_norm_eps",
        "pretrained_keel_alpha_init",
        "pretrained_keel_alpha_learnable",
        "pretrained_ffn_type",
        "pretrained_use_bias",
        "pretrained_initializer_range",
    )

    if cfg.backbone_type == "rope" and bool(cfg.from_scratch):
        set_pretrained = [name for name in pretrained_override_fields if getattr(cfg, name) is not None]
        if set_pretrained:
            raise ValueError(
                "These options apply only when model.from_scratch=false: " + ", ".join(sorted(set_pretrained))
            )

    if cfg.backbone_type == "rope" and not bool(cfg.from_scratch):
        invalid_sources: list[str] = []
        for field_name in ("discriminator_model_name_or_path", "generator_model_name_or_path"):
            src = getattr(cfg, field_name, None)
            if src and _looks_like_hf_deberta_checkpoint(str(src)):
                invalid_sources.append(f"{field_name}={src}")
        if invalid_sources:
            raise ValueError(
                "model.from_scratch=false with model.backbone_type='rope' requires DebertaRoPE checkpoints, "
                "not HF DeBERTa v2/v3 checkpoints. "
                "Use model.backbone_type='hf_deberta_v2' for HF DeBERTa weights, or keep model.from_scratch=true "
                "for RoPE model initialization. Invalid sources: " + ", ".join(sorted(invalid_sources))
            )

        defaults = ModelConfig()
        scratch_rope_pretrained_knobs = (
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
            "ffn_type",
            "use_bias",
            "swiglu_adjust_intermediate",
            "initializer_range",
        )
        changed_scratch_knobs = [
            name for name in scratch_rope_pretrained_knobs if getattr(cfg, name) != getattr(defaults, name)
        ]
        if changed_scratch_knobs:
            raise ValueError(
                "These options only affect scratch RoPE initialization and are not applied when "
                "model.from_scratch=false. Use explicit pretrained override fields instead "
                "(model.pretrained_*). Invalid options: " + ", ".join(sorted(changed_scratch_knobs))
            )

        if (
            cfg.pretrained_max_position_embeddings is not None
            and int(cfg.pretrained_max_position_embeddings) <= 0
        ):
            raise ValueError("model.pretrained_max_position_embeddings must be > 0 when provided.")
        if cfg.pretrained_rotary_pct is not None:
            pct = float(cfg.pretrained_rotary_pct)
            if pct <= 0.0 or pct > 1.0:
                raise ValueError("model.pretrained_rotary_pct must be in (0, 1] when provided.")
        if cfg.pretrained_norm_arch is not None:
            cfg.pretrained_norm_arch = _ensure_choice(
                "model.pretrained_norm_arch", cfg.pretrained_norm_arch, _NORM_ARCH_CHOICES
            )
        if cfg.pretrained_ffn_type is not None:
            cfg.pretrained_ffn_type = _ensure_choice(
                "model.pretrained_ffn_type", cfg.pretrained_ffn_type, _FFN_CHOICES
            )

    if cfg.generator_config_name_or_path or cfg.generator_model_name_or_path:
        derived_knobs = []
        for name in (
            "generator_num_hidden_layers",
            "generator_hidden_size",
            "generator_intermediate_size",
            "generator_num_attention_heads",
        ):
            if getattr(cfg, name) is not None:
                derived_knobs.append(name)
        if derived_knobs:
            raise ValueError(
                "These options are only used when deriving generator config and must be unset when "
                "model.generator_config_name_or_path or model.generator_model_name_or_path is provided: "
                + ", ".join(sorted(derived_knobs))
            )

    if not bool(cfg.from_scratch) and not cfg.generator_model_name_or_path:
        pretrained_shape_overrides = []
        for name in (
            "generator_hidden_size",
            "generator_intermediate_size",
            "generator_num_attention_heads",
        ):
            if getattr(cfg, name) is not None:
                pretrained_shape_overrides.append(name)
        if pretrained_shape_overrides:
            raise ValueError(
                "model.from_scratch=false with derived generator weights (generator_model_name_or_path unset) "
                "cannot use generator shape overrides because generator weights are loaded from the "
                "discriminator source. Set model.generator_model_name_or_path to an explicit generator "
                "checkpoint or unset: " + ", ".join(sorted(pretrained_shape_overrides))
            )

    if cfg.backbone_type == "rope":
        if int(cfg.hidden_size) <= 0:
            raise ValueError("model.hidden_size must be > 0.")
        if int(cfg.num_hidden_layers) <= 0:
            raise ValueError("model.num_hidden_layers must be > 0.")
        if int(cfg.num_attention_heads) <= 0:
            raise ValueError("model.num_attention_heads must be > 0.")
        if int(cfg.intermediate_size) <= 0:
            raise ValueError("model.intermediate_size must be > 0.")
        if int(cfg.hidden_size) % int(cfg.num_attention_heads) != 0:
            raise ValueError("model.hidden_size must be divisible by model.num_attention_heads.")

    # Warn on KEEL params when norm_arch="post" (they only apply when norm_arch="keel").
    defaults = ModelConfig()
    if cfg.norm_arch == "post":
        if cfg.keel_alpha_init is not None and cfg.keel_alpha_init != defaults.keel_alpha_init:
            warnings.warn(
                "model.keel_alpha_init has no effect when model.norm_arch='post'. "
                "Set model.norm_arch='keel' to use KEEL alpha scaling.",
                UserWarning,
                stacklevel=2,
            )
        if cfg.keel_alpha_learnable != defaults.keel_alpha_learnable:
            warnings.warn(
                "model.keel_alpha_learnable has no effect when model.norm_arch='post'. "
                "Set model.norm_arch='keel' to use learnable KEEL alpha.",
                UserWarning,
                stacklevel=2,
            )

    # Warn on swiglu_adjust_intermediate with ffn_type="mlp" (only applies to swiglu).
    if cfg.ffn_type == "mlp" and cfg.swiglu_adjust_intermediate != defaults.swiglu_adjust_intermediate:
        warnings.warn(
            "model.swiglu_adjust_intermediate has no effect when model.ffn_type='mlp'. "
            "The intermediate size scaling is only applied for ffn_type='swiglu'.",
            UserWarning,
            stacklevel=2,
        )


def validate_data_config(cfg: DataConfig) -> None:
    """Validate data-source and preprocessing option combinations.

    :param DataConfig cfg: Data configuration.
    """
    if cfg.load_from_disk:
        if cfg.streaming:
            raise ValueError(
                "data.streaming=true is not compatible with data.load_from_disk. Set data.streaming=false."
            )
        conflicting = []
        if cfg.dataset_name:
            conflicting.append("data.dataset_name")
        if cfg.data_files:
            conflicting.append("data.data_files")
        if cfg.dataset_config_name:
            conflicting.append("data.dataset_config_name")
        if conflicting:
            raise ValueError("data.load_from_disk cannot be combined with: " + ", ".join(sorted(conflicting)))

    if cfg.dataset_config_name and not cfg.dataset_name:
        raise ValueError("data.dataset_config_name requires data.dataset_name.")

    if not cfg.load_from_disk and not cfg.dataset_name and not cfg.data_files:
        raise ValueError(
            "No dataset source configured. Provide one of: data.load_from_disk, "
            "data.dataset_name, or data.data_files."
        )

    if int(cfg.max_seq_length) < 8:
        raise ValueError("data.max_seq_length must be >= 8 for pretraining.")
    if int(cfg.shuffle_buffer_size) < 0:
        raise ValueError("data.shuffle_buffer_size must be >= 0.")
    if not bool(cfg.streaming) and int(cfg.shuffle_buffer_size) not in {0, 1}:
        raise ValueError(
            "data.shuffle_buffer_size must be 0 or 1 when data.streaming=false "
            "(non-streaming datasets only support shuffle off/on)."
        )
    if not bool(cfg.pack_sequences) and bool(cfg.block_cross_document_attention):
        raise ValueError("data.block_cross_document_attention=true requires data.pack_sequences=true.")


def validate_train_config(cfg: TrainConfig) -> None:
    """Validate train config scalar ranges and constrained options.

    :param TrainConfig cfg: Training configuration.
    """
    cfg.report_to = _ensure_choice("train.report_to", cfg.report_to, _REPORT_TO_CHOICES)
    cfg.lr_scheduler_type = _ensure_choice(
        "train.lr_scheduler_type", cfg.lr_scheduler_type, _LR_SCHEDULER_CHOICES
    )
    cfg.sdpa_kernel = _normalize_sdpa_kernel(cfg.sdpa_kernel)
    cfg.torch_compile_mode = _normalize_torch_compile_mode(cfg.torch_compile_mode)
    cfg.torch_compile_scope = _normalize_torch_compile_scope(cfg.torch_compile_scope)
    cfg.torch_compile_backend = _normalize_torch_compile_backend(cfg.torch_compile_backend)
    cfg.wandb_watch = _normalize_wandb_watch(cfg.wandb_watch)

    cfg.mixed_precision = normalize_mixed_precision(cfg.mixed_precision)
    if cfg.output_dir is not None and not str(cfg.output_dir).strip():
        cfg.output_dir = None
    if not str(cfg.project_name).strip():
        raise ValueError("train.project_name must be non-empty.")

    if int(cfg.max_steps) <= 0:
        raise ValueError("train.max_steps must be > 0.")
    if int(cfg.per_device_train_batch_size) <= 0:
        raise ValueError("train.per_device_train_batch_size must be > 0.")
    if int(cfg.gradient_accumulation_steps) <= 0:
        raise ValueError("train.gradient_accumulation_steps must be > 0.")
    if int(cfg.warmup_steps) < 0:
        raise ValueError("train.warmup_steps must be >= 0.")
    if int(cfg.logging_steps) < 0:
        raise ValueError("train.logging_steps must be >= 0.")
    if int(cfg.save_steps) < 0:
        raise ValueError("train.save_steps must be >= 0.")
    if int(cfg.save_total_limit) < 0:
        raise ValueError("train.save_total_limit must be >= 0.")
    if int(cfg.wandb_watch_log_freq) <= 0:
        raise ValueError("train.wandb_watch_log_freq must be >= 1.")

    mlm = float(cfg.mlm_probability)
    if mlm <= 0.0 or mlm >= 1.0:
        raise ValueError("train.mlm_probability must be in (0, 1).")
    mask_p = float(cfg.mask_token_prob)
    rand_p = float(cfg.random_token_prob)
    if mask_p < 0.0 or rand_p < 0.0 or (mask_p + rand_p) > 1.0:
        raise ValueError(
            "Invalid masking probabilities: train.mask_token_prob + train.random_token_prob must be <= 1."
        )
    if int(cfg.mlm_max_ngram) < 1:
        raise ValueError("train.mlm_max_ngram must be >= 1.")
    if float(cfg.sampling_temperature) <= 0.0:
        raise ValueError("train.sampling_temperature must be > 0.")

    # Warn on torch_compile_scope when torch_compile=false (scope has no effect).
    if not bool(cfg.torch_compile) and cfg.torch_compile_scope != "auto":
        warnings.warn(
            "train.torch_compile_scope has no effect when train.torch_compile=false. "
            f"Current scope ({cfg.torch_compile_scope!r}) will be ignored.",
            UserWarning,
            stacklevel=2,
        )


def validate_training_workflow_options(
    *,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig | None = None,
) -> None:
    """Validate options tied to workflow support (for example, eval mode availability).

    :param DataConfig data_cfg: Data configuration.
    :param TrainConfig train_cfg: Training configuration.
    :param ModelConfig | None model_cfg: Optional model configuration for cross-surface checks.
    """
    sdpa_policy = str(train_cfg.sdpa_kernel).strip().lower()
    if (
        bool(data_cfg.pack_sequences)
        and bool(data_cfg.block_cross_document_attention)
        and sdpa_policy == "flash"
    ):
        raise ValueError(
            "train.sdpa_kernel=flash is not supported with data.pack_sequences=true. "
            "Packed batches may require 3D document-blocking attention masks that are incompatible "
            "with strict flash SDPA kernels. Use train.sdpa_kernel=auto|mem_efficient|math instead."
        )

    if model_cfg is not None:
        backbone_type = str(model_cfg.backbone_type).strip().lower()
        attn_impl = str(model_cfg.attention_implementation).strip().lower()
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
                "train.sdpa_kernel only affects rope attention when model.attention_implementation='sdpa'. "
                "Set train.sdpa_kernel=auto or switch model.attention_implementation=sdpa."
            )
        if (
            backbone_type != "rope"
            and bool(data_cfg.pack_sequences)
            and bool(data_cfg.block_cross_document_attention)
        ):
            raise ValueError(
                "data.block_cross_document_attention=true is only supported with model.backbone_type='rope'. "
                "Use data.block_cross_document_attention=false or switch to model.backbone_type='rope'."
            )
        embed_sharing = str(model_cfg.embedding_sharing).strip().lower()
        gen_lr = float(train_cfg.generator_learning_rate)
        if embed_sharing == "es" and gen_lr > 0 and gen_lr != float(train_cfg.learning_rate):
            raise ValueError(
                f"model.embedding_sharing='es' shares embedding parameters between generator and discriminator, "
                f"but train.generator_learning_rate ({gen_lr}) differs from train.learning_rate "
                f"({train_cfg.learning_rate}). Shared embeddings would silently use the generator LR. "
                f"Set generator_learning_rate=-1 (inherit) or match it to learning_rate, "
                f"or switch to embedding_sharing='gdes'/'none'."
            )
