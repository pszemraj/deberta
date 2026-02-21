"""Dataclass-based model, data, and training configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

_BACKBONE_CHOICES = {"rope", "hf_deberta_v2"}
_NORM_ARCH_CHOICES = {"post", "keel"}
_ATTN_IMPL_CHOICES = {"sdpa", "eager"}
_FFN_CHOICES = {"swiglu", "mlp"}
_EMBED_SHARING_CHOICES = {"none", "es", "gdes"}
_REPORT_TO_CHOICES = {"none", "wandb", "tensorboard"}
_LR_SCHEDULER_CHOICES = {
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
}
_SDPA_KERNEL_CHOICES = {"auto", "flash", "mem_efficient", "math", "flash_only"}
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
_MIXED_PRECISION_CHOICES = {"bf16", "bfloat16", "no", "none"}


@dataclass
class ModelConfig:
    """Model-related arguments.

    This codebase supports two backbone families:

      1) backbone_type="rope" (default): a modern encoder stack that replaces DeBERTa's
         disentangled position bias with RoPE (rotary embeddings) and uses RMSNorm with
         a Post-Norm or KEEL residual topology.

      2) backbone_type="hf_deberta_v2": compatibility path that instantiates Hugging Face
         DebertaV2Model backbones (DeBERTa v2/v3). This keeps the original disentangled
         attention implementation, and does NOT apply RoPE/RMSNorm/KEEL changes.

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
                "RoPE/RMSNorm/KEEL only apply to 'rope'."
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
                "KEEL alpha initial value (skip scaling). If unset, defaults to 2*num_hidden_layers for each model."
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

    initializer_range: float = field(
        default=0.02,
        metadata={"help": "Weight init std for rope backbone."},
    )

    hidden_dropout_prob: float | None = field(
        default=None,
        metadata={"help": "Override hidden dropout prob from config (rope or HF)."},
    )

    attention_probs_dropout_prob: float | None = field(
        default=None,
        metadata={"help": "Override attention dropout prob from config (rope or HF)."},
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

    eval_split: str | None = field(
        default=None,
        metadata={"help": "Reserved for future evaluation workflow. Currently unsupported."},
    )

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

    cache_dir: str | None = field(default=None, metadata={"help": "Optional datasets cache dir."})

    max_seq_length: int = field(
        default=512,
        metadata={"help": "Sequence length after packing (includes special tokens)."},
    )

    shuffle_buffer_size: int = field(
        default=10_000,
        metadata={"help": "Streaming shuffle buffer size. 0 disables streaming shuffle."},
    )

    # Non-streaming preprocessing
    preprocessing_num_workers: int = field(
        default=8,
        metadata={
            "help": "Legacy non-streaming pretokenization workers. Currently unused in unified packer path."
        },
    )


@dataclass
class TrainConfig:
    """Training-related arguments."""

    output_dir: str = field(
        default="runs/deberta_pretrain",
        metadata={"help": "Output directory (checkpoints, logs)."},
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

    run_name: str | None = field(
        default=None,
        metadata={"help": "Optional run name for experiment trackers."},
    )

    seed: int = field(default=42, metadata={"help": "Random seed."})

    max_steps: int = field(default=10_000, metadata={"help": "Total optimizer steps (streaming-friendly)."})

    per_device_train_batch_size: int = field(default=4, metadata={"help": "Train batch size per device."})

    per_device_eval_batch_size: int = field(
        default=4,
        metadata={
            "help": (
                "Reserved for future evaluation workflow. Currently unused "
                "(must remain default while eval is disabled)."
            )
        },
    )

    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Accumulate gradients this many steps before optimizer step."}
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

    eval_steps: int = field(
        default=0,
        metadata={"help": "Reserved for future evaluation workflow. Must remain 0."},
    )

    save_steps: int = field(default=1_000, metadata={"help": "Save checkpoint every N steps."})

    save_total_limit: int = field(
        default=3, metadata={"help": "Keep last N checkpoints. 0 disables deletion."}
    )

    report_to: str = field(
        default="none",
        metadata={"help": "Experiment tracker: none|wandb|tensorboard (accelerate loggers)."},
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

    tf32: bool = field(default=True, metadata={"help": "Enable TF32 matmul on Ampere+ GPUs."})

    sdpa_kernel: str = field(
        default="auto",
        metadata={
            "help": (
                "SDPA kernel policy: auto|flash|mem_efficient|math|flash_only. "
                "Best-effort preference for PyTorch SDPA backends on CUDA."
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


def validate_model_config(cfg: ModelConfig) -> None:
    """Validate model config semantics and normalize constrained values.

    :param ModelConfig cfg: Model configuration.
    """
    cfg.backbone_type = _ensure_choice("model.backbone_type", cfg.backbone_type, _BACKBONE_CHOICES)
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
        rope_only = (
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
            "initializer_range",
        )
        changed = [name for name in rope_only if getattr(cfg, name) != getattr(defaults, name)]
        if changed:
            raise ValueError(
                "These options are only valid when model.backbone_type='rope': " + ", ".join(sorted(changed))
            )

    if cfg.generator_config_name_or_path or cfg.generator_model_name_or_path:
        derived_only = []
        for name in (
            "generator_num_hidden_layers",
            "generator_hidden_size",
            "generator_intermediate_size",
            "generator_num_attention_heads",
        ):
            if getattr(cfg, name) is not None:
                derived_only.append(name)
        if derived_only:
            raise ValueError(
                "These options are only used when deriving generator config and must be unset when "
                "model.generator_config_name_or_path or model.generator_model_name_or_path is provided: "
                + ", ".join(sorted(derived_only))
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
    if int(cfg.preprocessing_num_workers) < 0:
        raise ValueError("data.preprocessing_num_workers must be >= 0.")
    default_preproc_workers = DataConfig().preprocessing_num_workers
    if int(cfg.preprocessing_num_workers) != int(default_preproc_workers):
        raise ValueError(
            "data.preprocessing_num_workers is currently unused in the unified packer path. "
            f"Keep the default ({default_preproc_workers}) to avoid inert config."
        )


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

    mp = cfg.mixed_precision
    if isinstance(mp, bool):
        pass
    else:
        v = str(mp).strip().lower()
        if v not in _MIXED_PRECISION_CHOICES:
            raise ValueError("train.mixed_precision must be one of: bf16|no (or synonyms bfloat16|none).")

    if int(cfg.max_steps) <= 0:
        raise ValueError("train.max_steps must be > 0.")
    if int(cfg.per_device_train_batch_size) <= 0:
        raise ValueError("train.per_device_train_batch_size must be > 0.")
    if int(cfg.per_device_eval_batch_size) <= 0:
        raise ValueError("train.per_device_eval_batch_size must be > 0.")
    if int(cfg.gradient_accumulation_steps) <= 0:
        raise ValueError("train.gradient_accumulation_steps must be > 0.")
    if int(cfg.warmup_steps) < 0:
        raise ValueError("train.warmup_steps must be >= 0.")
    if int(cfg.logging_steps) < 0:
        raise ValueError("train.logging_steps must be >= 0.")
    if int(cfg.eval_steps) < 0:
        raise ValueError("train.eval_steps must be >= 0.")
    if int(cfg.save_steps) < 0:
        raise ValueError("train.save_steps must be >= 0.")
    if int(cfg.save_total_limit) < 0:
        raise ValueError("train.save_total_limit must be >= 0.")

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


def validate_training_workflow_options(*, data_cfg: DataConfig, train_cfg: TrainConfig) -> None:
    """Validate options tied to workflow support (for example, eval mode availability).

    :param DataConfig data_cfg: Data configuration.
    :param TrainConfig train_cfg: Training configuration.
    """
    if data_cfg.eval_split is not None or int(train_cfg.eval_steps) > 0:
        raise ValueError(
            "Evaluation workflow is not implemented yet. Set data.eval_split=null and train.eval_steps=0."
        )

    default_eval_bs = TrainConfig().per_device_eval_batch_size
    if int(train_cfg.per_device_eval_batch_size) != int(default_eval_bs):
        raise ValueError(
            "train.per_device_eval_batch_size is reserved for future evaluation and is currently unused. "
            f"Keep the default ({default_eval_bs}) while evaluation is disabled."
        )

    sdpa_policy = str(train_cfg.sdpa_kernel).strip().lower()
    if bool(data_cfg.pack_sequences) and sdpa_policy == "flash_only":
        raise ValueError(
            "train.sdpa_kernel=flash_only is not supported with data.pack_sequences=true. "
            "Packed batches may require 3D document-blocking attention masks that are incompatible "
            "with flash-only SDPA kernels. Use train.sdpa_kernel=flash|auto|mem_efficient|math instead."
        )
