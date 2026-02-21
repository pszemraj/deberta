"""Dataclass-based model, data, and training configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass, field


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
        metadata={"help": "Hugging Face dataset name (e.g., c4, oscar, openwebtext)."},
    )

    dataset_config_name: str | None = field(
        default=None,
        metadata={"help": "Dataset config name (e.g., 'en' for c4)."},
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

    eval_split: str | None = field(default=None, metadata={"help": "Optional eval split name."})

    text_column_name: str = field(
        default="text",
        metadata={"help": "Text column to read from the dataset (for streaming packing)."},
    )

    streaming: bool = field(
        default=True,
        metadata={"help": "Use datasets streaming (IterableDataset). Recommended for pretraining."},
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
        metadata={"help": "Non-streaming tokenization workers."},
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

    per_device_eval_batch_size: int = field(default=4, metadata={"help": "Eval batch size per device."})

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

    eval_steps: int = field(default=0, metadata={"help": "Run eval every N steps. 0 disables."})

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
