"""Dataclass-based model, data, and training configuration definitions."""

from __future__ import annotations

import re
import types
import warnings
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from deberta.io_utils import load_json_mapping

_BACKBONE_CHOICES = {"rope", "hf_deberta_v2"}
_MODEL_PROFILE_CHOICES = {"modern", "deberta_v3_parity"}
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
    "deberta-v2",
    "deberta-v3",
    "mdeberta-v3",
)
_DENSE_DOC_BLOCK_WARN_SEQ_LEN = 2048
# Pre-stable policy: persisted run schemas may change when needed for correctness/simplicity.
# Backward checkpoint/resume compatibility is intentionally not guaranteed until a stable release.
RUN_CONFIG_SCHEMA_VERSION = 3
_VAR_FULL_RE = re.compile(r"^\$variables\.([A-Za-z0-9_.-]+)$")
_VAR_INLINE_RE = re.compile(r"\{\$variables\.([A-Za-z0-9_.-]+)\}")
_VAR_BRACE_RE = re.compile(r"\$\{variables\.([A-Za-z0-9_.-]+)\}")
_VAR_SUSPICIOUS_RE = re.compile(r"\$variables\.[A-Za-z0-9_.-]+")


@dataclass
class ModelConfig:
    """Model-related arguments.

    This codebase supports two backbone families:

      1) backbone_type="hf_deberta_v2" (default): DeBERTa-v2/v3-style encoder
         implementation (disentangled attention + LayerNorm), with configs/weights
         sourced from HF checkpoints when requested.

      2) backbone_type="rope" (opt-in): an experimental modern encoder stack that
         replaces DeBERTa's disentangled position bias with RoPE (rotary embeddings)
         and supports RMSNorm Post-Norm or KEEL residual topology.

    For RTD/ELECTRA pretraining, the discriminator config is the primary source; the
    generator config is either provided explicitly or derived from the discriminator.
    """

    tokenizer_name_or_path: str = field(
        default="microsoft/deberta-v3-base",
        metadata={"help": "Tokenizer name or local path."},
    )

    profile: str = field(
        default="modern",
        metadata={
            "help": (
                "Model/runtime profile. "
                "'modern' keeps this repo's default modernization choices; "
                "'deberta_v3_parity' applies DeBERTa-v3 parity-oriented defaults where unset."
            )
        },
    )

    tokenizer_allow_vocab_resize: bool = field(
        default=False,
        metadata={
            "help": (
                "Allow runtime tokenizer vocabulary growth via tokenizer.add_tokens(...) when "
                "vocab alignment controls require a larger size."
            )
        },
    )

    tokenizer_vocab_target: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional target tokenizer/model vocab size. When set, runtime validates that "
                "len(tokenizer) <= target and can add inert placeholder tokens (if enabled) "
                "to reach this size."
            )
        },
    )

    tokenizer_vocab_multiple: int = field(
        default=1,
        metadata={
            "help": (
                "Round resolved tokenizer vocab size up to this multiple (for tensor-core-friendly "
                "embedding dimensions). Use 1 to disable."
            )
        },
    )

    backbone_type: str = field(
        default="hf_deberta_v2",
        metadata={
            "help": (
                "Backbone implementation: 'hf_deberta_v2' (default) or 'rope' (experimental opt-in). "
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

    hf_model_size: str = field(
        default="base",
        metadata={
            "help": (
                "Repo-defined hf_deberta_v2 architecture preset "
                "(xsmall|small|base|large) used for synthesized configs."
            )
        },
    )

    # Model source paths are used for pretrained weight loading when from_scratch=false.
    pretrained_discriminator_path: str = field(
        default="",
        metadata={
            "help": (
                "Model name or path for discriminator weights when model.from_scratch=false. "
                "HF backbone configs are synthesized from repo defaults."
            )
        },
    )

    pretrained_generator_path: str | None = field(
        default=None,
        metadata={"help": "Optional HF model name/path for generator (config and weights)."},
    )

    hf_max_position_embeddings: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional max_position_embeddings override for hf_deberta_v2 configs. "
                "Use this instead of external edited JSON configs."
            )
        },
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
                "1/sqrt(2*num_hidden_layers) (two residual sublayers per transformer block)."
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
        default="mlp",
        metadata={
            "help": (
                "FFN block type for rope backbone: 'mlp' (default) or 'swiglu'. "
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
        default=0.0,
        metadata={
            "help": (
                "Hidden dropout probability for discriminator/generator configs. "
                "Default is 0.0 (repo standard override). Set null to preserve "
                "source/checkpoint-native dropout."
            )
        },
    )

    attention_probs_dropout_prob: float | None = field(
        default=0.0,
        metadata={
            "help": (
                "Attention dropout probability for discriminator/generator configs. "
                "Default is 0.0 (repo standard override). Set null to preserve "
                "source/checkpoint-native dropout."
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

    discriminator_learning_rate: float = field(
        default=-1.0,
        metadata={"help": "Optional override learning rate for discriminator params. -1 uses learning_rate."},
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

    decoupled_training: bool = field(
        default=True,
        metadata={"help": ("Enable true two-phase RTD training (generator step then discriminator step).")},
    )

    # Logging / eval / save
    logging_steps: int = field(default=50, metadata={"help": "Log every N optimizer steps."})

    debug_metrics: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable verbose local-only debug metric rows (for example zero-token window counters) "
                "in output_dir/metrics.jsonl.gz. These debug metrics are never sent to trackers."
            )
        },
    )

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

    resume_data_strategy: str = field(
        default="auto",
        metadata={
            "help": (
                "Resume data-iterator alignment policy when resuming from checkpoint: "
                "auto|replay|restart_epoch. "
                "'replay' replays consumed microbatches exactly (deterministic but slow), "
                "'restart_epoch' skips replay and shifts dataset epoch for O(1) resume, "
                "'auto' replays only when replay cost is below train.resume_replay_max_micro_batches."
            )
        },
    )

    resume_replay_max_micro_batches: int = field(
        default=10_000,
        metadata={
            "help": (
                "Replay threshold used by train.resume_data_strategy=auto. "
                "If consumed_micro_batches exceeds this value, resume switches to restart_epoch."
            )
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


@dataclass
class Config:
    """Top-level training config bundle."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _split_flat_dict(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split a flat mapping into model/data/train field mappings.

    :param dict[str, Any] raw: Flat config mapping.
    :raises ValueError: If unknown keys are present.
    :return tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: Section mappings.
    """
    model_keys = {f.name for f in fields(ModelConfig)}
    data_keys = {f.name for f in fields(DataConfig)}
    train_keys = {f.name for f in fields(TrainConfig)}

    model_dict: dict[str, Any] = {}
    data_dict: dict[str, Any] = {}
    train_dict: dict[str, Any] = {}
    unknown: dict[str, Any] = {}

    for key, value in raw.items():
        if key in model_keys:
            model_dict[key] = value
        elif key in data_keys:
            data_dict[key] = value
        elif key in train_keys:
            train_dict[key] = value
        else:
            unknown[key] = value

    if unknown:
        keys = ", ".join(sorted(unknown.keys()))
        raise ValueError(f"Unknown keys in config file (not in ModelConfig/DataConfig/TrainConfig): {keys}")

    return model_dict, data_dict, train_dict


def _split_nested_or_flat_sections(
    raw: dict[str, Any], *, format_name: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split nested/flat config mappings into model/data/train sections.

    :param dict[str, Any] raw: Parsed config mapping.
    :param str format_name: Format label for error messages.
    :raises ValueError: If config shape is invalid.
    :return tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: Section mappings.
    """
    if any(key in raw for key in ("model", "data", "train")):
        unknown_top = sorted(key for key in raw.keys() if key not in {"model", "data", "train"})
        if unknown_top:
            raise ValueError(
                f"Unknown top-level keys in nested {format_name} config "
                f"(expected only model/data/train): {', '.join(unknown_top)}"
            )
        model_dict = raw.get("model", {}) or {}
        data_dict = raw.get("data", {}) or {}
        train_dict = raw.get("train", {}) or {}
        if (
            not isinstance(model_dict, dict)
            or not isinstance(data_dict, dict)
            or not isinstance(train_dict, dict)
        ):
            raise ValueError(f"{format_name} config sections model/data/train must be dicts.")
        return model_dict, data_dict, train_dict

    return _split_flat_dict(raw)


def _resolve_variables(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``$variables.*`` references in a config mapping.

    Supported forms:
      - full replacement: ``$variables.foo``
      - inline replacement: ``{$variables.foo}`` and ``${variables.foo}``

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
        """Resolve one variable path from the variables mapping.

        :param str path: Dot-separated variables path.
        :raises ValueError: If a variable is unknown or circular.
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
        """Stringify and substitute a variable regex match.

        :param re.Match[str] match: Match containing the variable path.
        :return str: Resolved string value.
        """
        return str(_lookup_var(match.group(1)))

    def _resolve_value(value: Any) -> Any:
        """Resolve variables in a nested value tree.

        :param Any value: Nested value.
        :return Any: Resolved value.
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
        """Collect dotted leaf paths from a nested variable mapping.

        :param str prefix: Current path prefix.
        :param Any value: Nested variable value.
        :return list[str]: Leaf variable paths.
        """
        if isinstance(value, dict):
            out: list[str] = []
            for key, item in value.items():
                part = str(key).strip()
                child = f"{prefix}.{part}" if prefix else part
                out.extend(_collect_var_leaf_paths(child, item))
            return out
        return [prefix]

    # Resolve every variable leaf eagerly so missing/circular references fail fast,
    # even when a variable is not directly used in model/data/train sections.
    for var_path in _collect_var_leaf_paths("", raw_vars):
        _lookup_var(var_path)

    return {k: _resolve_value(v) for k, v in data.items() if k != "variables"}


def load_config_sections(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load strict model/data/train mappings from a YAML or JSON file.

    :param str | Path path: Config path.
    :raises ValueError: If file content is invalid for this config schema.
    :return tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: Parsed section mappings.
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
    resolved_raw = _resolve_variables(raw)
    return _split_nested_or_flat_sections(
        resolved_raw, format_name="YAML" if suffix in {".yaml", ".yml"} else "JSON"
    )


def _unwrap_optional_type(field_type: Any) -> tuple[Any, bool]:
    """Unwrap ``Optional[T]`` annotations.

    :param Any field_type: Raw type annotation.
    :return tuple[Any, bool]: Unwrapped type and whether ``None`` is allowed.
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
    """Apply one dotted override expression to a ``Config`` object.

    Format: ``section.field=value`` where ``section`` is one of model/data/train.

    :param Config cfg: Existing config bundle.
    :param str override: Override expression.
    :raises ValueError: If expression/section/field/value is invalid.
    :return Config: New config with one updated field.
    """
    text = str(override).strip()
    if "=" not in text:
        raise ValueError(f"Invalid override {override!r}. Expected format like model.hidden_size=768.")
    path, raw_value = text.split("=", 1)
    path = path.strip()
    raw_value = raw_value.strip()
    if "." not in path:
        raise ValueError(
            f"Invalid override path {path!r}. Expected format section.field (for example train.max_steps)."
        )
    section, field_name = path.split(".", 1)
    section = str(section).strip().lower()
    field_name = str(field_name).strip()
    if section not in {"model", "data", "train"}:
        raise ValueError(f"Unknown override section {section!r}; expected one of model|data|train.")
    section_obj = getattr(cfg, section)
    if not hasattr(section_obj, field_name):
        raise ValueError(f"Unknown override field {section}.{field_name!r}.")
    type_hints = get_type_hints(type(section_obj))
    field_type = type_hints.get(field_name, Any)
    coerced = _coerce_override_value(raw_value, field_type)
    new_section = replace(section_obj, **{field_name: coerced})
    return replace(cfg, **{section: new_section})


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
    :param bool replace_hyphen: Whether to normalize ``-`` to ``_`` before alias lookup.
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
        name="train.torch_compile_mode",
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
        name="train.torch_compile_scope",
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
        name="train.torch_compile_backend",
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
        name="train.wandb_watch",
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
        name="model.hf_attention_kernel",
        value=value,
        aliases=_HF_ATTN_KERNEL_ALIASES,
        choices=_HF_ATTN_KERNEL_CHOICES,
        replace_hyphen=True,
    )


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
            f"Expected {int(RUN_CONFIG_SCHEMA_VERSION)}. "
            "Backward resume/export compatibility is not guaranteed before stable release."
        )


_SnapshotConfigT = TypeVar("_SnapshotConfigT", "ModelConfig", "DataConfig")


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
    """Parse persisted ``model_config.json`` into ``ModelConfig``.

    :param dict[str, object] raw: Raw model config mapping.
    :param str source: Source path for error messages.
    :return ModelConfig: Parsed model configuration.
    """
    return _load_snapshot_dataclass(raw, cls=ModelConfig, source=source, config_name="model_config.json")


def load_data_config_snapshot(raw: dict[str, object], *, source: str) -> DataConfig:
    """Parse persisted ``data_config.json`` into ``DataConfig``.

    :param dict[str, object] raw: Raw data config mapping.
    :param str source: Source path for error messages.
    :return DataConfig: Parsed data configuration.
    """
    return _load_snapshot_dataclass(raw, cls=DataConfig, source=source, config_name="data_config.json")


def _looks_like_hf_deberta_checkpoint(value: str) -> bool:
    """Return whether a model source appears to be an HF DeBERTa v2/v3 checkpoint id.

    :param str value: Model source string.
    :return bool: True when source matches known HF DeBERTa hub-id prefixes.
    """
    v = f"/{str(value).strip().lower()}"
    return any(f"/{prefix}" in v for prefix in _HF_DEBERTA_PRETRAINED_PREFIXES)


# Canonical field-name sets for cross-backbone / scratch-vs-pretrained validation.
# rope_knobs = _ROPE_SCRATCH_ONLY_FIELDS | _ROPE_PRETRAINED_OVERRIDE_FIELDS | {"attention_implementation"}
_ROPE_SCRATCH_ONLY_FIELDS: frozenset[str] = frozenset(
    {
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
    }
)
_ROPE_PRETRAINED_OVERRIDE_FIELDS: frozenset[str] = frozenset(
    {
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
    }
)


def validate_model_config(cfg: ModelConfig) -> None:
    """Validate model config semantics and normalize constrained values.

    :param ModelConfig cfg: Model configuration.
    """
    cfg.backbone_type = _ensure_choice("model.backbone_type", cfg.backbone_type, _BACKBONE_CHOICES)
    cfg.profile = _ensure_choice("model.profile", cfg.profile, _MODEL_PROFILE_CHOICES)
    cfg.hf_attention_kernel = _normalize_hf_attention_kernel(cfg.hf_attention_kernel)
    cfg.hf_model_size = _ensure_choice("model.hf_model_size", cfg.hf_model_size, _HF_MODEL_SIZE_CHOICES)
    cfg.norm_arch = _ensure_choice("model.norm_arch", cfg.norm_arch, _NORM_ARCH_CHOICES)
    cfg.attention_implementation = _ensure_choice(
        "model.attention_implementation", cfg.attention_implementation, _ATTN_IMPL_CHOICES
    )
    cfg.ffn_type = _ensure_choice("model.ffn_type", cfg.ffn_type, _FFN_CHOICES)
    cfg.embedding_sharing = _ensure_choice(
        "model.embedding_sharing", cfg.embedding_sharing, _EMBED_SHARING_CHOICES
    )
    cfg.tokenizer_name_or_path = str(cfg.tokenizer_name_or_path).strip()
    cfg.pretrained_discriminator_path = str(cfg.pretrained_discriminator_path or "").strip()
    if cfg.pretrained_generator_path is not None:
        cfg.pretrained_generator_path = str(cfg.pretrained_generator_path).strip() or None

    if not cfg.tokenizer_name_or_path:
        raise ValueError("model.tokenizer_name_or_path must be a non-empty tokenizer source.")
    if not bool(cfg.from_scratch) and not cfg.pretrained_discriminator_path:
        raise ValueError(
            "model.pretrained_discriminator_path must be set when model.from_scratch=false "
            "(weights are loaded from this source)."
        )

    if cfg.max_position_embeddings is not None and int(cfg.max_position_embeddings) <= 0:
        raise ValueError("model.max_position_embeddings must be > 0 when provided.")
    if float(cfg.rotary_pct) <= 0.0 or float(cfg.rotary_pct) > 1.0:
        raise ValueError("model.rotary_pct must be in (0, 1].")
    if int(cfg.tokenizer_vocab_multiple) <= 0:
        raise ValueError("model.tokenizer_vocab_multiple must be >= 1.")
    if cfg.tokenizer_vocab_target is not None and int(cfg.tokenizer_vocab_target) <= 0:
        raise ValueError("model.tokenizer_vocab_target must be > 0 when provided.")

    defaults = ModelConfig()

    # Explicit dependency check: rope-only knobs are invalid in HF-compat mode.
    if cfg.backbone_type == "hf_deberta_v2":
        if cfg.hf_max_position_embeddings is not None and int(cfg.hf_max_position_embeddings) <= 0:
            raise ValueError("model.hf_max_position_embeddings must be > 0 when provided.")
        if cfg.hf_max_position_embeddings is not None and not bool(cfg.from_scratch):
            raise ValueError(
                "model.hf_max_position_embeddings is only supported when model.from_scratch=true for "
                "hf_deberta_v2 runs."
            )
        rope_knobs = (
            _ROPE_SCRATCH_ONLY_FIELDS | _ROPE_PRETRAINED_OVERRIDE_FIELDS | {"attention_implementation"}
        )
        changed = [name for name in rope_knobs if getattr(cfg, name) != getattr(defaults, name)]
        if changed:
            raise ValueError(
                "These options are only valid when model.backbone_type='rope': " + ", ".join(sorted(changed))
            )
    else:
        default_hf_kernel = defaults.hf_attention_kernel
        if cfg.hf_attention_kernel != default_hf_kernel:
            warnings.warn(
                "model.hf_attention_kernel only applies when model.backbone_type='hf_deberta_v2'. "
                f"Current value ({cfg.hf_attention_kernel!r}) has no effect on the rope backbone.",
                UserWarning,
                stacklevel=2,
            )
        if cfg.hf_max_position_embeddings is not None:
            warnings.warn(
                "model.hf_max_position_embeddings only applies when model.backbone_type='hf_deberta_v2'. "
                f"Current value ({cfg.hf_max_position_embeddings!r}) has no effect on the rope backbone.",
                UserWarning,
                stacklevel=2,
            )
        if cfg.hf_model_size != defaults.hf_model_size:
            warnings.warn(
                "model.hf_model_size only applies when model.backbone_type='hf_deberta_v2'. "
                f"Current value ({cfg.hf_model_size!r}) has no effect on the rope backbone.",
                UserWarning,
                stacklevel=2,
            )

    if cfg.backbone_type == "rope" and bool(cfg.from_scratch):
        set_pretrained = [name for name in _ROPE_PRETRAINED_OVERRIDE_FIELDS if getattr(cfg, name) is not None]
        if set_pretrained:
            raise ValueError(
                "These options apply only when model.from_scratch=false: " + ", ".join(sorted(set_pretrained))
            )

    if cfg.backbone_type == "rope" and not bool(cfg.from_scratch):
        invalid_sources: list[str] = []
        for field_name in ("pretrained_discriminator_path", "pretrained_generator_path"):
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

        changed_scratch_knobs = [
            name for name in _ROPE_SCRATCH_ONLY_FIELDS if getattr(cfg, name) != getattr(defaults, name)
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

    if cfg.pretrained_generator_path:
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
                "model.pretrained_generator_path is provided: " + ", ".join(sorted(derived_knobs))
            )

    if not bool(cfg.from_scratch) and not cfg.pretrained_generator_path:
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
                "model.from_scratch=false with derived generator weights (pretrained_generator_path unset) "
                "cannot use generator shape overrides because generator weights are loaded from the "
                "discriminator source. Set model.pretrained_generator_path to an explicit generator "
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
    if (
        bool(cfg.pack_sequences)
        and bool(cfg.block_cross_document_attention)
        and int(cfg.max_seq_length) > int(_DENSE_DOC_BLOCK_WARN_SEQ_LEN)
    ):
        warnings.warn(
            "data.block_cross_document_attention builds dense O(S^2) pairwise masks. "
            f"Configured data.max_seq_length={int(cfg.max_seq_length)} may be expensive; "
            f"consider reducing sequence length or disabling data.block_cross_document_attention "
            f"until sparse/segment-aware attention support lands (warning threshold: "
            f"{int(_DENSE_DOC_BLOCK_WARN_SEQ_LEN)}).",
            UserWarning,
            stacklevel=2,
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
    cfg.torch_compile_scope = _normalize_torch_compile_scope(cfg.torch_compile_scope)
    cfg.torch_compile_backend = _normalize_torch_compile_backend(cfg.torch_compile_backend)
    cfg.wandb_watch = _normalize_wandb_watch(cfg.wandb_watch)
    cfg.resume_data_strategy = _ensure_choice(
        "train.resume_data_strategy",
        cfg.resume_data_strategy,
        _RESUME_DATA_STRATEGY_CHOICES,
    )

    cfg.mixed_precision = normalize_mixed_precision(cfg.mixed_precision)
    if cfg.output_dir is not None and not str(cfg.output_dir).strip():
        cfg.output_dir = None
    if cfg.resume_from_checkpoint is not None:
        resume_from_checkpoint = str(cfg.resume_from_checkpoint).strip()
        cfg.resume_from_checkpoint = resume_from_checkpoint if resume_from_checkpoint else None
    if not str(cfg.project_name).strip():
        raise ValueError("train.project_name must be non-empty.")
    if bool(cfg.overwrite_output_dir) and bool(cfg.resume_from_checkpoint):
        raise ValueError(
            "train.overwrite_output_dir=true cannot be combined with train.resume_from_checkpoint. "
            "Overwrite would delete checkpoints before resume."
        )

    for _name, _min in (
        ("max_steps", 1),
        ("per_device_train_batch_size", 1),
        ("gradient_accumulation_steps", 1),
        ("warmup_steps", 0),
        ("logging_steps", 0),
        ("save_steps", 0),
        ("save_total_limit", 0),
        ("wandb_watch_log_freq", 1),
        ("resume_replay_max_micro_batches", 0),
    ):
        if int(getattr(cfg, _name)) < _min:
            raise ValueError(f"train.{_name} must be >= {_min}.")

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
    disc_lr = float(cfg.discriminator_learning_rate)
    if disc_lr != -1.0 and disc_lr <= 0.0:
        raise ValueError("train.discriminator_learning_rate must be -1 (inherit) or > 0.")
    if not isinstance(cfg.decoupled_training, bool):
        raise ValueError(
            "train.decoupled_training must be a boolean (true/false). "
            f"Got {type(cfg.decoupled_training).__name__}."
        )

    defaults = TrainConfig()

    # Warn when compile-specific knobs are configured while compile is disabled.
    if not bool(cfg.torch_compile):
        for _knob in ("torch_compile_scope", "torch_compile_mode", "torch_compile_backend"):
            if getattr(cfg, _knob) != getattr(defaults, _knob):
                _short = _knob.removeprefix("torch_compile_")
                warnings.warn(
                    f"train.{_knob} has no effect when train.torch_compile=false. "
                    f"Current {_short} ({getattr(cfg, _knob)!r}) will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )

    if cfg.report_to != "wandb":
        if cfg.wandb_watch != defaults.wandb_watch:
            warnings.warn(
                "train.wandb_watch only applies when train.report_to='wandb'. "
                f"Current report_to ({cfg.report_to!r}) will ignore watch mode {cfg.wandb_watch!r}.",
                UserWarning,
                stacklevel=2,
            )
        if int(cfg.wandb_watch_log_freq) != int(defaults.wandb_watch_log_freq):
            warnings.warn(
                "train.wandb_watch_log_freq only applies when train.report_to='wandb'. "
                f"Current report_to ({cfg.report_to!r}) will ignore watch frequency "
                f"{int(cfg.wandb_watch_log_freq)}.",
                UserWarning,
                stacklevel=2,
            )


def apply_profile_defaults(*, model_cfg: ModelConfig, train_cfg: TrainConfig) -> None:
    """Apply profile/backbone-specific defaults while preserving explicit non-default values.

    :param ModelConfig model_cfg: Model config to update in-place.
    :param TrainConfig train_cfg: Train config to update in-place.
    """

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

    explicit_model_fields = _explicit_fields(model_cfg)
    explicit_train_fields = _explicit_fields(train_cfg)

    profile = str(model_cfg.profile).strip().lower()
    model_defaults = ModelConfig()
    train_defaults = TrainConfig()

    if profile == "deberta_v3_parity":
        # Profile intends parity defaults on HF DeBERTa-v2 path unless user picked otherwise.
        if "backbone_type" not in explicit_model_fields and str(model_cfg.backbone_type) == str(
            model_defaults.backbone_type
        ):
            model_cfg.backbone_type = "hf_deberta_v2"

        if "embedding_sharing" not in explicit_model_fields and str(model_cfg.embedding_sharing) == str(
            model_defaults.embedding_sharing
        ):
            model_cfg.embedding_sharing = "gdes"

        if "hf_attention_kernel" not in explicit_model_fields and str(model_cfg.hf_attention_kernel) == str(
            model_defaults.hf_attention_kernel
        ):
            model_cfg.hf_attention_kernel = "dynamic"

    # Parity++ defaults apply to hf_deberta_v2 unless explicitly overridden.
    if str(model_cfg.backbone_type).strip().lower() == "hf_deberta_v2":
        if "mask_token_prob" not in explicit_train_fields and float(train_cfg.mask_token_prob) == float(
            train_defaults.mask_token_prob
        ):
            train_cfg.mask_token_prob = 1.0
        if "random_token_prob" not in explicit_train_fields and float(train_cfg.random_token_prob) == float(
            train_defaults.random_token_prob
        ):
            train_cfg.random_token_prob = 0.0
        if "disc_loss_weight" not in explicit_train_fields and float(train_cfg.disc_loss_weight) == float(
            train_defaults.disc_loss_weight
        ):
            train_cfg.disc_loss_weight = 10.0
        if "adam_epsilon" not in explicit_train_fields and float(train_cfg.adam_epsilon) == float(
            train_defaults.adam_epsilon
        ):
            train_cfg.adam_epsilon = 1e-6
        if "warmup_steps" not in explicit_train_fields and int(train_cfg.warmup_steps) == int(
            train_defaults.warmup_steps
        ):
            train_cfg.warmup_steps = 10_000
        if "token_weighted_gradient_accumulation" not in explicit_train_fields and bool(
            train_cfg.token_weighted_gradient_accumulation
        ) == bool(train_defaults.token_weighted_gradient_accumulation):
            train_cfg.token_weighted_gradient_accumulation = True


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
        if bool(train_cfg.decoupled_training) and embed_sharing == "es":
            raise ValueError(
                "train.decoupled_training is incompatible with model.embedding_sharing='es' because "
                "shared embedding parameters would be stepped in both generator and discriminator phases. "
                "Use embedding_sharing='gdes' or 'none' for decoupled training."
            )
