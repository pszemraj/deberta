#!/usr/bin/env python3
"""Generate the training config reference from dataclass schema."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import types
from pathlib import Path
from typing import Any, Union, get_args, get_origin


def _ensure_src_on_path() -> Path:
    """Add repo ``src`` to ``sys.path`` and return repo root."""

    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    return repo_root


REPO_ROOT = _ensure_src_on_path()

from deberta import config as cfg_mod  # noqa: E402
from deberta.config import Config, asdict_without_private, iter_leaf_paths_for_dataclass  # noqa: E402
from deberta.utils.mapping import flatten_mapping  # noqa: E402
from deberta.utils.types import unwrap_optional_type  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "docs" / "guides" / "config-reference.md"

TOP_LEVEL_NOTE = {
    "model": "Model/backbone, tokenizer, embedding sharing, dropout, and FlashDeBERTa policy.",
    "data": "Dataset source, streaming, packing length, and cross-document mask behavior.",
    "train": "Optimization loop, precision, DataLoader, compile, RTD objective, and checkpoint behavior.",
    "optim": "Optimizer, scheduler, learning rates, weight decay, and gradient clipping.",
    "logging": "Local tracker, W&B, output directories, and debug metric logging.",
}

VALID_VALUES = {
    "model.profile": sorted(cfg_mod._MODEL_PROFILE_CHOICES),
    "model.backbone_type": sorted(cfg_mod._BACKBONE_CHOICES),
    "model.embedding_sharing": sorted(cfg_mod._EMBED_SHARING_CHOICES),
    "model.hf.model_size": sorted(cfg_mod._HF_MODEL_SIZE_CHOICES),
    "model.hf.attention_kernel": sorted(cfg_mod._HF_ATTN_KERNEL_CHOICES),
    "model.hf.attention_impl": sorted(cfg_mod._HF_ATTN_IMPL_CHOICES),
    "model.rope.norm_arch": sorted(cfg_mod._NORM_ARCH_CHOICES),
    "model.rope.attention_implementation": sorted(cfg_mod._ATTN_IMPL_CHOICES),
    "model.rope.ffn_type": sorted(cfg_mod._FFN_CHOICES),
    "model.rope.pretrained.norm_arch": sorted(cfg_mod._NORM_ARCH_CHOICES),
    "model.rope.pretrained.ffn_type": sorted(cfg_mod._FFN_CHOICES),
    "train.mixed_precision": sorted(cfg_mod._MIXED_PRECISION_CANONICAL),
    "train.sdpa_kernel": sorted(cfg_mod._SDPA_KERNEL_CHOICES),
    "train.compile.mode": sorted(cfg_mod._TORCH_COMPILE_MODE_CHOICES),
    "train.compile.scope": sorted(cfg_mod._TORCH_COMPILE_SCOPE_CHOICES),
    "train.compile.backend": sorted(cfg_mod._TORCH_COMPILE_BACKEND_CHOICES),
    "train.checkpoint.resume_data_strategy": sorted(cfg_mod._RESUME_DATA_STRATEGY_CHOICES),
    "optim.scheduler.type": sorted(cfg_mod._LR_SCHEDULER_CHOICES),
    "logging.backend": sorted(cfg_mod._LOGGING_BACKEND_CHOICES),
    "logging.wandb.watch": sorted(cfg_mod._WANDB_WATCH_CHOICES),
}

ALIASES = {
    "model.hf.attention_kernel": cfg_mod._HF_ATTN_KERNEL_ALIASES,
    "train.mixed_precision": cfg_mod._MIXED_PRECISION_ALIASES,
    "train.sdpa_kernel": cfg_mod._SDPA_KERNEL_ALIASES,
    "train.compile.mode": cfg_mod._TORCH_COMPILE_MODE_ALIASES,
    "train.compile.scope": cfg_mod._TORCH_COMPILE_SCOPE_ALIASES,
    "train.compile.backend": cfg_mod._TORCH_COMPILE_BACKEND_ALIASES,
    "logging.wandb.watch": cfg_mod._WANDB_WATCH_ALIASES,
}

REQUIREDNESS = {
    "model.pretrained.discriminator_path": "conditional - required when `model.from_scratch=false`",
    "data.source.dataset_name": "conditional - one of `data.source.dataset_name`, `data.source.data_files`, or `data.source.load_from_disk` is required",
    "data.source.data_files": "conditional - one of `data.source.dataset_name`, `data.source.data_files`, or `data.source.load_from_disk` is required",
    "data.source.load_from_disk": "conditional - one of `data.source.dataset_name`, `data.source.data_files`, or `data.source.load_from_disk` is required",
}

FIELD_DOCS = {
    "model.profile": "Selects profile-level effective defaults. `deberta_v3_parity` preserves DeBERTa-v3 RTD conventions and changes several train/optim defaults when those keys are unset.",
    "model.backbone_type": "`hf_deberta_v2` is the native DeBERTa-v2/v3 parity backbone. `rope` is the modern RoPE/RMSNorm/SwiGLU-capable path and is the preferred speed baseline.",
    "model.from_scratch": "When `false`, `model.pretrained.discriminator_path` is required. RoPE pretrained mode requires DebertaRoPE checkpoints, not Microsoft HF DeBERTa checkpoints.",
    "model.embedding_sharing": "`gdes` is recommended. `es` shares embeddings and is incompatible with `train.decoupled_training=true`; if using `es`, `optim.lr.generator` must inherit or equal `optim.lr.base`.",
    "model.gradient_checkpointing": "Enables backbone gradient checkpointing. Useful for memory pressure, but slower and only worth enabling after checking throughput.",
    "model.tokenizer.name_or_path": "Tokenizer id/path. Must be non-empty. Use a vocab compatible with pretrained checkpoints unless `model.from_scratch=true`.",
    "model.tokenizer.allow_vocab_resize": "Allows embedding resize when tokenizer vocab differs from model config. Use only when intentionally changing vocab shape.",
    "model.tokenizer.vocab_target": "Optional target vocab size. If set, must be > 0. Usually leave `null`; `vocab_multiple` handles padding to efficient multiples.",
    "model.tokenizer.vocab_multiple": "Vocabulary size multiple for resize/padding. Must be >= 1. Use 128 for tensor-core-friendly vocab padding when resizing.",
    "model.hf.model_size": "Native HF DeBERTa-v2/v3 model size preset. Applies only when `model.backbone_type=hf_deberta_v2`; ignored with a warning for `rope`.",
    "model.hf.attention_kernel": "Native eager DeBERTa attention implementation choice. Applies only to `hf_deberta_v2`; ignored for `rope`.",
    "model.hf.attention_impl": "`eager` is the portability default. `flash` is valid only with `model.backbone_type=hf_deberta_v2`, requires dropout disabled, and uses the JSON route/kernel policy under `model.hf.flash.*`. Route/kernel defaults are capability-scoped (measured rows ship for `sm_120`); other GPUs run flash with conservative defaults - see `docs/advanced/gpu-support.md`.",
    "model.hf.flash.force_varlen": "Only used when `model.hf.attention_impl=flash`. Experimental routing override that forces padded batches onto varlen route; use only for benchmarking/debugging.",
    "model.hf.flash.varlen_min_seq_len": "Only used when `model.hf.attention_impl=flash`. Optional varlen threshold. `null` uses the JSON route table; if set, must be > 0.",
    "model.hf.flash.docblock_bias_seq_len": "Only used when `model.hf.attention_impl=flash`. Exact-length dense doc-block route override. `null` uses the JSON table, which selects dense `docblock_bias` for measured packed `1024`/`2048`/`4096` buckets on `sm_120` and ragged `docblock` elsewhere (other hardware and unlisted shapes). Set a positive exact sequence length to force dense only at that length on any GPU; set `0` to force-disable dense doc-block routing and use ragged segment metadata. A positive value bypasses the table's `max_batch_size`/`max_seq_len` safety bounds - the dense route saves a `(B,H,S,S)` bias for backward, so a stale knob plus a larger batch can OOM; training warns once when the knob forces dense past the bounds. Clear it after tuning experiments.",
    "model.hf.flash.local_bias_seq_len": "Only used when `model.hf.attention_impl=flash`. Dense local-bias route seq-len override for plain (non-packed) batches. `null` uses table policy (shipped rows are `sm_120`-scoped, so the route stays off on other GPUs); `0` disables the local-bias route; a positive value allows it only at that exact sequence length. On hardware without table rows, set this together with `local_bias_max_batch_size`. Independent of `model.hf.flash.docblock_bias_seq_len`.",
    "model.hf.flash.local_bias_max_batch_size": "Only used when `model.hf.attention_impl=flash`. Dense local-bias route cap. `null` uses table policy; `0` disables. On hardware without table rows, set this together with `local_bias_seq_len` to opt in. Dense routes can be memory-heavy and hardware-sensitive.",
    "model.hf.flash.eager_dense_max_seq_len": "Only used when `model.hf.attention_impl=flash`. Debugging fallback threshold. Must be >= 0. Leave `0` unless explicitly comparing eager dense attention inside flash experiments.",
    "model.hf.flash.kernel_overrides_path": "Only used when `model.hf.attention_impl=flash`. Optional JSON tuning table path for flash route/kernel policies. Experimental; keep run-local and record with checkpoints for reproducibility.",
    "model.hf.max_position_embeddings": "HF DeBERTa-v2/v3 max positions override. Only valid for `hf_deberta_v2` with `model.from_scratch=true`; must be > 0 if set.",
    "model.pretrained.discriminator_path": "Required when `model.from_scratch=false`. Use a discriminator checkpoint compatible with `model.backbone_type`.",
    "model.pretrained.generator_path": "Optional separate generator checkpoint. If set, `model.generator.*` shape overrides must be unset.",
    "model.generator.num_hidden_layers": "Scratch/derived generator layer-count override. Must be unset when `model.pretrained.generator_path` is provided.",
    "model.generator.hidden_size": "Scratch/derived generator hidden-size override. Invalid with `model.from_scratch=false` unless loading a separate generator checkpoint.",
    "model.generator.intermediate_size": "Scratch/derived generator FFN-size override. Leave `null` to derive from profile/backbone defaults.",
    "model.generator.num_attention_heads": "Scratch/derived generator attention-head override. Hidden size must remain divisible by heads in the built backbone.",
    "model.rope.hidden_size": "RoPE scratch hidden size. Only valid when `model.backbone_type=rope` and `model.from_scratch=true`; must be > 0 and divisible by heads.",
    "model.rope.num_hidden_layers": "RoPE scratch layer count. Only valid for scratch RoPE; must be > 0.",
    "model.rope.num_attention_heads": "RoPE scratch attention heads. Only valid for scratch RoPE; must be > 0 and divide `model.rope.hidden_size`.",
    "model.rope.intermediate_size": "RoPE scratch FFN width before optional SwiGLU adjustment. Must be > 0.",
    "model.rope.hidden_act": "RoPE scratch hidden activation name passed into the backbone. Use `gelu` for BERT-style parity unless intentionally experimenting.",
    "model.rope.rope_theta": "RoPE base theta for scratch models. Larger values extend long-context bias; default is standard 10000.",
    "model.rope.rotary_pct": "Fraction of each head using RoPE. Must be in (0, 1]. Use 1.0 unless reproducing partial-rotary ablations.",
    "model.rope.use_absolute_position_embeddings": "Adds learned absolute position embeddings to RoPE scratch models. Usually keep false; enabling changes the modern baseline.",
    "model.rope.max_position_embeddings": "Optional scratch RoPE max positions. If set, must be > 0. Leave `null` to derive from data/context builder.",
    "model.rope.type_vocab_size": "RoPE token-type embedding size. Use 0/1/2 depending on segment-id needs; default 2 matches BERT-style inputs.",
    "model.rope.norm_arch": "`post` is standard LayerNorm placement. `keel` enables KEEL scaling and activates `keel_alpha_*` controls.",
    "model.rope.norm_eps": "RoPE norm epsilon. Must be positive in practice; default 1e-6 is recommended for bf16 stability.",
    "model.rope.keel_alpha_init": "Only used when `model.rope.norm_arch=keel`; ignored with a warning for `post`.",
    "model.rope.keel_alpha_learnable": "Only used when `model.rope.norm_arch=keel`; ignored with a warning for `post`.",
    "model.rope.attention_implementation": "`sdpa` uses PyTorch scaled dot-product attention and is recommended. `eager` is for debugging/parity.",
    "model.rope.ffn_type": "`mlp` is BERT-style. `swiglu` is experimental-modern; if enabled, `swiglu_adjust_intermediate` controls width adjustment.",
    "model.rope.use_bias": "Adds linear biases in RoPE modules. Keep false for the modern baseline unless matching a checkpoint that used biases.",
    "model.rope.swiglu_adjust_intermediate": "Only used when `model.rope.ffn_type=swiglu`; ignored with warning for `mlp`.",
    "model.rope.initializer_range": "RoPE scratch initializer standard deviation. Default 0.02 follows BERT/DeBERTa convention.",
    "model.rope.pretrained.max_position_embeddings": "Pretrained RoPE-only override. Only valid when `model.backbone_type=rope` and `model.from_scratch=false`; must be > 0 if set.",
    "model.rope.pretrained.rope_theta": "Pretrained RoPE-only theta override. Use only to adapt a known compatible checkpoint.",
    "model.rope.pretrained.rotary_pct": "Pretrained RoPE-only rotary fraction override. If set, must be in (0, 1].",
    "model.rope.pretrained.use_absolute_position_embeddings": "Pretrained RoPE-only absolute-position override. Use only when the checkpoint/config expects it.",
    "model.rope.pretrained.type_vocab_size": "Pretrained RoPE-only token-type vocab override. Use only when adapting a compatible checkpoint.",
    "model.rope.pretrained.norm_arch": "Pretrained RoPE-only norm architecture override.",
    "model.rope.pretrained.norm_eps": "Pretrained RoPE-only norm epsilon override.",
    "model.rope.pretrained.keel_alpha_init": "Pretrained RoPE-only KEEL alpha init override. Applies only to KEEL checkpoints.",
    "model.rope.pretrained.keel_alpha_learnable": "Pretrained RoPE-only KEEL alpha learnable override. Applies only to KEEL checkpoints.",
    "model.rope.pretrained.ffn_type": "Pretrained RoPE-only FFN type override.",
    "model.rope.pretrained.use_bias": "Pretrained RoPE-only linear-bias override. Must match checkpoint architecture.",
    "model.rope.pretrained.initializer_range": "Pretrained RoPE-only initializer override. Usually irrelevant when loading weights.",
    "model.dropout.hidden_prob": "Hidden dropout probability or `null`. Flash attention requires this to be null or 0.0; default disables dropout for parity/throughput.",
    "model.dropout.attention_probs_prob": "Attention-prob dropout probability or `null`. Flash attention requires this to be null or 0.0.",
    "data.source.dataset_name": "HF dataset id/builder. At least one of `dataset_name`, `data_files`, or `load_from_disk` is required. Can combine with `data_files` for builders such as `text`.",
    "data.source.dataset_config_name": "HF dataset config/subset name. Requires `data.source.dataset_name`; cannot be used with `load_from_disk`.",
    "data.source.data_files": "Comma-separated local/remote data files. At least one source key is required. With no `dataset_name`, files are loaded via the HF `text` builder.",
    "data.source.load_from_disk": "Path to a dataset saved by HF `load_from_disk`. Requires `data.source.streaming=false` and cannot combine with `dataset_name`, `dataset_config_name`, or `data_files`.",
    "data.source.train_split": "Dataset split to train on. Must exist in the selected dataset/source.",
    "data.source.text_column_name": "Column containing raw text. Must exist in the loaded dataset examples.",
    "data.source.streaming": "Use HF streaming. Required false with `load_from_disk`; true is recommended for large remote datasets.",
    "data.source.shuffle_buffer_size": "Streaming reservoir shuffle buffer. Must be >= 0; for non-streaming data, only 0 or 1 are valid. Set 0 to disable shuffle.",
    "data.packing.enabled": "Pack text into fixed-length training blocks. Required for `block_cross_document_attention=true`; recommended true for throughput.",
    "data.packing.max_seq_length": "Packed block length / training sequence length. Must be >= 8. Longer doc-block masks can be O(S^2) on non-segment-aware backends.",
    "data.packing.block_cross_document_attention": "Prevents attention across packed document boundaries. Requires packing enabled. For RoPE with strict `train.sdpa_kernel=flash`, this is invalid; use auto/mem_efficient/math.",
    "train.seed": "Base random seed used by training, dataloader shuffling, and reproducibility setup.",
    "train.max_steps": "Number of optimizer steps. Must be >= 1. This is the primary duration control; there is no epoch-based scheduler in this trainer.",
    "train.per_device_train_batch_size": "Micro-batch size per process. Must be >= 1. Effective tokens per optimizer step also multiply by `gradient_accumulation_steps`.",
    "train.gradient_accumulation_steps": "Micro-batches per optimizer step. Must be >= 1. With token-weighted GA, losses are weighted by active token count.",
    "train.token_weighted_gradient_accumulation": "Weights accumulated losses by active tokens. Effective default stays true for `hf_deberta_v2`; recommended for packed/ragged batches.",
    "train.mixed_precision": "`bf16` is recommended on modern NVIDIA GPUs. `no` runs full precision and is slower/more memory-heavy.",
    "train.tf32": "Enables TensorFloat-32 matmul policy where supported. Recommended true for speed on Ampere+ GPUs.",
    "train.sdpa_kernel": "PyTorch SDPA kernel policy. Only affects `model.backbone_type=rope` with `model.rope.attention_implementation=sdpa`; ignored with warning for `hf_deberta_v2`. `flash` is invalid with packed cross-document blocking because strict flash SDPA cannot consume doc-block masks.",
    "train.decoupled_training": "Runs generator/discriminator optimizer phases separately. Incompatible with `model.embedding_sharing=es`; recommended true with `gdes`.",
    "train.dataloader.num_workers": "DataLoader worker count. Must be >= 0. Use 0 for debugging; use >0 for streaming throughput if the dataset source is stable.",
    "train.dataloader.pin_memory": "Pins CPU batches before GPU transfer. Recommended true for CUDA training.",
    "train.compile.enabled": "Enables `torch.compile` on selected model scopes. Experimental per route; compile failures are hard errors.",
    "train.compile.mode": "Torch compile mode. Only used when `train.compile.enabled=true`; otherwise ignored with a warning.",
    "train.compile.scope": "Compile target scope. Only used when compile is enabled. `auto` may downgrade for dynamic masks; `backbones` is the main stable scope.",
    "train.compile.backend": "Torch compile backend. Only used when compile is enabled; `inductor` is recommended for CUDA.",
    "train.objective.mlm_probability": "Fraction of active tokens selected for MLM/RTD corruption. Must be in (0, 1).",
    "train.objective.mask_token_prob": "Probability selected tokens become `[MASK]`. Must be >= 0 and with `random_token_prob` sum <= 1. Effective default becomes 1.0 for `hf_deberta_v2` when unset.",
    "train.objective.random_token_prob": "Probability selected tokens become random tokens. Must be >= 0 and with `mask_token_prob` sum <= 1. Effective default becomes 0.0 for `hf_deberta_v2` when unset.",
    "train.objective.mlm_max_ngram": "Maximum masking ngram span. Must be >= 1. Use 1 for simple token masking/parity.",
    "train.objective.sampling_temperature": "Generator sampling temperature for RTD replacements. Must be > 0.",
    "train.objective.gen_loss_weight": "Generator MLM loss weight. Use 1.0 unless intentionally rebalancing RTD phases.",
    "train.objective.disc_loss_weight": "Discriminator RTD loss weight. Effective default becomes 10.0 for `hf_deberta_v2` when unset; raw dataclass default is 50.0.",
    "train.checkpoint.output_dir": "Checkpoint directory. If null, runtime chooses/derives an output path. Also seeds logging output when `logging.output_dir=null`.",
    "train.checkpoint.overwrite_output_dir": "Deletes/reuses an existing output dir. Cannot combine with `resume_from_checkpoint`; dangerous if pointing at a run you need.",
    "train.checkpoint.save_steps": "Checkpoint interval in optimizer steps. Must be >= 0; 0 disables periodic saves. Final export is controlled separately.",
    "train.checkpoint.save_total_limit": "Number of checkpoints to keep. Must be >= 0; 0 means no retention limit enforcement.",
    "train.checkpoint.resume_from_checkpoint": "Checkpoint path to resume from, or null. Cannot combine with `overwrite_output_dir=true`.",
    "train.checkpoint.resume_data_strategy": "`auto` chooses replay/restart behavior. `replay` attempts deterministic dataloader replay; `restart_epoch` restarts data order.",
    "train.checkpoint.resume_replay_max_micro_batches": "Replay budget for resume data alignment. Must be >= 0. Higher values improve exact replay tolerance but cost startup time.",
    "train.checkpoint.export_hf_final": "Exports final discriminator HF artifacts after successful training. Disable for benchmark/probe runs to avoid extra final work.",
    "optim.lr.base": "Base AdamW learning rate. Must be > 0. Used by both phases unless generator/discriminator LR override is positive.",
    "optim.lr.generator": "`-1` inherits `optim.lr.base`; otherwise must be > 0. With `embedding_sharing=es`, must inherit or match base.",
    "optim.lr.discriminator": "`-1` inherits `optim.lr.base`; otherwise must be > 0.",
    "optim.adam.beta1": "Adam beta1. Typical value 0.9.",
    "optim.adam.beta2": "Adam beta2. Typical value 0.999.",
    "optim.adam.epsilon": "Adam epsilon. Effective default becomes 1e-6 for `hf_deberta_v2` when unset; raw default is 1e-8.",
    "optim.scheduler.type": "Learning-rate scheduler type passed to Transformers scheduler factory.",
    "optim.scheduler.warmup_steps": "Warmup optimizer steps. Must be >= 0. Effective default becomes 10000 for `hf_deberta_v2` when unset.",
    "optim.weight_decay": "AdamW weight decay. Must be >= 0. Default 0.01 is the recommended starting point.",
    "optim.max_grad_norm": "Gradient clipping norm. Must be >= 0; 0 disables clipping.",
    "logging.project_name": "Tracker/project name. Must be non-empty. Used for W&B project and local run metadata.",
    "logging.run_name": "Optional run name. If null, runtime derives one from output/run context.",
    "logging.output_dir": "Directory for resolved/original config snapshots and logs. If null, defaults to `train.checkpoint.output_dir`.",
    "logging.logging_steps": "Metric logging interval in optimizer steps. Must be >= 0; 0 disables periodic scalar logging.",
    "logging.backend": "Local backend when W&B is disabled. Ignored when `logging.wandb.enabled=true` because effective backend is W&B.",
    "logging.wandb.enabled": "Enables W&B tracking. Requires W&B auth/environment outside config. When true, `logging.backend` is not the effective tracker.",
    "logging.wandb.watch": "W&B model watch mode. Only applies when `logging.wandb.enabled=true`; otherwise ignored with warning if changed.",
    "logging.wandb.watch_log_freq": "W&B watch frequency. Must be >= 1 and only applies when W&B is enabled.",
    "logging.debug.metrics": "Logs extra debug metrics. Useful for diagnosis; can add overhead and noise in benchmark runs.",
}


def _type_name(tp: Any) -> str:
    """Render one dataclass field type in config-reference form."""

    target_t, allows_none = unwrap_optional_type(tp)
    origin = get_origin(tp)
    if origin in {Union, types.UnionType}:
        parts = [_type_name(arg) for arg in get_args(tp)]
        return " | ".join(dict.fromkeys(parts))
    if allows_none:
        return f"{_type_name(target_t)} | None"
    if target_t is type(None):
        return "None"
    if target_t is str:
        return "str"
    if target_t is int:
        return "int"
    if target_t is float:
        return "float"
    if target_t is bool:
        return "bool"
    if dataclasses.is_dataclass(target_t):
        return target_t.__name__
    return str(target_t).replace("<class '", "").replace("'>", "")


def _yaml_scalar(value: Any) -> str:
    """Render a scalar as YAML-compatible text."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def _comment_lines(text: str, *, indent: int) -> list[str]:
    """Wrap simple semicolon-separated comments at YAML indentation."""

    prefix = " " * indent + "# "
    lines: list[str] = []
    for part in text.split("\n"):
        part = part.strip()
        if not part:
            lines.append(" " * indent + "#")
        else:
            lines.append(prefix + part)
    return lines


def _leaf_entries() -> list[tuple[str, Any, Any]]:
    """Return generated leaf path, field type, and literal default tuples."""

    defaults = flatten_mapping(asdict_without_private(Config()))
    entries = [
        (path, field_type, defaults[path]) for path, field_type in iter_leaf_paths_for_dataclass(Config)
    ]
    paths = {path for path, _field_type, _default in entries}
    documented = set(FIELD_DOCS)
    missing = sorted(paths - documented)
    extra = sorted(documented - paths)
    if missing or extra:
        details = []
        if missing:
            details.append("missing FIELD_DOCS: " + ", ".join(missing))
        if extra:
            details.append("unknown FIELD_DOCS: " + ", ".join(extra))
        raise RuntimeError("; ".join(details))
    return entries


def render_config_reference() -> str:
    """Render the complete Markdown config reference."""

    entries = _leaf_entries()
    lines = [
        "# Config Reference",
        "",
        "`deberta train` accepts nested YAML/JSON config files and dotted CLI overrides.",
        "",
        "Only these top-level config sections are accepted: `model`, `data`, `train`, `optim`, and `logging`.",
        "`variables` may be used for interpolation before validation, but it is removed before dataclass",
        "construction and is not a runtime config section. Unknown keys fail validation.",
        "",
        "Dotted CLI overrides use the exact same leaf names shown below, for example",
        "`--train.max_steps 2000 --optim.scheduler.warmup_steps 200`. CLI overrides take precedence over",
        "config-file values. `deberta train --preset deberta-v3-base` is available; with a config file it only",
        "overrides model fields, and without a config file it supplies model/data/train/optim/logging defaults.",
        "`--dry-run` validates config and runtime preflight without training or writing checkpoints, but may",
        "still touch tokenizer/dataset network caches.",
        "",
        "Profile/backbone effective defaults run after file/CLI parsing. The YAML below shows literal dataclass",
        "defaults; keys whose effective default can change say so on the key itself.",
        "",
        "## Minimal Valid Skeleton",
        "",
        "```yaml",
        "data:",
        "  source:",
        '    dataset_name: "HuggingFaceFW/fineweb-edu"',
        "```",
        "",
        "Everything else can be omitted and will use documented defaults, but real training runs should set",
        "`train.checkpoint.output_dir`, `logging.output_dir`, and explicit model/data sizing for reproducibility.",
        "",
        "## Full Reference",
        "",
        "```yaml",
    ]

    current_section: str | None = None
    current_parents: list[str] = []
    for path, field_type, default in entries:
        parts = path.split(".")
        section = parts[0]
        if section != current_section:
            if current_section is not None:
                lines.append("")
            lines.extend(_comment_lines(TOP_LEVEL_NOTE[section], indent=0))
            lines.append(f"{section}:")
            current_section = section
            current_parents = [section]

        parents = parts[:-1]
        common = 0
        for old, new in zip(current_parents, parents, strict=False):
            if old != new:
                break
            common += 1
        for depth in range(common, len(parents)):
            parent = parents[depth]
            indent = depth * 2
            lines.append(" " * indent + f"{parent}:")
        current_parents = parents

        indent = (len(parts) - 1) * 2
        key = parts[-1]
        required = REQUIREDNESS.get(path, "no")
        meta = f"Type: {_type_name(field_type)}. Default: {_yaml_scalar(default)}. Required: {required}."
        if path in VALID_VALUES:
            meta += " Valid values: " + " | ".join(f"`{item}`" for item in VALID_VALUES[path]) + "."
        if path in ALIASES and ALIASES[path]:
            alias_text = ", ".join(
                f"`{alias}` -> `{canonical}`" for alias, canonical in sorted(ALIASES[path].items())
            )
            meta += f" Accepted CLI/config aliases: {alias_text}."
        lines.extend(_comment_lines(meta, indent=indent))
        lines.extend(_comment_lines("Guidance: " + FIELD_DOCS[path], indent=indent))
        lines.append(" " * indent + f"{key}: {_yaml_scalar(default)}")
    lines.extend(["```", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help=f"Write {OUTPUT_PATH.relative_to(REPO_ROOT)}.")
    group.add_argument("--check", action="store_true", help="Fail if the checked-in doc is stale.")
    args = parser.parse_args(argv)

    rendered = render_config_reference()
    if args.write:
        OUTPUT_PATH.write_text(rendered, encoding="utf-8")
        print(f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
        return 0
    if args.check:
        current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
        if current != rendered:
            print(
                f"{OUTPUT_PATH.relative_to(REPO_ROOT)} is stale. Run "
                "`conda run --name neobert --no-capture-output python tools/generate_config_reference.py --write`.",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT_PATH.relative_to(REPO_ROOT)} is up to date")
        return 0
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
