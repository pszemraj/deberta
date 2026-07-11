"""Concise config builders for behavior-focused tests."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, TypeVar

from deberta.config import DataConfig, LoggingConfig, ModelConfig, OptimConfig, TrainConfig
from deberta.utils.mapping import flatten_mapping

T = TypeVar("T")

_MODEL_PATHS = {
    "tokenizer_name_or_path": "tokenizer.name_or_path",
    "tokenizer_allow_vocab_resize": "tokenizer.allow_vocab_resize",
    "tokenizer_vocab_target": "tokenizer.vocab_target",
    "tokenizer_vocab_multiple": "tokenizer.vocab_multiple",
    "hf_attention_kernel": "hf.attention_kernel",
    "hf_attention_impl": "hf.attention_impl",
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


_DATA_PATHS = {
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


_TRAIN_PATHS = {
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

_OPTIM_PATHS = {
    "learning_rate": "lr.base",
    "generator_learning_rate": "lr.generator",
    "discriminator_learning_rate": "lr.discriminator",
    "adam_beta1": "adam.beta1",
    "adam_beta2": "adam.beta2",
    "adam_epsilon": "adam.epsilon",
    "lr_scheduler_type": "scheduler.type",
    "warmup_steps": "scheduler.warmup_steps",
}
_LOGGING_PATHS = {
    "wandb_watch": "wandb.watch",
    "wandb_watch_log_freq": "wandb.watch_log_freq",
    "debug_metrics": "debug.metrics",
}


def _replace_path(config: T, path: str, value: Any) -> T:
    head, *tail = path.split(".", 1)
    if not tail:
        return replace(config, **{head: value})
    child = getattr(config, head)
    return replace(config, **{head: _replace_path(child, tail[0], value)})


def _build(config: T, overrides: dict[str, Any], moved: dict[str, str] | None = None) -> T:
    valid = {item.name for item in fields(config)}
    paths: dict[str, Any] = {}
    for key, value in overrides.items():
        path = (moved or {}).get(key, key)
        if path in valid and isinstance(value, dict) and is_dataclass(getattr(config, path)):
            paths.update({f"{path}.{child}": item for child, item in flatten_mapping(value).items()})
        else:
            paths[path] = value
    for path, value in paths.items():
        config = _replace_path(config, path, value)
    return config


def make_model_config(**overrides: Any) -> ModelConfig:
    return _build(ModelConfig(), overrides, _MODEL_PATHS)


def make_data_config(**overrides: Any) -> DataConfig:
    return _build(DataConfig(), overrides, _DATA_PATHS)


def make_train_config(**overrides: Any) -> TrainConfig:
    return _build(TrainConfig(), overrides, _TRAIN_PATHS)


def make_optim_config(**overrides: Any) -> OptimConfig:
    return _build(OptimConfig(), overrides, _OPTIM_PATHS)


def make_logging_config(**overrides: Any) -> LoggingConfig:
    report_to = overrides.pop("report_to", None)
    config = _build(LoggingConfig(), overrides, _LOGGING_PATHS)
    if report_to is None:
        return config
    target = str(report_to).lower()
    if target not in {"none", "wandb"}:
        raise ValueError(f"Unsupported test tracker target: {target}")
    return _replace_path(config, "wandb.enabled", target == "wandb")
