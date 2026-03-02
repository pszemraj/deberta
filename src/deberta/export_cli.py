"""Checkpoint consolidation and standalone HF export CLI."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from deberta.checkpoint_utils import load_model_state_with_compile_key_remap, load_state_with_compile_fallback
from deberta.config import (
    ModelConfig,
    load_data_config_snapshot,
    load_model_config_snapshot,
    validate_data_config,
    validate_model_config,
    validate_run_metadata_schema,
)
from deberta.io_utils import load_json_mapping
from deberta.log_utils import setup_process_logging
from deberta.modeling import DebertaV3RTDPretrainer, build_backbone_configs, build_backbones
from deberta.modeling.export_utils import (
    clean_exported_config,
    load_intersection_state_dict,
    merge_embeddings_into_export_backbone,
    split_pretrainer_state_dict,
)

logger = logging.getLogger(__name__)


def _infer_run_dir(checkpoint_dir: Path) -> Path:
    """Infer parent run directory from checkpoint path.

    :param Path checkpoint_dir: Checkpoint directory path.
    :return Path: Parent run directory.
    """
    # checkpoint_dir = .../checkpoint-XXXX
    # run dir is usually parent.
    return checkpoint_dir.parent


def _validate_run_metadata_if_present(run_dir: Path) -> None:
    """Validate run-metadata schema when a metadata file is present.

    :param Path run_dir: Run directory potentially containing run metadata.
    :raises ValueError: If metadata schema is malformed or incompatible.
    """
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return
    raw = load_json_mapping(meta_path)
    validate_run_metadata_schema(raw, source=str(meta_path))


def _resolve_export_output_dir(*, output_dir: str | None, run_dir: Path) -> Path:
    """Resolve and preflight-validate export output directory.

    :param str | None output_dir: Requested output directory override.
    :param Path run_dir: Run directory used for default output location.
    :raises ValueError: If path exists as a non-directory or as a non-empty directory.
    :return Path: Resolved output path.
    """
    resolved = Path(output_dir).expanduser().resolve() if output_dir else (run_dir / "exported_hf")
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"output_dir exists and is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise ValueError(
                f"output_dir already exists and is not empty: {resolved}. "
                "Choose a new --output-dir or clear the directory."
            )
    return resolved


@dataclass
class ExportConfig:
    """Arguments for the consolidation/export tool."""

    checkpoint_dir: str
    output_dir: str | None = None
    run_dir: str | None = None

    export_what: str = "discriminator"  # discriminator|generator|both
    safe_serialization: bool = True

    # Memory knobs for FULL_STATE_DICT gather under FSDP
    offload_to_cpu: bool = True
    rank0: bool = True

    # Override embedding_sharing (normally read from model_config.json)
    embedding_sharing: str | None = None
    allow_partial_export: bool = False


def add_export_arguments(parser: argparse.ArgumentParser) -> None:
    """Register export CLI arguments on an argparse parser.

    :param argparse.ArgumentParser parser: Target parser.
    """
    parser.add_argument(
        "checkpoint_dir",
        help="Path to checkpoint-<step> directory saved by training.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for exported artifacts. Defaults to <run_dir>/exported_hf. "
            "If provided, it must not contain existing files."
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory containing model_config.json/data_config.json. Defaults to checkpoint parent.",
    )
    parser.add_argument(
        "--what",
        "--export-what",
        dest="export_what",
        default="discriminator",
        choices=("discriminator", "generator", "both"),
        help="Which component(s) to export.",
    )
    safe_group = parser.add_mutually_exclusive_group()
    safe_group.add_argument(
        "--safe-serialization",
        dest="safe_serialization",
        action="store_true",
        help="Use safetensors format when saving HF artifacts.",
    )
    safe_group.add_argument(
        "--no-safe-serialization",
        dest="safe_serialization",
        action="store_false",
        help="Disable safetensors format when saving HF artifacts.",
    )
    parser.set_defaults(safe_serialization=True)

    offload_group = parser.add_mutually_exclusive_group()
    offload_group.add_argument(
        "--offload-to-cpu",
        dest="offload_to_cpu",
        action="store_true",
        help="Offload consolidated full state dict to CPU under FSDP export.",
    )
    offload_group.add_argument(
        "--no-offload-to-cpu",
        dest="offload_to_cpu",
        action="store_false",
        help="Keep consolidated full state dict on accelerator memory under FSDP export.",
    )
    parser.set_defaults(offload_to_cpu=True)

    rank0_group = parser.add_mutually_exclusive_group()
    rank0_group.add_argument(
        "--rank0-only",
        dest="rank0",
        action="store_true",
        help="Gather full state dict on rank 0 only under FSDP export.",
    )
    rank0_group.add_argument(
        "--no-rank0-only",
        dest="rank0",
        action="store_false",
        help="Gather full state dict on all ranks under FSDP export.",
    )
    parser.set_defaults(rank0=True)
    parser.add_argument(
        "--embedding-sharing",
        default=None,
        choices=("none", "es", "gdes"),
        help="Override embedding sharing mode. Defaults to training config value.",
    )
    parser.add_argument(
        "--allow-partial-export",
        action="store_true",
        help=(
            "Allow partial backbone state loads when exporting. "
            "By default export fails on any missing/unexpected backbone keys."
        ),
    )


def namespace_to_export_config(ns: argparse.Namespace) -> ExportConfig:
    """Convert parsed argparse namespace into ExportConfig.

    :param argparse.Namespace ns: Parsed arguments.
    :return ExportConfig: Typed export config.
    """
    return ExportConfig(
        checkpoint_dir=ns.checkpoint_dir,
        output_dir=ns.output_dir,
        run_dir=ns.run_dir,
        export_what=ns.export_what,
        safe_serialization=bool(ns.safe_serialization),
        offload_to_cpu=bool(ns.offload_to_cpu),
        rank0=bool(ns.rank0),
        embedding_sharing=ns.embedding_sharing,
        allow_partial_export=bool(getattr(ns, "allow_partial_export", False)),
    )


def _build_export_backbone(
    model_cfg: ModelConfig, disc_config: Any, gen_config: Any, export_what: str
) -> tuple[Any | None, Any | None]:
    """Build export backbones for requested component(s).

    :param ModelConfig model_cfg: Model config.
    :param Any disc_config: Discriminator backbone config.
    :param Any gen_config: Generator backbone config.
    :param str export_what: Export target (discriminator|generator|both).
    :return tuple[Any | None, Any | None]: Discriminator and generator export models.
    """
    bt = (model_cfg.backbone_type or "rope").lower()
    export_what = export_what.lower()

    if bt == "hf_deberta_v2":
        from transformers import AutoModel

        if export_what in {"discriminator", "both"}:
            disc = AutoModel.from_config(disc_config)
        else:
            disc = None
        if export_what in {"generator", "both"}:
            gen = AutoModel.from_config(gen_config)
        else:
            gen = None
        return disc, gen

    # RoPE backbone
    from deberta.modeling.rope_encoder import DebertaRoPEModel

    disc = DebertaRoPEModel(disc_config) if export_what in {"discriminator", "both"} else None
    gen = DebertaRoPEModel(gen_config) if export_what in {"generator", "both"} else None
    return disc, gen


def _prepare_discriminator_state_for_strict_load(
    *,
    export_disc: Any,
    disc_sd: dict[str, torch.Tensor],
    embedding_sharing: str,
    strict_export_load: bool,
) -> dict[str, torch.Tensor]:
    """Normalize discriminator checkpoint keys for strict export loading.

    In ``embedding_sharing='gdes'``, discriminator checkpoints store embedding
    tensors as ``embeddings.<attr>.base_weight`` + ``embeddings.<attr>.bias``.
    Export backbones expect ``embeddings.<attr>.weight``; bias tensors are
    merged afterward via ``merge_embeddings_into_export_backbone``.

    :param Any export_disc: Export discriminator backbone module.
    :param dict[str, torch.Tensor] disc_sd: Raw discriminator checkpoint state dict.
    :param str embedding_sharing: Effective embedding sharing mode.
    :param bool strict_export_load: Whether strict export load is enabled.
    :return dict[str, torch.Tensor]: State dict prepared for strict backbone load.
    """
    if not strict_export_load or str(embedding_sharing).lower() != "gdes":
        return dict(disc_sd)

    model_keys = set(export_disc.state_dict().keys())
    prepared = dict(disc_sd)

    for key, value in list(disc_sd.items()):
        if not key.startswith("embeddings."):
            continue

        if key.endswith(".base_weight"):
            prefix = key[: -len(".base_weight")]
            weight_key = f"{prefix}.weight"
            # Export backbones expect .weight; checkpoint stores .base_weight.
            if weight_key in model_keys and weight_key not in prepared:
                prepared[weight_key] = value
            prepared.pop(key, None)
            continue

        if key.endswith(".bias"):
            prefix = key[: -len(".bias")]
            weight_key = f"{prefix}.weight"
            # Bias tensors are merge-only for GDES and should not participate in strict backbone load.
            if weight_key in model_keys and key not in model_keys:
                prepared.pop(key, None)

    return prepared


def run_export(cfg: ExportConfig) -> None:
    """Run checkpoint export flow.

    :param ExportConfig cfg: Export configuration.
    """
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required.") from e

    from accelerate import Accelerator
    from accelerate.utils import DistributedType

    accelerator = Accelerator()
    setup_process_logging(accelerator.is_main_process)

    checkpoint_dir = Path(cfg.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")

    run_dir = Path(cfg.run_dir).expanduser().resolve() if cfg.run_dir else _infer_run_dir(checkpoint_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    out_dir = _resolve_export_output_dir(output_dir=cfg.output_dir, run_dir=run_dir)

    model_cfg_path = run_dir / "model_config.json"
    data_cfg_path = run_dir / "data_config.json"

    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Expected {model_cfg_path} (produced during training)")
    if not data_cfg_path.exists():
        raise FileNotFoundError(f"Expected {data_cfg_path} (produced during training)")
    _validate_run_metadata_if_present(run_dir)

    # Pre-stable policy: export does not coerce legacy snapshot keys.
    # Stored configs must match current dataclass schemas.
    model_cfg = load_model_config_snapshot(load_json_mapping(model_cfg_path), source=str(model_cfg_path))
    data_cfg = load_data_config_snapshot(load_json_mapping(data_cfg_path), source=str(data_cfg_path))
    validate_model_config(model_cfg)
    validate_data_config(data_cfg)

    embedding_sharing = (cfg.embedding_sharing or model_cfg.embedding_sharing or "none").lower()
    strict_export_load = not bool(cfg.allow_partial_export)

    # Tokenizer (needed for configs, and we also export it)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name_or_path, use_fast=True)

    # Rebuild configs (must match training!)
    disc_config, gen_config = build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=int(data_cfg.max_seq_length),
    )

    # Build backbones + pretrainer container so accelerate.load_state can restore the exact structure.
    disc_backbone, gen_backbone = build_backbones(
        model_cfg=model_cfg,
        disc_config=disc_config,
        gen_config=gen_config,
        load_pretrained_weights=False,
    )
    if model_cfg.gradient_checkpointing:
        if hasattr(disc_backbone, "gradient_checkpointing_enable"):
            disc_backbone.gradient_checkpointing_enable()
        if hasattr(gen_backbone, "gradient_checkpointing_enable"):
            gen_backbone.gradient_checkpointing_enable()

    model = DebertaV3RTDPretrainer(
        discriminator_backbone=disc_backbone,
        generator_backbone=gen_backbone,
        disc_config=disc_config,
        gen_config=gen_config,
        embedding_sharing=embedding_sharing,
    )

    model = accelerator.prepare(model)

    # Load accelerate checkpoint (FSDP2 SHARDED_STATE_DICT is handled here by accelerate/torch.distributed.checkpoint).
    load_state_with_compile_fallback(
        accelerator=accelerator,
        model=model,
        checkpoint_dir=checkpoint_dir,
        context="export",
        remap_loader=load_model_state_with_compile_key_remap,
    )
    accelerator.wait_for_everyone()

    # Consolidate FULL_STATE_DICT when FSDP is enabled.
    full_sd: dict[str, torch.Tensor]
    if accelerator.distributed_type == DistributedType.FSDP:
        if bool(getattr(accelerator, "is_fsdp2", False)):
            try:
                from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
            except Exception as e:
                if not hasattr(accelerator, "get_state_dict"):
                    raise RuntimeError(
                        "accelerator.get_state_dict() is required for FSDP2 export when "
                        "torch.distributed.checkpoint state-dict APIs are unavailable."
                    ) from e
                full_sd = accelerator.get_state_dict(model)
            else:
                # Map CLI knobs to FSDP2 state-dict options.
                # - rank0=True  -> rank0 materializes full state and broadcasts tensor payloads.
                # - rank0=False -> all ranks materialize full state.
                opts = StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=bool(cfg.offload_to_cpu),
                    broadcast_from_rank0=bool(cfg.rank0),
                )
                full_sd = get_model_state_dict(model, options=opts)
        else:
            try:
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                is_fsdp_instance = isinstance(model, FSDP)
            except Exception:
                is_fsdp_instance = False

            if is_fsdp_instance:
                cfg_full = FullStateDictConfig(
                    offload_to_cpu=bool(cfg.offload_to_cpu),
                    rank0_only=bool(cfg.rank0),
                )
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg_full):
                    full_sd = model.state_dict()
            else:
                if not hasattr(accelerator, "get_state_dict"):
                    raise RuntimeError(
                        "accelerator.get_state_dict() is required for FSDP export on non-torch FSDP engines."
                    )
                full_sd = accelerator.get_state_dict(model)
    else:
        # Non-FSDP: unwrap DDP etc.
        full_sd = accelerator.unwrap_model(model).state_dict()

    if not accelerator.is_main_process:
        return

    disc_sd, gen_sd = split_pretrainer_state_dict(full_sd)
    if not disc_sd and not gen_sd:
        raise RuntimeError(
            "Failed to split discriminator/generator state dicts. "
            "This likely means the checkpoint structure does not match this code version."
        )

    # Build export models (backbones only)
    export_what = (cfg.export_what or "discriminator").lower()
    if export_what not in {"discriminator", "generator", "both"}:
        raise ValueError("export_what must be discriminator|generator|both")

    export_disc, export_gen = _build_export_backbone(model_cfg, disc_config, gen_config, export_what)

    stage_dir = out_dir.parent / f".{out_dir.name}.tmp-{uuid.uuid4().hex}"
    stage_dir.mkdir(parents=True, exist_ok=False)

    meta: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "run_dir": str(run_dir),
        "embedding_sharing": embedding_sharing,
        "backbone_type": model_cfg.backbone_type,
    }

    try:
        # Always export tokenizer at root for convenience
        tokenizer.save_pretrained(str(stage_dir))

        # Discriminator
        if export_disc is not None:
            disc_sd_for_load = _prepare_discriminator_state_for_strict_load(
                export_disc=export_disc,
                disc_sd=disc_sd,
                embedding_sharing=embedding_sharing,
                strict_export_load=bool(strict_export_load),
            )
            incompat_disc = load_intersection_state_dict(
                export_disc,
                disc_sd_for_load,
                strict=strict_export_load,
                context="export.discriminator",
            )
            if not strict_export_load:
                missing = list(getattr(incompat_disc, "missing_keys", []))
                unexpected = list(getattr(incompat_disc, "unexpected_keys", []))
                if missing or unexpected:
                    logger.warning(
                        "Discriminator export loaded partial state due --allow-partial-export: "
                        "missing=%d unexpected=%d",
                        len(missing),
                        len(unexpected),
                    )
            if embedding_sharing in {"es", "gdes"}:
                merge_embeddings_into_export_backbone(
                    export_model=export_disc,
                    disc_sd=disc_sd,
                    gen_sd=gen_sd,
                    mode=embedding_sharing,
                    fp32_accumulate=True,
                )

            export_disc.save_pretrained(
                str(stage_dir / "discriminator"), safe_serialization=bool(cfg.safe_serialization)
            )
            clean_exported_config(stage_dir / "discriminator" / "config.json", strict=True)
            meta["exported_discriminator"] = True

        # Generator
        if export_gen is not None:
            incompat_gen = load_intersection_state_dict(
                export_gen,
                gen_sd,
                strict=strict_export_load,
                context="export.generator",
            )
            if not strict_export_load:
                missing = list(getattr(incompat_gen, "missing_keys", []))
                unexpected = list(getattr(incompat_gen, "unexpected_keys", []))
                if missing or unexpected:
                    logger.warning(
                        "Generator export loaded partial state due --allow-partial-export: "
                        "missing=%d unexpected=%d",
                        len(missing),
                        len(unexpected),
                    )
            export_gen.save_pretrained(
                str(stage_dir / "generator"), safe_serialization=bool(cfg.safe_serialization)
            )
            clean_exported_config(stage_dir / "generator" / "config.json", strict=True)
            meta["exported_generator"] = True

        with (stage_dir / "export_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        if out_dir.exists():
            # At this point we already enforced "empty only".
            out_dir.rmdir()
        stage_dir.replace(out_dir)
        logger.info(f"Export complete: {out_dir}")
    except Exception:
        # Cleanup staged partial output so failed exports are re-runnable.
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> None:
    """Run checkpoint export CLI.

    :param list[str] | None argv: Optional CLI argv (excluding program name).
    """
    parser = argparse.ArgumentParser(
        prog="deberta export",
        description="Consolidate a training checkpoint and export standalone HF artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_export_arguments(parser)
    args = parser.parse_args(argv)
    cfg = namespace_to_export_config(args)
    run_export(cfg)


if __name__ == "__main__":
    main()
