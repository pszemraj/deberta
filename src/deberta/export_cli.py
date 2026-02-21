from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from deberta.config import DataConfig, ModelConfig
from deberta.modeling import DebertaV3RTDPretrainer, build_backbone_configs, build_backbones

logger = logging.getLogger(__name__)


def _setup_logging(is_main: bool) -> None:
    level = logging.INFO if is_main else logging.WARN
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_run_dir(checkpoint_dir: Path) -> Path:
    # checkpoint_dir = .../checkpoint-XXXX
    # run dir is usually parent.
    return checkpoint_dir.parent


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
    rank0_only: bool = True

    # Override embedding_sharing (normally read from model_config.json)
    embedding_sharing: str | None = None


def _split_state_dict(full_sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    disc: dict[str, torch.Tensor] = {}
    gen: dict[str, torch.Tensor] = {}

    for k, v in full_sd.items():
        if k.startswith("discriminator."):
            disc[k[len("discriminator.") :]] = v
        elif k.startswith("generator."):
            gen[k[len("generator.") :]] = v

    return disc, gen


def _merge_embeddings_into_export(
    *,
    export_model: Any,
    disc_sd: dict[str, torch.Tensor],
    gen_sd: dict[str, torch.Tensor],
    mode: str,
) -> None:
    if mode not in {"es", "gdes"}:
        return

    if not hasattr(export_model, "embeddings"):
        return

    def merge_attr(attr: str) -> None:
        if not hasattr(export_model.embeddings, attr):
            return
        gen_w = gen_sd.get(f"embeddings.{attr}.weight")
        if gen_w is None:
            return

        if mode == "es":
            merged = gen_w
        else:
            bias = disc_sd.get(f"embeddings.{attr}.bias")
            if bias is None:
                raise RuntimeError(f"Missing discriminator bias for embeddings.{attr}.bias (gdes)")
            # Merge in fp32 for numerical stability, then cast at the end.
            merged = gen_w.detach().float() + bias.detach().float()

        emb_mod = getattr(export_model.embeddings, attr)
        if hasattr(emb_mod, "weight") and emb_mod.weight is not None:
            emb_mod.weight.data.copy_(merged.to(emb_mod.weight.dtype))

    merge_attr("word_embeddings")
    merge_attr("position_embeddings")
    merge_attr("token_type_embeddings")


def _build_export_backbone(model_cfg: ModelConfig, disc_config: Any, gen_config: Any, export_what: str):
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


def main() -> None:
    try:
        from transformers import AutoTokenizer, HfArgumentParser
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required.") from e

    parser = HfArgumentParser(ExportConfig)
    (cfg,) = parser.parse_args_into_dataclasses()

    from accelerate import Accelerator
    from accelerate.utils import DistributedType

    accelerator = Accelerator()
    _setup_logging(accelerator.is_main_process)

    checkpoint_dir = Path(cfg.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")

    run_dir = Path(cfg.run_dir).expanduser().resolve() if cfg.run_dir else _infer_run_dir(checkpoint_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    model_cfg_path = run_dir / "model_config.json"
    data_cfg_path = run_dir / "data_config.json"

    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Expected {model_cfg_path} (produced during training)")
    if not data_cfg_path.exists():
        raise FileNotFoundError(f"Expected {data_cfg_path} (produced during training)")

    model_cfg = ModelConfig(**_load_json(model_cfg_path))
    data_cfg = DataConfig(**_load_json(data_cfg_path))

    embedding_sharing = (cfg.embedding_sharing or model_cfg.embedding_sharing or "none").lower()

    # Tokenizer (needed for configs, and we also export it)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer_name_or_path, use_fast=True)

    # Rebuild configs (must match training!)
    disc_config, gen_config = build_backbone_configs(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=int(data_cfg.max_seq_length),
    )

    # Build backbones + pretrainer container so accelerate.load_state can restore the exact structure.
    disc_backbone, gen_backbone = build_backbones(model_cfg=model_cfg, disc_config=disc_config, gen_config=gen_config)
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
    accelerator.load_state(str(checkpoint_dir))
    accelerator.wait_for_everyone()

    # Consolidate FULL_STATE_DICT on rank0 if FSDP is enabled.
    full_sd: dict[str, torch.Tensor]
    if accelerator.distributed_type == DistributedType.FSDP:
        try:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except Exception as e:  # pragma: no cover
            raise RuntimeError("torch.distributed.fsdp is required for FSDP export") from e

        if not isinstance(model, FSDP):
            raise RuntimeError(
                "Expected the prepared model to be an FSDP instance. "
                "Make sure you are launching with the same accelerate FSDP config used for training."
            )

        cfg_full = FullStateDictConfig(offload_to_cpu=bool(cfg.offload_to_cpu), rank0_only=bool(cfg.rank0_only))
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg_full):
            full_sd = model.state_dict()
    else:
        # Non-FSDP: unwrap DDP etc.
        full_sd = accelerator.unwrap_model(model).state_dict()

    if not accelerator.is_main_process:
        return

    disc_sd, gen_sd = _split_state_dict(full_sd)
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

    out_dir = Path(cfg.output_dir).expanduser().resolve() if cfg.output_dir else (run_dir / "exported_hf")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always export tokenizer at root for convenience
    tokenizer.save_pretrained(str(out_dir))

    meta: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "run_dir": str(run_dir),
        "embedding_sharing": embedding_sharing,
        "backbone_type": model_cfg.backbone_type,
    }

    # Discriminator
    if export_disc is not None:
        export_keys = set(export_disc.state_dict().keys())
        filtered = {k: v for k, v in disc_sd.items() if k in export_keys}
        export_disc.load_state_dict(filtered, strict=False)
        if embedding_sharing in {"es", "gdes"}:
            _merge_embeddings_into_export(export_model=export_disc, disc_sd=disc_sd, gen_sd=gen_sd, mode=embedding_sharing)

        export_disc.save_pretrained(str(out_dir / "discriminator"), safe_serialization=bool(cfg.safe_serialization))
        meta["exported_discriminator"] = True

    # Generator
    if export_gen is not None:
        export_keys = set(export_gen.state_dict().keys())
        filtered = {k: v for k, v in gen_sd.items() if k in export_keys}
        export_gen.load_state_dict(filtered, strict=False)
        export_gen.save_pretrained(str(out_dir / "generator"), safe_serialization=bool(cfg.safe_serialization))
        meta["exported_generator"] = True

    with (out_dir / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    logger.info(f"Export complete: {out_dir}")
