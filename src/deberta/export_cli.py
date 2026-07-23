"""Checkpoint consolidation and standalone HF export CLI."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from deberta.config import (
    ModelConfig,
    TrainConfig,
    load_data_config_snapshot,
    load_model_config_snapshot,
    load_train_config_snapshot,
    validate_data_config,
    validate_model_config,
)
from deberta.modeling import DebertaV3RTDPretrainer, build_backbone_configs, build_backbones
from deberta.modeling.export_utils import (
    clean_exported_config,
    merge_embeddings_into_export_backbone,
    split_pretrainer_state_dict,
    write_export_readme_and_license,
)
from deberta.run_layout import (
    DATA_CONFIG_FILENAME,
    MODEL_CONFIG_FILENAME,
    TRAIN_CONFIG_FILENAME,
    infer_run_dir_from_checkpoint,
    validate_run_metadata_file,
)
from deberta.utils.checkpoint import load_state_with_compile_fallback
from deberta.utils.io import load_json_mapping
from deberta.utils.log import setup_process_logging
from deberta.utils.paths import validate_existing_output_dir

logger = logging.getLogger(__name__)


def _normalize_export_target(value: str) -> str:
    """Normalize and validate export target selection.

    :param str value: Raw export target.
    :raises ValueError: If value is not one of discriminator|generator|both.
    :return str: Canonical lower-case export target.
    """
    normalized = str(value or "").strip().lower()
    if normalized not in {"discriminator", "generator", "both"}:
        raise ValueError("export_what must be discriminator|generator|both")
    return normalized


def _resolve_export_output_dir(*, output_dir: str | None, run_dir: Path) -> Path:
    """Resolve and preflight-validate export output directory.

    :param str | None output_dir: Requested output directory override.
    :param Path run_dir: Run directory used for default output location.
    :raises ValueError: If path exists as a non-directory or as a non-empty directory.
    :return Path: Resolved output path.
    """
    resolved = Path(output_dir).expanduser().resolve() if output_dir else (run_dir / "exported_hf")
    validate_existing_output_dir(
        output_dir=resolved,
        allow_nonempty=False,
        nonempty_error=(
            f"output_dir already exists and is not empty: {resolved}. "
            "Choose a new --output-dir or clear the directory."
        ),
        nondir_error=f"output_dir exists and is not a directory: {resolved}",
    )
    return resolved


def _load_optional_train_config(run_dir: Path) -> TrainConfig | None:
    """Best-effort load of ``train_config.json`` for export metadata rendering.

    :param Path run_dir: Run directory potentially containing ``train_config.json``.
    :return TrainConfig | None: Parsed train config when present/valid, otherwise ``None``.
    """
    train_cfg_path = run_dir / TRAIN_CONFIG_FILENAME
    if not train_cfg_path.exists():
        return None

    try:
        raw = load_json_mapping(train_cfg_path)
        return load_train_config_snapshot(raw, source=str(train_cfg_path))
    except Exception as exc:
        logger.warning("Failed to parse optional train config at %s: %s", train_cfg_path, exc)
        return None


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

    allow_partial_export: bool = False


def add_export_arguments(parser: argparse.ArgumentParser) -> None:
    """Register export CLI arguments on an argparse parser.

    :param argparse.ArgumentParser parser: Target parser.
    """
    parser.add_argument(
        "checkpoint_dir",
        help="Required path to an existing checkpoint-<step> directory saved by training.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for exported artifacts. Defaults to <run_dir>/exported_hf. "
            "The resolved directory must be absent or empty."
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Run directory containing required model_config.json and data_config.json snapshots; "
            "run_metadata.json is validated when present. Defaults to the checkpoint parent."
        ),
    )
    parser.add_argument(
        "--what",
        dest="export_what",
        default="discriminator",
        choices=("discriminator", "generator", "both"),
        help=(
            "Component to export. A single component uses a flat output directory; 'both' writes "
            "discriminator/ and generator/ subdirectories."
        ),
    )
    parser.add_argument(
        "--no-safe-serialization",
        dest="safe_serialization",
        action="store_false",
        help="Save model weights with PyTorch serialization instead of the default, recommended safetensors.",
    )

    parser.add_argument(
        "--no-offload-to-cpu",
        dest="offload_to_cpu",
        action="store_false",
        help=(
            "Keep the consolidated full state dict on accelerator memory instead of offloading it to CPU. "
            "The full state must fit accelerator memory; ignored for non-FSDP exports."
        ),
    )

    parser.add_argument(
        "--no-rank0-only",
        dest="rank0",
        action="store_false",
        help=(
            "Gather the full state dict on every rank instead of the default rank-0-only gather. "
            "Ignored for non-FSDP exports."
        ),
    )
    parser.add_argument(
        "--allow-partial-export",
        action="store_true",
        help=(
            "Recovery/debugging only: permit missing or unexpected backbone keys, which can leave "
            "parameters initialized rather than restored. Strict loading is the default."
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
    bt = (model_cfg.backbone_type or "hf_deberta_v2").lower()
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


# Source-state keys that exist only in training-time modules. Each predicate
# returns True when the export architecture omits the key by design, so the
# entry is training-only state rather than a genuine load mismatch. Add new
# native-vs-export exceptions here instead of inlining checks at load sites.
_TRAINING_ONLY_EXPORT_STATE_KEYS: dict[str, Callable[[Any], bool]] = {
    # Native RTD keeps this weight for Enhanced Mask Decoding; the standalone
    # HF encoder with position_biased_input=False neither defines nor consumes it.
    "embeddings.position_embeddings.weight": lambda export_model: (
        not bool(getattr(getattr(export_model, "config", None), "position_biased_input", True))
    ),
}


def _drop_training_only_state_for_strict_load(
    *,
    export_model: Any,
    state_dict: dict[str, torch.Tensor],
    strict_export_load: bool,
) -> dict[str, torch.Tensor]:
    """Drop known training-only source keys the export model does not define.

    :param Any export_model: Target export model instance.
    :param dict[str, torch.Tensor] state_dict: Component source state dict.
    :param bool strict_export_load: Strict source-state loading toggle.
    :return dict[str, torch.Tensor]: State dict safe for strict export loading.
    """
    prepared = dict(state_dict)
    if not strict_export_load:
        return prepared

    candidates = [
        key
        for key, is_training_only in _TRAINING_ONLY_EXPORT_STATE_KEYS.items()
        if key in prepared and is_training_only(export_model)
    ]
    if not candidates:
        return prepared

    model_keys = set(export_model.state_dict().keys())
    for key in candidates:
        if key not in model_keys:
            prepared.pop(key)
    return prepared


def _prepare_discriminator_state_for_strict_load(
    *,
    export_disc: Any,
    disc_sd: dict[str, torch.Tensor],
    embedding_sharing: str,
    strict_export_load: bool,
) -> dict[str, torch.Tensor]:
    """Map GDES discriminator embedding keys into strict export-backbone shape.

    :param Any export_disc: Export discriminator module.
    :param dict[str, torch.Tensor] disc_sd: Discriminator state dict.
    :param str embedding_sharing: Effective embedding-sharing mode.
    :param bool strict_export_load: Whether strict export-load checks are enabled.
    :return dict[str, torch.Tensor]: Prepared state dict for export-model loading.
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
            # Bias tensors are merge-only for GDES and should not participate in strict backbone load.
            if key not in model_keys:
                prepared.pop(key, None)

    return prepared


def _verify_staged_encoder_output_parity(
    *,
    component: str,
    export_model: Any,
    component_dir: Path,
) -> None:
    """Verify staged serialization preserves one materialized encoder's output.

    :param str component: Export component name used in failure diagnostics.
    :param Any export_model: In-memory materialized encoder.
    :param Path component_dir: Staged component directory to reload.
    """
    config = export_model.config
    seq_len = min(8, int(config.max_position_embeddings))
    vocab_size = int(config.vocab_size)
    input_ids = (torch.arange(seq_len, dtype=torch.long) + 1).remainder(vocab_size).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    export_model.to(device="cpu").eval()
    with torch.inference_mode():
        expected = export_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state

    staged_model = type(export_model).from_pretrained(str(component_dir)).to(device="cpu").eval()
    with torch.inference_mode():
        actual = staged_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state

    try:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-5, atol=2e-5)
    except AssertionError as exc:
        raise RuntimeError(
            f"Staged {component} encoder output does not match the materialized export model."
        ) from exc


def _export_component(
    *,
    component: str,
    export_model: Any | None,
    stage_dir: Path,
    export_what: str,
    safe_serialization: bool,
    strict_export_load: bool,
    model_cfg: ModelConfig,
    data_cfg: Any,
    train_cfg: Any,
    embedding_sharing: str,
    state_dict: dict[str, torch.Tensor],
    disc_sd: dict[str, torch.Tensor],
    gen_sd: dict[str, torch.Tensor],
) -> bool:
    """Load, merge, and save one exported component.

    :param str component: Component key (``discriminator`` or ``generator``).
    :param Any | None export_model: Target export model instance.
    :param Path stage_dir: Export staging directory.
    :param str export_what: Effective target selection.
    :param bool safe_serialization: Safetensors toggle for ``save_pretrained``.
    :param bool strict_export_load: Strict source-state loading toggle.
    :param ModelConfig model_cfg: Model config used to render README metadata.
    :param Any data_cfg: Data config used to render README metadata.
    :param Any train_cfg: Train config used to render README metadata.
    :param str embedding_sharing: Effective embedding-sharing mode.
    :param dict[str, torch.Tensor] state_dict: Component source state dict for loading.
    :param dict[str, torch.Tensor] disc_sd: Full discriminator source state dict.
    :param dict[str, torch.Tensor] gen_sd: Full generator source state dict.
    :return bool: True when export completed for this component.
    """
    if export_model is None:
        return False

    component_key = str(component).strip().lower()
    if component_key not in {"discriminator", "generator"}:
        raise ValueError(f"Unsupported export component: {component!r}")

    state_for_load = _drop_training_only_state_for_strict_load(
        export_model=export_model,
        state_dict=state_dict,
        strict_export_load=bool(strict_export_load),
    )

    if component_key == "discriminator":
        state_for_load = _prepare_discriminator_state_for_strict_load(
            export_disc=export_model,
            disc_sd=state_for_load,
            embedding_sharing=embedding_sharing,
            strict_export_load=bool(strict_export_load),
        )

    try:
        incompatible = export_model.load_state_dict(
            state_for_load,
            strict=bool(strict_export_load),
        )
    except RuntimeError as exc:
        raise RuntimeError(f"export.{component_key}: {exc}") from exc
    if not bool(strict_export_load):
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing or unexpected:
            logger.warning(
                "%s export loaded partial state due --allow-partial-export: missing=%d unexpected=%d",
                component_key.capitalize(),
                len(missing),
                len(unexpected),
            )

    if component_key == "discriminator" and embedding_sharing in {"es", "gdes"}:
        merge_embeddings_into_export_backbone(
            export_model=export_model,
            disc_sd=disc_sd,
            gen_sd=gen_sd,
            mode=embedding_sharing,
        )

    out_dir = stage_dir if export_what == component_key else (stage_dir / component_key)
    export_model.save_pretrained(str(out_dir), safe_serialization=bool(safe_serialization))
    clean_exported_config(out_dir / "config.json")
    write_export_readme_and_license(
        out_dir,
        model_cfg=model_cfg,
        export_config=getattr(export_model, "config", None),
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        embedding_sharing=embedding_sharing,
    )
    if bool(strict_export_load):
        _verify_staged_encoder_output_parity(
            component=component_key,
            export_model=export_model,
            component_dir=out_dir,
        )
    return True


def run_export(cfg: ExportConfig) -> None:
    """Run checkpoint export flow.

    :param ExportConfig cfg: Export configuration.
    """
    export_what = _normalize_export_target(cfg.export_what)

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

    run_dir = (
        Path(cfg.run_dir).expanduser().resolve()
        if cfg.run_dir
        else infer_run_dir_from_checkpoint(checkpoint_dir)
    )
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    out_dir = _resolve_export_output_dir(output_dir=cfg.output_dir, run_dir=run_dir)

    model_cfg_path = run_dir / MODEL_CONFIG_FILENAME
    data_cfg_path = run_dir / DATA_CONFIG_FILENAME

    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Expected {model_cfg_path} (produced during training)")
    if not data_cfg_path.exists():
        raise FileNotFoundError(f"Expected {data_cfg_path} (produced during training)")
    validate_run_metadata_file(run_dir)
    train_cfg = _load_optional_train_config(run_dir)

    # Pre-stable policy: export does not coerce legacy snapshot keys.
    # Stored configs must match current dataclass schemas.
    model_cfg = load_model_config_snapshot(load_json_mapping(model_cfg_path), source=str(model_cfg_path))
    data_cfg = load_data_config_snapshot(load_json_mapping(data_cfg_path), source=str(data_cfg_path))
    validate_model_config(model_cfg)
    validate_data_config(data_cfg)

    embedding_sharing = (model_cfg.embedding_sharing or "none").lower()
    strict_export_load = not bool(cfg.allow_partial_export)

    # Tokenizer (needed for configs, and we also export it)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer.name_or_path, use_fast=True)

    # Flash attention adds no parameters, so export rebuilds the checkpoint
    # container with eager attention. This keeps consolidation independent of
    # the optional FlashDeBERTa runtime while preserving the training shapes.
    export_model_cfg = replace(
        model_cfg,
        hf=replace(model_cfg.hf, attention_impl="eager"),
    )

    # Rebuild configs (must match training shapes and parameter names).
    disc_config, gen_config = build_backbone_configs(
        model_cfg=export_model_cfg,
        tokenizer=tokenizer,
        max_position_embeddings=int(data_cfg.packing.max_seq_length),
    )

    # Build backbones + pretrainer container so accelerate.load_state can restore the exact structure.
    disc_backbone, gen_backbone = build_backbones(
        model_cfg=export_model_cfg,
        disc_config=disc_config,
        gen_config=gen_config,
        load_pretrained_weights=False,
    )
    model = DebertaV3RTDPretrainer(
        discriminator_backbone=disc_backbone,
        generator_backbone=gen_backbone,
        disc_config=disc_config,
        gen_config=gen_config,
        embedding_sharing=embedding_sharing,
        additional_forbidden_token_ids=getattr(tokenizer, "all_special_ids", []),
    )

    model = accelerator.prepare(model)

    # Load accelerate checkpoint (FSDP2 SHARDED_STATE_DICT is handled here by accelerate/torch.distributed.checkpoint).
    load_state_with_compile_fallback(
        accelerator=accelerator,
        model=model,
        checkpoint_dir=checkpoint_dir,
        context="export",
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
        if (not bool(cfg.offload_to_cpu)) or (not bool(cfg.rank0)):
            logger.warning(
                "--no-offload-to-cpu/--no-rank0-only only apply to FSDP export; current distributed_type=%s, "
                "so those options are ignored.",
                accelerator.distributed_type,
            )
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
    export_disc, export_gen = _build_export_backbone(model_cfg, disc_config, gen_config, export_what)

    stage_dir = out_dir.parent / f".{out_dir.name}.tmp-{uuid.uuid4().hex}"
    stage_dir.mkdir(parents=True, exist_ok=False)

    embedding_materialization: dict[str, str] = {}
    if export_what in {"discriminator", "both"}:
        embedding_materialization["discriminator"] = {
            "none": "discriminator_checkpoint",
            "es": "generator_checkpoint_shared",
            "gdes": "generator_checkpoint_plus_discriminator_bias",
        }[embedding_sharing]
    if export_what in {"generator", "both"}:
        embedding_materialization["generator"] = "generator_checkpoint"

    meta: dict[str, Any] = {
        # Directory names only: exported directories ship to other machines
        # and the Hub, so provenance must not leak local absolute paths.
        "checkpoint_name": checkpoint_dir.name,
        "run_name": run_dir.name,
        "embedding_sharing": embedding_sharing,
        "backbone_type": model_cfg.backbone_type,
        "export_target": export_what,
        "artifact_type": (
            "rtd_pretrained_encoder_bundle" if export_what == "both" else "rtd_pretrained_encoder"
        ),
        "strict_state_load": bool(strict_export_load),
        "includes_rtd_head": False,
        "embedding_materialization": embedding_materialization,
    }

    try:
        # Always export tokenizer at root for convenience
        tokenizer.save_pretrained(str(stage_dir))

        for component, export_model, component_state in (
            ("discriminator", export_disc, disc_sd),
            ("generator", export_gen, gen_sd),
        ):
            if _export_component(
                component=component,
                export_model=export_model,
                stage_dir=stage_dir,
                export_what=export_what,
                safe_serialization=bool(cfg.safe_serialization),
                strict_export_load=bool(strict_export_load),
                model_cfg=model_cfg,
                data_cfg=data_cfg,
                train_cfg=train_cfg,
                embedding_sharing=embedding_sharing,
                state_dict=component_state,
                disc_sd=disc_sd,
                gen_sd=gen_sd,
            ):
                meta[f"exported_{component}"] = True

        with (stage_dir / "export_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        # Strict loads and staged encoder parity complete before this atomic publication step.
        if out_dir.exists():
            # At this point we already enforced "empty only".
            out_dir.rmdir()
        stage_dir.replace(out_dir)
        logger.info(f"Export complete: {out_dir}")
    except Exception:
        # Cleanup staged partial output so failed exports are re-runnable.
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
