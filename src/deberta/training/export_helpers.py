"""Export helpers used by the pretraining runtime."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from deberta.checkpoint_utils import canonicalize_state_dict_keys, unwrap_compiled_model
from deberta.io_utils import dump_json
from deberta.modeling.export_utils import (
    clean_exported_config as _clean_exported_config_impl,
)
from deberta.modeling.export_utils import (
    load_intersection_state_dict,
    merge_embeddings_into_export_backbone,
    write_export_readme_and_license,
)

logger = logging.getLogger(__name__)


def _write_export_readme(
    output_dir: Path,
    *,
    model_cfg: Any,
    export_config: Any | None = None,
    data_cfg: Any | None = None,
    train_cfg: Any,
    embedding_sharing: str,
) -> None:
    """Write README/Licensing files to the export directory.

    :param Path output_dir: Export destination directory.
    :param Any model_cfg: Model configuration dataclass.
    :param Any | None export_config: Optional exported backbone config (preferred source for architecture stats).
    :param Any | None data_cfg: Optional data configuration dataclass.
    :param Any train_cfg: Training configuration dataclass.
    :param str embedding_sharing: Embedding sharing mode.
    """
    write_export_readme_and_license(
        output_dir,
        model_cfg=model_cfg,
        export_config=export_config,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        embedding_sharing=embedding_sharing,
    )


def _export_discriminator_hf(
    *,
    accelerator: Any,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    embedding_sharing: str,
    model_cfg: Any = None,
    data_cfg: Any = None,
    train_cfg: Any = None,
) -> None:
    """Best-effort export of a standalone discriminator model.

    :param Any accelerator: Accelerate runtime object.
    :param Any model: Wrapped RTD pretrainer model.
    :param Any tokenizer: Tokenizer to export.
    :param Path output_dir: Export destination directory.
    :param str embedding_sharing: Sharing mode used during training.
    :param Any model_cfg: Optional model config for README generation.
    :param Any data_cfg: Optional data config for README generation.
    :param Any train_cfg: Optional training config for README generation.
    """

    try:
        unwrapped = unwrap_compiled_model(accelerator, model)
        disc_mod = getattr(unwrapped, "discriminator", None)
        gen_mod = getattr(unwrapped, "generator", None)
        if disc_mod is None or gen_mod is None:
            raise RuntimeError(
                "Unwrapped RTD model must expose discriminator and generator modules for export."
            )
        disc_sd_raw = accelerator.get_state_dict(disc_mod)
        gen_sd_raw = accelerator.get_state_dict(gen_mod)

        if not accelerator.is_main_process:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from transformers import AutoModel
        except Exception as e:
            logger.warning(f"Skipping HF export: transformers import failed: {e}")
            return

        try:
            from deberta.modeling.rope_encoder import DebertaRoPEConfig, DebertaRoPEModel
        except Exception:
            DebertaRoPEConfig = None  # type: ignore
            DebertaRoPEModel = None  # type: ignore

        disc_sd = canonicalize_state_dict_keys(dict(disc_sd_raw))
        gen_sd = canonicalize_state_dict_keys(dict(gen_sd_raw))

        if DebertaRoPEConfig is not None and isinstance(
            getattr(unwrapped, "disc_config", None), DebertaRoPEConfig
        ):
            export_disc = DebertaRoPEModel(unwrapped.disc_config)  # type: ignore[arg-type]
        else:
            export_disc = AutoModel.from_config(unwrapped.disc_config)

        missing = load_intersection_state_dict(export_disc, disc_sd)
        if missing.missing_keys:
            logger.info(
                f"HF export: missing keys (often expected with tied embeddings): {missing.missing_keys[:5]}..."
            )

        mode = (embedding_sharing or "none").lower()
        merge_embeddings_into_export_backbone(
            export_model=export_disc,
            disc_sd=disc_sd,
            gen_sd=gen_sd,
            mode=mode,
            fp32_accumulate=True,
        )

        tokenizer.save_pretrained(str(output_dir))
        export_disc.save_pretrained(str(output_dir), safe_serialization=True)
        _clean_exported_config_impl(output_dir / "config.json", strict=False)
        dump_json({"embedding_sharing": embedding_sharing}, output_dir / "export_meta.json")

        if model_cfg is not None and train_cfg is not None:
            _write_export_readme(
                output_dir,
                model_cfg=model_cfg,
                export_config=getattr(export_disc, "config", None),
                data_cfg=data_cfg,
                train_cfg=train_cfg,
                embedding_sharing=embedding_sharing,
            )

        logger.info(f"Exported discriminator to: {output_dir}")

    except Exception as e:
        logger.warning(
            "HF export failed (common under FSDP2 + SHARDED_STATE_DICT). "
            "Use `deberta export` after training for a guaranteed consolidation+export. "
            f"Error: {e}"
        )


def _export_discriminator_hf_subprocess(
    *,
    checkpoint_dir: Path,
    output_dir: Path,
) -> None:
    """Run discriminator export in an isolated subprocess.

    :param Path checkpoint_dir: Source checkpoint directory.
    :param Path output_dir: Destination directory for exported artifacts.
    """
    cmd = [
        sys.executable,
        "-m",
        "deberta",
        "export",
        str(checkpoint_dir),
        "--what",
        "discriminator",
        "--output-dir",
        str(output_dir),
        "--allow-partial-export",
    ]
    logger.info(
        "Running post-train export in subprocess from checkpoint %s.",
        checkpoint_dir,
    )
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        logger.warning(
            "Post-train export subprocess failed (exit=%d). Output:\n%s",
            int(proc.returncode),
            str(proc.stdout).strip(),
        )
        return
    logger.info("Post-train export complete: %s", output_dir)
