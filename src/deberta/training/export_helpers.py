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
)

logger = logging.getLogger(__name__)
_REPO_URL = "https://github.com/pszemraj/deberta"


def _write_export_readme(
    output_dir: Path,
    *,
    model_cfg: Any,
    data_cfg: Any | None = None,
    train_cfg: Any,
    embedding_sharing: str,
) -> None:
    """Write a basic README.md and LICENSE to the export directory.

    :param Path output_dir: Export destination directory.
    :param Any model_cfg: Model configuration dataclass.
    :param Any | None data_cfg: Optional data configuration dataclass.
    :param Any train_cfg: Training configuration dataclass.
    :param str embedding_sharing: Embedding sharing mode.
    """
    backbone = str(getattr(model_cfg, "backbone_type", "unknown"))
    hidden = int(getattr(model_cfg, "hidden_size", 0))
    layers = int(getattr(model_cfg, "num_hidden_layers", 0) or 0)
    heads = int(getattr(model_cfg, "num_attention_heads", 0) or 0)
    seq_len = int(getattr(data_cfg, "max_seq_length", 0) or 0)
    if seq_len == 0:
        seq_len = int(getattr(model_cfg, "max_position_embeddings", 0) or 0)
    steps = int(getattr(train_cfg, "max_steps", 0) or 0)

    if backbone == "rope":
        arch_desc = "RoPE encoder (RMSNorm, SwiGLU, rotary embeddings)"
        usage_snippet = """from transformers import AutoTokenizer
from deberta.modeling.rope_encoder import DebertaRoPEModel

model = DebertaRoPEModel.from_pretrained("path/to/this/dir")
tokenizer = AutoTokenizer.from_pretrained("path/to/this/dir")
"""
        compatibility_note = (
            "Note: RoPE exports use a custom `model_type` (`deberta-rope`) and are not currently "
            "loadable via `transformers.AutoModel.from_pretrained(...)` without custom auto-class registration."
        )
    else:
        arch_desc = "DeBERTa-v2 (disentangled attention, LayerNorm)"
        usage_snippet = """from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("path/to/this/dir")
tokenizer = AutoTokenizer.from_pretrained("path/to/this/dir")
"""
        compatibility_note = ""

    readme = f"""---
library_name: transformers
tags:
- deberta
- encoder
- rtd
- fill-mask
license: mit
---

# {backbone}-{hidden}h-{layers}L-{heads}H

RTD-pretrained encoder ({arch_desc}).

| Parameter | Value |
|---|---|
| Backbone | `{backbone}` |
| Hidden size | {hidden} |
| Layers | {layers} |
| Attention heads | {heads} |
| Max sequence length | {seq_len} |
| Embedding sharing | `{embedding_sharing}` |
| Training steps | {steps} |

## Training

Pretrained with replaced-token detection (RTD / ELECTRA-style) using
[pszemraj/deberta]({_REPO_URL}).

## Usage

```python
{usage_snippet}
```
{compatibility_note}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    mit_license = (
        "MIT License\n\n"
        "Copyright (c) 2025-2026 Peter Szemraj\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        'of this software and associated documentation files (the "Software"), to deal\n'
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n\n"
        "The above copyright notice and this permission notice shall be included in all\n"
        "copies or substantial portions of the Software.\n\n"
        'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n'
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
        "SOFTWARE.\n"
    )
    (output_dir / "LICENSE").write_text(mit_license, encoding="utf-8")


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
