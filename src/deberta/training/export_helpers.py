"""Export helpers used by the pretraining runtime."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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
