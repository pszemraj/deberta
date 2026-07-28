"""Export helpers used by the pretraining runtime."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def _export_discriminator_hf_subprocess(
    *,
    checkpoint_dir: Path,
    output_dir: Path,
) -> None:
    """Run and publish a strict discriminator export in an isolated subprocess.

    :param Path checkpoint_dir: Source checkpoint directory.
    :param Path output_dir: Managed destination directory for exported artifacts.
    """
    refresh_dir = output_dir.parent / f".{output_dir.name}.refresh-{uuid.uuid4().hex}"
    cmd = [
        sys.executable,
        "-m",
        "deberta",
        "export",
        str(checkpoint_dir),
        "--what",
        "discriminator",
        "--output-dir",
        str(refresh_dir),
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
        shutil.rmtree(refresh_dir, ignore_errors=True)
        raise RuntimeError(
            f"Post-train export subprocess failed (exit={int(proc.returncode)}). "
            f"Output:\n{str(proc.stdout).strip()}"
        )

    backup_dir: Path | None = None
    try:
        if output_dir.exists():
            backup_dir = output_dir.parent / f".{output_dir.name}.previous-{uuid.uuid4().hex}"
            output_dir.replace(backup_dir)
        try:
            refresh_dir.replace(output_dir)
        except Exception:
            if backup_dir is not None:
                backup_dir.replace(output_dir)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
    except Exception:
        shutil.rmtree(refresh_dir, ignore_errors=True)
        raise
    logger.info("Post-train export complete: %s", output_dir)
