"""Shared run-directory layout and metadata validation helpers."""

from __future__ import annotations

from pathlib import Path

from deberta.config import validate_run_metadata_schema
from deberta.io_utils import load_json_mapping

RUN_METADATA_FILENAME = "run_metadata.json"
MODEL_CONFIG_FILENAME = "model_config.json"
DATA_CONFIG_FILENAME = "data_config.json"
TRAIN_CONFIG_FILENAME = "train_config.json"
OPTIM_CONFIG_FILENAME = "optim_config.json"
LOGGING_CONFIG_FILENAME = "logging_config.json"
RESUME_SOURCE_FILENAME = "resume_source.json"
RUN_SNAPSHOT_FILENAMES: tuple[str, ...] = (
    MODEL_CONFIG_FILENAME,
    DATA_CONFIG_FILENAME,
    TRAIN_CONFIG_FILENAME,
    OPTIM_CONFIG_FILENAME,
    LOGGING_CONFIG_FILENAME,
    RUN_METADATA_FILENAME,
)


def infer_run_dir_from_checkpoint(checkpoint_dir: str | Path) -> Path:
    """Infer run directory from a checkpoint directory path.

    :param str | Path checkpoint_dir: Checkpoint directory path.
    :return Path: Parent run directory.
    """
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    return checkpoint_path.parent


def validate_run_metadata_file(run_dir: Path, *, required: bool) -> None:
    """Validate run metadata schema when present.

    :param Path run_dir: Run directory path.
    :param bool required: Whether a missing metadata file is an error.
    :raises ValueError: If metadata is missing when required, or schema-invalid.
    """
    meta_path = Path(run_dir) / RUN_METADATA_FILENAME
    if not meta_path.exists():
        if required:
            raise ValueError(f"Missing required run metadata file: {meta_path}")
        return

    raw = load_json_mapping(meta_path)
    validate_run_metadata_schema(raw, source=str(meta_path))
