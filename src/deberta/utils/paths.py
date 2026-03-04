"""Filesystem path helpers shared by training and export flows."""

from __future__ import annotations

from pathlib import Path


def validate_existing_output_dir(
    *,
    output_dir: Path,
    allow_nonempty: bool,
    nonempty_error: str,
    nondir_error: str,
) -> bool:
    """Validate output-dir shape/emptiness and return non-empty state.

    :param Path output_dir: Candidate output path.
    :param bool allow_nonempty: Whether existing non-empty directories are allowed.
    :param str nonempty_error: Error message for disallowed non-empty directories.
    :param str nondir_error: Error message for non-directory existing paths.
    :return bool: True when output_dir exists and is non-empty.
    """
    if not output_dir.exists():
        return False
    if not output_dir.is_dir():
        raise ValueError(nondir_error)
    is_nonempty = any(output_dir.iterdir())
    if is_nonempty and (not allow_nonempty):
        raise ValueError(nonempty_error)
    return bool(is_nonempty)
