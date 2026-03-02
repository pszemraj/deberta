"""Shared logging helpers."""

from __future__ import annotations

import logging


def setup_process_logging(is_main: bool) -> None:
    """Configure process-local logging.

    :param bool is_main: True for main process.
    """
    level = logging.INFO if is_main else logging.WARN
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
