#!/usr/bin/env python3
"""Compatibility shim for the removed FlashDeBERTa training wrapper."""

from __future__ import annotations

import sys


def main() -> int:
    """Report the supported config-driven FlashDeBERTa entrypoint.

    :return int: Process exit code.
    """

    sys.stderr.write(
        "tools/train_flashdeberta.py has been retired. Use the standard CLI with "
        "`deberta train ... --model.hf.attention_impl flash` after installing `.[flash]`.\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
