#!/usr/bin/env python
"""Deprecated wrapper for compile parity checks.

Use ``scratch/compile_parity_check.py`` directly.
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    print(
        "[DEPRECATED] scratch/compile_parity_check_gdes_sync.py is deprecated; "
        "forwarding to scratch/compile_parity_check.py"
    )
    target = Path(__file__).with_name("compile_parity_check.py")
    runpy.run_path(str(target), run_name="__main__")
