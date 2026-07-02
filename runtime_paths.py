from __future__ import annotations

import os
import sys
from pathlib import Path


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def ensure_app_working_dir() -> Path:
    base_dir = app_base_dir()
    os.chdir(base_dir)
    return base_dir
