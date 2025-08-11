"""Utilities to configure runtime dependencies for the dashboard."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


def configure() -> None:
    """Ensure battery_analysis is installed and DB env vars are set.

    Installs the local ``battery_analysis`` package if it's not already
    available and populates MongoDB environment variables with sensible
    defaults when they are missing.
    """
    try:
        importlib.import_module("battery_analysis")
    except ModuleNotFoundError:
        package_dir = (
            Path(__file__).resolve().parent.parent
            / "Python_Codes"
            / "BLeifer_Battery_Analysis"
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", str(package_dir)]
        )

    os.environ.setdefault("BATTERY_DB_HOST", "localhost")
    os.environ.setdefault("BATTERY_DB_PORT", "27017")
    os.environ.setdefault("BATTERY_DB_NAME", "battery_test_db")
