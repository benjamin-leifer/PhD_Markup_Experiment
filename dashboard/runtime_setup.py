"""Utilities to configure runtime dependencies for the dashboard."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

from battery_analysis.utils.config import load_config


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

    cfg = load_config()
    os.environ.setdefault("MONGO_URI", cfg["db_uri"])

    host = os.environ.pop("BATTERY_DB_HOST", None)
    port = os.environ.pop("BATTERY_DB_PORT", None)

    if host:
        os.environ.setdefault("MONGO_HOST", host)
    else:
        os.environ.setdefault("MONGO_HOST", "localhost")

    if port:
        os.environ.setdefault("MONGO_PORT", port)
    else:
        os.environ.setdefault("MONGO_PORT", "27017")

