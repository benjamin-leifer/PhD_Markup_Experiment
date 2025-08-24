"""Configuration loader for battery_analysis utilities.

This module reads optional user configuration from a TOML or INI file. The
configuration provides default values for command line utilities such as
``import_directory`` and ``import_watcher``.

Configuration files are searched in the user's home directory under the names
``.battery_analysis.toml`` or ``.battery_analysis.ini``.  The TOML format is
preferred when both exist.

Example TOML configuration::

    # ~/.battery_analysis.toml
    db_uri = "mongodb://localhost:27017/battery_test_db"
    workers = 4
    include = ["*.csv"]
    exclude = ["*/archive/*"]
    debounce = 1.0
    depth = 2

The same information may be expressed in INI format::

    # ~/.battery_analysis.ini
    [battery_analysis]
    db_uri = mongodb://localhost:27017/battery_test_db
    workers = 4
    include = *.csv,*.txt
    exclude = */archive/*
    debounce = 1.0
    depth = 2

Only keys relevant to the utilities are parsed; unknown entries are ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping
import configparser
import tomllib

__all__ = ["load_config"]

# Default configuration values used when no file is present
DEFAULTS: Dict[str, Any] = {
    "db_uri": "mongodb://localhost:27017/battery_test_db",
    "include": [],
    "exclude": [],
    "workers": None,
    "retries": 0,
    "debounce": 1.0,
    "depth": None,
    "slack_webhook": None,
    "email_recipients": [],
    "smtp_host": None,
    "email_sender": None,
    "redis_url": None,
    "redis_channel": "import_progress",
}


def _to_list(value: Any) -> list[str]:
    """Coerce ``value`` to a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        parts = [v.strip() for v in value.split(",")]
        return [p for p in parts if p]
    return []


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load configuration from ``path`` or the default user config file.

    Parameters
    ----------
    path:
        Optional explicit path to a configuration file. When omitted the loader
        searches ``~/.battery_analysis.toml`` and ``~/.battery_analysis.ini`` in
        that order.

    Returns
    -------
    dict
        Mapping with keys ``db_uri``, ``include``, ``exclude``, ``workers``,
        ``debounce`` and ``depth``. Missing keys fall back to sensible defaults.
    """

    cfg = DEFAULTS.copy()

    candidate: Path | None = None
    if path:
        candidate = Path(path).expanduser()
    else:
        home = Path.home()
        for name in (".battery_analysis.toml", ".battery_analysis.ini"):
            p = home / name
            if p.exists():
                candidate = p
                break

    if not candidate or not candidate.exists():
        return cfg

    try:
        if candidate.suffix.lower() == ".toml":
            with open(candidate, "rb") as fh:
                data = tomllib.load(fh)
        else:
            parser = configparser.ConfigParser()
            parser.read(candidate)
            # Prefer dedicated section but also allow defaults
            section: Mapping[str, str]
            if parser.has_section("battery_analysis"):
                section = parser["battery_analysis"]
            else:
                section = parser.defaults()
            data = dict(section)
    except Exception:
        # On any parsing error fall back to defaults
        return cfg

    # If TOML has dedicated table
    if isinstance(data, dict) and "battery_analysis" in data:
        data = data["battery_analysis"]

    if not isinstance(data, dict):
        return cfg

    for key in (
        "db_uri",
        "workers",
        "retries",
        "include",
        "exclude",
        "debounce",
        "depth",
        "slack_webhook",
        "email_recipients",
        "smtp_host",
        "email_sender",
        "redis_url",
        "redis_channel",
    ):
        if key not in data:
            continue
        value = data[key]
        if key in {"include", "exclude", "email_recipients"}:
            cfg[key] = _to_list(value)
        elif key in {"workers", "depth", "retries"}:
            try:
                cfg[key] = int(value) if value is not None else None
            except (TypeError, ValueError):
                pass
        elif key == "debounce":
            try:
                cfg[key] = float(value)
            except (TypeError, ValueError):
                pass
        else:
            cfg[key] = value

    return cfg
