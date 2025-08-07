"""Persist user dashboard preferences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

_PREF_FILE = Path(__file__).with_name("preferences.json")

DEFAULT_PREFERENCES: Dict[str, str] = {
    "theme": "light",
    "default_tab": "overview",
}


def load_preferences() -> Dict[str, str]:
    """Load preferences from disk, falling back to defaults."""
    if _PREF_FILE.exists():
        try:
            with _PREF_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                return {**DEFAULT_PREFERENCES, **data}
        except json.JSONDecodeError:
            pass
    return DEFAULT_PREFERENCES.copy()


def save_preferences(prefs: Dict[str, str]) -> None:
    """Persist preferences to disk."""
    _PREF_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _PREF_FILE.open("w", encoding="utf-8") as fh:
        json.dump(prefs, fh, indent=2)


__all__ = ["load_preferences", "save_preferences", "DEFAULT_PREFERENCES"]
