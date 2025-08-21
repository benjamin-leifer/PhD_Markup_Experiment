"""Persist user dashboard preferences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_PREF_FILE = Path(__file__).with_name("preferences.json")

# Include watcher configuration so directories to monitor can be persisted
# between application restarts.
DEFAULT_PREFERENCES: Dict[str, Any] = {
    "theme": "light",
    "default_tab": "overview",
    "watcher_dirs": [],
}


def load_preferences() -> Dict[str, Any]:
    """Load preferences from disk, falling back to defaults."""
    if _PREF_FILE.exists():
        try:
            with _PREF_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                return {**DEFAULT_PREFERENCES, **data}
        except json.JSONDecodeError:
            pass
    return DEFAULT_PREFERENCES.copy()


def save_preferences(prefs: Dict[str, Any]) -> None:
    """Persist preferences to disk."""
    _PREF_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _PREF_FILE.open("w", encoding="utf-8") as fh:
        json.dump(prefs, fh, indent=2)


__all__ = ["load_preferences", "save_preferences", "DEFAULT_PREFERENCES"]
