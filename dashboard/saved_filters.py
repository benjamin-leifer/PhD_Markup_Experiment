"""Manage persisted filter presets for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

# File storing saved filter definitions
_FILTER_FILE = Path(__file__).with_name("saved_filters.json")


def _load_all() -> List[Dict]:
    """Load all filters from disk."""
    if _FILTER_FILE.exists():
        with _FILTER_FILE.open("r", encoding="utf-8") as fh:
            try:
                return json.load(fh)
            except json.JSONDecodeError:
                return []
    return []


def _write_all(filters: List[Dict]) -> None:
    """Write the filter list to disk."""
    with _FILTER_FILE.open("w", encoding="utf-8") as fh:
        json.dump(filters, fh, indent=2)


def save_filter(name: str, filter_dict: Dict[str, str], description: Optional[str] = None) -> None:
    """Save or update a filter preset."""
    filters = _load_all()
    for f in filters:
        if f.get("name") == name:
            f["filter"] = filter_dict
            if description is not None:
                f["description"] = description
            break
    else:
        filters.append({"name": name, "filter": filter_dict, "description": description or ""})
    _write_all(filters)


def load_filter(name: str) -> Dict[str, str]:
    """Return the filter dictionary for ``name``."""
    for f in _load_all():
        if f.get("name") == name:
            return dict(f.get("filter", {}))
    raise KeyError(f"Filter '{name}' not found")


def delete_filter(name: str) -> None:
    """Delete a saved filter by name."""
    filters = [f for f in _load_all() if f.get("name") != name]
    _write_all(filters)


def list_filters() -> List[Dict[str, str]]:
    """Return metadata for all saved filters."""
    return [{"name": f.get("name", ""), "description": f.get("description", "")} for f in _load_all()]
