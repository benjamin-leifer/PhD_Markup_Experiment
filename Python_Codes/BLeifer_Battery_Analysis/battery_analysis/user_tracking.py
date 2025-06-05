"""Simple user tracking utilities."""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, Optional

_current_user: Optional[str] = None


def set_current_user(username: str) -> None:
    """Set the active username for the session."""
    global _current_user
    _current_user = username


def get_current_user() -> Optional[str]:
    """Return the currently active user, if any."""
    return _current_user


def _log(action: str, **details: Any) -> None:
    entry: Dict[str, Any] = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "user": _current_user,
        "action": action,
        "details": details,
    }
    logging.info("USER_ACTION %s", json.dumps(entry))


def log_filter_run(filter_name: str, params: Dict[str, Any]) -> None:
    """Record that a filter was executed."""
    _log("filter_run", filter=filter_name, params=params)


def log_flag(sample_id: str, reason: str) -> None:
    """Record that a sample was flagged or unflagged."""
    _log("flag", sample_id=sample_id, reason=reason)


def log_export(kind: str) -> None:
    """Record that a data export occurred."""
    _log("export", kind=kind)
