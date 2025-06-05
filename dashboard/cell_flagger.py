"""Utility module for flagging samples for review or retest."""

from __future__ import annotations

from typing import Dict, List

# in-memory storage for flagged samples
_FLAGS: Dict[str, str] = {}


def flag_sample(sample_id: str, reason: str) -> None:
    """Flag a sample for the given reason."""
    _FLAGS[sample_id] = reason
    try:
        from battery_analysis import user_tracking

        user_tracking.log_flag(sample_id, reason)
    except Exception:
        pass


def get_flags() -> List[Dict[str, str]]:
    """Return a list of all flagged samples."""
    return [{"sample_id": sid, "reason": reason} for sid, reason in _FLAGS.items()]


def clear_flag(sample_id: str) -> None:
    """Remove the flag for ``sample_id`` if present."""
    _FLAGS.pop(sample_id, None)
    try:
        from battery_analysis import user_tracking

        user_tracking.log_flag(sample_id, "cleared")
    except Exception:
        pass
