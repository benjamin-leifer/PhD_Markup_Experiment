"""Placeholder data access functions for the web dashboard."""

import datetime
from typing import List, Dict


def get_live_tests() -> List[Dict]:
    """Return example live test records."""
    now = datetime.datetime.now()
    return [
        {
            "cell_id": "Cell_A",
            "chemistry": "NMC",
            "cycle": 42,
            "status": "running",
            "last_update": now,
        },
        {
            "cell_id": "Cell_B",
            "chemistry": "LFP",
            "cycle": 10,
            "status": "paused",
            "last_update": now,
        },
    ]


def get_upcoming_tests() -> List[Dict]:
    """Return example upcoming test records."""
    return [
        {
            "cell_id": "Cell_X",
            "start_time": datetime.datetime.now()
            + datetime.timedelta(hours=1),  # noqa: E501
            "hardware": "Arbin_1",
        },
        {
            "cell_id": "Cell_Y",
            "start_time": datetime.datetime.now()
            + datetime.timedelta(hours=3),  # noqa: E501
            "hardware": "Arbin_2",
        },
    ]


def get_recent_results() -> List[Dict]:
    """Return example recent results."""
    return [
        {
            "cell_id": "Cell_Z",
            "finished": datetime.datetime.now() - datetime.timedelta(hours=4),
        },
        {
            "cell_id": "Cell_Q",
            "finished": datetime.datetime.now() - datetime.timedelta(hours=6),
        },
    ]


def get_summary_stats() -> Dict:
    """Return example summary statistics."""
    return {
        "running": 2,
        "completed_today": 5,
        "alerts": 0,
    }
