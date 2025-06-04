import datetime
from typing import List, Dict

# Placeholder functions for database access.
# These would normally interface with MongoDB via pymongo or mongoengine.


def get_running_tests() -> List[Dict]:
    """Return placeholder running test records."""
    now = datetime.datetime.now()
    return [
        {
            "cell_id": "Cell_001",
            "chemistry": "Gr|NMC",
            "test_type": "Cycle Life",
            "current_cycle": 120,
            "last_timestamp": now,
            "test_schedule": "1C charge/1C discharge",
            "status": "running",
        },
        {
            "cell_id": "Cell_002",
            "chemistry": "LFP",
            "test_type": "EIS",
            "current_cycle": 5,
            "last_timestamp": now,
            "test_schedule": "EIS every 10 cycles",
            "status": "paused",
        },
    ]


def get_upcoming_tests() -> List[Dict]:
    """Return placeholder upcoming test records."""
    return [
        {
            "cell_id": "Cell_010",
            "start_time": datetime.datetime.now() + datetime.timedelta(hours=2),
            "hardware": "Arbin_1",
            "notes": "Preconditioned",
        },
        {
            "cell_id": "Cell_011",
            "start_time": datetime.datetime.now() + datetime.timedelta(hours=4),
            "hardware": "Arbin_2",
            "notes": "High temperature",
        },
    ]


def get_summary_stats() -> Dict:
    """Return placeholder summary statistics."""
    return {
        "running": 2,
        "completed_today": 5,
        "failures": 0,
    }


def get_test_metadata(cell_id: str) -> Dict:
    """Return placeholder detailed metadata for a cell."""
    return {
        "cell_id": cell_id,
        "chemistry": "Gr|NMC",
        "formation_date": "2024-01-01",
        "notes": "Example cell used for demo purposes.",
    }
