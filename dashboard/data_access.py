import datetime
import io
from typing import List, Dict

from battery_analysis import user_tracking

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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


def add_new_material(name: str, chemistry: str, notes: str) -> None:
    """Placeholder for storing a new material entry."""
    print(f"New material added: {name}, {chemistry}, {notes}")


def get_running_tests_csv() -> str:
    """Return running tests data formatted as CSV."""
    df = pd.DataFrame(get_running_tests())
    user_tracking.log_export("running_csv")
    return df.to_csv(index=False)


def get_upcoming_tests_csv() -> str:
    """Return upcoming tests data formatted as CSV."""
    df = pd.DataFrame(get_upcoming_tests())
    user_tracking.log_export("upcoming_csv")
    return df.to_csv(index=False)


def _tests_to_pdf(rows: List[Dict]) -> bytes:
    """Helper to render test rows into a simple PDF."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    for row in rows:
        parts = []
        for key, val in row.items():
            if isinstance(val, datetime.datetime):
                parts.append(f"{key}: {val.strftime('%Y-%m-%d %H:%M')}")
            else:
                parts.append(f"{key}: {val}")
        pdf.drawString(40, y, " | ".join(parts))
        y -= 20
        if y < 40:
            pdf.showPage()
            y = height - 40
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def get_running_tests_pdf() -> bytes:
    """Return running tests data formatted as PDF bytes."""
    user_tracking.log_export("running_pdf")
    return _tests_to_pdf(get_running_tests())


def get_upcoming_tests_pdf() -> bytes:
    """Return upcoming tests data formatted as PDF bytes."""
    user_tracking.log_export("upcoming_pdf")
    return _tests_to_pdf(get_upcoming_tests())
