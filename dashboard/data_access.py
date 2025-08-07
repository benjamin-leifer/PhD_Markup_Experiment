import datetime
import io
import os
from pathlib import Path
from typing import Dict, List

from battery_analysis import user_tracking

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

try:  # pragma: no cover - database optional
    from battery_analysis import models
    from battery_analysis.utils.db import connect_with_fallback
except Exception:  # pragma: no cover - allow running without DB
    models = None
    connect_with_fallback = None


UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
_uploaded_files: List[Dict] = []
_DB_CONNECTED: bool | None = None

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def db_connected() -> bool:
    """Return True if the MongoDB backend is reachable."""
    global _DB_CONNECTED
    if _DB_CONNECTED is not None:
        return _DB_CONNECTED
    if models is None or connect_with_fallback is None:
        _DB_CONNECTED = False
        return False
    host = os.getenv("BATTERY_DB_HOST", "localhost")
    port = int(os.getenv("BATTERY_DB_PORT", "27017"))
    db_name = os.getenv("BATTERY_DB_NAME", "battery_test_db")
    connected = connect_with_fallback(
        db_name=db_name, host=host, port=port, ask_if_fails=False
    )
    if connected:
        try:  # pragma: no cover - requires database
            models.Sample.objects.first()  # type: ignore[attr-defined]
            _DB_CONNECTED = True
            return True
        except Exception:
            pass
    _DB_CONNECTED = False
    return False


# ---------------------------------------------------------------------------
# Data retrieval with graceful fallback to demo data
# ---------------------------------------------------------------------------


def _placeholder_running(now: datetime.datetime) -> List[Dict]:
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


def get_running_tests() -> List[Dict]:
    """Return currently running tests from the database or example data."""
    now = datetime.datetime.now()
    if db_connected():  # pragma: no cover - requires database
        try:
            tests = models.TestResult.objects(validated=False).order_by("-date")  # type: ignore[attr-defined]
            rows: List[Dict] = []
            for test in tests:
                try:
                    sample = (
                        test.sample.fetch()
                        if hasattr(test.sample, "fetch")
                        else test.sample
                    )
                    sample_name = getattr(sample, "name", str(sample.id))
                    chemistry = getattr(sample, "chemistry", "")
                except Exception:
                    sample_name, chemistry = "Unknown", ""
                cycle = (
                    test.cycle_count
                    if getattr(test, "cycle_count", None) is not None
                    else len(getattr(test, "cycles", []) or [])
                )
                rows.append(
                    {
                        "cell_id": sample_name,
                        "chemistry": chemistry or "Unknown",
                        "test_type": getattr(test, "test_type", ""),
                        "current_cycle": cycle,
                        "last_timestamp": getattr(test, "date", now),
                        "test_schedule": getattr(
                            getattr(test, "protocol", None), "name", test.name or ""
                        ),
                        "status": "running",
                    }
                )
            if rows:
                return rows
        except Exception:
            pass
    return _placeholder_running(now)


def _placeholder_upcoming(now: datetime.datetime) -> List[Dict]:
    return [
        {
            "cell_id": "Cell_010",
            "start_time": now + datetime.timedelta(hours=2),
            "hardware": "Arbin_1",
            "notes": "Preconditioned",
        },
        {
            "cell_id": "Cell_011",
            "start_time": now + datetime.timedelta(hours=4),
            "hardware": "Arbin_2",
            "notes": "High temperature",
        },
    ]


def get_upcoming_tests() -> List[Dict]:
    """Return upcoming scheduled tests or example data."""
    now = datetime.datetime.now()
    if db_connected():  # pragma: no cover - requires database
        try:
            tests = models.TestResult.objects(date__gt=now).order_by("date")  # type: ignore[attr-defined]
            rows: List[Dict] = []
            for test in tests:
                try:
                    sample = (
                        test.sample.fetch()
                        if hasattr(test.sample, "fetch")
                        else test.sample
                    )
                    sample_name = getattr(sample, "name", str(sample.id))
                except Exception:
                    sample_name = "Unknown"
                rows.append(
                    {
                        "cell_id": sample_name,
                        "start_time": getattr(test, "date", now),
                        "hardware": getattr(test, "tester", ""),
                        "notes": getattr(test, "notes", ""),
                    }
                )
            if rows:
                return rows
        except Exception:
            pass
    return _placeholder_upcoming(now)


def get_summary_stats() -> Dict:
    """Return summary statistics about tests."""
    if db_connected():  # pragma: no cover - requires database
        try:
            running = models.TestResult.objects(validated=False).count()  # type: ignore[attr-defined]
            today = datetime.datetime.combine(datetime.date.today(), datetime.time())
            completed = models.TestResult.objects(date__gte=today, validated=True).count()  # type: ignore[attr-defined]
            failures = models.TestResult.objects(notes__icontains="fail").count()  # type: ignore[attr-defined]
            return {
                "running": running,
                "completed_today": completed,
                "failures": failures,
            }
        except Exception:
            pass
    return {"running": 2, "completed_today": 5, "failures": 0}


def get_test_metadata(cell_id: str) -> Dict:
    """Return detailed metadata for ``cell_id``."""
    if db_connected():  # pragma: no cover - requires database
        try:
            sample = models.Sample.objects(name=cell_id).first()  # type: ignore[attr-defined]
            if sample:
                formation = getattr(sample, "formation_date", None)
                return {
                    "cell_id": cell_id,
                    "chemistry": getattr(sample, "chemistry", "Unknown"),
                    "formation_date": (
                        formation.strftime("%Y-%m-%d") if formation else "Unknown"
                    ),
                    "notes": getattr(sample, "notes", ""),
                }
        except Exception:
            pass
    return {
        "cell_id": cell_id,
        "chemistry": "Gr|NMC",
        "formation_date": "2024-01-01",
        "notes": "Example cell used for demo purposes.",
    }


def add_new_material(name: str, chemistry: str, notes: str) -> None:
    """Store a new material entry in the database or print a message."""
    if db_connected():  # pragma: no cover - requires database
        try:
            models.Sample(name=name, chemistry=chemistry, notes=notes).save()  # type: ignore[attr-defined]
            return
        except Exception:
            pass
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


def store_temp_upload(filename: str, content: bytes) -> str:
    """Save raw uploaded file content to server-side storage."""
    path = UPLOAD_DIR / filename
    with open(path, "wb") as f:
        f.write(content)
    return str(path)


def register_upload(filename: str, path: str, cycles, metadata) -> None:
    """Persist parsed upload information in memory."""
    _uploaded_files.append(
        {"filename": filename, "path": path, "cycles": cycles, "metadata": metadata}
    )


def get_uploaded_files() -> List[Dict]:
    """Return list of uploaded files and metadata."""
    return _uploaded_files
