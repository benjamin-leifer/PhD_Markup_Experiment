import datetime
import io
import os
from pathlib import Path
from typing import Any, Dict, List

try:  # pragma: no cover - optional dependency
    from battery_analysis import user_tracking
except Exception:  # pragma: no cover - provide dummy fallback

    class _UserTracking:
        @staticmethod
        def log_export(_name: str) -> None:
            """Fallback no-op logger."""

        @staticmethod
        def get_available_users() -> list[str]:  # type: ignore[override]
            return []

    user_tracking = _UserTracking()

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

try:  # pragma: no cover - database optional
    from battery_analysis import models
    from battery_analysis.utils.db import connect_with_fallback
except Exception:  # pragma: no cover - allow running without DB
    models = None
    connect_with_fallback = None


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
_uploaded_files: List[Dict] = []
_DB_CONNECTED: bool | None = None

# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------


def get_available_users() -> List[str]:
    """Return available usernames for the dashboard.

    The function attempts to retrieve the list from
    :mod:`battery_analysis.user_tracking`. When that package does not expose
    a helper or an error occurs, a small placeholder list is returned so the
    UI remains usable in offline or test environments.
    """

    for attr in ("get_available_users", "get_users", "available_users"):
        getter = getattr(user_tracking, attr, None)
        if callable(getter):
            try:
                users = list(getter())
                if users:
                    return users
            except Exception:
                pass
    return ["user1", "user2"]


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
# Data retrieval
# ---------------------------------------------------------------------------


def get_cell_dataset(cell_code: str):
    """Return the :class:`CellDataset` for ``cell_code`` if it exists.

    The function simply looks up the ``CellDataset`` by ``cell_code`` and
    returns ``None`` when the dataset is missing or the database is
    unavailable.
    """

    if not cell_code or models is None or not db_connected():
        return None

    try:  # pragma: no cover - depends on MongoDB
        return models.CellDataset.objects.get(cell_code=cell_code)  # type: ignore[attr-defined]
    except Exception:
        return None


def get_running_tests(
    limit: int | None = None,
    offset: int = 0,
    fields: List[str] | None = None,
) -> Dict[str, Any]:
    """Return currently running tests with minimal fields.

    Parameters
    ----------
    limit:
        Maximum number of rows to return. When ``None`` all rows are returned.
    offset:
        Number of initial rows to skip before returning results. Useful for
        implementing server-side pagination.
    fields:
        Optional list of :class:`TestResult` field names to fetch. When provided,
        MongoEngine's ``only`` is used to limit the fields retrieved from the
        database.

    Returns
    -------
    dict
        Dictionary with ``rows`` containing serialized test information and
        ``total`` with the total number of matching records.
    """

    now = datetime.datetime.now()
    if not db_connected():
        return {"rows": [], "total": 0}
    try:  # pragma: no cover - requires database
        tests = models.TestResult.objects(validated=False)  # type: ignore[attr-defined]
        total = tests.count()
        tests = tests.order_by("-date")
        if fields:
            tests = tests.only(*fields)
        if offset:
            tests = tests.skip(offset)
        if limit:
            tests = tests.limit(limit)
    except Exception as exc:  # pragma: no cover - requires database
        raise RuntimeError("Failed to query running tests") from exc
    rows: List[Dict] = []
    for test in tests:
        try:
            sample = (
                test.sample.fetch() if hasattr(test.sample, "fetch") else test.sample
            )
            sample_name = getattr(sample, "name", str(sample.id))
        except Exception:
            sample_name = "Unknown"
        rows.append(
            {
                "cell_id": sample_name,
                "test_type": getattr(test, "test_type", ""),
                "timestamp": getattr(test, "date", now),
            }
        )
    return {"rows": rows, "total": total}


def get_upcoming_tests(
    limit: int | None = None,
    offset: int = 0,
    fields: List[str] | None = None,
) -> Dict[str, Any]:
    """Return upcoming scheduled tests with minimal fields.

    Parameters
    ----------
    limit:
        Maximum number of rows to return. When ``None`` all rows are returned.
    offset:
        Number of initial rows to skip before returning results. Useful for
        implementing server-side pagination.
    fields:
        Optional list of :class:`TestResult` field names to fetch. When provided,
        MongoEngine's ``only`` is used to limit the fields retrieved from the
        database.

    Returns
    -------
    dict
        Dictionary with ``rows`` containing serialized test information and
        ``total`` with the total number of matching records.
    """

    now = datetime.datetime.now()
    if not db_connected():
        return {"rows": [], "total": 0}
    try:  # pragma: no cover - requires database
        tests = models.TestResult.objects(date__gt=now)  # type: ignore[attr-defined]
        total = tests.count()
        tests = tests.order_by("date")
        if fields:
            tests = tests.only(*fields)
        if offset:
            tests = tests.skip(offset)
        if limit:
            tests = tests.limit(limit)
    except Exception as exc:  # pragma: no cover - requires database
        raise RuntimeError("Failed to query upcoming tests") from exc
    rows: List[Dict] = []
    for test in tests:
        try:
            sample = (
                test.sample.fetch() if hasattr(test.sample, "fetch") else test.sample
            )
            sample_name = getattr(sample, "name", str(sample.id))
        except Exception:
            sample_name = "Unknown"
        rows.append(
            {
                "cell_id": sample_name,
                "test_type": getattr(test, "test_type", getattr(test, "name", "")),
                "timestamp": getattr(test, "date", now),
            }
        )
    return {"rows": rows, "total": total}


def get_summary_stats() -> Dict:
    """Return summary statistics about tests."""
    if not db_connected():
        return {}
    try:  # pragma: no cover - requires database
        running = models.TestResult.objects(validated=False).count()  # type: ignore[attr-defined]
        today = datetime.datetime.combine(datetime.date.today(), datetime.time())
        completed = models.TestResult.objects(date__gte=today, validated=True).count()  # type: ignore[attr-defined]
        failures = models.TestResult.objects(notes__icontains="fail").count()  # type: ignore[attr-defined]
        return {
            "running": running,
            "completed_today": completed,
            "failures": failures,
        }
    except Exception as exc:  # pragma: no cover - requires database
        raise RuntimeError("Failed to query summary stats") from exc


def get_test_metadata(cell_id: str) -> Dict:
    """Return detailed metadata for ``cell_id``."""
    if not db_connected():
        raise RuntimeError("Database unavailable")
    try:  # pragma: no cover - requires database
        sample = models.Sample.get_by_name(cell_id)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - requires database
        raise RuntimeError(f"Failed to fetch metadata for {cell_id}") from exc
    if not sample:
        raise RuntimeError(f"Sample {cell_id} not found")
    formation = getattr(sample, "formation_date", None)
    return {
        "cell_id": cell_id,
        "chemistry": getattr(sample, "chemistry", "Unknown"),
        "formation_date": formation.strftime("%Y-%m-%d") if formation else "Unknown",
        "notes": getattr(sample, "notes", ""),
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


def get_running_tests_csv(
    limit: int | None = None, fields: List[str] | None = None
) -> str:
    """Return running tests data formatted as CSV."""
    data = get_running_tests(limit=limit, fields=fields)["rows"]
    df = pd.DataFrame(data)
    user_tracking.log_export("running_csv")
    return df.to_csv(index=False)


def get_upcoming_tests_csv(
    limit: int | None = None, fields: List[str] | None = None
) -> str:
    """Return upcoming tests data formatted as CSV."""
    data = get_upcoming_tests(limit=limit, fields=fields)["rows"]
    df = pd.DataFrame(data)
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


def get_running_tests_pdf(
    limit: int | None = None, fields: List[str] | None = None
) -> bytes:
    """Return running tests data formatted as PDF bytes."""
    user_tracking.log_export("running_pdf")
    rows = get_running_tests(limit=limit, fields=fields)["rows"]
    return _tests_to_pdf(rows)


def get_upcoming_tests_pdf(
    limit: int | None = None, fields: List[str] | None = None
) -> bytes:
    """Return upcoming tests data formatted as PDF bytes."""
    user_tracking.log_export("upcoming_pdf")
    rows = get_upcoming_tests(limit=limit, fields=fields)["rows"]
    return _tests_to_pdf(rows)


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
