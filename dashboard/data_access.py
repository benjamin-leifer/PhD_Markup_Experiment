# flake8: noqa
# mypy: ignore-errors
import datetime
import io
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

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
    from battery_analysis.models import Sample
    from mongoengine import connect
except Exception:  # pragma: no cover - allow running without DB
    models = None
    Sample = None
    connect = None

try:  # pragma: no cover - manager optional
    from mongoengine.queryset.manager import QuerySetManager
except Exception:  # pragma: no cover - provide dummy fallback
    QuerySetManager = None  # type: ignore[assignment]

try:  # pragma: no cover - database utility optional
    from battery_analysis.utils.db import connect_with_fallback
except Exception:  # pragma: no cover - provide dummy fallback
    connect_with_fallback = None

from battery_analysis.utils.logging import get_logger

from Mongodb_implementation import get_client

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
_uploaded_files: List[Dict] = []
_DB_CONNECTED: bool | None = None
_DB_ERROR: str | None = None

# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_available_users() -> List[str]:
    """Return available usernames for the dashboard.

    The result is cached to avoid repeated lookups against the optional
    :mod:`battery_analysis.user_tracking` helper. When that package does not
    expose a helper or an error occurs, a small placeholder list is returned so
    the UI remains usable in offline or test environments.
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
    global _DB_CONNECTED, _DB_ERROR
    diagnostics: list[str] = []
    if _DB_CONNECTED is not None:
        logger.info("db_connected: returning cached status %s", _DB_CONNECTED)
        diagnostics.append(f"cached status: {_DB_CONNECTED}")
        return _DB_CONNECTED

    client = get_client()

    logger.info("Resolving MongoDB environment variables")
    diagnostics.append("Resolving environment variables")
    uri_env = os.getenv("MONGO_URI", "")
    host_env = os.getenv("MONGO_HOST", "localhost")
    port_env = os.getenv("MONGO_PORT", "27017")
    db_name = os.getenv("BATTERY_DB_NAME")
    if not db_name and uri_env:
        db_name = urlparse(uri_env).path.lstrip("/") or None
    db_name = db_name or "battery_test_db"
    uri = getattr(client, "_configured_uri", None) or uri_env or None
    host = getattr(client, "_configured_host", host_env)
    port = int(getattr(client, "_configured_port", port_env))
    logger.info(
        "Resolved env vars: MONGO_URI=%s MONGO_HOST=%s MONGO_PORT=%s DB_NAME=%s",
        uri_env,
        host_env,
        port_env,
        db_name,
    )
    diagnostics.append(
        f"Resolved env -> uri:{uri_env or 'None'} host:{host_env} port:{port_env} db:{db_name}"
    )
    logger.debug(
        "db_connected check using db=%s uri=%s host=%s port=%s",
        db_name,
        uri,
        host,
        port,
    )

    is_mock = False
    try:  # pragma: no cover - optional dependency
        import mongomock

        is_mock = isinstance(client, mongomock.MongoClient)
    except Exception:
        pass

    if not is_mock:
        logger.info(
            "Pinging MongoDB at %s",
            uri if uri else f"{host}:{port}",
        )
        diagnostics.append(f"Ping target: {uri if uri else f'{host}:{port}'}")
        try:
            client.admin.command("ping")
            logger.info("MongoDB ping succeeded")
            diagnostics.append("Ping succeeded")
        except Exception as exc:
            logger.warning(
                "MongoDB ping failed for %s:%s (uri=%s): %s",
                host,
                port,
                uri,
                exc,
            )
            logger.info("MongoDB ping failed: %s", exc)
            diagnostics.append(f"Ping failed: {exc}")
            is_mock = True

    if is_mock:
        logger.warning(
            "Using mongomock client; database operations will use in-memory mock",
        )
        diagnostics.append("Using mongomock client")
        _DB_ERROR = "Using mongomock in-memory client"
        _DB_CONNECTED = True
        return True

    logger.info("Checking database dependencies")
    diagnostics.append("Checking database dependencies")
    models_present = models is not None
    connect_present = connect is not None
    sample_present = Sample is not None
    sample_objects_present = (
        sample_present
        and QuerySetManager is not None
        and isinstance(Sample.__dict__.get("objects"), QuerySetManager)
    )
    logger.info("models module present: %s", models_present)
    logger.info("connect function present: %s", connect_present)
    logger.info("Sample class present: %s", sample_present)
    logger.info("Sample.objects present: %s", sample_objects_present)
    diagnostics.append(f"models module present: {models_present}")
    diagnostics.append(f"connect function present: {connect_present}")
    diagnostics.append(f"Sample class present: {sample_present}")
    diagnostics.append(f"Sample.objects present: {sample_objects_present}")
    essential_ok = all([models_present, connect_present])
    if not essential_ok:
        missing: list[str] = []
        if not models_present:
            missing.append("battery_analysis.models")
        if not connect_present:
            missing.append("mongoengine.connect")
        msg = f"Missing database dependencies: {', '.join(missing)}"
        logger.error(msg)
        _DB_ERROR = msg
        _DB_CONNECTED = False
        logger.info("DB connection diagnostics:\n%s", "\n".join(diagnostics))
        return False
    if not sample_present or not sample_objects_present:
        missing: list[str] = []
        if not sample_present:
            missing.append("Sample model")
        if sample_present and not sample_objects_present:
            missing.append("Sample.objects manager")
        msg = (
            "Missing optional database dependencies: "
            + ", ".join(missing)
            + "; proceeding with pymongo client"
        )
        logger.warning(msg)
        diagnostics.append(msg)
    logger.info(
        "Attempting MongoDB connection to %s",
        uri if uri else f"{host}:{port}",
    )
    diagnostics.append(f"Attempting connection to {uri if uri else f'{host}:{port}'}")
    try:
        connected = False
        if uri:
            logger.info("Connecting using URI")
            diagnostics.append("Connection branch: URI")
            if connect_with_fallback is not None:
                connected = connect_with_fallback(
                    db_name=db_name,
                    host=uri,
                    ask_if_fails=False,
                )
            else:
                connect(
                    db_name, host=uri, alias="default", serverSelectionTimeoutMS=2000
                )
                connected = True
            logger.info("URI connection result: %s", connected)
            diagnostics.append(f"URI connection result: {connected}")
        else:
            logger.info("Connecting using host/port")
            diagnostics.append("Connection branch: host/port")
            if connect_with_fallback is not None:
                connected = connect_with_fallback(
                    db_name=db_name,
                    host=host,
                    port=port,
                    ask_if_fails=False,
                )
            else:
                connect(
                    db_name,
                    host=host,
                    port=port,
                    alias="default",
                    serverSelectionTimeoutMS=2000,
                )
                connected = True
            logger.info("Host/port connection result: %s", connected)
            diagnostics.append(f"Host/port connection result: {connected}")
        if connected:
            logger.info(
                "MongoDB connection established to %s",
                uri if uri else f"{host}:{port}",
            )
            diagnostics.append("Connection established")
            if sample_objects_present:
                Sample.objects.first()  # type: ignore[attr-defined]
            else:
                diagnostics.append("Skipping Sample.objects check")
            _DB_ERROR = None
            _DB_CONNECTED = True
            return True
    except Exception as exc:
        _DB_ERROR = (
            f"MongoDB connection failed for {db_name} "
            f"(host={host} port={port}): {exc}"
        )
        logger.exception(_DB_ERROR)
        logger.info(
            "Connection attempt failed using uri=%s host=%s port=%s: %s",
            uri,
            host,
            port,
            exc,
        )
        diagnostics.append(f"Exception during connection: {exc}")
    else:
        last_err = None
        if connect_with_fallback is not None:
            last_err = getattr(connect_with_fallback, "last_error", None)
        if last_err:
            _DB_ERROR = (
                f"MongoDB connection failed for {db_name} "
                f"(host={host} port={port}): {last_err}"
            )
            diagnostics.append(f"Connection failed: {last_err}")
        else:
            _DB_ERROR = (
                f"MongoDB connection could not be established to "
                f"{uri if uri else f'{host}:{port}'}"
            )
            diagnostics.append("Connection failed: unknown error")
    logger.error(
        "MongoDB connection could not be established to %s; using demo data",
        uri if uri else f"{host}:{port}",
    )
    diagnostics.append(f"Using demo data for {uri if uri else f'{host}:{port}'}")
    logger.info("DB connection diagnostics:\n%s", "\n".join(diagnostics))
    _DB_CONNECTED = False
    return False


def get_db_error() -> str | None:
    """Return the reason why the MongoDB connection is unavailable."""
    return _DB_ERROR


# ---------------------------------------------------------------------------
# Data retrieval
# ---------------------------------------------------------------------------


def query_samples(
    query: Dict[str, Any] | None = None, fields: List[str] | None = None
) -> List[Any]:
    """Return samples matching ``query`` using available backend.

    When the optional :mod:`battery_analysis` models expose a mongoengine
    ``objects`` manager it is used with a raw query.  Otherwise the lightweight
    :func:`Mongodb_implementation.find_samples` helper is used which returns
    dictionaries.  The function logs which path was taken for easier debugging.
    """

    query = query or {}
    try:  # pragma: no cover - optional dependency
        from battery_analysis.models import Sample  # type: ignore

        if hasattr(Sample, "objects"):
            logger.debug("query_samples: using Sample.objects path")
            qs = Sample.objects  # type: ignore[attr-defined]
            if query:
                qs = qs(__raw__=query)
            if fields:
                qs = qs.only(*fields)
            return list(qs)
    except Exception:  # pragma: no cover - fallback when Sample unavailable
        logger.debug(
            "query_samples: Sample.objects unavailable; falling back to find_samples",
            exc_info=True,
        )

    logger.debug("query_samples: using find_samples path")
    from Mongodb_implementation import find_samples

    return find_samples(query)


def get_cell_dataset(cell_code: str):
    """Return the :class:`CellDataset` for ``cell_code`` if it exists.

    The function simply looks up the ``CellDataset`` by ``cell_code`` and
    returns ``None`` when the dataset is missing or the database is
    unavailable.
    """

    if not cell_code or models is None or not db_connected():
        return None

    try:  # pragma: no cover - depends on MongoDB
        return models.CellDataset.get_by_cell_code(cell_code)  # type: ignore[attr-defined]
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
    """Store a new material entry or log the details.

    When the database is available the helper uses
    :func:`Sample.get_or_create` to ensure a unique entry for the material.
    """

    if db_connected() and Sample is not None:  # pragma: no cover - requires database
        try:
            models.Sample.get_or_create(  # type: ignore[attr-defined]
                name, chemistry=chemistry, notes=notes
            )
            return
        except Exception:
            pass
    logger.info("New material added: %s, %s, %s", name, chemistry, notes)


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
