"""
Robust MongoDB connection helper.

Usage
-----
>>> from battery_analysis.utils import connect_to_database
>>> connected = connect_to_database()            # tries localhost first
>>> connected = connect_to_database(ask_if_fails=False)  # silent on failure
"""

try:  # pragma: no cover - depends on environment
    from mongoengine import connect
    from mongoengine.connection import ConnectionFailure, get_connection
except Exception:  # pragma: no cover - executed when mongoengine missing
    connect = None
    get_connection = None
    ConnectionFailure = Exception
import sys
from typing import Any

from battery_analysis.utils.logging import get_logger

logger = get_logger(__name__)


def connect_with_fallback(
    db_name: str = "battery_test_db",
    host: str = "localhost",
    port: int | None = None,
    ask_if_fails: bool = True,
    **connect_kwargs: Any,
) -> bool:
    """Attempt localhost connection first, then (optionally) prompt the user.

    Returns
    -------
    bool
        True if a connection was established, False otherwise.
    """
    if connect is None:
        logger.warning("mongoengine not available; cannot connect to database")
        connect_with_fallback.last_error = "mongoengine not available"  # type: ignore[attr-defined]  # noqa: E501
        return False
    try:
        is_uri = host.startswith(("mongodb://", "mongodb+srv://"))
        if is_uri:
            connect(
                db_name,
                host=host,
                serverSelectionTimeoutMS=2000,
                alias="default",
                **connect_kwargs,
            )
            logger.info("Connected to MongoDB via %s", host)
        else:
            port = port or 27017
            connect(
                db_name,
                host=host,
                port=port,
                serverSelectionTimeoutMS=2000,
                alias="default",
                **connect_kwargs,
            )
            logger.info("Connected to MongoDB at %s:%s", host, port)
        connect_with_fallback.last_error = None  # type: ignore[attr-defined]  # noqa: E501
        return True
    except Exception as exc:
        logger.warning("Local MongoDB connection failed: %s", exc)
        connect_with_fallback.last_error = str(exc)  # type: ignore[attr-defined]  # noqa: E501
        if not ask_if_fails:
            return False

    # -- interactive fallback ------------------------------------------------
    logger.warning("Could not reach MongoDB on localhost.")
    new_uri = input("MongoDB URI (blank to abort) > ").strip()
    if not new_uri:
        connect_with_fallback.last_error = "user aborted"  # type: ignore[attr-defined]  # noqa: E501
        return False
    try:
        connect(host=new_uri, alias="default", **connect_kwargs)
        logger.info("Connected via URI %s", new_uri)
        connect_with_fallback.last_error = None  # type: ignore[attr-defined]  # noqa: E501
        return True
    except Exception as exc:
        logger.error("Connection via %s also failed: %s", new_uri, exc)
        connect_with_fallback.last_error = str(exc)  # type: ignore[attr-defined]  # noqa: E501
        return False


def ensure_connection(**connect_kwargs: Any) -> bool:
    """Ensure a default MongoEngine connection exists.

    Parameters
    ----------
    **connect_kwargs
        Optional arguments passed through to :func:`connect_with_fallback` if a
        connection needs to be established.

    Returns
    -------
    bool
        ``True`` if a connection is available, ``False`` otherwise.
    """

    if connect is None or get_connection is None:
        msg = "mongoengine not available; cannot ensure DB connection"
        logger.warning(msg)
        return False

    try:
        # ``get_connection`` raises ``ConnectionFailure`` when the default
        # connection is missing.
        get_connection()
        return True
    except ConnectionFailure:
        return connect_with_fallback(ask_if_fails=False, **connect_kwargs)
    except Exception as exc:  # pragma: no cover - unexpected failures
        logger.error("Failed to ensure DB connection: %s", exc)
        return False


# Keep the legacy public name so existing imports work
connect_to_database = connect_with_fallback
connect_with_fallback.last_error = None  # type: ignore[attr-defined]  # noqa: E501


# Allow `python -m battery_analysis.utils.db` for a quick test
if __name__ == "__main__":
    sys.exit(0 if connect_with_fallback() else 1)
