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
    from mongoengine.connection import get_connection, ConnectionFailure
except Exception:  # pragma: no cover - executed when mongoengine missing
    connect = None
    get_connection = None
    ConnectionFailure = Exception
import sys

from battery_analysis.utils.logging import get_logger

logger = get_logger(__name__)


def connect_with_fallback(
    db_name: str = "battery_test_db",
    host: str = "localhost",
    port: int = 27017,
    ask_if_fails: bool = True,
    **connect_kwargs,
) -> bool:
    """Attempt localhost connection first, then (optionally) prompt the user.

    Returns
    -------
    bool
        True if a connection was established, False otherwise.
    """
    if connect is None:
        logger.warning("mongoengine not available; cannot connect to database")
        return False
    try:
        connect(
            db_name,
            host=host,
            port=port,
            serverSelectionTimeoutMS=2000,
            alias="default",
            **connect_kwargs,
        )
        logger.info("✅ Connected to MongoDB at %s:%s", host, port)
        return True
    except Exception as exc:
        logger.warning("Local MongoDB connection failed: %s", exc)
        if not ask_if_fails:
            return False

    # ── interactive fallback ───────────────────────────────────────────────
    logger.warning("⚠️  Couldn’t reach MongoDB on localhost.")
    new_uri = input("MongoDB URI (blank to abort) > ").strip()
    if not new_uri:
        return False
    try:
        connect(host=new_uri, alias="default", **connect_kwargs)
        logger.info("✅ Connected via URI %s", new_uri)
        return True
    except Exception as exc:
        logger.error("Connection via %s also failed: %s", new_uri, exc)
        return False


def ensure_connection(**connect_kwargs) -> bool:
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
        logger.warning("mongoengine not available; cannot ensure DB connection")
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


# Allow `python -m battery_analysis.utils.db` for a quick test
if __name__ == "__main__":
    sys.exit(0 if connect_with_fallback() else 1)
