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
except Exception:  # pragma: no cover - executed when mongoengine missing
    connect = None
import logging
import sys


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
        logging.warning("mongoengine not available; cannot connect to database")
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
        logging.info(f"✅ Connected to MongoDB at {host}:{port}")
        return True
    except Exception as exc:
        logging.warning(f"Local MongoDB connection failed: {exc}")
        if not ask_if_fails:
            return False

    # ── interactive fallback ───────────────────────────────────────────────
    print("⚠️  Couldn’t reach MongoDB on localhost.")
    new_uri = input("MongoDB URI (blank to abort) > ").strip()
    if not new_uri:
        return False
    try:
        connect(host=new_uri, alias="default", **connect_kwargs)
        logging.info(f"✅ Connected via URI {new_uri}")
        return True
    except Exception as exc:
        logging.error(f"Connection via {new_uri} also failed: {exc}")
        return False


# Keep the legacy public name so existing imports work
connect_to_database = connect_with_fallback


# Allow `python -m battery_analysis.utils.db` for a quick test
if __name__ == "__main__":
    sys.exit(0 if connect_with_fallback() else 1)
