"""Check MongoDB connectivity.

This script attempts to connect to MongoDB using ``get_client`` from
``Mongodb_implementation`` and reports detailed configuration and connection
errors to help diagnose why the dashboard falls back to demo data.
"""

import os
from urllib.parse import urlparse

from Mongodb_implementation import get_client


def main() -> None:
    try:
        client = get_client()
    except Exception as exc:  # noqa: BLE001
        print(f"Could not create MongoDB client: {exc}")
        return
    uri_env = os.getenv("MONGO_URI", "")
    host = getattr(
        client,
        "_configured_host",
        os.getenv("MONGO_HOST", "localhost"),
    )
    port = getattr(
        client,
        "_configured_port",
        os.getenv("MONGO_PORT", "27017"),
    )
    uri = getattr(client, "_configured_uri", None) or uri_env or ""
    db_name = os.getenv("BATTERY_DB_NAME")
    if not db_name and uri:
        db_name = urlparse(uri).path.lstrip("/") or None
    db_name = db_name or "battery_test_db"
    base = f"Configured MongoDB: db={db_name}, uri='{uri}', "
    msg = base + f"host={host}, port={port}"
    print(msg)
    is_mock = client.__class__.__module__.startswith("mongomock")
    backend = "mongomock" if is_mock else "real MongoDB server"
    try:
        client.admin.command("ping")
        print(f"Using {backend}. Ping successful.")
    except Exception as exc:  # noqa: BLE001
        print(f"Using {backend}. Ping failed: {exc}")


if __name__ == "__main__":
    main()
