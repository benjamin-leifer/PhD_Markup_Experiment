"""Check MongoDB connectivity.

This script attempts to connect to MongoDB using ``get_client`` from
``Mongodb_implementation`` and prints whether a real server or ``mongomock``
was used. Any connection errors are also reported.
"""

from Mongodb_implementation import get_client


def main() -> None:
    client = get_client()
    is_mock = client.__class__.__module__.startswith("mongomock")
    backend = "mongomock" if is_mock else "real MongoDB server"
    try:
        client.admin.command("ping")
        print(f"Using {backend}. Ping successful.")
    except Exception as exc:  # noqa: BLE001
        print(f"Using {backend}. Ping failed: {exc}")


if __name__ == "__main__":
    main()
