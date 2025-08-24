import sys
from pathlib import Path

# mypy: ignore-errors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mongomock  # noqa: E402
from pymongo.errors import ServerSelectionTimeoutError  # noqa: E402

import Mongodb_implementation  # noqa: E402


def test_get_client_falls_back_to_mongomock(monkeypatch):
    """Return a mongomock client when a real connection fails."""

    class FailingMongoClient:
        def __init__(*args, **kwargs):
            raise ServerSelectionTimeoutError("fail")

    monkeypatch.setattr(
        Mongodb_implementation,
        "MongoClient",
        FailingMongoClient,
    )

    client = Mongodb_implementation.get_client()
    assert isinstance(client, mongomock.MongoClient)
