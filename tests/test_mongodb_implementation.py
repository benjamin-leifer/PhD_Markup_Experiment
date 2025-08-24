import sys
from pathlib import Path

# mypy: ignore-errors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os  # noqa: E402

import mongomock  # noqa: E402
from pymongo.errors import ServerSelectionTimeoutError  # noqa: E402

import Mongodb_implementation  # noqa: E402


def test_get_client_falls_back_to_mongomock(monkeypatch):
    """If MongoDB is unreachable, get_client should return mongomock."""

    monkeypatch.delenv("USE_MONGO_MOCK", raising=False)

    class FailingMongoClient:
        def __init__(*args, **kwargs):
            raise ServerSelectionTimeoutError("fail")

    monkeypatch.setattr(
        Mongodb_implementation, "MongoClient", FailingMongoClient
    )  # noqa: E501

    client = Mongodb_implementation.get_client()
    assert isinstance(client, mongomock.MongoClient)
    expected_host = os.getenv("MONGO_HOST", "localhost")
    expected_port = int(os.getenv("MONGO_PORT", "27017"))
    assert getattr(client, "_configured_host") == expected_host
    assert getattr(client, "_configured_port") == expected_port
