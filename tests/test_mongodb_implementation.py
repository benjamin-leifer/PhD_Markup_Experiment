import sys
from pathlib import Path

# mypy: ignore-errors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os  # noqa: E402

import mongomock  # noqa: E402

import Mongodb_implementation  # noqa: E402


def test_get_client_uses_mongomock(monkeypatch):
    monkeypatch.setenv("USE_MONGO_MOCK", "1")
    client = Mongodb_implementation.get_client()
    assert isinstance(client, mongomock.MongoClient)
    expected_host = os.getenv("MONGO_HOST", "localhost")
    expected_port = int(os.getenv("MONGO_PORT", "27017"))
    assert getattr(client, "_configured_host") == expected_host
    assert getattr(client, "_configured_port") == expected_port


def test_get_client_falls_back_to_mongomock(monkeypatch):
    monkeypatch.delenv("USE_MONGO_MOCK", raising=False)
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.setenv("MONGO_HOST", "mongodb.invalid")
    monkeypatch.setenv("MONGO_PORT", "27017")
    client = Mongodb_implementation.get_client()
    assert isinstance(client, mongomock.MongoClient)
    assert getattr(client, "_configured_host") == "mongodb.invalid"
    assert getattr(client, "_configured_port") == 27017
