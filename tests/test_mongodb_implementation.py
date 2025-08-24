from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import mongomock
import Mongodb_implementation


def test_get_client_uses_mongomock(monkeypatch):
    monkeypatch.setenv("USE_MONGO_MOCK", "1")
    client = Mongodb_implementation.get_client()
    assert isinstance(client, mongomock.MongoClient)
    assert getattr(client, "_configured_host") == os.getenv("MONGO_HOST", "localhost")
    assert getattr(client, "_configured_port") == int(os.getenv("MONGO_PORT", "27017"))
