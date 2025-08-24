import sys
from pathlib import Path

# mypy: ignore-errors

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Mongodb_implementation  # noqa: E402
from dashboard import data_access  # noqa: E402


def test_db_connected_with_mongomock(monkeypatch) -> None:
    """db_connected returns True when a MongoDB connection fails."""

    # ensure cached state does not affect the test
    data_access._DB_CONNECTED = None

    class FailingMongoClient:
        def __init__(*args, **kwargs):
            from pymongo.errors import ServerSelectionTimeoutError

            raise ServerSelectionTimeoutError("fail")

    monkeypatch.setattr(
        Mongodb_implementation, "MongoClient", FailingMongoClient
    )  # noqa: E501

    assert data_access.db_connected() is True
    data_access._DB_CONNECTED = None


def test_db_connected_host_port_success(monkeypatch) -> None:
    """db_connected uses host/port when no URI is provided."""

    data_access._DB_CONNECTED = None
    data_access._DB_ERROR = None

    class DummySample:
        class objects:
            @staticmethod
            def first():
                return None

    monkeypatch.setattr(data_access, "Sample", DummySample)
    monkeypatch.setattr(data_access, "models", object())
    monkeypatch.setattr(data_access, "connect", lambda *a, **k: None)

    class DummyClient:
        def __init__(self):
            def _command(_self, _cmd):
                return {"ok": 1}

            self.admin = type("A", (), {"command": _command})()
            self._configured_host = "localhost"
            self._configured_port = 27017

    monkeypatch.setattr(data_access, "get_client", lambda: DummyClient())

    called: dict[str, int | str | None] = {}

    def fake_connect_with_fallback(db_name, host, port=None, **kwargs):
        called.update({"host": host, "port": port})
        return True

    monkeypatch.setattr(
        data_access, "connect_with_fallback", fake_connect_with_fallback
    )

    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.setenv("MONGO_HOST", "localhost")
    monkeypatch.setenv("MONGO_PORT", "27017")

    assert data_access.db_connected() is True
    assert called == {"host": "localhost", "port": 27017}
    data_access._DB_CONNECTED = None


def test_db_connected_uri_success(monkeypatch) -> None:
    """db_connected uses URI without passing a port."""

    data_access._DB_CONNECTED = None
    data_access._DB_ERROR = None

    class DummySample:
        class objects:
            @staticmethod
            def first():
                return None

    monkeypatch.setattr(data_access, "Sample", DummySample)
    monkeypatch.setattr(data_access, "models", object())
    monkeypatch.setattr(data_access, "connect", lambda *a, **k: None)

    uri = "mongodb://example.com/testdb"

    class DummyClient:
        def __init__(self):
            def _command(_self, _cmd):
                return {"ok": 1}

            self.admin = type("A", (), {"command": _command})()
            self._configured_uri = uri

    monkeypatch.setattr(data_access, "get_client", lambda: DummyClient())

    called: dict[str, int | str | None] = {}

    def fake_connect_with_fallback(db_name, host, port=None, **kwargs):
        called.update({"host": host, "port": port})
        return True

    monkeypatch.setattr(
        data_access, "connect_with_fallback", fake_connect_with_fallback
    )

    monkeypatch.setenv("MONGO_URI", uri)

    assert data_access.db_connected() is True
    assert called == {"host": uri, "port": None}
    data_access._DB_CONNECTED = None
    monkeypatch.delenv("MONGO_URI", raising=False)


def test_db_connected_reports_error(monkeypatch) -> None:
    """db_connected exposes the underlying connection failure."""

    data_access._DB_CONNECTED = None
    data_access._DB_ERROR = None

    class DummySample:
        class objects:
            @staticmethod
            def first():
                return None

    monkeypatch.setattr(data_access, "Sample", DummySample)
    monkeypatch.setattr(data_access, "models", object())
    monkeypatch.setattr(data_access, "connect", lambda *a, **k: None)

    class DummyClient:
        def __init__(self):
            def _command(_self, _cmd):
                return {"ok": 1}

            self.admin = type("A", (), {"command": _command})()
            self._configured_host = "localhost"
            self._configured_port = 27017

    monkeypatch.setattr(data_access, "get_client", lambda: DummyClient())

    def fake_connect_with_fallback(db_name, host, port=None, **kwargs):
        fake_connect_with_fallback.last_error = "boom"
        return False

    monkeypatch.setattr(
        data_access, "connect_with_fallback", fake_connect_with_fallback
    )

    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.setenv("MONGO_HOST", "localhost")
    monkeypatch.setenv("MONGO_PORT", "27017")

    assert data_access.db_connected() is False
    error = data_access.get_db_error() or ""
    assert "boom" in error
    data_access._DB_CONNECTED = None
