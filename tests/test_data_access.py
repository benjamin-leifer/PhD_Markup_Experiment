from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mongomock  # noqa: E402
from dashboard import data_access  # noqa: E402
from mongoengine import connect, disconnect  # noqa: E402


def test_db_connected_with_mongomock(monkeypatch):
    """db_connected returns True when a mock MongoDB is available."""
    # ensure cached state does not affect the test
    data_access._DB_CONNECTED = None
    connect("testdb", mongo_client_class=mongomock.MongoClient, alias="default")
    monkeypatch.setattr(data_access, "connect_with_fallback", lambda **_: True)
    assert data_access.db_connected() is True
    disconnect()
    data_access._DB_CONNECTED = None
