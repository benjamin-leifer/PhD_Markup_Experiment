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
