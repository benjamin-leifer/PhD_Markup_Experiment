from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
MODELS_ROOT = PACKAGE_ROOT / "battery_analysis" / "models"
for path in (ROOT, PACKAGE_ROOT, MODELS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mongomock  # noqa: E402
from mongoengine import connect, disconnect  # noqa: E402

from sample import Sample  # noqa: E402


def setup_module() -> None:
    disconnect()
    connect("testdb", mongo_client_class=mongomock.MongoClient, alias="default")


def teardown_module() -> None:
    disconnect()


def test_creates_new_sample() -> None:
    Sample.drop_collection()
    sample = Sample.get_or_create("S1")
    assert sample.name == "S1"
    assert Sample.objects(name="S1").count() == 1


def test_returns_existing_sample() -> None:
    Sample.drop_collection()
    existing = Sample(name="S2").save()
    fetched = Sample.get_or_create("S2")
    assert fetched.id == existing.id
    assert Sample.objects(name="S2").count() == 1
