from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
MODELS_ROOT = PACKAGE_ROOT / "battery_analysis" / "models"
for path in (ROOT, PACKAGE_ROOT, MODELS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mongomock  # noqa: E402
import pytest  # noqa: E402
from mongoengine import connect, disconnect, ValidationError  # noqa: E402

from sample import Sample  # noqa: E402


def setup_module() -> None:
    disconnect()
    connect("testdb", mongo_client_class=mongomock.MongoClient, alias="default")


def teardown_module() -> None:
    disconnect()


def test_self_reference_rejected() -> None:
    Sample.drop_collection()
    sample = Sample(name="S1").save()
    sample.anode = sample
    with pytest.raises(ValidationError, match="cannot reference self"):
        sample.clean()


def test_unsaved_component_reference() -> None:
    Sample.drop_collection()
    s1 = Sample(name="S1")
    s2 = Sample(name="S2")  # Unsaved component
    s1.anode = s2
    with pytest.raises(ValidationError, match="saved Sample"):
        s1.clean()
