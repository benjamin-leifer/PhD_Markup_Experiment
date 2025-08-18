from pathlib import Path
import sys
import datetime
import types

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
MODELS_ROOT = PACKAGE_ROOT / "battery_analysis" / "models"
for path in (ROOT, PACKAGE_ROOT, MODELS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mongomock
from mongoengine import connect, disconnect

from sample import Sample

# ---------------------------------------------------------------------------
# Load TestResult with stubbed dependencies to avoid heavy imports
# ---------------------------------------------------------------------------
battery_analysis_pkg = types.ModuleType("battery_analysis")
utils_pkg = types.ModuleType("battery_analysis.utils")
gridfs_conversion = types.ModuleType("battery_analysis.utils.gridfs_conversion")
gridfs_conversion.data_to_gridfs = lambda *args, **kwargs: None
gridfs_conversion.gridfs_to_data = lambda *args, **kwargs: None
db_mod = types.ModuleType("battery_analysis.utils.db")
db_mod.ensure_connection = lambda: True
models_pkg = types.ModuleType("battery_analysis.models")
stages_mod = types.ModuleType("battery_analysis.models.stages")
stages_mod.inherit_metadata = lambda obj: {}

sys.modules.update(
    {
        "battery_analysis": battery_analysis_pkg,
        "battery_analysis.utils": utils_pkg,
        "battery_analysis.utils.gridfs_conversion": gridfs_conversion,
        "battery_analysis.utils.db": db_mod,
        "battery_analysis.models": models_pkg,
        "battery_analysis.models.stages": stages_mod,
    }
)

import importlib.util

spec = importlib.util.spec_from_file_location(
    "battery_analysis.models.testresult", MODELS_ROOT / "testresult.py"
)
testresult_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = testresult_module
spec.loader.exec_module(testresult_module)
TestResult = testresult_module.TestResult


def setup_module() -> None:
    disconnect()
    connect("testdb", mongo_client_class=mongomock.MongoClient, alias="default")


def teardown_module() -> None:
    disconnect()


def test_sample_add_note_records_timestamp_and_author() -> None:
    Sample.drop_collection()
    sample = Sample(name="S1").save()
    sample.add_note("first note")
    sample.add_note("second note", author="Bob")
    assert len(sample.notes_log) == 2
    first, second = sample.notes_log
    assert first["text"] == "first note"
    assert first["author"] is None
    assert isinstance(first["timestamp"], datetime.datetime)
    assert second["text"] == "second note"
    assert second["author"] == "Bob"
    assert isinstance(second["timestamp"], datetime.datetime)


def test_testresult_add_note_records_timestamp_and_author() -> None:
    Sample.drop_collection()
    TestResult.drop_collection()
    sample = Sample(name="S2").save()
    test_result = TestResult(sample=sample, tester="Arbin").save()
    test_result.add_note("initial analysis", author="Alice")
    test_result.add_note("follow up")
    assert len(test_result.notes_log) == 2
    first, second = test_result.notes_log
    assert first["text"] == "initial analysis"
    assert first["author"] == "Alice"
    assert isinstance(first["timestamp"], datetime.datetime)
    assert second["text"] == "follow up"
    assert second["author"] is None
    assert isinstance(second["timestamp"], datetime.datetime)
