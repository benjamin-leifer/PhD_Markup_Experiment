from __future__ import annotations

# mypy: ignore-errors

import sys
import types
import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Stub optional SciPy dependency used elsewhere
if "scipy" not in sys.modules:

    class _ScipyStub(types.ModuleType):
        def __getattr__(self, name: str) -> types.ModuleType:  # pragma: no cover - stub
            mod = types.ModuleType(name)
            sys.modules[f"scipy.{name}"] = mod
            setattr(self, name, mod)
            return mod

    sys.modules["scipy"] = _ScipyStub("scipy")

import mongomock  # noqa: E402
from mongoengine import connect, disconnect  # noqa: E402

disconnect()
connect("validation_test", mongo_client_class=mongomock.MongoClient, alias="default")

from battery_analysis.models import Sample  # noqa: E402
from battery_analysis.utils import data_update  # noqa: E402
from battery_analysis import parsers  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db() -> None:
    Sample._registry.clear()
    disconnect()
    connect(
        "validation_test", mongo_client_class=mongomock.MongoClient, alias="default"
    )
    yield
    disconnect()


def test_missing_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    file_path = tmp_path / "test.csv"
    file_path.write_text("dummy")
    sample = Sample.get_or_create("S1")

    def fake_parse(path: str):
        return [], {"tester": "X", "name": "t"}

    monkeypatch.setattr(parsers, "parse_file", fake_parse)

    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError):
            data_update.process_file_with_update(str(file_path), sample)

    assert "Missing required metadata" in caplog.text


def test_invalid_cycle_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    file_path = tmp_path / "test.csv"
    file_path.write_text("dummy")
    sample = Sample.get_or_create("S1")

    def fake_parse(path: str):
        cycles = [{"cycle_index": 1, "charge_capacity": 1.0}]  # missing fields
        meta = {"tester": "X", "name": "t", "date": datetime.datetime.utcnow()}
        return cycles, meta

    monkeypatch.setattr(parsers, "parse_file", fake_parse)

    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError):
            data_update.process_file_with_update(str(file_path), sample)

    assert "Invalid cycle data" in caplog.text
