from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Tuple

import mongomock
from mongoengine import connect, disconnect

# mypy: ignore-errors


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = (
    ROOT / "Python_Codes" / "BLeifer_Battery_Analysis" / "battery_analysis"
)  # noqa: E501
UTILS_DIR = PACKAGE_DIR / "utils"

# ---------------------------------------------------------------------------
# Lightweight package stubs to avoid heavy imports from battery_analysis
# ---------------------------------------------------------------------------
package_stub = types.ModuleType("battery_analysis")
package_stub.__path__ = [str(PACKAGE_DIR)]
sys.modules.setdefault("battery_analysis", package_stub)

utils_stub = types.ModuleType("battery_analysis.utils")
utils_stub.__path__ = [str(UTILS_DIR)]
sys.modules.setdefault("battery_analysis.utils", utils_stub)

models = importlib.import_module("battery_analysis.models")
Sample = models.Sample
TestResult = models.TestResult

spec = importlib.util.spec_from_file_location(
    "battery_analysis.utils.search_tests",
    UTILS_DIR / "search_tests.py",
)
search_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search_tests)  # type: ignore[arg-type]


def setup_module() -> None:
    disconnect()
    connect(
        "testdb",
        mongo_client_class=mongomock.MongoClient,
        alias="default",
    )


def teardown_module() -> None:
    disconnect()


def _prepare() -> Tuple[TestResult, TestResult]:
    Sample.drop_collection()
    TestResult.drop_collection()
    s1 = Sample(name="S1", chemistry="NMC").save()
    s2 = Sample(name="S2", chemistry="LFP").save()
    t1 = TestResult(
        sample=s1,
        tester="Arbin",
        initial_capacity=1.0,
        final_capacity=0.9,
        capacity_retention=0.9,
        date=datetime(2024, 1, 15),
    ).save()
    t2 = TestResult(
        sample=s2,
        tester="Arbin",
        initial_capacity=2.0,
        final_capacity=1.8,
        capacity_retention=0.9,
        date=datetime(2024, 2, 1),
    ).save()
    return t1, t2


def test_filter_by_sample(capsys) -> None:
    t1, t2 = _prepare()
    search_tests.main(["--sample", "S1"])
    out = capsys.readouterr().out
    assert str(t1.id) in out
    assert str(t2.id) not in out


def test_filter_by_chemistry_and_date(capsys) -> None:
    t1, t2 = _prepare()
    search_tests.main(["--chemistry", "LFP"])
    out = capsys.readouterr().out
    assert str(t2.id) in out
    assert str(t1.id) not in out

    capsys.readouterr()  # clear
    search_tests.main(["--date-range", "2024-01-01:2024-01-31"])
    out = capsys.readouterr().out
    assert str(t1.id) in out
    assert str(t2.id) not in out
