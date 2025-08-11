import sys
from pathlib import Path

import mongomock
import pytest
from mongoengine import connect, disconnect

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from battery_analysis import advanced_analysis, models  # noqa: E402


def _create_test_result():
    sample = models.Sample(name="S").save()
    cycles = [
        models.CycleSummary(
            cycle_index=i,
            charge_capacity=100.0,
            discharge_capacity=100 - i,
            coulombic_efficiency=1.0,
        )
        for i in range(10)
    ]
    test = models.TestResult(
        sample=sample,
        tester="Arbin",
        cycles=cycles,
        cycle_count=len(cycles),
    ).save()
    return test


def test_custom_eol_and_models():
    disconnect()
    connect("testdb", mongo_client_class=mongomock.MongoClient, alias="default")
    try:
        test = _create_test_result()
        result = advanced_analysis.capacity_fade_analysis(
            str(test.id), eol_percent=70, models=["linear"]
        )
        assert set(result["fade_models"].keys()) == {"linear"}
        assert result["predicted_eol_cycle"] == pytest.approx(30)
    finally:
        disconnect()

