from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import types  # noqa: E402

sys.modules.setdefault("advanced_analysis", types.ModuleType("advanced_analysis"))
sys.modules.setdefault("battery_analysis.analysis", types.ModuleType("analysis"))
sys.modules.setdefault("analysis", sys.modules["battery_analysis.analysis"])

from battery_analysis.models.sample import Sample  # noqa: E402
from battery_analysis.models.testresult import TestResult  # noqa: E402
from battery_analysis.models.cycle_summary import CycleSummary  # noqa: E402
from battery_analysis.models.cell_dataset import CellDataset  # noqa: E402
from battery_analysis.utils.cell_dataset_builder import (  # noqa: E402
    gather_tests,
    merge_tests,
    update_cell_dataset,
    rollback,
)
import mongomock  # noqa: E402
from mongoengine import connect, disconnect  # noqa: E402


def _cycle(idx: int) -> CycleSummary:
    return CycleSummary(
        cycle_index=idx,
        charge_capacity=1.0,
        discharge_capacity=0.9,
        coulombic_efficiency=0.9,
    )


def test_builder_flow() -> None:
    connect("builder_db", mongo_client_class=mongomock.MongoClient, alias="default")
    sample = Sample(name="S1").save()
    t1 = TestResult(
        sample=sample,
        tester="Arbin",
        name="t1",
        cell_code="CN1",
        date=datetime.utcnow() - timedelta(days=1),
        cycles=[_cycle(1)],
    ).save()
    t2 = TestResult(
        sample=sample,
        tester="Arbin",
        name="t2",
        cell_code="CN1",
        date=datetime.utcnow(),
        cycles=[_cycle(1)],
    ).save()

    tests = gather_tests("CN1")
    assert tests == [t1, t2]

    combined = merge_tests(tests)
    assert [c.cycle_index for c in combined] == [1, 2]

    dataset1 = update_cell_dataset("CN1")
    assert isinstance(dataset1, CellDataset)
    assert len(dataset1.tests) == 2
    assert [c.cycle_index for c in dataset1.combined_cycles] == [1, 2]
    assert dataset1.version == 1
    assert dataset1.previous_id is None

    TestResult(
        sample=sample,
        tester="Arbin",
        name="t3",
        cell_code="CN1",
        date=datetime.utcnow() + timedelta(days=1),
        cycles=[_cycle(1)],
    ).save()
    dataset2 = update_cell_dataset("CN1")
    assert len(dataset2.tests) == 3
    assert [c.cycle_index for c in dataset2.combined_cycles] == [1, 2, 3]
    assert dataset2.version == 2
    assert dataset2.previous_id == dataset1.id

    disconnect()


def test_rollback() -> None:
    connect("rollback_db", mongo_client_class=mongomock.MongoClient, alias="default")
    sample = Sample(name="S1").save()
    TestResult(
        sample=sample,
        tester="Arbin",
        name="t1",
        cell_code="CN1",
        date=datetime.utcnow() - timedelta(days=1),
        cycles=[_cycle(1)],
    ).save()
    ds1 = update_cell_dataset("CN1")
    TestResult(
        sample=sample,
        tester="Arbin",
        name="t2",
        cell_code="CN1",
        date=datetime.utcnow(),
        cycles=[_cycle(1)],
    ).save()
    ds2 = update_cell_dataset("CN1")
    assert ds2.version == 2
    assert ds2.previous_id == ds1.id

    rolled = rollback("CN1", 1)
    assert rolled is not None
    assert rolled.version == 3
    assert rolled.previous_id == ds2.id
    assert len(rolled.tests) == 1

    disconnect()
