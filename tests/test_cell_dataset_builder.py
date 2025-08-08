from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from battery_analysis.models import (  # noqa: E402
    Sample,
    TestResult,
    CycleSummary,
    CellDataset,
)
from battery_analysis.utils.cell_dataset_builder import (  # noqa: E402
    gather_tests,
    merge_tests,
    update_cell_dataset,
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


def test_builder_flow():
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

    dataset = update_cell_dataset("CN1")
    assert isinstance(dataset, CellDataset)
    assert len(dataset.tests) == 2
    assert [c.cycle_index for c in dataset.combined_cycles] == [1, 2]

    TestResult(
        sample=sample,
        tester="Arbin",
        name="t3",
        cell_code="CN1",
        date=datetime.utcnow() + timedelta(days=1),
        cycles=[_cycle(1)],
    ).save()
    dataset = update_cell_dataset("CN1")
    assert len(dataset.tests) == 3
    assert [c.cycle_index for c in dataset.combined_cycles] == [1, 2, 3]

    disconnect()
