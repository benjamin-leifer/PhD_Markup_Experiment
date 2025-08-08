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
import mongomock  # noqa: E402
from mongoengine import connect, disconnect  # noqa: E402


def _cycle(idx: int) -> CycleSummary:
    return CycleSummary(
        cycle_index=idx,
        charge_capacity=1.0,
        discharge_capacity=0.9,
        coulombic_efficiency=0.9,
    )


def test_build_and_append_dataset():
    connect("testdb", mongo_client_class=mongomock.MongoClient, alias="default")
    sample = Sample(name="S1").save()
    tr1 = TestResult(
        sample=sample,
        tester="Arbin",
        name="t1",
        cell_code="CN1",
        cycles=[_cycle(1)],
    ).save()
    tr2 = TestResult(
        sample=sample,
        tester="Arbin",
        name="t2",
        cell_code="CN1",
        cycles=[_cycle(2)],
    ).save()

    dataset = CellDataset.build_from_tests([tr1])
    sample.reload()
    assert dataset.cell_code == "CN1"
    assert dataset.sample == sample
    assert dataset.tests == [tr1]
    assert len(dataset.combined_cycles) == 1
    assert sample.default_dataset == dataset

    dataset.append_test(tr2)
    assert dataset.tests == [tr1, tr2]
    assert len(dataset.combined_cycles) == 2
    disconnect()
