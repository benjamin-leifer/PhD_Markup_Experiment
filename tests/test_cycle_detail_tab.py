import io
import pickle
import sys
from pathlib import Path

import mongomock
from mongomock.gridfs import enable_gridfs_integration
from mongoengine import connect

# Ensure the dashboard package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.cycle_detail_tab import _get_cycle_indices
from battery_analysis.models import Sample, TestResult, CycleDetailData, CycleSummary
from battery_analysis.utils.detailed_data_manager import get_detailed_cycle_data


def _setup_db():
    enable_gridfs_integration()
    connect(
        "battery_test_db",
        host="mongodb://localhost",
        mongo_client_class=mongomock.MongoClient,
    )
    Sample.drop_collection()
    TestResult.drop_collection()
    CycleDetailData.drop_collection()


def test_cycle_indices_from_gridfs():
    _setup_db()
    sample = Sample(name="S1").save()
    test = TestResult(name="T1", sample=sample, tester="Other").save()

    # Store a cycle detail entry without touching TestResult.cycles
    detail = CycleDetailData(test_result=test, cycle_index=1)
    charge = io.BytesIO()
    pickle.dump({"voltage": [1, 2], "capacity": [0.1, 0.2]}, charge)
    charge.seek(0)
    detail.charge_data.put(
        charge, content_type="application/python-pickle", filename="c.pkl"
    )
    discharge = io.BytesIO()
    pickle.dump({"voltage": [2, 1], "capacity": [0.2, 0.1]}, discharge)
    discharge.seek(0)
    detail.discharge_data.put(
        discharge, content_type="application/python-pickle", filename="d.pkl"
    )
    detail.save()

    # get_detailed_cycle_data now includes the cycle even without summaries
    assert list(get_detailed_cycle_data(str(test.id)).keys()) == [1]

    # our helper should still find the available cycle
    assert _get_cycle_indices(str(test.id)) == [1]


def test_exclude_incomplete_cycles():
    _setup_db()
    sample = Sample(name="S1").save()
    cycles = [
        CycleSummary(cycle_index=1, charge_capacity=1.0, discharge_capacity=1.0, coulombic_efficiency=1.0),
        CycleSummary(cycle_index=2, charge_capacity=1.0, discharge_capacity=0.0, coulombic_efficiency=0.0),
    ]
    test = TestResult(name="T1", sample=sample, tester="Other", cycles=cycles).save()
    for idx in (1, 2):
        detail = CycleDetailData(test_result=test, cycle_index=idx)
        ch = io.BytesIO()
        pickle.dump({"voltage": [1, 2], "capacity": [0.1, 0.2]}, ch)
        ch.seek(0)
        detail.charge_data.put(ch, content_type="application/python-pickle", filename=f"c{idx}.pkl")
        dis = io.BytesIO()
        pickle.dump({"voltage": [2, 1], "capacity": [0.2, 0.1]}, dis)
        dis.seek(0)
        detail.discharge_data.put(dis, content_type="application/python-pickle", filename=f"d{idx}.pkl")
        detail.save()

    assert _get_cycle_indices(str(test.id)) == [1]
    data = get_detailed_cycle_data(str(test.id))
    assert list(data.keys()) == [1]
