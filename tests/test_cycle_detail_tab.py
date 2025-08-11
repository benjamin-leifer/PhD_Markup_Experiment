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
from battery_analysis.models import Sample, TestResult, CycleDetailData
from battery_analysis.utils.detailed_data_manager import get_detailed_cycle_data


def _setup_db():
    enable_gridfs_integration()
    connect(
        "battery_test_db",
        host="mongodb://localhost",
        mongo_client_class=mongomock.MongoClient,
    )


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

    # get_detailed_cycle_data ignores the entry because test.cycles is empty
    assert get_detailed_cycle_data(str(test.id)) == {}

    # our helper should still find the available cycle
    assert _get_cycle_indices(str(test.id)) == [1]
