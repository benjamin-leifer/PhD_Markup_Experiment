import os
import sys
import io
import pickle

import mongomock
from mongomock.gridfs import enable_gridfs_integration
enable_gridfs_integration()
from mongoengine import connect, disconnect

# Ensure package root on path
ROOT = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(ROOT, "..", "Python_Codes", "BLeifer_Battery_Analysis"))
for path in (ROOT, PACKAGE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from battery_analysis import models  # noqa: E402
from battery_analysis.utils import data_update  # noqa: E402


def _store_cycle_detail(test_result, cycle_index, charge_cap, discharge_cap):
    detail = models.CycleDetailData(test_result=test_result, cycle_index=cycle_index)
    charge_bytes = io.BytesIO()
    pickle.dump({"capacity": [0, charge_cap]}, charge_bytes)
    charge_bytes.seek(0)
    detail.charge_data.put(charge_bytes, content_type="application/python-pickle")

    discharge_bytes = io.BytesIO()
    pickle.dump({"capacity": [0, discharge_cap]}, discharge_bytes)
    discharge_bytes.seek(0)
    detail.discharge_data.put(discharge_bytes, content_type="application/python-pickle")

    detail.save()


def test_backfill_cycle_summaries():
    disconnect()
    connect("testdb", mongo_client_class=mongomock.MongoClient)
    try:
        sample = models.Sample(name="S1").save()
        test = models.TestResult(sample=sample, tester="Arbin", name="t1").save()
        sample.update(push__tests=test.id)

        _store_cycle_detail(test, 1, 1.0, 0.8)

        updated = data_update.backfill_cycle_summaries([test.id])
        assert updated == 1

        test.reload()
        assert len(test.cycles) == 1
        assert abs(test.cycles[0].discharge_capacity - 0.8) < 1e-6
    finally:
        disconnect()
