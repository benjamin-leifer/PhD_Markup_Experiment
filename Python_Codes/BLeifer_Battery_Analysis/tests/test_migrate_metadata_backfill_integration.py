import datetime as dt
import os
import sys

import pytest

mongomock = pytest.importorskip("mongomock")
from mongoengine import connect, disconnect

TESTS_DIR = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from battery_analysis import models
from battery_analysis.utils.migrate_metadata_backfill import migrate_test_results


def test_migrate_test_results_backfills_missing_timestamps_from_date() -> None:
    disconnect()
    connect("testdb", mongo_client_class=mongomock.MongoClient)
    try:
        sample = models.Sample(name="S1").save()
        known_date = dt.datetime(2024, 1, 2, 3, 4, 5)
        test = models.TestResult(
            sample=sample.id,
            tester="Arbin",
            name="run-1",
            date=known_date,
            cycles=[],
        )
        test.created_at = None
        test.updated_at = None
        test.save()

        counts = migrate_test_results(models.TestResult.objects())
        updated = models.TestResult.objects(id=test.id).first()

        assert counts.scanned == 1
        assert counts.matched == 1
        assert counts.changed == 1
        assert updated is not None
        assert updated.created_at == known_date
        assert updated.updated_at == known_date
    finally:
        disconnect()
