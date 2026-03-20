from __future__ import annotations

import datetime as dt
import time

from battery_analysis.models import Sample, TestResult
from battery_analysis.utils.migrate_metadata_backfill import (
    FIXED_EPOCH,
    derive_testresult_timestamp_updates,
)


class _ObjectIdStub:
    def __init__(self, generation_time: dt.datetime):
        self.generation_time = generation_time


def test_timestamps_are_set_and_updated():
    sample = Sample(name="S1")
    test = TestResult(sample=sample, tester="Arbin")
    assert test.created_at is not None
    assert test.updated_at is not None

    first_created = test.created_at
    first_updated = test.updated_at

    time.sleep(0.01)
    test.clean()
    assert test.created_at == first_created
    assert test.updated_at > first_updated


def test_timestamp_backfill_prefers_test_date_and_is_deterministic():
    sample = Sample(name="S2")
    known_date = dt.datetime(2024, 5, 1, 12, 30, 0)
    test = TestResult(sample=sample, tester="Arbin")
    test.created_at = None
    test.updated_at = None
    test.date = known_date
    test.id = _ObjectIdStub(dt.datetime(2023, 1, 1, 0, 0, 0))

    updates = derive_testresult_timestamp_updates(test)

    assert updates == {"created_at": known_date, "updated_at": known_date}
    assert derive_testresult_timestamp_updates(test) == updates


def test_timestamp_backfill_falls_back_to_object_id_then_epoch():
    sample = Sample(name="S3")
    object_id_time = dt.datetime(2022, 7, 8, 9, 10, 11)
    test = TestResult(sample=sample, tester="Arbin")
    test.created_at = None
    test.updated_at = None
    test.date = None
    test.metadata = {}
    test.id = _ObjectIdStub(object_id_time)

    assert derive_testresult_timestamp_updates(test) == {
        "created_at": object_id_time,
        "updated_at": object_id_time,
    }

    test.id = None
    assert derive_testresult_timestamp_updates(test) == {
        "created_at": FIXED_EPOCH,
        "updated_at": FIXED_EPOCH,
    }
