import logging
import os
import sys

import pytest

# Ensure package root on path
TESTS_DIR = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from mongoengine import connect, disconnect  # noqa: E402
import mongomock  # noqa: E402

from battery_analysis import models  # noqa: E402
from battery_analysis.utils import data_update  # noqa: E402


def test_sequential_file_names_match() -> None:
    """Sequential files with suffixes like _Wb_1 should map to the same test."""
    disconnect()
    connect("testdb", mongo_client_class=mongomock.MongoClient)
    try:
        sample = models.Sample(name="S1")
        sample.save()
        test = models.TestResult(
            sample=sample.id,
            tester="Arbin",
            name="BL-LL-EU02_RT_Rate_Test_Channel_30",
            file_path="/tmp/BL-LL-EU02_RT_Rate_Test_Channel_30_Wb_1.xlsx",
            cycles=[],
        )
        test.save()
        sample.update(push__tests=test.id)
        sample.reload()

        metadata = {"name": "BL-LL-EU02_RT_Rate_Test_Channel_30", "tester": "Arbin"}
        identifiers = data_update.extract_test_identifiers(
            "/tmp/BL-LL-EU02_RT_Rate_Test_Channel_30_Wb_2.xlsx",
            [],
            metadata,
        )
        matches = data_update.find_matching_tests(identifiers, sample.id)
        assert matches and matches[0].id == test.id
    finally:
        disconnect()


def _make_cycle(idx: int) -> dict[str, float | int]:
    return {
        "cycle_index": idx,
        "charge_capacity": 1.0,
        "discharge_capacity": 1.0,
        "coulombic_efficiency": 1.0,
    }


def test_duplicate_cycle_indices_warn(caplog: pytest.LogCaptureFixture) -> None:
    disconnect()
    connect("testdb", mongo_client_class=mongomock.MongoClient)
    try:
        sample = models.Sample(name="S1")
        sample.save()
        test = models.TestResult(
            sample=sample.id,
            tester="Arbin",
            name="T1",
            file_path="/tmp/t1.xlsx",
            cycles=[
                models.CycleSummary(
                    cycle_index=1,
                    charge_capacity=1.0,
                    discharge_capacity=1.0,
                    coulombic_efficiency=1.0,
                )
            ],
        )
        test.save()
        sample.update(push__tests=test.id)
        sample.reload()

        new_cycles = [_make_cycle(1), _make_cycle(2)]
        with caplog.at_level(logging.WARNING):
            data_update.update_test_data(test, new_cycles, metadata=None)
        assert any("already exists" in rec.message for rec in caplog.records)
        assert [c.cycle_index for c in test.cycles] == [1, 2]
    finally:
        disconnect()


def test_missing_cycle_index_error(caplog: pytest.LogCaptureFixture) -> None:
    disconnect()
    connect("testdb", mongo_client_class=mongomock.MongoClient)
    try:
        sample = models.Sample(name="S1")
        sample.save()
        test = models.TestResult(
            sample=sample.id,
            tester="Arbin",
            name="T1",
            file_path="/tmp/t1.xlsx",
            cycles=[
                models.CycleSummary(
                    cycle_index=1,
                    charge_capacity=1.0,
                    discharge_capacity=1.0,
                    coulombic_efficiency=1.0,
                )
            ],
        )
        test.save()
        sample.update(push__tests=test.id)
        sample.reload()

        new_cycles = [_make_cycle(3)]
        with caplog.at_level(logging.ERROR):
            data_update.update_test_data(test, new_cycles, metadata=None)
        assert any("not strictly increasing" in rec.message for rec in caplog.records)
    finally:
        disconnect()
