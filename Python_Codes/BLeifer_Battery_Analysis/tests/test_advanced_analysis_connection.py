"""Tests for MongoEngine connection handling in ``get_voltage_capacity_data``."""

import os
import sys

import mongomock
import pytest
from mongoengine import connect, disconnect
from mongoengine.connection import get_connection

# Ensure package root is on sys.path so ``battery_analysis`` can be imported
TESTS_DIR = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PACKAGE_ROOT not in sys.path:  # pragma: no cover - defensive
    sys.path.insert(0, PACKAGE_ROOT)

from battery_analysis import advanced_analysis


def _patch_testresult_objects(monkeypatch):
    """Patch ``TestResult.objects`` to avoid real database access."""

    class DummyQuery:
        def first(self):
            return None

    monkeypatch.setattr(
        advanced_analysis.models.TestResult,
        "objects",
        lambda *args, **kwargs: DummyQuery(),
    )


def test_get_voltage_capacity_data_with_connection(monkeypatch):
    """Function should work when a connection is already established."""

    connect("testdb", alias="default", mongo_client_class=mongomock.MongoClient)
    _patch_testresult_objects(monkeypatch)

    with pytest.raises(ValueError, match="Test with ID"):
        advanced_analysis.get_voltage_capacity_data("abc")

    disconnect(alias="default")


def test_get_voltage_capacity_data_without_connection(monkeypatch):
    """Function should create a connection when none exists."""

    # Ensure no existing connection
    disconnect(alias="default")

    # Patch connect to use mongomock when invoked by ``advanced_analysis``
    import mongoengine

    original_connect = mongoengine.connect

    def _mocked_connect(db, **kwargs):
        kwargs.setdefault("alias", "default")
        kwargs.setdefault("mongo_client_class", mongomock.MongoClient)
        return original_connect(db, **kwargs)

    monkeypatch.setattr(mongoengine, "connect", _mocked_connect)
    os.environ["BATTERY_DB_NAME"] = "testdb"

    _patch_testresult_objects(monkeypatch)

    with pytest.raises(ValueError, match="Test with ID"):
        advanced_analysis.get_voltage_capacity_data("abc")

    # Connection should now exist thanks to automatic connect
    get_connection(alias="default")

    disconnect(alias="default")

