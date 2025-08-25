import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard import comparison_tab  # noqa: E402


def test_get_sample_options_db_unavailable(
    monkeypatch: Any, caplog: Any
) -> None:  # noqa: E501
    monkeypatch.setattr(comparison_tab, "db_connected", lambda: False)
    monkeypatch.setattr(comparison_tab, "get_db_error", lambda: "no db")
    with caplog.at_level("ERROR", logger=comparison_tab.logger.name):
        opts, error = comparison_tab._get_sample_options()
    assert opts == [{"label": "Sample_001", "value": "sample1"}]
    assert error == "Database not connected; using demo data (no db)"
    assert "Database not connected: no db; using demo data" in caplog.text


def test_get_sample_options_db_available(
    monkeypatch: Any, caplog: Any
) -> None:  # noqa: E501
    monkeypatch.setattr(comparison_tab, "db_connected", lambda: True)
    samples = [
        {"_id": "id1", "name": "SampleA"},
        {"_id": "id2", "name": "SampleB"},
    ]
    monkeypatch.setattr(comparison_tab, "find_samples", lambda: samples)
    with caplog.at_level("WARNING", logger=comparison_tab.logger.name):
        opts, error = comparison_tab._get_sample_options()
    assert opts == [
        {"label": "SampleA", "value": "id1"},
        {"label": "SampleB", "value": "id2"},
    ]
    assert error is None
    assert caplog.text == ""


def test_get_sample_data_pymongo(monkeypatch: Any) -> None:
    """Sample data retrieval using raw pymongo helpers."""

    # Fake ``battery_analysis`` package with a ``Sample`` lacking ``objects``
    fake_pkg = types.ModuleType("battery_analysis")
    fake_models = types.ModuleType("battery_analysis.models")

    @dataclass
    class Sample:
        id: str
        name: str

    fake_models.Sample = Sample
    fake_models.TestResult = object
    fake_pkg.models = fake_models
    monkeypatch.setitem(sys.modules, "battery_analysis", fake_pkg)
    monkeypatch.setitem(sys.modules, "battery_analysis.models", fake_models)

    sample_doc = {"_id": "s1", "name": "PymongoSample"}
    cycle_doc = {
        "cycle_index": 1,
        "discharge_capacity": 1.5,
        "coulombic_efficiency": 0.95,
    }
    monkeypatch.setattr(comparison_tab, "find_samples", lambda q: [sample_doc])
    monkeypatch.setattr(
        comparison_tab, "find_test_results", lambda q: [{"cycle_summaries": [cycle_doc]}]
    )

    name, data, err = comparison_tab._get_sample_data("s1")
    assert err is None
    assert name == "PymongoSample"
    assert data["cycle"].tolist() == [1]
    assert data["capacity"].tolist() == [1.5]
    assert data["ce"].tolist() == [0.95]
    assert np.isnan(data["impedance"]).all()


def test_get_sample_data_mongoengine(monkeypatch: Any) -> None:
    """Sample data retrieval using mongoengine-style ``objects``."""

    fake_pkg = types.ModuleType("battery_analysis")
    fake_models = types.ModuleType("battery_analysis.models")

    @dataclass
    class Cycle:
        cycle_index: int = 1
        discharge_capacity: float = 2.0
        coulombic_efficiency: float = 0.9

    @dataclass
    class Dataset:
        combined_cycles: list[Cycle]

    @dataclass
    class Sample:
        id: str
        name: str
        default_dataset: Dataset | None = None

    sample_obj = Sample("m1", "MongoSample", Dataset([Cycle()]))

    class Objects:
        def __call__(self, **kwargs: Any) -> "Objects":
            return self

        def first(self) -> Sample:
            return sample_obj

    Sample.objects = Objects()

    fake_models.Sample = Sample
    fake_models.TestResult = object
    fake_pkg.models = fake_models
    monkeypatch.setitem(sys.modules, "battery_analysis", fake_pkg)
    monkeypatch.setitem(sys.modules, "battery_analysis.models", fake_models)

    name, data, err = comparison_tab._get_sample_data("m1")
    assert err is None
    assert name == "MongoSample"
    assert data["cycle"].tolist() == [1]
    assert data["capacity"].tolist() == [2.0]
    assert data["ce"].tolist() == [0.9]
    assert np.isnan(data["impedance"]).all()
