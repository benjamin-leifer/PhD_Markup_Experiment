"""Regression tests for the Missing Data dashboard tab."""

# flake8: noqa
# mypy: ignore-errors

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import dash

ROOT = Path(__file__).resolve().parents[1]


def test_incomplete_sample_appears_in_table(monkeypatch):
    """A test result lacking sample components is listed in the table."""

    class _Sample:
        anode = cathode = separator = electrolyte = None

    class _Test:
        id = "T1"
        cell_code = "C1"
        sample = _Sample()
        name = "file1"

    class _Query:
        def only(self, *fields):
            return [_Test()]

    class _TestResult:
        @staticmethod
        def _get_collection():
            class _Coll:
                def create_index(self, _field):
                    pass

            return _Coll()

        @staticmethod
        def objects(__raw__):
            return _Query()

    models = types.SimpleNamespace(TestResult=_TestResult)
    fake_ba = types.ModuleType("battery_analysis")
    fake_ba.models = models
    monkeypatch.setitem(sys.modules, "battery_analysis", fake_ba)
    monkeypatch.setitem(sys.modules, "battery_analysis.models", models)

    spec = importlib.util.spec_from_file_location(
        "missing_data_tab", ROOT / "dashboard" / "missing_data_tab.py"
    )
    missing_data_tab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(missing_data_tab)

    app = dash.Dash(__name__)
    app.layout = missing_data_tab.layout()
    missing_data_tab.register_callbacks(app)

    records = missing_data_tab._get_missing_data()
    assert records == [
        {
            "test_id": "T1",
            "cell_code": "C1",
            "missing": ["anode", "cathode", "separator", "electrolyte"],
        }
    ]

    render_table = app.callback_map["missing-data-table.data"]["callback"].__wrapped__
    rows = render_table(records)
    assert rows == [
        {
            "test_id": "T1",
            "cell_code": "C1",
            "missing": "anode, cathode, separator, electrolyte",
            "resolve": "[Resolve](#)",
        }
    ]
