import sys
import types
from dataclasses import dataclass, field
from typing import Any, List

import plotly.graph_objects as go
import pytest
from pathlib import Path
import dash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard import cycle_detail_tab

SAMPLES = [{"_id": "s1", "name": "SampleA"}]
TEST_DOC = {
    "_id": "t1",
    "name": "TestA",
    "cycles": [
        {"cycle_index": 1, "charge_capacity": 1.0, "discharge_capacity": 1.0},
        {"cycle_index": 2, "charge_capacity": 1.0, "discharge_capacity": 0.0},
    ],
}
CYCLE_DATA = {
    1: {
        "charge": {"voltage": [1, 2], "capacity": [0.1, 0.2]},
        "discharge": {"voltage": [2, 1], "capacity": [0.2, 0.1]},
    }
}


def _patch_models(monkeypatch: Any, with_manager: bool) -> None:
    """Install minimal dataclass-based models."""
    fake_pkg = types.ModuleType("battery_analysis")
    fake_models = types.ModuleType("battery_analysis.models")

    @dataclass
    class Sample:
        id: str
        name: str
        default_dataset: Any = None

    @dataclass
    class CycleSummary:
        cycle_index: int
        charge_capacity: float
        discharge_capacity: float

    @dataclass
    class TestResult:
        id: str
        sample: str
        name: str
        cycles: List[CycleSummary] = field(default_factory=list)
        test_type: str = ""

    fake_models.Sample = Sample
    fake_models.TestResult = TestResult
    fake_models.CycleSummary = CycleSummary
    fake_pkg.models = fake_models

    monkeypatch.setitem(sys.modules, "battery_analysis", fake_pkg)
    monkeypatch.setitem(sys.modules, "battery_analysis.models", fake_models)

    if with_manager:
        class Manager(list):
            def __call__(self, **query):
                filtered = [
                    o for o in self if all(getattr(o, k) == v for k, v in query.items())
                ]
                return Manager(filtered)

            def only(self, *_fields):
                return self

            def first(self):
                return self[0] if self else None

        Sample.objects = Manager([Sample(id="s1", name="SampleA")])
        cycles = [CycleSummary(1, 1.0, 1.0)]
        TestResult.objects = Manager(
            [TestResult(id="t1", sample="s1", name="TestA", cycles=cycles)]
        )


@pytest.mark.parametrize("with_manager", [False, True])
def test_cycle_detail_paths(monkeypatch: Any, with_manager: bool) -> None:
    _patch_models(monkeypatch, with_manager)

    monkeypatch.setattr(cycle_detail_tab, "db_connected", lambda: True)
    monkeypatch.setattr(cycle_detail_tab, "get_cell_dataset", None)
    monkeypatch.setattr(cycle_detail_tab, "find_samples", lambda: SAMPLES)
    monkeypatch.setattr(cycle_detail_tab, "find_test_results", lambda q: [TEST_DOC])
    monkeypatch.setattr(
        cycle_detail_tab,
        "get_detailed_cycle_data",
        lambda test_id, cycle_index=None: CYCLE_DATA,
    )

    test_opts = cycle_detail_tab._get_test_options("s1")
    assert test_opts == [{"label": "TestA", "value": "t1"}]

    indices = cycle_detail_tab._get_cycle_indices("t1")
    assert indices == [1]
    app = dash.Dash(__name__)
    cycle_detail_tab.register_callbacks(app)
    key = next(k for k in app.callback_map if k.startswith(f"..{cycle_detail_tab.GRAPH}.figure"))
    cb = app.callback_map[key]["callback"].__wrapped__
    fig, modal = cb("t1", 1)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert modal.to_dict() == fig.to_dict()
