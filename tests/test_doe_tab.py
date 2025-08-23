from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import importlib.util
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Minimal battery_analysis package structure so dashboard modules import
utils_spec = importlib.util.spec_from_file_location(
    "battery_analysis.utils.doe_builder",
    PACKAGE_ROOT / "battery_analysis" / "utils" / "doe_builder.py",
)
assert utils_spec and utils_spec.loader
utils_mod = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(utils_mod)
utils_pkg = types.ModuleType("battery_analysis.utils")
utils_pkg.__path__ = [str(PACKAGE_ROOT / "battery_analysis" / "utils")]
utils_pkg.doe_builder = utils_mod
utils_pkg.import_watcher = types.ModuleType("import_watcher")
sys.modules.setdefault("battery_analysis.utils", utils_pkg)
sys.modules.setdefault("battery_analysis.utils.doe_builder", utils_mod)
sys.modules.setdefault("battery_analysis.utils.import_watcher", utils_pkg.import_watcher)

sys.modules.setdefault("battery_analysis.models", types.ModuleType("models"))

import dashboard.doe_tab as doe_tab  # noqa: E402

def test_auto_linking(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclass
    class Plan:
        name: str = "P"
        factors: dict[str, Any] = None
        matrix: list[dict[str, Any]] = None

        def __post_init__(self) -> None:
            if self.factors is None:
                self.factors = {"tag": ["r"]}
            if self.matrix is None:
                self.matrix = [{"tag": "r"}]

        def save(self) -> None:  # pragma: no cover - placeholder
            return None

    class PlanManager:
        def __init__(self) -> None:
            self.plan = Plan()

        def __iter__(self):  # pragma: no cover - simple iteration
            return iter([self.plan])

        def only(self, *a: Any, **k: Any) -> "PlanManager":
            return self

    @dataclass
    class TestResult:
        metadata: dict[str, Any]
        id: str = "t1"

    class TestManager:
        def __init__(self) -> None:
            self.tests = [TestResult({"tag": "r"})]

        def __iter__(self):  # pragma: no cover - simple iteration
            return iter(self.tests)

        def only(self, *a: Any, **k: Any) -> "TestManager":
            return self

    models = types.SimpleNamespace(
        ExperimentPlan=types.SimpleNamespace(objects=PlanManager()),
        TestResult=types.SimpleNamespace(objects=TestManager()),
    )
    monkeypatch.setitem(sys.modules, "battery_analysis", types.SimpleNamespace(models=models))

    linked: list[dict[str, Any]] = []

    def fake_link(test: TestResult, metadata: dict[str, Any]) -> None:
        plan = next(iter(models.ExperimentPlan.objects))
        plan.matrix[0].setdefault("tests", []).append({"id": test.id})
        linked.append(metadata)

    monkeypatch.setattr(
        doe_tab,
        "doe_builder",
        types.SimpleNamespace(link_test_to_plan=fake_link),
    )

    plans = doe_tab._load_plans()
    assert linked and plans[0]["matrix"][0]["tests"]
