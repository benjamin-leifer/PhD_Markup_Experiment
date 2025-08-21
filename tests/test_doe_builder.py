from __future__ import annotations

import importlib.util
import types
import json
from pathlib import Path
from typing import Any, Iterable
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pkg = types.ModuleType("battery_analysis")
pkg.__path__ = [str(PACKAGE_ROOT / "battery_analysis")]
sys.modules.setdefault("battery_analysis", pkg)
sys.modules.setdefault(
    "battery_analysis.utils", types.ModuleType("battery_analysis.utils")
)
models_pkg = types.ModuleType("battery_analysis.models")
models_pkg.__path__ = [str(PACKAGE_ROOT / "battery_analysis" / "models")]
sys.modules.setdefault("battery_analysis.models", models_pkg)

experiment_plan_module = load_module(
    "battery_analysis.models.experiment_plan",
    PACKAGE_ROOT / "battery_analysis" / "models" / "experiment_plan.py",
)
setattr(
    sys.modules["battery_analysis.models"],
    "ExperimentPlan",
    experiment_plan_module.ExperimentPlan,
)
ExperimentPlan = experiment_plan_module.ExperimentPlan
doe_builder = load_module(
    "battery_analysis.utils.doe_builder",
    PACKAGE_ROOT / "battery_analysis" / "utils" / "doe_builder.py",
)


def test_load_from_files(tmp_path: Path) -> None:
    csv_file = tmp_path / "plan.csv"
    csv_file.write_text("A,B\n1,2\n1,3\n", encoding="utf-8")
    json_file = tmp_path / "plan.json"
    json_file.write_text(
        json.dumps([{"A": "1", "B": "2"}, {"A": "1", "B": "3"}]), encoding="utf-8"
    )

    f_csv, m_csv = doe_builder.load_from_csv(csv_file)
    assert f_csv == {"A": ["1"], "B": ["2", "3"]}
    assert m_csv == [{"A": "1", "B": "2"}, {"A": "1", "B": "3"}]

    f_json, m_json = doe_builder.load_from_json(json_file)
    assert f_json == f_csv
    assert m_json == m_csv


def test_main_import_and_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store: dict[str, Any] = {}

    def fake_get_by_name(cls: Any, name: str) -> Any:
        return store.get(name)

    monkeypatch.setattr(ExperimentPlan, "get_by_name", classmethod(fake_get_by_name))

    original_save_plan = doe_builder.save_plan

    def wrapped_save_plan(
        name: str,
        factors: dict[str, Iterable[Any]],
        matrix: list[dict[str, Any]],
        sample_ids: Iterable[Any] | None = None,
    ) -> Any:
        plan = original_save_plan(name, factors, matrix, sample_ids)
        store[name] = plan
        return plan

    monkeypatch.setattr(doe_builder, "save_plan", wrapped_save_plan)

    csv_file = tmp_path / "plan.csv"
    csv_file.write_text("A,B\n1,2\n1,3\n", encoding="utf-8")
    doe_builder.main(["--name", "demo", "--input", str(csv_file), "--save"])
    assert len(store["demo"].matrix) == 2

    json_file = tmp_path / "plan.json"
    json_file.write_text(
        json.dumps([{"A": "1", "B": "2"}, {"A": "1", "B": "4"}]), encoding="utf-8"
    )
    doe_builder.main(["--name", "demo", "--input", str(json_file), "--save"])

    plan = store["demo"]
    assert len(plan.matrix) == 3
    assert {"A": "1", "B": "4"} in plan.matrix
