from __future__ import annotations

import importlib.util
import types
import json
from pathlib import Path
from typing import Any, Iterable
from dataclasses import dataclass
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

# Provide a minimal ``mongoengine`` stub so model definitions import without the
# real dependency.
mongoengine_stub = types.ModuleType("mongoengine")
mongoengine_stub.Document = type("Document", (), {})
mongoengine_stub.fields = types.SimpleNamespace(
    StringField=lambda *a, **k: None,
    DictField=lambda *a, **k: None,
    ListField=lambda *a, **k: None,
    ReferenceField=lambda *a, **k: None,
    DateTimeField=lambda *a, **k: None,
)
sys.modules["mongoengine"] = mongoengine_stub

@dataclass
class ExperimentPlan:
    name: str
    factors: dict[str, Any]
    matrix: list[dict[str, Any]]
    sample_ids: list[Any]

    @classmethod
    def get_by_name(cls, name: str) -> Any:
        return None

    def save(self) -> None:  # pragma: no cover - placeholder
        return None


setattr(sys.modules["battery_analysis.models"], "ExperimentPlan", ExperimentPlan)

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


def test_progress_exports(tmp_path: Path) -> None:
    factors = {"A": [1, 2]}
    plan = doe_builder.save_plan("progress", factors)
    plan.matrix[0].setdefault("tests", []).append({"id": "t1"})
    csv_path = tmp_path / "progress.csv"
    html_path = tmp_path / "progress.html"
    doe_builder.export_progress_csv(plan, csv_path)
    doe_builder.export_progress_html(plan, html_path)
    csv_lines = csv_path.read_text().splitlines()
    header = csv_lines[0].split(",")
    assert "completed" in header
    idx = header.index("completed")
    row_true = csv_lines[1].split(",")[idx]
    row_false = csv_lines[2].split(",")[idx]
    assert row_true == "True"
    assert row_false == "False"
    html_text = html_path.read_text()
    assert "doe-progress-table" in html_text
    assert "True" in html_text and "False" in html_text
