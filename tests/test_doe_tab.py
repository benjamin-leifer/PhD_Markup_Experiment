from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Callable, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Stub out heavy optional dependencies
if "scipy" not in sys.modules:

    class _ScipyStub(types.ModuleType):
        def __getattr__(self, name: str) -> types.ModuleType:
            # pragma: no cover
            mod = types.ModuleType(name)
            sys.modules[f"scipy.{name}"] = mod
            setattr(self, name, mod)
            return mod

    sys.modules["scipy"] = _ScipyStub("scipy")

import dashboard.app as app_module  # noqa: E402
import dashboard.doe_tab as doe_tab  # noqa: E402


def _get_callback(
    cb_map: dict[str, Any], trigger_id: str
) -> Callable[..., Any]:  # noqa: E501
    key = next(
        (
            k
            for k, v in cb_map.items()
            if any(i["id"] == trigger_id for i in v["inputs"])
        )
    )
    return cast(Callable[..., Any], cb_map[key]["callback"].__wrapped__)


def test_plan_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    app = app_module.create_app()
    cb = app.callback_map

    add_factor = _get_callback(cb, doe_tab.ADD_FACTOR)
    add_level = _get_callback(cb, doe_tab.ADD_LEVEL)
    add_row = _get_callback(cb, doe_tab.ADD_ROW)
    save_cb = _get_callback(cb, doe_tab.SAVE_PLAN)

    plan: dict[str, Any] = {"factors": {}, "matrix": []}
    plan, _, _, _ = add_factor(1, "anode", plan)
    plan, _, _, _ = add_level(1, "anode", "A", plan)
    plan, _, _ = add_row(1, '{"anode": "A"}', plan)

    saved: dict[str, Any] = {}

    def fake_save(
        name: str, factors: dict[str, Any], matrix: list[dict[str, Any]]
    ) -> None:
        saved.update(name=name, factors=factors, matrix=matrix)

    monkeypatch.setattr(app_module, "save_plan", fake_save)

    result = save_cb(1, "New Plan", plan)
    assert saved["name"] == "New Plan"
    assert cast(dict[str, Any], saved["factors"]) == {"anode": ["A"]}
    assert cast(list[dict[str, Any]], saved["matrix"]) == [{"anode": "A"}]
    assert result[2] is True and result[4] == "Success"


def test_plan_editing(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = {
        "name": "Base",
        "factors": {"anode": ["A"], "cathode": ["X"]},
        "matrix": [{"anode": "A", "cathode": "X"}],
    }
    monkeypatch.setattr(doe_tab, "_load_plans", lambda: [existing])
    app = app_module.create_app()
    cb = app.callback_map

    load_plan = _get_callback(cb, doe_tab.PLAN_DROPDOWN)
    add_level = _get_callback(cb, doe_tab.ADD_LEVEL)
    add_row = _get_callback(cb, doe_tab.ADD_ROW)
    save_cb = _get_callback(cb, doe_tab.SAVE_PLAN)

    _, plan, _, _, _ = load_plan("Base")
    plan, _, _, _ = add_level(1, "anode", "B", plan)
    plan, _, _ = add_row(1, '{"anode": "B", "cathode": "X"}', plan)

    saved: dict[str, Any] = {}

    def fake_save(
        name: str, factors: dict[str, Any], matrix: list[dict[str, Any]]
    ) -> None:
        saved.update(name=name, factors=factors, matrix=matrix)

    monkeypatch.setattr(app_module, "save_plan", fake_save)
    save_cb(1, "Base", plan)
    assert cast(dict[str, Any], saved["factors"])["anode"] == ["A", "B"]
    assert len(cast(list[dict[str, Any]], saved["matrix"])) == 2
