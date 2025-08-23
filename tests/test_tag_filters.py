import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for p in (ROOT, PACKAGE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from battery_analysis.models import Sample, TestResult  # noqa: E402
import types
import importlib.machinery

# Provide minimal mongoengine stub to satisfy imports
if "mongoengine" not in sys.modules:
    me = types.ModuleType("mongoengine")
    me.__path__ = []  # type: ignore[attr-defined]
    me.__spec__ = importlib.machinery.ModuleSpec("mongoengine", loader=None, is_package=True)
    sys.modules["mongoengine"] = me
    qs = types.ModuleType("mongoengine.queryset")
    qs.__path__ = []  # type: ignore[attr-defined]
    qs.__spec__ = importlib.machinery.ModuleSpec("mongoengine.queryset", loader=None, is_package=True)
    visitor = types.ModuleType("mongoengine.queryset.visitor")
    visitor.__spec__ = importlib.machinery.ModuleSpec(
        "mongoengine.queryset.visitor", loader=None, is_package=False
    )
    me.queryset = qs  # type: ignore[attr-defined]
    qs.visitor = visitor  # type: ignore[attr-defined]

    class _Q:  # pragma: no cover - simple stub
        pass

    visitor.Q = _Q
    sys.modules["mongoengine.queryset"] = qs
    sys.modules["mongoengine.queryset.visitor"] = visitor

from battery_analysis.utils import import_directory  # noqa: E402
from dashboard.trait_filter_tab import build_query  # noqa: E402


def test_process_file_with_tags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_path = tmp_path / "S1" / "test.csv"
    data_path.parent.mkdir()
    data_path.write_text("dummy")

    sample = Sample(name="S1")

    def fake_process(path: str, sample: Sample) -> tuple[TestResult, bool]:
        return TestResult(sample=sample), False

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )

    test, _ = import_directory.process_file_with_update(
        str(data_path), sample, tags=["alpha", "beta"], archive=False
    )

    assert set(sample.tags) == {"alpha", "beta"}
    assert set(getattr(test, "tags", [])) == {"alpha", "beta"}


def test_build_query_tags() -> None:
    q = build_query(
        chemistry=None,
        manufacturer=None,
        additives=None,
        additive_mode="any",
        tags=["x", "y"],
        tag_mode="all",
        cycle_min=None,
        cycle_max=None,
        thick_min=None,
        thick_max=None,
    )
    assert q == {"tags": {"$all": ["x", "y"]}}

