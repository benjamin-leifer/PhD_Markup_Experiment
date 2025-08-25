import sys
from dataclasses import dataclass
from pathlib import Path
import types
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard import advanced_analysis_tab, cycle_detail_tab, eis_tab


SAMPLES = [{"_id": "s1", "name": "SampleA"}]
TESTS = [{"_id": "t1", "name": "TestA"}]


def _patch_env(monkeypatch: Any, module: Any) -> None:
    """Patch module environment to use fallback data providers."""
    monkeypatch.setattr(module, "db_connected", lambda: True)
    monkeypatch.setattr(module, "find_samples", lambda: SAMPLES)
    monkeypatch.setattr(module, "find_test_results", lambda query: TESTS)


def _patch_models(monkeypatch: Any) -> None:
    """Provide minimal dataclass models lacking ``objects`` managers."""
    fake_pkg = types.ModuleType("battery_analysis")
    fake_models = types.ModuleType("battery_analysis.models")

    @dataclass
    class Sample:
        id: str
        name: str

    @dataclass
    class TestResult:
        id: str
        sample: str
        name: str
        test_type: str = ""

    fake_models.Sample = Sample
    fake_models.TestResult = TestResult
    fake_pkg.models = fake_models
    monkeypatch.setitem(sys.modules, "battery_analysis", fake_pkg)
    monkeypatch.setitem(sys.modules, "battery_analysis.models", fake_models)


def test_advanced_analysis_tab_options(monkeypatch: Any) -> None:
    _patch_models(monkeypatch)
    _patch_env(monkeypatch, advanced_analysis_tab)

    sample_opts = advanced_analysis_tab._get_sample_options()
    assert sample_opts == [{"label": "SampleA", "value": "s1"}]

    test_opts = advanced_analysis_tab._get_test_options("s1")
    assert test_opts == [{"label": "TestA", "value": "t1"}]


def test_cycle_detail_tab_options(monkeypatch: Any) -> None:
    _patch_models(monkeypatch)
    _patch_env(monkeypatch, cycle_detail_tab)

    sample_opts = cycle_detail_tab._get_sample_options()
    assert sample_opts == [{"label": "SampleA", "value": "s1"}]

    test_opts = cycle_detail_tab._get_test_options("s1")
    assert test_opts == [{"label": "TestA", "value": "t1"}]


def test_eis_tab_options(monkeypatch: Any) -> None:
    _patch_models(monkeypatch)
    _patch_env(monkeypatch, eis_tab)

    sample_opts = eis_tab._get_sample_options()
    assert sample_opts == [{"label": "SampleA", "value": "s1"}]

    test_opts = eis_tab._get_test_options("s1")
    assert test_opts == [{"label": "TestA", "value": "t1"}]
