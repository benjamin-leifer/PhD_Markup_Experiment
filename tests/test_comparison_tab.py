import sys
import types
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard import comparison_tab  # noqa: E402


def test_get_sample_options_empty(monkeypatch: Any) -> None:
    class DummyQueryset:
        def only(self, *args: Any, **kwargs: Any) -> List[Any]:
            return []

    class DummySample:
        objects = DummyQueryset()

    models = types.SimpleNamespace(Sample=DummySample)
    ba_module = types.ModuleType("battery_analysis")
    ba_module.models = models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "battery_analysis", ba_module)

    opts, error = comparison_tab._get_sample_options()
    assert opts == [{"label": "Sample_001", "value": "sample1"}]
    assert error is not None
