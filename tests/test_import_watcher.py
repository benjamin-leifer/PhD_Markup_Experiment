from __future__ import annotations

from pathlib import Path
import sys
import types
import time
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for p in (ROOT, PACKAGE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Stub out heavy optional dependencies
if "scipy" not in sys.modules:
    class _ScipyStub(types.ModuleType):
        def __getattr__(self, name: str) -> types.ModuleType:
            mod = types.ModuleType(name)
            sys.modules[f"scipy.{name}"] = mod
            setattr(self, name, mod)
            return mod
    sys.modules["scipy"] = _ScipyStub("scipy")

if "matplotlib" not in sys.modules:
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.get_backend = lambda: "agg"
    matplotlib.use = lambda *args, **kwargs: None
    pyplot = types.ModuleType("pyplot")
    pyplot.figure = lambda *args, **kwargs: types.SimpleNamespace(
        canvas=types.SimpleNamespace(manager=None)
    )
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot

analysis_stub = types.ModuleType("analysis")
analysis_stub.compute_metrics = lambda cycles: {}
analysis_stub.update_sample_properties = lambda *a, **k: None
analysis_stub.create_test_result = lambda *a, **k: types.SimpleNamespace(cycles=[])
analysis_stub.summarize_detailed_cycles = lambda *a, **k: []
sys.modules.setdefault("battery_analysis.analysis", analysis_stub)
sys.modules.setdefault("battery_analysis.report", types.ModuleType("report"))
for name in ["advanced_analysis", "plots", "eis", "outlier_analysis"]:
    sys.modules.setdefault(f"battery_analysis.{name}", types.ModuleType(name))

import mongomock  # noqa: E402
from mongoengine import connect, disconnect  # noqa: E402

disconnect()
connect("watcher_test", mongo_client_class=mongomock.MongoClient, alias="default")

from battery_analysis.models import Sample, ImportJob  # noqa: E402
from battery_analysis.utils import import_watcher, import_directory  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db() -> None:
    Sample._registry.clear()
    ImportJob._registry.clear()
    disconnect()
    connect("watcher_test", mongo_client_class=mongomock.MongoClient, alias="default")
    yield
    disconnect()


def test_watcher_triggers_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, Sample]] = []

    def fake_process(path: str, sample: Sample) -> None:
        calls.append((Path(path), sample))

    monkeypatch.setattr(import_directory, "process_file_with_update", fake_process)

    observer = import_watcher.watch(str(tmp_path), debounce=0.1)
    try:
        sample_dir = tmp_path / "S1"
        sample_dir.mkdir()
        file_path = sample_dir / "test.csv"
        file_path.write_text("data")

        for _ in range(20):
            if calls:
                break
            time.sleep(0.1)

        assert calls, "process_file_with_update was not called"
        assert calls[0][0] == file_path
        assert calls[0][1].name == "S1"
    finally:
        observer.stop()
        observer.join()


def test_start_stop_api(tmp_path: Path) -> None:
    """The programmatic API should track running watchers."""

    import_watcher.start_watcher(str(tmp_path), debounce=0.1)
    try:
        running = import_watcher.list_watchers()
        assert any(w["directory"] == str(tmp_path) for w in running)
    finally:
        import_watcher.stop_watcher(str(tmp_path))
    assert all(w["directory"] != str(tmp_path) for w in import_watcher.list_watchers())
