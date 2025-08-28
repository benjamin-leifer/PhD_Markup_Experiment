from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

# mypy: ignore-errors
# flake8: noqa



ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis" / "battery_analysis"

# ---------------------------------------------------------------------------
# Lightweight package stubs so the module can be imported without heavy deps
# ---------------------------------------------------------------------------
package_stub = types.ModuleType("battery_analysis")
package_stub.__path__ = [str(PACKAGE_DIR)]  # type: ignore[attr-defined]
sys.modules["battery_analysis"] = package_stub

utils_stub = types.ModuleType("battery_analysis.utils")
utils_stub.__path__ = [str(PACKAGE_DIR / "utils")]  # type: ignore[attr-defined]
sys.modules["battery_analysis.utils"] = utils_stub


class DummyTestResult:
    _store: list["DummyTestResult"] = []

    def __init__(
        self,
        id: str,
        name: str = "",
        file_path: str | None = None,
        cell_code: str | None = None,
    ):
        self.id = id
        self.name = name
        self.file_path = file_path
        self.cell_code = cell_code
        self.cycles: list = []
        self.metadata: dict = {}

    def save(self) -> None:  # pragma: no cover - no-op
        pass

    @classmethod
    def objects(cls, **query):
        objs = cls._store
        if query:
            key, value = next(iter(query.items()))
            objs = [o for o in objs if getattr(o, key) == value]
        return DummyQuery(objs)


class DummyQuery:
    def __init__(self, objs):
        self._objs = list(objs)

    def count(self) -> int:
        return len(self._objs)

    def skip(self, n: int) -> "DummyQuery":
        return DummyQuery(self._objs[n:])

    def limit(self, n: int) -> "DummyQuery":
        return DummyQuery(self._objs[:n])

    def __iter__(self):
        return iter(self._objs)


class DummyRefactorJob:
    _registry: dict[str, "DummyRefactorJob"] = {}

    def __init__(self, status: str = "running"):
        import datetime
        import uuid

        self.id = str(uuid.uuid4())
        self.start_time = datetime.datetime.utcnow()
        self.end_time = None
        self.current_test = None
        self.processed_count = 0
        self.total_count = 0
        self.errors: list[str] = []
        self.status = status

    def save(self) -> "DummyRefactorJob":
        self.__class__._registry[self.id] = self
        return self

    @classmethod
    def objects(cls, **query):
        if "id" in query:
            obj = cls._registry.get(str(query["id"]))
            return DummyJobQuery([obj] if obj else [])
        return DummyJobQuery(list(cls._registry.values()))

    def delete(self) -> None:  # pragma: no cover - not used
        self.__class__._registry.pop(self.id, None)


class DummyJobQuery(list):
    def first(self):
        return self[0] if self else None

    def count(self) -> int:  # pragma: no cover - simple helper
        return len(self)


models_stub = types.ModuleType("battery_analysis.models")
models_stub.TestResult = DummyTestResult
models_stub.RefactorJob = DummyRefactorJob
sys.modules["battery_analysis.models"] = models_stub

fs_stub = types.SimpleNamespace(save_raw=lambda *a, **k: None)
sys.modules["battery_analysis.utils.file_storage"] = fs_stub

cdb_stub = types.SimpleNamespace(update_cell_dataset=lambda code: None)
sys.modules["battery_analysis.utils.cell_dataset_builder"] = cdb_stub


def _normalize_identifier(name: str | None) -> str | None:
    return name


def update_test_data(test, cycles, metadata, strategy="replace") -> None:
    pass


data_update_stub = types.SimpleNamespace(
    _normalize_identifier=_normalize_identifier, update_test_data=update_test_data
)
sys.modules["battery_analysis.utils.data_update"] = data_update_stub

sys.modules["battery_analysis.utils.db"] = types.SimpleNamespace(
    ensure_connection=lambda: None
)

spec = importlib.util.spec_from_file_location(
    "battery_analysis.utils.refactor_data", PACKAGE_DIR / "utils" / "refactor_data.py"
)
refactor_data = importlib.util.module_from_spec(spec)
sys.modules["battery_analysis.utils.refactor_data"] = refactor_data
assert spec.loader is not None
spec.loader.exec_module(refactor_data)


@pytest.fixture(autouse=True)
def clear_stores() -> None:
    DummyTestResult._store.clear()
    DummyRefactorJob._registry.clear()


def test_resume_updates_job_and_skips_processed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for idx in range(3):
        DummyTestResult._store.append(DummyTestResult(f"T{idx+1}", name=f"N{idx+1}"))
    job = DummyRefactorJob().save()
    job.processed_count = 1
    job.save()

    processed: list[str] = []

    def fake_update(test, cycles, metadata, strategy="replace"):
        processed.append(test.id)

    monkeypatch.setattr(refactor_data, "update_test_data", fake_update)

    report = tmp_path / "report.json"
    refactor_data.refactor_tests(batch_size=1, job_id=job.id, report_path=report)

    assert processed == ["T2", "T3"]
    saved = DummyRefactorJob._registry[job.id]
    assert saved.processed_count == 3
    assert saved.total_count == 3
    assert saved.status == "completed"
    assert saved.end_time is not None

    data = json.loads(report.read_text())
    assert [d["test"] for d in data] == ["T2", "T3"]
    assert all(d["status"] == "ok" for d in data)
