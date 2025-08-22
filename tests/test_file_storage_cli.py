from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis" / "battery_analysis"

# ---------------------------------------------------------------------------
# Lightweight stubs of the package structure to avoid heavy imports
# ---------------------------------------------------------------------------
package_stub = types.ModuleType("battery_analysis")
package_stub.__path__ = [str(PACKAGE_DIR)]  # type: ignore[attr-defined]
sys.modules["battery_analysis"] = package_stub

utils_stub = types.ModuleType("battery_analysis.utils")
utils_stub.__path__ = [str(PACKAGE_DIR / "utils")]  # type: ignore[attr-defined]
sys.modules["battery_analysis.utils"] = utils_stub


# In-memory model stand-ins -------------------------------------------------
class DummyRawDataFile:
    _store: list["DummyRawDataFile"] = []

    def __init__(self, id: str):
        self.id = id

    @classmethod
    def objects(cls, **query):
        class Query:
            def __init__(self, objs):
                self._objs = objs

            def first(self):
                return self._objs[0] if self._objs else None

            def __iter__(self):
                return iter(self._objs)

        res = [
            o
            for o in cls._store
            if all(getattr(o, k, None) == v for k, v in query.items())
        ]
        return Query(res)

    def delete(self) -> None:
        self.__class__._store.remove(self)

    class file_data:  # noqa: D401 - simple stub
        @staticmethod
        def delete() -> None:
            pass


class DummyTestResult:
    _store: list["DummyTestResult"] = []

    def __init__(self, id: str, *, file_hash: str | None = None, file_id: str | None = None):
        self.id = id
        self.file_hash = file_hash
        self.file_id = file_id

    def save(self) -> None:
        if self not in self.__class__._store:
            self.__class__._store.append(self)

    @classmethod
    def objects(cls, **query):
        class Query:
            def __init__(self, objs):
                self._objs = objs

            def first(self):
                return self._objs[0] if self._objs else None

        res = [
            o
            for o in cls._store
            if all(getattr(o, k, None) == v for k, v in query.items())
        ]
        return Query(res)


models_stub = types.ModuleType("battery_analysis.models")
models_stub.TestResult = DummyTestResult
models_stub.RawDataFile = DummyRawDataFile
sys.modules["battery_analysis.models"] = models_stub

# Stubbed file_storage module for CLI tests (real module imported separately)
file_storage_stub = types.ModuleType("battery_analysis.utils.file_storage")
file_storage_stub.retrieve_raw = lambda *a, **k: b""  # overwritten in tests
file_storage_stub.cleanup_orphaned = lambda: 0
sys.modules["battery_analysis.utils.file_storage"] = file_storage_stub

db_stub = types.ModuleType("battery_analysis.utils.db")
db_stub.ensure_connection = lambda: True
sys.modules["battery_analysis.utils.db"] = db_stub

# Import the CLI module which uses the above stubs
spec = importlib.util.spec_from_file_location(
    "battery_analysis.utils.raw_file_cli", str(PACKAGE_DIR / "utils" / "raw_file_cli.py")
)
raw_file_cli = importlib.util.module_from_spec(spec)
sys.modules["battery_analysis.utils.raw_file_cli"] = raw_file_cli
spec.loader.exec_module(raw_file_cli)  # type: ignore

# Load the real file_storage implementation under a different name
spec_fs = importlib.util.spec_from_file_location(
    "battery_analysis.utils.file_storage_real", str(PACKAGE_DIR / "utils" / "file_storage.py")
)
file_storage_real = importlib.util.module_from_spec(spec_fs)
sys.modules["battery_analysis.utils.file_storage_real"] = file_storage_real
spec_fs.loader.exec_module(file_storage_real)  # type: ignore


# ---------------------------------------------------------------------------
# CLI behaviour
# ---------------------------------------------------------------------------

def test_download_writes_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_retrieve(fid: str, as_file_path: bool = False) -> bytes:
        calls.append((fid, as_file_path))
        return b"abc"

    monkeypatch.setattr(raw_file_cli.file_storage, "retrieve_raw", fake_retrieve)

    raw_file_cli.main(["download", "F1"])

    assert capsys.readouterr().out == "abc"
    assert calls == [("F1", False)]


def test_by_test_saves_to_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = types.SimpleNamespace(file_id="FILE")

    class DummyQuery:
        def first(self) -> object:
            return dummy

    def objects(**query):
        assert query == {"id": "T1"}
        return DummyQuery()

    monkeypatch.setattr(raw_file_cli, "TestResult", types.SimpleNamespace(objects=objects))

    src = tmp_path / "src.bin"
    src.write_bytes(b"data")

    calls: list[tuple[str, bool]] = []

    def fake_retrieve(fid: str, as_file_path: bool = False) -> str:
        calls.append((fid, as_file_path))
        return str(src)

    monkeypatch.setattr(raw_file_cli.file_storage, "retrieve_raw", fake_retrieve)

    out = tmp_path / "out.bin"
    raw_file_cli.main(["by-test", "T1", "--out", str(out)])

    assert out.read_bytes() == b"data"
    assert calls == [("FILE", True)]


def test_cleanup_invokes_gc(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[bool] = []

    def fake_cleanup() -> int:
        calls.append(True)
        return 2

    monkeypatch.setattr(raw_file_cli.file_storage, "cleanup_orphaned", fake_cleanup)

    raw_file_cli.main(["cleanup"])

    out = capsys.readouterr().out
    assert "2" in out
    assert calls == [True]


# ---------------------------------------------------------------------------
# file_storage helpers
# ---------------------------------------------------------------------------

def test_save_raw_deduplicates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    DummyTestResult._store.clear()

    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")

    ids = iter(["RID1", "RID2"])

    def fake_store(path: str, test_result=None, file_type=None):
        rid = next(ids)
        return types.SimpleNamespace(id=rid)

    monkeypatch.setattr(file_storage_real, "store_raw_data_file", fake_store)

    t1 = DummyTestResult("T1")
    t1.save()
    out1 = file_storage_real.save_raw(str(path), test_result=t1)

    t2 = DummyTestResult("T2")
    t2.save()
    out2 = file_storage_real.save_raw(str(path), test_result=t2)

    assert out1 == out2 == "RID1"
    assert t1.file_id == t2.file_id == "RID1"


def test_cleanup_orphaned_removes_unlinked() -> None:
    DummyRawDataFile._store = [DummyRawDataFile("R1"), DummyRawDataFile("R2")]
    DummyTestResult._store = [DummyTestResult("T", file_id="R1", file_hash="h")]

    removed = file_storage_real.cleanup_orphaned()

    assert removed == 1
    assert [f.id for f in DummyRawDataFile._store] == ["R1"]

