from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis" / "battery_analysis"

# Stub out the package structure to avoid heavy imports
package_stub = types.ModuleType("battery_analysis")
package_stub.__path__ = [str(PACKAGE_DIR)]  # type: ignore[attr-defined]
sys.modules["battery_analysis"] = package_stub

utils_stub = types.ModuleType("battery_analysis.utils")
utils_stub.__path__ = [str(PACKAGE_DIR / "utils")]  # type: ignore[attr-defined]
sys.modules["battery_analysis.utils"] = utils_stub

models_stub = types.ModuleType("battery_analysis.models")
models_stub.TestResult = type("TestResult", (), {})  # placeholder
sys.modules["battery_analysis.models"] = models_stub

file_storage_stub = types.ModuleType("battery_analysis.utils.file_storage")
file_storage_stub.retrieve_raw = lambda *a, **k: b""  # overwritten in tests
sys.modules["battery_analysis.utils.file_storage"] = file_storage_stub

db_stub = types.ModuleType("battery_analysis.utils.db")
db_stub.ensure_connection = lambda: True
sys.modules["battery_analysis.utils.db"] = db_stub

spec = importlib.util.spec_from_file_location(
    "battery_analysis.utils.raw_file_cli", str(PACKAGE_DIR / "utils" / "raw_file_cli.py")
)
raw_file_cli = importlib.util.module_from_spec(spec)
sys.modules["battery_analysis.utils.raw_file_cli"] = raw_file_cli
spec.loader.exec_module(raw_file_cli)  # type: ignore


def test_download_writes_stdout(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
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
