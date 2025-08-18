from __future__ import annotations

from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mongomock  # noqa: E402
from mongoengine import connect, disconnect  # noqa: E402
from battery_analysis.models import Sample, TestResult  # noqa: E402
from battery_analysis.utils import import_directory  # noqa: E402
from battery_analysis.utils import data_update  # noqa: E402
from battery_analysis import parsers  # noqa: E402


def _make_file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / "S1" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("dummy")
    return path


def _setup_db() -> None:
    disconnect()
    connect("import_test", mongo_client_class=mongomock.MongoClient, alias="default")


def test_new_file_creates_testresult(tmp_path: Path) -> None:
    _setup_db()
    _make_file(tmp_path, "test.csv")

    import_directory.import_directory(tmp_path)

    assert Sample.objects.count() == 1
    assert TestResult.objects.count() == 1
    disconnect()


def test_sequential_files_append_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_db()
    _make_file(tmp_path, "run_Wb_1.csv")
    _make_file(tmp_path, "run_Wb_2.csv")

    def fake_parse(path: str) -> tuple[list[dict[str, float]], dict[str, object]]:
        name = Path(path).name
        if "Wb_1" in name:
            cycles = [
                {
                    "cycle_index": 1,
                    "charge_capacity": 1.0,
                    "discharge_capacity": 1.0,
                    "coulombic_efficiency": 1.0,
                },
                {
                    "cycle_index": 2,
                    "charge_capacity": 1.0,
                    "discharge_capacity": 1.0,
                    "coulombic_efficiency": 1.0,
                },
            ]
        else:
            cycles = [
                {
                    "cycle_index": 3,
                    "charge_capacity": 1.0,
                    "discharge_capacity": 1.0,
                    "coulombic_efficiency": 1.0,
                },
                {
                    "cycle_index": 4,
                    "charge_capacity": 1.0,
                    "discharge_capacity": 1.0,
                    "coulombic_efficiency": 1.0,
                },
            ]
        return cycles, {"tester": "Other", "name": name, "date": None}

    monkeypatch.setattr(parsers, "parse_file", fake_parse)

    import_directory.import_directory(tmp_path)

    assert Sample.objects.count() == 1
    assert TestResult.objects.count() == 1
    test = TestResult.objects.first()
    assert [c.cycle_index for c in test.cycles] == [1, 2, 3, 4]
    disconnect()


def test_include_exclude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_db()
    keep = _make_file(tmp_path, "keep.csv")
    _make_file(tmp_path, "skip.csv")

    processed: list[Path] = []

    def fake_process(path: str, sample: Sample) -> tuple[object, bool]:
        processed.append(Path(path))
        from types import SimpleNamespace

        return SimpleNamespace(id="x"), False

    monkeypatch.setattr(data_update, "process_file_with_update", fake_process)
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    import_directory.import_directory(tmp_path, include=["*keep*"], exclude=["*skip*"])

    assert processed == [keep]
    disconnect()
