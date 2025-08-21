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

disconnect()
connect("import_test", mongo_client_class=mongomock.MongoClient, alias="default")
from battery_analysis.models import Sample  # noqa: E402
from battery_analysis.utils import import_directory  # noqa: E402
from battery_analysis import parsers  # noqa: E402


def _make_file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / "S1" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("dummy")
    return path


def _setup_db() -> None:
    Sample._registry.clear()
    disconnect()
    connect("import_test", mongo_client_class=mongomock.MongoClient, alias="default")


def test_new_file_creates_testresult(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_db()
    _make_file(tmp_path, "test.csv")

    def fake_process(path: str, sample: Sample) -> tuple[object, bool]:
        cycles, _ = parsers.parse_file(path)
        if sample.tests:
            sample.tests[0].cycles.extend(cycles)
            return sample.tests[0], True
        test = type("Test", (), {"cycles": cycles})()
        sample.tests.append(test)
        return test, False

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    import_directory.import_directory(tmp_path, workers=1)

    assert len(Sample._registry) == 1
    sample = Sample.get_by_name("S1")
    assert sample is not None
    assert len(sample.tests) == 1
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

    def fake_process(path: str, sample: Sample) -> tuple[object, bool]:
        cycles, _ = parsers.parse_file(path)
        if sample.tests:
            sample.tests[0].cycles.extend(cycles)
            return sample.tests[0], True
        test = type("Test", (), {"cycles": cycles})()
        sample.tests.append(test)
        return test, False

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    import_directory.import_directory(tmp_path, workers=1)

    assert len(Sample._registry) == 1
    sample = Sample.get_by_name("S1")
    assert sample is not None
    assert len(sample.tests) == 1
    test = sample.tests[0]
    assert [c["cycle_index"] for c in test.cycles] == [1, 2, 3, 4]
    disconnect()


def test_state_skips_and_reset_reimports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_db()
    _make_file(tmp_path, "test.csv")

    calls: list[str] = []

    def fake_process(path: str, sample: Sample) -> tuple[object, bool]:
        calls.append(path)

        class Dummy:
            pass

        return Dummy(), False

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    import_directory.import_directory(tmp_path, workers=1)
    assert len(calls) == 1
    assert (tmp_path / ".import_state.json").exists()

    import_directory.import_directory(tmp_path, workers=1)
    assert len(calls) == 1

    import_directory.import_directory(tmp_path, reset=True, workers=1)
    assert len(calls) == 2
    disconnect()


def test_dry_run_skips_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_db()
    _make_file(tmp_path, "test.csv")

    def fail(
        *_args: object, **_kwargs: object
    ) -> None:  # pragma: no cover - should not be called
        raise AssertionError("Should not be called in dry run")

    monkeypatch.setattr(import_directory.data_update, "process_file_with_update", fail)
    monkeypatch.setattr(import_directory, "update_cell_dataset", fail)

    import_directory.import_directory(tmp_path, dry_run=True, workers=1)

    assert len(Sample._registry) == 0
    assert not (tmp_path / ".import_state.json").exists()
    disconnect()


def test_include_exclude_patterns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_db()
    keep_dir = tmp_path / "keep"
    skip_dir = tmp_path / "skip"
    (keep_dir / "keep.csv").parent.mkdir(parents=True, exist_ok=True)
    (keep_dir / "keep.csv").write_text("dummy")
    (keep_dir / "ignore.txt").write_text("dummy")
    (skip_dir / "skip.csv").parent.mkdir(parents=True, exist_ok=True)
    (skip_dir / "skip.csv").write_text("dummy")

    calls: list[str] = []

    def fake_process(path: str, sample: Sample) -> tuple[object, bool]:
        calls.append(Path(path).name)
        return object(), False

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    import_directory.import_directory(
        tmp_path,
        workers=1,
        include=["*.csv", "*keep*"],
        exclude=["*.txt", "*skip*"],
    )

    assert calls == ["keep.csv"]
    disconnect()
