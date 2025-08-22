from __future__ import annotations

from pathlib import Path
import sys
import json
import os
import hashlib
import pytest
import types
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Stub out heavy optional dependencies
if "scipy" not in sys.modules:

    class _ScipyStub(types.ModuleType):
        def __getattr__(self, name: str) -> types.ModuleType:  # pragma: no cover - stub
            mod = types.ModuleType(name)
            sys.modules[f"scipy.{name}"] = mod
            setattr(self, name, mod)
            return mod

    sys.modules["scipy"] = _ScipyStub("scipy")

import mongomock  # noqa: E402
from mongoengine import connect, disconnect  # noqa: E402

disconnect()
connect("import_test", mongo_client_class=mongomock.MongoClient, alias="default")
from battery_analysis.models import Sample, ImportJob, TestResult  # noqa: E402
from battery_analysis.utils import import_directory  # noqa: E402
from battery_analysis import parsers  # noqa: E402

parsers.register_parser(".csv", lambda path: ([], {}))


@pytest.fixture
def import_dir(tmp_path: Path) -> tuple[Path, Callable[[str, str, str], Path]]:
    def make(name: str, content: str = "dummy", sample: str = "S1") -> Path:
        path = tmp_path / sample / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    return tmp_path, make


@pytest.fixture(autouse=True)
def fresh_db() -> None:
    Sample._registry.clear()
    ImportJob._registry.clear()
    disconnect()
    connect("import_test", mongo_client_class=mongomock.MongoClient, alias="default")
    yield
    disconnect()


@pytest.mark.parallel
def test_progress_logging_and_summary(
    import_dir: tuple[Path, Callable[[str, str, str], Path]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root, make = import_dir
    for i in range(3):
        make(f"run{i}.csv")

    def fake_process(path: str, sample: Sample) -> tuple[object, bool]:
        return object(), False

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    with caplog.at_level("INFO"):
        import_directory.import_directory(root, workers=2)

    assert "Processed 3/3" in caplog.text
    assert "Summary: created=3, updated=0, skipped=0" in caplog.text


def test_new_file_creates_testresult(
    import_dir: tuple[Path, Callable[[str, str, str], Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, make = import_dir
    make("test.csv")

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

    import_directory.import_directory(root, workers=1)

    assert len(Sample._registry) == 1
    sample = Sample.get_by_name("S1")
    assert sample is not None
    assert len(sample.tests) == 1


def test_process_file_archives_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    from battery_analysis.utils import file_storage

    data_path = tmp_path / "S1" / "test.csv"
    data_path.parent.mkdir()
    content = b"abc"
    data_path.write_bytes(content)

    sample = Sample(name="S1")

    def fake_process(path: str, sample: Sample) -> tuple[TestResult, bool]:
        test = TestResult(sample=sample)
        return test, False

    stored: dict[str, bytes] = {}

    def fake_save_raw(path: str, **_: object) -> str:
        fid = "1"
        stored[fid] = Path(path).read_bytes()
        return fid

    def fake_retrieve_raw(file_id: str, as_file_path: bool = False):
        return stored[file_id]

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )
    monkeypatch.setattr(file_storage, "save_raw", fake_save_raw)
    monkeypatch.setattr(file_storage, "retrieve_raw", fake_retrieve_raw)

    test, was_update = import_directory.process_file_with_update(str(data_path), sample)
    assert not was_update
    assert test.file_hash == hashlib.sha256(content).hexdigest()
    assert test.file_id == "1"
    assert stored["1"] == content


def test_process_file_no_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    from battery_analysis.utils import file_storage

    data_path = tmp_path / "S1" / "test.csv"
    data_path.parent.mkdir()
    content = b"xyz"
    data_path.write_bytes(content)

    sample = Sample(name="S1")

    def fake_process(path: str, sample: Sample) -> tuple[TestResult, bool]:
        test = TestResult(sample=sample)
        return test, False

    called = False

    def fake_save_raw(path: str, **_: object) -> str:  # pragma: no cover - safety
        nonlocal called
        called = True
        return "id"

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )
    monkeypatch.setattr(file_storage, "save_raw", fake_save_raw)

    test, _ = import_directory.process_file_with_update(
        str(data_path), sample, archive=False
    )
    assert test.file_hash == hashlib.sha256(content).hexdigest()
    assert getattr(test, "file_id", None) is None
    assert not called


def test_renamed_duplicate_skipped(
    import_dir: tuple[Path, Callable[[str, str, str], Path]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root, make = import_dir
    original = make("run1.csv")

    def fake_process(path: str, sample: Sample) -> tuple[TestResult, bool]:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for t in sample.tests:
            if getattr(t, "file_hash", None) == digest:
                return t, True
        test = TestResult(sample=sample)
        test.file_hash = digest
        sample.tests.append(test)
        return test, False

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    import_directory.import_directory(root, workers=1, archive=False)
    caplog.clear()

    renamed = original.parent / "renamed.csv"
    renamed.write_text(original.read_text())
    original.unlink()

    with caplog.at_level("INFO"):
        import_directory.import_directory(root, workers=1, archive=False)
    assert "Summary: created=0, updated=1, skipped=0" in caplog.text

    sample = Sample.get_by_name("S1")
    assert sample is not None
    assert len(sample.tests) == 1


def test_incomplete_metadata_skips_file(
    import_dir: tuple[Path, Callable[[str, str, str], Path]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root, make = import_dir
    make("bad.csv")

    def fake_parse(path: str) -> tuple[list[dict[str, float]], dict[str, object]]:
        # Return metadata missing required keys like 'name' and 'date'
        return [], {"tester": "X"}

    monkeypatch.setattr(parsers, "parse_file", fake_parse)
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    with caplog.at_level("ERROR"):
        import_directory.import_directory(root, workers=1)

    # Validation error should be logged and no tests created
    assert "Missing required metadata" in caplog.text
    sample = Sample.get_by_name("S1")
    assert sample is not None
    assert len(sample.tests) == 0


@pytest.mark.slow
def test_sequential_files_append_cycles(
    import_dir: tuple[Path, Callable[[str, str, str], Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, make = import_dir
    make("run_Wb_1.csv")
    make("run_Wb_2.csv")

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

    import_directory.import_directory(root, workers=1)

    assert len(Sample._registry) == 1
    sample = Sample.get_by_name("S1")
    assert sample is not None
    assert len(sample.tests) == 1
    test = sample.tests[0]
    assert sorted(c["cycle_index"] for c in test.cycles) == [1, 2, 3, 4]


def test_state_skips_and_reset_reimports(
    import_dir: tuple[Path, Callable[[str, str, str], Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, make = import_dir
    file_path = make("test.csv")

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

    import_directory.import_directory(root, workers=1)
    assert len(calls) == 1
    state_path = root / ".import_state.json"
    assert state_path.exists()
    data = json.loads(state_path.read_text())
    entry = data[str(file_path.resolve())]
    assert {"mtime", "hash"} <= set(entry)

    import_directory.import_directory(root, workers=1)
    assert len(calls) == 1

    import_directory.import_directory(root, reset=True, workers=1)
    assert len(calls) == 2


def test_hash_change_triggers_processing(
    import_dir: tuple[Path, Callable[[str, str, str], Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, make = import_dir
    file_path = make("test.csv")

    calls: list[str] = []

    def fake_process(path: str, sample: Sample) -> tuple[object, bool]:
        calls.append(path)
        return object(), False

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    import_directory.import_directory(root, workers=1)
    assert len(calls) == 1

    state = json.loads((root / ".import_state.json").read_text())
    mtime = state[str(file_path.resolve())]["mtime"]

    file_path.write_text("modified")
    os.utime(file_path, (mtime, mtime))

    import_directory.import_directory(root, workers=1)
    assert len(calls) == 2


def test_missing_hash_migrates_state(
    import_dir: tuple[Path, Callable[[str, str, str], Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, make = import_dir
    file_path = make("test.csv")
    mtime = os.path.getmtime(file_path)
    state_path = root / ".import_state.json"
    state_path.write_text(json.dumps({str(file_path.resolve()): mtime}))

    calls: list[str] = []

    def fake_process(path: str, sample: Sample) -> tuple[object, bool]:
        calls.append(path)
        return object(), False

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    import_directory.import_directory(root, workers=1)
    assert calls == []

    data = json.loads(state_path.read_text())
    entry = data[str(file_path.resolve())]
    assert {"mtime", "hash"} <= set(entry)


def test_dry_run_skips_processing(
    import_dir: tuple[Path, Callable[[str, str, str], Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, make = import_dir
    make("test.csv")

    def fail(
        *_args: object, **_kwargs: object
    ) -> None:  # pragma: no cover - should not be called
        raise AssertionError("Should not be called in dry run")

    monkeypatch.setattr(import_directory.data_update, "process_file_with_update", fail)
    monkeypatch.setattr(import_directory, "update_cell_dataset", fail)

    import_directory.import_directory(root, dry_run=True, workers=1)

    assert len(Sample._registry) == 0
    assert not (root / ".import_state.json").exists()


@pytest.mark.parallel
def test_include_exclude_patterns(
    import_dir: tuple[Path, Callable[[str, str, str], Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _make = import_dir
    keep_dir = root / "keep"
    skip_dir = root / "skip"
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
        root,
        workers=1,
        include=["*.csv", "*keep*"],
        exclude=["*.txt", "*skip*"],
    )

    assert calls == ["keep.csv"]


def test_config_file_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config file values are used when CLI options are absent."""

    from importlib import reload
    from battery_analysis.utils import config as config_mod
    from battery_analysis.utils import import_directory as import_directory_mod

    # Create configuration file in temporary home directory
    home = tmp_path / "home"
    home.mkdir()
    cfg = home / ".battery_analysis.toml"
    cfg.write_text('include = ["*.csv"]\nexclude = ["*skip*"]\n')

    # Patch Path.home and reload modules to pick up config
    monkeypatch.setattr(Path, "home", lambda: home)
    reload(config_mod)
    reload(import_directory_mod)

    root = tmp_path / "data"
    keep_dir = root / "keep"
    skip_dir = root / "skip"
    keep_dir.mkdir(parents=True)
    skip_dir.mkdir(parents=True)
    (keep_dir / "keep.csv").write_text("dummy")
    (skip_dir / "skip.csv").write_text("dummy")

    calls: list[str] = []

    def fake_process(path: str, sample: Sample) -> tuple[object, bool]:
        calls.append(Path(path).name)
        return object(), False

    monkeypatch.setattr(
        import_directory_mod.data_update, "process_file_with_update", fake_process
    )
    monkeypatch.setattr(import_directory_mod, "update_cell_dataset", lambda name: None)

    import_directory_mod.main([str(root), "--workers", "1"])

    assert calls == ["keep.csv"]

    # Restore modules to default state for other tests
    monkeypatch.undo()
    reload(config_mod)
    reload(import_directory_mod)


def test_import_job_and_rollback(
    import_dir: tuple[Path, Callable[[str, str, str], Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, make = import_dir
    file_path = make("test.csv")

    def fake_process(path: str, sample: Sample) -> tuple[TestResult, bool]:
        test = TestResult(sample=sample)
        setattr(test, "tester", "Other")
        setattr(test, "file_path", path)
        sample.tests.append(test)
        return test, False

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    import_directory.import_directory(root, workers=1)

    assert ImportJob._registry
    job = next(iter(ImportJob._registry.values()))
    assert job.end_time is not None
    assert job.files and job.files[0]["path"] == str(file_path)
    sample = Sample.get_by_name("S1")
    assert sample and sample.tests

    import_directory.rollback_job(job.id)
    assert Sample.get_by_name("S1").tests == []


def test_resume_skips_processed_files(
    import_dir: tuple[Path, Callable[[str, str, str], Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, make = import_dir
    f1 = make("first.csv")
    f2 = make("second.csv")

    processed: list[str] = []

    def fake_process(path: str, sample: Sample) -> tuple[object, bool]:
        processed.append(Path(path).name)
        return object(), False

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    job = ImportJob(processed_count=1, total_count=2, files=[{"path": str(f1)}])
    job.save()

    mtime = os.path.getmtime(f1)
    h = hashlib.md5(f1.read_bytes()).hexdigest()
    state = {str(f1.resolve()): {"mtime": mtime, "hash": h}}
    (root / ".import_state.json").write_text(json.dumps(state))

    import_directory.import_directory(root, workers=1, resume=str(job.id))

    assert processed == ["second.csv"]
    job = ImportJob.objects(id=job.id).first()
    assert job is not None
    assert job.processed_count == 2
    assert any(e["path"] == str(f2) for e in job.files)


def test_status_outputs_jobs(capsys: pytest.CaptureFixture[str]) -> None:
    job = ImportJob(processed_count=1, total_count=2, errors=["err"]).save()
    import_directory.show_status()
    out = capsys.readouterr().out
    assert str(job.id) in out
    assert "1/2" in out
    assert "err" in out


def test_preview_samples_skips_processing(
    import_dir: tuple[Path, Callable[[str, str, str], Path]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, make = import_dir
    path = make("a.mpt")

    called = False

    def fake_process(path: str, sample: Sample) -> tuple[object, bool]:
        nonlocal called
        called = True
        return object(), False

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )

    import_directory.import_directory(root, preview_samples=True, workers=1)

    assert not called
    out = capsys.readouterr().out
    assert str(path.resolve()) in out
    assert "S1" in out


def test_sample_map_overrides_names(
    import_dir: tuple[Path, Callable[[str, str, str], Path]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, make = import_dir
    f1 = make("a.mpt", sample="d1")

    map_path = tmp_path / "map.csv"
    map_path.write_text(f"file_path,sample\n{f1.resolve()},Custom\n")

    names: list[str] = []

    def fake_process(path: str, sample: Sample) -> tuple[object, bool]:
        if not path.endswith("map.csv"):
            names.append(sample.name)
        return object(), False

    monkeypatch.setattr(
        import_directory.data_update, "process_file_with_update", fake_process
    )
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    import_directory.import_directory(
        root,
        workers=1,
        sample_map=str(map_path),
        preview_samples=True,
        confirm=True,
    )

    assert names == ["Custom"]
    assert Sample.get_by_name("Custom") is not None
    assert Sample.get_by_name("d1") is None
