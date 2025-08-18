from __future__ import annotations

from pathlib import Path
import sys
import types
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mongomock  # noqa: E402
from mongoengine import connect, disconnect  # noqa: E402
for name in [
    "battery_analysis.analysis",
    "battery_analysis.report",
    "battery_analysis.plots",
    "battery_analysis.eis",
    "battery_analysis.outlier_analysis",
    "battery_analysis.advanced_analysis",
]:
    if name not in sys.modules:
        stub = types.ModuleType(name)
        if name == "battery_analysis.analysis":
            stub.compute_metrics = lambda cycles: {}
            stub.update_sample_properties = lambda sample, test: None
            stub.create_test_result = (
                lambda sample, cycles, tester="Other", metadata=None: None
            )
            stub.summarize_detailed_cycles = lambda test, detailed: None
        sys.modules[name] = stub

from battery_analysis.models import Sample, TestResult, CycleSummary  # noqa: E402
from battery_analysis.utils import import_directory  # noqa: E402
from battery_analysis import parsers  # noqa: E402


def _make_file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / "S1" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("dummy")
    return path


def _setup_db() -> None:
    disconnect()
    connect("import_test", mongo_client_class=mongomock.MongoClient, alias="default")


@pytest.fixture
def fake_processing(monkeypatch: pytest.MonkeyPatch):
    processed: list[str] = []
    tests: dict[tuple[str, str], TestResult] = {}

    def fake_process(file_path: str, sample: Sample):
        processed.append(Path(file_path).name)
        cycles, _ = parsers.parse_file(file_path)
        key = (sample.name, Path(file_path).stem.split("_Wb_")[0])
        test = tests.get(key)
        was_update = False
        if test is None:
            test = TestResult(
                sample=sample,
                tester="Other",
                name=Path(file_path).name,
                file_path=str(file_path),
                cycles=[],
            )
            test.save()
            tests[key] = test
        else:
            was_update = True
        for c in cycles:
            test.cycles.append(CycleSummary(**c))
        test.save()
        return test, was_update

    monkeypatch.setattr(import_directory.data_update, "process_file_with_update", fake_process)
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)
    return processed


def test_new_file_creates_testresult(tmp_path: Path, fake_processing) -> None:
    _setup_db()
    _make_file(tmp_path, "test.csv")

    import_directory.import_directory(tmp_path)

    assert Sample.objects.count() == 1
    assert TestResult.objects.count() == 1
    disconnect()


def test_sequential_files_append_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_processing
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


def test_parse_error_logged_and_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _setup_db()
    _make_file(tmp_path, "bad.csv")

    def boom(*args, **kwargs):
        raise ValueError("parse boom")

    monkeypatch.setattr(import_directory.data_update, "process_file_with_update", boom)
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    with caplog.at_level("ERROR"):
        import_directory.import_directory(tmp_path)

    assert "Failed to process" in caplog.text
    assert TestResult.objects.count() == 0
    disconnect()


def test_dry_run_leaves_db_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_db()
    _make_file(tmp_path, "dry.csv")

    def boom(*args, **kwargs):
        raise AssertionError("should not be called")

    monkeypatch.setattr(import_directory.data_update, "process_file_with_update", boom)
    monkeypatch.setattr(Sample, "get_or_create", classmethod(boom))
    monkeypatch.setattr(import_directory, "update_cell_dataset", lambda name: None)

    import_directory.import_directory(tmp_path, dry_run=True)

    assert Sample.objects.count() == 0
    assert TestResult.objects.count() == 0
    disconnect()


def test_include_exclude_filters(
    tmp_path: Path, fake_processing
) -> None:
    _setup_db()
    _make_file(tmp_path, "keep.csv")
    _make_file(tmp_path, "skip.csv")

    import_directory.import_directory(
        tmp_path, include=["*keep*"], exclude=["*skip*"]
    )

    assert Sample.objects.count() == 1
    assert TestResult.objects.count() == 1
    disconnect()


def test_resume_from_manifest(
    tmp_path: Path, fake_processing
) -> None:
    _setup_db()
    f1 = _make_file(tmp_path, "first.csv")
    f2 = _make_file(tmp_path, "second.csv")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([str(f1.relative_to(tmp_path))]))

    import_directory.import_directory(tmp_path, resume_manifest=str(manifest))

    assert Sample.objects.count() == 1
    assert TestResult.objects.count() == 1
    data = json.loads(manifest.read_text())
    assert set(data) == {str(f1.relative_to(tmp_path)), str(f2.relative_to(tmp_path))}
    disconnect()
