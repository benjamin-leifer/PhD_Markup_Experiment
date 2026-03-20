import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "Python_Codes/BLeifer_Battery_Analysis")

from battery_analysis.models import Sample
from battery_analysis.utils import import_directory as import_directory_module


def _reset_model_state() -> None:
    if hasattr(Sample, "_registry"):
        Sample._registry.clear()
    if hasattr(import_directory_module.ImportJob, "_registry"):
        import_directory_module.ImportJob._registry.clear()
    if hasattr(import_directory_module.ImportJobSummary, "_registry"):
        import_directory_module.ImportJobSummary._registry.clear()


class _SavedTestResult:
    def __init__(self, test_id: str):
        self.id = test_id


class _DummySample:
    def __init__(self, name: str):
        self.name = name
        self.tags: list[str] = []

    def save(self):
        return self


def test_import_directory_skips_parser_metadata_without_sample_lookup(tmp_path, monkeypatch):
    _reset_model_state()
    data_dir = tmp_path / "batch_a"
    data_dir.mkdir()
    (data_dir / "run.csv").write_text("cycle,data\n1,2\n", encoding="utf-8")

    parse_calls: list[str] = []
    monkeypatch.setattr(import_directory_module.parsers, "get_supported_formats", lambda: [".csv"])
    monkeypatch.setattr(import_directory_module.parsers, "parse_file", lambda path: parse_calls.append(path))
    monkeypatch.setattr(import_directory_module, "ensure_connection", lambda **kwargs: True)

    result = import_directory_module.import_directory(
        str(tmp_path),
        dry_run=True,
        sample_lookup=False,
        workers=2,
    )

    assert result == 0
    assert parse_calls == []


def test_import_directory_overlaps_discovery_and_processing(tmp_path, monkeypatch):
    _reset_model_state()
    first_dir = tmp_path / "batch_one"
    second_dir = tmp_path / "batch_two"
    first_dir.mkdir()
    second_dir.mkdir()
    first_file = first_dir / "first.csv"
    second_file = second_dir / "second.csv"
    first_file.write_text("a", encoding="utf-8")
    second_file.write_text("b", encoding="utf-8")

    process_started = threading.Event()
    overlap_seen: list[bool] = []

    monkeypatch.setattr(import_directory_module.parsers, "get_supported_formats", lambda: [".csv"])
    monkeypatch.setattr(import_directory_module, "ensure_connection", lambda **kwargs: True)
    monkeypatch.setattr(import_directory_module, "update_cell_dataset", lambda name: None)
    monkeypatch.setattr(
        import_directory_module.Sample,
        "get_or_create",
        classmethod(lambda cls, name, **attrs: _DummySample(name)),
    )

    def fake_process(path, sample, **kwargs):
        process_started.set()
        time.sleep(0.05)
        return _SavedTestResult(Path(path).stem), False

    def fake_walk(root):
        yield (str(first_dir), [], [first_file.name])
        time.sleep(0.2)
        overlap_seen.append(process_started.is_set())
        yield (str(second_dir), [], [second_file.name])

    monkeypatch.setattr(import_directory_module, "process_file_with_update", fake_process)
    monkeypatch.setattr(import_directory_module.os, "walk", fake_walk)

    result = import_directory_module.import_directory(str(tmp_path), workers=1)

    assert result == 0
    assert overlap_seen == [True]
