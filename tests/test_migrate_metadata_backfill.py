from __future__ import annotations

from dataclasses import dataclass, field

from battery_analysis.utils.migrate_metadata_backfill import (
    derive_raw_data_file_updates,
    migrate_raw_data_files,
)


@dataclass
class FakeSample:
    name: str


@dataclass
class FakeTestResult:
    id: str
    sample: FakeSample | None = None
    tester: str | None = None
    name: str | None = None
    file_path: str | None = None
    tags: list[str] = field(default_factory=list)
    created_by: str | None = None


@dataclass
class FakeRawDataFile:
    filename: str
    test_result: FakeTestResult | None = None
    sample: FakeSample | None = None
    source_path: str | None = None
    operator: str | None = None
    acquisition_device: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    save_calls: int = 0

    def save(self) -> None:
        self.save_calls += 1


def test_raw_data_file_backfill_derives_values_from_test_and_path() -> None:
    sample = FakeSample(name="CELL-01")
    test = FakeTestResult(
        id="TR-1",
        sample=sample,
        tester="BioLogic",
        name="Formation Run",
        file_path="/imports/operators/alice/CELL-01/run1.mpt",
        tags=["formation", "cycler"],
        created_by="alice",
    )
    raw = FakeRawDataFile(
        filename="run1.mpt",
        test_result=test,
        source_path="/imports/operators/alice/CELL-01/run1.mpt",
        metadata={"notes": "kept"},
    )

    updates = derive_raw_data_file_updates(raw, sample_lookup=lambda name: sample)

    assert updates["sample"] is sample
    assert updates["operator"] == "alice"
    assert updates["acquisition_device"] == "BioLogic"
    assert updates["tags"] == ["formation", "cycler"]
    assert updates["metadata"] == {
        "notes": "kept",
        "filename": "run1.mpt",
        "file_path": "/imports/operators/alice/CELL-01/run1.mpt",
        "source_path": "/imports/operators/alice/CELL-01/run1.mpt",
        "test_result_id": "TR-1",
        "test_name": "Formation Run",
        "tester": "BioLogic",
        "sample_name": "CELL-01",
        "sample_code": "CELL-01",
        "operator": "alice",
        "acquisition_device": "BioLogic",
        "tags": ["formation", "cycler"],
    }


def test_raw_data_file_migration_is_idempotent_and_dry_run_safe() -> None:
    sample = FakeSample(name="CELL-02")
    test = FakeTestResult(id="TR-2", sample=sample, tester="Arbin", tags=["rate"])
    raw = FakeRawDataFile(
        filename="run2.csv",
        test_result=test,
        source_path="/imports/CELL-02/run2.csv",
    )

    dry_run_counts = migrate_raw_data_files(
        [raw], sample_lookup=lambda name: sample, dry_run=True
    )
    assert dry_run_counts.matched == 1
    assert dry_run_counts.changed == 1
    assert raw.save_calls == 0
    assert raw.sample is None

    apply_counts = migrate_raw_data_files(
        [raw], sample_lookup=lambda name: sample, dry_run=False
    )
    assert apply_counts.changed == 1
    assert raw.save_calls == 1
    assert raw.sample is sample
    assert raw.acquisition_device == "Arbin"
    assert raw.tags == ["rate"]

    rerun_counts = migrate_raw_data_files(
        [raw], sample_lookup=lambda name: sample, dry_run=False
    )
    assert rerun_counts.matched == 0
    assert rerun_counts.changed == 0
    assert raw.save_calls == 1
