"""Model definitions for the battery_analysis package.

This module attempts to import the real MongoEngine based models. If
MongoEngine is not installed (for example when running the lightweight unit
tests in this repository), simple dataclass based stand-ins are provided so
that the rest of the package can be imported without requiring MongoDB or
MongoEngine.
"""

from __future__ import annotations

# mypy: ignore-errors

# Try to import the real MongoEngine models
try:  # pragma: no cover - behaviour depends on environment
    from mongoengine import (  # type: ignore  # noqa: F401
        CASCADE,
        Document,
        EmbeddedDocument,
        fields,
    )

    try:
        from .cycle_summary import CycleSummary  # type: ignore
        from .sample import Sample  # type: ignore

        # Convenience export to create or fetch samples by name
        get_or_create_sample = Sample.get_or_create  # type: ignore
        from .cell_dataset import CellDataset  # type: ignore
        from .experiment_plan import ExperimentPlan  # type: ignore
        from .import_job import ImportJob, ImportJobSummary  # type: ignore
        from .raw_file import RawDataFile  # type: ignore
        from .refactor_job import RefactorJob  # type: ignore
        from .stages import CathodeMaterial, Cell, Electrode, Slurry, inherit_metadata
        from .test_protocol import TestProtocol  # type: ignore
        from .testresult import CycleDetailData, TestResult  # type: ignore
    except ImportError:  # pragma: no cover - allow running as script
        import importlib

        CycleSummary = importlib.import_module(
            ".cycle_summary", __name__
        ).CycleSummary  # type: ignore
        Sample = importlib.import_module(".sample", __name__).Sample  # type: ignore
        # Convenience export to create or fetch samples by name
        get_or_create_sample = Sample.get_or_create  # type: ignore
        testresult_mod = importlib.import_module(".testresult", __name__)
        TestResult = testresult_mod.TestResult  # type: ignore
        CycleDetailData = testresult_mod.CycleDetailData  # type: ignore
        RawDataFile = importlib.import_module(
            ".raw_file", __name__
        ).RawDataFile  # type: ignore
        RefactorJob = importlib.import_module(
            ".refactor_job", __name__
        ).RefactorJob  # type: ignore
        TestProtocol = importlib.import_module(
            ".test_protocol", __name__
        ).TestProtocol  # type: ignore
        CellDataset = importlib.import_module(
            ".cell_dataset", __name__
        ).CellDataset  # type: ignore
        mod = importlib.import_module(".import_job", __name__)
        ImportJob = mod.ImportJob  # type: ignore
        ImportJobSummary = mod.ImportJobSummary  # type: ignore
        stages_mod = importlib.import_module(".stages", __name__)
        CathodeMaterial = stages_mod.CathodeMaterial  # type: ignore
        Slurry = stages_mod.Slurry  # type: ignore
        Electrode = stages_mod.Electrode  # type: ignore
        Cell = stages_mod.Cell  # type: ignore
        inherit_metadata = stages_mod.inherit_metadata  # type: ignore

    __all__ = [
        "Sample",
        "get_or_create_sample",
        "TestResult",
        "CycleSummary",
        "RawDataFile",
        "CycleDetailData",
        "TestProtocol",
        "CellDataset",
        "ExperimentPlan",
        "ImportJob",
        "ImportJobSummary",
        "RefactorJob",
        "CathodeMaterial",
        "Slurry",
        "Electrode",
        "Cell",
        "inherit_metadata",
    ]
except Exception:  # pragma: no cover - executed when mongoengine is missing
    import logging
    import traceback

    logging.exception("Failed to import MongoEngine models")
    traceback.print_exc()

    # Provide very small dataclass implementations used in tests
    import datetime
    import uuid
    from dataclasses import dataclass
    from dataclasses import field as dc_field
    from typing import ClassVar

    @dataclass
    class CycleSummary:  # type: ignore
        cycle_index: int
        charge_capacity: float
        discharge_capacity: float
        coulombic_efficiency: float

    def inherit_metadata(obj):  # type: ignore
        merged: dict = {}
        current = obj
        while current is not None:
            data = getattr(current, "metadata", {}) or {}
            merged = {**data, **merged}
            current = getattr(current, "parent", None)
        return merged

    @dataclass
    class CathodeMaterial:  # type: ignore
        name: str
        composition: str | None = None
        manufacturer: str | None = None
        metadata: dict = dc_field(default_factory=dict)
        parent: "CathodeMaterial | None" = None

        @classmethod
        def from_parent(cls, parent: "CathodeMaterial | None" = None, **kwargs):
            obj = cls(parent=parent, **kwargs)
            obj.metadata = inherit_metadata(obj)
            return obj

    @dataclass
    class Slurry:  # type: ignore
        parent: CathodeMaterial
        solids_content: float | None = None
        mixing_time: float | None = None
        metadata: dict = dc_field(default_factory=dict)

        @classmethod
        def from_parent(cls, parent: CathodeMaterial, **kwargs):
            obj = cls(parent=parent, **kwargs)
            obj.metadata = inherit_metadata(obj)
            return obj

    @dataclass
    class Electrode:  # type: ignore
        parent: Slurry
        loading: float | None = None
        thickness: float | None = None
        metadata: dict = dc_field(default_factory=dict)

        @classmethod
        def from_parent(cls, parent: Slurry, **kwargs):
            obj = cls(parent=parent, **kwargs)
            obj.metadata = inherit_metadata(obj)
            return obj

    @dataclass
    class Cell:  # type: ignore
        parent: Electrode
        format: str | None = None
        nominal_capacity: float | None = None
        metadata: dict = dc_field(default_factory=dict)

        @classmethod
        def from_parent(cls, parent: Electrode, **kwargs):
            obj = cls(parent=parent, **kwargs)
            obj.metadata = inherit_metadata(obj)
            return obj

    @dataclass
    class TestResult:  # type: ignore
        parent: Cell | None = None
        sample: Sample | None = None
        initial_capacity: float = 0.0
        final_capacity: float = 0.0
        capacity_retention: float = 0.0
        avg_coulombic_eff: float = 0.0
        metadata: dict = dc_field(default_factory=dict)
        last_cycle_complete: bool | None = None
        c_rates: list = dc_field(default_factory=list)
        protocol: "TestProtocol | None" = None
        tags: list[str] = dc_field(default_factory=list)
        created_by: str | None = None
        last_modified_by: str | None = None
        notes_log: list = dc_field(default_factory=list)

        @classmethod
        def from_parent(cls, parent: Cell, **kwargs):  # type: ignore
            obj = cls(parent=parent, **kwargs)
            obj.metadata = inherit_metadata(obj)
            return obj

    @dataclass
    class Sample:  # type: ignore
        name: str
        tests: list = dc_field(default_factory=list)
        nominal_capacity: float | None = None
        avg_initial_capacity: float | None = None
        avg_final_capacity: float | None = None
        avg_capacity_retention: float | None = None
        avg_coulombic_eff: float | None = None
        avg_energy_efficiency: float | None = None
        median_internal_resistance: float | None = None
        parent: "Sample | None" = None
        anode: "Sample | None" = None
        cathode: "Sample | None" = None
        separator: "Sample | None" = None
        electrolyte: "Sample | None" = None
        tags: list[str] = dc_field(default_factory=list)
        created_by: str | None = None
        last_modified_by: str | None = None
        notes_log: list = dc_field(default_factory=list)
        default_dataset: "CellDataset | None" = None

        _registry: ClassVar[dict[str, "Sample"]] = {}

        @classmethod
        def get_by_name(cls, name: str) -> "Sample | None":
            return cls._registry.get(name)

        def save(self) -> "Sample":
            self.__class__._registry[self.name] = self
            return self

        @classmethod
        def get_or_create(cls, name: str, **attrs) -> "Sample":
            sample = cls.get_by_name(name)
            if sample is None:
                sample = cls(name=name, **attrs)
                sample.save()
            return sample

    @dataclass
    class CellDataset:  # type: ignore
        cell_code: str
        sample: Sample | None = None
        tests: list = dc_field(default_factory=list)
        combined_cycles: list = dc_field(default_factory=list)

        @classmethod
        def build_from_tests(cls, test_results):
            tests = list(test_results)
            if not tests:
                raise ValueError("test_results must not be empty")
            dataset = cls(
                cell_code=getattr(tests[0], "cell_code", None),
                sample=getattr(tests[0], "sample", None),
            )
            for tr in tests:
                dataset.append_test(tr)
            if dataset.sample is not None:
                dataset.sample.default_dataset = dataset
            return dataset

        def append_test(self, test_result):
            self.tests.append(test_result)
            cycles = getattr(test_result, "cycles", None) or []
            self.combined_cycles.extend(cycles)
            return self

    @dataclass
    class RawDataFile:  # type: ignore
        filename: str
        file_data: bytes | None = None
        file_type: str | None = None
        upload_date: datetime.datetime | None = None
        test_result: "TestResult | None" = None
        sample: "Sample | None" = None
        operator: str | None = None
        acquisition_device: str | None = None
        tags: list[str] = dc_field(default_factory=list)
        metadata: dict = dc_field(default_factory=dict)
        source_path: str | None = None
        import_job_id: str | None = None

    class CycleDetailData:  # type: ignore
        pass

    @dataclass
    class TestProtocol:  # type: ignore
        name: str
        summary: str
        c_rates: list = dc_field(default_factory=list)

    @dataclass
    class ExperimentPlan:  # type: ignore
        name: str
        factors: dict
        matrix: list
        sample_ids: list = dc_field(default_factory=list)

        _registry: ClassVar[dict[str, "ExperimentPlan"]] = {}

        @classmethod
        def get_by_name(cls, name: str):
            return cls._registry.get(name)

        def save(self) -> "ExperimentPlan":
            self.__class__._registry[self.name] = self
            return self

    @dataclass
    class ImportJob:  # type: ignore
        id: str = dc_field(default_factory=lambda: str(uuid.uuid4()))
        start_time: datetime.datetime = dc_field(
            default_factory=datetime.datetime.utcnow
        )
        end_time: datetime.datetime | None = None
        files: list = dc_field(default_factory=list)
        errors: list[str] = dc_field(default_factory=list)
        current_file: str | None = None
        processed_count: int = 0
        total_count: int = 0

        _registry: ClassVar[dict[str, "ImportJob"]] = {}

        def save(self) -> "ImportJob":
            self.__class__._registry[self.id] = self
            return self

        @classmethod
        def objects(cls, **query):
            class _Q(list):
                def first(self):
                    return self[0] if self else None

            if "id" in query:
                obj = cls._registry.get(str(query["id"]))
                return _Q([obj] if obj else [])
            return _Q(cls._registry.values())

        def delete(self) -> None:
            self.__class__._registry.pop(self.id, None)

    @dataclass
    class ImportJobSummary:  # type: ignore
        id: str = dc_field(default_factory=lambda: str(uuid.uuid4()))
        start_time: datetime.datetime = dc_field(
            default_factory=datetime.datetime.utcnow
        )
        end_time: datetime.datetime | None = None
        total_count: int = 0
        processed_count: int = 0
        created_count: int = 0
        updated_count: int = 0
        skipped_count: int = 0
        errors: list[str] = dc_field(default_factory=list)
        status: str = "running"

        _registry: ClassVar[dict[str, "ImportJobSummary"]] = {}

        def save(self) -> "ImportJobSummary":
            self.__class__._registry[self.id] = self
            return self

        @classmethod
        def objects(cls, **query):
            class _Q(list):
                def first(self):
                    return self[0] if self else None

                def count(self):
                    return len(self)

            if "id" in query:
                obj = cls._registry.get(str(query["id"]))
                return _Q([obj] if obj else [])
            return _Q(cls._registry.values())

    @dataclass
    class RefactorJob:  # type: ignore
        id: str = dc_field(default_factory=lambda: str(uuid.uuid4()))
        start_time: datetime.datetime = dc_field(
            default_factory=datetime.datetime.utcnow
        )
        end_time: datetime.datetime | None = None
        current_test: str | None = None
        processed_count: int = 0
        total_count: int = 0
        errors: list[str] = dc_field(default_factory=list)
        status: str = "running"

        _registry: ClassVar[dict[str, "RefactorJob"]] = {}

        def save(self) -> "RefactorJob":
            self.__class__._registry[self.id] = self
            return self

        @classmethod
        def objects(cls, **query):
            class _Q(list):
                def first(self):
                    return self[0] if self else None

                def count(self):  # pragma: no cover - simple container
                    return len(self)

            if "id" in query:
                obj = cls._registry.get(str(query["id"]))
                return _Q([obj] if obj else [])
            return _Q(cls._registry.values())

        def delete(self) -> None:
            self.__class__._registry.pop(self.id, None)

    # Convenience export to create or fetch samples by name
    get_or_create_sample = Sample.get_or_create

    __all__ = [
        "Sample",
        "get_or_create_sample",
        "TestResult",
        "CycleSummary",
        "RawDataFile",
        "CycleDetailData",
        "TestProtocol",
        "CellDataset",
        "ExperimentPlan",
        "ImportJob",
        "ImportJobSummary",
        "RefactorJob",
        "CathodeMaterial",
        "Slurry",
        "Electrode",
        "Cell",
        "inherit_metadata",
    ]
