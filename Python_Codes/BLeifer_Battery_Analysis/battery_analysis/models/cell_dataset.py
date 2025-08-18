# battery_analysis/models/cell_dataset.py

"""Aggregated dataset combining multiple test results for a single cell."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, cast

from mongoengine import Document, fields

try:
    from .cycle_summary import CycleSummary
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    CycleSummary = importlib.import_module("cycle_summary").CycleSummary

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .sample import Sample
    from .test_result import TestResult


class CellDataset(Document):  # type: ignore[misc]
    """Represents a collection of tests aggregated for a given cell code."""

    cell_code = fields.StringField(required=True, unique=True)
    sample = fields.ReferenceField("Sample", required=True)
    tests = fields.ListField(fields.LazyReferenceField("TestResult"), default=list)
    combined_cycles = fields.ListField(
        fields.EmbeddedDocumentField(CycleSummary), default=list
    )

    meta = {"collection": "cell_datasets", "indexes": ["cell_code"]}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @classmethod
    def get_by_cell_code(cls, code: str) -> "CellDataset | None":
        """Return the dataset for ``code`` or ``None`` if missing."""
        return cast("CellDataset | None", cls.objects(cell_code=code).first())

    @classmethod
    def get_or_create(cls, code: str, sample: "Sample", **attrs: Any) -> "CellDataset":
        """Retrieve a dataset by ``code`` or create and save a new one."""
        dataset = cls.get_by_cell_code(code)
        if dataset is None:
            dataset = cls(cell_code=code, sample=sample, **attrs)
            dataset.save()
        return dataset

    @classmethod
    def build_from_tests(cls, test_results: Iterable["TestResult"]) -> "CellDataset":
        """Construct a dataset from an iterable of :class:`TestResult` objects."""
        tests = list(test_results)
        if not tests:
            raise ValueError("test_results must not be empty")

        first = tests[0]
        sample_ref = first.sample
        if hasattr(sample_ref, "fetch") and getattr(sample_ref, "id", None):
            sample = sample_ref.fetch()
        else:
            sample = sample_ref
        dataset = cls(cell_code=first.cell_code, sample=sample)
        for tr in tests:
            dataset.append_test(tr)
        try:  # ensure dataset has an id for reference fields
            dataset.save()
        except Exception:  # pragma: no cover - saving may fail without DB
            pass
        if hasattr(sample, "default_dataset"):
            sample.default_dataset = dataset
            try:  # persist link if possible
                sample.save()
            except Exception:  # pragma: no cover
                pass
        return dataset

    def append_test(self, test_result: "TestResult") -> "CellDataset":
        """Add a test result and extend the combined cycle summaries."""
        self.tests.append(test_result)
        cycles = getattr(test_result, "cycles", None) or []
        self.combined_cycles.extend(cycles)
        return self
