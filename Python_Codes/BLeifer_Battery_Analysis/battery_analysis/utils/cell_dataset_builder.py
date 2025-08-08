"""Helpers for constructing :class:`CellDataset` documents.

These utilities aggregate multiple :class:`~battery_analysis.models.TestResult`
objects for a given cell code into a single :class:`~battery_analysis.models.CellDataset`.
"""

from __future__ import annotations

from typing import Iterable, List

from battery_analysis.models import CellDataset, TestResult, CycleSummary


# ---------------------------------------------------------------------------
# Data gathering and merging
# ---------------------------------------------------------------------------


def gather_tests(cell_code: str) -> List[TestResult]:
    """Return all tests for ``cell_code`` sorted by ``date``.

    Parameters
    ----------
    cell_code:
        Identifier of the cell whose tests should be gathered.
    """

    if not cell_code:
        return []
    try:
        qs = TestResult.objects(cell_code=cell_code).order_by("date")  # type: ignore[attr-defined]
    except Exception:
        return []
    return list(qs)


def merge_tests(tests: Iterable[TestResult]):
    """Concatenate cycle summaries from ``tests`` with sequential indices.

    Each cycle's ``cycle_index`` is updated so that the merged list has
    continuous numbering starting at 1.
    """

    combined: List[CycleSummary] = []
    next_idx = 1
    for test in tests:
        for cyc in getattr(test, "cycles", []) or []:
            new_cyc = CycleSummary(**cyc.to_mongo().to_dict())
            new_cyc.cycle_index = next_idx
            combined.append(new_cyc)
            next_idx += 1
    return combined


# ---------------------------------------------------------------------------
# CellDataset construction
# ---------------------------------------------------------------------------


def update_cell_dataset(cell_code: str) -> CellDataset | None:
    """Build or refresh the :class:`CellDataset` for ``cell_code``.

    The dataset is created if absent; otherwise its tests and combined
    cycles are rebuilt from all available :class:`TestResult` records.
    """

    tests = gather_tests(cell_code)
    if not tests:
        return None

    dataset = CellDataset.objects(cell_code=cell_code).first()
    if not dataset:
        dataset = CellDataset.build_from_tests(tests)
        dataset.combined_cycles = merge_tests(tests)
        dataset.save()
        sample = dataset.sample
    else:
        dataset.tests = tests
        dataset.combined_cycles = merge_tests(tests)

        sample_ref = tests[0].sample
        sample = sample_ref.fetch() if hasattr(sample_ref, "fetch") else sample_ref
        dataset.sample = sample
        dataset.save()

    if getattr(sample, "default_dataset", None) != dataset:
        sample.default_dataset = dataset
        try:
            sample.save()
        except Exception:
            pass
    return dataset
