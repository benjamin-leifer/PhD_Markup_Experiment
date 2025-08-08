"""Helpers for constructing :class:`CellDataset` documents.

These utilities aggregate multiple :class:`~battery_analysis.models.TestResult`
objects for a given cell code into a single :class:`~battery_analysis.models.CellDataset`.
"""

from __future__ import annotations

from typing import Iterable, List

import logging

from battery_analysis.models import CellDataset, TestResult, CycleSummary


# ---------------------------------------------------------------------------
# Data gathering and merging
# ---------------------------------------------------------------------------


logger = logging.getLogger(__name__)


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


def merge_tests(
    tests: Iterable[TestResult], start_index: int = 1
) -> List[CycleSummary]:
    """Concatenate cycle summaries from ``tests`` with sequential indices.

    Parameters
    ----------
    tests:
        Iterable of :class:`TestResult` instances whose cycles should be
        concatenated.
    start_index:
        Starting value for the sequential ``cycle_index``. Defaults to ``1``.

    Each cycle's ``cycle_index`` is updated so that the merged list has
    continuous numbering beginning at ``start_index``.
    """

    combined: List[CycleSummary] = []
    next_idx = start_index
    for test in tests:
        cycles = getattr(test, "cycles", []) or []
        logger.info(
            "Merging %s with %d cycles", getattr(test, "name", "<unknown>"), len(cycles)
        )
        for cyc in cycles:
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

    A dataset is created if absent; otherwise any newly available tests are
    appended and the combined cycles re-indexed accordingly.
    """

    tests = gather_tests(cell_code)
    if not tests:
        return None

    dataset = CellDataset.objects(cell_code=cell_code).first()
    if not dataset:
        dataset = CellDataset.build_from_tests(tests)
        dataset.combined_cycles = merge_tests(tests)
        sample = dataset.sample
    else:
        existing_ids = {t.id for t in dataset.tests}
        new_tests = [t for t in tests if t.id not in existing_ids]
        if new_tests:
            logger.info(
                "Appending %d new tests to dataset %s: %s",
                len(new_tests),
                dataset.id,
                [getattr(t, "name", str(t.id)) for t in new_tests],
            )
            dataset.tests.extend(new_tests)
            start = len(dataset.combined_cycles) + 1
            dataset.combined_cycles.extend(merge_tests(new_tests, start_index=start))
        sample_ref = tests[0].sample
        sample = sample_ref.fetch() if hasattr(sample_ref, "fetch") else sample_ref
        dataset.sample = sample

    dataset.tests.sort(key=lambda t: t.date)
    dataset.save()

    if getattr(sample, "default_dataset", None) != dataset:
        sample.default_dataset = dataset
        try:
            sample.save()
        except Exception:
            pass
    return dataset
