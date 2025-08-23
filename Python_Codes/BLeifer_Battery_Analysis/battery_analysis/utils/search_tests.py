"""Search for :class:`TestResult` records with simple filters.

The module exposes a small command line interface that queries
``TestResult`` documents in MongoDB. Matching records are printed as a
plain table containing the test identifier, the associated sample and a
few key metrics.

Usage examples::

    # List tests for a particular sample
    python -m battery_analysis.utils.search_tests --sample S1

    # Filter by chemistry and restrict the date range
    python -m battery_analysis.utils.search_tests \
        --chemistry NMC --date-range 2024-01-01:2024-06-30
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Iterable, List

from battery_analysis.models import Sample, TestResult
from battery_analysis.utils.db import ensure_connection

DATE_FMT = "%Y-%m-%d"


def _parse_date_range(value: str) -> tuple[dt.datetime, dt.datetime]:
    """Return ``(start, end)`` datetimes from ``YYYY-MM-DD:YYYY-MM-DD``."""
    try:
        start_str, end_str = value.split(":", 1)
        start = dt.datetime.fromisoformat(start_str)
        end = dt.datetime.fromisoformat(end_str)
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"Invalid date range '{value}'"
        raise argparse.ArgumentTypeError(msg) from exc
    return start, end


def build_parser() -> argparse.ArgumentParser:
    """Return the :mod:`argparse` parser used for the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", help="Filter by Sample name")
    parser.add_argument("--chemistry", help="Filter by Sample chemistry")
    parser.add_argument(
        "--date-range",
        help="Inclusive date span 'YYYY-MM-DD:YYYY-MM-DD'",
    )
    return parser


def _format_float(value: float | None) -> str:
    """Return ``value`` formatted to three decimals or ``""``."""

    return f"{value:.3f}" if isinstance(value, (int, float)) else ""


def _fetch_sample(test: TestResult) -> Sample | None:
    """Return the Sample referenced by ``test`` if available."""

    try:
        if hasattr(test.sample, "fetch"):
            return test.sample.fetch()
        return test.sample
    except Exception:  # pragma: no cover - best effort
        return None


def _collect_rows(results: Iterable[TestResult]) -> List[List[str]]:
    rows: List[List[str]] = []
    for test in results:
        sample = _fetch_sample(test)
        sample_name = getattr(sample, "name", "")
        chemistry = getattr(sample, "chemistry", "")
        date = getattr(test, "date", None)
        if isinstance(date, dt.datetime):
            date_str = date.strftime(DATE_FMT)
        else:
            date_str = ""
        rows.append(
            [
                str(getattr(test, "id", "")),
                sample_name,
                chemistry,
                date_str,
                _format_float(getattr(test, "initial_capacity", None)),
                _format_float(getattr(test, "final_capacity", None)),
                _format_float(getattr(test, "capacity_retention", None)),
            ]
        )
    return rows


def _render_table(rows: List[List[str]]) -> str:
    headers = [
        "ID",
        "Sample",
        "Chemistry",
        "Date",
        "InitCap",
        "FinalCap",
        "Retention",
    ]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = " ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers)]
    for row in rows:
        lines.append(fmt.format(*row))
    return "\n".join(lines)


def query_tests(
    sample: str | None = None,
    chemistry: str | None = None,
    date_range: str | None = None,
) -> List[TestResult]:
    """Return ``TestResult`` objects matching the supplied filters."""

    qs = TestResult.objects
    if sample:
        samp = Sample.objects(name=sample).first()
        if not samp:
            return []
        qs = qs.filter(sample=samp)
    if chemistry:
        # fmt: off
        sample_ids = [
            s.id
            for s in Sample.objects(chemistry=chemistry).only("id")
        ]
        # fmt: on
        if not sample_ids:
            return []
        qs = qs.filter(sample__in=sample_ids)
    if date_range:
        start, end = _parse_date_range(date_range)
        qs = qs.filter(date__gte=start, date__lte=end)
    return list(qs)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_connection()
    results = query_tests(args.sample, args.chemistry, args.date_range)
    if not results:
        print("No matching tests found.")
        return
    rows = _collect_rows(results)
    print(_render_table(rows))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
