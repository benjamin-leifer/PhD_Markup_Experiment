"""Utilities for refreshing :class:`TestResult` records.

This module provides a :func:`refactor_tests` helper and a small CLI which can
be invoked via ``python -m battery_analysis.utils.refactor_data``.  The utility
recomputes common identifiers, archives raw data files and rewrites cycle
summaries so they conform to the current schema.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import time
from pathlib import Path
from typing import Iterable, List, Set

from battery_analysis.models import RefactorJob, TestResult
from battery_analysis.utils import file_storage
from battery_analysis.utils.cell_dataset_builder import update_cell_dataset
from battery_analysis.utils.data_update import _normalize_identifier, update_test_data
from battery_analysis.utils.db import ensure_connection

logger = logging.getLogger(__name__)

CONTROL_FILE = Path(__file__).resolve().parents[4] / ".refactor_control"


def _read_control_command() -> str | None:
    """Return the current command from the control file if it exists."""

    try:
        cmd = CONTROL_FILE.read_text(encoding="utf-8").strip().lower()
        return cmd or None
    except FileNotFoundError:
        return None


def _cycle_dicts(test: TestResult) -> List[dict[str, object]]:
    """Return ``test.cycles`` as plain dictionaries."""

    cycles: List[dict[str, object]] = []
    for cyc in getattr(test, "cycles", []) or []:
        if hasattr(cyc, "to_mongo"):
            cycles.append(cyc.to_mongo().to_dict())
        else:  # dataclass fallback used in tests
            data = {key: getattr(cyc, key) for key in vars(cyc)}
            cycles.append(data)
    return cycles


def refactor_tests(
    filter: dict[str, object] | None = None,
    *,
    dry_run: bool = False,
    batch_size: int = 50,
    job_id: str | None = None,
) -> RefactorJob:
    """Refresh ``TestResult`` documents matching ``filter``.

    Parameters
    ----------
    filter:
        Optional MongoEngine-style query parameters restricting the processed
        tests.
    dry_run:
        When ``True`` database writes are skipped.
    batch_size:
        Number of ``TestResult`` objects processed at once.
    job_id:
        Identifier of an existing :class:`RefactorJob` to resume.
    """

    ensure_connection()
    if job_id:
        job = RefactorJob.objects(id=job_id).first()
        if not job:
            raise ValueError(f"Unknown RefactorJob id {job_id}")
    else:
        job = RefactorJob(status="running")
        job.save()

    qs = TestResult.objects(**(filter or {}))
    total = qs.count()
    job.total_count = total
    job.status = "running"
    job.save()

    processed = job.processed_count
    errors: List[str] = list(job.errors or [])

    for start in range(processed, total, batch_size):
        batch = qs.skip(start).limit(batch_size)
        affected: Set[str] = set()
        logger.info(
            "Processing tests %d-%d of %d",
            start + 1,
            min(start + batch_size, total),
            total,
        )
        for test in batch:
            cmd = _read_control_command()
            if cmd == "cancel":
                logger.info("Refactor cancelled via control file")
                job.status = "cancelled"
                job.end_time = dt.datetime.utcnow()
                job.save()
                return job
            if cmd == "pause":
                logger.info("Refactor paused via control file")
                while True:
                    time.sleep(0.2)
                    cmd = _read_control_command()
                    if cmd == "cancel":
                        logger.info("Refactor cancelled via control file")
                        job.status = "cancelled"
                        job.end_time = dt.datetime.utcnow()
                        job.save()
                        return job
                    if cmd != "pause":
                        logger.info("Refactor resumed")
                        break
            try:
                if getattr(test, "name", None):
                    test.base_test_name = _normalize_identifier(test.name)
                if getattr(test, "file_path", None):
                    fname = os.path.basename(test.file_path)
                    test.base_file_name = _normalize_identifier(fname)
                    if not dry_run and os.path.exists(test.file_path):
                        file_storage.save_raw(test.file_path, test_result=test)

                cycles = _cycle_dicts(test)
                metadata: dict[str, object] = getattr(test, "metadata", {}) or {}
                if not dry_run:
                    update_test_data(test, cycles, metadata, strategy="replace")
                    test.save()

                if getattr(test, "cell_code", None):
                    affected.add(test.cell_code)
            except Exception as exc:  # pragma: no cover - best effort
                errors.append(str(exc))
            processed += 1
            job.current_test = str(getattr(test, "id", ""))
            job.processed_count = processed
            job.errors = errors
            job.save()

        if not dry_run:
            for code in affected:
                try:
                    update_cell_dataset(code)
                except Exception:  # pragma: no cover - best effort
                    pass

    job.end_time = dt.datetime.utcnow()
    job.status = "completed"
    job.save()
    return job


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", metavar="CODE", help="Process tests for a single sample"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without saving changes"
    )
    parser.add_argument("--job-id", help="Resume a previous refactor job")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    filt = {"cell_code": args.sample} if args.sample else None
    refactor_tests(filt, dry_run=args.dry_run, job_id=args.job_id)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
