"""Audit raw test files against stored hashes in MongoDB and GridFS.

This utility walks a directory tree, computes the SHA256 hash for each file
and compares it with :class:`battery_analysis.models.TestResult` entries and
archived raw files stored in GridFS.  Discrepancies are reported in a summary
table that can be emitted as JSON or CSV.

Example
-------
Run directly as a module to audit a directory::

    python -m battery_analysis.utils.verify_import DATA_DIR --format json

The command exits with a non-zero status code if any files are missing from the
database, missing on disk, or have mismatched hashes.
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Iterable

from battery_analysis.models import TestResult, Sample
from battery_analysis.utils.db import ensure_connection
from battery_analysis.utils import file_storage


def _sha256(path: Path) -> str:
    """Return the SHA256 hex digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_directory(root: str) -> List[Dict[str, str]]:
    """Verify ``root`` against ``TestResult`` records and GridFS archives.

    Parameters
    ----------
    root:
        Directory containing raw data files.

    Returns
    -------
    list of dict
        Each dict describes a discrepancy with keys ``path``, ``status``,
        ``expected_hash``, ``actual_hash``, ``gridfs_hash`` and ``test_id``
        (if applicable).  The list is empty when no discrepancies are found.
    """

    ensure_connection()
    root_path = Path(root).resolve()

    file_hashes: Dict[str, str] = {}
    for path in root_path.rglob("*"):
        if path.is_file():
            file_hashes[str(path)] = _sha256(path)

    discrepancies: List[Dict[str, str]] = []

    # Collect tests from database or in-memory registry
    if hasattr(TestResult, "objects"):
        tests: Iterable = TestResult.objects(file_path__startswith=str(root_path))
    else:  # Fallback for lightweight dataclass models used in tests
        tests = []
        registry = getattr(Sample, "_registry", {})
        for sample in registry.values():  # type: ignore[assignment]
            for t in getattr(sample, "tests", []) or []:
                if str(getattr(t, "file_path", "")).startswith(str(root_path)):
                    tests.append(t)

    for test in tests:
        stored_path = str(Path(test.file_path).resolve()) if getattr(test, "file_path", None) else None
        expected = getattr(test, "file_hash", "") or ""
        test_id = str(getattr(test, "id", ""))
        file_id = getattr(test, "file_id", None)

        actual = ""
        if stored_path and stored_path in file_hashes:
            actual = file_hashes.pop(stored_path)
            if expected and expected != actual:
                discrepancies.append(
                    {
                        "path": stored_path,
                        "status": "hash_mismatch",
                        "expected_hash": expected,
                        "actual_hash": actual,
                        "gridfs_hash": "",
                        "test_id": test_id,
                    }
                )
        else:
            discrepancies.append(
                {
                    "path": stored_path or "",
                    "status": "missing_file",
                    "expected_hash": expected,
                    "actual_hash": "",
                    "gridfs_hash": "",
                    "test_id": test_id,
                }
            )

        if file_id:
            try:
                raw = file_storage.get_raw_data_file_by_id(file_id)
            except Exception:
                discrepancies.append(
                    {
                        "path": stored_path or "",
                        "status": "missing_gridfs",
                        "expected_hash": expected,
                        "actual_hash": actual,
                        "gridfs_hash": "",
                        "test_id": test_id,
                    }
                )
            else:
                if isinstance(raw, str):
                    with open(raw, "rb") as fh:
                        raw_bytes = fh.read()
                else:
                    raw_bytes = raw
                gridfs_hash = hashlib.sha256(raw_bytes).hexdigest()
                if expected and gridfs_hash != expected:
                    discrepancies.append(
                        {
                            "path": stored_path or "",
                            "status": "gridfs_mismatch",
                            "expected_hash": expected,
                            "actual_hash": actual,
                            "gridfs_hash": gridfs_hash,
                            "test_id": test_id,
                        }
                    )
        else:
            discrepancies.append(
                {
                    "path": stored_path or "",
                    "status": "missing_gridfs",
                    "expected_hash": expected,
                    "actual_hash": actual,
                    "gridfs_hash": "",
                    "test_id": test_id,
                }
            )

    # Files without DB entries
    for path, actual in file_hashes.items():
        discrepancies.append(
            {
                "path": path,
                "status": "missing_db",
                "expected_hash": "",
                "actual_hash": actual,
                "gridfs_hash": "",
                "test_id": "",
            }
        )

    return discrepancies


def summarize_discrepancies(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
    """Return counts of added, mismatched and missing items."""
    counts = {"added": 0, "mismatched": 0, "missing": 0}
    for row in rows:
        status = row.get("status")
        if status == "missing_db":
            counts["added"] += 1
        elif status in {"hash_mismatch", "gridfs_mismatch"}:
            counts["mismatched"] += 1
        elif status in {"missing_file", "missing_gridfs"}:
            counts["missing"] += 1
    return counts


def _emit_csv(rows: Iterable[Dict[str, str]]) -> None:
    fieldnames = [
        "path",
        "status",
        "expected_hash",
        "actual_hash",
        "gridfs_hash",
        "test_id",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Directory of raw files to audit")
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format for the discrepancy report",
    )
    args = parser.parse_args(argv)

    rows = verify_directory(args.root)
    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        _emit_csv(rows)

    summary = summarize_discrepancies(rows)
    print(
        f"Added: {summary['added']} | Mismatched: {summary['mismatched']} | Missing: {summary['missing']}"
    )

    return 0 if not rows else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    import sys

    raise SystemExit(main())
