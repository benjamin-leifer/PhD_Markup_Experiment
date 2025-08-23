"""Verify raw test files against recorded hashes.

This helper walks a directory tree, computes SHA256 hashes for each file and
compares them with the ``file_hash`` stored on :class:`battery_analysis.models.TestResult`
records.  Any missing or mismatched files are returned in a list of
dictionaries and the command line interface can emit the results as either
JSON or CSV.

The module exits with a non-zero status code when discrepancies are found::

    python -m battery_analysis.utils.verify_import DATA_DIR --format json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from battery_analysis.models import Sample, TestResult
from battery_analysis.utils.db import ensure_connection

__all__ = [
    "verify_directory",
    "summarize_discrepancies",
    "write_report",
    "main",
]


def _sha256(path: Path) -> str:
    """Return the SHA256 hex digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_tests(root_path: Path) -> Iterable[TestResult]:
    """Yield :class:`TestResult` objects with file paths under ``root_path``."""
    if hasattr(TestResult, "objects"):
        return TestResult.objects(file_path__startswith=str(root_path))
    registry = getattr(Sample, "_registry", {})
    tests: List[TestResult] = []
    for sample in registry.values():
        for t in getattr(sample, "tests", []) or []:
            fp = getattr(t, "file_path", "")
            if fp and str(Path(fp).resolve()).startswith(str(root_path)):
                tests.append(t)
    return tests


def verify_directory(root: str) -> List[Dict[str, str]]:
    """Verify ``root`` against :class:`TestResult` records.

    Parameters
    ----------
    root:
        Directory containing raw data files.

    Returns
    -------
    list of dict
        Each dict describes a discrepancy with keys ``path``, ``status``,
        ``expected_hash``, ``actual_hash`` and ``test_id``.  The list is empty
        when no discrepancies are found.
    """
    ensure_connection()
    root_path = Path(root).resolve()

    file_hashes: Dict[str, str] = {}
    for path in root_path.rglob("*"):
        if path.is_file():
            file_hashes[str(path.resolve())] = _sha256(path)

    discrepancies: List[Dict[str, str]] = []

    for test in _collect_tests(root_path):
        stored_path = (
            str(Path(test.file_path).resolve())
            if getattr(test, "file_path", None)
            else None
        )
        expected = getattr(test, "file_hash", "") or ""
        test_id = str(getattr(test, "id", ""))

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
                    "test_id": test_id,
                }
            )

    # Files without corresponding DB entries
    for file_path, actual in file_hashes.items():
        discrepancies.append(
            {
                "path": file_path,
                "status": "missing_db",
                "expected_hash": "",
                "actual_hash": actual,
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
        elif status == "hash_mismatch":
            counts["mismatched"] += 1
        elif status == "missing_file":
            counts["missing"] += 1
    return counts


def _emit_csv(rows: Iterable[Dict[str, str]]) -> None:
    fieldnames = ["path", "status", "expected_hash", "actual_hash", "test_id"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def write_report(rows: Iterable[Dict[str, str]], path: str) -> None:
    """Write ``rows`` to ``path`` in JSON or CSV format."""
    dst = Path(path)
    data = list(rows)
    if dst.suffix.lower() == ".json":
        dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        fieldnames = ["path", "status", "expected_hash", "actual_hash", "test_id"]
        with dst.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
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

    return 0 if not rows else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
