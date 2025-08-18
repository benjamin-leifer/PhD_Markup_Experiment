"""Command-line utility to import test files from a directory.

This module scans a directory tree for files supported by
:func:`battery_analysis.parsers.parse_file`, imports each test using
``process_file_with_update`` and refreshes any affected cell datasets. Samples
are retrieved or created via :func:`Sample.get_or_create`.

The script can be executed directly::

    python -m battery_analysis.utils.import_directory ROOT_DIR

Use ``--sample-lookup`` to attempt detecting the sample from parser metadata
(e.g. a ``sample_code`` field).  Without this option the parent directory name
is used as the sample identifier.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Set

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from battery_analysis import parsers
from battery_analysis.models import Sample
from battery_analysis.utils.db import ensure_connection
from battery_analysis.utils import data_update
from battery_analysis.utils.cell_dataset_builder import update_cell_dataset

logger = logging.getLogger(__name__)


def import_directory(
    root: str, *, sample_lookup: bool = False, dry_run: bool = False
) -> int:
    """Import all supported files within ``root``.

    Parameters
    ----------
    root:
        Root directory to search for files.
    sample_lookup:
        When ``True`` the parser is invoked to extract metadata (such as a
        ``sample_code``) to determine the sample.  Otherwise the parent
        directory name is used.
    dry_run:
        Parse files and report actions without writing to the database.

    Returns
    -------
    int
        ``0`` if processing completed, ``1`` if the database connection was not
        available.
    """

    if not ensure_connection():
        logger.error("Database connection not available")
        return 1

    supported = {ext.lower() for ext in parsers.get_supported_formats()}
    processed_samples: Set[str] = set()
    files: list[str] = []

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported:
                continue
            files.append(os.path.join(dirpath, filename))

    total = len(files)
    tests_created = tests_updated = failures = 0
    iterator = tqdm(files, unit="file") if tqdm else files

    for idx, file_path in enumerate(iterator, 1):
        if not tqdm and (idx % 10 == 0 or idx == total):
            logger.info("Processed %d/%d files", idx, total)

        parsed_data = metadata = None
        if sample_lookup or dry_run:
            try:
                parsed_data, metadata = parsers.parse_file(file_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to parse %s: %s", file_path, exc)
                failures += 1
                continue

        name = metadata.get("sample_code") if (sample_lookup and metadata) else None
        if not name:
            name = os.path.basename(os.path.dirname(file_path)) or "unknown"

        attrs: Dict[str, object] = {}
        if metadata:
            attrs = {k: v for k, v in metadata.items() if k != "sample_code"}

        if dry_run:
            sample = Sample.objects(name=name).first()
            action = "created"
            if sample and parsed_data and metadata:
                identifiers = data_update.extract_test_identifiers(
                    file_path, parsed_data, metadata
                )
                matches = data_update.find_matching_tests(identifiers, sample.id)
                if matches:
                    action = "updated"
            logger.info("(dry-run) Would have %s test for sample %s", action, name)
        else:
            sample = Sample.get_or_create(name, **attrs)
            try:
                test, was_update = data_update.process_file_with_update(
                    file_path, sample
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to process %s: %s", file_path, exc)
                failures += 1
                continue

            action = "updated" if was_update else "created"
            logger.info(
                "%s test %s for sample %s",
                action.title(),
                getattr(test, "id", None),
                sample.name,
            )
            processed_samples.add(sample.name)

        if action == "updated":
            tests_updated += 1
        else:
            tests_created += 1

    if not dry_run:
        for name in processed_samples:
            try:
                update_cell_dataset(name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to refresh dataset for %s: %s", name, exc)

    print(
        f"Processed {total} files: {tests_created} created, {tests_updated} updated, {failures} failures"
    )

    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for command-line execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Root directory containing data files")
    parser.add_argument(
        "--sample-lookup",
        action="store_true",
        help="Lookup sample using parser metadata instead of directory name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and report actions without writing to the database",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    return import_directory(
        args.root, sample_lookup=args.sample_lookup, dry_run=args.dry_run
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
