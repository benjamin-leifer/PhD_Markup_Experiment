"""Command-line utility to import test files from a directory.

This module scans a directory tree for files supported by
:func:`battery_analysis.parsers.parse_file`, imports each test using
``process_file_with_update`` and refreshes any affected cell datasets. Samples
are retrieved or created via :func:`Sample.get_or_create`.

The script can be executed directly::

    python -m battery_analysis.utils.import_directory ROOT_DIR

Use ``--sample-lookup`` to attempt detecting the sample from parser metadata
(e.g. a ``sample_code`` field). Without this option the parent directory name
is used as the sample identifier.

A manifest file (``.import_state.json``) in the root directory records the
modification time of processed files so subsequent runs skip unchanged inputs.
Use ``--reset`` to ignore this state and re-import everything.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Set, Tuple

from battery_analysis import parsers
from battery_analysis.models import Sample
from battery_analysis.utils.db import ensure_connection
from battery_analysis.utils import data_update
from battery_analysis.utils.cell_dataset_builder import update_cell_dataset

logger = logging.getLogger(__name__)


def import_directory(
    root: str,
    *,
    sample_lookup: bool = False,
    reset: bool = False,
    dry_run: bool = False,
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

    reset:
        When ``True`` any existing import state is ignored and all files are
        reprocessed.
    dry_run:
        When ``True`` parse files and report what would happen without creating
        samples, importing tests, or refreshing datasets.

    Returns
    -------
    int
        ``0`` if processing completed, ``1`` if the database connection was not
        available.
    """

    if not dry_run and not ensure_connection():
        logger.error("Database connection not available")
        return 1

    supported = {ext.lower() for ext in parsers.get_supported_formats()}
    processed: Set[str] = set()
    eligible: List[Tuple[str, float]] = []
    skipped = 0

    state_path = os.path.join(root, ".import_state.json")
    state: Dict[str, float] = {}
    if not reset and os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load state from %s: %s", state_path, exc)

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported:
                continue
            file_path = os.path.join(dirpath, filename)
            abs_path = os.path.abspath(file_path)
            mtime = os.path.getmtime(abs_path)
            if not reset and state.get(abs_path) == mtime:
                logger.info("Skipping %s; already imported", abs_path)
                skipped += 1
                continue
            eligible.append((abs_path, mtime))

    total = len(eligible)
    created = 0
    updated = 0

    for idx, (abs_path, mtime) in enumerate(eligible, 1):
        metadata = None
        if sample_lookup or dry_run:
            try:
                _, metadata = parsers.parse_file(abs_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to parse %s: %s", abs_path, exc)
                skipped += 1
                continue

        name = metadata.get("sample_code") if metadata else None
        if not name:
            name = os.path.basename(os.path.dirname(abs_path)) or "unknown"

        attrs: Dict[str, object] = {}
        if metadata:
            attrs = {k: v for k, v in metadata.items() if k != "sample_code"}

        processed.add(name)

        if dry_run:
            logger.info("Would process %s for sample %s", abs_path, name)
        else:
            sample = Sample.get_or_create(name, **attrs)
            try:
                test, was_update = data_update.process_file_with_update(
                    abs_path, sample
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to process %s: %s", abs_path, exc)
                skipped += 1
                continue

            action = "updated" if was_update else "created"
            if was_update:
                updated += 1
            else:
                created += 1
            logger.info(
                "%s test %s for sample %s",
                action.title(),
                getattr(test, "id", None),
                sample.name,
            )
            state[abs_path] = mtime
            try:
                with open(state_path, "w", encoding="utf-8") as fh:
                    json.dump(state, fh, indent=2, sort_keys=True)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to write state to %s: %s", state_path, exc)

        if idx % 10 == 0 or idx == total:
            logger.info("Processed %s/%s", idx, total)

    if dry_run:
        for name in processed:
            logger.info("Would refresh dataset for %s", name)
    else:
        for name in processed:
            try:
                update_cell_dataset(name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to refresh dataset for %s: %s", name, exc)

    print(f"Summary: created={created}, updated={updated}, skipped={skipped}")

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
        "--reset",
        action="store_true",
        help="Ignore existing import state and reprocess all files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files but do not import or update datasets",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    return import_directory(
        args.root,
        sample_lookup=args.sample_lookup,
        reset=args.reset,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
