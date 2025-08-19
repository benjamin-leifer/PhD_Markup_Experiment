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
import fnmatch
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Set

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
    dry_run: bool = False,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    resume_manifest: str | None = None,
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
        When ``True`` files are logged but no database changes are made.
    include:
        Optional iterable of glob patterns. Only files matching at least one
        pattern are processed.
    exclude:
        Optional iterable of glob patterns. Files matching any pattern are
        skipped.
    resume_manifest:
        Path to a JSON file listing previously processed file paths. When
        provided, files listed are skipped and newly processed files are added.

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
    processed: Set[str] = set()

    include = list(include or [])
    exclude = list(exclude or [])

    manifest_path: Path | None = None
    manifest_processed: Set[str] = set()
    if resume_manifest:
        manifest_path = Path(resume_manifest)
        if manifest_path.exists():
            try:
                manifest_processed = set(json.load(manifest_path.open()))
            except Exception:  # pragma: no cover - defensive
                manifest_processed = set()

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported:
                continue
            file_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(file_path, root)

            if include and not any(fnmatch.fnmatch(rel_path, pat) for pat in include):
                continue
            if exclude and any(fnmatch.fnmatch(rel_path, pat) for pat in exclude):
                continue
            if resume_manifest and rel_path in manifest_processed:
                continue

            metadata = None
            if sample_lookup:
                try:
                    _, metadata = parsers.parse_file(file_path)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Failed to parse %s: %s", file_path, exc)
                    continue

            name = metadata.get("sample_code") if metadata else None
            if not name:
                name = os.path.basename(os.path.dirname(file_path)) or "unknown"

            attrs: Dict[str, object] = {}
            if metadata:
                attrs = {k: v for k, v in metadata.items() if k != "sample_code"}

            try:
                if dry_run:
                    logger.info(
                        "Dry run: would process %s for sample %s", file_path, name
                    )
                else:
                    sample = Sample.get_or_create(name, **attrs)
                    test, was_update = data_update.process_file_with_update(
                        file_path, sample
                    )
                    action = "updated" if was_update else "created"
                    logger.info(
                        "%s test %s for sample %s",
                        action.title(),
                        getattr(test, "id", None),
                        sample.name,
                    )
                    processed.add(sample.name)
                if resume_manifest:
                    manifest_processed.add(rel_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to process %s: %s", file_path, exc)
                continue

    if not dry_run:
        for name in processed:
            try:
                update_cell_dataset(name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to refresh dataset for %s: %s", name, exc)

    if resume_manifest and manifest_path is not None:
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(sorted(manifest_processed), fh, indent=2)

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
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    return import_directory(args.root, sample_lookup=args.sample_lookup)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
