"""Command-line utility to import test files from a directory.

This module scans a directory tree for files supported by
:func:`battery_analysis.parsers.parse_file`, imports each test using
``process_file_with_update`` and refreshes any affected cell datasets. Samples
are retrieved or created via :func:`Sample.get_or_create`.

The script can be executed directly::

    python -m battery_analysis.utils.import_directory ROOT_DIR

Limit processing with glob patterns::

    python -m battery_analysis.utils.import_directory data \
        --include "*_Wb_*.csv" --exclude "*/old/*"

Use ``--sample-lookup`` to attempt detecting the sample from parser metadata
(e.g. a ``sample_code`` field).  Without this option the parent directory name
is used as the sample identifier.  Supply ``--include`` and ``--exclude``
patterns to limit which files are processed.
"""

from __future__ import annotations

import argparse
import logging
import os
from fnmatch import fnmatch
from typing import Dict, Iterable, List, Set

from battery_analysis import parsers
from battery_analysis.models import Sample
from battery_analysis.utils.db import ensure_connection
from battery_analysis.utils import data_update
from battery_analysis.utils.cell_dataset_builder import update_cell_dataset

logger = logging.getLogger(__name__)


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch(path, pat) for pat in patterns)


def import_directory(
    root: str,
    *,
    sample_lookup: bool = False,
    include: List[str] | None = None,
    exclude: List[str] | None = None,
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
    include:
        List of glob patterns for paths to include.  Patterns are matched
        against the relative directory or file path within ``root``.  When
        omitted, all files are considered.
    exclude:
        List of glob patterns for paths to exclude.  Paths matching any pattern
        are skipped.

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

    include = include or ["*"]
    exclude = exclude or []

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if exclude and _matches(rel_dir, exclude):
            continue
        if include and rel_dir != "." and not _matches(rel_dir, include):
            continue

        dirnames[:] = [
            d
            for d in dirnames
            if not _matches(os.path.join(rel_dir, d), exclude)
            and (not include or _matches(os.path.join(rel_dir, d), include))
        ]

        for filename in filenames:
            rel_file = os.path.join(rel_dir, filename)
            if exclude and _matches(rel_file, exclude):
                continue
            if include and not _matches(rel_file, include):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported:
                continue
            file_path = os.path.join(dirpath, filename)

            metadata = None
            if sample_lookup:
                try:
                    _, metadata = parsers.parse_file(file_path)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Failed to parse %s: %s", file_path, exc)
                    metadata = None

            name = metadata.get("sample_code") if metadata else None
            if not name:
                name = os.path.basename(os.path.dirname(file_path)) or "unknown"

            attrs: Dict[str, object] = {}
            if metadata:
                attrs = {k: v for k, v in metadata.items() if k != "sample_code"}

            sample = Sample.get_or_create(name, **attrs)

            try:
                test, was_update = data_update.process_file_with_update(
                    file_path, sample
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to process %s: %s", file_path, exc)
                continue

            action = "updated" if was_update else "created"
            logger.info(
                "%s test %s for sample %s",
                action.title(),
                getattr(test, "id", None),
                sample.name,
            )
            processed.add(sample.name)

    for name in processed:
        try:
            update_cell_dataset(name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to refresh dataset for %s: %s", name, exc)

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
        "--include",
        action="append",
        default=None,
        help="Glob pattern of files or directories to include (may repeat)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Glob pattern of files or directories to exclude (may repeat)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    return import_directory(
        args.root,
        sample_lookup=args.sample_lookup,
        include=args.include,
        exclude=args.exclude,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
