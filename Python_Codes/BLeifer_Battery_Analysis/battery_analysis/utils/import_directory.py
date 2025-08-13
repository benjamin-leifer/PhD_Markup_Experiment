"""Command-line utility to import test files from a directory.

This module scans a directory tree for files supported by
:func:`battery_analysis.parsers.parse_file`, imports each test using
``process_file_with_update`` and refreshes any affected cell datasets.

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

from battery_analysis import models, parsers
from battery_analysis.utils.db import ensure_connection
from battery_analysis.utils import data_update
from battery_analysis.utils.cell_dataset_builder import update_cell_dataset

logger = logging.getLogger(__name__)


def _get_or_create_sample(
    file_path: str, metadata: Dict[str, object] | None = None
) -> models.Sample:
    """Return the :class:`~battery_analysis.models.Sample` for ``file_path``.

    Parameters
    ----------
    file_path:
        Path to the data file.  The parent directory name is used when a
        ``sample_code`` is not present in ``metadata``.
    metadata:
        Optional metadata returned by the parser.  If it contains a
        ``sample_code`` key that value is used as the sample name.
    """

    name = None
    if metadata:
        name = metadata.get("sample_code")
    if not name:
        name = os.path.basename(os.path.dirname(file_path)) or "unknown"

    sample = models.Sample.objects(name=name).first()
    if not sample:
        sample = models.Sample(name=name)
        sample.save()
        logger.info("Created sample %s", name)
    return sample


def import_directory(root: str, *, sample_lookup: bool = False) -> int:
    """Import all supported files within ``root``.

    Parameters
    ----------
    root:
        Root directory to search for files.
    sample_lookup:
        When ``True`` the parser is invoked to extract metadata (such as a
        ``sample_code``) to determine the sample.  Otherwise the parent
        directory name is used.

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

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
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

            sample = _get_or_create_sample(file_path, metadata)

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
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    return import_directory(args.root, sample_lookup=args.sample_lookup)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
