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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    workers: int | None = None,
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
    workers:
        Maximum number of worker threads for parallel processing. ``None``
        uses the default determined by :class:`ThreadPoolExecutor`.

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
    # Gather all eligible file paths grouped by sample
    by_sample: Dict[str, Tuple[Sample, List[str]]] = {}
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

            name = metadata.get("sample_code") if metadata else None
            if not name:
                name = os.path.basename(os.path.dirname(file_path)) or "unknown"

            attrs: Dict[str, object] = {}
            if metadata:
                attrs = {k: v for k, v in metadata.items() if k != "sample_code"}

            sample = Sample.get_or_create(name, **attrs)
            info = by_sample.setdefault(sample.name, (sample, []))
            info[1].append(file_path)

    processed: Set[str] = set()

    def _process_sample(sample: Sample, paths: List[str]) -> List[Tuple[object, bool]]:
        """Worker function to process all files for a sample sequentially."""
        if not ensure_connection():  # defensive
            raise RuntimeError("Database connection not available")
        results: List[Tuple[object, bool]] = []
        for path in paths:
            try:
                test, was_update = data_update.process_file_with_update(path, sample)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to process %s: %s", path, exc)
                continue
            results.append((test, was_update))
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_sample = {
            executor.submit(_process_sample, sample, paths): sample.name
            for sample, paths in by_sample.values()
        }

        for future in as_completed(future_to_sample):
            sample_name = future_to_sample[future]
            try:
                results = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Processing failed for sample %s: %s", sample_name, exc)
                results = []

            for test, was_update in results:
                action = "updated" if was_update else "created"
                logger.info(
                    "%s test %s for sample %s",
                    action.title(),
                    getattr(test, "id", None),
                    sample_name,
                )
            processed.add(sample_name)

            try:
                update_cell_dataset(sample_name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to refresh dataset for %s: %s", sample_name, exc)

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
        "--workers",
        type=int,
        default=None,
        help="Maximum number of worker threads (default: library default)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    return import_directory(
        args.root,
        sample_lookup=args.sample_lookup,
        workers=args.workers,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
