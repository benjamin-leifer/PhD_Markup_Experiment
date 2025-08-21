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

``--include`` and ``--exclude`` options accept glob patterns to filter
directories or filenames. Multiple patterns may be supplied by repeating the
option. For example, to import only ``.csv`` files while skipping anything in
``archive`` folders::

    python -m battery_analysis.utils.import_directory data \
        --include "*.csv" --exclude "*/archive/*"

A manifest file (``.import_state.json``) in the root directory records the
modification time and content hash of processed files so subsequent runs skip
unchanged inputs. Use ``--reset`` to ignore this state and re-import
everything.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import logging
import os
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import BinaryIO, Dict, List, Set, Tuple, cast
import fnmatch

from battery_analysis import parsers
from battery_analysis.models import Sample, TestResult, ImportJob
from battery_analysis.utils.db import ensure_connection
from battery_analysis.utils import data_update
from battery_analysis.utils.cell_dataset_builder import update_cell_dataset
from battery_analysis.utils.config import load_config

logger = logging.getLogger(__name__)

# Load configuration at module import so CLI defaults can reference it
CONFIG = load_config()


def process_file_with_update(path: str, sample: Sample) -> tuple[TestResult, bool]:
    """Public wrapper around :func:`data_update.process_file_with_update`.

    This helper is exposed so external tools like
    :mod:`battery_analysis.utils.import_watcher` can reuse the same file
    processing logic as the directory importer.

    Parameters
    ----------
    path:
        Path to the data file.
    sample:
        :class:`Sample` instance the file belongs to.

    Returns
    -------
    tuple
        The ``(TestResult, was_update)`` tuple returned by
        :func:`battery_analysis.utils.data_update.process_file_with_update`.
    """

    return cast(
        tuple[TestResult, bool], data_update.process_file_with_update(path, sample)
    )


def import_directory(
    root: str,
    *,
    sample_lookup: bool = False,
    reset: bool = False,
    dry_run: bool = False,
    workers: int | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
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
    workers:
        Number of worker threads to use when importing files. ``None`` uses the
        CPU count.

    include:
        Glob patterns that must match either the directory path or filename for
        a file to be processed. If omitted, all paths are included.
    exclude:
        Glob patterns that, when matched against the directory path or filename,
        cause the file to be skipped.

    Returns
    -------
    int
        ``0`` if processing completed, ``1`` if the database connection was not
        available.
    """

    if not dry_run:
        db_kwargs: Dict[str, object] = {}
        if CONFIG.get("db_uri"):
            db_kwargs["host"] = CONFIG["db_uri"]
        if not ensure_connection(**db_kwargs):
            logger.error("Database connection not available")
            return 1

    job: ImportJob | None = None
    if not dry_run:
        try:
            job = ImportJob().save()
        except Exception:  # pragma: no cover - best effort
            job = None

    include = include or []
    exclude = exclude or []

    def _match(path: str, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatch(path, pat) for pat in patterns)

    supported = {ext.lower() for ext in parsers.get_supported_formats()}
    processed: Set[str] = set()
    eligible: List[Tuple[str, float, str]] = []
    skipped = 0

    state_path = os.path.join(root, ".import_state.json")
    state: Dict[str, object] = {}
    state_dirty = False
    if not reset and os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load state from %s: %s", state_path, exc)

    for dirpath, _, filenames in os.walk(root):
        if exclude and _match(dirpath, exclude):
            continue
        dir_included = True if not include else _match(dirpath, include)
        for filename in filenames:
            if exclude and _match(filename, exclude):
                continue
            if include and not (dir_included or _match(filename, include)):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported:
                continue
            file_path = os.path.join(dirpath, filename)
            abs_path = os.path.abspath(file_path)
            mtime = os.path.getmtime(abs_path)

            # compute hash of file contents
            h = hashlib.md5()
            with open(abs_path, "rb") as bin_fh:
                reader = cast(BinaryIO, bin_fh)
                for chunk in iter(lambda: reader.read(8192), b""):
                    h.update(chunk)
            file_hash = h.hexdigest()

            if not reset and abs_path in state:
                entry = state[abs_path]
                if isinstance(entry, dict):
                    prev_mtime = entry.get("mtime")
                    prev_hash = entry.get("hash")
                else:
                    prev_mtime = entry
                    prev_hash = None

                if prev_mtime == mtime and prev_hash == file_hash:
                    logger.info("Skipping %s; already imported", abs_path)
                    skipped += 1
                    continue

                if prev_mtime == mtime and prev_hash is None:
                    logger.info("Skipping %s; already imported", abs_path)
                    if not dry_run:
                        state[abs_path] = {"mtime": mtime, "hash": file_hash}
                        state_dirty = True
                    skipped += 1
                    continue

            eligible.append((abs_path, mtime, file_hash))

    total = len(eligible)
    created = 0
    updated = 0

    workers = workers or (os.cpu_count() or 1)

    def _process(
        abs_path: str, mtime: float, file_hash: str
    ) -> tuple[str, str, str, float, str, object | None, str | None]:
        """Worker function processing a single file."""
        if not ensure_connection():
            logger.error("Database connection not available")
            return (
                "",
                "skipped",
                abs_path,
                mtime,
                file_hash,
                None,
                "Database connection not available",
            )

        metadata = None
        if sample_lookup or dry_run:
            try:
                _, metadata = parsers.parse_file(abs_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to parse %s: %s", abs_path, exc)
                return "", "skipped", abs_path, mtime, file_hash, None, str(exc)

        name = metadata.get("sample_code") if metadata else None
        if not name:
            name = os.path.basename(os.path.dirname(abs_path)) or "unknown"

        attrs: Dict[str, object] = {}
        if metadata:
            attrs = {k: v for k, v in metadata.items() if k != "sample_code"}

        if dry_run:
            logger.info("Would process %s for sample %s", abs_path, name)
            return name, "dry_run", abs_path, mtime, file_hash, None, None

        sample = Sample.get_or_create(name, **attrs)
        try:
            test, was_update = data_update.process_file_with_update(abs_path, sample)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to process %s: %s", abs_path, exc)
            return name, "skipped", abs_path, mtime, file_hash, None, str(exc)

        action = "updated" if was_update else "created"
        logger.info(
            "%s test %s for sample %s",
            action.title(),
            getattr(test, "id", None),
            sample.name,
        )
        return name, action, abs_path, mtime, file_hash, getattr(test, "id", None), None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process, p, m, h) for p, m, h in eligible]
        for idx, fut in enumerate(as_completed(futures), 1):
            name, action, abs_path, mtime, file_hash, test_id, error = fut.result()
            if name:
                processed.add(name)
            if action == "updated":
                updated += 1
            elif action == "created":
                created += 1
            elif action == "skipped":
                skipped += 1
            if action in {"updated", "created"} and not dry_run:
                state[abs_path] = {"mtime": mtime, "hash": file_hash}
                try:
                    with open(state_path, "w", encoding="utf-8") as fh:
                        json.dump(state, fh, indent=2, sort_keys=True)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Failed to write state to %s: %s", state_path, exc)
            if job is not None:
                entry = {"path": abs_path, "action": action}
                if test_id is not None:
                    entry["test_id"] = str(test_id)
                if error is not None:
                    entry["error"] = error
                    job.errors.append(error)
                job.files.append(entry)
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

    if state_dirty and not dry_run:
        try:
            with open(state_path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2, sort_keys=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to write state to %s: %s", state_path, exc)

    print(f"Summary: created={created}, updated={updated}, skipped={skipped}")

    if job is not None:
        job.end_time = datetime.datetime.utcnow()
        try:
            job.save()
        except Exception:  # pragma: no cover - best effort
            pass

    return 0


def rollback_job(job_id: str) -> int:
    """Remove ``TestResult`` entries created during a previous import job."""

    if not ensure_connection():
        logger.error("Database connection not available")
        return 1

    job = ImportJob.objects(id=job_id).first()
    if not job:
        logger.error("ImportJob %s not found", job_id)
        return 1

    for entry in getattr(job, "files", []):
        test_id = entry.get("test_id")
        path = entry.get("path")
        if test_id:
            try:
                TestResult.objects(id=test_id).delete()
            except Exception:  # pragma: no cover - defensive
                logger.error("Failed to delete TestResult %s", test_id)
        elif path:
            # Dataclass fallback: remove tests by matching file_path
            for sample in getattr(Sample, "_registry", {}).values():
                sample.tests = [
                    t for t in sample.tests if getattr(t, "file_path", None) != path
                ]
    logger.info("Rolled back import job %s", job_id)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for command-line execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", help="Root directory containing data files")
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
    parser.add_argument(
        "--workers",
        type=int,
        default=CONFIG.get("workers") or (os.cpu_count() or 1),
        help="Number of worker threads to use for importing files",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Glob pattern to include (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Glob pattern to exclude (repeatable)",
    )
    parser.add_argument(
        "--rollback",
        metavar="JOB_ID",
        help="Remove TestResult entries created during the specified job",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    if args.rollback:
        return rollback_job(args.rollback)
    if not args.root:
        parser.error("root is required unless --rollback is specified")

    include = args.include if args.include is not None else CONFIG.get("include")
    exclude = args.exclude if args.exclude is not None else CONFIG.get("exclude")

    return import_directory(
        args.root,
        sample_lookup=args.sample_lookup,
        reset=args.reset,
        dry_run=args.dry_run,
        workers=args.workers,
        include=include,
        exclude=exclude,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
