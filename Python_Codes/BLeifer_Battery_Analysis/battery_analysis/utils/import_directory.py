"""Command-line utility to import test files from a directory.

This module scans a directory tree for files supported by
:func:`battery_analysis.parsers.parse_file`, imports each test using
``process_file_with_update`` and refreshes any affected cell datasets. Samples
are retrieved or created via :func:`Sample.get_or_create`.  Each imported file
is archived to MongoDB's GridFS for future retrieval unless ``--no-archive`` is
specified.

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
import os
import datetime
import csv
import tomllib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import BinaryIO, Dict, List, Set, Tuple, Iterator, cast
import fnmatch
from pathlib import Path

from pandas.errors import ParserError

try:
    import redis
except Exception:  # pragma: no cover - optional dependency
    redis = None

from battery_analysis import parsers
from battery_analysis.models import Sample, TestResult, ImportJob
from battery_analysis.utils.db import ensure_connection
from battery_analysis.utils import data_update, file_storage
from battery_analysis.utils.cell_dataset_builder import update_cell_dataset
from battery_analysis.utils.config import load_config
from battery_analysis.utils import notifications
from battery_analysis.utils.logging import get_logger

logger = get_logger(__name__)

# Load configuration at module import so CLI defaults can reference it
CONFIG = load_config()

# Exceptions that trigger a retry when processing files
RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (ParserError, ConnectionError)
# Base delay used for exponential backoff between retries
RETRY_BASE_DELAY = 0.5

CONTROL_FILE = Path(__file__).resolve().parents[4] / ".import_control"


def _read_control_command() -> str | None:
    """Return the current command from the control file if it exists."""
    try:
        cmd = CONTROL_FILE.read_text(encoding="utf-8").strip().lower()
        return cmd or None
    except FileNotFoundError:
        return None


def _chunked(
    seq: List[Dict[str, object]], size: int
) -> Iterator[List[Dict[str, object]]]:
    """Yield lists of up to ``size`` items from ``seq``."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _load_sample_map(path: str) -> Dict[str, str]:
    """Load a mapping of file paths to sample names from ``path``.

    The file may be CSV with ``file_path,sample`` columns or a TOML file
    containing a ``[samples]`` table mapping paths to names.
    """

    mapping: Dict[str, str] = {}
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not row:
                    continue
                fp = row.get("file_path") or next(iter(row.values()))
                sample = row.get("sample") or list(row.values())[1]
                mapping[str(fp)] = str(sample)
    elif path.lower().endswith(".toml"):
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        table = data.get("samples", data)
        for fp, sample in table.items():
            mapping[str(fp)] = str(sample)
    else:  # pragma: no cover - defensive
        raise ValueError("Unsupported mapping format; use CSV or TOML")
    return mapping


def _write_sample_map(path: str, pairs: List[Tuple[str, str]]) -> None:
    """Write ``pairs`` to ``path`` in CSV or TOML format."""

    if path.lower().endswith(".csv"):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file_path", "sample"])
            for fp, sample in pairs:
                writer.writerow([fp, sample])
    elif path.lower().endswith(".toml"):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("[samples]\n")
            for fp, sample in pairs:
                fp_esc = fp.replace("\\", "\\\\").replace('"', '\\"')
                sample_esc = sample.replace("\\", "\\\\").replace('"', '\\"')
                fh.write(f'"{fp_esc}" = "{sample_esc}"\n')
    else:  # pragma: no cover - defensive
        raise ValueError("Unsupported mapping format; use CSV or TOML")


def process_file_with_update(
    path: str,
    sample: Sample,
    *,
    archive: bool = True,
    job: ImportJob | None = None,
    tags: list[str] | None = None,
) -> tuple[TestResult, bool]:
    """Process ``path`` for ``sample`` and optionally archive the raw file.

    This helper wraps :func:`battery_analysis.utils.data_update.process_file_with_update`
    to attach a SHA256 hash of the raw bytes and persist the original file in
    GridFS.  External tools such as :mod:`battery_analysis.utils.import_watcher`
    use this function to ensure consistent processing.

    Parameters
    ----------
    path:
        Path to the data file.
    sample:
        :class:`Sample` instance the file belongs to.
    archive:
        When ``True`` (default) the raw file is saved to GridFS via
        :func:`battery_analysis.utils.file_storage.save_raw` and the resulting
        ``file_id`` recorded on the :class:`~battery_analysis.models.TestResult`.
    job:
        Optional :class:`~battery_analysis.models.ImportJob` to record on the
        archived raw file.

    Returns
    -------
    tuple
        The ``(TestResult, was_update)`` tuple returned by
        :func:`battery_analysis.utils.data_update.process_file_with_update`.
    """

    abs_path = os.path.abspath(path)
    test, was_update = data_update.process_file_with_update(abs_path, sample)

    h = hashlib.sha256()
    with open(abs_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    digest = h.hexdigest()

    if isinstance(test, TestResult):
        test.file_hash = digest
        if tags:
            existing = getattr(test, "tags", []) or []
            test.tags = list({*existing, *tags})
            sample_tags = list({*(getattr(sample, "tags", []) or []), *tags})
            try:
                sample.tags = sample_tags
                sample.save()
            except Exception:  # pragma: no cover - best effort
                pass

        if archive:
            try:
                file_id = file_storage.save_raw(
                    path,
                    test_result=test,
                    source_path=abs_path,
                    import_job=job,
                )
                test.file_id = file_id
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Failed to archive %s: %s", path, exc)

        try:
            test.save()
        except Exception:  # pragma: no cover - best effort
            pass

    return cast(tuple[TestResult, bool], (test, was_update))


def import_directory(
    root: str,
    *,
    sample_lookup: bool = False,
    reset: bool = False,
    dry_run: bool = False,
    workers: int | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    notify: bool = False,
    archive: bool = True,
    preview_samples: bool = False,
    confirm: bool = False,
    sample_map: str | None = None,
    resume: str | None = None,
    report: str | None = None,
    retries: int = 0,
    tags: list[str] | None = None,
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
    notify:
        When ``True`` send a completion notification.
    archive:
        When ``True`` (default) archive raw files to GridFS. Disable with
        ``--no-archive``.
    preview_samples:
        When ``True`` display inferred sample names for each file before
        processing. Import stops after preview unless ``confirm`` is also
        supplied.
    confirm:
        Continue with the import after previewing samples. Has no effect unless
        ``preview_samples`` is ``True``.
    sample_map:
        Optional path to a CSV or TOML file mapping file paths to final sample
        names. When used with ``preview_samples`` a mapping file is created if it
        does not exist so users may edit the names before confirming the import.
    resume:
        Identifier of an :class:`ImportJob` to continue. Files already recorded
        for the job are skipped and new imports are appended to the existing
        job record.
    report:
        Optional path to write a per-file processing report in CSV or JSON
        format.
    retries:
        Number of times to retry processing a file when certain transient
        exceptions occur.

    Returns
    -------
    int
        ``0`` if processing completed, ``1`` if the database connection was not
        available.
    """

    if preview_samples and not confirm:
        dry_run = True

    if not dry_run:
        db_kwargs: Dict[str, object] = {}
        if CONFIG.get("db_uri"):
            db_kwargs["host"] = CONFIG["db_uri"]
        if not ensure_connection(**db_kwargs):
            logger.error("Database connection not available")
            if notify:
                notifications.send(
                    "Import job failed: database connection not available"
                )
            return 1

    job: ImportJob | None = None
    start_idx = 0
    processed_paths: Set[str] = set()
    if not dry_run:
        if resume:
            job = ImportJob.objects(id=resume).first()
            if not job:
                logger.error("ImportJob %s not found", resume)
                return 1
            start_idx = job.processed_count or 0
            processed_paths = {e.get("path") for e in job.files if e.get("path")}
        else:
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
    entries: List[Dict[str, object]] = []
    skipped = 0
    report_entries: List[Tuple[str, str, object | None]] = []

    state_path = os.path.join(root, ".import_state.json")
    state: Dict[str, object] = {}
    state_dirty = False
    original_state: Dict[str, object] = {}
    if not reset and os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load state from %s: %s", state_path, exc)
        original_state = dict(state)
    else:
        original_state = {}

    current_paths: Set[str] = set()

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
            current_paths.add(abs_path)
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

            metadata = None
            if sample_lookup or dry_run or preview_samples:
                try:
                    _, metadata = parsers.parse_file(abs_path)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Failed to parse %s: %s", abs_path, exc)

            name = metadata.get("sample_code") if metadata and sample_lookup else None
            if not name:
                name = os.path.basename(os.path.dirname(abs_path)) or "unknown"
            attrs: Dict[str, object] = {}
            if metadata and sample_lookup:
                attrs = {
                    k: v
                    for k, v in metadata.items()
                    if k not in {"sample_code", "name"}
                }

            entries.append(
                {
                    "path": abs_path,
                    "mtime": mtime,
                    "hash": file_hash,
                    "sample": name,
                    "attrs": attrs,
                }
            )

    if not reset:
        missing_paths = set(state.keys()) - current_paths
        if missing_paths:
            for path in missing_paths:
                state.pop(path, None)
            state_dirty = True

    if resume and processed_paths:
        entries = [e for e in entries if e["path"] not in processed_paths]

    total = start_idx + len(entries)
    created = 0
    updated = 0

    pub = None
    channel = CONFIG.get("redis_channel", "import_progress")
    if redis and CONFIG.get("redis_url"):
        try:  # pragma: no cover - optional
            pub = redis.from_url(CONFIG["redis_url"])
        except Exception:
            pub = None

    if job is not None:
        job.total_count = max(job.total_count, total)
        job.processed_count = start_idx
        try:
            job.save()
        except Exception:
            pass
        if pub:
            try:
                pub.publish(
                    channel,
                    json.dumps({"job_id": str(job.id), "total": job.total_count}),
                )
            except Exception:
                pass

    pairs = [(e["path"], e["sample"]) for e in entries]

    if sample_map:
        if os.path.exists(sample_map):
            mapping = _load_sample_map(sample_map)
            for e in entries:
                if e["path"] in mapping:
                    e["sample"] = mapping[e["path"]]
        elif preview_samples:
            _write_sample_map(sample_map, pairs)
        pairs = [(e["path"], e["sample"]) for e in entries]

    if preview_samples:
        max_len = max((len(p) for p, _ in pairs), default=10)
        header = "File Path".ljust(max_len) + " | Sample"
        print(header)
        print("-" * len(header))
        for fp, sample in pairs:
            print(fp.ljust(max_len) + " | " + sample)
        if sample_map and not os.path.exists(sample_map):
            print(f"Sample map written to {sample_map}")
        if not confirm:
            return 0

    workers = workers or (os.cpu_count() or 1)

    def _process(
        abs_path: str,
        mtime: float,
        file_hash: str,
        name: str,
        attrs: Dict[str, object],
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

        if dry_run:
            logger.info("Would process %s for sample %s", abs_path, name)
            return name, "dry_run", abs_path, mtime, file_hash, None, None

        sample = Sample.get_or_create(name, **attrs)
        if tags:
            try:
                sample.tags = list({*(getattr(sample, "tags", []) or []), *tags})
                sample.save()
            except Exception:
                pass
        attempt = 0
        while True:
            try:
                test, was_update = process_file_with_update(
                    abs_path, sample, archive=archive, job=job, tags=tags
                )
                break
            except RETRY_EXCEPTIONS as exc:
                if attempt >= retries:
                    msg = f"{exc} after {attempt} retries"
                    logger.error(
                        "Failed to process %s after %s retries: %s",
                        abs_path,
                        attempt,
                        exc,
                    )
                    return name, "skipped", abs_path, mtime, file_hash, None, msg
                time.sleep(2**attempt * RETRY_BASE_DELAY)
                attempt += 1
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

    idx = 0
    cancelled = False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch in _chunked(entries, max(1, workers)):
            futures = [
                executor.submit(
                    _process, e["path"], e["mtime"], e["hash"], e["sample"], e["attrs"]
                )
                for e in batch
            ]
            for fut in as_completed(futures):
                idx += 1
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
                detail = str(test_id) if test_id is not None else error
                report_entries.append((abs_path, action, detail))
                if job is not None:
                    entry = {"path": abs_path, "action": action}
                    if test_id is not None:
                        entry["test_id"] = str(test_id)
                    if error is not None:
                        entry["error"] = error
                        job.errors.append(error)
                    job.files.append(entry)
                    job.current_file = abs_path
                    job.processed_count = start_idx + idx
                    if (start_idx + idx) % 5 == 0 or (start_idx + idx) == total:
                        try:
                            job.save()
                        except Exception:
                            pass
                        if pub:
                            try:
                                pub.publish(
                                    channel,
                                    json.dumps(
                                        {
                                            "job_id": str(job.id),
                                            "current_file": abs_path,
                                            "processed": start_idx + idx,
                                            "total": total,
                                        }
                                    ),
                                )
                            except Exception:
                                pass
                if (start_idx + idx) % 10 == 0 or (start_idx + idx) == total:
                    logger.info("Processed %s/%s", start_idx + idx, total)

            cmd = _read_control_command()
            if cmd == "cancel":
                logger.info("Import cancelled via control file")
                cancelled = True
                break
            if cmd == "pause":
                logger.info("Import paused via control file")
                while True:
                    time.sleep(0.2)
                    cmd = _read_control_command()
                    if cmd == "cancel":
                        logger.info("Import cancelled via control file")
                        cancelled = True
                        break
                    if cmd != "pause":
                        logger.info("Import resumed")
                        break
                if cancelled:
                    break

    if cancelled:
        return 0

    if dry_run:
        for name in processed:
            logger.info("Would refresh dataset for %s", name)
    else:
        for name in processed:
            try:
                update_cell_dataset(name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to refresh dataset for %s: %s", name, exc)

    if (state_dirty or (not reset and state != original_state)) and not dry_run:
        try:
            with open(state_path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2, sort_keys=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to write state to %s: %s", state_path, exc)

    logger.info(
        "Summary: created=%s, updated=%s, skipped=%s",
        created,
        updated,
        skipped,
    )

    if job is not None:
        job.end_time = datetime.datetime.utcnow()
        job.current_file = None
        job.processed_count = total
        try:
            job.save()
        except Exception:  # pragma: no cover - best effort
            pass
        if pub:
            try:
                pub.publish(
                    channel, json.dumps({"job_id": str(job.id), "status": "completed"})
                )
            except Exception:
                pass

    if notify:
        msg = f"Import job completed: created={created}, updated={updated}, skipped={skipped}"
        if job is not None and job.errors:
            msg += f" with {len(job.errors)} errors"
        notifications.send(msg)

    if report:
        try:
            Path(report).parent.mkdir(parents=True, exist_ok=True)
            if report.lower().endswith(".json"):
                rows = [
                    {"file_path": fp, "status": status, "detail": detail}
                    for fp, status, detail in report_entries
                ]
                with open(report, "w", encoding="utf-8") as fh:
                    json.dump(rows, fh, indent=2)
            else:
                with open(report, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["file_path", "status", "detail"])
                    for fp, status, detail in report_entries:
                        writer.writerow([fp, status, detail or ""])
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to write report to %s: %s", report, exc)

    return 0


def show_status(job_id: str | None = None) -> int:
    """Print status information about import jobs."""

    if not ensure_connection():
        logger.error("Database connection not available")
        return 1

    jobs: list[ImportJob]
    if callable(getattr(ImportJob, "objects", None)):
        qs = ImportJob.objects(id=job_id) if job_id else ImportJob.objects()
        if job_id:
            job = qs.first() if hasattr(qs, "first") else next(iter(qs), None)
            if not job:
                logger.error("ImportJob %s not found", job_id)
                return 1
            jobs = [job]
        else:
            jobs = list(qs)
            jobs.sort(key=lambda j: getattr(j, "start_time", None), reverse=True)
    else:  # pragma: no cover - defensive
        jobs = []
    if not jobs:
        print("No import jobs found")
        return 0

    for job in jobs:
        start = job.start_time.isoformat() if job.start_time else "N/A"
        end = job.end_time.isoformat() if job.end_time else "N/A"
        processed = f"{job.processed_count}/{job.total_count}"
        errs = "; ".join(job.errors) if job.errors else "None"
        print(
            f"{job.id} | start: {start} | end: {end} | processed: {processed} | errors: {errs}"
        )
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
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send notification on completion or error",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Do not store raw files in GridFS",
    )
    parser.add_argument(
        "--preview-samples",
        action="store_true",
        help="Preview inferred sample names and exit unless --confirm is supplied",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Continue with import after previewing samples",
    )
    parser.add_argument(
        "--sample-map",
        metavar="PATH",
        help="CSV or TOML mapping file to override sample names",
    )
    parser.add_argument(
        "--status",
        nargs="?",
        const="",
        metavar="JOB_ID",
        help="Show status of import jobs; optionally provide JOB_ID",
    )
    parser.add_argument(
        "--resume",
        metavar="JOB_ID",
        help="Resume a previously interrupted import job",
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        help="Write processing report to PATH (CSV or JSON)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=CONFIG.get("retries") or 0,
        help="Number of times to retry failed file processing",
    )
    parser.add_argument(
        "--tags",
        action="append",
        default=None,
        help="Tag to apply to imported samples and tests (repeatable)",
    )
    args = parser.parse_args(argv)

    from battery_analysis.utils.logging import get_logger

    _logger = get_logger(__name__)
    if args.status is not None:
        return show_status(args.status or None)
    if args.rollback:
        return rollback_job(args.rollback)
    if not args.root:
        parser.error("root is required unless --rollback or --status is specified")

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
        notify=args.notify,
        archive=not args.no_archive,
        preview_samples=args.preview_samples,
        confirm=args.confirm,
        sample_map=args.sample_map,
        resume=args.resume,
        report=args.report,
        retries=args.retries,
        tags=args.tags,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
