# flake8: noqa
# mypy: ignore-errors
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

The command also understands remote locations.  Supplying ``--remote`` with an
``sftp://`` or ``s3://`` URI will fetch the files to a temporary directory using
:mod:`battery_analysis.utils.remote_import` before processing them locally.
"""

from __future__ import annotations

import argparse
import csv
import datetime
from dataclasses import dataclass, field
import fnmatch
import hashlib
import inspect
import json
import os
import tarfile
import tempfile
import time
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib
import zipfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, Set, Tuple, cast

from pandas.errors import ParserError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

try:
    import redis
except Exception:  # pragma: no cover - optional dependency
    redis = None

from battery_analysis import parsers
from battery_analysis.models import ImportJob, ImportJobSummary, Sample, TestResult
from battery_analysis.utils import data_update, file_storage, notifications


def update_cell_dataset(name: str) -> None:
    """Lazy wrapper so tests can monkeypatch update_cell_dataset."""
    from battery_analysis.utils.cell_dataset_builder import update_cell_dataset as _update

    _update(name)
from battery_analysis.utils.config import load_config
from battery_analysis.utils.db import ensure_connection
from battery_analysis.utils.logging import get_logger

logger = get_logger(__name__)

# Load configuration at module import so CLI defaults can reference it
CONFIG = load_config()

# Exceptions that trigger a retry when processing files
RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (ParserError, ConnectionError)
# Base delay used for exponential backoff between retries
RETRY_BASE_DELAY = 0.5

CONTROL_FILE = Path(__file__).resolve().parents[4] / ".import_control"


@dataclass(frozen=True)
class _DiscoveredFile:
    """Filesystem candidate yielded by the discovery stage."""

    index: int
    path: str
    mtime: float


@dataclass
class _PreparedFile:
    """Worker output produced after hashing and optional metadata lookup."""

    index: int
    path: str
    mtime: float
    file_hash: str
    sample: str
    attrs: Dict[str, object] = field(default_factory=dict)
    status: str = "ready"
    state_entry: Dict[str, object] | None = None


@dataclass
class _ImportResult:
    """Final per-file import result consumed by the coordinator stage."""

    prepared: _PreparedFile
    sample_name: str
    action: str
    test_id: object | None = None
    error: str | None = None


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


def _hash_file(path: str) -> str:
    """Return the MD5 digest for ``path`` using chunked reads."""

    h = hashlib.md5()
    with open(path, "rb") as bin_fh:
        reader = cast(BinaryIO, bin_fh)
        for chunk in iter(lambda: reader.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_sample_name(path: str) -> str:
    """Infer the default sample name from ``path`` without parser metadata."""

    return os.path.basename(os.path.dirname(path)) or "unknown"


def _iter_candidate_files(
    root: str,
    *,
    include: list[str],
    exclude: list[str],
    supported: set[str],
) -> Iterator[_DiscoveredFile]:
    """Yield supported files from ``root`` in discovery order.

    Discovery remains single-threaded so include/exclude filtering and walk
    ordering behave exactly as before, while downstream worker stages can start
    hashing and importing each yielded path immediately.
    """

    def _match(path: str, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatch(path, pat) for pat in patterns)

    idx = 0
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
            abs_path = os.path.abspath(os.path.join(dirpath, filename))
            yield _DiscoveredFile(index=idx, path=abs_path, mtime=os.path.getmtime(abs_path))
            idx += 1


def _prepare_file(
    candidate: _DiscoveredFile,
    *,
    reset: bool,
    previous_state: Dict[str, object],
    sample_lookup: bool,
    sample_map_data: Dict[str, str],
) -> _PreparedFile:
    """Hash a candidate file and resolve its sample metadata.

    The worker only calls :func:`battery_analysis.parsers.parse_file` when
    ``sample_lookup`` requires parser metadata. This keeps plain imports from
    eagerly parsing every file during discovery.
    """

    abs_path = candidate.path
    file_hash = _hash_file(abs_path)

    if not reset and abs_path in previous_state:
        entry = previous_state[abs_path]
        if isinstance(entry, dict):
            prev_mtime = entry.get("mtime")
            prev_hash = entry.get("hash")
        else:
            prev_mtime = entry
            prev_hash = None

        if prev_mtime == candidate.mtime and prev_hash == file_hash:
            return _PreparedFile(
                index=candidate.index,
                path=abs_path,
                mtime=candidate.mtime,
                file_hash=file_hash,
                sample=sample_map_data.get(abs_path, _default_sample_name(abs_path)),
                status="unchanged",
            )

        if prev_mtime == candidate.mtime and prev_hash is None:
            return _PreparedFile(
                index=candidate.index,
                path=abs_path,
                mtime=candidate.mtime,
                file_hash=file_hash,
                sample=sample_map_data.get(abs_path, _default_sample_name(abs_path)),
                status="unchanged",
                state_entry={"mtime": candidate.mtime, "hash": file_hash},
            )

    metadata: Dict[str, object] = {}
    if sample_lookup:
        try:
            _, parsed_metadata = parsers.parse_file(abs_path)
            if parsed_metadata:
                metadata = dict(parsed_metadata)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to parse %s: %s", abs_path, exc)

    name = _default_sample_name(abs_path)
    if sample_lookup:
        name = cast(str, metadata.get("sample_code") or name)
    if abs_path in sample_map_data:
        name = sample_map_data[abs_path]

    attrs: Dict[str, object] = {}
    if sample_lookup and metadata:
        attrs = {
            k: v
            for k, v in metadata.items()
            if k not in {"sample_code", "name"}
        }

    return _PreparedFile(
        index=candidate.index,
        path=abs_path,
        mtime=candidate.mtime,
        file_hash=file_hash,
        sample=name,
        attrs=attrs,
        state_entry={"mtime": candidate.mtime, "hash": file_hash},
    )


def _process_prepared_file(
    prepared: _PreparedFile,
    *,
    archive: bool,
    dry_run: bool,
    job: ImportJob | None,
    retries: int,
    tags: list[str] | None,
) -> _ImportResult:
    """Import a prepared file or report what would happen during dry runs."""

    if not ensure_connection():
        logger.error("Database connection not available")
        return _ImportResult(
            prepared=prepared,
            sample_name=prepared.sample,
            action="skipped",
            error="Database connection not available",
        )

    if dry_run:
        logger.info("Would process %s for sample %s", prepared.path, prepared.sample)
        return _ImportResult(
            prepared=prepared,
            sample_name=prepared.sample,
            action="dry_run",
        )

    sample = Sample.get_or_create(prepared.sample, **prepared.attrs)
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
                prepared.path, sample, archive=archive, job=job, tags=tags
            )
            break
        except RETRY_EXCEPTIONS as exc:
            if attempt >= retries:
                msg = f"{exc} after {attempt} retries"
                logger.error(
                    "Failed to process %s after %s retries: %s",
                    prepared.path,
                    attempt,
                    exc,
                )
                return _ImportResult(
                    prepared=prepared,
                    sample_name=prepared.sample,
                    action="skipped",
                    error=msg,
                )
            time.sleep(2**attempt * RETRY_BASE_DELAY)
            attempt += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to process %s: %s", prepared.path, exc)
            return _ImportResult(
                prepared=prepared,
                sample_name=prepared.sample,
                action="skipped",
                error=str(exc),
            )

    action = "updated" if was_update else "created"
    logger.info(
        "%s test %s for sample %s",
        action.title(),
        getattr(test, "id", None),
        sample.name,
    )
    return _ImportResult(
        prepared=prepared,
        sample_name=prepared.sample,
        action=action,
        test_id=getattr(test, "id", None),
    )


def _process_candidate_file(
    candidate: _DiscoveredFile,
    *,
    reset: bool,
    previous_state: Dict[str, object],
    sample_lookup: bool,
    sample_map_data: Dict[str, str],
    archive: bool,
    dry_run: bool,
    job: ImportJob | None,
    retries: int,
    tags: list[str] | None,
) -> _ImportResult:
    """Prepare and import a candidate file inside one worker task."""

    prepared = _prepare_file(
        candidate,
        reset=reset,
        previous_state=previous_state,
        sample_lookup=sample_lookup,
        sample_map_data=sample_map_data,
    )
    if prepared.status == "unchanged":
        return _ImportResult(
            prepared=prepared,
            sample_name=prepared.sample,
            action="unchanged",
        )
    return _process_prepared_file(
        prepared,
        archive=archive,
        dry_run=dry_run,
        job=job,
        retries=retries,
        tags=tags,
    )


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
    process = data_update.process_file_with_update
    params = inspect.signature(process).parameters
    if "sync_sample" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    ):
        test, was_update = process(
            abs_path,
            sample,
            sync_sample=True,
            recompute_sample=False,
        )
    else:
        test, was_update = process(abs_path, sample)

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
            test.save(sync_sample=True, recompute_sample_metrics=False)
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

    The importer now runs as a three-stage pipeline:

    1. **Discovery** walks the directory tree and yields supported file paths.
    2. **Worker tasks** hash files, compare ``.import_state.json``, optionally
       parse metadata for ``sample_lookup``, resolve sample names, and then
       import each prepared file.
    3. **Coordination** runs on the calling thread and is solely responsible for
       ordered bookkeeping: updating :class:`ImportJob` /
       :class:`ImportJobSummary`, publishing Redis progress messages, updating
       ``.import_state.json``, and writing the optional report.

    Because discovery and worker tasks overlap, operators may see progress begin
    before the full walk finishes. The reported ``total`` grows as files finish
    preparation and are confirmed as work items. ``preview_samples`` still lists
    samples before import; in that mode discovery overlaps only with the
    preparation stage and import starts only after preview confirmation.

    Parameters
    ----------
    root:
        Root directory to search for files.
    sample_lookup:
        When ``True`` parser metadata (for example ``sample_code``) is used to
        determine the sample and any extra sample attributes.
    reset:
        When ``True`` any existing import state is ignored and all files are
        reprocessed.
    dry_run:
        When ``True`` report what would happen without creating samples,
        importing tests, or refreshing datasets.
    workers:
        Number of worker threads to use when preparing and importing files.
        ``None`` uses the CPU count.
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
        for the job are skipped and new imports are appended to the existing job
        record.
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
    summary: ImportJobSummary | None = None
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
        try:
            summary = ImportJobSummary().save()
        except Exception:  # pragma: no cover - best effort
            summary = None

    include = include or []
    exclude = exclude or []
    supported = {ext.lower() for ext in parsers.get_supported_formats()}
    workers = workers or (os.cpu_count() or 1)
    max_in_flight = max(1, workers * 2)

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

    sample_map_data: Dict[str, str] = {}
    if sample_map and os.path.exists(sample_map):
        sample_map_data = _load_sample_map(sample_map)

            metadata = None
            try:
                _, metadata = parsers.parse_file(abs_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to parse %s: %s", abs_path, exc)

            name = metadata.get("sample_code") if metadata else None
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

    entries.sort(key=lambda entry: cast(str, entry["path"]))

    total = start_idx + len(entries)
    processed_samples: Set[str] = set()
    report_entries: List[Tuple[str, str, object | None]] = []
    current_paths: Set[str] = set()
    prepared_entries: list[_PreparedFile] = []
    discovered_work = 0
    completed_work = 0
    created = 0
    updated = 0
    skipped = 0
    cancelled = False
    paused = False

    pub = None
    channel = CONFIG.get("redis_channel", "import_progress")
    if redis and CONFIG.get("redis_url"):
        try:  # pragma: no cover - optional
            pub = redis.from_url(CONFIG["redis_url"])
        except Exception:
            pub = None

    def _save_state_file() -> None:
        if dry_run:
            return
        try:
            with open(state_path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2, sort_keys=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to write state to %s: %s", state_path, exc)

    def _publish_progress(payload: Dict[str, object]) -> None:
        if not pub:
            return
        try:
            pub.publish(channel, json.dumps(payload))
        except Exception:
            pass

    def _sync_progress(*, force: bool = False, current_file: str | None = None) -> None:
        if summary is not None:
            summary.total_count = start_idx + discovered_work
            summary.processed_count = start_idx + completed_work
            summary.created_count = created
            summary.updated_count = updated
            summary.skipped_count = skipped
            try:
                if force or ((start_idx + completed_work) % 5 == 0):
                    summary.save()
            except Exception:
                pass

        if job is not None:
            job.total_count = max(job.total_count, start_idx + discovered_work)
            job.processed_count = start_idx + completed_work
            if current_file is not None:
                job.current_file = current_file
            try:
                if force or ((start_idx + completed_work) % 5 == 0):
                    job.save()
            except Exception:
                pass
            payload: Dict[str, object] = {
                "job_id": str(job.id),
                "processed": start_idx + completed_work,
                "total": job.total_count,
            }
            if current_file is not None:
                payload["current_file"] = current_file
            if force or current_file is not None:
                _publish_progress(payload)

    def _handle_control() -> None:
        nonlocal cancelled, paused
        cmd = _read_control_command()
        if cmd == "cancel":
            logger.info("Import cancelled via control file")
            cancelled = True
            return
        if cmd != "pause":
            return

        if not paused:
            logger.info("Import paused via control file")
            paused = True
        while True:
            time.sleep(0.2)
            cmd = _read_control_command()
            if cmd == "cancel":
                logger.info("Import cancelled via control file")
                cancelled = True
                return
            if cmd != "pause":
                logger.info("Import resumed")
                paused = False
                return

    _sync_progress(force=True)

    if job is not None:
        _publish_progress({"job_id": str(job.id), "total": job.total_count})

    prepare_futures: Dict[Future[_PreparedFile], _DiscoveredFile] = {}
    work_futures: Dict[Future[_ImportResult], _PreparedFile | _DiscoveredFile] = {}

    def _submit_import(executor: ThreadPoolExecutor, prepared: _PreparedFile) -> None:
        nonlocal discovered_work
        discovered_work += 1
        prepared_entries.append(prepared)
        future = executor.submit(
            _process_prepared_file,
            prepared,
            archive=archive,
            dry_run=dry_run,
            job=job,
            retries=retries,
            tags=tags,
        )
        work_futures[future] = prepared
        _sync_progress(force=True)

    def _record_completed_result(result: _ImportResult) -> None:
        nonlocal completed_work, created, updated, skipped, state_dirty

        completed_work += 1
        if result.action == "updated":
            updated += 1
            processed_samples.add(result.sample_name)
        elif result.action == "created":
            created += 1
            processed_samples.add(result.sample_name)
        elif result.action == "dry_run":
            processed_samples.add(result.sample_name)
        elif result.action == "skipped":
            skipped += 1

        if result.action in {"updated", "created"} and not dry_run:
            state[result.prepared.path] = result.prepared.state_entry or {
                "mtime": result.prepared.mtime,
                "hash": result.prepared.file_hash,
            }
            state_dirty = True
            _save_state_file()

        detail = str(result.test_id) if result.test_id is not None else result.error
        report_entries.append((result.prepared.path, result.action, detail))

        if summary is not None and result.error is not None:
            summary.errors.append(result.error)
        if job is not None:
            entry = {"path": result.prepared.path, "action": result.action}
            if result.test_id is not None:
                entry["test_id"] = str(result.test_id)
            if result.error is not None:
                entry["error"] = result.error
                job.errors.append(result.error)
            job.files.append(entry)

        _sync_progress(
            force=((start_idx + completed_work) == (start_idx + discovered_work)),
            current_file=result.prepared.path,
        )
        if (start_idx + completed_work) % 10 == 0 or (start_idx + completed_work) == (
            start_idx + discovered_work
        ):
            logger.info("Processed %s/%s", start_idx + completed_work, start_idx + discovered_work)

    def _drain_futures(
        executor: ThreadPoolExecutor,
        *,
        block: bool,
        allow_import_submission: bool,
    ) -> None:
        nonlocal discovered_work, skipped, state_dirty

        while True:
            active = list(prepare_futures) + list(work_futures)
            if not active:
                return
            timeout = None if block else 0
            done, _ = wait(active, timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                return
            for future in done:
                if future in prepare_futures:
                    prepare_futures.pop(future)
                    prepared = future.result()
                    if prepared.status == "unchanged":
                        skipped += 1
                        if prepared.state_entry is not None and not dry_run:
                            state[prepared.path] = prepared.state_entry
                            state_dirty = True
                        _sync_progress(force=True)
                        _handle_control()
                        if cancelled:
                            return
                        continue
                    if allow_import_submission:
                        _submit_import(executor, prepared)
                    else:
                        prepared_entries.append(prepared)
                else:
                    source = work_futures.pop(future)
                    result = future.result()
                    if result.action == "unchanged":
                        skipped += 1
                        if result.prepared.state_entry is not None and not dry_run:
                            state[result.prepared.path] = result.prepared.state_entry
                            state_dirty = True
                        _sync_progress(force=True)
                    else:
                        if isinstance(source, _DiscoveredFile):
                            discovered_work += 1
                        _record_completed_result(result)
                _handle_control()
                if cancelled:
                    return
            if not block:
                return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for candidate in _iter_candidate_files(
            root,
            include=include,
            exclude=exclude,
            supported=supported,
        ):
            current_paths.add(candidate.path)
            if resume and candidate.path in processed_paths:
                continue

            while len(prepare_futures) + len(work_futures) >= max_in_flight:
                _drain_futures(executor, block=True, allow_import_submission=not preview_samples)
                if cancelled:
                    break
            if cancelled:
                break

            if preview_samples:
                future = executor.submit(
                    _prepare_file,
                    candidate,
                    reset=reset,
                    previous_state=original_state,
                    sample_lookup=sample_lookup,
                    sample_map_data=sample_map_data,
                )
                prepare_futures[future] = candidate
            else:
                future = executor.submit(
                    _process_candidate_file,
                    candidate,
                    reset=reset,
                    previous_state=original_state,
                    sample_lookup=sample_lookup,
                    sample_map_data=sample_map_data,
                    archive=archive,
                    dry_run=dry_run,
                    job=job,
                    retries=retries,
                    tags=tags,
                )
                work_futures[future] = candidate
            _drain_futures(executor, block=False, allow_import_submission=not preview_samples)
            if cancelled:
                break

        if not cancelled:
            _drain_futures(executor, block=True, allow_import_submission=not preview_samples)

        if not reset:
            missing_paths = set(state.keys()) - current_paths
            if missing_paths:
                for missing_path in missing_paths:
                    state.pop(missing_path, None)
                state_dirty = True

        if preview_samples:
            prepared_entries.sort(key=lambda entry: entry.index)
            pairs = [(entry.path, entry.sample) for entry in prepared_entries]
            wrote_sample_map = False
            if sample_map and not os.path.exists(sample_map):
                _write_sample_map(sample_map, pairs)
                wrote_sample_map = True
            if sample_map and os.path.exists(sample_map):
                mapping = _load_sample_map(sample_map)
                for entry in prepared_entries:
                    if entry.path in mapping:
                        entry.sample = mapping[entry.path]
                pairs = [(entry.path, entry.sample) for entry in prepared_entries]

            max_len = max((len(p) for p, _ in pairs), default=10)
            header = "File Path".ljust(max_len) + " | Sample"
            print(header)
            print("-" * len(header))
            for fp, sample in pairs:
                print(fp.ljust(max_len) + " | " + sample)
            if sample_map and wrote_sample_map:
                print(f"Sample map written to {sample_map}")
            if not confirm or cancelled:
                return 0
            else:
                for prepared in prepared_entries:
                    while len(work_futures) >= max_in_flight:
                        _drain_futures(executor, block=True, allow_import_submission=False)
                        if cancelled:
                            break
                    if cancelled:
                        break
                    _submit_import(executor, prepared)
                    _drain_futures(executor, block=False, allow_import_submission=False)
                if not cancelled:
                    _drain_futures(executor, block=True, allow_import_submission=False)

    if cancelled:
        return 0

    if dry_run:
        for name in processed:
            logger.info("Would consolidate sample metrics for %s", name)
            logger.info("Would refresh dataset for %s", name)
    else:
        for name in processed:
            sample = Sample.get_by_name(name)
            if sample is None:
                continue
            try:
                sample.recompute_metrics()
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to consolidate sample %s: %s", name, exc)
                continue
        for name in processed_samples:
            logger.info("Would refresh dataset for %s", name)
    else:
        for name in processed_samples:
            try:
                update_cell_dataset(name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to refresh dataset for %s: %s", name, exc)

    if (state_dirty or (not reset and state != original_state)) and not dry_run:
        _save_state_file()

    logger.info(
        "Summary: created=%s, updated=%s, skipped=%s",
        created,
        updated,
        skipped,
    )

    final_total = start_idx + discovered_work
    if summary is not None:
        summary.end_time = datetime.datetime.utcnow()
        summary.total_count = final_total
        summary.processed_count = start_idx + completed_work
        summary.created_count = created
        summary.updated_count = updated
        summary.skipped_count = skipped
        summary.status = "failed" if summary.errors else "completed"
        try:
            summary.save()
        except Exception:
            pass

    if job is not None:
        job.end_time = datetime.datetime.utcnow()
        job.current_file = None
        job.total_count = max(job.total_count, final_total)
        job.processed_count = start_idx + completed_work
        try:
            job.save()
        except Exception:  # pragma: no cover - best effort
            pass
        _publish_progress({"job_id": str(job.id), "status": "completed"})

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


def show_history() -> int:
    """Print history of import job summaries."""

    if not ensure_connection():
        logger.error("Database connection not available")
        return 1

    summaries: list[ImportJobSummary]
    if callable(getattr(ImportJobSummary, "objects", None)):
        summaries = list(ImportJobSummary.objects())
        summaries.sort(
            key=lambda s: getattr(s, "start_time", None), reverse=True
        )
    else:  # pragma: no cover - defensive
        summaries = []
    if not summaries:
        print("No import summaries found")
        return 0

    for s in summaries:
        start = s.start_time.isoformat() if s.start_time else "N/A"
        end = s.end_time.isoformat() if s.end_time else "N/A"
        counts = (
            f"created={getattr(s, 'created_count', 0)} "
            f"updated={getattr(s, 'updated_count', 0)} "
            f"skipped={getattr(s, 'skipped_count', 0)}"
        )
        errs = "; ".join(getattr(s, "errors", [])) or "None"
        status = getattr(s, "status", "")
        print(
            f"{s.id} | start: {start} | end: {end} | {counts} | status: {status} | errors: {errs}"
        )
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
        "--remote",
        metavar="URI",
        help="Remote SFTP or S3 path to import (e.g. sftp://user@host/path)",
    )
    parser.add_argument(
        "--archive",
        metavar="PATH",
        help="ZIP or TAR.GZ archive to import",
    )
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
        "--history",
        action="store_true",
        help="Show import job summaries and exit",
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify imported files against stored hashes after processing",
    )
    parser.add_argument(
        "--verify-report",
        metavar="PATH",
        help="Write verification report to PATH (CSV or JSON)",
    )
    args = parser.parse_args(argv)

    from battery_analysis.utils.logging import get_logger

    _logger = get_logger(__name__)
    if args.history:
        return show_history()
    if args.status is not None:
        return show_status(args.status or None)
    if args.rollback:
        return rollback_job(args.rollback)
    if args.remote and args.archive:
        parser.error("--archive cannot be used with --remote")

    if not args.root and not args.remote and not args.archive and not args.history:
        parser.error(
            "root is required unless --rollback, --status, --history, --remote or --archive is specified",
        )
    include = args.include if args.include is not None else CONFIG.get("include")
    exclude = args.exclude if args.exclude is not None else CONFIG.get("exclude")

    if args.remote:
        from battery_analysis.utils import remote_import

        with remote_import.remote_files(args.remote) as tmpdir:
            root_path = tmpdir
            result = import_directory(
                tmpdir,
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
    elif args.archive:
        with tempfile.TemporaryDirectory() as tmpdir:
            if args.archive.lower().endswith('.zip'):
                with zipfile.ZipFile(args.archive) as zf:
                    zf.extractall(tmpdir)
            elif args.archive.lower().endswith(('.tar.gz', '.tgz')):
                with tarfile.open(args.archive, 'r:gz') as tf:
                    tf.extractall(tmpdir)
            else:
                parser.error('Unsupported archive format: use .zip or .tar.gz')
            root_path = tmpdir
            result = import_directory(
                tmpdir,
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
    else:
        root_path = args.root
        result = import_directory(
            root_path,
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
    if args.verify and result == 0:
        from battery_analysis.utils import verify_import

        rows = verify_import.verify_directory(root_path)
        if args.verify_report:
            verify_import.write_report(rows, args.verify_report)
        summary = verify_import.summarize_discrepancies(rows)
        print(
            f"Added: {summary['added']} | Mismatched: {summary['mismatched']} | Missing: {summary['missing']}"
        )
        if rows:
            return 1

    return result


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
