"""Watch directories for new or modified data files.

This module provides utilities to monitor a directory tree and automatically
process newly created or modified files using
``import_directory.process_file_with_update``.  The watcher is based on the
`watchdog <https://python-watchdog.readthedocs.io/>`_ package.

Example
-------
The watcher can be executed directly as a module::

    python -m battery_analysis.utils.import_watcher DATA_DIR

This will watch ``DATA_DIR`` recursively for files with extensions supported by
:func:`battery_analysis.parsers.get_supported_formats`.  When a file is created
or changed, the watcher determines the sample from the parent directory name and
invokes :func:`import_directory.process_file_with_update`.

The command line interface exposes options for a debounce interval (time to wait
before processing after a change) and maximum recursion depth.
"""

from __future__ import annotations

import argparse
import os
import threading
import time

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from battery_analysis import parsers
from battery_analysis.models import Sample
from battery_analysis.utils import import_directory
from battery_analysis.utils.db import ensure_connection
from battery_analysis.utils.config import load_config
from battery_analysis.utils.logging import get_logger

logger = get_logger(__name__)

# Load configuration at import so CLI defaults and watch() can use it
CONFIG = load_config()


class _ImportEventHandler(FileSystemEventHandler):  # type: ignore[misc]
    """Handle filesystem events and trigger imports."""

    def __init__(
        self, root: str, debounce: float, max_depth: int | None, tags: list[str] | None
    ) -> None:
        super().__init__()
        self.root = os.path.abspath(root)
        self.debounce = debounce
        self.max_depth = max_depth
        self.tags = tags or []
        self.supported = {ext.lower() for ext in parsers.get_supported_formats()}
        self._timers: dict[str, threading.Timer] = {}

    # ``watchdog`` may emit multiple events for a single file write.  To avoid
    # redundant work we debounce events using a ``threading.Timer``.
    def _queue(self, src_path: str) -> None:
        if os.path.isdir(src_path):
            return
        ext = os.path.splitext(src_path)[1].lower()
        if ext not in self.supported:
            return
        rel = os.path.relpath(src_path, self.root)
        if rel.startswith(".."):
            return
        if self.max_depth is not None and rel.count(os.sep) > self.max_depth:
            return
        timer = self._timers.pop(src_path, None)
        if timer:
            timer.cancel()
        timer = threading.Timer(self.debounce, self._process, args=(src_path,))
        self._timers[src_path] = timer
        timer.start()

    def on_created(
        self, event: FileSystemEvent
    ) -> None:  # pragma: no cover - behaviour tested via _queue
        self._queue(event.src_path)

    def on_modified(
        self, event: FileSystemEvent
    ) -> None:  # pragma: no cover - behaviour tested via _queue
        self._queue(event.src_path)

    def _process(self, src_path: str) -> None:
        self._timers.pop(src_path, None)
        sample_name = os.path.basename(os.path.dirname(src_path)) or "unknown"
        sample = Sample.get_or_create(sample_name)
        try:
            import_directory.process_file_with_update(src_path, sample, tags=self.tags)
        except Exception:  # pragma: no cover - best effort logging
            logger.exception("Failed to process %s", src_path)


def watch(
    root: str,
    *,
    debounce: float = 1.0,
    depth: int | None = None,
    tags: list[str] | None = None,
) -> Observer:
    """Start watching ``root`` and return the active :class:`Observer`.

    Parameters
    ----------
    root:
        Directory to monitor.
    debounce:
        Seconds to wait after the last event before processing a file.
    depth:
        Maximum recursion depth relative to ``root``. ``None`` means unlimited.
    """

    if not ensure_connection(host=CONFIG.get("db_uri")):
        raise RuntimeError("Database connection not available")

    handler = _ImportEventHandler(root, debounce, depth, tags)
    observer = Observer()
    observer.schedule(handler, root, recursive=True)
    observer.start()
    return observer


# --- Programmatic watcher management ------------------------------------

# Store active observers keyed by their root directory.  The start time is
# tracked so callers can report how long a watcher has been running.
_WATCHERS: dict[str, tuple[Observer, float]] = {}
_WATCHERS_LOCK = threading.Lock()


def start_watcher(
    root: str, *, debounce: float = 1.0, depth: int | None = None, tags: list[str] | None = None
) -> None:
    """Start watching ``root`` and remember the observer.

    This is a thin wrapper around :func:`watch` that keeps track of active
    watchers so they can later be queried or stopped programmatically.
    ``watch`` is still exposed for backwards compatibility.
    """

    with _WATCHERS_LOCK:
        if root in _WATCHERS:
            return
        observer = watch(root, debounce=debounce, depth=depth, tags=tags)
        _WATCHERS[root] = (observer, time.time())


def stop_watcher(root: str) -> None:
    """Stop the watcher monitoring ``root`` if it is running."""

    with _WATCHERS_LOCK:
        info = _WATCHERS.pop(root, None)
    if not info:
        return
    observer, _ = info
    observer.stop()
    observer.join()


def list_watchers() -> list[dict[str, float]]:
    """Return information about all active watchers.

    Each dictionary contains ``directory`` and ``uptime`` (in seconds).
    """

    with _WATCHERS_LOCK:
        items = list(_WATCHERS.items())
    now = time.time()
    return [
        {"directory": path, "uptime": now - start}
        for path, (_, start) in items
    ]


def is_watching(root: str) -> bool:
    """Return ``True`` if ``root`` currently has an active watcher."""

    with _WATCHERS_LOCK:
        return root in _WATCHERS


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the import watcher."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Directory to monitor for imports")
    parser.add_argument(
        "--debounce",
        type=float,
        default=CONFIG.get("debounce", 1.0),
        help="Seconds to wait after changes before processing",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=CONFIG.get("depth"),
        help="Maximum recursion depth to monitor (default: unlimited)",
    )
    parser.add_argument(
        "--tags",
        action="append",
        default=None,
        help="Tag to apply to imported samples and tests (repeatable)",
    )
    args = parser.parse_args(argv)

    observer = watch(args.path, debounce=args.debounce, depth=args.depth, tags=args.tags)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping watcher")
    finally:
        observer.stop()
        observer.join()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    raise SystemExit(main())
