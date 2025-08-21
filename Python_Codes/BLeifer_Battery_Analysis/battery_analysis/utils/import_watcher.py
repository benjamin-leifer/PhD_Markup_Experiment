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
import logging
import os
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from battery_analysis import parsers
from battery_analysis.models import Sample
from battery_analysis.utils import import_directory

logger = logging.getLogger(__name__)


class _ImportEventHandler(FileSystemEventHandler):
    """Handle filesystem events and trigger imports."""

    def __init__(self, root: str, debounce: float, max_depth: int | None) -> None:
        super().__init__()
        self.root = os.path.abspath(root)
        self.debounce = debounce
        self.max_depth = max_depth
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

    def on_created(self, event):  # pragma: no cover - behaviour tested via _queue
        self._queue(event.src_path)

    def on_modified(self, event):  # pragma: no cover - behaviour tested via _queue
        self._queue(event.src_path)

    def _process(self, src_path: str) -> None:
        self._timers.pop(src_path, None)
        sample_name = os.path.basename(os.path.dirname(src_path)) or "unknown"
        sample = Sample.get_or_create(sample_name)
        try:
            import_directory.process_file_with_update(src_path, sample)
        except Exception:  # pragma: no cover - best effort logging
            logger.exception("Failed to process %s", src_path)


def watch(root: str, *, debounce: float = 1.0, depth: int | None = None) -> Observer:
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

    handler = _ImportEventHandler(root, debounce, depth)
    observer = Observer()
    observer.schedule(handler, root, recursive=True)
    observer.start()
    return observer


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the import watcher."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Directory to monitor for imports")
    parser.add_argument(
        "--debounce",
        type=float,
        default=1.0,
        help="Seconds to wait after changes before processing",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Maximum recursion depth to monitor (default: unlimited)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    observer = watch(args.path, debounce=args.debounce, depth=args.depth)
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
