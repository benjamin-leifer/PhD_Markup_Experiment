"""Command-line helper to retrieve raw files archived in GridFS.

This utility provides a small interface for downloading raw files stored
with :func:`battery_analysis.utils.file_storage.save_raw`.

Usage examples::

    # Fetch by RawDataFile id and write to stdout
    python -m battery_analysis.utils.raw_file_cli download <FILE_ID>

    # Fetch the file linked to a TestResult and save locally
    python -m battery_analysis.utils.raw_file_cli by-test <TEST_ID> --out result.csv
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from battery_analysis.models import TestResult
from battery_analysis.utils import file_storage
from battery_analysis.utils.db import ensure_connection


def _write_output(data: bytes | str, out: str | None) -> None:
    """Write ``data`` to ``out`` or stdout.

    ``data`` is either the raw bytes of the file or the path to a temporary
    file produced by :func:`file_storage.retrieve_raw` when ``as_file_path`` is
    ``True``.
    """

    if out:
        dest = Path(out)
        if isinstance(data, str):
            shutil.copyfile(data, dest)
            try:
                os.remove(data)
            except OSError:
                pass
        else:
            dest.write_bytes(data)
    else:
        if isinstance(data, str):
            with open(data, "rb") as fh:
                shutil.copyfileobj(fh, sys.stdout.buffer)
            try:
                os.remove(data)
            except OSError:
                pass
        else:
            sys.stdout.buffer.write(data)


def _download(file_id: str, out: str | None) -> None:
    data = file_storage.retrieve_raw(file_id, as_file_path=bool(out))
    _write_output(data, out)


def _by_test(test_id: str, out: str | None) -> None:
    test = TestResult.objects(id=test_id).first()
    if not test or not getattr(test, "file_id", None):
        raise SystemExit(f"No raw data file linked to test {test_id}")
    _download(test.file_id, out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="fetch by RawDataFile id")
    p_dl.add_argument("file_id", help="RawDataFile identifier")
    p_dl.add_argument("--out", "-o", help="output path; defaults to stdout")

    p_bt = sub.add_parser(
        "by-test", help="fetch file linked to a TestResult identifier"
    )
    p_bt.add_argument("test_id", help="TestResult identifier")
    p_bt.add_argument("--out", "-o", help="output path; defaults to stdout")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_connection()

    if args.cmd == "download":
        _download(args.file_id, args.out)
    elif args.cmd == "by-test":
        _by_test(args.test_id, args.out)
    else:  # pragma: no cover - argparse enforces choices
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - manual use
    main()
