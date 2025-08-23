"""Helpers for importing data from remote locations.

This module provides a :func:`remote_files` context manager that downloads
files from an SFTP or S3 location into a temporary directory.  The caller can
process the files locally while the context manager cleans up the temporary
storage when finished.
"""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import ParseResult, urlparse

# mypy: ignore-errors


# These third-party libraries are optional at runtime.  They are only imported
# when the corresponding URI scheme is used so the package can function without
# heavyweight dependencies installed.
try:  # pragma: no cover - optional dependency
    import boto3
except Exception:  # pragma: no cover - allow running without boto3
    boto3 = None

try:  # pragma: no cover - optional dependency
    import paramiko
except Exception:  # pragma: no cover - allow running without paramiko
    paramiko = None


__all__ = ["remote_files"]


@contextmanager
def remote_files(uri: str) -> Iterator[str]:
    """Yield a local directory containing files from ``uri``.

    Parameters
    ----------
    uri:
        ``sftp://user@host/path`` or ``s3://bucket/prefix`` style URI.

    Yields
    ------
    str
        Path to a temporary directory populated with the remote files.  The
        directory and its contents are removed when the context manager exits.
    """

    parsed = urlparse(uri)
    with tempfile.TemporaryDirectory() as tmpdir:
        if parsed.scheme == "sftp":
            _fetch_sftp(parsed, tmpdir)
        elif parsed.scheme == "s3":
            _fetch_s3(parsed, tmpdir)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported remote scheme: {parsed.scheme}")
        yield tmpdir


def _fetch_sftp(parsed: ParseResult, dst: str) -> None:
    """Download an SFTP tree into ``dst``.

    Parameters
    ----------
    parsed:
        Parsed result of :func:`urllib.parse.urlparse` for the SFTP URI.
    dst:
        Destination directory to populate with downloaded files.
    """

    if paramiko is None:  # pragma: no cover - dependency not installed
        raise RuntimeError("paramiko is required for SFTP imports")
    username = parsed.username or os.getenv("SFTP_USERNAME")
    password = parsed.password or os.getenv("SFTP_PASSWORD")
    host = parsed.hostname or "localhost"
    port = parsed.port or 22
    root = parsed.path or "/"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=username, password=password)
    sftp = client.open_sftp()
    try:
        for remote_path in _sftp_walk(sftp, root):
            rel = os.path.relpath(remote_path, root).lstrip("./")
            local_path = Path(dst, rel)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_path, str(local_path))
    finally:
        sftp.close()
        client.close()


def _sftp_walk(sftp: "paramiko.SFTPClient", path: str) -> Iterator[str]:
    for entry in sftp.listdir_attr(path):
        remote = f"{path.rstrip('/')}/{entry.filename}"
        if stat.S_ISDIR(entry.st_mode):
            yield from _sftp_walk(sftp, remote)
        else:
            yield remote


def _fetch_s3(parsed: ParseResult, dst: str) -> None:
    """Download objects from an S3 bucket into ``dst``.

    Parameters
    ----------
    parsed:
        Parsed S3 URI.
    dst:
        Directory where objects should be downloaded.
    """

    if boto3 is None:  # pragma: no cover - dependency not installed
        raise RuntimeError("boto3 is required for S3 imports")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = os.path.relpath(key, prefix) if prefix else key
            local_path = Path(dst, rel)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))
