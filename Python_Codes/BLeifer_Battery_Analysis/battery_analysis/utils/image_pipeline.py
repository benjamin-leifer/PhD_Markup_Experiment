"""Utility functions for ingesting and validating image files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Union, Any

from .file_storage import store_raw_data_file


_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff"}


def collect_image_paths(source: Any) -> Iterator[str]:
    """Yield image file paths from ``source``.

    Parameters
    ----------
    source:
        May be a directory path, an iterable of paths or file-like objects,
        a single file path, or an uploaded file-like object providing a ``read``
        method (e.g. ``werkzeug.FileStorage``).

    Yields
    ------
    str
        Paths to image files on the local filesystem. Uploaded file-like
        objects are written to a temporary file before yielding the path.
    """

    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_dir():
            for entry in path.iterdir():
                if entry.is_file():
                    yield str(entry)
        elif path.exists():
            # single file
            yield str(path)
        else:
            raise ValueError(f"Path does not exist: {source}")
    elif isinstance(source, Iterable) and not hasattr(source, "read"):
        for item in source:
            # Recursively handle nested iterables or file-like objects
            yield from collect_image_paths(item)
    elif hasattr(source, "read"):
        # Handle uploaded file-like object
        filename = getattr(source, "filename", "")
        suffix = Path(filename).suffix if filename else ""
        data = source.read()
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as tmp:
                tmp.write(data)
        finally:
            try:
                source.close()
            except Exception:
                pass
        yield temp_path
    else:
        raise TypeError(f"Unsupported source type: {type(source)!r}")


def validate_image(
    path: Union[str, os.PathLike[str]],
    max_bytes: int = 10_000_000,
    allowed_ext: set[str] = _ALLOWED_EXTENSIONS,
) -> None:
    """Validate image file constraints.

    Parameters
    ----------
    path:
        Path to the image file.
    max_bytes:
        Maximum allowed file size in bytes. Defaults to 10 MB.
    allowed_ext:
        Set of allowed file extensions. Defaults to PNG/JPEG/TIFF formats.

    Raises
    ------
    ValueError
        If the file does not exist, exceeds size limits, or has an
        unsupported extension.
    """

    p = Path(path)

    if not p.is_file():
        raise ValueError(f"Image file not found: {p}")

    ext = p.suffix.lower()
    if ext not in allowed_ext:
        raise ValueError(f"Unsupported image extension: {ext}")

    size = p.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"Image file too large: {size} bytes exceeds limit of {max_bytes}"
        )


def ingest_image_file(
    path: Union[str, os.PathLike[str]],
    sample: Any = None,
    test_result: Any = None,
    operator: Any = None,
    device: Any = None,
    tags: Any = None,
) -> Any:
    """Validate and store an image file in persistent storage.

    Parameters
    ----------
    path:
        Path to the image file.
    sample, test_result, operator, device, tags:
        Optional metadata to associate with the stored image. These values are
        forwarded to :func:`store_raw_data_file` when supported.
    """

    abs_path = os.path.abspath(path)
    validate_image(abs_path)

    metadata = {
        "sample": sample,
        "test_result": test_result,
        "operator": operator,
        "device": device,
        "tags": tags,
        "file_type": "image",
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    try:
        return store_raw_data_file(abs_path, **metadata)
    except TypeError:
        # Fallback for versions of store_raw_data_file that don't support
        # the extended metadata parameters.
        fallback = {
            "test_result": test_result,
            "file_type": "image",
        }
        fallback = {k: v for k, v in fallback.items() if v is not None}
        return store_raw_data_file(abs_path, **fallback)
