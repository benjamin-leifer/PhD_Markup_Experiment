"""Utility helpers for ingesting, retrieving, and validating image files.

Examples
--------
>>> raw = ingest_image_file("foo.png", create_thumbnail=True)
>>> data = get_image(raw)  # doctest: +SKIP
>>> thumb_path = get_thumbnail(raw, as_file_path=True)  # doctest: +SKIP

``get_image`` and ``get_thumbnail`` accept either a :class:`RawDataFile` instance
or its string ``id``. When ``as_file_path`` is ``True`` the file is written to a
temporary location and the filesystem path is returned; otherwise the raw bytes
are provided.
"""

from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Iterator, Union

from PIL import Image

from battery_analysis.models import RawDataFile

from .file_storage import get_raw_data_file_by_id, store_raw_data_file


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


def generate_thumbnail(
    raw_file: RawDataFile, size: tuple[int, int] = (256, 256)
) -> RawDataFile:
    """Create and store a thumbnail for ``raw_file``.

    The original image is read from GridFS, resized using :mod:`Pillow`, and
    stored as a new :class:`RawDataFile` with ``file_type='thumbnail'`` and a
    ``"thumbnail"`` tag. Basic metadata such as image dimensions are recorded
    on the stored document.

    Parameters
    ----------
    raw_file:
        The original image file stored in GridFS.
    size:
        Maximum size (width, height) of the generated thumbnail. Defaults to
        ``(256, 256)``.

    Returns
    -------
    RawDataFile
        The stored thumbnail document.
    """

    raw_file.file_data.seek(0)
    image_bytes = raw_file.file_data.read()

    with Image.open(BytesIO(image_bytes)) as img:
        img.thumbnail(size)
        buf = BytesIO()
        img.save(buf, format=img.format or "PNG")
        buf.seek(0)

        suffix = f".{(img.format or 'png').lower()}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(buf.read())
            temp_path = tmp.name

    try:
        try:
            thumb = store_raw_data_file(
                temp_path,
                sample=getattr(raw_file, "sample", None),
                test_result=getattr(raw_file, "test_result", None),
                tags=["thumbnail"],
                file_type="thumbnail",
            )
        except TypeError:
            thumb = store_raw_data_file(
                temp_path,
                test_result=getattr(raw_file, "test_result", None),
                file_type="thumbnail",
            )
            if getattr(raw_file, "sample", None) is not None:
                thumb.sample = raw_file.sample
            thumb.tags = ["thumbnail"]

        thumb.metadata = {
            "width": img.width,
            "height": img.height,
            "source_file_id": str(raw_file.id),
            "format": img.format,
        }
        thumb.save()
        return thumb
    finally:
        os.remove(temp_path)


def get_image(raw_file: RawDataFile | str, as_file_path: bool = False):
    """Retrieve the original image data for ``raw_file``.

    Parameters
    ----------
    raw_file:
        The original :class:`RawDataFile` instance or its ``id`` as a string.
    as_file_path:
        When ``True`` a temporary file path is returned instead of bytes.
    """

    file_id = str(raw_file.id) if isinstance(raw_file, RawDataFile) else str(raw_file)
    return get_raw_data_file_by_id(file_id, as_file_path=as_file_path)


def get_thumbnail(raw_file: RawDataFile | str, as_file_path: bool = False):
    """Retrieve the stored thumbnail corresponding to ``raw_file``.

    The thumbnail is searched by looking for a :class:`RawDataFile` whose
    ``metadata['source_file_id']`` matches the id of ``raw_file`` and which is
    tagged with ``"thumbnail"`` or has ``file_type='thumbnail'``.

    Parameters
    ----------
    raw_file:
        The original :class:`RawDataFile` or its ``id``.
    as_file_path:
        When ``True`` a temporary file path is returned instead of bytes.
    """

    file_id = str(raw_file.id) if isinstance(raw_file, RawDataFile) else str(raw_file)

    thumb = RawDataFile.objects(
        metadata__source_file_id=str(file_id), tags="thumbnail"
    ).first()
    if not thumb:
        thumb = RawDataFile.objects(
            metadata__source_file_id=str(file_id), file_type="thumbnail"
        ).first()

    if not thumb:
        raise ValueError(f"No thumbnail found for raw file {file_id}")

    return get_raw_data_file_by_id(str(thumb.id), as_file_path=as_file_path)


def ingest_image_file(
    path: Union[str, os.PathLike[str]],
    sample: Any = None,
    test_result: Any = None,
    operator: Any = None,
    device: Any = None,
    tags: Any = None,
    create_thumbnail: bool = False,
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
        raw_file = store_raw_data_file(abs_path, **metadata)
    except TypeError:
        # Fallback for versions of store_raw_data_file that don't support
        # the extended metadata parameters.
        fallback = {
            "test_result": test_result,
            "file_type": "image",
        }
        fallback = {k: v for k, v in fallback.items() if v is not None}
        raw_file = store_raw_data_file(abs_path, **fallback)
        if sample is not None:
            raw_file.sample = sample

    if sample is not None and hasattr(sample, "images"):
        sample.images.append(raw_file)
        sample.save()

    if create_thumbnail:
        generate_thumbnail(raw_file)

    return raw_file
