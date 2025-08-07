"""Utility helpers for storing and retrieving Python objects in GridFS.

This module provides simple functions to serialize Python objects (typically
small dictionaries or arrays) to MongoDB's GridFS and read them back.  Data is
stored using Python's :mod:`pickle` module which keeps the structure intact.
"""

from __future__ import annotations

import io
import pickle
from typing import Any


def data_to_gridfs(file_field, data: Any, filename: str) -> None:
    """Serialize ``data`` and store it in ``file_field``.

    Parameters
    ----------
    file_field:
        A :class:`mongoengine.fields.GridFSProxy` or similar object returned
        from a ``FileField`` on a document.
    data:
        The Python object to serialize.  This should be pickleable.
    filename:
        The filename to store alongside the GridFS file.
    """
    buffer = io.BytesIO()
    pickle.dump(data, buffer)
    buffer.seek(0)
    file_field.put(
        buffer,
        filename=filename,
        content_type="application/python-pickle",
    )


def gridfs_to_data(file_field) -> Any:
    """Read and deserialize data from ``file_field``.

    Parameters
    ----------
    file_field:
        The GridFS proxy object from which data should be read.

    Returns
    -------
    Any
        The deserialized Python object, or ``None`` if ``file_field`` is empty.
    """
    if not file_field:
        return None
    data_bytes = file_field.read()
    return pickle.loads(data_bytes)
