"""Shared logging configuration for battery analysis utilities.

This module exposes a :func:`get_logger` helper that returns a module-level
logger configured with a rotating file handler.  When a ``mongo_uri`` is
provided and :mod:`pymongo` is available, log records are also inserted into
``mongo_db.mongo_collection``.

The log file location can be customised via the ``log_file`` argument or the
``BATTERY_LOG_FILE`` environment variable.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
from typing import Optional

try:  # pragma: no cover - optional dependency
    from pymongo import MongoClient  # type: ignore
except Exception:  # pragma: no cover - the handler simply won't be added
    MongoClient = None  # type: ignore

# Default log file path; can be overridden via environment variable
DEFAULT_LOG_FILE = Path(os.getenv("BATTERY_LOG_FILE", "battery_analysis.log"))


def get_logger(
    name: str = "battery_analysis",
    *,
    log_file: Optional[str | Path] = None,
    mongo_uri: Optional[str] = None,
    mongo_db: str = "logs",
    mongo_collection: str = "logs",
) -> logging.Logger:
    """Return a configured :class:`logging.Logger` instance.

    Parameters
    ----------
    name:
        Name of the logger to retrieve.
    log_file:
        Optional path to the log file.  Defaults to
        ``DEFAULT_LOG_FILE`` when omitted.
    mongo_uri:
        Optional MongoDB connection URI.  When provided and
        :mod:`pymongo` is installed, log records are inserted into the
        specified collection in ``mongo_db``.
    mongo_db, mongo_collection:
        Database and collection names used when ``mongo_uri`` is supplied.
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    path = Path(log_file) if log_file else DEFAULT_LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if mongo_uri and MongoClient is not None:
        try:
            client = MongoClient(mongo_uri)
            collection = client[mongo_db][mongo_collection]

            class MongoHandler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    try:
                        collection.insert_one(
                            {
                                "name": record.name,
                                "level": record.levelname,
                                "message": record.getMessage(),
                                "created": record.created,
                            }
                        )
                    except Exception:
                        pass  # pragma: no cover - best effort

            logger.addHandler(MongoHandler())
        except Exception:  # pragma: no cover - failing to log to Mongo shouldn't abort
            pass

    return logger


def get_log_file() -> Path:
    """Return the path of the configured log file."""

    return DEFAULT_LOG_FILE
