"""Utility helpers for working with MongoDB.

This module centralizes MongoDB access so scripts throughout the project
can share the same connection logic. Connection details are pulled from
environment variables to avoid hard-coding credentials in source files.

Environment variables:

``MONGO_URI``
    Full MongoDB connection string. If provided this takes precedence over
    the host/port variables.

``MONGO_HOST``
    Hostname of the MongoDB server. Defaults to ``"localhost"``.

``MONGO_PORT``
    Port number of the MongoDB server. Defaults to ``27017``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from pymongo import MongoClient
import os


def get_client() -> MongoClient:
    """Return a :class:`~pymongo.mongo_client.MongoClient` instance.

    The connection parameters are pulled from ``MONGO_URI`` or the
    ``MONGO_HOST``/``MONGO_PORT`` environment variables. Defaults are
    provided so the database can be accessed in development without any
    configuration. The resulting client stores the parameters used so
    callers can reuse them without establishing a connection.
    """

    if os.getenv("USE_MONGO_MOCK"):
        import mongomock

        host = os.getenv("MONGO_HOST", "localhost")
        port = int(os.getenv("MONGO_PORT", "27017"))
        client = mongomock.MongoClient()
        client._configured_host = host  # type: ignore[attr-defined]
        client._configured_port = port  # type: ignore[attr-defined]
        return client

    uri = os.getenv("MONGO_URI")
    if uri:
        client = MongoClient(uri)
        # Stash the URI so other modules can reuse it without inspecting the client
        client._configured_uri = uri  # type: ignore[attr-defined]
        return client

    host = os.getenv("MONGO_HOST", "localhost")
    port = int(os.getenv("MONGO_PORT", "27017"))
    client = MongoClient(host, port)
    client._configured_host = host  # type: ignore[attr-defined]
    client._configured_port = port  # type: ignore[attr-defined]
    return client


def _get_collection(
    db_name: str, collection_name: str, client: Optional[MongoClient] = None
):
    client = client or get_client()
    return client[db_name][collection_name]


def insert_sample(
    sample: Dict[str, Any],
    *,
    db_name: str = "battery_test_db",
    collection_name: str = "samples",
    client: Optional[MongoClient] = None,
) -> str:
    """Insert a sample document and return its ID."""

    coll = _get_collection(db_name, collection_name, client)
    result = coll.insert_one(sample)
    return str(result.inserted_id)


def insert_test_result(
    result: Dict[str, Any],
    *,
    db_name: str = "battery_test_db",
    collection_name: str = "test_results",
    client: Optional[MongoClient] = None,
) -> str:
    """Insert a test result document and return its ID."""

    coll = _get_collection(db_name, collection_name, client)
    result = coll.insert_one(result)
    return str(result.inserted_id)


def find_samples(
    query: Optional[Dict[str, Any]] = None,
    *,
    db_name: str = "battery_test_db",
    collection_name: str = "samples",
    client: Optional[MongoClient] = None,
) -> List[Dict[str, Any]]:
    """Return a list of sample documents matching ``query``."""

    coll = _get_collection(db_name, collection_name, client)
    return list(coll.find(query or {}))


def find_test_results(
    query: Optional[Dict[str, Any]] = None,
    *,
    db_name: str = "battery_test_db",
    collection_name: str = "test_results",
    client: Optional[MongoClient] = None,
) -> List[Dict[str, Any]]:
    """Return a list of test result documents matching ``query``."""

    coll = _get_collection(db_name, collection_name, client)
    return list(coll.find(query or {}))


__all__ = [
    "get_client",
    "insert_sample",
    "insert_test_result",
    "find_samples",
    "find_test_results",
]

