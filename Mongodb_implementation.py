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

import logging
import os
from typing import Any, Dict, List, Optional

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pymongo.uri_parser import parse_uri

# mypy: ignore-errors


def get_client() -> MongoClient:
    """Return a :class:`~pymongo.mongo_client.MongoClient` instance.

    The connection parameters are pulled from ``MONGO_URI`` or the
    ``MONGO_HOST``/``MONGO_PORT`` environment variables. Defaults are
    provided so the database can be accessed in development without any
    configuration. The resulting client stores the parameters used so
    callers can reuse them without establishing a connection.
    """

    logger = logging.getLogger(__name__)

    use_mock_env = os.getenv("USE_MONGO_MOCK")
    uri = os.getenv("MONGO_URI")
    host = os.getenv("MONGO_HOST", "localhost")
    port = int(os.getenv("MONGO_PORT", "27017"))

    if uri:
        try:
            parsed = parse_uri(uri)
            host, port = parsed["nodelist"][0]
        except Exception:
            # Leave host and port from environment if parsing fails
            pass

    logger.info(
        "Mongo params - USE_MONGO_MOCK=%s uri=%s host=%s port=%s",
        bool(use_mock_env),
        uri,
        host,
        port,
    )

    if use_mock_env:
        import mongomock

        client = mongomock.MongoClient()
        client._configured_host = host
        client._configured_port = port
        if uri:
            client._configured_uri = uri
        logger.info(
            "USE_MONGO_MOCK set. Created mongomock client for host=%s port=%s",
            host,
            port,
        )
        logger.info(
            "Mongo client configured: uri=%s host=%s port=%s",
            getattr(client, "_configured_uri", None),
            client._configured_host,
            client._configured_port,
        )
        return client

    try:
        if uri:
            client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        else:
            client = MongoClient(host, port, serverSelectionTimeoutMS=2000)

        logger.info("Pinging MongoDB at %s:%s", host, port)
        client.admin.command("ping")
        logger.info("MongoDB ping succeeded")
    except PyMongoError as exc:
        logger.info(
            "Failed to connect to MongoDB with uri=%s host=%s port=%s: %s. "
            "Falling back to mongomock.",
            uri,
            host,
            port,
            exc,
        )
        import mongomock

        client = mongomock.MongoClient()
        client._configured_host = host
        client._configured_port = port
        if uri:
            client._configured_uri = uri
        logger.info("mongomock client created")
        logger.info(
            "Mongo client configured: uri=%s host=%s port=%s",
            getattr(client, "_configured_uri", None),
            client._configured_host,
            client._configured_port,
        )
        return client

    if uri:
        client._configured_uri = uri
    client._configured_host = host
    client._configured_port = port
    logger.info(
        "Mongo client configured: uri=%s host=%s port=%s",
        getattr(client, "_configured_uri", None),
        client._configured_host,
        client._configured_port,
    )
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
