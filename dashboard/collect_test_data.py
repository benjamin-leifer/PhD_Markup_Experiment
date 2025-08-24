"""Utility to export a small MongoDB dataset for offline debugging.

This script connects to the configured MongoDB instance and exports
approximately ten ``TestResult`` records along with their related
``Sample`` and ``CycleDetailData`` documents. Any GridFS files referenced
by the ``CycleDetailData`` documents are written to the
``dashboard/test_data`` directory. A JSON manifest describing the
exported objects is also written to that directory so the dashboard can
load the data when MongoDB is unavailable.

The script relies only on ``pymongo`` and ``gridfs`` so it can run in
minimal environments. The MongoDB connection itself is handled by
``Mongodb_implementation.get_client`` which pulls settings from the
``MONGO_URI`` (or ``MONGO_HOST``/``MONGO_PORT``) environment variables.
The database name is taken from the path portion of ``MONGO_URI`` when
available and otherwise defaults to ``battery_test_db``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import gridfs

from Mongodb_implementation import get_client

from battery_analysis.utils.logging import get_logger

logger = get_logger(__name__)


def export_dataset(limit: int = 10) -> Path:
    """Export ``limit`` test results and return path to manifest file."""
    uri = os.getenv("MONGO_URI", "")
    db_name = os.getenv("BATTERY_DB_NAME")
    if not db_name and uri:
        db_name = urlparse(uri).path.lstrip("/") or None
    db_name = db_name or "battery_test_db"

    client = get_client()
    db = client[db_name]
    fs = gridfs.GridFS(db)

    out_dir = Path(__file__).resolve().parent / "test_data"
    out_dir.mkdir(exist_ok=True)

    manifest: Dict[str, Any] = {"tests": [], "upcoming": [], "summary": {}}

    for test in db["test_results"].find().limit(limit):
        sample = db["samples"].find_one({"_id": test.get("sample")})
        cycles = list(db["cycle_detail_data"].find({"test_result": test["_id"]}))
        cycle_info = []
        for cycle in cycles:
            c: Dict[str, str | int] = {"cycle_index": cycle.get("cycle_index", 0)}
            for field in ("charge_data", "discharge_data"):
                file_id = cycle.get(field)
                if file_id:
                    data = fs.get(file_id).read()
                    filename = f"{file_id}.bin"
                    with open(out_dir / filename, "wb") as fh:
                        fh.write(data)
                    c[field] = filename
            cycle_info.append(c)
        manifest["tests"].append(
            {
                "cell_id": sample.get("name") if sample else "Unknown",
                "chemistry": sample.get("chemistry") if sample else "",
                "test_type": test.get("test_type", ""),
                "current_cycle": test.get("cycle_count", 0),
                "last_timestamp": test.get("date", datetime.utcnow()).isoformat(),
                "test_schedule": test.get("name", ""),
                "status": "running",
                "cycles": cycle_info,
            }
        )

    manifest_path = out_dir / "offline_dump.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest_path


def main() -> None:
    path = export_dataset()
    logger.info("Export complete: %s", path)


if __name__ == "__main__":  # pragma: no cover - manual utility
    main()
