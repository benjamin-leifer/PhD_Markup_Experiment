from pathlib import Path
import sys
import json
import shutil
from datetime import datetime
from typing import Any

import mongomock
from mongomock.gridfs import enable_gridfs_integration
import gridfs
from bson import ObjectId

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_export_dataset_creates_manifest_and_cycle_files(
    monkeypatch: Any, tmp_path: Path
) -> None:
    enable_gridfs_integration()
    client = mongomock.MongoClient()
    db = client["battery_test_db"]
    fs = gridfs.GridFS(db)
    charge_id = fs.put(b"charge")
    discharge_id = fs.put(b"discharge")

    sample_id = (
        db["samples"]
        .insert_one(
            {
                "_id": ObjectId(),
                "name": "CellA",
                "chemistry": "Li-ion",
            }
        )
        .inserted_id
    )

    test_id = (
        db["test_results"]
        .insert_one(
            {
                "_id": ObjectId(),
                "sample": sample_id,
                "test_type": "cycle",
                "cycle_count": 1,
                "name": "Test1",
                "date": datetime(2024, 1, 1),
            }
        )
        .inserted_id
    )

    db["cycle_detail_data"].insert_one(
        {
            "_id": ObjectId(),
            "test_result": test_id,
            "cycle_index": 1,
            "charge_data": charge_id,
            "discharge_data": discharge_id,
        }
    )

    import Mongodb_implementation

    monkeypatch.setattr(Mongodb_implementation, "get_client", lambda: client)

    import dashboard.collect_test_data as ctd

    fake_path = tmp_path / "pkg" / "collect_test_data.py"
    fake_path.parent.mkdir()
    fake_path.write_text("")
    monkeypatch.setattr(ctd, "__file__", str(fake_path))

    manifest_path = ctd.export_dataset(limit=1)
    out_dir = manifest_path.parent
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        assert set(manifest) >= {"tests", "upcoming", "summary"}
        assert len(manifest["tests"]) == 1
        test_info = manifest["tests"][0]
        for field in (
            "cell_id",
            "chemistry",
            "test_type",
            "current_cycle",
            "last_timestamp",
            "test_schedule",
            "status",
            "cycles",
        ):
            assert field in test_info
        cycle = test_info["cycles"][0]
        for file_field in ("charge_data", "discharge_data"):
            file_path = out_dir / cycle[file_field]
            assert file_path.is_file()
    finally:
        shutil.rmtree(out_dir)
    assert not out_dir.exists()
