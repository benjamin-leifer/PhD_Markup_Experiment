import os
import pickle
import types

import gridfs
import pytest
from battery_analysis.utils import detailed_data_manager as dm
from mongomock.gridfs import enable_gridfs_integration

from Mongodb_implementation import get_client


def test_get_detailed_cycle_data_raw_mongo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ensure mongomock with gridfs
    os.environ["USE_MONGO_MOCK"] = "1"
    enable_gridfs_integration()
    client = get_client()
    db = client["battery_test_db"]
    fs = gridfs.GridFS(db)
    tr_coll = db["test_results"]
    cd_coll = db["cycle_detail_data"]

    test_id = tr_coll.insert_one(
        {
            "cycles": [
                {
                    "cycle_index": 1,
                    "charge_capacity": 1.0,
                    "discharge_capacity": 1.0,
                },
                {
                    "cycle_index": 2,
                    "charge_capacity": 1.0,
                    "discharge_capacity": 0.0,
                },
            ]
        }
    ).inserted_id

    # insert cycle detail documents
    def _store(idx: int) -> None:
        ch = fs.put(
            pickle.dumps({"v": [1, 2]}),
            filename=f"c{idx}.pkl",
            content_type="application/python-pickle",
        )
        dis = fs.put(
            pickle.dumps({"v": [2, 1]}),
            filename=f"d{idx}.pkl",
            content_type="application/python-pickle",
        )
        cd_coll.insert_one(
            {
                "test_result": test_id,
                "cycle_index": idx,
                "charge_data": ch,
                "discharge_data": dis,
            }
        )

    _store(1)
    _store(2)

    # remove mongoengine attributes
    monkeypatch.setattr(dm, "TestResult", types.SimpleNamespace())
    monkeypatch.setattr(dm, "CycleDetailData", types.SimpleNamespace())

    data = dm.get_detailed_cycle_data(str(test_id))
    assert list(data.keys()) == [1]
    one = dm.get_detailed_cycle_data(str(test_id), 1)
    assert one[1]["charge"]["v"] == [1, 2]
