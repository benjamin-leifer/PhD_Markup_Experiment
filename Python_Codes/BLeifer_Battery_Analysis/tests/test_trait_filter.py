import sys
import os

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from dashboard.trait_filter_tab import build_query


def test_build_query_any():
    q = build_query(
        chemistry=None,
        manufacturer=None,
        additives=["FEC", "VC"],
        additive_mode="any",
        tags=None,
        tag_mode="any",
        cycle_min=None,
        cycle_max=None,
        thick_min=None,
        thick_max=None,
    )
    assert q == {"additives": {"$in": ["FEC", "VC"]}}


def test_build_query_numeric_and_all():
    q = build_query(
        chemistry="NMC",
        manufacturer=None,
        additives=["FEC", "VC"],
        additive_mode="all",
        tags=["fast"],
        tag_mode="exclude",
        cycle_min=10,
        cycle_max=100,
        thick_min=None,
        thick_max=1.0,
    )
    assert "$and" in q
    assert {"chemistry": "NMC"} in q["$and"]
    assert {"additives": {"$all": ["FEC", "VC"]}} in q["$and"]
    assert {"tags": {"$nin": ["fast"]}} in q["$and"]
    assert {"cycle_count": {"$gt": 10, "$lt": 100}} in q["$and"]
    assert {"thickness": {"$lt": 1.0}} in q["$and"]
