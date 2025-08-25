from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard import trait_filter_tab


def test_get_sample_names_dict(monkeypatch):
    docs = [{"name": "Alpha"}, {"name": "Beta"}]
    def fake_query(query, fields=None):
        regex = query.get("name", {}).get("$regex", "")
        prefix = regex.lstrip("^")
        return [d for d in docs if d["name"].startswith(prefix)]

    monkeypatch.setattr(trait_filter_tab, "query_samples", fake_query)
    assert trait_filter_tab.get_sample_names("A") == ["Alpha"]


def test_filter_samples_dict(monkeypatch):
    sample_doc = {
        "name": "S1",
        "chemistry": "NMC",
        "manufacturer": "M",
        "cycle_count": 40,
        "avg_final_capacity": 1.2,
        "median_internal_resistance": 0.05,
        "avg_coulombic_eff": 0.98,
        "date": "2024-01-01",
        "tags": ["A"],
    }
    monkeypatch.setattr(trait_filter_tab, "query_samples", lambda q: [sample_doc])
    rows = trait_filter_tab.filter_samples("NMC", "M", sample="S1")
    assert rows[0]["name"] == "S1"
    assert rows[0]["sample_obj"].name == "S1"
