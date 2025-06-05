import os
import sys
import json

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from dashboard import export_handler


def sample_rows():
    return [
        {
            "name": "S1",
            "chemistry": "NMC",
            "manufacturer": "ABC",
            "capacity": 1.0,
            "resistance": 0.05,
            "ce": 0.95,
            "date": "2024-01-01",
        },
        {
            "name": "S2",
            "chemistry": "LFP",
            "manufacturer": "XYZ",
            "capacity": 2.0,
            "resistance": 0.06,
            "ce": 0.96,
            "date": "2024-01-02",
        },
    ]


def test_export_csv():
    csv_str = export_handler.export_filtered_results(sample_rows(), format="csv")
    assert "S1" in csv_str
    assert "chemistry" in csv_str


def test_export_excel_bytes():
    data = export_handler.export_filtered_results(sample_rows(), format="excel")
    assert isinstance(data, bytes)
    assert data[:2] == b"PK"


def test_export_json():
    json_str = export_handler.export_filtered_results(sample_rows(), format="json")
    data = json.loads(json_str)
    assert data[0]["test_id"] == "S1"
