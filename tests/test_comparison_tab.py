import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard import comparison_tab  # noqa: E402


def test_get_sample_options_db_unavailable(
    monkeypatch: Any, caplog: Any
) -> None:  # noqa: E501
    monkeypatch.setattr(comparison_tab, "db_connected", lambda: False)
    monkeypatch.setattr(comparison_tab, "get_db_error", lambda: "no db")
    with caplog.at_level("ERROR", logger=comparison_tab.logger.name):
        opts, error = comparison_tab._get_sample_options()
    assert opts == [{"label": "Sample_001", "value": "sample1"}]
    assert error == "Database not connected; using demo data (no db)"
    assert "Database not connected: no db; using demo data" in caplog.text


def test_get_sample_options_db_available(
    monkeypatch: Any, caplog: Any
) -> None:  # noqa: E501
    monkeypatch.setattr(comparison_tab, "db_connected", lambda: True)
    samples = [
        {"_id": "id1", "name": "SampleA"},
        {"_id": "id2", "name": "SampleB"},
    ]
    monkeypatch.setattr(comparison_tab, "find_samples", lambda: samples)
    with caplog.at_level("WARNING", logger=comparison_tab.logger.name):
        opts, error = comparison_tab._get_sample_options()
    assert opts == [
        {"label": "SampleA", "value": "id1"},
        {"label": "SampleB", "value": "id2"},
    ]
    assert error is None
    assert caplog.text == ""
