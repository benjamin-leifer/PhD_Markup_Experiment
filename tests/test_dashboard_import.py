"""Dashboard import directory tests using dash.testing."""

from __future__ import annotations

from unittest.mock import Mock

import pytest


def test_import_directory_starts_job_and_redirects(monkeypatch, tmp_path, dash_duo):
    """Entering a path and clicking start triggers import and redirects."""
    create_app = pytest.importorskip("dashboard.app").create_app
    app = create_app()

    # Mock heavy import routine
    mock_import = Mock(return_value="job-id")
    monkeypatch.setattr(
        "battery_analysis.utils.import_directory.import_directory",
        mock_import,
    )

    dash_duo.start_server(app)
    dash_duo.wait_for_page(dash_duo.server_url + "/data-import")

    # Simulate entering directory and starting import
    dash_duo.find_element("#import-dir-path").send_keys(str(tmp_path))
    dash_duo.find_element("#import-dir-start").click()

    dash_duo.wait_for_page(dash_duo.server_url + "/import-jobs")
    assert dash_duo.driver.current_url.endswith("/import-jobs")

    mock_import.assert_called_once_with(
        str(tmp_path),
        include=["*.csv", "*.xlsx", "*.xls", "*.mpt"],
    )
