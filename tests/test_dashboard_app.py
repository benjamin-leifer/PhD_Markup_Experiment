"""Basic integration tests for the dashboard Dash app."""

from pathlib import Path
import sys
import requests
from dash.testing.application_runners import ThreadedRunner

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dashboard.app import create_app  # noqa: E402


def test_layout_and_export_modal():
    """The app layout renders and export modal toggles via callback."""
    app = create_app()
    with ThreadedRunner() as runner:
        runner.start(app)
        response = requests.get(runner.url)
        assert "Battery Test Dashboard" in response.text

        toggle = app.callback_map["export-modal.is_open"]["callback"].__wrapped__
        assert toggle(1, None, False) is True
