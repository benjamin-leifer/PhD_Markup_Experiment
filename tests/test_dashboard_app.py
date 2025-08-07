"""Integration tests for the dashboard Dash app."""

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

TAB_LABELS = [
    "Overview",
    "New Material",
    "Data Import",
    "Export",
    "Comparison",
    "Advanced Analysis",
    "Cycle Detail",
    "EIS",
    "Document Status",
    "Missing Data",
    "Trait Filter",
    "Flags",
]


def test_all_tabs_render():
    """The app layout renders and includes all tab labels."""
    app = create_app()
    with ThreadedRunner() as runner:
        runner.start(app)
        response = requests.get(runner.url)
        for label in TAB_LABELS:
            assert label in response.text


def test_basic_callbacks():
    """Each tab's primary callback executes without error."""
    app = create_app()
    cb = app.callback_map

    toggle = cb["export-modal.is_open"]["callback"].__wrapped__
    assert toggle(1, None, False) is True

    aa_key = next(k for k in cb if "dqdv-options.style" in k)
    toggle_opts = cb[aa_key]["callback"].__wrapped__
    styles = toggle_opts("dqdv")
    assert styles[0] == {"display": "block"}

    update_tests = cb["cd-test.options"]["callback"].__wrapped__
    assert update_tests("sample1")

    update_graph = cb["compare-graph.figure"]["callback"].__wrapped__
    fig = update_graph(None, "capacity")
    assert fig.data == ()

    eis_key = next(k for k in cb if "eis-file-div.style" in k)
    toggle_source = cb[eis_key]["callback"].__wrapped__
    file_style, db_style = toggle_source("file")
    assert file_style["display"] == "block" and db_style["display"] == "none"

    render_alerts = cb["missing-data-alerts.children"]["callback"].__wrapped__
    alerts = render_alerts([{"test_id": "A", "missing": ["cathode"]}])
    assert alerts

    update_results = cb["trait-results.children"]["callback"].__wrapped__
    result = update_results(1, None, None, None, None, None, None, None, None, None, [])
    assert result is not None
