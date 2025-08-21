"""Dashboard tab that streams recent log entries."""

from __future__ import annotations

from pathlib import Path
from dash import dcc, html, Input, Output, callback

from battery_analysis.utils.logging import get_log_file

LOG_FILE = get_log_file()


def _tail(path: Path, lines: int = 50) -> str:
    """Return the last ``lines`` lines from ``path``.

    Failures simply return an empty string so the component remains functional
    even when the log file does not exist.
    """

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = fh.readlines()
        return "".join(data[-lines:])
    except Exception:
        return ""


def layout() -> html.Div:
    """Return the layout for the logs tab."""

    return html.Div(
        [
            dcc.Interval(id="logs-interval", interval=2000, n_intervals=0),
            html.Pre(id="logs-output", style={"height": "400px", "overflowY": "scroll"}),
        ]
    )


@callback(Output("logs-output", "children"), Input("logs-interval", "n_intervals"))
def update_logs(_: int) -> str:
    """Callback to refresh displayed logs."""

    return _tail(LOG_FILE)
