"""Dash tab displaying import job summaries with filters."""

from __future__ import annotations

from typing import Dict, List

import dash
from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc
import requests

from . import layout as layout_components

try:  # pragma: no cover - optional dependency
    from battery_analysis.utils.config import load_config
except Exception:  # pragma: no cover - fallback when package missing
    def load_config() -> dict:  # type: ignore
        return {}

cfg = load_config()
API_URL = cfg.get("api_url", "http://localhost:8000")

TABLE_CONTAINER = "import-jobs-table-container"
STATUS_FILTER = "import-jobs-status-filter"
PROGRESS_INTERVAL = "import-jobs-progress"


def _get_jobs(status: str | None = None) -> List[Dict[str, str]]:
    """Return import job summaries for the table."""
    records: List[Dict[str, str]] = []
    try:  # pragma: no cover - requires API server
        resp = requests.get(f"{API_URL}/import-job-summaries", timeout=2)
        data = resp.json().get("jobs", [])
        for job in data:
            if status and job.get("status") != status:
                continue
            records.append(
                {
                    "id": job.get("id", ""),
                    "start_time": (job.get("start_time") or "")[:16],
                    "end_time": (job.get("end_time") or "")[:16],
                    "created": str(job.get("created", "")),
                    "updated": str(job.get("updated", "")),
                    "skipped": str(job.get("skipped", "")),
                    "errors": "; ".join(job.get("errors", [])),
                    "status": job.get("status", ""),
                }
            )
    except Exception:
        pass
    return records


def layout() -> html.Div:
    """Layout for the Import Jobs tab."""
    jobs = _get_jobs()
    table = layout_components.import_jobs_table(jobs)
    components = [
        html.H4("Import Jobs"),
        dcc.Dropdown(
            id=STATUS_FILTER,
            options=[
                {"label": "All", "value": ""},
                {"label": "Completed", "value": "completed"},
                {"label": "Failed", "value": "failed"},
                {"label": "Running", "value": "running"},
            ],
            value="",
            clearable=False,
            className="mb-2",
        ),
        html.Div(table, id=TABLE_CONTAINER),
        dcc.Interval(id=PROGRESS_INTERVAL, interval=2000),
    ]
    return html.Div(components)


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for the Import Jobs tab."""

    @app.callback(
        Output(TABLE_CONTAINER, "children"),
        Input(PROGRESS_INTERVAL, "n_intervals"),
        Input(STATUS_FILTER, "value"),
        prevent_initial_call=True,
    )
    def _refresh(_: int, status: str) -> dbc.Table:  # pragma: no cover - callback
        return layout_components.import_jobs_table(_get_jobs(status or None))

