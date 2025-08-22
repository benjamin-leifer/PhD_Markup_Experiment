"""Dash tab displaying import job history and rollback actions."""

from __future__ import annotations

from typing import List, Dict
from pathlib import Path

import dash
from dash import html, Input, Output, ALL, ctx, dcc
import dash_bootstrap_components as dbc
import requests

from . import layout as layout_components

try:  # pragma: no cover - optional dependencies
    from battery_analysis.utils import import_directory
    from battery_analysis.utils.config import load_config
except Exception:  # pragma: no cover - fallback when package missing
    import_directory = None  # type: ignore

    def load_config() -> dict:  # type: ignore
        return {}

cfg = load_config()
API_URL = cfg.get("api_url", "http://localhost:8000")

TABLE_CONTAINER = "import-jobs-table-container"
STATUS_MESSAGE = "import-jobs-status"
PROGRESS_INTERVAL = "import-jobs-progress"
CONTROL_FILE = Path(__file__).resolve().parents[1] / ".import_control"


def _get_jobs() -> List[Dict[str, str]]:
    """Return import job information suitable for the table."""
    records: List[Dict[str, str]] = []
    try:  # pragma: no cover - requires API server
        resp = requests.get(f"{API_URL}/import-jobs", timeout=2)
        data = resp.json().get("jobs", [])
        for job in data:
            records.append(
                {
                    "id": job.get("id", ""),
                    "start_time": (job.get("start_time") or "")[:16],
                    "end_time": (job.get("end_time") or "")[:16],
                    "file_count": str(job.get("processed_count", "")),
                    "errors": "; ".join(job.get("errors", [])),
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
        html.Div(
            [
                dbc.Button(
                    "Pause Imports",
                    id="pause-imports",
                    color="warning",
                    className="me-2",
                ),
                dbc.Button(
                    "Cancel Imports",
                    id="cancel-imports",
                    color="danger",
                ),
            ],
            className="mb-2",
        ),
        html.Div(table, id=TABLE_CONTAINER),
        html.Div(id=STATUS_MESSAGE),
        dcc.Interval(id=PROGRESS_INTERVAL, interval=2000),
    ]
    return html.Div(components)


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for the Import Jobs tab."""

    @app.callback(
        Output(STATUS_MESSAGE, "children", allow_duplicate=True),
        Input("pause-imports", "n_clicks"),
        Input("cancel-imports", "n_clicks"),
        prevent_initial_call=True,
    )
    def _control_imports(_: int, __: int) -> str:  # pragma: no cover - callback
        trigger = ctx.triggered_id
        if not trigger:
            raise dash.exceptions.PreventUpdate
        cmd = "pause" if trigger == "pause-imports" else "cancel"
        try:
            CONTROL_FILE.write_text(cmd)
        except Exception:  # pragma: no cover - best effort
            return "Failed to write control command"
        return f"Sent {cmd} command"

    @app.callback(
        Output(TABLE_CONTAINER, "children"),
        Output(STATUS_MESSAGE, "children"),
        Input({"type": "rollback-job", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _rollback_job(
        _: List[int],
    ) -> tuple[dbc.Table, str]:  # pragma: no cover - callback
        trigger = ctx.triggered_id
        if not trigger:
            raise dash.exceptions.PreventUpdate
        job_id = trigger.get("index")  # type: ignore[assignment]
        if import_directory and job_id:
            import_directory.rollback_job(job_id)
        refreshed = layout_components.import_jobs_table(_get_jobs())
        message = f"Rolled back job {job_id}" if job_id else ""
        return refreshed, message

    @app.callback(
        Output(TABLE_CONTAINER, "children", allow_duplicate=True),
        Input(PROGRESS_INTERVAL, "n_intervals"),
        prevent_initial_call=True,
    )
    def _refresh_progress(_: int) -> dbc.Table:  # pragma: no cover - callback
        return layout_components.import_jobs_table(_get_jobs())

