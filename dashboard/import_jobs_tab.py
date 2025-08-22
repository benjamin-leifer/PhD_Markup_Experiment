"""Dash tab displaying import job history and rollback actions."""

from __future__ import annotations

from typing import List, Dict
from pathlib import Path

import dash
from dash import html, Input, Output, ALL, ctx, dcc
import dash_bootstrap_components as dbc

from . import layout as layout_components

try:  # pragma: no cover - database may not be available
    from battery_analysis import models
    from battery_analysis.utils import import_directory
    from battery_analysis.utils.config import load_config
    import redis  # type: ignore
except Exception:  # pragma: no cover - fallback when package missing
    models = None  # type: ignore
    import_directory = None  # type: ignore
    def load_config() -> dict:  # type: ignore
        return {}

    redis = None  # type: ignore

cfg = load_config()
_pubsub = None
if redis and cfg.get("redis_url"):
    try:
        _redis = redis.from_url(cfg["redis_url"])
        _pubsub = _redis.pubsub(ignore_subscribe_messages=True)
        _pubsub.subscribe(cfg.get("redis_channel", "import_progress"))
    except Exception:  # pragma: no cover - optional
        _pubsub = None

TABLE_CONTAINER = "import-jobs-table-container"
STATUS_MESSAGE = "import-jobs-status"
PROGRESS_INTERVAL = "import-jobs-progress"
CONTROL_FILE = Path(__file__).resolve().parents[1] / ".import_control"


def _get_jobs() -> List[Dict[str, str]]:
    """Return import job information suitable for the table."""
    records: List[Dict[str, str]] = []
    if not models:
        return records
    try:  # pragma: no cover - requires database
        jobs = models.ImportJob.objects.order_by("-start_time")  # type: ignore[attr-defined]
        for job in jobs:
            records.append(
                {
                    "id": str(job.id),
                    "start_time": (
                        job.start_time.strftime("%Y-%m-%d %H:%M")
                        if job.start_time
                        else ""
                    ),
                    "end_time": (
                        job.end_time.strftime("%Y-%m-%d %H:%M") if job.end_time else ""
                    ),
                    "file_count": str(len(getattr(job, "files", []))),
                    "errors": "; ".join(getattr(job, "errors", [])),
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
    ]
    if _pubsub:
        components.append(dcc.Interval(id=PROGRESS_INTERVAL, interval=2000))
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

    if _pubsub:

        @app.callback(
            Output(TABLE_CONTAINER, "children", allow_duplicate=True),
            Input(PROGRESS_INTERVAL, "n_intervals"),
            prevent_initial_call=True,
        )
        def _refresh_progress(_: int) -> dbc.Table:  # pragma: no cover - callback
            message = _pubsub.get_message()
            if not message:
                raise dash.exceptions.PreventUpdate
            return layout_components.import_jobs_table(_get_jobs())
