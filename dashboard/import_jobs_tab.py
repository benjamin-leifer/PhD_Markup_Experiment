"""Dash tab displaying import job history and rollback actions."""

from __future__ import annotations

from typing import List, Dict

import dash
from dash import html, Input, Output, ALL, ctx
import dash_bootstrap_components as dbc

from . import layout as layout_components

try:  # pragma: no cover - database may not be available
    from battery_analysis import models
    from battery_analysis.utils import import_directory
except Exception:  # pragma: no cover - fallback when package missing
    models = None  # type: ignore
    import_directory = None  # type: ignore

TABLE_CONTAINER = "import-jobs-table-container"
STATUS_MESSAGE = "import-jobs-status"


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
                    "start_time": job.start_time.strftime("%Y-%m-%d %H:%M") if job.start_time else "",
                    "end_time": job.end_time.strftime("%Y-%m-%d %H:%M") if job.end_time else "",
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
    return html.Div(
        [html.H4("Import Jobs"), html.Div(table, id=TABLE_CONTAINER), html.Div(id=STATUS_MESSAGE)]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for the Import Jobs tab."""

    @app.callback(
        Output(TABLE_CONTAINER, "children"),
        Output(STATUS_MESSAGE, "children"),
        Input({"type": "rollback-job", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _rollback_job(_: List[int]) -> tuple[dbc.Table, str]:  # pragma: no cover - callback
        trigger = ctx.triggered_id
        if not trigger:
            raise dash.exceptions.PreventUpdate
        job_id = trigger.get("index")  # type: ignore[assignment]
        if import_directory and job_id:
            import_directory.rollback_job(job_id)
        refreshed = layout_components.import_jobs_table(_get_jobs())
        message = f"Rolled back job {job_id}" if job_id else ""
        return refreshed, message
