"""Dash tab for monitoring refactor jobs and controlling active refactors."""

from __future__ import annotations

from pathlib import Path
from threading import Thread
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, ctx, dcc, html

try:
    from . import auth
    from . import layout as layout_components
except ImportError:  # running as a script
    import auth
    import layout as layout_components

try:  # pragma: no cover - optional dependency
    from battery_analysis.models import RefactorJob
except Exception:  # pragma: no cover - allow running without models
    RefactorJob = None

try:  # pragma: no cover - optional dependency
    from battery_analysis.utils import refactor_data
except Exception:  # pragma: no cover - allow running without utility
    refactor_data = None


CONTROL_FILE = Path(__file__).resolve().parent.parent / ".refactor_control"
TABLE_CONTAINER = "refactor-jobs-table-container"
STATUS_MESSAGE = "refactor-jobs-status"
REFRESH_INTERVAL = "refactor-jobs-refresh"
PAUSE_BTN = "refactor-jobs-pause"
RESUME_BTN = "refactor-jobs-resume"
CANCEL_BTN = "refactor-jobs-cancel"
START_BTN = "refactor-jobs-start"
JOB_STORE = "refactor-jobs-job"


def _load_jobs() -> List[Dict[str, str]]:
    """Return RefactorJob records formatted for the table."""

    jobs: List[Dict[str, str]] = []
    if not RefactorJob:
        return jobs
    try:  # pragma: no cover - requires database
        for job in RefactorJob.objects.order_by("-start_time")[:20]:
            start = getattr(job, "start_time", None)
            end = getattr(job, "end_time", None)
            processed = getattr(job, "processed_count", 0) or 0
            total = getattr(job, "total_count", 0) or 0
            errors = getattr(job, "errors", []) or []
            end_formatter = getattr(end, "strftime", lambda _fmt: "")
            end_str = end_formatter("%Y-%m-%d %H:%M")
            jobs.append(
                {
                    "id": str(getattr(job, "id", "")),
                    "start": getattr(start, "strftime", lambda _fmt: "")(
                        "%Y-%m-%d %H:%M"
                    ),
                    "end": end_str,
                    "progress": f"{processed}/{total}",
                    "errors": str(len(errors)),
                }
            )
    except Exception:
        pass
    return jobs


def layout() -> html.Div:
    """Layout for the Refactor Jobs tab."""

    table = layout_components.refactor_jobs_table(_load_jobs())
    buttons = dbc.ButtonGroup(
        [
            dbc.Button("Start", id=START_BTN, color="primary", size="sm"),
            dbc.Button("Pause", id=PAUSE_BTN, color="warning", size="sm"),
            dbc.Button("Resume", id=RESUME_BTN, color="success", size="sm"),
            dbc.Button("Cancel", id=CANCEL_BTN, color="danger", size="sm"),
        ],
        className="mb-2",
    )
    return html.Div(
        [
            html.H4("Refactor Jobs"),
            buttons,
            html.Div(id=STATUS_MESSAGE, className="mb-2"),
            html.Div(table, id=TABLE_CONTAINER),
            dcc.Interval(id=REFRESH_INTERVAL, interval=2000),
            dcc.Store(id=JOB_STORE),
        ]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for the Refactor Jobs tab."""

    @app.callback(  # type: ignore[misc]
        Output(TABLE_CONTAINER, "children"),
        Input(REFRESH_INTERVAL, "n_intervals"),
        Input(JOB_STORE, "data"),
    )
    def _refresh(_: int, __: str | None) -> dbc.Table:  # pragma: no cover - callback
        return layout_components.refactor_jobs_table(_load_jobs())

    @app.callback(  # type: ignore[misc]
        Output(STATUS_MESSAGE, "children", allow_duplicate=True),
        Output(JOB_STORE, "data"),
        Input(START_BTN, "n_clicks"),
        State("user-role", "data"),
        prevent_initial_call=True,
    )
    def _start(n_clicks: int, role: str | None) -> tuple[html.Component, object]:
        """Start a new refactor job in the background."""

        if not auth.can_manage_refactor_jobs(role or ""):
            raise dash.exceptions.PreventUpdate
        if not RefactorJob or not refactor_data:
            return (
                dbc.Alert("Refactor not supported", color="danger", dismissable=True),
                dash.no_update,
            )
        try:
            job = RefactorJob().save()
            Thread(
                target=refactor_data.refactor_tests,
                kwargs={"job_id": str(getattr(job, "id", ""))},
                daemon=True,
            ).start()
            return (
                dbc.Alert(
                    f"Started job {getattr(job, 'id', '')}",
                    color="success",
                    dismissable=True,
                ),
                str(getattr(job, "id", "")),
            )
        except Exception as exc:  # pragma: no cover - best effort
            return dbc.Alert(str(exc), color="danger", dismissable=True), dash.no_update

    @app.callback(  # type: ignore[misc]
        Output(STATUS_MESSAGE, "children"),
        Input(PAUSE_BTN, "n_clicks"),
        Input(RESUME_BTN, "n_clicks"),
        Input(CANCEL_BTN, "n_clicks"),
        State("user-role", "data"),
        prevent_initial_call=True,
    )
    def _control(
        pause: int | None,
        resume: int | None,
        cancel: int | None,
        role: str | None,
    ) -> html.Component:  # pragma: no cover - callback
        if not auth.can_manage_refactor_jobs(role or ""):
            raise dash.exceptions.PreventUpdate
        trigger = ctx.triggered_id
        if trigger is None:
            raise dash.exceptions.PreventUpdate
        cmd_map = {
            PAUSE_BTN: "pause",
            RESUME_BTN: "resume",
            CANCEL_BTN: "cancel",
        }
        cmd = cmd_map.get(trigger)
        if cmd is None:
            raise dash.exceptions.PreventUpdate
        try:
            CONTROL_FILE.write_text(cmd, encoding="utf-8")
            return dbc.Alert(
                f"Sent {cmd} command",
                color="info",
                dismissable=True,
            )
        except Exception as exc:  # pragma: no cover - file writing error
            return dbc.Alert(str(exc), color="danger", dismissable=True)


__all__ = ["layout", "register_callbacks"]
