from __future__ import annotations

"""Dash tab for managing directory import watchers."""

from typing import List, Dict, Any

import dash
from dash import html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

from . import layout as layout_components
from . import preferences

try:  # pragma: no cover - battery_analysis may be optional in tests
    from battery_analysis.utils import import_watcher
except Exception:  # pragma: no cover - allow running without package
    import_watcher = None  # type: ignore

TABLE_CONTAINER = "watcher-table-container"
STATUS_MESSAGE = "watcher-status"
ADD_WATCHER = "add-watcher"


def _format_uptime(seconds: float) -> str:
    mins, sec = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f"{hrs}h {mins}m {sec}s"
    return f"{mins}m {sec}s"


def _gather_watchers() -> List[Dict[str, Any]]:
    prefs = preferences.load_preferences()
    dirs = prefs.get("watcher_dirs", [])
    running = {}
    if import_watcher:
        for info in import_watcher.list_watchers():
            running[info["directory"]] = info
    watchers: List[Dict[str, Any]] = []
    for idx, path in enumerate(dirs):
        info = running.get(path)
        watchers.append(
            {
                "index": idx,
                "path": path,
                "running": bool(info),
                "uptime": _format_uptime(info["uptime"]) if info else "",
            }
        )
    return watchers


def layout() -> html.Div:
    """Layout for the Watchers tab."""
    watchers = _gather_watchers()
    return html.Div(
        [
            html.H4("Directory Watchers"),
            dbc.Button("Add Watcher", id=ADD_WATCHER, color="secondary", size="sm", className="mb-2"),
            html.Div(layout_components.watcher_table(watchers), id=TABLE_CONTAINER),
            html.Div(id=STATUS_MESSAGE, className="mt-2"),
        ]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register Dash callbacks for the Watchers tab."""

    @app.callback(
        Output(TABLE_CONTAINER, "children"),
        Input(ADD_WATCHER, "n_clicks"),
        prevent_initial_call=True,
    )
    def _add_row(_: int) -> dbc.Table:  # pragma: no cover - callback
        prefs = preferences.load_preferences()
        prefs.setdefault("watcher_dirs", []).append("")
        preferences.save_preferences(prefs)
        return layout_components.watcher_table(_gather_watchers())

    @app.callback(
        Output(TABLE_CONTAINER, "children", allow_duplicate=True),
        Input({"type": "watcher-path", "index": dash.ALL}, "value"),
        prevent_initial_call=True,
    )
    def _update_paths(values: List[str]) -> dbc.Table:  # pragma: no cover - callback
        prefs = preferences.load_preferences()
        prefs["watcher_dirs"] = [v for v in values if v]
        preferences.save_preferences(prefs)
        return layout_components.watcher_table(_gather_watchers())

    @app.callback(
        Output(TABLE_CONTAINER, "children", allow_duplicate=True),
        Output(STATUS_MESSAGE, "children"),
        Input({"type": "watcher-toggle", "index": dash.ALL}, "n_clicks"),
        State({"type": "watcher-path", "index": dash.ALL}, "value"),
        prevent_initial_call=True,
    )
    def _toggle_watcher(btns: List[int], paths: List[str]) -> tuple[dbc.Table, str]:  # pragma: no cover - callback
        trigger = ctx.triggered_id
        if trigger is None:
            raise dash.exceptions.PreventUpdate
        idx = trigger["index"]
        path = paths[idx]
        if not path or import_watcher is None:
            return dash.no_update, "Invalid path"
        if import_watcher.is_watching(path):
            import_watcher.stop_watcher(path)
            msg = f"Stopped watcher for {path}"
        else:
            import_watcher.start_watcher(path)
            msg = f"Started watcher for {path}"
        return layout_components.watcher_table(_gather_watchers()), msg
