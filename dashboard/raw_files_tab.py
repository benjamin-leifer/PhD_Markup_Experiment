"""Dash tab for searching and downloading raw data files."""

from __future__ import annotations

from typing import Dict, List

import dash
from dash import Input, Output, State, dcc, html, dash_table
import dash_bootstrap_components as dbc

from . import auth

TABLE_ID = "raw-files-table"
SEARCH_INPUT = "raw-search-input"
SEARCH_BUTTON = "raw-search-btn"


def _search_files(query: str) -> List[Dict[str, str]]:
    """Return raw file records matching ``query``.

    The implementation attempts to query the MongoDB backend but falls back to
    example data when the database is unavailable so the interface remains
    functional in offline contexts.
    """

    try:  # pragma: no cover - requires database
        from battery_analysis.models import RawDataFile

        flt: Dict[str, object] = {}
        if query:
            flt = {
                "$or": [
                    {"_id": query},
                    {"filename": {"$regex": query, "$options": "i"}},
                ]
            }
        files: List[Dict[str, str]] = []
        for f in RawDataFile.objects(__raw__=flt).order_by("-upload_date"):
            sample = getattr(getattr(f, "test_result", None), "sample", None)
            sample_name = getattr(sample, "name", "")
            ts = getattr(f, "upload_date", None)
            ts_str = getattr(ts, "isoformat", lambda: "")()
            files.append(
                {"file_id": str(f.id), "sample": sample_name, "timestamp": ts_str}
            )
        return files
    except Exception:
        return [
            {"file_id": "F1", "sample": "Sample_A", "timestamp": "2024-01-01"},
            {"file_id": "F2", "sample": "Sample_B", "timestamp": "2024-01-02"},
        ]


def layout() -> html.Div:
    """Return the tab layout."""

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            id=SEARCH_INPUT,
                            placeholder="Search by TestResult ID or metadata",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Button("Search", id=SEARCH_BUTTON, color="primary"),
                        width="auto",
                    ),
                ],
                className="mb-3",
            ),
            dash_table.DataTable(
                id=TABLE_ID,
                columns=[
                    {"name": "File ID", "id": "file_id"},
                    {"name": "Sample", "id": "sample"},
                    {"name": "Timestamp", "id": "timestamp"},
                    {
                        "name": "Download",
                        "id": "download",
                        "presentation": "markdown",
                    },
                ],
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left"},
            ),
            dcc.Store(id="user-role"),
        ]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for the raw files tab."""

    @app.callback(
        Output(TABLE_ID, "data"),
        Input(SEARCH_BUTTON, "n_clicks"),
        State(SEARCH_INPUT, "value"),
        State("user-role", "data"),
        prevent_initial_call=True,
    )  # type: ignore[misc]
    def _update_table(
        n_clicks: int, query: str | None, role: str | None
    ) -> List[Dict[str, str]]:
        records = _search_files(query or "")
        can_dl = auth.can_download_raw(role or "")
        for r in records:
            r["download"] = f"[Download](/raw/{r['file_id']})" if can_dl else ""
        return records


__all__ = ["layout", "register_callbacks"]
