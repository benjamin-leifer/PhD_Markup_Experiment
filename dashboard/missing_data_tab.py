"""Dash tab highlighting tests missing component assignments."""

from __future__ import annotations

from typing import List, Dict

from dash import html, dcc, Input, Output, State, dash_table
import dash
import dash_bootstrap_components as dbc

DATA_STORE = "missing-data-store"
TABLE_ID = "missing-data-table"
MODAL_ID = "missing-data-modal"
SELECTED_TEST_STORE = "missing-data-selected-test"


def _get_missing_data() -> List[Dict[str, object]]:
    """Return tests with missing component references.

    Tries to query the MongoDB backend, otherwise returns example data so the
    interface remains functional without a database.
    """
    try:  # pragma: no cover - requires database
        from battery_analysis import models

        records: List[Dict[str, object]] = []
        for test in models.TestResult.objects():  # type: ignore[attr-defined]
            sample = test.sample.fetch() if hasattr(test.sample, "fetch") else test.sample
            missing = [
                f
                for f in ("anode", "cathode", "separator", "electrolyte")
                if getattr(sample, f, None) is None
            ]
            if missing:
                records.append({"test_id": str(test.id), "missing": missing})
        return records
    except Exception:
        return [
            {"test_id": "Test_A", "missing": ["cathode", "separator"]},
            {"test_id": "Test_B", "missing": ["electrolyte"]},
        ]


def layout() -> html.Div:
    """Return layout for the missing data tab."""
    return html.Div(
        [
            dcc.Store(id=DATA_STORE, data=_get_missing_data()),
            dcc.Store(id=SELECTED_TEST_STORE),
            dash_table.DataTable(
                id=TABLE_ID,
                columns=[
                    {"name": "Test ID", "id": "test_id"},
                    {"name": "Missing Components", "id": "missing"},
                    {"name": "Resolve", "id": "resolve", "presentation": "markdown"},
                ],
                data=[],
                style_cell={"textAlign": "left"},
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Resolve Missing Data")),
                    dbc.ModalBody(id=f"{MODAL_ID}-body"),
                    dbc.ModalFooter(
                        dbc.Button("Mark Resolved", id="confirm-resolve", color="primary")
                    ),
                ],
                id=MODAL_ID,
                is_open=False,
            ),
        ]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for resolving missing data alerts."""

    @app.callback(Output(TABLE_ID, "data"), Input(DATA_STORE, "data"))
    def _render_table(data: List[Dict[str, object]]):
        if not data:
            return []
        rows = []
        for rec in data:
            rows.append(
                {
                    "test_id": rec["test_id"],
                    "missing": ", ".join(rec["missing"]),
                    "resolve": "[Resolve](#)",
                }
            )
        return rows
    @app.callback(
        Output(MODAL_ID, "is_open"),
        Output(SELECTED_TEST_STORE, "data"),
        Output(f"{MODAL_ID}-body", "children"),
        Input(TABLE_ID, "active_cell"),
        State(TABLE_ID, "data"),
        prevent_initial_call=True,
    )
    def _open_modal(active_cell, rows):
        if not active_cell or active_cell.get("column_id") != "resolve":
            return dash.no_update, dash.no_update, dash.no_update
        row = rows[active_cell["row"]]
        body = f"Resolve missing components for test {row['test_id']}?"
        return True, row["test_id"], body

    @app.callback(
        Output(DATA_STORE, "data"),
        Output(MODAL_ID, "is_open"),
        Input("confirm-resolve", "n_clicks"),
        State(SELECTED_TEST_STORE, "data"),
        State(DATA_STORE, "data"),
        prevent_initial_call=True,
    )
    def _resolve_issue(n_clicks, test_id, data: List[Dict[str, object]]):
        if not n_clicks or not test_id:
            return data, False
        return [rec for rec in data if rec.get("test_id") != test_id], False

    return None
