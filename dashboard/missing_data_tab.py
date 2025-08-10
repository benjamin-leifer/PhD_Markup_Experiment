"""Dash tab highlighting tests missing component assignments."""

from __future__ import annotations

from typing import List, Dict, Optional

from dash import html, dcc, Input, Output, State, dash_table, ALL
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


def _build_modal_body(
    missing: List[str],
    values: Optional[Dict[str, str]] = None,
    error: Optional[str] = None,
) -> List[html.Component]:
    """Return a list of input fields for the modal.

    Parameters
    ----------
    missing:
        List of component field names that require user input.
    values:
        Optional mapping of field name to the current value.  Used when
        re-rendering the modal after a validation error.
    error:
        Optional error message to display at the top of the modal body.
    """

    body: List[html.Component] = []
    if error:
        body.append(dbc.Alert(error, color="danger", id="resolve-error"))
    for field in missing:
        value = values.get(field, "") if values else ""
        body.append(
            dbc.FormGroup(
                [
                    dbc.Label(field.capitalize()),
                    dbc.Input(id={"type": "missing-component", "field": field}, value=value),
                ]
            )
        )
    return body


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
        missing_list = (
            [m.strip() for m in row["missing"].split(",")]
            if isinstance(row["missing"], str)
            else row["missing"]
        )
        body = _build_modal_body(missing_list)
        return True, {"test_id": row["test_id"], "missing": missing_list}, body

    @app.callback(
        Output(DATA_STORE, "data"),
        Output(MODAL_ID, "is_open"),
        Output(f"{MODAL_ID}-body", "children"),
        Input("confirm-resolve", "n_clicks"),
        State(SELECTED_TEST_STORE, "data"),
        State(DATA_STORE, "data"),
        State({"type": "missing-component", "field": ALL}, "value"),
        prevent_initial_call=True,
    )
    def _resolve_issue(n_clicks, selected, data: List[Dict[str, object]], values):
        if not n_clicks or not selected:
            return data, False, dash.no_update

        missing: List[str] = selected.get("missing", [])
        provided = {m: v for m, v in zip(missing, values or [])}

        if not all(v and str(v).strip() for v in provided.values()):
            body = _build_modal_body(missing, provided, "All fields are required")
            return data, True, body

        # Attempt to update the database; failures surface to the user.
        try:  # pragma: no cover - requires database
            from battery_analysis import models

            test = models.TestResult.objects(id=selected["test_id"]).first()
            if not test:
                raise RuntimeError("Test not found")
            sample = test.sample.fetch() if hasattr(test.sample, "fetch") else test.sample
            for field, name in provided.items():
                model_cls = getattr(models, field.capitalize())
                query = model_cls.objects(name=name)  # type: ignore[attr-defined]
                obj = query.first() if hasattr(query, "first") else None
                if not obj:
                    obj = model_cls(name=name)
                    if hasattr(obj, "save"):
                        obj.save()
                setattr(sample, field, obj)
            if hasattr(sample, "save"):
                sample.save()
        except Exception as exc:
            body = _build_modal_body(missing, provided, str(exc))
            return data, True, body

        new_data = [rec for rec in data if rec.get("test_id") != selected["test_id"]]
        return new_data, False, []

    return None
