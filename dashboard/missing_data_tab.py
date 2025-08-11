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

        coll = models.TestResult._get_collection()  # type: ignore[attr-defined]
        for field in (
            "sample.anode",
            "sample.cathode",
            "sample.separator",
            "sample.electrolyte",
        ):
            coll.create_index(field)

        query = {
            "$or": [
                {"sample.anode": None},
                {"sample.cathode": None},
                {"sample.separator": None},
                {"sample.electrolyte": None},
            ]
        }
        records: List[Dict[str, object]] = []
        for test in models.TestResult.objects(__raw__=query).only(
            "id", "cell_code", "sample"
        ):  # type: ignore[attr-defined]
            sample = (
                test.sample.fetch() if hasattr(test.sample, "fetch") else test.sample
            )
            missing = [
                f
                for f in ("anode", "cathode", "separator", "electrolyte")
                if getattr(sample, f, None) is None
            ]
            if missing:
                records.append(
                    {
                        "test_id": str(test.id),
                        "cell_code": getattr(test, "cell_code", ""),
                        "missing": missing,
                    }
                )
        return records
    except Exception:
        return [
            {
                "test_id": "Test_A",
                "cell_code": "Cell_A",
                "missing": ["cathode", "separator"],
            },
            {
                "test_id": "Test_B",
                "cell_code": "Cell_B",
                "missing": ["electrolyte"],
            },
        ]


def _suggest_values(test_id: str, missing: List[str]) -> Dict[str, str]:
    """Return best-guess values for ``missing`` components of ``test_id``.

    The implementation queries the :mod:`battery_analysis` models and uses
    :func:`similarity_suggestions.suggest_similar_samples` to locate comparable
    samples.  Any attribute names found on those samples are offered as initial
    values.  Failures to access the database simply result in an empty mapping
    so the interface remains functional in offline contexts.
    """

    guesses: Dict[str, str] = {}
    try:  # pragma: no cover - requires database
        from battery_analysis import models
        from similarity_suggestions import suggest_similar_samples

        test = models.TestResult.objects(id=test_id).first()  # type: ignore[attr-defined]
        if not test:
            return {}
        sample = test.sample.fetch() if hasattr(test.sample, "fetch") else test.sample

        suggestions = suggest_similar_samples(str(sample.id))
        for suggestion in suggestions:
            other = models.Sample.objects(id=suggestion["sample_id"]).first()  # type: ignore[attr-defined]
            if not other:
                continue
            for field in missing:
                if field in guesses:
                    continue
                obj = getattr(other, field, None)
                name = getattr(obj, "name", None)
                if name:
                    guesses[field] = name
            if len(guesses) == len(missing):
                break
    except Exception:
        return {}

    return guesses


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
        Optional mapping of field name to the current value or suggested
        default.  Used both for initial rendering with best guesses and when
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
            html.Div(
                [
                    dbc.Label(field.capitalize()),
                    dbc.Input(
                        id={"type": "missing-component", "field": field}, value=value
                    ),
                ],
                className="mb-3",
            )
        )
    return body


def layout() -> html.Div:
    """Return layout for the missing data tab."""
    return html.Div(
        [
            dcc.Store(id=DATA_STORE, data=[]),
            dcc.Store(id=SELECTED_TEST_STORE),
            dash_table.DataTable(
                id=TABLE_ID,
                columns=[
                    {"name": "Cell Code", "id": "cell_code"},
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
                        dbc.Button(
                            "Mark Resolved", id="confirm-resolve", color="primary"
                        )
                    ),
                ],
                id=MODAL_ID,
                is_open=False,
            ),
        ]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for resolving missing data alerts."""

    @app.callback(
        Output(DATA_STORE, "data", allow_duplicate=True),
        Input("tabs", "value"),
        State(DATA_STORE, "data"),
        prevent_initial_call=True,
    )
    def _load_data(active_tab, data):
        if active_tab == "missing-data" and not data:
            return _get_missing_data()
        return dash.no_update

    @app.callback(Output(TABLE_ID, "data"), Input(DATA_STORE, "data"))
    def _render_table(data: List[Dict[str, object]]):
        if not data:
            return []
        rows = []
        for rec in data:
            rows.append(
                {
                    "test_id": rec["test_id"],
                    "cell_code": rec.get("cell_code", ""),
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
        """Display the modal for the selected test row.

        The DataTable stores the missing components as a comma separated
        string; this callback converts that representation back into a list and
        builds the modal body dynamically.  Returning ``dash.no_update`` for
        all outputs when another column is clicked keeps the current modal
        state intact.
        """

        if not active_cell or active_cell.get("column_id") != "resolve":
            return dash.no_update, dash.no_update, dash.no_update

        row = rows[active_cell["row"]]
        missing_list = (
            [m.strip() for m in row["missing"].split(",")]
            if isinstance(row["missing"], str)
            else row["missing"]
        )
        suggestions = _suggest_values(row["test_id"], missing_list)
        body = [
            html.P(f"Cell Code: {row.get('cell_code', '')}", id="cell-code-display")
        ] + _build_modal_body(missing_list, suggestions)
        selected = {
            "test_id": row["test_id"],
            "cell_code": row.get("cell_code", ""),
            "missing": missing_list,
            "suggestions": suggestions,
        }
        return True, selected, body

    @app.callback(
        Output(DATA_STORE, "data", allow_duplicate=True),
        Output(MODAL_ID, "is_open", allow_duplicate=True),
        Output(f"{MODAL_ID}-body", "children", allow_duplicate=True),
        Input("confirm-resolve", "n_clicks"),
        State(SELECTED_TEST_STORE, "data"),
        State(DATA_STORE, "data"),
        State({"type": "missing-component", "field": ALL}, "value"),
        prevent_initial_call=True,
    )
    def _resolve_issue(n_clicks, selected, data: List[Dict[str, object]], values):
        """Validate user input and persist component assignments.

        The callback creates or fetches component objects using
        :mod:`battery_analysis.models`, attaches them to the selected test's
        :class:`~battery_analysis.models.Sample`, and updates the table to hide
        resolved tests.  Any exceptions bubble up as an error message within
        the modal so users receive immediate feedback.
        """

        if not n_clicks or not selected:
            return data, False, dash.no_update

        missing: List[str] = selected.get("missing", [])
        suggested = selected.get("suggestions", {})
        provided = {
            m: (v if v and str(v).strip() else suggested.get(m, ""))
            for m, v in zip(missing, values or [])
        }

        if not all(v and str(v).strip() for v in provided.values()):
            body = [
                html.P(
                    f"Cell Code: {selected.get('cell_code', '')}",
                    id="cell-code-display",
                )
            ] + _build_modal_body(missing, provided, "All fields are required")
            return data, True, body

        # Attempt to update the database; failures surface to the user.
        try:  # pragma: no cover - requires database
            from battery_analysis import models

            test = models.TestResult.objects(id=selected["test_id"]).first()
            if not test:
                raise RuntimeError("Test not found")
            sample = (
                test.sample.fetch() if hasattr(test.sample, "fetch") else test.sample
            )

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
            body = [
                html.P(
                    f"Cell Code: {selected.get('cell_code', '')}",
                    id="cell-code-display",
                )
            ] + _build_modal_body(missing, provided, str(exc))
            return data, True, body

        new_data = [rec for rec in data if rec.get("test_id") != selected["test_id"]]
        return new_data, False, []

    return None
