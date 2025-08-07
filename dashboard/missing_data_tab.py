"""Dash tab highlighting tests missing component assignments."""

from __future__ import annotations

from typing import List, Dict

from dash import html, dcc, Input, Output, State
import dash
import dash_bootstrap_components as dbc

ALERT_CONTAINER = "missing-data-alerts"
DATA_STORE = "missing-data-store"


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
        [dcc.Store(id=DATA_STORE, data=_get_missing_data()), html.Div(id=ALERT_CONTAINER)]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for resolving missing data alerts."""

    @app.callback(Output(ALERT_CONTAINER, "children"), Input(DATA_STORE, "data"))
    def _render_alerts(data: List[Dict[str, object]]):
        if not data:
            return [dbc.Alert("No missing data detected", color="success")]
        alerts = []
        for rec in data:
            msg = ", ".join(rec["missing"])
            alerts.append(
                dbc.Alert(
                    [
                        html.H6(f"Test {rec['test_id']}", className="mb-1"),
                        html.P(f"Missing components: {msg}", className="mb-1"),
                        dbc.Button(
                            "Mark Resolved",
                            id={"type": "resolve-btn", "index": rec["test_id"]},
                            color="link",
                        ),
                    ],
                    color="warning",
                    className="mb-2",
                )
            )
        return alerts

    @app.callback(
        Output(DATA_STORE, "data"),
        Input({"type": "resolve-btn", "index": dash.dependencies.ALL}, "n_clicks"),
        State(DATA_STORE, "data"),
        prevent_initial_call=True,
    )
    def _resolve_issue(n_clicks, data: List[Dict[str, object]]):
        ctx = dash.callback_context
        if not ctx.triggered:
            return data
        triggered = ctx.triggered[0]["prop_id"].split(".")[0]
        import json

        index = json.loads(triggered)["index"]
        return [rec for rec in data if rec.get("test_id") != index]

    return None
