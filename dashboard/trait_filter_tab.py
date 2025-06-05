"""Trait filtering tab for the dashboard.

Provides simple UI components to filter Sample records by certain traits.
Database queries are stubbed so the module can run without MongoDB.
"""

from __future__ import annotations

from typing import List, Dict, Optional

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State


# Component IDs used in callbacks
CHEMISTRY_DROPDOWN = "trait-chemistry"
MANUFACTURER_DROPDOWN = "trait-manufacturer"
FILTER_BUTTON = "trait-filter-btn"
RESULTS_DIV = "trait-results"
PLOT_DIV = "trait-plot-area"


def get_distinct_values(field: str) -> List[str]:
    """Return distinct values for ``field`` from :class:`Sample` records.

    When MongoEngine is not available or the query fails, demo values are
    returned so the dashboard still functions in a standalone manner.
    """
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis.models import Sample  # type: ignore

        return list(Sample.objects.distinct(field))  # type: ignore[attr-defined]
    except Exception:
        # Fallback demo values
        if field == "chemistry":
            return ["NMC", "LFP", "LCO"]
        if field == "manufacturer":
            return ["ABC Batteries", "XYZ Cells"]
        return []


def filter_samples(
    chemistry: Optional[str], manufacturer: Optional[str]
) -> List[Dict[str, str]]:
    """Query samples matching the provided traits.

    Parameters
    ----------
    chemistry:
        Desired chemistry string or ``None`` to ignore.
    manufacturer:
        Desired manufacturer string or ``None`` to ignore.

    Returns
    -------
    list of dict
        Each dict contains ``name``, ``chemistry`` and ``manufacturer`` keys.
    """
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis.models import Sample  # type: ignore

        qs = Sample.objects  # type: ignore[attr-defined]
        if chemistry:
            qs = qs.filter(chemistry=chemistry)
        if manufacturer:
            qs = qs.filter(manufacturer=manufacturer)
        return [
            {
                "name": s.name,
                "chemistry": getattr(s, "chemistry", ""),
                "manufacturer": getattr(s, "manufacturer", ""),
            }
            for s in qs
        ]
    except Exception:
        # Fallback demo data
        return [
            {
                "name": "Sample_001",
                "chemistry": chemistry or "NMC",
                "manufacturer": manufacturer or "ABC Batteries",
            }
        ]


def _build_table(rows: List[Dict[str, str]]) -> dbc.Table:
    header = html.Thead(html.Tr([html.Th("Name"), html.Th("Chemistry"), html.Th("Manufacturer")]))
    body_rows = [
        html.Tr([html.Td(r["name"]), html.Td(r["chemistry"]), html.Td(r["manufacturer"])])
        for r in rows
    ]
    body = html.Tbody(body_rows)
    return dbc.Table([header, body], bordered=True, hover=True, striped=True)


def layout() -> html.Div:
    """Return the layout for the trait filter tab."""
    chem_opts = [{"label": c, "value": c} for c in get_distinct_values("chemistry")]
    manu_opts = [{"label": m, "value": m} for m in get_distinct_values("manufacturer")]

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            options=chem_opts,
                            id=CHEMISTRY_DROPDOWN,
                            placeholder="Chemistry",
                            clearable=True,
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            options=manu_opts,
                            id=MANUFACTURER_DROPDOWN,
                            placeholder="Manufacturer",
                            clearable=True,
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Button("Filter", id=FILTER_BUTTON, color="primary"),
                        width="auto",
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(id=RESULTS_DIV)),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div("Plot placeholder", id=PLOT_DIV)),
                ],
                className="mt-3",
            ),
        ]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register Dash callbacks for the tab."""

    @app.callback(
        Output(RESULTS_DIV, "children"),
        Output(PLOT_DIV, "children"),
        Input(FILTER_BUTTON, "n_clicks"),
        State(CHEMISTRY_DROPDOWN, "value"),
        State(MANUFACTURER_DROPDOWN, "value"),
        prevent_initial_call=True,
    )
    def _update_results(n_clicks, chemistry, manufacturer):
        rows = filter_samples(chemistry, manufacturer)
        table = _build_table(rows) if rows else dbc.Alert("No results", color="warning")
        # Plot placeholder. In the future this could be a dcc.Graph figure.
        plot_placeholder = html.Div("Plot placeholder")
        return table, plot_placeholder

