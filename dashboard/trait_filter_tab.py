"""Trait filtering tab for the dashboard.

Provides simple UI components to filter Sample records by certain traits.
Database queries are stubbed so the module can run without MongoDB.
"""

from __future__ import annotations

from typing import List, Dict, Optional

import normalization_utils as norm_utils

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
NORMALIZE_CHECKBOX = "trait-normalize"


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
        Each dict contains ``name``, ``chemistry``, ``manufacturer`` and basic metrics.
    """
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis.models import Sample  # type: ignore

        qs = Sample.objects  # type: ignore[attr-defined]
        if chemistry:
            qs = qs.filter(chemistry=chemistry)
        if manufacturer:
            qs = qs.filter(manufacturer=manufacturer)
        rows = []
        for s in qs:
            rows.append(
                {
                    "name": s.name,
                    "chemistry": getattr(s, "chemistry", ""),
                    "manufacturer": getattr(s, "manufacturer", ""),
                    "sample_obj": s,
                    "capacity": getattr(s, "avg_final_capacity", None),
                    "resistance": getattr(s, "median_internal_resistance", None),
                    "ce": getattr(s, "avg_coulombic_eff", None),
                }
            )
        return rows
    except Exception:
        # Fallback demo data
        return [
            {
                "name": "Sample_001",
                "chemistry": chemistry or "NMC",
                "manufacturer": manufacturer or "ABC Batteries",
                "sample_obj": None,
                "capacity": 1.0,
                "resistance": 0.05,
                "ce": 0.98,
            }
        ]


def _build_table(rows: List[Dict[str, str]], normalized: bool) -> dbc.Table:
    cap_header = "Capacity (mAh/cm²)" if normalized else "Capacity (mAh)"
    res_header = "Resistance (Ω·cm²)" if normalized else "Resistance (Ω)"
    header = html.Thead(
        html.Tr(
            [
                html.Th("Name"),
                html.Th("Chemistry"),
                html.Th("Manufacturer"),
                html.Th(cap_header),
                html.Th(res_header),
                html.Th("CE %"),
            ]
        )
    )
    body_rows = []
    for r in rows:
        body_rows.append(
            html.Tr(
                [
                    html.Td(r["name"]),
                    html.Td(r["chemistry"]),
                    html.Td(r["manufacturer"]),
                    html.Td("{:.3f}".format(r["capacity"]) if r.get("capacity") is not None else "-"),
                    html.Td("{:.3f}".format(r["resistance"]) if r.get("resistance") is not None else "-"),
                    html.Td("{:.1f}".format(r["ce"]) if r.get("ce") is not None else "-"),
                ]
            )
        )
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
                    dbc.Col(
                        dcc.Checklist(
                            options=[{"label": "Normalize metrics", "value": "norm"}],
                            value=[],
                            id=NORMALIZE_CHECKBOX,
                            switch=True,
                        ),
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
        State(NORMALIZE_CHECKBOX, "value"),
        prevent_initial_call=True,
    )
    def _update_results(n_clicks, chemistry, manufacturer, normalize_value):
        normalize = normalize_value and "norm" in normalize_value
        rows = filter_samples(chemistry, manufacturer)
        for r in rows:
            sample = r.get("sample_obj")
            if normalize and sample is not None:
                cap = norm_utils.normalize_capacity(sample)
                res = norm_utils.normalize_impedance(sample)
                ce = norm_utils.coulombic_efficiency_percent(sample)
            else:
                cap = r.get("capacity")
                res = r.get("resistance")
                ce = r.get("ce")
            r["capacity"] = cap
            r["resistance"] = res
            r["ce"] = ce

        table = _build_table(rows, normalize) if rows else dbc.Alert("No results", color="warning")
        plot_placeholder = html.Div("Plot placeholder")
        return table, plot_placeholder

