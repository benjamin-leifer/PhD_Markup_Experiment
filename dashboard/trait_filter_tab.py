"""Trait filtering tab for the dashboard.

Simplified module providing UI components and query helpers for filtering
sample data.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State

from . import saved_filters, plot_overlay, export_handler
import normalization_utils as norm_utils

# Component IDs used in callbacks
CHEMISTRY_DROPDOWN = "trait-chemistry"
MANUFACTURER_DROPDOWN = "trait-manufacturer"
FILTER_BUTTON = "trait-filter-btn"
RESULTS_DIV = "trait-results"
PLOT_DIV = "trait-plot-area"
EXPORT_BUTTON = "trait-export-btn"
EXPORT_DOWNLOAD = "trait-export-download"
NORMALIZE_CHECKBOX = "trait-normalize"
METRIC_RADIO = "overlay-metric"


def get_distinct_values(field: str) -> List[str]:
    """Return placeholder distinct values for ``field``."""
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis.models import Sample  # type: ignore

        return list(Sample.objects.distinct(field))  # type: ignore[attr-defined]
    except Exception:
        if field == "chemistry":
            return ["NMC", "LFP", "LCO"]
        if field == "manufacturer":
            return ["ABC Batteries", "XYZ Cells"]
        return []


def filter_samples(chemistry: Optional[str], manufacturer: Optional[str]) -> List[Dict[str, Any]]:
    """Return placeholder sample rows matching the given traits."""
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


def build_query(
    chemistry: Optional[str],
    manufacturer: Optional[str],
    additives: Optional[List[str]],
    additive_mode: str,
    tags: Optional[List[str]],
    tag_mode: str,
    cycle_min: Optional[float],
    cycle_max: Optional[float],
    thick_min: Optional[float],
    thick_max: Optional[float],
) -> Dict[str, Any]:
    """Construct a MongoDB-style query dict from UI selections."""

    conditions: List[Dict[str, Any]] = []

    if chemistry:
        conditions.append({"chemistry": chemistry})
    if manufacturer:
        conditions.append({"manufacturer": manufacturer})

    if additives:
        if additive_mode == "all":
            conditions.append({"additives": {"$all": additives}})
        elif additive_mode == "exclude":
            conditions.append({"additives": {"$nin": additives}})
        else:
            conditions.append({"additives": {"$in": additives}})

    if tags:
        field = "tags"
        if tag_mode == "all":
            conditions.append({field: {"$all": tags}})
        elif tag_mode == "exclude":
            conditions.append({field: {"$nin": tags}})
        else:
            conditions.append({field: {"$in": tags}})

    if cycle_min is not None or cycle_max is not None:
        comp: Dict[str, Any] = {}
        if cycle_min is not None:
            comp["$gt"] = cycle_min
        if cycle_max is not None:
            comp["$lt"] = cycle_max
        if comp:
            conditions.append({"cycle_count": comp})

    if thick_min is not None or thick_max is not None:
        comp: Dict[str, Any] = {}
        if thick_min is not None:
            comp["$gt"] = thick_min
        if thick_max is not None:
            comp["$lt"] = thick_max
        if comp:
            conditions.append({"thickness": comp})

    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _build_table(rows: List[Dict[str, Any]], normalized: bool) -> dbc.Table:
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
    body_rows = [
        html.Tr(
            [
                html.Td(r["name"]),
                html.Td(r["chemistry"]),
                html.Td(r["manufacturer"]),
                html.Td(f"{r['capacity']:.3f}" if r.get("capacity") is not None else "-"),
                html.Td(f"{r['resistance']:.3f}" if r.get("resistance") is not None else "-"),
                html.Td(f"{r['ce']:.1f}" if r.get("ce") is not None else "-"),
            ]
        )
        for r in rows
    ]
    body = html.Tbody(body_rows)
    return dbc.Table([header, body], bordered=True, hover=True, striped=True)


def layout() -> html.Div:
    chem_opts = [{"label": c, "value": c} for c in get_distinct_values("chemistry")]
    manu_opts = [{"label": m, "value": m} for m in get_distinct_values("manufacturer")]
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(options=chem_opts, id=CHEMISTRY_DROPDOWN, placeholder="Chemistry", clearable=True),
                        width=3,
                    ),
                    dbc.Col(
                        dcc.Dropdown(options=manu_opts, id=MANUFACTURER_DROPDOWN, placeholder="Manufacturer", clearable=True),
                        width=3,
                    ),
                    dbc.Col(dbc.Button("Filter", id=FILTER_BUTTON, color="primary"), width="auto"),
                    dbc.Col(dbc.Button("Export Results", id=EXPORT_BUTTON, color="secondary"), width="auto"),
                    dcc.Download(id=EXPORT_DOWNLOAD),
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
            dbc.Row([dbc.Col(html.Div(id=RESULTS_DIV))]),
            dbc.Row([dbc.Col(html.Div(id=PLOT_DIV))]),
        ]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register Dash callbacks for the tab."""

    @app.callback(
        Output(RESULTS_DIV, "children"),
        Input(FILTER_BUTTON, "n_clicks"),
        State(CHEMISTRY_DROPDOWN, "value"),
        State(MANUFACTURER_DROPDOWN, "value"),
        State(NORMALIZE_CHECKBOX, "value"),
        prevent_initial_call=True,
    )
    def _update_results(n_clicks, chemistry, manufacturer, normalize_value):
        normalize = normalize_value and "norm" in normalize_value
        rows = filter_samples(chemistry, manufacturer)
        table = _build_table(rows, normalize) if rows else dbc.Alert("No results", color="warning")
        return table

    @app.callback(
        Output(EXPORT_DOWNLOAD, "data"),
        Input(EXPORT_BUTTON, "n_clicks"),
        State(CHEMISTRY_DROPDOWN, "value"),
        State(MANUFACTURER_DROPDOWN, "value"),
        State(NORMALIZE_CHECKBOX, "value"),
        prevent_initial_call=True,
    )
    def _export_results(n_clicks, chemistry, manufacturer, normalize_value):
        normalize = normalize_value and "norm" in normalize_value
        rows = filter_samples(chemistry, manufacturer)
        if normalize:
            for r in rows:
                sample = r.get("sample_obj")
                if sample is not None:
                    r["capacity"] = norm_utils.normalize_capacity(sample)
                    r["resistance"] = norm_utils.normalize_impedance(sample)
                    r["ce"] = norm_utils.coulombic_efficiency_percent(sample)
        csv_str = export_handler.export_filtered_results(rows, format="csv")
        return dcc.send_string(csv_str, "filtered_results.csv")
