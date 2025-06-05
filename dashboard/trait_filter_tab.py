"""Trait filtering tab for the dashboard.

Provides simple UI components to filter Sample records by certain traits.
Database queries are stubbed so the module can run without MongoDB.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any

from . import saved_filters

import normalization_utils as norm_utils

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State
from . import plot_overlay


# Component IDs used in callbacks
CHEMISTRY_DROPDOWN = "trait-chemistry"
MANUFACTURER_DROPDOWN = "trait-manufacturer"
ADDITIVE_DROPDOWN = "trait-additive"
ADDITIVE_MODE = "trait-additive-mode"
TAG_DROPDOWN = "trait-tags"
TAG_MODE = "trait-tag-mode"
CYCLE_MIN = "trait-cycle-min"
CYCLE_MAX = "trait-cycle-max"
THICK_MIN = "trait-thick-min"
THICK_MAX = "trait-thick-max"
FILTER_BUTTON = "trait-filter-btn"
SAVED_FILTER_DROPDOWN = "trait-saved-filter"
RESULTS_DIV = "trait-results"
PLOT_DIV = "trait-plot-area"

NORMALIZE_CHECKBOX = "trait-normalize"
METRIC_RADIO = "overlay-metric"



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
        if field == "additive":
            return ["FEC", "VC", "PS"]
        if field == "tags":
            return ["high_energy", "fast_charge", "demo"]
        return []


def filter_samples(query: Dict[str, Any]) -> List[Dict[str, str]]:
    """Query samples matching the provided traits.

    Parameters
    ----------
    query:
        MongoDB-style query dictionary built by :func:`build_query`.

    Returns
    -------
    list of dict
        Each dict contains ``name``, ``chemistry``, ``manufacturer`` and basic metrics.
    """
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis.models import Sample  # type: ignore
        from battery_analysis import user_tracking

        user_tracking.log_filter_run(
            "trait_filter",
            {"chemistry": chemistry, "manufacturer": manufacturer},
        )

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
        if query:
            qs = qs.filter(**query)
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
        chem = None
        manu = None
        if isinstance(query, dict):
            chem = query.get("chemistry")
            manu = query.get("manufacturer")
            if not chem or not manu:
                # Look into $and clauses
                for cond in query.get("$and", []):
                    if not chem:
                        chem = cond.get("chemistry")
                    if not manu:
                        manu = cond.get("manufacturer")

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
                "chemistry": chem or "NMC",
                "manufacturer": manu or "ABC Batteries",
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


def _build_table(rows: List[Dict[str, str]]) -> dbc.Table:
    header = html.Thead(
        html.Tr([html.Th("Name"), html.Th("Chemistry"), html.Th("Manufacturer")])
    )
    body_rows = [
        html.Tr(
            [html.Td(r["name"]), html.Td(r["chemistry"]), html.Td(r["manufacturer"])]
        )
        for r in rows
    ]
    body = html.Tbody(body_rows)
    return dbc.Table([header, body], bordered=True, hover=True, striped=True)


def layout() -> html.Div:
    """Return the layout for the trait filter tab."""
    chem_opts = [{"label": c, "value": c} for c in get_distinct_values("chemistry")]
    manu_opts = [{"label": m, "value": m} for m in get_distinct_values("manufacturer")]
    add_opts = [{"label": a, "value": a} for a in get_distinct_values("additive")]
    tag_opts = [{"label": t, "value": t} for t in get_distinct_values("tags")]
    saved_opts = [{"label": f["name"], "value": f["name"]} for f in saved_filters.list_filters()]


    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            options=saved_opts,
                            id=SAVED_FILTER_DROPDOWN,
                            placeholder="Saved Filters",
                            clearable=True,
                        ),
                        width=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            options=chem_opts,
                            id=CHEMISTRY_DROPDOWN,
                            placeholder="Chemistry",
                            clearable=True,
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            options=manu_opts,
                            id=MANUFACTURER_DROPDOWN,
                            placeholder="Manufacturer",
                            clearable=True,
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Button("Filter", id=FILTER_BUTTON, color="primary"),
                        width="auto",
                    ),
                ],
                className="mb-3",
            ),
            dcc.Tabs(
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

                    dbc.Col(
                        dcc.Dropdown(
                            options=add_opts,
                            id=ADDITIVE_DROPDOWN,
                            placeholder="Additives",
                            multi=True,
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.RadioItems(
                            options=[
                                {"label": "Any of", "value": "any"},
                                {"label": "All of", "value": "all"},
                                {"label": "Exclude", "value": "exclude"},
                            ],
                            value="any",
                            id=ADDITIVE_MODE,
                            inline=True,
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            options=tag_opts,
                            id=TAG_DROPDOWN,
                            placeholder="Tags",
                            multi=True,
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.RadioItems(
                            options=[
                                {"label": "Any of", "value": "any"},
                                {"label": "All of", "value": "all"},
                                {"label": "Exclude", "value": "exclude"},
                            ],
                            value="any",
                            id=TAG_MODE,
                            inline=True,
                        ),
                        width=3,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Input(
                            type="number",
                            id=CYCLE_MIN,
                            placeholder="Min cycles",
                        ),
                        width=2,
                    ),
                    dbc.Col(
                        dcc.Input(
                            type="number",
                            id=CYCLE_MAX,
                            placeholder="Max cycles",
                        ),
                        width=2,
                    ),
                    dbc.Col(
                        dcc.Input(
                            type="number",
                            id=THICK_MIN,
                            placeholder="Min thickness",
                        ),
                        width=2,
                    ),
                    dbc.Col(
                        dcc.Input(
                            type="number",
                            id=THICK_MAX,
                            placeholder="Max thickness",
                        ),
                        width=2,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(id=RESULTS_DIV)),

                    dcc.Tab(html.Div(id=RESULTS_DIV), label="Results"),
                    dcc.Tab(
                        html.Div(
                            [
                                dcc.RadioItems(
                                    id=METRIC_RADIO,
                                    options=[
                                        {
                                            "label": "Voltage vs Capacity",
                                            "value": "voltage_vs_capacity",
                                        },
                                        {
                                            "label": "CE vs Cycle",
                                            "value": "ce_vs_cycle",
                                        },
                                        {
                                            "label": "Impedance vs Cycle",
                                            "value": "impedance_vs_cycle",
                                        },
                                    ],
                                    value="voltage_vs_capacity",
                                    inline=True,
                                ),
                                html.Div(id=PLOT_DIV, className="mt-3"),
                            ]
                        ),
                        label="Overlay Plots",
                    ),

                ]
            ),
        ]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register Dash callbacks for the tab."""

    @app.callback(
        Output(CHEMISTRY_DROPDOWN, "value"),
        Output(MANUFACTURER_DROPDOWN, "value"),
        Input(SAVED_FILTER_DROPDOWN, "value"),
        prevent_initial_call=True,
    )
    def _apply_saved_filter(name):
        if not name:
            raise dash.exceptions.PreventUpdate
        try:
            filt = saved_filters.load_filter(name)
        except KeyError:
            return dash.no_update, dash.no_update
        return filt.get("chemistry"), filt.get("manufacturer")

    @app.callback(
        Output(RESULTS_DIV, "children"),
        Output(PLOT_DIV, "children"),
        Input(FILTER_BUTTON, "n_clicks"),
        Input(SAVED_FILTER_DROPDOWN, "value"),
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

        State(ADDITIVE_DROPDOWN, "value"),
        State(ADDITIVE_MODE, "value"),
        State(TAG_DROPDOWN, "value"),
        State(TAG_MODE, "value"),
        State(CYCLE_MIN, "value"),
        State(CYCLE_MAX, "value"),
        State(THICK_MIN, "value"),
        State(THICK_MAX, "value"),
        prevent_initial_call=True,
    )
    def _update_results(
        n_clicks,
        chemistry,
        manufacturer,
        additives,
        additive_mode,
        tags,
        tag_mode,
        cycle_min,
        cycle_max,
        thick_min,
        thick_max,
    ):
        query = build_query(
            chemistry,
            manufacturer,
            additives,
            additive_mode,
            tags,
            tag_mode,
            cycle_min,
            cycle_max,
            thick_min,
            thick_max,
        )
        rows = filter_samples(query)
        State(METRIC_RADIO, "value"),
        prevent_initial_call=True,
    )

    def _update_results(n_clicks, _saved_name, chemistry, manufacturer):

        rows = filter_samples(chemistry, manufacturer)
        table = _build_table(rows) if rows else dbc.Alert("No results", color="warning")

        fig = plot_overlay.plot_overlay(rows, metric=metric)
        import io, base64
        import matplotlib.pyplot as plt

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt_data = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close(fig)
        img = html.Img(
            src="data:image/png;base64," + plt_data, style={"max-width": "100%"}
        )

        return table, img
