"""Trait filtering tab for the dashboard.

Provides simple UI components to filter Sample records by certain traits.
Database queries are stubbed so the module can run without MongoDB.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State


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
        Each dict contains ``name``, ``chemistry`` and ``manufacturer`` keys.
    """
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis.models import Sample  # type: ignore

        qs = Sample.objects  # type: ignore[attr-defined]
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
    add_opts = [{"label": a, "value": a} for a in get_distinct_values("additive")]
    tag_opts = [{"label": t, "value": t} for t in get_distinct_values("tags")]

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
            dbc.Row(
                [
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
        table = _build_table(rows) if rows else dbc.Alert("No results", color="warning")
        # Plot placeholder. In the future this could be a dcc.Graph figure.
        plot_placeholder = html.Div("Plot placeholder")
        return table, plot_placeholder

