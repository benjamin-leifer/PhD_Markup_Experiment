"""Trait filtering tab for the dashboard.

Simplified module providing UI components and query helpers for filtering
sample data.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any

import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash import Input, Output, State

from . import saved_filters, export_handler
import normalization_utils as norm_utils

# Component IDs used in callbacks
CHEMISTRY_DROPDOWN = "trait-chemistry"
MANUFACTURER_DROPDOWN = "trait-manufacturer"
SAMPLE_DROPDOWN = "trait-sample"
DATE_RANGE = "trait-date-range"
CYCLE_MIN_INPUT = "trait-cycle-min"
CYCLE_MAX_INPUT = "trait-cycle-max"
CE_MIN_INPUT = "trait-ce-min"
CE_MAX_INPUT = "trait-ce-max"
FILTER_BUTTON = "trait-filter-btn"
RESULTS_DIV = "trait-results"
PLOT_DIV = "trait-plot-area"
EXPORT_BUTTON = "trait-export-btn"
EXPORT_DOWNLOAD = "trait-export-download"
NORMALIZE_CHECKBOX = "trait-normalize"
METRIC_RADIO = "overlay-metric"
# Components for cross-tab communication
RESULTS_TABLE = "trait-results-table"
SELECTION_STORE = "trait-selected-sample"

# Saved filter components
FILTER_NAME_INPUT = "trait-filter-name"
SAVE_FILTER_BUTTON = "trait-save-filter"
SAVED_FILTER_DROPDOWN = "trait-saved-filters"


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


def get_sample_names(prefix: str = "") -> List[str]:
    """Return placeholder sample names starting with ``prefix``."""
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis.models import Sample  # type: ignore

        if prefix:
            return [
                s.name
                for s in Sample.objects(name__istartswith=prefix).only("name")  # type: ignore[attr-defined]
            ]
        return list(Sample.objects.distinct("name"))  # type: ignore[attr-defined]
    except Exception:
        samples = ["Sample_001", "Sample_002", "Demo_Cell"]
        if prefix:
            return [s for s in samples if s.lower().startswith(prefix.lower())]
        return samples


def filter_samples(
    chemistry: Optional[str],
    manufacturer: Optional[str],
    sample: Optional[str] = None,
    date_range: Optional[tuple[str, str]] = None,
    cycle_min: Optional[float] = None,
    cycle_max: Optional[float] = None,
    ce_min: Optional[float] = None,
    ce_max: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return sample dictionaries matching the given traits.

    The function attempts to query :mod:`battery_analysis`'s ``Sample`` model to
    obtain real data. If the database is unavailable, lightweight demo rows are
    returned instead so the dashboard still functions for documentation builds
    and tests.
    """

    rows: List[Dict[str, Any]] = []
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis.models import Sample  # type: ignore

        qs = Sample.objects  # type: ignore[attr-defined]
        if chemistry:
            qs = qs.filter(chemistry=chemistry)
        if manufacturer:
            qs = qs.filter(manufacturer=manufacturer)
        if sample:
            qs = qs.filter(name=sample)

        samples = list(qs)
        for s in samples:
            rows.append(
                {
                    "name": getattr(s, "name", ""),
                    "chemistry": getattr(s, "chemistry", ""),
                    "manufacturer": getattr(s, "manufacturer", ""),
                    "sample_obj": s,
                    "capacity": getattr(s, "avg_final_capacity", None),
                    "resistance": getattr(s, "median_internal_resistance", None),
                    "ce": getattr(s, "avg_coulombic_eff", None),
                    "date": getattr(s, "created_at", None),
                    "cycle_count": getattr(s, "cycle_count", None),
                }
            )
    except Exception:
        sample_name = sample or "Sample_001"
        rows = [
            {
                "name": sample_name,
                "chemistry": chemistry or "NMC",
                "manufacturer": manufacturer or "ABC Batteries",
                "sample_obj": None,
                "capacity": 1.0,
                "resistance": 0.05,
                "ce": 0.98,
                "date": "2024-01-01",
                "cycle_count": 50,
            }
        ]

    def _in_range(
        val: Optional[float], lo: Optional[float], hi: Optional[float]
    ) -> bool:
        if val is None:
            return True
        if lo is not None and val < lo:
            return False
        if hi is not None and val > hi:
            return False
        return True

    filtered: List[Dict[str, Any]] = []
    for r in rows:
        if date_range:
            start, end = date_range
            date = r.get("date")
            if start and date and date < start:
                continue
            if end and date and date > end:
                continue
        if not _in_range(r.get("cycle_count"), cycle_min, cycle_max):
            continue
        if not _in_range(r.get("ce"), ce_min, ce_max):
            continue
        filtered.append(r)

    return filtered


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
    sample: Optional[str] = None,
    date_range: Optional[tuple[str, str]] = None,
    ce_min: Optional[float] = None,
    ce_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Construct a MongoDB-style query dict from UI selections."""

    conditions: List[Dict[str, Any]] = []

    if chemistry:
        conditions.append({"chemistry": chemistry})
    if manufacturer:
        conditions.append({"manufacturer": manufacturer})
    if sample:
        conditions.append({"name": sample})

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

    if date_range is not None:
        start, end = date_range
        comp: Dict[str, Any] = {}
        if start:
            comp["$gte"] = start
        if end:
            comp["$lte"] = end
        if comp:
            conditions.append({"date": comp})

    if ce_min is not None or ce_max is not None:
        comp: Dict[str, Any] = {}
        if ce_min is not None:
            comp["$gt"] = ce_min
        if ce_max is not None:
            comp["$lt"] = ce_max
        if comp:
            conditions.append({"ce": comp})

    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _build_table(rows: List[Dict[str, Any]], normalized: bool) -> dash_table.DataTable:
    cap_header = "Capacity (mAh/cm²)" if normalized else "Capacity (mAh)"
    res_header = "Resistance (Ω·cm²)" if normalized else "Resistance (Ω)"

    # calculate normalized metrics when requested
    if normalized:
        for r in rows:
            sample = r.get("sample_obj")
            if sample is not None:
                r["capacity"] = norm_utils.normalize_capacity(sample)
                r["resistance"] = norm_utils.normalize_impedance(sample)
                r["ce"] = norm_utils.coulombic_efficiency_percent(sample)

    # round numeric values for display
    for r in rows:
        if r.get("capacity") is not None:
            r["capacity"] = round(float(r["capacity"]), 3)
        if r.get("resistance") is not None:
            r["resistance"] = round(float(r["resistance"]), 3)
        if r.get("ce") is not None:
            r["ce"] = round(float(r["ce"]), 1)

    columns = [
        {"name": "Name", "id": "name"},
        {"name": "Chemistry", "id": "chemistry"},
        {"name": "Manufacturer", "id": "manufacturer"},
        {"name": cap_header, "id": "capacity", "type": "numeric"},
        {"name": res_header, "id": "resistance", "type": "numeric"},
        {"name": "CE %", "id": "ce", "type": "numeric"},
    ]

    return dash_table.DataTable(
        id=RESULTS_TABLE,
        columns=columns,
        data=rows,
        filter_action="native",
        sort_action="native",
        row_selectable="single",
        cell_selectable=False,
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left"},
    )


def layout() -> html.Div:
    chem_opts = [{"label": c, "value": c} for c in get_distinct_values("chemistry")]
    manu_opts = [{"label": m, "value": m} for m in get_distinct_values("manufacturer")]
    sample_opts = [{"label": s, "value": s} for s in get_sample_names()]
    saved_opts = [
        {"label": f["name"], "value": f["name"]} for f in saved_filters.list_filters()
    ]
    return html.Div(
        [
            dcc.Store(id=SELECTION_STORE),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            options=saved_opts,
                            id=SAVED_FILTER_DROPDOWN,
                            placeholder="Saved presets",
                            clearable=True,
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dcc.Input(
                            id=FILTER_NAME_INPUT,
                            placeholder="Preset name",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Save Preset", id=SAVE_FILTER_BUTTON, color="secondary"
                        ),
                        width="auto",
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
                        dcc.Dropdown(
                            options=sample_opts,
                            id=SAMPLE_DROPDOWN,
                            placeholder="Sample",
                            clearable=True,
                            searchable=True,
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dcc.DatePickerRange(id=DATE_RANGE),
                        width="auto",
                    ),
                    dbc.Col(
                        dcc.Input(
                            id=CYCLE_MIN_INPUT,
                            type="number",
                            placeholder="Cycle Min",
                            style={"width": "6em"},
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dcc.Input(
                            id=CYCLE_MAX_INPUT,
                            type="number",
                            placeholder="Cycle Max",
                            style={"width": "6em"},
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dcc.Input(
                            id=CE_MIN_INPUT,
                            type="number",
                            placeholder="CE Min",
                            style={"width": "6em"},
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dcc.Input(
                            id=CE_MAX_INPUT,
                            type="number",
                            placeholder="CE Max",
                            style={"width": "6em"},
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button("Filter", id=FILTER_BUTTON, color="primary"),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Export Results", id=EXPORT_BUTTON, color="secondary"
                        ),
                        width="auto",
                    ),
                    dcc.Download(id=EXPORT_DOWNLOAD),
                    dbc.Col(
                        dbc.Checklist(
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
        State(SAMPLE_DROPDOWN, "value"),
        State(DATE_RANGE, "start_date"),
        State(DATE_RANGE, "end_date"),
        State(CYCLE_MIN_INPUT, "value"),
        State(CYCLE_MAX_INPUT, "value"),
        State(CE_MIN_INPUT, "value"),
        State(CE_MAX_INPUT, "value"),
        State(NORMALIZE_CHECKBOX, "value"),
        prevent_initial_call=True,
    )
    def _update_results(
        n_clicks,
        chemistry,
        manufacturer,
        sample,
        start_date,
        end_date,
        cycle_min,
        cycle_max,
        ce_min,
        ce_max,
        normalize_value,
    ):
        normalize = normalize_value and "norm" in normalize_value
        rows = filter_samples(
            chemistry,
            manufacturer,
            sample=sample,
            date_range=(start_date, end_date),
            cycle_min=cycle_min,
            cycle_max=cycle_max,
            ce_min=ce_min,
            ce_max=ce_max,
        )
        table = (
            _build_table(rows, normalize)
            if rows
            else dbc.Alert("No results", color="warning")
        )
        return table

    @app.callback(
        Output(EXPORT_DOWNLOAD, "data"),
        Input(EXPORT_BUTTON, "n_clicks"),
        State(CHEMISTRY_DROPDOWN, "value"),
        State(MANUFACTURER_DROPDOWN, "value"),
        State(SAMPLE_DROPDOWN, "value"),
        State(DATE_RANGE, "start_date"),
        State(DATE_RANGE, "end_date"),
        State(CYCLE_MIN_INPUT, "value"),
        State(CYCLE_MAX_INPUT, "value"),
        State(CE_MIN_INPUT, "value"),
        State(CE_MAX_INPUT, "value"),
        State(NORMALIZE_CHECKBOX, "value"),
        prevent_initial_call=True,
    )
    def _export_results(
        n_clicks,
        chemistry,
        manufacturer,
        sample,
        start_date,
        end_date,
        cycle_min,
        cycle_max,
        ce_min,
        ce_max,
        normalize_value,
    ):
        normalize = normalize_value and "norm" in normalize_value
        rows = filter_samples(
            chemistry,
            manufacturer,
            sample=sample,
            date_range=(start_date, end_date),
            cycle_min=cycle_min,
            cycle_max=cycle_max,
            ce_min=ce_min,
            ce_max=ce_max,
        )
        if normalize:
            for r in rows:
                sample = r.get("sample_obj")
                if sample is not None:
                    r["capacity"] = norm_utils.normalize_capacity(sample)
                    r["resistance"] = norm_utils.normalize_impedance(sample)
                    r["ce"] = norm_utils.coulombic_efficiency_percent(sample)
        csv_str = export_handler.export_filtered_results(rows, format="csv")
        return dcc.send_string(csv_str, "filtered_results.csv")

    @app.callback(
        Output(SAVED_FILTER_DROPDOWN, "options"),
        Output(SAVED_FILTER_DROPDOWN, "value"),
        Input(SAVE_FILTER_BUTTON, "n_clicks"),
        State(FILTER_NAME_INPUT, "value"),
        State(CHEMISTRY_DROPDOWN, "value"),
        State(MANUFACTURER_DROPDOWN, "value"),
        State(SAMPLE_DROPDOWN, "value"),
        State(DATE_RANGE, "start_date"),
        State(DATE_RANGE, "end_date"),
        State(CYCLE_MIN_INPUT, "value"),
        State(CYCLE_MAX_INPUT, "value"),
        State(CE_MIN_INPUT, "value"),
        State(CE_MAX_INPUT, "value"),
        prevent_initial_call=True,
    )
    def _save_filter(
        n_clicks,
        name,
        chemistry,
        manufacturer,
        sample,
        start_date,
        end_date,
        cycle_min,
        cycle_max,
        ce_min,
        ce_max,
    ):
        if not name:
            return dash.no_update, dash.no_update
        filt = {
            "chemistry": chemistry,
            "manufacturer": manufacturer,
            "sample": sample,
            "start_date": start_date,
            "end_date": end_date,
            "cycle_min": cycle_min,
            "cycle_max": cycle_max,
            "ce_min": ce_min,
            "ce_max": ce_max,
        }
        saved_filters.save_filter(name, filt)
        options = [
            {"label": f["name"], "value": f["name"]}
            for f in saved_filters.list_filters()
        ]
        return options, name

    @app.callback(
        Output(SELECTION_STORE, "data"),
        Input(RESULTS_TABLE, "selected_rows"),
        State(RESULTS_TABLE, "data"),
        prevent_initial_call=True,
    )
    def _update_store(selected_rows, table_data):
        if not selected_rows or not table_data:
            return dash.no_update
        idx = selected_rows[0]
        if idx >= len(table_data):
            return dash.no_update
        row = table_data[idx]
        return {"sample": row.get("name")}

    @app.callback(
        Output(CHEMISTRY_DROPDOWN, "value"),
        Output(MANUFACTURER_DROPDOWN, "value"),
        Output(SAMPLE_DROPDOWN, "value"),
        Output(DATE_RANGE, "start_date"),
        Output(DATE_RANGE, "end_date"),
        Output(CYCLE_MIN_INPUT, "value"),
        Output(CYCLE_MAX_INPUT, "value"),
        Output(CE_MIN_INPUT, "value"),
        Output(CE_MAX_INPUT, "value"),
        Input(SAVED_FILTER_DROPDOWN, "value"),
        prevent_initial_call=True,
    )
    def _load_filter(name):
        if not name:
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        filt = saved_filters.load_filter(name)
        return (
            filt.get("chemistry"),
            filt.get("manufacturer"),
            filt.get("sample"),
            filt.get("start_date"),
            filt.get("end_date"),
            filt.get("cycle_min"),
            filt.get("cycle_max"),
            filt.get("ce_min"),
            filt.get("ce_max"),
        )
