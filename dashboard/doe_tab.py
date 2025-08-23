"""Design-of-experiments tab displaying plan heatmaps."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import dash
import plotly.graph_objs as go
from dash import Input, Output, dcc, html
import pandas as pd
import tempfile
from pathlib import Path
import types

try:  # pragma: no cover - optional dependency
    from battery_analysis.utils import doe_builder
except Exception:  # pragma: no cover - gracefully handle missing package
    doe_builder = None  # type: ignore

# Component ids used across the DOE tab.  They are centralized here so the
# callbacks in :mod:`dashboard.app` can reference them without duplicating the
# string literals.
PLAN_DROPDOWN = "doe-plan"
PLAN_NAME = "doe-plan-name"
FACTOR_INPUT = "doe-factor-name"
ADD_FACTOR = "doe-add-factor"
FACTOR_SELECT = "doe-factor-select"
LEVEL_INPUT = "doe-level-name"
ADD_LEVEL = "doe-add-level"
MATRIX_INPUT = "doe-matrix-row"
ADD_ROW = "doe-add-row"
FACTORS_DIV = "doe-factors"
MATRIX_DIV = "doe-matrix"
PLAN_STORE = "doe-plan-store"
SAVE_PLAN = "doe-save-plan"
FEEDBACK_DIV = "doe-feedback"
HEATMAP_GRAPH = "doe-heatmap"
PROGRESS_CSV_BTN = "doe-progress-csv-btn"
PROGRESS_HTML_BTN = "doe-progress-html-btn"
PROGRESS_CSV_DOWNLOAD = "doe-progress-csv-download"
PROGRESS_HTML_DOWNLOAD = "doe-progress-html-download"


def _load_plans() -> List[Dict[str, Any]]:
    """Return available :class:`ExperimentPlan` documents.

    Attempts to query the :mod:`battery_analysis` models for real data.
    When the database or package is unavailable, a small placeholder list is
    returned so the interface remains functional in offline environments.
    """

    try:  # pragma: no cover - requires database
        from battery_analysis import models

        plans = list(
            models.ExperimentPlan.objects.only(
                "name",
                "factors",
                "matrix",
            )
        )
        return [
            {
                "name": p.name,
                "factors": p.factors,
                "matrix": p.matrix,
            }
            for p in plans
        ]
    except Exception:
        return [
            {
                "name": "Demo Plan",
                "factors": {"anode": ["A", "B"], "cathode": ["X", "Y"]},
                "matrix": [
                    {"anode": "A", "cathode": "X"},
                    {"anode": "A", "cathode": "Y"},
                    {"anode": "B", "cathode": "X"},
                    {"anode": "B", "cathode": "Y"},
                ],
            }
        ]


def layout() -> html.Div:
    """Return the layout for the DOE heatmap tab."""

    options = [{"label": p["name"], "value": p["name"]} for p in _load_plans()]
    return html.Div(
        [
            dcc.Store(id=PLAN_STORE, data={"factors": {}, "matrix": []}),
            dcc.Dropdown(
                options=options,
                id=PLAN_DROPDOWN,
                placeholder="Select experiment plan",
            ),
            dcc.Input(id=PLAN_NAME, placeholder="Plan name"),
            html.Div(
                [
                    dcc.Input(id=FACTOR_INPUT, placeholder="Factor"),
                    html.Button("Add Factor", id=ADD_FACTOR),
                ]
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id=FACTOR_SELECT,
                        placeholder="Select factor",
                    ),
                    dcc.Input(id=LEVEL_INPUT, placeholder="Level"),
                    html.Button("Add Level", id=ADD_LEVEL),
                ]
            ),
            html.Div(id=FACTORS_DIV),
            dcc.Input(id=MATRIX_INPUT, placeholder="Matrix row (JSON)"),
            html.Button("Add Row", id=ADD_ROW),
            html.Div(id=MATRIX_DIV),
            html.Button("Save Plan", id=SAVE_PLAN),
            html.Div(id=FEEDBACK_DIV, className="text-danger"),
            html.Div(
                [
                    html.Button("Download Progress CSV", id=PROGRESS_CSV_BTN),
                    html.Button("Download Progress HTML", id=PROGRESS_HTML_BTN),
                    dcc.Download(id=PROGRESS_CSV_DOWNLOAD),
                    dcc.Download(id=PROGRESS_HTML_DOWNLOAD),
                ]
            ),
            dcc.Graph(id=HEATMAP_GRAPH),
        ]
    )


def _compute_status(plan: Dict[str, Any]) -> Dict[int, bool]:
    """Return mapping of matrix index to completion status."""

    matrix = plan.get("matrix", [])
    completed: Dict[int, bool] = {}
    try:  # pragma: no cover - requires database
        from battery_analysis import models

        for idx, combo in enumerate(matrix):
            query = {f"metadata.{k}": v for k, v in combo.items()}
            test = models.TestResult.objects(__raw__=query).first()
            completed[idx] = bool(test)
    except Exception:
        for idx, _ in enumerate(matrix):
            completed[idx] = idx % 2 == 0
    return completed


def _build_figure(plan: Dict[str, Any]) -> go.Figure:
    """Return heatmap figure for ``plan``."""

    factors = list(plan.get("factors", {}).keys())
    if len(factors) < 2:
        fig = go.Figure()
        fig.update_layout(title="Plan requires at least two factors")
        return fig
    x_factor, y_factor = factors[:2]
    x_levels = list(plan["factors"].get(x_factor, []))
    y_levels = list(plan["factors"].get(y_factor, []))
    status = [[0 for _ in x_levels] for _ in y_levels]
    completed = _compute_status(plan)
    for idx, combo in enumerate(plan.get("matrix", [])):
        xv = combo.get(x_factor)
        yv = combo.get(y_factor)
        if xv in x_levels and yv in y_levels:
            x_idx = x_levels.index(xv)
            y_idx = y_levels.index(yv)
            status[y_idx][x_idx] = 1 if completed.get(idx) else 0
    colorscale = [[0, "rgb(255,0,0)"], [1, "rgb(0,200,0)"]]
    fig = go.Figure(
        data=go.Heatmap(
            z=status,
            x=x_levels,
            y=y_levels,
            colorscale=colorscale,
            showscale=False,
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=plan.get("name", "Experiment Plan"),
        xaxis_title=x_factor,
        yaxis_title=y_factor,
    )
    return fig


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for the DOE heatmap tab."""

    @app.callback(  # type: ignore[misc]
        Output(HEATMAP_GRAPH, "figure"),
        Input(PLAN_DROPDOWN, "value"),
    )
    def _update_heatmap(plan_name: Optional[str]) -> go.Figure:
        if not plan_name:
            return go.Figure()
        plan = next((p for p in _load_plans() if p["name"] == plan_name), None)
        if not plan:
            return go.Figure()
        return _build_figure(plan)

    @app.callback(  # type: ignore[misc]
        Output(PROGRESS_CSV_DOWNLOAD, "data"),
        Input(PROGRESS_CSV_BTN, "n_clicks"),
        State(PLAN_DROPDOWN, "value"),
        prevent_initial_call=True,
    )
    def _download_progress_csv(n_clicks: int, plan_name: Optional[str]):
        if not plan_name:
            return dash.no_update
        plan = next((p for p in _load_plans() if p["name"] == plan_name), None)
        if not plan:
            return dash.no_update
        rows = [
            {**row, "completed": bool(row.get("tests"))}
            for row in plan.get("matrix", [])
        ]
        if doe_builder is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.close()
            pseudo = types.SimpleNamespace(name=plan_name, matrix=rows)
            doe_builder.export_progress_csv(pseudo, tmp.name)  # type: ignore[arg-type]
            csv_str = Path(tmp.name).read_text()
            Path(tmp.name).unlink(missing_ok=True)
        else:
            df = pd.DataFrame(rows)
            csv_str = df.to_csv(index=False)
        return dcc.send_string(csv_str, f"{plan_name}_progress.csv")

    @app.callback(  # type: ignore[misc]
        Output(PROGRESS_HTML_DOWNLOAD, "data"),
        Input(PROGRESS_HTML_BTN, "n_clicks"),
        State(PLAN_DROPDOWN, "value"),
        prevent_initial_call=True,
    )
    def _download_progress_html(n_clicks: int, plan_name: Optional[str]):
        if not plan_name:
            return dash.no_update
        plan = next((p for p in _load_plans() if p["name"] == plan_name), None)
        if not plan:
            return dash.no_update
        rows = [
            {**row, "completed": bool(row.get("tests"))}
            for row in plan.get("matrix", [])
        ]
        if doe_builder is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            tmp.close()
            pseudo = types.SimpleNamespace(name=plan_name, matrix=rows)
            doe_builder.export_progress_html(pseudo, tmp.name)  # type: ignore[arg-type]
            html_str = Path(tmp.name).read_text()
            Path(tmp.name).unlink(missing_ok=True)
        else:
            df = pd.DataFrame(rows)
            table_html = df.to_html(
                index=False, table_id="doe-progress-table", classes="display"
            )
            html_str = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<link rel=\"stylesheet\" href=\"https://cdn.datatables.net/1.13.5/css/jquery.dataTables.min.css\" />
<script src=\"https://code.jquery.com/jquery-3.6.0.min.js\"></script>
<script src=\"https://cdn.datatables.net/1.13.5/js/jquery.dataTables.min.js\"></script>
</head>
<body>
{table_html}
<script>
$(document).ready(function() {{
    $('#doe-progress-table').DataTable();
}});
</script>
</body>
</html>"""
        return dcc.send_string(html_str, f"{plan_name}_progress.html")
