"""Design-of-experiments tab displaying plan heatmaps."""

from __future__ import annotations

from typing import Dict, List, Optional

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go


PLAN_DROPDOWN = "doe-plan"
HEATMAP_GRAPH = "doe-heatmap"


def _load_plans() -> List[Dict[str, object]]:
    """Return available :class:`ExperimentPlan` documents.

    Attempts to query the :mod:`battery_analysis` models for real data. When the
    database or package is unavailable, a small placeholder list is returned so
    the interface remains functional in offline environments.
    """

    try:  # pragma: no cover - requires database
        from battery_analysis import models

        plans = list(
            models.ExperimentPlan.objects.only("name", "factors", "matrix")  # type: ignore[attr-defined]
        )
        return [
            {"name": p.name, "factors": p.factors, "matrix": p.matrix} for p in plans
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
            dcc.Dropdown(
                options=options,
                id=PLAN_DROPDOWN,
                placeholder="Select experiment plan",
            ),
            dcc.Graph(id=HEATMAP_GRAPH),
        ]
    )


def _compute_status(plan: Dict[str, object]) -> Dict[int, bool]:
    """Return mapping of matrix index to completion status."""

    matrix = plan.get("matrix", [])
    completed: Dict[int, bool] = {}
    try:  # pragma: no cover - requires database
        from battery_analysis import models

        for idx, combo in enumerate(matrix):
            query = {f"metadata.{k}": v for k, v in combo.items()}
            test = models.TestResult.objects(__raw__=query).first()  # type: ignore[attr-defined]
            completed[idx] = bool(test)
    except Exception:
        for idx, _ in enumerate(matrix):
            completed[idx] = idx % 2 == 0
    return completed


def _build_figure(plan: Dict[str, object]) -> go.Figure:
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
            z=status, x=x_levels, y=y_levels, colorscale=colorscale, showscale=False
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

    @app.callback(Output(HEATMAP_GRAPH, "figure"), Input(PLAN_DROPDOWN, "value"))
    def _update_heatmap(plan_name: Optional[str]):
        if not plan_name:
            return go.Figure()
        plan = next((p for p in _load_plans() if p["name"] == plan_name), None)
        if not plan:
            return go.Figure()
        return _build_figure(plan)
