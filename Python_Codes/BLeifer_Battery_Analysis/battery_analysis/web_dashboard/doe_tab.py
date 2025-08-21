"""DOE heatmap tab for the battery analysis web dashboard."""

from __future__ import annotations

from typing import Any, List, Optional

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go

PLAN_DROPDOWN = "doe-plan"
HEATMAP_GRAPH = "doe-heatmap"


def _load_plans() -> List[Any]:
    """Return available :class:`ExperimentPlan` documents.

    Tries to query the real database but falls back to a demo plan when running
    offline or without MongoDB.
    """
    try:  # pragma: no cover - requires database
        from battery_analysis.models import ExperimentPlan

        return list(ExperimentPlan.objects.only("name", "factors", "matrix"))
    except Exception:
        from types import SimpleNamespace

        return [
            SimpleNamespace(
                name="Demo Plan",
                factors={"anode": ["A", "B"], "cathode": ["X", "Y"]},
                matrix=[
                    {"anode": "A", "cathode": "X", "tests": [1]},
                    {"anode": "A", "cathode": "Y"},
                    {"anode": "B", "cathode": "X"},
                    {"anode": "B", "cathode": "Y", "tests": [1]},
                ],
            )
        ]


def layout() -> html.Div:
    """Return layout containing a plan selector and heatmap."""
    options = [{"label": p.name, "value": p.name} for p in _load_plans()]
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


def _plan_by_name(name: str) -> Any | None:
    """Return plan object with ``name`` or ``None`` if missing."""
    for plan in _load_plans():
        if plan.name == name:
            return plan
    return None


def _build_figure(plan: Any) -> go.Figure:
    """Return heatmap figure for ``plan`` highlighting completion status."""
    factors = list(getattr(plan, "factors", {}).keys())
    if len(factors) < 2:
        fig = go.Figure()
        fig.update_layout(title="Plan requires at least two factors")
        return fig
    x_factor, y_factor = factors[:2]
    x_levels = list(plan.factors.get(x_factor, []))
    y_levels = list(plan.factors.get(y_factor, []))
    matrix = list(getattr(plan, "matrix", []))

    try:  # pragma: no cover - requires database
        from battery_analysis.utils import doe_builder

        remaining = doe_builder.remaining_combinations(plan)
    except Exception:
        remaining = [row for row in matrix if not row.get("tests")]

    status = [[0 for _ in x_levels] for _ in y_levels]
    for row in matrix:
        xv, yv = row.get(x_factor), row.get(y_factor)
        if xv in x_levels and yv in y_levels:
            xi = x_levels.index(xv)
            yi = y_levels.index(yv)
            status[yi][xi] = 0 if row in remaining else 1

    colorscale = [[0, "rgb(255,0,0)"], [1, "rgb(0,200,0)"]]
    fig = go.Figure(
        data=go.Heatmap(
            z=status, x=x_levels, y=y_levels, colorscale=colorscale, showscale=False
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=getattr(plan, "name", "Experiment Plan"),
        xaxis_title=x_factor,
        yaxis_title=y_factor,
    )
    return fig


def register_callbacks(app: dash.Dash) -> None:
    """Register Dash callbacks for the DOE tab."""

    @app.callback(  # type: ignore[misc]
        Output(HEATMAP_GRAPH, "figure"), Input(PLAN_DROPDOWN, "value")
    )
    def _update_heatmap(plan_name: Optional[str]) -> go.Figure:
        if not plan_name:
            return go.Figure()
        plan = _plan_by_name(plan_name)
        if plan is None:
            return go.Figure()
        return _build_figure(plan)
