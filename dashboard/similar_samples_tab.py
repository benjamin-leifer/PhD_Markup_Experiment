"""Dash tab for suggesting similar samples.

This tab provides a small form where a user can enter a reference sample
identifier.  Suggested samples are displayed in a table with links to the
comparison view of the dashboard.  The heavy lifting for computing
similarities is delegated to :func:`similarity_suggestions.suggest_similar_samples`.
"""

from __future__ import annotations

from typing import List, Dict

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State

SAMPLE_INPUT = "similar-sample-id"
SUGGEST_BUTTON = "similar-suggest-btn"
RESULTS_DIV = "similar-results"


def _render_table(ref_id: str, suggestions: List[Dict[str, str]]) -> dbc.Table:
    """Return a ``Table`` component for ``suggestions``."""

    header = html.Thead(
        html.Tr([html.Th("Sample"), html.Th("Score"), html.Th("Differences")])
    )
    rows = []
    for s in suggestions:
        link = dcc.Link(s["sample_id"], href=f"#/comparison?ref={ref_id}&other={s['sample_id']}")
        rows.append(
            html.Tr(
                [html.Td(link), html.Td(s["score"]), html.Td(s.get("differences", ""))]
            )
        )
    body = html.Tbody(rows)
    return dbc.Table([header, body], bordered=True, hover=True, striped=True, responsive=True)


def layout() -> html.Div:
    """Return the layout for the tab."""

    return html.Div(
        [
            dbc.Input(id=SAMPLE_INPUT, placeholder="Sample ID"),
            dbc.Button("Suggest", id=SUGGEST_BUTTON, color="primary", className="mt-2"),
            html.Div(id=RESULTS_DIV, className="mt-3"),
        ]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for the tab."""

    @app.callback(
        Output(RESULTS_DIV, "children"),
        Input(SUGGEST_BUTTON, "n_clicks"),
        State(SAMPLE_INPUT, "value"),
        prevent_initial_call=True,
    )
    def _suggest(n_clicks, sample_id):  # pragma: no cover - simple callback wiring
        if not sample_id:
            return html.Div("Please provide a sample id")
        try:
            from similarity_suggestions import suggest_similar_samples
        except Exception:
            return html.Div("Suggestion service unavailable")
        suggestions = suggest_similar_samples(sample_id)
        if not suggestions:
            return html.Div("No similar samples found")
        return _render_table(sample_id, suggestions)
