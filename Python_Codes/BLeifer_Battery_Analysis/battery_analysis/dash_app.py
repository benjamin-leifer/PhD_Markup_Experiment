"""Simple Dash app mirroring the Tkinter tabs.

The original Tkinter application contains a number of tabs that group related
analysis tools.  This module recreates those tabs in a Dash application using
``dcc.Tabs`` for navigation and ``dash_bootstrap_components`` for a responsive
layout.
"""

from __future__ import annotations

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc


def create_app() -> dash.Dash:
    """Create and configure the Dash application.

    Returns
    -------
    dash.Dash
        Configured Dash app instance.
    """

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    tabs = [
        dcc.Tab(label="Dashboard", value="dashboard", children=html.Div("Dashboard")),
        dcc.Tab(
            label="Comparison", value="comparison", children=html.Div("Comparison")
        ),
        dcc.Tab(
            label="Advanced Analysis",
            value="advanced",
            children=html.Div("Advanced Analysis"),
        ),
        dcc.Tab(label="EIS", value="eis", children=html.Div("EIS")),
        dcc.Tab(label="PyBAMM", value="pybamm", children=html.Div("PyBAMM")),
        dcc.Tab(
            label="Missing Data",
            value="missing",
            children=html.Div("Missing Data"),
        ),
        dcc.Tab(
            label="Document Flow",
            value="doc_flow",
            children=html.Div("Document Flow"),
        ),
        dcc.Tab(
            label="Trait Filter",
            value="trait_filter",
            children=html.Div("Trait Filter"),
        ),
    ]

    app.layout = dbc.Container(
        dcc.Tabs(id="tabs", value="dashboard", children=tabs), fluid=True
    )

    return app


def main() -> None:  # pragma: no cover - CLI entry point
    """Run the Dash development server."""

    create_app().run(debug=True)


if __name__ == "__main__":  # pragma: no cover - module execution
    main()
