"""Dash application mirroring the Tkinter tab structure.

This module provides a small Dash application with a tabbed interface that
matches the tab order of the Tkinter GUI. It serves as a lightweight starting
point for a web-based dashboard.
"""

from __future__ import annotations

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# Optional feature detection mirroring the Tkinter GUI
try:  # pragma: no cover - optional dependency
    from . import advanced_analysis, MISSING_ADVANCED_PACKAGES  # type: ignore
    HAS_ADVANCED = advanced_analysis is not None
except Exception:  # pragma: no cover - optional dependency
    HAS_ADVANCED = False

try:  # pragma: no cover - optional dependency
    from . import eis  # type: ignore
    HAS_EIS = True
except Exception:  # pragma: no cover - optional dependency
    HAS_EIS = False

try:  # pragma: no cover - optional dependency
    from . import pybamm_models  # type: ignore
    HAS_PYBAMM = getattr(pybamm_models, "HAS_PYBAMM", False)
except Exception:  # pragma: no cover - optional dependency
    HAS_PYBAMM = False


def create_app() -> dash.Dash:
    """Create and configure the Dash application.

    Returns
    -------
    dash.Dash
        Configured Dash app instance.
    """

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    tabs = [
        dcc.Tab(label="Data Upload", value="data", children=html.Div("Data Upload")),
        dcc.Tab(label="Analysis", value="analysis", children=html.Div("Analysis")),
        dcc.Tab(
            label="Comparison", value="comparison", children=html.Div("Comparison")
        ),
    ]

    if HAS_ADVANCED:
        tabs.append(
            dcc.Tab(
                label="Advanced Analysis",
                value="advanced",
                children=html.Div("Advanced Analysis"),
            )
        )

    if HAS_EIS:
        tabs.append(
            dcc.Tab(label="EIS Analysis", value="eis", children=html.Div("EIS"))
        )

    if HAS_PYBAMM:
        tabs.append(
            dcc.Tab(
                label="PyBAMM Modeling",
                value="pybamm",
                children=html.Div("PyBAMM Modeling"),
            )
        )

    tabs.extend(
        [
            dcc.Tab(
                label="Dashboard", value="dashboard", children=html.Div("Dashboard")
            ),
            dcc.Tab(
                label="Document Flow",
                value="doc_flow",
                children=html.Div("Document Flow"),
            ),
            dcc.Tab(
                label="Missing Data",
                value="missing",
                children=html.Div("Missing Data"),
            ),
            dcc.Tab(
                label="Trait Filter",
                value="trait_filter",
                children=html.Div("Trait Filter"),
            ),
            dcc.Tab(
                label="Settings", value="settings", children=html.Div("Settings")
            ),
        ]
    )

    app.layout = dbc.Container(
        dcc.Tabs(id="tabs", value=tabs[0].value, children=tabs), fluid=True
    )

    return app


def main() -> None:  # pragma: no cover - CLI entry point
    """Run the Dash development server."""

    create_app().run(debug=True)


if __name__ == "__main__":  # pragma: no cover - module execution
    main()
