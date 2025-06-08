"""Dash application for the battery analysis dashboard."""

import dash
import dash_bootstrap_components as dbc

# Allow running as a standalone script by fixing the package context
if __name__ == "__main__" and __package__ is None:  # pragma: no cover - manual run
    import pathlib
    import sys

    # Add the package root directory so that "battery_analysis" can be
    # imported when this file is executed as a script.
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "battery_analysis.web_dashboard"

from . import layout as layout_components
from . import data_access


def create_app() -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    def serve_layout():
        stats = data_access.get_summary_stats()
        live_tests = data_access.get_live_tests()
        upcoming_tests = data_access.get_upcoming_tests()
        recent_results = data_access.get_recent_results()

        return dbc.Container(
            [
                layout_components.navbar(),
                layout_components.dashboard_layout(
                    stats, live_tests, upcoming_tests, recent_results
                ),
                layout_components.export_modal(),
            ],
            fluid=True,
        )

    app.layout = serve_layout

    return app


if __name__ == "__main__":  # pragma: no cover - manual run
    create_app().run_server(debug=True)
