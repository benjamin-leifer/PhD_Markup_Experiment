"""Dash application for the battery analysis dashboard."""

import dash
import dash_bootstrap_components as dbc

try:
    from . import layout as layout_components
    from . import data_access, doe_tab
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    layout_components = importlib.import_module("layout")
    data_access = importlib.import_module("data_access")
    doe_tab = importlib.import_module("doe_tab")


def create_app() -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    def serve_layout() -> dbc.Container:
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

    doe_tab.register_callbacks(app)

    return app


if __name__ == "__main__":  # pragma: no cover - manual run
    create_app().run(debug=True)
