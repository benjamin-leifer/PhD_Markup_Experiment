"""Dash application for battery test monitoring."""

import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import Input, Output, State

from . import data_access
from . import layout as layout_components


def create_app() -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    def serve_layout():
        running = data_access.get_running_tests()
        upcoming = data_access.get_upcoming_tests()
        stats = data_access.get_summary_stats()
        return dbc.Container(
            [
                html.H2("Battery Test Dashboard", className="mt-2"),
                layout_components.summary_layout(stats),
                html.H4("Running Tests"),
                layout_components.running_tests_table(running),
                html.H4("Upcoming Tests"),
                layout_components.upcoming_tests_table(upcoming),
                layout_components.metadata_modal(),
            ],
            fluid=True,
        )

    app.layout = serve_layout

    @app.callback(
        Output("metadata-modal", "is_open"),
        Output("metadata-content", "children"),
        Input("running-tests-table", "active_cell"),
        Input("close-metadata", "n_clicks"),
        State("metadata-modal", "is_open"),
    )
    def display_metadata(active_cell, close_clicks, is_open):
        # If a table cell is clicked, show metadata modal
        if active_cell and active_cell.get("row") is not None:
            row = active_cell["row"]
            tests = data_access.get_running_tests()
            if row < len(tests):
                cell_id = tests[row]["cell_id"]
                meta = data_access.get_test_metadata(cell_id)
                body = html.Div([
                    html.P(f"Cell ID: {meta['cell_id']}", className="mb-1"),
                    html.P(f"Chemistry: {meta['chemistry']}", className="mb-1"),
                    html.P(f"Formation Date: {meta['formation_date']}", className="mb-1"),
                    html.P(meta['notes']),
                ])
                return True, body
        if close_clicks and is_open:
            return False, dash.no_update
        return is_open, dash.no_update

    return app


if __name__ == "__main__":  # pragma: no cover - manual execution
    create_app().run_server(debug=True)
