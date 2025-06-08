"""Dash application for battery test monitoring."""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import Input, Output, State
import json

from . import cell_flagger

from . import data_access
from . import layout as layout_components
from . import trait_filter_tab


def create_app() -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    def serve_layout():
        running = data_access.get_running_tests()
        upcoming = data_access.get_upcoming_tests()
        stats = data_access.get_summary_stats()
        flags = cell_flagger.get_flags()
        return dbc.Container(
            [
                html.H2("Battery Test Dashboard", className="mt-2"),
                dcc.Dropdown(
                    id="current-user",
                    options=[
                        {"label": "User 1", "value": "user1"},
                        {"label": "User 2", "value": "user2"},
                    ],
                    placeholder="Select User",
                ),
                html.Div(id="user-set-out", style={"display": "none"}),
                dcc.Tabs(
                    [
                        dcc.Tab(
                            [
                                layout_components.summary_layout(stats),
                                html.H4("Running Tests"),
                                layout_components.running_tests_table(running),
                                html.H4("Upcoming Tests"),
                                layout_components.upcoming_tests_table(upcoming),
                            ],
                            label="Overview",
                        ),
                        dcc.Tab(
                            layout_components.new_material_form(),
                            label="New Material",
                        ),
                        dcc.Tab(
                            layout_components.export_button(),
                            label="Export",
                        ),
                        dcc.Tab(
                            trait_filter_tab.layout(),
                            label="Trait Filter",
                        ),
                        dcc.Tab(
                            html.Div(
                                layout_components.flagged_table(flags),
                                id="flagged-container",
                            ),
                            label="Flags",
                        ),
                    ]
                ),
                layout_components.metadata_modal(),
                layout_components.export_modal(),
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
                body = html.Div(
                    [
                        html.P(f"Cell ID: {meta['cell_id']}", className="mb-1"),
                        html.P(f"Chemistry: {meta['chemistry']}", className="mb-1"),
                        html.P(
                            f"Formation Date: {meta['formation_date']}",
                            className="mb-1",
                        ),
                        html.P(meta["notes"]),
                    ]
                )
                return True, body
        if close_clicks and is_open:
            return False, dash.no_update
        return is_open, dash.no_update

    @app.callback(
        Output("material-submit-feedback", "children"),
        Input("submit-material", "n_clicks"),
        State("material-name", "value"),
        State("material-chemistry", "value"),
        State("material-notes", "value"),
        prevent_initial_call=True,
    )
    def submit_material(n_clicks, name, chemistry, notes):
        data_access.add_new_material(name or "", chemistry or "", notes or "")
        return dbc.Alert("Material submitted", color="success", dismissable=True)

    @app.callback(
        Output("export-modal", "is_open"),
        Input("open-export", "n_clicks"),
        Input("close-export", "n_clicks"),
        State("export-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_export(open_clicks, close_clicks, is_open):
        if open_clicks or close_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("download-data", "data"),
        Input("download-csv", "n_clicks"),
        State("export-choice", "value"),
        prevent_initial_call=True,
    )
    def export_csv(n_clicks, choice):
        if choice == "running":
            csv_str = data_access.get_running_tests_csv()
            filename = "running_tests.csv"
        else:
            csv_str = data_access.get_upcoming_tests_csv()
            filename = "upcoming_tests.csv"
        return dcc.send_string(csv_str, filename)

    @app.callback(
        Output("download-pdf-file", "data"),
        Input("download-pdf", "n_clicks"),
        State("export-choice", "value"),
        prevent_initial_call=True,
    )
    def export_pdf(n_clicks, choice):
        if choice == "running":
            pdf_bytes = data_access.get_running_tests_pdf()
            filename = "running_tests.pdf"
        else:
            pdf_bytes = data_access.get_upcoming_tests_pdf()
            filename = "upcoming_tests.pdf"
        return dcc.send_bytes(pdf_bytes, filename)

    @app.callback(
        Output("flagged-container", "children"),
        Input({"type": "flag-review", "index": dash.ALL}, "n_clicks"),
        Input({"type": "flag-retest", "index": dash.ALL}, "n_clicks"),
        Input({"type": "clear-flag", "index": dash.ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def update_flags(review_clicks, retest_clicks, clear_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        comp_id = json.loads(prop_id)
        sample_id = comp_id["index"]
        if comp_id["type"] == "flag-review":
            cell_flagger.flag_sample(sample_id, "manual review")
        elif comp_id["type"] == "flag-retest":
            cell_flagger.flag_sample(sample_id, "retest")
        elif comp_id["type"] == "clear-flag":
            cell_flagger.clear_flag(sample_id)
        return layout_components.flagged_table(cell_flagger.get_flags())

    @app.callback(Output("user-set-out", "children"), Input("current-user", "value"))
    def set_user(user):
        if user:
            try:
                from battery_analysis import user_tracking

                user_tracking.set_current_user(user)
            except Exception:
                pass
        return ""

    trait_filter_tab.register_callbacks(app)

    return app


if __name__ == "__main__":  # pragma: no cover - manual execution
    create_app().run(debug=True)
