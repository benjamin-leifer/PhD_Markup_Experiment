"""Dash application for battery test monitoring."""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import Input, Output, State
import json
import base64

from battery_analysis.parsers import parse_file

try:
    from . import cell_flagger
    from . import data_access
    from . import layout as layout_components
    from . import (
        trait_filter_tab,
        advanced_analysis_tab,
        ad_hoc_analysis_tab,
        cycle_detail_tab,
        eis_tab,
        comparison_tab,
        document_flow_tab,
        missing_data_tab,
    )
    from . import auth
    from . import preferences
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    cell_flagger = importlib.import_module("cell_flagger")
    data_access = importlib.import_module("data_access")
    layout_components = importlib.import_module("layout")
    trait_filter_tab = importlib.import_module("trait_filter_tab")
    advanced_analysis_tab = importlib.import_module("advanced_analysis_tab")
    ad_hoc_analysis_tab = importlib.import_module("ad_hoc_analysis_tab")
    cycle_detail_tab = importlib.import_module("cycle_detail_tab")
    eis_tab = importlib.import_module("eis_tab")
    comparison_tab = importlib.import_module("comparison_tab")
    document_flow_tab = importlib.import_module("document_flow_tab")
    missing_data_tab = importlib.import_module("missing_data_tab")
    auth = importlib.import_module("auth")
    preferences = importlib.import_module("preferences")


def create_app(test_role: str | None = None, enable_login: bool = False) -> dash.Dash:
    """Create and configure the Dash application.

    Parameters
    ----------
    test_role:
        Optional explicit role to start the app with.
    enable_login:
        If ``True`` the login page is shown and users must authenticate. When
        ``False`` (the default) the application skips the login page and starts
        with an admin role.
    """
    import diskcache
    from dash.background_callback import DiskcacheManager

    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        background_callback_manager=background_callback_manager,
    )

    if enable_login:
        auth.register_callbacks(app)
    else:
        test_role = test_role or "admin"

    def dashboard_layout(user_role: str) -> html.Div:
        prefs = preferences.load_preferences()
        running = data_access.get_running_tests()
        upcoming = data_access.get_upcoming_tests()
        stats = data_access.get_summary_stats()
        flags = cell_flagger.get_flags()
        navbar = dbc.NavbarSimple(
            dbc.Switch(
                id="theme-toggle",
                label="Dark mode",
                value=prefs.get("theme", "light") == "dark",
                className="ms-2",
            ),
            brand="Battery Test Dashboard",
            color="primary",
            dark=True,
            className="mb-3",
        )
        status_bar = dbc.Navbar(
            dbc.Container(
                [
                    html.Span(
                        "Status: Ready", id="status-text", className="navbar-text"
                    ),
                    html.Span(
                        "Database: Not Connected",
                        id="db-status",
                        className="ms-auto navbar-text",
                    ),
                ]
            ),
            color="light",
            fixed="bottom",
        )
        is_admin = user_role == "admin"
        tabs = dcc.Tabs(
            [
                # Order mirrors the original Tkinter GUI tabs
                dcc.Tab(
                    [
                        layout_components.summary_layout(stats),
                        html.H4("Running Tests"),
                        layout_components.running_tests_table(running),
                        html.H4("Upcoming Tests"),
                        layout_components.upcoming_tests_table(upcoming),
                    ],
                    label="Overview",
                    value="overview",
                ),
                dcc.Tab(
                    layout_components.new_material_form(),
                    label="New Material",
                    value="new-material",
                ),
                dcc.Tab(
                    layout_components.data_import_layout(),
                    label="Data Import",
                    disabled=not is_admin,
                    value="data-import",
                ),
                dcc.Tab(
                    layout_components.export_button(),
                    label="Export",
                    disabled=not is_admin,
                    value="export",
                ),
                dcc.Tab(
                    comparison_tab.layout(),
                    label="Comparison",
                    value="comparison",
                ),
                dcc.Tab(
                    advanced_analysis_tab.layout(),
                    label="Advanced Analysis",
                    disabled=not is_admin,
                    value="advanced-analysis",
                ),
                dcc.Tab(
                    ad_hoc_analysis_tab.layout(),
                    label="Ad Hoc Analysis",
                    value="ad-hoc",
                ),
                dcc.Tab(
                    cycle_detail_tab.layout(),
                    label="Cycle Detail",
                    value="cycle-detail",
                ),
                dcc.Tab(
                    eis_tab.layout(),
                    label="EIS",
                    value="eis",
                ),
                dcc.Tab(
                    document_flow_tab.layout(),
                    label="Document Status",
                    value="document-status",
                ),
                dcc.Tab(
                    missing_data_tab.layout(),
                    label="Missing Data",
                    value="missing-data",
                ),
                dcc.Tab(
                    trait_filter_tab.layout(),
                    label="Trait Filter",
                    disabled=not is_admin,
                    value="trait-filter",
                ),
                dcc.Tab(
                    html.Div(
                        layout_components.flagged_table(flags),
                        id="flagged-container",
                    ),
                    label="Flags",
                    value="flags",
                ),
            ],
            id="tabs",
            value=prefs.get("default_tab", "overview"),
        )
        return dbc.Container(
            [
                navbar,
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id="current-user",
                                options=[
                                    {"label": u, "value": u}
                                    for u in data_access.get_available_users()
                                ],
                                placeholder="Select User",
                                persistence=True,
                                persistence_type="local",
                            ),
                            md=4,
                        )
                    ],
                    className="mb-3",
                ),
                html.Div(id="user-set-out", style={"display": "none"}),
                dbc.Row([dbc.Col(tabs, width=12)]),
                layout_components.metadata_modal(),
                layout_components.export_modal(),
                dcc.Interval(id="refresh-interval", interval=60 * 1000, n_intervals=0),
                status_bar,
            ],
            fluid=True,
        )

    def serve_layout() -> html.Div:
        prefs = preferences.load_preferences()
        theme_href = (
            dbc.themes.DARKLY if prefs.get("theme") == "dark" else dbc.themes.BOOTSTRAP
        )
        return html.Div(
            [
                dcc.Store(id="user-role", data=test_role),
                dcc.Store(id="preferences", storage_type="local", data=prefs),
                html.Link(rel="stylesheet", href=theme_href, id="theme"),
                html.Div(id="page-content"),
            ]
        )

    app.layout = serve_layout

    @app.callback(Output("page-content", "children"), Input("user-role", "data"))
    def display_page(role):
        if enable_login and not role:
            return auth.layout()
        return dashboard_layout(role or "admin")

    @app.callback(Output("theme", "href"), Input("preferences", "data"))
    def apply_theme(prefs):
        theme = (prefs or {}).get("theme", "light")
        return dbc.themes.DARKLY if theme == "dark" else dbc.themes.BOOTSTRAP

    @app.callback(
        Output("preferences", "data"),
        Input("theme-toggle", "value"),
        Input("tabs", "value"),
        State("preferences", "data"),
    )
    def update_prefs(dark_mode, tab, data):
        prefs = data or {}
        prefs["theme"] = "dark" if dark_mode else "light"
        prefs["default_tab"] = tab
        preferences.save_preferences(prefs)
        return prefs

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
        Output("running-tests-table", "children"),
        Output("upcoming-tests-table", "children"),
        Input("refresh-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def refresh_test_tables(_):
        running = data_access.get_running_tests()
        upcoming = data_access.get_upcoming_tests()
        return (
            layout_components.running_tests_table(running).children,
            layout_components.upcoming_tests_table(upcoming).children,
        )

    @app.callback(
        Output("db-status", "children"), Input("refresh-interval", "n_intervals")
    )
    def refresh_db_status(_):
        status = "Connected" if data_access.db_connected() else "Not Connected"
        return f"Database: {status}"

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
        Output("upload-form", "style"),
        Output("meta-sample-code", "value"),
        Output("meta-chemistry", "value"),
        Output("meta-notes", "value"),
        Output("upload-info", "data"),
        Output("upload-status", "children"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        prevent_initial_call=True,
    )
    def handle_upload(contents, filename):
        if contents is None:
            raise dash.exceptions.PreventUpdate
        try:
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            path = data_access.store_temp_upload(filename, decoded)
            cycles, metadata = parse_file(path)
            info = {
                "filename": filename,
                "path": path,
                "cycles": cycles,
                "metadata": metadata,
            }
            style = {}
            return (
                style,
                metadata.get("sample_code", ""),
                metadata.get("chemistry", ""),
                metadata.get("notes", ""),
                info,
                dash.no_update,
            )
        except Exception as err:  # pragma: no cover - simple error handling
            alert = dbc.Alert(
                f"Failed to parse {filename}: {err}", color="danger", dismissable=True
            )
            return {"display": "none"}, "", "", "", None, alert

    @app.callback(
        Output("upload-status", "children", allow_duplicate=True),
        Output("upload-form", "style", allow_duplicate=True),
        Output("uploaded-files-list", "children"),
        Input("save-metadata", "n_clicks"),
        Input("cancel-metadata", "n_clicks"),
        State("meta-sample-code", "value"),
        State("meta-chemistry", "value"),
        State("meta-notes", "value"),
        State("upload-info", "data"),
        prevent_initial_call=True,
    )
    def save_metadata(save_clicks, cancel_clicks, sample_code, chemistry, notes, info):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger == "save-metadata" and info:
            try:
                metadata = info.get("metadata", {}) or {}
                metadata.update(
                    {
                        "sample_code": sample_code or "",
                        "chemistry": chemistry or "",
                        "notes": notes or "",
                    }
                )
                data_access.register_upload(
                    info["filename"], info["path"], info["cycles"], metadata
                )
                files = data_access.get_uploaded_files()
                items = [html.Li(f["filename"]) for f in files]
                status = dbc.Alert(
                    f"Saved {info['filename']}", color="success", dismissable=True
                )
                return status, {"display": "none"}, items
            except Exception as err:  # pragma: no cover - simple error handling
                alert = dbc.Alert(
                    f"Error saving metadata: {err}", color="danger", dismissable=True
                )
                return alert, dash.no_update, dash.no_update
        return (
            dbc.Alert("Upload canceled", color="secondary", dismissable=True),
            {"display": "none"},
            dash.no_update,
        )

    @app.callback(
        Output("flagged-container", "children"),
        Input({"type": "flag-review", "index": dash.ALL}, "n_clicks"),
        Input({"type": "flag-retest", "index": dash.ALL}, "n_clicks"),
        Input({"type": "clear-flag", "index": dash.ALL}, "n_clicks"),
        Input("refresh-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def update_flags(review_clicks, retest_clicks, clear_clicks, _):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        prop_id = ctx.triggered[0]["prop_id"]
        if prop_id == "refresh-interval.n_intervals":
            return layout_components.flagged_table(cell_flagger.get_flags())
        comp_id = json.loads(prop_id.split(".")[0])
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
    comparison_tab.register_callbacks(app)
    advanced_analysis_tab.register_callbacks(app)
    ad_hoc_analysis_tab.register_callbacks(app)
    cycle_detail_tab.register_callbacks(app)
    eis_tab.register_callbacks(app)
    document_flow_tab.register_callbacks(app)
    missing_data_tab.register_callbacks(app)

    return app


if __name__ == "__main__":  # pragma: no cover - manual execution
    create_app().run(debug=True)
