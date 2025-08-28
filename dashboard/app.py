"""Dash application for battery test monitoring."""

# flake8: noqa

# mypy: ignore-errors

# Automatically configure dependencies so the dashboard works out of the box
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import dash
import dash_bootstrap_components as dbc
from battery_analysis.utils.doe_builder import save_plan
from dash import Input, Output, State, dcc, html

try:  # pragma: no cover - runtime helper is simple
    from .runtime_setup import configure as _configure_runtime
except ImportError:  # pragma: no cover - allow running as script
    from runtime_setup import configure as _configure_runtime

_configure_runtime()

from battery_analysis.parsers import parse_file  # noqa: E402

try:
    from battery_analysis.utils import import_watcher

    from . import (
        ad_hoc_analysis_tab,
        advanced_analysis_tab,
        auth,
        cell_flagger,
        comparison_tab,
        cycle_detail_tab,
        data_access,
        document_flow_tab,
        doe_tab,
        eis_tab,
        import_jobs_tab,
        import_stats_tab,
    )
    from . import layout as layout_components
    from . import (
        missing_data_tab,
        preferences,
        raw_files_tab,
        refactor_jobs_tab,
        similar_samples_tab,
        trait_filter_tab,
        watcher_tab,
    )
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
    similar_samples_tab = importlib.import_module("similar_samples_tab")
    document_flow_tab = importlib.import_module("document_flow_tab")
    missing_data_tab = importlib.import_module("missing_data_tab")
    doe_tab = importlib.import_module("doe_tab")
    import_jobs_tab = importlib.import_module("import_jobs_tab")
    refactor_jobs_tab = importlib.import_module("refactor_jobs_tab")
    import_stats_tab = importlib.import_module("import_stats_tab")
    watcher_tab = importlib.import_module("watcher_tab")
    raw_files_tab = importlib.import_module("raw_files_tab")
    auth = importlib.import_module("auth")
    preferences = importlib.import_module("preferences")
    try:
        import_watcher = importlib.import_module(
            "battery_analysis.utils.import_watcher"
        )
    except Exception:  # pragma: no cover
        import_watcher = None  # type: ignore


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

    page_size = 25

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        background_callback_manager=background_callback_manager,
    )

    # Start configured import watchers so they persist across restarts
    if import_watcher:
        try:  # pragma: no cover - best effort startup
            for path in preferences.load_preferences().get("watcher_dirs", []):
                import_watcher.start_watcher(path)
        except Exception:
            pass

    if enable_login:
        auth.register_callbacks(app)
    else:
        test_role = test_role or "admin"

    def _overview_layout() -> html.Div:
        stats = data_access.get_summary_stats()
        running = data_access.get_running_tests(limit=page_size)["rows"]
        upcoming = data_access.get_upcoming_tests(limit=page_size)["rows"]
        return html.Div(
            [
                layout_components.summary_layout(stats),
                html.H4("Running Tests"),
                dcc.Loading(
                    id="running-tests-loading",
                    children=layout_components.running_tests_table(running),
                ),
                html.H4("Upcoming Tests"),
                dcc.Loading(
                    id="upcoming-tests-loading",
                    children=layout_components.upcoming_tests_table(upcoming),
                ),
            ]
        )

    tab_layouts = {
        "overview": _overview_layout,
        "new-material": layout_components.new_material_form,
        "data-import": layout_components.data_import_layout,
        "export": layout_components.export_button,
        "import-jobs": import_jobs_tab.layout,
        "refactor-jobs": refactor_jobs_tab.layout,
        "import-stats": import_stats_tab.layout,
        "watchers": watcher_tab.layout,
        "comparison": comparison_tab.layout,
        "similar-samples": similar_samples_tab.layout,
        "advanced-analysis": advanced_analysis_tab.layout,
        "ad-hoc": ad_hoc_analysis_tab.layout,
        "cycle-detail": cycle_detail_tab.layout,
        "eis": eis_tab.layout,
        "document-status": document_flow_tab.layout,
        "missing-data": missing_data_tab.layout,
        "doe-heatmap": doe_tab.layout,
        "trait-filter": trait_filter_tab.layout,
        "raw-files": raw_files_tab.layout,
        "flags": lambda: html.Div(
            layout_components.flagged_table(cell_flagger.get_flags()),
            id="flagged-container",
        ),
    }

    def dashboard_layout(user_role: str) -> html.Div:
        prefs = preferences.load_preferences()
        connected = data_access.db_connected()
        if connected:
            db_status = "Connected"
        else:
            err = data_access.get_db_error() or "Unknown error"
            db_status = f"Not Connected: {err}"
        navbar = dbc.Navbar(
            dbc.Container(
                [
                    dbc.NavbarToggler(id="navbar-toggler", className="me-2 d-md-none"),
                    dbc.NavbarBrand("Battery Test Dashboard", className="me-auto"),
                    dbc.Switch(
                        id="theme-toggle",
                        label="Dark mode",
                        value=prefs.get("theme", "light") == "dark",
                        className="ms-auto",
                    ),
                    dbc.Tooltip(
                        "Toggle dark mode",
                        target="theme-toggle",
                        placement="bottom",
                    ),
                ]
            ),
            color="primary",
            dark=True,
            className="mb-3",
        )
        status_bar = dbc.Navbar(
            dbc.Container(
                [
                    html.Span(
                        "Status: Ready",
                        id="status-text",
                        className="navbar-text",
                        **{"aria-live": "polite"},
                    ),
                    html.Span(
                        f"Database: {db_status}",
                        id="db-status",
                        className="ms-auto navbar-text",
                        **{"aria-live": "polite"},
                    ),
                ]
            ),
            color="light",
            fixed="bottom",
        )

        is_admin = user_role == "admin"

        def can(perm: str) -> bool:
            return auth.has_permission(user_role, perm)

        def nav_link(label: str, value: str, disabled: bool = False) -> dbc.NavLink:
            return dbc.NavLink(
                label, href=f"/{value}", disabled=disabled, active="exact"
            )

        nav_sections = {
            "Import": [
                nav_link("Data Import", "data-import", not can("data-import")),
                nav_link("Import Jobs", "import-jobs", not can("import-jobs")),
                nav_link("Refactor Jobs", "refactor-jobs", not can("refactor-jobs")),
                nav_link("Import Stats", "import-stats", not can("import-stats")),
                nav_link("Watchers", "watchers", not is_admin),
                nav_link("Raw Files", "raw-files", not can("raw-files")),
            ],
            "Analysis": [
                nav_link("Overview", "overview", not can("overview")),
                nav_link("Comparison", "comparison", not can("comparison")),
                nav_link("Similar Samples", "similar-samples"),
                nav_link(
                    "Advanced Analysis",
                    "advanced-analysis",
                    not can("advanced-analysis"),
                ),
                nav_link("Ad Hoc Analysis", "ad-hoc", not can("ad-hoc")),
                nav_link("Cycle Detail", "cycle-detail", not can("cycle-detail")),
                nav_link("EIS", "eis", not can("eis")),
                nav_link(
                    "Document Status", "document-status", not can("document-status")
                ),
                nav_link("Missing Data", "missing-data", not can("missing-data")),
                nav_link("DOE Heatmap", "doe-heatmap", not can("doe-heatmap")),
                nav_link("Trait Filter", "trait-filter", not can("trait-filter")),
            ],
            "Utilities": [
                nav_link("New Material", "new-material", not can("new-material")),
                nav_link("Export", "export", not can("export")),
                nav_link("Flags", "flags", not can("flags")),
            ],
        }

        def _sidebar_children():
            children: list[dash.Component] = []
            for section, links in nav_sections.items():
                children.append(html.H4(section))
                children.append(
                    dbc.Nav(links, vertical=True, pills=True, className="mb-4")
                )
            return children

        sidebar = html.Div(_sidebar_children())
        sidebar_offcanvas = dbc.Offcanvas(
            _sidebar_children(),
            id="sidebar-offcanvas",
            placement="start",
            className="d-md-none",
            title="Menu",
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
                dbc.Row(
                    [
                        dbc.Col(sidebar, md=2, className="d-none d-md-block"),
                        dbc.Col(
                            html.Div(
                                tab_layouts.get(
                                    prefs.get("default_tab", "overview"),
                                    _overview_layout,
                                )(),
                                id="tab-content",
                            ),
                            md=10,
                        ),
                    ]
                ),
                sidebar_offcanvas,
                layout_components.metadata_modal(),
                layout_components.export_modal(),
                layout_components.import_dir_job_store(),
                dcc.Interval(id="refresh-interval", interval=60 * 1000, n_intervals=0),
                layout_components.toast_container(),
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
                html.A(
                    "Skip to main content",
                    href="#main-content",
                    className="visually-hidden-focusable",
                ),
                dcc.Location(
                    id="url", pathname=f"/{prefs.get('default_tab', 'overview')}"
                ),
                layout_components.user_role_store(test_role),
                dcc.Store(id="preferences", storage_type="local", data=prefs),
                dcc.Store(id="active-tab", data=prefs.get("default_tab", "overview")),
                html.Link(rel="stylesheet", href=theme_href, id="theme"),
                dbc.Switch(
                    id="theme-toggle",
                    value=prefs.get("theme", "light") == "dark",
                    style={"display": "none"},
                ),
                html.Div(id="main-content"),
            ]
        )

    app.layout = serve_layout
    # Validation layout includes dynamic components so callbacks referencing
    # them are recognized even if those components aren't in the initial
    # layout. This prevents "nonexistent object" errors for elements like
    # the theme toggle.
    if enable_login:
        app.validation_layout = html.Div(
            [serve_layout(), auth.layout(), dashboard_layout(test_role or "admin")]
        )
    else:
        app.validation_layout = html.Div(
            [serve_layout(), dashboard_layout(test_role or "admin")]
        )

    @app.callback(Output("main-content", "children"), Input("user-role", "data"))
    def display_page(role):
        if enable_login and not role:
            return auth.layout()
        return dashboard_layout(role or "admin")

    @app.callback(
        Output("active-tab", "data"),
        Output("tab-content", "children"),
        Input("url", "pathname"),
        prevent_initial_call=True,
    )
    def render_tab(pathname):
        tab = pathname.lstrip("/") or "overview"
        layout_fn = tab_layouts.get(tab, _overview_layout)
        return tab, layout_fn()

    @app.callback(Output("theme", "href"), Input("preferences", "data"))
    def apply_theme(prefs):
        theme = (prefs or {}).get("theme", "light")
        return dbc.themes.DARKLY if theme == "dark" else dbc.themes.BOOTSTRAP

    @app.callback(
        Output("preferences", "data"),
        Input("theme-toggle", "value"),
        Input("active-tab", "data"),
        State("preferences", "data"),
    )
    def update_prefs(dark_mode, tab, data):
        prefs = data or {}
        prefs["theme"] = "dark" if dark_mode else "light"
        prefs["default_tab"] = tab
        preferences.save_preferences(prefs)
        return prefs

    @app.callback(
        Output("sidebar-offcanvas", "is_open"),
        Input("navbar-toggler", "n_clicks"),
        State("sidebar-offcanvas", "is_open"),
    )
    def toggle_sidebar(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("metadata-modal", "is_open"),
        Output("metadata-content", "children"),
        Input("running-tests-table", "active_cell", allow_optional=True),
        Input("close-metadata", "n_clicks"),
        State("metadata-modal", "is_open"),
        State("running-tests-table", "data", allow_optional=True),
    )
    def display_metadata(active_cell, close_clicks, is_open, rows):
        # If a table cell is clicked, show metadata modal
        if active_cell and active_cell.get("row") is not None:
            row = active_cell["row"]
            if rows and row < len(rows):
                cell_id = rows[row]["cell_id"]
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
        State("user-role", "data"),
        prevent_initial_call=True,
    )
    def submit_material(n_clicks, name, chemistry, notes, role):
        if not auth.has_permission(role, "new-material"):
            raise dash.exceptions.PreventUpdate
        data_access.add_new_material(name or "", chemistry or "", notes or "")
        return dbc.Alert("Material submitted", color="success", dismissable=True)

    @app.callback(
        Output("export-modal", "is_open"),
        Input("open-export", "n_clicks"),
        Input("close-export", "n_clicks"),
        State("export-modal", "is_open"),
        State("user-role", "data"),
        prevent_initial_call=True,
    )
    def toggle_export(open_clicks, close_clicks, is_open, role):
        if not auth.has_permission(role, "export"):
            raise dash.exceptions.PreventUpdate
        if open_clicks or close_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("running-tests-table", "data", allow_duplicate=True),
        Output("upcoming-tests-table", "data", allow_duplicate=True),
        Input("refresh-interval", "n_intervals"),
        State("running-tests-table", "data", allow_optional=True),
        State("upcoming-tests-table", "data", allow_optional=True),
        prevent_initial_call=True,
    )
    def refresh_test_tables(_, running_rows, upcoming_rows):
        running_limit = len(running_rows or [])
        upcoming_limit = len(upcoming_rows or [])
        running = data_access.get_running_tests(limit=running_limit)["rows"]
        upcoming = data_access.get_upcoming_tests(limit=upcoming_limit)["rows"]
        return (
            layout_components.running_tests_rows(running),
            layout_components.upcoming_tests_rows(upcoming),
        )

    @app.callback(
        Output("running-tests-table", "data", allow_duplicate=True),
        Input("running-tests-table", "derived_viewport_indices", allow_optional=True),
        State("running-tests-table", "data", allow_optional=True),
        prevent_initial_call=True,
    )
    def paginate_running(viewport, current):
        if not viewport:
            raise dash.exceptions.PreventUpdate
        last = viewport[-1]
        if last < len(current or []) - 1:
            raise dash.exceptions.PreventUpdate
        new_rows = data_access.get_running_tests(
            limit=page_size, offset=len(current or [])
        )["rows"]
        if not new_rows:
            raise dash.exceptions.PreventUpdate
        return (current or []) + layout_components.running_tests_rows(new_rows)

    @app.callback(
        Output("upcoming-tests-table", "data", allow_duplicate=True),
        Input("upcoming-tests-table", "derived_viewport_indices", allow_optional=True),
        State("upcoming-tests-table", "data", allow_optional=True),
        prevent_initial_call=True,
    )
    def paginate_upcoming(viewport, current):
        if not viewport:
            raise dash.exceptions.PreventUpdate
        last = viewport[-1]
        if last < len(current or []) - 1:
            raise dash.exceptions.PreventUpdate
        new_rows = data_access.get_upcoming_tests(
            limit=page_size, offset=len(current or [])
        )["rows"]
        if not new_rows:
            raise dash.exceptions.PreventUpdate
        return (current or []) + layout_components.upcoming_tests_rows(new_rows)

    @app.callback(
        Output("db-status", "children"),
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Output("notification-toast", "header", allow_duplicate=True),
        Output("notification-toast", "icon", allow_duplicate=True),
        Input("refresh-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def refresh_db_status(_):
        connected = data_access.db_connected()
        status = "Connected" if connected else "Not Connected"
        if connected:
            return (
                f"Database: {status}",
                False,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        error = data_access.get_db_error() or "Unable to connect to the database."
        return (
            f"Database: {status} - {error}",
            True,
            error,
            "Database Error",
            "danger",
        )

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
        Output("upload-status", "children", allow_duplicate=True),
        Input("upload-data", "loading_state"),
        prevent_initial_call=True,
    )
    def show_upload_progress(loading_state):
        if loading_state and loading_state.get("is_loading"):
            return "Parsing file..."
        return dash.no_update

    @app.callback(
        Output("upload-form", "style"),
        Output("meta-sample-code", "value"),
        Output("meta-chemistry", "value"),
        Output("meta-notes", "value"),
        Output("upload-info", "data"),
        Output("upload-status", "children", allow_duplicate=True),
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Output("notification-toast", "header", allow_duplicate=True),
        Output("notification-toast", "icon", allow_duplicate=True),
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
                "",
                True,
                f"Parsed {filename}",
                "Success",
                "success",
            )
        except Exception as err:  # pragma: no cover - simple error handling
            return (
                {"display": "none"},
                "",
                "",
                "",
                None,
                "",
                True,
                f"Failed to parse {filename}: {err}",
                "Error",
                "danger",
            )

    @app.callback(
        Output("upload-status", "children", allow_duplicate=True),
        Output("upload-form", "style", allow_duplicate=True),
        Output("uploaded-files-list", "children"),
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Output("notification-toast", "header", allow_duplicate=True),
        Output("notification-toast", "icon", allow_duplicate=True),
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
                return (
                    "",
                    {"display": "none"},
                    items,
                    True,
                    f"Saved {info['filename']}",
                    "Success",
                    "success",
                )
            except Exception as err:  # pragma: no cover - simple error handling
                return (
                    "",
                    dash.no_update,
                    dash.no_update,
                    True,
                    f"Error saving metadata: {err}",
                    "Error",
                    "danger",
                )
        return (
            "",
            {"display": "none"},
            dash.no_update,
            True,
            "Upload canceled",
            "Info",
            "secondary",
        )

    @app.callback(
        Output("import-dir-job", "data"),
        Output("upload-status", "children", allow_duplicate=True),
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Output("notification-toast", "header", allow_duplicate=True),
        Output("notification-toast", "icon", allow_duplicate=True),
        Output("url", "pathname"),
        Input("import-dir-start", "n_clicks"),
        State("import-dir-path", "value"),
        State("user-role", "data"),
        prevent_initial_call=True,
    )
    def start_import_directory(n_clicks, path, role):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not path or not os.path.isdir(path):
            msg = "Invalid directory"
            return (
                dash.no_update,
                msg,
                True,
                msg,
                "Error",
                "danger",
                dash.no_update,
            )
        if not auth.has_permission(role or "", "data-import"):
            msg = "Not authorized"
            return (
                dash.no_update,
                msg,
                True,
                msg,
                "Error",
                "danger",
                dash.no_update,
            )

        def _run() -> object | None:
            from battery_analysis.utils.import_directory import import_directory

            return import_directory(path, include=["*.csv", "*.xlsx", "*.xls", "*.mpt"])

        try:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_run)
            job_id = None
            try:
                job_id = future.result(timeout=0)
            except TimeoutError:
                pass
            return (
                job_id,
                "",
                True,
                f"Started import for {path}",
                "Import Started",
                "success",
                "/import-jobs",
            )
        except Exception as err:  # pragma: no cover - simple error handling
            msg = str(err)
            return (
                dash.no_update,
                msg,
                True,
                msg,
                "Error",
                "danger",
                dash.no_update,
            )

    @app.callback(
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Output("notification-toast", "header", allow_duplicate=True),
        Output("notification-toast", "icon", allow_duplicate=True),
        Input("arbin-dir-start", "n_clicks"),
        State("arbin-dir-path", "value"),
        State("user-role", "data"),
        prevent_initial_call=True,
    )
    def import_arbin_directory(n_clicks, path, role):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if not path or not os.path.isdir(path):
            msg = "Invalid directory"
            return True, msg, "Error", "danger"
        if not auth.has_permission(role or "", "data-import"):
            msg = "Not authorized"
            return True, msg, "Error", "danger"
        try:
            from battery_analysis.utils.import_directory import import_directory

            import_directory(path, include=["*.csv", "*.xls", "*.xlsx"])
            msg = f"Imported Arbin directory {path}"
            return True, msg, "Import Complete", "success"
        except Exception as err:  # pragma: no cover - simple error handling
            msg = str(err)
            return True, msg, "Error", "danger"

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

    # ------------------------------------------------------------------
    # DOE plan builder callbacks
    # ------------------------------------------------------------------

    def _factor_items(factors: dict[str, list[str]]) -> list[html.Li]:
        return [
            html.Li(f"{name}: {', '.join(levels)}") for name, levels in factors.items()
        ]

    def _matrix_items(matrix: list[dict[str, str]]) -> list[html.Li]:
        return [
            html.Li(", ".join(f"{k}={v}" for k, v in row.items())) for row in matrix
        ]

    @app.callback(
        Output(doe_tab.PLAN_NAME, "value"),
        Output(doe_tab.PLAN_STORE, "data"),
        Output(doe_tab.FACTORS_DIV, "children"),
        Output(doe_tab.MATRIX_DIV, "children"),
        Output(doe_tab.FACTOR_SELECT, "options"),
        Input(doe_tab.PLAN_DROPDOWN, "value"),
    )
    def load_plan(plan_name):
        if not plan_name:
            empty = {"factors": {}, "matrix": []}
            return "", empty, [], [], []
        plan = next((p for p in doe_tab._load_plans() if p["name"] == plan_name), None)
        if not plan:
            empty = {"factors": {}, "matrix": []}
            return "", empty, [], [], []
        factors = plan.get("factors", {})
        matrix = plan.get("matrix", [])
        return (
            plan_name,
            {"factors": factors, "matrix": matrix},
            html.Ul(_factor_items(factors)),
            html.Ul(_matrix_items(matrix)),
            [{"label": f, "value": f} for f in factors.keys()],
        )

    @app.callback(
        Output(doe_tab.PLAN_STORE, "data", allow_duplicate=True),
        Output(doe_tab.FACTOR_INPUT, "value"),
        Output(doe_tab.FACTORS_DIV, "children", allow_duplicate=True),
        Output(doe_tab.FACTOR_SELECT, "options", allow_duplicate=True),
        Input(doe_tab.ADD_FACTOR, "n_clicks"),
        State(doe_tab.FACTOR_INPUT, "value"),
        State(doe_tab.PLAN_STORE, "data"),
        prevent_initial_call=True,
    )
    def add_factor(n_clicks, name, plan):
        if not n_clicks or not name:
            raise dash.exceptions.PreventUpdate
        plan = plan or {"factors": {}, "matrix": []}
        plan["factors"].setdefault(name, [])
        factors_children = html.Ul(_factor_items(plan["factors"]))
        options = [{"label": f, "value": f} for f in plan["factors"].keys()]
        return plan, "", factors_children, options

    @app.callback(
        Output(doe_tab.PLAN_STORE, "data", allow_duplicate=True),
        Output(doe_tab.LEVEL_INPUT, "value"),
        Output(doe_tab.FACTORS_DIV, "children", allow_duplicate=True),
        Output(doe_tab.FACTOR_SELECT, "options", allow_duplicate=True),
        Input(doe_tab.ADD_LEVEL, "n_clicks"),
        State(doe_tab.FACTOR_SELECT, "value"),
        State(doe_tab.LEVEL_INPUT, "value"),
        State(doe_tab.PLAN_STORE, "data"),
        prevent_initial_call=True,
    )
    def add_level(n_clicks, factor, level, plan):
        if not n_clicks or not factor or not level:
            raise dash.exceptions.PreventUpdate
        plan = plan or {"factors": {}, "matrix": []}
        levels = plan["factors"].setdefault(factor, [])
        if level not in levels:
            levels.append(level)
        factors_children = html.Ul(_factor_items(plan["factors"]))
        options = [{"label": f, "value": f} for f in plan["factors"].keys()]
        return plan, "", factors_children, options

    @app.callback(
        Output(doe_tab.PLAN_STORE, "data", allow_duplicate=True),
        Output(doe_tab.MATRIX_INPUT, "value"),
        Output(doe_tab.MATRIX_DIV, "children", allow_duplicate=True),
        Input(doe_tab.ADD_ROW, "n_clicks"),
        State(doe_tab.MATRIX_INPUT, "value"),
        State(doe_tab.PLAN_STORE, "data"),
        prevent_initial_call=True,
    )
    def add_row(n_clicks, row_text, plan):
        if not n_clicks or not row_text:
            raise dash.exceptions.PreventUpdate
        plan = plan or {"factors": {}, "matrix": []}
        try:
            combo = json.loads(row_text)
        except Exception:
            return (
                plan,
                row_text,
                html.Div("Invalid row format", className="text-danger"),
            )
        plan["matrix"].append(combo)
        matrix_children = html.Ul(_matrix_items(plan["matrix"]))
        return plan, "", matrix_children

    @app.callback(
        Output(doe_tab.PLAN_DROPDOWN, "options", allow_duplicate=True),
        Output(doe_tab.FEEDBACK_DIV, "children"),
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Output("notification-toast", "header", allow_duplicate=True),
        Output("notification-toast", "icon", allow_duplicate=True),
        Input(doe_tab.SAVE_PLAN, "n_clicks"),
        State(doe_tab.PLAN_NAME, "value"),
        State(doe_tab.PLAN_STORE, "data"),
        prevent_initial_call=True,
    )
    def save_plan_callback(n_clicks, name, plan):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        errors: list[str] = []
        if not name:
            errors.append("Plan name is required.")
        if not plan or not plan.get("factors"):
            errors.append("At least one factor is required.")
        if not plan or not plan.get("matrix"):
            errors.append("At least one matrix row is required.")
        if errors:
            return (
                dash.no_update,
                html.Ul([html.Li(e) for e in errors]),
                True,
                "Invalid plan",
                "Error",
                "danger",
            )
        try:
            save_plan(name, plan.get("factors", {}), plan.get("matrix", []))
        except Exception as err:
            msg = f"Failed to save plan: {err}"
            return dash.no_update, html.Div(msg), True, msg, "Error", "danger"
        options = [
            {"label": p["name"], "value": p["name"]} for p in doe_tab._load_plans()
        ]
        if name not in [opt["value"] for opt in options]:
            options.append({"label": name, "value": name})
        return options, "", True, f"Saved plan {name}", "Success", "success"

    trait_filter_tab.register_callbacks(app)
    comparison_tab.register_callbacks(app)
    advanced_analysis_tab.register_callbacks(app)
    ad_hoc_analysis_tab.register_callbacks(app)
    cycle_detail_tab.register_callbacks(app)
    eis_tab.register_callbacks(app)
    document_flow_tab.register_callbacks(app)
    missing_data_tab.register_callbacks(app)
    doe_tab.register_callbacks(app)
    import_jobs_tab.register_callbacks(app)
    refactor_jobs_tab.register_callbacks(app)
    import_stats_tab.register_callbacks(app)
    watcher_tab.register_callbacks(app)
    raw_files_tab.register_callbacks(app)
    similar_samples_tab.register_callbacks(app)

    return app


if __name__ == "__main__":  # pragma: no cover - manual execution
    create_app().run(debug=True)
