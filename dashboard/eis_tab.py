"""Dash components for Electrochemical Impedance Spectroscopy (EIS) analysis."""

from __future__ import annotations

import base64
import json
import logging
import tempfile
from multiprocessing import Process
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from bson import ObjectId
from bson.errors import InvalidId
from dash import Input, Output, State, dcc, html
from plotly.subplots import make_subplots

from dashboard.data_access import db_connected, get_db_error
from Mongodb_implementation import find_samples, find_test_results

# mypy: ignore-errors

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from battery_analysis import eis

    HAS_EIS = True
    HAS_IMPEDANCE = getattr(eis, "HAS_IMPEDANCE", False)
except Exception:  # pragma: no cover - handle missing dependency
    eis = None  # type: ignore
    HAS_EIS = False
    HAS_IMPEDANCE = False

# Component IDs
DATA_SOURCE = "eis-source"
FILE_UPLOAD = "eis-file-upload"
SAMPLE_DROPDOWN = "eis-sample"
TEST_DROPDOWN = "eis-test"
LOAD_BUTTON = "eis-load"
DATA_STORE = "eis-data"
FMIN_INPUT = "eis-fmin"
FMAX_INPUT = "eis-fmax"
INDUCTIVE_CHECK = "eis-inductive"
PROCESS_BUTTON = "eis-process"
PLOT_TYPE = "eis-plot-type"
HIGHLIGHT_CHECK = "eis-highlight"
CIRCUIT_DROPDOWN = "eis-circuit"
CUSTOM_CIRCUIT_INPUT = "eis-custom-circuit"
USE_CUSTOM_CHECK = "eis-use-custom"
FIT_BUTTON = "eis-fit"
GRAPH_COMPONENT = "eis-graph"
MPL_POPOUT_BUTTON = "eis-mpl-popout-btn"


def _get_sample_options() -> List[Dict[str, str]]:
    if not db_connected():
        reason = get_db_error() or "unknown reason"
        logger.error("Database not connected: %s; using demo data", reason)
        return [{"label": "Sample_001", "value": "sample1"}]
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis import models

        sample_manager = getattr(models.Sample, "objects", None)
        if sample_manager:
            samples = sample_manager.only("name")  # type: ignore[attr-defined]
            opts = [{"label": s.name, "value": str(s.id)} for s in samples]
        else:
            samples = find_samples()
            opts = [
                {
                    "label": s.get("name", ""),
                    "value": str(s.get("_id", s.get("name", ""))),
                }
                for s in samples
                if s.get("name")
            ]
        if not opts:
            logger.warning("No sample options found; using demo data")
            return [{"label": "Sample_001", "value": "sample1"}]
        return opts
    except Exception:
        logger.exception("Failed to load sample options")
        return [{"label": "Sample_001", "value": "sample1"}]


def _get_test_options(sample_id: str) -> List[Dict[str, str]]:
    if not sample_id:
        return []
    if not db_connected():
        reason = get_db_error() or "unknown reason"
        logger.error(
            "Database not connected: %s; using demo data for EIS tests", reason
        )
        return [{"label": "EIS_Test", "value": str(ObjectId())}]
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis import models

        sample_manager = getattr(models.Sample, "objects", None)
        test_manager = getattr(models.TestResult, "objects", None)
        if sample_manager and test_manager:
            tests = test_manager(sample=sample_id, test_type="EIS").only("name")
            return [{"label": t.name, "value": str(t.id)} for t in tests]

        try:
            sample_oid = ObjectId(sample_id)
        except InvalidId:
            sample_oid = sample_id
        tests = find_test_results({"sample": sample_oid, "test_type": "EIS"})
        opts = [
            {"label": t.get("name", ""), "value": str(t.get("_id", ""))}
            for t in tests
            if t.get("name")
        ]
        if not opts:
            logger.warning(
                "No EIS tests found for sample %s; using demo data", sample_id
            )
            return [{"label": "EIS_Test", "value": str(ObjectId())}]
        return opts
    except Exception:
        logger.exception("Failed to load EIS tests for sample %s", sample_id)
        return [{"label": "EIS_Test", "value": str(ObjectId())}]


def layout() -> html.Div:
    sample_opts = _get_sample_options()
    data_source = dbc.RadioItems(
        options=[
            {"label": "File", "value": "file"},
            {"label": "Database", "value": "database"},
        ],
        value="file",
        id=DATA_SOURCE,
        inline=True,
    )

    upload = dcc.Upload(
        id=FILE_UPLOAD,
        children=dbc.Button("Upload EIS Data", color="primary"),
        multiple=False,
        disabled=not HAS_EIS,
    )
    upload_tooltip = (
        dbc.Tooltip("EIS functionality not available", target=FILE_UPLOAD)
        if not HAS_EIS
        else None
    )
    file_div = html.Div(
        [upload, upload_tooltip] if upload_tooltip else [upload], id="eis-file-div"
    )

    db_div = html.Div(
        [
            dcc.Dropdown(
                id=SAMPLE_DROPDOWN,
                options=sample_opts,
                placeholder="Sample",
                clearable=True,
            ),
            dcc.Dropdown(id=TEST_DROPDOWN, placeholder="EIS Test", clearable=True),
            dbc.Button(
                "Load Data", id=LOAD_BUTTON, color="primary", disabled=not HAS_EIS
            ),
        ],
        id="eis-db-div",
        style={"display": "none"},
    )

    preprocess_div = html.Div(
        [
            dcc.Input(
                id=FMIN_INPUT,
                type="number",
                placeholder="Min Hz",
                style={"width": "8em"},
            ),
            dcc.Input(
                id=FMAX_INPUT,
                type="number",
                placeholder="Max Hz",
                style={"width": "8em"},
            ),
            dbc.Checklist(
                options=[{"label": "Filter Inductive", "value": "ind"}],
                value=["ind"],
                id=INDUCTIVE_CHECK,
                switch=True,
            ),
            dbc.Button(
                "Process Data",
                id=PROCESS_BUTTON,
                color="secondary",
                disabled=not HAS_EIS,
            ),
            dcc.Loading(html.Div(id="eis-processing")),
        ],
        className="mb-2",
    )

    plot_opts = html.Div(
        [
            dcc.RadioItems(
                options=[
                    {"label": "Nyquist", "value": "nyquist"},
                    {"label": "Bode", "value": "bode"},
                    {"label": "DRT", "value": "drt", "disabled": not HAS_IMPEDANCE},
                ],
                value="nyquist",
                id=PLOT_TYPE,
                inline=True,
            ),
            dbc.Checklist(
                options=[{"label": "Highlight Frequencies", "value": "h"}],
                value=[],
                id=HIGHLIGHT_CHECK,
                switch=True,
            ),
            (
                dbc.Tooltip("impedance.py not installed", target=PLOT_TYPE)
                if not HAS_IMPEDANCE
                else None
            ),
        ],
        className="mb-2",
    )

    circuit_div = html.Div(
        [
            dcc.Dropdown(
                id=CIRCUIT_DROPDOWN,
                options=[
                    {"label": c, "value": c}
                    for c in [
                        "R0",
                        "R0-p(R1,C1)",
                        "R0-p(R1,CPE1)",
                        "R0-p(R1,C1)-W2",
                        "R0-p(R1,CPE1)-W2",
                    ]
                ],
                value="R0-p(R1,CPE1)-W2",
                disabled=not HAS_IMPEDANCE,
            ),
            dcc.Input(id=CUSTOM_CIRCUIT_INPUT, placeholder="Custom Circuit"),
            dbc.Checklist(
                options=[
                    {
                        "label": "Use Custom",
                        "value": "use",
                        "disabled": not HAS_IMPEDANCE,
                    }
                ],
                value=[],
                id=USE_CUSTOM_CHECK,
                switch=True,
            ),
            dbc.Button(
                "Fit Circuit",
                id=FIT_BUTTON,
                color="secondary",
                disabled=not HAS_IMPEDANCE,
            ),
            (
                dbc.Tooltip(
                    "impedance.py not installed",
                    target=FIT_BUTTON,
                )
                if not HAS_IMPEDANCE
                else None
            ),
        ],
        className="mb-2",
    )

    return html.Div(
        [
            data_source,
            file_div,
            db_div,
            preprocess_div,
            plot_opts,
            circuit_div,
            dbc.Button(
                "Open in Matplotlib",
                id=MPL_POPOUT_BUTTON,
                color="secondary",
            ),
            dcc.Store(id=DATA_STORE),
            dcc.Loading(dcc.Graph(id=GRAPH_COMPONENT)),
        ]
    )


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("eis-file-div", "style"),
        Output("eis-db-div", "style"),
        Input(DATA_SOURCE, "value"),
    )
    def _toggle_source(src):
        file_style = {"display": "block" if src == "file" else "none"}
        db_style = {"display": "block" if src == "database" else "none"}
        return file_style, db_style

    @app.callback(Output(TEST_DROPDOWN, "options"), Input(SAMPLE_DROPDOWN, "value"))
    def _update_tests(sample_id):
        if not sample_id:
            return []
        return _get_test_options(sample_id)

    @app.callback(
        Output(DATA_STORE, "data"),
        Input(FILE_UPLOAD, "contents"),
        State(FILE_UPLOAD, "filename"),
        prevent_initial_call=True,
    )
    def _upload_file(contents, filename):
        if not contents or not HAS_EIS or eis is None:
            raise dash.exceptions.PreventUpdate
        _, content_string = contents.split(",")
        data = base64.b64decode(content_string)
        with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        parsed = eis.parse_eis_file(tmp_path)
        return parsed

    @app.callback(
        Output(DATA_STORE, "data", allow_duplicate=True),
        Input(LOAD_BUTTON, "n_clicks"),
        State(TEST_DROPDOWN, "value"),
        prevent_initial_call=True,
    )
    def _load_from_db(n_clicks, test_id):
        if not n_clicks or not test_id or not HAS_EIS or eis is None:
            raise dash.exceptions.PreventUpdate
        data = eis.get_eis_data(test_id)
        return data

    @app.callback(
        Output(DATA_STORE, "data", allow_duplicate=True),
        Output("eis-processing", "children"),
        Input(PROCESS_BUTTON, "n_clicks"),
        State(DATA_STORE, "data"),
        State(FMIN_INPUT, "value"),
        State(FMAX_INPUT, "value"),
        State(INDUCTIVE_CHECK, "value"),
        running=[(Output(PROCESS_BUTTON, "disabled"), True, False)],
        prevent_initial_call=True,
        background=True,
    )
    def _process_data(n_clicks, data, fmin, fmax, inductive_vals):
        if not n_clicks or data is None or eis is None:
            raise dash.exceptions.PreventUpdate
        inductive = bool(inductive_vals and "ind" in inductive_vals)
        freq, z_real, z_imag = eis.preprocess_eis_data(
            np.array(data["frequency"]),
            np.array(data["Z_real"]),
            np.array(data["Z_imag"]),
            f_min=fmin,
            f_max=fmax,
            inductive_filter=inductive,
        )
        data.update(
            {
                "frequency": freq,
                "Z_real": z_real,
                "Z_imag": z_imag,
            }
        )
        return data, ""

    @app.callback(
        Output(GRAPH_COMPONENT, "figure"),
        Input(DATA_STORE, "data"),
        Input(PLOT_TYPE, "value"),
        prevent_initial_call=True,
        background=True,
    )
    def _update_graph(data, plot_type):
        if not data or not HAS_EIS:
            raise dash.exceptions.PreventUpdate
        freq = np.array(data["frequency"])
        z_real = np.array(data["Z_real"])
        z_imag = np.array(data["Z_imag"])
        fig = go.Figure()
        if plot_type == "nyquist":
            fig.add_trace(
                go.Scatter(x=z_real, y=-z_imag, mode="lines+markers", name="Nyquist")
            )
            fig.update_layout(
                xaxis_title="Z_real (Ω)",
                yaxis_title="-Z_imag (Ω)",
                template="plotly_white",
                yaxis_scaleanchor="x",
                yaxis_scaleratio=1,
            )
        elif plot_type == "bode":
            mag = np.sqrt(z_real**2 + z_imag**2)
            phase = np.arctan2(-z_imag, z_real) * 180 / np.pi
            fig = make_bode_plot(freq, mag, phase)
        elif plot_type == "drt" and HAS_IMPEDANCE:
            from impedance.models.circuits.drt import calculate_drt

            z = z_real + 1j * z_imag
            _, gamma, tau, _, _ = calculate_drt(freq, z)
            fig.add_trace(go.Scatter(x=tau, y=gamma, mode="lines"))
            fig.update_layout(
                xaxis_type="log",
                yaxis_title="γ (Ω/ln(s))",
                xaxis_title="Time constant (s)",
                template="plotly_white",
            )
        else:
            fig.update_layout(template="plotly_white")
        return fig

    @app.callback(
        Output(MPL_POPOUT_BUTTON, "n_clicks"),
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Output("notification-toast", "header", allow_duplicate=True),
        Output("notification-toast", "icon", allow_duplicate=True),
        Input(MPL_POPOUT_BUTTON, "n_clicks"),
        State(GRAPH_COMPONENT, "figure"),
        prevent_initial_call=True,
    )
    def _popout_matplotlib(n_clicks, fig_dict):
        import importlib.util

        if not n_clicks or not fig_dict:
            raise dash.exceptions.PreventUpdate

        backend = None
        for module, candidate in (
            ("PyQt5", "Qt5Agg"),
            ("PySide2", "Qt5Agg"),
            ("PyQt6", "QtAgg"),
            ("PySide6", "QtAgg"),
        ):
            if importlib.util.find_spec(module):
                backend = candidate
                break

        if backend is None:
            return (
                0,
                True,
                "Qt bindings not available; install PyQt5/PyQt6/PySide2/PySide6.",
                "Error",
                "danger",
            )

        try:
            proc = Process(
                target=_render_matplotlib, args=(fig_dict, backend), daemon=True
            )
            proc.start()
            if not proc.is_alive():
                raise OSError("Matplotlib process failed to start")
        except OSError:
            return (
                0,
                True,
                "Failed to launch Matplotlib pop-out.",
                "Error",
                "danger",
            )
        return (0, dash.no_update, dash.no_update, dash.no_update, dash.no_update)


def _render_matplotlib(fig_dict, backend: str | None = None):
    import json
    import matplotlib
    import plotly.graph_objects as go
    from plotly.utils import PlotlyJSONDecoder

    if backend:
        try:
            matplotlib.use(backend, force=True)
        except Exception:
            logging.exception("Failed to set Matplotlib backend to %s", backend)
    import matplotlib.pyplot as plt

    fig = go.Figure(json.loads(json.dumps(fig_dict), cls=PlotlyJSONDecoder))
    try:
        plt.figure()
        for tr in fig.data:
            if isinstance(tr, go.Scatter):
                plt.plot(tr.x, tr.y, label=tr.name)
        if any(tr.name for tr in fig.data):
            plt.legend()
        plt.show()
    except Exception:
        logging.exception("Matplotlib pop-out failed")
        raise SystemExit


def make_bode_plot(freq, mag, phase):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scatter(x=freq, y=mag, mode="lines+markers", name="|Z|"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=freq, y=phase, mode="lines+markers", name="Phase"), row=2, col=1
    )
    fig.update_xaxes(type="log", title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="|Z| (Ω)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (°)", row=2, col=1)
    fig.update_layout(template="plotly_white")
    return fig
