"""Dash components for Electrochemical Impedance Spectroscopy (EIS) analysis."""

from __future__ import annotations

import base64
import tempfile
from typing import List, Dict

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import Input, Output, State
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

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


def _get_sample_options() -> List[Dict[str, str]]:
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis import models

        samples = models.Sample.objects.only("name")  # type: ignore[attr-defined]
        return [{"label": s.name, "value": str(s.id)} for s in samples]
    except Exception:
        return [{"label": "Sample_001", "value": "sample1"}]


def _get_test_options(sample_id: str) -> List[Dict[str, str]]:
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis import models

        tests = models.TestResult.objects(sample=sample_id, test_type="EIS").only(
            "name"
        )  # type: ignore[attr-defined]
        return [{"label": t.name, "value": str(t.id)} for t in tests]
    except Exception:
        return [{"label": "EIS_Test", "value": "test1"}]


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
                options=[{"label": "Use Custom", "value": "use"}],
                value=[],
                id=USE_CUSTOM_CHECK,
                switch=True,
                disabled=not HAS_IMPEDANCE,
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
            dcc.Store(id=DATA_STORE),
            dcc.Graph(id=GRAPH_COMPONENT),
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
        for key in ["frequency", "Z_real", "Z_imag"]:
            parsed[key] = np.array(parsed[key]).tolist()
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
        for key in ["frequency", "Z_real", "Z_imag"]:
            data[key] = np.array(data[key]).tolist()
        return data

    @app.callback(
        Output(DATA_STORE, "data", allow_duplicate=True),
        Input(PROCESS_BUTTON, "n_clicks"),
        State(DATA_STORE, "data"),
        State(FMIN_INPUT, "value"),
        State(FMAX_INPUT, "value"),
        State(INDUCTIVE_CHECK, "value"),
        prevent_initial_call=True,
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
                "frequency": freq.tolist(),
                "Z_real": z_real.tolist(),
                "Z_imag": z_imag.tolist(),
            }
        )
        return data

    @app.callback(
        Output(GRAPH_COMPONENT, "figure"),
        Input(DATA_STORE, "data"),
        Input(PLOT_TYPE, "value"),
        prevent_initial_call=True,
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
