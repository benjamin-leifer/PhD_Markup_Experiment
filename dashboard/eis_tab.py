"""Dash components for Electrochemical Impedance Spectroscopy (EIS) analysis."""

from __future__ import annotations

import base64
import tempfile
from typing import Optional

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import Input, Output, State
from plotly import graph_objs as go

try:  # pragma: no cover - optional dependency
    from battery_analysis import eis
    HAS_EIS = True
except Exception:  # pragma: no cover - handle missing dependency
    eis = None  # type: ignore
    HAS_EIS = False

UPLOAD_COMPONENT = "eis-upload"
GRAPH_COMPONENT = "eis-graph"


def layout() -> html.Div:
    """Return the Dash layout for the EIS analysis tab."""
    upload = dcc.Upload(
        id=UPLOAD_COMPONENT,
        children=dbc.Button("Upload EIS Data", color="primary"),
        multiple=False,
        disabled=not HAS_EIS,
    )
    tooltip = (
        dbc.Tooltip("EIS functionality not available", target=UPLOAD_COMPONENT)
        if not HAS_EIS
        else None
    )
    return html.Div([
        html.Div([upload, tooltip] if tooltip else [upload], className="mb-2"),
        dcc.Graph(id=GRAPH_COMPONENT),
    ])


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for EIS analysis."""

    @app.callback(
        Output(GRAPH_COMPONENT, "figure"),
        Input(UPLOAD_COMPONENT, "contents"),
        State(UPLOAD_COMPONENT, "filename"),
        prevent_initial_call=True,
    )
    def _update_graph(contents: Optional[str], _filename: Optional[str]):
        if not contents or not HAS_EIS or eis is None:
            raise dash.exceptions.PreventUpdate
        _, content_string = contents.split(",")
        data = base64.b64decode(content_string)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            parsed = eis.parse_eis_file(tmp_path)
        finally:
            pass
        fig = go.Figure()
        z_real = parsed["Z_real"]
        z_imag = parsed["Z_imag"]
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
        return fig
