"""Comparison tab with Plotly overlays and export."""

from __future__ import annotations

from typing import List, Dict, Any, Optional

import io

import numpy as np
import pandas as pd

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State
import plotly.graph_objs as go

try:
    from .trait_filter_tab import get_sample_names
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    get_sample_names = importlib.import_module("trait_filter_tab").get_sample_names

# Component IDs
SAMPLE_DROPDOWN = "compare-samples"
METRIC_RADIO = "compare-metric"
GRAPH = "compare-graph"
EXPORT_BUTTON = "compare-export-btn"
EXPORT_DOWNLOAD = "compare-export-download"
EXPORT_IMG_BUTTON = "compare-export-img-btn"
EXPORT_IMG_DOWNLOAD = "compare-export-img-download"


def _get_sample_data(sample: str) -> Dict[str, np.ndarray]:
    """Return cycle data for ``sample``.

    The function tries to pull real cycle data from the ``battery_analysis``
    package. If that fails (e.g. database not available) deterministic demo data
    is generated instead so that the dashboard remains functional.
    """

    try:  # pragma: no cover - depends on battery_analysis and database
        from battery_analysis import analysis  # type: ignore  # noqa: F401
        from battery_analysis.models import Sample  # type: ignore

        s = Sample.objects.get(name=sample)  # type: ignore[attr-defined]
        cycles = np.arange(1, getattr(s, "cycle_count", 0) or 1 + 1)
        capacity = np.array(getattr(s, "capacity_trace", []))
        ce = np.array(getattr(s, "ce_trace", []))
        impedance = np.array(getattr(s, "impedance_trace", []))
        if not len(cycles):
            raise ValueError("no cycle data")
        if len(capacity) != len(cycles):
            capacity = np.full_like(cycles, np.nan, dtype=float)
        if len(ce) != len(cycles):
            ce = np.full_like(cycles, np.nan, dtype=float)
        if len(impedance) != len(cycles):
            impedance = np.full_like(cycles, np.nan, dtype=float)
        return {"cycle": cycles, "capacity": capacity, "ce": ce, "impedance": impedance}
    except Exception:
        rng = np.random.default_rng(abs(hash(sample)) % (2**32))
        cycles = np.arange(1, 51)
        capacity = 1.0 + 0.1 * rng.standard_normal(len(cycles))
        ce = 0.98 + 0.01 * rng.standard_normal(len(cycles))
        impedance = 100 + 5 * rng.standard_normal(len(cycles))
        return {"cycle": cycles, "capacity": capacity, "ce": ce, "impedance": impedance}


def layout() -> html.Div:
    """Return the layout for the comparison tab."""
    sample_opts = [{"label": s, "value": s} for s in get_sample_names()]
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            options=sample_opts,
                            id=SAMPLE_DROPDOWN,
                            multi=True,
                            placeholder="Select samples",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dcc.RadioItems(
                            options=[
                                {"label": "Capacity", "value": "capacity"},
                                {
                                    "label": "Normalized Capacity",
                                    "value": "norm_capacity",
                                },
                                {"label": "Coulombic Efficiency", "value": "ce"},
                                {"label": "Impedance", "value": "impedance"},
                            ],
                            value="capacity",
                            id=METRIC_RADIO,
                            inline=True,
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button("Export Data", id=EXPORT_BUTTON, color="secondary"),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Export Plot", id=EXPORT_IMG_BUTTON, color="secondary"
                        ),
                        width="auto",
                    ),
                    dcc.Download(id=EXPORT_DOWNLOAD),
                    dcc.Download(id=EXPORT_IMG_DOWNLOAD),
                ],
                className="mb-3",
            ),
            dbc.Row([dbc.Col(dcc.Graph(id=GRAPH))]),
        ]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register Dash callbacks for the comparison tab."""

    @app.callback(
        Output(GRAPH, "figure"),
        Input(SAMPLE_DROPDOWN, "value"),
        Input(METRIC_RADIO, "value"),
    )
    def _update_graph(samples: Optional[List[str]], metric: str):
        fig = go.Figure()
        if not samples:
            fig.update_layout(template="plotly_white", xaxis_title="Cycle")
            return fig
        for name in samples:
            data = _get_sample_data(name)
            y = data[metric] if metric in data else data["capacity"]
            if metric == "norm_capacity":
                y = data["capacity"] / data["capacity"][0]
            fig.add_trace(go.Scatter(x=data["cycle"], y=y, mode="lines", name=name))
        y_labels = {
            "capacity": "Capacity (mAh)",
            "norm_capacity": "Normalized Capacity",
            "ce": "Coulombic Efficiency",
            "impedance": "Impedance (Ohm)",
        }
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Cycle",
            yaxis_title=y_labels.get(metric, metric),
        )
        return fig

    @app.callback(
        Output(EXPORT_DOWNLOAD, "data"),
        Input(EXPORT_BUTTON, "n_clicks"),
        State(SAMPLE_DROPDOWN, "value"),
        State(METRIC_RADIO, "value"),
        prevent_initial_call=True,
    )
    def _export_data(n_clicks, samples, metric):
        if not samples:
            return dash.no_update
        records: List[Dict[str, Any]] = []
        for name in samples:
            data = _get_sample_data(name)
            y = data[metric] if metric in data else data["capacity"]
            if metric == "norm_capacity":
                y = data["capacity"] / data["capacity"][0]
            for cycle, val in zip(data["cycle"], y):
                records.append(
                    {"sample": name, "cycle": int(cycle), metric: float(val)}
                )
        df = pd.DataFrame(records)
        csv_str = df.to_csv(index=False)
        return dcc.send_string(csv_str, "comparison_data.csv")

    @app.callback(
        Output(EXPORT_IMG_DOWNLOAD, "data"),
        Input(EXPORT_IMG_BUTTON, "n_clicks"),
        State(GRAPH, "figure"),
        prevent_initial_call=True,
    )
    def _export_plot(n_clicks, fig_dict):
        if not fig_dict:
            return dash.no_update
        fig = go.Figure(fig_dict)
        buffer = io.BytesIO()
        fig.write_image(buffer, format="png")
        buffer.seek(0)
        return dcc.send_bytes(buffer.getvalue(), "comparison_plot.png")
