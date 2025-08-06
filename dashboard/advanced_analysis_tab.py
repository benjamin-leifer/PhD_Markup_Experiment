"""Dash components for advanced electrochemical analysis."""

from __future__ import annotations

from typing import List, Dict

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import Input, Output, State
from plotly import graph_objs as go

try:  # pragma: no cover - optional dependencies
    from battery_analysis import advanced_analysis, MISSING_ADVANCED_PACKAGES
except Exception:  # pragma: no cover - gracefully handle missing packages
    advanced_analysis = None  # type: ignore
    MISSING_ADVANCED_PACKAGES = ["advanced analysis"]

HAS_ADVANCED = not MISSING_ADVANCED_PACKAGES

SAMPLE_DROPDOWN = "aa-sample"
TEST_DROPDOWN = "aa-test"
ANALYSIS_RADIO = "aa-analysis"
CYCLE_INPUT = "aa-cycle"
SMOOTH_CHECK = "aa-smooth"
WINDOW_INPUT = "aa-window"
RUN_BUTTON = "aa-run"
RESULT_GRAPH = "aa-graph"
RESULT_TEXT = "aa-results"


def _get_sample_options() -> List[Dict[str, str]]:
    """Return dropdown options for available samples."""
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis import models

        samples = models.Sample.objects.only("name")  # type: ignore[attr-defined]
        return [{"label": s.name, "value": str(s.id)} for s in samples]
    except Exception:
        return [{"label": "Sample_001", "value": "sample1"}]


def _get_test_options(sample_id: str) -> List[Dict[str, str]]:
    """Return dropdown options for tests belonging to ``sample_id``."""
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis import models

        tests = models.TestResult.objects(sample=sample_id).only("name")  # type: ignore[attr-defined]
        return [{"label": t.name, "value": str(t.id)} for t in tests]
    except Exception:
        return [{"label": "Test_A", "value": "testA"}]


def layout() -> html.Div:
    """Return the Dash layout for the advanced analysis tab."""
    sample_opts = _get_sample_options()
    layout_children = [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        options=sample_opts,
                        id=SAMPLE_DROPDOWN,
                        placeholder="Sample",
                        clearable=True,
                    ),
                    width=3,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id=TEST_DROPDOWN,
                        placeholder="Test",
                        clearable=True,
                    ),
                    width=3,
                ),
                dbc.Col(
                    dcc.RadioItems(
                        options=[
                            {"label": "dQ/dV", "value": "dqdv"},
                            {"label": "Capacity Fade", "value": "fade"},
                        ],
                        value="dqdv",
                        id=ANALYSIS_RADIO,
                        inline=True,
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dcc.Input(
                        id=CYCLE_INPUT,
                        type="number",
                        placeholder="Cycle",
                        value=1,
                        style={"width": "6em"},
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Checklist(
                        options=[{"label": "Smooth", "value": "smooth"}],
                        value=["smooth"],
                        id=SMOOTH_CHECK,
                        switch=True,
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dcc.Input(
                        id=WINDOW_INPUT,
                        type="number",
                        placeholder="Window",
                        value=11,
                        style={"width": "6em"},
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button("Run Analysis", id=RUN_BUTTON, color="primary", disabled=not HAS_ADVANCED),
                    width="auto",
                ),
                dbc.Col(dbc.Tooltip(
                    "Missing packages: " + ", ".join(MISSING_ADVANCED_PACKAGES),
                    target=RUN_BUTTON,
                ), width="auto") if not HAS_ADVANCED else html.Div(),
            ],
            className="mb-2",
        ),
        dbc.Row([dbc.Col(dcc.Graph(id=RESULT_GRAPH))]),
        dbc.Row([dbc.Col(html.Div(id=RESULT_TEXT))]),
    ]
    return html.Div(layout_children)


def register_callbacks(app: dash.Dash) -> None:
    """Register Dash callbacks for advanced analysis."""

    @app.callback(Output(TEST_DROPDOWN, "options"), Input(SAMPLE_DROPDOWN, "value"))
    def _update_tests(sample_id):
        if not sample_id:
            return []
        return _get_test_options(sample_id)

    @app.callback(
        Output(RESULT_GRAPH, "figure"),
        Output(RESULT_TEXT, "children"),
        Input(RUN_BUTTON, "n_clicks"),
        State(TEST_DROPDOWN, "value"),
        State(CYCLE_INPUT, "value"),
        State(SMOOTH_CHECK, "value"),
        State(WINDOW_INPUT, "value"),
        prevent_initial_call=True,
    )
    def _run_analysis(n_clicks, test_id, cycle, smooth_vals, window):
        if not HAS_ADVANCED or not advanced_analysis:
            raise dash.exceptions.PreventUpdate
        smooth = smooth_vals and "smooth" in smooth_vals
        try:
            voltage, capacity = advanced_analysis.get_voltage_capacity_data(test_id, cycle)
            v_mid, dqdv = advanced_analysis.compute_dqdv(
                capacity,
                voltage,
                smooth=smooth,
                window_size=window or 11,
                polyorder=3,
            )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=v_mid, y=dqdv, mode="lines", name="dQ/dV")
            )
            fig.update_layout(
                xaxis_title="Voltage (V)",
                yaxis_title="dQ/dV",
                template="plotly_white",
            )
            text = f"Computed dQ/dV for cycle {cycle}"
            return fig, text
        except Exception as e:  # pragma: no cover - runtime errors
            fig = go.Figure()
            return fig, f"Error: {e}"
