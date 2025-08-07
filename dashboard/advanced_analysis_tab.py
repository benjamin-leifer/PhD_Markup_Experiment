"""Dash components for advanced electrochemical analysis."""

from __future__ import annotations

from typing import List, Dict

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import Input, Output, State
from plotly import graph_objs as go
import numpy as np
import base64
import tempfile
import io

try:  # pragma: no cover - optional dependencies
    from battery_analysis import advanced_analysis, MISSING_ADVANCED_PACKAGES
except Exception:  # pragma: no cover - gracefully handle missing packages
    advanced_analysis = None  # type: ignore
    MISSING_ADVANCED_PACKAGES = ["advanced analysis"]

HAS_ADVANCED = not MISSING_ADVANCED_PACKAGES

SAMPLE_DROPDOWN = "aa-sample"
TEST_DROPDOWN = "aa-test"
ANALYSIS_RADIO = "aa-analysis"

# dQ/dV options
CYCLE_INPUT = "aa-cycle"
SMOOTH_CHECK = "aa-smooth"
WINDOW_INPUT = "aa-window"

# Capacity fade options
EOL_INPUT = "aa-eol"
FADE_MODELS = "aa-fade-models"

# Anomaly detection options
METRIC_RADIO = "aa-metric"
THRESHOLD_INPUT = "aa-threshold"

# Energy analysis options
WEIGHT_INPUT = "aa-weight"
VOLUME_INPUT = "aa-volume"

# Clustering options
CLUSTER_METRIC = "aa-cluster-metric"
METHOD_RADIO = "aa-method"
N_CLUSTERS = "aa-n-clusters"

# Josh request options
JOSH_UPLOAD = "aa-josh-upload"
JOSH_SHEET = "aa-josh-sheet"
JOSH_MASS = "aa-josh-mass"

RUN_BUTTON = "aa-run"
EXPORT_BUTTON = "aa-export-btn"
EXPORT_DOWNLOAD = "aa-export-download"
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
                            {"label": "Anomalies", "value": "anomalies"},
                            {"label": "Energy", "value": "energy"},
                            {"label": "Clustering", "value": "clustering"},
                            {"label": "Josh dQ/dV", "value": "Josh_request_Dq_dv"},
                        ],
                        value="dqdv",
                        id=ANALYSIS_RADIO,
                        inline=True,
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button(
                        "Run Analysis",
                        id=RUN_BUTTON,
                        color="primary",
                        disabled=not HAS_ADVANCED,
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button(
                        "Export Plot", id=EXPORT_BUTTON, color="secondary"
                    ),
                    width="auto",
                ),
                (
                    dbc.Col(
                        dbc.Tooltip(
                            "Missing packages: " + ", ".join(MISSING_ADVANCED_PACKAGES),
                            target=RUN_BUTTON,
                        ),
                        width="auto",
                    )
                    if not HAS_ADVANCED
                    else html.Div()
                ),
                dcc.Download(id=EXPORT_DOWNLOAD),
            ],
            className="mb-2",
        ),
        html.Div(
            [
                # dQ/dV options
                html.Div(
                    [
                        dcc.Input(
                            id=CYCLE_INPUT,
                            type="number",
                            value=1,
                            placeholder="Cycle",
                            style={"width": "6em"},
                        ),
                        dbc.Checklist(
                            options=[{"label": "Smooth", "value": "smooth"}],
                            value=["smooth"],
                            id=SMOOTH_CHECK,
                            switch=True,
                        ),
                        dcc.Input(
                            id=WINDOW_INPUT,
                            type="number",
                            value=11,
                            placeholder="Window",
                            style={"width": "6em"},
                        ),
                    ],
                    id="dqdv-options",
                    className="mb-2",
                ),
                # Capacity fade options
                html.Div(
                    [
                        dcc.Input(
                            id=EOL_INPUT,
                            type="number",
                            value=80,
                            placeholder="EOL %",
                            style={"width": "6em"},
                        ),
                        dbc.Checklist(
                            options=[
                                {"label": "Linear", "value": "linear"},
                                {"label": "Power", "value": "power"},
                                {"label": "Exponential", "value": "exponential"},
                            ],
                            value=["linear", "power", "exponential"],
                            id=FADE_MODELS,
                            inline=True,
                        ),
                    ],
                    id="fade-options",
                    className="mb-2",
                    style={"display": "none"},
                ),
                # Anomaly options
                html.Div(
                    [
                        dcc.RadioItems(
                            options=[
                                {
                                    "label": "Discharge Capacity",
                                    "value": "discharge_capacity",
                                },
                                {
                                    "label": "Charge Capacity",
                                    "value": "charge_capacity",
                                },
                                {
                                    "label": "Coulombic Efficiency",
                                    "value": "coulombic_efficiency",
                                },
                            ],
                            value="discharge_capacity",
                            id=METRIC_RADIO,
                            inline=True,
                        ),
                        dcc.Input(
                            id=THRESHOLD_INPUT,
                            type="number",
                            value=3.0,
                            placeholder="Threshold σ",
                            style={"width": "8em"},
                        ),
                    ],
                    id="anomaly-options",
                    className="mb-2",
                    style={"display": "none"},
                ),
                # Energy options
                html.Div(
                    [
                        dcc.Input(
                            id=WEIGHT_INPUT,
                            type="number",
                            value=0,
                            placeholder="Weight g",
                            style={"width": "8em"},
                        ),
                        dcc.Input(
                            id=VOLUME_INPUT,
                            type="number",
                            value=0,
                            placeholder="Volume cm³",
                            style={"width": "8em"},
                        ),
                    ],
                    id="energy-options",
                    className="mb-2",
                    style={"display": "none"},
                ),
                # Clustering options
                html.Div(
                    [
                        dcc.RadioItems(
                            options=[
                                {
                                    "label": "Capacity Retention",
                                    "value": "avg_capacity_retention",
                                },
                                {
                                    "label": "Initial Capacity",
                                    "value": "avg_initial_capacity",
                                },
                                {
                                    "label": "Final Capacity",
                                    "value": "avg_final_capacity",
                                },
                                {
                                    "label": "Coulombic Efficiency",
                                    "value": "avg_coulombic_eff",
                                },
                            ],
                            value="avg_capacity_retention",
                            id=CLUSTER_METRIC,
                            inline=True,
                        ),
                        dcc.RadioItems(
                            options=[
                                {"label": "Hierarchical", "value": "hierarchical"},
                                {"label": "K-means", "value": "kmeans"},
                            ],
                            value="hierarchical",
                            id=METHOD_RADIO,
                            inline=True,
                        ),
                        dcc.Input(
                            id=N_CLUSTERS,
                            type="number",
                            value=3,
                            placeholder="Clusters",
                            style={"width": "6em"},
                        ),
                    ],
                    id="clustering-options",
                    className="mb-2",
                    style={"display": "none"},
                ),
                # Josh options
                html.Div(
                    [
                        dcc.Upload(id=JOSH_UPLOAD, children=dbc.Button("Upload Excel")),
                        dcc.Input(
                            id=JOSH_SHEET, value="Channel51_1", placeholder="Sheet Name"
                        ),
                        dcc.Input(
                            id=JOSH_MASS,
                            type="number",
                            value=0.0015,
                            placeholder="Mass g",
                            style={"width": "8em"},
                        ),
                    ],
                    id="josh-options",
                    className="mb-2",
                    style={"display": "none"},
                ),
            ]
        ),
        dbc.Row([dbc.Col(dcc.Loading(dcc.Graph(id=RESULT_GRAPH)))]),
        dbc.Row([dbc.Col(dcc.Loading(html.Div(id=RESULT_TEXT)))]),
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
        Output("dqdv-options", "style"),
        Output("fade-options", "style"),
        Output("anomaly-options", "style"),
        Output("energy-options", "style"),
        Output("clustering-options", "style"),
        Output("josh-options", "style"),
        Input(ANALYSIS_RADIO, "value"),
    )
    def _toggle_options(analysis):
        show = {"display": "block"}
        hide = {"display": "none"}
        return (
            show if analysis == "dqdv" else hide,
            show if analysis == "fade" else hide,
            show if analysis == "anomalies" else hide,
            show if analysis == "energy" else hide,
            show if analysis == "clustering" else hide,
            show if analysis == "Josh_request_Dq_dv" else hide,
        )

    @app.callback(
        Output(RESULT_GRAPH, "figure"),
        Output(RESULT_TEXT, "children"),
        Input(RUN_BUTTON, "n_clicks"),
        State(ANALYSIS_RADIO, "value"),
        State(SAMPLE_DROPDOWN, "value"),
        State(TEST_DROPDOWN, "value"),
        State(CYCLE_INPUT, "value"),
        State(SMOOTH_CHECK, "value"),
        State(WINDOW_INPUT, "value"),
        State(EOL_INPUT, "value"),
        State(FADE_MODELS, "value"),
        State(METRIC_RADIO, "value"),
        State(THRESHOLD_INPUT, "value"),
        State(WEIGHT_INPUT, "value"),
        State(VOLUME_INPUT, "value"),
        State(CLUSTER_METRIC, "value"),
        State(METHOD_RADIO, "value"),
        State(N_CLUSTERS, "value"),
        State(JOSH_UPLOAD, "contents"),
        State(JOSH_UPLOAD, "filename"),
        State(JOSH_SHEET, "value"),
        State(JOSH_MASS, "value"),
        running=[(Output(RUN_BUTTON, "disabled"), True, False)],
        prevent_initial_call=True,
        background=True,
    )
    def _run_analysis(
        n_clicks,
        analysis,
        sample_id,
        test_id,
        cycle,
        smooth_vals,
        window,
        eol,
        fade_models,
        metric,
        threshold,
        weight,
        volume,
        cluster_metric,
        method,
        n_clusters,
        josh_contents,
        josh_filename,
        josh_sheet,
        josh_mass,
    ):
        if not HAS_ADVANCED or not advanced_analysis:
            raise dash.exceptions.PreventUpdate
        try:
            if analysis == "dqdv":
                smooth = smooth_vals and "smooth" in smooth_vals
                voltage, capacity = advanced_analysis.get_voltage_capacity_data(
                    test_id, cycle
                )
                v_mid, dqdv = advanced_analysis.compute_dqdv(
                    capacity,
                    voltage,
                    smooth=smooth,
                    window_size=window or 11,
                    polyorder=3,
                )
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=v_mid, y=dqdv, mode="lines", name="dQ/dV"))
                fig.update_layout(
                    xaxis_title="Voltage (V)",
                    yaxis_title="dQ/dV",
                    template="plotly_white",
                )
                text = f"Computed dQ/dV for cycle {cycle}"
                return fig, text

            if analysis == "fade":
                from battery_analysis import models

                test = models.TestResult.objects(id=test_id).first()
                if not test:
                    return go.Figure(), "Test not found"
                cycle_nums = [c.cycle_index for c in test.cycles]
                discharge_caps = [c.discharge_capacity for c in test.cycles]
                result = advanced_analysis.capacity_fade_analysis(test_id)
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=cycle_nums,
                        y=discharge_caps,
                        mode="markers+lines",
                        name="Capacity",
                    )
                )
                best = result.get("best_model")
                model_info = result.get("fade_models", {}).get(best, {})
                if best and model_info:
                    x = np.array(cycle_nums)
                    params = model_info.get("params", {})
                    if best == "linear":
                        fit = params.get("slope", 0) * x + params.get("intercept", 0)
                    elif best == "power":
                        fit = params.get("a", 0) * np.power(
                            x, params.get("b", 0)
                        ) + params.get("c", 0)
                    else:
                        fit = params.get("a", 0) * np.exp(
                            params.get("b", 0) * x
                        ) + params.get("c", 0)
                    fig.add_trace(
                        go.Scatter(
                            x=cycle_nums,
                            y=fit,
                            mode="lines",
                            name=f"{model_info.get('name')} fit",
                        )
                    )
                fig.update_layout(
                    xaxis_title="Cycle",
                    yaxis_title="Discharge Capacity",
                    template="plotly_white",
                )
                text = f"Fade rate {result['fade_rate_pct_per_cycle']:.2f}%/cycle"
                eol_cycle = result.get("predicted_eol_cycle")
                if eol_cycle:
                    text += f"; predicted EOL cycle {int(eol_cycle)}"
                return fig, text

            if analysis == "anomalies":
                from battery_analysis import models

                test = models.TestResult.objects(id=test_id).first()
                if not test:
                    return go.Figure(), "Test not found"
                cycle_nums = [c.cycle_index for c in test.cycles]
                if metric == "discharge_capacity":
                    values = [c.discharge_capacity for c in test.cycles]
                elif metric == "charge_capacity":
                    values = [c.charge_capacity for c in test.cycles]
                else:
                    values = [c.coulombic_efficiency for c in test.cycles]
                result = advanced_analysis.detect_anomalies(
                    test_id, metric, n_sigma=threshold or 3.0
                )
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=cycle_nums, y=values, mode="lines+markers", name=metric
                    )
                )
                anomalies = result.get("anomalies", [])
                if anomalies:
                    fig.add_trace(
                        go.Scatter(
                            x=[a["cycle"] for a in anomalies],
                            y=[a["value"] for a in anomalies],
                            mode="markers",
                            marker=dict(color="red", size=10),
                            name="Anomalies",
                        )
                    )
                fig.update_layout(
                    xaxis_title="Cycle",
                    yaxis_title=metric.replace("_", " ").title(),
                    template="plotly_white",
                )
                text = f"Found {result['anomaly_count']} anomalies"
                return fig, text

            if analysis == "energy":
                result = advanced_analysis.energy_analysis(test_id)
                fig = go.Figure()
                init_e = result.get("initial_discharge_energy")
                final_e = result.get("final_discharge_energy")
                if init_e is not None and final_e is not None:
                    fig.add_trace(go.Bar(x=["Initial", "Final"], y=[init_e, final_e]))
                    fig.update_layout(
                        yaxis_title="Energy (Wh)", template="plotly_white"
                    )
                text = (
                    f"Energy retention: {result.get('energy_retention'):.2f}"
                    if result.get("energy_retention") is not None
                    else "No energy data"
                )
                return fig, text

            if analysis == "clustering":
                test_ids: List[str] = []
                if sample_id:
                    test_opts = _get_test_options(sample_id)
                    test_ids = [t["value"] for t in test_opts]
                elif test_id:
                    test_ids = test_id if isinstance(test_id, list) else [test_id]
                if len(test_ids) < 2:
                    return go.Figure(), "Select a sample with at least two tests"
                result = advanced_analysis.cluster_tests(
                    test_ids,
                    metrics=[cluster_metric],
                    method=method or "hierarchical",
                    n_clusters=n_clusters,
                )
                fig = go.Figure()
                pcs = result.get("principal_components", [])
                tests_info = result.get("tests", [])
                clusters = result.get("clusters", {})
                for cluster_id, tests in clusters.items():
                    x_vals = []
                    y_vals = []
                    texts = []
                    for t in tests:
                        try:
                            idx = tests_info.index(t)
                        except ValueError:
                            continue
                        if idx < len(pcs) and len(pcs[idx]) >= 2:
                            x_vals.append(pcs[idx][0])
                            y_vals.append(pcs[idx][1])
                            texts.append(t.get("test_name", t.get("test_id", "")))
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="markers",
                            name=f"Cluster {cluster_id}",
                            text=texts,
                        )
                    )
                fig.update_layout(
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                    template="plotly_white",
                )
                summary_items = [
                    html.Li(
                        f"Cluster {cid}: "
                        + ", ".join(
                            t.get("test_name", t.get("test_id", "")) for t in tlist
                        )
                    )
                    for cid, tlist in clusters.items()
                ]
                text = (
                    html.Div([html.Ul(summary_items)])
                    if summary_items
                    else "No clusters"
                )
                return fig, text

            if analysis == "Josh_request_Dq_dv":
                if not josh_contents:
                    return go.Figure(), "Please upload an Excel file"
                _, content_string = josh_contents.split(",")
                data = base64.b64decode(content_string)
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=josh_filename or ""
                ) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                df = advanced_analysis.josh_request_dq_dv(
                    tmp_path,
                    sheet_name=josh_sheet or "Channel51_1",
                    mass_g=josh_mass or 0.0015,
                )
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=df["V"], y=df["dQdV_sm"], mode="lines", name="dQ/dV")
                )
                fig.update_layout(
                    xaxis_title="Voltage (V)",
                    yaxis_title="dQ/dV",
                    template="plotly_white",
                )
                text = "Computed dQ/dV from uploaded file"
                return fig, text

            return go.Figure(), "Unknown analysis"
        except Exception as e:  # pragma: no cover - runtime errors
            return go.Figure(), f"Error: {e}"

    @app.callback(
        Output(EXPORT_DOWNLOAD, "data"),
        Input(EXPORT_BUTTON, "n_clicks"),
        State(RESULT_GRAPH, "figure"),
        prevent_initial_call=True,
    )
    def _export_plot(n_clicks, fig_dict):
        if not fig_dict:
            return dash.no_update
        fig = go.Figure(fig_dict)
        buffer = io.BytesIO()
        fig.write_image(buffer, format="png")
        buffer.seek(0)
        return dcc.send_bytes(buffer.getvalue(), "advanced_analysis.png")
