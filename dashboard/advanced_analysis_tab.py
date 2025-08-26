"""Dash components for advanced electrochemical analysis."""

from __future__ import annotations

import base64
import io
import json
import logging
import tempfile
from multiprocessing import Process
from types import SimpleNamespace
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from bson import ObjectId
from bson.errors import InvalidId
from dash import Input, Output, State, dcc, html
import plotly.graph_objects as go
try:  # pragma: no cover
    from plotly.utils import PlotlyJSONDecoder
except Exception:  # pragma: no cover
    from json import JSONDecoder as PlotlyJSONDecoder  # type: ignore

import normalization_utils
from dashboard.data_access import db_connected, get_db_error
from Mongodb_implementation import find_samples, find_test_results

# mypy: ignore-errors

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - gracefully handle missing SciPy
    savgol_filter = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    from battery_analysis import MISSING_ADVANCED_PACKAGES, advanced_analysis
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
MPL_POPOUT_BUTTON = "aa-mpl-popout-btn"
RESULT_GRAPH = "aa-graph"
RESULT_TEXT = "aa-results"
NORMALIZATION_OUTPUT = "aa-normalization"


def compute_dqdv_pitt(
    voltage: np.ndarray,
    capacity: np.ndarray,
    *,
    smooth: bool = True,
    bin_width: float = 0.003,
    window_pre: int = 301,
    poly_pre: int = 3,
    window_post: int = 21,
    poly_post: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute dQ/dV using the patched PITT protocol.

    This implementation follows the approach in
    ``Dq_DV_PITT_Integration_patched.py`` by binning voltage data and
    applying Savitzky–Golay smoothing before and after differentiation.
    """

    df = pd.DataFrame({"V": np.asarray(voltage), "QmAh": np.asarray(capacity)})
    df = df.dropna().sort_values("V")
    if df.empty:
        raise ValueError("No valid voltage/capacity data")

    # Fixed-width binning of voltage
    df = df.assign(_vbin=np.round(df["V"] / bin_width) * bin_width)
    df_bin = (
        df.groupby("_vbin", as_index=False)["QmAh"]
        .mean()
        .rename(columns={"_vbin": "V"})
        .sort_values("V")
    )

    def _savgol(y: np.ndarray, window: int, poly: int) -> np.ndarray:
        if savgol_filter is None or window <= 1:
            return y
        if window % 2 == 0:
            window -= 1
        window = min(window, len(y) - (len(y) % 2 == 0))
        if window < poly + 2 + (poly % 2 == 0):
            window = poly + 2 + (poly % 2 == 0)
        return savgol_filter(y, window, poly)

    v = df_bin["V"].to_numpy()
    q = df_bin["QmAh"].to_numpy() / 1000.0  # mAh → Ah
    order = np.argsort(v)
    v, q = v[order], q[order]
    q_sm = _savgol(q, window_pre, poly_pre) if smooth else q
    dq = np.diff(q_sm)
    dv = np.diff(v)
    v_mid = 0.5 * (v[:-1] + v[1:])
    dqdv = np.divide(dq, dv, out=np.full_like(dq, np.nan), where=dv != 0)
    dqdv = _savgol(dqdv, window_post, poly_post) if smooth else dqdv
    return v_mid, dqdv


def _get_sample_options() -> List[Dict[str, str]]:
    """Return dropdown options for available samples."""
    if not db_connected():
        reason = get_db_error() or "unknown reason"
        logger.error("Database not connected: %s; using demo data", reason)
        return [{"label": "Sample_001", "value": "sample1"}]
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis import models

        if hasattr(models.Sample, "objects"):
            samples = models.Sample.objects.only("name")  # type: ignore[attr-defined]
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
    """Return dropdown options for tests belonging to ``sample_id``."""
    if not sample_id:
        return []
    if not db_connected():
        reason = get_db_error() or "unknown reason"
        logger.error("Database not connected: %s; using demo data for tests", reason)
        return [{"label": f"{sample_id}-TestA", "value": str(ObjectId())}]
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis import models

        if hasattr(models.Sample, "objects") and hasattr(models.TestResult, "objects"):
            tests = models.TestResult.objects(sample=sample_id).only(
                "name"
            )  # type: ignore[attr-defined]
            return [{"label": t.name, "value": str(t.id)} for t in tests]

        try:
            sample_oid = ObjectId(sample_id)
        except InvalidId:
            sample_oid = sample_id
        tests = find_test_results({"sample": sample_oid})
        opts = [
            {"label": t.get("name", ""), "value": str(t.get("_id", ""))}
            for t in tests
            if t.get("name")
        ]
        if not opts:
            logger.warning(
                "No test options found for sample %s; using demo data", sample_id
            )
            return [{"label": f"{sample_id}-TestA", "value": str(ObjectId())}]
        return opts
    except Exception:
        logger.exception("Failed to load test options for sample %s", sample_id)
        return [{"label": f"{sample_id}-TestA", "value": str(ObjectId())}]


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
                    dbc.Button("Export Plot", id=EXPORT_BUTTON, color="secondary"),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button(
                        "Open in Matplotlib",
                        id=MPL_POPOUT_BUTTON,
                        color="secondary",
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
        dbc.Row(dbc.Col(html.Div(id=NORMALIZATION_OUTPUT))),
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
                            value=301,
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
        Output(NORMALIZATION_OUTPUT, "children"), Input(SAMPLE_DROPDOWN, "value")
    )
    def _show_normalization(sample_id):
        if not sample_id:
            return ""
        if not db_connected():
            reason = get_db_error() or "unknown reason"
            logger.warning(
                "Database not connected: %s; cannot show normalization", reason
            )
            return ""
        try:  # pragma: no cover - depends on MongoDB
            from battery_analysis import models

            if hasattr(models.Sample, "objects"):
                sample = models.Sample.objects(id=sample_id).first()
            else:
                try:
                    oid = ObjectId(sample_id)
                except InvalidId:
                    oid = sample_id
                docs = find_samples({"_id": oid})
                sample = SimpleNamespace(**docs[0]) if docs else None
        except Exception:  # pragma: no cover - fallback when DB unavailable
            logger.exception("Failed to load sample for normalization")
            sample = None
        if not sample:
            return ""
        cap = normalization_utils.normalize_capacity(sample)
        imp = normalization_utils.normalize_impedance(sample)
        parts = []
        if cap is not None:
            parts.append(f"Normalized Capacity: {cap:.3f}")
        if imp is not None:
            parts.append(f"Normalized Impedance: {imp:.3f}")
        return " | ".join(parts) if parts else ""

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
        if analysis != "Josh_request_Dq_dv" and not db_connected():
            reason = get_db_error() or "unknown reason"
            logger.warning("Database not connected: %s; cannot run analysis", reason)
            return go.Figure(), f"Database not connected ({reason})"
        try:
            if analysis == "dqdv":
                from battery_analysis import models

                if not hasattr(models.TestResult, "objects"):
                    return (
                        go.Figure(),
                        (
                            "Advanced analysis requires MongoEngine models; "
                            "please install MongoEngine and retry."
                        ),
                    )

                smooth = smooth_vals and "smooth" in smooth_vals
                voltage, capacity = advanced_analysis.get_voltage_capacity_data(
                    test_id, cycle
                )
                v_mid, dqdv = compute_dqdv_pitt(
                    voltage,
                    capacity,
                    smooth=smooth,
                    window_pre=window or 301,
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

                if hasattr(models.TestResult, "objects"):
                    test = models.TestResult.objects(id=test_id).first()
                else:
                    try:
                        oid = ObjectId(test_id)
                    except InvalidId:
                        oid = test_id
                    docs = find_test_results({"_id": oid})
                    if docs:
                        cycles = docs[0].get("cycles") or docs[0].get(
                            "cycle_summaries", []
                        )
                        test = SimpleNamespace(
                            cycles=[SimpleNamespace(**c) for c in cycles]
                        )
                    else:
                        test = None
                if not test:
                    logger.warning("Test %s not found; using demo data", test_id)
                    return go.Figure(), "Test not found"
                if len(test.cycles) < 10:
                    err = (
                        "Need at least 10 cycles for fade analysis, "
                        f"found {len(test.cycles)}"
                    )
                    return go.Figure(), err
                cycle_nums = [c.cycle_index for c in test.cycles]
                discharge_caps = [c.discharge_capacity for c in test.cycles]
                result = advanced_analysis.capacity_fade_analysis(
                    test_id,
                    eol_percent=eol if eol is not None else 80,
                    models=fade_models or [],
                )
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

                if hasattr(models.TestResult, "objects"):
                    test = models.TestResult.objects(id=test_id).first()
                else:
                    try:
                        oid = ObjectId(test_id)
                    except InvalidId:
                        oid = test_id
                    docs = find_test_results({"_id": oid})
                    if docs:
                        cycles = docs[0].get("cycles") or docs[0].get(
                            "cycle_summaries", []
                        )
                        test = SimpleNamespace(
                            cycles=[SimpleNamespace(**c) for c in cycles]
                        )
                    else:
                        test = None
                if not test:
                    logger.warning("Test %s not found; using demo data", test_id)
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

    @app.callback(
        Output(MPL_POPOUT_BUTTON, "n_clicks"),
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Output("notification-toast", "header", allow_duplicate=True),
        Output("notification-toast", "icon", allow_duplicate=True),
        Input(MPL_POPOUT_BUTTON, "n_clicks"),
        State(RESULT_GRAPH, "figure"),
        prevent_initial_call=True,
    )
    def _popout_matplotlib(n_clicks, fig_dict):
        import matplotlib

        if not n_clicks or not fig_dict:
            raise dash.exceptions.PreventUpdate

        if matplotlib.get_backend().lower() == "agg":
            return (
                0,
                True,
                "An interactive Matplotlib backend is required for pop-out plots.",
                "Error",
                "danger",
            )

        Process(target=_render_matplotlib, args=(fig_dict,), daemon=True).start()
        return (0, dash.no_update, dash.no_update, dash.no_update, dash.no_update)


def _render_matplotlib(fig_dict):
    import matplotlib.pyplot as plt
    fig = go.Figure(
        json.loads(json.dumps(fig_dict), cls=PlotlyJSONDecoder)
    )
    plt.figure()
    for tr in fig.data:
        if isinstance(tr, go.Scatter):
            plt.plot(tr.x, tr.y, label=tr.name)
    if any(tr.name for tr in fig.data):
        plt.legend()
    plt.show()
