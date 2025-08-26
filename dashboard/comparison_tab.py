"""Comparison tab with Plotly overlays and export."""

# flake8: noqa

from __future__ import annotations

import io
import logging
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, State, dcc, html

# Database helpers
from dashboard.data_access import db_connected, get_db_error
from Mongodb_implementation import find_samples, find_test_results

# mypy: ignore-errors

# Component IDs
SAMPLE_DROPDOWN = "compare-samples"
METRIC_RADIO = "compare-metric"
GRAPH = "compare-graph"
EXPORT_BUTTON = "compare-export-btn"
EXPORT_DOWNLOAD = "compare-export-download"
EXPORT_IMG_BUTTON = "compare-export-img-btn"
EXPORT_IMG_DOWNLOAD = "compare-export-img-download"
ERROR_ALERT = "compare-error-alert"
MPL_POPOUT_BUTTON = "compare-mpl-popout-btn"

logger = logging.getLogger(__name__)


def _get_sample_options() -> Tuple[List[Dict[str, str]], Optional[str]]:
    """Return dropdown options for available samples and an error message."""
    if not db_connected():
        reason = get_db_error() or "unknown reason"
        logger.error("Database not connected: %s; using demo data", reason)
        return (
            [{"label": "Sample_001", "value": "sample1"}],
            f"Database not connected; using demo data ({reason})",
        )

    try:
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
            return (
                [{"label": "Sample_001", "value": "sample1"}],
                "No sample options available; using demo data",
            )
        return opts, None
    except Exception as exc:
        logger.exception("Failed to load sample options")
        return (
            [{"label": "Sample_001", "value": "sample1"}],
            f"Could not load sample options: {exc}",
        )


def _get_sample_data(
    sample_id: str,
) -> Tuple[str, Dict[str, np.ndarray], Optional[str]]:
    """Return (sample name, cycle data, error message) for ``sample_id``.

    The function first attempts to pull real cycle data from the
    ``battery_analysis`` package.  If the ``Sample`` model exposes a
    mongoengine-style ``objects`` attribute the existing ORM-based logic is
    used.  Otherwise raw MongoDB helpers :func:`find_samples` and
    :func:`find_test_results` are used to fetch documents directly.  If any of
    these steps fail (e.g. database not available) deterministic demo data is
    generated so the dashboard remains functional.
    """

    logger.debug("Starting _get_sample_data for sample %s", sample_id)

    try:  # pragma: no cover - depends on battery_analysis and database
        from battery_analysis.models import Sample, TestResult

        from dashboard.data_access import get_cell_dataset

        if hasattr(Sample, "objects"):
            logger.debug("sample %s: using mongoengine path", sample_id)
            # Mongoengine models are available; use them directly
            s = Sample.objects(id=sample_id).first()
            if not s:
                raise ValueError("sample not found")

            sample_name = getattr(s, "name", str(sample_id))
            dataset = getattr(s, "default_dataset", None)
            if not dataset:
                dataset = get_cell_dataset(sample_name)

            cycles: List[int] = []
            capacity: List[float] = []
            ce: List[float] = []

            if dataset and getattr(dataset, "combined_cycles", None):
                for c in dataset.combined_cycles:
                    cycles.append(c.cycle_index)
                    capacity.append(c.discharge_capacity)
                    ce.append(c.coulombic_efficiency)
            else:
                tests = TestResult.objects(sample=s.id).order_by("date")
                for t in tests:
                    summaries = getattr(t, "cycle_summaries", None)
                    if summaries is None:
                        summaries = getattr(t, "cycles", [])
                    for c in summaries:
                        cycles.append(getattr(c, "cycle_index", len(cycles) + 1))
                        capacity.append(getattr(c, "discharge_capacity", np.nan))
                        ce.append(getattr(c, "coulombic_efficiency", np.nan))
        else:
            logger.debug("sample %s: using pymongo path", sample_id)
            # ``Sample`` lacks ``objects``; fall back to raw pymongo helpers
            from bson import ObjectId
            from bson.errors import InvalidId

            try:
                oid = ObjectId(sample_id)
            except InvalidId:
                logger.warning("Invalid sample id %s; using raw value", sample_id)
                oid = sample_id

            sample_query = {"_id": oid}
            logger.debug("find_samples query: %s", sample_query)
            sample_docs = find_samples(sample_query)
            if not sample_docs:
                logger.warning("find_samples returned no documents")
                raise ValueError("sample not found")
            logger.debug("find_samples returned %d documents", len(sample_docs))
            s = sample_docs[0]
            sample_name = s.get("name", str(sample_id))

            cycles = []
            capacity = []
            ce = []

            tests_query = {"sample": oid}
            logger.debug("find_test_results query: %s", tests_query)
            tests = find_test_results(tests_query)
            if not tests:
                logger.warning("find_test_results returned no documents")
            else:
                logger.debug("find_test_results returned %d documents", len(tests))
            for t in tests:
                summaries = t.get("cycle_summaries")
                if summaries is None:
                    summaries = t.get("cycles", [])
                for c in summaries:
                    cycles.append(c.get("cycle_index", len(cycles) + 1))
                    capacity.append(c.get("discharge_capacity", np.nan))
                    ce.append(c.get("coulombic_efficiency", np.nan))

            logger.debug("Parsed %d cycles from summaries", len(cycles))
            if not cycles:
                logger.warning("No cycle summaries found for sample %s", sample_id)

        if not cycles:
            raise ValueError("no cycle data")

        cycles_arr = np.array(cycles, dtype=int)
        capacity_arr = np.array(capacity, dtype=float)
        ce_arr = np.array(ce, dtype=float)
        impedance_arr = np.full_like(cycles_arr, np.nan, dtype=float)
        return (
            sample_name,
            {
                "cycle": cycles_arr,
                "capacity": capacity_arr,
                "ce": ce_arr,
                "impedance": impedance_arr,
            },
            None,
        )
    except Exception as exc:
        logger.exception("Failed to load data for sample %s", sample_id)
        sample_name = str(sample_id)
        rng = np.random.default_rng(abs(hash(sample_name)) % (2**32))
        cycles = np.arange(1, 51)
        capacity = 1.0 + 0.1 * rng.standard_normal(len(cycles))
        ce = 0.98 + 0.01 * rng.standard_normal(len(cycles))
        impedance = 100 + 5 * rng.standard_normal(len(cycles))
        return (
            sample_name,
            {
                "cycle": cycles,
                "capacity": capacity,
                "ce": ce,
                "impedance": impedance,
            },
            f"Could not load data for {sample_id}: {exc}",
        )


def layout() -> html.Div:
    """Return the layout for the comparison tab."""
    sample_opts, error_msg = _get_sample_options()
    return html.Div(
        [
            dbc.Alert(
                error_msg,
                color="warning",
                is_open=bool(error_msg),
                id=ERROR_ALERT,
                className="mb-3",
            ),
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
                    dbc.Col(
                        dbc.Button(
                            "Open in Matplotlib",
                            id=MPL_POPOUT_BUTTON,
                            color="secondary",
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
        Output(ERROR_ALERT, "children"),
        Output(ERROR_ALERT, "is_open"),
        Input(SAMPLE_DROPDOWN, "value"),
        Input(METRIC_RADIO, "value"),
    )
    def _update_graph(samples: Optional[List[str]], metric: str):
        fig = go.Figure()
        if not samples:
            fig.update_layout(template="plotly_white", xaxis_title="Cycle")
            return fig, dash.no_update, dash.no_update
        errors: List[str] = []
        for sample_id in samples:
            name, data, err = _get_sample_data(sample_id)
            if err:
                errors.append(err)
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
        error_msg = "; ".join(errors)
        return fig, error_msg, bool(error_msg)

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
        for sample_id in samples:
            name, data, _ = _get_sample_data(sample_id)
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
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Output("notification-toast", "header", allow_duplicate=True),
        Output("notification-toast", "icon", allow_duplicate=True),
        Input(EXPORT_IMG_BUTTON, "n_clicks"),
        State(GRAPH, "figure"),
        prevent_initial_call=True,
    )
    def _export_plot(n_clicks, fig_dict):
        if not fig_dict:
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        fig = go.Figure(fig_dict)
        buffer = io.BytesIO()
        try:
            fig.write_image(buffer, format="png")
        except (ValueError, ImportError):
            return (
                dash.no_update,
                True,
                "Install the 'kaleido' package to enable image export.",
                "Error",
                "danger",
            )
        buffer.seek(0)
        return (
            dcc.send_bytes(buffer.getvalue(), "comparison_plot.png"),
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    @app.callback(
        Output(MPL_POPOUT_BUTTON, "n_clicks"),
        Output("notification-toast", "is_open", allow_duplicate=True),
        Output("notification-toast", "children", allow_duplicate=True),
        Output("notification-toast", "header", allow_duplicate=True),
        Output("notification-toast", "icon", allow_duplicate=True),
        Input(MPL_POPOUT_BUTTON, "n_clicks"),
        State(GRAPH, "figure"),
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

        Thread(target=_render_matplotlib, args=(fig_dict,), daemon=True).start()
        return (0, dash.no_update, dash.no_update, dash.no_update, dash.no_update)


def _render_matplotlib(fig_dict: Dict[str, Any]) -> None:
    """Render ``fig_dict`` using Matplotlib."""
    import json

    import matplotlib.pyplot as plt

    def _prepare(vals):
        if not vals:
            return []
        return [
            json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for v in vals
        ]

    plt.figure()
    for trace in fig_dict.get("data", []):
        if trace.get("type") == "scatter":
            x = _prepare(trace.get("x", []))
            y = _prepare(trace.get("y", []))
            name = trace.get("name")
            plt.plot(x, y, label=name if name else None)

    handles, labels = plt.gca().get_legend_handles_labels()
    if any(label and not label.startswith("_") for label in labels):
        plt.legend()

    plt.show()
