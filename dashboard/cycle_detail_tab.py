"""Plotly-based cycle detail tab with selectable cells and pop-out modal."""

from __future__ import annotations

from typing import List, Dict, Optional

import io

import numpy as np
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State
import plotly.graph_objects as go
from bson import ObjectId

try:  # pragma: no cover - depends on optional packages
    from battery_analysis.utils.detailed_data_manager import (
        get_detailed_cycle_data as _get_detailed_cycle_data,
    )
except Exception:  # pragma: no cover - fallback for demo environments

    def _get_detailed_cycle_data(
        test_id: str, cycle_index: Optional[int] = None
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """Return placeholder detailed cycle data."""
        rng = np.random.default_rng(abs(hash(test_id)) % (2**32))
        cycles = range(1, 6)
        data: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
        for idx in cycles:
            capacity = np.linspace(0, 1.0, 50)
            charge_voltage = 3.0 + 0.1 * rng.standard_normal(capacity.size)
            discharge_voltage = 3.0 - 0.1 * rng.standard_normal(capacity.size)
            data[idx] = {
                "charge": {
                    "capacity": capacity,
                    "voltage": charge_voltage,
                },
                "discharge": {
                    "capacity": capacity,
                    "voltage": discharge_voltage,
                },
            }
        if cycle_index is None:
            return data
        return {cycle_index: data.get(cycle_index, {})}


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
        from .data_access import get_cell_dataset

        sample = models.Sample.objects(id=sample_id).first()  # type: ignore[attr-defined]
        if not sample:
            raise ValueError("sample not found")

        dataset = getattr(sample, "default_dataset", None)
        if not dataset:
            dataset = get_cell_dataset(getattr(sample, "name", ""))

        options: List[Dict[str, str]] = []
        if dataset:
            for t_ref in getattr(dataset, "tests", []):
                try:
                    t_obj = t_ref.fetch() if hasattr(t_ref, "fetch") else t_ref
                    options.append({"label": t_obj.name, "value": str(t_obj.id)})
                except Exception:
                    pass
        if options:
            return options

        tests = models.TestResult.objects(sample=sample_id).only("name")  # type: ignore[attr-defined]
        return [{"label": t.name, "value": str(t.id)} for t in tests]
    except Exception:
        return [
            {"label": f"{sample_id}-TestA", "value": str(ObjectId())},
            {"label": f"{sample_id}-TestB", "value": str(ObjectId())},
        ]


SAMPLE_DROPDOWN = "cd-sample"
TEST_DROPDOWN = "cd-test"
CYCLE_DROPDOWN = "cd-cycle"
POP_BUTTON = "cd-popout"
EXPORT_BUTTON = "cd-export-btn"
EXPORT_DOWNLOAD = "cd-export-download"
GRAPH = "cd-graph"
MODAL = "cd-modal"
MODAL_GRAPH = "cd-modal-graph"
SELECTION_STORE = "trait-selected-sample"


def layout() -> html.Div:
    """Return the layout for the cycle detail tab."""
    sample_opts = _get_sample_options()
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            options=sample_opts,
                            id=SAMPLE_DROPDOWN,
                            placeholder="Sample",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id=TEST_DROPDOWN,
                            placeholder="Test",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id=CYCLE_DROPDOWN,
                            placeholder="Cycle",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Button("Export Plot", id=EXPORT_BUTTON, color="secondary"),
                        md="auto",
                    ),
                    dbc.Col(
                        dbc.Button("Pop-out", id=POP_BUTTON),
                        md="auto",
                    ),
                    dcc.Download(id=EXPORT_DOWNLOAD),
                ],
                className="gy-2",
            ),
            dcc.Graph(id=GRAPH),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Cycle Detail")),
                    dbc.ModalBody(dcc.Graph(id=MODAL_GRAPH)),
                ],
                id=MODAL,
                size="xl",
            ),
        ]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register Dash callbacks for the cycle detail tab."""

    @app.callback(
        Output(SAMPLE_DROPDOWN, "value"),
        Input(SELECTION_STORE, "data"),
        State(SAMPLE_DROPDOWN, "options"),
        prevent_initial_call=True,
    )
    def _prefill_sample(data, options):
        if not data or not data.get("sample"):
            return dash.no_update
        sample_name = data["sample"]
        for opt in options:
            if opt.get("label") == sample_name or opt.get("value") == sample_name:
                return opt.get("value")
        return dash.no_update

    @app.callback(Output(TEST_DROPDOWN, "options"), Input(SAMPLE_DROPDOWN, "value"))
    def _update_tests(sample_id: Optional[str]) -> List[Dict[str, str]]:
        if not sample_id:
            return []
        return _get_test_options(sample_id)

    @app.callback(Output(CYCLE_DROPDOWN, "options"), Input(TEST_DROPDOWN, "value"))
    def _update_cycles(test_id: Optional[str]) -> List[Dict[str, int]]:
        if not test_id:
            return []
        data = _get_detailed_cycle_data(test_id)
        return [{"label": f"Cycle {idx}", "value": idx} for idx in sorted(data.keys())]

    @app.callback(
        Output(GRAPH, "figure"),
        Output(MODAL_GRAPH, "figure"),
        Input(TEST_DROPDOWN, "value"),
        Input(CYCLE_DROPDOWN, "value"),
    )
    def _update_figure(test_id: Optional[str], cycle_index: Optional[int]):
        if not test_id or cycle_index is None:
            return go.Figure(), go.Figure()
        data = _get_detailed_cycle_data(test_id, cycle_index)
        if cycle_index not in data:
            return go.Figure(), go.Figure()
        cycle_data = data[cycle_index]
        charge = cycle_data.get("charge", {})
        discharge = cycle_data.get("discharge", {})
        fig = go.Figure()
        if "voltage" in charge and "capacity" in charge:
            fig.add_trace(
                go.Scatter(
                    x=charge["capacity"],
                    y=charge["voltage"],
                    mode="lines",
                    name="Charge",
                    line=dict(color="blue"),
                )
            )
        if "voltage" in discharge and "capacity" in discharge:
            fig.add_trace(
                go.Scatter(
                    x=discharge["capacity"],
                    y=discharge["voltage"],
                    mode="lines",
                    name="Discharge",
                    line=dict(color="red"),
                )
            )
        fig.update_layout(
            xaxis_title="Capacity (mAh)",
            yaxis_title="Voltage (V)",
            title=f"Voltage vs. Capacity - Cycle {cycle_index}",
            template="plotly_white",
            legend_title_text="Segment",
        )
        return fig, fig

    @app.callback(
        Output(MODAL, "is_open"), Input(POP_BUTTON, "n_clicks"), State(MODAL, "is_open")
    )
    def _toggle_modal(n_clicks: Optional[int], is_open: bool) -> bool:
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output(EXPORT_DOWNLOAD, "data"),
        Input(EXPORT_BUTTON, "n_clicks"),
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
        return dcc.send_bytes(buffer.getvalue(), "cycle_detail.png")


__all__ = ["layout", "register_callbacks"]
