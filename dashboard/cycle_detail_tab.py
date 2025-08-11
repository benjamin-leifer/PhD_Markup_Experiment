"""Plotly-based cycle detail tab with selectable cells and pop-out modal."""

from __future__ import annotations

from typing import List, Dict, Optional

import io

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State
import plotly.graph_objects as go

from battery_analysis.utils.detailed_data_manager import (
    get_detailed_cycle_data,
)


def _get_sample_options() -> List[Dict[str, str]]:
    """Return dropdown options for available samples."""
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis import models

        samples = models.Sample.objects.only("name")  # type: ignore[attr-defined]
        return [{"label": s.name, "value": str(s.id)} for s in samples]
    except Exception:
        return []


def _get_test_options(sample_id: str) -> List[Dict[str, str]]:
    """Return dropdown options for tests belonging to ``sample_id``."""
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis import models
        from .data_access import get_cell_dataset

        sample = models.Sample.objects(id=sample_id).first()  # type: ignore[attr-defined]
        if not sample:
            return []

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
        return []


def _get_cycle_indices(test_id: str) -> List[int]:
    """Return available cycle indices for ``test_id``.

    The function first queries :class:`~battery_analysis.models.CycleDetailData`
    directly so that even tests without populated ``cycles`` arrays still return
    the indices of stored detailed data. When that query fails (for example when
    the database is unavailable), it falls back to
    :func:`get_detailed_cycle_data` which may consult inline cycle summaries.
    """

    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis import models

        cycles = models.CycleDetailData.objects(test_result=test_id).only(
            "cycle_index"
        )  # type: ignore[attr-defined]
        indices = [c.cycle_index for c in cycles]
        if indices:
            return sorted(indices)
    except Exception:
        pass

    data = get_detailed_cycle_data(test_id)
    return sorted(data.keys())


SAMPLE_DROPDOWN = "cd-sample"
TEST_DROPDOWN = "cd-test"
CYCLE_DROPDOWN = "cd-cycle"
POP_BUTTON = "cd-popout"
EXPORT_BUTTON = "cd-export-btn"
EXPORT_DOWNLOAD = "cd-export-download"
MPL_POPOUT_BUTTON = "cd-mpl-popout-btn"
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
                    dbc.Col(
                        dbc.Button(
                            "Open in Matplotlib",
                            id=MPL_POPOUT_BUTTON,
                            color="secondary",
                        ),
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
        indices = _get_cycle_indices(test_id)
        return [{"label": f"Cycle {idx}", "value": idx} for idx in indices]

    @app.callback(
        Output(GRAPH, "figure"),
        Output(MODAL_GRAPH, "figure"),
        Input(TEST_DROPDOWN, "value"),
        Input(CYCLE_DROPDOWN, "value"),
    )
    def _update_figure(test_id: Optional[str], cycle_index: Optional[int]):
        if not test_id or cycle_index is None:
            return go.Figure(), go.Figure()
        data = get_detailed_cycle_data(test_id, cycle_index)
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

    @app.callback(
        Output(MPL_POPOUT_BUTTON, "n_clicks"),
        Input(MPL_POPOUT_BUTTON, "n_clicks"),
        State(GRAPH, "figure"),
        prevent_initial_call=True,
    )
    def _popout_matplotlib(n_clicks, fig_dict):
        import json
        import matplotlib.pyplot as plt

        if not n_clicks or not fig_dict:
            raise dash.exceptions.PreventUpdate

        def _prepare(vals):
            if not vals:
                return []
            return [
                json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                for v in vals
            ]

        plt.figure()
        for trace in fig_dict.get("data", []):
            if trace.get("type") == "scatter":
                plt.plot(
                    _prepare(trace.get("x", [])),
                    _prepare(trace.get("y", [])),
                    label=trace.get("name"),
                )
        plt.legend()
        plt.show()
        return 0


__all__ = ["layout", "register_callbacks"]
