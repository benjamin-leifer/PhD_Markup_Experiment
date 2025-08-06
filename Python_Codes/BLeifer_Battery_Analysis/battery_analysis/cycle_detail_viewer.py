"""Dash-based cycle detail viewer using Plotly."""

from __future__ import annotations

from typing import List, Optional

import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc

from battery_analysis.utils.detailed_data_manager import (
    get_detailed_cycle_data,
)


def layout(test_options: Optional[List[dict]] = None) -> html.Div:
    """Create the layout for the cycle detail viewer.

    Parameters
    ----------
    test_options:
        Optional list of dictionaries with ``label`` and ``value`` keys used to
        populate the test selection dropdown.

    Returns
    -------
    dash.html.Div
        Layout container for embedding in a Dash application.
    """

    test_options = test_options or []

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="cycle-viewer-test",
                            options=test_options,
                            placeholder="Select test",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="cycle-viewer-cycle",
                            placeholder="Select cycle",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Button("Pop-out", id="cycle-viewer-popout"),
                        md="auto",
                    ),
                ],
                className="gy-2",
            ),
            dcc.Graph(id="cycle-viewer-graph"),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Cycle Detail")),
                    dbc.ModalBody(dcc.Graph(id="cycle-viewer-modal-graph")),
                ],
                id="cycle-viewer-modal",
                size="xl",
            ),
        ]
    )


def register_callbacks(app) -> None:
    """Register Dash callbacks for the cycle detail viewer."""

    @app.callback(
        Output("cycle-viewer-cycle", "options"),
        Input("cycle-viewer-test", "value"),
    )
    def _update_cycle_options(test_id: Optional[str]) -> List[dict]:
        if not test_id:
            return []
        data = get_detailed_cycle_data(test_id)
        options: List[dict] = []
        for idx in sorted(data.keys()):
            options.append({"label": f"Cycle {idx}", "value": idx})
        return options

    @app.callback(
        Output("cycle-viewer-graph", "figure"),
        Output("cycle-viewer-modal-graph", "figure"),
        Input("cycle-viewer-test", "value"),
        Input("cycle-viewer-cycle", "value"),
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
        Output("cycle-viewer-modal", "is_open"),
        Input("cycle-viewer-popout", "n_clicks"),
        State("cycle-viewer-modal", "is_open"),
    )
    def _toggle_modal(n_clicks: Optional[int], is_open: bool) -> bool:
        if n_clicks:
            return not is_open
        return is_open


__all__ = ["layout", "register_callbacks"]
