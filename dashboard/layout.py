"""Dash layout components for the battery dashboard."""

import dash_bootstrap_components as dbc
from dash import html, dcc
from typing import List, Dict


def summary_layout(stats: Dict) -> dbc.Row:
    """Summary statistics section."""
    cards = [
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Running"),
                dbc.CardBody(html.H4(str(stats.get("running", 0)), className="card-title")),
            ]),
            width=2,
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Completed Today"),
                dbc.CardBody(html.H4(str(stats.get("completed_today", 0)), className="card-title")),
            ]),
            width=2,
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Failures"),
                dbc.CardBody(html.H4(str(stats.get("failures", 0)), className="card-title")),
            ]),
            width=2,
        ),
    ]
    return dbc.Row(cards, className="mb-4")


def running_tests_table(tests: List[Dict]) -> dbc.Table:
    """Table of running tests."""
    header = html.Thead(
        html.Tr([
            html.Th("Cell ID"),
            html.Th("Chemistry"),
            html.Th("Test Type"),
            html.Th("Current Cycle"),
            html.Th("Last Timestamp"),
            html.Th("Schedule"),
            html.Th("Status"),
        ])
    )
    rows = []
    for t in tests:
        rows.append(
            html.Tr(
                [
                    html.Td(t["cell_id"]),
                    html.Td(t["chemistry"]),
                    html.Td(t["test_type"]),
                    html.Td(t["current_cycle"]),
                    html.Td(t["last_timestamp"].strftime("%Y-%m-%d %H:%M")),
                    html.Td(t["test_schedule"]),
                    html.Td(t["status"]),
                ]
            )
        )
    body = html.Tbody(rows)
    return dbc.Table([header, body], id="running-tests-table", bordered=True, hover=True, responsive=True, striped=True)


def upcoming_tests_table(tests: List[Dict]) -> dbc.Table:
    """Table of upcoming tests."""
    header = html.Thead(
        html.Tr([
            html.Th("Cell ID"),
            html.Th("Start Time"),
            html.Th("Hardware"),
            html.Th("Notes"),
        ])
    )
    rows = []
    for t in tests:
        rows.append(
            html.Tr(
                [
                    html.Td(t["cell_id"]),
                    html.Td(t["start_time"].strftime("%Y-%m-%d %H:%M")),
                    html.Td(t["hardware"]),
                    html.Td(t["notes"]),
                ]
            )
        )
    body = html.Tbody(rows)
    return dbc.Table([header, body], id="upcoming-tests-table", bordered=True, hover=True, responsive=True, striped=True)


def metadata_modal() -> dbc.Modal:
    """Modal dialog placeholder for metadata view."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Test Metadata")),
            dbc.ModalBody(id="metadata-content"),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-metadata", className="ms-auto", n_clicks=0)
            ),
        ],
        id="metadata-modal",
        is_open=False,
        size="lg",
    )
