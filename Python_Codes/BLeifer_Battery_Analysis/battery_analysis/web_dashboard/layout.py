"""Layout components for the web dashboard."""

import dash_bootstrap_components as dbc
from dash import html
from typing import List, Dict


def navbar() -> dbc.Navbar:
    """Top navigation bar."""
    return dbc.NavbarSimple(
        brand="Battery Dashboard",
        color="primary",
        dark=True,
    )


def summary_panel(stats: Dict) -> dbc.Card:
    """Summary statistics card."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Running", className="card-title"),
                html.H2(stats.get("running", 0)),
                html.H5("Completed Today", className="mt-3"),
                html.H2(stats.get("completed_today", 0)),
                html.H5("Alerts", className="mt-3"),
                html.H2(stats.get("alerts", 0)),
            ]
        )
    )


def live_tests_panel(tests: List[Dict]) -> dbc.Card:
    """Panel showing live tests."""
    rows = [html.Li(f"{t['cell_id']} - {t['status']}") for t in tests]
    return dbc.Card(
        [
            dbc.CardHeader("Live Tests"),
            dbc.CardBody(html.Ul(rows)),
        ]
    )


def upcoming_tests_panel(tests: List[Dict]) -> dbc.Card:
    """Panel showing upcoming tests."""
    # fmt: off
    rows = [
        html.Li(
            f"{t['cell_id']} @ {t['start_time'].strftime('%H:%M')}"
        )
        for t in tests
    ]
    # fmt: on
    return dbc.Card(
        [
            dbc.CardHeader("Upcoming Tests"),
            dbc.CardBody(html.Ul(rows)),
        ]
    )


def recent_results_panel(results: List[Dict]) -> dbc.Card:
    """Panel showing recent results."""
    # fmt: off
    rows = [
        html.Li(f"{r['cell_id']} finished") for r in results
    ]
    # fmt: on
    return dbc.Card(
        [
            dbc.CardHeader("Recent Results"),
            dbc.CardBody(html.Ul(rows)),
        ]
    )


def dashboard_layout(
    stats,
    live_tests,
    upcoming_tests,
    recent_results,
) -> html.Div:
    """Main dashboard layout."""
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(summary_panel(stats), width=3),
                    dbc.Col(live_tests_panel(live_tests), width=4),
                    dbc.Col(upcoming_tests_panel(upcoming_tests), width=5),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        recent_results_panel(recent_results),
                        width=12,
                    )
                ]
            ),
        ]
    )


def export_modal() -> dbc.Modal:
    """Placeholder export modal."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Export")),
            dbc.ModalBody("Coming soon"),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-export", className="ms-auto")
            ),
        ],
        id="export-modal",
        is_open=False,
    )
