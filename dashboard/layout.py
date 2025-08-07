"""Dash layout components for the battery dashboard."""

import dash_bootstrap_components as dbc
from dash import html, dcc
from typing import List, Dict


def flagged_table(flags: List[Dict[str, str]]) -> dbc.Table:
    """Return a table of flagged samples."""

    header = html.Thead(
        html.Tr([html.Th("Sample ID"), html.Th("Reason"), html.Th("Actions")])
    )
    rows = []
    for f in flags:
        rows.append(
            html.Tr(
                [
                    html.Td(f["sample_id"]),
                    html.Td(f["reason"]),
                    html.Td(
                        dbc.Button(
                            "Clear",
                            id={"type": "clear-flag", "index": f["sample_id"]},
                            color="secondary",
                            size="sm",
                        )
                    ),
                ]
            )
        )
    body = html.Tbody(rows)
    return dbc.Table(
        [header, body],
        id="flagged-table",
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
    )


def summary_layout(stats: Dict) -> dbc.Row:
    """Summary statistics section."""
    cards = [
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader("Running"),
                    dbc.CardBody(
                        html.H4(
                            str(stats.get("running", 0)),
                            className="card-title",
                        )
                    ),
                ]
            ),
            width=2,
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader("Completed Today"),
                    dbc.CardBody(
                        html.H4(
                            str(stats.get("completed_today", 0)),
                            className="card-title",
                        )
                    ),
                ]
            ),
            width=2,
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader("Failures"),
                    dbc.CardBody(
                        html.H4(
                            str(stats.get("failures", 0)),
                            className="card-title",
                        )
                    ),
                ]
            ),
            width=2,
        ),
    ]
    return dbc.Row(cards, className="mb-4")


def running_tests_table(tests: List[Dict]) -> dbc.Table:
    """Table of running tests."""
    header = html.Thead(
        html.Tr(
            [
                html.Th("Cell ID"),
                html.Th("Chemistry"),
                html.Th("Test Type"),
                html.Th("Current Cycle"),
                html.Th("Last Timestamp"),
                html.Th("Schedule"),
                html.Th("Status"),
                html.Th("Actions"),
            ]
        )
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
                    html.Td(
                        dbc.DropdownMenu(
                            [
                                dbc.DropdownMenuItem(
                                    "Flag for Review",
                                    id={
                                        "type": "flag-review",
                                        "index": t["cell_id"],
                                    },
                                ),
                                dbc.DropdownMenuItem(
                                    "Flag for Retest",
                                    id={
                                        "type": "flag-retest",
                                        "index": t["cell_id"],
                                    },
                                ),
                            ],
                            label="Flag",
                            color="secondary",
                            size="sm",
                        )
                    ),
                ]
            )
        )
    body = html.Tbody(rows)
    return dbc.Table(
        [header, body],
        id="running-tests-table",
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
    )


def upcoming_tests_table(tests: List[Dict]) -> dbc.Table:
    """Table of upcoming tests."""
    header = html.Thead(
        html.Tr(
            [
                html.Th("Cell ID"),
                html.Th("Start Time"),
                html.Th("Hardware"),
                html.Th("Notes"),
                html.Th("Actions"),
            ]
        )
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
                    html.Td(
                        dbc.DropdownMenu(
                            [
                                dbc.DropdownMenuItem(
                                    "Flag for Review",
                                    id={
                                        "type": "flag-review",
                                        "index": t["cell_id"],
                                    },
                                ),
                                dbc.DropdownMenuItem(
                                    "Flag for Retest",
                                    id={
                                        "type": "flag-retest",
                                        "index": t["cell_id"],
                                    },
                                ),
                            ],
                            label="Flag",
                            color="secondary",
                            size="sm",
                        )
                    ),
                ]
            )
        )
    body = html.Tbody(rows)
    return dbc.Table(
        [header, body],
        id="upcoming-tests-table",
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
    )


def metadata_modal() -> dbc.Modal:
    """Modal dialog placeholder for metadata view."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Test Metadata")),
            dbc.ModalBody(id="metadata-content"),
            dbc.ModalFooter(
                dbc.Button(
                    "Close",
                    id="close-metadata",
                    className="ms-auto",
                    n_clicks=0,
                )
            ),
        ],
        id="metadata-modal",
        is_open=False,
        size="lg",
    )


def new_material_form() -> dbc.Form:
    """Form for entering new experimental material info."""
    return dbc.Form(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Label("Material Name"), width=2),
                    dbc.Col(
                        dbc.Input(
                            id="material-name",
                            placeholder="e.g. NMC811",
                        ),
                        width=10,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Label("Chemistry"), width=2),
                    dbc.Col(
                        dbc.Input(
                            id="material-chemistry",
                            placeholder="e.g. LiNiMnCoO2",
                        ),
                        width=10,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Label("Notes"), width=2),
                    dbc.Col(dbc.Textarea(id="material-notes"), width=10),
                ],
                className="mb-3",
            ),
            dbc.Button(
                "Submit",
                id="submit-material",
                color="primary",
                className="mt-2",
            ),
            html.Div(id="material-submit-feedback", className="mt-2"),
        ]
    )


def export_button() -> dbc.Button:
    """Button to open the export modal."""
    return dbc.Button("Export Data", id="open-export", color="secondary")


def export_modal() -> dbc.Modal:
    """Modal dialog for exporting test data."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Export")),
            dbc.ModalBody(
                dbc.Stack(
                    [
                        dbc.Row(
                            [
                                dbc.Label(
                                    "Dataset",
                                    html_for="export-choice",
                                    width="auto",
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        options=[
                                            {
                                                "label": "Running Tests",
                                                "value": "running",
                                            },
                                            {
                                                "label": "Upcoming Tests",
                                                "value": "upcoming",
                                            },
                                        ],
                                        id="export-choice",
                                        value="running",
                                        clearable=False,
                                    ),
                                    className="mb-2",
                                ),
                            ]
                        ),
                        dbc.Button(
                            "Download CSV",
                            id="download-csv",
                            color="primary",
                            className="mb-2",
                        ),
                        dbc.Button(
                            "Download PDF",
                            id="download-pdf",
                            color="secondary",
                            className="mb-2",
                        ),
                        dcc.Download(id="download-data"),
                        dcc.Download(id="download-pdf-file"),
                    ]
                )
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-export", className="ms-auto")
            ),
        ],
        id="export-modal",
        is_open=False,
    )


def data_import_layout() -> html.Div:
    """Layout for the Data Import tab with file upload and metadata form."""
    return html.Div(
        [
            dcc.Upload(
                id="upload-data",
                children=html.Div(
                    ["Drag and Drop or ", html.A("Select Files")]
                ),
                style={
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                },
            ),
            dbc.Form(
                [
                    dbc.Label("Sample Code"),
                    dbc.Input(id="meta-sample-code"),
                    dbc.Label("Chemistry"),
                    dbc.Input(id="meta-chemistry"),
                    dbc.Label("Notes"),
                    dbc.Textarea(id="meta-notes"),
                    dbc.Button("Save", id="save-metadata", color="primary", className="me-2"),
                    dbc.Button("Cancel", id="cancel-metadata", color="secondary"),
                ],
                id="upload-form",
                style={"display": "none"},
            ),
            dcc.Store(id="upload-info"),
            html.Div(id="upload-status"),
            html.Ul(id="uploaded-files-list"),
        ]
    )

