"""Dash tab displaying document status information."""

from __future__ import annotations

from typing import List, Dict

from dash import html, dash_table
import dash_bootstrap_components as dbc

STATUS_TABLE = "doc-status-table"


def _get_document_statuses() -> List[Dict[str, str]]:
    """Return status info for key document models.

    Attempts to query the MongoDB backend for counts. If the backend is not
    available, placeholder values are returned so that the UI still renders.
    """
    try:  # pragma: no cover - requires database
        from battery_analysis import models

        docs = [models.Sample, models.TestResult, models.CycleSummary]
        statuses = []
        for doc in docs:
            count = doc.objects.count()  # type: ignore[attr-defined]
            statuses.append(
                {
                    "document": doc.__name__,
                    "count": count,
                    "status": "Available" if count else "Missing",
                }
            )
        return statuses
    except Exception:
        # Fallback demo data
        return [
            {"document": "Sample", "count": 1, "status": "Available"},
            {"document": "TestResult", "count": 1, "status": "Available"},
            {"document": "CycleSummary", "count": 0, "status": "Missing"},
        ]


def layout() -> html.Div:
    """Return layout for the document status tab."""
    data = _get_document_statuses()
    table = dash_table.DataTable(
        id=STATUS_TABLE,
        columns=[
            {"name": "Document", "id": "document"},
            {"name": "Records", "id": "count"},
            {"name": "Status", "id": "status"},
        ],
        data=data,
        style_table={"overflowX": "auto"},
    )
    return html.Div([
        html.H4("Document Status"),
        table,
    ])


def register_callbacks(app):  # pragma: no cover - no callbacks yet
    """Placeholder for potential future callbacks."""
    return None
