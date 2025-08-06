"""Dash tab highlighting tests missing component assignments."""

from __future__ import annotations

from typing import List, Dict

from dash import html
import dash_bootstrap_components as dbc

ALERT_CONTAINER = "missing-data-alerts"


def _get_missing_data() -> List[Dict[str, object]]:
    """Return tests with missing component references.

    Tries to query the MongoDB backend, otherwise returns example data so the
    interface remains functional without a database.
    """
    try:  # pragma: no cover - requires database
        from battery_analysis import models

        records: List[Dict[str, object]] = []
        for test in models.TestResult.objects():  # type: ignore[attr-defined]
            sample = test.sample.fetch() if hasattr(test.sample, "fetch") else test.sample
            missing = [
                f
                for f in ("anode", "cathode", "separator", "electrolyte")
                if getattr(sample, f, None) is None
            ]
            if missing:
                records.append({"test_id": str(test.id), "missing": missing})
        return records
    except Exception:
        return [
            {"test_id": "Test_A", "missing": ["cathode", "separator"]},
            {"test_id": "Test_B", "missing": ["electrolyte"]},
        ]


def layout() -> html.Div:
    """Return layout for the missing data tab."""
    alerts = []
    for rec in _get_missing_data():
        msg = ", ".join(rec["missing"])
        alerts.append(
            dbc.Alert(
                [
                    html.H6(f"Test {rec['test_id']}", className="mb-1"),
                    html.P(f"Missing components: {msg}", className="mb-1"),
                    dbc.Button(
                        "Resolve",
                        color="link",
                        href=f"/resolve/{rec['test_id']}",
                        external_link=True,
                    ),
                ],
                color="warning",
                className="mb-2",
            )
        )
    if not alerts:
        alerts = [dbc.Alert("No missing data detected", color="success")]
    return html.Div(alerts, id=ALERT_CONTAINER)


def register_callbacks(app):  # pragma: no cover - no callbacks yet
    """Placeholder for future callbacks."""
    return None
