"""Dashboard tab showing import statistics over time."""

import datetime
from typing import Optional

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

try:
    from . import data_access
except ImportError:  # running as a script
    import data_access  # type: ignore

try:  # pragma: no cover - optional dependency
    from battery_analysis.models import RawDataFile, Sample, TestResult
except Exception:  # pragma: no cover - allow running without models
    TestResult = Sample = RawDataFile = None

DATE_RANGE = "import-stats-range"
WEEKLY_GRAPH = "import-stats-weekly"
MONTHLY_GRAPH = "import-stats-monthly"


def _load_records(
    start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]
) -> pd.DataFrame:
    """Fetch import-related records from the database."""

    records: list[dict[str, object]] = []
    if not data_access.db_connected() or not (
        TestResult and Sample and RawDataFile
    ):  # noqa: E501
        return pd.DataFrame(records, columns=["date", "type"])

    for model, field, name in [
        (TestResult, "date", "TestResult"),
        (Sample, "created_at", "Sample"),
        (RawDataFile, "upload_date", "RawFile"),
    ]:
        try:  # pragma: no cover - requires database
            qs = model.objects
            if start is not None:
                qs = qs.filter(**{f"{field}__gte": start})
            if end is not None:
                qs = qs.filter(**{f"{field}__lte": end})
            for obj in qs:
                dt = getattr(obj, field, None)
                if dt is not None:
                    records.append({"date": dt, "type": name})
        except Exception:
            continue
    return pd.DataFrame.from_records(records)


def aggregate_counts(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate import records by ``freq`` returning counts per type."""

    if df.empty:
        cols = ["period", "TestResult", "Sample", "RawFile"]
        return pd.DataFrame(columns=cols).set_index("period").astype(int)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    grouped = df.groupby("type").resample(freq).size().unstack(0).fillna(0)
    grouped.index.name = "period"
    return grouped.astype(int)


def layout() -> html.Div:
    """Layout for the Import Stats tab."""

    today = datetime.date.today()
    start = today - datetime.timedelta(days=30)
    return html.Div(
        [
            html.H4("Import Stats"),
            dcc.DatePickerRange(
                id=DATE_RANGE, start_date=start, end_date=today
            ),  # noqa: E501
            dcc.Graph(id=WEEKLY_GRAPH),
            dcc.Graph(id=MONTHLY_GRAPH),
        ]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for Import Stats tab."""

    @app.callback(  # type: ignore[misc]
        Output(WEEKLY_GRAPH, "figure"),
        Output(MONTHLY_GRAPH, "figure"),
        Input(DATE_RANGE, "start_date"),
        Input(DATE_RANGE, "end_date"),
    )
    def _update_figures(
        start_date: str, end_date: str
    ) -> tuple[go.Figure, go.Figure]:  # pragma: no cover - callback
        start = pd.to_datetime(start_date) if start_date else None
        end = pd.to_datetime(end_date) if end_date else None
        records = _load_records(start, end)
        weekly = aggregate_counts(records, "W").reset_index()
        monthly = aggregate_counts(records, "M").reset_index()
        weekly_long = weekly.melt(
            id_vars="period", var_name="type", value_name="count"
        )  # noqa: E501
        monthly_long = monthly.melt(
            id_vars="period", var_name="type", value_name="count"
        )
        fig_w = px.bar(weekly_long, x="period", y="count", color="type")
        fig_m = px.line(monthly_long, x="period", y="count", color="type")
        return fig_w, fig_m
