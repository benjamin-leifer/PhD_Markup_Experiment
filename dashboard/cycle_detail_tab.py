"""Plotly-based cycle detail tab with selectable cells and pop-out modal."""

from __future__ import annotations

import io
import logging
from multiprocessing import Process
from typing import Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from battery_analysis.utils.detailed_data_manager import get_detailed_cycle_data
from bson import ObjectId
from bson.errors import InvalidId
from dash import Input, Output, State, dcc, html

from dashboard.data_access import db_connected, get_db_error
from Mongodb_implementation import find_samples, find_test_results

# mypy: ignore-errors


logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from dashboard.data_access import get_cell_dataset
except ImportError:  # pragma: no cover - handled during testing
    get_cell_dataset = None
    logger.warning(
        "get_cell_dataset could not be imported; dataset lookups will be skipped"
    )


def _get_sample_options() -> List[Dict[str, str]]:
    """Return dropdown options for available samples."""
    if not db_connected():
        reason = get_db_error() or "unknown reason"
        logger.error("Database not connected: %s; using demo data", reason)
        return [{"label": "Sample_001", "value": "sample1"}]
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis import models

        if hasattr(models.Sample, "objects"):
            samples = models.Sample.objects.only("name")
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
            return [{"label": "Sample_001", "value": "sample1"}]
        return opts
    except Exception:
        logger.exception("Failed to load sample options")
        return [{"label": "Sample_001", "value": "sample1"}]


def _get_test_options(sample_id: str) -> List[Dict[str, str]]:
    """Return dropdown options for tests belonging to ``sample_id``."""
    if not sample_id:
        return []
    if not db_connected():
        reason = get_db_error() or "unknown reason"
        logger.error("Database not connected: %s; using demo data for tests", reason)
        return [{"label": f"{sample_id}-TestA", "value": str(ObjectId())}]
    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis import models

        if hasattr(models.Sample, "objects") and hasattr(models.TestResult, "objects"):
            logger.info(
                "Using mongoengine backend for test options; sample_id=%s", sample_id
            )
            sample = models.Sample.objects(id=sample_id).first()
            if not sample:
                logger.info(
                    "Mongoengine query returned 0 tests for sample %s", sample_id
                )
                return []

            dataset = getattr(sample, "default_dataset", None)
            if not dataset and get_cell_dataset:
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
                logger.info(
                    "Mongoengine dataset provided %d test options for sample %s",
                    len(options),
                    sample_id,
                )
                return options

            tests = models.TestResult.objects(sample=sample_id).only("name")
            opts = [{"label": t.name, "value": str(t.id)} for t in tests]
            logger.info(
                "Mongoengine TestResult query sample=%s returned %d tests",
                sample_id,
                len(opts),
            )
            return opts

        # Fallback to raw MongoDB helpers
        try:
            sample_oid = ObjectId(sample_id)
        except InvalidId as exc:
            logger.warning("InvalidId for sample_id %s: %s", sample_id, exc)
            sample_oid = sample_id
        query = {"sample": sample_oid}
        logger.info("Using PyMongo backend for test options; query=%s", query)
        tests = find_test_results(query)
        opts = [
            {"label": t.get("name", ""), "value": str(t.get("_id", ""))}
            for t in tests
            if t.get("name")
        ]
        logger.info(
            "PyMongo backend returned %d test options for sample %s",
            len(opts),
            sample_id,
        )
        if not opts:
            logger.warning(
                "No test options found for sample %s; using demo data", sample_id
            )
            return [{"label": f"{sample_id}-TestA", "value": str(ObjectId())}]
        return opts
    except Exception:
        logger.exception("Failed to load test options for sample %s", sample_id)
        return [{"label": f"{sample_id}-TestA", "value": str(ObjectId())}]


def _get_cycle_indices(test_id: str) -> List[int]:
    """Return available cycle indices for ``test_id``.

    The function prefers indices from the ``TestResult`` record, filtering out
    cycles that lack both charge and discharge capacity. If no cycle summaries
    are available it queries :class:`~battery_analysis.models.CycleDetailData`
    directly so that even tests without populated ``cycles`` arrays still return
    available detailed data. When that query fails (for example when the
    database is unavailable), it falls back to :func:`get_detailed_cycle_data`
    which may consult inline cycle summaries.
    """

    logger.debug("Fetching cycle indices for test %s", test_id)
    if not db_connected():
        reason = get_db_error() or "unknown reason"
        logger.warning("Database not connected: %s; using fallback cycle data", reason)
        data = get_detailed_cycle_data(test_id)
        logger.info(
            "Fallback cycle data returned %d cycles for test %s",
            len(data),
            test_id,
        )
        return sorted(data.keys())

    try:  # pragma: no cover - depends on MongoDB
        from battery_analysis import models

        if hasattr(models.TestResult, "objects"):
            logger.info(
                "Using mongoengine backend for cycle indices; test_id=%s", test_id
            )
            test = models.TestResult.objects(id=test_id).only("cycles").first()
            if test and getattr(test, "cycles", None):
                indices = [
                    c.cycle_index
                    for c in test.cycles
                    if getattr(c, "charge_capacity", 0) > 0
                    and getattr(c, "discharge_capacity", 0) > 0
                ]
                logger.info(
                    "Mongoengine TestResult.cycles returned %d indices for test %s",
                    len(indices),
                    test_id,
                )
                if indices:
                    return sorted(indices)

            cycles = models.CycleDetailData.objects(test_result=test_id).only(
                "cycle_index"
            )
            indices = [c.cycle_index for c in cycles]
            logger.info(
                "Mongoengine CycleDetailData query test_result=%s returned %d indices",
                test_id,
                len(indices),
            )
            if indices:
                return sorted(indices)
        else:
            try:
                test_oid = ObjectId(test_id)
            except InvalidId as exc:
                logger.warning("InvalidId for test_id %s: %s", test_id, exc)
                test_oid = test_id
            query = {"_id": test_oid}
            logger.info("Using PyMongo backend for cycle indices; query=%s", query)
            tests = find_test_results(query)
            if tests:
                test_doc = tests[0]
                cycles = test_doc.get("cycles", [])
                indices = [
                    c.get("cycle_index")
                    for c in cycles
                    if c.get("charge_capacity", 0) > 0
                    and c.get("discharge_capacity", 0) > 0
                ]
                logger.info(
                    "PyMongo backend returned %d cycle indices for test %s",
                    len(indices),
                    test_id,
                )
                if indices:
                    return sorted(indices)
    except Exception:
        logger.warning("Failed to load cycle indices from DB", exc_info=True)

    data = get_detailed_cycle_data(test_id)
    logger.info(
        "get_detailed_cycle_data returned %d cycles for test %s",
        len(data),
        test_id,
    )
    if not data:
        logger.warning(
            "No cycle data found for test %s (db_connected=%s)",
            test_id,
            db_connected(),
        )
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
        logger.info(
            "Update figure called with test_id=%s cycle_index=%s (db_connected=%s)",
            test_id,
            cycle_index,
            db_connected(),
        )
        if not test_id or cycle_index is None:
            return go.Figure(), go.Figure()
        data = get_detailed_cycle_data(test_id, cycle_index)
        logger.info("Retrieved %d cycles for test %s", len(data), test_id)
        if cycle_index not in data:
            logger.warning(
                "Cycle %s missing from test %s data (db_connected=%s)",
                cycle_index,
                test_id,
                db_connected(),
            )
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
                "Qt backend not available; install PyQt5/PySide2.",
                "Error",
                "danger",
            )

        try:
            proc = Process(target=_render_matplotlib, args=(fig_dict,), daemon=True)
            proc.start()
            if not proc.is_alive():
                raise OSError("Matplotlib process failed to start")
        except OSError:
            return (
                0,
                True,
                "Failed to launch Matplotlib pop-out.",
                "Error",
                "danger",
            )
        return (0, dash.no_update, dash.no_update, dash.no_update, dash.no_update)


__all__ = ["layout", "register_callbacks"]


def _render_matplotlib(fig_dict):
    import matplotlib.pyplot as plt

    fig = go.Figure(fig_dict)
    try:
        plt.figure()
        for tr in fig.data:
            if isinstance(tr, go.Scatter):
                plt.plot(tr.x, tr.y, label=tr.name)
        if any(tr.name for tr in fig.data):
            plt.legend()
        plt.show()
    except Exception:
        logging.exception("Matplotlib pop-out failed")
        raise SystemExit
