from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Callable

import dash
import dash_bootstrap_components as dbc
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import refactor_jobs_tab  # noqa: E402


def test_start_callback_runs_job(monkeypatch: pytest.MonkeyPatch) -> None:
    started: dict[str, str] = {}

    class DummyRefactorJob:
        def __init__(self, status: str = "running") -> None:
            import uuid

            self.id = str(uuid.uuid4())
            self.status = status
            self.start_time = None
            self.end_time = None
            self.current_test = None
            self.processed_count = 0
            self.total_count = 0
            self.errors: list[str] = []

        def save(self) -> "DummyRefactorJob":
            return self

    def fake_refactor_tests(*, job_id: str | None = None) -> None:
        started["job_id"] = job_id or ""

    class DummyThread:
        def __init__(
            self,
            target: Callable[..., None],
            kwargs: dict[str, object] | None = None,
            daemon: bool | None = None,
        ) -> None:
            self.target = target
            self.kwargs = kwargs or {}

        def start(self) -> None:
            self.target(**self.kwargs)

    monkeypatch.setattr(refactor_jobs_tab, "RefactorJob", DummyRefactorJob)
    monkeypatch.setattr(
        refactor_jobs_tab,
        "refactor_data",
        types.SimpleNamespace(refactor_tests=fake_refactor_tests),
    )
    monkeypatch.setattr(refactor_jobs_tab, "Thread", DummyThread)

    app = dash.Dash(__name__)
    refactor_jobs_tab.register_callbacks(app)
    key = next(k for k in app.callback_map if refactor_jobs_tab.JOB_STORE in k)
    callback = app.callback_map[key]["callback"].__wrapped__

    alert, job_id = callback(1, "admin")

    assert started["job_id"] == job_id
    assert isinstance(alert, dbc.Alert)
    assert "Started job" in "".join(str(c) for c in alert.children)
