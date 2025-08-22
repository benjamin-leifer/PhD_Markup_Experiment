from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for p in (ROOT, PACKAGE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Stub out heavy optional dependencies
if "scipy" not in sys.modules:

    class _ScipyStub(types.ModuleType):
        def __getattr__(self, name: str) -> types.ModuleType:  # pragma: no cover - stub
            mod = types.ModuleType(name)
            sys.modules[f"scipy.{name}"] = mod
            setattr(self, name, mod)
            return mod

    sys.modules["scipy"] = _ScipyStub("scipy")

import battery_analysis.api as api  # noqa: E402

client = TestClient(api.app)

ADMIN = "admin-token"
VIEWER = "viewer-token"


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def _clear_jobs() -> None:
    api.ImportJob._registry.clear()
    yield
    api.ImportJob._registry.clear()


def test_requires_token() -> None:
    resp = client.get("/tests")
    assert resp.status_code == 403


def test_viewer_can_list_tests() -> None:
    resp = client.get("/tests", headers=_auth(VIEWER))
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["tests"], list)


def test_import_and_doe(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, str] = {}

    def fake_import(path: str, dry_run: bool = True) -> int:
        called["import"] = path
        return 0

    monkeypatch.setattr(api, "import_directory", fake_import)

    resp = client.post("/import", json={"path": "data"}, headers=_auth(ADMIN))
    assert resp.status_code == 200
    assert called["import"] == "data"

    def fake_save(name: str, factors: dict[str, list[int]]) -> types.SimpleNamespace:
        called["save"] = name
        return types.SimpleNamespace(name=name, matrix=[{"a": 1}])

    monkeypatch.setattr(api, "save_plan", fake_save)

    resp = client.post(
        "/doe-plans",
        json={"name": "p1", "factors": {"a": [1]}},
        headers=_auth(ADMIN),
    )
    assert resp.status_code == 200
    assert called["save"] == "p1"

    resp = client.post("/import", json={"path": "data"}, headers=_auth(VIEWER))
    assert resp.status_code == 403


def test_import_job_submission_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Exec:
        def submit(self, fn, *a, **kw):
            fn(*a, **kw)
            class _Fut:
                def result(self) -> None:
                    return None

            return _Fut()

    monkeypatch.setattr(api, "executor", _Exec())

    def fake_import_directory(path: str, resume: str) -> int:
        job = api.ImportJob._registry[resume]
        job.processed_count = 5
        job.errors = ["oops"]
        job.save()
        return 0

    monkeypatch.setattr(api, "import_directory", fake_import_directory)

    resp = client.post("/import-jobs", json={"path": "data"}, headers=_auth(ADMIN))
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    resp_forbidden = client.post(
        "/import-jobs", json={"path": "data"}, headers=_auth(VIEWER)
    )
    assert resp_forbidden.status_code == 403

    resp2 = client.get(f"/import-jobs/{job_id}", headers=_auth(VIEWER))
    assert resp2.status_code == 200
    data = resp2.json()["job"]
    assert data["processed_count"] == 5
    assert data["errors"] == ["oops"]
