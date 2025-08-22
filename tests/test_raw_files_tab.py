from __future__ import annotations

from pathlib import Path
import sys
import types

import dash
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
import dashboard.raw_files_tab as raw_files_tab  # noqa: E402

client = TestClient(api.app)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_raw_endpoint_enforces_role(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.file_storage, "retrieve_raw", lambda fid: b"DATA")
    resp = client.get("/raw/F1", headers=_auth("admin-token"))
    assert resp.status_code == 200
    assert resp.content == b"DATA"
    resp2 = client.get("/raw/F1", headers=_auth("viewer-token"))
    assert resp2.status_code == 403


def test_search_callback_respects_permissions(monkeypatch: pytest.MonkeyPatch) -> None:
    app = dash.Dash(__name__)
    app.layout = raw_files_tab.layout()
    raw_files_tab.register_callbacks(app)
    cb = app.callback_map[f"{raw_files_tab.TABLE_ID}.data"]["callback"].__wrapped__
    monkeypatch.setattr(
        raw_files_tab,
        "_search_files",
        lambda q: [{"file_id": "F1", "sample": "S1", "timestamp": "T"}],
    )
    admin_rows = cb(1, "F1", "admin")
    assert admin_rows == [
        {
            "file_id": "F1",
            "sample": "S1",
            "timestamp": "T",
            "download": "[Download](/raw/F1)",
        }
    ]
    viewer_rows = cb(1, "F1", "viewer")
    assert viewer_rows[0]["download"] == ""
