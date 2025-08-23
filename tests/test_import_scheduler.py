import json
import sys
import types
from pathlib import Path

import pytest

mod = types.ModuleType("import_directory")
mod.import_directory = lambda *a, **k: None  # type: ignore[attr-defined]
sys.modules["battery_analysis.utils.import_directory"] = mod

import battery_analysis.utils.import_scheduler as scheduler  # noqa: E402


def test_load_and_save_jobs_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "jobs.json"
    data = {"jobs": [{"directory": "/tmp/data", "cron": "* * * * *"}]}
    cfg.write_text(json.dumps(data))
    jobs = scheduler.load_jobs(path=cfg)
    assert jobs == data["jobs"]

    new_jobs = [{"directory": "/tmp/other", "cron": "0 * * * *"}]
    scheduler.save_jobs(new_jobs, path=cfg)
    reloaded = json.loads(cfg.read_text())
    assert reloaded["jobs"] == new_jobs


def test_load_jobs_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tmp_path / "jobs.toml"
    cfg.write_text('[[jobs]]\ndirectory = "/tmp/data"\ncron = "* * * * *"\n')
    jobs = scheduler.load_jobs(path=cfg)
    assert jobs == [{"directory": "/tmp/data", "cron": "* * * * *"}]
