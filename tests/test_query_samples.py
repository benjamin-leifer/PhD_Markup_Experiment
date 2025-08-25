from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard import data_access
from dashboard.data_access import query_samples
import Mongodb_implementation as mi


def test_query_samples_mongoengine(monkeypatch, caplog):
    fake_pkg = types.ModuleType("battery_analysis")
    fake_models = types.ModuleType("battery_analysis.models")

    sample_obj = SimpleNamespace(name="ObjSample")

    class Objects:
        def __call__(self, __raw__=None):
            assert __raw__ == {"name": "S1"}
            return self

        def only(self, *fields):
            assert fields == ("name",)
            return self

        def __iter__(self):
            yield sample_obj

    fake_models.Sample = SimpleNamespace(objects=Objects())
    fake_pkg.models = fake_models
    monkeypatch.setitem(sys.modules, "battery_analysis", fake_pkg)
    monkeypatch.setitem(sys.modules, "battery_analysis.models", fake_models)

    with caplog.at_level("DEBUG", logger=data_access.logger.name):
        res = query_samples({"name": "S1"}, fields=["name"])
    assert res == [sample_obj]
    assert "Sample.objects path" in caplog.text


def test_query_samples_find_samples(monkeypatch, caplog):
    fake_pkg = types.ModuleType("battery_analysis")
    fake_models = types.ModuleType("battery_analysis.models")

    class Sample:
        pass

    fake_models.Sample = Sample
    fake_pkg.models = fake_models
    monkeypatch.setitem(sys.modules, "battery_analysis", fake_pkg)
    monkeypatch.setitem(sys.modules, "battery_analysis.models", fake_models)

    sample_doc = {"name": "DocSample"}
    monkeypatch.setattr(mi, "find_samples", lambda q: [sample_doc])

    with caplog.at_level("DEBUG", logger=data_access.logger.name):
        res = query_samples({"name": "DocSample"})
    assert res == [sample_doc]
    assert "find_samples path" in caplog.text
