from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

# flake8: noqa
# mypy: ignore-errors

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if "scipy" not in sys.modules:

    class _ScipyStub(types.ModuleType):
        def __getattr__(self, name: str):  # pragma: no cover
            mod = types.ModuleType(name)
            sys.modules[f"scipy.{name}"] = mod
            setattr(self, name, mod)
            return mod

    sys.modules["scipy"] = _ScipyStub("scipy")

import mongomock  # noqa: E402

# Replace stubbed mongoengine with real module for tests
sys.modules.pop("mongoengine", None)
sys.modules["mongoengine"] = importlib.import_module("mongoengine")  # noqa: E402
from mongoengine import connect, disconnect  # noqa: E402

disconnect()
connect("import_test", mongo_client_class=mongomock.MongoClient, alias="default")
from battery_analysis import parsers  # noqa: E402
from battery_analysis.utils import import_directory  # noqa: E402

parsers.register_parser(".csv", lambda path: ([], {}))


@mock_aws
@pytest.mark.parallel
def test_remote_s3_import(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="remote")
    s3.put_object(Bucket="remote", Key="S1/data.csv", Body=b"1,2,3")

    monkeypatch.setattr(import_directory, "ensure_connection", lambda **_: True)

    with caplog.at_level("INFO"):
        rc = import_directory.main(["--remote", "s3://remote", "--dry-run"])
    assert rc == 0
    assert any("Would process" in r.message for r in caplog.records)
