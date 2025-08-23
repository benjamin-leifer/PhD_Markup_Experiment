import os
import sys
import types
import hashlib

# Create stub battery_analysis package and models
TESTS_DIR = os.path.dirname(__file__)
PACKAGE_DIR = os.path.abspath(os.path.join(TESTS_DIR, "..", "battery_analysis"))
package_stub = types.ModuleType("battery_analysis")
package_stub.__path__ = [PACKAGE_DIR]
sys.modules["battery_analysis"] = package_stub

models_stub = types.ModuleType("battery_analysis.models")

class Sample:
    _registry: dict[str, "Sample"] = {}

    def __init__(self, name: str):
        self.name = name
        self.tests: list[TestResult] = []  # type: ignore[name-defined]

    def save(self) -> "Sample":
        self.__class__._registry[self.name] = self
        return self


class TestResult:
    def __init__(self):
        self.file_path = ""
        self.file_hash = ""
        self.file_id = ""

models_stub.Sample = Sample
models_stub.TestResult = TestResult
sys.modules["battery_analysis.models"] = models_stub

# Stub out utils package and required modules
utils_stub = types.ModuleType("battery_analysis.utils")
utils_stub.__path__ = [os.path.join(PACKAGE_DIR, "utils")]
sys.modules["battery_analysis.utils"] = utils_stub

db_stub = types.ModuleType("battery_analysis.utils.db")
db_stub.ensure_connection = lambda **kwargs: True
sys.modules["battery_analysis.utils.db"] = db_stub

file_storage_stub = types.ModuleType("battery_analysis.utils.file_storage")
file_storage_stub.get_raw_data_file_by_id = lambda *a, **k: b""
sys.modules["battery_analysis.utils.file_storage"] = file_storage_stub

import importlib.util

spec = importlib.util.spec_from_file_location(
    "battery_analysis.utils.verify_import",
    os.path.join(PACKAGE_DIR, "utils", "verify_import.py"),
)
verify_import = importlib.util.module_from_spec(spec)
sys.modules["battery_analysis.utils.verify_import"] = verify_import
spec.loader.exec_module(verify_import)


def test_verify_import_gridfs(tmp_path, monkeypatch):
    sample = Sample("S1").save()

    ok_path = tmp_path / "ok.txt"
    ok_bytes = b"good"
    ok_path.write_bytes(ok_bytes)
    test_ok = TestResult()
    test_ok.file_path = str(ok_path)
    test_ok.file_hash = hashlib.sha256(ok_bytes).hexdigest()
    test_ok.file_id = "ok"
    sample.tests.append(test_ok)

    mismatch_path = tmp_path / "mismatch.txt"
    mismatch_bytes = b"foo"
    mismatch_path.write_bytes(mismatch_bytes)
    test_mismatch = TestResult()
    test_mismatch.file_path = str(mismatch_path)
    test_mismatch.file_hash = hashlib.sha256(mismatch_bytes).hexdigest()
    test_mismatch.file_id = "bad"
    sample.tests.append(test_mismatch)

    test_missing = TestResult()
    test_missing.file_path = str(tmp_path / "missing.txt")
    test_missing.file_hash = ""
    test_missing.file_id = "missing"
    sample.tests.append(test_missing)

    extra_path = tmp_path / "extra.txt"
    extra_path.write_text("extra")

    def fake_get_raw(file_id, as_file_path=False):
        if file_id == "ok":
            return ok_bytes
        if file_id == "bad":
            return b"bad"  # different content
        raise ValueError("not found")

    monkeypatch.setattr(verify_import.file_storage, "get_raw_data_file_by_id", fake_get_raw)

    rows = verify_import.verify_directory(str(tmp_path))
    summary = verify_import.summarize_discrepancies(rows)

    assert summary == {"added": 1, "mismatched": 1, "missing": 2}
    statuses = {r["status"] for r in rows}
    assert {"missing_db", "gridfs_mismatch", "missing_file", "missing_gridfs"} <= statuses
