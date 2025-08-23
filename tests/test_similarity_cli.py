import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis" / "battery_analysis"

# ---------------------------------------------------------------------------
# Lightweight package stubs so the CLI can be imported without heavy deps
# ---------------------------------------------------------------------------
package_stub = types.ModuleType("battery_analysis")
package_stub.__path__ = [str(PACKAGE_DIR)]  # type: ignore[attr-defined]
sys.modules["battery_analysis"] = package_stub

utils_stub = types.ModuleType("battery_analysis.utils")
utils_stub.__path__ = [str(PACKAGE_DIR / "utils")]  # type: ignore[attr-defined]
sys.modules["battery_analysis.utils"] = utils_stub

spec = importlib.util.spec_from_file_location(
    "battery_analysis.utils.similarity_cli", PACKAGE_DIR / "utils" / "similarity_cli.py"
)
similarity_cli = importlib.util.module_from_spec(spec)
sys.modules["battery_analysis.utils.similarity_cli"] = similarity_cli
assert spec.loader is not None
spec.loader.exec_module(similarity_cli)  # type: ignore[arg-type]

import similarity_suggestions


def test_cli_prints_suggestions(capsys, monkeypatch):
    monkeypatch.setattr(
        similarity_suggestions,
        "suggest_similar_samples",
        lambda sid, N=5: [
            {"sample_id": "S1", "score": "0.500", "differences": "anode=Li"},
            {"sample_id": "S2", "score": "0.250", "differences": ""},
        ],
    )

    similarity_cli.main(["suggest", "REF", "--count", "2"])
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["S1 0.500 anode=Li", "S2 0.250"]


def test_cli_handles_no_results(capsys, monkeypatch):
    monkeypatch.setattr(
        similarity_suggestions, "suggest_similar_samples", lambda sid, N=5: []
    )

    similarity_cli.main(["suggest", "X"])
    out = capsys.readouterr().out.strip()
    assert out == "No similar samples found"
