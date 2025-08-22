import sys
from pathlib import Path

path = Path(__file__).resolve().parents[1]
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

import normalize_cli
import normalization_utils


def test_cli_outputs_normalized_metrics(capsys, monkeypatch):
    sample = normalization_utils.Sample(
        name="S1",
        avg_final_capacity=3.0,
        median_internal_resistance=0.05,
        tests=[],
    )
    setattr(sample, "area", 2.0)
    setattr(sample, "thickness", 0.1)

    monkeypatch.setattr(normalize_cli, "fetch_sample", lambda code: sample)

    normalize_cli.main(["S1"])
    out = capsys.readouterr().out.strip()
    assert out == "S1 capacity=1.500 impedance=0.005"


def test_cli_handles_missing_sample(capsys, monkeypatch):
    monkeypatch.setattr(normalize_cli, "fetch_sample", lambda code: None)
    normalize_cli.main(["X"])
    out = capsys.readouterr().out.strip()
    assert out == "X: sample not found"
