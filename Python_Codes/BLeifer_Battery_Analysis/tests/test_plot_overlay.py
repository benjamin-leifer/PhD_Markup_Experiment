import os
import sys
import matplotlib
import pytest

matplotlib.use("Agg")

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from dashboard import plot_overlay
import matplotlib.figure


def sample_data():
    return [
        {
            "name": "S1",
            "capacity": [0, 1, 2],
            "voltage": [3.0, 3.5, 4.0],
            "cycle_index": [1, 2, 3],
            "coulombic_efficiency": [0.9, 0.95, 0.96],
            "impedance": [100, 102, 104],
            "electrolyte": {"additive": "A"},
        },
        {
            "name": "S2",
            "capacity": [0, 1, 2],
            "voltage": [3.1, 3.6, 4.1],
            "cycle_index": [1, 2, 3],
            "coulombic_efficiency": [0.91, 0.94, 0.95],
            "impedance": [101, 103, 105],
            "electrolyte": {"additive": "B"},
        },
    ]


def test_overlay_returns_figure():
    samples = sample_data()
    fig = plot_overlay.plot_overlay(samples)
    assert isinstance(fig, matplotlib.figure.Figure)
    fig = plot_overlay.plot_overlay(samples, metric="ce_vs_cycle")
    assert isinstance(fig, matplotlib.figure.Figure)
    fig = plot_overlay.plot_overlay(samples, metric="impedance_vs_cycle")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_missing_data_raises_error():
    samples = [{"name": "S1", "voltage": [3.0], "electrolyte": {"additive": "A"}}]
    with pytest.raises(ValueError):
        plot_overlay.plot_overlay(samples)
