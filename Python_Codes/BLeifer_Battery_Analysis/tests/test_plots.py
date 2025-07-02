import os
import sys
import matplotlib

matplotlib.use("Agg")

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from battery_analysis import plots
import matplotlib.axes


def sample_datasets():
    return [
        {"Q": [0, 1, 2], "V": [3.0, 3.5, 4.0]},
        {"Q": [0, 1, 2], "V": [3.1, 3.6, 4.1]},
    ]


def test_plot_formation_cycle_returns_axes():
    datasets = sample_datasets()
    ax = plots.plot_formation_cycle(datasets, ["a", "b"], ["r", "b"])
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_diff_capacity_returns_axes():
    datasets = sample_datasets()
    ax = plots.plot_diff_capacity(datasets, ["a", "b"], ["r", "b"])
    assert isinstance(ax, matplotlib.axes.Axes)

