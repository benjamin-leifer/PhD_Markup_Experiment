"""Plotting utilities for battery analysis."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_formation_cycle(
    datasets: Sequence[dict],
    labels: Sequence[str],
    colors: Sequence[str],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay voltage vs. capacity for formation cycles.

    Parameters
    ----------
    datasets : Sequence[dict]
        Each dataset must contain "Q" (capacity) and "V" (voltage) arrays.
    labels : Sequence[str]
        Labels for the legend corresponding to each dataset.
    colors : Sequence[str]
        Colors for each dataset line.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If ``None``, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    max_q = 0.0
    for data, label, color in zip(datasets, labels, colors):
        q = data.get("Q", [])
        v = data.get("V", [])
        ax.plot(q, v, label=label, color=color)
        if len(q) > 0:
            max_q = max(max_q, np.max(q))

    ax.set_xlabel("Q (mAh g$^{-1}$)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True)
    ax.legend()
    ax.set_xlim(0, max_q)
    return ax


def plot_diff_capacity(
    datasets: Sequence[dict],
    labels: Sequence[str],
    colors: Sequence[str],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot differential capacity curves for the charge branch.

    Parameters
    ----------
    datasets : Sequence[dict]
        Each dataset with ``"Q"`` and ``"V"`` arrays.
    labels : Sequence[str]
        Legend labels for each dataset.
    colors : Sequence[str]
        Colors for each dataset line.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If ``None``, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    for data, label, color in zip(datasets, labels, colors):
        q = np.asarray(data.get("Q", []), dtype=float)
        v = np.asarray(data.get("V", []), dtype=float)
        if len(q) < 2 or len(v) < 2:
            continue
        sort_idx = np.argsort(v)
        v_sorted = v[sort_idx]
        q_sorted = q[sort_idx]
        dq_dv = np.gradient(q_sorted, v_sorted)
        ax.plot(v_sorted, dq_dv, label=label, color=color)

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("dQ/dV (mAh g$^{-1}$ V$^{-1}$)")
    ax.grid(True)
    ax.legend()
    return ax
