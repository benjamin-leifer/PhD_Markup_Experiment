"""Utility for overlaying test curves by trait."""

from __future__ import annotations

from typing import Iterable, Mapping, Any, Optional

import matplotlib.pyplot as plt
import numpy as np


def _get_nested(data: Mapping[str, Any], path: str) -> Any:
    """Return nested value from a dictionary using dotted path."""
    val: Any = data
    for part in path.split("."):
        if isinstance(val, Mapping) and part in val:
            val = val[part]
        else:
            return None
    return val


def plot_overlay(
    samples: Iterable[Mapping[str, Any]],
    metric: str = "voltage_vs_capacity",
    group_by: str = "electrolyte.additive",
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (6, 4),
    **plot_kwargs: Any,
) -> plt.Figure:
    """Return a Matplotlib figure overlaying curves by trait.

    Parameters
    ----------
    samples:
        Iterable of sample dictionaries containing measurement arrays.
    metric:
        Which metric to plot. Options are ``"voltage_vs_capacity"``,
        ``"ce_vs_cycle"``, or ``"impedance_vs_cycle"``.
    group_by:
        Dotted key path used to color code samples by trait.
    ax:
        Existing :class:`~matplotlib.axes.Axes` to draw on. If ``None`` a new
        figure and axes are created.
    figsize:
        Figure size passed to :func:`matplotlib.pyplot.subplots` when ``ax`` is
        ``None``.
    **plot_kwargs:
        Additional keyword arguments forwarded to
        :meth:`matplotlib.axes.Axes.plot`.

    Raises
    ------
    ValueError
        If the required data for the chosen metric is missing from a sample.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Determine groups and assign colors
    groups = list({ _get_nested(s, group_by) for s in samples })
    groups = [g if g is not None else "Unknown" for g in groups]
    cmap = plt.get_cmap("tab10")
    colors = {g: cmap(i % 10) for i, g in enumerate(groups)}

    for sample in samples:
        grp = _get_nested(sample, group_by) or "Unknown"
        color = colors.get(grp, "C0")

        if metric == "voltage_vs_capacity":
            x = sample.get("capacity")
            y = sample.get("voltage")
            if x is None or y is None:
                raise ValueError("Sample lacks capacity/voltage data")
            ax.plot(
                x,
                y,
                label=sample.get("name", str(sample)),
                color=color,
                **plot_kwargs,
            )
            ax.set_xlabel("Capacity (mAh)")
            ax.set_ylabel("Voltage (V)")
        elif metric == "ce_vs_cycle":
            x = sample.get("cycle_index")
            y = sample.get("coulombic_efficiency")
            if x is None or y is None:
                raise ValueError(
                    "Sample lacks cycle index/coulombic efficiency data"
                )
            ax.plot(
                x,
                y,
                label=sample.get("name", str(sample)),
                color=color,
                **plot_kwargs,
            )
            ax.set_xlabel("Cycle")
            ax.set_ylabel("Coulombic Efficiency")
        elif metric == "impedance_vs_cycle":
            x = sample.get("cycle_index")
            y = sample.get("impedance")
            if x is None or y is None:
                raise ValueError("Sample lacks cycle index/impedance data")
            ax.plot(
                x,
                y,
                label=sample.get("name", str(sample)),
                color=color,
                **plot_kwargs,
            )
            ax.set_xlabel("Cycle")
            ax.set_ylabel("Impedance (Ohm)")
        else:
            raise ValueError(f"Unknown metric: {metric}")

    ax.legend(title=group_by)
    ax.grid(True)
    fig.tight_layout()
    return fig
