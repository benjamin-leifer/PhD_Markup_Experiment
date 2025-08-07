"""Utility for detecting outlier samples."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

try:  # pragma: no cover - optional dependency
    import seaborn as sns
except Exception:  # pragma: no cover - gracefully handle missing seaborn
    sns = None

try:
    from . import models, utils
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    models = importlib.import_module("models")
    utils = importlib.import_module("utils")


_valid_metrics = [
    "avg_initial_capacity",
    "avg_final_capacity",
    "avg_capacity_retention",
    "avg_coulombic_eff",
    "avg_energy_efficiency",
    "median_internal_resistance",
    "capacity_retention",
    "initial_capacity",
    "final_capacity",
]


def _get_sample(obj):
    if hasattr(obj, "id") and hasattr(obj, "name"):
        return obj
    try:  # pragma: no cover - depends on MongoDB
        return models.Sample.objects(id=obj).first()
    except Exception:
        return None


def detect_outliers(
    samples: Iterable, metric: str = "capacity_retention"
) -> Tuple[List[str], plt.Figure]:
    """Detect statistical outliers in ``samples`` based on ``metric``.

    Both IQR and z-score methods are used. Outliers from either method are
    highlighted on a boxplot with an overlaid swarmplot.

    Parameters
    ----------
    samples:
        Iterable of :class:`Sample` objects or sample IDs.
    metric:
        Attribute on the sample to analyze. Defaults to ``"capacity_retention"``.

    Returns
    -------
    list[str]
        List of sample IDs identified as outliers.
    matplotlib.figure.Figure
        Figure showing the distribution and labelled outliers.
    """
    if metric not in _valid_metrics:
        raise ValueError(f"Invalid metric: {metric}")

    ids = []
    values = []

    for obj in samples:
        s = _get_sample(obj)
        if s is None:
            continue
        val = getattr(s, metric, None)
        if val is None:
            continue
        ids.append(str(s.id))
        values.append(float(val))

    if not values:
        raise ValueError("No valid metric values found")

    arr = np.array(values)

    # z-score method
    z_scores = np.abs((arr - np.mean(arr)) / np.std(arr))
    z_mask = z_scores > 3

    # IQR method
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_mask = (arr < lower) | (arr > upper)

    outlier_indices = set(np.where(z_mask)[0]) | set(np.where(iqr_mask)[0])
    outlier_ids = [ids[i] for i in sorted(outlier_indices)]

    # plotting
    fig, ax = plt.subplots(figsize=(5, 4))
    if sns is not None:  # pragma: no cover - optional
        sns.boxplot(y=arr, ax=ax, color="lightgray")
        sns.swarmplot(y=arr, ax=ax, color="tab:blue")
    else:  # pragma: no cover - fallback
        ax.boxplot(
            arr, vert=True, patch_artist=True, boxprops={"facecolor": "lightgray"}
        )
        ax.scatter(np.zeros_like(arr), arr, color="tab:blue")
    for i in outlier_indices:
        ax.text(0, arr[i], ids[i], ha="left", va="center", color="red")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xticks([])
    ax.set_title("Outlier Detection")
    fig.tight_layout()

    return outlier_ids, fig
