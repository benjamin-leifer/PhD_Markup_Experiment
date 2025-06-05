"""Widget for analyzing multiple sample performance metrics.

Provides checkbox selection for metrics, displays a table of
values for each sample and plots bar charts or scatter plots to
compare metrics pairwise.
"""

from __future__ import annotations

import itertools
import tkinter as tk
from tkinter import ttk
from typing import Iterable, List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .. import models


class MultiMetricAnalysis(ttk.Frame):
    """Analysis widget showing multiple metrics for a list of samples."""

    METRICS = [
        ("Initial Capacity", "avg_initial_capacity"),
        ("Final Capacity", "avg_final_capacity"),
        ("Capacity Retention", "avg_capacity_retention"),
        ("Coulombic Efficiency", "avg_coulombic_eff"),
        ("Energy Efficiency", "avg_energy_efficiency"),
        ("Internal Resistance", "median_internal_resistance"),
    ]

    def __init__(self, parent: tk.Widget, samples: Iterable[models.Sample] | None = None) -> None:
        super().__init__(parent)
        self.samples: List[models.Sample] = list(samples) if samples else []
        self.metric_vars = {key: tk.BooleanVar(value=False) for _, key in self.METRICS}
        self._build_ui()

    def _build_ui(self) -> None:
        control = ttk.LabelFrame(self, text="Metrics")
        control.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        for text, key in self.METRICS:
            ttk.Checkbutton(
                control,
                text=text,
                variable=self.metric_vars[key],
                command=self.update_view,
            ).pack(anchor=tk.W)

        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        table_frame = ttk.Frame(right_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.table = ttk.Treeview(table_frame, show="headings")
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.configure(yscrollcommand=scroll.set)

        plot_frame = ttk.Frame(right_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.fig = plt.Figure(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_view()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_samples(self, samples: Iterable[models.Sample]) -> None:
        """Replace the displayed samples."""
        self.samples = list(samples)
        self.update_view()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _selected_metrics(self) -> List[str]:
        return [key for key, var in self.metric_vars.items() if var.get()]

    def update_view(self) -> None:
        self._update_table()
        self._update_plot()

    def _update_table(self) -> None:
        metrics = self._selected_metrics()
        columns = ["name"] + metrics
        self.table.configure(columns=columns)
        for col in columns:
            self.table.heading(col, text=col.replace("_", " ").title())
            self.table.column(col, width=120, anchor=tk.W)
        self.table.delete(*self.table.get_children())
        for s in self.samples:
            values = [getattr(s, "name", "")] + [self._fmt(getattr(s, m, None)) for m in metrics]
            self.table.insert("", tk.END, values=values)

    def _update_plot(self) -> None:
        metrics = self._selected_metrics()
        self.fig.clear()
        if not metrics or not self.samples:
            self.canvas.draw()
            return
        if len(metrics) == 1:
            m = metrics[0]
            ax = self.fig.add_subplot(111)
            names = [getattr(s, "name", "") for s in self.samples]
            vals = [getattr(s, m, float("nan")) for s in self.samples]
            ax.bar(names, vals)
            ax.set_ylabel(m.replace("_", " ").title())
            ax.tick_params(axis="x", rotation=45)
        else:
            pairs = list(itertools.combinations(metrics, 2))
            cols = 2
            rows = (len(pairs) + cols - 1) // cols
            for i, (m1, m2) in enumerate(pairs):
                ax = self.fig.add_subplot(rows, cols, i + 1)
                x = [getattr(s, m1, float("nan")) for s in self.samples]
                y = [getattr(s, m2, float("nan")) for s in self.samples]
                ax.scatter(x, y)
                ax.set_xlabel(m1.replace("_", " ").title())
                ax.set_ylabel(m2.replace("_", " ").title())
        self.fig.tight_layout()
        self.canvas.draw()

    @staticmethod
    def _fmt(val: float | None) -> str:
        return "" if val is None else f"{val:.3g}"
