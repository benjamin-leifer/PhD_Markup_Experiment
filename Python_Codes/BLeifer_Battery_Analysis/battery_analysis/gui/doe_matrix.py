"""Heatmap visualization of Design of Experiments matrices."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DOEHeatmap(ttk.Frame):
    """Display a heatmap of trait combinations.

    Parameters
    ----------
    parent:
        Parent Tk widget.
    matrix:
        Mapping of ``(row_label, col_label)`` to a list of cell IDs.
    rows:
        Iterable of row labels.
    cols:
        Iterable of column labels.
    """

    def __init__(
        self,
        parent: tk.Widget,
        matrix: Dict[Tuple[str, str], List[str]],
        rows: Iterable[str],
        cols: Iterable[str],
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.matrix = matrix
        self.rows = list(rows)
        self.cols = list(cols)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.info_var = tk.StringVar(value="Click a cell to see IDs")
        ttk.Label(self, textvariable=self.info_var).pack(fill=tk.X)

        self.draw_heatmap()
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def draw_heatmap(self) -> None:
        counts = [
            [len(self.matrix.get((r, c), [])) for c in self.cols]
            for r in self.rows
        ]
        self.ax.clear()
        im = self.ax.imshow(counts, cmap="Blues")
        self.ax.set_xticks(range(len(self.cols)), labels=self.cols, rotation=45)
        self.ax.set_yticks(range(len(self.rows)), labels=self.rows)

        # Annotate counts on each cell
        for i, row in enumerate(counts):
            for j, val in enumerate(row):
                self.ax.text(j, i, str(val), ha="center", va="center", color="black")

        self.fig.colorbar(im, ax=self.ax, label="# Tests")
        self.fig.tight_layout()
        self.canvas.draw()

    def on_click(self, event) -> None:
        if event.inaxes != self.ax:
            return
        i, j = int(round(event.ydata)), int(round(event.xdata))
        if 0 <= i < len(self.rows) and 0 <= j < len(self.cols):
            ids = self.matrix.get((self.rows[i], self.cols[j]), [])
            if ids:
                message = ", ".join(ids)
            else:
                message = "No tests"
            self.info_var.set(f"{self.rows[i]} × {self.cols[j]}: {message}")
            messagebox.showinfo("Cell IDs", message)


def demo_matrix() -> Tuple[List[str], List[str], Dict[Tuple[str, str], List[str]]]:
    """Return demo DOE matrix data for standalone use."""
    rows = ["Anode_A", "Anode_B"]
    cols = ["Cathode_1", "Cathode_2"]
    matrix = {
        ("Anode_A", "Cathode_1"): ["C001", "C002"],
        ("Anode_A", "Cathode_2"): ["C003"],
        ("Anode_B", "Cathode_1"): [],
        ("Anode_B", "Cathode_2"): ["C004", "C005", "C006"],
    }
    return rows, cols, matrix
