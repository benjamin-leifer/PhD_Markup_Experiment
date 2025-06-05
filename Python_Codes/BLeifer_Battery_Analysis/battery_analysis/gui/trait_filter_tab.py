"""Trait filter tab for the Tkinter GUI.

Provides widgets to filter :class:`battery_analysis.models.Sample` documents by
chemistry and manufacturer. Database queries fall back to demo data so the GUI
can run without MongoDB.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional

from .. import models
from .multi_metric_analysis import MultiMetricAnalysis


def get_distinct_values(field: str) -> List[str]:
    """Return distinct values for ``field`` from Sample records."""
    try:  # pragma: no cover - depends on MongoDB
        return list(models.Sample.objects.distinct(field))
    except Exception:
        if field == "chemistry":
            return ["NMC", "LFP", "LCO"]
        if field == "manufacturer":
            return ["ABC Batteries", "XYZ Cells"]
        return []


def filter_samples(
    chemistry: Optional[str], manufacturer: Optional[str]
) -> List[models.Sample]:
    """Query samples matching the provided traits."""
    try:  # pragma: no cover - depends on MongoDB
        qs = models.Sample.objects
        if chemistry:
            qs = qs.filter(chemistry=chemistry)
        if manufacturer:
            qs = qs.filter(manufacturer=manufacturer)
        return list(qs)
    except Exception:
        dummy = models.Sample(name="Sample_001")
        dummy.chemistry = chemistry or "NMC"
        dummy.manufacturer = manufacturer or "ABC Batteries"
        dummy.avg_initial_capacity = 1.0
        dummy.avg_final_capacity = 0.9
        dummy.avg_capacity_retention = 0.9
        dummy.avg_coulombic_eff = 0.99
        dummy.avg_energy_efficiency = 0.98
        dummy.median_internal_resistance = 0.1
        return [dummy]


class TraitFilterTab(ttk.Frame):
    """Tab for filtering samples by traits."""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.create_widgets()
        self.refresh_options()

    def create_widgets(self):
        """Create the widgets for this tab."""
        control = ttk.LabelFrame(self, text="Filter Options")
        control.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(control, text="Chemistry:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.chem_var = tk.StringVar()
        self.chem_box = ttk.Combobox(control, textvariable=self.chem_var, width=20)
        self.chem_box.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(control, text="Manufacturer:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.man_var = tk.StringVar()
        self.man_box = ttk.Combobox(control, textvariable=self.man_var, width=20)
        self.man_box.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        self.filter_btn = ttk.Button(control, text="Filter", command=self.apply_filter)
        self.filter_btn.grid(row=0, column=4, padx=5, pady=5)

        self.refresh_btn = ttk.Button(control, text="Refresh", command=self.refresh_options)
        self.refresh_btn.grid(row=0, column=5, padx=5, pady=5)

        result_frame = ttk.Frame(self)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        columns = ("name", "chemistry", "manufacturer")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col.capitalize())
            self.tree.column(col, width=150, anchor=tk.W)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.config(yscrollcommand=scrollbar.set)

        analysis_frame = ttk.LabelFrame(self, text="Metric Analysis")
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.metric_analysis = MultiMetricAnalysis(analysis_frame)
        self.metric_analysis.pack(fill=tk.BOTH, expand=True)

    def refresh_options(self):
        """Refresh dropdown options from the database."""
        self.chem_box['values'] = get_distinct_values("chemistry")
        self.man_box['values'] = get_distinct_values("manufacturer")
        self.chem_box.set("")
        self.man_box.set("")

    def apply_filter(self):
        """Filter samples based on the selected traits."""
        chemistry = self.chem_var.get() or None
        manufacturer = self.man_var.get() or None
        samples = filter_samples(chemistry, manufacturer)
        for r in self.tree.get_children():
            self.tree.delete(r)
        for s in samples:
            self.tree.insert(
                "",
                tk.END,
                values=(
                    getattr(s, "name", ""),
                    getattr(s, "chemistry", ""),
                    getattr(s, "manufacturer", ""),
                ),
            )
        self.metric_analysis.update_samples(samples)
        self.main_app.update_status(f"Found {len(samples)} sample(s)")
