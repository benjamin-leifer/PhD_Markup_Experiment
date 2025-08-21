"""Simple Tkinter form for building and saving DOE plans."""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, cast

from ..models import ExperimentPlan
from ..utils.doe_builder import (
    export_csv,
    export_pdf,
    generate_combinations,
    load_from_csv,
    load_from_json,
    save_plan,
)


class DOEBuilderApp(tk.Tk):
    """GUI for creating :class:`ExperimentPlan` records."""

    def __init__(self) -> None:
        super().__init__()
        self.title("DOE Builder")

        self.name_var = tk.StringVar()
        ttk.Label(self, text="Plan Name:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.name_var).grid(row=0, column=1, sticky="ew")

        ttk.Label(self, text="Factors (JSON)").grid(row=1, column=0, sticky="nw")
        self.factors_text = tk.Text(self, width=40, height=10)
        self.factors_text.grid(row=1, column=1, sticky="nsew")

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(btn_frame, text="Preview", command=self.on_preview).pack(
            side=tk.LEFT
        )
        ttk.Button(btn_frame, text="Save", command=self.on_save).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Export CSV", command=self.on_export_csv).pack(
            side=tk.LEFT
        )
        ttk.Button(btn_frame, text="Export PDF", command=self.on_export_pdf).pack(
            side=tk.LEFT
        )

        self.preview = tk.Text(self, width=40, height=10, state="disabled")
        self.preview.grid(row=3, column=0, columnspan=2, sticky="nsew")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)

    # ------------------------------------------------------------------
    def parse_factors(self) -> Dict[str, Any] | None:
        text = self.factors_text.get("1.0", tk.END)
        try:
            return cast(Dict[str, Any], json.loads(text))
        except Exception as exc:  # pragma: no cover - user input
            messagebox.showerror("Invalid JSON", str(exc))
            return None

    def on_preview(self) -> None:
        factors = self.parse_factors()
        if factors is None:
            return
        matrix = generate_combinations(factors)
        self.preview.configure(state="normal")
        self.preview.delete("1.0", tk.END)
        for combo in matrix:
            self.preview.insert(tk.END, f"{combo}\n")
        self.preview.configure(state="disabled")

    def on_save(self) -> None:
        plan = self._build_plan()
        if plan is None:
            return
        messagebox.showinfo(
            "Plan Saved",
            f"Saved '{plan.name}' with {len(plan.matrix)} combinations",
        )

    def on_export_csv(self) -> None:
        plan = self._build_plan()
        if plan is None:
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV Files", "*.csv")]
        )
        if filename:
            export_csv(plan, filename)
            messagebox.showinfo("Exported", f"CSV exported to {filename}")

    def on_export_pdf(self) -> None:
        plan = self._build_plan()
        if plan is None:
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")]
        )
        if filename:
            export_pdf(plan, filename)
            messagebox.showinfo("Exported", f"PDF exported to {filename}")

    def _build_plan(self) -> ExperimentPlan | None:
        factors = self.parse_factors()
        if factors is None:
            return None
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Missing Name", "Please enter a plan name")
            return None
        return save_plan(name, factors)


# ----------------------------------------------------------------------


def launch(argv: list[str] | None = None) -> None:
    """Launch the DOE builder GUI."""

    parser = argparse.ArgumentParser(description="DOE Builder GUI")
    parser.add_argument("--input", help="Path to DOE definition file")
    args = parser.parse_args(argv)

    pre_factors: Dict[str, Any] | None = None
    if args.input:
        path = Path(args.input)
        if path.suffix.lower() == ".csv":
            pre_factors, _ = load_from_csv(path)
        else:
            pre_factors, _ = load_from_json(path)

    app = DOEBuilderApp()
    if pre_factors:
        app.factors_text.insert("1.0", json.dumps(pre_factors, indent=2))
    app.mainloop()


__all__ = ["DOEBuilderApp", "launch"]
