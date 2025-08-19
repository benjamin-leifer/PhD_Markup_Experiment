"""Simple Tkinter form for building and saving DOE plans."""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import ttk, messagebox

from ..utils.doe_builder import generate_combinations, save_plan


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
        ttk.Button(btn_frame, text="Preview", command=self.on_preview).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Save", command=self.on_save).pack(side=tk.LEFT)

        self.preview = tk.Text(self, width=40, height=10, state="disabled")
        self.preview.grid(row=3, column=0, columnspan=2, sticky="nsew")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)

    # ------------------------------------------------------------------
    def parse_factors(self) -> dict | None:
        text = self.factors_text.get("1.0", tk.END)
        try:
            return json.loads(text)
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
        factors = self.parse_factors()
        if factors is None:
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Missing Name", "Please enter a plan name")
            return
        plan = save_plan(name, factors)
        messagebox.showinfo(
            "Plan Saved",
            f"Saved '{plan.name}' with {len(plan.matrix)} combinations",
        )


# ----------------------------------------------------------------------

def launch() -> None:
    """Launch the DOE builder GUI."""

    app = DOEBuilderApp()
    app.mainloop()


__all__ = ["DOEBuilderApp", "launch"]
