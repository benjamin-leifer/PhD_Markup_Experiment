"""GUI tab displaying tests missing component assignments."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from .. import models


class MissingDataTab(ttk.Frame):
    """Show tests with samples missing component references."""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self._item_map: dict[str, dict] = {}
        self.create_widgets()

    def create_widgets(self):
        control = ttk.Frame(self)
        control.pack(fill=tk.X, padx=10, pady=5)
        self.refresh_btn = ttk.Button(control, text="Refresh", command=self.refresh_table)
        self.refresh_btn.pack(side=tk.LEFT)
        self.resolve_btn = ttk.Button(control, text="Resolve Selected", command=self.resolve_selected)
        self.resolve_btn.pack(side=tk.RIGHT)

        columns = ("test", "missing")
        self.tree = ttk.Treeview(self, columns=columns, show="headings")
        self.tree.heading("test", text="Test")
        self.tree.heading("missing", text="Missing Components")
        self.tree.column("test", width=200)
        self.tree.column("missing", width=200)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.config(yscrollcommand=scrollbar.set)

        self.refresh_table()

    def refresh_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self._item_map.clear()
        for rec in getattr(self.main_app, "missing_data", []):
            item = self.tree.insert("", tk.END, values=(rec.get("test_id"), ", ".join(rec.get("missing", []))))
            self._item_map[item] = rec

    def resolve_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        rec = self._item_map.get(sel[0])
        if not rec:
            return
        test = models.TestResult.objects(id=rec["test_id"]).first()
        if not test:
            messagebox.showerror("Error", "Test not found")
            return
        sample = test.sample.fetch() if hasattr(test.sample, "fetch") else test.sample
        fields = rec["missing"]
        dlg = tk.Toplevel(self)
        dlg.title("Resolve Components")
        entries = {}
        for f in fields:
            ttk.Label(dlg, text=f.capitalize()+":").pack(anchor=tk.W, padx=5, pady=2)
            e = ttk.Entry(dlg)
            e.pack(fill=tk.X, padx=5)
            entries[f] = e
        def save():
            for f, ent in entries.items():
                name = ent.get().strip()
                if not name:
                    continue
                comp = models.Sample.objects(name=name).first()
                if not comp:
                    comp = models.Sample(name=name)
                    comp.save()
                setattr(sample, f, comp)
            sample.save()
            if rec in self.main_app.missing_data:
                self.main_app.missing_data.remove(rec)
            dlg.destroy()
            self.refresh_table()
        ttk.Button(dlg, text="Save", command=save).pack(pady=5)
        dlg.transient(self)
        dlg.grab_set()
