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
        self.reassign_btn = ttk.Button(control, text="Reassign Test", command=self.reassign_selected)
        self.reassign_btn.pack(side=tk.RIGHT, padx=(0, 5))

        columns = ("test", "missing")
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        self.tree.heading("test", text="Test")
        self.tree.heading("missing", text="Missing Components")
        self.tree.column("test", width=200)
        self.tree.column("missing", width=200)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        h_scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.tree.xview)
        h_scrollbar.pack(fill=tk.X)

        self.tree.config(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

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
        dlg.geometry("400x300")
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

    def reassign_selected(self):
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

        dlg = tk.Toplevel(self)
        dlg.title("Reassign Test")
        dlg.geometry("300x150")

        ttk.Label(dlg, text="New Sample Name:").pack(anchor=tk.W, padx=5, pady=5)
        entry = ttk.Entry(dlg)
        entry.pack(fill=tk.X, padx=5)

        def save():
            name = entry.get().strip()
            if not name:
                messagebox.showerror("Error", "Sample name required")
                return
            sample = models.Sample.objects(name=name).first()
            if not sample:
                sample = models.Sample(name=name)
                sample.save()

            # remove from old sample
            old_sample = test.sample.fetch() if hasattr(test.sample, "fetch") else test.sample
            if old_sample and test in getattr(old_sample, "tests", []):
                old_sample.tests.remove(test)
                old_sample.save()

            test.sample = sample
            test.save()
            if test not in getattr(sample, "tests", []):
                sample.tests.append(test)
                sample.save()

            missing = [f for f in ("anode", "cathode", "separator", "electrolyte") if getattr(sample, f, None) is None]
            if missing:
                self.main_app.missing_data.append({"test_id": str(test.id), "sample_id": str(sample.id), "missing": missing})

            if rec in self.main_app.missing_data:
                self.main_app.missing_data.remove(rec)
            dlg.destroy()
            self.refresh_table()

        ttk.Button(dlg, text="Save", command=save).pack(pady=5)
        dlg.transient(self)
        dlg.grab_set()
