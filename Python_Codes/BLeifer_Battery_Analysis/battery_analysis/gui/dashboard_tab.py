"""Dashboard tab for monitoring running tests."""

import tkinter as tk
from tkinter import ttk, messagebox
import datetime

from battery_analysis import models


class DashboardTab(ttk.Frame):
    """Tab that displays cells currently undergoing testing."""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.refresh_interval = 10000  # ms
        self.create_widgets()
        self.after(self.refresh_interval, self.refresh_data)

    def create_widgets(self):
        """Create widgets for the dashboard."""
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        self.refresh_btn = ttk.Button(control_frame, text="Refresh", command=self.refresh_data)
        self.refresh_btn.pack(side=tk.RIGHT)

        columns = ("sample", "test", "start", "cycle", "status")
        self.tree = ttk.Treeview(self, columns=columns, show="headings")
        self.tree.heading("sample", text="Sample")
        self.tree.heading("test", text="Test Name")
        self.tree.heading("start", text="Start Time")
        self.tree.heading("cycle", text="Cycles")
        self.tree.heading("status", text="Status")

        self.tree.column("sample", width=150)
        self.tree.column("test", width=200)
        self.tree.column("start", width=150)
        self.tree.column("cycle", width=80, anchor=tk.E)
        self.tree.column("status", width=100)

        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.config(yscrollcommand=scrollbar.set)

    def refresh_data(self):
        """Refresh the dashboard table with running tests."""
        for row in self.tree.get_children():
            self.tree.delete(row)
        try:
            running_tests = models.TestResult.objects(validated=False).order_by("-date")
            for test in running_tests:
                try:
                    sample_name = test.sample.fetch().name if test.sample else "Unknown"
                except Exception:
                    sample_name = "Unknown"
                start_time = test.date.strftime("%Y-%m-%d %H:%M") if test.date else ""
                if test.cycle_count is not None:
                    cycle = test.cycle_count
                elif test.cycles:
                    cycle = len(test.cycles)
                else:
                    cycle = "?"
                status = "Running"
                self.tree.insert("", tk.END, values=(sample_name, test.name or "", start_time, cycle, status))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dashboard data: {e}")
        self.after(self.refresh_interval, self.refresh_data)
