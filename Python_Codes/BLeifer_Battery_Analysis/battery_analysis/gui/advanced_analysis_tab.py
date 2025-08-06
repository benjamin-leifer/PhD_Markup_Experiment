"""
Advanced Analysis tab for the battery analysis GUI.

This module provides a tab for performing advanced analysis techniques, including
differential capacity analysis, capacity fade modeling, anomaly detection, and more.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from battery_analysis.gui.custom_toolbar import CustomToolbar
from battery_analysis.utils import popout_figure
import numpy as np
import pandas as pd
import threading
import logging
import os

from battery_analysis import models
from battery_analysis import advanced_analysis

try:  # pragma: no cover - optional component
    from battery_analysis.cycle_detail_viewer import CycleDetailViewer
except Exception:  # pragma: no cover - fallback if viewer unavailable
    CycleDetailViewer = None  # type: ignore


class AdvancedAnalysisTab(ttk.Frame):
    """Tab for advanced electrochemical analysis."""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.current_sample = None
        self.current_test = None
        self.current_dqdv_data = None
        self.current_fade_analysis = None
        self.create_widgets()

    def create_widgets(self):
        """Create the widgets for the advanced analysis tab."""
        # Create a frame for the selection and controls
        self.top_frame = ttk.Frame(self)
        self.top_frame.pack(fill=tk.X, padx=10, pady=10)

        # Sample and Test Selection
        selection_frame = ttk.LabelFrame(
            self.top_frame, text="Sample and Test Selection"
        )
        selection_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

        # Sample selection
        sample_frame = ttk.Frame(selection_frame)
        sample_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(sample_frame, text="Sample:").pack(side=tk.LEFT, padx=5)
        self.sample_combobox = ttk.Combobox(sample_frame, width=30)
        self.sample_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.sample_combobox.bind("<<ComboboxSelected>>", self.on_sample_selected)

        self.refresh_samples_btn = ttk.Button(
            sample_frame, text="Refresh", command=self.refresh_samples
        )
        self.refresh_samples_btn.pack(side=tk.RIGHT, padx=5)

        # Test selection
        test_frame = ttk.Frame(selection_frame)
        test_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(test_frame, text="Test:").pack(side=tk.LEFT, padx=5)
        self.test_combobox = ttk.Combobox(test_frame, width=30)
        self.test_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.test_combobox.bind("<<ComboboxSelected>>", self.on_test_selected)

        # Analysis selection
        analysis_frame = ttk.LabelFrame(self.top_frame, text="Analysis Type")
        analysis_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(20, 5), pady=5)

        self.analysis_type_var = tk.StringVar(value="dqdv")
        analyses = [
            ("Differential Capacity (dQ/dV)", "dqdv"),
            ("Capacity Fade Modeling", "fade"),
            ("Anomaly Detection", "anomalies"),
            ("Energy Analysis", "energy"),
            ("Test Clustering", "clustering"),
            ("Josh_request_Dq_dv", "Josh_request_Dq_dv"),
        ]

        for i, (text, value) in enumerate(analyses):
            ttk.Radiobutton(
                analysis_frame,
                text=text,
                variable=self.analysis_type_var,
                value=value,
                command=self.on_analysis_type_changed,
            ).pack(anchor=tk.W, padx=5, pady=3)

        # Analysis options frame (contents will change based on analysis type)
        self.options_frame = ttk.LabelFrame(self.top_frame, text="Analysis Options")
        self.options_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Initial options frame content (for dQ/dV)
        self.dqdv_options = ttk.Frame(self.options_frame)
        self.dqdv_options.pack(fill=tk.BOTH, expand=True)

        ttk.Label(self.dqdv_options, text="Cycle Number:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )
        self.cycle_entry = ttk.Entry(self.dqdv_options, width=10)
        self.cycle_entry.insert(0, "1")
        self.cycle_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.dqdv_options, text="Smoothing:").grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W
        )
        self.smooth_var = tk.BooleanVar(value=True)
        self.smooth_check = ttk.Checkbutton(self.dqdv_options, variable=self.smooth_var)
        self.smooth_check.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(self.dqdv_options, text="Window Size:").grid(
            row=2, column=0, padx=5, pady=5, sticky=tk.W
        )
        self.window_entry = ttk.Entry(self.dqdv_options, width=10)
        self.window_entry.insert(0, "11")
        self.window_entry.grid(row=2, column=1, padx=5, pady=5)

        # Run analysis button
        self.run_btn = ttk.Button(
            self.top_frame, text="Run Analysis", command=self.run_analysis
        )
        self.run_btn.pack(side=tk.LEFT, padx=20, pady=5, ipadx=10, ipady=10)

        # In your create_widgets method, find your options frame and add:
        self.view_cycle_btn = ttk.Button(
            self.options_frame,  # Make sure to use the correct frame name from your code
            text="View Cycle Details",
            command=self.view_cycle_details,
        )
        self.view_cycle_btn.pack(fill=tk.X, padx=5, pady=5)

        # Create a notebook for results
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Plot frame for visualization
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="Plots")

        # Create a matplotlib figure for plots
        self.fig = plt.figure(figsize=(9, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add a toolbar with editing support
        self.toolbar = CustomToolbar(self.canvas, self.plot_frame)
        self.toolbar.update()

        # Button to open plot in a standalone window
        self.popout_btn = ttk.Button(
            self.plot_frame,
            text="Open in Window",
            command=lambda: popout_figure(self.fig),
        )
        self.popout_btn.pack(anchor=tk.NE, padx=5, pady=5)

        # Results frame for text information
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")

        # Add a text widget for displaying results
        self.results_text = tk.Text(self.results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.config(state=tk.DISABLED)

        # Add a scrollbar to the results text
        self.results_scrollbar = ttk.Scrollbar(
            self.results_text, command=self.results_text.yview
        )
        self.results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=self.results_scrollbar.set)

        # Initially configure the tab for dQ/dV analysis
        self.on_analysis_type_changed()

    def refresh_samples(self):
        """Refresh the sample list from the database."""
        if not self.main_app.db_connected:
            messagebox.showwarning(
                "Not Connected", "Please connect to the database first."
            )
            return

        try:
            # Get all samples from the database
            samples = models.Sample.objects().all()

            # Update the combobox
            sample_names = [sample.name for sample in samples]
            self.sample_combobox["values"] = sample_names

            if sample_names:
                self.sample_combobox.current(0)
                self.on_sample_selected(None)

            self.main_app.log_message(
                f"Loaded {len(sample_names)} samples for advanced analysis"
            )
        except Exception as e:
            self.main_app.log_message(f"Error loading samples: {str(e)}", logging.ERROR)

    def on_sample_selected(self, event):
        """Handle sample selection event."""
        sample_name = self.sample_combobox.get()
        if not sample_name:
            return

        try:
            # Get the sample from the database
            sample = models.Sample.objects(name=sample_name).first()

            if not sample:
                messagebox.showerror(
                    "Sample Not Found", f"Sample '{sample_name}' not found in database."
                )
                return

            # Store the current sample
            self.current_sample = sample

            # Get tests for this sample
            tests = []
            for test_ref in sample.tests:
                test = models.TestResult.objects(id=test_ref.id).first()
                if test:
                    tests.append(test)

            # Update the test combobox
            test_names = [test.name for test in tests]
            self.test_combobox["values"] = test_names

            if test_names:
                self.test_combobox.current(0)
                self.on_test_selected(None)
            else:
                self.test_combobox.set("")
                self.current_test = None

                # Clear results
                self.clear_results()

            self.main_app.log_message(
                f"Loaded {len(tests)} tests for advanced analysis of sample {sample_name}"
            )
        except Exception as e:
            self.main_app.log_message(f"Error loading tests: {str(e)}", logging.ERROR)

    def on_test_selected(self, event):
        """Handle test selection event."""
        if not self.current_sample:
            return

        test_name = self.test_combobox.get()
        if not test_name:
            self.current_test = None
            self.clear_results()
            return

        try:
            # Get the test from the database
            test = models.TestResult.objects(
                sample=self.current_sample, name=test_name
            ).first()

            if not test:
                messagebox.showerror(
                    "Test Not Found", f"Test '{test_name}' not found for this sample."
                )
                return

            # Store the current test
            self.current_test = test

            self.main_app.log_message(
                f"Selected test {test_name} for advanced analysis"
            )

            # Clear previous results
            self.clear_results()
        except Exception as e:
            self.main_app.log_message(f"Error selecting test: {str(e)}", logging.ERROR)

    def on_analysis_type_changed(self):
        """Handle analysis type change."""
        # Clear previous options
        for widget in self.options_frame.winfo_children():
            widget.destroy()

        analysis_type = self.analysis_type_var.get()

        if analysis_type == "dqdv":
            # dQ/dV analysis options
            frame = ttk.Frame(self.options_frame)
            frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            ttk.Label(frame, text="Cycle Number:").grid(
                row=0, column=0, padx=5, pady=5, sticky=tk.W
            )
            self.cycle_entry = ttk.Entry(frame, width=10)
            self.cycle_entry.insert(0, "1")
            self.cycle_entry.grid(row=0, column=1, padx=5, pady=5)

            ttk.Label(frame, text="Smoothing:").grid(
                row=1, column=0, padx=5, pady=5, sticky=tk.W
            )
            self.smooth_var = tk.BooleanVar(value=True)
            self.smooth_check = ttk.Checkbutton(frame, variable=self.smooth_var)
            self.smooth_check.grid(row=1, column=1, padx=5, pady=5)

            ttk.Label(frame, text="Window Size:").grid(
                row=2, column=0, padx=5, pady=5, sticky=tk.W
            )
            self.window_entry = ttk.Entry(frame, width=10)
            self.window_entry.insert(0, "11")
            self.window_entry.grid(row=2, column=1, padx=5, pady=5)

        elif analysis_type == "fade":
            # Capacity fade analysis options
            frame = ttk.Frame(self.options_frame)
            frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            ttk.Label(frame, text="EOL Capacity (%):").grid(
                row=0, column=0, padx=5, pady=5, sticky=tk.W
            )
            self.eol_entry = ttk.Entry(frame, width=10)
            self.eol_entry.insert(0, "80")
            self.eol_entry.grid(row=0, column=1, padx=5, pady=5)

            ttk.Label(frame, text="Models to Try:").grid(
                row=1, column=0, padx=5, pady=5, sticky=tk.W
            )

            self.linear_var = tk.BooleanVar(value=True)
            self.linear_check = ttk.Checkbutton(
                frame, text="Linear", variable=self.linear_var
            )
            self.linear_check.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

            self.power_var = tk.BooleanVar(value=True)
            self.power_check = ttk.Checkbutton(
                frame, text="Power Law", variable=self.power_var
            )
            self.power_check.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

            self.exp_var = tk.BooleanVar(value=True)
            self.exp_check = ttk.Checkbutton(
                frame, text="Exponential", variable=self.exp_var
            )
            self.exp_check.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

        elif analysis_type == "anomalies":
            # Anomaly detection options
            frame = ttk.Frame(self.options_frame)
            frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            ttk.Label(frame, text="Metric:").grid(
                row=0, column=0, padx=5, pady=5, sticky=tk.W
            )
            self.metric_var = tk.StringVar(value="discharge_capacity")
            metrics = [
                ("Discharge Capacity", "discharge_capacity"),
                ("Charge Capacity", "charge_capacity"),
                ("Coulombic Efficiency", "coulombic_efficiency"),
            ]

            for i, (text, value) in enumerate(metrics):
                ttk.Radiobutton(
                    frame, text=text, variable=self.metric_var, value=value
                ).grid(row=i + 1, column=0, sticky=tk.W, padx=20, pady=2)

            ttk.Label(frame, text="Detection Threshold (σ):").grid(
                row=4, column=0, padx=5, pady=5, sticky=tk.W
            )
            self.threshold_entry = ttk.Entry(frame, width=10)
            self.threshold_entry.insert(0, "3.0")
            self.threshold_entry.grid(row=4, column=1, padx=5, pady=5)

        elif analysis_type == "energy":
            # Energy analysis options
            frame = ttk.Frame(self.options_frame)
            frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            ttk.Label(frame, text="Cell Weight (g):").grid(
                row=0, column=0, padx=5, pady=5, sticky=tk.W
            )
            self.weight_entry = ttk.Entry(frame, width=10)
            self.weight_entry.insert(0, "0")
            self.weight_entry.grid(row=0, column=1, padx=5, pady=5)

            ttk.Label(frame, text="Cell Volume (cm³):").grid(
                row=1, column=0, padx=5, pady=5, sticky=tk.W
            )
            self.volume_entry = ttk.Entry(frame, width=10)
            self.volume_entry.insert(0, "0")
            self.volume_entry.grid(row=1, column=1, padx=5, pady=5)

            ttk.Label(frame, text="Note: Enter 0 if unknown").grid(
                row=2, column=0, columnspan=2, padx=5, pady=5
            )

        elif analysis_type == "clustering":
            # Clustering options
            frame = ttk.Frame(self.options_frame)
            frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            ttk.Label(frame, text="Metric:").grid(
                row=0, column=0, padx=5, pady=5, sticky=tk.W
            )
            self.cluster_metric_var = tk.StringVar(value="capacity_retention")
            metrics = [
                ("Capacity Retention", "avg_capacity_retention"),
                ("Initial Capacity", "avg_initial_capacity"),
                ("Final Capacity", "avg_final_capacity"),
                ("Coulombic Efficiency", "avg_coulombic_eff"),
            ]

            for i, (text, value) in enumerate(metrics):
                ttk.Radiobutton(
                    frame, text=text, variable=self.cluster_metric_var, value=value
                ).grid(row=i + 1, column=0, sticky=tk.W, padx=20, pady=2)

            ttk.Label(frame, text="Clustering Method:").grid(
                row=0, column=1, padx=5, pady=5, sticky=tk.W
            )
            self.method_var = tk.StringVar(value="hierarchical")
            methods = [("Hierarchical", "hierarchical"), ("K-means", "kmeans")]

            for i, (text, value) in enumerate(methods):
                ttk.Radiobutton(
                    frame, text=text, variable=self.method_var, value=value
                ).grid(row=i + 1, column=1, sticky=tk.W, padx=20, pady=2)

            ttk.Label(frame, text="Number of Clusters:").grid(
                row=5, column=0, padx=5, pady=5, sticky=tk.W
            )
            self.n_clusters_entry = ttk.Entry(frame, width=10)
            self.n_clusters_entry.insert(0, "3")
            self.n_clusters_entry.grid(row=5, column=1, padx=5, pady=5)

            ttk.Label(frame, text="Note: This feature needs at least 5 tests").grid(
                row=6, column=0, columnspan=2, padx=5, pady=5
            )

        elif analysis_type == "Josh_request_Dq_dv":
            frame = ttk.Frame(self.options_frame)
            frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            ttk.Label(frame, text="Excel File:").grid(
                row=0, column=0, padx=5, pady=5, sticky=tk.W
            )
            self.josh_file_entry = ttk.Entry(frame, width=30)
            self.josh_file_entry.grid(row=0, column=1, padx=5, pady=5)
            ttk.Button(frame, text="Browse", command=self.browse_josh_file).grid(
                row=0, column=2, padx=5, pady=5
            )

            ttk.Label(frame, text="Sheet Name:").grid(
                row=1, column=0, padx=5, pady=5, sticky=tk.W
            )
            self.josh_sheet_entry = ttk.Entry(frame, width=20)
            self.josh_sheet_entry.insert(0, "Channel51_1")
            self.josh_sheet_entry.grid(row=1, column=1, padx=5, pady=5)

            ttk.Label(frame, text="Mass (g):").grid(
                row=2, column=0, padx=5, pady=5, sticky=tk.W
            )
            self.josh_mass_entry = ttk.Entry(frame, width=10)
            self.josh_mass_entry.insert(0, "0.0015")
            self.josh_mass_entry.grid(row=2, column=1, padx=5, pady=5)

    def clear_results(self):
        """Clear current results."""
        # Clear the figure
        self.fig.clear()
        self.canvas.draw()

        # Clear the results text
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)

        # Reset stored results
        self.current_dqdv_data = None
        self.current_fade_analysis = None

    def run_analysis(self):
        """Run the selected analysis type."""
        analysis_type = self.analysis_type_var.get()
        if analysis_type != "Josh_request_Dq_dv" and not self.current_test:
            messagebox.showinfo("No Test Selected", "Please select a test to analyze.")
            return

        # Update the status
        self.main_app.update_status(f"Running {analysis_type} analysis...")
        self.run_btn.config(state=tk.DISABLED)

        # Clear previous results
        self.clear_results()

        # Run the analysis in a thread
        threading.Thread(
            target=self._run_analysis_thread, args=(analysis_type,), daemon=True
        ).start()

    def _run_analysis_thread(self, analysis_type):
        """Thread function to run the analysis."""
        try:
            if analysis_type == "dqdv":
                self._run_dqdv_analysis()
            elif analysis_type == "fade":
                self._run_fade_analysis()
            elif analysis_type == "anomalies":
                self._run_anomaly_detection()
            elif analysis_type == "energy":
                self._run_energy_analysis()
            elif analysis_type == "clustering":
                self._run_clustering_analysis()
            elif analysis_type == "Josh_request_Dq_dv":
                self._run_josh_request_dqdv()

            self.main_app.update_status(
                f"{analysis_type.capitalize()} analysis completed"
            )

        except Exception as e:
            self.main_app.log_message(
                f"Error in {analysis_type} analysis: {str(e)}", logging.ERROR
            )
            messagebox.showerror(
                "Analysis Error", f"Error running {analysis_type} analysis: {str(e)}"
            )
            self.main_app.update_status(f"Analysis failed: {str(e)}")

        finally:
            # Re-enable the run button
            self.run_btn.config(state=tk.NORMAL)

    def _run_dqdv_analysis(self):
        """Run differential capacity analysis."""
        test_id = str(self.current_test.id)

        # Get options
        try:
            cycle_number = int(self.cycle_entry.get())
        except ValueError:
            cycle_number = 1

        smooth = self.smooth_var.get()

        try:
            window_size = int(self.window_entry.get())
        except ValueError:
            window_size = 11

        self.main_app.log_message(
            f"Running dQ/dV analysis for test {self.current_test.name}, "
            + f"cycle {cycle_number}, smooth={smooth}, window_size={window_size}"
        )

        # Get voltage and capacity data
        try:
            voltage, capacity = advanced_analysis.get_voltage_capacity_data(
                test_id, cycle_number
            )

            if smooth:
                self.plot_smoothed_dqdv(
                    voltage,
                    capacity,
                    cycle_number,
                    window_size=window_size,
                    polyorder=3,
                )
            else:
                from battery_analysis.advanced_analysis import compute_dqdv

                v_mid, dqdv = compute_dqdv(
                    capacity,
                    voltage,
                    smooth=False,
                    window_size=window_size,
                    polyorder=3,
                )

                self.current_dqdv_data = {
                    "voltage": voltage,
                    "capacity": capacity,
                    "v_centers": v_mid,
                    "dq_dv": dqdv,
                    "cycle": cycle_number,
                }

                self.plot_dqdv_results()

            # Update the results text
            self.update_dqdv_results_text()

            # Switch to the plot tab
            self.notebook.select(0)

            self.main_app.log_message(
                f"dQ/dV analysis complete for cycle {cycle_number}"
            )

        except Exception as e:
            self.main_app.log_message(
                f"Error in dQ/dV analysis: {str(e)}", logging.ERROR
            )
            raise

    def plot_dqdv_results(self):
        """Plot differential capacity analysis results."""
        if not self.current_dqdv_data:
            return

        # Clear the figure
        self.fig.clear()

        # Create a 2x1 grid
        gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1])

        # Top plot: dQ/dV
        ax1 = self.fig.add_subplot(gs[0])
        ax1.plot(
            self.current_dqdv_data["v_centers"],
            self.current_dqdv_data["dq_dv"],
            "-",
            linewidth=2,
        )
        ax1.set_xlabel("Voltage (V)")
        ax1.set_ylabel("dQ/dV (mAh/V)")
        ax1.set_title(
            f'Differential Capacity Analysis - Cycle {self.current_dqdv_data["cycle"]}'
        )
        ax1.grid(True, linestyle="--", alpha=0.7)

        # Find peaks
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(
            self.current_dqdv_data["dq_dv"],
            height=0.1 * max(self.current_dqdv_data["dq_dv"]),
            distance=50,
        )

        # Highlight peaks
        for peak_idx in peaks:
            ax1.plot(
                self.current_dqdv_data["v_centers"][peak_idx],
                self.current_dqdv_data["dq_dv"][peak_idx],
                "ro",
                markersize=8,
            )
            ax1.annotate(
                f"{self.current_dqdv_data['v_centers'][peak_idx]:.3f} V",
                (
                    self.current_dqdv_data["v_centers"][peak_idx],
                    self.current_dqdv_data["dq_dv"][peak_idx],
                ),
                xytext=(10, 10),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

        # Bottom plot: Voltage vs Capacity
        ax2 = self.fig.add_subplot(gs[1])
        ax2.plot(
            self.current_dqdv_data["capacity"], self.current_dqdv_data["voltage"], "-"
        )
        ax2.set_xlabel("Capacity (mAh)")
        ax2.set_ylabel("Voltage (V)")
        ax2.set_title(f'Voltage vs. Capacity - Cycle {self.current_dqdv_data["cycle"]}')
        ax2.grid(True, linestyle="--", alpha=0.7)

        # Adjust layout
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_smoothed_dqdv(
        self, voltage, capacity, cycle, *, window_size=11, polyorder=3
    ):
        """Compute and plot a smoothed differential capacity curve."""
        from battery_analysis.advanced_analysis import compute_dqdv

        v_mid, dqdv = compute_dqdv(
            capacity,
            voltage,
            smooth=True,
            window_size=window_size,
            polyorder=polyorder,
        )

        self.current_dqdv_data = {
            "voltage": voltage,
            "capacity": capacity,
            "v_centers": v_mid,
            "dq_dv": dqdv,
            "cycle": cycle,
        }

        self.plot_dqdv_results()

    def update_dqdv_results_text(self):
        """Update the results text with dQ/dV analysis information."""
        if not self.current_dqdv_data:
            return

        # Find peaks
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(
            self.current_dqdv_data["dq_dv"],
            height=0.1 * max(self.current_dqdv_data["dq_dv"]),
            distance=50,
        )

        # Enable editing
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        # Add title
        self.results_text.insert(
            tk.END, "Differential Capacity Analysis Results\n\n", "title"
        )

        # Test and cycle info
        self.results_text.insert(
            tk.END, f"Sample: {self.current_sample.name}\n", "heading"
        )
        self.results_text.insert(tk.END, f"Test: {self.current_test.name}\n", "heading")
        self.results_text.insert(
            tk.END, f"Cycle: {self.current_dqdv_data['cycle']}\n\n", "heading"
        )

        # Peak information
        self.results_text.insert(
            tk.END, f"Detected Peaks (Phase Transitions)\n", "heading"
        )

        if len(peaks) == 0:
            self.results_text.insert(tk.END, "No significant peaks detected.\n\n")
        else:
            for i, peak_idx in enumerate(peaks):
                voltage = self.current_dqdv_data["v_centers"][peak_idx]
                intensity = self.current_dqdv_data["dq_dv"][peak_idx]

                self.results_text.insert(
                    tk.END,
                    f"Peak {i + 1}: {voltage:.3f} V (intensity: {intensity:.2f} mAh/V)\n",
                )

            self.results_text.insert(tk.END, "\n")

        # Voltage range
        v_min = min(self.current_dqdv_data["voltage"])
        v_max = max(self.current_dqdv_data["voltage"])

        self.results_text.insert(tk.END, f"Voltage Range\n", "heading")
        self.results_text.insert(tk.END, f"Min: {v_min:.3f} V\n")
        self.results_text.insert(tk.END, f"Max: {v_max:.3f} V\n")
        self.results_text.insert(tk.END, f"Range: {v_max - v_min:.3f} V\n\n")

        # Capacity range
        c_min = min(self.current_dqdv_data["capacity"])
        c_max = max(self.current_dqdv_data["capacity"])

        self.results_text.insert(tk.END, f"Capacity Range\n", "heading")
        self.results_text.insert(tk.END, f"Min: {c_min:.3f} mAh\n")
        self.results_text.insert(tk.END, f"Max: {c_max:.3f} mAh\n")
        self.results_text.insert(tk.END, f"Range: {c_max - c_min:.3f} mAh\n\n")

        # Add interpretation
        self.results_text.insert(tk.END, "Interpretation\n", "heading")

        if len(peaks) > 0:
            # Add information about what peaks might indicate
            self.results_text.insert(
                tk.END,
                "The peaks in the dQ/dV curve correspond to phase transitions in the active material.\n\n",
            )

            # For Li-ion cells, specific voltage ranges might indicate:
            if any(3.0 <= self.current_dqdv_data["v_centers"][p] <= 3.5 for p in peaks):
                self.results_text.insert(
                    tk.END,
                    "Peaks in the 3.0-3.5V range typically correspond to graphite staging reactions.\n",
                )

            if any(3.4 <= self.current_dqdv_data["v_centers"][p] <= 3.6 for p in peaks):
                self.results_text.insert(
                    tk.END,
                    "Peaks around 3.4-3.6V may indicate LiFePO₄ phase transitions (if applicable).\n",
                )

            if any(3.7 <= self.current_dqdv_data["v_centers"][p] <= 4.1 for p in peaks):
                self.results_text.insert(
                    tk.END,
                    "Peaks in the 3.7-4.1V range are often associated with NMC/NCA phase transitions.\n",
                )
        else:
            self.results_text.insert(
                tk.END,
                "No significant phase transitions detected. This might indicate a solid solution behavior rather than distinct phase changes.\n",
            )

        # Apply styles
        self.results_text.tag_configure(
            "title", font=("Arial", 14, "bold"), justify="center"
        )
        self.results_text.tag_configure("heading", font=("Arial", 11, "bold"))

        # Disable editing
        self.results_text.config(state=tk.DISABLED)

    def _run_fade_analysis(self):
        """Run capacity fade analysis."""
        test_id = str(self.current_test.id)

        self.main_app.log_message(
            f"Running capacity fade analysis for test {self.current_test.name}"
        )

        try:
            # Run capacity fade analysis
            self.current_fade_analysis = advanced_analysis.capacity_fade_analysis(
                test_id
            )

            # Plot the results
            self.plot_fade_results()

            # Update the results text
            self.update_fade_results_text()

            # Switch to the plot tab
            self.notebook.select(0)

            self.main_app.log_message(f"Capacity fade analysis complete")

        except Exception as e:
            self.main_app.log_message(
                f"Error in capacity fade analysis: {str(e)}", logging.ERROR
            )
            raise

    def plot_fade_results(self):
        """Plot capacity fade analysis results."""
        if not self.current_fade_analysis:
            return

        # Clear the figure
        self.fig.clear()

        # Create a 2x1 grid
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1])

        # Extract data
        cycle_count = self.current_test.cycle_count
        cycles = np.array([c.cycle_index for c in self.current_test.cycles])
        capacities = np.array([c.discharge_capacity for c in self.current_test.cycles])

        # Normalize capacities
        normalized_caps = capacities / capacities[0] * 100

        # Top plot: Capacity vs Cycle with fits
        ax1 = self.fig.add_subplot(gs[0])
        ax1.plot(cycles, capacities, "o", label="Measured Data")
        ax1.set_xlabel("Cycle Number")
        ax1.set_ylabel("Discharge Capacity (mAh)")
        ax1.set_title("Capacity Fade Analysis")
        ax1.grid(True, linestyle="--", alpha=0.7)

        # Plot the fitted models
        if "fade_models" in self.current_fade_analysis:
            fade_models = self.current_fade_analysis["fade_models"]

            for model_name, model_data in fade_models.items():
                if "error" in model_data:
                    continue  # Skip failed models

                if model_name == "linear" and "params" in model_data:
                    a, b = (
                        model_data["params"]["slope"],
                        model_data["params"]["intercept"],
                    )
                    x_fit = np.linspace(min(cycles), max(cycles) * 2, 100)
                    y_fit = a * x_fit + b
                    ax1.plot(
                        x_fit,
                        y_fit,
                        "-",
                        label=f'Linear (R²={model_data["r_squared"]:.3f})',
                    )

                elif model_name == "power" and "params" in model_data:
                    a, b, c = (
                        model_data["params"]["a"],
                        model_data["params"]["b"],
                        model_data["params"]["c"],
                    )
                    x_fit = np.linspace(min(cycles), max(cycles) * 2, 100)
                    y_fit = a * np.power(x_fit, b) + c
                    ax1.plot(
                        x_fit,
                        y_fit,
                        "-",
                        label=f'Power (R²={model_data["r_squared"]:.3f})',
                    )

                elif model_name == "exponential" and "params" in model_data:
                    a, b, c = (
                        model_data["params"]["a"],
                        model_data["params"]["b"],
                        model_data["params"]["c"],
                    )
                    x_fit = np.linspace(min(cycles), max(cycles) * 2, 100)
                    y_fit = a * np.exp(b * x_fit) + c
                    ax1.plot(
                        x_fit,
                        y_fit,
                        "-",
                        label=f'Exponential (R²={model_data["r_squared"]:.3f})',
                    )

        ax1.legend(loc="best")

        # Bottom plot: Normalized Capacity vs Cycle
        ax2 = self.fig.add_subplot(gs[1])
        ax2.plot(cycles, normalized_caps, "o", label="Measured Data")
        ax2.set_xlabel("Cycle Number")
        ax2.set_ylabel("Normalized Capacity (%)")
        ax2.set_title("Normalized Capacity Fade")
        ax2.grid(True, linestyle="--", alpha=0.7)

        # Add prediction to 80% if available
        if self.current_fade_analysis["predicted_eol_cycle"]:
            eol_cycle = self.current_fade_analysis["predicted_eol_cycle"]
            ax2.axvline(x=eol_cycle, color="r", linestyle="--")
            ax2.axhline(y=80, color="r", linestyle="--")
            ax2.plot(eol_cycle, 80, "ro", markersize=8)
            ax2.annotate(
                f"EOL at cycle {int(eol_cycle)}",
                (eol_cycle, 80),
                xytext=(30, -20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

            # Add confidence level if available
            if self.current_fade_analysis["confidence"]:
                conf = self.current_fade_analysis["confidence"] * 100
                ax2.annotate(
                    f"Confidence: {conf:.1f}%",
                    (eol_cycle, 80),
                    xytext=(30, -40),
                    textcoords="offset points",
                )

        # Add a horizontal line at 80% for reference
        ax2.axhline(y=80, color="gray", linestyle="--", alpha=0.5)

        # Adjust layout
        self.fig.tight_layout()
        self.canvas.draw()

    def update_fade_results_text(self):
        """Update the results text with fade analysis information."""
        if not self.current_fade_analysis:
            return

        # Enable editing
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        # Add title
        self.results_text.insert(tk.END, "Capacity Fade Analysis Results\n\n", "title")

        # Test info
        self.results_text.insert(
            tk.END, f"Sample: {self.current_sample.name}\n", "heading"
        )
        self.results_text.insert(tk.END, f"Test: {self.current_test.name}\n", "heading")
        self.results_text.insert(
            tk.END,
            f"Cycles: {self.current_fade_analysis['cycle_count']}\n\n",
            "heading",
        )

        # Basic metrics
        self.results_text.insert(tk.END, "Capacity Metrics\n", "heading")
        self.results_text.insert(
            tk.END,
            f"Initial Capacity: {self.current_fade_analysis['initial_capacity']:.2f} mAh\n",
        )
        self.results_text.insert(
            tk.END,
            f"Final Capacity: {self.current_fade_analysis['final_capacity']:.2f} mAh\n",
        )
        self.results_text.insert(
            tk.END,
            f"Capacity Retention: {self.current_fade_analysis['capacity_retention'] * 100:.2f}%\n",
        )
        self.results_text.insert(
            tk.END,
            f"Fade Rate: {self.current_fade_analysis['fade_rate_pct_per_cycle']:.4f}% per cycle\n\n",
        )

        # Model information
        if (
            "fade_models" in self.current_fade_analysis
            and len(self.current_fade_analysis["fade_models"]) > 0
        ):
            self.results_text.insert(tk.END, "Fade Models\n", "heading")

            for model_name, model_data in self.current_fade_analysis[
                "fade_models"
            ].items():
                if "error" in model_data:
                    self.results_text.insert(
                        tk.END,
                        f"{model_data['name']}: Failed to fit ({model_data['error']})\n",
                    )
                    continue

                self.results_text.insert(
                    tk.END, f"{model_data['name']} Model\n", "subheading"
                )
                self.results_text.insert(
                    tk.END, f"R² Value: {model_data['r_squared']:.4f}\n"
                )

                # Parameters
                self.results_text.insert(tk.END, "Parameters:\n")
                for param_name, param_value in model_data["params"].items():
                    self.results_text.insert(
                        tk.END, f"  {param_name}: {param_value:.6e}\n"
                    )

                # EOL prediction
                if model_data["eol_cycle"] is not None:
                    self.results_text.insert(
                        tk.END,
                        f"Predicted EOL Cycle (80%): {int(model_data['eol_cycle'])}\n",
                    )

                self.results_text.insert(tk.END, "\n")

            # Best model information
            if self.current_fade_analysis["best_model"]:
                best_model = self.current_fade_analysis["best_model"]
                best_model_data = self.current_fade_analysis["fade_models"][best_model]

                self.results_text.insert(tk.END, "Best Model\n", "heading")
                self.results_text.insert(tk.END, f"Model: {best_model_data['name']}\n")
                self.results_text.insert(
                    tk.END, f"R²: {best_model_data['r_squared']:.4f}\n"
                )

                if self.current_fade_analysis["predicted_eol_cycle"]:
                    self.results_text.insert(
                        tk.END,
                        f"Predicted EOL Cycle (80%): {int(self.current_fade_analysis['predicted_eol_cycle'])}\n",
                    )

                    if self.current_fade_analysis["confidence"]:
                        self.results_text.insert(
                            tk.END,
                            f"Prediction Confidence: {self.current_fade_analysis['confidence'] * 100:.1f}%\n",
                        )

        # Interpretation
        self.results_text.insert(tk.END, "\nInterpretation\n", "heading")

        # Fade rate interpretation
        fade_rate = self.current_fade_analysis["fade_rate_pct_per_cycle"]
        if fade_rate < 0.01:
            self.results_text.insert(
                tk.END,
                "Very low fade rate: <0.01% per cycle. Excellent cycling stability.\n",
            )
        elif fade_rate < 0.05:
            self.results_text.insert(
                tk.END,
                f"Low fade rate: {fade_rate:.4f}% per cycle. Good cycling stability.\n",
            )
        elif fade_rate < 0.1:
            self.results_text.insert(
                tk.END,
                f"Moderate fade rate: {fade_rate:.4f}% per cycle. Acceptable performance.\n",
            )
        else:
            self.results_text.insert(
                tk.END,
                f"High fade rate: {fade_rate:.4f}% per cycle. Significant capacity loss over cycling.\n",
            )

        # Best model interpretation
        if self.current_fade_analysis["best_model"]:
            best_model = self.current_fade_analysis["best_model"]

            if best_model == "linear":
                self.results_text.insert(
                    tk.END,
                    "Linear fade model fits best: Constant capacity loss per cycle.\n",
                )
            elif best_model == "power":
                self.results_text.insert(
                    tk.END,
                    "Power law model fits best: Fade rate decreases over time, indicating stabilization.\n",
                )
            elif best_model == "exponential":
                self.results_text.insert(
                    tk.END,
                    "Exponential model fits best: Initial rapid fade followed by slower fade rate.\n",
                )

        # Cycle life prediction
        if self.current_fade_analysis["predicted_eol_cycle"]:
            eol = int(self.current_fade_analysis["predicted_eol_cycle"])
            completed = self.current_fade_analysis["cycle_count"]

            if eol <= completed:
                self.results_text.insert(
                    tk.END,
                    f"Cell has already reached end-of-life criterion (80% capacity).\n",
                )
            else:
                self.results_text.insert(
                    tk.END,
                    f"Projected to reach 80% capacity at cycle {eol} "
                    + f"({eol - completed} more cycles from current state).\n",
                )

                if self.current_fade_analysis["confidence"]:
                    conf = self.current_fade_analysis["confidence"] * 100
                    if conf > 80:
                        self.results_text.insert(
                            tk.END, f"High prediction confidence: {conf:.1f}%\n"
                        )
                    elif conf > 50:
                        self.results_text.insert(
                            tk.END, f"Medium prediction confidence: {conf:.1f}%\n"
                        )
                    else:
                        self.results_text.insert(
                            tk.END,
                            f"Low prediction confidence: {conf:.1f}% - Take prediction with caution\n",
                        )

        # Apply styles
        self.results_text.tag_configure(
            "title", font=("Arial", 14, "bold"), justify="center"
        )
        self.results_text.tag_configure("heading", font=("Arial", 11, "bold"))
        self.results_text.tag_configure("subheading", font=("Arial", 10, "bold"))

        # Disable editing
        self.results_text.config(state=tk.DISABLED)

    def _run_anomaly_detection(self):
        """Run anomaly detection analysis."""
        test_id = str(self.current_test.id)

        # Get options
        metric = self.metric_var.get()

        try:
            threshold = float(self.threshold_entry.get())
        except ValueError:
            threshold = 3.0

        self.main_app.log_message(
            f"Running anomaly detection for test {self.current_test.name}, "
            + f"metric={metric}, threshold={threshold}"
        )

        try:
            # Run anomaly detection
            anomaly_results = advanced_analysis.detect_anomalies(
                test_id, metric, threshold
            )

            # Plot the results
            self.plot_anomaly_results(anomaly_results, metric)

            # Update the results text
            self.update_anomaly_results_text(anomaly_results, metric)

            # Switch to the plot tab
            self.notebook.select(0)

            self.main_app.log_message(f"Anomaly detection complete")

        except Exception as e:
            self.main_app.log_message(
                f"Error in anomaly detection: {str(e)}", logging.ERROR
            )
            raise

    def plot_anomaly_results(self, anomaly_results, metric):
        """Plot anomaly detection results."""
        # Get cycle data
        cycle_data = analysis.get_cycle_data(str(self.current_test.id))

        # Extract data for plotting
        cycles = [c["cycle_index"] for c in cycle_data["cycles"]]

        if metric == "discharge_capacity":
            values = [c["discharge_capacity"] for c in cycle_data["cycles"]]
            y_label = "Discharge Capacity (mAh)"
        elif metric == "charge_capacity":
            values = [c["charge_capacity"] for c in cycle_data["cycles"]]
            y_label = "Charge Capacity (mAh)"
        elif metric == "coulombic_efficiency":
            values = [c["coulombic_efficiency"] * 100 for c in cycle_data["cycles"]]
            y_label = "Coulombic Efficiency (%)"

        # Clear the figure
        self.fig.clear()

        # Create a single plot
        ax = self.fig.add_subplot(111)

        # Plot the metric values
        ax.plot(cycles, values, "o-", label="Data", zorder=1)

        # Highlight anomalies
        anomaly_cycles = [a["cycle"] for a in anomaly_results["anomalies"]]
        anomaly_values = []

        for cycle in anomaly_cycles:
            idx = cycles.index(cycle) if cycle in cycles else -1
            if idx >= 0:
                anomaly_values.append(values[idx])

        if anomaly_cycles:
            ax.plot(
                anomaly_cycles,
                anomaly_values,
                "ro",
                markersize=10,
                label="Anomalies",
                zorder=2,
            )

            # Annotate anomalies
            for cycle, value in zip(anomaly_cycles, anomaly_values):
                ax.annotate(
                    f"Cycle {cycle}",
                    (cycle, value),
                    xytext=(10, 10),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", lw=1.5),
                )

        # Add mean and threshold lines
        mean_value = anomaly_results["normal_range"]["mean"]
        std_dev = anomaly_results["normal_range"]["std_dev"]
        threshold = anomaly_results["normal_range"]["threshold"]

        ax.axhline(y=mean_value, color="green", linestyle="--", label="Mean")
        ax.axhline(y=mean_value + threshold, color="red", linestyle="--", label="+3σ")
        ax.axhline(y=mean_value - threshold, color="red", linestyle="--", label="-3σ")

        # Set plot properties
        ax.set_xlabel("Cycle Number")
        ax.set_ylabel(y_label)
        ax.set_title(f"Anomaly Detection - {metric}")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="best")

        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def update_anomaly_results_text(self, anomaly_results, metric):
        """Update the results text with anomaly detection information."""
        # Enable editing
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        # Add title
        self.results_text.insert(tk.END, "Anomaly Detection Results\n\n", "title")

        # Test info
        self.results_text.insert(
            tk.END, f"Sample: {self.current_sample.name}\n", "heading"
        )
        self.results_text.insert(tk.END, f"Test: {self.current_test.name}\n", "heading")

        # Metric info
        if metric == "discharge_capacity":
            metric_name = "Discharge Capacity"
        elif metric == "charge_capacity":
            metric_name = "Charge Capacity"
        elif metric == "coulombic_efficiency":
            metric_name = "Coulombic Efficiency"
        else:
            metric_name = metric

        self.results_text.insert(
            tk.END, f"Analyzed Metric: {metric_name}\n\n", "heading"
        )

        # Normal range information
        mean_value = anomaly_results["normal_range"]["mean"]
        std_dev = anomaly_results["normal_range"]["std_dev"]
        threshold = anomaly_results["normal_range"]["threshold"]

        self.results_text.insert(tk.END, "Normal Range Parameters\n", "heading")
        self.results_text.insert(tk.END, f"Mean: {mean_value:.3f}\n")
        self.results_text.insert(tk.END, f"Standard Deviation: {std_dev:.3f}\n")
        self.results_text.insert(tk.END, f"Detection Threshold: {threshold:.3f}\n")
        self.results_text.insert(
            tk.END,
            f"Valid Range: {mean_value - threshold:.3f} to {mean_value + threshold:.3f}\n\n",
        )

        # Anomaly information
        self.results_text.insert(tk.END, "Detected Anomalies\n", "heading")

        if anomaly_results["anomaly_count"] == 0:
            self.results_text.insert(tk.END, "No anomalies detected.\n\n")
        else:
            self.results_text.insert(
                tk.END, f"Total Anomalies: {anomaly_results['anomaly_count']}\n\n"
            )

            # Table header
            self.results_text.insert(
                tk.END, "Cycle\tValue\tMethod\tSignificance\n", "table_header"
            )

            # Table rows
            for anomaly in anomaly_results["anomalies"]:
                self.results_text.insert(
                    tk.END,
                    f"{anomaly['cycle']}\t{anomaly['value']:.3f}\t{anomaly['detection_method']}\t{anomaly['significance']:.3f}\n",
                )

            self.results_text.insert(tk.END, "\n")

        # Interpretation
        self.results_text.insert(tk.END, "Interpretation\n", "heading")

        if anomaly_results["anomaly_count"] == 0:
            self.results_text.insert(
                tk.END,
                f"No anomalies detected in the {metric_name.lower()} data. "
                + f"The cell shows consistent behavior throughout cycling.\n",
            )
        else:
            # Group anomalies by method
            z_score_anomalies = [
                a
                for a in anomaly_results["anomalies"]
                if a["detection_method"] == "z-score"
            ]
            mavg_anomalies = [
                a
                for a in anomaly_results["anomalies"]
                if a["detection_method"] == "moving_average"
            ]

            if z_score_anomalies:
                self.results_text.insert(
                    tk.END,
                    f"Found {len(z_score_anomalies)} statistical outliers using z-score method. "
                    + f"These are cycles that deviate significantly from the average behavior.\n\n",
                )

            if mavg_anomalies:
                self.results_text.insert(
                    tk.END,
                    f"Found {len(mavg_anomalies)} anomalies using moving average method. "
                    + f"These represent sudden changes in performance compared to recent cycles.\n\n",
                )

            # Potential causes
            self.results_text.insert(tk.END, "Potential Causes for Anomalies:\n")
            self.results_text.insert(
                tk.END, "- Measurement errors or equipment issues\n"
            )
            self.results_text.insert(
                tk.END, "- Temperature fluctuations during testing\n"
            )
            self.results_text.insert(
                tk.END, "- Cell internal short circuits or other failures\n"
            )
            self.results_text.insert(tk.END, "- Changes in testing conditions\n")
            self.results_text.insert(
                tk.END,
                "- Actual cell degradation events (SEI breakdown, lithium plating, etc.)\n\n",
            )

            # Recommendations
            self.results_text.insert(tk.END, "Recommendations:\n")
            self.results_text.insert(
                tk.END, "- Investigate cycles with anomalous behavior\n"
            )
            self.results_text.insert(
                tk.END,
                "- Consider excluding anomalous cycles when modeling capacity fade\n",
            )
            self.results_text.insert(
                tk.END, "- Check testing equipment and conditions for consistency\n"
            )

        # Apply styles
        self.results_text.tag_configure(
            "title", font=("Arial", 14, "bold"), justify="center"
        )
        self.results_text.tag_configure("heading", font=("Arial", 11, "bold"))
        self.results_text.tag_configure("subheading", font=("Arial", 10, "bold"))
        self.results_text.tag_configure("table_header", font=("Courier", 10, "bold"))

        # Disable editing
        self.results_text.config(state=tk.DISABLED)

    def _run_energy_analysis(self):
        """Run energy analysis."""
        test_id = str(self.current_test.id)

        # Get cell weight and volume if provided
        try:
            weight = float(self.weight_entry.get())
            weight = weight if weight > 0 else None
        except ValueError:
            weight = None

        try:
            volume = float(self.volume_entry.get())
            volume = volume if volume > 0 else None
        except ValueError:
            volume = None

        self.main_app.log_message(
            f"Running energy analysis for test {self.current_test.name}"
        )

        try:
            # Run energy analysis
            energy_results = advanced_analysis.energy_analysis(test_id)

            # If weight/volume provided, calculate energy density
            if weight or volume:
                # Get the initial discharge energy
                if (
                    "initial_discharge_energy" in energy_results
                    and energy_results["initial_discharge_energy"]
                ):
                    energy_wh = energy_results["initial_discharge_energy"]

                    # Calculate energy density
                    densities = utils.calculate_energy_density(
                        energy_results["initial_discharge_capacity"],
                        energy_wh / energy_results["initial_discharge_capacity"],
                        weight,
                        volume,
                    )

                    # Add to results
                    energy_results["energy_density"].update(densities)

            # Plot the results
            self.plot_energy_results(energy_results)

            # Update the results text
            self.update_energy_results_text(energy_results)

            # Switch to the plot tab
            self.notebook.select(0)

            self.main_app.log_message(f"Energy analysis complete")

        except Exception as e:
            self.main_app.log_message(
                f"Error in energy analysis: {str(e)}", logging.ERROR
            )
            raise

    def plot_energy_results(self, energy_results):
        """Plot energy analysis results."""
        # Get cycle data
        cycle_data = analysis.get_cycle_data(str(self.current_test.id))

        # Extract data for plotting
        cycles = [c["cycle_index"] for c in cycle_data["cycles"]]

        # Extract energy values if available
        discharge_energy = []
        energy_efficiency = []

        for c in cycle_data["cycles"]:
            if "discharge_energy" in c and c["discharge_energy"] is not None:
                discharge_energy.append(c["discharge_energy"])
            else:
                # Estimate from capacity if energy not directly available
                # Assuming average voltage of 3.7V for Li-ion
                discharge_energy.append(
                    c["discharge_capacity"] * 3.7 / 1000
                )  # Convert to Wh

            if "energy_efficiency" in c and c["energy_efficiency"] is not None:
                energy_efficiency.append(
                    c["energy_efficiency"] * 100
                )  # Convert to percentage
            else:
                # Estimate from coulombic efficiency if energy efficiency not available
                # Energy efficiency is typically a bit lower than coulombic efficiency
                energy_efficiency.append(c["coulombic_efficiency"] * 95)  # 95% of CE

        # Clear the figure
        self.fig.clear()

        # Create a 2x1 grid
        gs = self.fig.add_gridspec(2, 1)

        # Top plot: Discharge Energy
        ax1 = self.fig.add_subplot(gs[0])
        ax1.plot(cycles, discharge_energy, "o-", color="blue")
        ax1.set_xlabel("Cycle Number")
        ax1.set_ylabel("Discharge Energy (Wh)")
        ax1.set_title("Discharge Energy vs. Cycle Number")
        ax1.grid(True, linestyle="--", alpha=0.7)

        # Add initial and final values annotation
        if discharge_energy:
            ax1.annotate(
                f"Initial: {discharge_energy[0]:.3f} Wh",
                xy=(cycles[0], discharge_energy[0]),
                xytext=(10, 10),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

            ax1.annotate(
                f"Final: {discharge_energy[-1]:.3f} Wh",
                xy=(cycles[-1], discharge_energy[-1]),
                xytext=(10, 10),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

        # Bottom plot: Energy Efficiency
        ax2 = self.fig.add_subplot(gs[1])
        ax2.plot(cycles, energy_efficiency, "o-", color="green")
        ax2.set_xlabel("Cycle Number")
        ax2.set_ylabel("Energy Efficiency (%)")
        ax2.set_title("Energy Efficiency vs. Cycle Number")
        ax2.grid(True, linestyle="--", alpha=0.7)

        # Set y-axis range to highlight efficiency differences
        if energy_efficiency:
            min_eff = min(80, min(energy_efficiency) - 5)
            ax2.set_ylim(min_eff, 100)

        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def update_energy_results_text(self, energy_results):
        """Update the results text with energy analysis information."""
        # Enable editing
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        # Add title
        self.results_text.insert(tk.END, "Energy Analysis Results\n\n", "title")

        # Test info
        self.results_text.insert(
            tk.END, f"Sample: {self.current_sample.name}\n", "heading"
        )
        self.results_text.insert(
            tk.END, f"Test: {self.current_test.name}\n\n", "heading"
        )

        # Energy metrics
        self.results_text.insert(tk.END, "Energy Metrics\n", "heading")

        initial_charge = energy_results.get("initial_charge_energy")
        if initial_charge is not None:
            self.results_text.insert(
                tk.END, f"Initial Charge Energy: {initial_charge:.3f} Wh\n"
            )

        final_charge = energy_results.get("final_charge_energy")
        if final_charge is not None:
            self.results_text.insert(
                tk.END, f"Final Charge Energy: {final_charge:.3f} Wh\n"
            )

        initial_discharge = energy_results.get("initial_discharge_energy")
        if initial_discharge is not None:
            self.results_text.insert(
                tk.END, f"Initial Discharge Energy: {initial_discharge:.3f} Wh\n"
            )

        final_discharge = energy_results.get("final_discharge_energy")
        if final_discharge is not None:
            self.results_text.insert(
                tk.END, f"Final Discharge Energy: {final_discharge:.3f} Wh\n"
            )

        avg_charge = energy_results.get("avg_charge_energy")
        if avg_charge is not None:
            self.results_text.insert(
                tk.END, f"Average Charge Energy: {avg_charge:.3f} Wh\n"
            )

        avg_discharge = energy_results.get("avg_discharge_energy")
        if avg_discharge is not None:
            self.results_text.insert(
                tk.END, f"Average Discharge Energy: {avg_discharge:.3f} Wh\n"
            )

        self.results_text.insert(tk.END, "\n")

        # Efficiency metrics
        self.results_text.insert(tk.END, "Efficiency Metrics\n", "heading")

        avg_efficiency = energy_results.get("avg_energy_efficiency")
        if avg_efficiency is not None:
            self.results_text.insert(
                tk.END, f"Average Energy Efficiency: {avg_efficiency * 100:.2f}%\n"
            )

        retention = energy_results.get("energy_retention")
        if retention is not None:
            self.results_text.insert(
                tk.END, f"Energy Retention: {retention * 100:.2f}%\n"
            )

        fade_rate = energy_results.get("energy_fade_rate_pct_per_cycle")
        if fade_rate is not None:
            self.results_text.insert(
                tk.END, f"Energy Fade Rate: {fade_rate:.4f}% per cycle\n"
            )

        self.results_text.insert(tk.END, "\n")

        # Energy density
        energy_density = energy_results.get("energy_density", {})
        if energy_density:
            self.results_text.insert(tk.END, "Energy Density\n", "heading")

            grav_density = energy_density.get("gravimetric")
            if grav_density is not None:
                self.results_text.insert(
                    tk.END, f"Gravimetric Energy Density: {grav_density:.1f} Wh/kg\n"
                )

            vol_density = energy_density.get("volumetric")
            if vol_density is not None:
                self.results_text.insert(
                    tk.END, f"Volumetric Energy Density: {vol_density:.1f} Wh/L\n"
                )

            self.results_text.insert(tk.END, "\n")

        # Interpretation
        self.results_text.insert(tk.END, "Interpretation\n", "heading")

        # Interpret energy efficiency
        if avg_efficiency is not None:
            eff_pct = avg_efficiency * 100
            if eff_pct > 95:
                self.results_text.insert(
                    tk.END,
                    f"Very high energy efficiency ({eff_pct:.1f}%): Excellent conversion with minimal losses.\n",
                )
            elif eff_pct > 90:
                self.results_text.insert(
                    tk.END,
                    f"High energy efficiency ({eff_pct:.1f}%): Good performance with low energy losses.\n",
                )
            elif eff_pct > 85:
                self.results_text.insert(
                    tk.END,
                    f"Moderate energy efficiency ({eff_pct:.1f}%): Typical for many Li-ion cells.\n",
                )
            else:
                self.results_text.insert(
                    tk.END,
                    f"Low energy efficiency ({eff_pct:.1f}%): Significant energy losses during cycling.\n",
                )

        # Interpret energy density
        if "gravimetric" in energy_density:
            grav_density = energy_density["gravimetric"]
            if grav_density > 250:
                self.results_text.insert(
                    tk.END,
                    f"High gravimetric energy density ({grav_density:.1f} Wh/kg): Excellent for portable applications.\n",
                )
            elif grav_density > 150:
                self.results_text.insert(
                    tk.END,
                    f"Moderate gravimetric energy density ({grav_density:.1f} Wh/kg): Good for mobile applications.\n",
                )
            else:
                self.results_text.insert(
                    tk.END,
                    f"Low gravimetric energy density ({grav_density:.1f} Wh/kg): Better suited for stationary applications.\n",
                )

        # Apply styles
        self.results_text.tag_configure(
            "title", font=("Arial", 14, "bold"), justify="center"
        )
        self.results_text.tag_configure("heading", font=("Arial", 11, "bold"))
        self.results_text.tag_configure("subheading", font=("Arial", 10, "bold"))

        # Disable editing
        self.results_text.config(state=tk.DISABLED)

    def _run_clustering_analysis(self):
        """Run test clustering analysis."""
        # This requires multiple tests to be meaningful

        try:
            # Get test IDs for the current sample
            test_ids = []

            # If sample has enough tests, run direct clustering
            sample_tests = models.TestResult.objects(sample=self.current_sample)

            if len(sample_tests) < 5:
                # Not enough tests in this sample, look for more samples
                # Get all samples
                all_samples = models.Sample.objects().all()

                test_count = 0
                for sample in all_samples:
                    sample_test_ids = [
                        str(t.id) for t in models.TestResult.objects(sample=sample)
                    ]
                    test_ids.extend(sample_test_ids)
                    test_count += len(sample_test_ids)

                if test_count < 5:
                    messagebox.showinfo(
                        "Insufficient Data",
                        "Clustering analysis requires at least 5 tests in total. "
                        + f"Found only {test_count} tests across all samples.",
                    )
                    return
            else:
                # Use tests from the current sample only
                for test in sample_tests:
                    test_ids.append(str(test.id))

            # Get clustering options
            metric = self.cluster_metric_var.get()
            method = self.method_var.get()

            try:
                n_clusters = int(self.n_clusters_entry.get())
            except ValueError:
                n_clusters = 3

            self.main_app.log_message(
                f"Running test clustering analysis with {len(test_ids)} tests, "
                + f"metric={metric}, method={method}, n_clusters={n_clusters}"
            )

            # Run clustering analysis
            cluster_results = advanced_analysis.cluster_tests(
                test_ids, metrics=[metric], method=method, n_clusters=n_clusters
            )

            # Plot the results
            self.plot_clustering_results(cluster_results)

            # Update the results text
            self.update_clustering_results_text(cluster_results)

            # Switch to the plot tab
            self.notebook.select(0)

            self.main_app.log_message(f"Clustering analysis complete")

        except Exception as e:
            self.main_app.log_message(
                f"Error in clustering analysis: {str(e)}", logging.ERROR
            )
            raise

    def plot_clustering_results(self, cluster_results):
        """Plot clustering analysis results."""
        # Clear the figure
        self.fig.clear()

        # Create a single plot
        ax = self.fig.add_subplot(111)

        # Prepare data for plotting
        test_info = cluster_results["tests"]
        metrics = cluster_results["metrics_used"]
        primary_metric = metrics[0] if metrics else "unknown_metric"

        # Try to perform PCA to visualize in 2D
        try:
            # Extract coordinates from PCA
            principal_components = cluster_results.get("principal_components", [])

            # If no PCA components in results, create artificial x,y coordinates
            if not principal_components or len(principal_components) < 1:
                # Create a simple scatter based on cluster assignments
                x_coords = []
                y_coords = []

                for i, test in enumerate(test_info):
                    x_coords.append(i % 5)  # Arbitrary distribution
                    y_coords.append(i // 5)

                principal_components = list(zip(x_coords, y_coords))

            # Plot each test as a point, colored by cluster
            clusters = set(test["cluster"] for test in test_info)
            colors = plt.cm.tab10.colors

            for cluster_id in clusters:
                cluster_tests = [
                    test for test in test_info if test.get("cluster") == cluster_id
                ]

                # Get x, y coordinates for each test in this cluster
                x_vals = []
                y_vals = []
                labels = []

                for test in cluster_tests:
                    test_idx = test_info.index(test)
                    if test_idx < len(principal_components):
                        if len(principal_components[test_idx]) >= 2:
                            x_vals.append(principal_components[test_idx][0])
                            y_vals.append(principal_components[test_idx][1])
                            labels.append(test.get("test_name", f"Test {test_idx}"))

                # Plot this cluster
                ax.scatter(
                    x_vals,
                    y_vals,
                    color=colors[cluster_id % len(colors)],
                    label=f"Cluster {cluster_id}",
                    s=100,
                )

                # Add labels
                for i, label in enumerate(labels):
                    ax.annotate(
                        label,
                        (x_vals[i], y_vals[i]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )

            # Set plot properties
            ax.set_title(f"Test Clustering by {primary_metric}")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.legend(loc="best")
            ax.grid(True, linestyle="--", alpha=0.7)

        except Exception as e:
            # Fallback visualization if PCA fails
            ax.text(
                0.5,
                0.5,
                f"Could not create cluster visualization: {str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def update_clustering_results_text(self, cluster_results):
        """Update the results text with clustering information."""
        # Enable editing
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        # Add title
        self.results_text.insert(tk.END, "Test Clustering Results\n\n", "title")

        # Clustering info
        self.results_text.insert(tk.END, "Clustering Information\n", "heading")

        self.results_text.insert(tk.END, f"Method: {cluster_results['method']}\n")
        self.results_text.insert(
            tk.END, f"Number of Clusters: {cluster_results['n_clusters']}\n"
        )
        self.results_text.insert(
            tk.END, f"Metrics Used: {', '.join(cluster_results['metrics_used'])}\n"
        )
        self.results_text.insert(
            tk.END, f"Total Tests: {len(cluster_results['tests'])}\n\n"
        )

        # Cluster details
        self.results_text.insert(tk.END, "Cluster Details\n", "heading")

        for cluster_id, tests in cluster_results["clusters"].items():
            self.results_text.insert(
                tk.END, f"Cluster {cluster_id} ({len(tests)} tests)\n", "subheading"
            )

            for test in tests:
                self.results_text.insert(
                    tk.END,
                    f"  • {test['test_name']} (Sample: {test.get('sample_name', 'Unknown')})\n",
                )

            self.results_text.insert(tk.END, "\n")

        # Interpretation
        self.results_text.insert(tk.END, "Interpretation\n", "heading")
        self.results_text.insert(
            tk.END,
            "The clustering analysis groups tests with similar performance characteristics together. ",
        )
        self.results_text.insert(
            tk.END,
            "Tests in the same cluster exhibit similar behavior regarding the selected metrics.\n\n",
        )

        self.results_text.insert(tk.END, "Possible applications of these clusters:\n")
        self.results_text.insert(
            tk.END, "• Identifying groups of cells with similar performance\n"
        )
        self.results_text.insert(
            tk.END, "• Finding outliers that perform differently from the majority\n"
        )
        self.results_text.insert(
            tk.END, "• Categorizing cells by their performance characteristics\n"
        )
        self.results_text.insert(
            tk.END, "• Identifying manufacturing batches with consistent properties\n\n"
        )

        self.results_text.insert(tk.END, "Next steps:\n")
        self.results_text.insert(
            tk.END, "• Investigate what makes each cluster unique\n"
        )
        self.results_text.insert(
            tk.END,
            "• Compare physical or chemical properties of cells in different clusters\n",
        )
        self.results_text.insert(
            tk.END,
            "• Use clustering results to select representative cells for further testing\n",
        )

        # Apply styles
        self.results_text.tag_configure(
            "title", font=("Arial", 14, "bold"), justify="center"
        )
        self.results_text.tag_configure("heading", font=("Arial", 11, "bold"))
        self.results_text.tag_configure("subheading", font=("Arial", 10, "bold"))

        # Disable editing
        self.results_text.config(state=tk.DISABLED)

    def view_cycle_details(self):
        """View detailed data for a specific cycle."""
        if not self.current_test:
            messagebox.showinfo("No Test Selected", "Please select a test to analyze.")
            return

        # Ask which cycle to view
        cycle_dialog = tk.Toplevel(self)
        cycle_dialog.title("Select Cycle")
        cycle_dialog.geometry("300x150")
        cycle_dialog.transient(self.main_app)
        cycle_dialog.grab_set()

        ttk.Label(cycle_dialog, text="Enter cycle number to view:").pack(pady=(20, 5))

        cycle_var = tk.StringVar()
        cycle_entry = ttk.Entry(cycle_dialog, textvariable=cycle_var, width=10)
        cycle_entry.pack(pady=5)
        cycle_entry.insert(0, "1")

        def on_view():
            try:
                cycle_num = int(cycle_var.get())
                cycle_dialog.destroy()
                self.show_cycle_detail_window(str(self.current_test.id), cycle_num)
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid cycle number")

        ttk.Button(cycle_dialog, text="View", command=on_view).pack(pady=10)

    def show_cycle_detail_window(self, test_id, cycle_num):
        """Show a window with detailed cycle data."""
        try:
            CycleDetailViewer(self.main_app, test_id, cycle_num)
        except Exception as e:
            self.main_app.log_message(
                f"Error viewing cycle details: {str(e)}", logging.ERROR
            )
            messagebox.showerror("Error", f"Error viewing cycle details: {str(e)}")

    def browse_josh_file(self):
        """Prompt user to select an Excel file for the Josh_request_Dq_dv analysis."""
        file_path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
        )
        if file_path:
            self.josh_file_entry.delete(0, tk.END)
            self.josh_file_entry.insert(0, file_path)

    def _run_josh_request_dqdv(self):
        """Run the custom dQ/dV analysis requested by Josh."""
        file_path = self.josh_file_entry.get()
        sheet = self.josh_sheet_entry.get() or "Channel51_1"
        try:
            mass = float(self.josh_mass_entry.get())
        except ValueError:
            mass = 0.0015

        if not file_path:
            messagebox.showerror("Error", "Please select an Excel file")
            return

        try:
            df = advanced_analysis.josh_request_dq_dv(file_path, sheet, mass)

            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.plot(df["V"], df["dQdV_sm"], linewidth=1)
            ax.set_xlim(2.0, 3.7)
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("dQ/dV (mAh g$^{-1}$ V$^{-1}$)")
            ax.set_title("Smoothed Differential Capacity (Cycle 1 Charge)")
            ax.grid(True)

            self.canvas.draw()
            self.notebook.select(0)
        except Exception as e:
            self.main_app.log_message(
                f"Error running Josh_request_Dq_dv: {str(e)}", logging.ERROR
            )
            messagebox.showerror("Error", f"Failed to run analysis: {e}")
