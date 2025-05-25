"""
EIS Analysis tab for the battery analysis GUI.

This module provides a tab for Electrochemical Impedance Spectroscopy (EIS) analysis,
including importing, fitting, and visualizing EIS data.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd
import threading
import logging
import os

from battery_analysis import models
from battery_analysis import eis


class EISTab(ttk.Frame):
    """Tab for EIS analysis."""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.current_sample = None
        self.current_eis_data = None
        self.current_fit_result = None

        # Check if impedance.py is available
        self.has_impedance = eis.HAS_IMPEDANCE

        self.create_widgets()

    def create_widgets(self):
        """Create the widgets for the EIS tab."""
        # Create a paned window to allow resizable sections
        self.paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame for controls
        self.left_frame = ttk.Frame(self.paned_window, width=300)
        self.paned_window.add(self.left_frame, weight=1)

        # Right frame for visualization
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=3)

        # ===== Left Frame Contents =====
        # Data Source section
        data_source_frame = ttk.LabelFrame(self.left_frame, text="Data Source")
        data_source_frame.pack(fill=tk.X, padx=5, pady=5)

        # Radio buttons for data source
        self.data_source_var = tk.StringVar(value="file")
        ttk.Radiobutton(
            data_source_frame, text="Import from File", variable=self.data_source_var,
            value="file", command=self.toggle_data_source
        ).pack(anchor=tk.W, padx=5, pady=2)

        ttk.Radiobutton(
            data_source_frame, text="Load from Database", variable=self.data_source_var,
            value="database", command=self.toggle_data_source
        ).pack(anchor=tk.W, padx=5, pady=2)

        # File import frame
        self.file_frame = ttk.Frame(data_source_frame)
        self.file_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.file_frame, text="File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.file_entry = ttk.Entry(self.file_frame, width=20)
        self.file_entry.grid(row=0, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        self.browse_btn = ttk.Button(self.file_frame, text="Browse...", command=self.browse_file)
        self.browse_btn.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(self.file_frame, text="Associate with Sample:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.file_sample_combobox = ttk.Combobox(self.file_frame, width=20)
        self.file_sample_combobox.grid(row=1, column=1, columnspan=2, sticky=tk.W + tk.E, padx=5, pady=5)

        self.import_btn = ttk.Button(self.file_frame, text="Import", command=self.import_file)
        self.import_btn.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky=tk.E + tk.W)

        # Database selection frame (initially hidden)
        self.db_frame = ttk.Frame(data_source_frame)

        ttk.Label(self.db_frame, text="Sample:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.sample_combobox = ttk.Combobox(self.db_frame, width=20)
        self.sample_combobox.grid(row=0, column=1, sticky=tk.W + tk.E, padx=5, pady=5)
        self.sample_combobox.bind("<<ComboboxSelected>>", self.on_sample_selected)

        ttk.Button(
            self.db_frame, text="Refresh", command=self.refresh_samples
        ).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(self.db_frame, text="EIS Test:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.test_combobox = ttk.Combobox(self.db_frame, width=20)
        self.test_combobox.grid(row=1, column=1, columnspan=2, sticky=tk.W + tk.E, padx=5, pady=5)

        self.load_btn = ttk.Button(self.db_frame, text="Load Data", command=self.load_eis_data)
        self.load_btn.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky=tk.E + tk.W)

        # Initial state based on data source
        self.toggle_data_source()

        # Data Preprocessing section
        preprocess_frame = ttk.LabelFrame(self.left_frame, text="Data Preprocessing")
        preprocess_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(preprocess_frame, text="Min Frequency (Hz):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.fmin_entry = ttk.Entry(preprocess_frame, width=10)
        self.fmin_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(preprocess_frame, text="Max Frequency (Hz):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.fmax_entry = ttk.Entry(preprocess_frame, width=10)
        self.fmax_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(preprocess_frame, text="Filter Inductive Points:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.inductive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preprocess_frame, variable=self.inductive_var).grid(row=2, column=1, sticky=tk.W, padx=5,
                                                                            pady=5)

        ttk.Button(
            preprocess_frame, text="Process Data", command=self.preprocess_data
        ).grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.E + tk.W)

        # Visualization section
        viz_frame = ttk.LabelFrame(self.left_frame, text="Visualization")
        viz_frame.pack(fill=tk.X, padx=5, pady=5)

        self.plot_type_var = tk.StringVar(value="nyquist")
        ttk.Radiobutton(
            viz_frame, text="Nyquist Plot", variable=self.plot_type_var,
            value="nyquist", command=self.update_plot
        ).pack(anchor=tk.W, padx=5, pady=2)

        ttk.Radiobutton(
            viz_frame, text="Bode Plot", variable=self.plot_type_var,
            value="bode", command=self.update_plot
        ).pack(anchor=tk.W, padx=5, pady=2)

        if self.has_impedance:
            ttk.Radiobutton(
                viz_frame, text="DRT Plot", variable=self.plot_type_var,
                value="drt", command=self.update_plot
            ).pack(anchor=tk.W, padx=5, pady=2)

        self.highlight_freq_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            viz_frame, text="Highlight Frequencies",
            variable=self.highlight_freq_var,
            command=self.update_plot
        ).pack(anchor=tk.W, padx=5, pady=5)

        # Circuit Fitting section (only if impedance.py is available)
        if self.has_impedance:
            fitting_frame = ttk.LabelFrame(self.left_frame, text="Circuit Fitting")
            fitting_frame.pack(fill=tk.X, padx=5, pady=5)

            ttk.Label(fitting_frame, text="Circuit:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

            self.circuit_var = tk.StringVar(value="R0-p(R1,CPE1)-W2")
            common_circuits = [
                "R0",  # Simple resistor
                "R0-p(R1,C1)",  # Randles without diffusion
                "R0-p(R1,CPE1)",  # Randles with CPE
                "R0-p(R1,C1)-W2",  # Simple Randles with Warburg
                "R0-p(R1,CPE1)-W2",  # Randles with CPE and Warburg
                "R0-p(R1,C1)-p(R2,C2)",  # Two time constants
                "R0-p(R1,CPE1)-p(R2,CPE2)",  # Two time constants with CPEs
                "R0-p(R1,CPE1)-p(R2-W2,CPE2)"  # Complete battery model
            ]

            self.circuit_combobox = ttk.Combobox(fitting_frame, textvariable=self.circuit_var, width=25)
            self.circuit_combobox['values'] = common_circuits
            self.circuit_combobox.grid(row=0, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

            ttk.Label(fitting_frame, text="Custom Circuit:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
            self.custom_circuit_entry = ttk.Entry(fitting_frame, width=25)
            self.custom_circuit_entry.grid(row=1, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

            self.custom_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                fitting_frame, text="Use Custom Circuit",
                variable=self.custom_var
            ).grid(row=2, column=0, columnspan=2, padx=5, pady=5)

            ttk.Button(
                fitting_frame, text="Fit Circuit", command=self.fit_circuit
            ).grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.E + tk.W)

            # Export options
            ttk.Button(
                fitting_frame, text="Generate Report", command=self.generate_report
            ).grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky=tk.E + tk.W)

        # If impedance.py is not available, show a warning
        else:
            warning_frame = ttk.LabelFrame(self.left_frame, text="Circuit Fitting Unavailable")
            warning_frame.pack(fill=tk.X, padx=5, pady=5)

            ttk.Label(
                warning_frame,
                text="impedance.py package is not installed.\nInstall with: pip install impedance",
                foreground="red"
            ).pack(padx=10, pady=10)

        # ===== Right Frame Contents =====
        # Create a notebook for different views
        self.view_notebook = ttk.Notebook(self.right_frame)
        self.view_notebook.pack(fill=tk.BOTH, expand=True)

        # Plot tab
        self.plot_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.plot_frame, text="Plots")

        # Create matplotlib figure
        self.fig = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add a toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

        # Data tab
        self.data_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.data_frame, text="Data Table")

        # Create a table for the data
        self.data_table_frame = ttk.Frame(self.data_frame)
        self.data_table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create treeview for data table
        self.data_table = ttk.Treeview(
            self.data_table_frame,
            columns=("frequency", "z_real", "z_imag", "z_mag", "z_phase"),
            show="headings"
        )

        self.data_table.heading("frequency", text="Frequency (Hz)")
        self.data_table.heading("z_real", text="Z' (Ω)")
        self.data_table.heading("z_imag", text="Z\" (Ω)")
        self.data_table.heading("z_mag", text="|Z| (Ω)")
        self.data_table.heading("z_phase", text="Phase (°)")

        self.data_table.column("frequency", width=120, anchor=tk.E)
        self.data_table.column("z_real", width=120, anchor=tk.E)
        self.data_table.column("z_imag", width=120, anchor=tk.E)
        self.data_table.column("z_mag", width=120, anchor=tk.E)
        self.data_table.column("z_phase", width=120, anchor=tk.E)

        self.data_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add scrollbar to table
        table_scrollbar = ttk.Scrollbar(self.data_table_frame, orient=tk.VERTICAL, command=self.data_table.yview)
        table_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_table.config(yscrollcommand=table_scrollbar.set)

        # Results tab
        self.results_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.results_frame, text="Results")

        # Create a text widget for the results
        self.results_text = tk.Text(self.results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text.config(state=tk.DISABLED)

        # Add scrollbar to the results
        results_scrollbar = ttk.Scrollbar(self.results_text, orient=tk.VERTICAL, command=self.results_text.yview)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=results_scrollbar.set)

        # Parameters tab (for circuit fit results)
        if self.has_impedance:
            self.params_frame = ttk.Frame(self.view_notebook)
            self.view_notebook.add(self.params_frame, text="Parameters")

            # Create a frame for the parameters
            params_inner_frame = ttk.Frame(self.params_frame)
            params_inner_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Create a treeview for parameters
            self.params_table = ttk.Treeview(
                params_inner_frame,
                columns=("parameter", "value", "unit", "description"),
                show="headings"
            )

            self.params_table.heading("parameter", text="Parameter")
            self.params_table.heading("value", text="Value")
            self.params_table.heading("unit", text="Unit")
            self.params_table.heading("description", text="Description")

            self.params_table.column("parameter", width=100, anchor=tk.W)
            self.params_table.column("value", width=120, anchor=tk.E)
            self.params_table.column("unit", width=80, anchor=tk.W)
            self.params_table.column("description", width=200, anchor=tk.W)

            self.params_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Add scrollbar to table
            params_scrollbar = ttk.Scrollbar(params_inner_frame, orient=tk.VERTICAL, command=self.params_table.yview)
            params_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.params_table.config(yscrollcommand=params_scrollbar.set)

    def toggle_data_source(self):
        """Toggle between file and database data sources."""
        data_source = self.data_source_var.get()

        if data_source == "file":
            # Hide database frame, show file frame
            self.db_frame.pack_forget()
            self.file_frame.pack(fill=tk.X, padx=5, pady=5)

            # Refresh sample list for file association
            self.refresh_file_samples()
        else:
            # Hide file frame, show database frame
            self.file_frame.pack_forget()
            self.db_frame.pack(fill=tk.X, padx=5, pady=5)

            # Refresh samples in database
            self.refresh_samples()

    def refresh_file_samples(self):
        """Refresh the sample list for file association."""
        if not self.main_app.db_connected:
            self.file_sample_combobox['values'] = []
            return

        try:
            # Get all samples from the database
            samples = models.Sample.objects().all()
            sample_names = [sample.name for sample in samples]
            self.file_sample_combobox['values'] = sample_names

            if sample_names:
                self.file_sample_combobox.current(0)
        except Exception as e:
            self.main_app.log_message(f"Error loading samples for file association: {str(e)}", logging.ERROR)

    def refresh_samples(self):
        """Refresh the sample list from the database."""
        if not self.main_app.db_connected:
            messagebox.showwarning("Not Connected", "Please connect to the database first.")
            return

        try:
            # Get all samples from the database
            samples = models.Sample.objects().all()

            # Update the combobox
            sample_names = [sample.name for sample in samples]
            self.sample_combobox['values'] = sample_names

            if sample_names:
                self.sample_combobox.current(0)
                self.on_sample_selected(None)

            self.main_app.log_message(f"Loaded {len(sample_names)} samples for EIS analysis")
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
                messagebox.showerror("Sample Not Found", f"Sample '{sample_name}' not found in database.")
                return

            # Store the current sample
            self.current_sample = sample

            # Get EIS tests for this sample
            tests = []
            for test_ref in sample.tests:
                test = models.TestResult.objects(id=test_ref.id).first()
                if test and hasattr(test, 'test_type') and test.test_type == 'EIS':
                    tests.append(test)
                # Also check custom_data for EIS data
                elif test and hasattr(test, 'custom_data') and 'eis_data' in test.custom_data:
                    tests.append(test)

            # Update the test combobox
            test_names = [test.name for test in tests]
            self.test_combobox['values'] = test_names

            if test_names:
                self.test_combobox.current(0)
            else:
                self.test_combobox.set("")
                self.main_app.log_message(f"No EIS tests found for sample {sample_name}")

            self.main_app.log_message(f"Loaded {len(tests)} EIS tests for sample {sample_name}")
        except Exception as e:
            self.main_app.log_message(f"Error loading tests: {str(e)}", logging.ERROR)

    def browse_file(self):
        """Open a file dialog to select an EIS data file."""
        filetypes = [
            ("EIS Data Files", "*.csv *.txt *.dta *.z *.mpt *.dfr"),
            ("Text/CSV Files", "*.csv *.txt"),
            ("Gamry Files", "*.dta"),
            ("BioLogic Files", "*.z *.mpt"),
            ("Autolab Files", "*.dfr"),
            ("All Files", "*.*")
        ]

        file_path = filedialog.askopenfilename(
            title="Select EIS Data File",
            filetypes=filetypes
        )

        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def import_file(self):
        """Import an EIS file."""
        file_path = self.file_entry.get()

        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Invalid File", "Please select a valid EIS data file.")
            return

        # Get the selected sample if one is chosen
        sample_name = self.file_sample_combobox.get()
        sample_id = None

        if sample_name and self.main_app.db_connected:
            sample = models.Sample.objects(name=sample_name).first()
            if sample:
                sample_id = str(sample.id)

        self.main_app.update_status(f"Importing EIS file: {os.path.basename(file_path)}...")

        # Disable the import button to prevent multiple imports
        self.import_btn.config(state=tk.DISABLED)

        # Run import in a thread
        threading.Thread(
            target=self._import_file_thread,
            args=(file_path, sample_id),
            daemon=True
        ).start()

    def _import_file_thread(self, file_path, sample_id):
        """Thread function to import an EIS file."""
        try:
            # Import the file
            eis_data = eis.import_eis_data(file_path, sample_id)

            # Store the data
            self.current_eis_data = eis_data

            # Update the GUI
            self.main_app.update_status("EIS data imported successfully")
            self.main_app.log_message(f"Imported EIS data from {os.path.basename(file_path)}")

            # Update data table
            self.update_data_table()

            # Update plots
            self.update_plot()

            # Switch to the plot tab
            self.view_notebook.select(0)

        except Exception as e:
            self.main_app.log_message(f"Error importing EIS file: {str(e)}", logging.ERROR)
            messagebox.showerror("Import Error", f"Error importing EIS file: {str(e)}")
            self.main_app.update_status("Import failed")

        finally:
            # Re-enable the import button
            self.import_btn.config(state=tk.NORMAL)

    def load_eis_data(self):
        """Load EIS data from the database."""
        sample_name = self.sample_combobox.get()
        test_name = self.test_combobox.get()

        if not sample_name or not test_name:
            messagebox.showinfo("Selection Required", "Please select a sample and test.")
            return

        try:
            # Get the sample from the database
            sample = models.Sample.objects(name=sample_name).first()

            if not sample:
                messagebox.showerror("Sample Not Found", f"Sample '{sample_name}' not found in database.")
                return

            # Get the test from the database
            test = models.TestResult.objects(sample=sample, name=test_name).first()

            if not test:
                messagebox.showerror("Test Not Found", f"Test '{test_name}' not found for sample {sample_name}.")
                return

            self.main_app.update_status(f"Loading EIS data for test: {test_name}...")

            # Disable the load button to prevent multiple loads
            self.load_btn.config(state=tk.DISABLED)

            # Run load in a thread
            threading.Thread(
                target=self._load_eis_data_thread,
                args=(str(test.id),),
                daemon=True
            ).start()

        except Exception as e:
            self.main_app.log_message(f"Error loading EIS data: {str(e)}", logging.ERROR)
            messagebox.showerror("Load Error", f"Error loading EIS data: {str(e)}")
            self.main_app.update_status("Load failed")

    def _load_eis_data_thread(self, test_id):
        """Thread function to load EIS data from database."""
        try:
            # Load the data
            eis_data = eis.get_eis_data(test_id)

            # Store the data
            self.current_eis_data = eis_data

            # Update the GUI
            self.main_app.update_status("EIS data loaded successfully")
            self.main_app.log_message(f"Loaded EIS data for test ID: {test_id}")

            # Update data table
            self.update_data_table()

            # Update plots
            self.update_plot()

            # Switch to the plot tab
            self.view_notebook.select(0)

        except Exception as e:
            self.main_app.log_message(f"Error loading EIS data: {str(e)}", logging.ERROR)
            messagebox.showerror("Load Error", f"Error loading EIS data: {str(e)}")
            self.main_app.update_status("Load failed")

        finally:
            # Re-enable the load button
            self.load_btn.config(state=tk.NORMAL)

    def preprocess_data(self):
        """Preprocess the EIS data."""
        if not self.current_eis_data:
            messagebox.showinfo("No Data", "Please import or load EIS data first.")
            return

        # Get preprocessing options
        try:
            f_min = float(self.fmin_entry.get()) if self.fmin_entry.get() else None
        except ValueError:
            f_min = None

        try:
            f_max = float(self.fmax_entry.get()) if self.fmax_entry.get() else None
        except ValueError:
            f_max = None

        inductive_filter = self.inductive_var.get()

        # Preprocess the data
        try:
            frequency = self.current_eis_data['frequency']
            z_real = self.current_eis_data['Z_real']
            z_imag = self.current_eis_data['Z_imag']

            # Apply preprocessing
            frequency_proc, z_real_proc, z_imag_proc = eis.preprocess_eis_data(
                frequency, z_real, z_imag, f_min, f_max, inductive_filter
            )

            # Update the current data with processed data
            self.current_eis_data['frequency'] = frequency_proc
            self.current_eis_data['Z_real'] = z_real_proc
            self.current_eis_data['Z_imag'] = z_imag_proc
            self.current_eis_data['Z_mag'] = np.sqrt(z_real_proc ** 2 + z_imag_proc ** 2)
            self.current_eis_data['Z_phase'] = np.arctan2(z_imag_proc, z_real_proc) * (180 / np.pi)

            # If there was a previous fit, clear it
            self.current_fit_result = None

            # Update data table
            self.update_data_table()

            # Update plots
            self.update_plot()

            # Update status
            self.main_app.update_status("Data preprocessing complete")
            self.main_app.log_message(
                f"Preprocessed EIS data: {len(frequency_proc)} points " +
                f"(f_min={f_min}, f_max={f_max}, inductive_filter={inductive_filter})"
            )

        except Exception as e:
            self.main_app.log_message(f"Error preprocessing data: {str(e)}", logging.ERROR)
            messagebox.showerror("Preprocessing Error", f"Error preprocessing data: {str(e)}")
            self.main_app.update_status("Preprocessing failed")

    def update_data_table(self):
        """Update the data table with current EIS data."""
        if not self.current_eis_data:
            return

        # Clear the table
        for item in self.data_table.get_children():
            self.data_table.delete(item)

        # Add the data
        frequency = self.current_eis_data['frequency']
        z_real = self.current_eis_data['Z_real']
        z_imag = self.current_eis_data['Z_imag']
        z_mag = self.current_eis_data['Z_mag']
        z_phase = self.current_eis_data['Z_phase']

        for i in range(len(frequency)):
            self.data_table.insert("", "end", values=(
                f"{frequency[i]:.4e}",
                f"{z_real[i]:.4e}",
                f"{z_imag[i]:.4e}",
                f"{z_mag[i]:.4e}",
                f"{z_phase[i]:.2f}"
            ))

    def update_plot(self):
        """Update the plot based on current plot type and data."""
        if not self.current_eis_data:
            return

        plot_type = self.plot_type_var.get()

        if plot_type == "nyquist":
            self.plot_nyquist()
        elif plot_type == "bode":
            self.plot_bode()
        elif plot_type == "drt" and self.has_impedance:
            self.plot_drt()

    def plot_nyquist(self):
        """Plot the Nyquist plot."""
        # Clear the figure
        self.fig.clear()

        # Create a single plot
        ax = self.fig.add_subplot(111)

        # Get the data
        z_real = self.current_eis_data['Z_real']
        z_imag = self.current_eis_data['Z_imag']
        frequency = self.current_eis_data['frequency']

        # Plot the data
        ax.plot(z_real, -z_imag, 'o-', label='Data')

        # If there's a fit result, add it to the plot
        if self.current_fit_result and 'fitted_impedance' in self.current_fit_result:
            z_fit = self.current_fit_result['fitted_impedance']
            ax.plot(z_fit.real, -z_fit.imag, '--', color='red', label='Fit')

        # Highlight frequencies if requested
        if self.highlight_freq_var.get():
            # Define frequencies to highlight (in Hz)
            highlight_freqs = [0.01, 0.1, 1, 10, 100, 1000, 10000]

            # Find closest indices to highlight frequencies
            for freq in highlight_freqs:
                if frequency.min() <= freq <= frequency.max():
                    idx = np.argmin(np.abs(frequency - freq))
                    ax.plot(z_real[idx], -z_imag[idx], 'o', markersize=8,
                            markerfacecolor='none', markeredgecolor='red')
                    ax.annotate(f"{freq} Hz", (z_real[idx], -z_imag[idx]),
                                xytext=(5, 5), textcoords='offset points')

        # Set plot properties
        ax.set_xlabel("Z' (Ω)")
        ax.set_ylabel("-Z\" (Ω)")
        ax.set_title("Nyquist Plot")
        ax.grid(True, alpha=0.3)

        # Equal aspect ratio for Nyquist plot
        ax.set_aspect('equal')

        # Add legend if needed
        if self.current_fit_result and 'fitted_impedance' in self.current_fit_result:
            ax.legend(loc='best')

        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_bode(self):
        """Plot Bode plots (magnitude and phase vs frequency)."""
        # Clear the figure
        self.fig.clear()

        # Create two subplots
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212, sharex=ax1)

        # Get the data
        frequency = self.current_eis_data['frequency']
        z_mag = self.current_eis_data['Z_mag']
        z_phase = self.current_eis_data['Z_phase']

        # Plot magnitude
        ax1.loglog(frequency, z_mag, 'o-', label='Data')
        ax1.set_ylabel("|Z| (Ω)")
        ax1.set_title("Bode Magnitude Plot")
        ax1.grid(True, which='both', alpha=0.3)

        # Plot phase
        ax2.semilogx(frequency, -z_phase, 'o-', label='Data')
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (°)")
        ax2.set_title("Bode Phase Plot")
        ax2.grid(True, which='both', alpha=0.3)

        # If there's a fit result, add it to the plots
        if self.current_fit_result and 'fitted_impedance' in self.current_fit_result:
            z_fit = self.current_fit_result['fitted_impedance']

            # Calculate magnitude and phase
            z_fit_mag = np.abs(z_fit)
            z_fit_phase = np.angle(z_fit, deg=True)

            # Plot fits
            ax1.loglog(frequency, z_fit_mag, '--', color='red', label='Fit')
            ax2.semilogx(frequency, -z_fit_phase, '--', color='red', label='Fit')

            # Add legends
            ax1.legend(loc='best')
            ax2.legend(loc='best')

        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_drt(self):
        """Plot Distribution of Relaxation Times (DRT)."""
        if not self.has_impedance:
            messagebox.showinfo(
                "Feature Unavailable",
                "DRT plotting requires the impedance.py package.\n" +
                "Install it with: pip install impedance"
            )
            return

        # Clear the figure
        self.fig.clear()

        try:
            # Create the DRT plot
            fig = eis.plot_drt(self.current_eis_data)

            # Replace our figure with the returned one
            self.fig.clear()

            # Copy axes from the returned figure to our figure
            for ax in fig.get_axes():
                new_ax = self.fig.add_subplot(111)
                for line in ax.lines:
                    new_ax.plot(line.get_xdata(), line.get_ydata(),
                                linestyle=line.get_linestyle(),
                                marker=line.get_marker(),
                                color=line.get_color(),
                                label=line.get_label())

                # Copy other properties
                new_ax.set_xlabel(ax.get_xlabel())
                new_ax.set_ylabel(ax.get_ylabel())
                new_ax.set_title(ax.get_title())
                new_ax.set_xscale(ax.get_xscale())
                new_ax.set_yscale(ax.get_yscale())
                new_ax.grid(True, which='both', alpha=0.3)

                # Add legend if needed
                if ax.get_legend():
                    new_ax.legend(loc='best')

            # Update the canvas
            self.fig.tight_layout()
            self.canvas.draw()

            # Clean up the temporary figure
            plt.close(fig)

        except Exception as e:
            self.main_app.log_message(f"Error plotting DRT: {str(e)}", logging.ERROR)
            messagebox.showerror("DRT Error", f"Error plotting Distribution of Relaxation Times: {str(e)}")

            # Show the error in the plot
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error plotting DRT: {str(e)}",
                    ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()

    def fit_circuit(self):
        """Fit an equivalent circuit model to the data."""
        if not self.has_impedance:
            messagebox.showinfo(
                "Feature Unavailable",
                "Circuit fitting requires the impedance.py package.\n" +
                "Install it with: pip install impedance"
            )
            return

        if not self.current_eis_data:
            messagebox.showinfo("No Data", "Please import or load EIS data first.")
            return

        # Get the circuit to fit
        if self.custom_var.get():
            circuit_string = self.custom_circuit_entry.get()
            if not circuit_string:
                messagebox.showwarning("Missing Circuit", "Please enter a custom circuit string.")
                return
        else:
            circuit_string = self.circuit_var.get()

        self.main_app.update_status(f"Fitting circuit: {circuit_string}...")

        # Run the fit in a thread
        threading.Thread(
            target=self._fit_circuit_thread,
            args=(circuit_string,),
            daemon=True
        ).start()

    def _fit_circuit_thread(self, circuit_string):
        """Thread function to fit an equivalent circuit model."""
        try:
            # Get the data
            frequency = self.current_eis_data['frequency']
            z_real = self.current_eis_data['Z_real']
            z_imag = self.current_eis_data['Z_imag']

            # Fit the circuit
            fit_result = eis.fit_circuit_model(frequency, z_real, z_imag, circuit_string)

            # Store the result
            self.current_fit_result = fit_result

            # Update the plot
            self.update_plot()

            # Update the results text
            self.update_results_text()

            # Update the parameters table
            self.update_parameters_table()

            # Update status
            self.main_app.update_status("Circuit fitting complete")
            self.main_app.log_message(f"Fitted circuit: {circuit_string}, R² = {fit_result['r_squared']:.4f}")

        except Exception as e:
            self.main_app.log_message(f"Error fitting circuit: {str(e)}", logging.ERROR)
            messagebox.showerror("Fitting Error", f"Error fitting circuit: {str(e)}")
            self.main_app.update_status("Fitting failed")

    def update_results_text(self):
        """Update the results text with circuit fitting information."""
        if not self.current_fit_result:
            return

        # Enable editing
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        # Add title
        self.results_text.insert(tk.END, "Circuit Fitting Results\n\n", "title")

        # Add circuit information
        circuit = self.current_fit_result.get('circuit_string', 'Unknown')
        r_squared = self.current_fit_result.get('r_squared', 0)
        chi_squared = self.current_fit_result.get('chi_squared', 0)

        self.results_text.insert(tk.END, f"Circuit: {circuit}\n", "heading")
        self.results_text.insert(tk.END, f"R² Value: {r_squared:.4f}\n")
        self.results_text.insert(tk.END, f"Chi² Value: {chi_squared:.4e}\n\n")

        # Add parameter values
        self.results_text.insert(tk.END, "Circuit Parameters\n", "heading")

        for name, value in self.current_fit_result.get('parameters', {}).items():
            self.results_text.insert(tk.END, f"{name}: {value:.6e}\n")

        self.results_text.insert(tk.END, "\n")

        # Add characteristic frequencies
        try:
            char_freqs = eis.compute_characteristic_frequencies(
                self.current_eis_data['frequency'],
                self.current_eis_data['Z_real'],
                self.current_eis_data['Z_imag']
            )

            if char_freqs['characteristic_frequencies']:
                self.results_text.insert(tk.END, "Characteristic Frequencies\n", "heading")

                for i, freq_data in enumerate(char_freqs['characteristic_frequencies']):
                    self.results_text.insert(
                        tk.END,
                        f"Peak {i + 1}: {freq_data['frequency']:.2e} Hz (τ = {freq_data['time_constant']:.2e} s)\n"
                    )

            self.results_text.insert(tk.END, "\n")
        except Exception as e:
            self.main_app.log_message(f"Error computing characteristic frequencies: {str(e)}", logging.WARNING)

        # Apply styles
        self.results_text.tag_configure("title", font=("Arial", 14, "bold"), justify='center')
        self.results_text.tag_configure("heading", font=("Arial", 11, "bold"))

        # Disable editing
        self.results_text.config(state=tk.DISABLED)

        # Switch to the Results tab
        self.view_notebook.select(2)  # Index 2 = Results tab

    def update_parameters_table(self):
        """Update the parameters table with physical parameters from the fit."""
        if not self.has_impedance or not self.current_fit_result:
            return

        # Clear the table
        for item in self.params_table.get_children():
            self.params_table.delete(item)

        try:
            # Extract physical parameters
            params = eis.extract_physical_parameters(self.current_fit_result)

            # Add each parameter to the table
            for name, param_data in params.items():
                self.params_table.insert("", "end", values=(
                    name,
                    f"{param_data['value']:.6e}" if isinstance(param_data['value'], (int, float)) else param_data[
                        'value'],
                    param_data.get('unit', ''),
                    param_data.get('description', '')
                ))

            # Switch to the Parameters tab
            if params:
                self.view_notebook.select(3)  # Index 3 = Parameters tab

        except Exception as e:
            self.main_app.log_message(f"Error updating parameters table: {str(e)}", logging.ERROR)

    def generate_report(self):
        """Generate a comprehensive EIS report."""
        if not self.current_eis_data:
            messagebox.showinfo("No Data", "Please import or load EIS data first.")
            return

        # Ask for the output file location
        file_path = filedialog.asksaveasfilename(
            title="Save EIS Report",
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )

        if not file_path:
            return  # User cancelled

        self.main_app.update_status("Generating EIS report...")

        # Run report generation in a thread
        threading.Thread(
            target=self._generate_report_thread,
            args=(file_path,),
            daemon=True
        ).start()

    def _generate_report_thread(self, file_path):
        """Thread function to generate an EIS report."""
        try:
            # Generate the report
            report_file = eis.generate_comprehensive_eis_report(
                self.current_eis_data,
                fit_results=self.current_fit_result,
                filename=file_path
            )

            # Update the GUI
            self.main_app.update_status("EIS report generated successfully")
            self.main_app.log_message(f"Generated EIS report: {report_file}")

            # Ask if the user wants to open the report
            if messagebox.askyesno(
                    "Report Generated",
                    f"EIS report saved to {report_file}. Open now?"
            ):
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(report_file)}")

        except Exception as e:
            self.main_app.log_message(f"Error generating EIS report: {str(e)}", logging.ERROR)
            messagebox.showerror("Report Error", f"Error generating EIS report: {str(e)}")
            self.main_app.update_status("Report generation failed")