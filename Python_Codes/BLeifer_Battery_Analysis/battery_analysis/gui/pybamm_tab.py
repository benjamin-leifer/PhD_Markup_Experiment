"""
PyBAMM Modeling tab for the battery analysis GUI.

This module provides a tab for theoretical battery modeling using PyBAMM,
including model setup, simulation, parameter fitting, and comparison with experimental data.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from battery_analysis.gui.custom_toolbar import CustomToolbar
from battery_analysis.utils import popout_figure
import numpy as np
import threading
import logging

try:
    from .. import models
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    models = importlib.import_module("models")

try:
    from .. import pybamm_models

    HAS_PYBAMM = pybamm_models.HAS_PYBAMM
except ImportError:  # pragma: no cover - allow running as script
    try:
        import importlib

        pybamm_models = importlib.import_module("pybamm_models")
        HAS_PYBAMM = pybamm_models.HAS_PYBAMM
    except Exception:
        HAS_PYBAMM = False


class PyBAMMTab(ttk.Frame):
    """Tab for PyBAMM battery modeling."""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.current_sample = None
        self.current_test = None
        self.simulation_results = None
        self.comparison_results = None
        self.create_widgets()

    def create_widgets(self):
        """Create the widgets for the PyBAMM tab."""
        # Check if PyBAMM is available
        if not HAS_PYBAMM:
            # Show a message about missing PyBAMM
            message_frame = ttk.Frame(self)
            message_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

            ttk.Label(
                message_frame,
                text="PyBAMM is not installed. Please install it to use theoretical modeling features.",
                font=("Arial", 14),
            ).pack(pady=20)

            ttk.Label(
                message_frame,
                text="Install with: pip install pybamm",
                font=("Arial", 12),
            ).pack(pady=10)

            ttk.Button(
                message_frame,
                text="Reload After Installation",
                command=self.check_pybamm,
            ).pack(pady=20)

            return

        # Create a paned window to allow resizable sections
        self.paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for model configuration
        self.left_panel = ttk.Frame(self.paned_window, width=350)
        self.paned_window.add(self.left_panel, weight=1)

        # Right panel for results
        self.right_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_panel, weight=2)

        # ===== Left Panel Contents =====
        # Experimental Data section
        exp_data_frame = ttk.LabelFrame(self.left_panel, text="Experimental Data")
        exp_data_frame.pack(fill=tk.X, padx=5, pady=5)

        # Sample selection
        sample_frame = ttk.Frame(exp_data_frame)
        sample_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(sample_frame, text="Sample:").pack(side=tk.LEFT, padx=5)
        self.sample_combobox = ttk.Combobox(sample_frame, width=20)
        self.sample_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.sample_combobox.bind("<<ComboboxSelected>>", self.on_sample_selected)

        self.refresh_samples_btn = ttk.Button(
            sample_frame, text="Refresh", command=self.refresh_samples
        )
        self.refresh_samples_btn.pack(side=tk.RIGHT, padx=5)

        # Test selection for validation/comparison
        test_frame = ttk.Frame(exp_data_frame)
        test_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(test_frame, text="Test:").pack(side=tk.LEFT, padx=5)
        self.test_combobox = ttk.Combobox(test_frame, width=20)
        self.test_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.test_combobox.bind("<<ComboboxSelected>>", self.on_test_selected)

        # Extract parameters button
        self.extract_params_btn = ttk.Button(
            exp_data_frame,
            text="Extract Parameters from Sample",
            command=self.extract_sample_parameters,
        )
        self.extract_params_btn.pack(fill=tk.X, padx=5, pady=5)

        # Model Configuration section
        model_frame = ttk.LabelFrame(self.left_panel, text="Model Configuration")
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        # Model selection
        ttk.Label(model_frame, text="Model Type:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.model_var = tk.StringVar(value="SPM")

        model_combobox = ttk.Combobox(
            model_frame, textvariable=self.model_var, width=25
        )
        model_combobox["values"] = list(pybamm_models.AVAILABLE_MODELS.keys())
        model_combobox.grid(row=0, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        # When hovering over the model combobox, show a tooltip with model descriptions
        ToolTip(
            model_combobox,
            "\n".join([f"{k}: {v}" for k, v in pybamm_models.AVAILABLE_MODELS.items()]),
        )

        # Chemistry selection
        ttk.Label(model_frame, text="Chemistry:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.chemistry_var = tk.StringVar(value="NMC")

        chemistry_combobox = ttk.Combobox(
            model_frame, textvariable=self.chemistry_var, width=25
        )
        chemistry_combobox["values"] = ["NMC", "LFP", "NCA", "LCO", "LMO"]
        chemistry_combobox.grid(row=1, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        # Parameter section
        params_frame = ttk.LabelFrame(self.left_panel, text="Parameter Values")
        params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create a frame with a canvas for scrollable content
        canvas_frame = ttk.Frame(params_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.params_canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=self.params_canvas.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.params_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.params_canvas.configure(yscrollcommand=scrollbar.set)

        # Frame for parameters that will be inside the canvas
        self.params_inner_frame = ttk.Frame(self.params_canvas)
        self.params_canvas_window = self.params_canvas.create_window(
            (0, 0), window=self.params_inner_frame, anchor=tk.NW
        )

        self.params_inner_frame.bind("<Configure>", self._configure_params_canvas)
        self.params_canvas.bind("<Configure>", self._configure_params_canvas_window)

        # Add some initial parameters
        self.parameter_entries = {}
        self.add_parameter_entry("Electrode height [m]", "0.1")
        self.add_parameter_entry("Negative electrode thickness [m]", "8.0e-05")
        self.add_parameter_entry("Positive electrode thickness [m]", "7.0e-05")
        self.add_parameter_entry("Initial concentration [mol.m-3]", "1000.0")

        # Parameter file operations
        param_file_frame = ttk.Frame(params_frame)
        param_file_frame.pack(fill=tk.X, padx=2, pady=5)

        ttk.Button(
            param_file_frame, text="Load Parameters", command=self.load_parameters
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            param_file_frame, text="Save Parameters", command=self.save_parameters
        ).pack(side=tk.RIGHT, padx=5)

        # Experiment configuration
        exp_frame = ttk.LabelFrame(self.left_panel, text="Experiment Configuration")
        exp_frame.pack(fill=tk.X, padx=5, pady=5)

        # C-rates
        c_rates_frame = ttk.Frame(exp_frame)
        c_rates_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(c_rates_frame, text="Charge C-rate:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.charge_rate_entry = ttk.Entry(c_rates_frame, width=10)
        self.charge_rate_entry.insert(0, "0.5")
        self.charge_rate_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(c_rates_frame, text="Discharge C-rate:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.discharge_rate_entry = ttk.Entry(c_rates_frame, width=10)
        self.discharge_rate_entry.insert(0, "1.0")
        self.discharge_rate_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Voltage limits
        v_limits_frame = ttk.Frame(exp_frame)
        v_limits_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(v_limits_frame, text="Upper Voltage [V]:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.upper_v_entry = ttk.Entry(v_limits_frame, width=10)
        self.upper_v_entry.insert(0, "4.2")
        self.upper_v_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(v_limits_frame, text="Lower Voltage [V]:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.lower_v_entry = ttk.Entry(v_limits_frame, width=10)
        self.lower_v_entry.insert(0, "2.5")
        self.lower_v_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Temperature
        temp_frame = ttk.Frame(exp_frame)
        temp_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(temp_frame, text="Temperature [°C]:").pack(side=tk.LEFT, padx=5)
        self.temp_entry = ttk.Entry(temp_frame, width=10)
        self.temp_entry.insert(0, "25")
        self.temp_entry.pack(side=tk.LEFT, padx=5)

        # Number of cycles
        cycles_frame = ttk.Frame(exp_frame)
        cycles_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(cycles_frame, text="Number of Cycles:").pack(side=tk.LEFT, padx=5)
        self.cycles_entry = ttk.Entry(cycles_frame, width=10)
        self.cycles_entry.insert(0, "1")
        self.cycles_entry.pack(side=tk.LEFT, padx=5)

        # Run Simulation button
        self.run_button = ttk.Button(
            self.left_panel, text="Run Simulation", command=self.run_simulation
        )
        self.run_button.pack(fill=tk.X, padx=5, pady=10)

        # ===== Right Panel Contents =====
        # Create a notebook for different views
        self.view_notebook = ttk.Notebook(self.right_panel)
        self.view_notebook.pack(fill=tk.BOTH, expand=True)

        # Plot tab
        self.plot_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.plot_frame, text="Plots")

        # Create a matplotlib figure for plots
        self.fig = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add a toolbar with editing support
        self.toolbar = CustomToolbar(self.canvas, self.plot_frame)
        self.toolbar.update()

        self.popout_btn = ttk.Button(
            self.plot_frame,
            text="Open in Window",
            command=lambda: popout_figure(self.fig),
        )
        self.popout_btn.pack(anchor=tk.NE, padx=5, pady=5)

        # Comparison tab
        self.comparison_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.comparison_frame, text="Comparison")

        # Create a matplotlib figure for comparison plots
        self.comparison_fig = plt.figure(figsize=(8, 6))
        self.comparison_canvas = FigureCanvasTkAgg(
            self.comparison_fig, master=self.comparison_frame
        )
        self.comparison_canvas.draw()
        self.comparison_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add a toolbar with editing support
        self.comparison_toolbar = CustomToolbar(
            self.comparison_canvas, self.comparison_frame
        )
        self.comparison_toolbar.update()

        self.comparison_popout_btn = ttk.Button(
            self.comparison_frame,
            text="Open in Window",
            command=lambda: popout_figure(self.comparison_fig),
        )
        self.comparison_popout_btn.pack(anchor=tk.NE, padx=5, pady=5)

        # Parameters tab
        self.fitted_params_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.fitted_params_frame, text="Parameters")

        # Create a text widget for parameter display
        self.params_text = tk.Text(self.fitted_params_frame, wrap=tk.WORD)
        self.params_text.pack(fill=tk.BOTH, expand=True)
        self.params_text.config(state=tk.DISABLED)

        # Add a scrollbar to the parameters text
        params_scrollbar = ttk.Scrollbar(
            self.params_text, orient=tk.VERTICAL, command=self.params_text.yview
        )
        params_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.params_text.config(yscrollcommand=params_scrollbar.set)

        # Add toolbar at the top of the right panel for common operations
        toolbar_frame = ttk.Frame(self.right_panel)
        toolbar_frame.pack(fill=tk.X, padx=5, pady=5, before=self.view_notebook)

        # Plot type selection
        ttk.Label(toolbar_frame, text="Plot:").pack(side=tk.LEFT, padx=5)

        self.plot_type_var = tk.StringVar(value="voltage")
        plot_types = [
            ("Voltage", "voltage"),
            ("Current", "current"),
            ("Capacity", "capacity"),
            ("Concentrations", "concentrations"),
        ]

        for text, value in plot_types:
            ttk.Radiobutton(
                toolbar_frame,
                text=text,
                variable=self.plot_type_var,
                value=value,
                command=self.update_plot,
            ).pack(side=tk.LEFT, padx=5)

        # Comparison toggle
        self.compare_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            toolbar_frame,
            text="Compare with Experiment",
            variable=self.compare_var,
            command=self.update_plot,
        ).pack(side=tk.RIGHT, padx=10)

        # Parameter Estimation button
        self.fit_button = ttk.Button(
            toolbar_frame, text="Fit Parameters", command=self.fit_parameters
        )
        self.fit_button.pack(side=tk.RIGHT, padx=5)

    def check_pybamm(self):
        """Check if PyBAMM is installed and reload the tab if it is."""
        # Try to import PyBAMM
        try:
            import pybamm  # noqa: F401

            # If successful, destroy and recreate the tab
            for widget in self.winfo_children():
                widget.destroy()
            self.create_widgets()
            self.main_app.log_message(
                "PyBAMM detected. Modeling tab reloaded successfully."
            )
        except ImportError:
            messagebox.showinfo(
                "PyBAMM Not Found", "PyBAMM is still not installed or not available."
            )

    def _configure_params_canvas(self, event):
        """Handle parameter frame configuration events."""
        self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all"))

    def _configure_params_canvas_window(self, event):
        """Adjust the width of the params window when canvas resizes."""
        self.params_canvas.itemconfig(self.params_canvas_window, width=event.width)

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
                f"Loaded {len(sample_names)} samples for PyBAMM modeling"
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

            self.main_app.log_message(
                f"Selected sample {sample_name} for PyBAMM modeling"
            )

            # Extract chemistry from the sample and update the interface
            if sample.chemistry:
                # Try to determine chemistry type
                chemistry = self._extract_chemistry_from_string(sample.chemistry)
                if chemistry and chemistry in self.chemistry_var._values:
                    self.chemistry_var.set(chemistry)

        except Exception as e:
            self.main_app.log_message(f"Error loading tests: {str(e)}", logging.ERROR)

    def _extract_chemistry_from_string(self, chemistry_string):
        """Extract the most likely chemistry from a string description."""
        chemistry_string = chemistry_string.lower()

        if "nmc" in chemistry_string:
            return "NMC"
        elif "lfp" in chemistry_string or "lifepo4" in chemistry_string:
            return "LFP"
        elif "nca" in chemistry_string:
            return "NCA"
        elif "lco" in chemistry_string or "licoo2" in chemistry_string:
            return "LCO"
        elif "lmo" in chemistry_string or "limn2o4" in chemistry_string:
            return "LMO"

        return None

    def on_test_selected(self, event):
        """Handle test selection event."""
        if not self.current_sample:
            return

        test_name = self.test_combobox.get()
        if not test_name:
            self.current_test = None
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

            # Update interface with test parameters if available
            if hasattr(test, "charge_rate") and test.charge_rate is not None:
                self.charge_rate_entry.delete(0, tk.END)
                self.charge_rate_entry.insert(0, str(test.charge_rate))

            if hasattr(test, "discharge_rate") and test.discharge_rate is not None:
                self.discharge_rate_entry.delete(0, tk.END)
                self.discharge_rate_entry.insert(0, str(test.discharge_rate))

            if (
                hasattr(test, "upper_cutoff_voltage")
                and test.upper_cutoff_voltage is not None
            ):
                self.upper_v_entry.delete(0, tk.END)
                self.upper_v_entry.insert(0, str(test.upper_cutoff_voltage))

            if (
                hasattr(test, "lower_cutoff_voltage")
                and test.lower_cutoff_voltage is not None
            ):
                self.lower_v_entry.delete(0, tk.END)
                self.lower_v_entry.insert(0, str(test.lower_cutoff_voltage))

            if hasattr(test, "temperature") and test.temperature is not None:
                self.temp_entry.delete(0, tk.END)
                self.temp_entry.insert(0, str(test.temperature))

            self.main_app.log_message(
                f"Selected test {test_name} for comparison with PyBAMM modeling"
            )

        except Exception as e:
            self.main_app.log_message(f"Error selecting test: {str(e)}", logging.ERROR)

    def add_parameter_entry(self, param_name, default_value="0.0"):
        """Add a parameter entry field to the parameters frame."""
        # Create a frame for this parameter
        param_frame = ttk.Frame(self.params_inner_frame)
        param_frame.pack(fill=tk.X, padx=2, pady=2)

        # Add label and entry
        ttk.Label(param_frame, text=param_name + ":").pack(side=tk.LEFT, padx=5)
        entry = ttk.Entry(param_frame, width=15)
        entry.insert(0, default_value)
        entry.pack(side=tk.RIGHT, padx=5)

        # Store the entry widget for later access
        self.parameter_entries[param_name] = entry

    def extract_sample_parameters(self):
        """Extract parameters from the selected sample."""
        if not self.current_sample:
            messagebox.showinfo("No Sample", "Please select a sample first.")
            return

        try:
            # Use PyBAMM module to extract parameters
            parameters = pybamm_models.extract_parameters_from_sample(
                self.current_sample.id
            )

            # Update interface with extracted parameters
            if "cathode_material" in parameters:
                cath_material = parameters["cathode_material"]
                if cath_material in ["NMC", "LFP", "NCA", "LCO", "LMO"]:
                    self.chemistry_var.set(cath_material)

            # Update nominal capacity if available
            if "nominal_capacity" in parameters:
                if "Initial concentration [mol.m-3]" in self.parameter_entries:
                    # This is a simplified mapping, in reality more complex calculations would be needed
                    nom_cap = parameters["nominal_capacity"]
                    # Assuming roughly 1000 mol/m³ per 1Ah capacity for demonstration
                    conc = 1000 * nom_cap / 1000  # Convert mAh to Ah, then scale
                    self.parameter_entries["Initial concentration [mol.m-3]"].delete(
                        0, tk.END
                    )
                    self.parameter_entries["Initial concentration [mol.m-3]"].insert(
                        0, str(conc)
                    )

            # If validation data is available, update experiment settings
            if "validation_data" in parameters and parameters["validation_data"]:
                test_data = parameters["validation_data"][0]  # Use first test

                # Update C-rates
                if "charge_C_rate" in test_data:
                    self.charge_rate_entry.delete(0, tk.END)
                    self.charge_rate_entry.insert(0, str(test_data["charge_C_rate"]))

                if "discharge_C_rate" in test_data:
                    self.discharge_rate_entry.delete(0, tk.END)
                    self.discharge_rate_entry.insert(
                        0, str(test_data["discharge_C_rate"])
                    )

                # Update voltage limits
                if "upper_voltage" in test_data:
                    self.upper_v_entry.delete(0, tk.END)
                    self.upper_v_entry.insert(0, str(test_data["upper_voltage"]))

                if "lower_voltage" in test_data:
                    self.lower_v_entry.delete(0, tk.END)
                    self.lower_v_entry.insert(0, str(test_data["lower_voltage"]))

                # Update temperature
                if "temperature" in test_data:
                    self.temp_entry.delete(0, tk.END)
                    self.temp_entry.insert(0, str(test_data["temperature"]))

            self.main_app.log_message(
                f"Extracted parameters from sample {self.current_sample.name}"
            )

        except Exception as e:
            self.main_app.log_message(
                f"Error extracting parameters: {str(e)}", logging.ERROR
            )
            messagebox.showerror(
                "Parameter Extraction Error", f"Error extracting parameters: {str(e)}"
            )

    def load_parameters(self):
        """Load parameters from a JSON file."""
        file_path = filedialog.askopenfilename(
            title="Load Parameters",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if not file_path:
            return  # User cancelled

        try:
            # Load the parameters
            parameters = pybamm_models.load_model_parameters(file_path)

            # Update the interface
            if "model" in parameters:
                self.model_var.set(parameters["model"])

            if "chemistry" in parameters:
                self.chemistry_var.set(parameters["chemistry"])

            if "parameters" in parameters:
                for name, value in parameters["parameters"].items():
                    if name in self.parameter_entries:
                        self.parameter_entries[name].delete(0, tk.END)
                        self.parameter_entries[name].insert(0, str(value))
                    else:
                        # Add a new parameter entry if it doesn't exist
                        self.add_parameter_entry(name, str(value))

            if "experiment" in parameters:
                exp = parameters["experiment"]
                if "charge_C_rate" in exp:
                    self.charge_rate_entry.delete(0, tk.END)
                    self.charge_rate_entry.insert(0, str(exp["charge_C_rate"]))

                if "discharge_C_rate" in exp:
                    self.discharge_rate_entry.delete(0, tk.END)
                    self.discharge_rate_entry.insert(0, str(exp["discharge_C_rate"]))

                if "upper_voltage" in exp:
                    self.upper_v_entry.delete(0, tk.END)
                    self.upper_v_entry.insert(0, str(exp["upper_voltage"]))

                if "lower_voltage" in exp:
                    self.lower_v_entry.delete(0, tk.END)
                    self.lower_v_entry.insert(0, str(exp["lower_voltage"]))

                if "temperature_C" in exp:
                    self.temp_entry.delete(0, tk.END)
                    self.temp_entry.insert(0, str(exp["temperature_C"]))

                if "cycles" in exp:
                    self.cycles_entry.delete(0, tk.END)
                    self.cycles_entry.insert(0, str(exp["cycles"]))

            self.main_app.log_message(f"Loaded parameters from {file_path}")

        except Exception as e:
            self.main_app.log_message(
                f"Error loading parameters: {str(e)}", logging.ERROR
            )
            messagebox.showerror("Load Error", f"Error loading parameters: {str(e)}")

    def save_parameters(self):
        """Save parameters to a JSON file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Parameters",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if not file_path:
            return  # User cancelled

        try:
            # Collect parameters from the interface
            parameters = {
                "model": self.model_var.get(),
                "chemistry": self.chemistry_var.get(),
                "parameters": {},
                "experiment": {
                    "charge_C_rate": float(self.charge_rate_entry.get()),
                    "discharge_C_rate": float(self.discharge_rate_entry.get()),
                    "upper_voltage": float(self.upper_v_entry.get()),
                    "lower_voltage": float(self.lower_v_entry.get()),
                    "temperature_C": float(self.temp_entry.get()),
                    "cycles": int(self.cycles_entry.get()),
                },
            }

            # Add parameters from entries
            for name, entry in self.parameter_entries.items():
                try:
                    value = float(entry.get())
                    parameters["parameters"][name] = value
                except ValueError:
                    # Skip non-numeric values
                    pass

            # Save to file
            success = pybamm_models.save_model_parameters(parameters, file_path)

            if success:
                self.main_app.log_message(f"Saved parameters to {file_path}")
            else:
                self.main_app.log_message(
                    f"Error saving parameters to {file_path}", logging.ERROR
                )
                messagebox.showerror(
                    "Save Error", f"Error saving parameters to {file_path}"
                )

        except Exception as e:
            self.main_app.log_message(
                f"Error saving parameters: {str(e)}", logging.ERROR
            )
            messagebox.showerror("Save Error", f"Error saving parameters: {str(e)}")

    def run_simulation(self):
        """Run a PyBAMM simulation with the configured parameters."""
        # Disable the run button to prevent multiple simulations
        self.run_button.config(state=tk.DISABLED)
        self.main_app.update_status("Running PyBAMM simulation...")

        # Collect parameters
        try:
            model_type = self.model_var.get()
            chemistry = self.chemistry_var.get()

            # Collect custom parameters
            custom_parameters = {}
            for name, entry in self.parameter_entries.items():
                try:
                    value = float(entry.get())
                    custom_parameters[name] = value
                except ValueError:
                    # Skip non-numeric values
                    pass

            # Experiment parameters
            try:
                charge_rate = float(self.charge_rate_entry.get())
                discharge_rate = float(self.discharge_rate_entry.get())
                upper_v = float(self.upper_v_entry.get())
                lower_v = float(self.lower_v_entry.get())
                temperature_c = float(self.temp_entry.get())
                cycles = int(self.cycles_entry.get())
            except ValueError as e:
                raise ValueError(f"Invalid experiment parameter: {str(e)}")

            # Convert temperature from C to K
            temperature_k = temperature_c + 273.15

            # Create parameter dictionary
            parameters = {
                "chemistry": chemistry,
                "custom_parameters": custom_parameters,
            }

            # Create experiment dictionary
            experiment = {
                "period_specs": [
                    {
                        "type": "CC_charge",
                        "rate": charge_rate,
                        "cutoff_voltage": upper_v,
                    },
                    {
                        "type": "CC_discharge",
                        "rate": discharge_rate,
                        "cutoff_voltage": lower_v,
                    },
                ]
                * cycles
            }

        except ValueError as e:
            self.main_app.log_message(
                f"Error in simulation parameters: {str(e)}", logging.ERROR
            )
            messagebox.showerror(
                "Parameter Error", f"Error in simulation parameters: {str(e)}"
            )
            self.run_button.config(state=tk.NORMAL)
            self.main_app.update_status("Simulation failed")
            return

        # Run the simulation in a thread
        threading.Thread(
            target=self._run_simulation_thread,
            args=(model_type, parameters, experiment, temperature_k),
            daemon=True,
        ).start()

    def _run_simulation_thread(self, model_type, parameters, experiment, temperature_k):
        """Thread function to run the simulation."""
        try:
            # Run the simulation
            results = pybamm_models.run_simulation(
                model_type=model_type, parameters=parameters, experiment=experiment
            )

            # Store the results
            self.simulation_results = results

            # Clear any previous comparison
            self.comparison_results = None

            # If a test is selected, compare with it
            if self.current_test and self.compare_var.get():
                self._compare_with_experiment(results)

            # Update the plots
            self.update_plot()

            # Update parameters tab
            self.update_parameters_text(results)

            # Update status
            self.main_app.update_status("PyBAMM simulation completed successfully")
            self.main_app.log_message(
                f"Completed PyBAMM simulation with model: {model_type}"
            )

            # Switch to the plot tab
            self.view_notebook.select(0)

        except Exception as e:
            self.main_app.log_message(
                f"Error in PyBAMM simulation: {str(e)}", logging.ERROR
            )
            messagebox.showerror(
                "Simulation Error", f"Error in PyBAMM simulation: {str(e)}"
            )
            self.main_app.update_status("Simulation failed")

        finally:
            # Re-enable the run button
            self.run_button.config(state=tk.NORMAL)

    def _compare_with_experiment(self, simulation_results):
        """Compare simulation results with experimental data."""
        try:
            # Run the comparison
            self.comparison_results = pybamm_models.compare_simulation_with_experiment(
                simulation_results, str(self.current_test.id)
            )

            # Update the comparison plot
            self.update_comparison_plot()

            # Log the comparison
            metrics = self.comparison_results.get("metrics", {})
            if metrics:
                error = metrics.get("mean_capacity_error_pct", float("nan"))
                self.main_app.log_message(
                    f"Comparison with {self.current_test.name}: Mean capacity error = {error:.2f}%"
                )

        except Exception as e:
            self.main_app.log_message(
                f"Error in comparison with experiment: {str(e)}", logging.ERROR
            )

    def update_plot(self):
        """Update the plot with current simulation results."""
        if not self.simulation_results:
            return

        # Clear the figure
        self.fig.clear()

        # Get plot type
        plot_type = self.plot_type_var.get()

        if plot_type == "voltage":
            self._plot_voltage_curves()
        elif plot_type == "current":
            self._plot_current_curves()
        elif plot_type == "capacity":
            self._plot_capacity_curves()
        elif plot_type == "concentrations":
            self._plot_concentration_curves()

        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_voltage_curves(self):
        """Plot voltage vs. time curves."""
        # Create a single subplot
        ax = self.fig.add_subplot(111)

        # Get the first cycle from the results
        cycles = self.simulation_results.get("cycles", {})
        if not cycles:
            ax.text(0.5, 0.5, "No cycle data available", ha="center", va="center")
            return

        # Plot each cycle
        for cycle_num, cycle_data in cycles.items():
            time = np.array(cycle_data["t"])
            variables = cycle_data.get("variables", {})

            if "Terminal voltage [V]" in variables:
                voltage = np.array(variables["Terminal voltage [V]"])
                ax.plot(time / 3600, voltage, label=f"Cycle {cycle_num}")

        # Set plot properties
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("Voltage vs. Time")
        ax.grid(True)

        if len(cycles) > 1:
            ax.legend()

    def _plot_current_curves(self):
        """Plot current vs. time curves."""
        # Create a single subplot
        ax = self.fig.add_subplot(111)

        # Get the first cycle from the results
        cycles = self.simulation_results.get("cycles", {})
        if not cycles:
            ax.text(0.5, 0.5, "No cycle data available", ha="center", va="center")
            return

        # Plot each cycle
        for cycle_num, cycle_data in cycles.items():
            time = np.array(cycle_data["t"])
            variables = cycle_data.get("variables", {})

            if "Current [A]" in variables:
                current = np.array(variables["Current [A]"])
                ax.plot(time / 3600, current, label=f"Cycle {cycle_num}")

        # Set plot properties
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Current (A)")
        ax.set_title("Current vs. Time")
        ax.grid(True)

        if len(cycles) > 1:
            ax.legend()

    def _plot_capacity_curves(self):
        """Plot capacity vs. voltage (discharge curve)."""
        # Create a single subplot
        ax = self.fig.add_subplot(111)

        # Get cycles from the results
        cycles = self.simulation_results.get("cycles", {})
        if not cycles:
            ax.text(0.5, 0.5, "No cycle data available", ha="center", va="center")
            return

        # Plot each cycle
        for cycle_num, cycle_data in cycles.items():
            variables = cycle_data.get("variables", {})

            if (
                "Terminal voltage [V]" in variables
                and "Discharge capacity [A.h]" in variables
            ):
                voltage = np.array(variables["Terminal voltage [V]"])
                capacity = np.array(variables["Discharge capacity [A.h]"])

                # Sort by capacity for proper plotting
                sort_idx = np.argsort(capacity)
                voltage = voltage[sort_idx]
                capacity = capacity[sort_idx]

                ax.plot(capacity, voltage, label=f"Cycle {cycle_num}")

        # Set plot properties
        ax.set_xlabel("Capacity (Ah)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("Voltage vs. Capacity")
        ax.grid(True)

        if len(cycles) > 1:
            ax.legend()

    def _plot_concentration_curves(self):
        """Plot particle and electrolyte concentrations."""
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, num=self.fig.number)

        # Get the first cycle from the results
        cycles = self.simulation_results.get("cycles", {})
        if not cycles:
            ax1.text(0.5, 0.5, "No cycle data available", ha="center", va="center")
            return

        # Use only the first cycle for this plot
        cycle_data = cycles.get(1, next(iter(cycles.values())))

        time = np.array(cycle_data["t"])
        variables = cycle_data.get("variables", {})

        # Plot particle surface concentrations
        if (
            "Negative particle surface concentration" in variables
            and "Positive particle surface concentration" in variables
        ):
            neg_conc = np.array(variables["Negative particle surface concentration"])
            pos_conc = np.array(variables["Positive particle surface concentration"])

            ax1.plot(time / 3600, neg_conc, label="Negative Electrode")
            ax1.plot(time / 3600, pos_conc, label="Positive Electrode")

            ax1.set_xlabel("Time (hours)")
            ax1.set_ylabel("Surface Concentration (normalized)")
            ax1.set_title("Particle Surface Concentration")
            ax1.grid(True)
            ax1.legend()

        # Plot electrolyte concentration
        if "X-averaged electrolyte concentration [mol.m-3]" in variables:
            elyte_conc = np.array(
                variables["X-averaged electrolyte concentration [mol.m-3]"]
            )

            ax2.plot(time / 3600, elyte_conc)

            ax2.set_xlabel("Time (hours)")
            ax2.set_ylabel("Concentration (mol/m³)")
            ax2.set_title("Electrolyte Concentration")
            ax2.grid(True)

    def update_comparison_plot(self):
        """Update the comparison plot between simulation and experiment."""
        if not self.simulation_results or not self.comparison_results:
            return

        # Clear the figure
        self.comparison_fig.clear()

        # Create a single subplot
        ax = self.comparison_fig.add_subplot(111)

        # Get data from the comparison
        cycles = self.comparison_results.get("cycles", {})
        if not cycles:
            ax.text(0.5, 0.5, "No comparison data available", ha="center", va="center")
            return

        # Plot each cycle's capacity comparison
        cycle_nums = []
        sim_capacities = []
        exp_capacities = []
        errors = []

        for cycle_num, data in sorted(cycles.items()):
            cycle_nums.append(cycle_num)
            sim_capacities.append(data["sim_capacity"])
            exp_capacities.append(data["exp_capacity"])
            errors.append(data["capacity_error_pct"])

        # Bar chart showing capacities side by side
        x = np.arange(len(cycle_nums))
        width = 0.35

        ax.bar(x - width / 2, sim_capacities, width, label="Simulation")
        ax.bar(x + width / 2, exp_capacities, width, label="Experiment")

        # Set chart properties
        ax.set_xlabel("Cycle Number")
        ax.set_ylabel("Discharge Capacity (mAh)")
        ax.set_title("Simulated vs. Experimental Capacity")
        ax.set_xticks(x)
        ax.set_xticklabels(cycle_nums)
        ax.legend()

        # Add error text
        for i, error in enumerate(errors):
            ax.text(
                x[i],
                max(sim_capacities[i], exp_capacities[i]) + 50,
                f"{error:.1f}%",
                ha="center",
                va="bottom",
                color="red" if abs(error) > 10 else "green",
            )

        # Add overall metrics if available
        metrics = self.comparison_results.get("metrics", {})
        if metrics:
            mean_error = metrics.get("mean_capacity_error_pct", float("nan"))
            rmse = metrics.get("rmse_capacity", float("nan"))

            text = f"Mean Error: {mean_error:.2f}%\nRMSE: {rmse:.2f}%"
            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        # Update the canvas
        self.comparison_fig.tight_layout()
        self.comparison_canvas.draw()

    def update_parameters_text(self, results=None):
        """Update the parameters text with model information."""
        # Enable editing
        self.params_text.config(state=tk.NORMAL)
        self.params_text.delete(1.0, tk.END)

        if results is None:
            results = self.simulation_results

        if not results:
            self.params_text.insert(tk.END, "No simulation results available.")
            self.params_text.config(state=tk.DISABLED)
            return

        # Add title
        self.params_text.insert(tk.END, "PyBAMM Simulation Parameters\n\n", "title")

        # Model information
        self.params_text.insert(tk.END, "Model Information\n", "heading")
        self.params_text.insert(tk.END, f"Model Type: {self.model_var.get()}\n")
        self.params_text.insert(tk.END, f"Chemistry: {self.chemistry_var.get()}\n")
        self.params_text.insert(
            tk.END,
            f"Description: {pybamm_models.AVAILABLE_MODELS.get(self.model_var.get(), '')}\n\n",
        )

        # Experiment information
        self.params_text.insert(tk.END, "Experiment Configuration\n", "heading")
        self.params_text.insert(
            tk.END, f"Charge C-rate: {self.charge_rate_entry.get()}\n"
        )
        self.params_text.insert(
            tk.END, f"Discharge C-rate: {self.discharge_rate_entry.get()}\n"
        )
        self.params_text.insert(
            tk.END, f"Upper Voltage: {self.upper_v_entry.get()} V\n"
        )
        self.params_text.insert(
            tk.END, f"Lower Voltage: {self.lower_v_entry.get()} V\n"
        )
        self.params_text.insert(tk.END, f"Temperature: {self.temp_entry.get()} °C\n")
        self.params_text.insert(tk.END, f"Cycles: {self.cycles_entry.get()}\n\n")

        # Model parameters
        self.params_text.insert(tk.END, "Model Parameters\n", "heading")
        for param_name, entry in sorted(self.parameter_entries.items()):
            value = entry.get()
            self.params_text.insert(tk.END, f"{param_name}: {value}\n")

        # Add comparison metrics if available
        if self.comparison_results and "metrics" in self.comparison_results:
            self.params_text.insert(tk.END, "\nComparison with Experiment\n", "heading")

            metrics = self.comparison_results["metrics"]
            for name, value in sorted(metrics.items()):
                self.params_text.insert(tk.END, f"{name}: {value:.4f}\n")

        # Apply text styles
        self.params_text.tag_configure(
            "title", font=("Arial", 14, "bold"), justify="center"
        )
        self.params_text.tag_configure("heading", font=("Arial", 11, "bold"))

        # Disable editing
        self.params_text.config(state=tk.DISABLED)

    def fit_parameters(self):
        """Fit model parameters to experimental data."""
        if not self.current_test:
            messagebox.showinfo(
                "No Test Selected",
                "Please select an experimental test to fit parameters to.",
            )
            return

        # Collect parameters to fit
        fit_dialog = tk.Toplevel(self)
        fit_dialog.title("Parameter Fitting Configuration")
        fit_dialog.geometry("500x400")
        fit_dialog.transient(self)
        fit_dialog.grab_set()

        # Model selection
        model_frame = ttk.Frame(fit_dialog)
        model_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(model_frame, text="Model Type:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        fit_model_var = tk.StringVar(value=self.model_var.get())

        fit_model_combobox = ttk.Combobox(
            model_frame, textvariable=fit_model_var, width=25
        )
        fit_model_combobox["values"] = list(pybamm_models.AVAILABLE_MODELS.keys())
        fit_model_combobox.grid(row=0, column=1, sticky=tk.W + tk.E, padx=5, pady=5)

        # Parameters to fit
        param_frame = ttk.LabelFrame(fit_dialog, text="Parameters to Fit")
        param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a canvas with scrollbar for parameters
        canvas_frame = ttk.Frame(param_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        fit_canvas = tk.Canvas(canvas_frame)
        fit_scrollbar = ttk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=fit_canvas.yview
        )
        fit_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        fit_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fit_canvas.configure(yscrollcommand=fit_scrollbar.set)

        # Frame for parameter checkboxes
        fit_params_frame = ttk.Frame(fit_canvas)
        fit_canvas.create_window((0, 0), window=fit_params_frame, anchor=tk.NW)

        # Checkboxes for common parameters to fit
        fittable_params = [
            "Negative electrode diffusivity [m2.s-1]",
            "Positive electrode diffusivity [m2.s-1]",
            "Negative electrode reaction rate constant [m.s-1]",
            "Positive electrode reaction rate constant [m.s-1]",
            "Negative electrode Bruggeman coefficient (electrode)",
            "Positive electrode Bruggeman coefficient (electrode)",
            "Electrolyte diffusivity [m2.s-1]",
            "Electrolyte conductivity [S.m-1]",
        ]

        param_vars = {}
        for i, param in enumerate(fittable_params):
            var = tk.BooleanVar(value=i < 2)  # Default: select first two
            param_vars[param] = var
            ttk.Checkbutton(fit_params_frame, text=param, variable=var).grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=2
            )

        # Make sure scrolling works
        fit_params_frame.update_idletasks()
        fit_canvas.config(scrollregion=fit_canvas.bbox("all"))

        # Buttons
        button_frame = ttk.Frame(fit_dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            button_frame,
            text="Fit Parameters",
            command=lambda: self._run_parameter_fitting(
                fit_model_var.get(),
                {k: v.get() for k, v in param_vars.items()},
                fit_dialog,
            ),
        ).pack(side=tk.RIGHT, padx=5)

        ttk.Button(button_frame, text="Cancel", command=fit_dialog.destroy).pack(
            side=tk.RIGHT, padx=5
        )

    def _run_parameter_fitting(self, model_type, param_selections, dialog):
        """
        Run parameter fitting after configuration.

        Args:
            model_type: Type of model to use
            param_selections: Dictionary of parameter name to bool indicating if it should be fitted
            dialog: The configuration dialog to close
        """
        # Close the dialog
        dialog.destroy()

        # Get parameters to fit
        params_to_fit = [
            param for param, selected in param_selections.items() if selected
        ]

        if not params_to_fit:
            messagebox.showwarning(
                "No Parameters Selected", "Please select at least one parameter to fit."
            )
            return

        # Disable interface during fitting
        self.fit_button.config(state=tk.DISABLED)
        self.main_app.update_status(
            "Fitting parameters, this may take several minutes..."
        )

        # Run fitting in a thread
        threading.Thread(
            target=self._parameter_fitting_thread,
            args=(model_type, params_to_fit),
            daemon=True,
        ).start()

    def _parameter_fitting_thread(self, model_type, params_to_fit):
        """Thread function to run parameter fitting."""
        try:
            # Collect initial parameters
            initial_params = {}
            for name, entry in self.parameter_entries.items():
                try:
                    value = float(entry.get())
                    initial_params[name] = value
                except ValueError:
                    # Skip non-numeric values
                    pass

            # Run parameter fitting
            fitted_params = pybamm_models.estimate_model_parameters(
                str(self.current_test.id),
                model_type=model_type,
                initial_params=initial_params,
                params_to_fit=params_to_fit,
            )

            # Update the interface with fitted parameters
            self.main_app.update_status("Parameter fitting completed")

            # Show results
            self._show_fitting_results(fitted_params)

            # Update parameters in the interface
            for param, value in fitted_params.get("fitted_parameters", {}).items():
                if param in self.parameter_entries:
                    self.parameter_entries[param].delete(0, tk.END)
                    self.parameter_entries[param].insert(0, f"{value:.6e}")
                else:
                    # Add a new parameter entry if it doesn't exist
                    self.add_parameter_entry(param, f"{value:.6e}")

            self.main_app.log_message("Parameter fitting completed successfully")

        except Exception as e:
            self.main_app.log_message(
                f"Error in parameter fitting: {str(e)}", logging.ERROR
            )
            messagebox.showerror(
                "Fitting Error", f"Error in parameter fitting: {str(e)}"
            )
            self.main_app.update_status("Parameter fitting failed")

        finally:
            # Re-enable the fit button
            self.fit_button.config(state=tk.NORMAL)

    def _show_fitting_results(self, fitted_params):
        """Show parameter fitting results."""
        # Create a dialog to show results
        results_dialog = tk.Toplevel(self)
        results_dialog.title("Parameter Fitting Results")
        results_dialog.geometry("600x400")
        results_dialog.transient(self)

        # Create a text widget for the results
        results_text = tk.Text(results_dialog, wrap=tk.WORD)
        results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(
            results_text, orient=tk.VERTICAL, command=results_text.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results_text.config(yscrollcommand=scrollbar.set)

        # Add title
        results_text.insert(tk.END, "Parameter Fitting Results\n\n", "title")

        # Add fitted parameters
        results_text.insert(tk.END, "Fitted Parameters\n", "heading")

        for param, value in fitted_params.get("fitted_parameters", {}).items():
            results_text.insert(tk.END, f"{param}: {value:.6e}\n")

        # Add goodness of fit metrics
        results_text.insert(tk.END, "\nGoodness of Fit\n", "heading")

        for metric, value in fitted_params.get("goodness_of_fit", {}).items():
            results_text.insert(tk.END, f"{metric}: {value:.6f}\n")

        # Add optimization info
        results_text.insert(
            tk.END, f"\nIterations: {fitted_params.get('iteration_count', 'N/A')}\n"
        )

        # Apply text styles
        results_text.tag_configure(
            "title", font=("Arial", 14, "bold"), justify="center"
        )
        results_text.tag_configure("heading", font=("Arial", 11, "bold"))

        # Disable editing
        results_text.config(state=tk.DISABLED)

        # Add a close button
        ttk.Button(results_dialog, text="Close", command=results_dialog.destroy).pack(
            pady=10
        )

        # Center the dialog on the parent window
        results_dialog.update_idletasks()
        x = (
            self.winfo_rootx()
            + (self.winfo_width() - results_dialog.winfo_width()) // 2
        )
        y = (
            self.winfo_rooty()
            + (self.winfo_height() - results_dialog.winfo_height()) // 2
        )
        results_dialog.geometry(f"+{x}+{y}")


class ToolTip:
    """Simple tooltip implementation for Tkinter widgets."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        # Create a toplevel window
        self.tooltip = tk.Toplevel(self.widget)
        # Remove decoration
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(
            self.tooltip,
            text=self.text,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
        )
        label.pack(padx=2, pady=2)

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None
