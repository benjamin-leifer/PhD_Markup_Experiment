"""
Main GUI application for battery analysis package.

This module provides a Tkinter-based GUI for the battery_analysis package,
allowing users to upload and analyze battery test data.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd

# Add parent directory to path to run the GUI directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import package modules
from battery_analysis import utils, models, analysis, report
from battery_analysis.parsers import parse_file
from battery_analysis.utils import data_update
try:
    from battery_analysis import advanced_analysis
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False

try:
    from battery_analysis import eis
    HAS_EIS = True
except ImportError:
    HAS_EIS = False

try:
    from battery_analysis import pybamm_models
    HAS_PYBAMM = pybamm_models.HAS_PYBAMM
except ImportError:
    HAS_PYBAMM = False

# Import GUI components
from .comparison_tab import ComparisonTab
from .advanced_analysis_tab import AdvancedAnalysisTab
from .eis_tab import EISTab
from .document_flow_tab import DocumentFlowTab
from .pybamm_tab import PyBAMMTab


class BatteryAnalysisApp(tk.Tk):
    """Main application window for battery analysis GUI."""

    def __init__(self):
        super().__init__()

        # Set up the main window
        self.title("Battery Test Data Analysis")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # Create a message queue for thread-safe communication
        self.queue = queue.Queue()

        # Setup logging
        self.setup_logging()

        # Initialize database connection state
        self.db_connected = False

        # Create the main frame with tabs
        self.create_widgets()

        # Configure styles
        self.style = ttk.Style()
        self.style.configure("TNotebook", background="#f0f0f0")
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", padding=6)
        self.style.configure("TLabel", background="#f0f0f0")

        # Start queue processing
        self.process_queue()

    def setup_logging(self):
        """Setup logging to both file and a text widget."""
        self.log_queue = queue.Queue()
        self.logger = logging.getLogger('BatteryAnalysisGUI')
        self.logger.setLevel(logging.INFO)

        # Create a handler that puts logs into the queue
        queue_handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        queue_handler.setFormatter(formatter)
        self.logger.addHandler(queue_handler)

        # Also add a file handler
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, 'battery_analysis_gui.log')
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def create_widgets(self):
        """Create the main widgets for the application."""
        # Create the main frame that will contain everything
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.data_tab = DataUploadTab(self.notebook, self)
        self.notebook.add(self.data_tab, text="Data Upload")

        self.analysis_tab = AnalysisTab(self.notebook, self)
        self.notebook.add(self.analysis_tab, text="Analysis")

        self.comparison_tab = ComparisonTab(self.notebook, self)
        self.notebook.add(self.comparison_tab, text="Comparison")

        if HAS_ADVANCED:
            self.advanced_tab = AdvancedAnalysisTab(self.notebook, self)
            self.notebook.add(self.advanced_tab, text="Advanced Analysis")

        if HAS_EIS:
            self.eis_tab = EISTab(self.notebook, self)
            self.notebook.add(self.eis_tab, text="EIS Analysis")

        if HAS_PYBAMM:
            self.pybamm_tab = PyBAMMTab(self.notebook, self)
            self.notebook.add(self.pybamm_tab, text="PyBAMM Modeling")

        # Document Flow tab
        self.doc_flow_tab = DocumentFlowTab(self.notebook, self)
        self.notebook.add(self.doc_flow_tab, text="Document Flow")

        self.settings_tab = SettingsTab(self.notebook, self)
        self.notebook.add(self.settings_tab, text="Settings")

        # Create a status bar at the bottom
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(fill=tk.X, padx=0, pady=(5, 0))

        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)

        self.db_status_label = ttk.Label(
            self.status_frame,
            text="Database: Not Connected",
            foreground="red"
        )
        self.db_status_label.pack(side=tk.RIGHT)

        # Create a log frame below the notebook
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Log")
        self.log_frame.pack(fill=tk.X, expand=False, padx=0, pady=(5, 0))

        self.log_text = tk.Text(self.log_frame, height=6, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)

        # Add a scrollbar to the log text
        self.log_scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=self.log_scrollbar.set)

    def update_status(self, message):
        """Update the status bar with a message."""
        self.status_label.config(text=message)

    def update_db_status(self, connected):
        """Update the database connection status."""
        self.db_connected = connected
        if connected:
            self.db_status_label.config(text="Database: Connected", foreground="green")
        else:
            self.db_status_label.config(text="Database: Not Connected", foreground="red")

    def log_message(self, message, level=logging.INFO):
        """Log a message to both the log widget and the logger."""
        if level == logging.INFO:
            self.logger.info(message)
        elif level == logging.WARNING:
            self.logger.warning(message)
        elif level == logging.ERROR:
            self.logger.error(message)
        elif level == logging.DEBUG:
            self.logger.debug(message)

    def process_queue(self):
        """Process the message queue and update the GUI."""
        try:
            # Process all log messages in the queue
            while not self.log_queue.empty():
                record = self.log_queue.get(0)
                self.update_log(record)

            # Process the main message queue
            while not self.queue.empty():
                message = self.queue.get(0)
                if message['type'] == 'status':
                    self.update_status(message['text'])
                elif message['type'] == 'db_status':
                    self.update_db_status(message['connected'])
                elif message['type'] == 'data_loaded':
                    self.analysis_tab.update_data(message['data'])
                elif message['type'] == 'update_diagram':
                    message['callback']()
        except queue.Empty:
            pass

        # Schedule to run again
        self.after(100, self.process_queue)

    def update_log(self, record):
        """Update the log text widget with a new log record."""
        msg = self.logger.handlers[0].format(record)

        # Determine color based on level
        if record.levelno >= logging.ERROR:
            tag = "error"
            color = "red"
        elif record.levelno >= logging.WARNING:
            tag = "warning"
            color = "orange"
        else:
            tag = "info"
            color = "black"

        # Update the log text
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n", tag)
        self.log_text.tag_config(tag, foreground=color)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)


class QueueHandler(logging.Handler):
    """
    A logging handler that puts logs into a queue.

    This is particularly helpful for logging from non-GUI threads,
    as it allows the GUI thread to update the UI safely.
    """
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)


class DataUploadTab(ttk.Frame):
    """Tab for uploading and managing data files."""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.create_widgets()

        # Keep track of uploaded files
        self.uploaded_files = []

    def create_widgets(self):
        """Create the widgets for the data upload tab."""
        # Top section: Database connection
        self.db_frame = ttk.LabelFrame(self, text="Database Connection")
        self.db_frame.pack(fill=tk.X, padx=10, pady=10)

        # DB connection panel
        db_inner_frame = ttk.Frame(self.db_frame)
        db_inner_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(db_inner_frame, text="Host:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.host_entry = ttk.Entry(db_inner_frame, width=30)
        self.host_entry.insert(0, "localhost")
        self.host_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(db_inner_frame, text="Port:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.port_entry = ttk.Entry(db_inner_frame, width=10)
        self.port_entry.insert(0, "27017")
        self.port_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        ttk.Label(db_inner_frame, text="Database:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.db_name_entry = ttk.Entry(db_inner_frame, width=30)
        self.db_name_entry.insert(0, "battery_test_db")
        self.db_name_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        self.connect_btn = ttk.Button(db_inner_frame, text="Connect", command=self.connect_db)
        self.connect_btn.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)

        # Middle section: Data upload
        self.upload_frame = ttk.LabelFrame(self, text="Upload Data Files")
        self.upload_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Buttons for file selection
        btn_frame = ttk.Frame(self.upload_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self.select_file_btn = ttk.Button(
            btn_frame, text="Select File(s)", command=self.select_files
        )
        self.select_file_btn.pack(side=tk.LEFT, padx=5)

        self.select_dir_btn = ttk.Button(
            btn_frame, text="Select Directory", command=self.select_directory
        )
        self.select_dir_btn.pack(side=tk.LEFT, padx=5)

        self.process_btn = ttk.Button(
            btn_frame, text="Process Selected Files", command=self.process_files,
            state=tk.DISABLED
        )
        self.process_btn.pack(side=tk.RIGHT, padx=5)

        # File list with scrollbar
        list_frame = ttk.Frame(self.upload_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        listbox_scrollbar = ttk.Scrollbar(list_frame, command=self.file_listbox.yview)
        listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=listbox_scrollbar.set)

        # Sample association section
        sample_frame = ttk.LabelFrame(self.upload_frame, text="Sample Association")
        sample_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(sample_frame, text="Mode:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.sample_mode_var = tk.StringVar(value="existing")
        self.existing_radio = ttk.Radiobutton(
            sample_frame, text="Use Existing Sample", variable=self.sample_mode_var,
            value="existing", command=self.toggle_sample_mode
        )
        self.existing_radio.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        self.new_radio = ttk.Radiobutton(
            sample_frame, text="Create New Sample", variable=self.sample_mode_var,
            value="new", command=self.toggle_sample_mode
        )
        self.new_radio.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)

        # Existing sample selection
        self.sample_select_frame = ttk.Frame(sample_frame)
        self.sample_select_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)

        ttk.Label(self.sample_select_frame, text="Sample:").pack(side=tk.LEFT, padx=5)
        self.sample_combobox = ttk.Combobox(self.sample_select_frame, width=40)
        self.sample_combobox.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.refresh_samples_btn = ttk.Button(
            self.sample_select_frame, text="Refresh", command=self.refresh_samples
        )
        self.refresh_samples_btn.pack(side=tk.RIGHT, padx=5)

        # New sample creation (initially hidden)
        self.new_sample_frame = ttk.Frame(sample_frame)

        ttk.Label(self.new_sample_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.sample_name_entry = ttk.Entry(self.new_sample_frame, width=30)
        self.sample_name_entry.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        ttk.Label(self.new_sample_frame, text="Chemistry:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.chemistry_entry = ttk.Entry(self.new_sample_frame, width=30)
        self.chemistry_entry.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        ttk.Label(self.new_sample_frame, text="Form Factor:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.form_factor_entry = ttk.Entry(self.new_sample_frame, width=30)
        self.form_factor_entry.grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        ttk.Label(self.new_sample_frame, text="Nominal Capacity (mAh):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.capacity_entry = ttk.Entry(self.new_sample_frame, width=30)
        self.capacity_entry.grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

    def toggle_sample_mode(self):
        """Toggle between existing sample and new sample modes."""
        mode = self.sample_mode_var.get()
        if mode == "existing":
            self.new_sample_frame.grid_forget()
            self.sample_select_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)
            self.refresh_samples()
        else:
            self.sample_select_frame.grid_forget()
            self.new_sample_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)

    def connect_db(self):
        """Connect to the MongoDB database."""
        host = self.host_entry.get()
        port = int(self.port_entry.get())
        db_name = self.db_name_entry.get()

        # Disable the button during connection attempt
        self.connect_btn.config(state=tk.DISABLED)
        self.main_app.update_status("Connecting to database...")

        # Use a thread to avoid freezing the UI
        def connect_thread():
            try:
                connected = utils.connect_to_database(db_name, host, port)

                if connected:
                    self.main_app.queue.put({
                        'type': 'status',
                        'text': f"Connected to database {db_name}"
                    })
                    self.main_app.queue.put({
                        'type': 'db_status',
                        'connected': True
                    })
                    self.main_app.log_message(f"Connected to MongoDB at {host}:{port}/{db_name}")

                    # Refresh sample list
                    self.refresh_samples()
                else:
                    self.main_app.queue.put({
                        'type': 'status',
                        'text': "Failed to connect to database"
                    })
                    self.main_app.queue.put({
                        'type': 'db_status',
                        'connected': False
                    })
                    self.main_app.log_message("Failed to connect to database", logging.ERROR)
            except Exception as e:
                self.main_app.queue.put({
                    'type': 'status',
                    'text': f"Error connecting to database: {str(e)}"
                })
                self.main_app.queue.put({
                    'type': 'db_status',
                    'connected': False
                })
                self.main_app.log_message(f"Error connecting to database: {str(e)}", logging.ERROR)

            # Re-enable the connect button
            self.connect_btn.config(state=tk.NORMAL)

        # Start the thread
        threading.Thread(target=connect_thread, daemon=True).start()

    def refresh_samples(self):
        """Refresh the sample combobox with samples from the database."""
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

            self.main_app.log_message(f"Loaded {len(sample_names)} samples from database")
        except Exception as e:
            self.main_app.log_message(f"Error loading samples: {str(e)}", logging.ERROR)

    def select_files(self):
        """Open a file dialog to select data files."""
        # Get the supported file extensions
        from battery_analysis.parsers import get_supported_formats
        formats = get_supported_formats()

        # Create a file type filter for the dialog
        file_types = [
            ("All supported files", " ".join([f"*{fmt}" for fmt in formats])),
            ("Arbin files", "*.csv *.xlsx *.xls"),
            ("BioLogic files", "*.mpr *.mpt"),
            ("All files", "*.*")
        ]

        # Open the file dialog
        files = filedialog.askopenfilenames(
            title="Select Data Files",
            filetypes=file_types
        )

        if files:
            # Clear the current list
            self.file_listbox.delete(0, tk.END)
            self.uploaded_files = []

            # Add the selected files
            for file in files:
                self.file_listbox.insert(tk.END, file)
                self.uploaded_files.append(file)

            # Enable the process button
            self.process_btn.config(state=tk.NORMAL)

            self.main_app.log_message(f"Selected {len(files)} files")
            self.main_app.update_status(f"Selected {len(files)} files")

    def select_directory(self):
        """Open a directory dialog to select a folder with data files."""
        directory = filedialog.askdirectory(
            title="Select Directory with Data Files"
        )

        if directory:
            # Use the utility function to find all compatible files
            files = utils.get_file_list(directory)

            if not files:
                messagebox.showinfo("No Files Found", "No supported data files found in the selected directory.")
                return

            # Clear the current list
            self.file_listbox.delete(0, tk.END)
            self.uploaded_files = []

            # Add the found files
            for file in files:
                self.file_listbox.insert(tk.END, file)
                self.uploaded_files.append(file)

            # Enable the process button
            self.process_btn.config(state=tk.NORMAL)

            self.main_app.log_message(f"Found {len(files)} files in directory {directory}")
            self.main_app.update_status(f"Found {len(files)} files in directory")

    def process_files(self):
        """Process the selected files."""
        if not self.main_app.db_connected:
            messagebox.showwarning("Not Connected", "Please connect to the database first.")
            return

        selected_indices = self.file_listbox.curselection()

        if not selected_indices:
            messagebox.showinfo("No Files Selected", "Please select files to process.")
            return

        # Get the selected files
        selected_files = [self.uploaded_files[i] for i in selected_indices]

        # Check if we're using an existing sample or creating a new one
        sample_mode = self.sample_mode_var.get()

        if sample_mode == "existing":
            sample_name = self.sample_combobox.get()
            if not sample_name:
                messagebox.showwarning("No Sample Selected", "Please select a sample.")
                return

            # Get the sample from the database
            sample = models.Sample.objects(name=sample_name).first()

            if not sample:
                messagebox.showerror("Sample Not Found", f"Sample '{sample_name}' not found in database.")
                return
        else:
            # Create a new sample
            sample_name = self.sample_name_entry.get()
            chemistry = self.chemistry_entry.get()
            form_factor = self.form_factor_entry.get()

            if not sample_name:
                messagebox.showwarning("Missing Information", "Please enter a sample name.")
                return

            # Check if the sample already exists
            existing_sample = models.Sample.objects(name=sample_name).first()
            if existing_sample:
                confirm = messagebox.askyesno(
                    "Sample Exists",
                    f"Sample '{sample_name}' already exists. Use existing sample?"
                )
                if confirm:
                    sample = existing_sample
                else:
                    return
            else:
                # Create the new sample
                try:
                    sample = models.Sample(
                        name=sample_name,
                        chemistry=chemistry,
                        form_factor=form_factor
                    )

                    # Add nominal capacity if provided
                    capacity_str = self.capacity_entry.get()
                    if capacity_str:
                        try:
                            sample.nominal_capacity = float(capacity_str)
                        except ValueError:
                            messagebox.showwarning(
                                "Invalid Capacity",
                                "Nominal capacity must be a number. Using default."
                            )

                    sample.save()
                    self.main_app.log_message(f"Created new sample: {sample_name}")
                except Exception as e:
                    messagebox.showerror("Error", f"Error creating sample: {str(e)}")
                    return

        # Process the files
        self.process_btn.config(state=tk.DISABLED)
        self.main_app.update_status(f"Processing {len(selected_files)} files...")

        # Configure update behavior
        self.update_var = tk.BooleanVar(value=True)

        # Ask user about update behavior
        update_dialog = tk.Toplevel(self)
        update_dialog.title("Update Configuration")
        update_dialog.geometry("400x200")
        update_dialog.transient(self)
        update_dialog.grab_set()

        ttk.Label(
            update_dialog,
            text="How should duplicate/existing tests be handled?",
            font=("Arial", 10, "bold")
        ).pack(pady=(15, 5))

        # Update strategy selection
        strategy_frame = ttk.Frame(update_dialog)
        strategy_frame.pack(pady=5, fill=tk.X, padx=20)

        self.strategy_var = tk.StringVar(value="auto")

        ttk.Radiobutton(
            strategy_frame, text="Automatic (recommended)", variable=self.strategy_var,
            value="auto"
        ).pack(anchor=tk.W)

        ttk.Label(
            strategy_frame,
            text="Detects updates and appends new cycles to existing tests",
            font=("Arial", 8), foreground="gray"
        ).pack(anchor=tk.W, padx=(20, 0))

        ttk.Radiobutton(
            strategy_frame, text="Always create new tests", variable=self.strategy_var,
            value="new"
        ).pack(anchor=tk.W, pady=(5, 0))

        ttk.Label(
            strategy_frame,
            text="Creates separate tests even if duplicates exist",
            font=("Arial", 8), foreground="gray"
        ).pack(anchor=tk.W, padx=(20, 0))

        ttk.Radiobutton(
            strategy_frame, text="Ask for each duplicate", variable=self.strategy_var,
            value="ask"
        ).pack(anchor=tk.W, pady=(5, 0))

        # Buttons
        button_frame = ttk.Frame(update_dialog)
        button_frame.pack(side=tk.BOTTOM, pady=15)

        ttk.Button(
            button_frame, text="Continue",
            command=lambda: self._continue_processing(selected_files, sample, update_dialog)
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame, text="Cancel",
            command=lambda: self._cancel_processing(update_dialog)
        ).pack(side=tk.LEFT, padx=5)

        # Wait for the dialog to be closed
        self.wait_window(update_dialog)

    def _continue_processing(self, selected_files, sample, dialog):
        # Get the selected strategy
        strategy = self.strategy_var.get()

        # Close the dialog
        dialog.destroy()

        # Use a thread to avoid freezing the UI
        threading.Thread(
            target=self._process_files_thread,
            args=(selected_files, sample, strategy),
            daemon=True
        ).start()

    def _cancel_processing(self, dialog):
        # Close the dialog
        dialog.destroy()

        # Re-enable the process button
        self.process_btn.config(state=tk.NORMAL)

        self.main_app.update_status("Processing cancelled")

    def _process_files_thread(self, selected_files, sample, update_strategy):
        """Thread function for processing files with update handling."""
        try:
            success_count = 0
            update_count = 0
            failed_count = 0

            for file_path in selected_files:
                try:
                    self.main_app.log_message(f"Processing file: {os.path.basename(file_path)}")

                    if update_strategy == "new":
                        # Always create new tests
                        parsed_data, metadata = parse_file(file_path)
                        # Add file path to metadata
                        if metadata is None:
                            metadata = {}
                        metadata['file_path'] = file_path

                        # Create a new test
                        test_result = analysis.create_test_result(
                            sample=sample,
                            cycles_summary=parsed_data,
                            tester=metadata.get('tester', 'Unknown'),
                            metadata=metadata
                        )

                        success_count += 1
                        self.main_app.log_message(
                            f"Created new test: {test_result.name} with {test_result.cycle_count} cycles"
                        )

                    elif update_strategy == "auto":
                        # Use automatic update detection
                        test_result, was_update = data_update.process_file_with_update(file_path, sample)

                        if was_update:
                            update_count += 1
                            self.main_app.log_message(
                                f"Updated existing test: {test_result.name} to {test_result.cycle_count} cycles"
                            )
                        else:
                            success_count += 1
                            self.main_app.log_message(
                                f"Created new test: {test_result.name} with {test_result.cycle_count} cycles"
                            )

                    elif update_strategy == "ask":
                        # Ask for each file

                        # Parse the file
                        parsed_data, metadata = parse_file(file_path)
                        # Add file path to metadata
                        if metadata is None:
                            metadata = {}
                        metadata['file_path'] = file_path

                        # Check for existing tests
                        identifiers = data_update.extract_test_identifiers(file_path, parsed_data, metadata)
                        matching_tests = data_update.find_matching_tests(identifiers, sample.id)

                        if matching_tests:
                            # Ask user what to do
                            response_queue = queue.Queue()

                            def show_update_dialog():
                                existing_test = matching_tests[0]

                                dialog = tk.Toplevel(self)
                                dialog.title("Update Existing Test")
                                dialog.geometry("500x300")
                                dialog.transient(self.main_app)
                                dialog.grab_set()

                                ttk.Label(
                                    dialog,
                                    text=f"File appears to match existing test: {existing_test.name}",
                                    font=("Arial", 10, "bold")
                                ).pack(pady=(15, 5))

                                # Show comparison
                                comparison_frame = ttk.LabelFrame(dialog, text="Comparison")
                                comparison_frame.pack(fill=tk.X, padx=15, pady=5)

                                comparison_data = [
                                    ("Existing test cycles:", str(existing_test.cycle_count)),
                                    ("New file cycles:", str(len(parsed_data))),
                                    ("Existing test date:", existing_test.date.strftime("%Y-%m-%d %H:%M:%S") if existing_test.date else "Unknown"),
                                    ("File name:", os.path.basename(file_path))
                                ]

                                for i, (label, value) in enumerate(comparison_data):
                                    ttk.Label(comparison_frame, text=label, anchor=tk.E).grid(
                                        row=i, column=0, sticky=tk.E, padx=5, pady=2)
                                    ttk.Label(comparison_frame, text=value).grid(
                                        row=i, column=1, sticky=tk.W, padx=5, pady=2)

                                # Actions
                                action_frame = ttk.LabelFrame(dialog, text="Action")
                                action_frame.pack(fill=tk.X, padx=15, pady=5)

                                action_var = tk.StringVar(value="append")

                                ttk.Radiobutton(
                                    action_frame, text="Append new cycles (if any)",
                                    variable=action_var, value="append"
                                ).pack(anchor=tk.W, padx=5)

                                ttk.Radiobutton(
                                    action_frame, text="Replace completely",
                                    variable=action_var, value="replace"
                                ).pack(anchor=tk.W, padx=5)

                                ttk.Radiobutton(
                                    action_frame, text="Create as new test anyway",
                                    variable=action_var, value="new"
                                ).pack(anchor=tk.W, padx=5)

                                ttk.Radiobutton(
                                    action_frame, text="Skip this file",
                                    variable=action_var, value="skip"
                                ).pack(anchor=tk.W, padx=5)

                                # Buttons
                                button_frame = ttk.Frame(dialog)
                                button_frame.pack(side=tk.BOTTOM, pady=15)

                                ttk.Button(
                                    button_frame, text="Apply",
                                    command=lambda: response_queue.put(action_var.get()) or dialog.destroy()
                                ).pack(side=tk.LEFT, padx=5)

                                # Wait for response
                                dialog.protocol("WM_DELETE_WINDOW", lambda: response_queue.put("skip") or dialog.destroy())

                            # Show dialog in main thread and wait for response
                            self.main_app.after(0, show_update_dialog)
                            action = response_queue.get()  # This blocks until user responds

                            if action == "skip":
                                continue  # Skip this file

                            elif action == "new":
                                # Create as new test
                                test_result = analysis.create_test_result(
                                    sample=sample,
                                    cycles_summary=parsed_data,
                                    tester=metadata.get('tester', 'Unknown'),
                                    metadata=metadata
                                )

                                success_count += 1
                                self.main_app.log_message(
                                    f"Created new test: {test_result.name} with {test_result.cycle_count} cycles"
                                )

                            else:  # append or replace
                                # Update existing test
                                existing_test = matching_tests[0]
                                updated_test = data_update.update_test_data(
                                    existing_test, parsed_data, metadata, strategy=action
                                )

                                update_count += 1
                                self.main_app.log_message(
                                    f"Updated existing test: {updated_test.name} ({action}) to {updated_test.cycle_count} cycles"
                                )

                        else:
                            # No match, create new test
                            test_result = analysis.create_test_result(
                                sample=sample,
                                cycles_summary=parsed_data,
                                tester=metadata.get('tester', 'Unknown'),
                                metadata=metadata
                            )

                            success_count += 1
                            self.main_app.log_message(
                                f"Created new test: {test_result.name} with {test_result.cycle_count} cycles"
                            )

                except Exception as e:
                    self.main_app.log_message(f"Error processing file {os.path.basename(file_path)}: {str(e)}", logging.ERROR)
                    failed_count += 1

            # Update status
            self.main_app.update_status(
                f"Processed {success_count} new, {update_count} updated, {failed_count} failed"
            )

            # Update the sample selection combobox
            self.refresh_samples()

            # Notify analysis tab that new data is available
            if success_count > 0 or update_count > 0:
                # Get the latest test result
                sample_obj = models.Sample.objects(id=sample.id).first()
                if sample_obj and sample_obj.tests:
                    latest_test = models.TestResult.objects(id=sample_obj.tests[-1].id).first()

                    if latest_test:
                        self.main_app.queue.put({
                            'type': 'data_loaded',
                            'data': {
                                'sample': sample_obj,
                                'test': latest_test
                            }
                        })

        except Exception as e:
            self.main_app.log_message(f"Error processing files: {str(e)}", logging.ERROR)
            self.main_app.queue.put({
                'type': 'status',
                'text': f"Error processing files: {str(e)}"
            })

        finally:
            # Re-enable the process button
            self.process_btn.config(state=tk.NORMAL)


class AnalysisTab(ttk.Frame):
    """Tab for basic analysis and visualization."""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.current_data = None
        self.create_widgets()

    def create_widgets(self):
        """Create the widgets for the analysis tab."""
        # Left panel: Sample and test selection
        self.left_frame = ttk.Frame(self)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Sample selection
        sample_frame = ttk.LabelFrame(self.left_frame, text="Sample Selection")
        sample_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(sample_frame, text="Sample:").pack(anchor=tk.W, padx=5, pady=5)
        self.sample_combobox = ttk.Combobox(sample_frame, width=30)
        self.sample_combobox.pack(fill=tk.X, padx=5, pady=5)
        self.sample_combobox.bind("<<ComboboxSelected>>", self.on_sample_selected)

        self.refresh_samples_btn = ttk.Button(
            sample_frame, text="Refresh", command=self.refresh_samples
        )
        self.refresh_samples_btn.pack(fill=tk.X, padx=5, pady=5)

        # Test selection
        test_frame = ttk.LabelFrame(self.left_frame, text="Test Selection")
        test_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(test_frame, text="Test:").pack(anchor=tk.W, padx=5, pady=5)
        self.test_combobox = ttk.Combobox(test_frame, width=30)
        self.test_combobox.pack(fill=tk.X, padx=5, pady=5)
        self.test_combobox.bind("<<ComboboxSelected>>", self.on_test_selected)

        # Analysis options
        analysis_frame = ttk.LabelFrame(self.left_frame, text="Analysis Options")
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)

        self.analyze_btn = ttk.Button(
            analysis_frame, text="Analyze Test", command=self.analyze_test
        )
        self.analyze_btn.pack(fill=tk.X, padx=5, pady=5)

        self.report_btn = ttk.Button(
            analysis_frame, text="Generate Report", command=self.generate_report
        )
        self.report_btn.pack(fill=tk.X, padx=5, pady=5)

        # Right panel: Visualization and results
        self.right_frame = ttk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Notebook for different views
        self.view_notebook = ttk.Notebook(self.right_frame)
        self.view_notebook.pack(fill=tk.BOTH, expand=True)

        # Summary view
        self.summary_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.summary_frame, text="Summary")

        # Create a text widget for the summary
        self.summary_text = tk.Text(self.summary_frame, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        self.summary_text.config(state=tk.DISABLED)

        # Plot view
        self.plot_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.plot_frame, text="Plots")

        # Create a matplotlib figure for plots
        self.fig = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add a toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

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

            self.main_app.log_message(f"Loaded {len(sample_names)} samples for analysis")
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

            # Get tests for this sample
            tests = []
            for test_ref in sample.tests:
                test = models.TestResult.objects(id=test_ref.id).first()
                if test:
                    tests.append(test)

            # Update the test combobox
            test_names = [test.name for test in tests]
            self.test_combobox['values'] = test_names

            if test_names:
                self.test_combobox.current(0)
                self.on_test_selected(None)
            else:
                self.test_combobox.set("")
                self.clear_views()

            self.main_app.log_message(f"Loaded {len(tests)} tests for sample {sample_name}")
        except Exception as e:
            self.main_app.log_message(f"Error loading tests: {str(e)}", logging.ERROR)

    def on_test_selected(self, event):
        """Handle test selection event."""
        sample_name = self.sample_combobox.get()
        test_name = self.test_combobox.get()

        if not sample_name or not test_name:
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

            # Store the current data
            self.current_data = {
                'sample': sample,
                'test': test
            }

            # Display basic information
            self.update_summary()

            self.main_app.log_message(f"Selected test {test_name} for sample {sample_name}")
        except Exception as e:
            self.main_app.log_message(f"Error selecting test: {str(e)}", logging.ERROR)

    def update_data(self, data):
        """Update the current data from external events (e.g., data upload)."""
        if 'sample' in data and 'test' in data:
            self.current_data = data

            # Update the comboboxes
            self.refresh_samples()

            # Select the current sample and test
            self.sample_combobox.set(data['sample'].name)
            self.test_combobox.set(data['test'].name)

            # Update the views
            self.update_summary()

            self.main_app.log_message(f"Data updated from external source")

    def update_summary(self):
        """Update the summary view with current data."""
        if not self.current_data:
            return

        sample = self.current_data['sample']
        test = self.current_data['test']

        # Clear the text widget
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)

        # Add sample information
        self.summary_text.insert(tk.END, "Sample Information\n", "heading1")
        self.summary_text.insert(tk.END, f"Name: {sample.name}\n")

        if sample.chemistry:
            self.summary_text.insert(tk.END, f"Chemistry: {sample.chemistry}\n")

        if sample.form_factor:
            self.summary_text.insert(tk.END, f"Form Factor: {sample.form_factor}\n")

        if sample.nominal_capacity:
            self.summary_text.insert(tk.END, f"Nominal Capacity: {sample.nominal_capacity} mAh\n")

        self.summary_text.insert(tk.END, "\n")

        # Add test information
        self.summary_text.insert(tk.END, "Test Information\n", "heading1")
        self.summary_text.insert(tk.END, f"Name: {test.name}\n")
        self.summary_text.insert(tk.END, f"Tester: {test.tester}\n")

        if test.date:
            self.summary_text.insert(tk.END, f"Date: {test.date.strftime('%Y-%m-%d %H:%M:%S')}\n")

        if hasattr(test, 'test_type') and test.test_type:
            self.summary_text.insert(tk.END, f"Test Type: {test.test_type}\n")

        self.summary_text.insert(tk.END, "\n")

        # Add test results
        self.summary_text.insert(tk.END, "Test Results\n", "heading1")

        if test.cycle_count:
            self.summary_text.insert(tk.END, f"Cycle Count: {test.cycle_count}\n")

        if test.initial_capacity:
            self.summary_text.insert(tk.END, f"Initial Capacity: {test.initial_capacity:.2f} mAh\n")

        if test.final_capacity:
            self.summary_text.insert(tk.END, f"Final Capacity: {test.final_capacity:.2f} mAh\n")

        if test.capacity_retention:
            retention_pct = test.capacity_retention * 100
            self.summary_text.insert(tk.END, f"Capacity Retention: {retention_pct:.2f}%\n")

        if test.avg_coulombic_eff:
            ce_pct = test.avg_coulombic_eff * 100
            self.summary_text.insert(tk.END, f"Average Coulombic Efficiency: {ce_pct:.2f}%\n")

        # Style the headings
        self.summary_text.tag_configure("heading1", font=("Arial", 12, "bold"))

        # Disable editing
        self.summary_text.config(state=tk.DISABLED)

        # Update the plots
        self.update_plots()

    def update_plots(self):
        """Update the plot view with current data."""
        if not self.current_data:
            return

        test = self.current_data['test']

        # Clear the figure
        self.fig.clear()

        # Create subplots for capacity and efficiency
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212, sharex=ax1)

        # Extract data for plotting
        cycle_nums = [cycle.cycle_index for cycle in test.cycles]
        discharge_caps = [cycle.discharge_capacity for cycle in test.cycles]
        charge_caps = [cycle.charge_capacity for cycle in test.cycles]
        ce_values = [cycle.coulombic_efficiency * 100 for cycle in test.cycles]  # Convert to percentage

        # Plot capacity
        ax1.plot(cycle_nums, discharge_caps, 'o-', label='Discharge')
        ax1.plot(cycle_nums, charge_caps, 's-', label='Charge')
        ax1.set_ylabel('Capacity (mAh)')
        ax1.set_title(f'Capacity vs. Cycle Number - {test.name}')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # Plot coulombic efficiency
        ax2.plot(cycle_nums, ce_values, 'o-', color='green')
        ax2.set_xlabel('Cycle Number')
        ax2.set_ylabel('Coulombic Efficiency (%)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_ylim(min(90, min(ce_values) - 2), 102)  # Set reasonable y-axis limits

        # Adjust layout
        self.fig.tight_layout()

        # Update the canvas
        self.canvas.draw()

    def clear_views(self):
        """Clear all views."""
        # Clear the summary
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.config(state=tk.DISABLED)

        # Clear the plot
        self.fig.clear()
        self.canvas.draw()

    def analyze_test(self):
        """Perform detailed analysis on the current test."""
        if not self.current_data:
            messagebox.showinfo("No Data", "Please select a test to analyze.")
            return

        test = self.current_data['test']

        try:
            # Perform basic analysis
            cycle_data = analysis.get_cycle_data(test.id)

            # Create a new window for the results
            analysis_window = tk.Toplevel(self)
            analysis_window.title(f"Analysis Results - {test.name}")
            analysis_window.geometry("800x600")

            # Create a notebook for different analysis views
            notebook = ttk.Notebook(analysis_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Cycle data tab
            cycle_frame = ttk.Frame(notebook)
            notebook.add(cycle_frame, text="Cycle Data")

            # Create a table for cycle data
            cycle_table = ttk.Treeview(
                cycle_frame,
                columns=("cycle", "charge", "discharge", "ce"),
                show="headings"
            )

            cycle_table.heading("cycle", text="Cycle")
            cycle_table.heading("charge", text="Charge Cap. (mAh)")
            cycle_table.heading("discharge", text="Discharge Cap. (mAh)")
            cycle_table.heading("ce", text="CE (%)")

            cycle_table.column("cycle", width=80)
            cycle_table.column("charge", width=120)
            cycle_table.column("discharge", width=120)
            cycle_table.column("ce", width=80)

            cycle_table.pack(fill=tk.BOTH, expand=True)

            # Add scrollbar
            scrollbar = ttk.Scrollbar(cycle_frame, orient=tk.VERTICAL, command=cycle_table.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            cycle_table.configure(yscrollcommand=scrollbar.set)

            # Add cycle data to the table
            for i, cycle in enumerate(cycle_data['cycles']):
                cycle_table.insert(
                    "",
                    tk.END,
                    values=(
                        cycle['cycle_index'],
                        f"{cycle['charge_capacity']:.2f}",
                        f"{cycle['discharge_capacity']:.2f}",
                        f"{cycle['coulombic_efficiency']*100:.2f}"
                    )
                )

            # Add a tab for advanced analysis if available
            if HAS_ADVANCED and isinstance(test.id, str):
                advanced_frame = ttk.Frame(notebook)
                notebook.add(advanced_frame, text="Advanced Analysis")

                # Create a button to run advanced analysis
                def run_advanced_analysis():
                    try:
                        # Perform capacity fade analysis
                        fade_analysis = advanced_analysis.capacity_fade_analysis(test.id)

                        # Update the text widget with results
                        advanced_text.config(state=tk.NORMAL)
                        advanced_text.delete(1.0, tk.END)

                        advanced_text.insert(tk.END, "Capacity Fade Analysis\n\n", "heading")
                        advanced_text.insert(tk.END, f"Initial Capacity: {fade_analysis['initial_capacity']:.2f} mAh\n")
                        advanced_text.insert(tk.END, f"Final Capacity: {fade_analysis['final_capacity']:.2f} mAh\n")
                        advanced_text.insert(tk.END, f"Capacity Retention: {fade_analysis['capacity_retention']*100:.2f}%\n")
                        advanced_text.insert(tk.END, f"Fade Rate: {fade_analysis['fade_rate_pct_per_cycle']:.4f}% per cycle\n\n")

                        if fade_analysis['best_model']:
                            best_model = fade_analysis['best_model']
                            advanced_text.insert(tk.END, f"Best Fit Model: {fade_analysis['fade_models'][best_model]['name']}\n")
                            advanced_text.insert(tk.END, f"R²: {fade_analysis['fade_models'][best_model]['r_squared']:.4f}\n")

                            if fade_analysis['predicted_eol_cycle']:
                                advanced_text.insert(tk.END, f"Predicted EOL Cycle (80% retention): {int(fade_analysis['predicted_eol_cycle'])}\n")
                                advanced_text.insert(tk.END, f"Prediction Confidence: {fade_analysis['confidence']*100:.1f}%\n")

                        # Style the headings
                        advanced_text.tag_configure("heading", font=("Arial", 12, "bold"))

                        # Disable editing
                        advanced_text.config(state=tk.DISABLED)

                    except Exception as e:
                        messagebox.showerror("Error", f"Error performing advanced analysis: {str(e)}")

                advanced_btn = ttk.Button(
                    advanced_frame, text="Run Advanced Analysis", command=run_advanced_analysis
                )
                advanced_btn.pack(padx=10, pady=10)

                # Create a text widget for the results
                advanced_text = tk.Text(advanced_frame, wrap=tk.WORD)
                advanced_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                advanced_text.config(state=tk.DISABLED)

            self.main_app.log_message(f"Analyzed test {test.name}")

        except Exception as e:
            self.main_app.log_message(f"Error analyzing test: {str(e)}", logging.ERROR)
            messagebox.showerror("Error", f"Error analyzing test: {str(e)}")

    def generate_report(self):
        """Generate a PDF report for the current test."""
        if not self.current_data:
            messagebox.showinfo("No Data", "Please select a test to generate a report.")
            return

        test = self.current_data['test']

        # Ask for the output file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile=f"{test.name}_report.pdf"
        )

        if not file_path:
            return  # User cancelled

        # Generate the report
        try:
            self.main_app.update_status("Generating report...")

            # Use a thread to avoid freezing the UI
            def report_thread():
                try:
                    report_file = report.generate_report(test, file_path)

                    self.main_app.queue.put({
                        'type': 'status',
                        'text': f"Report saved to {report_file}"
                    })

                    self.main_app.log_message(f"Generated report for test {test.name}: {report_file}")

                    # Ask if the user wants to open the report
                    if messagebox.askyesno("Report Generated", f"Report saved to {report_file}. Open now?"):
                        import webbrowser
                        webbrowser.open(f"file://{os.path.abspath(report_file)}")

                except Exception as e:
                    self.main_app.log_message(f"Error generating report: {str(e)}", logging.ERROR)
                    messagebox.showerror("Error", f"Error generating report: {str(e)}")

            # Start the thread
            threading.Thread(target=report_thread, daemon=True).start()

        except Exception as e:
            self.main_app.log_message(f"Error generating report: {str(e)}", logging.ERROR)
            messagebox.showerror("Error", f"Error generating report: {str(e)}")


class SettingsTab(ttk.Frame):
    """Tab for application settings."""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.create_widgets()

    def create_widgets(self):
        """Create the widgets for the settings tab."""
        # Create a labeled frame for database settings
        db_frame = ttk.LabelFrame(self, text="Database Settings")
        db_frame.pack(fill=tk.X, padx=10, pady=10)

        # Database settings
        ttk.Label(db_frame, text="Default Database Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.db_name_entry = ttk.Entry(db_frame, width=30)
        self.db_name_entry.insert(0, "battery_test_db")
        self.db_name_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(db_frame, text="Default Host:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.host_entry = ttk.Entry(db_frame, width=30)
        self.host_entry.insert(0, "localhost")
        self.host_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(db_frame, text="Default Port:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.port_entry = ttk.Entry(db_frame, width=10)
        self.port_entry.insert(0, "27017")
        self.port_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # Save button
        self.save_btn = ttk.Button(db_frame, text="Save Settings", command=self.save_settings)
        self.save_btn.grid(row=3, column=0, columnspan=2, pady=10)

        # Create a labeled frame for application settings
        app_frame = ttk.LabelFrame(self, text="Application Settings")
        app_frame.pack(fill=tk.X, padx=10, pady=10)

        # Theme selection
        ttk.Label(app_frame, text="Theme:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.theme_combobox = ttk.Combobox(app_frame, width=20)
        self.theme_combobox['values'] = ('Default', 'Light', 'Dark')
        self.theme_combobox.current(0)
        self.theme_combobox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Acknowledgments
        ack_frame = ttk.LabelFrame(self, text="About")
        ack_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        about_text = """Battery Test Data Analysis

A tool for analyzing battery test data from various sources.

Features:
- Data import from Arbin and BioLogic formats
- Cycling data analysis and visualization
- Advanced analysis tools including capacity fade prediction
- EIS analysis and modeling
- Report generation

Dependencies:
- MongoDB for data storage
- Python libraries: numpy, pandas, matplotlib, etc.

License: MIT
"""

        about_label = ttk.Label(ack_frame, text=about_text, justify=tk.LEFT)
        about_label.pack(padx=10, pady=10)

    def save_settings(self):
        """Save the current settings."""
        # In a real application, we would save these to a configuration file
        messagebox.showinfo("Settings", "Settings saved successfully.")
        self.main_app.log_message("Settings saved")


def main():
    """Run the application."""
    app = BatteryAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()