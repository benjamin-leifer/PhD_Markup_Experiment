"""
Cycle Detail Viewer component for the battery analysis GUI.

This module provides functionality for viewing detailed cycle data
that is stored in GridFS.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from battery_analysis.gui.custom_toolbar import CustomToolbar
from battery_analysis.utils import popout_figure
import numpy as np
import pandas as pd
import logging
import os

from battery_analysis.utils.detailed_data_manager import get_detailed_cycle_data


class CycleDetailViewer:
    """A class for viewing detailed cycle data."""

    def __init__(self, parent, test_id, cycle_num):
        """Initialize the cycle detail viewer.

        Args:
            parent: The parent tkinter widget
            test_id: ID of the TestResult to view
            cycle_num: The cycle number to view
        """
        self.parent = parent
        self.test_id = test_id
        self.cycle_num = cycle_num
        self.detailed_data = None

        # Create a new window
        self.window = tk.Toplevel(parent)
        self.window.title(f"Cycle {cycle_num} Details")
        self.window.geometry("900x700")

        # Load the detailed data
        self.load_data()

        # Create the UI if data was loaded successfully
        if self.detailed_data:
            self.create_ui()
        else:
            # Show error message if data couldn't be loaded
            ttk.Label(
                self.window,
                text=f"No detailed data available for cycle {cycle_num}. \n\n"
                     "This could be because the data wasn't stored in GridFS during import,\n"
                     "or because the cycle doesn't exist.",
                foreground="red",
                font=("Arial", 12)
            ).pack(expand=True, padx=20, pady=20)

            # Add close button
            ttk.Button(
                self.window, text="Close", command=self.window.destroy
            ).pack(pady=10)

    def load_data(self):
        """Load detailed cycle data from GridFS."""
        try:
            cycle_data = get_detailed_cycle_data(self.test_id, self.cycle_num)
            if cycle_data and self.cycle_num in cycle_data:
                self.detailed_data = cycle_data[self.cycle_num]
                return True
            return False
        except Exception as e:
            logging.error(f"Error loading detailed data for cycle {self.cycle_num}: {e}")
            messagebox.showerror(
                "Data Load Error",
                f"Error loading detailed data:\n{str(e)}"
            )
            return False

    def create_ui(self):
        """Create the user interface for viewing cycle details."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.plot_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)
        self.dqdv_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.plot_tab, text="Voltage-Capacity Plot")
        self.notebook.add(self.data_tab, text="Raw Data")
        self.notebook.add(self.dqdv_tab, text="dQ/dV Analysis")

        # Create voltage-capacity plot
        self.create_voltage_capacity_plot()

        # Create data tables
        self.create_data_tables()

        # Create dQ/dV analysis
        self.create_dqdv_analysis()

        # Add export button at the bottom
        export_frame = ttk.Frame(self.window)
        export_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            export_frame, text="Export to CSV", command=self.export_data
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            export_frame, text="Export Plot", command=self.export_plot
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            export_frame, text="Close", command=self.window.destroy
        ).pack(side=tk.RIGHT, padx=5)

    def create_voltage_capacity_plot(self):
        """Create the voltage vs capacity plot."""
        # Create figure and axes
        fig = plt.Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # Get charge and discharge data
        charge_data = self.detailed_data.get('charge', {})
        discharge_data = self.detailed_data.get('discharge', {})

        # Plot charge data if available
        if 'voltage' in charge_data and 'capacity' in charge_data:
            ax.plot(
                charge_data['capacity'],
                charge_data['voltage'],
                'b-',
                label='Charge'
            )

        # Plot discharge data if available
        if 'voltage' in discharge_data and 'capacity' in discharge_data:
            ax.plot(
                discharge_data['capacity'],
                discharge_data['voltage'],
                'r-',
                label='Discharge'
            )

        # Set plot properties
        ax.set_xlabel('Capacity (mAh)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Voltage vs. Capacity - Cycle {self.cycle_num}')
        ax.grid(True)
        ax.legend()

        # Create canvas
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar with editing support
        toolbar = CustomToolbar(self.canvas, self.plot_tab)
        toolbar.update()

        ttk.Button(
            self.plot_tab,
            text="Open in Window",
            command=lambda: popout_figure(fig),
        ).pack(pady=5, anchor=tk.NE)

        # Store figure for export
        self.figure = fig

    def create_data_tables(self):
        """Create tables to display the raw data."""
        # Create notebook for charge/discharge tabs
        data_notebook = ttk.Notebook(self.data_tab)
        data_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create charge data tab
        charge_tab = ttk.Frame(data_notebook)
        data_notebook.add(charge_tab, text="Charge Data")

        # Create discharge data tab
        discharge_tab = ttk.Frame(data_notebook)
        data_notebook.add(discharge_tab, text="Discharge Data")

        # Get data
        charge_data = self.detailed_data.get('charge', {})
        discharge_data = self.detailed_data.get('discharge', {})

        # Create charge data table
        if charge_data and 'voltage' in charge_data:
            charge_columns = ["Index"]
            data_columns = []

            # Add available columns
            if 'voltage' in charge_data:
                charge_columns.append("Voltage (V)")
                data_columns.append('voltage')
            if 'current' in charge_data:
                charge_columns.append("Current (A)")
                data_columns.append('current')
            if 'capacity' in charge_data:
                charge_columns.append("Capacity (mAh)")
                data_columns.append('capacity')
            if 'time' in charge_data:
                charge_columns.append("Time (s)")
                data_columns.append('time')

            # Create treeview
            charge_tree = ttk.Treeview(charge_tab, columns=charge_columns, show="headings")

            # Add headings
            for col in charge_columns:
                charge_tree.heading(col, text=col)
                charge_tree.column(col, width=100)

            # Add data rows
            for i in range(len(charge_data['voltage'])):
                values = [i]
                for col in data_columns:
                    if i < len(charge_data[col]):
                        values.append(f"{charge_data[col][i]:.4f}")
                    else:
                        values.append("")
                charge_tree.insert("", "end", values=values)

            # Add scrollbar
            charge_scroll = ttk.Scrollbar(charge_tab, orient="vertical", command=charge_tree.yview)
            charge_tree.configure(yscrollcommand=charge_scroll.set)

            # Pack widgets
            charge_scroll.pack(side="right", fill="y")
            charge_tree.pack(side="left", fill="both", expand=True)
        else:
            ttk.Label(
                charge_tab,
                text="No charge data available",
                foreground="red"
            ).pack(expand=True, padx=20, pady=20)

        # Create discharge data table
        if discharge_data and 'voltage' in discharge_data:
            discharge_columns = ["Index"]
            data_columns = []

            # Add available columns
            if 'voltage' in discharge_data:
                discharge_columns.append("Voltage (V)")
                data_columns.append('voltage')
            if 'current' in discharge_data:
                discharge_columns.append("Current (A)")
                data_columns.append('current')
            if 'capacity' in discharge_data:
                discharge_columns.append("Capacity (mAh)")
                data_columns.append('capacity')
            if 'time' in discharge_data:
                discharge_columns.append("Time (s)")
                data_columns.append('time')

            # Create treeview
            discharge_tree = ttk.Treeview(discharge_tab, columns=discharge_columns, show="headings")

            # Add headings
            for col in discharge_columns:
                discharge_tree.heading(col, text=col)
                discharge_tree.column(col, width=100)

            # Add data rows
            for i in range(len(discharge_data['voltage'])):
                values = [i]
                for col in data_columns:
                    if i < len(discharge_data[col]):
                        values.append(f"{discharge_data[col][i]:.4f}")
                    else:
                        values.append("")
                discharge_tree.insert("", "end", values=values)

            # Add scrollbar
            discharge_scroll = ttk.Scrollbar(discharge_tab, orient="vertical", command=discharge_tree.yview)
            discharge_tree.configure(yscrollcommand=discharge_scroll.set)

            # Pack widgets
            discharge_scroll.pack(side="right", fill="y")
            discharge_tree.pack(side="left", fill="both", expand=True)
        else:
            ttk.Label(
                discharge_tab,
                text="No discharge data available",
                foreground="red"
            ).pack(expand=True, padx=20, pady=20)

    def create_dqdv_analysis(self):
        """Create the dQ/dV analysis plot."""
        try:
            # Create figure and axes
            fig = plt.Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)

            # Calculate dQ/dV for charge and discharge
            charge_data = self.detailed_data.get('charge', {})
            discharge_data = self.detailed_data.get('discharge', {})

            # Process charge data
            if 'voltage' in charge_data and 'capacity' in charge_data:
                v = np.array(charge_data['voltage'])
                q = np.array(charge_data['capacity'])

                # Sort by voltage (increasing)
                sort_idx = np.argsort(v)
                v_sorted = v[sort_idx]
                q_sorted = q[sort_idx]

                # Remove duplicates in voltage
                unique_idx = np.concatenate(([True], np.diff(v_sorted) > 1e-5))
                v_unique = v_sorted[unique_idx]
                q_unique = q_sorted[unique_idx]

                # Calculate dQ/dV
                if len(v_unique) > 3:
                    dq = np.diff(q_unique)
                    dv = np.diff(v_unique)
                    dqdv = dq / dv
                    v_centers = (v_unique[1:] + v_unique[:-1]) / 2

                    # Apply smoothing
                    try:
                        from scipy.signal import savgol_filter
                        window = min(11, len(dqdv) - 2 if len(dqdv) % 2 == 0 else len(dqdv) - 1)
                        window = max(3, window - 1 if window % 2 == 0 else window)
                        dqdv_smooth = savgol_filter(dqdv, window, 1)

                        # Plot
                        ax.plot(v_centers, dqdv_smooth, 'b-', label='Charge')
                    except:
                        # If smoothing fails, use raw data
                        ax.plot(v_centers, dqdv, 'b-', label='Charge')

            # Process discharge data
            if 'voltage' in discharge_data and 'capacity' in discharge_data:
                v = np.array(discharge_data['voltage'])
                q = np.array(discharge_data['capacity'])

                # Sort by voltage (decreasing for discharge)
                sort_idx = np.argsort(v)[::-1]
                v_sorted = v[sort_idx]
                q_sorted = q[sort_idx]

                # Remove duplicates in voltage
                unique_idx = np.concatenate(([True], np.diff(v_sorted) < -1e-5))
                v_unique = v_sorted[unique_idx]
                q_unique = q_sorted[unique_idx]

                # Calculate dQ/dV
                if len(v_unique) > 3:
                    dq = np.diff(q_unique)
                    dv = np.diff(v_unique)
                    dqdv = dq / dv
                    v_centers = (v_unique[1:] + v_unique[:-1]) / 2

                    # Apply smoothing
                    try:
                        from scipy.signal import savgol_filter
                        window = min(11, len(dqdv) - 2 if len(dqdv) % 2 == 0 else len(dqdv) - 1)
                        window = max(3, window - 1 if window % 2 == 0 else window)
                        dqdv_smooth = savgol_filter(dqdv, window, 1)

                        # Plot
                        ax.plot(v_centers, dqdv_smooth, 'r-', label='Discharge')
                    except:
                        # If smoothing fails, use raw data
                        ax.plot(v_centers, dqdv, 'r-', label='Discharge')

            # Set plot properties
            ax.set_xlabel('Voltage (V)')
            ax.set_ylabel('dQ/dV (mAh/V)')
            ax.set_title(f'Differential Capacity Analysis - Cycle {self.cycle_num}')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.dqdv_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Add toolbar with editing support
            toolbar = CustomToolbar(canvas, self.dqdv_tab)
            toolbar.update()

            ttk.Button(
                self.dqdv_tab,
                text="Open in Window",
                command=lambda f=fig: popout_figure(f),
            ).pack(pady=5, anchor=tk.NE)

            # Store figure for export
            self.dqdv_figure = fig

        except Exception as e:
            logging.error(f"Error creating dQ/dV analysis: {e}")
            ttk.Label(
                self.dqdv_tab,
                text=f"Error creating dQ/dV analysis:\n{str(e)}",
                foreground="red"
            ).pack(expand=True, padx=20, pady=20)

    def export_data(self):
        """Export the detailed data to a CSV file."""
        from tkinter import filedialog

        # Ask for the output file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"cycle_{self.cycle_num}_data.csv"
        )

        if not file_path:
            return  # User cancelled

        try:
            # Get data
            charge_data = self.detailed_data.get('charge', {})
            discharge_data = self.detailed_data.get('discharge', {})

            # Prepare DataFrames for charge and discharge data
            charge_rows = []
            discharge_rows = []

            # Process charge data
            if 'voltage' in charge_data:
                for i in range(len(charge_data['voltage'])):
                    row = {
                        'Segment': 'Charge',
                        'Voltage (V)': charge_data['voltage'][i] if i < len(charge_data['voltage']) else None,
                        'Current (A)': charge_data.get('current', [])[i] if i < len(
                            charge_data.get('current', [])) else None,
                        'Capacity (mAh)': charge_data.get('capacity', [])[i] if i < len(
                            charge_data.get('capacity', [])) else None,
                        'Time (s)': charge_data.get('time', [])[i] if i < len(charge_data.get('time', [])) else None
                    }
                    charge_rows.append(row)

            # Process discharge data
            if 'voltage' in discharge_data:
                for i in range(len(discharge_data['voltage'])):
                    row = {
                        'Segment': 'Discharge',
                        'Voltage (V)': discharge_data['voltage'][i] if i < len(discharge_data['voltage']) else None,
                        'Current (A)': discharge_data.get('current', [])[i] if i < len(
                            discharge_data.get('current', [])) else None,
                        'Capacity (mAh)': discharge_data.get('capacity', [])[i] if i < len(
                            discharge_data.get('capacity', [])) else None,
                        'Time (s)': discharge_data.get('time', [])[i] if i < len(
                            discharge_data.get('time', [])) else None
                    }
                    discharge_rows.append(row)

            # Combine into one DataFrame
            all_rows = charge_rows + discharge_rows
            df = pd.DataFrame(all_rows)

            # Save to CSV
            df.to_csv(file_path, index=False)

            messagebox.showinfo(
                "Export Complete",
                f"Data saved to {file_path}"
            )

        except Exception as e:
            logging.error(f"Error exporting data: {e}")
            messagebox.showerror(
                "Export Error",
                f"Error exporting data:\n{str(e)}"
            )

    def export_plot(self):
        """Export the current plot as an image."""
        from tkinter import filedialog

        # Ask for the output file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ],
            initialfile=f"cycle_{self.cycle_num}_plot.png"
        )

        if not file_path:
            return  # User cancelled

        try:
            # Determine which figure to save based on current tab
            if self.notebook.index(self.notebook.select()) == 0:
                figure = self.figure  # Voltage-capacity plot
            elif self.notebook.index(self.notebook.select()) == 2:
                figure = self.dqdv_figure  # dQ/dV plot
            else:
                messagebox.showerror(
                    "Export Error",
                    "Cannot export plot from current tab"
                )
                return

            # Save the figure
            figure.savefig(file_path, dpi=300, bbox_inches='tight')

            messagebox.showinfo(
                "Export Complete",
                f"Plot saved to {file_path}"
            )

        except Exception as e:
            logging.error(f"Error exporting plot: {e}")
            messagebox.showerror(
                "Export Error",
                f"Error exporting plot:\n{str(e)}"
            )
