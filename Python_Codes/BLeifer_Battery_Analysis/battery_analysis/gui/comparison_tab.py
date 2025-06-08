"""
Comparison tab for the battery analysis GUI.

This module provides a tab for comparing multiple tests/samples, generating comparison
plots, and performing statistical analysis of battery test data.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from battery_analysis.gui.custom_toolbar import CustomToolbar
from battery_analysis.utils import popout_figure
import numpy as np
import pandas as pd
import threading
import logging
import os
import traceback

from battery_analysis import models, analysis, report


class ComparisonTab(ttk.Frame):
    """Tab for comparing multiple tests and samples."""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.selected_tests = []
        self.comparison_data = {}
        self.create_widgets()

    def create_widgets(self):
        """Create the widgets for the comparison tab."""
        # Split the tab into left and right panels
        self.left_panel = ttk.Frame(self)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10, ipadx=5, ipady=5)

        self.right_panel = ttk.Frame(self)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ===== Left Panel: Test Selection =====
        selection_frame = ttk.LabelFrame(self.left_panel, text="Test Selection")
        selection_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Sample selection
        ttk.Label(selection_frame, text="Sample:").pack(anchor=tk.W, padx=5, pady=5)
        self.sample_combobox = ttk.Combobox(selection_frame, width=30)
        self.sample_combobox.pack(fill=tk.X, padx=5, pady=2)
        self.sample_combobox.bind("<<ComboboxSelected>>", self.on_sample_selected)

        ttk.Button(
            selection_frame, text="Refresh Samples", command=self.refresh_samples
        ).pack(fill=tk.X, padx=5, pady=5)

        # Test selection listbox
        ttk.Label(selection_frame, text="Available Tests:").pack(anchor=tk.W, padx=5, pady=5)
        test_list_frame = ttk.Frame(selection_frame)
        test_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.test_listbox = tk.Listbox(test_list_frame, selectmode=tk.EXTENDED, height=10)
        self.test_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        test_scrollbar = ttk.Scrollbar(test_list_frame, orient=tk.VERTICAL, command=self.test_listbox.yview)
        test_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.test_listbox.config(yscrollcommand=test_scrollbar.set)

        # Selected tests listbox
        ttk.Label(selection_frame, text="Selected for Comparison:").pack(anchor=tk.W, padx=5, pady=5)
        selected_list_frame = ttk.Frame(selection_frame)
        selected_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.selected_listbox = tk.Listbox(selected_list_frame, selectmode=tk.EXTENDED, height=10)
        self.selected_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        selected_scrollbar = ttk.Scrollbar(selected_list_frame, orient=tk.VERTICAL, command=self.selected_listbox.yview)
        selected_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.selected_listbox.config(yscrollcommand=selected_scrollbar.set)

        # Buttons for moving tests between listboxes
        btn_frame = ttk.Frame(selection_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Add →", command=self.add_selected_tests).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="← Remove", command=self.remove_selected_tests).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear All", command=self.clear_selected_tests).pack(side=tk.RIGHT, padx=5)

        # Comparison options
        options_frame = ttk.LabelFrame(self.left_panel, text="Comparison Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(options_frame, text="Plot Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.plot_type_var = tk.StringVar(value="capacity_vs_cycle")
        plot_types = [
            ("Capacity vs Cycle", "capacity_vs_cycle"),
            ("Normalized Capacity", "normalized_capacity"),
            ("Coulombic Efficiency", "coulombic_efficiency"),
            ("First/Last 10 Cycles", "first_last_cycles"),
            ("Statistical Comparison", "statistical_comparison")
        ]

        for i, (text, value) in enumerate(plot_types):
            ttk.Radiobutton(
                options_frame, text=text, variable=self.plot_type_var, value=value
            ).grid(row=i + 1, column=0, sticky=tk.W, padx=20, pady=2)

        # Option to overlay box/whisker plot of all samples binned by cycle
        self.boxplot_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text="Show Box Plot by Cycle",
            variable=self.boxplot_var,
        ).grid(row=len(plot_types) + 1, column=0, sticky=tk.W, padx=20, pady=2)

        # Option to hide individual curves when showing box plot
        self.boxplot_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text="Hide Individual Curves",
            variable=self.boxplot_only_var,
        ).grid(row=len(plot_types) + 2, column=0, sticky=tk.W, padx=20, pady=2)

        # Create comparison button
        ttk.Button(
            self.left_panel, text="Generate Comparison", command=self.generate_comparison
        ).pack(fill=tk.X, padx=5, pady=10)

        # Export button
        ttk.Button(
            self.left_panel, text="Export Comparison Data", command=self.export_comparison_data
        ).pack(fill=tk.X, padx=5, pady=5)

        # Generate report button
        ttk.Button(
            self.left_panel, text="Generate Comparison Report", command=self.generate_comparison_report
        ).pack(fill=tk.X, padx=5, pady=5)

        # ===== Right Panel: Visualization =====
        # Create a notebook for different views
        self.view_notebook = ttk.Notebook(self.right_panel)
        self.view_notebook.pack(fill=tk.BOTH, expand=True)

        # Plot view
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

        # Button to open plot in a standalone window
        self.popout_btn = ttk.Button(
            self.plot_frame,
            text="Open in Window",
            command=lambda: popout_figure(self.fig),
        )
        self.popout_btn.pack(anchor=tk.NE, padx=5, pady=5)

        # Data view
        self.data_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.data_frame, text="Data Table")

        # Create a frame for the data table
        table_frame = ttk.Frame(self.data_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create a Treeview widget for the data table
        self.data_table = ttk.Treeview(table_frame)
        self.data_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar
        table_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.data_table.yview)
        table_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_table.config(yscrollcommand=table_scrollbar.set)

        # Statistics view
        self.stats_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.stats_frame, text="Statistics")

        # Create a text widget for statistics
        self.stats_text = tk.Text(self.stats_frame, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.stats_text.config(state=tk.DISABLED)

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

            self.main_app.log_message(f"Loaded {len(sample_names)} samples for comparison")
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

            # Clear the test listbox
            self.test_listbox.delete(0, tk.END)

            # Get tests for this sample
            tests = []
            for test_ref in sample.tests:
                test = models.TestResult.objects(id=test_ref.id).first()
                if test:
                    tests.append(test)

            # Add tests to the listbox
            self.test_data = {}
            for test in tests:
                self.test_listbox.insert(tk.END, f"{test.name} ({test.cycle_count} cycles)")
                self.test_data[f"{test.name} ({test.cycle_count} cycles)"] = {
                    'test_id': str(test.id),
                    'test_name': test.name,
                    'sample_name': sample_name,
                    'cycle_count': test.cycle_count
                }

            self.main_app.log_message(f"Loaded {len(tests)} tests for sample {sample_name}")
        except Exception as e:
            self.main_app.log_message(f"Error loading tests: {str(e)}", logging.ERROR)

    def add_selected_tests(self):
        """Add selected tests to the comparison list."""
        selected_indices = self.test_listbox.curselection()
        if not selected_indices:
            return

        for i in selected_indices:
            test_label = self.test_listbox.get(i)
            # Only add if not already in the selected list
            if test_label not in [self.selected_listbox.get(j) for j in range(self.selected_listbox.size())]:
                self.selected_listbox.insert(tk.END, test_label)
                self.selected_tests.append(self.test_data[test_label])

        self.main_app.log_message(f"Added {len(selected_indices)} tests to comparison")

    def remove_selected_tests(self):
        """Remove selected tests from the comparison list."""
        selected_indices = self.selected_listbox.curselection()
        if not selected_indices:
            return

        # Convert indices to values
        removed = []
        for i in sorted(selected_indices, reverse=True):
            test_label = self.selected_listbox.get(i)
            removed.append(test_label)
            self.selected_listbox.delete(i)

            # Find and remove from selected_tests
            for j, test in enumerate(self.selected_tests):
                if f"{test['test_name']} ({test['cycle_count']} cycles)" == test_label:
                    self.selected_tests.pop(j)
                    break

        self.main_app.log_message(f"Removed {len(selected_indices)} tests from comparison")

    def clear_selected_tests(self):
        """Clear all selected tests."""
        self.selected_listbox.delete(0, tk.END)
        self.selected_tests = []
        self.main_app.log_message("Cleared all selected tests")

    def generate_comparison(self):
        """Generate comparison plots for the selected tests."""
        if not self.selected_tests:
            messagebox.showinfo("No Tests", "Please select tests to compare.")
            return

        # Disable the button during processing
        self.main_app.update_status("Generating comparison...")

        # Use a thread to avoid freezing the UI
        def comparison_thread():
            try:
                # Collect data for all selected tests
                cycles_data = {}
                test_info = {}

                for test_data in self.selected_tests:
                    test_id = test_data['test_id']

                    # Get cycle data from the test
                    try:
                        cycle_data = analysis.get_cycle_data(test_id)
                        cycles_data[test_id] = cycle_data
                        test_info[test_id] = {
                            'name': test_data['test_name'],
                            'sample_name': test_data['sample_name'],
                            'cycle_count': test_data['cycle_count']
                        }
                    except Exception as e:
                        tb = traceback.format_exc()
                        self.main_app.log_message(
                            f"Error getting cycle data for test {test_data['test_name']} ({test_id}): {str(e)}",
                            logging.ERROR,
                        )
                        self.main_app.log_message(tb, logging.ERROR)
                        # Continue with other tests

                if not cycles_data:
                    messagebox.showinfo("No Data", "Could not retrieve cycle data for any of the selected tests.")
                    self.main_app.update_status("Comparison failed")
                    return

                # Store the comparison data for later use
                self.comparison_data = {
                    'cycles_data': cycles_data,
                    'test_info': test_info
                }

                # Plot data based on selected plot type
                plot_type = self.plot_type_var.get()
                show_box = self.boxplot_var.get()
                box_only = self.boxplot_only_var.get()

                if plot_type == "capacity_vs_cycle":
                    self.plot_capacity_vs_cycle(cycles_data, test_info, show_box, box_only)
                elif plot_type == "normalized_capacity":
                    self.plot_normalized_capacity(cycles_data, test_info)
                elif plot_type == "coulombic_efficiency":
                    self.plot_coulombic_efficiency(cycles_data, test_info)
                elif plot_type == "first_last_cycles":
                    self.plot_first_last_cycles(cycles_data, test_info)
                elif plot_type == "statistical_comparison":
                    self.plot_statistical_comparison(cycles_data, test_info)

                # Update the data table
                self.update_data_table(cycles_data, test_info)

                # Update statistics view
                self.update_statistics_view(cycles_data, test_info)

                # Update status
                self.main_app.update_status("Comparison generated successfully")
                self.main_app.log_message(f"Generated comparison for {len(cycles_data)} tests")

                # Switch to the plot tab
                self.view_notebook.select(0)  # First tab (Plots)

            except Exception as e:
                self.main_app.log_message(f"Error generating comparison: {str(e)}", logging.ERROR)
                messagebox.showerror("Error", f"Error generating comparison: {str(e)}")
                self.main_app.update_status("Comparison failed")

        # Start the thread
        threading.Thread(target=comparison_thread, daemon=True).start()

    def plot_capacity_vs_cycle(self, cycles_data, test_info, show_box=False, box_only=False):
        """Plot discharge capacity vs cycle number for all tests.

        If ``show_box`` is True, a box/whisker plot is displayed using the
        discharge capacities from all tests. When ``box_only`` is True, the
        individual curves are omitted.
        """
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if not box_only:
            # Plot each test
            for test_id, cycle_data in cycles_data.items():
                info = test_info[test_id]

                # Extract cycle numbers and discharge capacities
                cycle_nums = [c['cycle_index'] for c in cycle_data['cycles']]
                discharge_caps = [c['discharge_capacity'] for c in cycle_data['cycles']]

                # Plot this test
                ax.plot(cycle_nums, discharge_caps, 'o-', label=f"{info['name']} ({info['sample_name']})")

        if show_box:
            # Gather discharge capacities by cycle index for all tests
            max_cycle = max(len(data['cycles']) for data in cycles_data.values())
            grouped = []
            positions = []
            for idx in range(max_cycle):
                vals = [data['cycles'][idx]['discharge_capacity']
                        for data in cycles_data.values()
                        if idx < len(data['cycles'])]
                if vals:
                    grouped.append(vals)
                    positions.append(idx + 1)

            if grouped:
                ax.boxplot(
                    grouped,
                    positions=positions,
                    widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor='none', color='black'),
                    medianprops=dict(color='red'),
                )

        # Set plot properties
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Discharge Capacity (mAh)')
        ax.set_title('Discharge Capacity Comparison')
        ax.grid(True, linestyle='--', alpha=0.7)
        if not box_only:
            ax.legend(loc='best')

        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_normalized_capacity(self, cycles_data, test_info):
        """Plot normalized discharge capacity vs cycle number for all tests."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Plot each test
        for test_id, cycle_data in cycles_data.items():
            info = test_info[test_id]

            # Extract cycle numbers and discharge capacities
            cycle_nums = [c['cycle_index'] for c in cycle_data['cycles']]
            discharge_caps = [c['discharge_capacity'] for c in cycle_data['cycles']]

            # Normalize to the first cycle capacity
            first_capacity = discharge_caps[0] if discharge_caps else 1.0
            normalized_caps = [cap / first_capacity * 100 for cap in discharge_caps]

            # Plot this test
            ax.plot(cycle_nums, normalized_caps, 'o-', label=f"{info['name']} ({info['sample_name']})")

        # Set plot properties
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Normalized Capacity (%)')
        ax.set_title('Normalized Capacity Comparison')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')

        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_coulombic_efficiency(self, cycles_data, test_info):
        """Plot coulombic efficiency vs cycle number for all tests."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Plot each test
        for test_id, cycle_data in cycles_data.items():
            info = test_info[test_id]

            # Extract cycle numbers and coulombic efficiencies
            cycle_nums = [c['cycle_index'] for c in cycle_data['cycles']]
            ce_values = [c['coulombic_efficiency'] * 100 for c in cycle_data['cycles']]  # Convert to percentage

            # Plot this test
            ax.plot(cycle_nums, ce_values, 'o-', label=f"{info['name']} ({info['sample_name']})")

        # Set plot properties
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Coulombic Efficiency (%)')
        ax.set_title('Coulombic Efficiency Comparison')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')

        # Set y-axis limits to highlight differences (typically CE > 95%)
        ax.set_ylim(min(90, min(min([c['coulombic_efficiency'] * 100 for c in data['cycles']])
                                for data in cycles_data.values()) - 2), 101)

        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_first_last_cycles(self, cycles_data, test_info):
        """Plot first and last cycles for each test."""
        self.fig.clear()

        # Create a 2x1 grid of subplots
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        # Colors for different tests
        colors = plt.cm.tab10.colors

        # Plot each test
        for i, (test_id, cycle_data) in enumerate(cycles_data.items()):
            info = test_info[test_id]
            color = colors[i % len(colors)]

            # Get first cycle data
            if len(cycle_data['cycles']) > 0:
                first_cycle = cycle_data['cycles'][0]
                first_capacity = first_cycle['discharge_capacity']

                # Plot first cycle bar
                ax1.bar(i, first_capacity, color=color,
                        label=f"{info['name']} ({info['sample_name']})")

            # Get last cycle data
            if len(cycle_data['cycles']) > 0:
                last_cycle = cycle_data['cycles'][-1]
                last_capacity = last_cycle['discharge_capacity']

                # Plot last cycle bar
                ax2.bar(i, last_capacity, color=color)

        # Set plot properties
        ax1.set_title('First Cycle Capacity')
        ax1.set_ylabel('Discharge Capacity (mAh)')
        ax1.set_xticks(range(len(test_info)))
        ax1.set_xticklabels([f"{i + 1}" for i in range(len(test_info))], rotation=45)

        ax2.set_title('Last Cycle Capacity')
        ax2.set_ylabel('Discharge Capacity (mAh)')
        ax2.set_xticks(range(len(test_info)))
        ax2.set_xticklabels([f"{i + 1}" for i in range(len(test_info))], rotation=45)

        # Add a legend
        handles, labels = ax1.get_legend_handles_labels()
        self.fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
                        ncol=min(3, len(test_info)))

        # Update the canvas
        self.fig.tight_layout(rect=[0, 0.1, 1, 1])  # Make room for the legend
        self.canvas.draw()

    def plot_statistical_comparison(self, cycles_data, test_info):
        """Plot statistical comparison of key metrics for all tests."""
        self.fig.clear()

        # Create a figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6), num=self.fig.number)

        # Prepare data for plotting
        labels = []
        initial_caps = []
        final_caps = []
        retentions = []
        avg_ce = []

        for test_id, cycle_data in cycles_data.items():
            info = test_info[test_id]
            labels.append(info['name'])

            cycles = cycle_data['cycles']
            if cycles:
                initial_caps.append(cycles[0]['discharge_capacity'])
                final_caps.append(cycles[-1]['discharge_capacity'])
                retentions.append(cycles[-1]['discharge_capacity'] / cycles[0]['discharge_capacity'] * 100)
                avg_ce.append(np.mean([c['coulombic_efficiency'] * 100 for c in cycles]))

        # Plot initial capacity
        ax1.bar(range(len(labels)), initial_caps)
        ax1.set_title('Initial Capacity')
        ax1.set_ylabel('mAh')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')

        # Plot final capacity
        ax2.bar(range(len(labels)), final_caps)
        ax2.set_title('Final Capacity')
        ax2.set_ylabel('mAh')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')

        # Plot capacity retention
        ax3.bar(range(len(labels)), retentions)
        ax3.set_title('Capacity Retention')
        ax3.set_ylabel('%')
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45, ha='right')

        # Plot average coulombic efficiency
        ax4.bar(range(len(labels)), avg_ce)
        ax4.set_title('Avg. Coulombic Efficiency')
        ax4.set_ylabel('%')
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=45, ha='right')

        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def update_data_table(self, cycles_data, test_info):
        """Update the data table with comparison data."""
        # Clear the existing table
        for item in self.data_table.get_children():
            self.data_table.delete(item)

        # Configure the columns
        self.data_table['columns'] = ['test', 'sample', 'cycles', 'initial_cap', 'final_cap', 'retention', 'avg_ce']

        self.data_table.heading('test', text='Test Name')
        self.data_table.heading('sample', text='Sample')
        self.data_table.heading('cycles', text='Cycles')
        self.data_table.heading('initial_cap', text='Initial Cap. (mAh)')
        self.data_table.heading('final_cap', text='Final Cap. (mAh)')
        self.data_table.heading('retention', text='Retention (%)')
        self.data_table.heading('avg_ce', text='Avg. CE (%)')

        self.data_table.column('test', width=150)
        self.data_table.column('sample', width=100)
        self.data_table.column('cycles', width=60, anchor='center')
        self.data_table.column('initial_cap', width=120, anchor='center')
        self.data_table.column('final_cap', width=120, anchor='center')
        self.data_table.column('retention', width=100, anchor='center')
        self.data_table.column('avg_ce', width=100, anchor='center')

        # Add data for each test
        for test_id, cycle_data in cycles_data.items():
            info = test_info[test_id]
            cycles = cycle_data['cycles']

            if cycles:
                initial_cap = cycles[0]['discharge_capacity']
                final_cap = cycles[-1]['discharge_capacity']
                retention = final_cap / initial_cap * 100 if initial_cap > 0 else 0
                avg_ce = np.mean([c['coulombic_efficiency'] * 100 for c in cycles])

                self.data_table.insert('', 'end', values=(
                    info['name'],
                    info['sample_name'],
                    info['cycle_count'],
                    f"{initial_cap:.2f}",
                    f"{final_cap:.2f}",
                    f"{retention:.2f}",
                    f"{avg_ce:.2f}"
                ))

    def update_statistics_view(self, cycles_data, test_info):
        """Update the statistics view with comparison analysis."""
        # Enable the text widget for editing
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)

        # Add a title
        self.stats_text.insert(tk.END, "Statistical Comparison\n\n", "title")

        # Calculate statistical measures
        if cycles_data:
            # Prepare data
            test_names = []
            initial_caps = []
            final_caps = []
            retentions = []
            avg_ces = []
            avg_fade_rates = []

            for test_id, cycle_data in cycles_data.items():
                info = test_info[test_id]
                cycles = cycle_data['cycles']

                if cycles:
                    test_names.append(info['name'])
                    initial_cap = cycles[0]['discharge_capacity']
                    final_cap = cycles[-1]['discharge_capacity']

                    initial_caps.append(initial_cap)
                    final_caps.append(final_cap)

                    retention = final_cap / initial_cap * 100 if initial_cap > 0 else 0
                    retentions.append(retention)

                    avg_ce = np.mean([c['coulombic_efficiency'] * 100 for c in cycles])
                    avg_ces.append(avg_ce)

                    # Calculate average fade rate (%/cycle)
                    cycle_count = len(cycles)
                    if cycle_count > 1 and initial_cap > 0:
                        fade_rate = ((initial_cap - final_cap) / initial_cap) * (100 / cycle_count)
                    else:
                        fade_rate = 0
                    avg_fade_rates.append(fade_rate)

            # Add summary statistics
            self.stats_text.insert(tk.END, "Summary Statistics\n", "heading")

            # Initial Capacity
            self.stats_text.insert(tk.END, "Initial Capacity (mAh):\n", "subheading")
            self.stats_text.insert(tk.END, f"  Mean: {np.mean(initial_caps):.2f}\n")
            self.stats_text.insert(tk.END, f"  Std Dev: {np.std(initial_caps):.2f}\n")
            self.stats_text.insert(tk.END, f"  Range: {min(initial_caps):.2f} - {max(initial_caps):.2f}\n\n")

            # Final Capacity
            self.stats_text.insert(tk.END, "Final Capacity (mAh):\n", "subheading")
            self.stats_text.insert(tk.END, f"  Mean: {np.mean(final_caps):.2f}\n")
            self.stats_text.insert(tk.END, f"  Std Dev: {np.std(final_caps):.2f}\n")
            self.stats_text.insert(tk.END, f"  Range: {min(final_caps):.2f} - {max(final_caps):.2f}\n\n")

            # Capacity Retention
            self.stats_text.insert(tk.END, "Capacity Retention (%):\n", "subheading")
            self.stats_text.insert(tk.END, f"  Mean: {np.mean(retentions):.2f}\n")
            self.stats_text.insert(tk.END, f"  Std Dev: {np.std(retentions):.2f}\n")
            self.stats_text.insert(tk.END, f"  Range: {min(retentions):.2f} - {max(retentions):.2f}\n\n")

            # Average Coulombic Efficiency
            self.stats_text.insert(tk.END, "Average Coulombic Efficiency (%):\n", "subheading")
            self.stats_text.insert(tk.END, f"  Mean: {np.mean(avg_ces):.2f}\n")
            self.stats_text.insert(tk.END, f"  Std Dev: {np.std(avg_ces):.2f}\n")
            self.stats_text.insert(tk.END, f"  Range: {min(avg_ces):.2f} - {max(avg_ces):.2f}\n\n")

            # Fade Rate
            self.stats_text.insert(tk.END, "Average Fade Rate (%/cycle):\n", "subheading")
            self.stats_text.insert(tk.END, f"  Mean: {np.mean(avg_fade_rates):.4f}\n")
            self.stats_text.insert(tk.END, f"  Std Dev: {np.std(avg_fade_rates):.4f}\n")
            self.stats_text.insert(tk.END, f"  Range: {min(avg_fade_rates):.4f} - {max(avg_fade_rates):.4f}\n\n")

            # Add rankings
            self.stats_text.insert(tk.END, "Performance Rankings\n", "heading")

            # Rank by capacity retention
            self.stats_text.insert(tk.END, "Ranked by Capacity Retention:\n", "subheading")
            retention_ranks = sorted(zip(test_names, retentions), key=lambda x: x[1], reverse=True)
            for i, (name, ret) in enumerate(retention_ranks):
                self.stats_text.insert(tk.END, f"  {i + 1}. {name}: {ret:.2f}%\n")
            self.stats_text.insert(tk.END, "\n")

            # Rank by coulombic efficiency
            self.stats_text.insert(tk.END, "Ranked by Coulombic Efficiency:\n", "subheading")
            ce_ranks = sorted(zip(test_names, avg_ces), key=lambda x: x[1], reverse=True)
            for i, (name, ce) in enumerate(ce_ranks):
                self.stats_text.insert(tk.END, f"  {i + 1}. {name}: {ce:.2f}%\n")
            self.stats_text.insert(tk.END, "\n")

            # Rank by fade rate (lower is better)
            self.stats_text.insert(tk.END, "Ranked by Fade Rate (lower is better):\n", "subheading")
            fade_ranks = sorted(zip(test_names, avg_fade_rates), key=lambda x: x[1])
            for i, (name, rate) in enumerate(fade_ranks):
                self.stats_text.insert(tk.END, f"  {i + 1}. {name}: {rate:.4f}%/cycle\n")

        # Apply text styles
        self.stats_text.tag_configure("title", font=("Arial", 14, "bold"), justify='center')
        self.stats_text.tag_configure("heading", font=("Arial", 12, "bold"))
        self.stats_text.tag_configure("subheading", font=("Arial", 10, "bold"))

        # Disable editing
        self.stats_text.config(state=tk.DISABLED)

    def export_comparison_data(self):
        """Export comparison data to CSV."""
        if not self.comparison_data:
            messagebox.showinfo("No Data", "No comparison data available to export.")
            return

        # Ask for the output file location
        file_path = tk.filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="comparison_data.csv"
        )

        if not file_path:
            return  # User cancelled

        try:
            # Create a DataFrame with comparison data
            rows = []

            for test_id, cycle_data in self.comparison_data['cycles_data'].items():
                info = self.comparison_data['test_info'][test_id]
                test_name = info['name']
                sample_name = info['sample_name']

                for cycle in cycle_data['cycles']:
                    rows.append({
                        'Sample': sample_name,
                        'Test': test_name,
                        'Cycle': cycle['cycle_index'],
                        'Charge Capacity (mAh)': cycle['charge_capacity'],
                        'Discharge Capacity (mAh)': cycle['discharge_capacity'],
                        'Coulombic Efficiency (%)': cycle['coulombic_efficiency'] * 100
                    })

            df = pd.DataFrame(rows)

            # Save to CSV
            df.to_csv(file_path, index=False)

            self.main_app.log_message(f"Exported comparison data to: {file_path}")
            messagebox.showinfo("Export Complete", f"Comparison data saved to {file_path}")

        except Exception as e:
            self.main_app.log_message(f"Error exporting data: {str(e)}", logging.ERROR)
            messagebox.showerror("Export Error", f"Error exporting data: {str(e)}")

    def generate_comparison_report(self):
        """Generate a PDF report comparing the selected tests."""
        if not self.comparison_data:
            messagebox.showinfo("No Data", "No comparison data available for reporting.")
            return

        # Ask for the output file location
        file_path = tk.filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile="comparison_report.pdf"
        )

        if not file_path:
            return  # User cancelled

        try:
            self.main_app.update_status("Generating comparison report...")

            # Use a thread to avoid freezing the UI
            def report_thread():
                try:
                    # Get all tests from IDs
                    test_objects = []

                    for test_id in self.comparison_data['cycles_data'].keys():
                        test = models.TestResult.objects(id=test_id).first()
                        if test:
                            test_objects.append(test)

                    # Generate the comparison report
                    report_file = report.generate_comparison_report(
                        test_objects,
                        filename=file_path
                    )

                    self.main_app.queue.put({
                        'type': 'status',
                        'text': f"Comparison report saved to {report_file}"
                    })

                    self.main_app.log_message(f"Generated comparison report: {report_file}")

                    # Ask if the user wants to open the report
                    if messagebox.askyesno("Report Generated", f"Report saved to {report_file}. Open now?"):
                        import webbrowser
                        webbrowser.open(f"file://{os.path.abspath(report_file)}")

                except Exception as e:
                    self.main_app.log_message(f"Error generating report: {str(e)}", logging.ERROR)
                    messagebox.showerror("Error", f"Error generating report: {str(e)}")
                    self.main_app.queue.put({
                        'type': 'status',
                        'text': f"Error generating report: {str(e)}"
                    })

            # Start the thread
            threading.Thread(target=report_thread, daemon=True).start()

        except Exception as e:
            self.main_app.log_message(f"Error setting up report generation: {str(e)}", logging.ERROR)
            messagebox.showerror("Error", f"Error generating report: {str(e)}")
