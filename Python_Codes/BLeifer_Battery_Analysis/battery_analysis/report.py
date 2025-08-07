"""
Report generation module for battery test data.

This module provides functions to generate PDF reports with summaries, metrics,
and visualizations of battery test results.
"""

import os
import time
import datetime
import tempfile
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

try:
    from . import utils
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    utils = importlib.import_module("utils")


def generate_report(test_result, filename=None):
    """
    Generate a PDF report for the given TestResult.

    Args:
        test_result: TestResult document to report on
        filename: Output PDF filename (default: auto-generated based on test name)

    Returns:
        str: Path to the generated PDF file
    """
    # If filename not provided, create one based on test name
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join([c if c.isalnum() else "_" for c in test_result.name])
        filename = f"report_{safe_name}_{timestamp}.pdf"

    # Create capacity vs cycle plots
    plot_paths = generate_plots(test_result)

    # Create PDF
    create_pdf_report(test_result, plot_paths, filename)

    # Clean up temporary plot files
    for plot_path in plot_paths:
        if os.path.exists(plot_path):
            try:
                os.remove(plot_path)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {plot_path}: {e}")

    return filename


def generate_plots(test_result):
    """
    Generate plots for the test result.

    Args:
        test_result: TestResult document to generate plots for

    Returns:
        list: List of paths to generated plot image files
    """
    plot_paths = []

    # 1. Create a capacity vs cycle plot using matplotlib
    cycles = test_result.cycles
    cycle_numbers = [c.cycle_index for c in cycles]
    discharge_capacities = [c.discharge_capacity for c in cycles]
    charge_capacities = [c.charge_capacity for c in cycles]
    coulombic_efficiencies = [
        c.coulombic_efficiency * 100 for c in cycles
    ]  # Convert to percentage

    # Plot 1: Capacity vs Cycle
    plt.figure(figsize=(8, 6))
    plt.plot(
        cycle_numbers,
        discharge_capacities,
        marker="o",
        linestyle="-",
        label="Discharge Capacity",
        color="blue",
    )
    plt.plot(
        cycle_numbers,
        charge_capacities,
        marker="s",
        linestyle="-",
        label="Charge Capacity",
        color="red",
    )

    plt.title(f"Capacity vs. Cycle Number - {test_result.sample.name}", fontsize=12)
    plt.xlabel("Cycle Number")
    plt.ylabel("Capacity (mAh)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")

    # Use integer ticks for x-axis (cycle numbers)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add retention annotation
    if test_result.capacity_retention is not None:
        retention_pct = test_result.capacity_retention * 100
        plt.annotate(
            f"Capacity Retention: {retention_pct:.1f}%",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    # Save the plot as a temporary image
    capacity_plot_path = os.path.join(
        tempfile.gettempdir(), f"capacity_plot_{int(time.time())}.png"
    )
    plt.savefig(capacity_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    plot_paths.append(capacity_plot_path)

    # Plot 2: Coulombic Efficiency vs Cycle
    plt.figure(figsize=(8, 6))
    plt.plot(
        cycle_numbers, coulombic_efficiencies, marker="o", linestyle="-", color="green"
    )

    plt.title(
        f"Coulombic Efficiency vs. Cycle Number - {test_result.sample.name}",
        fontsize=12,
    )
    plt.xlabel("Cycle Number")
    plt.ylabel("Coulombic Efficiency (%)")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Set y-axis range to make efficiency variations more visible (typically CE is >95%)
    min_ce = max(80, min(coulombic_efficiencies) - 5) if coulombic_efficiencies else 80
    plt.ylim(min_ce, 102)

    # Use integer ticks for x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add average CE annotation
    if test_result.avg_coulombic_eff is not None:
        avg_ce = test_result.avg_coulombic_eff * 100
        plt.annotate(
            f"Average CE: {avg_ce:.2f}%",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    # Save the plot
    ce_plot_path = os.path.join(
        tempfile.gettempdir(), f"ce_plot_{int(time.time())}.png"
    )
    plt.savefig(ce_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    plot_paths.append(ce_plot_path)

    # Plot 3: Normalized Capacity vs Cycle (if enough cycles)
    if len(cycle_numbers) > 5:
        plt.figure(figsize=(8, 6))

        # Normalize to first cycle capacity
        first_capacity = discharge_capacities[0] if discharge_capacities else 1.0
        normalized_capacities = [
            cap / first_capacity * 100 for cap in discharge_capacities
        ]

        plt.plot(
            cycle_numbers,
            normalized_capacities,
            marker="o",
            linestyle="-",
            color="purple",
        )

        plt.title(
            f"Normalized Discharge Capacity - {test_result.sample.name}", fontsize=12
        )
        plt.xlabel("Cycle Number")
        plt.ylabel("Normalized Capacity (%)")
        plt.grid(True, linestyle="--", alpha=0.7)

        # Use integer ticks for x-axis
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save the plot
        norm_plot_path = os.path.join(
            tempfile.gettempdir(), f"norm_plot_{int(time.time())}.png"
        )
        plt.savefig(norm_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(norm_plot_path)

    return plot_paths


def create_pdf_report(test_result, plot_paths, filename):
    """
    Create a PDF report with test result data and plots.

    Args:
        test_result: TestResult document to report on
        plot_paths: List of paths to plot image files
        filename: Output PDF filename

    Returns:
        None
    """
    # Create a PDF document
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    # Container for the 'Flowable' objects
    elements = []

    # Styles
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Title",
            fontName="Helvetica-Bold",
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )

    styles.add(
        ParagraphStyle(
            name="Heading2",
            fontName="Helvetica-Bold",
            fontSize=14,
            alignment=TA_LEFT,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(name="Normal", fontName="Helvetica", fontSize=10, spaceAfter=6)
    )

    # Title
    elements.append(Paragraph("Battery Test Report", styles["Title"]))
    elements.append(Spacer(1, 0.1 * inch))

    # Sample and Test Information
    elements.append(Paragraph("Sample Information", styles["Heading2"]))

    # Sample data table
    sample_data = [
        ["Sample Name:", test_result.sample.name],
        ["Chemistry:", getattr(test_result.sample, "chemistry", "N/A")],
        ["Manufacturer:", getattr(test_result.sample, "manufacturer", "N/A")],
        ["Form Factor:", getattr(test_result.sample, "form_factor", "N/A")],
    ]

    # Add nominal capacity if available
    if (
        hasattr(test_result.sample, "nominal_capacity")
        and test_result.sample.nominal_capacity is not None
    ):
        sample_data.append(
            ["Nominal Capacity:", f"{test_result.sample.nominal_capacity:.2f} mAh"]
        )

    sample_table = Table(sample_data, colWidths=[1.5 * inch, 4.5 * inch])
    sample_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(sample_table)
    elements.append(Spacer(1, 0.2 * inch))

    # Test Information
    elements.append(Paragraph("Test Information", styles["Heading2"]))

    test_data = [
        ["Test Name:", test_result.name],
        [
            "Test Date:",
            (
                test_result.date.strftime("%Y-%m-%d %H:%M:%S")
                if test_result.date
                else "N/A"
            ),
        ],
        ["Tester:", test_result.tester],
        ["Test Type:", getattr(test_result, "test_type", "Cycling")],
    ]

    # Add test parameters if available
    if hasattr(test_result, "temperature") and test_result.temperature is not None:
        test_data.append(["Temperature:", f"{test_result.temperature:.1f} °C"])

    if (
        hasattr(test_result, "upper_cutoff_voltage")
        and test_result.upper_cutoff_voltage is not None
    ):
        test_data.append(
            ["Upper Cutoff Voltage:", f"{test_result.upper_cutoff_voltage:.3f} V"]
        )

    if (
        hasattr(test_result, "lower_cutoff_voltage")
        and test_result.lower_cutoff_voltage is not None
    ):
        test_data.append(
            ["Lower Cutoff Voltage:", f"{test_result.lower_cutoff_voltage:.3f} V"]
        )

    if hasattr(test_result, "charge_rate") and test_result.charge_rate is not None:
        test_data.append(["Charge Rate:", f"{test_result.charge_rate:.2f}C"])

    if (
        hasattr(test_result, "discharge_rate")
        and test_result.discharge_rate is not None
    ):
        test_data.append(["Discharge Rate:", f"{test_result.discharge_rate:.2f}C"])

    test_table = Table(test_data, colWidths=[1.5 * inch, 4.5 * inch])
    test_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(test_table)
    elements.append(Spacer(1, 0.2 * inch))

    # Performance Metrics
    elements.append(Paragraph("Performance Metrics", styles["Heading2"]))

    # Format performance metrics table
    metrics_data = [
        ["Total Cycles:", str(test_result.cycle_count)],
        ["Initial Discharge Capacity:", f"{test_result.initial_capacity:.3f} mAh"],
        ["Final Discharge Capacity:", f"{test_result.final_capacity:.3f} mAh"],
        ["Capacity Retention:", f"{test_result.capacity_retention * 100:.2f}%"],
        [
            "Average Coulombic Efficiency:",
            f"{test_result.avg_coulombic_eff * 100:.3f}%",
        ],
    ]

    # Add energy efficiency if available
    if (
        hasattr(test_result, "avg_energy_efficiency")
        and test_result.avg_energy_efficiency is not None
    ):
        metrics_data.append(
            [
                "Average Energy Efficiency:",
                f"{test_result.avg_energy_efficiency * 100:.3f}%",
            ]
        )

    # Calculate capacity fade rate per cycle
    if test_result.cycle_count > 1 and test_result.initial_capacity > 0:
        fade_rate = (
            (test_result.initial_capacity - test_result.final_capacity)
            / test_result.initial_capacity
        ) * (100.0 / test_result.cycle_count)
        metrics_data.append(["Capacity Fade Rate:", f"{fade_rate:.4f}% per cycle"])

    metrics_table = Table(metrics_data, colWidths=[2 * inch, 4 * inch])
    metrics_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(metrics_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Add plots
    elements.append(Paragraph("Performance Plots", styles["Heading2"]))

    # Add each plot image to the report
    for i, plot_path in enumerate(plot_paths):
        # Add plot description
        if i == 0:
            elements.append(Paragraph("Capacity vs. Cycle Number", styles["Normal"]))
        elif i == 1:
            elements.append(
                Paragraph("Coulombic Efficiency vs. Cycle Number", styles["Normal"])
            )
        elif i == 2:
            elements.append(
                Paragraph("Normalized Capacity vs. Cycle Number", styles["Normal"])
            )

        # Add the image
        try:
            img = Image(plot_path, width=6.5 * inch, height=4.5 * inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2 * inch))
        except Exception as e:
            elements.append(
                Paragraph(f"Error including plot {i + 1}: {e}", styles["Normal"])
            )

    # Cycle Data Table (first few and last few cycles)
    elements.append(Paragraph("Cycle Data Summary", styles["Heading2"]))

    # Table header
    cycle_table_data = [
        ["Cycle", "Charge Cap. (mAh)", "Discharge Cap. (mAh)", "Coulombic Eff. (%)"]
    ]

    # Get first 5 cycles
    first_n = min(5, len(test_result.cycles))
    for i in range(first_n):
        cycle = test_result.cycles[i]
        cycle_table_data.append(
            [
                str(cycle.cycle_index),
                f"{cycle.charge_capacity:.3f}",
                f"{cycle.discharge_capacity:.3f}",
                f"{cycle.coulombic_efficiency * 100:.2f}",
            ]
        )

    # Add ellipsis row if there are more than 10 cycles
    if len(test_result.cycles) > 10:
        cycle_table_data.append(["...", "...", "...", "..."])

    # Get last 5 cycles
    if len(test_result.cycles) > first_n:
        last_n = min(5, len(test_result.cycles) - first_n)
        for i in range(len(test_result.cycles) - last_n, len(test_result.cycles)):
            cycle = test_result.cycles[i]
            cycle_table_data.append(
                [
                    str(cycle.cycle_index),
                    f"{cycle.charge_capacity:.3f}",
                    f"{cycle.discharge_capacity:.3f}",
                    f"{cycle.coulombic_efficiency * 100:.2f}",
                ]
            )

    # Create and style the table
    cycle_table = Table(cycle_table_data)
    cycle_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 4),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    elements.append(cycle_table)

    # Add report generation timestamp
    elements.append(Spacer(1, 0.5 * inch))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Report generated: {timestamp}", styles["Normal"]))

    # Build the PDF
    doc.build(elements)

    return filename


def generate_comparison_report(test_results, filename=None):
    """
    Generate a comparative PDF report for multiple test results.

    Args:
        test_results: List of TestResult documents to compare
        filename: Output PDF filename (default: auto-generated)

    Returns:
        str: Path to the generated PDF file
    """
    # If filename not provided, create one
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_report_{timestamp}.pdf"

    # Create PDF
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    # Container for the 'Flowable' objects
    elements = []

    # Styles
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Title",
            fontName="Helvetica-Bold",
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )

    styles.add(
        ParagraphStyle(
            name="Heading2",
            fontName="Helvetica-Bold",
            fontSize=14,
            alignment=TA_LEFT,
            spaceAfter=8,
        )
    )

    # Title
    elements.append(Paragraph("Battery Test Comparison Report", styles["Title"]))
    elements.append(Spacer(1, 0.1 * inch))

    # Create comparison plots for all tests
    cycle_nums = []
    discharge_caps = []
    sample_names = []

    # Generate comparative plot with all tests
    plt.figure(figsize=(10, 6))

    for test in test_results:
        cycles = test.cycles
        cycle_numbers = [c.cycle_index for c in cycles]
        discharge_capacities = [c.discharge_capacity for c in cycles]

        # Plot discharge capacity
        plt.plot(
            cycle_numbers,
            discharge_capacities,
            marker="o",
            markersize=4,
            linestyle="-",
            label=f"{utils.get_sample_name(test.sample)}",
        )

        # Store for normalized plot
        cycle_nums.append(cycle_numbers)
        discharge_caps.append(discharge_capacities)
        sample_names.append(utils.get_sample_name(test.sample))

    plt.title("Discharge Capacity Comparison", fontsize=12)
    plt.xlabel("Cycle Number")
    plt.ylabel("Discharge Capacity (mAh)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")

    # Use integer ticks for x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save the plot
    comparison_plot_path = os.path.join(
        tempfile.gettempdir(), f"comparison_plot_{int(time.time())}.png"
    )
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Normalized capacity comparison plot
    plt.figure(figsize=(10, 6))

    for i, (cycles, capacities, name) in enumerate(
        zip(cycle_nums, discharge_caps, sample_names)
    ):
        first_capacity = capacities[0] if capacities else 1.0
        normalized = [cap / first_capacity * 100 for cap in capacities]

        plt.plot(
            cycles, normalized, marker="o", markersize=4, linestyle="-", label=f"{name}"
        )

    plt.title("Normalized Discharge Capacity Comparison", fontsize=12)
    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity (%)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save the normalized plot
    normalized_plot_path = os.path.join(
        tempfile.gettempdir(), f"normalized_comparison_{int(time.time())}.png"
    )
    plt.savefig(normalized_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Summary table of all tests
    elements.append(Paragraph("Test Summary", styles["Heading2"]))

    # Table header
    summary_data = [
        [
            "Sample",
            "Test",
            "Cycles",
            "Initial Cap. (mAh)",
            "Final Cap. (mAh)",
            "Retention (%)",
            "Avg. CE (%)",
        ]
    ]

    # Add data for each test
    for test in test_results:
        summary_data.append(
            [
                utils.get_sample_name(test.sample),
                test.name,
                str(test.cycle_count),
                f"{test.initial_capacity:.3f}",
                f"{test.final_capacity:.3f}",
                f"{test.capacity_retention * 100:.2f}",
                f"{test.avg_coulombic_eff * 100:.2f}",
            ]
        )

    # Create and style the table
    summary_table = Table(summary_data)
    summary_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 4),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    elements.append(summary_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Add plots
    elements.append(Paragraph("Comparison Plots", styles["Heading2"]))

    # Add absolute capacity comparison plot
    elements.append(Paragraph("Discharge Capacity Comparison", styles["Normal"]))
    img = Image(comparison_plot_path, width=7 * inch, height=4 * inch)
    elements.append(img)
    elements.append(Spacer(1, 0.2 * inch))

    # Add normalized capacity comparison plot
    elements.append(
        Paragraph("Normalized Discharge Capacity Comparison", styles["Normal"])
    )
    img = Image(normalized_plot_path, width=7 * inch, height=4 * inch)
    elements.append(img)

    # Add report generation timestamp
    elements.append(Spacer(1, 0.5 * inch))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Report generated: {timestamp}", styles["Normal"]))

    # Build the PDF
    doc.build(elements)

    # Clean up temporary files
    for path in [comparison_plot_path, normalized_plot_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {path}: {e}")

    return filename
