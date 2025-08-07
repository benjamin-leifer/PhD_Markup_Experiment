#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mongo_dqdv_plot.py – Plot dQ/dV from MongoDB (GridFS detailed data)

 • Connects to MongoDB
 • Retrieves tests by cell code
 • Extracts voltage-capacity from GridFS detailed data
 • Computes and plots dQ/dV for charge and discharge
 • Computes difference in dQ/dV between 1st and 2nd charge cycles
"""

import logging
import re
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from mongoengine import connect
from battery_analysis import models
from battery_analysis.utils.detailed_data_manager import get_detailed_cycle_data
from battery_analysis.advanced_analysis import calculate_differential_capacity

# ────────────────────────── CONFIG ──────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ────────────────────────── HELPERS ──────────────────────────
def find_tests_by_cell_code(code: str | List[str]):
    """Return tests matching cell codes (select longest cycling test)."""
    if isinstance(code, list):
        results = []
        for c in code:
            regex = re.compile(c, re.IGNORECASE)
            tests = models.TestResult.objects(name__regex=regex)
            if tests:
                results.append(max(tests, key=lambda t: len(t.cycles)))
        return results
    else:
        regex = re.compile(code, re.IGNORECASE)
        tests = models.TestResult.objects(name__regex=regex)
        return [max(tests, key=lambda t: len(t.cycles))] if tests else []


def extract_dqdv_from_gridfs(test, cycle: int):
    """Get charge and discharge dQ/dV from GridFS detailed data."""
    return calculate_differential_capacity(test.id, cycle, smoothing=True)


# ────────────────────────── PLOTTING ──────────────────────────
def plot_dqdv_from_mongo(cell_codes: List[str], db="battery_test_db", host="localhost", port=27017, cycle=1):
    """Plot dQ/dV for multiple cells by code."""
    connect(db, host=host, port=port)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    for idx, code in enumerate(cell_codes):
        tests = find_tests_by_cell_code(code)
        if not tests:
            logging.warning(f"No tests found for {code}")
            continue

        test = tests[0]
        logging.info(f"Processing {test.name} (cycle {cycle})")

        try:
            dqdv_data = extract_dqdv_from_gridfs(test, cycle)
        except Exception as e:
            logging.error(f"Could not extract dQ/dV for {code}: {e}")
            continue

        color = cmap(idx % 10)

        # Plot charge
        if dqdv_data.get("charge"):
            ax.plot(
                dqdv_data["charge"]["voltage"],
                dqdv_data["charge"]["dqdv"],
                label=f"{code} (Charge)",
                color=color,
                linestyle="-"
            )

        # Plot discharge
        if dqdv_data.get("discharge"):
            ax.plot(
                dqdv_data["discharge"]["voltage"],
                dqdv_data["discharge"]["dqdv"],
                label=f"{code} (Discharge)",
                color=color,
                linestyle="--"
            )

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("dQ/dV (mAh/V)")
    ax.set_title(f"dQ/dV Analysis – Cycle {cycle}")
    ax.legend(fontsize="small")
    plt.tight_layout()
    plt.show()


# ────────────────────────── DIFFERENCE PLOT ──────────────────────────
def plot_dqdv_difference_first_second(cell_codes: List[str], db="battery_test_db", host="localhost", port=27017):
    """
    Plot difference in dQ/dV (charge) between cycle 1 and 2 for multiple cells.
    """
    connect(db, host=host, port=port)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    for idx, code in enumerate(cell_codes):
        tests = find_tests_by_cell_code(code)
        if not tests:
            logging.warning(f"No tests found for {code}")
            continue

        test = tests[0]
        logging.info(f"Processing difference plot for {test.name}")

        try:
            dqdv_1 = extract_dqdv_from_gridfs(test, 1)
            dqdv_2 = extract_dqdv_from_gridfs(test, 2)
        except Exception as e:
            logging.error(f"Could not extract dQ/dV difference for {code}: {e}")
            continue

        if "charge" not in dqdv_1 or "charge" not in dqdv_2:
            logging.warning(f"No charge dQ/dV data for {code}")
            continue

        v1, dq1 = np.array(dqdv_1["charge"]["voltage"]), np.array(dqdv_1["charge"]["dqdv"])
        v2, dq2 = np.array(dqdv_2["charge"]["voltage"]), np.array(dqdv_2["charge"]["dqdv"])

        # Common voltage grid
        v_min, v_max = max(v1.min(), v2.min()), min(v1.max(), v2.max())
        v_common = np.linspace(v_min, v_max, 500)

        dq1_interp = np.interp(v_common, v1, dq1)
        dq2_interp = np.interp(v_common, v2, dq2)

        dq_diff = dq1_interp - dq2_interp

        ax.plot(v_common, dq_diff, label=f"{code} Δ(1st-2nd)", color=cmap(idx % 10))

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Δ dQ/dV (mAh/V)")
    ax.set_title("dQ/dV Difference (Cycle 1 – Cycle 2, Charge)")
    ax.legend(fontsize="small")
    plt.tight_layout()
    plt.show()


# ────────────────────────── MAIN ──────────────────────────
if __name__ == "__main__":
    # Example usage:
    plot_dqdv_from_mongo(["GJ01", "GK01"], db="battery_test_db", host="localhost", port=27017, cycle=1)
    plot_dqdv_difference_first_second(["GJ01", "GK01"], db="battery_test_db", host="localhost", port=27017)
