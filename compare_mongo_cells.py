#!/usr/bin/env python3
"""Compare cycling tests stored in MongoDB by cell code.

This script replicates the ``compare_cells_on_same_plot`` logic from
``Scratch_t6`` but pulls data from the ``test_results`` collection
instead of Excel files. Tests are selected by searching the ``name``
field for a cell code (two letters followed by two digits).
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
from mongoengine import connect

from battery_analysis import models


CycleData = Tuple[List[int], List[float], List[float], List[float]]


def extract_cycle_data(test: models.TestResult, normalized: bool = False) -> CycleData:
    """Return cycle numbers, charge capacity, discharge capacity and CE."""
    cycles_sorted = sorted(test.cycles, key=lambda c: c.cycle_index)
    cycle_numbers = [c.cycle_index for c in cycles_sorted]
    charge_caps = [c.charge_capacity for c in cycles_sorted]
    discharge_caps = [c.discharge_capacity for c in cycles_sorted]
    ce = [c.coulombic_efficiency * 100 for c in cycles_sorted]

    if normalized and discharge_caps:
        norm = discharge_caps[0] or 1.0
        charge_caps = [v / norm * 100 for v in charge_caps]
        discharge_caps = [v / norm * 100 for v in discharge_caps]

    return cycle_numbers, charge_caps, discharge_caps, ce


def format_key(name: str) -> str:
    """Return ``name`` without the trailing cell code."""
    match = re.search(r"([A-Za-z]{2}\d{2})", name)
    if match:
        return name.replace(match.group(1), "").strip()
    return name


def compare_tests_on_same_plot(
    tests: Iterable[models.TestResult],
    *,
    normalized: bool = False,
    x_bounds: Tuple[int, int] = (0, 100),
    save_str: str | None = None,
    color_scheme: dict[str, str] | None = None,
) -> None:
    """Plot capacity and coulombic efficiency for multiple tests."""
    tests = list(tests)
    if not tests:
        raise ValueError("No tests provided for comparison.")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    c_rate_labels = {2: "Form", 4: "C/10", 7: "C/8", 10: "C/4", 13: "C/2", 16: "1C", 19: "2C"}
    annotated: set[int] = set()

    for test in tests:
        cycles, charge_caps, discharge_caps, ce = extract_cycle_data(test, normalized)
        name = test.name or "Test"
        cell_match = re.search(r"([A-Za-z]{2}\d{2})", name)
        cell_code = cell_match.group(1) if cell_match else name
        color = None
        if color_scheme and cell_code in color_scheme:
            color = color_scheme[cell_code]

        ax1.scatter(cycles, charge_caps, color=color, label=format_key(name))
        ax2.scatter(cycles, ce, marker="D", color=color, label=f"{format_key(name)} (CE)")

        if x_bounds[1] < 20:
            for cycle, label in c_rate_labels.items():
                if cycle in cycles and cycle not in annotated:
                    ax1.text(cycle - 1, 105 if normalized else max(charge_caps), label,
                             fontsize=10, ha="center", color="black")
                    annotated.add(cycle)

    for cycle in [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5]:
        ax1.axvline(x=cycle, color="black", linestyle="--")

    ax1.set_xlabel("Cycle Number")
    ax1.set_xlim(x_bounds)
    if normalized:
        ax1.set_ylabel("Capacity (%)")
        ax1.set_ylim(0, 120)
    else:
        ax1.set_ylabel("Capacity (mAh)")
    ax2.set_ylabel("Coulombic Efficiency (%)")
    ax2.set_ylim(0, 120)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper center",
               bbox_to_anchor=(0.5, -0.15), fontsize="small", ncol=2)

    plt.tight_layout()
    if save_str:
        plt.savefig(f"{save_str}_comparison.png", dpi=300)
    plt.show()


def find_tests_by_cell_code(code: str) -> List[models.TestResult]:
    """Return tests where ``name`` contains ``code`` (case-insensitive)."""
    regex = re.compile(code, re.IGNORECASE)
    return list(models.TestResult.objects(name__regex=regex))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MongoDB cycling data by cell code")
    parser.add_argument("codes", nargs="+", help="Cell codes to search for, e.g. AA01")
    parser.add_argument("--db", default="battery_test_db", help="Database name")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=27017)
    parser.add_argument("--normalized", action="store_true", help="Normalize capacity")
    parser.add_argument("--save", help="File prefix for saving the plot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    connect(args.db, host=args.host, port=args.port)

    tests: List[models.TestResult] = []
    for code in args.codes:
        results = find_tests_by_cell_code(code)
        if not results:
            logging.warning("No tests found for code %s", code)
        tests.extend(results)

    if not tests:
        logging.error("No matching tests found.")
        return

    compare_tests_on_same_plot(tests, normalized=args.normalized, save_str=args.save)


if __name__ == "__main__":
    main()
