#!/usr/bin/env python3
"""Compare cycling tests stored in MongoDB by cell code.

This script replicates the ``compare_cells_on_same_plot`` logic from
``Scratch_t6`` but pulls data from the ``test_results`` collection
instead of Excel files. Tests are selected by searching the ``name``
field for a cell code (two letters followed by two digits). The legend
shows ``<cell code> - <electrolyte>`` based on the Spring 2025 cell list.
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import Iterable, List, Tuple, Dict
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
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


def load_lookup_table(path: str) -> Dict[str, str]:
    """Return mapping of cell code to electrolyte from ``path``.

    The Excel file is expected to contain ``Cell Code`` and ``Electrolyte``
    columns. If the file cannot be read or the columns are missing, an empty
    mapping is returned.
    """
    try:
        df = pd.read_excel(path)
    except Exception as exc:  # File not found or parse error
        logging.warning("Could not read lookup table %s: %s", path, exc)
        return {}

    if "Cell Code" not in df.columns or "Electrolyte" not in df.columns:
        logging.warning(
            "Lookup table %s missing required columns", path
        )
        return {}

    return (
        df.set_index("Cell Code")
        ["Electrolyte"]
        .astype(str)
        .to_dict()
    )


def compare_tests_on_same_plot(
    tests: Iterable[models.TestResult],
    *,
    normalized: bool = False,
    x_bounds: Tuple[int, int] = (0, 100),
    save_str: str | None = None,
    color_scheme: dict[str, str] | None = None,
    electrolyte_lookup: Dict[str, str] | None = None,
) -> None:
    """Plot capacity and coulombic efficiency for multiple tests."""
    tests = list(tests)
    if not tests:
        raise ValueError("No tests provided for comparison.")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    c_rate_labels = {2: "Form", 4: "C/10", 7: "C/8", 10: "C/4", 13: "C/2", 16: "1C", 19: "2C"}
    annotated: set[int] = set()
    electrolyte_lookup = electrolyte_lookup or {}

    for test in tests:
        cycles, charge_caps, discharge_caps, ce = extract_cycle_data(test, normalized)
        name = test.name or "Test"
        cell_match = re.search(r"([A-Za-z]{2}\d{2})", name)
        cell_code = cell_match.group(1) if cell_match else name
        letters_match = re.match(r"[A-Za-z]+", cell_code)
        letters = letters_match.group(0) if letters_match else cell_code
        electrolyte = electrolyte_lookup.get(letters, "Unknown")
        label_base = f"{cell_code} - {electrolyte}"
        color = None
        if color_scheme and cell_code in color_scheme:
            color = color_scheme[cell_code]

        ax1.scatter(cycles, charge_caps, color=color, label=label_base)
        ax2.scatter(cycles, ce, marker="D", color=color, label=f"{label_base} (CE)")

        if x_bounds[1] < 20:
            for cycle, label in c_rate_labels.items():
                if cycle in cycles and cycle not in annotated:
                    ax1.text(cycle - 1, 105 if normalized else max(charge_caps), label,
                             fontsize=10, ha="center", color="black")
                    annotated.add(cycle)

    # for cycle in [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5]:
    #     ax1.axvline(x=cycle, color="black", linestyle="--")

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


def find_tests_by_cell_code(code: str | List[str]) -> List[models.TestResult]:
    """Return tests where `name` contains `code` (case-insensitive), selecting the test with the greatest number of cycles."""
    if isinstance(code, list):
        results = []
        for single_code in code:
            regex = re.compile(single_code, re.IGNORECASE)
            all_tests = models.TestResult.objects(name__regex=regex)
            if all_tests:
                # Select the test with the greatest number of cycles
                best_test = max(all_tests, key=lambda test: len(test.cycles))
                results.append(best_test)
        return results
    else:
        regex = re.compile(code, re.IGNORECASE)
        all_tests = models.TestResult.objects(name__regex=regex)
        if all_tests:
            return [max(all_tests, key=lambda test: len(test.cycles))]
        return []


def plot_mean_capacity_with_std(
    grouped_cell_codes: List[List[str]],
    *,
    normalized: bool = False,
    x_bounds: Tuple[int, int] = (0, 100),
    save_str: str | None = None,
    electrolyte_lookup: Dict[str, str] | None = None,
) -> None:
    """
    Draw one figure with a mean-capacity curve (±STD band) for every
    sub-list in ``grouped_cell_codes``.
    """
    electrolyte_lookup = electrolyte_lookup or {}

    fig, ax = plt.subplots(figsize=(5, 3))

    for group_idx, cell_codes in enumerate(grouped_cell_codes):
        group_charge_caps = []
        group_cycles = None

        # ----- gather data for this group -----
        for code in cell_codes:
            for test in find_tests_by_cell_code(code):
                cycles, charge_caps, _, _ = extract_cycle_data(test, normalized)
                group_charge_caps.append(charge_caps)
                if group_cycles is None:
                    group_cycles = cycles

        if not group_charge_caps:
            logging.warning("No data found for group %d", group_idx + 1)
            continue

        # ----- pad to common length -----
        max_len = max(len(c) for c in group_charge_caps)
        padded = [c + [np.nan] * (max_len - len(c)) for c in group_charge_caps]
        group_cycles = (group_cycles + [np.nan] * (max_len - len(group_cycles)))[:max_len]

        mean_caps = np.nanmean(padded, axis=0)
        std_caps  = np.nanstd(padded,  axis=0)

        # ----- plot -----
        line, = ax.plot(group_cycles, mean_caps, label=f"Group {group_idx+1}")
        ax.fill_between(group_cycles,
                        mean_caps - std_caps,
                        mean_caps + std_caps,
                        alpha=0.20,
                        color=line.get_color())

    # ----- cosmetics -----
    ax.set_xlabel("Cycle Number")
    ax.set_xlim(x_bounds)
    ax.set_ylabel("Capacity (%)" if normalized else "Capacity (mAh)")
    if normalized:
        ax.set_ylim(0, 120)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fontsize="small", ncol=2)
    plt.tight_layout()
    if save_str:
        plt.savefig(f"{save_str}_mean_groups.png", dpi=300)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MongoDB cycling data by cell code")
    parser.add_argument(
        "codes",
        nargs="*",  # now optional
        default=[
            #['EU02', 'EU03'],
            #['FT02', 'FT03','FT04','FT05'],
            #['FU01','FU02', 'FU03','FU04','FU05','FS01', 'FS02',],
            ['FU01', 'FU04', ],#MF91
            ['GB01', 'GB02', ],#DTFV1411
            ['FT04', 'FT05', ], #DTFV1422
            ['GJ04', ], #DTFV1452
            ['GK05',], #DTFV1425
            #['GB01', 'GB02', 'GB03', 'GB04', 'GB06','GB07',],# Example default codes
            #["FM01", "FM02", "FM03"],  # Example default codes
            #["FM04", "FM05", "FM06"],
            #["FK01", "FK02", "FK03"],
            #["FK04", "FK05",],
            #["FS03", "FS04", "FS05"],
            #['FM01', 'FM06', 'FK02', 'FK05', 'FS03'],
            #['FS06',],
            #['FU06',],
            #[ 'FF02',],
            #['FJ02', 'FJ04',],
            #['FR03', ],
            #['FR06','FT06']# Example default codes
            ],  # <= put any sensible default(s) here
        help="Cell codes to search for (default: AA01)",
    )
    parser.add_argument("--db", default="battery_test_db", help="Database name")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=27017)
    parser.add_argument("--normalized", action="store_true", help="Normalize capacity")
    parser.add_argument(
        "--lookup",
        default=r'C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx',
        help="Path to Spring 2025 Cell List.xlsx",
    )
    parser.add_argument("--save", help="File prefix for saving the plot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    connect(args.db, host=args.host, port=args.port)

    electrolyte_lookup = load_lookup_table(args.lookup)

    tests: List[models.TestResult] = []
    for code in args.codes:
        results = find_tests_by_cell_code(code)
        if not results:
            logging.warning("No tests found for code %s", code)
        tests.extend(results)

    if not tests:
        logging.error("No matching tests found.")
        return

    # ---------- NEW: mean ± std first ----------
    grouped_codes = args.codes if isinstance(args.codes[0], list) else [args.codes]  # treat all CLI codes as one group
    plot_mean_capacity_with_std(
        grouped_codes,
        normalized=args.normalized,
        save_str=f"{args.save}_mean" if args.save else None,
        electrolyte_lookup=electrolyte_lookup,
    )

    compare_tests_on_same_plot(
        tests,
        normalized=args.normalized,
        save_str=args.save,
        electrolyte_lookup=electrolyte_lookup,
    )

    for group_idx, cell_codes in enumerate(args.codes):
            # Generate and save the mean ± std plot
            plot_mean_capacity_with_std(
                [cell_codes],
                normalized=args.normalized,
                save_str=f"{args.save}_group_{group_idx + 1}_mean" if args.save else None,
                electrolyte_lookup=electrolyte_lookup,
            )

            # Find tests for the current group
            tests: List[models.TestResult] = []
            for code in cell_codes:
                results = find_tests_by_cell_code(code)
                if not results:
                    logging.warning("No tests found for code %s", code)
                tests.extend(results)

            if not tests:
                logging.warning("No matching tests found for group %d", group_idx + 1)
                continue

            # Generate and save the discharge capacity/CE vs. cycle number plot
            compare_tests_on_same_plot(
                tests,
                normalized=args.normalized,
                save_str=f"{args.save}_group_{group_idx + 1}" if args.save else None,
                electrolyte_lookup=electrolyte_lookup,
            )

if __name__ == "__main__":
    main()



