#!/usr/bin/env python3
"""
compare_mongo_on_same_cell.py  —  standalone version
------------------------------------------------------------------
Plots (1) mean capacity ± standard deviation and (2) individual
capacity / Coulombic-efficiency curves for one or more cell codes
stored in a MongoDB database.

If you double-click or run the script with no arguments it will:

  • connect to the default Mongo DB
  • look up the default cell code(s)
  • create two figures (mean±σ first, then the overlaid curves)

All command-line flags are optional so power users can still
customise behaviour when needed.

Dependencies
------------
• mongoengine (or pymongo if you adapt the connect() call)
• matplotlib, pandas, numpy, etc. (whatever your helper
  functions require)
• local helper modules:
      models.py
      utils.py (must expose the functions imported below)

Adjust the DEFAULT_* constants to suit your environment.
"""

from __future__ import annotations

import argparse
import logging
from typing import List

from mongoengine import connect           # or change to pymongo.MongoClient



# ---- user-editable defaults ------------------------------------------------
DEFAULT_CODES: list[str] = ["AA01"]                     # fallback cell code(s)
DEFAULT_DB: str = "battery_test_db"
DEFAULT_HOST: str = "localhost"
DEFAULT_PORT: int = 27017
DEFAULT_LOOKUP_PATH: str = (
    r"C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx"
)
# ---------------------------------------------------------------------------
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

def parse_args() -> argparse.Namespace:
    """Parse CLI options but make every flag optional."""
    parser = argparse.ArgumentParser(
        description="Plot cycling data from MongoDB by cell code (defaults allow zero-arg execution)."
    )
    parser.add_argument(
        "codes",
        nargs="*",
        default=DEFAULT_CODES,
        help=f"Cell codes (default: {DEFAULT_CODES})",
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="MongoDB database name")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    parser.add_argument(
        "--lookup",
        default=DEFAULT_LOOKUP_PATH,
        help="Path to Excel cell-lookup table",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Normalize capacity curves (off by default)",
    )
    parser.add_argument(
        "--save",
        help="Filename prefix for saving plots (if omitted, plots show interactively)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ------------------------------------------------------------------ DB --
    connect(args.db, host=args.host, port=args.port)
    logging.info("Connected to MongoDB '%s' @ %s:%s", args.db, args.host, args.port)

    electrolyte_lookup = load_lookup_table(args.lookup)

    # -------------------------------- mean ± std FIRST ----------------------
    grouped_codes = [args.codes]  # treat all requested codes as one group
    plot_mean_capacity_with_std(
        grouped_codes,
        normalized=args.normalized,
        save_str=f"{args.save}_mean" if args.save else None,
        electrolyte_lookup=electrolyte_lookup,
    )

    # ----------------------------- individual curves ------------------------
    tests: List[models.TestResult] = []
    for code in args.codes:
        tests.extend(find_tests_by_cell_code(code))

    if not tests:
        logging.error("No matching tests for codes: %s", args.codes)
        return

    compare_tests_on_same_plot(
        tests,
        normalized=args.normalized,
        save_str=args.save,
        electrolyte_lookup=electrolyte_lookup,
    )

    logging.info("Plotting complete.")


if __name__ == "__main__":
    main()
