#!/usr/bin/env python3
"""
Generate per-cell Voltage–Capacity plots (charge & discharge) split by cycle
from Arbin CSV exports with headers at line 26.

Usage examples:
  python plot_vq_by_cycle.py --input "path/to/-21°C Cycling" --out plots_21C
  python plot_vq_by_cycle.py --input "path/to/-21°C Cycling" --max-cycles 10
  python plot_vq_by_cycle.py --input "path/to/-21°C Cycling" --only-discharge
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_arbin_csv(csv_path: Path, header_line: int = 25) -> pd.DataFrame:
    """
    Load an Arbin CSV that contains metadata rows followed by a header row.
    Default: header at line 26 (0-indexed header=25).
    """
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python", header=header_line, on_bad_lines="skip")
    except Exception as e:
        raise RuntimeError(f"Failed to read {csv_path}: {e}")

    # Coerce relevant numeric columns; silently ignore if a column is missing
    for col in ["Voltage (V)", "Charge Capacity (mAh)", "Discharge Capacity (mAh)", "Cycle"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def detect_cell_label(df: pd.DataFrame, fallback: str = "Cell") -> str:
    """
    Use 'Cell ID' column if present; otherwise fallback.
    """
    if "Cell ID" in df.columns and pd.api.types.is_string_dtype(df["Cell ID"]):
        first = df["Cell ID"].dropna().astype(str)
        if not first.empty and first.iloc[0].strip():
            return first.iloc[0].strip()
    return fallback


def plot_voltage_capacity_per_cycle(
    df: pd.DataFrame,
    cell_label: str,
    save_path: Path,
    max_cycles: int | None = None,
    skip_cycles: set[int] | None = None,
    only_charge: bool = False,
    only_discharge: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    dpi: int = 150,
):
    """
    Create a per-cycle V–Q plot. Charge = solid, Discharge = dashed.
    """
    required = ["Cycle", "Voltage (V)"]
    if not all(c in df.columns for c in required):
        missing = [c for c in required if c not in df.columns]
        raise ValueError(f"Missing columns for plotting: {missing}")

    # Filter and prep
    df = df.copy()
    df = df.dropna(subset=["Voltage (V)"])
    # Helpful to fill NaNs in capacity with 0 for aesthetic axes, but safer to leave NaNs out:
    # We'll drop rows with NaN for the specific series we plot.

    # Determine cycle list
    cycles = (
        df["Cycle"]
        .dropna()
        .astype(int)
        .tolist()
    )
    if not cycles:
        raise ValueError("No cycle numbers found.")
    unique_cycles = sorted(set(c for c in cycles if c > 0))  # skip 0/idle/etc

    if skip_cycles:
        unique_cycles = [c for c in unique_cycles if c not in skip_cycles]
    if max_cycles is not None:
        unique_cycles = unique_cycles[:max_cycles]

    # Make the plot
    plt.figure(figsize=(9, 7))

    for cyc in unique_cycles:
        grp = df[df["Cycle"] == cyc]

        # Charge curve
        if not only_discharge and "Charge Capacity (mAh)" in df.columns:
            chg = grp.dropna(subset=["Charge Capacity (mAh)", "Voltage (V)"])
            if not chg.empty:
                plt.plot(
                    chg["Charge Capacity (mAh)"],
                    chg["Voltage (V)"],
                    label=f"Cycle {cyc} - Chg",
                    linewidth=1.6,
                )

        # Discharge curve
        if not only_charge and "Discharge Capacity (mAh)" in df.columns:
            dis = grp.dropna(subset=["Discharge Capacity (mAh)", "Voltage (V)"])
            if not dis.empty:
                plt.plot(
                    dis["Discharge Capacity (mAh)"],
                    dis["Voltage (V)"],
                    linestyle="--",
                    label=f"Cycle {cyc} - Dis",
                    linewidth=1.6,
                )

    plt.title(f"Voltage vs Capacity — {cell_label} (−21°C)")
    plt.xlabel("Capacity (mAh)")
    plt.ylabel("Voltage (V)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9, frameon=True)

    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def find_csvs(input_dir: Path) -> list[Path]:
    """
    Recursively find CSV files under input_dir.
    """
    return [p for p in input_dir.rglob("*.csv") if p.is_file()]

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ... (keep all function definitions unchanged)

def main():
    input_dir = Path(r"C:\Users\benja\Downloads\149 Data-20251006T145752Z-1-001\149 Data\-21°C Cycling")
    out_dir = Path(r"C:\Users\benja\Downloads\149 Data-20251006T145752Z-1-001\149 Data\-21°C Cycling\vq_plots_-21C")
    max_cycles = None
    skip_cycles = set()
    only_charge = False
    only_discharge = False
    xlim = None
    ylim = [2.8, 4.4]
    dpi = 150

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    csvs = find_csvs(input_dir)
    if not csvs:
        print(f"No CSV files found under: {input_dir}")
        return

    print(f"Found {len(csvs)} CSV file(s). Generating plots...")
    for csv_path in sorted(csvs):
        try:
            df = load_arbin_csv(csv_path)
            cell_label = detect_cell_label(df, fallback=csv_path.parent.name or "Cell")
            safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in cell_label)
            out_path = out_dir / f"{safe_label}_VQ_by_cycle.png"

            plot_voltage_capacity_per_cycle(
                df=df,
                cell_label=cell_label,
                save_path=out_path,
                max_cycles=max_cycles,
                skip_cycles=skip_cycles,
                only_charge=only_charge,
                only_discharge=only_discharge,
                xlim=xlim,
                ylim=ylim,
                dpi=dpi,
            )
            print(f"  ✓ {cell_label}: {out_path}")
        except Exception as e:
            print(f"  ✗ Failed on {csv_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

