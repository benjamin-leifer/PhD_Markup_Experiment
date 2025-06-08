"""Utilities for analyzing raw Arbin cycling files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt


def _find_column(df: pd.DataFrame, patterns: list[str]) -> str | None:
    """Return first column name containing any of the patterns."""
    for col in df.columns:
        for pat in patterns:
            if pat in str(col).lower():
                return col
    return None


def read_arbin_file(file_path: str) -> pd.DataFrame:
    """Load a raw Arbin cycling file into a DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the raw Arbin file (CSV, DAT, or MPT).

    Returns
    -------
    pandas.DataFrame
        DataFrame with standardized column names: ``cycle``, ``step_type``,
        ``current_A``, ``capacity_mAh``, ``voltage_V``, ``timestamp``.
    """
    ext = Path(file_path).suffix.lower()
    if ext not in {".csv", ".dat", ".mpt", ".txt"}:
        raise ValueError(f"Unsupported file extension: {ext}")

    df = pd.read_csv(file_path, sep=None, engine="python")

    cycle_col = _find_column(df, ["cycle", "cycle_index", "cycle number"])
    if not cycle_col:
        raise ValueError("Cycle number column not found")

    step_col = _find_column(df, ["step type", "mode", "state"])
    current_col = _find_column(df, ["current", "curr", "i("])
    capacity_col = _find_column(df, ["capacity"])
    voltage_col = _find_column(df, ["volt"])
    time_col = _find_column(df, ["timestamp", "time", "date"])

    df = df.rename(
        columns={
            cycle_col: "cycle",
            **({step_col: "step_type"} if step_col else {}),
            current_col: "current_A",
            capacity_col: "capacity_mAh",
            voltage_col: "voltage_V",
            time_col: "timestamp",
        }
    )

    required = ["cycle", "current_A", "capacity_mAh", "voltage_V", "timestamp"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in file")
    if "step_type" not in df.columns:
        df["step_type"] = ""

    return df[["cycle", "step_type", "current_A", "capacity_mAh", "voltage_V", "timestamp"]]


def split_cycles_by_rate(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split cycles into predefined rate segments.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame returned by :func:`read_arbin_file`.

    Returns
    -------
    dict
        Mapping of segment names to DataFrames filtered to those cycles.
    """
    cycle_col = _find_column(df, ["cycle"])
    if not cycle_col:
        raise ValueError("Cycle column not found")

    cycles = df[cycle_col].astype(int)
    segments: Dict[str, pd.DataFrame] = {}

    def subset(cycle_list):
        return df[cycles.isin(cycle_list)].copy()

    segments["formation"] = subset([1])

    blocks = {
        "rate_C10": range(2, 5),
        "rate_C8": range(5, 8),
        "rate_C4": range(8, 11),
        "rate_C2": range(11, 14),
        "rate_1C": range(14, 17),
        "rate_2C": range(17, 20),
    }
    max_cycle = cycles.max()
    for name, rng in blocks.items():
        rng_list = [c for c in rng if c <= max_cycle]
        segments[name] = subset(rng_list)

    segments["long_term"] = df[cycles >= 20].copy()
    segments["cap_check_50"] = df[cycles % 50 == 0].copy()
    return segments


def plot_cycle_segments(segments: Dict[str, pd.DataFrame], out_dir: str, overlay: bool = False) -> None:
    """Plot voltage vs. capacity for each cycle segment.

    Parameters
    ----------
    segments : dict
        Segment dictionary returned by :func:`split_cycles_by_rate`.
    out_dir : str
        Directory where PNG files will be saved.
    overlay : bool, optional
        If ``True`` overlay all cycles of a segment in a single plot.
    """
    os.makedirs(out_dir, exist_ok=True)

    for name, seg in segments.items():
        if seg.empty:
            continue
        cycle_col = _find_column(seg, ["cycle"])
        cap_col = _find_column(seg, ["capacity"])
        volt_col = _find_column(seg, ["volt"])
        fig, ax = plt.subplots()
        if overlay:
            for cyc, grp in seg.groupby(cycle_col):
                ax.plot(grp[cap_col], grp[volt_col], label=f"cycle {cyc}")
            if seg[cycle_col].nunique() > 1:
                ax.legend()
        else:
            ax.plot(seg[cap_col], seg[volt_col])
        ax.set_xlabel("Capacity (mAh)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}.png"))
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Command-line interface for analyzing Arbin files."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Arbin cycling files")
    parser.add_argument("files", nargs="+", help="Arbin files to analyze")
    parser.add_argument("--out-dir", default="arbin_plots", help="Output directory")
    parser.add_argument("--overlay", action="store_true", help="Overlay rate step cycles")
    args = parser.parse_args(argv)

    for fp in args.files:
        df = read_arbin_file(fp)
        segments = split_cycles_by_rate(df)
        plot_cycle_segments(segments, args.out_dir, overlay=args.overlay)
        print(f"Summary for {fp}:")
        for name, seg in segments.items():
            print(f"  {name:12s} : {seg['cycle'].nunique()} cycles")
        print()


if __name__ == "__main__":
    main()
