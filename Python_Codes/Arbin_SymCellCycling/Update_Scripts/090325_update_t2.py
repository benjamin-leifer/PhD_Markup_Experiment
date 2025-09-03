#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Stream-processing + alpha-group mean±stdev bands

Pipeline
--------
1) Discover Excel files recursively under --root.
2) For each file:
   • Parse Arbin-like Sheet 1 (Cycle/Current/ChargeCap/DischargeCap).
   • Compute per-cycle metrics (Qdis, Qchg, CE%).
   • Append to --out-csv immediately.
   • Skip any (Cell, Cycle) rows already in the CSV (resume-safe).
3) Reload the CSV once and plot alpha-group mean ± stdev bands:
   • Left y-axis: Discharge Capacity (mAh)
   • Right y-axis: Coulombic Efficiency (%)
   • Legend entries are per-alpha (e.g., "AA (cells=3)").

Notes
-----
- Styling keeps the crisp vibe used across your figures, but per-alpha bands
  use a clean, deterministic color per alpha (solid line + shaded ±1σ).
- If you want mAh g⁻¹ later, we can extend with --mass-csv (Cell -> mass_g).
"""

from __future__ import annotations
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ───────────────────────── Logging ─────────────────────────
def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ───────────────────────── Aesthetics ─────────────────────────
# Base palette (stable, high-contrast)
PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
    "#56B4E9", "#F0E442", "#000000", "#999999", "#7F7F7F",
]
# Fixed palette: Okabe–Ito colorblind-safe, plus MF91 override
ELECTROLYTE_COLOR = {
    "DTFV1411": "#0072B2",  # blue
    "DTFV1422": "#E69F00",  # orange
    "DTFV1452": "#009E73",  # green
    "DTFV1425": "#D55E00",  # vermillion
    "DTV1410":  "#CC79A7",  # reddish purple
    "DTV142":   "#56B4E9",  # sky blue
    "DTF1410":  "#F0E442",  # yellow
    "DTF142":   "#000000",  # black
    "MF91":     "#FF0000",  # red (special case)
    "DTFV1411 -repeat": "#0072B2",  # blue
    "DTFV1422 - repeat": "#E69F00",  # orange
}
# Marker styles to distinguish close electrolyte variants
ELECTROLYTE_MARKER = {
    "DTFV1411": "o",
    "DTFV1422": "o",
    "DTFV1452": "o",
    "DTFV1452-new": "o",   # distinguish variant by marker
    "DTFV1425": "o",
    "DTV1410":  "o",
    "DTV142":   "o",
    "DTF1410":  "o",
    "DTF142":   "o",
    "MF91":     "o",
    "DTFV1411 -repeat":"*",
    "DTFV1422 - repeat": "*",
}
# Manual overrides: force certain alphas to use a specific cell
# Example: {"GN": "GN03", "AS": "AS05"}
CELL_OVERRIDES: Dict[str, str] = {
    "GW":"GW01",
    "GT": "GT01",
    "GV": "GV02",
}


# Map alpha codes (two-letter prefixes) to electrolyte formulations
electrolyte_lookup: Dict[str, str] = {
    "AS": "DT14",
    "AT": "DTFV1425",
    "AU": "DT14",
    "FT": "DTFV1422",
    "FU": "MF91",
    "GB": "DTFV1411",
    "GN": "DTFV1452",
    "GO": "DTFV1425",
    "GW": "DTFV1411 -repeat",
    "GX": "DTFV1422 - repeat",
    "GV": "DTV1410",
    "GU": "DTV142",
    "GT": "DTF1410",
    "GS": "DTF142",
    "GY": "DT14",
    "GJ": "DTFV1452",
    "GK": "DTFV1425",


    # extend as needed...
}

_alpha_color_cache: Dict[str, str] = {}

def color_for_alpha(alpha: str) -> str:
    if alpha not in _alpha_color_cache:
        _alpha_color_cache[alpha] = PALETTE[abs(hash(alpha)) % len(PALETTE)]
    return _alpha_color_cache[alpha]


# ───────────────────────── File discovery ─────────────────────────
def find_excel_files(root: Path) -> List[Path]:
    exts = (".xlsx", ".xls")
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    logging.info("Found %d Excel files under %s", len(files), root)
    return files


# ───────────────────────── Helpers ─────────────────────────
def cell_code_from_name(fname: str) -> str:
    """Extract codes like 'AA01' from filename; fall back to last 4 chars of stem."""
    m = re.search(r"[A-Za-z]{2}\d{2}", fname)
    if m:
        return m.group(0).upper()
    return Path(fname).stem[-4:].upper()

def alpha_bucket(cell_code: str) -> str:
    """Return 2-letter alphabetical bucket from codes like 'AA' in 'AA01'."""
    m = re.match(r"([A-Za-z]{2})\d{0,2}", cell_code)
    return m.group(1).upper() if m else cell_code[:2].upper()

def dataset_id_from_name(fname: str) -> str:
    """
    Return a stable dataset id for chunked Excel files.
    Example:
      'BL-LL-AS01_Channel_4_Wb_1.xlsx' -> 'BL-LL-AS01_Channel_4'
      'BL-LL-AS01_Channel_4_Wb_2.xlsx' -> 'BL-LL-AS01_Channel_4'
    We strip a trailing '_Wb_<num>' that appears before the extension.
    """
    stem = Path(fname).stem  # filename w/o extension
    # Remove a final '_Wb_<digits>' chunk, case-insensitive
    stem = re.sub(r"(?i)_wb_\d+$", "", stem)
    return stem
def plot_best_cells_from_csv(csv_path: Path, save_path: Optional[Path] = None, dpi: int = 200) -> None:
    """
    Load CSV and plot the *best single cell* per alpha group.
    Best cell = one with highest maximum discharge capacity (Qdis).
    Legend includes cell code and electrolyte formulation (from alpha).
    Excludes the final cycle from plotting.
    """
    tab = pd.read_csv(csv_path)

    if tab.empty:
        logging.error("CSV is empty; nothing to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(8.8, 5.4))
    ax2 = ax1.twinx()

    for alpha, g in tab.groupby("Alpha"):
        # Choose cell: use override if available, else pick best
        if alpha in CELL_OVERRIDES:
            best_cell = CELL_OVERRIDES[alpha]
            logging.info("Override: using %s for alpha %s", best_cell, alpha)
        else:
            best_cell = (
                g.groupby("Cell")["Qdis"]
                .max()
                .sort_values(ascending=False)
                .index[0]
            )

        g_best = g[g["Cell"] == best_cell].sort_values("Cycle")

        # Exclude last cycle
        if len(g_best) > 1:
            g_best = g_best.iloc[:-1]

        # Map alpha → electrolyte and pick style
        electrolyte = electrolyte_lookup.get(alpha, "Unknown")
        color = ELECTROLYTE_COLOR.get(electrolyte, "#7F7F7F")  # default gray
        marker = ELECTROLYTE_MARKER.get(electrolyte, "o")

        legend_label = f"{best_cell} ({electrolyte})"

        # Plot discharge capacity
        ax1.plot(
            g_best["Cycle"], g_best["Qdis"],
            color=color, lw=2.2, marker=marker, label=legend_label
        )
        # Plot CE
        ax2.plot(
            g_best["Cycle"], g_best["CE_pct"],
            color=color, lw=1.5, linestyle=":"
        )

    # Labels & style
    ax1.set_xlabel("Cycle Number")
    ax1.set_ylabel("Discharge Capacity (mAh)")
    ax2.set_ylabel("Coulombic Efficiency (%)")
    ax1.set_xlim(0, 100)
    ax2.set_ylim(0, 110)

    ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis="y", direction="in", right=True)
    ax1.grid(True, alpha=0.22)

    ax1.legend(loc="best", fontsize="small", frameon=True)
    ax1.set_title("Best Cell per Alpha: Discharge Capacity & CE vs Cycle (Excluding Last Cycle)")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logging.info("Saved figure to %s (dpi=%d)", save_path, dpi)

    plt.show()



def _clean(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ───────────────────────── Data ingest ─────────────────────────
def load_arbin_sheet1(path: Path) -> pd.DataFrame:
    """
    Return dataframe with columns:
      ['Cycle','Current_A','Qchg_Ah','Qdis_Ah']
    Pulls sheet index 1 (2nd sheet), Arbin-style.
    """
    engine = "openpyxl" if path.suffix.lower() == ".xlsx" else None
    df0 = pd.read_excel(path, sheet_name=1, engine=engine)

    cmap = {_clean(c): c for c in df0.columns}
    cyc_key = next((cmap[k] for k in cmap if k.startswith("cycleindex") or k.startswith("cyclenumber")), None)
    cur_key = next((cmap[k] for k in cmap if k.startswith("currenta") or k.startswith("current")), None)
    chg_key = next((cmap[k] for k in cmap if k.startswith("chargecapacityah")), None)
    dch_key = next((cmap[k] for k in cmap if k.startswith("dischargecapacityah")), None)

    logging.debug("Column mapping in %s -> Cycle=%s, Current=%s, Qchg=%s, Qdis=%s",
                  path.name, cyc_key, cur_key, chg_key, dch_key)

    if None in (cyc_key, cur_key, chg_key, dch_key):
        raise KeyError(f"Missing required columns in {path.name}")

    df = df0[[cyc_key, cur_key, chg_key, dch_key]].copy()
    df.columns = ["Cycle", "Current_A", "Qchg_Ah", "Qdis_Ah"]
    return df


def per_cycle_metrics(df: pd.DataFrame, mass_g: Optional[float] = None) -> pd.DataFrame:
    """
    Compute per-cycle discharge capacity (mAh or mAh g^-1) and CE%.
    CE_n = Qdis_n / Qchg_n * 100 (0 if denom is 0).
    """
    out = []
    for cyc, g in df.groupby("Cycle"):
        g = g.copy()
        g["Qchg_mAh"] = g["Qchg_Ah"] * 1000.0
        g["Qdis_mAh"] = g["Qdis_Ah"] * 1000.0

        chg = g.loc[g["Current_A"] > 0, "Qchg_mAh"].max() if (g["Current_A"] > 0).any() else g["Qchg_mAh"].max()
        dis = g.loc[g["Current_A"] < 0, "Qdis_mAh"].max() if (g["Current_A"] < 0).any() else g["Qdis_mAh"].max()

        if mass_g and mass_g > 0:
            chg, dis = chg / mass_g, dis / mass_g

        ce = 100.0 * (dis / chg) if chg and np.isfinite(chg) and chg != 0 else 0.0
        out.append({"Cycle": int(cyc), "Qdis": dis, "Qchg": chg, "CE_pct": ce})

    res = pd.DataFrame(out).sort_values("Cycle").reset_index(drop=True)
    logging.debug("Per-cycle metrics: %d cycles; head:\n%s", len(res), res.head(3))
    return res


# ───────────────────────── Streaming CSV I/O ─────────────────────────
# old:
# CSV_COLS = ["Cell", "Alpha", "Cycle", "Qdis", "Qchg", "CE_pct"]

# new:
CSV_COLS = ["Cell", "Alpha", "Dataset", "Cycle", "Qdis", "Qchg", "CE_pct", "SourceFile"]


def load_already_imported(out_csv: Path) -> set[tuple[str, str, int]]:
    """
    Return a set of (Cell, Dataset, Cycle) already present in out_csv.
    Backward-compatible: if CSV lacks 'Dataset', we use '*' as a wildcard dataset.
    """
    if not out_csv.exists():
        return set()
    try:
        prev = pd.read_csv(out_csv)
        # Backward compatibility (old CSV without Dataset column)
        if "Dataset" not in prev.columns:
            prev["Dataset"] = "*"
        s = {
            (str(r.Cell).upper(), str(r.Dataset), int(r.Cycle))
            for r in prev.itertuples(index=False)
            if "Cell" in prev.columns and "Cycle" in prev.columns
        }
        logging.info("Resume mode: %d rows already present in %s", len(prev), out_csv)
        return s
    except Exception as e:
        logging.exception("Failed to read existing CSV (%s); proceeding without resume: %s", out_csv, e)
        return set()


def append_metrics_to_csv(cell: str, alpha: str, dataset: str, met: pd.DataFrame, out_csv: Path, source_path: Path) -> int:
    """
    Append rows to CSV with schema:
    Cell, Alpha, Dataset, Cycle, Qdis, Qchg, CE_pct, SourceFile
    """
    df2 = met.copy()
    df2["Cell"] = cell
    df2["Alpha"] = alpha
    df2["Dataset"] = dataset
    df2["SourceFile"] = str(source_path)
    df2 = df2[["Cell", "Alpha", "Dataset", "Cycle", "Qdis", "Qchg", "CE_pct", "SourceFile"]]
    mode = "a" if out_csv.exists() else "w"
    header = not out_csv.exists()
    df2.to_csv(out_csv, mode=mode, header=header, index=False)
    return len(df2)



# ───────────────────────── Plotting (alpha bands) ─────────────────────────
def plot_alpha_bands_from_csv(csv_path: Path, save_path: Optional[Path] = None, dpi: int = 200) -> None:
    """
    Load the accumulated CSV and draw per-alpha mean ± stdev bands
    for Discharge Capacity and CE%.
    """
    tab = pd.read_csv(csv_path)

    if tab.empty:
        logging.error("CSV is empty; nothing to plot.")
        return

    # Compute per-alpha, per-cycle stats
    grouped = tab.groupby(["Alpha", "Cycle"]).agg(
        Qdis_mean=("Qdis", "mean"),
        Qdis_std=("Qdis", "std"),
        CE_mean=("CE_pct", "mean"),
        CE_std=("CE_pct", "std"),
        n=("Cell", "nunique"),  # number of unique cells contributing to this point
    ).reset_index()

    # Also get per-alpha unique cell counts for legend
    cells_per_alpha = tab.groupby("Alpha")["Cell"].nunique().to_dict()

    fig, ax1 = plt.subplots(figsize=(8.8, 5.4))
    ax2 = ax1.twinx()

    for alpha, g in grouped.groupby("Alpha"):
        g = g.sort_values("Cycle")
        color = color_for_alpha(alpha)
        n_cells = cells_per_alpha.get(alpha, int(g["n"].max() if len(g) else 0))

        # Discharge capacity mean line + ±1σ band
        ax1.plot(g["Cycle"], g["Qdis_mean"], color=color, lw=2.2, label=f"{alpha} (cells={n_cells})")
        if np.isfinite(g["Qdis_std"]).any():
            y1 = g["Qdis_mean"] - g["Qdis_std"]
            y2 = g["Qdis_mean"] + g["Qdis_std"]
            ax1.fill_between(g["Cycle"], y1, y2, color=color, alpha=0.2, linewidth=0)

        # CE mean line + ±1σ band (dotted line, same color)
        ax2.plot(g["Cycle"], g["CE_mean"], color=color, lw=1.9, linestyle=":")
        if np.isfinite(g["CE_std"]).any():
            y1c = g["CE_mean"] - g["CE_std"]
            y2c = g["CE_mean"] + g["CE_std"]
            ax2.fill_between(g["Cycle"], y1c, y2c, color=color, alpha=0.12, linewidth=0)

    # Labels & styling
    ax1.set_xlabel("Cycle Number")
    ax1.set_ylabel("Discharge Capacity (mAh)")
    ax2.set_ylabel("Coulombic Efficiency (%)")
    ax2.set_ylim(0, 110)

    ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis="y", direction="in", right=True)
    ax1.grid(True, alpha=0.22)

    leg = ax1.legend(loc="best", fontsize="small", frameon=True, title="Alpha groups")
    if leg and hasattr(leg, "get_frame"):
        leg.get_frame().set_linewidth(1.2)

    ax1.set_title("Discharge Capacity (mAh) and CE (%) vs Cycle — Alpha Mean ± 1σ")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logging.info("Saved figure to %s (dpi=%d)", save_path, dpi)

    plt.show()


# ───────────────────────── Main ─────────────────────────
# ───────────────────────── Main ─────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Stream Excel → CSV; then plot alpha-group mean±stdev bands from CSV."
    )
    # Set your default paths here so PyCharm runs without CLI args
    ap.add_argument("--root", type=str, default=r"C:\Users\benja\Downloads\Temp\C_10 Cycling\2025\08\C_10 Sept Update",
                    help="Root folder to scan recursively for Excel files.")
    ap.add_argument("--out-csv", type=str, default=r"C:\Users\benja\Downloads\Temp\C_10 Cycling\2025\08\C_10 Sept Update\metrics.csv",
                    help="Path to append per-cycle metrics CSV.")
    ap.add_argument("--save", type=str, default=r"C:\Users\benja\Downloads\Temp\C_10 Cycling\2025\08\C_10 Sept Update\plot.png",
                    help="Optional path to save the plotted figure (e.g., out.png)")
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI when saving (default: 200)")
    ap.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR")

    # If you want defaults when running inside PyCharm, force empty args:
    args = ap.parse_args([])

    setup_logging(args.log_level)
    root = Path(args.root)
    out_csv = Path(args.out_csv)

    if not root.exists():
        logging.error("Root path does not exist: %s", root)
        return
    save_path = Path(args.save) if args.save else None
    plot_best_cells_from_csv(out_csv, save_path=save_path, dpi=args.dpi)
    # Resume set of (Cell, Cycle)
    imported = load_already_imported(out_csv)

    files = find_excel_files(root)
    if not files:
        logging.error("No Excel files found.")
        return

    processed_files = 0
    for idx, f in enumerate(files, start=1):
        logging.info("[%d/%d] %s", idx, len(files), f.name)
        cell = cell_code_from_name(f.name)
        alpha = alpha_bucket(cell)
        dataset = dataset_id_from_name(f.name)  # <-- NEW

        try:
            df = load_arbin_sheet1(f)
            met = per_cycle_metrics(df)

            # Skip cycles already imported FOR THIS (cell, dataset)
            if imported:
                mask = ~met["Cycle"].astype(int).apply(lambda c: (cell, dataset, int(c)) in imported)
                met_new = met.loc[mask].copy()
            else:
                met_new = met

            if met_new.empty:
                logging.info("  Skip %s (%s): all cycles already in CSV", cell, dataset)
                continue

            appended = append_metrics_to_csv(cell, alpha, dataset, met_new, out_csv, f)
            imported.update({(cell, dataset, int(r.Cycle)) for r in met_new.itertuples()})
            logging.info("  Appended %d cycle rows for %s (alpha=%s, dataset=%s)", appended, cell, alpha, dataset)

            if appended > 0:
                processed_files += 1


        except Exception as e:
            logging.exception("  Failed on %s: %s", f.name, e)

    if processed_files == 0 and not out_csv.exists():
        logging.error("No data appended and CSV does not exist. Nothing to plot.")
        return

    save_path = Path(args.save) if args.save else None
    plot_best_cells_from_csv(out_csv, save_path=save_path, dpi=args.dpi)

    logging.info("Done.")



if __name__ == "__main__":
    main()
