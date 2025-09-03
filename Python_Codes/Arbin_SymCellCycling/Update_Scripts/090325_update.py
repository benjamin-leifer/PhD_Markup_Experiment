#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Plot discharge capacity (y1) and Coulombic efficiency (y2) vs cycle number
for all Excel cycling files found under a directory (recursively).

• Data ingestion mirrors your "7_24" approach to Arbin exports (sheet 1,
  fuzzy column matching for Cycle/Current/ChargeCap/DischargeCap).
• Visualization matches your Dq_DV top-of-file style guidelines:
  - Chemistry-driven color families
  - Marker choice by additive flags (F/V)
  - Linestyle by test type (e.g., "C/20", "PITT")
  - Legend as "{Cell code} - {Electrolyte}"

Extras:
- Detailed logging of file discovery, parsing, column mapping, per-file cycle stats.
- Optional save to image (--save) and CSV export of per-file metrics (--export-csv).
- Grouping utilities to compute mean±stdev by alphabetical code (e.g., "AA" in "AA01").

Usage:
  python plot_capacity_ce_by_cycle.py --root "E:\DataRoot" --save out.png --export-csv metrics.csv --log-level INFO

Author: ChatGPT for Ben
Date: 2025-09-03
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ───────────────────────── Logging Setup ─────────────────────────
def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ───────────────────────── Style System ─────────────────────────
# Keep this aligned with your Dq_DV script's top-of-file mapping
PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
    "#56B4E9", "#F0E442", "#000000", "#999999", "#7F7F7F",
]
BASE_COLOR = {
    "DTFV1411": PALETTE[0],
    "DTFV1422": PALETTE[1],
    "DTFV1452": PALETTE[2],
    "DTFV1425": PALETTE[3],
    "DTV1410":  PALETTE[4],
    "DTV142":   PALETTE[5],
    "DTF1410":  PALETTE[6],
    "DTF142":   PALETTE[7],
    "MF91":     PALETTE[8],
}
TEST_LS = {"C/20": "-", "PITT": "--", "Unknown": "-."}

# Electrolyte labels for legend text
# Extend as you add cells so legends read "{Cell code} - {Electrolyte}"
electrolyte_lookup: Dict[str, str] = {
    "AS01": "DT14 - C/20",
    "AT03": "DTFV1425 - C/20",
    "AU02": "DT14 - C/20",
    "FZ01": "DTFV1422 - C/20",
    "GN01": "DTFV1452 (new) - C/20",
    "GO01": "DTFV1425 - C/20",
    "GW05": "DTFV1411 - C/20",
    "GV05": "DTV1410 - C/20",
    "GU05": "DTV142 - C/20",
    "GT05": "DTF1410 - C/20",
    "GS05": "DTF142 - C/20",
    "GY01": "DT14 - C/20",
    "GY02": "DT14 - C/20",
    # ...update as needed
}

_fallback_color_cache: Dict[str, str] = {}

def canonicalize_base_token(s: str) -> str:
    s = re.sub(r"\s*\(.*?\)\s*", "", s or "")
    return s.split()[0]

def split_base_and_test(elec_label: str) -> Tuple[str, str]:
    if not elec_label or elec_label.lower() == "unknown":
        return "Unknown", "Unknown"
    parts = [p.strip() for p in elec_label.split(" - ") if p.strip()]
    base = parts[0] if parts else "Unknown"
    test = parts[-1] if len(parts) > 1 else "Unknown"
    if "pitt" in test.lower():
        test = "PITT"
    return base, test

def color_for_base(base: str) -> str:
    if base in BASE_COLOR:
        return BASE_COLOR[base]
    if base not in _fallback_color_cache:
        slot = abs(hash(base)) % len(PALETTE)
        _fallback_color_cache[base] = PALETTE[slot]
    return _fallback_color_cache[base]

def parse_electrolyte_code(code: str) -> Dict[str, int | bool]:
    m = re.match(r'^(DT)(F?)(V?)(\d{2})(\d*)$', (code or "").strip().upper())
    if not m:
        return {"has_f": False, "has_v": False, "f_pct": 0, "v_pct": 0}
    _, fflag, vflag, _, tail = m.groups()
    has_f = (fflag == 'F'); has_v = (vflag == 'V')
    f_pct = v_pct = 0
    if has_f and has_v:
        if len(tail) >= 2:
            f_pct, v_pct = int(tail[0]), int(tail[1])
        elif len(tail) == 1:
            f_pct = int(tail[0])
    elif has_f:
        f_pct = int(tail) if tail else 0
    elif has_v:
        v_pct = int(tail) if tail else 0
    return {"has_f": has_f, "has_v": has_v, "f_pct": f_pct, "v_pct": v_pct}

def style_from_code(base_token: str) -> Dict[str, object]:
    info = parse_electrolyte_code(base_token)
    if info["has_f"] and info["has_v"]:
        marker = "o"
    elif info["has_f"]:
        marker = "s"
    elif info["has_v"]:
        marker = "^"
    else:
        marker = "D"
    total = min(10, info["f_pct"] + info["v_pct"])
    lw = 1.8 + 0.12 * total
    return {"marker": marker, "lw": lw, "markevery": 30}

def style_for_cell(cell_id: str, idx_hint: int = 0) -> dict:
    label = electrolyte_lookup.get(cell_id, "Unknown")
    base_raw, test = split_base_and_test(label)
    base = canonicalize_base_token(base_raw)
    color = color_for_base(base)
    linest = TEST_LS.get(test, TEST_LS["Unknown"])
    code_style = style_from_code(base)
    legend = f"{cell_id} - {label}"
    return {
        "color": color,
        "linestyle": linest,
        "marker": code_style["marker"],
        "lw": code_style["lw"],
        "markevery": code_style["markevery"],
        "legend": legend,
    }


# ─────────────────────── Helpers: filenames & grouping ───────────────────
def cell_code_from_name(fname: str) -> str:
    m = re.search(r"[A-Za-z]{2}\d{2}", fname)
    if m:
        return m.group(0).upper()
    m = re.search(r"LL-([A-Za-z0-9]{4})", fname, flags=re.I)
    if m:
        return m.group(1).upper()
    stem = Path(fname).stem.upper()
    m = re.findall(r"[A-Za-z0-9]", stem)
    return "".join(m[-4:]) if len(m) >= 4 else stem

def alpha_bucket(cell_code: str) -> str:
    m = re.match(r"([A-Za-z]{2})\d{0,2}", cell_code)
    return m.group(1).upper() if m else cell_code[:2].upper()


# ───────────────────────── Find Excel files ─────────────────────────
def find_excel_files(root: Path) -> List[Path]:
    exts = (".xlsx", ".xls")
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    logging.info("Discovered %d Excel files under %s", len(files), root)
    for p in files[:5]:
        logging.debug("  example file: %s", p)
    if len(files) > 5:
        logging.debug("  ... (%d more)", len(files) - 5)
    return files


# ───────────────────────── Data ingestion ─────────────────────────
def _clean(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def load_arbin_sheet1(path: Path) -> pd.DataFrame:
    """
    Returns dataframe with columns:
    ['Cycle','Current_A','Qchg_Ah','Qdis_Ah']
    """
    engine = "openpyxl" if path.suffix.lower() == ".xlsx" else None
    df0 = pd.read_excel(path, sheet_name=1, engine=engine)
    cmap = {_clean(c): c for c in df0.columns}

    # Fuzzy matches
    cyc_key = next((cmap[k] for k in cmap if k.startswith("cycleindex") or k.startswith("cyclenumber")), None)
    cur_key = next((cmap[k] for k in cmap if k.startswith("currenta") or k.startswith("current")), None)
    chg_key = next((cmap[k] for k in cmap if k.startswith("chargecapacityah")), None)
    dch_key = next((cmap[k] for k in cmap if k.startswith("dischargecapacityah")), None)

    logging.debug(
        "Column mapping in %s -> Cycle=%s, Current=%s, Qchg=%s, Qdis=%s",
        path.name, cyc_key, cur_key, chg_key, dch_key
    )

    missing = [name for name in [cyc_key, cur_key, chg_key, dch_key] if name is None]
    if missing:
        raise KeyError(f"Missing required columns in {path.name}")

    df = df0[[cyc_key, cur_key, chg_key, dch_key]].copy()
    df.columns = ["Cycle", "Current_A", "Qchg_Ah", "Qdis_Ah"]

    # Basic sanity
    nrows = len(df)
    ncycles = df["Cycle"].nunique()
    logging.debug("Parsed %s: %d rows, %d cycles", path.name, nrows, ncycles)

    return df


# ─────────────────────── Per-cycle metrics ───────────────────────
def per_cycle_metrics(df: pd.DataFrame, mass_g: Optional[float] = None) -> pd.DataFrame:
    """
    Compute per-cycle discharge capacity (mAh or mAh g^-1) and Coulombic efficiency (%).
    CE_n = Qdis_n / Qchg_n * 100 (0 if denominator is 0).
    """
    out = []
    for cyc, g in df.groupby("Cycle"):
        g = g.copy()
        g["Qchg_mAh"] = g["Qchg_Ah"] * 1000.0
        g["Qdis_mAh"] = g["Qdis_Ah"] * 1000.0

        chg = g.loc[g["Current_A"] > 0, "Qchg_mAh"].max() if (g["Current_A"] > 0).any() else g["Qchg_mAh"].max()
        dis = g.loc[g["Current_A"] < 0, "Qdis_mAh"].max() if (g["Current_A"] < 0).any() else g["Qdis_mAh"].max()

        if mass_g and mass_g > 0:
            chg = chg / mass_g
            dis = dis / mass_g

        ce = 100.0 * (dis / chg) if chg and np.isfinite(chg) and chg != 0 else 0.0
        out.append({"Cycle": int(cyc), "Qdis": dis, "Qchg": chg, "CE_pct": ce})

    res = pd.DataFrame(out).sort_values("Cycle").reset_index(drop=True)
    logging.debug("Computed per-cycle metrics: %d cycles (first rows):\n%s", len(res), res.head(3))
    return res


# ─────────────────────── Grouping (mean & stdev) ───────────────────────
def summarize_by_alpha(per_file_series: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    {cell_code -> metrics df}  ->  {alpha_code -> summary df with mean/std per cycle}
    """
    rows = []
    for cell, df in per_file_series.items():
        a = alpha_bucket(cell)
        for _, r in df.iterrows():
            rows.append({"alpha": a, "Cycle": int(r["Cycle"]), "Qdis": r["Qdis"], "CE_pct": r["CE_pct"]})
    tall = pd.DataFrame(rows)
    if tall.empty:
        logging.warning("summarize_by_alpha: no data available")
        return {}

    out = {}
    for a, g in tall.groupby("alpha"):
        gsum = g.groupby("Cycle").agg(
            Qdis_mean=("Qdis", "mean"), Qdis_std=("Qdis", "std"),
            CE_mean=("CE_pct", "mean"), CE_std=("CE_pct", "std"),
        ).reset_index()
        out[a] = gsum
        logging.debug("Alpha %s summary (first rows):\n%s", a, gsum.head(3))
    return out


# ───────────────────────── Plotting ─────────────────────────
def plot_all(files: List[Path],
             save_path: Optional[Path] = None,
             dpi: int = 200,
             masses_g: Optional[Dict[str, float]] = None) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict {cell_code: per_cycle_metrics_df}.
    """
    if not files:
        logging.error("No Excel files found to plot.")
        return {}

    fig, ax1 = plt.subplots(figsize=(8.6, 5.2))
    ax2 = ax1.twinx()

    used_labels = set()
    per_file_metrics: Dict[str, pd.DataFrame] = {}

    logging.info("Beginning parse/plot for %d files ...", len(files))

    for idx, f in enumerate(files, start=1):
        logging.info("[%d/%d] %s", idx, len(files), f.name)
        try:
            df = load_arbin_sheet1(f)
        except Exception as e:
            logging.exception("Failed to load %s: %s", f.name, e)
            continue

        cell = cell_code_from_name(f.name)
        mass = (masses_g or {}).get(cell)
        met = per_cycle_metrics(df, mass_g=mass)
        per_file_metrics[cell] = met

        sty = style_for_cell(cell, idx_hint=idx)

        # Discharge capacity (left axis)
        ax1.plot(
            met["Cycle"], met["Qdis"],
            label=sty["legend"],
            color=sty["color"],
            linestyle=sty["linestyle"],
            marker=sty["marker"],
            lw=sty["lw"],
            markevery=sty["markevery"] if sty["marker"] else None,
        )

        # Coulombic efficiency (right axis) — same color, dotted style for clarity
        ax2.plot(
            met["Cycle"], met["CE_pct"],
            color=sty["color"],
            linestyle=":",
            marker=None,
            lw=max(1.0, sty["lw"] * 0.9),
        )

        if sty["legend"] not in used_labels:
            used_labels.add(sty["legend"])

        logging.debug("  %s: cycles=%d, Qdis[min=%.3f,max=%.3f], CE[min=%.2f,max=%.2f]",
                      cell,
                      len(met),
                      float(np.nanmin(met['Qdis'])) if len(met) else float('nan'),
                      float(np.nanmax(met['Qdis'])) if len(met) else float('nan'),
                      float(np.nanmin(met['CE_pct'])) if len(met) else float('nan'),
                      float(np.nanmax(met['CE_pct'])) if len(met) else float('nan'))

    # Axes labels/limits and visual polish
    ax1.set_xlabel("Cycle Number")
    y1lab = "Discharge Capacity (mAh g$^{-1}$)" if (masses_g and len(masses_g) > 0) else "Discharge Capacity (mAh)"
    ax1.set_ylabel(y1lab)
    ax2.set_ylabel("Coulombic Efficiency (%)")
    ax2.set_ylim(0, 110)

    # Match your crisp style vibe
    ax1.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis='y', direction='in', right=True)
    ax1.grid(True, alpha=0.22)

    # Legend
    leg = ax1.legend(loc="best", fontsize="small", frameon=True)
    if leg is not None and hasattr(leg, "get_frame"):
        leg.get_frame().set_linewidth(1.2)

    title_units = " (mAh g$^{-1}$)" if (masses_g and len(masses_g) > 0) else " (mAh)"
    ax1.set_title(f"Discharge Capacity{title_units} and CE vs Cycle")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logging.info("Figure saved to: %s (dpi=%d)", save_path, dpi)

    plt.show()
    logging.info("Plotting complete.")

    return per_file_metrics


# ───────────────────────── CSV Export ─────────────────────────
def export_metrics_csv(per_file_metrics: Dict[str, pd.DataFrame], out_csv: Path) -> None:
    rows = []
    for cell, df in per_file_metrics.items():
        for _, r in df.iterrows():
            rows.append({
                "Cell": cell,
                "Cycle": int(r["Cycle"]),
                "Qdis": float(r["Qdis"]),
                "Qchg": float(r["Qchg"]),
                "CE_pct": float(r["CE_pct"]),
                "Alpha": alpha_bucket(cell),
                "Legend": f"{cell} - {electrolyte_lookup.get(cell, 'Unknown')}",
            })
    tab = pd.DataFrame(rows).sort_values(["Cell", "Cycle"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_csv, index=False)
    logging.info("Exported metrics CSV: %s  (rows=%d)", out_csv, len(tab))


# ───────────────────────── Main / CLI ─────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot capacity & CE vs cycle from Excel files recursively.")
    parser.add_argument("--root", type=str, default="", help="Root folder to search (omit to open a folder dialog)")
    parser.add_argument("--save", type=str, default="", help="Optional path to save the figure image (e.g., out.png)")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI when saving (default: 200)")
    parser.add_argument("--export-csv", type=str, default="", help="Optional path to export per-cycle metrics as CSV")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    # Optional: per-cell mass normalization table (CSV with columns: Cell,mass_g)
    parser.add_argument("--mass-csv", type=str, default="", help="CSV mapping Cell -> mass_g to normalize to mAh g^-1")
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Resolve root folder
    if args.root:
        root = Path(args.root)
        if not root.exists():
            logging.error("Root path does not exist: %s", root)
            return
    else:
        # Try a folder chooser if available
        try:
            import tkinter as tk
            from tkinter import filedialog
            tk.Tk().withdraw()
            chosen = filedialog.askdirectory(title="Select root folder containing Excel cycling files")
            if not chosen:
                logging.error("No folder selected; exiting.")
                return
            root = Path(chosen)
        except Exception:
            logging.error("tkinter not available; please provide --root PATH")
            return

    # Optional mass normalization
    masses_g: Dict[str, float] = {}
    if args.mass_csv:
        mpath = Path(args.mass_csv)
        if mpath.exists():
            try:
                mtab = pd.read_csv(mpath)
                # Expect columns: Cell, mass_g
                for _, r in mtab.iterrows():
                    c = str(r["Cell"]).strip().upper()
                    mg = float(r["mass_g"])
                    masses_g[c] = mg
                logging.info("Loaded mass table for normalization: %d entries", len(masses_g))
            except Exception as e:
                logging.exception("Failed to load mass CSV (%s): %s", mpath, e)
        else:
            logging.warning("Mass CSV not found: %s", mpath)

    files = find_excel_files(root)
    if not files:
        logging.error("No Excel files found under %s", root)
        return

    save_path = Path(args.save) if args.save else None
    per_file_metrics = plot_all(files, save_path=save_path, dpi=args.dpi, masses_g=masses_g if masses_g else None)

    if args.export_csv:
        export_metrics_csv(per_file_metrics, Path(args.export_csv))

    logging.info("Done.")


if __name__ == "__main__":
    main()
