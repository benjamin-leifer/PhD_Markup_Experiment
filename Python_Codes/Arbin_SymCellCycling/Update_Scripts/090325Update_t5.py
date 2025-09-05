#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Parallel + cached streaming importer for Arbin Excel → CSV, then plot:
  • Crawls --root recursively for .xlsx/.xls
  • Per file: parse Sheet 1, compute per-cycle metrics
  • Uses caching to skip unchanged files
  • Uses resume/skip so previously-imported (Cell,Dataset,Cycle) aren’t duplicated
  • Writes per-file temp CSV parts in parallel, then appends once to --out-csv
  • Plots best cell per alpha (highest max Qdis), excluding the LAST cycle
  • Legend includes cell code and electrolyte (by alpha)
  • Colors tied to electrolyte family; MF91 is red; markers separate close variants

CSV schema (canonical 8 cols):
  Cell, Alpha, Dataset, Cycle, Qdis, Qchg, CE_pct, SourceFile
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter


# ========= Styling / Lookups =========

# Fixed electrolyte colors (Okabe–Ito palette) with MF91 forced red
ELECTROLYTE_COLOR: Dict[str, str] = {
    "DTFV1411": "#0072B2",  # blue
    "DTFV1422": "#E69F00",  # orange
    "DTFV1452": "#009E73",  # green
    "DTFV1425": "#D55E00",  # vermillion
    "DTV1410":  "#CC79A7",  # reddish purple
    "DTV142":   "#56B4E9",  # sky blue
    "DTF1410":  "#F0E442",  # yellow
    "DTF142":   "#000000",  # black
    "MF91":     "#FF0000",  # red
}

# Markers to separate close variants (same color family)
ELECTROLYTE_MARKER: Dict[str, str] = {
    "DTFV1411": "o",
    "DTFV1422": "s",
    "DTFV1452": "D",
    "DTFV1452-new": "^",  # variant distinguished by marker
    "DTFV1425": "v",
    "DTV1410":  "P",
    "DTV142":   "X",
    "DTF1410":  "h",
    "DTF142":   "*",
    "MF91":     "o",
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
    "GW": "DTFV1411",
    "GX": "DTFV1422",
    "GV": "DTV1410",
    "GU": "DTV142",
    "GT": "DTF1410",
    "GS": "DTF142",
    "GY": "DT14",
    "GJ": "DTFV1452",
    "GK": "DTFV1425",
    "FR": "DTFV1422",
    "FS": "MF91",
    "GC": "DTFV1422",
    "GD": "DTFV1452",

    # extend as needed...
}
# Optional: manual overrides { "GN": "GN03", "AS": "AS05", ... }
CELL_OVERRIDES: Dict[str, str] = {}

CSV_COLS = ["Cell", "Alpha", "Dataset", "Cycle", "Qdis", "Qchg", "CE_pct", "SourceFile"]


# ========= Logging =========

def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")


# ========= Helpers: names, parsing =========

def alpha_bucket(cell_code: str) -> str:
    m = re.match(r"([A-Za-z]{2})\d{0,2}", cell_code)
    return m.group(1).upper() if m else cell_code[:2].upper()

def cell_code_from_name(fname: str) -> str:
    m = re.search(r"[A-Za-z]{2}\d{2}", fname)
    if m:
        return m.group(0).upper()
    return Path(fname).stem[-4:].upper()

def dataset_id_from_name(fname: str) -> str:
    stem = Path(fname).stem
    # Strip trailing _Wb_<num>
    stem = re.sub(r"(?i)_wb_\d+$", "", stem)
    # Reduce to a concise dataset id (before any extra suffixes like _Wb)
    return stem

def _clean(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def find_excel_files(root: Path) -> List[Path]:
    exts = (".xlsx", ".xls")
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    logging.info("Discovered %d Excel files under %s", len(files), root)
    return files


# ========= Data ingestion =========

def load_arbin_sheet1(path: Path) -> pd.DataFrame:
    engine = "openpyxl" if path.suffix.lower() == ".xlsx" else None
    df0 = pd.read_excel(path, sheet_name=1, engine=engine)

    cmap = {_clean(c): c for c in df0.columns}
    cyc_key = next((cmap[k] for k in cmap if k.startswith("cycleindex") or k.startswith("cyclenumber")), None)
    cur_key = next((cmap[k] for k in cmap if k.startswith("currenta") or k.startswith("current")), None)
    chg_key = next((cmap[k] for k in cmap if k.startswith("chargecapacityah")), None)
    dch_key = next((cmap[k] for k in cmap if k.startswith("dischargecapacityah")), None)
    if None in (cyc_key, cur_key, chg_key, dch_key):
        raise KeyError(f"Missing required columns in {path.name}")

    df = df0[[cyc_key, cur_key, chg_key, dch_key]].copy()
    df.columns = ["Cycle", "Current_A", "Qchg_Ah", "Qdis_Ah"]
    return df

def per_cycle_metrics(df: pd.DataFrame, mass_g: Optional[float] = None) -> pd.DataFrame:
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
    return pd.DataFrame(out).sort_values("Cycle").reset_index(drop=True)


# ========= CSV migration / resume index =========

def migrate_csv_if_needed(out_csv: Path) -> None:
    if not out_csv.exists():
        return
    try:
        df = pd.read_csv(out_csv, engine="python", on_bad_lines="skip")
    except Exception as e:
        logging.exception("Failed to read %s for migration: %s", out_csv, e)
        return

    for col in CSV_COLS:
        if col not in df.columns:
            if col == "Dataset":
                df[col] = "*"
            elif col == "SourceFile":
                df[col] = ""
            else:
                df[col] = np.nan

    df = df[CSV_COLS]
    df["Cell"] = df["Cell"].astype(str).str.upper()
    df["Alpha"] = df["Alpha"].astype(str).str.upper()
    df["Dataset"] = df["Dataset"].astype(str)
    df["Cycle"] = pd.to_numeric(df["Cycle"], errors="coerce").astype("Int64")
    for k in ["Qdis", "Qchg", "CE_pct"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")

    tmp = out_csv.with_suffix(".migrating.tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(out_csv)
    logging.info("Migrated CSV to unified schema: %s", out_csv)

def build_import_index(out_csv: Path) -> pd.DataFrame:
    if not out_csv.exists():
        return pd.DataFrame(columns=["Cell", "Dataset", "Cycle"])
    df = pd.read_csv(out_csv, usecols=["Cell", "Dataset", "Cycle"])
    df["Cell"] = df["Cell"].astype(str).str.upper()
    df["Dataset"] = df["Dataset"].astype(str)
    df["Cycle"] = pd.to_numeric(df["Cycle"], errors="coerce").astype("Int64")
    return df.dropna(subset=["Cell", "Dataset", "Cycle"]).reset_index(drop=True)

def import_sets_and_max(import_index: pd.DataFrame) -> Tuple[Dict[Tuple[str,str], Set[int]], Dict[Tuple[str,str], int]]:
    """Return dicts: {(Cell,Dataset): set(cycles)}, {(Cell,Dataset): max_cycle}"""
    cycsets: Dict[Tuple[str,str], Set[int]] = {}
    maxes: Dict[Tuple[str,str], int] = {}
    if import_index.empty:
        return cycsets, maxes
    for (cell, ds), g in import_index.groupby(["Cell", "Dataset"]):
        s = set(int(c) for c in g["Cycle"].dropna().astype(int).tolist())
        cycsets[(cell, ds)] = s
        maxes[(cell, ds)] = max(s) if s else -1
    return cycsets, maxes


# ========= Cache =========

@dataclass
class FileCacheInfo:
    mtime: float
    max_cycle: int

def cache_path_for(out_csv: Path) -> Path:
    return out_csv.with_suffix(".cache.json")

def load_cache(out_csv: Path) -> Dict:
    p = cache_path_for(out_csv)
    if not p.exists():
        return {"files": {}, "datasets": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"files": {}, "datasets": {}}

def save_cache(out_csv: Path, cache: Dict) -> None:
    p = cache_path_for(out_csv)
    p.write_text(json.dumps(cache, indent=2), encoding="utf-8")


# ========= Parallel worker =========

def worker_process_file(args) -> Tuple[str, str, str, int, Optional[Dict]]:
    """
    Returns: (cell, dataset, sourcefile, appended_rows, file_cache_update or None)

    Skips reading if:
      - file mtime unchanged AND file_max_cycle <= imported_max_cycle for (cell,dataset)
    """
    (path_str, cell, dataset, imported_cycles, imported_max_cycle, out_part_dir) = args
    path = Path(path_str)
    try:
        mtime = path.stat().st_mtime
    except Exception:
        return cell, dataset, path_str, 0, None

    # Load cached info from a sibling json if present (optional pre-pass done in parent)
    file_cache_update = {"mtime": mtime, "max_cycle": -1}

    # If we previously parsed this file and it hasn't changed, and our cached max_cycle
    # is not beyond what's already imported for (cell,dataset), we can skip.
    # (Parent provides cached file info via imported_max_cycle logic)
    # We still need to know file_max_cycle; without reading we rely on cache:
    # Parent already compared cached file_max_cycle to imported_max_cycle to decide to queue this file.
    # If we are here, either cache was missing/stale OR parent chose to process anyway.

    # Read Excel + compute metrics
    t0 = perf_counter()
    df = load_arbin_sheet1(path)
    t1 = perf_counter()
    met = per_cycle_metrics(df)
    t2 = perf_counter()

    # Determine file's max cycle
    file_max_cycle = int(met["Cycle"].max()) if not met.empty else -1
    file_cache_update["max_cycle"] = file_max_cycle

    # If nothing new beyond imported max, also skip quickly
    if imported_max_cycle is not None and file_max_cycle <= imported_max_cycle:
        return cell, dataset, str(path), 0, file_cache_update

    # Filter out already-imported cycles for this (cell,dataset)
    if imported_cycles:
        keep_mask = ~met["Cycle"].astype(int).isin(imported_cycles)
        met_new = met.loc[keep_mask].copy()
    else:
        met_new = met

    if met_new.empty:
        return cell, dataset, str(path), 0, file_cache_update

    # Write a per-file temp part CSV
    part_name = f"{cell}__{dataset}__{abs(hash(path)) & 0xffffffff}.part.csv"
    out_part = out_part_dir / part_name

    # Build output columns
    alpha = alpha_bucket(cell)
    out_df = met_new.copy()
    out_df["Cell"] = cell
    out_df["Alpha"] = alpha
    out_df["Dataset"] = dataset
    out_df["SourceFile"] = str(path)
    out_df = out_df[CSV_COLS]

    out_df.to_csv(out_part, index=False)

    logging.debug("Worker %s: read %.1f ms, metrics %.1f ms, wrote %d cycles -> %s",
                  path.name, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, len(out_df), out_part.name)
    return cell, dataset, str(path), len(out_df), file_cache_update


# ========= Plotting: Best cell per alpha (exclude last cycle) =========

def color_and_marker_for_alpha(alpha: str) -> Tuple[str, str, str]:
    elec = electrolyte_lookup.get(alpha, "Unknown")
    color = ELECTROLYTE_COLOR.get(elec, "#7F7F7F")
    marker = ELECTROLYTE_MARKER.get(elec, "o")
    return elec, color, marker

def plot_best_cells_from_csv(csv_path: Path, save_path: Optional[Path] = None, dpi: int = 200) -> None:
    try:
        tab = pd.read_csv(csv_path)
    except Exception as e:
        logging.warning("Standard read_csv failed (%s). Retrying with engine='python' and on_bad_lines='skip'.", e)
        tab = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    if tab.empty:
        logging.error("CSV is empty; nothing to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(8.8, 5.4))
    ax2 = ax1.twinx()

    for alpha, g in tab.groupby("Alpha"):
        # Choose override or best
        if alpha in CELL_OVERRIDES:
            best_cell = CELL_OVERRIDES[alpha]
            logging.info("Override: using %s for alpha %s", best_cell, alpha)
        else:
            best_cell = (g.groupby("Cell")["Qdis"].max().sort_values(ascending=False).index[0])

        g_best = g[g["Cell"] == best_cell].sort_values("Cycle")
        if len(g_best) > 1:
            g_best = g_best.iloc[:-1]  # exclude last cycle

        electrolyte, color, marker = color_and_marker_for_alpha(alpha)
        legend_label = f"{best_cell} ({electrolyte})"

        ax1.plot(g_best["Cycle"], g_best["Qdis"], color=color, lw=2.2, marker=marker, label=legend_label)
        ax2.plot(g_best["Cycle"], g_best["CE_pct"], color=color, lw=1.5, linestyle=":")

    ax1.set_xlabel("Cycle Number")
    ax1.set_ylabel("Discharge Capacity (mAh)")
    ax2.set_ylabel("Coulombic Efficiency (%)")
    ax2.set_ylim(0, 110)
    ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis="y", direction="in", right=True)
    ax1.grid(True, alpha=0.22)
    ax1.legend(loc="best", fontsize="small", frameon=True)
    ax1.set_title("Best Cell per Alpha: Capacity & CE vs Cycle (Last Cycle Excluded)")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logging.info("Saved figure to %s (dpi=%d)", save_path, dpi)
    plt.show()

def plot_all_cells_per_alpha_from_csv(
    csv_path: Path,
    target_alphas: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
    dpi: int = 200,
    exclude_last_cycle: bool = True,
) -> None:
    """
    For each requested alpha (e.g., ["AS","GN"]), produce its own figure with
    ALL cells of that alpha plotted together. Color is determined by the alpha's
    electrolyte (same convention as the rest of the script), and linestyles
    rotate per cell for readability.

    Capacity (left y) and Coulombic Efficiency (right y).
    """
    # robust CSV read (handles mixed old/new rows if any linger)
    try:
        tab = pd.read_csv(csv_path)
    except Exception:
        tab = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    if tab.empty:
        logging.error("CSV is empty; nothing to plot.")
        return

    # figure out which alphas to plot
    all_alphas = sorted(a for a in tab["Alpha"].dropna().astype(str).str.upper().unique())
    if target_alphas is None or len(target_alphas) == 0:
        alphas = all_alphas
    else:
        alphas = [a.strip().upper() for a in target_alphas if a.strip()]
        missing = [a for a in alphas if a not in all_alphas]
        if missing:
            logging.warning("Requested alphas not found in CSV: %s", ", ".join(missing))
            alphas = [a for a in alphas if a in all_alphas]
        if not alphas:
            logging.error("No matching alphas to plot.")
            return

    # distinct linestyles per cell while keeping the color fixed per electrolyte
    LS = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    for alpha in alphas:
        g_alpha = tab[tab["Alpha"].astype(str).str.upper() == alpha].copy()
        if g_alpha.empty:
            continue

        electrolyte = electrolyte_lookup.get(alpha, "Unknown")
        color = ELECTROLYTE_COLOR.get(electrolyte, "#7F7F7F")  # same palette; MF91 stays red
        marker = ELECTROLYTE_MARKER.get(electrolyte, "o")

        # one figure per alpha
        fig, ax1 = plt.subplots(figsize=(9.0, 5.2))
        ax2 = ax1.twinx()

        # each cell gets its own linestyle
        cells = sorted(g_alpha["Cell"].dropna().astype(str).unique())
        for i, cell in enumerate(cells):
            dfc = g_alpha[g_alpha["Cell"] == cell].sort_values("Cycle")
            if exclude_last_cycle and len(dfc) > 1:
                dfc = dfc.iloc[:-1]

            # Capacity
            ax1.plot(
                dfc["Cycle"],
                dfc["Qdis"],
                color=color,
                lw=2.0,
                linestyle=LS[i % len(LS)],
                marker=None,
                label=cell,
            )
            # CE
            ax2.plot(
                dfc["Cycle"],
                dfc["CE_pct"],
                color=color,
                lw=1.2,
                linestyle=LS[i % len(LS)],
            )

        # labels & styling
        ax1.set_xlabel("Cycle Number")
        ax1.set_ylabel("Discharge Capacity (mAh)")
        ax2.set_ylabel("Coulombic Efficiency (%)")
        ax2.set_ylim(0, 110)
        ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        ax2.tick_params(axis="y", direction="in", right=True)
        ax1.grid(True, alpha=0.22)

        # legend lists cells; title shows alpha + electrolyte
        ax1.legend(loc="best", fontsize="small", frameon=True, title="Cells")
        ax1.set_title(f"Alpha {alpha}: Capacity & CE vs Cycle — {electrolyte}")

        plt.tight_layout()

        # optional save per alpha
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            out_png = save_dir / f"{alpha}_all_cells.png"
            plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
            logging.info("Saved %s", out_png)

        plt.show()

def plot_electrolytes_subplots_from_csv(
    csv_path: Path,
    target_electrolytes: Optional[List[str]] = None,  # None => all present
    exclude_alphas: Optional[List[str]] = None,       # e.g. ["EU","ZZ"]
    exclude_cells: Optional[List[str]] = None,        # e.g. ["EU01","GN03"]
    save_path: Optional[Path] = None,
    dpi: int = 200,
    exclude_last_cycle: bool = True,
    min_capacity_mAh: float = 1.0,                    # drop cycles below this capacity
) -> None:
    """
    Parent figure with one subplot per electrolyte. Each subplot contains every
    cell (replicate) mapped to that electrolyte (via alpha → electrolyte lookup).
    Color = electrolyte color (fixed), marker = varies by cell code.

    Also excludes cycles where Qdis < min_capacity_mAh.
    """
    # robust read
    try:
        tab = pd.read_csv(csv_path)
    except Exception:
        tab = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    if tab.empty:
        logging.error("CSV is empty; nothing to plot.")
        return

    # Normalize
    tab["Alpha"] = tab["Alpha"].astype(str).str.upper()
    tab["Cell"]  = tab["Cell"].astype(str).str.upper()

    # Exclude by alpha
    if exclude_alphas:
        excl = {a.strip().upper() for a in exclude_alphas if a.strip()}
        before = len(tab)
        tab = tab[~tab["Alpha"].isin(excl)].copy()
        logging.info("Excluded alphas %s (%d → %d rows)", sorted(excl), before, len(tab))

    # Exclude by full cell code
    if exclude_cells:
        exclc = {c.strip().upper() for c in exclude_cells if c.strip()}
        before = len(tab)
        tab = tab[~tab["Cell"].isin(exclc)].copy()
        logging.info("Excluded cells %s (%d → %d rows)", sorted(exclc), before, len(tab))

    if tab.empty:
        logging.error("No rows left after exclusion; nothing to plot.")
        return

    # Map each row to an electrolyte via alpha
    tab["Electrolyte"] = tab["Alpha"].map(lambda a: electrolyte_lookup.get(a, "Unknown"))

    # If target_electrolytes specified, filter
    if target_electrolytes:
        keep = {e.strip() for e in target_electrolytes if e.strip()}
        before = len(tab)
        tab = tab[tab["Electrolyte"].isin(keep)].copy()
        logging.info("Filtered electrolytes %s (%d → %d rows)", sorted(keep), before, len(tab))
        if tab.empty:
            logging.error("No rows match requested electrolytes.")
            return

    # Determine electrolytes to plot (stable order)
    electrolytes = sorted(tab["Electrolyte"].unique())
    if not electrolytes:
        logging.error("No electrolytes to plot.")
        return

    # Build per-electrolyte cell roster
    per_elec_cells: Dict[str, List[str]] = {
        e: sorted(tab.loc[tab["Electrolyte"] == e, "Cell"].unique())
        for e in electrolytes
    }

    # Marker palette (rotates across cells in each subplot)
    MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h", "*", "<", ">", "8", "p"]

    # Grid layout (up to 3 cols)
    import math
    n = len(electrolytes)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(6.6 * ncols, 4.8 * nrows))
    fig.suptitle("Capacity & CE vs Cycle — All Replicates per Electrolyte", y=0.995)

    for idx, elec in enumerate(electrolytes, start=1):
        ax1 = fig.add_subplot(nrows, ncols, idx)
        ax2 = ax1.twinx()

        color = ELECTROLYTE_COLOR.get(elec, "#7F7F7F")  # MF91 stays red via your map
        cells = per_elec_cells[elec]
        marker_map = {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(cells)}

        g_e = tab[tab["Electrolyte"] == elec]

        for cell in cells:
            dfc = g_e[g_e["Cell"] == cell].sort_values("Cycle")

            # Exclude low-capacity cycles
            if "Qdis" in dfc.columns:
                dfc = dfc[dfc["Qdis"] >= float(min_capacity_mAh)]

            # Exclude the last cycle
            if exclude_last_cycle and len(dfc) > 1:
                dfc = dfc.iloc[:-1]

            if dfc.empty:
                continue

            # Capacity (left axis)
            ax1.plot(
                dfc["Cycle"], dfc["Qdis"],
                color=color, marker=marker_map[cell], lw=1.9, ms=4.5, label=cell,
            )
            # CE (right axis)
            ax2.plot(
                dfc["Cycle"], dfc["CE_pct"],
                color=color, linestyle=":", lw=1.2,
            )

        ax1.set_title(f"{elec}")
        ax1.set_xlabel("Cycle Number")
        ax1.set_ylabel("Discharge Capacity (mAh)")
        ax2.set_ylabel("Coulombic Efficiency (%)")
        ax2.set_ylim(0, 110)
        ax1.grid(True, alpha=0.22)

        # Legend beneath each subplot
        ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            fontsize="small",
            frameon=True,
            ncol=3,
            title="Cells"
        )

        ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        ax2.tick_params(axis="y", direction="in", right=True)
        ax1.set_ylim(bottom=0, top=5)
        ax2.set_ylim(bottom=0, top=110)
        #ax1.set_xlim(left=0, right=dfc["Cycle"].max() + 1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logging.info("Saved multi-subplot figure to %s", save_path)
    plt.show()



# ========= Main =========

def main():
    ap = argparse.ArgumentParser(description="Parallel + cached Excel importer → CSV → plot best per alpha.")
    ap.add_argument("--root", type=str, default=r"C:\Users\benja\Downloads\Temp\C_10 Cycling",
                    help="Root folder to scan recursively for Excel files.")
    ap.add_argument("--out-csv", type=str, default=r"C:\Users\benja\Downloads\Temp\C_10 Cycling\metrics.csv",
                    help="Path to per-cycle metrics CSV (appended).")
    ap.add_argument("--save", type=str, default=r"C:\Users\benja\Downloads\Temp\C_10 Cycling\plot.png",
                    help="Optional path to save the plotted figure.")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                    help="Number of parallel worker processes.")

    # Plotting controls
    ap.add_argument("--per-electrolyte-plot", action="store_true",
                    help="Produce the multi-subplot electrolytes figure (one subplot per electrolyte).")
    ap.add_argument("--electrolytes", type=str, default="",
                    help='Comma-separated electrolyte codes to include (e.g., "DTFV1452,MF91"). Blank = all.')
    ap.add_argument("--exclude-alphas", type=str, default="EU",
                    help='Comma-separated alpha prefixes to exclude (e.g., "EU,ZZ").')
    ap.add_argument("--exclude-cells", type=str, default="FS07, FS06, FS05, FS04, FS03, FR07, FR06, FR05, FR04, FR03,",
                    help='Comma-separated full cell codes to exclude (e.g., "EU01,GN03").')
    ap.add_argument("--min-capacity", type=float, default=1.0,
                    help="Drop cycles with Qdis below this capacity (mAh). Default: 1.0")

    # New: skip ingestion and only plot
    ap.add_argument("--csv-done", action="store_true", default=True,
                    help="Skip ingestion; assume out-csv already exists and only run plotting.")

    # Force defaults in PyCharm; switch to ap.parse_args() if you want CLI args
    args = ap.parse_args([])

    setup_logging(args.log_level)

    # Common paths / parsed list args
    root = Path(args.root)
    out_csv = Path(args.out_csv)
    parts_dir = out_csv.parent / (out_csv.stem + "_parts")
    save_path = Path(args.save) if args.save else None

    electrolytes = [s.strip() for s in args.electrolytes.split(",") if s.strip()] if args.electrolytes else None
    exclude_alphas = [s.strip() for s in args.exclude_alphas.split(",") if s.strip()] if args.exclude_alphas else None
    exclude_cells = [s.strip() for s in args.exclude_cells.split(",") if s.strip()] if args.exclude_cells else None

    # === Fast path: CSV already built, just plot ===
    if args.csv_done:
        if not out_csv.exists():
            logging.error("CSV does not exist at %s; cannot plot with --csv-done.", out_csv)
            return

        # Plot choice
        if args.per_electrolyte_plot:
            plot_electrolytes_subplots_from_csv(
                out_csv,
                target_electrolytes=electrolytes,
                exclude_alphas=exclude_alphas,
                exclude_cells=exclude_cells,
                save_path=save_path.with_name("electrolytes_subplots.png") if save_path else None,
                dpi=args.dpi,
                exclude_last_cycle=True,
                min_capacity_mAh=args.min_capacity,
            )
        else:
            plot_electrolytes_subplots_from_csv(
                out_csv,
                target_electrolytes=electrolytes,
                exclude_alphas=exclude_alphas,
                exclude_cells=exclude_cells,
                save_path=save_path.with_name("electrolytes_subplots.png") if save_path else None,
                dpi=args.dpi,
                exclude_last_cycle=True,
                min_capacity_mAh=args.min_capacity,
            )
            plot_best_cells_from_csv(out_csv, save_path=save_path, dpi=args.dpi)
        return  # ✅ skip ingestion entirely

    # === Normal path: build/update CSV then plot ===
    parts_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        logging.error("Root path does not exist: %s", root)
        return

    # Normalize any existing CSV
    migrate_csv_if_needed(out_csv)

    # Build resume index and per-(cell,dataset) skip sets
    import_index = build_import_index(out_csv)
    imported_sets, imported_max = import_sets_and_max(import_index)

    # Load cache and seed dataset max from imported
    cache = load_cache(out_csv)
    cache.setdefault("files", {})
    cache.setdefault("datasets", {})
    for (cell, ds), m in imported_max.items():
        cache["datasets"][f"{cell}|{ds}"] = {"max_cycle_imported": int(m)}

    # Discover files
    files = find_excel_files(root)
    if not files:
        logging.error("No Excel files found.")
        # Still attempt plotting if CSV exists
        if out_csv.exists():
            if args.per_electrolyte_plot:
                plot_electrolytes_subplots_from_csv(
                    out_csv,
                    target_electrolytes=electrolytes,
                    exclude_alphas=exclude_alphas,
                    exclude_cells=exclude_cells,
                    save_path=save_path.with_name("electrolytes_subplots.png") if save_path else None,
                    dpi=args.dpi,
                    exclude_last_cycle=True,
                    min_capacity_mAh=args.min_capacity,
                )
            else:
                plot_best_cells_from_csv(out_csv, save_path=save_path, dpi=args.dpi)
        return

    # Build task list with caching decisions
    tasks = []
    skipped_due_to_cache = 0
    for f in files:
        cell = cell_code_from_name(f.name)
        ds = dataset_id_from_name(f.name)
        mtime = f.stat().st_mtime
        imported_cycles = imported_sets.get((cell, ds), set())
        imported_max_cycle = imported_max.get((cell, ds), -1)

        finfo = cache["files"].get(str(f))
        # If file unchanged and its cached max_cycle <= imported_max_cycle, skip
        if finfo and abs(finfo.get("mtime", 0.0) - mtime) < 1e-6 and finfo.get("max_cycle", -1) <= imported_max_cycle:
            skipped_due_to_cache += 1
            continue

        # Otherwise queue for processing
        tasks.append((str(f), cell, ds, imported_cycles, imported_max_cycle, parts_dir))

    logging.info("Queued %d files (skipped %d via cache)", len(tasks), skipped_due_to_cache)

    # Parallel ingest
    t_all0 = perf_counter()
    appended_total = 0
    updates_for_cache: Dict[str, Dict] = {}

    if tasks:
        from concurrent.futures import as_completed
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = [ex.submit(worker_process_file, t) for t in tasks]
            total = len(futs)
            for i, fut in enumerate(as_completed(futs), 1):
                cell, ds, sourcefile, appended, file_cache_update = fut.result()
                logging.info("[%d/%d] Finished %s: %d new cycles", i, total, Path(sourcefile).name, appended)
                appended_total += appended

                # Update in-memory resume structures for subsequent tasks in SAME run
                if appended > 0:
                    imported_sets.setdefault((cell, ds), set())
                    imported_max[(cell, ds)] = max(
                        imported_max.get((cell, ds), -1),
                        cache["datasets"].get(f"{cell}|{ds}", {}).get("max_cycle_imported", -1)
                    )

                if file_cache_update is not None:
                    updates_for_cache[sourcefile] = file_cache_update

    t_all1 = perf_counter()
    logging.info("Workers finished in %.2f s; total appended cycles (in parts): %d",
                 (t_all1 - t_all0), appended_total)

    # Append all parts into the main CSV once (much faster than per-row appends)
    part_files = sorted(parts_dir.glob("*.part.csv"))
    if part_files:
        header_needed = not out_csv.exists()
        with out_csv.open("a", newline="", encoding="utf-8") as outfh:
            for pf in part_files:
                with pf.open("r", encoding="utf-8") as infh:
                    if header_needed:
                        outfh.write(infh.read())
                        header_needed = False
                    else:
                        # skip first header line
                        first = True
                        for line in infh:
                            if first:
                                first = False
                                continue
                            outfh.write(line)
        # clean up parts
        for pf in part_files:
            try:
                pf.unlink()
            except Exception:
                pass

        logging.info("Appended %d part files into %s", len(part_files), out_csv)

        # Update dataset max imported in cache from the CSV (cheap groupby)
        import_index = build_import_index(out_csv)
        _, imported_max = import_sets_and_max(import_index)
        for (cell, ds), m in imported_max.items():
            cache["datasets"][f"{cell}|{ds}"] = {"max_cycle_imported": int(m)}

        # Apply file cache updates
        for src, info in updates_for_cache.items():
            cache["files"][src] = {"mtime": float(info["mtime"]), "max_cycle": int(info["max_cycle"])}

        save_cache(out_csv, cache)
    else:
        logging.info("No part files produced; nothing to append to CSV.")

    # Plot (post-ingestion)
    if args.per_electrolyte_plot:
        plot_electrolytes_subplots_from_csv(
            out_csv,
            target_electrolytes=electrolytes,
            exclude_alphas=exclude_alphas,
            exclude_cells=exclude_cells,
            save_path=save_path.with_name("electrolytes_subplots.png") if save_path else None,
            dpi=args.dpi,
            exclude_last_cycle=True,
            min_capacity_mAh=args.min_capacity,
        )
    else:
        plot_best_cells_from_csv(out_csv, save_path=save_path, dpi=args.dpi)

    logging.info("Done.")



if __name__ == "__main__":
    main()
