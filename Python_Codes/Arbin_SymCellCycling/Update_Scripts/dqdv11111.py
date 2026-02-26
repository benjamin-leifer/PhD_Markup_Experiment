#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_dqdv_from_cell_index.py
----------------------------
Uses cell_file_index.xlsx to plot dQ/dV (cycle 1, charge) grouped by electrolyte.
Saves one figure per electrolyte to Downloads/Dq_Dv_scans/ as "<electrolyte>.png".
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ───────────────────────────── USER SETTINGS ─────────────────────────────
INDEX_XLSX = Path(r"C:\Users\benja\Downloads\cell_file_index.xlsx")

OUTPUT_DIR = Path(r"C:\Users\benja\Downloads\Dq_Dv_scans")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# dQ/dV settings
CYCLE_HUMAN = 1          # you asked for cycle 1
CHARGE = True
BIN_W = 0.003            # 3 mV grid

# Smoothing (optional)
DQDV_SMOOTH = False      # False = raw numerical derivative (like your script default)
WIN_PRE, POLY_PRE = 301, 3
WIN_POST, POLY_POST = 21, 2

# File parsing
CHUNK = 25_000
ENC = "cp1252"
DEBUG = False

V_MIN, V_MAX = 1.5, 3.5
Y_MIN, Y_MAX = -0.01, 0.10


# If you want ALL curves normalized by a single active mass (g), set to a float (e.g., 0.0159)
# Otherwise keep None (no mass normalization).
ACTIVE_MASS_G = 0.02496886674 / 1000  # g


# Which rows from the index to use (optional filters)
# e.g., only HiFi: INDEX_TAG_FILTER = "HiFi"
INDEX_TAG_FILTER: str | None = None
# ────────────────────────────────────────────────────────────────────────


def _dbg(msg: str):
    if DEBUG:
        print(msg)

def build_prefix_pattern(prefixes: set[str]) -> re.Pattern:
    """
    Match any allowed prefix ONLY when followed by optional separators and then a digit.
    e.g. AA01, BL-LL-AA01, AA_01, AA-01
    """

    esc = sorted((re.escape(p) for p in prefixes if p), key=len, reverse=True)
    if not esc:
        return re.compile(r"(?!x)x")

    # IMPORTANT: put '-' at the end of the character class to avoid range issues
    pattern = (
        r"(?<![A-Z0-9])"
        r"(" + "|".join(esc) + r")"
        r"(?=(?:[ _-]*\d))"
    )

    return re.compile(pattern, flags=re.IGNORECASE)

def sanitize_filename(s: str) -> str:
    s = str(s).strip()
    if not s:
        return "Unknown"
    # Windows-illegal filename chars: \ / : * ? " < > |
    return re.sub(r'[\\/:*?"<>|]+', "_", s)


def fixed_bin(df: pd.DataFrame, bin_w: float) -> pd.DataFrame:
    """Average capacity inside fixed-width voltage bins."""
    df = df.assign(_vbin=np.round(df["V"] / bin_w) * bin_w)
    out = (df.groupby("_vbin", as_index=False)["QmAh"]
             .mean()
             .rename(columns={"_vbin": "V"})
             .sort_values("V", ignore_index=True))
    return out


def raw_dqdv_mAh(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Raw dQ/dV using Q in mAh (=> units mAh/V)."""
    if df.empty:
        raise ValueError("empty trace")
    v = df["V"].to_numpy()
    q = df["QmAh"].to_numpy()  # mAh
    order = np.argsort(v)
    v, q = v[order], q[order]
    dq = np.diff(q)
    dv = np.diff(v)
    v_mid = 0.5 * (v[:-1] + v[1:])
    y = np.divide(dq, dv, out=np.full_like(dq, np.nan, dtype=float), where=dv != 0)
    return v_mid, y


def savgol_dqdv_mAh(df: pd.DataFrame,
                    w_pre=301, p_pre=3,
                    w_post=21, p_post=2) -> Tuple[np.ndarray, np.ndarray]:
    """Smoothed dQ/dV using Savitzky–Golay on Q then on dQ/dV; Q in mAh."""
    if df.empty:
        raise ValueError("empty trace")

    v = df["V"].to_numpy()
    q = df["QmAh"].to_numpy()  # mAh
    order = np.argsort(v)
    v, q = v[order], q[order]

    def _fit(win: int, poly: int, n: int) -> int:
        win = min(win if win % 2 else win - 1, n if n % 2 else n - 1)
        if win <= poly:
            win = poly + 2 + (poly % 2 == 0)  # next odd > poly
            win = min(win, n if n % 2 else n - 1)
        return win

    w_pre = _fit(w_pre, p_pre, len(q))
    w_post = _fit(w_post, p_post, len(q) - 1)

    q_sm = savgol_filter(q, w_pre, p_pre)
    dq, dv = np.diff(q_sm), np.diff(v)
    v_mid = 0.5 * (v[:-1] + v[1:])
    y = np.divide(dq, dv, out=np.full_like(dq, np.nan, dtype=float), where=dv != 0)
    y = savgol_filter(y, w_post, p_post)
    return v_mid, y


# ───────── EC-Lab (.mpt) loader (adapted from your existing script) ─────────
def eclab_header_row(fp: Path) -> Tuple[int, List[str]]:
    with open(fp, "r", encoding=ENC, errors="ignore") as f:
        for line in f:
            if line.lower().startswith("nb header lines"):
                hdr = int(line.split(":")[-1].strip()) - 1
                break
        else:
            raise RuntimeError("EC-Lab header count not found")
    with open(fp, "r", encoding=ENC, errors="ignore") as f:
        for i, l in enumerate(f):
            if i == hdr:
                return hdr, l.rstrip().split("\t")
    raise RuntimeError("EC-Lab header row not found")


def eclab_pick(cols: List[str], charge: bool):
    def m(pats): return next((c for c in cols if re.search(pats, c, re.I)), None)
    v = m(r"(ewe|ecell).*v")
    if charge:
        q = m(r"q.*charge.*m?a\.?h")
    else:
        q = m(r"q.*discharge.*m?a\.?h")
    cyc = m(r"cycle.*number|cycle.*index") or m(r"\bNs\b")
    half = m(r"half\s*cycle")
    if not (v and q):
        raise KeyError("EC-Lab voltage or capacity column missing")
    return v, q, cyc, half


def load_eclab(fp: Path, cycle0: int, charge: bool) -> pd.DataFrame:
    hdr, cols = eclab_header_row(fp)
    v, q, cyc, half = eclab_pick(cols, charge)
    sel = [v, q] + [c for c in (cyc, half) if c]

    dfs: List[pd.DataFrame] = []
    for ch in pd.read_csv(fp, sep="\t", header=None, names=cols,
                          skiprows=range(hdr + 1), usecols=sel,
                          chunksize=CHUNK, engine="python", encoding=ENC):
        if cyc:
            ch = ch[ch[cyc] == cycle0]
        if half:
            ch = ch[ch[half] == (0 if charge else 1)]
        dfs.append(ch[[v, q]])

    df = pd.concat(dfs, ignore_index=True).astype(float)
    df.columns = ["V", "QmAh"]  # EC-Lab is already mA.h
    return df


# ───────── Arbin (.xlsx/.xlsm/.xls) loader (adapted from your existing script) ─────────
_clean = lambda s: re.sub(r"[^a-z]", "", str(s).lower())

def load_arbin(fp: Path, cycle1: int, charge: bool) -> pd.DataFrame:
    df0 = pd.read_excel(fp, sheet_name=1, engine="openpyxl")
    cmap = {_clean(c): c for c in df0.columns}

    v_key = next((k for k in cmap if k.startswith("voltage")), None)
    q_key = next((k for k in cmap if k.startswith("chargecapacity" if charge else "dischargecapacity")), None)
    cyc_key = next((k for k in cmap if k.startswith(("cycleindex", "cyclenumber"))), None)
    half_key = next((k for k in cmap if k.startswith("halfcycle")), None)

    if not (v_key and q_key):
        raise KeyError(f"Required Arbin columns missing in {fp.name}")

    v_col = cmap[v_key]
    q_col = cmap[q_key]

    df = df0[[v_col, q_col]].copy()

    if cyc_key:
        df = df[df0[cmap[cyc_key]] == cycle1]
    if half_key:
        half = df0[cmap[half_key]]
        df = df[half == (0 if charge else 1)]

    df.columns = ["V", "Qraw"]
    df = df.dropna()

    # Convert Q to mAh robustly:
    # - If header suggests Ah (and not mAh), multiply by 1000
    header = str(q_col).lower()
    q = df["Qraw"].astype(float).to_numpy()

    if ("ah" in header) and ("mah" not in header):
        q = q * 1000.0  # Ah -> mAh
    else:
        # Heuristic fallback: if values look like Ah (<~10), scale to mAh
        if np.nanmax(q) < 10:
            q = q * 1000.0

    out = pd.DataFrame({"V": df["V"].astype(float).to_numpy(), "QmAh": q})
    return out.reset_index(drop=True)


def load_any_trace(fp: Path, cycle_human: int, charge: bool) -> pd.DataFrame:
    ext = fp.suffix.lower()
    if ext == ".mpt":
        # EC-Lab cycles commonly 0-based -> cycle1 becomes 0
        return load_eclab(fp, cycle_human - 1, charge)
    if ext in (".xls", ".xlsx", ".xlsm"):
        # Arbin cycles commonly 1-based
        return load_arbin(fp, cycle_human, charge)
    raise ValueError(f"Unsupported file type: {fp.name}")


def main():
    # Load the index
    df = pd.read_excel(INDEX_XLSX, engine="openpyxl")

    required_cols = {"electrolyte", "file_path"}
    missing = required_cols - set(map(str, df.columns))
    if missing:
        raise ValueError(f"cell_file_index.xlsx missing columns: {missing}. Found: {list(df.columns)}")

    # normalize column access
    # (your file likely has these, but keep it safe)
    if "cell_code" not in df.columns:
        df["cell_code"] = ""
    if "tags" not in df.columns:
        df["tags"] = ""
    if "modified_time" not in df.columns:
        df["modified_time"] = ""

    # Optional tag filter (e.g., only HiFi)
    if INDEX_TAG_FILTER:
        df = df[df["tags"].astype(str).str.contains(INDEX_TAG_FILTER, case=False, na=False)].copy()

    # De-dupe (keep newest by modified_time) within (electrolyte, cell_code, tags)
    df["modified_time_dt"] = pd.to_datetime(df["modified_time"], errors="coerce")
    df = df.sort_values(["modified_time_dt"], ascending=False, kind="stable")
    df = df.drop_duplicates(subset=["file_path"], keep="first")
    df = df.drop_duplicates(subset=["electrolyte", "cell_code", "tags"], keep="first")

    # Group by electrolyte
    df["electrolyte"] = df["electrolyte"].astype(str).fillna("").replace({"": "Unknown"})
    grouped = df.groupby("electrolyte", sort=True)

    print(f"Electrolytes found: {len(grouped)}")
    for electrolyte, g in grouped:
        electrolyte_name = str(electrolyte).strip() if str(electrolyte).strip() else "Unknown"
        safe_name = sanitize_filename(electrolyte_name)

        fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
        cmap = plt.get_cmap("tab10")

        plotted = 0
        for i, row in enumerate(g.itertuples(index=False)):
            fp = Path(getattr(row, "file_path"))
            if not fp.exists():
                print(f"  [skip missing] {fp}")
                continue

            label_cell = str(getattr(row, "cell_code", "")).strip()
            label_tags = str(getattr(row, "tags", "")).strip()
            label = label_cell if label_cell else fp.stem
            if label_tags:
                label = f"{label} | {label_tags}"

            try:
                df_raw = load_any_trace(fp, CYCLE_HUMAN, CHARGE)

                # keep only charge 1 between 1.5–3.5 V
                df_raw = df_raw[(df_raw["V"] >= V_MIN) & (df_raw["V"] <= V_MAX)].copy()
                if df_raw.empty or len(df_raw) < 5:
                    print(f"  [skip out of window/short] {fp.name}")
                    continue

                df_b = fixed_bin(df_raw, BIN_W)

                if len(df_b) < 5:
                    print(f"  [skip short] {fp.name}")
                    continue

                if DQDV_SMOOTH:
                    v_mid, y = savgol_dqdv_mAh(df_b, WIN_PRE, POLY_PRE, WIN_POST, POLY_POST)
                else:
                    v_mid, y = raw_dqdv_mAh(df_b)

                # optional normalization
                if ACTIVE_MASS_G:
                    y = y / ACTIVE_MASS_G  # mAh/V/g = mAh g^-1 V^-1

                ax.plot(
                    v_mid, y,
                    marker="o", mfc="none", ms=2.3, lw=1.25,
                    label=label,
                    color=cmap(plotted % 10)
                )
                plotted += 1
            except Exception as e:
                print(f"  [skip error] {fp.name}: {e}")
                continue

        if plotted == 0:
            plt.close(fig)
            print(f"  [no plots] {electrolyte_name}")
            continue

        # Cosmetics
        ax.set_xlabel("Voltage (V)")
        if ACTIVE_MASS_G:
            ax.set_ylabel(r"dQ/dV (mAh g$^{-1}$ V$^{-1}$)")
        else:
            ax.set_ylabel(r"dQ/dV (mAh V$^{-1}$)")

        # Title: electrolyte name
        ax.set_title(electrolyte_name)

        # Legend: compact
        ax.legend(fontsize="x-small", ncol=1, frameon=False)

        # Save
        out_path = OUTPUT_DIR / f"{safe_name}.png"
        ax.set_xlim(V_MIN, V_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"  saved: {out_path}  (n={plotted})")


if __name__ == "__main__":
    main()
