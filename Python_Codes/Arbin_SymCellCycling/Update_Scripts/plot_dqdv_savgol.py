#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_dqdv_savgol.py – EC-Lab (.mpt) + Arbin (.xlsx) dQ/dV on a 3 mV grid
========================================================================
* Streams each file in chunks (memory-safe)
* Auto-detects voltage & capacity columns for both back-ends
* Bins every trace on a common 3 mV voltage lattice
* Two Savitzky–Golay passes (wide on Q, narrow on dQ/dV) with auto-shrink
* Optional mass-normalisation to mAh g⁻¹ V⁻¹ via MASS_MG
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import re, warnings, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ────────────────────────────── USER SETTINGS ──────────────────────────────
DATA_DIR = Path(r"C:\Users\benja\Downloads\DQ_DV Work\Lab Arbin_DQ_DV_2025_07_15\07\Dq_DV")

FILES: List[str] = [
    "BL-LL-FZ01_RT_C_20_Charge_02_CP_C04.mpt",
    #"BL-LL-FY01_RT_C_20_Charge_02_CP_C04.mpt",
    #"BL-LL-FX01_RT_C_20_Charge_02_CP_C02.mpt",
    #"BL-LL-FW01_RT_C_20_Charge_02_CP_C01.mpt",
    "BL-LL-GA01_RT_C_20_Charge_02_CP_C02.mpt",
    "BL-LL-GA02_RT_C_20_Form_HighFid_Channel_64_Wb_1.xlsx",
    "BL-LL-FZ02_RT_C_20_Form_HighFid_Channel_63_Wb_1.xlsx",
    #"BL-LL-FY02_RT_C_20_Form_HighFid_Channel_62_Wb_1.xlsx",
    "BL-LL-FW02_RT_C_20_Form_HighFid_Channel_60_Wb_1.xlsx",
    "BL-LL-FX02_RT_C_20_Form_HighFid_Channel_61_Wb_1.xlsx",
]

# Active-material mass **mg**
_mass_each_mg = 0.02496886674/1000
MASS_MG = {id_: _mass_each_mg for id_ in
           ["FZ01","FY01","FX01","FW01","GA01",
            "FZ02","FY02","FX02","FW02","GA02"]}

# Script parameters
CYCLE    = 1          # 0-based
CHARGE   = True
BIN_W    = 0.003      # 3 mV bin
WIN_PRE, POLY_PRE   = 301, 3   # SG on Q
WIN_POST, POLY_POST = 21,  2   # SG on dQ/dV
CHUNK    = 25_000     # streamed rows
ENC      = "cp1252"   # EC-Lab encoding
DEBUG    = True
# Toggle smoothing in dQ/dV calculation
DQDV_SMOOTH = False   # set False to plot raw (unsmoothed) numerical derivative

# ────────────────────────────────────────────────────────────────────────────


# ───────── common helpers ─────────
def _dbg(msg: str):
    if DEBUG:
        print(" ", msg)

def fixed_bin(df: pd.DataFrame, bin_w: float) -> pd.DataFrame:
    """Average capacity inside fixed-width voltage bins."""
    df = df.assign(_vbin=np.round(df["V"] / bin_w) * bin_w)
    out = (df.groupby("_vbin", as_index=False)["QmAh"]
             .mean()
             .rename(columns={"_vbin": "V"})
             .sort_values("V", ignore_index=True))
    return out


def savgol_dqdv(df: pd.DataFrame,
                w_pre=301, p_pre=3,
                w_post=21, p_post=2):
    """Return (V_mid, dQ/dV) with auto-shrink windows."""
    if df.empty:
        raise ValueError("empty trace fed to savgol_dqdv")

    v = df["V"].to_numpy()
    q = (df["QmAh"] / 1000.0).to_numpy()   # mAh → Ah
    order = np.argsort(v);  v, q = v[order], q[order]

    # --- make sure windows are valid ---
    def _fit(win: int, poly: int, n: int) -> int:
        win = min(win if win % 2 else win - 1, n if n % 2 else n - 1)
        if win <= poly:
            win = poly + 2 + (poly % 2 == 0)  # next odd > poly
            win = min(win, n if n % 2 else n - 1)
        return win

    w_pre  = _fit(w_pre,  p_pre,  len(q))
    w_post = _fit(w_post, p_post, len(q) - 1)

    _dbg(f"    SG windows → pre:{w_pre} post:{w_post}")

    q_sm = savgol_filter(q, w_pre, p_pre)
    dq, dv = np.diff(q_sm), np.diff(v)
    v_mid  = 0.5 * (v[:-1] + v[1:])
    y = np.divide(dq, dv, out=np.full_like(dq, np.nan), where=dv != 0)
    y = savgol_filter(y, w_post, p_post)
    return v_mid, y

def raw_dqdv(df: pd.DataFrame):
    """
    Minimal dQ/dV: straight diff of binned capacity (mAh→Ah) vs V.
    No smoothing; NaNs where dv == 0.
    """
    if df.empty:
        raise ValueError("empty trace fed to raw_dqdv")

    v = df["V"].to_numpy()
    q = (df["QmAh"] / 1000.0).to_numpy()   # mAh → Ah
    order = np.argsort(v); v, q = v[order], q[order]

    dq = np.diff(q)
    dv = np.diff(v)
    v_mid = 0.5 * (v[:-1] + v[1:])
    y = np.divide(dq, dv, out=np.full_like(dq, np.nan), where=dv != 0)
    return v_mid, y


# ───────── EC-Lab loader ─────────
def eclab_header_row(fp: Path) -> tuple[int, list[str]]:
    with open(fp, "r", encoding=ENC, errors="ignore") as f:
        for line in f:
            if line.lower().startswith("nb header lines"):
                hdr = int(line.split(':')[-1].strip()) - 1        # ← no regex, always works
                break
        else:
            raise RuntimeError("EC-Lab header count not found")
    with open(fp, "r", encoding=ENC, errors="ignore") as f:
        for i, l in enumerate(f):
            if i == hdr:
                return hdr, l.rstrip().split("\t")


def eclab_pick(cols: list[str], charge: bool):
    def m(pats): return next((c for c in cols if re.search(pats, c, re.I)), None)

    v = m(r"(ewe|ecell).*v")  # matches <Ewe/V>, <Ecell/V>, etc.
    if charge:
        q = m(r"q.*charge.*m?a\.?h")  # Q charge/mA.h or charge/discharge
    else:
        q = m(r"q.*discharge.*m?a\.?h")  # Q discharge/mA.h
    cyc = m(r"cycle.*number|cycle.*index") or m(r"\bNs\b")
    half = m(r"half\\s*cycle")
    if not (v and q):
        raise KeyError("EC-Lab voltage or capacity column missing")
    return v, q, cyc, half


def load_eclab(fp: Path, cycle: int, charge: bool) -> pd.DataFrame:
    hdr, cols = eclab_header_row(fp)
    if DEBUG:
        print("  EC-Lab headers:", cols[:15])  # ← add
    v, q, cyc, half = eclab_pick(cols, charge)
    sel = [v, q] + [c for c in (cyc, half) if c]

    dfs: List[pd.DataFrame] = []
    for ch in pd.read_csv(fp, sep="\t", header=None, names=cols,
                          skiprows=range(hdr + 1), usecols=sel,
                          chunksize=CHUNK, engine="python", encoding=ENC):

        if DEBUG and not dfs:  # show only once
            print("  › first 5 rows raw")
            print(ch.head())

        if cyc:
            ch = ch[ch[cyc] == cycle]
        if half:
            ch = ch[ch[half] == (0 if charge else 1)]
        dfs.append(ch[[v, q]])
    df = pd.concat(dfs, ignore_index=True).astype(float)
    df.columns = ["V", "QmAh"]
    _dbg(f"  EC-Lab rows : {len(df)}")
    return df


# ───────── Arbin loader ─────────
_clean = lambda s: re.sub(r"[^a-z]", "", s.lower())  # strip non-letters

def load_arbin(fp: Path, cycle: int, charge: bool) -> pd.DataFrame:
    df0 = pd.read_excel(fp, sheet_name=1, engine="openpyxl")
    if DEBUG:
        print("  Headers:", list(df0.columns)[:15])
    # sheet 2 (0-based 1)
    cmap = {_clean(c): c for c in df0.columns}


    v_key  = next((k for k in cmap if k.startswith("voltage")), None)
    q_key  = next((k for k in cmap if k.startswith(
             "chargecapacity" if charge else "dischargecapacity")), None)
    cyc_key  = next((k for k in cmap if k.startswith(("cycleindex","cyclenumber"))), None)
    half_key = next((k for k in cmap if k.startswith("halfcycle")), None)

    if not (v_key and q_key):
        raise KeyError(f"Required columns missing in {fp.name}\\nHeaders: {list(df0.columns)[:10]}")

    df = df0[[cmap[v_key], cmap[q_key]]].copy()
    if cyc_key:
        df = df[df0[cmap[cyc_key]] == cycle]
    if half_key:
        df["half"] = df0[cmap[half_key]]
        df = df[df["half"] == (0 if charge else 1)]

    # keep only voltage & capacity for the derivative
    # keep only voltage & capacity (other helper cols already filtered)
    df_out = df[[cmap[v_key], cmap[q_key]]].copy()
    df_out.columns = ["V", "QmAh"]
    df_out = df_out.dropna()
    _dbg(f"  Arbin rows  : {len(df_out)}")
    return df_out.astype(float).reset_index(drop=True)

# def make_charge_curve(df: pd.DataFrame, cell_mass_mg: float | None = None):
#     """Return V-axis (ascending) and Q (mAh g⁻¹ or mAh) for a charge curve."""
#     # sort by V for smooth trace
#     df = df.sort_values("V", ignore_index=True)
#     q = df["QmAh"].to_numpy()
#     if cell_mass_mg:
#         q = 1000000*q / (cell_mass_mg )          # → mAh g⁻¹
#     return df["V"].to_numpy(), q

def make_charge_curve(df: pd.DataFrame, cell_mass_mg: float | None = None):
    """Return V-axis (ascending) and Q (mAh g⁻¹ or mAh) for a charge curve."""
    # Sort by V for smooth trace
    df = df.sort_values("V", ignore_index=True)
    q = df["QmAh"].to_numpy()

    # Debug raw values
    print("Raw Q values:", q[:10])

    if cell_mass_mg:
        q = 1000000 * q / cell_mass_mg  # → mAh g⁻¹
        # Debug scaled values
        print("Scaled Q values:", q[:10])

    return df["V"].to_numpy(), q

# ───────── main ─────────
# def main():
#     # side-by-side: left = dQ/dV, right = charge curve
#     fig, (ax_dqdv, ax_charge) = plt.subplots(
#         1, 2, figsize=(10, 4), sharex=False, constrained_layout=True
#     )
#     cmap = plt.get_cmap("tab10")
#
#     for idx, fname in enumerate(FILES):
#         fp = DATA_DIR / fname
#         stem = Path(fname).stem
#         m = re.search(r"LL-([A-Za-z0-9]{4})", stem, re.I)  # e.g. LL-FZ01
#         cell_id = m.group(1).upper() if m else stem  # fallback → full stem
#
#         ext = fp.suffix.lower()
#         print(f"Processing {fname}")
#         try:
#             if ext == ".mpt":
#                 # Biologic cycles are 0-based → Option 2 adjustment
#                 df_raw = load_eclab(fp, CYCLE - 1, CHARGE)
#             elif ext in (".xls", ".xlsx"):
#                 # Arbin cycles are 1-based; use CYCLE unchanged
#                 df_raw = load_arbin(fp, CYCLE, CHARGE)
#             else:
#                 print("  ✗ Unknown file type → skipped");
#                 continue
#         except Exception as e:
#             print("  ✗", e);
#             continue
#
#         # common 3 mV lattice
#         df_binned = fixed_bin(df_raw, BIN_W)
#         _dbg(f"  → after binning: {len(df_binned)} rows")
#         if len(df_binned) <= POLY_PRE + 2:
#             print("  ✗ Trace too short after binning → skipped");
#             continue
#
#         # dQ/dV (uses binned data)
#         if DQDV_SMOOTH:
#             v_mid, y = savgol_dqdv(df_binned, WIN_PRE, POLY_PRE, WIN_POST, POLY_POST)
#         else:
#             v_mid, y = raw_dqdv(df_binned)
#
#         if cell_id in MASS_MG:
#             y /= (MASS_MG[cell_id] * 1000.0)  # mg → g
#         ax_dqdv.plot(
#             v_mid, y,
#             marker="o", mfc="none", ms=2.3, lw=1.25,
#             label=cell_id, color=cmap(idx % 10)
#         )
#
#         # charge curve (capacity vs V) — use *binned* so it lines up w/ dQ/dV
#         v_curve = df_binned["V"].to_numpy()
#         q_curve = df_binned["QmAh"].to_numpy()
#         if cell_id in MASS_MG:
#             q_curve = q_curve / (MASS_MG[cell_id] * 1000)  # → mAh g^-1
#         ax_charge.plot(
#             q_curve, v_curve,
#             lw=1.25, label=cell_id, color=cmap(idx % 10)
#         )
#
#     # axis cosmetics
#     if MASS_MG:
#         ax_dqdv.set_ylabel("dQ/dV (mAh g$^{-1}$ V$^{-1}$)")
#         ax_charge.set_ylabel("Voltage (V)")
#     else:
#         ax_dqdv.set_ylabel("dQ/dV (Ah V$^{-1}$)")
#         ax_charge.set_ylabel("Capacity (mAh)")
#
#     for ax in (ax_dqdv, ax_charge):
#         ax.set_xlabel("Voltage (V)")
#         ax.set_xlabel("Capacity (mAh g$^{-1}$)")
#         #ax.set_xlim(2.4, 3.6)
#         #ax.grid(False, alpha=0.3)
#     ax_charge.set_xlabel("Capacity (mAh g$^{-1}$)")
#     smooth_tag = "smoothed" if DQDV_SMOOTH else "raw"
#     ax_dqdv.set_title(f"dQ/dV ({smooth_tag}) – {'charge' if CHARGE else 'discharge'} C{CYCLE}")
#
#     ax_charge.set_title(f"Charge curves – C{CYCLE}")
#
#     ax_dqdv.legend(ncol=2, fontsize="x-small")
#     ax_charge.legend(ncol=2, fontsize="x-small")
#
#     plt.show()
# Update the main function to scale Arbin files
def main():
    # side-by-side: left = dQ/dV, right = charge curve
    fig, (ax_dqdv, ax_charge) = plt.subplots(
        1, 2, figsize=(10, 4), sharex=False, constrained_layout=True
    )
    cmap = plt.get_cmap("tab10")

    for idx, fname in enumerate(FILES):
        fp = DATA_DIR / fname
        stem = Path(fname).stem
        m = re.search(r"LL-([A-Za-z0-9]{4})", stem, re.I)  # e.g. LL-FZ01
        cell_id = m.group(1).upper() if m else stem  # fallback → full stem

        ext = fp.suffix.lower()
        print(f"Processing {fname}")
        try:
            if ext == ".mpt":
                # Biologic cycles are 0-based → Option 2 adjustment
                df_raw = load_eclab(fp, CYCLE - 1, CHARGE)
            elif ext in (".xls", ".xlsx"):
                # Arbin cycles are 1-based; use CYCLE unchanged
                df_raw = load_arbin(fp, CYCLE, CHARGE)
                # Scale Arbin files by 1000
                df_raw["QmAh"] *= 1000
            else:
                print("  ✗ Unknown file type → skipped")
                continue
        except Exception as e:
            print("  ✗", e)
            continue

        # common 3 mV lattice
        df_binned = fixed_bin(df_raw, BIN_W)
        _dbg(f"  → after binning: {len(df_binned)} rows")
        if len(df_binned) <= POLY_PRE + 2:
            print("  ✗ Trace too short after binning → skipped")
            continue

        # dQ/dV (uses binned data)
        if DQDV_SMOOTH:
            v_mid, y = savgol_dqdv(df_binned, WIN_PRE, POLY_PRE, WIN_POST, POLY_POST)
        else:
            v_mid, y = raw_dqdv(df_binned)

        if cell_id in MASS_MG:
            y /= (MASS_MG[cell_id] * 1000.0)  # mg → g
        ax_dqdv.plot(
            v_mid, y,
            marker="o", mfc="none", ms=2.3, lw=1.25,
            label=cell_id, color=cmap(idx % 10)
        )

        # charge curve (capacity vs V) — use *binned* so it lines up w/ dQ/dV
        v_curve = df_binned["V"].to_numpy()
        q_curve = df_binned["QmAh"].to_numpy()
        if cell_id in MASS_MG:
            q_curve = q_curve / (MASS_MG[cell_id] * 1000)  # → mAh g^-1
        ax_charge.plot(
            q_curve, v_curve,
            lw=1.25, label=cell_id, color=cmap(idx % 10)
        )

    # axis cosmetics
    if MASS_MG:
        ax_dqdv.set_ylabel("dQ/dV (mAh g$^{-1}$ V$^{-1}$)")
        ax_charge.set_ylabel("Voltage (V)")
    else:
        ax_dqdv.set_ylabel("dQ/dV (Ah V$^{-1}$)")
        ax_charge.set_ylabel("Capacity (mAh)")

    for ax in (ax_dqdv, ax_charge):
        ax.set_xlabel("Voltage (V)")
        ax.set_xlabel("Capacity (mAh g$^{-1}$)")
    ax_charge.set_xlabel("Capacity (mAh g$^{-1}$)")
    ax_dqdv.set_xlabel("Voltage (V)")
    smooth_tag = "smoothed" if DQDV_SMOOTH else "raw"
    ax_dqdv.set_title(f"dQ/dV ({smooth_tag}) – {'charge' if CHARGE else 'discharge'} C{CYCLE}")

    ax_charge.set_title(f"Charge curves – C{CYCLE}")

    ax_dqdv.legend(ncol=2, fontsize="x-small")
    ax_charge.legend(ncol=2, fontsize="x-small")

    plt.show()


if __name__ == "__main__":
    main()
