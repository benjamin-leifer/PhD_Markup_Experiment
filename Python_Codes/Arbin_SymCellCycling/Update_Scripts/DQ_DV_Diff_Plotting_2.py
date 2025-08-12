#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
GCPL_dqdv_cycle_diff.py — dQ/dV comparison between two selected charge cycles
=============================================================================
What it does
- Scans a file or directory for Bio-Logic EC-Lab GCPL data files matching "*_03_GCPL_*.mpt".
- Extracts the CHARGE half-cycle for two user-selected cycles (e.g., 1 & 2, 1 & 3, etc.).
- Bins to a fixed 3 mV lattice, computes dQ/dV for each cycle, and plots both + the difference (Cj − Ci).
- Auto-detects whether 'cycle number' is 0-based or 1-based; falls back robustly.

How to use
1) Edit the CONFIG section below:
   - TARGET_PATH: path to a .mpt file or a directory.
   - CYCLES_TO_COMPARE: tuple of two integers, e.g., (1, 2).
   - SAVE_PNG: set True to export a PNG next to each input file.

2) Run:
   python GCPL_dqdv_cycle_diff.py
"""

# =========================
# CONFIG — EDIT THESE ONLY
# =========================
TARGET_PATH = r"C:\path\to\your\file_or_folder"  # file (.mpt) or directory
CYCLES_TO_COMPARE = (1, 2)                        # choose any two cycles to compare (1-based)
SAVE_PNG = True                                   # save a PNG next to each input file
BIN_W = 0.003                                     # 3 mV voltage bin width
ENC = "cp1252"                                    # EC-Lab default text encoding
CHUNK = 50_000                                    # CSV chunk size for large files

# Optional: mass normalization (g) keyed by short ID (last token in first filename field, e.g., "GN01")
MASS_G = {
    # "GN01": 0.0249,
}

# =========================
# IMPLEMENTATION (no edits)
# =========================


def cell_short_id(fname: str) -> str:
    first = fname.split("_")[0]  # e.g., BL-LL-GN01
    return first.split("-")[-1] if "-" in first else first

def find_header_and_cols(fp: Path) -> tuple[int, list[str]]:
    """Return (header_line_index, column_names_list)."""
    with fp.open("r", encoding=ENC, errors="ignore") as f:
        hdr = None
        for line in f:
            if line.lower().startswith("nb header lines"):
                hdr = int(line.split(":")[-1].strip()) - 1
                break
    if hdr is None:
        raise RuntimeError("EC-Lab header count not found")

    with fp.open("r", encoding=ENC, errors="ignore") as f:
        for i, l in enumerate(f):
            if i == hdr:
                return hdr, l.rstrip().split("\t")
    raise RuntimeError("Failed to read header row")

def _match(cols: list[str], pattern: str) -> str | None:
    return next((c for c in cols if re.search(pattern, c, re.I)), None)

def find_cols(cols: list[str], want_charge: bool):
    """
    Robustly locate key columns:
      - Voltage: prefer 'Ewe/V' (or 'Ecell/V')
      - Quantity: prefer explicit 'Q charge/mA.h' or 'Q discharge/mA.h' (fallback to mixed)
      - Cycle index: prefer 'cycle number' (fallback 'cycle index' or 'Ns')
      - Half-cycle: 'half cycle' if present (0=charge, 1=discharge)
    """
    # Voltage
    v = (_match(cols, r"^(Ewe|Ecell)\s*/\s*V") or
         _match(cols, r"\b(Ewe|Ecell)\b"))
    # Quantity (mAh)
    q_charge_explicit   = _match(cols, r"^Q\s*charge\s*/\s*mA\.?h")
    q_discharge_explicit= _match(cols, r"^Q\s*discharge\s*/\s*mA\.?h")
    q_mixed             = _match(cols, r"^Q\s*charge/discharge\s*/\s*mA\.?h")

    if want_charge:
        q = q_charge_explicit or _match(cols, r"^Q.*charge.*mA\.?h") or q_mixed
    else:
        q = q_discharge_explicit or _match(cols, r"^Q.*discharge.*mA\.?h") or q_mixed

    cyc = (_match(cols, r"^cycle\s*number$") or
           _match(cols, r"cycle\s*(index|number)") or
           _match(cols, r"^\bNs\b"))
    half = _match(cols, r"^half\s*cycle$") or _match(cols, r"half\s*cycle")

    if not v:
        raise KeyError("Voltage column not found (e.g., 'Ewe/V').")
    if not q:
        raise KeyError(f"Q column not found for want_charge={want_charge} (e.g., 'Q charge/mA.h').")
    return v, q, cyc, half

def detect_cycle_base(fp: Path, hdr: int, cols: list[str], cyc_col: str | None) -> bool | None:
    """
    Return True if cycle column appears 0-based (min==0), False if 1-based (min==1), or None if unknown.
    """
    if cyc_col is None:
        return None
    for ch in pd.read_csv(fp, sep="\t", header=None, names=cols,
                          skiprows=range(hdr+1), usecols=[cyc_col],
                          chunksize=CHUNK, engine="python", encoding=ENC):
        vals = pd.to_numeric(ch[cyc_col], errors="coerce").dropna()
        if len(vals):
            mn = int(np.floor(vals.min()))
            if mn == 0:
                return True   # 0-based
            if mn == 1:
                return False  # 1-based
    return None

def load_charge_halfcycle(fp: Path, cycle_number_1based: int) -> pd.DataFrame:
    """
    Return DataFrame with columns V (V) and QmAh (mAh) for the CHARGE half-cycle
    of the requested (1-based) cycle number. Auto-detects 0- vs 1-based cycles.
    """
    hdr, cols = find_header_and_cols(fp)
    v, q, cyc, half = find_cols(cols, want_charge=True)
    use = [v, q] + [c for c in (cyc, half) if c]

    is_zero_based = detect_cycle_base(fp, hdr, cols, cyc)

    pieces = []
    for ch in pd.read_csv(fp, sep="\t", header=None, names=cols,
                          skiprows=range(hdr+1), usecols=use,
                          chunksize=CHUNK, engine="python", encoding=ENC):
        chf = ch

        # Filter to desired cycle
        if cyc is not None:
            desired = ((cycle_number_1based - 1) if is_zero_based else cycle_number_1based
                       if is_zero_based is not None else cycle_number_1based)
            chf = chf[chf[cyc] == desired]

            # If empty and base is unknown, try the other base as fallback
            if chf.empty and is_zero_based is None:
                alt_desired = cycle_number_1based - 1
                chf = ch[ch[cyc] == alt_desired]

        # Keep only charge half-cycle when present
        if half is not None and half in chf:
            chf = chf[chf[half] == 0]  # 0 = charge, 1 = discharge

        if not chf.empty:
            pieces.append(chf[[v, q]])

    if not pieces:
        raise RuntimeError(
            f"No rows for cycle {cycle_number_1based} (charge) in {fp.name}."
        )

    df = pd.concat(pieces, ignore_index=True)
    df = df.rename(columns={v: "V", q: "QmAh"}).apply(pd.to_numeric, errors="coerce").dropna()
    return df.reset_index(drop=True)

def to_fixed_bins(df: pd.DataFrame, bin_w: float = BIN_W) -> pd.DataFrame:
    out = df.copy()
    out["_vbin"] = np.round(out["V"] / bin_w) * bin_w
    out = (out.groupby("_vbin", as_index=False)["QmAh"]
              .mean()
              .rename(columns={"_vbin": "V"})
              .sort_values("V", ignore_index=True))
    return out

def dqdv_from_binned(df: pd.DataFrame) -> pd.DataFrame:
    v = df["V"].to_numpy()
    q = (df["QmAh"] / 1000.0).to_numpy()  # mAh -> Ah for stability
    o = np.argsort(v)
    v, q = v[o], q[o]
    dv = np.diff(v)
    dq = np.diff(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        dqdv = np.divide(dq, dv, out=np.full_like(dq, np.nan), where=dv != 0)
    vmid = 0.5*(v[:-1] + v[1:])
    res = pd.DataFrame({"V": vmid, "dQ/dV": dqdv * 1000.0})  # back to mAh/V
    return res.dropna()

def compute_curves(fp: Path, cycle_i: int, cycle_j: int, mass_g: float|None = None):
    c_i_raw = load_charge_halfcycle(fp, cycle_i)
    c_j_raw = load_charge_halfcycle(fp, cycle_j)
    b_i = to_fixed_bins(c_i_raw, BIN_W)
    b_j = to_fixed_bins(c_j_raw, BIN_W)
    d_i = dqdv_from_binned(b_i).rename(columns={"dQ/dV": "dQdV"})
    d_j = dqdv_from_binned(b_j).rename(columns={"dQ/dV": "dQdV"})
    if mass_g and mass_g > 0:
        d_i["dQdV"] /= mass_g
        d_j["dQdV"] /= mass_g
    merged = pd.merge(d_i, d_j, on="V", how="inner", suffixes=(f"_c{cycle_i}", f"_c{cycle_j}"))
    diff = pd.DataFrame({
        "V": merged["V"],
        "dQdV": merged[f"dQdV_c{cycle_j}"] - merged[f"dQdV_c{cycle_i}"]
    })
    return d_i, d_j, diff

def plot_curves(fp: Path, cycle_i: int, cycle_j: int, save: bool=False):
    cid = cell_short_id(fp.name)
    mass = MASS_G.get(cid)
    d_i, d_j, diff = compute_curves(fp, cycle_i, cycle_j, mass_g=mass)

    plt.figure(figsize=(8,5))
    plt.plot(d_i["V"], d_i["dQdV"], lw=2, label=f"Charge (Cycle {cycle_i})")
    plt.plot(d_j["V"], d_j["dQdV"], lw=2, label=f"Charge (Cycle {cycle_j})")
    plt.plot(diff["V"], diff["dQdV"], lw=2, linestyle="--", label=f"Difference (C{cycle_j} − C{cycle_i})")
    ylab = "dQ/dV (mAh V$^{-1}$)" if not mass else "dQ/dV (mAh g$^{-1}$ V$^{-1}$)"
    plt.xlabel("Voltage (V)")
    plt.ylabel(ylab)
    plt.title(f"{cid} — dQ/dV: Charge Cycle {cycle_i} vs {cycle_j} and Difference")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save:
        out = fp.with_suffix("").as_posix() + f"_dqdv_C{cycle_i}_vs_C{cycle_j}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

def find_gcpl_targets(p: Path):
    if p.is_file() and "_03_GCPL_" in p.name and p.suffix.lower()==".mpt":
        return [p]
    return sorted(x for x in p.rglob("*_03_GCPL_*.mpt"))

def main():
    target = Path(
        r"C:\Users\benja\Downloads\DQ_DV Work\Lab Arbin_DQ_DV_2025_07_15\07\Dq_DV"
)
    files = find_gcpl_targets(target)
    if not files:
        print("No *_03_GCPL_*.mpt files found at:", target)
        sys.exit(2)

    c1, c2 = 1,1
    for fp in files:
        print(f"Processing {fp.name} for Charge Cycle {c1} vs {c2} dQ/dV plot...")
        try:
            plot_curves(fp, c1, c2, save=SAVE_PNG)
        except Exception as e:
            print(f"  ✗ {fp.name}: {e}")

if __name__ == "__main__":
    main()
