#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gcpl_dqdv_cycle_diff.py — dQ/dV comparison between two selected CHARGE cycles
- Works with Bio-Logic EC-Lab GCPL ASCII (*.mpt)
- Bins to a fixed 3 mV grid
- Plots Charge(Ci), Charge(Cj), and Difference (Cj − Ci) on the same axes
"""
from __future__ import annotations
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# =========================
# CONFIG — EDIT THESE ONLY
# =========================
TARGET_PATH = r"C:\Users\benja\Downloads\DQ_DV Work\Lab Arbin_DQ_DV_2025_07_15\07\Dq_DV" # .mpt file or directory
CYCLE_I = 2                                      # first cycle to compare (1-based)
CYCLE_J = 2                                      # second cycle to compare (1-based)
SAVE_PNG = True                                  # save figure next to each file
BIN_W = 0.003                                    # 3 mV bin width
ENC = "cp1252"                                   # EC-Lab default
CHUNK = 50_000                                   # CSV chunk size

# Optional mass normalization (grams) by short ID (e.g., "GN01")
MASS_G = {
    # "GN01": 0.0249,
}

# =========================
# IMPLEMENTATION (no edits)
# =========================


def _short_id(fname: str) -> str:
    first = fname.split("_")[0]         # e.g., BL-LL-GN01
    return first.split("-")[-1] if "-" in first else first

def _find_header_and_cols(fp: Path) -> tuple[int, list[str]]:
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

def _m(cols: list[str], pattern: str) -> str | None:
    return next((c for c in cols if re.search(pattern, c, re.I)), None)

def _find_cols(cols: list[str], want_charge: bool):
    """
    Prefer explicit columns:
      Voltage: 'Ewe/V' (or 'Ecell/V')
      Quantity: 'Q charge/mA.h' or 'Q discharge/mA.h' (fallback to mixed)
      Cycle: 'cycle number' (or 'cycle index' or 'Ns')
      Half-cycle: 'half cycle' if present (0=charge, 1=discharge)
    """
    v = _m(cols, r"^(Ewe|Ecell)\s*/\s*V") or _m(cols, r"\b(Ewe|Ecell)\b")
    q_charge = _m(cols, r"^Q\s*charge\s*/\s*mA\.?h")
    q_discharge = _m(cols, r"^Q\s*discharge\s*/\s*mA\.?h")
    q_mixed = _m(cols, r"^Q\s*charge/discharge\s*/\s*mA\.?h")

    if want_charge:
        q = q_charge or _m(cols, r"^Q.*charge.*mA\.?h") or q_mixed
    else:
        q = q_discharge or _m(cols, r"^Q.*discharge.*mA\.?h") or q_mixed

    cyc = (_m(cols, r"^cycle\s*number$") or
           _m(cols, r"cycle\s*(index|number)") or
           _m(cols, r"^\bNs\b"))
    half = _m(cols, r"^half\s*cycle$") or _m(cols, r"half\s*cycle")

    if not v:
        raise KeyError("Voltage column not found (e.g., 'Ewe/V').")
    if not q:
        raise KeyError(f"Q column not found for want_charge={want_charge} (e.g., 'Q charge/mA.h').")
    return v, q, cyc, half

def _detect_cycle_base(fp: Path, hdr: int, cols: list[str], cyc_col: str | None) -> bool | None:
    """
    Return True if cycle appears 0-based (min==0), False if 1-based (min==1), None if unknown.
    """
    if cyc_col is None:
        return None
    use = [cyc_col]
    for ch in pd.read_csv(fp, sep="\t", header=None, names=cols,
                          skiprows=range(hdr+1), usecols=use,
                          chunksize=CHUNK, engine="python", encoding=ENC):
        vals = pd.to_numeric(ch[cyc_col], errors="coerce").dropna()
        if len(vals):
            mn = int(np.floor(vals.min()))
            if mn == 0:
                return True
            if mn == 1:
                return False
    return None

def _load_charge_halfcycle(fp: Path, cycle_number_1based: int) -> pd.DataFrame:
    """
    Return DataFrame with columns ['V','QmAh'] for the CHARGE half-cycle of the requested cycle.
    Auto-detects 0- vs 1-based 'cycle number'. Uses 'half cycle == 0' when present.
    """
    hdr, cols = _find_header_and_cols(fp)
    v, q, cyc, half = _find_cols(cols, want_charge=True)
    use = [v, q] + [c for c in (cyc, half) if c]

    zero_based = _detect_cycle_base(fp, hdr, cols, cyc)

    parts = []
    for ch in pd.read_csv(fp, sep="\t", header=None, names=cols,
                          skiprows=range(hdr+1), usecols=use,
                          chunksize=CHUNK, engine="python", encoding=ENC):
        sel = ch
        if cyc is not None:
            desired = (cycle_number_1based - 1) if zero_based else cycle_number_1based
            if zero_based is None:
                # try strict 1-based first
                sel = sel[sel[cyc] == cycle_number_1based]
                if sel.empty:
                    sel = ch[ch[cyc] == (cycle_number_1based - 1)]
            else:
                sel = sel[sel[cyc] == desired]

        if half is not None and half in sel:
            sel = sel[sel[half] == 0]  # 0=charge

        if not sel.empty:
            parts.append(sel[[v, q]])

    if not parts:
        raise RuntimeError(f"No rows for cycle {cycle_number_1based} (charge) in {fp.name}.")

    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns={v: "V", q: "QmAh"}).apply(pd.to_numeric, errors="coerce").dropna()
    return df.reset_index(drop=True)

def _to_fixed_bins(df: pd.DataFrame, bin_w: float = BIN_W) -> pd.DataFrame:
    out = df.copy()
    out["_vbin"] = np.round(out["V"].to_numpy().ravel() / bin_w) * bin_w
    out = (out.groupby("_vbin", as_index=False)["QmAh"]
              .mean()
              .rename(columns={"_vbin": "V"})
              .sort_values("V", ignore_index=True))
    return out

def _dqdv_from_binned(df: pd.DataFrame) -> pd.DataFrame:
    v = df["V"].to_numpy().ravel()
    q = (df["QmAh"].to_numpy().ravel() / 1000.0)  # mAh -> Ah
    ord_idx = np.argsort(v)
    v, q = v[ord_idx], q[ord_idx]
    dv = np.diff(v)
    dq = np.diff(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        dqdv = np.divide(dq, dv, out=np.full_like(dq, np.nan), where=dv != 0)
    vmid = 0.5 * (v[:-1] + v[1:])
    res = pd.DataFrame({"V": vmid, "dQdV": dqdv * 1000.0})  # back to mAh/V
    return res.dropna()

def compute_curves(fp: Path, cycle_i: int, cycle_j: int, mass_g: float | None = None):
    di_raw = _load_charge_halfcycle(fp, cycle_i)
    dj_raw = _load_charge_halfcycle(fp, cycle_j)
    bi = _to_fixed_bins(di_raw, BIN_W)
    bj = _to_fixed_bins(dj_raw, BIN_W)
    di = _dqdv_from_binned(bi)
    dj = _dqdv_from_binned(bj)
    if mass_g and mass_g > 0:
        di["dQdV"] /= mass_g
        dj["dQdV"] /= mass_g
    merged = pd.merge(di, dj, on="V", how="inner", suffixes=("_i", "_j"))
    diff = pd.DataFrame({"V": merged["V"].to_numpy().ravel(),
                         "dQdV": (merged["dQdV_j"].to_numpy().ravel()
                                  - merged["dQdV_i"].to_numpy().ravel())})
    return di, dj, diff

def plot_curves(fp: Path, cycle_i: int, cycle_j: int, save: bool = False):
    sid = _short_id(fp.name)
    mass = MASS_G.get(sid)
    di, dj, diff = compute_curves(fp, cycle_i, cycle_j, mass_g=mass)

    x_i = di["V"].to_numpy().ravel(); y_i = di["dQdV"].to_numpy().ravel()
    x_j = dj["V"].to_numpy().ravel(); y_j = dj["dQdV"].to_numpy().ravel()
    x_d = diff["V"].to_numpy().ravel(); y_d = diff["dQdV"].to_numpy().ravel()

    plt.figure(figsize=(8, 5))
    plt.plot(x_i, y_i, lw=2, label=f"Charge (Cycle {cycle_i})")
    plt.plot(x_j, y_j, lw=2, label=f"Charge (Cycle {cycle_j})")
    plt.plot(x_d, y_d, lw=2, linestyle="--", label=f"Difference (C{cycle_j} − C{cycle_i})")
    ylab = "dQ/dV (mAh V$^{-1}$)" if not mass else "dQ/dV (mAh g$^{-1}$ V$^{-1}$)"
    plt.xlabel("Voltage (V)")
    plt.ylabel(ylab)
    plt.title(f"{sid} — dQ/dV: Charge Cycle {cycle_i} vs {cycle_j} and Difference")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save:
        out = fp.with_suffix("").as_posix() + f"_dqdv_C{cycle_i}_vs_C{cycle_j}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

def _find_targets(p: Path):
    if p.is_file() and "_03_GCPL_" in p.name and p.suffix.lower() == ".mpt":
        return [p]
    return sorted(x for x in p.rglob("*_03_GCPL_*.mpt"))

def main():
    target = Path(TARGET_PATH)
    files = _find_targets(target)
    if not files:
        print("No *_03_GCPL_*.mpt files found at:", target)
        sys.exit(2)

    for fp in files:
        print(f"Processing {fp.name} for Charge Cycle {CYCLE_I} vs {CYCLE_J} dQ/dV plot...")
        try:
            plot_curves(fp, CYCLE_I, CYCLE_J, save=SAVE_PNG)
        except Exception as e:
            print(f"  ✗ {fp.name}: {e}")

import pandas as pd
import re
from pathlib import Path

ENCODING = "cp1252"  # Bio-Logic default
CHUNK_SIZE = 50_000  # for large files

def find_header_and_cols(fp: Path):
    """Locate header line count and column names in a Bio-Logic MPT file."""
    hdr_lines = None
    with fp.open("r", encoding=ENCODING, errors="ignore") as f:
        for line in f:
            if line.lower().startswith("nb header lines"):
                hdr_lines = int(line.split(":")[-1].strip()) - 1
                break
    if hdr_lines is None:
        raise RuntimeError("Could not find 'Nb header lines' in file header.")
    with fp.open("r", encoding=ENCODING, errors="ignore") as f:
        for i, line in enumerate(f):
            if i == hdr_lines:
                cols = line.strip().split("\t")
                return hdr_lines, cols
    raise RuntimeError("Could not read column names.")

def print_cycle_numbers(fp: Path):
    """Print unique cycle numbers and row counts from an MPT file."""
    hdr, cols = find_header_and_cols(fp)
    cyc_col = next((c for c in cols if re.search(r"cycle\s*number", c, re.I)), None)
    if not cyc_col:
        raise RuntimeError("No 'cycle number' column found.")

    print(f"Found cycle number column: '{cyc_col}'")
    all_cycles = []
    for chunk in pd.read_csv(fp, sep="\t", header=None, names=cols,
                             skiprows=range(hdr+1), usecols=[cyc_col],
                             chunksize=CHUNK_SIZE, engine="python", encoding=ENCODING):
        all_cycles.extend(chunk[cyc_col].dropna().astype(int).tolist())

    counts = pd.Series(all_cycles).value_counts().sort_index()
    print("\nCycle Number : Row Count")
    for cyc, count in counts.items():
        print(f"{cyc:<5} : {count}")


if __name__ == "__main__":
    mpt_path = Path(
        r"C:\Users\benja\Downloads\DQ_DV Work\Lab Arbin_DQ_DV_2025_07_15\07\Dq_DV\BL-LL-GN01_RT_No_Formation_03_GCPL_C01.mpt")
    print_cycle_numbers(mpt_path)
    main()
