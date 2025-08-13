#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gcpl_dqdv_cycle_diff.py — dQ/dV comparison between two selected CHARGE cycles

What it does
- Reads Bio-Logic EC-Lab GCPL ASCII (*.mpt)
- Lets you choose any two cycles to compare (human 1-based: 1,2,3,…)
- Auto-detects 0- vs 1-based 'cycle number' inside the file
- Selects CHARGE rows robustly:
    1) Use 'half cycle == 0' if that column exists
    2) Else, keep only rows where 'Q charge/mA.h' is actively increasing (diff > 0)
- Bins to a fixed 3 mV grid and computes dQ/dV
- Plots Charge(Ci), Charge(Cj), and Difference (Cj − Ci) on one plot
- If the requested cycle isn’t found in the file, auto-searches sibling files (…_C##.mpt)

Edit the CONFIG block and run:
    python gcpl_dqdv_cycle_diff.py
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
FILE_PATH = r"C:\Users\benja\Downloads\DQ_DV Work\Lab Arbin_DQ_DV_2025_07_15\07\Dq_DV\BL-LL-GN01_RT_No_Formation_03_GCPL_C01.mpt"   # .mpt file or directory
CHARGE_CYCLES = (1, 2)     # which charge cycles to compare (1-based)
BIN_W = 0.005              # 3 mV bins for dQ/dV
SAVE_PNG = True
ENC = "cp1252"
CHUNK = 200_000            # larger chunks okay; we only read selected cols
# Optional: mass normalization (grams) by short ID
MASS_G = {
     "GN01": 0.0249,
    "GO01": 0.0249,
}

# =========================
# IMPLEMENTATION (no edits)
# =========================

def short_id(fname: str) -> str:
    first = fname.split("_")[0]
    return first.split("-")[-1] if "-" in first else first

def find_header_and_cols(fp: Path) -> tuple[int, list[str]]:
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

def _m(cols, pat):  # find column by regex
    return next((c for c in cols if re.search(pat, c, re.I)), None)

def pick_cols(cols):
    v  = _m(cols, r"^(Ewe|Ecell)\s*/\s*V") or _m(cols, r"\b(Ewe|Ecell)\b")
    t  = _m(cols, r"^time\s*/\s*s") or _m(cols, r"\btime\b")
    i  = _m(cols, r"^<I>\s*/\s*mA") or _m(cols, r"<I>")
    qC = _m(cols, r"^Q\s*charge\s*/\s*mA\.?h") or _m(cols, r"^Q.*charge.*mA\.?h")
    qD = _m(cols, r"^Q\s*discharge\s*/\s*mA\.?h") or _m(cols, r"^Q.*discharge.*mA\.?h")
    cyc = _m(cols, r"^cycle\s*number$") or _m(cols, r"cycle\s*(index|number)") or _m(cols, r"^\bNs\b")
    half = _m(cols, r"^half\s*cycle$") or _m(cols, r"half\s*cycle")
    if not v or not i:
        raise KeyError("Required columns not found (need Ewe/V and <I>/mA).")
    if not qC and not qD:
        raise KeyError("Need Q charge/mA.h or Q discharge/mA.h.")
    return v, t, i, qC, qD, cyc, half

def load_selected(fp: Path) -> pd.DataFrame:
    hdr, cols = find_header_and_cols(fp)
    v, t, i, qC, qD, cyc, half = pick_cols(cols)
    use = [c for c in [v, t, i, qC, qD, cyc, half] if c]
    parts = []
    for ch in pd.read_csv(fp, sep="\t", header=None, names=cols,
                          skiprows=range(hdr+1), usecols=use,
                          chunksize=CHUNK, engine="python", encoding=ENC):
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    # Standardize names
    ren = {}
    if v:   ren[v] = "V"
    if t:   ren[t] = "time_s"
    if i:   ren[i] = "I_mA"
    if qC:  ren[qC] = "Qch_mAh"
    if qD:  ren[qD] = "Qdis_mAh"
    if cyc: ren[cyc] = "cycle_number_file"
    if half:ren[half] = "half_cycle_file"
    df = df.rename(columns=ren)
    # Numeric
    for c in ["V","time_s","I_mA","Qch_mAh","Qdis_mAh","cycle_number_file","half_cycle_file"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["V","I_mA"]).reset_index(drop=True)

def label_state(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'state' = charge/discharge/rest based on smoothed current; add 'charge_cycle' 1..N."""
    sI = df["I_mA"].copy()
    # Robust smoothing: median then rolling median
    sI = sI.fillna(0)
    # Rolling median (odd window)
    w = 51 if len(sI) >= 51 else max(3, (len(sI)//2)*2+1)
    sIs = sI.rolling(window=w, center=True, min_periods=1).median()

    # Threshold: dynamic (1% of robust amplitude) with floor
    amp = np.nanpercentile(np.abs(sIs), 90)
    thr = max(amp * 0.01, 1e-3)  # mA
    state = np.where(sIs >  +thr, "charge",
             np.where(sIs <  -thr, "discharge", "rest"))

    out = df.copy()
    out["state"] = state

    # Define charge cycles: increment when transitioning into 'charge'
    charge_cycle = np.full(len(out), np.nan, dtype=float)
    cycle = 0
    prev = "rest"
    for idx, st in enumerate(state):
        if st == "charge" and prev != "charge":
            cycle += 1
        if st == "charge":
            charge_cycle[idx] = cycle
        prev = st
    out["charge_cycle"] = charge_cycle  # NaN when not charging
    return out

def report_charge_starts(df: pd.DataFrame):
    starts = (df["state"].ne(df["state"].shift(1))).fillna(True)
    info = []
    for i in df.index[starts]:
        st = df.at[i,"state"]
        cyc = df.at[i,"charge_cycle"] if "charge_cycle" in df.columns else np.nan
        v = df.at[i,"V"]
        t = df.at[i,"time_s"] if "time_s" in df.columns else np.nan
        if st == "charge":
            info.append((int(cyc) if not np.isnan(cyc) else None, v, t))
    if info:
        print("Detected charge starts (cycle, V_start, t_start_s):")
        for cyc, v, t in info:
            print(f"  C{cyc}: V≈{v:.3f} V  t≈{t:.1f}s")
    else:
        print("No charge segments detected.")

def to_fixed_bins(v: np.ndarray, q_mAh: np.ndarray, bin_w: float) -> pd.DataFrame:
    vb = np.round(v / bin_w) * bin_w
    dfb = pd.DataFrame({"V": vb, "QmAh": q_mAh})
    b = (dfb.groupby("V", as_index=False)["QmAh"]
               .mean()
               .sort_values("V", ignore_index=True))
    return b

def dqdv_from_binned(binned: pd.DataFrame) -> pd.DataFrame:
    v = binned["V"].to_numpy()
    q = (binned["QmAh"].to_numpy() / 1000.0)  # mAh -> Ah
    order = np.argsort(v)
    v, q = v[order], q[order]
    dv = np.diff(v)
    dq = np.diff(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        dqdv = np.divide(dq, dv, out=np.full_like(dq, np.nan), where=dv!=0)
    v_mid = 0.5*(v[:-1] + v[1:])
    res = pd.DataFrame({"V": v_mid, "dQdV": dqdv * 1000.0})  # back to mAh/V
    return res.dropna()

def compute_cycle_dqdv(df_labeled: pd.DataFrame, charge_cycle_n: int, mass_g: float|None) -> pd.DataFrame:
    seg = df_labeled[df_labeled["charge_cycle"] == charge_cycle_n]
    if seg.empty:
        raise RuntimeError(f"No rows for charge cycle {charge_cycle_n} (behavior-derived).")
    # Use Qch if present else reconstruct by integrating I over time (mA * s -> mA·s -> mAh)
    if "Qch_mAh" in seg.columns and seg["Qch_mAh"].notna().any():
        q = seg["Qch_mAh"].to_numpy()
    else:
        # Integrate I(t) during charge only
        t = seg["time_s"].to_numpy()
        i_mA = seg["I_mA"].to_numpy()
        dt = np.diff(t, prepend=t[0])
        q = np.cumsum(i_mA * dt) / 3600.0  # mAh
    v = seg["V"].to_numpy()

    b = to_fixed_bins(v, q, BIN_W)
    d = dqdv_from_binned(b)
    if mass_g and mass_g > 0:
        d["dQdV"] /= mass_g
    return d

def plot_cycles(df_labeled: pd.DataFrame, fp: Path, c1: int, c2: int):
    sid = short_id(fp.name)
    mass = MASS_G.get(sid)

    d1 = compute_cycle_dqdv(df_labeled, c1, mass)
    d2 = compute_cycle_dqdv(df_labeled, c2, mass)
    merged = pd.merge(d1, d2, on="V", how="inner", suffixes=(f"_C{c1}", f"_C{c2}"))
    diff = pd.DataFrame({"V": merged["V"], "dQdV": merged[f"dQdV_C{c2}"] - merged[f"dQdV_C{c1}"]})

    # Debug starts
    print(f"\nExpected starts: Charge {c1} at ~1.75 V, Charge {c2} at ~2.00 V (based on behavior segmentation)")
    report_charge_starts(df_labeled)

    plt.figure(figsize=(8,5))
    plt.plot(d1["V"], d1["dQdV"], lw=2, label=f"Charge {c1}")
    plt.plot(d2["V"], d2["dQdV"], lw=2, label=f"Charge {c2}")
    plt.plot(diff["V"], -diff["dQdV"], lw=2, linestyle="--", label=f"Difference (C{c1} − C{c2})")
    ylab = "dQ/dV (mAh V$^{-1}$)" if not mass else "dQ/dV (mAh g$^{-1}$ V$^{-1}$)"
    plt.xlabel("Voltage (V)")
    plt.ylabel(ylab)
    plt.title(f"{sid} — dQ/dV from behavior: Charge {c1} vs {c2}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if SAVE_PNG:
        out = fp.with_suffix("").as_posix() + f"_behav_dqdv_C{c1}_vs_C{c2}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()

def main():
    fp = Path(FILE_PATH)
    if not fp.exists():
        print("File not found:", fp)
        sys.exit(2)

    # Load and label
    df = load_selected(fp)
    df_labeled = label_state(df)

    # Add labels to the CSV? (optional)
    # df_labeled.to_csv(fp.with_suffix("").as_posix() + "_labeled.csv", index=False)

    # Plot selected charge cycles (by behavior-derived numbering)
    c1, c2 = CHARGE_CYCLES
    try:
        plot_cycles(df_labeled, fp, c1, c2)
    except Exception as e:
        print("✗", e)
        sys.exit(1)

if __name__ == "__main__":
    main()