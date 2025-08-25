#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_dqdv_plus_pcga.py  –  dQ/dV  +  charge curves  +  PCGA step capacities
===========================================================================

 • Keeps the full functionality of plot_dqdv_savgol.py.
 • Adds a scan for *_03_PCGA_*.mpt staircase files and overlays their
   step-resolved capacities as scatter points on the dQ/dV axis
   (secondary Y-axis).

Author: ChatGPT integration for Benjamin – 2025-07-25
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import warnings, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ───────────────────────── USER CONFIG SECTION ──────────────────────────
DATA_DIR = Path(r"C:\Users\benja\Downloads\DQ_DV Work\Lab Arbin_DQ_DV_2025_07_15\07\Dq_DV")  # master folder

FILES: List[str] = [
    # ---------- your dQ/dV & charge-curve input files ----------
    #"BL-LL-FZ01_RT_C_20_Charge_02_CP_C04.mpt",
    #"BL-LL-GA01_RT_C_20_Charge_02_CP_C02.mpt",
    #"BL-LL-GN01_RT_No_Formation_03_GCPL_C01.mpt",
    #"BL-LL-GX05_RT_No_Formation_02_CP_C04.mpt",
    "BL-LL-GW05_RT_No_Formation_02_CP_C03.mpt",#DTFV1411
    "BL-LL-GV05_RT_No_Formation_02_CP_C01.mpt",#DTV1410
    #"BL-LL-GU05_RT_No_Formation_02_CP_C04.mpt",#DTV142
    "BL-LL-GT05_RT_No_Formation_02_CP_C03.mpt",#DTF1410
    #"BL-LL-GS05_RT_No_Formation_02_CP_C01.mpt",#DTF142
    #"BL-LL-GA02_RT_C_20_Form_HighFid_Channel_64_Wb_1.xlsx",
    #"BL-LL-FZ02_RT_C_20_Form_HighFid_Channel_63_Wb_1.xlsx",
    #"BL-LL-FW02_RT_C_20_Form_HighFid_Channel_60_Wb_1.xlsx",
    #"BL-LL-FX02_RT_C_20_Form_HighFid_Channel_61_Wb_1.xlsx",
]

# active-material masses (mg) if you want normalised units
MASS_MG = {id_: 0.02496886674 / 1000  # mg → g
           for id_ in ["FZ01","FY01","FX01","FW01","GA01",
                        "FZ02","FY02","FX02","FW02","GA02", "GN01",
                       "GX05","GW05","GV05","GU05","GT05","GS05"]}
MASS_G = {
    #"GD01": 0.02496886674,   # example masses
    #"GC01": 0.02496886674,
    #"GC02": 0.02496886674*10000,
    #"GD02": 0.02496886674*10000,
    #"GJ06": 0.02496886674*10000,
    #"GK06": 0.02496886674*10000,
    #"GL01": 0.02496886674*10000,
    #"GM01": 0.02496886674*10000,

}

electrolyte_lookup = {
    "GC02": "DTFV1422 - PITT",
    "GD02": "DTFV1452 - Old - PITT",
    "GJ06": "DTFV1452 - New - PITT",
    "GK06": "DTFV1425 - PITT",
    "GL01": "DTFV1411 - PITT",
    "GM01": "MF91 - PITT",
    "GS05": "DTF142 - C/20",
    "GT05": "DTF1410 - C/20",
    "GW05": "DTFV1411 - C/20",
    "GV05": "DTV1410 - C/20",
    "GU05": "DTV142 - C/20",
    "GA01": "DTFV1411 - C/20",
    "GN01": "DTFV1411 - C/20",
    "FZ01": "DTFV1411 - C/20",
}

# Analysis parameters (unchanged from original script)
CYCLE = 1                     # 0-based for Bio-Logic, 1-based for Arbin
CHARGE = True
BIN_W = 0.003                 # 3 mV lattice
WIN_PRE, POLY_PRE = 301, 3
WIN_POST, POLY_POST = 21, 2
CHUNK = 25_000
ENC = "cp1252"
DQDV_SMOOTH = False           # True → Savitzky–Golay, False → raw diff
DEBUG = False                 # print parser diagnostics
# ─────────────────────────────────────────────────────────────────────────

# ═════════════════════════ helper functions (dQ/dV) ═════════════════════
def _dbg(msg: str):  # lightweight debug print
    if DEBUG:
        print(" ", msg)

def fixed_bin(df: pd.DataFrame, bin_w: float) -> pd.DataFrame:
    df = df.assign(_vbin=np.round(df["V"] / bin_w) * bin_w)
    out = (df.groupby("_vbin", as_index=False)["QmAh"]
             .mean()
             .rename(columns={"_vbin": "V"})
             .sort_values("V", ignore_index=True))
    return out

def savgol_dqdv(df: pd.DataFrame,
                w_pre=301, p_pre=3,
                w_post=21, p_post=2):
    v, q = df["V"].to_numpy(), df["QmAh"].to_numpy() / 1000  # mAh → Ah
    order = np.argsort(v); v, q = v[order], q[order]
    # ensure window validity
    fit = lambda w,p,n: max(min(w if w%2 else w-1, n-(n%2==0)), p+2+(p%2==0))
    w_pre, w_post = fit(w_pre, p_pre, len(q)), fit(w_post, p_post, len(q)-1)
    q_sm = savgol_filter(q, w_pre, p_pre)
    dq, dv = np.diff(q_sm), np.diff(v)
    v_mid = 0.5*(v[:-1] + v[1:])
    y = np.divide(dq, dv, out=np.full_like(dq, np.nan), where=dv!=0)
    y = savgol_filter(y, w_post, p_post)
    return v_mid, y

raw_dqdv = lambda df: savgol_dqdv(df, 1, 0, 1, 0)  # degenerate case, no smooth

# --- Bio-Logic (.mpt) utilities from original script --------------------
def eclab_header_row(fp: Path):
    with fp.open("r", encoding=ENC, errors="ignore") as f:
        for line in f:
            if line.lower().startswith("nb header lines"):
                hdr = int(line.split(":")[-1].strip()) - 1
                break
        else:
            raise RuntimeError("EC-Lab header count not found")
    with fp.open("r", encoding=ENC, errors="ignore") as f:
        for i,l in enumerate(f):
            if i == hdr:
                return hdr, l.rstrip().split("\t")

def eclab_pick(cols, charge):
    m = lambda p: next((c for c in cols if re.search(p, c, re.I)), None)
    v  = m(r"(ewe|ecell).*v")
    q  = m(r"q.*charge.*m?a\.?h" if charge else r"q.*discharge.*m?a\.?h")
    cyc, half = m(r"cycle.*(index|number)|\bNs\b"), m(r"half\s*cycle")
    if not (v and q):
        raise KeyError("V or Q column missing")
    return v,q,cyc,half

def load_eclab(fp: Path, cycle: int, charge: bool) -> pd.DataFrame:
    hdr, cols = eclab_header_row(fp)
    v,q,cyc,half = eclab_pick(cols, charge)
    use = [v,q] + [c for c in (cyc,half) if c]
    chunks, rows = [], 0
    for ch in pd.read_csv(fp, sep="\t", header=None, names=cols,
                          skiprows=range(hdr+1), usecols=use,
                          chunksize=CHUNK, engine="python", encoding=ENC):
        if cyc:  ch = ch[ch[cyc] == cycle-1]      # Bio-Logic = 0-based
        if half: ch = ch[ch[half] == (0 if charge else 1)]
        chunks.append(ch[[v,q]])
        rows += len(ch)
    df = pd.concat(chunks, ignore_index=True).astype(float)
    df.columns = ["V","QmAh"]; _dbg(f"  EC-Lab rows {rows}→{len(df)}")
    return df

# --- Arbin (.xlsx) utilities -------------------------------------------
_clean = lambda s: re.sub(r"[^a-z]", "", s.lower())
def load_arbin(fp: Path, cycle: int, charge: bool) -> pd.DataFrame:
    df0 = pd.read_excel(fp, sheet_name=1, engine="openpyxl")
    cmap = {_clean(c): c for c in df0.columns}
    v_key  = next(k for k in cmap if k.startswith("voltage"))
    q_key  = next(k for k in cmap if k.startswith(
               "chargecapacity" if charge else "dischargecapacity"))
    cyc_key  = next((k for k in cmap if k.startswith(("cycleindex","cyclenumber"))), None)
    half_key = next((k for k in cmap if k.startswith("halfcycle")), None)
    df = df0[[cmap[v_key], cmap[q_key]]].copy()
    if cyc_key:  df = df[df0[cmap[cyc_key]] == cycle]
    if half_key: df = df[df0[cmap[half_key]] == (0 if charge else 1)]
    df.columns = ["V","QmAh"]; df.dropna(inplace=True)
    _dbg(f"  Arbin rows {len(df)}")
    return df.reset_index(drop=True).astype(float)

# ═════════════════════ PCGA step-capacity helpers ═══════════════════════
def read_mpt_header(path: Path) -> int:
    with path.open("r", errors="ignore") as fh:
        for i,ln in enumerate(fh):
            if "Nb header lines" in ln:
                return int(ln.split(":")[1].strip())
            if i>100: break
    raise RuntimeError(f"Header count not found in {path.name}")

def integrate_pcga_steps(df, masses, cell_id, EPS_DV=0.003):
    ctrl = next((c for c in df.columns if re.search(r"control.*?/V", c, re.I)), "Ewe/V")
    v_target = df[ctrl].astype(float).reset_index(drop=True)
    v_meas   = df["Ewe/V"].astype(float).reset_index(drop=True)
    i_col    = next(c for c in df.columns if re.search(r"I.*?/mA", c))
    cur      = df[i_col].astype(float).reset_index(drop=True)
    t        = df["time/s"].astype(float).reset_index(drop=True)

    # robust step boundaries
    idx = None
    if "Ns" in df.columns:
        Ns = pd.to_numeric(df["Ns"], errors="coerce").ffill().astype(int).reset_index(drop=True)
        if Ns.nunique() > 2:
            idx = [0] + [k for k in range(1, len(Ns)) if Ns[k] != Ns[k-1]] + [len(Ns)]
    # 1 mV rounding on target
    if idx is None or len(idx) < 12:
        v_round = (v_target / 1e-3).round().astype("Int64")
        idx2 = [0] + [k for k in range(1, len(v_round)) if v_round[k] != v_round[k-1]] + [len(v_round)]
        if len(idx2) > (len(idx) if idx else 0):
            idx = idx2
    # adaptive as fallback
    if idx is None or len(idx) < 12:
        diffs = (v_target.shift(-1) - v_target).abs().to_numpy()[:-1]
        nz = diffs[diffs > 0]
        if nz.size:
            thr = max(0.25 * nz.min(), 5e-4)  # at least 0.5 mV
            idx3 = [0] + [k for k in range(1, len(v_target)) if abs(v_target.iloc[k] - v_target.iloc[k-1]) > thr] + [len(v_target)]
            if len(idx3) > (len(idx) if idx else 0):
                idx = idx3
    # measured Ewe/V rounding last
    if idx is None or len(idx) < 12:
        v_round_m = (v_meas / 1e-3).round().astype("Int64")
        idx = [0] + [k for k in range(1, len(v_round_m)) if v_round_m[k] != v_round_m[k-1]] + [len(v_round_m)]

    Vmid, dQ_dV = [], []
    m = masses.get(cell_id)
    if m is None:
        raise KeyError(f"Mass not defined for cell {cell_id}")
    step_dv = 0.003  # 3 mV as specified
    for s, e in zip(idx[:-1], idx[1:]):
        if e - s < 3:
            continue
        dt = t.iloc[s:e].diff().fillna(0.0)
        Q_mAh = (cur.iloc[s:e] * dt).sum() / 3600.0
        if Q_mAh == 0:
            continue
        q_mAh_g = Q_mAh / m
        dQdV = q_mAh_g / step_dv
        Vmid.append(v_meas.iloc[s:e].mean())
        dQ_dV.append(dQdV)

    return np.array(Vmid), np.array(dQ_dV)
    return np.array(Vmid), np.array(dQ_dV)

def pcga_files_in(dir_: Path) -> list[Path]:
    return sorted(p for p in dir_.rglob("*_03_PCGA_*.mpt"))

def cell_short_id(fname: str) -> str:
    first = fname.split("_")[0]                 # BL-LL-GD01
    return first.split("-")[-1] if "-" in first else first
# ════════════════════════════════════════════════════════════════════════

def main():
    fig, (ax_dqdv, ax_charge) = plt.subplots(
        1, 2, figsize=(11, 4.5), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    # --------------- dQ/dV & charge-curve processing --------------------
    for idx,fname in enumerate(FILES):
        fp = DATA_DIR / fname
        stem = Path(fname).stem
        m = re.search(r"LL-([A-Za-z0-9]{4})", stem, re.I)
        cell_id = m.group(1).upper() if m else stem
        print(f"Processing dQ/dV file  {fname}")

        try:
            if fp.suffix.lower() == ".mpt":
                if "03_GCPL" in fname:
                    df_raw = load_eclab(fp, cycle=1, charge=CHARGE)  # first cycle for GCPL
                else:
                    df_raw = load_eclab(fp, CYCLE, CHARGE)
            elif fp.suffix.lower() in (".xls",".xlsx"):
                df_raw = load_arbin(fp, CYCLE, CHARGE)
                df_raw["QmAh"] *= 1000
            else:
                print("   (unknown format, skipped)")
                continue
        except Exception as e:
            print("   ✗",e); continue

        df_bin = fixed_bin(df_raw, BIN_W)
        if len(df_bin) <= POLY_PRE+2:
            print("   (too few points, skipped)"); continue

        v_mid,y = (savgol_dqdv if DQDV_SMOOTH else raw_dqdv)(df_bin)
        if cell_id in MASS_MG:
            y /= (MASS_MG[cell_id]*1000)
        label = f"{cell_id}: {electrolyte_lookup.get(cell_id, 'Unknown')}"
        ax_dqdv.plot(v_mid,y,lw=1.3,label=label,
                     color=cmap(idx%10))

        v_curve = df_bin["V"].to_numpy()
        q_curve = df_bin["QmAh"].to_numpy()
        if cell_id in MASS_MG:
            q_curve /= (MASS_MG[cell_id]*1000)
        label = f"{cell_id}: {electrolyte_lookup.get(cell_id, 'Unknown')}"
        ax_charge.plot(q_curve, v_curve,lw=1.3,label=label,
                       color=cmap(idx%10))

    # -------------------- PCGA overlay (mAh g^-1 V^-1) ------------------
    pcga_files = pcga_files_in(DATA_DIR)
    print(f"\nFound {len(pcga_files)} *_03_PCGA_*.mpt files")

    used_labels = set()
    for p in pcga_files:
        try:
            hdr = read_mpt_header(p)
            df_pcga = pd.read_csv(
                p,
                sep="\t",
                skiprows=hdr - 1,
                header=0,
                engine="python",
                encoding="ISO-8859-1",
                on_bad_lines="skip",
            )
            cid = cell_short_id(p.name)

            if cid not in MASS_G:
                print(f" ⚠︎  MASS_G entry missing for '{cid}' – file skipped")
                continue
            EPS_DV = 0.003
            V, dQdV = integrate_pcga_steps(df_pcga, MASS_G, cid, EPS_DV)
            print(f"   {p.name}  →  {len(V)} valid steps")

            if len(V) == 0:
                continue  # nothing to plot

            # create one legend label per cell
            label = electrolyte_lookup[cid]
            if label in used_labels:
                label = "_nolegend_"
            else:
                used_labels.add(label)
            #ax_pcga = ax_dqdv.twinx()  # secondary Y-axis for PCGA
            # ax_dqdv.scatter(
            #     V, dQdV,
            #     marker="o",  # filled circle
            #     s=48,  # bigger
            #     linewidths=0.5,
            #     edgecolors="black",
            #     alpha=0.9,
            #     zorder=10,  # sit on top of curves
            #     label=label,
            # )
            label = f"{cell_id}: {electrolyte_lookup.get(cell_id, 'Unknown')}"
            ax_dqdv.plot(V, dQdV,
                         color=cmap(len(used_labels) % 10),
                         linestyle= "-",  # dashed line
                         label=label)

        except Exception as e:
            print(f"   ✗  {p.name}  →  {e}")

    # --------------- cosmetics ------------------------------------------
    ax_dqdv.set_xlabel("Voltage (V)")
    ax_dqdv.set_ylabel("dQ/dV (mAh g$^{-1}$ V$^{-1}$)")
    #ax_pcga.set_ylabel("PCGA dQ/dV (mAh g$^{-1}$ V$^{-1}$)")
    tag = "smoothed" if DQDV_SMOOTH else "raw"
    ax_dqdv.set_title(f"dQ/dV ({tag}) – C{CYCLE}")
    ax_dqdv.set_ylim(0, 0.07)
    ax_dqdv.set_xlim(1.8, 3.6)
    ax_charge.set_xlabel("Capacity (mAh g$^{-1}$)" if MASS_MG
                         else "Capacity (mAh)")
    ax_charge.set_ylabel("Voltage (V)")
    ax_charge.set_title(f"Charge curves – C{CYCLE}")

    ax_dqdv.legend(loc="best", fontsize="x-small")
    ax_charge.legend(loc="best", fontsize="x-small")
    #plt.savefig(DATA_DIR / "dq_dv_pitt.png", dpi=300, bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    main()
