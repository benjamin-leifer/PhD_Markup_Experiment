import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# --- Helpers ---
def detect_skiprows(path):
    with open(path, "r", encoding="latin1", errors="ignore") as f:
        for i, line in enumerate(f):
            m = re.search(r"Nb header lines\s*:\s*(\d+)", line)
            if m:
                return max(int(m.group(1)) - 1, 0)
    return 74

def monotonic_trim(capacity_values):
    """Strictly enforce monotonic increase: keep points up to the first decrease."""
    keep_len = 1
    for i in range(1, len(capacity_values)):
        if capacity_values[i] >= capacity_values[i-1]:
            keep_len += 1
        else:
            break
    return keep_len

def extract_segments_trimmed(df, cycle, i_col, cap_col, v_col, current_cutoff_mA=0.01, trim_ends=2):
    """Split into charge (I>0) and discharge (I<0). Trim first/last N points and cut at first non-monotonic point."""
    cyc = df[df["cycle number"] == cycle].copy()
    if cyc.empty:
        return [], []
    # filter current cutoff
    cyc = cyc[cyc[i_col].abs() >= current_cutoff_mA].copy()
    if cyc.empty:
        return [], []
    # ensure time order
    if "time/s" in cyc.columns:
        cyc.sort_values("time/s", inplace=True)
    # split
    charge = cyc[cyc[i_col] > 0].copy()
    discharge = cyc[cyc[i_col] < 0].copy()
    segs_charge = []
    segs_discharge = []
    for seg, out in [(charge, segs_charge), (discharge, segs_discharge)]:
        if seg.empty:
            continue
        # optional end trim
        if len(seg) > 2*trim_ends:
            seg = seg.iloc[trim_ends:-trim_ends].copy()
        # strict monotonic trim (increasing capacity expected from instrument)
        cap_vals = seg[cap_col].values
        k = monotonic_trim(cap_vals)
        seg = seg.iloc[:k].copy()
        if len(seg) >= 2:
            out.append((seg[cap_col].values, seg[v_col].values))
    return segs_charge, segs_discharge

# --- Inputs ---
path = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Results\Zn Ion GPE\2024\Comparisons\Cycling"
file_gcpl_liquid = path+"/0126_02_0.5MZnTFSI_ZnC_GF_Ch_InsideGB_0hr_RT_t3_03_GCPL_C02.mpt"  # liquid EC/PC, RT
file_gcpl_gpe50 = path+"/0225_04_0.5MZnTFSI_Zn_GPE_Ch_InsideGB_0hr_50C_04_GCPL_C05.mpt"    # GPE, 50C

active_mass_g = 0.0006
current_cutoff_mA = 0.01
sel_cycles = [3, 5, 10, 20]
markers = {3: "o", 5: "s", 10: "D", 20: "^"}

# --- Build combined plot ---
plt.figure(figsize=(8,6))

for color, (path, label) in zip(["tab:blue", "tab:orange"], [
    (file_gcpl_liquid, "ZnTFSI liquid EC/PC RT"),
    (file_gcpl_gpe50, "ZnTFSI GPE 50°C"),
]):
    skip = detect_skiprows(path)
    df = pd.read_csv(path, sep="\t", skiprows=skip, encoding="latin1")
    # detect cols
    i_col = "<I>/mA" if "<I>/mA" in df.columns else ("I/mA" if "I/mA" in df.columns else None)
    if i_col is None or not {"Ewe/V", "Capacity/mA.h", "cycle number"}.issubset(df.columns):
        continue
    df = df.dropna(subset=["Ewe/V", "Capacity/mA.h", "cycle number"]).copy()
    df["Capacity (mAh/g)"] = df["Capacity/mA.h"] / active_mass_g
    for cyc in sel_cycles:
        ch_segs, dis_segs = extract_segments_trimmed(
            df, cyc, i_col, "Capacity (mAh/g)", "Ewe/V",
            current_cutoff_mA=current_cutoff_mA, trim_ends=2
        )
        # plot charge segments
        for cap, volt in ch_segs:
            markevery = max(1, len(cap) // 20)
            plt.plot(cap, volt, color=color, marker=markers.get(cyc, "o"), markevery=markevery,
                     linestyle="-", label=f"{label} – Cycle {cyc}")
        # plot discharge segments
        for cap, volt in dis_segs:
            markevery = max(1, len(cap) // 20)
            plt.plot(cap, volt, color=color, marker=markers.get(cyc, "o"), markevery=markevery,
                     linestyle="--",)

plt.xlabel("Capacity (mAh/g)")
plt.ylabel("Voltage (V)")
plt.title("RT Chevrel Performance vs. 50C Performance")
plt.legend(fontsize=8, ncol=1, loc="center right")
plt.tight_layout()
plt.show()
