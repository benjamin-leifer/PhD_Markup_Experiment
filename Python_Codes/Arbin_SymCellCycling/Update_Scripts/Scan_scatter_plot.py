"""
scatter_time_at_full_charge_vs_minus51C_capacity.py

Creates:
  (1) Combined scatter: time-at-full-charge vs -51C discharge capacity (mAh/g)
  (2) Per-electrolyte small-multiples grid (same axes)

Color = Electrolyte
Marker = Replicate Set (trial repeat)

Normalization to mAh/g:
  REF_CAP_MAH = 4.0
  REF_SPEC_MAH = 160.6
  CONV_AH_TO_MAHG = 1000.0 * REF_SPEC_MAH / REF_CAP_MAH  # 40150.0
So:
  Ah  -> mAh/g : Ah * CONV_AH_TO_MAHG
  mAh -> mAh/g : (mAh/1000) * CONV_AH_TO_MAHG  == mAh * (REF_SPEC_MAH/REF_CAP_MAH)
"""

from __future__ import annotations

import os
import math
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------- USER CONFIG ----------------------
INPUT_CSV = r"C:\Users\benja\Downloads\_summary_exports\charge_discharge_metrics_minus51C.csv"
OUTPUT_DIR = r"C:\Users\benja\Downloads\_summary_exports\_plots_time_at_full_charge"
SAVE_FIGS = True
DPI = 200

# columns expected in your CSV
TIME_COL = "ChargeEnd_to_DischargeStart_h"     # hours between end-of-charge and discharge start
CELL_COL = "CellCode"
# choose ONE of these depending on your CSV. Script will try to infer if not found.
CAP_COL_CANDIDATES = ["DischargeCapacity_mAh", "DischargeCapacity_Ah", "DischargeCap_mAh", "DischargeCap_Ah"]

# If you want to average multiple files/rows per CellCode before plotting:
AGGREGATE_TO_CELL_MEAN = True

# marker mapping for replicate set
REP_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}

# ---------------------- NORMALIZATION ----------------------
REF_CAP_MAH = 4.0
REF_SPEC_MAH = 160.6
CONV_AH_TO_MAHG = 1000.0 * REF_SPEC_MAH / REF_CAP_MAH  # 40150.0
CONV_MAH_TO_MAHG = REF_SPEC_MAH / REF_CAP_MAH          # 40.15


# ---------------------- REPLICATE/ELECTROLYTE MAP ----------------------
# Paste of the table you provided (tab-delimited). We only need Cell ID, Replicate Set, Electrolyte.
REPLICATE_TABLE_TSV = """Cell ID\tReplicate Set\tElectrolyte\tAnode\tCathode\tSpecific Discharge Capacity at -51C, (mAh/g) (Cutoff at 2V)
FA01\t1\tDT14\tLi\tNMC532\t70.7
EN04\t1\tDTF14-1\tGr\tNMC532\t8.3
DU06\t1\tDTF14-2\tGr\tNMC532\t23.6
EO05\t1\tDTF14-5\tGr\tNMC532\t0.2
EJ05\t1\tDTF14-10\tGr\tNMC532\t0.9
FC04\t1\tDTFV1411\tGr\tNMC532\t54.2
FD04\t1\tDTFV1412\tGr\tNMC532\t54
FE04\t1\tDTFV1421\tGr\tNMC532\t8.8
FF05\t1\tDTFV1422\tGr\tNMC532\t59.1
FG05\t1\tDTFV1452\tGr\tNMC532\t39
ES05\t1\tDTFV14102\tGr\tNMC532\t39.3
HU01\t2\tDTF141\tGr\tNMC532\t70.8
HV01\t2\tDTF142\tGr\tNMC532\t67.8
HW04\t2\tDTF145\tGr\tNMC532\t74.7
HX01\t2\tDTF1410\tGr\tNMC532\t70.3
IA01\t2\tDTFV1411\tGr\tNMC532\t67.4
IB01\t2\tDTFV1421\tGr\tNMC532\t55.8
IC04\t2\tDTFV1412\tGr\tNMC532\t38.4
ID02\t2\tDTFV1422\tGr\tNMC532\t35.9
IE02\t2\tDTFV1452\tGr\tNMC532\t37.3
IF02\t2\tDTFV14102\tGr\tNMC532\t16.0
IP04\t3\tDTFV1411\tGr\tNMC532\t71.7
IQ02\t3\tDTFV1421\tGr\tNMC532\t73.1
IR03\t3\tDTFV1412\tGr\tNMC532\t69.1
IS03\t3\tDTFV1422\tGr\tNMC532\t76.7
IT01\t3\tDTFV1452\tGr\tNMC532\t80.3
IU03\t3\tDTFV14102\tGr\tNMC532\t75.2
IV03\t3\tDTF141\tGr\tNMC532\t35.6
IW04\t3\tDTF142\tGr\tNMC532\t79.8
IX03\t3\tDTF145\tGr\tNMC532\t77.4
IY01\t3\tDTF1410\tGr\tNMC532\t57.4
"""


def load_replicate_map() -> pd.DataFrame:
    m = pd.read_csv(StringIO(REPLICATE_TABLE_TSV), sep="\t")
    m = m.rename(columns={"Cell ID": "CellCode", "Replicate Set": "ReplicateSet"})
    m["CellCode"] = m["CellCode"].astype(str).str.strip()
    m["Electrolyte"] = m["Electrolyte"].astype(str).str.strip()
    m["ReplicateSet"] = pd.to_numeric(m["ReplicateSet"], errors="coerce")
    return m[["CellCode", "ReplicateSet", "Electrolyte"]]


def infer_capacity_column(df: pd.DataFrame) -> tuple[str, str]:
    """
    Returns (capacity_col, units) where units in {"mAh","Ah"}.
    """
    cols = set(df.columns)
    for c in CAP_COL_CANDIDATES:
        if c in cols:
            if c.lower().endswith("_ah"):
                return c, "Ah"
            if c.lower().endswith("_mah"):
                return c, "mAh"
            # fallback inference
            return c, "mAh"

    # last-resort: look for something that resembles discharge capacity
    for c in df.columns:
        cl = c.lower()
        if "discharge" in cl and "cap" in cl:
            # infer units by name
            if "ah" in cl and "mah" not in cl:
                return c, "Ah"
            return c, "mAh"

    raise ValueError(
        "Could not infer discharge capacity column. "
        f"Looked for: {CAP_COL_CANDIDATES}. Available columns: {list(df.columns)}"
    )


def to_mAh_per_g(cap_values: pd.Series, units: str) -> pd.Series:
    """
    Convert capacity to mAh/g using your scheme.
    """
    x = pd.to_numeric(cap_values, errors="coerce")
    if units == "Ah":
        return x * CONV_AH_TO_MAHG
    if units == "mAh":
        return x * CONV_MAH_TO_MAHG
    raise ValueError(f"Unknown units: {units}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    if CELL_COL not in df.columns:
        raise ValueError(f"Expected '{CELL_COL}' column in CSV. Found: {list(df.columns)}")
    if TIME_COL not in df.columns:
        raise ValueError(f"Expected '{TIME_COL}' column in CSV. Found: {list(df.columns)}")

    df[CELL_COL] = df[CELL_COL].astype(str).str.strip()

    cap_col, cap_units = infer_capacity_column(df)
    df["DischargeCapacity_mAh_g"] = to_mAh_per_g(df[cap_col], cap_units)
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")

    # Merge replicate/electrolyte map (only keeps rows that appear in your table)
    map_df = load_replicate_map()
    d = df.merge(map_df, on="CellCode", how="inner")

    # Clean
    d = d[np.isfinite(d[TIME_COL]) & np.isfinite(d["DischargeCapacity_mAh_g"])]
    d = d[d[TIME_COL] >= 0]

    if d.empty:
        raise RuntimeError(
            "No rows after merge/cleaning. "
            "Check that your CSV CellCode values match the IDs in the replicate table."
        )

    # Optionally aggregate multiple rows per cell (e.g., multiple files)
    if AGGREGATE_TO_CELL_MEAN:
        d_plot = (
            d.groupby(["CellCode", "Electrolyte", "ReplicateSet"], as_index=False)
             .agg({TIME_COL: "mean", "DischargeCapacity_mAh_g": "mean"})
        )
    else:
        d_plot = d.copy()

    # Color map by electrolyte
    electrolytes = sorted(d_plot["Electrolyte"].unique())
    cmap = plt.get_cmap("tab20")
    color_map = {elyte: cmap(i % cmap.N) for i, elyte in enumerate(electrolytes)}

    # --------------- FIG 1: Combined scatter ---------------
    fig1 = plt.figure(figsize=(9, 6))
    ax1 = plt.gca()

    for rep_set, sub_rep in d_plot.groupby("ReplicateSet"):
        rep_set_int = int(rep_set) if np.isfinite(rep_set) else -1
        marker = REP_MARKERS.get(rep_set_int, "o")
        for elyte, sub in sub_rep.groupby("Electrolyte"):
            ax1.scatter(
                sub[TIME_COL], sub["DischargeCapacity_mAh_g"],
                marker=marker,
                c=[color_map[elyte]] * len(sub),
                edgecolors="black", linewidths=0.5,
                s=60,
            )

    ax1.set_xlabel("Time at full charge before -51C discharge (h)")
    ax1.set_ylabel("Discharge capacity at -51C (mAh/g)")
    ax1.set_title("Time at full charge vs -51C discharge capacity")
    ax1.grid(True, alpha=0.25)

    # Two legends: colors (electrolyte) + markers (replicate set)
    handles_colors = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=color_map[e], markeredgecolor="black",
               label=e, markersize=8)
        for e in electrolytes
    ]

    rep_sets_present = sorted([int(x) for x in d_plot["ReplicateSet"].dropna().unique()])
    handles_reps = [
        Line2D([0], [0], marker=REP_MARKERS.get(r, "o"), linestyle="None",
               color="black", markerfacecolor="white", markeredgecolor="black",
               label=f"Replicate set {r}", markersize=8)
        for r in rep_sets_present
    ]

    leg1 = ax1.legend(handles=handles_colors, title="Electrolyte", loc="upper right", frameon=False)
    ax1.add_artist(leg1)
    ax1.legend(handles=handles_reps, title="Trial repeat", loc="lower left", frameon=False)

    fig1.tight_layout()
    if SAVE_FIGS:
        out1 = os.path.join(OUTPUT_DIR, "scatter_time_full_charge_vs_cap_minus51C_mAhg.png")
        fig1.savefig(out1, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {out1}")

    # --------------- FIG 2: Per-electrolyte small multiples ---------------
    n = len(electrolytes)
    cols = 3 if n > 2 else n
    rows = int(math.ceil(n / cols))

    fig2, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.0), squeeze=False)

    # shared axis limits
    xlim = (0, max(1.0, float(d_plot[TIME_COL].max()) * 1.05))
    ylim = (0, max(1.0, float(d_plot["DischargeCapacity_mAh_g"].max()) * 1.05))

    for i, elyte in enumerate(electrolytes):
        ax = axes[i // cols][i % cols]
        sub_e = d_plot[d_plot["Electrolyte"] == elyte]

        for rep_set, sub in sub_e.groupby("ReplicateSet"):
            rep_set_int = int(rep_set) if np.isfinite(rep_set) else -1
            ax.scatter(
                sub[TIME_COL], sub["DischargeCapacity_mAh_g"],
                marker=REP_MARKERS.get(rep_set_int, "o"),
                c=[color_map[elyte]] * len(sub),
                edgecolors="black", linewidths=0.5,
                s=55,
                label=f"Set {rep_set_int}",
            )

        ax.set_title(elyte)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("Hours")
        ax.set_ylabel("mAh/g")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="best")

    # Hide unused panels
    for j in range(i + 1, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig2.suptitle("Per-electrolyte: time at full charge vs -51C discharge capacity (mAh/g)", y=1.02)
    fig2.tight_layout()

    if SAVE_FIGS:
        out2 = os.path.join(OUTPUT_DIR, "small_multiples_time_full_charge_vs_cap_minus51C_mAhg.png")
        fig2.savefig(out2, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {out2}")

    plt.show()


if __name__ == "__main__":
    main()
