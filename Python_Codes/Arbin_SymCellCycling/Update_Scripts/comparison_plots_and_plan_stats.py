# Comparison_plots_and_plan_stats.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Import your already-parsed data + styling from Cell_build_2.py
# -----------------------------------------------------------------------------
try:
    import Cell_build_2 as cb
except Exception as e:
    raise ImportError(
        "Couldn't import Cell_build_2.py. "
        "Make sure this script is in the same directory and that Cell_build_2.py runs cleanly first.\n"
        f"Original error: {e}"
    )

# -----------------------------------------------------------------------------
# Output folder
# -----------------------------------------------------------------------------
COMP_DIR = os.path.join(cb.OUT_DIR, "comparisons")
os.makedirs(COMP_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (7, 5)
plt.rcParams["font.size"] = 11


# -----------------------------------------------------------------------------
# Groups (based on your experiment plan)
# -----------------------------------------------------------------------------
RT_BASES = ["AC-5", "AE-2", "AE-3", "BC-3", "BE-1", "BE-2", "CC-3", "CE-1", "CE-2"]
COLD_BASES = RT_BASES.copy()
LARGE_BASES = ["E1", "E2", "F1", "F2"]

PLANS = {
    "RT_small_format":   {"bases": RT_BASES,   "suffix": "RT"},
    "COLD_-21C_small":   {"bases": COLD_BASES, "suffix": "-21C"},
    "Large_format":      {"bases": LARGE_BASES,"suffix": None},
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_cycles_for_base(base, suffix):
    """Fetch cycles dict for base+suffix the way your main script stores it."""
    key = base if suffix is None else f"{base} - {suffix}"
    return cb.cell_cycles_data.get(key)

def style_for_base(base, suffix):
    meta = cb.CELL_META.get(base, cb.CellMeta("experimental", None))
    color = cb.BASE_COLOR_MAP.get(base, "gray")
    lw = cb.LW_BY_FILL.get(meta.fill_mL, cb.DEFAULT_LW)
    alpha = cb.ALPHA_BY_TEMP.get(suffix, 1.0) if suffix else 1.0
    return meta, color, lw, alpha

def curve_to_x(curve, base, xmode):
    """Convert list of (Ah, V) to desired x-axis."""
    if xmode == "Ah":
        return [q for q, _ in curve]
    elif xmode == "mAhg":
        mass_g = cb.ACTIVE_MASS_G.get(base, np.nan)
        if not np.isfinite(mass_g):
            return [np.nan] * len(curve)
        return [(q * 1000.0 / mass_g) for q, _ in curve]
    else:
        raise ValueError("xmode must be 'Ah' or 'mAhg'")

def make_axes_pretty(ax):
    ax.set_ylim(2.3, 4.4)
    ax.set_xlim(left=0)
    ax.grid(False)
    ax.tick_params(which='both', axis='both', direction='in',
                   bottom=True, left=True, right=True, top=True,
                   labelbottom=True, labelleft=True)


# -----------------------------------------------------------------------------
# Plotting: experimental vs control comparisons per plan
# -----------------------------------------------------------------------------
def plot_plan(plan_name, bases, suffix, xmode="Ah"):
    fig, ax = plt.subplots()

    for base in bases:
        cycles = get_cycles_for_base(base, suffix)
        if not cycles:
            continue

        meta, color, lw, alpha = style_for_base(base, suffix)

        # Plot cycles 1–5 in same style as individual plots.
        # Legend: only label cycle 1 curves so it stays readable.
        for cyc in range(1, 6):
            cd = cycles.get(cyc, {"charge": [], "discharge": []})
            marker = cb.CYCLE_MARKERS[cyc]

            # charge
            if cd["charge"]:
                x = curve_to_x(cd["charge"], base, xmode)
                y = [v for _, v in cd["charge"]]
                label = f"{base} (chg)" if cyc == 1 else "_nolegend_"
                me = max(1, int(len(x) / 12))
                ax.plot(
                    x, y, "-",
                    color=color, lw=lw, alpha=alpha,
                    marker=marker, markevery=me, ms=5,
                    label=label
                )

            # discharge
            if cd["discharge"]:
                x = curve_to_x(cd["discharge"], base, xmode)
                y = [v for _, v in cd["discharge"]]
                label = f"{base} (dis)" if cyc == 1 else "_nolegend_"
                me = max(1, int(len(x) / 12))
                ax.plot(
                    x, y, "--",
                    color=color, lw=lw, alpha=alpha,
                    marker=marker, markevery=me, ms=5,
                    label=label
                )

    # Titles + labels
    ax.set_title(f"{plan_name} | {suffix or 'n/a'} | x={xmode}")
    ax.set_xlabel("Capacity (Ah)" if xmode == "Ah" else "Specific Capacity (mAh/g)")
    ax.set_ylabel("Voltage (V)")
    make_axes_pretty(ax)

    # Legend grouped and readable
    ax.legend(fontsize=8, ncol=2, loc="best", frameon=False)

    fig.tight_layout()
    out_png = os.path.join(COMP_DIR, f"{plan_name}_{suffix or 'NA'}_{xmode}.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print("Saved comparison plot:", out_png)


# Make all comparison plots (Ah + mAh/g for each plan)
for plan_name, cfg in PLANS.items():
    plot_plan(plan_name, cfg["bases"], cfg["suffix"], xmode="Ah")
    plot_plan(plan_name, cfg["bases"], cfg["suffix"], xmode="mAhg")


# -----------------------------------------------------------------------------
# Plan summary statistics
# -----------------------------------------------------------------------------
def build_plan_rows(bases, suffix):
    rows = []
    for base in bases:
        cycles = get_cycles_for_base(base, suffix)
        if not cycles:
            continue

        meta = cb.CELL_META.get(base, cb.CellMeta("experimental", None))
        mass_g = cb.ACTIVE_MASS_G.get(base, np.nan)

        for cyc in range(1, 6):
            cd = cycles.get(cyc, {"charge": [], "discharge": []})

            Vc_avg, Qc_Ah = cb.avg_voltage(cd["charge"])
            Vd_avg, Qd_Ah = cb.avg_voltage(cd["discharge"])
            CE = (Qd_Ah / Qc_Ah * 100.0) if Qc_Ah and Qc_Ah > 1e-9 else np.nan

            Qc_mAhg = (Qc_Ah * 1000.0 / mass_g) if np.isfinite(mass_g) else np.nan
            Qd_mAhg = (Qd_Ah * 1000.0 / mass_g) if np.isfinite(mass_g) else np.nan

            rows.append({
                "Base Cell": base,
                "Temp": suffix or "Large",
                "Electrolyte Type": meta.electrolyte_type,
                "Fill (mL)": meta.fill_mL,
                "Active Mass (g)": mass_g,
                "Cycle": cyc,
                "Charge Capacity (Ah)": Qc_Ah,
                "Discharge Capacity (Ah)": Qd_Ah,
                "Charge Spec Cap (mAh/g)": Qc_mAhg,
                "Discharge Spec Cap (mAh/g)": Qd_mAhg,
                "Avg Charge V (V)": Vc_avg,
                "Avg Discharge V (V)": Vd_avg,
                "Coulombic Eff (%)": CE
            })
    return pd.DataFrame(rows)


def summarize_plan(df):
    """
    Plan-level summary:
      - split by electrolyte type (control vs experimental)
      - report cycle 1 stats and cycles 2–5 average stats
    """
    out_blocks = []

    for etype in ["control", "experimental"]:
        dfe = df[df["Electrolyte Type"] == etype].copy()
        if dfe.empty:
            continue

        # cycle 1 only
        c1 = dfe[dfe["Cycle"] == 1]
        # cycles 2–5
        c25 = dfe[dfe["Cycle"].between(2, 5)]

        def stats_block(label, sub):
            return {
                "Electrolyte Type": etype,
                "Window": label,
                "n_cells": sub["Base Cell"].nunique(),
                "Discharge Spec Cap mean (mAh/g)": sub["Discharge Spec Cap (mAh/g)"].mean(),
                "Discharge Spec Cap std (mAh/g)": sub["Discharge Spec Cap (mAh/g)"].std(),
                "Avg Discharge V mean (V)": sub["Avg Discharge V (V)"].mean(),
                "Avg Discharge V std (V)": sub["Avg Discharge V (V)"].std(),
                "CE mean (%)": sub["Coulombic Eff (%)"].mean(),
                "CE std (%)": sub["Coulombic Eff (%)"].std(),
            }

        if not c1.empty:
            out_blocks.append(stats_block("Cycle 1", c1))
        if not c25.empty:
            out_blocks.append(stats_block("Cycles 2–5", c25))

    return pd.DataFrame(out_blocks)


# Build + save stats per experimental plan
for plan_name, cfg in PLANS.items():
    df_plan = build_plan_rows(cfg["bases"], cfg["suffix"])

    # per-cell/per-cycle table
    df_plan_out = os.path.join(COMP_DIR, f"{plan_name}_per_cell_table.xlsx")
    df_plan.to_excel(df_plan_out, index=False)
    print("Saved per-cell table:", df_plan_out)

    # plan-level experimental vs control summary
    df_summary = summarize_plan(df_plan)
    df_summary_out_csv = os.path.join(COMP_DIR, f"{plan_name}_summary_stats.csv")
    df_summary_out_xlsx = os.path.join(COMP_DIR, f"{plan_name}_summary_stats.xlsx")

    df_summary.to_csv(df_summary_out_csv, index=False)
    df_summary.to_excel(df_summary_out_xlsx, index=False)
    print("Saved plan summary stats:", df_summary_out_csv)
    print("Saved plan summary stats:", df_summary_out_xlsx)
