# python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 14,           # default font size
    'axes.labelsize': 18,      # x/y label size
    'axes.titlesize': 18,      # title size
    'xtick.labelsize': 16,     # x tick label size
    'ytick.labelsize': 16,     # y tick label size
    'legend.fontsize': 14
})
from datetime import datetime

EXCEL_PATH = r"C:\Users\benja\Downloads\10_22_2025 Temp\Slide Trials\Local_Images\LP\capacity_ce_export_v3.xlsx"
SHEET_NAME = "Plot"
CAPACITY_COL = "Specific Charge Capacity (mAh/g)"
OUT_DIR = Path(r"C:\Users\benja\Downloads\10_22_2025 Temp\Slide Trials\Local_Images\LP\LPcyclelife_plots")

def _norm(s):
    return s.strip().lower()

def _find_cycle_and_ce(df: pd.DataFrame, capacity_col: str):
    cycle_key = next((c for c in df.columns if "cycle" in _norm(c)), df.columns[0])
    ce_key = next((c for c in df.columns if "coulombic" in _norm(c) or _norm(c)=="ce"), None)
    #if ce_key is None:
    #    ce_key = next((c for c in df.columns if c not in (cycle_key, capacity_col)), df.columns[-1])
    return cycle_key, ce_key

def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    cycle_key, ce_key = _find_cycle_and_ce(df, CAPACITY_COL)

    plot_df = df[[cycle_key, CAPACITY_COL, ce_key]].copy()
    for c in [cycle_key, CAPACITY_COL, ce_key]:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
    plot_df = plot_df.dropna(subset=[cycle_key, CAPACITY_COL, ce_key])
    #plot_df = plot_df[plot_df[cycle_key] != 1]
    plot_df = plot_df.sort_values(by=cycle_key)

    ce_vals = plot_df[ce_key].copy()
    if ce_vals.max() <= 1.01 and ce_vals.min() >= 0.99:
        ce_vals = ce_vals * 100.0

    fig, ax1 = plt.subplots(figsize=(9,5))
    ax2 = ax1.twinx()

    lns1 = ax1.plot(
        plot_df[cycle_key],
        plot_df[CAPACITY_COL],
        marker='o', markersize=5, color='blue',
        label="Specific Capacity (mAh/g)"
    )

    # CE: diamond marker, not filled
    lns2 = ax2.plot(
        plot_df[cycle_key],
        ce_vals,
        marker='D', markersize=6,
        linestyle='-', color='blue',
        markerfacecolor='none', markeredgecolor='blue',
        label="Coulombic Efficiency (%)"
    )

    ax1.set_xlabel("Cycle Number", fontsize=20, fontweight='bold')
    ax1.set_ylabel("Specific Capacity (mAh/g)", fontsize=20, fontweight='bold')
    ax2.set_ylabel("Coulombic Efficiency (%)", fontsize=20, fontweight='bold')
    #ax1.set_title("Specific Capacity & Coulombic Efficiency vs Cycle Number of LP")
    ax1.set_ylim([0, 200])
    ax2.set_ylim([0, 110])

    # ticks pointing inward for both axes and show ticks on top/right
    ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax2.tick_params(axis='y', which='both', direction='in')

    # combined legend
    lns = lns1 + lns2
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc="lower left")

    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / f"capacity_CE_Sheet2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    plt.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    #plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
