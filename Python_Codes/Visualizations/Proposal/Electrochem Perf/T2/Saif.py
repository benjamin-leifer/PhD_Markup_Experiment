"""
Arbin Specific Capacity vs. Cycle Plotter
------------------------------------------
Requirements: pip install pandas openpyxl matplotlib

Usage:
  1. Set FILE_CONFIGS below to your .xlsx file paths and labels.
  2. Set NUM_CYCLES to the number of cycles you want to plot (or None for all).
  3. Run: python arbin_capacity_plot.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# ── Configuration ─────────────────────────────────────────────────────────────

THEORETICAL_CAP = 160.6  # mAh/g

FILE_CONFIGS = [
    {"path": r"SA-LP--40C-103123_Channel_39_Wb_1.xlsx", "label": "SA_LP_NCM_-30C"},
    {"path": r"SA-LP-NCM-110223_Channel_45_Wb_1.xlsx",     "label": "SA_LP_RT"},
    #{"path": "SADT14NCM110223_Channel_44_Wb_1.xlsx",   "label": "SADT14NCM"},
]

NUM_CYCLES = 3   # Set to None to plot all cycles
COLORS     = ["#e63946", "#1d7bc4", "#f59e0b"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_value_in_sheet(df, keywords):
    """Search a DataFrame (from Global_Info) for a keyword and return the adjacent numeric value."""
    for _, row in df.iterrows():
        for val in row:
            if isinstance(val, str) and any(k.lower() in val.lower() for k in keywords):
                # Return the next numeric value in the same row
                found = False
                for v in row:
                    if found and isinstance(v, (int, float)) and not pd.isna(v):
                        return float(v)
                    if v == val:
                        found = True
    return None


def get_norm_factor(global_df):
    """Return normalization info dict from Global_Info sheet."""
    mass = find_value_in_sheet(global_df, ["mass"])
    if mass and mass > 0:
        print(f"  → Normalization: Mass = {mass:.4f} g")
        return {"type": "mass", "val": mass}

    cap = find_value_in_sheet(global_df, ["capacity"])
    if cap and cap > 0:
        print(f"  → Normalization: Capacity = {cap*1000:.2f} mAh")
        return {"type": "cap", "val": cap}

    print("  ⚠ No normalization found in Global_Info — reporting raw mAh")
    return None


def apply_norm(cap_ah, nf):
    """Convert Ah to mAh/g using the normalization factor."""
    if nf is None:
        return cap_ah * 1000
    if nf["type"] == "mass":
        return cap_ah * 1000 / nf["val"]
    if nf["type"] == "cap":
        return (cap_ah / nf["val"]) * THEORETICAL_CAP
    return cap_ah * 1000


def parse_file(path, num_cycles):
    """Read an Arbin xlsx and return per-cycle charge/discharge specific capacity."""
    print(f"\nReading: {os.path.basename(path)}")
    xl = pd.ExcelFile(path)
    print(f"  Sheets: {xl.sheet_names}")

    global_sheet = next((s for s in xl.sheet_names if "global" in s.lower()), xl.sheet_names[0])
    data_sheet   = next((s for s in xl.sheet_names if s != global_sheet), xl.sheet_names[1])
    print(f"  Global sheet: '{global_sheet}' | Data sheet: '{data_sheet}'")

    global_df = xl.parse(global_sheet, header=None)
    nf = get_norm_factor(global_df)

    data_df = xl.parse(data_sheet)
    data_df.columns = data_df.columns.str.strip()
    print(f"  Columns: {list(data_df.columns)}")
    print(f"  Rows: {len(data_df)}")

    # Find relevant columns flexibly
    def find_col(pattern, exclude=None):
        for c in data_df.columns:
            if pd.Series(c).str.contains(pattern, case=False, regex=True).iloc[0]:
                if exclude is None or not pd.Series(c).str.contains(exclude, case=False, regex=True).iloc[0]:
                    return c
        return None

    cycle_col = find_col(r"cycle.?index")
    chg_col   = find_col(r"charge.?cap", exclude="dis")
    dis_col   = find_col(r"discharge.?cap")

    print(f"  Using → Cycle: '{cycle_col}' | Charge: '{chg_col}' | Discharge: '{dis_col}'")

    if not all([cycle_col, chg_col, dis_col]):
        raise ValueError("Could not find required columns. Check column names above.")

    # Get max capacity per cycle (cumulative Arbin convention)
    grouped = data_df.groupby(cycle_col).agg(
        chg=(chg_col, "max"),
        dis=(dis_col, "max"),
    ).reset_index()
    grouped.columns = ["cycle", "chg_ah", "dis_ah"]
    grouped = grouped.sort_values("cycle")

    if num_cycles is not None:
        grouped = grouped.head(num_cycles)

    grouped["chg_mAhg"] = grouped["chg_ah"].apply(lambda x: apply_norm(x, nf))
    grouped["dis_mAhg"] = grouped["dis_ah"].apply(lambda x: apply_norm(x, nf))

    return grouped

# ── Main ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))

for i, cfg in enumerate(FILE_CONFIGS):
    df = parse_file(cfg["path"], NUM_CYCLES)
    color = COLORS[i % len(COLORS)]
    ax.plot(df["cycle"], df["chg_mAhg"], color=color, marker="o", linewidth=2, label=f"{cfg['label']} Charge")
    ax.plot(df["cycle"], df["dis_mAhg"], color=color, marker="^", linewidth=2, linestyle="--", label=f"{cfg['label']} Discharge")

ax.set_xlabel("Cycle Number", fontsize=12)
ax.set_ylabel("Specific Capacity (mAh/g)", fontsize=12)
ax.set_title("Charge / Discharge Specific Capacity vs. Cycle", fontsize=13)
ax.legend(fontsize=10, framealpha=0.7)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("capacity_vs_cycle.png", dpi=150)
plt.show()
print("\nPlot saved as capacity_vs_cycle.png")