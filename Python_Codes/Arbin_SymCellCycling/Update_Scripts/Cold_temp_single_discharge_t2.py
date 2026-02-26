import os
import pandas as pd
import matplotlib.pyplot as plt
# add near the top with imports
from electrolyte_style import clean_elyte_str, style_for_electrolyte, pretty_label, save_curve_legend_png


# =========================
# USER INPUT
# =========================

CELL_CODES = [
    "JC05",
    "JD04",
    "JE05",
    # "HU03",
]
CELL_CODES = [
    "IV03",
    "IW04",
    "IX03",
    "IY01"
]
CELL_CODES = [
    "IP04",
    "IQ02",
    "IR03",
    "IS03",
    "IT01",
    "IU03",
]

base_dir = r"C:\Users\benja\Downloads\Dilute THF Data\11_25_25\-51C_Repeats"
old_directory = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\-51C_discharges"

lookup_table_path = r"C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx"
plots_dir = os.path.join(base_dir, "plots_-51C_combined_DTFV")
os.makedirs(plots_dir, exist_ok=True)


# Capacity normalization (same as your grid scripts)
REF_CAP_MAH = 4.0
REF_SPEC_MAH_G = 160.6
CONV_AH_TO_MAHG = 1000.0 * REF_SPEC_MAH_G / REF_CAP_MAH

# === Visual scaling (set >1.0 to enlarge fonts/markers) ===
FONT_SCALE = 3.0  # change this to taste (1.0 = original sizes)
FIG_SCALE = 2.0

# Derived plotting sizes
FIG_W = 7.5 * FIG_SCALE
FIG_H = 6.0 * FIG_SCALE
LINE_WIDTH = 2.0 * FONT_SCALE
MARKER_SIZE = 6.0 * FONT_SCALE
XLABEL_FS = 12 * FONT_SCALE
YLABEL_FS = 12 * FONT_SCALE
TICK_FS = 10 * FONT_SCALE
LEGEND_FS = 10 * FONT_SCALE

import matplotlib as mpl
# Ensure tick appearance is explicit and large enough to be visible
mpl.rcParams.update({
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 6 * FONT_SCALE,
    'xtick.minor.size': 3 * FONT_SCALE,
    'ytick.major.size': 6 * FONT_SCALE,
    'ytick.minor.size': 3 * FONT_SCALE,
    'xtick.major.width': 1.0 * FONT_SCALE,
    'ytick.major.width': 1.0 * FONT_SCALE,
    'xtick.color': 'black',
    'ytick.color': 'black',
})


# =========================
# LOOKUP
# =========================

def load_lookup_table(path: str) -> dict:
    df = pd.read_excel(path)
    key_col = df.columns[0]
    df[key_col] = df[key_col].astype(str).str.upper()

    lookup = {}
    for _, row in df.iterrows():
        key = str(row[key_col]).strip().upper()
        lookup[key] = {c: row[c] for c in df.columns[1:]}
    return lookup


def get_electrolyte(cell_code: str, lookup: dict) -> str:
    code = cell_code.upper()

    if code in lookup and pd.notna(lookup[code].get("Electrolyte")):
        return str(lookup[code]["Electrolyte"])

    alpha = "".join([c for c in code if c.isalpha()])
    if alpha in lookup and pd.notna(lookup[alpha].get("Electrolyte")):
        return str(lookup[alpha]["Electrolyte"])

    return code


# =========================
# FILE SEARCH
# =========================

def get_cell_code_from_filename(path: str) -> str:
    base = os.path.basename(path)
    root = base.split("_")[0]
    return root.split("-")[-1].upper()


def get_channel_sheet_name(path: str) -> str:
    xls = pd.ExcelFile(path)
    return xls.sheet_names[1]


def find_51C_discharge_files():
    search_dirs = [d for d in [base_dir, old_directory] if os.path.isdir(d)]
    out = []
    seen = set()

    for root_dir in search_dirs:
        for r, _, files in os.walk(root_dir):
            parent_lower = os.path.basename(r).lower()

            for fn in files:
                fn_lower = fn.lower()
                if not (fn_lower.endswith(".xlsx") and "dis" in fn_lower):
                    continue

                if "-51c" not in fn_lower and "-51c" not in parent_lower:
                    continue

                p = os.path.join(r, fn)
                norm = os.path.normcase(os.path.normpath(p))
                if norm in seen:
                    continue

                seen.add(norm)
                out.append(p)

    return out


# =========================
# DISCHARGE 1 EXTRACTION
# =========================

def slice_first_discharge(df_dis):
    # Use cycle column if present
    for cyc_col in ["Cycle Index", "Cycle", "Cycle_Index", "CycleIndex"]:
        if cyc_col in df_dis.columns:
            first_cycle = df_dis[cyc_col].dropna().min()
            return df_dis[df_dis[cyc_col] == first_cycle]

    # Otherwise detect reset
    cap = df_dis["Discharge Capacity (Ah)"].to_numpy()
    if len(cap) < 3:
        return df_dis

    max_cap = cap.max()
    thresh = max(0.005 * max_cap, 5e-5)
    start_cap = cap[0]

    for i in range(1, len(cap)):
        if (cap[i-1] - cap[i]) > thresh and cap[i] <= start_cap + 1e-6:
            return df_dis.iloc[:i]

    return df_dis


def load_discharge1(path):
    sheet = get_channel_sheet_name(path)
    df = pd.read_excel(path, sheet_name=sheet)

    df_dis = df[df["Current (A)"] < 0].copy()
    df_dis = df_dis.dropna(subset=["Voltage (V)", "Discharge Capacity (Ah)"])
    df_dis = slice_first_discharge(df_dis)

    Q = df_dis["Discharge Capacity (Ah)"].to_numpy() * CONV_AH_TO_MAHG
    V = df_dis["Voltage (V)"].to_numpy()

    order = Q.argsort()
    return Q[order], V[order]


# =========================
# MAIN
# =========================

def main():
    lookup = load_lookup_table(lookup_table_path)
    all_files = find_51C_discharge_files()

    if not all_files:
        print("No -51C files found.")
        return

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    plotted_any = False

    legend_dir = r"C:\Users\benja\Downloads\Final Countdown\Proposal Slide Figures - Cycling Plots\Legend"
    os.makedirs(legend_dir, exist_ok=True)

    for cell_code in CELL_CODES:
        ely_raw = get_electrolyte(cell_code, lookup)
        ely = clean_elyte_str(ely_raw)

        cell_files = [
            p for p in all_files
            if get_cell_code_from_filename(p) == cell_code.upper()
        ]

        for p in sorted(cell_files):
            try:
                Q, V = load_discharge1(p)
                # plot with markers (scaled) and a reasonable markevery to avoid overplotting
                markevery = max(len(Q) // 30, 1) if len(Q) > 1 else 1
                markevery = max(len(Q) // 30, 1) if len(Q) > 1 else 1
                st = style_for_electrolyte(
                    ely,
                    lw_base=LINE_WIDTH,
                    lw=LINE_WIDTH,
                    markevery=markevery,
                    ms_scale=FONT_SCALE,  # scale markers with your figure scaling
                    mew_scale=FONT_SCALE,
                )

                ax.plot(
                    Q, V,
                    label=pretty_label(ely),  # or show_details=True
                    alpha=0.95,
                    **st,
                )
                plotted_any = True
            except Exception as e:
                print(f"Skipping {p}: {e}")

    if not plotted_any:
        print("Nothing plotted.")
        plt.close(fig)
        return

    # No title (as requested)
    ax.set_xlabel("Specific Capacity (mAh/g)", fontsize=XLABEL_FS)
    ax.set_ylabel("Voltage (V)", fontsize=YLABEL_FS)
    ax.set_ylim(0, 4.5)
    ax.set_xlim(0, 100)
    #ax.grid(True, alpha=0.25)


    # Ensure spines are visible and scaled to FONT_SCALE so ticks have a reference
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0 * FONT_SCALE)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_zorder(3)

    # Enable minor ticks and make ticks point inward on all sides, with explicit lengths
    ax.minorticks_on()

    # Explicitly ensure ticks are drawn on both sides and visible
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.tick_params(
        which='both',            # major and minor
        axis='both',             # x and y
        direction='in',          # point ticks inward
        top=True, right=True,    # show ticks on top and right
        length=6 * FONT_SCALE,   # major tick length
        width=1.0 * FONT_SCALE,  # major tick width
        labelsize=TICK_FS,
        colors='black',
    )
    # Minor ticks shorter
    ax.tick_params(
        which='minor',
        axis='both',
        length=3 * FONT_SCALE,
        width=max(0.8 * FONT_SCALE, 0.5),
        colors='black'
    )

    # Make sure ticklines are above the axes background but below plotted lines
    for t in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        t.set_zorder(4)
        t.set_color('black')


    # Remove duplicate legend entries and place legend below the plot with 2 columns
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    leg = ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=LEGEND_FS,
        ncol=2,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
    )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    legend_png = os.path.join(legend_dir, "SelectedCells_-51C_Discharge1_LEGEND.png")
    save_curve_legend_png(
        list(unique.values()),
        list(unique.keys()),
        legend_png,
        ncol=2,
        fontsize=int(LEGEND_FS),
    )
    # enlarge legend handles to match scaled plot elements
    # Some Matplotlib versions expose legend handles as 'legendHandles' and others as 'legend_handles'.
    legend_handles = getattr(leg, 'legendHandles', None) or getattr(leg, 'legend_handles', None)
    if legend_handles is not None:
        for lh in legend_handles:
            try:
                lh.set_linewidth(max(1.0, LINE_WIDTH))
            except Exception:
                pass
            try:
                lh.set_markersize(MARKER_SIZE)
            except Exception:
                pass

    out_path = os.path.join(plots_dir, "SelectedCells_-51C_Discharge1.png")
    # Use tight bbox when saving to avoid clipping ticks/legend
    fig.tight_layout()
    # Save with explicit facecolor/edgecolor to avoid unexpected transparency or styling
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.12, facecolor='white', edgecolor='white')
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
