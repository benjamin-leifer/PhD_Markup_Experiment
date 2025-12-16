import os
import re
import math
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import cycle

# --- USER SETTINGS ---
base_dir = r"C:\Users\benja\Downloads\Dilute THF Data\11_25_25\DPE Room Temperature Data"  # top-level directory to walk through
save_dir = os.path.join(base_dir, "RateTest_Plots_norm")
os.makedirs(save_dir, exist_ok=True)

# --- LOOKUP TABLE (cold_t2-style labeling) ---
lookup_table_path = r"C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx"

# --- NORMALIZATION CONSTANTS (Cold_t1/Cold_t2-style) ---
# 4 mAh corresponds to 160.6 mAh/g
REF_CAP_MAH = 4.0
REF_SPEC_MAH = 160.6
CONV_AH_TO_MAHG = 1000.0 * REF_SPEC_MAH / REF_CAP_MAH  # 40150.0

# Skip absurdly large "RateTest" workbooks that aren't Arbin exports
MAX_FILE_MB = 30


# --- HELPERS: file finding / codes ---
def find_rate_files(root_dir):
    """Find all Excel files containing 'RateTest' in their filenames."""
    matches = []
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".xlsx") and "ratetest" in fn.lower():
                path = os.path.join(r, fn)
                matches.append(path)
    return sorted(matches)


def get_cell_code_from_basename(basename: str) -> str:
    """
    Try to extract cell code like IO01 from BL-LL-IO01_RateTest_...xlsx.
    Fallback to AA or AA## regex if needed.
    """
    # BL-LL-IO01_RateTest_...xlsx -> IO01
    root = basename.split("_")[0]          # e.g. BL-LL-IO01
    parts = root.split("-")
    if len(parts) >= 3:
        code = parts[-1]
    else:
        # fallback: AA or AA## pattern
        m = re.search(r"[A-Z]{2}\d{0,2}", basename)
        code = m.group(0) if m else "Unknown"
    return code


def get_cell_code(path: str) -> str:
    return get_cell_code_from_basename(os.path.basename(path))


# --- LOOKUP TABLE + LABELING (from cold_t2) ---
def load_lookup_table(path):
    """
    Load Spring 2025 Cell List and index by 'Cell Code'.
    Ensures 'Electrolyte', 'Cathode', and 'Anode' columns exist.
    """
    df = pd.read_excel(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "Cell Code" not in df.columns:
        raise KeyError("Lookup table missing 'Cell Code' column")
    if "Electrolyte" not in df.columns:
        df["Electrolyte"] = ""
    if "Cathode" not in df.columns:
        df["Cathode"] = ""
    if "Anode" not in df.columns:
        df["Anode"] = ""
    df["Cell Code"] = df["Cell Code"].str.strip()
    df["Electrolyte"] = df["Electrolyte"].fillna("").astype(str).str.strip()
    df["Cathode"] = df["Cathode"].fillna("").astype(str).str.strip()
    df["Anode"] = df["Anode"].fillna("").astype(str).str.strip()
    return df.set_index("Cell Code")


def get_display_label(cell_code, lookup):
    """
    Build 'Electrolyte-SampleNumber' for a given cell_code.
    Looks up Electrolyte by alpha prefix, and extracts sample number from suffix.
    (Same logic intent as in cold_t2.)
    """
    alpha = cell_code[:2]
    row = None

    if cell_code in lookup.index:
        row = lookup.loc[cell_code]
    elif alpha in lookup.index:
        row = lookup.loc[alpha]

    electrolyte = ""
    if row is not None:
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        electrolyte = str(row.get("Electrolyte", "")).strip()

    suffix = cell_code[2:]
    sample_num = suffix.lstrip("0") or suffix or ""

    if electrolyte and sample_num:
        return f"{electrolyte}-{sample_num}"
    elif electrolyte:
        return electrolyte
    else:
        return cell_code


def get_cell_chemistry(code, lookup):
    """
    Return '{Cathode}|{Anode}' using either the full cell code or the alpha (first two letters).
    If neither is present or fields are empty, return ''.
    """
    candidates = [code]
    if len(code) >= 2:
        candidates.append(code[:2])

    for key in candidates:
        if key in lookup.index:
            row = lookup.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            cath = str(row.get("Cathode", "")).strip()
            an = str(row.get("Anode", "")).strip()
            if cath or an:
                if cath and an:
                    return f"{cath}|{an}"
                else:
                    return cath or an
    return ""


# --- DATA LOADING ---
def load_statistics_sheet(path):
    """
    Load the 'StatisticsByCycle...' sheet from an Excel file.
    - Skips files > MAX_FILE_MB.
    - Prefers sheet names containing 'StatisticsByCycle'.
    """
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise ValueError(f"{path} is {size_mb:.1f} MB (> {MAX_FILE_MB} MB), skipping")

    # Just reading sheet names is cheap compared to full parse
    xls = pd.ExcelFile(path, engine="openpyxl")
    stats_sheets = [s for s in xls.sheet_names if "StatisticsByCycle" in s]

    if stats_sheets:
        sheet_name = stats_sheets[0]
    else:
        # fallback: use 2nd sheet if no explicit StatisticsByCycle
        if len(xls.sheet_names) < 2:
            raise ValueError(f"{path} has fewer than 2 sheets and no 'StatisticsByCycle' sheet")
        sheet_name = xls.sheet_names[1]

    print(f"  Reading '{sheet_name}' from {os.path.basename(path)} (size {size_mb:.1f} MB)")
    df = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl")

    # Check expected columns
    required_cols = ["Cycle Index", "Charge Capacity (Ah)", "Discharge Capacity (Ah)"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"{path} missing expected column: {col}")
    return df


def combine_statistics_for_cell(paths):
    """
    For a given cell (e.g. IO01), combine data from *all* RateTest files:
    - Load each StatisticsByCycle sheet
    - Concatenate
    - Group by Cycle Index and take max capacities
    Returns a grouped DataFrame or None if nothing usable.
    """
    dfs = []
    for path in paths:
        print(f"Processing file for combination: {path}")
        try:
            df = load_statistics_sheet(path)
        except Exception as e:
            print(f"  Skipping {path}: {e}")
            continue

        # Keep only the required columns (drop weird extras to simplify combine)
        sub = df[["Cycle Index", "Charge Capacity (Ah)", "Discharge Capacity (Ah)"]].copy()
        dfs.append(sub)

    if not dfs:
        print("  No valid StatisticsByCycle data for this cell.")
        return None

    df_all = pd.concat(dfs, ignore_index=True)

    # Group by cycle index in case there is overlap between files
    df_grouped = df_all.groupby("Cycle Index", as_index=False).max()
    df_grouped = df_grouped.sort_values("Cycle Index")
    return df_grouped


# --- GROUP BUILDERS & STYLE MAPS ---
def build_alpha_groups(groups):
    """
    Build mapping alpha -> list of (cell_code, paths).
    groups is cell_code -> [paths].
    """
    alpha_groups = defaultdict(list)
    for cell_code, paths in groups.items():
        alpha = cell_code[:2]
        alpha_groups[alpha].append((cell_code, paths))
    return alpha_groups


def assign_alpha_colors(alpha_groups):
    """
    Assign each alpha a consistent color.
    """
    base_colors = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    ]
    color_cycle = cycle(base_colors)
    alpha_to_color = {}
    for alpha in sorted(alpha_groups.keys()):
        alpha_to_color[alpha] = next(color_cycle)
    return alpha_to_color


def assign_cell_markers(groups):
    """
    Assign each full cell code a distinct marker, cycling through a list.
    """
    base_markers = ["o", "s", "D", "^", "v", "P", "X", "h", ">", "<"]
    marker_cycle = cycle(base_markers)
    cell_to_marker = {}
    for cell_code in sorted(groups.keys()):
        cell_to_marker[cell_code] = next(marker_cycle)
    return cell_to_marker


# --- PLOTTING ---
def plot_per_cell(groups, lookup):
    """
    For each full cell code (e.g. IO01), combine all its RateTest files
    and plot a single normalized charge/discharge specific capacity vs cycle scatter.
    Title uses chemistry + cold_t2-style display label.
    """
    for cell_code, paths in groups.items():
        print(f"\n=== Cell {cell_code} ===")

        df_grouped = combine_statistics_for_cell(paths)
        if df_grouped is None or df_grouped.empty:
            print(f"  No combined data for {cell_code}, skipping plot.")
            continue

        display_label = get_display_label(cell_code, lookup)
        chem = get_cell_chemistry(cell_code, lookup)

        fig, ax = plt.subplots(figsize=(7, 5))

        # Normalize to mAh/g
        spec_charge = df_grouped["Charge Capacity (Ah)"] * CONV_AH_TO_MAHG
        spec_discharge = df_grouped["Discharge Capacity (Ah)"] * CONV_AH_TO_MAHG

        ax.scatter(
            df_grouped["Cycle Index"],
            spec_charge,
            label="Combined — Qchg",
            marker="o",
            s=30,
            alpha=0.9,
            edgecolor="k",
            linewidth=0.3,
        )
        ax.scatter(
            df_grouped["Cycle Index"],
            spec_discharge,
            label="Combined — Qdis",
            marker="s",
            s=30,
            alpha=0.9,
            edgecolor="k",
            linewidth=0.3,
        )

        ax.set_xlabel("Cycle Number")
        ax.set_ylabel("Specific Capacity (mAh/g)")

        title_parts = []
        if chem:
            title_parts.append(chem)
        if display_label:
            title_parts.append(display_label)
        title_parts.append("Rate Test Capacity (normalized, combined)")
        ax.set_title(" — ".join(title_parts))

        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="x-small", loc="best")

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{cell_code}_RateTest_SpecCapacity_combined.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved per-cell combined plot: {out_path}")


def plot_alpha_group_on_axis(ax, alpha, entries, lookup, color, cell_to_marker):
    """
    Plot a single alpha group (e.g. IO) on the provided axis.
    Returns (handles, labels) for legend construction.
    """
    plotted_any = False
    alpha_handles = []
    alpha_labels = []

    # Get group electrolyte + chemistry
    electrolyte = ""
    if alpha in lookup.index:
        row = lookup.loc[alpha]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        electrolyte = str(row.get("Electrolyte", "")).strip()
    chem = get_cell_chemistry(alpha, lookup)

    for cell_code, paths in entries:
        print(f"  Combining data for cell {cell_code} in group {alpha}")
        df_grouped = combine_statistics_for_cell(paths)
        if df_grouped is None or df_grouped.empty:
            print(f"    No combined data for {cell_code}, skipping.")
            continue

        spec_discharge = df_grouped["Discharge Capacity (Ah)"] * CONV_AH_TO_MAHG
        label = get_display_label(cell_code, lookup)
        marker = cell_to_marker.get(cell_code, "o")

        sc = ax.scatter(
            df_grouped["Cycle Index"],
            spec_discharge,
            label=label,
            marker=marker,
            s=25,
            alpha=0.85,
            edgecolor="k",
            linewidth=0.2,
            c=color,
        )
        alpha_handles.append(sc)
        alpha_labels.append(label)
        plotted_any = True

    if not plotted_any:
        return [], []

    title_parts = []
    if chem:
        title_parts.append(chem)
    if electrolyte:
        title_parts.append(electrolyte)
    else:
        title_parts.append(f"{alpha} group")
    #title_parts.append("Rate Test Discharge (normalized, combined)")
    ax.set_title(" — ".join(title_parts))

    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Specific Discharge Capacity (mAh/g)")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.tick_params(direction="in", top=True, right=True)

    return alpha_handles, alpha_labels

# python
import itertools
import os

def plot_alpha_pairs_side_by_side(alpha_groups, lookup, alpha_to_color, cell_to_marker, save_dir,
                                  xlim=None, ylim=None, figsize=(12, 5), pad_frac=0.03):
    """
    Loop through all 2-combinations of alphas and make a side-by-side figure for each pair.
    - Reuses plot_alpha_group_on_axis for each subplot so color/marker mapping is identical.
    - If xlim/ylim are None, compute combined limits from plotted data and apply to both axes.
    - Saves one PNG per alpha pair to `save_dir`.
    """
    os.makedirs(save_dir, exist_ok=True)
    alphas = sorted(alpha_groups.keys())
    if len(alphas) < 2:
        print("Need at least two alphas to make pairs.")
        return

    for a1, a2 in itertools.combinations(alphas, 2):
        entries1 = alpha_groups.get(a1, [])
        entries2 = alpha_groups.get(a2, [])

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        ax1, ax2 = axes if isinstance(axes, (list, tuple)) else (axes, )

        # Plot each alpha on its axis using existing helper
        color1 = alpha_to_color.get(a1, "k")
        color2 = alpha_to_color.get(a2, "k")

        handles1, labels1 = plot_alpha_group_on_axis(ax1, a1, entries1, lookup, color1, cell_to_marker)
        handles2, labels2 = plot_alpha_group_on_axis(ax2, a2, entries2, lookup, color2, cell_to_marker)

        # If nothing plotted in both axes, skip
        if (not handles1) and (not handles2):
            plt.close(fig)
            print(f"Skipping pair {a1}-{a2}: no data.")
            continue

        # Determine common x/y limits if not explicitly provided
        if xlim is None or ylim is None:
            # Use dataLim from axes to determine plotted ranges
            x0s = [ax.dataLim.x0 for ax in (ax1, ax2)]
            x1s = [ax.dataLim.x1 for ax in (ax1, ax2)]
            y0s = [ax.dataLim.y0 for ax in (ax1, ax2)]
            y1s = [ax.dataLim.y1 for ax in (ax1, ax2)]

            xmin, xmax = min(x0s), max(x1s)
            ymin, ymax = min(y0s), max(y1s)

            # if dataLim is degenerate (single point) guard against zero span
            xspan = max(xmax - xmin, 1.0)
            yspan = max(ymax - ymin, 1.0)

            padx = xspan * pad_frac
            pady = yspan * pad_frac

            common_xlim = (xmin - padx, xmax + padx)
            common_ylim = (max(0.0, ymin - pady), ymax + pady)  # keep lower bound >= 0 for capacities
        else:
            common_xlim = tuple(xlim)
            common_ylim = tuple(ylim)

        ax1.set_xlim(common_xlim)
        ax2.set_xlim(common_xlim)
        ax1.set_ylim(common_ylim)
        ax2.set_ylim(common_ylim)

        # Build a single combined legend (unique labels)
        global_handles = {}
        for ax in (ax1, ax2):
            hs, ls = ax.get_legend_handles_labels()
            for h, l in zip(hs, ls):
                if l not in global_handles:
                    global_handles[l] = h

        if global_handles:
            # place legend above the plots
            fig.legend(
                list(global_handles.values()),
                list(global_handles.keys()),
                loc="upper center",
                fontsize="xx-small",
                ncol=min(6, len(global_handles)),
                bbox_to_anchor=(0.5, 0.99),
            )

        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
        out_path = os.path.join(save_dir, f"{a1}_{a2}_alpha_pair.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved alpha-pair plot: {out_path}")

def plot_groups_by_alpha(alpha_groups, lookup, alpha_to_color, cell_to_marker):
    """
    One figure per alpha (IO, AA, etc.).
    Each cell in that alpha has a different marker; alpha has its own color.
    """
    for alpha, entries in sorted(alpha_groups.items()):
        print(f"\n=== Alpha group {alpha} ===")

        fig, ax = plt.subplots(figsize=(7, 5))
        color = alpha_to_color.get(alpha, "k")

        handles, labels = plot_alpha_group_on_axis(
            ax, alpha, entries, lookup, color, cell_to_marker
        )

        if handles:
            ax.legend(
                handles,
                labels,
                fontsize="xx-small",
                ncol=1,
                loc="best",
            )

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{alpha}_group_RateTest_SpecCapacity_combined.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved alpha-group combined plot: {out_path}")


def plot_all_alphas_side_by_side(alpha_groups, lookup, alpha_to_color, cell_to_marker, max_cols=3):
    """
    Make a single figure with subplots for each alpha, arranged side by side (in a grid).
    Alpha controls color; cell controls marker. Shared y-axis for easier comparison.
    """
    n_alpha = len(alpha_groups)
    if n_alpha == 0:
        print("No alpha groups to plot side-by-side.")
        return

    ncols = min(max_cols, n_alpha)
    nrows = int(math.ceil(n_alpha / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4 * nrows),
        sharey=True,
    )

    # axes can be scalar, 1D, or 2D; normalize to flat list
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    elif nrows == 1 or ncols == 1:
        axes_flat = list(axes.flatten())
    else:
        axes_flat = [ax for row in axes for ax in row]

    global_handles = {}
    for ax, (alpha, entries) in zip(axes_flat, sorted(alpha_groups.items())):
        color = alpha_to_color.get(alpha, "k")
        handles, labels = plot_alpha_group_on_axis(
            ax, alpha, entries, lookup, color, cell_to_marker
        )
        for h, lab in zip(handles, labels):
            if lab not in global_handles:
                global_handles[lab] = h

    # Hide any unused axes if n_alpha < nrows * ncols
    for ax in axes_flat[n_alpha:]:
        ax.set_visible(False)

    if global_handles:
        fig.legend(
            list(global_handles.values()),
            list(global_handles.keys()),
            loc="upper center",
            fontsize="xx-small",
            ncol=min(4, len(global_handles)),
            bbox_to_anchor=(0.5, 0.99),
        )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    out_path = os.path.join(save_dir, "AllAlphaGroups_RateTest_SpecCapacity_side_by_side.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved side-by-side alpha-group plot: {out_path}")


# --- MAIN ---
def main():
    rate_files = find_rate_files(base_dir)
    if not rate_files:
        print(f"No RateTest Excel files found under {base_dir}")
        return

    print(f"Found {len(rate_files)} RateTest files under {base_dir}")

    # Load lookup for cold_t2-style labels and chemistry
    try:
        lookup = load_lookup_table(lookup_table_path)
    except Exception as e:
        print(f"Failed to load lookup table '{lookup_table_path}': {e}")
        return

    # group by full cell code (IO01, AA01, AB03, etc.)
    groups = {}
    for p in rate_files:
        code = get_cell_code(p)
        groups.setdefault(code, []).append(p)

    # Build alpha groups and style maps
    alpha_groups = build_alpha_groups(groups)
    alpha_to_color = assign_alpha_colors(alpha_groups)
    cell_to_marker = assign_cell_markers(groups)

    # Per-cell combined plots (normalized) with cold_t2-style display label & chemistry in title
    plot_per_cell(groups, lookup)

    # One figure per alpha, plus side-by-side overview
    plot_groups_by_alpha(alpha_groups, lookup, alpha_to_color, cell_to_marker)
    plot_all_alphas_side_by_side(alpha_groups, lookup, alpha_to_color, cell_to_marker)


if __name__ == "__main__":
    main()
