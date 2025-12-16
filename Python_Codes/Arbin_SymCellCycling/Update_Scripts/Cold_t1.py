# python
import os
import pandas as pd
import matplotlib.pyplot as plt

# --- user settings (adjust paths as needed) ---
base_dir = r"C:\Users\benja\Downloads\Dilute THF Data\11_25_25\-51C_Repeats"
lookup_table_path = r"C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx"
plots_dir = os.path.join(base_dir, "plots_t1")
os.makedirs(plots_dir, exist_ok=True)

# reuse existing constants from Cold_t1
TEMP_TAG = "-51C"
REQUIRE_DIS_IN_NAME = True
REF_CAP_MAH = 4.0
REF_SPEC_MAH = 160.6
CONV_AH_TO_MAHG = 1000.0 * REF_SPEC_MAH / REF_CAP_MAH  # 40150.0

# --- helpers (adapted from Cold_t1) ---
def find_discharge_files(root_dir, temp_tag=TEMP_TAG, require_discharge_name=REQUIRE_DIS_IN_NAME):
    matches = []
    temp_tag_lower = temp_tag.lower()
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if not fn.lower().endswith(".xlsx"):
                continue
            name_lower = fn.lower()
            if temp_tag_lower in name_lower and ((not require_discharge_name) or ("dis" in name_lower)):
                matches.append(os.path.join(r, fn))
    return sorted(matches)

def get_cell_code(path):
    base = os.path.basename(path)
    root = base.split("_")[0]
    return root.split("-")[-1]  # e.g. 'HU01' or 'DN06'

def get_channel_sheet_name(path):
    xls = pd.ExcelFile(path)
    if len(xls.sheet_names) < 2:
        raise ValueError(f"{path} has no channel sheet")
    return xls.sheet_names[1]

def load_discharge_curve(path):
    cell_code = get_cell_code(path)
    sheet_name = get_channel_sheet_name(path)
    df = pd.read_excel(path, sheet_name=sheet_name)
    required_cols = ["Voltage (V)", "Current (A)", "Discharge Capacity (Ah)"]
    if not all(col in df.columns for col in required_cols):
        raise KeyError(f"{path} missing one of {required_cols}")
    df_dis = df[df["Current (A)"] < 0].copy()
    if df_dis.empty:
        raise ValueError(f"{path}: no rows with Current (A) < 0")
    df_dis = df_dis.dropna(subset=["Discharge Capacity (Ah)", "Voltage (V)"])
    df_dis["Spec Discharge Capacity (mAh/g)"] = df_dis["Discharge Capacity (Ah)"] * CONV_AH_TO_MAHG
    df_dis = df_dis.sort_values("Spec Discharge Capacity (mAh/g)")
    x_spec = df_dis["Spec Discharge Capacity (mAh/g)"].values
    y_volt = df_dis["Voltage (V)"].values
    return x_spec, y_volt, cell_code

# --- lookup table ---
def load_lookup_table(path):
    df = pd.read_excel(path, dtype=str)
    # normalize column names and strip whitespace
    df.columns = [c.strip() for c in df.columns]
    if "Cell Code" not in df.columns:
        raise KeyError("Lookup table missing 'Cell Code' column")
    # ensure 'Electrolyte' exists (create empty if not)
    if "Electrolyte" not in df.columns:
        df["Electrolyte"] = ""
    df["Cell Code"] = df["Cell Code"].str.strip()
    df["Electrolyte"] = df["Electrolyte"].fillna("").astype(str).str.strip()
    return df.set_index("Cell Code")

# python
from collections import defaultdict
import os
import matplotlib.pyplot as plt

def plot_groups_by_alpha(groups, lookup, plots_dir):
    """
    groups: dict mapping full cell_code (e.g. 'AA01') -> list of file paths
    lookup: dataframe returned by load_lookup_table (indexed by two-letter code)
    plots_dir: directory to save group plots
    """
    alpha_groups = defaultdict(list)
    for cell_code, paths in groups.items():
        alpha = cell_code[:2]
        for p in paths:
            alpha_groups[alpha].append((cell_code, p))

    os.makedirs(plots_dir, exist_ok=True)

    for alpha, entries in sorted(alpha_groups.items()):
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted_any = False
        for cell_code, path in entries:
            try:
                x_spec, y_volt, _ = load_discharge_curve(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue
            plotted_any = True
            label = f"{cell_code} — {os.path.basename(path)}"
            ax.plot(x_spec, y_volt, label=label, linewidth=1.5)

        if not plotted_any:
            plt.close(fig)
            continue

        # Title uses lookup if available (lookup keyed by two-letter code)
        electrolyte = ""
        if alpha in lookup.index:
            electrolyte = lookup.loc[alpha, "Electrolyte"]
        ax.set_title(f"{alpha} group — {electrolyte}")
        ax.set_xlabel("Discharge Specific Capacity (mAh/g)")
        ax.set_ylabel("Voltage (V)")
        ax.set_ylim(0, 4.5)
        ax.set_xlim(-4, 160)
        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="xx-small", ncol=1, loc="best")

        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"{alpha}_group.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")

# In main(), after building `groups` and `lookup`, call:
# plot_groups_by_alpha(groups, lookup, plots_dir)
# python

def plot_groups_by_electrolyte(groups, plots_dir):
    """
    Group by electrolyte sets (DTF / DTFV) and plot each full cell (e.g. 'HU01') with
    its own color and marker. Saves one PNG per electrolyte group to `plots_dir`.
    """
    electrolyte_groups = {
        "DTF": ["HU", "HV", "HW", "HX"],
        "DTFV": ["IA", "IB", "IC", "ID", "IE", "IF"],
    }
    markers = ["o", "s", "D", "^", "v", "P", "X", "h", "+", "*", "<", ">"]
    os.makedirs(plots_dir, exist_ok=True)

    available_codes = sorted(groups.keys())

    for e_name, prefixes in electrolyte_groups.items():
        # find all full cell codes matching any prefix in this electrolyte group
        matched_cells = sorted([c for c in available_codes if any(c.startswith(pref) for pref in prefixes)])
        if not matched_cells:
            print(f"No cells found for electrolyte group {e_name}, skipping.")
            continue

        # assign one distinct color per full cell using a categorical colormap
        cmap = plt.get_cmap("tab20")
        n_cells = len(matched_cells)
        colors = [cmap(i % cmap.N) for i in range(n_cells)]

        fig, ax = plt.subplots(figsize=(8, 6))
        plotted_any = False

        for i, cell_code in enumerate(matched_cells):
            paths = groups.get(cell_code, [])
            x_all = []
            y_all = []

            for p in sorted(paths):
                try:
                    x_spec, y_volt, _ = load_discharge_curve(p)
                except Exception as exc:
                    print(f"Skipping {p}: {exc}")
                    continue
                # extend with arrays/lists safely
                x_all.extend(x_spec.tolist() if hasattr(x_spec, "tolist") else list(x_spec))
                y_all.extend(y_volt.tolist() if hasattr(y_volt, "tolist") else list(y_volt))

            if not x_all:
                continue

            color = colors[i]
            marker = markers[i % len(markers)]

            ax.scatter(
                x_all,
                y_all,
                label=cell_code,
                marker=marker,
                s=32,
                alpha=0.9,
                edgecolor="k",
                linewidth=0.3,
                color=color,
            )
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            print(f"No valid data in electrolyte group {e_name}, skipping plot.")
            continue

        ax.set_title(f"{e_name} group — per-cell colors & markers")
        ax.set_xlabel("Discharge Specific Capacity (mAh/g)")
        ax.set_ylabel("Voltage (V)")
        ax.set_ylim(0, 4.5)
        ax.set_xlim(-4, 160)
        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="xx-small", ncol=1, loc="best")

        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"{e_name}_group_per_cell_color_scatter.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")

def plot_groups_by_electrolyte_combined(groups, plots_dir):
    """
    Combine all RateTest files per cell (e.g., IO01),
    then plot grouped by electrolyte type (DTF / DTFV).

    Colors are consistent per electrolyte family,
    markers vary per cell.
    """
    electrolyte_groups = {
        "DTF": ["HU", "HV", "HW", "HX"],
        "DTFV": ["IA", "IB", "IC", "ID", "IE", "IF"],
    }

    # Define consistent colors for each electrolyte family
    electrolyte_colors = {
        "DTF": "#1f77b4",   # blue
        "DTFV": "#ff7f0e",  # orange
    }

    # Markers for different cells
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]

    os.makedirs(plots_dir, exist_ok=True)

    # Build combined discharge data per cell
    def combine_statistics_for_cell(paths):
        dfs = []
        for path in paths:
            try:
                xls = pd.ExcelFile(path, engine="openpyxl")
                sheet_name = [s for s in xls.sheet_names if "StatisticsByCycle" in s][0]
                df = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl")
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue

            if not {"Cycle Index", "Discharge Capacity (Ah)"}.issubset(df.columns):
                continue

            dfs.append(df[["Cycle Index", "Discharge Capacity (Ah)"]])

        if not dfs:
            return None

        df_all = pd.concat(dfs, ignore_index=True)
        df_grouped = df_all.groupby("Cycle Index", as_index=False).max()
        df_grouped["Specific Discharge (mAh/g)"] = df_grouped["Discharge Capacity (Ah)"] * 40150
        return df_grouped

    # Now plot by electrolyte family
    for e_name, prefixes in electrolyte_groups.items():
        matched_cells = sorted([c for c in groups.keys() if any(c.startswith(pref) for pref in prefixes)])
        if not matched_cells:
            print(f"No cells found for electrolyte {e_name}, skipping.")
            continue

        color = electrolyte_colors[e_name]
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted_any = False

        for i, cell_code in enumerate(matched_cells):
            df_grouped = combine_statistics_for_cell(groups[cell_code])
            if df_grouped is None or df_grouped.empty:
                continue

            marker = markers[i % len(markers)]
            ax.plot(
                df_grouped["Cycle Index"],
                df_grouped["Specific Discharge (mAh/g)"],
                label=cell_code,
                color=color,
                marker=marker,
                linewidth=2,
                markersize=6,
            )
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        ax.set_title(f"{e_name} Group — Combined Rate Test (normalized)")
        ax.set_xlabel("Cycle Number")
        ax.set_ylabel("Specific Discharge Capacity (mAh/g)")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(fontsize="small", ncol=2, loc="best")

        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"{e_name}_RateTest_Group_Combined.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")


# --- main per-cell plotting ---
def main():
    files = find_discharge_files(base_dir)
    if not files:
        print(f"No matching .xlsx files found under: {base_dir}")
        return

    lookup = load_lookup_table(lookup_table_path)

    # group files by exact cell code (e.g. 'HU01')
    groups = {}
    for p in files:
        code = get_cell_code(p)
        groups.setdefault(code, []).append(p)

    for cell_code, paths in groups.items():
        # look up using prefix as in Scratch_t7 (first two chars)
        lookup_key = cell_code[:2]
        electrolyte = ""
        if lookup_key in lookup.index:
            electrolyte = lookup.loc[lookup_key, "Electrolyte"]
        title_label = f"{cell_code}: - {electrolyte}"

        fig, ax = plt.subplots(figsize=(8, 6))
        plotted_any = False
        for p in paths:
            try:
                x_spec, y_volt, code = load_discharge_curve(p)
            except Exception as e:
                print(f"Skipping {p}: {e}")
                continue
            plotted_any = True
            legend_label = os.path.basename(p)
            ax.plot(x_spec, y_volt, label=legend_label, linewidth=2)

        if not plotted_any:
            plt.close(fig)
            continue

        ax.set_xlabel("Discharge Specific Capacity (mAh/g)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(title_label)
        ax.set_ylim(0, 4.5)
        ax.set_xlim(-4, 160)
        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="xx-small", ncol=1, loc="best")
        plt.tight_layout()

        out_name = f"{cell_code}_discharge.png"
        out_path = os.path.join(plots_dir, out_name)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")

    plot_groups_by_alpha(groups, lookup, plots_dir)
    plot_groups_by_electrolyte_combined(groups, plots_dir)

if __name__ == "__main__":
    main()