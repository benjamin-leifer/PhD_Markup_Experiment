import os
import re
import math
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import cycle

# --- USER SETTINGS ---
base_dir = r"C:\Users\benja\Downloads\Dilute THF Data\KRI 2026_01_06"  # top-level directory to walk through
save_dir = os.path.join(base_dir, "RateTest_Plots_norm6")
os.makedirs(save_dir, exist_ok=True)

# --- LOOKUP TABLE (cold_t2-style labeling) ---
lookup_table_path = r"C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx"

REF_CAP_MAH = 4.0
REF_SPEC_MAH = 160.6
CONV_AH_TO_MAHG = 1000.0 * REF_SPEC_MAH / REF_CAP_MAH  # 40150.0

# Representative cycles for rate test (Table 3 style)
SUMMARY_CYCLE_MAP = {
    "C/10": 3,
    "C/8": 6,
    "C/4": 9,
    "C/2": 12,
    "1C": 15,
    "2C": 18,
}


# --- FILE DISCOVERY / CELL CODE HELPERS ---

def find_rate_files(base_dir):
    """
    Recursively find Excel files with 'RateTest' in the name under base_dir.
    """
    rate_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".xlsx", ".xlsm", ".xls")) and "ratetest" in f.lower():
                rate_files.append(os.path.join(root, f))
    return sorted(rate_files)


def extract_digits(s):
    """Extract trailing digits from a string, or return None if none."""
    m = re.search(r"(\d+)$", s)
    return int(m.group(1)) if m else None


def get_cell_code_from_basename(basename: str) -> str:
    """
    Infer cell code (AA, AA01, IO01, etc.) from the RateTest filename.
    We assume the Spring 2025 naming pattern is present somewhere in the basename.
    """
    name, _ = os.path.splitext(basename)

    # If something like BL-LL-IO01_... is present, take IO01
    m = re.search(r"BL-LL-([A-Z]{2}\d{0,2})", name)
    if m:
        return m.group(1)

    # Fallbacks:
    parts = re.split(r"[_\s\-]+", name)
    for p in parts:
        if re.match(r"^[A-Z]{2}\d{0,2}$", p):
            return p

    # If we still don't have it, take first AA or AA## pattern
    m = re.search(r"[A-Z]{2}\d{0,2}", basename)
    if m:
        return m.group(0)

    return "Unknown"


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


def get_electrolyte_name(cell_code, lookup):
    """Return the electrolyte label for a given cell code using the lookup table.
    Falls back from full cell code to alpha (first two letters)."""
    alpha = cell_code[:2]
    row = None

    if cell_code in lookup.index:
        row = lookup.loc[cell_code]
    elif alpha in lookup.index:
        row = lookup.loc[alpha]

    if row is None:
        return ""
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return str(row.get("Electrolyte", "")).strip()


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
            cathode = str(row.get("Cathode", "")).strip()
            anode = str(row.get("Anode", "")).strip()
            if cathode or anode:
                if cathode and anode:
                    return f"{cathode}|{anode}"
                return cathode or anode
    return ""


def get_display_label(cell_code, lookup):
    """
    Build a display label like 'DT14-1 (IO)' using cold_t2-style logic:
      - electrolyte from lookup
      - numeric suffix as sample number
      - alpha prefix from cell code
    """
    electrolyte = get_electrolyte_name(cell_code, lookup)
    alpha = cell_code[:2]
    sample_number = extract_digits(cell_code)
    if electrolyte:
        if sample_number is not None:
            return f"{electrolyte}-{sample_number} ({alpha})"
        else:
            return f"{electrolyte} ({alpha})"
    else:
        if sample_number is not None:
            return f"{cell_code} ({alpha})"
        else:
            return cell_code


# --- STATISTICS-BY-CYCLE HANDLING ---

def load_statistics_sheet(path):
    """
    Load the 'StatisticsByCycle' (or similar) sheet from a RateTest Excel file.
    This is used to build the CE / specific-capacity summary (Table-3-like).
    """
    xls = pd.ExcelFile(path, engine="openpyxl")
    sheet_name = None
    for s in xls.sheet_names:
        if "StatisticsByCycle" in s:
            sheet_name = s
            break
    if sheet_name is None:
        raise ValueError(f"No 'StatisticsByCycle' sheet found in {path}")

    df = pd.read_excel(xls, sheet_name=sheet_name)
    if "Cycle Index" not in df.columns:
        raise KeyError(f"'Cycle Index' column not found in {path} sheet {sheet_name}")

    required_cols = [
        "Cycle Index",
        "Charge Capacity (Ah)",
        "Discharge Capacity (Ah)",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"StatisticsByCycle sheet {sheet_name} in {path} missing '{c}'")

    df = df.copy()
    df["Cycle Index"] = pd.to_numeric(df["Cycle Index"], errors="coerce")
    df = df.dropna(subset=["Cycle Index"])
    df["Cycle Index"] = df["Cycle Index"].astype(int)
    return df


def combine_statistics_for_cell(paths):
    """
    Combine StatisticsByCycle data across multiple RateTest files for a single cell.
    If the same cycle index appears in multiple files, take the row with the
    larger discharge capacity (heuristic).
    """
    frames = []
    for p in paths:
        try:
            df = load_statistics_sheet(p)
            frames.append(df)
        except Exception as e:
            print(f"Skipping {p} for statistics: {e}")
    if not frames:
        return None

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.sort_values(["Cycle Index", "Discharge Capacity (Ah)"], ascending=[True, False])
    df_agg = df_all.groupby("Cycle Index", as_index=False).first()
    return df_agg


# --- GROUPING / STYLING HELPERS ---

def build_alpha_groups(groups):
    """
    From a dict: {cell_code: [paths]}, build alpha_groups:
      {alpha: [(cell_code, [paths]), ...]}.
    """
    alpha_groups = defaultdict(list)
    for cell_code, paths in groups.items():
        alpha = cell_code[:2]
        alpha_groups[alpha].append((cell_code, paths))
    return alpha_groups


def assign_alpha_colors(alpha_groups):
    """
    Assign a distinct color to each alpha using matplotlib's tab colors.
    """
    tab_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    color_cycler = cycle(tab_colors)
    alpha_to_color = {}
    for alpha in sorted(alpha_groups.keys()):
        alpha_to_color[alpha] = next(color_cycler)
    return alpha_to_color


def assign_cell_markers(groups):
    """
    Assign a marker per cell code for group plots.
    """
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]
    marker_cycler = cycle(markers)
    cell_to_marker = {}
    for cell_code in sorted(groups.keys()):
        cell_to_marker[cell_code] = next(marker_cycler)
    return cell_to_marker



# --- OLD vs NEW COMPARISON HELPERS (master-grid v10 style) ---

# New alpha prefixes corresponding to the new repeat experiments (IP–IU, IV–IY)
NEW_ALPHA_PREFIXES = {
    "IV", "IW", "IX", "IY",  # new DTF repeats
    "IP", "IQ", "IR", "IS", "IT", "IU",  # new DTFV repeats
}

def get_alpha_prefix(code_or_alpha: str) -> str:
    """Return the first two characters of a cell code / alpha, e.g. 'HU01' -> 'HU'."""
    return str(code_or_alpha)[:2]

def is_new_alpha(alpha: str) -> bool:
    """True if an alpha (like 'HU' or 'IP') is part of the newer repeat set."""
    return get_alpha_prefix(alpha) in NEW_ALPHA_PREFIXES

def get_line_style_for_alpha(alpha: str) -> str:
    """Solid line for original / legacy data, dashed line for new repeat experiments."""
    return "--" if is_new_alpha(alpha) else "-"


# Deterministic markers by cell number suffix (e.g., HU01 -> 1, HU03 -> 3)
CELLNUM_MARKERS = {
    1: "o",
    2: "s",
    3: "^",
    4: "D",
    5: "v",
    6: "P",
    7: "X",
    8: "*",
    9: "h",
    10: "<",
    11: ">",
}

def get_cell_number(cell_code: str):
    s = str(cell_code)
    suffix = s[2:] if len(s) > 2 else ""
    digits = "".join(ch for ch in suffix if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits.lstrip("0") or "0")
    except Exception:
        return None

def marker_for_cell_code(cell_code: str) -> str:
    n = get_cell_number(cell_code)
    if n is None:
        return "o"
    return CELLNUM_MARKERS.get(n, "o")


def safe_filename(s: str, max_len: int = 120) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s[:max_len] if len(s) > max_len else s


def assign_electrolyte_colors(electrolytes):
    """Assign one color per electrolyte for comparison plots."""
    tab_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    color_cycler = cycle(tab_colors)
    out = {}
    for e in sorted({str(x).strip() for x in electrolytes if str(x).strip()}):
        out[e] = next(color_cycler)
    return out


# Default alpha sets copied from cold_t4_master_grid_v10 grouping logic
DEFAULT_ELECTROLYTE_SETS = {
    "DTF_new": ["HU", "HV", "HW", "HX", "IV", "IW", "IX", "IY"],
    "DTFV_new": ["IA", "IB", "IC", "ID", "IE", "IF", "IP", "IQ", "IR", "IS", "IT", "IU"],
    "FEC_Mod": ["FA", "EC", "HU", "HV", "HW", "HX", "IV", "IW", "IX", "IY"],
    "FEC_1wtVC": ["FA", "EC", "IA", "IB", "IP", "IQ"],
    "FEC_2wtVC": ["FA", "EC", "IC", "ID", "IE", "IF", "IR", "IS", "IT", "IU"],
    "VC_1wtFEC": ["FA", "EC", "IA", "ID", "IP", "IS"],
    "VC_2wtFEC": ["FA", "EC", "IC", "ID", "IR", "IS"],
    "2wtAdd": ["FA", "EC", "HV", "IA", "IB", "IC", "IP", "IQ", "IR", "IW"],
}


def make_old_new_comparison_index_rate(summary_df: pd.DataFrame, out_dir: str):
    """
    Build a CSV index that lists, for each electrolyte:
      - which alphas appear in old vs new
      - counts of old/new cells
    """
    if summary_df is None or summary_df.empty or "Electrolyte" not in summary_df.columns:
        return None

    by_electrolyte = defaultdict(lambda: {"old_alphas": set(), "new_alphas": set(), "old_cells": set(), "new_cells": set()})

    for _, r in summary_df.iterrows():
        electrolyte = str(r.get("Electrolyte", "")).strip()
        cell_code = str(r.get("Cell Code", "")).strip()
        alpha = get_alpha_prefix(r.get("Alpha", cell_code))
        if not electrolyte or not cell_code:
            continue

        if is_new_alpha(alpha):
            by_electrolyte[electrolyte]["new_alphas"].add(alpha)
            by_electrolyte[electrolyte]["new_cells"].add(cell_code)
        else:
            by_electrolyte[electrolyte]["old_alphas"].add(alpha)
            by_electrolyte[electrolyte]["old_cells"].add(cell_code)

    records = []
    for electrolyte, d in sorted(by_electrolyte.items()):
        if not d["old_cells"] and not d["new_cells"]:
            continue
        records.append({
            "Electrolyte": electrolyte,
            "OldAlphas": ", ".join(sorted(d["old_alphas"])),
            "NewAlphas": ", ".join(sorted(d["new_alphas"])),
            "N_OldCells": len(d["old_cells"]),
            "N_NewCells": len(d["new_cells"]),
        })

    if not records:
        return None

    df_index = pd.DataFrame(records)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "old_new_comparison_index_RateTest.csv")
    df_index.to_csv(out_path, index=False)
    print(f"Saved old/new comparison index (RateTest): {out_path}")
    return df_index


def plot_old_vs_new_rate_by_electrolyte(
    summary_df: pd.DataFrame,
    lookup,
    out_dir: str,
    electrolyte_colors: dict,
):
    """
    For each electrolyte that appears in both old and new alpha prefixes,
    make a comparison plot for RateTest data:

      - X-axis: rate labels (C/10 ... 2C)
      - Y-axis: Q_dis at the representative cycles (mAh/g)
      - Solid: old data
      - Dashed: new repeat data
      - Color: per-electrolyte (one color for the plot)
      - Marker: per-cell number suffix (01, 02, ...)
    """
    if summary_df is None or summary_df.empty:
        return

    rate_labels = list(SUMMARY_CYCLE_MAP.keys())
    rate_cols = [f"Q_dis({lab}) (mAh/g)" for lab in rate_labels]

    os.makedirs(out_dir, exist_ok=True)

    # Group rows by electrolyte, and split old/new
    by_e = defaultdict(lambda: {"old": [], "new": []})
    for _, r in summary_df.iterrows():
        electrolyte = str(r.get("Electrolyte", "")).strip()
        cell_code = str(r.get("Cell Code", "")).strip()
        alpha = get_alpha_prefix(r.get("Alpha", cell_code))
        if not electrolyte or not cell_code:
            continue
        gen = "new" if is_new_alpha(alpha) else "old"
        by_e[electrolyte][gen].append(r)

    for electrolyte, gens in sorted(by_e.items()):
        if not gens["old"] or not gens["new"]:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        plotted_any = False
        used_labels = set()

        color = electrolyte_colors.get(electrolyte, "tab:blue")

        # Plot old then new for consistent legend grouping
        for gen_label, linestyle in [("old", "-"), ("new", "--")]:
            for r in sorted(gens[gen_label], key=lambda rr: str(rr.get("Cell Code", ""))):
                cell_code = str(r.get("Cell Code", "")).strip()
                if not cell_code:
                    continue

                y = [pd.to_numeric(r.get(c, math.nan), errors="coerce") for c in rate_cols]
                x = list(range(len(rate_labels)))
                # mask finite values
                x_f = [xx for xx, yy in zip(x, y) if pd.notna(yy)]
                y_f = [yy for yy in y if pd.notna(yy)]
                if len(x_f) < 2:
                    continue

                display_label = get_display_label(cell_code, lookup)
                label = f"{display_label} ({gen_label})"
                if label in used_labels:
                    label = None
                else:
                    used_labels.add(label)

                ax.plot(
                    x_f,
                    y_f,
                    color=color,
                    linestyle=linestyle,
                    marker=marker_for_cell_code(cell_code),
                    linewidth=1.6,
                    markersize=5,
                    alpha=0.9,
                    label=label,
                )
                plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        # Title includes chemistry if available
        rep_cell = str(gens["old"][0].get("Cell Code", "") or gens["new"][0].get("Cell Code", ""))
        chem = get_cell_chemistry(rep_cell, lookup)
        title_parts = [electrolyte, "old vs new", "Rate test"]
        if chem:
            title_parts.append(chem)
        ax.set_title(" — ".join(title_parts))

        ax.set_xlabel("Rate step")
        ax.set_ylabel("Discharge Specific Capacity (mAh/g)")
        ax.set_xticks(range(len(rate_labels)))
        ax.set_xticklabels(rate_labels)
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="xx-small", ncol=2, loc="best")

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{safe_filename(electrolyte)}_RateTest_old_vs_new.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved old vs new rate comparison: {out_path}")



def plot_old_vs_new_capacity_vs_cycle_by_electrolyte(
    groups: dict,
    lookup,
    out_dir: str,
    electrolyte_colors: dict,
):
    """
    Master-grid-style old vs new comparison, but for RateTest capacity-vs-cycle traces.

      - X-axis: cycle index
      - Y-axis: specific discharge capacity (mAh/g)
      - Solid: old data
      - Dashed: new repeat data
      - Color: per-electrolyte (one color per plot)
      - Marker: per-cell number suffix (01, 02, ...)
    """
    if not groups:
        return

    os.makedirs(out_dir, exist_ok=True)

    by_e = defaultdict(lambda: {"old": [], "new": []})
    for cell_code, paths in groups.items():
        electrolyte = get_electrolyte_name(cell_code, lookup).strip()
        if not electrolyte:
            continue
        alpha = get_alpha_prefix(cell_code)
        gen = "new" if is_new_alpha(alpha) else "old"
        by_e[electrolyte][gen].append((cell_code, paths))

    for electrolyte, gens in sorted(by_e.items()):
        if not gens["old"] or not gens["new"]:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        plotted_any = False
        used_labels = set()

        color = electrolyte_colors.get(electrolyte, "tab:blue")

        for gen_label, linestyle in [("old", "-"), ("new", "--")]:
            for cell_code, paths in sorted(gens[gen_label], key=lambda x: x[0]):
                df_grouped = combine_statistics_for_cell(paths)
                if df_grouped is None or df_grouped.empty:
                    continue

                x = df_grouped["Cycle Index"].values
                y = (df_grouped["Discharge Capacity (Ah)"] * CONV_AH_TO_MAHG).values
                if len(x) < 2:
                    continue

                display_label = get_display_label(cell_code, lookup)
                label = f"{display_label} ({gen_label})"
                if label in used_labels:
                    label = None
                else:
                    used_labels.add(label)

                markevery = max(len(x) // 30, 1)

                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    marker=marker_for_cell_code(cell_code),
                    linewidth=1.6,
                    markersize=4,
                    markevery=markevery,
                    alpha=0.9,
                    label=label,
                )
                plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        # Title includes chemistry if available
        rep_cell = (gens["old"][0][0] if gens["old"] else gens["new"][0][0])
        chem = get_cell_chemistry(rep_cell, lookup)
        title_parts = [electrolyte, "old vs new", "Capacity vs cycle"]
        if chem:
            title_parts.append(chem)
        ax.set_title(" — ".join(title_parts))

        ax.set_xlabel("Cycle Number")
        ax.set_ylabel("Specific Discharge Capacity (mAh/g)")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="xx-small", ncol=2, loc="best")

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{safe_filename(electrolyte)}_CapacityVsCycle_old_vs_new.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved old vs new capacity-vs-cycle comparison: {out_path}")



def plot_rate_group_sets(
    summary_df: pd.DataFrame,
    lookup,
    out_dir: str,
    alpha_to_color: dict,
    electrolyte_sets=None,
):
    """
    Master-grid-style grouped comparisons, but for RateTest (Q_dis vs rate).

    One figure per group (DTF_new, DTFV_new, etc.). Each line is a cell.
      - Color: alpha color (consistent with other RateTest plots)
      - Linestyle: old solid, new dashed
      - Marker: cell number suffix
    """
    if summary_df is None or summary_df.empty:
        return

    rate_labels = list(SUMMARY_CYCLE_MAP.keys())
    rate_cols = [f"Q_dis({lab}) (mAh/g)" for lab in rate_labels]

    if electrolyte_sets is None:
        electrolyte_sets = DEFAULT_ELECTROLYTE_SETS

    os.makedirs(out_dir, exist_ok=True)

    for group_name, alpha_list in electrolyte_sets.items():
        df_sel = summary_df[summary_df["Alpha"].astype(str).str[:2].isin(alpha_list)].copy()
        if df_sel.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        plotted_any = False
        used_labels = set()

        for _, r in df_sel.sort_values(["Alpha", "Cell Code"]).iterrows():
            cell_code = str(r.get("Cell Code", "")).strip()
            alpha = get_alpha_prefix(r.get("Alpha", cell_code))
            if not cell_code:
                continue

            y = [pd.to_numeric(r.get(c, math.nan), errors="coerce") for c in rate_cols]
            x = list(range(len(rate_labels)))
            x_f = [xx for xx, yy in zip(x, y) if pd.notna(yy)]
            y_f = [yy for yy in y if pd.notna(yy)]
            if len(x_f) < 2:
                continue

            color = alpha_to_color.get(alpha, "tab:gray")
            ls = get_line_style_for_alpha(alpha)

            label = get_display_label(cell_code, lookup)
            if label in used_labels:
                label = None
            else:
                used_labels.add(label)

            ax.plot(
                x_f,
                y_f,
                color=color,
                linestyle=ls,
                marker=marker_for_cell_code(cell_code),
                linewidth=1.4,
                markersize=5,
                alpha=0.85,
                label=label,
            )
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        ax.set_title(f"{group_name} — Rate test (Q_dis vs rate)")
        ax.set_xlabel("Rate step")
        ax.set_ylabel("Discharge Specific Capacity (mAh/g)")
        ax.set_xticks(range(len(rate_labels)))
        ax.set_xticklabels(rate_labels)
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="xx-small", ncol=2, loc="best")

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{group_name}_RateTest_group_comparison.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved group-set comparison: {out_path}")



# --- SUMMARY TABLE (TABLE 3 STYLE) ---

def build_rate_summary_table(groups, lookup):
    """Build a summary table similar to Table 3 in the writeup.

    For each cell (full cell code), we use the combined StatisticsByCycle data and pull:
      - Formation CE from cycle 1 (discharge/charge * 100).
      - Discharge specific capacities (mAh/g) at cycles 3, 6, 9, 12, 15, 18.
    """
    records = []

    for cell_code, paths in groups.items():
        df_grouped = combine_statistics_for_cell(paths)
        if df_grouped is None or df_grouped.empty:
            continue

        row = {
            "Cell Code": cell_code,
            "Alpha": cell_code[:2],
        }

        electrolyte = get_electrolyte_name(cell_code, lookup)
        chem = get_cell_chemistry(cell_code, lookup)
        if electrolyte:
            row["Electrolyte"] = electrolyte
        else:
            row["Electrolyte"] = ""
        if chem:
            row["Chemistry"] = chem
        else:
            row["Chemistry"] = ""

        cyc1 = df_grouped.loc[df_grouped["Cycle Index"] == 1]
        if not cyc1.empty:
            chg = float(cyc1["Charge Capacity (Ah)"].iloc[0])
            dis = float(cyc1["Discharge Capacity (Ah)"].iloc[0])
            if chg > 0:
                row["CE Cycle 1 (%)"] = 100.0 * dis / chg
            else:
                row["CE Cycle 1 (%)"] = math.nan
        else:
            row["CE Cycle 1 (%)"] = math.nan

        for rate_label, c_idx in SUMMARY_CYCLE_MAP.items():
            cyc = df_grouped.loc[df_grouped["Cycle Index"] == c_idx]
            col_name = f"Q_dis({rate_label}) (mAh/g)"
            if cyc.empty:
                row[col_name] = math.nan
            else:
                q_ah = float(cyc["Discharge Capacity (Ah)"].iloc[0])
                row[col_name] = q_ah * CONV_AH_TO_MAHG

        records.append(row)

    if not records:
        return None
    return pd.DataFrame(records)


# --- RATE SUMMARY PLOTS (OLD: CAPACITY VS CYCLE) ---

def plot_rate_summary_per_cell(groups, lookup):
    """
    For each cell, make a small figure with CE vs cycle (cycle 1)
    plus specific discharge capacity vs cycle for the representative
    rate-test cycles (3, 6, 9, 12, 15, 18).
    """
    for cell_code, paths in groups.items():
        df_grouped = combine_statistics_for_cell(paths)
        if df_grouped is None or df_grouped.empty:
            continue

        df_grouped = df_grouped.copy()
        df_grouped["CE (%)"] = 100.0 * df_grouped["Discharge Capacity (Ah)"] / df_grouped["Charge Capacity (Ah)"]

        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = ax1.twinx()

        ax1.plot(
            df_grouped["Cycle Index"],
            df_grouped["Discharge Capacity (Ah)"] * CONV_AH_TO_MAHG,
            "o-",
            label="Q_dis (mAh/g)",
        )
        ax2.plot(
            df_grouped["Cycle Index"],
            df_grouped["CE (%)"],
            "s--",
            color="tab:red",
            label="CE (%)",
        )

        ax1.set_xlabel("Cycle Index")
        ax1.set_ylabel("Specific Discharge Capacity (mAh/g)")
        ax2.set_ylabel("CE (%)")

        display_label = get_display_label(cell_code, lookup)
        chem = get_cell_chemistry(cell_code, lookup)

        title_parts = []
        if chem:
            title_parts.append(chem)
        if display_label:
            title_parts.append(display_label)
        else:
            title_parts.append(cell_code)
        ax1.set_title(" — ".join(title_parts))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize="x-small", loc="best")

        fig.tight_layout()
        out_path = os.path.join(save_dir, f"{cell_code}_RateTest_Summary_vsCycle.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved rate-summary vs cycle plot: {out_path}")


# --- PER-CELL COMBINED CAPACITY VS CYCLE ---

def plot_per_cell(groups, lookup):
    """
    For each cell, make a combined normalized plot:
      - X-axis: cycle index
      - Y-axis: specific discharge capacity (mAh/g)
    """
    for cell_code, paths in groups.items():
        df_grouped = combine_statistics_for_cell(paths)
        if df_grouped is None or df_grouped.empty:
            continue

        spec_discharge = df_grouped["Discharge Capacity (Ah)"] * CONV_AH_TO_MAHG

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            df_grouped["Cycle Index"],
            spec_discharge,
            s=25,
            alpha=0.85,
            edgecolor="k",
            linewidth=0.2,
            c="tab:blue",
            label="Q_dis (mAh/g)",
        )

        display_label = get_display_label(cell_code, lookup)
        chem = get_cell_chemistry(cell_code, lookup)
        electrolyte = get_electrolyte_name(cell_code, lookup)

        title_parts = []
        if chem:
            title_parts.append(chem)
        if electrolyte:
            title_parts.append(electrolyte)
        if display_label and display_label not in title_parts:
            title_parts.append(display_label)
        if not title_parts:
            title_parts.append(cell_code)

        ax.set_title(" — ".join(title_parts))
        ax.set_xlabel("Cycle Number")
        ax.set_ylabel("Specific Discharge Capacity (mAh/g)")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(fontsize="x-small", loc="best")

        fig.tight_layout()
        out_path = os.path.join(save_dir, f"{cell_code}_RateTest_SpecCapacity_vsCycle.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved per-cell capacity-vs-cycle plot: {out_path}")


# --- GROUP PLOTS BY ALPHA (CAPACITY VS CYCLE) ---

def plot_groups_by_alpha(alpha_groups, lookup, alpha_to_color, cell_to_marker):
    """
    One figure per alpha:
      - X-axis: cycle index
      - Y-axis: specific discharge capacity (mAh/g)
    """
    for alpha, entries in sorted(alpha_groups.items()):
        color = alpha_to_color.get(alpha, "k")

        fig, ax = plt.subplots(figsize=(7, 5))
        plotted_any = False

        electrolyte = ""
        if alpha in lookup.index:
            row = lookup.loc[alpha]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            electrolyte = str(row.get("Electrolyte", "")).strip()
        chem = get_cell_chemistry(alpha, lookup)

        for cell_code, paths in entries:
            df_grouped = combine_statistics_for_cell(paths)
            if df_grouped is None or df_grouped.empty:
                continue

            spec_discharge = df_grouped["Discharge Capacity (Ah)"] * CONV_AH_TO_MAHG
            label = get_display_label(cell_code, lookup)
            marker = cell_to_marker.get(cell_code, "o")

            ax.scatter(
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
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        title_parts = []
        if chem:
            title_parts.append(chem)
        if electrolyte:
            title_parts.append(electrolyte)
        else:
            title_parts.append(f"{alpha} group")

        ax.set_title(" — ".join(title_parts))
        ax.set_xlabel("Cycle Number")
        ax.set_ylabel("Specific Discharge Capacity (mAh/g)")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(fontsize="xx-small", ncol=1, loc="best")

        fig.tight_layout()
        out_path = os.path.join(save_dir, f"{alpha}_group_RateTest_SpecCapacity.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved group-by-alpha plot: {out_path}")


def plot_all_alphas_side_by_side(alpha_groups, lookup, alpha_to_color, cell_to_marker):
    """
    Overview plot: all alphas on one figure (capacity vs cycle).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_any = False

    handles = []
    labels = []

    for alpha, entries in sorted(alpha_groups.items()):
        color = alpha_to_color.get(alpha, "k")

        for cell_code, paths in entries:
            df_grouped = combine_statistics_for_cell(paths)
            if df_grouped is None or df_grouped.empty:
                continue

            spec_discharge = df_grouped["Discharge Capacity (Ah)"] * CONV_AH_TO_MAHG
            marker = cell_to_marker.get(cell_code, "o")
            label = get_display_label(cell_code, lookup)

            sc = ax.scatter(
                df_grouped["Cycle Index"],
                spec_discharge,
                marker=marker,
                s=25,
                alpha=0.7,
                edgecolor="k",
                linewidth=0.2,
                c=color,
            )
            if label not in labels:
                handles.append(sc)
                labels.append(label)
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Specific Discharge Capacity (mAh/g)")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(handles, labels, fontsize="xx-small", ncol=2, loc="best")
    ax.set_title("RateTest Overview — all alphas")

    fig.tight_layout()
    out_path = os.path.join(save_dir, "All_alphas_RateTest_SpecCapacity.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved all-alphas overview plot: {out_path}")

def plot_rt_from_cycle20_combined(groups, lookup):
    """
    Combined RT cycling plot (cycles ≥ 20) for IK02, IL03, IM01, IN01, IO01.
    Each cell is color-coded with its own dashed reference line at
    the C/2 (cycle 12) discharge capacity.
    """
    target_cells = ["IK02", "IL03", "IM01", "IN01", "IO01"]
    colors = {
        "IK02": "tab:blue",
        "IL03": "tab:orange",
        "IM01": "tab:green",
        "IN01": "tab:red",
        "IO01": "tab:purple",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for cell_code in target_cells:
        if cell_code not in groups:
            print(f"Skipping {cell_code} — no RateTest data found.")
            continue

        df_grouped = combine_statistics_for_cell(groups[cell_code])
        if df_grouped is None or df_grouped.empty:
            print(f"Skipping {cell_code} — empty or missing StatisticsByCycle data.")
            continue

        # Find cycle 12 discharge capacity (C/2 reference)
        cyc12 = df_grouped[df_grouped["Cycle Index"] == 12]
        q12_ref = None
        if not cyc12.empty:
            q12_ref = float(cyc12["Discharge Capacity (Ah)"].iloc[0]) * CONV_AH_TO_MAHG

        # Filter to cycles ≥ 20
        df_rt = df_grouped[df_grouped["Cycle Index"] >= 20].copy()
        if df_rt.empty:
            continue

        df_rt["Qdis (mAh/g)"] = df_rt["Discharge Capacity (Ah)"] * CONV_AH_TO_MAHG
        label = get_display_label(cell_code, lookup)
        color = colors.get(cell_code, None)

        # Main curve
        ax.plot(
            df_rt["Cycle Index"][:-1],
            df_rt["Qdis (mAh/g)"][:-1],
            marker="o",
            lw=1.5,
            ms=4,
            alpha=0.9,
            color=color,
            label=label,
        )

        # Individual dashed reference line (not in legend)
        if q12_ref is not None:
            ax.axhline(y=q12_ref, color=color, linestyle="--", lw=1.2, alpha=0.8)

    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Specific Discharge Capacity (mAh/g)")
    ax.set_title("Room-Temperature Cycling ≥ Cycle 20 (with C/2 Reference Lines)")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize="x-small", loc="best")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "RT_Cycles20plus_AllCells_withIndividualRefs.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved combined RT plot with individual reference lines: {out_path}")



def plot_rt_from_cycle20(groups, lookup):
    """
    Plot room-temperature (RT) cycling curves for specified cells:
      IK02, IL03, IM01, IN01, IO01
    starting from cycle 20 onwards.
    Adds a dashed horizontal line at the C/2 (cycle 12) discharge capacity.
    """
    target_cells = ["IK02", "IL03", "IM01", "IN01", "IO01"]

    for cell_code in target_cells:
        if cell_code not in groups:
            print(f"Skipping {cell_code} — no RateTest data found.")
            continue

        df_grouped = combine_statistics_for_cell(groups[cell_code])
        if df_grouped is None or df_grouped.empty:
            print(f"Skipping {cell_code} — empty or missing StatisticsByCycle data.")
            continue

        # Filter to cycles ≥ 20
        df_rt = df_grouped[df_grouped["Cycle Index"] >= 20].copy()
        if df_rt.empty:
            print(f"No cycles ≥ 20 found for {cell_code}.")
            continue

        # Compute normalized discharge capacity
        df_rt["Qdis (mAh/g)"] = df_rt["Discharge Capacity (Ah)"] * CONV_AH_TO_MAHG

        # Determine reference (cycle 12) discharge capacity for dashed line
        cyc12 = df_grouped[df_grouped["Cycle Index"] == 12]
        if cyc12.empty:
            q12_ref = None
        else:
            q12_ref = float(cyc12["Discharge Capacity (Ah)"].iloc[0]) * CONV_AH_TO_MAHG

        # Plot
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(df_rt["Cycle Index"], df_rt["Qdis (mAh/g)"], "o-", c="tab:blue",
                lw=1.5, ms=4, label="Discharge capacity (mAh/g)")

        if q12_ref is not None:
            ax.axhline(y=q12_ref, color="tab:red", ls="--", lw=1.2,
                       label=f"C/2 ref (Cycle 12 Qdis ≈ {q12_ref:.0f} mAh/g)")

        display_label = get_display_label(cell_code, lookup)
        chem = get_cell_chemistry(cell_code, lookup)
        electrolyte = get_electrolyte_name(cell_code, lookup)

        title = " — ".join(filter(None, [chem, electrolyte, display_label, "RT Cycling (≥ Cycle 20)"]))
        ax.set_title(title)
        ax.set_xlabel("Cycle Number")
        ax.set_ylabel("Specific Discharge Capacity (mAh/g)")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend(fontsize="x-small", loc="best")

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{cell_code}_RT_cycles20plus_withC2ref.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved RT plot: {out_path}")

# --- EARLY/LATE ALPHA PLOTS (CAPACITY VS CYCLE) ---

def plot_alpha_group_window_on_axis(
    ax,
    alpha,
    entries,
    lookup,
    color,
    cell_to_marker,
    cycle_min=None,
    cycle_max=None,
    exclude_last=False,
    window_label="",
):
    """Plot an alpha group on a given axis, restricted to a cycle window.

    Parameters
    ----------
    cycle_min : int or None
        Minimum cycle index to include (inclusive).
    cycle_max : int or None
        Maximum cycle index to include (inclusive). If None, no explicit upper bound.
    exclude_last : bool
        If True, exclude the last cycle for each cell (useful for "2nd to last" windows).
    window_label : str
        Extra string appended to the plot title to indicate the cycle window.
    """
    plotted_any = False

    electrolyte = ""
    if alpha in lookup.index:
        row = lookup.loc[alpha]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        electrolyte = str(row.get("Electrolyte", "")).strip()
    chem = get_cell_chemistry(alpha, lookup)

    for cell_code, paths in entries:
        df_grouped = combine_statistics_for_cell(paths)
        if df_grouped is None or df_grouped.empty:
            continue

        df_filt = df_grouped.copy()

        if exclude_last and not df_filt.empty:
            max_cycle = df_filt["Cycle Index"].max()
            df_filt = df_filt[df_filt["Cycle Index"] < max_cycle]

        if cycle_min is not None:
            df_filt = df_filt[df_filt["Cycle Index"] >= cycle_min]
        if cycle_max is not None:
            df_filt = df_filt[df_filt["Cycle Index"] <= cycle_max]

        if df_filt.empty:
            continue

        spec_discharge = df_filt["Discharge Capacity (Ah)"] * CONV_AH_TO_MAHG
        label = get_display_label(cell_code, lookup)
        marker = cell_to_marker.get(cell_code, "o")

        ax.scatter(
            df_filt["Cycle Index"],
            spec_discharge,
            label=label,
            marker=marker,
            s=25,
            alpha=0.85,
            edgecolor="k",
            linewidth=0.2,
            c=color,
        )
        plotted_any = True

    if not plotted_any:
        return False

    title_parts = []
    if chem:
        title_parts.append(chem)
    if electrolyte:
        title_parts.append(electrolyte)
    else:
        title_parts.append(f"{alpha} group")
    if window_label:
        title_parts.append(window_label)

    ax.set_title(" — ".join(title_parts))
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Specific Discharge Capacity (mAh/g)")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.tick_params(direction="in", top=True, right=True)

    return True


def plot_alpha_early_late(alpha_groups, lookup, alpha_to_color, cell_to_marker):
    """For each alpha, make two plots:
        1) Cycles 1–19
        2) Cycles 19–(second-to-last) for each cell in that alpha group.
    """
    for alpha, entries in sorted(alpha_groups.items()):
        color = alpha_to_color.get(alpha, "k")

        # Early cycles: 1–19
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        plotted_early = plot_alpha_group_window_on_axis(
            ax1,
            alpha,
            entries,
            lookup,
            color,
            cell_to_marker,
            cycle_min=1,
            cycle_max=19,
            exclude_last=False,
            window_label="Cycles 1–19",
        )
        if plotted_early:
            plt.tight_layout()
            out_path1 = os.path.join(
                save_dir, f"{alpha}_group_RateTest_SpecCapacity_cycles_1_19.png"
            )
            fig1.savefig(out_path1, dpi=300)
            print(f"Saved alpha early-cycle plot: {out_path1}")
        plt.close(fig1)

        # Later cycles: 19–(second-to-last)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        plotted_late = plot_alpha_group_window_on_axis(
            ax2,
            alpha,
            entries,
            lookup,
            color,
            cell_to_marker,
            cycle_min=19,
            cycle_max=None,
            exclude_last=True,
            window_label="Cycles 19–(2nd to last)",
        )
        if plotted_late:
            plt.tight_layout()
            out_path2 = os.path.join(
                save_dir, f"{alpha}_group_RateTest_SpecCapacity_cycles_19_to_second_last.png"
            )
            fig2.savefig(out_path2, dpi=300)
            print(f"Saved alpha late-cycle plot: {out_path2}")
        plt.close(fig2)


# --- NEW: RATE-TEST VOLTAGE–CAPACITY CURVES PER CELL ---

def get_channel_sheet_name_for_rate(path):
    """Return the sheet name for the main channel data (2nd sheet).

    For the rate-test files, the detailed voltage–capacity data live on the
    second sheet (index 1), just like the -51 °C discharge scripts.
    """
    xls = pd.ExcelFile(path, engine="openpyxl")
    if len(xls.sheet_names) < 2:
        raise ValueError(f"{path} has no channel sheet with time-series data")
    return xls.sheet_names[1]


def load_rate_channel_data_for_cell(paths):
    """Load and concatenate the per-point channel data for all RateTest files
    belonging to a single cell.

    Returns a DataFrame with at least:
      - 'Cycle Index'
      - 'Voltage (V)'
      - 'Current (A)'
      - one of 'Charge Capacity (Ah)', 'Discharge Capacity (Ah)', or 'Capacity (Ah)'
    or None if nothing usable is found.
    """
    dfs = []
    for p in paths:
        try:
            sheet_name = get_channel_sheet_name_for_rate(p)
            df = pd.read_excel(p, sheet_name=sheet_name, engine="openpyxl")
            if "Cycle Index" not in df.columns:
                print(f"{p} has no 'Cycle Index' column; skipping for rate curves.")
                continue
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {p} for rate curves: {e}")
    if not dfs:
        return None
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def plot_rate_voltage_curves_per_cell(groups, lookup):
    """For each cell, plot charge and discharge curves as mAh/g vs voltage.

    - X-axis: specific capacity (mAh/g)
    - Y-axis: voltage (V)
    - Curves: charge (dashed) and discharge (solid) for the representative
      rate-test cycles (3, 6, 9, 12, 15, 18).
    - Legend: one entry per rate (C/10, C/8, ..., 2C); color encodes rate,
      line style encodes direction (charge vs discharge).

    This uses the full time-series 'channel' sheets rather than StatisticsByCycle.
    """
    # Map each representative cycle index to a distinct color
    rate_labels = list(SUMMARY_CYCLE_MAP.keys())
    cycle_indices = [SUMMARY_CYCLE_MAP[k] for k in rate_labels]

    base_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]
    color_cycle = cycle(base_colors)
    rate_to_color = {rate: next(color_cycle) for rate in rate_labels}

    for cell_code, paths in groups.items():
        df_all = load_rate_channel_data_for_cell(paths)
        if df_all is None or df_all.empty:
            continue

        required_cols = {"Voltage (V)", "Current (A)"}
        if not required_cols.issubset(df_all.columns):
            print(f"{cell_code}: missing one of {required_cols} in channel data; skipping rate curves.")
            continue

        has_chg = "Charge Capacity (Ah)" in df_all.columns
        has_dis = "Discharge Capacity (Ah)" in df_all.columns
        has_cap = "Capacity (Ah)" in df_all.columns

        if not (has_chg or has_dis or has_cap):
            print(f"{cell_code}: no capacity columns found in channel data; skipping rate curves.")
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        rate_handles = {}
        plotted_any = False

        for rate_label, c_idx in zip(rate_labels, cycle_indices):
            df_cyc = df_all[df_all["Cycle Index"] == c_idx].copy()
            if df_cyc.empty:
                continue

            color = rate_to_color[rate_label]

            # Charge segment: Current > 0
            df_chg = df_cyc[df_cyc["Current (A)"] > 0].copy()
            if not df_chg.empty:
                if has_chg:
                    cap = df_chg["Charge Capacity (Ah)"]
                elif has_cap:
                    cap = df_chg["Capacity (Ah)"]
                else:
                    cap = None
                if cap is not None:
                    x_chg = cap * CONV_AH_TO_MAHG
                    y_chg = df_chg["Voltage (V)"]
                    mask = (~x_chg.isna()) & (~y_chg.isna())
                    x_chg = x_chg[mask]
                    y_chg = y_chg[mask]
                    if not x_chg.empty:
                        order = x_chg.argsort()
                        x_chg = x_chg.iloc[order]
                        y_chg = y_chg.iloc[order]
                        h_chg, = ax.plot(
                            x_chg,
                            y_chg,
                            linestyle="--",
                            linewidth=1.5,
                            color=color,
                        )
                        # Use one handle per rate for the legend (charge curve)
                        if rate_label not in rate_handles:
                            rate_handles[rate_label] = h_chg
                        plotted_any = True

            # Discharge segment: Current < 0
            df_dis = df_cyc[df_cyc["Current (A)"] < 0].copy()
            if not df_dis.empty:
                if has_dis:
                    cap = df_dis["Discharge Capacity (Ah)"]
                elif has_cap:
                    cap = df_dis["Capacity (Ah)"]
                else:
                    cap = None
                if cap is not None:
                    x_dis = cap * CONV_AH_TO_MAHG
                    y_dis = df_dis["Voltage (V)"]
                    mask = (~x_dis.isna()) & (~y_dis.isna())
                    x_dis = x_dis[mask]
                    y_dis = y_dis[mask]
                    if not x_dis.empty:
                        order = x_dis.argsort()
                        x_dis = x_dis.iloc[order]
                        y_dis = y_dis.iloc[order]
                        ax.plot(
                            x_dis,
                            y_dis,
                            linestyle="-",
                            linewidth=1.5,
                            color=color,
                        )
                        # Don't add a second legend entry; style difference carries the direction info
                        plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        ax.set_xlabel("Specific Capacity (mAh/g)")
        ax.set_ylabel("Voltage (V)")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)

        display_label = get_display_label(cell_code, lookup)
        chem = get_cell_chemistry(cell_code, lookup)
        electrolyte = get_electrolyte_name(cell_code, lookup)

        title_parts = []
        if chem:
            title_parts.append(chem)
        if electrolyte:
            title_parts.append(electrolyte)
        elif display_label:
            title_parts.append(display_label)
        else:
            title_parts.append(cell_code)
        title_parts.append("Rate test (charge/discharge curves)")
        ax.set_title(" — ".join(title_parts))

        if rate_handles:
            legend_labels = list(rate_handles.keys())
            legend_handles = [rate_handles[k] for k in legend_labels]
            ax.legend(
                legend_handles,
                legend_labels,
                fontsize="x-small",
                loc="best",
                title="Rate",
            )

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{cell_code}_RateTest_V_vs_mAhg_curves.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved rate-test V–capacity curves: {out_path}")


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

    # Build a Table-3-style summary table and save it
    summary_df = build_rate_summary_table(groups, lookup)
    if summary_df is not None:
        summary_csv = os.path.join(save_dir, "RateTest_Summary_Table_like_Table3.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved rate-summary table (CSV): {summary_csv}")

        try:
            summary_xlsx = os.path.join(save_dir, "RateTest_Summary_Table_like_Table3.xlsx")
            summary_df.to_excel(summary_xlsx, index=False)
            print(f"Saved rate-summary table (Excel): {summary_xlsx}")
        except Exception as e:
            print(f"Could not save Excel summary table: {e}")

        # --- master-grid v10 style comparisons (RateTest) ---
        comparisons_dir = os.path.join(save_dir, "RateTest_Old_vs_New_Comparisons")
        os.makedirs(comparisons_dir, exist_ok=True)

        electrolyte_colors = assign_electrolyte_colors(summary_df["Electrolyte"].dropna().unique())
        make_old_new_comparison_index_rate(summary_df, comparisons_dir)
        plot_old_vs_new_rate_by_electrolyte(summary_df, lookup, comparisons_dir, electrolyte_colors)

        cap_cycle_dir = os.path.join(save_dir, "RateTest_Old_vs_New_CapacityVsCycle")
        os.makedirs(cap_cycle_dir, exist_ok=True)
        plot_old_vs_new_capacity_vs_cycle_by_electrolyte(groups, lookup, cap_cycle_dir, electrolyte_colors)

        groupsets_dir = os.path.join(save_dir, "RateTest_GroupSet_Comparisons")
        os.makedirs(groupsets_dir, exist_ok=True)
        plot_rate_group_sets(summary_df, lookup, groupsets_dir, alpha_to_color)

    # Per-cell rate-test voltage–capacity curves (charge/discharge) with rate legend
    #plot_rate_voltage_curves_per_cell(groups, lookup)

    # Per-cell rate-summary plots at the representative Table-3 cycles
    #plot_rate_summary_per_cell(groups, lookup)

    # Per-cell combined plots (normalized) with cold_t2-style display label & chemistry in title
    #plot_per_cell(groups, lookup)

    # One figure per alpha, plus side-by-side overview
    #plot_groups_by_alpha(alpha_groups, lookup, alpha_to_color, cell_to_marker)
    #plot_all_alphas_side_by_side(alpha_groups, lookup, alpha_to_color, cell_to_marker)

    # Alpha-level early/late cycle plots (1–19 and 19–2nd-to-last)
    #plot_alpha_early_late(alpha_groups, lookup, alpha_to_color, cell_to_marker)
    # Room-temperature cycling plots for target cells (≥ cycle 20)
    plot_rt_from_cycle20_combined(groups, lookup)

    #plot_rt_from_cycle20(groups, lookup)



if __name__ == "__main__":
    main()
