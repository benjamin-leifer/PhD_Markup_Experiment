import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# --- user settings ---
base_dir = r"C:\Users\benja\Downloads\Dilute THF Data\11_25_25\-51C_Repeats"
old_directory = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\-51C_discharges'
lookup_table_path = r"C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx"
plots_dir = os.path.join(base_dir, "plots_t19")
os.makedirs(plots_dir, exist_ok=True)

# --- FEC x VC master-grid settings ---
# alpha_select: None -> include all alphas; "HU" -> only HU##; "best" -> best-per-bin across all alphas
GRID_ALPHA_SELECT = None
# selection: "all" overlays all replicates; "best" selects best cell per (FEC,VC) bin (still overlays its replicates)
GRID_SELECTION = "all"
# Voltage used to score "best" (mAh/g at this voltage, via interpolation)
GRID_BEST_VOLTAGE = 2.0
# Always include these alpha prefixes as references in every subplot
#GRID_REFERENCE_ALPHAS = ("FA", "EC")
GRID_REFERENCE_ALPHAS = ("FA", "IZ")
# Grid layout
GRID_FEC_LEVELS = (1, 2, 5, 10)
GRID_VC_LEVELS = (0, 1, 2)
# Markers on curves (cell-code markers); set False if you want clean lines
GRID_SHOW_MARKERS = True

# --- metrics / trend settings ---
CAPACITY_VOLTAGE = 2.5  # voltage used for capacity-at-V metrics/trend plots


# If True, also save one plot per cell (can be a lot of files)
MAKE_PER_CELL_PLOTS = False



# --- effect (trend) plots settings ---
# If True, generate "effect of FEC" and "effect of VC" trend plots from the per-cell metrics table.
MAKE_EFFECT_TREND_PLOTS = True

# Overlay per-trial mean ± 1σ in each subplot (uses TRIAL_LINESTYLES for Trial 1/2/3)
EFFECT_OVERLAY_TRIAL_MEAN_SD = True
REF_CAP_MAH = 4.0
REF_SPEC_MAH = 160.6
CONV_AH_TO_MAHG = 1000.0 * REF_SPEC_MAH / REF_CAP_MAH  # 40150.0
non_16mm_Ref = REF_CAP_MAH * 1.606 / 2


# ---------- helpers ----------
def debug_marker_collisions(df_metrics, trial=3):
    d = df_metrics[df_metrics["Trial"] == trial].copy()

    # ensure these exist even if something upstream changed
    if "CellNumber" not in d.columns:
        d["CellNumber"] = d["CellCode"].apply(get_cell_number)

    d["Marker"] = d["CellNumber"].apply(marker_for_cell_number)

    # (A) show rows that will default to "o" because they have no mapping
    mapped_keys = set(CELLNUM_MARKERS.keys())
    bad = d[d["CellNumber"].isna() | (~d["CellNumber"].isin(mapped_keys))].copy()
    if not bad.empty:
        print("\n=== Rows with missing/unmapped CellNumber (marker will default to 'o') ===")
        cols = ["CellCode", "Alpha", "Electrolyte", "FEC_wt", "VC_wt", "CellNumber", "Marker"]
        cols = [c for c in cols if c in bad.columns]
        print(bad[cols].sort_values(["FEC_wt", "VC_wt", "CellCode"]).to_string(index=False))

    # (B) show true collisions: same (FEC,VC) has repeated marker
    print("\n=== Marker collisions by (FEC_wt, VC_wt) ===")
    any_collisions = False
    for (fec, vc), g in d.groupby(["FEC_wt", "VC_wt"]):
        if g["Marker"].nunique() != len(g):
            any_collisions = True
            print(f"\n-- FEC={fec} wt%, VC={vc} wt% --")
            cols = ["CellCode", "Alpha", "Electrolyte", "CellNumber", "Marker"]
            cols = [c for c in cols if c in g.columns]
            print(g[cols].sort_values(["Marker", "CellCode"]).to_string(index=False))

    if not any_collisions:
        print("No collisions found.")


def find_discharge_files(base_dir, old_directory=None, temp_tag=None):
    """
    Search `base_dir` and optional `old_directory` for discharge .xlsx files.
    Returns a list of (path, source) tuples where source is 'base' or 'old'.
    Deduplicates paths while preserving order (base_dir files first).

    If temp_tag is provided (e.g. "-51"), only files whose FILENAME or
    IMMEDIATE PARENT FOLDER explicitly contains "-51C" (case-insensitive)
    are included. This avoids accidentally pulling in non-51C data that
    happen to live under a higher-level "-51C_..." directory.
    """
    dirs_to_search = []
    if base_dir and os.path.isdir(base_dir):
        dirs_to_search.append((base_dir, "base"))
    if old_directory and os.path.isdir(old_directory):
        dirs_to_search.append((old_directory, "old"))

    all_results = []
    seen = set()

    temp_tag_norm = temp_tag.lower() if temp_tag else None
    canonical_temp = None
    if temp_tag_norm:
        # "-51", "51", "-51c" -> "51"
        canonical_temp = temp_tag_norm.lstrip("-").rstrip("c")

    for root_dir, src in dirs_to_search:
        for r, _, files in os.walk(root_dir):
            for fn in sorted(files):
                fn_lower = fn.lower()
                if not (fn_lower.endswith(".xlsx") and "dis" in fn_lower):
                    continue

                p = os.path.join(r, fn)
                norm = os.path.normcase(os.path.normpath(p))

                # robust -51C filter, based on file name and immediate parent folder
                if canonical_temp:
                    base_lower = fn_lower
                    parent_lower = os.path.basename(r).lower()
                    pattern = f"-{canonical_temp}c"   # e.g. "-51c"
                    if pattern not in base_lower and pattern not in parent_lower:
                        # skip anything that isn't explicitly tagged as -51C
                        continue

                if norm in seen:
                    continue
                seen.add(norm)
                all_results.append((p, src))
    return all_results


def get_cell_code(path: str) -> str:
    base = os.path.basename(path)
    root = base.split("_")[0]
    return root.split("-")[-1]


def get_channel_sheet_name(path):
    xls = pd.ExcelFile(path)
    if len(xls.sheet_names) < 2:
        raise ValueError(f"{path} has no channel sheet")
    return xls.sheet_names[1]


def load_discharge_curve(path, lookup=None):
    """
    Load discharge curve from `path`. If `lookup` (DataFrame) is provided,
    use the cathode value to decide whether to normalize with REF_CAP_MAH
    (default) or non_16mm_Ref when the cathode does not contain `16mm` / `16 mm`.
    Returns (x_spec, y_volt, cell_code).
    """
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

    # decide which reference capacity to use
    cap_ah = REF_CAP_MAH
    try:
        if lookup is not None:
            _, cathode = get_anode_cathode(cell_code, lookup)
            if cathode:
                low = cathode.lower()
                if ("16mm" not in low) and ("16 mm" not in low):
                    cap_ah = non_16mm_Ref
    except Exception:
        # on any error, fall back to default REF_CAP_MAH
        cap_ah = REF_CAP_MAH

    conv = 1000.0 * REF_SPEC_MAH / cap_ah
    df_dis["Spec Discharge Capacity (mAh/g)"] = df_dis["Discharge Capacity (Ah)"] * conv
    df_dis = df_dis.sort_values("Spec Discharge Capacity (mAh/g)")
    x_spec = df_dis["Spec Discharge Capacity (mAh/g)"].values
    y_volt = df_dis["Voltage (V)"].values
    return x_spec, y_volt, cell_code


def aggregate_discharge_for_cell(paths, lookup=None):
    """
    Aggregate discharge points across files in `paths`. Pass `lookup` to
    load_discharge_curve so per-file normalization is applied.
    """
    x_all = []
    y_all = []
    for p in paths:
        try:
            x_spec, y_volt, _ = load_discharge_curve(p, lookup)
        except Exception as e:
            print(f"Skipping {p} for aggregation: {e}")
            continue
        x_all.extend(x_spec.tolist())
        y_all.extend(y_volt.tolist())
    if not x_all:
        return None, None
    x = np.asarray(x_all)
    y = np.asarray(y_all)
    order = np.argsort(x)
    return x[order], y[order]


def load_lookup_table(path):
    df = pd.read_excel(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "Cell Code" not in df.columns:
        raise KeyError("Lookup table missing 'Cell Code' column")
    if "Electrolyte" not in df.columns:
        df["Electrolyte"] = ""
    df["Cell Code"] = df["Cell Code"].str.strip()
    df["Electrolyte"] = df["Electrolyte"].fillna("").astype(str).str.strip()
    return df.set_index("Cell Code")


def get_electrolyte_name(cell_code: str, lookup: pd.DataFrame) -> str:
    """
    Return the electrolyte string for a given cell code.
    First tries full cell code, then falls back to alpha (first two letters).
    """
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


# ---------- label helper ----------
def get_display_label(cell_code, lookup):
    """
    Build 'Electrolyte-# (alpha)' for a given cell_code.
    Looks up Electrolyte by alpha / full code and extracts sample number
    from the suffix.
    """
    electrolyte = get_electrolyte_name(cell_code, lookup)
    alpha = cell_code[:2]
    suffix = cell_code[2:]
    sample_num = suffix.lstrip("0") or suffix or ""

    if electrolyte and sample_num:
        # e.g. "DT14-1 (HU)"
        return f"{electrolyte}-{sample_num} ({alpha})"
    elif electrolyte:
        # e.g. "DT14 (FA)" when there is no numeric suffix
        return f"{electrolyte} ({alpha})"
    else:
        return cell_code


def get_anode_cathode_sheet_names(path):
    """
    Return (anode_sheet_name, cathode_sheet_name) if found in `path` Excel file.
    Matches sheet names containing 'anode' or 'cathode' (case-insensitive).
    Currently not used for titles, but available if needed.
    """
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return "", ""
    anode = ""
    cathode = ""
    for s in xls.sheet_names:
        low = s.lower()
        if "anode" in low and not anode:
            anode = s
        if "cathode" in low and not cathode:
            cathode = s
        if anode and cathode:
            break
    return anode, cathode


def get_anode_cathode(code: str, lookup: pd.DataFrame):
    """
    Look up cathode and anode for a given code (full cell code or alpha)
    from the Spring 2025 Cell List.

    Tries:
      1. Full cell code (e.g. 'HU01')
      2. Alpha prefix (e.g. 'HU')

    Expects columns 'Cathode' and 'Anode' if present; falls back gracefully
    if they are missing.
    """
    anode = ""
    cathode = ""

    if not isinstance(lookup, pd.DataFrame):
        return anode, cathode

    keys_to_try = [code]
    alpha = code[:2]
    if alpha not in keys_to_try:
        keys_to_try.append(alpha)

    for key in keys_to_try:
        if key in lookup.index:
            row = lookup.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if "Cathode" in lookup.columns:
                cathode = str(row.get("Cathode", "")).strip()
            if "Anode" in lookup.columns:
                anode = str(row.get("Anode", "")).strip()
            break

    return anode, cathode


# ---------- additive parsing + linewidth ----------
def get_total_additive(electrolyte: str) -> float:
    """
    Heuristic parser for total additive content (wt%) from electrolyte names like:
    - DTF141  -> FEC 1%
    - DTF145  -> FEC 5%
    - DTF1410 -> FEC 10%
    - DTFV1411  -> FEC 1%, VC 1%   (total 2)
    - DTFV1421  -> 2 + 1 = 3
    - DTFV1452  -> 5 + 2 = 7
    - DTFV14102 -> 10 + 2 = 12
    Returns F% + V% (approx) or 0 if it can't be parsed.
    """
    if not electrolyte:
        return 0.0

    e = electrolyte.upper()

    def extract_digits(after: str) -> str:
        return "".join(ch for ch in after if ch.isdigit())

    # DTFV14(Fwt%,Vwt%)
    if "DTFV14" in e:
        tail = e.split("DTFV14", 1)[1]
        digits = extract_digits(tail)
        if not digits:
            return 0.0
        best = None
        # try all splits into F and V, choose the one with largest F+V within a sane range
        for i in range(1, len(digits)):
            F = int(digits[:i])
            V = int(digits[i:])
            if 0 <= F <= 20 and 0 <= V <= 20:
                s = F + V
                if best is None or s > best:
                    best = s
        if best is not None:
            return float(best)
        return float(int(digits))  # fallback: treat all as one number

    # DTF14(Fwt%)
    if "DTF14" in e:
        tail = e.split("DTF14", 1)[1]
        digits = extract_digits(tail)
        if digits:
            return float(int(digits))

    # base electrolyte, no additives
    return 0.0


def compute_linewidth(total_additive: float, max_additive: float,
                      base_width: float = 1.5, max_width: float = 3.0) -> float:
    """
    Map total additive content → line width between base_width and max_width.
    """
    if max_additive <= 0:
        return base_width
    frac = max(total_additive, 0.0) / max_additive
    frac = max(0.0, min(1.0, frac))
    return base_width + frac * (max_width - base_width)


# ---------- generation & linestyle helpers ----------
# New alpha prefixes corresponding to the new repeat experiments (IP–IU, IV–IY)
NEW_ALPHA_PREFIXES = {
    "IV", "IW", "IX", "IY",  # new DTF repeats
    "IP", "IQ", "IR", "IS", "IT", "IU",  # new DTFV repeats
}


def get_alpha_prefix(code_or_alpha: str) -> str:
    """
    Return the first two characters of a cell code / alpha, e.g. 'HU01' -> 'HU'.
    """
    return str(code_or_alpha)[:2]


def is_new_alpha(alpha: str) -> bool:
    """
    True if an alpha (like 'HU' or 'IP') is part of the newer repeat set.
    """
    return get_alpha_prefix(alpha) in NEW_ALPHA_PREFIXES


def get_line_style_for_alpha(alpha: str) -> str:
    """
    Solid line for original / legacy data, dashed line for new repeat experiments.
    """
    return "--" if is_new_alpha(alpha) else "-"


# ---------- trial repeat helpers ----------
# Trials are defined by alpha prefix ranges:
#   Trial 1: up to and including FG
#   Trial 2: up to and including IF
#   Trial 3: anything after IF
TRIAL1_MAX_ALPHA = "FG"
TRIAL2_MAX_ALPHA = "IF"
TRIAL_LINESTYLES = {1: "-", 2: "--", 3: ":"}


def get_trial_number(alpha: str) -> int:
    a = get_alpha_prefix(alpha).upper()
    if a <= TRIAL1_MAX_ALPHA:
        return 1
    if a <= TRIAL2_MAX_ALPHA:
        return 2
    return 3


def get_line_style_for_trial_alpha(alpha: str) -> str:
    return TRIAL_LINESTYLES[get_trial_number(alpha)]


# ---------- marker helpers ----------
# Marker is set by the cell number suffix (e.g., HU01 -> 1, HU03 -> 3) so it's consistent everywhere.
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


def marker_for_cell_number(n):
    if n is None:
        return "o"
    try:
        n = int(n)
    except Exception:
        return "o"
    return CELLNUM_MARKERS.get(n, "o")


def marker_for_cell_code(cell_code: str) -> str:
    return marker_for_cell_number(get_cell_number(cell_code))



# ---------- alpha grouping ----------
def plot_groups_by_alpha(groups, lookup, out_dir, cell_meta, electrolyte_colors, max_additive):
    """
    Per-alpha grouped plots.
    - Color: by electrolyte (consistent across all plots via electrolyte_colors)
    - Linewidth: by total additive content (via cell_meta + max_additive)
    - Marker: per cell_code within each alpha (same marker for repeats)
    - Title: includes cathode|anode and electrolyte.
    """
    alpha_groups = defaultdict(list)
    for cell_code, paths in groups.items():
        alpha = cell_code[:2]
        for p in paths:
            alpha_groups[alpha].append((cell_code, p))

    os.makedirs(out_dir, exist_ok=True)

    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]

    for alpha, entries in sorted(alpha_groups.items()):
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted_any = False

        # unique cell codes in this alpha, assign marker per cell
        alpha_cells = sorted({cell_code for (cell_code, _) in entries})
        cell_marker = {
            cell_code: markers[i % len(markers)]
            for i, cell_code in enumerate(alpha_cells)
        }

        for cell_code, path in entries:
            try:
                x_spec, y_volt, _ = load_discharge_curve(path, lookup=lookup)
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue

            meta = cell_meta.get(cell_code, {})
            electrolyte = meta.get("electrolyte", "")
            total_add = meta.get("total_additive", 0.0)

            color = alpha_colors.get(cell_code[:2], "tab:gray")
            lw = compute_linewidth(total_add, max_additive)
            marker = cell_marker.get(cell_code, "o")

            if len(x_spec) > 1:
                markevery = max(len(x_spec) // 30, 1)
            else:
                markevery = 1

            label = get_display_label(cell_code, lookup)
            ax.plot(
                x_spec,
                y_volt,
                label=label,
                linewidth=lw,
                color=color,
                marker=marker,
                markersize=4,
                markevery=markevery,
            )
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        # Build group title: alpha group — Cathode|Anode — Electrolyte
        electrolyte_label = ""
        if alpha in lookup.index:
            row = lookup.loc[alpha]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            electrolyte_label = str(row.get("Electrolyte", "")).strip()

        anode_alpha, cathode_alpha = get_anode_cathode(alpha, lookup)
        chem_title = ""
        if cathode_alpha or anode_alpha:
            if cathode_alpha and anode_alpha:
                chem_title = f"{cathode_alpha}|{anode_alpha}"
            elif cathode_alpha:
                chem_title = cathode_alpha
            else:
                chem_title = anode_alpha

        title_parts = [f"{alpha} group"]
        if chem_title:
            title_parts.append(chem_title)
        if electrolyte_label:
            title_parts.append(electrolyte_label)

        ax.set_title(" — ".join(title_parts))
        ax.set_xlabel("Discharge Specific Capacity (mAh/g)")
        ax.set_ylabel("Voltage (V)")
        ax.set_ylim(0, 4.5)
        ax.set_xlim(-4, 160)
        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="xx-small", ncol=1, loc="best")

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{alpha}_group.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")


# ---------- DTF/DTFV grouped plots ----------
def plot_dtf_dtfv_groups(groups, lookup, out_dir, cell_meta, electrolyte_colors, max_additive):
    electrolyte_sets = {
        "DTF_new": ["HU", "HV", "HW", "HX", "IV", "IW", "IX", "IY"],
        "DTFV_new": ["IA", "IB", "IC", "ID", "IE", "IF",
                     "IP", "IQ", "IR", "IS", "IT", "IU"],

        "FEC_Mod": ["FA", "EC",
                    "HU", "HV", "HW", "HX",
                    "IV", "IW", "IX", "IY"],

        "FEC_1wtVC": ["FA", "EC",
                      "IA", "IB",
                      "IP", "IQ"],

        "FEC_2wtVC": ["FA", "EC",
                      "IC", "ID", "IE", "IF",
                      "IR", "IS", "IT", "IU"],

        "VC_1wtFEC": ["FA", "EC",
                      "IA", "ID",
                      "IP", "IS"],

        "VC_2wtFEC": ["FA", "EC",
                      "IC", "ID",
                      "IR", "IS"],

        "2wtAdd": ["FA", "EC",
                   "HV",
                   "IA", "IB", "IC",
                   "IP", "IQ", "IR", "IW"],
    }

    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]

    os.makedirs(out_dir, exist_ok=True)

    for group_name, alpha_list in electrolyte_sets.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted_any = False

        labeled_cells = set()

        # get cathode/anode representative for this group
        group_anode, group_cathode = "", ""
        for alpha in alpha_list:
            group_anode, group_cathode = get_anode_cathode(alpha, lookup)
            if group_anode or group_cathode:
                break

        for alpha in alpha_list:
            alpha_cells = sorted(c for c in groups.keys() if c.startswith(alpha))
            if not alpha_cells:
                continue

            # solid for old, dashed for new based on alpha prefix
            ls = get_line_style_for_alpha(alpha)

            for c_idx, cell_code in enumerate(alpha_cells):
                paths = groups[cell_code]
                marker = markers[c_idx % len(markers)]

                meta = cell_meta.get(cell_code, {})
                electrolyte = meta.get("electrolyte", "")
                total_add = meta.get("total_additive", 0.0)
                color = alpha_colors.get(cell_code[:2], "tab:gray")
                lw = compute_linewidth(total_add, max_additive)

                if cell_code not in labeled_cells:
                    cell_label = get_display_label(cell_code, lookup)
                    labeled_cells.add(cell_code)
                else:
                    cell_label = None

                for p in sorted(paths):
                    try:
                        x_spec, y_volt, _ = load_discharge_curve(p, lookup=lookup)
                    except Exception as exc:
                        print(f"Skipping {p} in {group_name} plot: {exc}")
                        continue

                    markevery = max(len(x_spec) // 30, 1) if len(x_spec) > 1 else 1

                    ax.plot(
                        x_spec,
                        y_volt,
                        color=color,
                        marker=marker,
                        linestyle=ls,
                        linewidth=lw,
                        markersize=4,
                        markevery=markevery,
                        label=cell_label,
                    )
                    cell_label = None
                    plotted_any = True

        if not plotted_any:
            plt.close(fig)
            print(f"No valid data for {group_name}, skipping plot.")
            continue

        chem_title = ""
        if group_cathode or group_anode:
            if group_cathode and group_anode:
                chem_title = f"{group_cathode}|{group_anode}"
            elif group_cathode:
                chem_title = group_cathode
            else:
                chem_title = group_anode

        title_parts = [f"{group_name} group"]
        # if chem_title:
        #     title_parts.append(chem_title)
        # title_parts.append("DTF/DTFV discharge")

        ax.set_title(" — ".join(title_parts))
        ax.set_xlabel("Discharge Specific Capacity (mAh/g)")
        ax.set_ylabel("Voltage (V)")
        ax.set_ylim(0, 4.5)
        ax.set_xlim(-4, 160)
        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="xx-small", ncol=2, loc="best")

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{group_name}_DTF_DTFV_group.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")


# ---------- electrolyte-wise old vs new comparison ----------
def plot_old_vs_new_comparisons(groups, lookup, out_dir, cell_meta, electrolyte_colors, max_additive):
    """
    For each electrolyte that appears in both 'old' and 'new' alpha prefixes,
    make a comparison plot overlaying all old vs all new cells.

    - Solid line: old data (including legacy FA/EC, HU–HX, IA–IF, etc.)
    - Dashed line: new repeat data (IV–IY, IP–IU)
    - Color: one color per electrolyte (from electrolyte_colors)
    """
    by_electrolyte = defaultdict(lambda: {"old": [], "new": []})

    for cell_code, paths in groups.items():
        alpha = get_alpha_prefix(cell_code)
        meta = cell_meta.get(cell_code, {})
        electrolyte = meta.get("electrolyte", "").strip()
        if not electrolyte:
            continue
        generation = "new" if is_new_alpha(alpha) else "old"
        by_electrolyte[electrolyte][generation].append((cell_code, paths))

    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]

    for electrolyte, gens in by_electrolyte.items():
        old_entries = gens["old"]
        new_entries = gens["new"]
        if not old_entries or not new_entries:
            continue  # need both to compare

        fig, ax = plt.subplots(figsize=(8, 6))
        plotted_any = False
        used_labels = set()

        color = alpha_colors.get(cell_code[:2], "tab:gray")

        def plot_group(entries, linestyle, group_label):
            nonlocal plotted_any
            for idx, (cell_code, paths) in enumerate(sorted(entries, key=lambda x: x[0])):
                marker = markers[idx % len(markers)]
                display_label = get_display_label(cell_code, lookup)
                # display_label is already "electrolyte-# (alpha)"
                label = f"{display_label} ({group_label})"
                if label in used_labels:
                    label = None
                else:
                    used_labels.add(label)

                for p in sorted(paths):
                    try:
                        x_spec, y_volt, _ = load_discharge_curve(p, lookup=lookup)
                    except Exception as exc:
                        print(f"Skipping {p} in comparison for {electrolyte}: {exc}")
                        continue

                    markevery = max(len(x_spec) // 30, 1) if len(x_spec) > 1 else 1

                    ax.plot(
                        x_spec,
                        y_volt,
                        color=color,
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=compute_linewidth(
                            cell_meta.get(cell_code, {}).get("total_additive", 0.0),
                            max_additive,
                        ),
                        markersize=4,
                        markevery=markevery,
                        label=label,
                    )
                    label = None
                    plotted_any = True

        # Old (solid) then new (dashed)
        plot_group(old_entries, "-", "old")
        plot_group(new_entries, "--", "new")

        if not plotted_any:
            plt.close(fig)
            continue

        # Build title using cathode|anode if available
        all_cell_codes = [c for c, _ in old_entries + new_entries]
        rep_code = all_cell_codes[0]
        anode, cathode = get_anode_cathode(rep_code, lookup)
        chem_title = ""
        if cathode or anode:
            if cathode and anode:
                chem_title = f"{cathode}|{anode}"
            elif cathode:
                chem_title = cathode
            else:
                chem_title = anode

        title_parts = [electrolyte, "old vs new"]
        if chem_title:
            title_parts.append(chem_title)

        ax.set_title(" — ".join(title_parts))
        ax.set_xlabel("Discharge Specific Capacity (mAh/g)")
        ax.set_ylabel("Voltage (V)")
        ax.set_ylim(0, 4.5)
        ax.set_xlim(-4, 160)
        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="xx-small", ncol=2, loc="best")

        plt.tight_layout()
        safe_name = (
            electrolyte.replace("/", "_")
                       .replace(" ", "_")
                       .replace(":", "_")
        )
        out_path = os.path.join(out_dir, f"{safe_name}_old_vs_new.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved comparison plot: {out_path}")


# ---------- CSV index of old vs new comparisons ----------
def make_old_new_index(groups, lookup, cell_meta, out_dir):
    """
    Build a CSV index that summarizes, for each electrolyte, which alphas/cells
    are contributing to the 'old' and 'new' sets, and whether FA/EC are present.

    Columns:
      - Electrolyte
      - OldAlphas
      - NewAlphas
      - Has_FA
      - Has_EC
      - N_OldCells
      - N_NewCells
    """
    by_electrolyte = defaultdict(lambda: {
        "old_alphas": set(),
        "new_alphas": set(),
        "has_FA": False,
        "has_EC": False,
        "old_cells": set(),
        "new_cells": set(),
    })

    for cell_code, paths in groups.items():
        alpha = get_alpha_prefix(cell_code)
        meta = cell_meta.get(cell_code, {})
        electrolyte = meta.get("electrolyte", "").strip()
        if not electrolyte:
            continue

        gen = "new" if is_new_alpha(alpha) else "old"
        entry = by_electrolyte[electrolyte]

        if gen == "new":
            entry["new_alphas"].add(alpha)
            entry["new_cells"].add(cell_code)
        else:
            entry["old_alphas"].add(alpha)
            entry["old_cells"].add(cell_code)

        if cell_code.startswith("FA"):
            entry["has_FA"] = True
        if cell_code.startswith("EC"):
            entry["has_EC"] = True

    records = []
    for electrolyte, d in sorted(by_electrolyte.items()):
        if not d["old_alphas"] and not d["new_alphas"]:
            continue
        records.append({
            "Electrolyte": electrolyte,
            "OldAlphas": ", ".join(sorted(d["old_alphas"])),
            "NewAlphas": ", ".join(sorted(d["new_alphas"])),
            "Has_FA": d["has_FA"],
            "Has_EC": d["has_EC"],
            "N_OldCells": len(d["old_cells"]),
            "N_NewCells": len(d["new_cells"]),
        })

    if not records:
        print("No old/new comparison index produced (no electrolytes with data).")
        return

    df_index = pd.DataFrame(records)
    index_path = os.path.join(out_dir, "old_new_comparison_index.csv")
    df_index.to_csv(index_path, index=False)
    print(f"Saved old/new comparison index: {index_path}")
    print(df_index)


# ---------- summary table ----------
def make_dtf_dtfv_summary(groups, out_dir, target_voltage=2.5):
    electrolyte_sets = {
        "DTF_new": ["HU", "HV", "HW", "HX", "IV", "IW", "IX", "IY"],
        "DTFV_new": ["IA", "IB", "IC", "ID", "IE", "IF",
                     "IP", "IQ", "IR", "IS", "IT", "IU"],

        "FEC_Mod": ["FA", "EC",
                    "HU", "HV", "HW", "HX",
                    "IV", "IW", "IX", "IY"],

        "FEC_1wtVC": ["FA", "EC",
                      "IA", "IB",
                      "IP", "IQ"],

        "FEC_2wtVC": ["FA", "EC",
                      "IC", "ID", "IE", "IF",
                      "IR", "IS", "IT", "IU"],

        "VC_1wtFEC": ["FA", "EC",
                      "IA", "ID",
                      "IP", "IS"],

        "VC_2wtFEC": ["FA", "EC",
                      "IC", "ID",
                      "IR", "IS"],

        "2wtAdd": ["FA", "EC",
                   "HV",
                   "IA", "IB", "IC",
                   "IP", "IQ", "IR", "IW"],
    }

    rows = []
    for group_name, alpha_list in electrolyte_sets.items():
        for alpha in alpha_list:
            alpha_cells = sorted(c for c in groups.keys() if c.startswith(alpha))
            for cell_code in alpha_cells:
                x, y = aggregate_discharge_for_cell(groups[cell_code])
                if x is None or len(x) < 2:
                    continue
                Q = x
                V = y
                Q0, Q1 = Q[0], Q[-1]
                dQ = Q1 - Q0
                if dQ <= 0:
                    avg_V = np.nan
                else:
                    energy = np.trapezoid(V, Q)  # modern NumPy syntax
                    avg_V = energy / dQ

                cap_at_target = np.nan
                for i in range(len(Q) - 1):
                    v1, v2 = V[i], V[i + 1]
                    if (v1 >= target_voltage and v2 <= target_voltage) or (v1 <= target_voltage and v2 >= target_voltage):
                        if v1 == v2:
                            cap_at_target = Q[i]
                        else:
                            frac = (target_voltage - v1) / (v2 - v1)
                            cap_at_target = Q[i] + frac * (Q[i + 1] - Q[i])
                        break

                rows.append({
                    "ElectrolyteGroup": group_name,
                    "Alpha": alpha,
                    "CellCode": cell_code,
                    "Capacity_at_{:.1f}V_mAh_g".format(target_voltage): cap_at_target,
                    "AverageDischargeVoltage_V": avg_V,
                })

    if not rows:
        print("No DTF/DTFV summary data produced.")
        return

    df_summary = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, "DTF_DTFV_discharge_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved summary table: {summary_path}")
    print(df_summary)
# python
def make_alpha_best_summary(groups, lookup, cell_meta, out_dir, target_voltage=2.0):
    """
    For each alpha (first two chars of cell code) find the cell with the
    largest Specific Discharge Capacity at `target_voltage` (mAh/g),
    and write an Excel sheet summarizing:
      - Cell ID
      - Electrolyte
      - Anode
      - Cathode
      - Specific Discharge Capacity at 2V
    """
    rows = []
    alphas = sorted({get_alpha_prefix(c) for c in groups.keys()})
    for alpha in alphas:
        alpha_cells = sorted(c for c in groups.keys() if c.startswith(alpha))
        best_cell = None
        best_cap = -np.inf

        for cell_code in alpha_cells:
            x, y = aggregate_discharge_for_cell(groups[cell_code], lookup=lookup)
            if x is None or len(x) < 2:
                continue

            cap_at_target = np.nan
            # interpolate capacity (x) at target voltage using (y = voltage, x = capacity)
            for i in range(len(x) - 1):
                v1, v2 = y[i], y[i + 1]
                if (v1 >= target_voltage and v2 <= target_voltage) or (v1 <= target_voltage and v2 >= target_voltage):
                    if v1 == v2:
                        cap_at_target = x[i]
                    else:
                        frac = (target_voltage - v1) / (v2 - v1)
                        cap_at_target = x[i] + frac * (x[i + 1] - x[i])
                    break

            if np.isnan(cap_at_target):
                continue

            if cap_at_target > best_cap:
                best_cap = cap_at_target
                best_cell = cell_code

        if best_cell is not None:
            electrolyte = cell_meta.get(best_cell, {}).get("electrolyte", get_electrolyte_name(best_cell, lookup))
            anode, cathode = get_anode_cathode(best_cell, lookup)
            rows.append({
                "Cell ID": best_cell,
                "Electrolyte": electrolyte,
                "Anode": anode,
                "Cathode": cathode,
                "Specific Discharge Capacity at 2V (mAh/g)": best_cap,
            })

    if not rows:
        print("No best-performance rows found for any alpha.")
        return

    df_best = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "alpha_best_performance.xlsx")
    df_best.to_excel(out_path, index=False)
    print(f"Saved alpha best-performance summary: {out_path}")
    print(df_best)

# --- main ---

# ---------- FEC x VC parsing ----------
def parse_fec_vc(electrolyte: str, fec_levels=(1, 2, 5, 10), vc_levels=(0, 1, 2)):
    '''
    Parse (FEC_wt%, VC_wt%) from electrolyte naming conventions:
      - DTF14(F)        -> (F, 0)
      - DTFV14(FV)      -> (F, V)  where FV is concatenated digits (e.g. 14102 => F=10,V=2)

    Returns (fec, vc) as ints, or (None, None) if it can't be parsed.
    '''
    if not electrolyte:
        return None, None
    e = str(electrolyte).upper()

    def extract_digits(s: str) -> str:
        return "".join(ch for ch in s if ch.isdigit())

    # DTFV14(F,V)
    if "DTFV14" in e:
        tail = e.split("DTFV14", 1)[1]
        digits = extract_digits(tail)
        if not digits or len(digits) < 2:
            return None, None

        best = None  # (score, F, V)
        for i in range(1, len(digits)):
            try:
                F = int(digits[:i])
                V = int(digits[i:])
            except ValueError:
                continue
            if not (0 <= F <= 20 and 0 <= V <= 20):
                continue

            # Prefer splits that land exactly on the grid levels
            score = 0
            if F in fec_levels:
                score += 1000
            if V in vc_levels:
                score += 1000

            # tie-breaker: higher additive sum (consistent with earlier heuristic)
            score += (F + V)

            # slight preference for longer F chunk (helps pick 10 over 1 when ambiguous)
            score += i * 0.01

            if best is None or score > best[0]:
                best = (score, F, V)

        if best is None:
            return None, None
        return best[1], best[2]

    # DTF14(F)
    if "DTF14" in e:
        tail = e.split("DTF14", 1)[1]
        digits = extract_digits(tail)
        if digits:
            try:
                return int(digits), 0
            except ValueError:
                return None, None

    return None, None


def _capacity_at_voltage(Q_mAh_g, V, target_voltage=2.0):
    '''
    Return interpolated capacity Q at the first crossing of target_voltage during discharge.
    If no crossing found, returns np.nan.
    '''
    if Q_mAh_g is None or V is None or len(Q_mAh_g) < 2 or len(V) < 2:
        return np.nan

    Q = np.asarray(Q_mAh_g, dtype=float)
    VV = np.asarray(V, dtype=float)

    for i in range(len(Q) - 1):
        v1, v2 = VV[i], VV[i + 1]
        if (v1 >= target_voltage and v2 <= target_voltage) or (v1 <= target_voltage and v2 >= target_voltage):
            if v1 == v2:
                return float(Q[i])
            frac = (target_voltage - v1) / (v2 - v1)
            return float(Q[i] + frac * (Q[i + 1] - Q[i]))

    return np.nan

def _voltage_at_capacity(Q_mAh_g, V, target_capacity):
    '''
    Return interpolated voltage V at a given discharge capacity Q (mAh/g).
    If target_capacity is outside the available Q range, returns np.nan.
    '''
    if Q_mAh_g is None or V is None or len(Q_mAh_g) < 2 or len(V) < 2:
        return np.nan

    Q = np.asarray(Q_mAh_g, dtype=float)
    VV = np.asarray(V, dtype=float)

    # ensure increasing Q for interpolation
    order = np.argsort(Q)
    Q = Q[order]
    VV = VV[order]

    if not (np.isfinite(target_capacity) and np.isfinite(Q[0]) and np.isfinite(Q[-1])):
        return np.nan
    if target_capacity < Q[0] or target_capacity > Q[-1]:
        return np.nan

    try:
        return float(np.interp(target_capacity, Q, VV))
    except Exception:
        return np.nan



# ---------- single-figure master grid ----------
def plot_fec_vc_master_grid(groups, lookup, out_dir, cell_meta, alpha_colors, max_additive,
                            fec_levels=(1, 2, 5, 10), vc_levels=(0, 1, 2),
                            alpha_select=None, selection="all",
                            reference_alphas=("FA", "EC"),
                            best_voltage=2.0, show_markers=True):
    '''
    Create ONE figure containing a grid of subplots:
      rows = FEC wt% (ascending), columns = VC wt% (ascending).
    Each panel overlays curves for all matching cells in that (FEC,VC) bin.

    Options:
      - alpha_select: None (all), "HU" (one alpha), or "best" (best cell per bin across all alphas)
      - selection: "all" (overlay all replicates), "best" (best cell per bin; still overlays its replicates)
      - reference_alphas: alpha prefixes always plotted in every panel (FA/EC)
      - Legends show cell codes.
    '''
    os.makedirs(out_dir, exist_ok=True)

    # Normalize alpha_select / selection
    if isinstance(alpha_select, str) and alpha_select.lower() == "best":
        selection = "best"
        alpha_select = None

    fec_levels = tuple(fec_levels)
    vc_levels = tuple(vc_levels)

    nrows, ncols = len(fec_levels), len(vc_levels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), sharex=True, sharey=True)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])
    # marker by cell number (suffix), consistent across all plots
    eligible_cells = sorted(groups.keys())
    cell_marker = {cc: marker_for_cell_code(cc) for cc in eligible_cells}

    ref_set = set(reference_alphas)
    ref_cells = sorted([cc for cc in groups.keys() if cc[:2] in ref_set])

    def include_cell(cc: str) -> bool:
        if cc in ref_cells:
            return False
        if alpha_select is None:
            return True
        return cc.startswith(str(alpha_select))

    # Build bins: (fec, vc) -> list of cell_codes
    bin_cells = {(f, v): [] for f in fec_levels for v in vc_levels}
    for cc in groups.keys():
        if not include_cell(cc):
            continue
        meta = cell_meta.get(cc, {})
        electrolyte = meta.get("electrolyte", "")
        fec, vc = parse_fec_vc(electrolyte, fec_levels=fec_levels, vc_levels=vc_levels)
        if fec in fec_levels and vc in vc_levels:
            bin_cells[(fec, vc)].append(cc)

    # Select best per bin if requested (by max capacity@best_voltage across replicates)
    if str(selection).lower() == "best":
        best_bin_cells = {(f, v): [] for f in fec_levels for v in vc_levels}
        for (fec, vc), cells in bin_cells.items():
            best_cc = None
            best_score = -np.inf
            for cc in cells:
                max_cap = -np.inf
                for p in sorted(groups.get(cc, [])):
                    try:
                        Q, V, _ = load_discharge_curve(p, lookup=lookup)
                    except Exception:
                        continue
                    cap = _capacity_at_voltage(Q, V, target_voltage=best_voltage)
                    if np.isfinite(cap):
                        max_cap = max(max_cap, cap)
                if max_cap > best_score:
                    best_score = max_cap
                    best_cc = cc

            if best_cc is not None and np.isfinite(best_score):
                best_bin_cells[(fec, vc)] = [best_cc]
        bin_cells = best_bin_cells

    # Plot each panel
    for r, fec in enumerate(fec_levels):
        for c, vc in enumerate(vc_levels):
            ax = axes[r, c]
            ax.set_title(f"FEC {fec} wt%  |  VC {vc} wt%")

            plotted_any = False
            labeled_in_panel = set()

            # References in every panel
            for ref_cc in ref_cells:
                paths = groups.get(ref_cc, [])
                meta = cell_meta.get(ref_cc, {})
                electrolyte = meta.get("electrolyte", "")
                total_add = meta.get("total_additive", 0.0)

                color = alpha_colors.get(ref_cc[:2], "tab:gray")
                lw = compute_linewidth(total_add, max_additive)
                marker = cell_marker.get(ref_cc, "o")
                ls = get_line_style_for_trial_alpha(ref_cc[:2])

                label = ref_cc if ref_cc not in labeled_in_panel else None
                for p in sorted(paths):
                    try:
                        Q, V, _ = load_discharge_curve(p, lookup=lookup)
                    except Exception:
                        continue
                    markevery = max(len(Q) // 30, 1) if len(Q) > 1 else 1
                    ax.plot(
                        Q, V,
                        color=color,
                        linestyle=ls,
                        linewidth=lw,
                        marker=(marker if show_markers else None),
                        markersize=4,
                        markevery=markevery,
                        label=label,
                        alpha=0.85,
                    )
                    label = None
                    labeled_in_panel.add(ref_cc)
                    plotted_any = True

            # Bin cells
            for cc in sorted(bin_cells[(fec, vc)]):
                paths = groups.get(cc, [])
                if not paths:
                    continue

                meta = cell_meta.get(cc, {})
                electrolyte = meta.get("electrolyte", "")
                total_add = meta.get("total_additive", 0.0)

                color = alpha_colors.get(cc[:2], "tab:gray")
                lw = compute_linewidth(total_add, max_additive)
                marker = cell_marker.get(cc, "o")
                ls = get_line_style_for_trial_alpha(cc[:2])

                label = cc if cc not in labeled_in_panel else None
                for p in sorted(paths):
                    try:
                        Q, V, _ = load_discharge_curve(p, lookup=lookup)
                    except Exception:
                        continue

                    markevery = max(len(Q) // 30, 1) if len(Q) > 1 else 1

                    ax.plot(
                        Q, V,
                        color=color,
                        linestyle=ls,
                        linewidth=lw,
                        marker=(marker if show_markers else None),
                        markersize=4,
                        markevery=markevery,
                        label=label,
                    )
                    label = None
                    labeled_in_panel.add(cc)
                    plotted_any = True

            ax.set_ylim(0, 4.5)
            ax.set_xlim(-4, 160)
            ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize="xx-small", ncol=1, loc="best")

            if not plotted_any:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=10)

    for ax in axes[-1, :]:
        ax.set_xlabel("Discharge Specific Capacity (mAh/g)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Voltage (V)")

    plt.tight_layout()

    tag = "ALL" if alpha_select is None else str(alpha_select)
    sel_tag = str(selection).upper()
    out_path = os.path.join(out_dir, f"FECxVC_master_{tag}_{sel_tag}.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved master grid: {out_path}")



# ---------- per-cell discharge metrics + scatter plots ----------
def compute_cell_discharge_metrics(groups, lookup, cell_meta,
                                   fec_levels=(1, 2, 5, 10), vc_levels=(0, 1, 2),
                                   capacity_voltage=2.5):
    """
    Build a per-cell metrics table for scatter/summary plots.

    Metrics:
      - InitialDischargeV_V: first voltage point at start of discharge (after sorting by capacity)
      - Capacity_at_{capacity_voltage}V_mAh_g: capacity at the specified voltage (interpolated)
      - AvgDischargeV_V: average discharge voltage (∫V dQ / ∫dQ)
      - Trial: from alpha prefix ranges (<=FG, <=IF, >IF)
      - CellNumber: numeric suffix from the cell code (e.g., HU01 -> 1)
      - FEC_wt / VC_wt: parsed from electrolyte string when possible (DTF/DTFV naming)
    """
    rows = []
    for cell_code, paths in sorted(groups.items()):
        if not paths:
            continue

        alpha = get_alpha_prefix(cell_code)
        trial = get_trial_number(alpha)
        cell_num = get_cell_number(cell_code)

        meta = cell_meta.get(cell_code, {})
        electrolyte = meta.get("electrolyte", "")

        # parse condition (may be None/None for FA/EC etc.)
        fec_wt, vc_wt = parse_fec_vc(electrolyte, fec_levels=fec_levels, vc_levels=vc_levels)

        Q, V = aggregate_discharge_for_cell(paths, lookup=lookup)
        if Q is None or V is None or len(Q) < 2:
            continue

        init_v = float(V[0]) if len(V) else np.nan
        cap_at_v = _capacity_at_voltage(Q, V, target_voltage=capacity_voltage)

        dQ = float(Q[-1] - Q[0])
        if dQ > 0:
            avg_v = float(np.trapezoid(V, Q) / dQ)
            mid_q = float(Q[0] + 0.5 * dQ)
            mid_v = _voltage_at_capacity(Q, V, mid_q)
        else:
            avg_v = np.nan
            mid_q = np.nan
            mid_v = np.nan

        rows.append({
            "CellCode": cell_code,
            "Alpha": alpha,
            "Trial": trial,
            "CellNumber": cell_num,
            "Electrolyte": electrolyte,
            "FEC_wt": fec_wt,
            "VC_wt": vc_wt,
            "InitialDischargeV_V": init_v,
            f"Capacity_at_{capacity_voltage:.1f}V_mAh_g": cap_at_v,
            f"cap_at_{capacity_voltage:.1f}V": cap_at_v,
            "AvgDischargeV_V": avg_v,
            "MidpointDischargeV_V": mid_v,
            "MidpointDischargeQ_mAh_g": mid_q,
        })

    return pd.DataFrame(rows)


def plot_scatter_by_trial(df, alpha_colors, out_dir,
                          x_col, y_col, y_label, title, filename):
    """
    Make a 1x3 subplot scatter (Trial 1/2/3).
      - Point color: alpha
      - Point marker: cell number
      - Per-alpha vertical error bars: ± 1 SD in y (computed across cells in that alpha, within that trial)
      - Legends are placed underneath the plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    trials = [1, 2, 3]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)

    # plot points + per-alpha y-std bars
    for ax, t in zip(axes, trials):
        dft = df[df["Trial"] == t].copy()
        ax.set_title(f"Trial {t}")

        # individual points
        for _, row in dft.iterrows():
            alpha = row.get("Alpha", "")
            color = alpha_colors.get(alpha, "tab:gray")
            marker = marker_for_cell_number(row.get("CellNumber"))
            ax.scatter(
                row.get(x_col, np.nan),
                row.get(y_col, np.nan),
                color=color,
                marker=marker,
                s=55,
                alpha=0.9,
            )

        # per-alpha vertical 1σ bars in y
        for alpha, g in dft.groupby("Alpha"):
            x_vals = pd.to_numeric(g[x_col], errors="coerce").to_numpy()
            y_vals = pd.to_numeric(g[y_col], errors="coerce").to_numpy()
            x_mean = np.nanmean(x_vals) if len(x_vals) else np.nan
            y_mean = np.nanmean(y_vals) if len(y_vals) else np.nan
            if len(y_vals) >= 2:
                y_std = np.nanstd(y_vals, ddof=1)
            else:
                y_std = np.nan

            if np.isfinite(x_mean) and np.isfinite(y_mean) and np.isfinite(y_std) and y_std > 0:
                ax.errorbar(
                    [x_mean], [y_mean],
                    yerr=[y_std],
                    fmt="none",
                    ecolor=alpha_colors.get(alpha, "tab:gray"),
                    elinewidth=1.6,
                    capsize=4,
                    alpha=0.9,
                )

        ax.set_xlabel("Initial discharge potential (V)")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(y_label)
    fig.suptitle(title, y=0.98)

    # ---- legends underneath ----
    from matplotlib.lines import Line2D

    # alpha color legend (only alphas present in the df)
    present_alphas = sorted([a for a in df["Alpha"].dropna().unique() if str(a).strip()])
    alpha_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markersize=6, color=alpha_colors.get(a, "tab:gray"), label=a)
        for a in present_alphas
    ]

    # marker legend (only numbers present)
    present_nums = sorted([n for n in df["CellNumber"].dropna().unique() if pd.notna(n)])
    marker_handles = [
        Line2D([0], [0], marker=marker_for_cell_number(n), linestyle="None",
               markersize=7, color="black", label=str(int(n)))
        for n in present_nums
    ]

    # Leave room for legends
    fig.subplots_adjust(bottom=0.28)

    if alpha_handles:
        leg1 = fig.legend(
            handles=alpha_handles,
            title="Alpha (color)",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.10),
            ncol=min(10, max(1, len(alpha_handles))),
            fontsize="x-small",
            title_fontsize="small",
            frameon=False,
        )
        fig.add_artist(leg1)

    if marker_handles:
        fig.legend(
            handles=marker_handles,
            title="Cell # (marker)",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(12, max(1, len(marker_handles))),
            fontsize="x-small",
            title_fontsize="small",
            frameon=False,
        )

    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter: {out_path}")



# ---------- effect / trend plots ----------
def _make_alpha_legend_handles(df, alpha_colors):
    """Return handles/labels for alpha color legend using only alphas present in df."""
    from matplotlib.lines import Line2D
    present_alphas = sorted([a for a in df["Alpha"].dropna().unique() if str(a).strip()])
    handles = []
    labels = []
    for a in present_alphas:
        handles.append(Line2D([0], [0], marker="o", linestyle="None", color=alpha_colors.get(a, "tab:gray"), markersize=6))
        labels.append(str(a))
    return handles, labels


def _make_marker_legend_handles(df):
    """Return handles/labels for cell-number marker legend using only cell numbers present in df."""
    from matplotlib.lines import Line2D
    nums = sorted({int(n) for n in df["CellNumber"].dropna().unique() if str(n).strip().isdigit()})
    handles = []
    labels = []
    for n in nums:
        handles.append(Line2D([0], [0], marker=marker_for_cell_number(n), linestyle="None", color="black", markersize=6))
        labels.append(str(n))
    return handles, labels


def _make_trial_mean_handles():
    """Return handles/labels for the trial mean±1σ overlays."""
    from matplotlib.lines import Line2D
    handles = []
    labels = []
    for t in (1, 2, 3):
        handles.append(Line2D([0], [0], linestyle=TRIAL_LINESTYLES[t], color="black", linewidth=2))
        labels.append(f"Trial {t} mean ± 1σ")
    return handles, labels


def plot_effect_of_fec_shift(df_metrics, alpha_colors, out_dir,
                             y_col, y_label, title, filename,
                             fec_levels=(1, 2, 5, 10), vc_levels=(0, 1, 2),
                             overlay_trial_mean_sd=True):
    """
    Effect of FEC: for each VC (subplot), scatter per-cell metrics vs FEC.
      - point color: alpha
      - point marker: cell number
      - optional overlay: per-trial mean ± 1σ at each FEC (black, linestyle per trial)
    """
    os.makedirs(out_dir, exist_ok=True)

    df = df_metrics.copy()
    df["FEC_wt"] = pd.to_numeric(df["FEC_wt"], errors="coerce")
    df["VC_wt"] = pd.to_numeric(df["VC_wt"], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    df = df[df["FEC_wt"].isin(list(fec_levels)) & df["VC_wt"].isin(list(vc_levels))]
    df = df.dropna(subset=["FEC_wt", "VC_wt", y_col, "Alpha"])

    ncols = len(vc_levels)
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 5.2), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, vc in zip(axes, vc_levels):
        d = df[df["VC_wt"] == vc]
        ax.set_title(f"VC = {int(vc)} wt%")
        ax.set_xlabel("FEC (wt%)")
        ax.set_xticks(list(fec_levels))
        ax.grid(True, alpha=0.25)

        # scatter all points
        for _, row in d.iterrows():
            a = row["Alpha"]
            color = alpha_colors.get(a, "tab:gray")
            marker = marker_for_cell_number(row.get("CellNumber"))
            ax.scatter(row["FEC_wt"], row[y_col], color=color, marker=marker, s=55, alpha=0.9)

        # overlay trial mean ± 1σ
        if overlay_trial_mean_sd:
            for t in (1, 2, 3):
                dt = d[d["Trial"] == t]
                if dt.empty:
                    continue
                g = dt.groupby("FEC_wt")[y_col]
                means = g.mean()
                sds = g.std()
                xs = np.array(sorted(means.index.to_list()), dtype=float)
                ys = np.array([means.loc[x] for x in xs], dtype=float)
                es = np.array([sds.loc[x] if x in sds.index else np.nan for x in xs], dtype=float)
                ax.errorbar(xs, ys, yerr=es, color="black", linestyle=TRIAL_LINESTYLES[t],
                            marker="o", markersize=4, linewidth=2, capsize=3, alpha=0.85)

    axes[0].set_ylabel(y_label)
    fig.suptitle(title, y=0.98)

    # ---- legends underneath ----
    alpha_handles, alpha_labels = _make_alpha_legend_handles(df, alpha_colors)
    marker_handles, marker_labels = _make_marker_legend_handles(df)
    trial_handles, trial_labels = _make_trial_mean_handles()

    # place three legends stacked
    y0 = -0.02
    if alpha_handles:
        fig.legend(alpha_handles, alpha_labels, loc="upper center",
                   bbox_to_anchor=(0.5, y0), ncol=min(16, max(1, len(alpha_handles))),
                   fontsize="x-small", title="Alpha (color)", title_fontsize="small", frameon=False)
        y0 -= 0.08

    if marker_handles:
        fig.legend(marker_handles, marker_labels, loc="upper center",
                   bbox_to_anchor=(0.5, y0), ncol=min(12, max(1, len(marker_handles))),
                   fontsize="x-small", title="Cell # (marker)", title_fontsize="small", frameon=False)
        y0 -= 0.08

    if overlay_trial_mean_sd:
        fig.legend(trial_handles, trial_labels, loc="upper center",
                   bbox_to_anchor=(0.5, y0), ncol=3,
                   fontsize="x-small", title="Trial overlay", title_fontsize="small", frameon=False)

    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FEC-effect plot: {out_path}")


def plot_effect_of_vc_shift(df_metrics, alpha_colors, out_dir,
                            y_col, y_label, title, filename,
                            fec_levels=(1, 2, 5, 10), vc_levels=(0, 1, 2),
                            overlay_trial_mean_sd=True):
    """
    Effect of VC: for each FEC (subplot), scatter per-cell metrics vs VC.
      - point color: alpha
      - point marker: cell number
      - optional overlay: per-trial mean ± 1σ at each VC (black, linestyle per trial)
    """
    os.makedirs(out_dir, exist_ok=True)

    df = df_metrics.copy()
    df["FEC_wt"] = pd.to_numeric(df["FEC_wt"], errors="coerce")
    df["VC_wt"] = pd.to_numeric(df["VC_wt"], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    df = df[df["FEC_wt"].isin(list(fec_levels)) & df["VC_wt"].isin(list(vc_levels))]
    df = df.dropna(subset=["FEC_wt", "VC_wt", y_col, "Alpha"])

    # Layout: 2x2 if 4 FEC levels, otherwise 1xN
    if len(fec_levels) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), sharey=True, sharex=True)
        axes = axes.ravel()
    else:
        ncols = len(fec_levels)
        fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 5.2), sharey=True)
        if ncols == 1:
            axes = [axes]

    for ax, fec in zip(axes, fec_levels):
        d = df[df["FEC_wt"] == fec]
        ax.set_title(f"FEC = {int(fec)} wt%")
        ax.set_xlabel("VC (wt%)")
        ax.set_xticks(list(vc_levels))
        ax.grid(True, alpha=0.25)

        # scatter all points
        for _, row in d.iterrows():
            a = row["Alpha"]
            color = alpha_colors.get(a, "tab:gray")
            marker = marker_for_cell_number(row.get("CellNumber"))
            ax.scatter(row["VC_wt"], row[y_col], color=color, marker=marker, s=55, alpha=0.9)

        # overlay trial mean ± 1σ
        if overlay_trial_mean_sd:
            for t in (1, 2, 3):
                dt = d[d["Trial"] == t]
                if dt.empty:
                    continue
                g = dt.groupby("VC_wt")[y_col]
                means = g.mean()
                sds = g.std()
                xs = np.array(sorted(means.index.to_list()), dtype=float)
                ys = np.array([means.loc[x] for x in xs], dtype=float)
                es = np.array([sds.loc[x] if x in sds.index else np.nan for x in xs], dtype=float)
                ax.errorbar(xs, ys, yerr=es, color="black", linestyle=TRIAL_LINESTYLES[t],
                            marker="o", markersize=4, linewidth=2, capsize=3, alpha=0.85)

    # If 2x2, set left y-labels
    if isinstance(axes, (list, tuple, np.ndarray)):
        axes[0].set_ylabel(y_label)
    else:
        axes.set_ylabel(y_label)

    fig.suptitle(title, y=0.98)

    # ---- legends underneath ----
    alpha_handles, alpha_labels = _make_alpha_legend_handles(df, alpha_colors)
    marker_handles, marker_labels = _make_marker_legend_handles(df)
    trial_handles, trial_labels = _make_trial_mean_handles()

    y0 = -0.02
    if alpha_handles:
        fig.legend(alpha_handles, alpha_labels, loc="upper center",
                   bbox_to_anchor=(0.5, y0), ncol=min(16, max(1, len(alpha_handles))),
                   fontsize="x-small", title="Alpha (color)", title_fontsize="small", frameon=False)
        y0 -= 0.08

    if marker_handles:
        fig.legend(marker_handles, marker_labels, loc="upper center",
                   bbox_to_anchor=(0.5, y0), ncol=min(12, max(1, len(marker_handles))),
                   fontsize="x-small", title="Cell # (marker)", title_fontsize="small", frameon=False)
        y0 -= 0.08

    if overlay_trial_mean_sd:
        fig.legend(trial_handles, trial_labels, loc="upper center",
                   bbox_to_anchor=(0.5, y0), ncol=3,
                   fontsize="x-small", title="Trial overlay", title_fontsize="small", frameon=False)

    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved VC-effect plot: {out_path}")


# ---------- effect trends with discharge curves (split by Trial) ----------
def plot_effect_of_fec_shift_with_curves(df, groups, lookup, alpha_colors, out_dir,
                                        title_prefix="Effect of FEC Shift",
                                        y_col="cap_at_2.5V",
                                        y_label="Metric",
                                        capacity_voltage=CAPACITY_VOLTAGE,
                                        fec_levels=GRID_FEC_LEVELS,
                                        vc_levels=GRID_VC_LEVELS):
    """
    For each VC panel, plot y_col vs FEC (scatter + mean±sd) on the top row,
    and plot representative discharge curves (best by cap_at_{capacity_voltage}V) on the bottom row.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap_col = f"cap_at_{capacity_voltage:.1f}V"

    ncols = len(vc_levels)
    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(5 * ncols, 9), sharey="row")
    if ncols == 1:
        axes = np.array(axes).reshape(2, 1)

    # --- enforce identical axes across ALL top-row panels ---
    # x-limits based on provided FEC levels
    try:
        x0 = float(np.nanmin(list(fec_levels)))
        x1 = float(np.nanmax(list(fec_levels)))
        dx = 0.15 * (x1 - x0) if np.isfinite(x0) and np.isfinite(x1) and (x1 > x0) else 0.5
        top_xlim = (x0 - dx, x1 + dx)
    except Exception:
        top_xlim = None

    # y-limits based on all finite values of the selected metric
    y_all = df.get(y_col)
    if y_all is not None:
        y_all = np.asarray(y_all, dtype=float)
        y_all = y_all[np.isfinite(y_all)]
    else:
        y_all = np.array([], dtype=float)

    if y_all.size:
        y0 = float(np.nanmin(y_all))
        y1 = float(np.nanmax(y_all))
        dy = 0.08 * (y1 - y0) if (y1 > y0) else (0.05 * abs(y0) if y0 != 0 else 0.05)
        top_ylim = (y0 - dy, y1 + dy)
    else:
        top_ylim = None

    for j, vc in enumerate(vc_levels):
        col = df[df["VC_wt"] == vc].copy()

        # --- top row: scatter + mean±sd ---
        ax = axes[0, j]
        ax.set_title(f"VC {vc} wt%")
        ax.set_xlabel("FEC (wt%)")
        if j == 0:
            ax.set_ylabel(y_label)

        for _, r in col.iterrows():
            fec = r.get("FEC_wt", np.nan)
            yv = r.get(y_col, np.nan)
            if not (np.isfinite(fec) and np.isfinite(yv)):
                continue
            alpha = r.get("Alpha", "")
            cc = r.get("CellCode", "")
            ax.scatter(fec, yv,
                       color=alpha_colors.get(alpha, "tab:gray"),
                       marker=marker_for_cell_code(cc),
                       s=35, alpha=0.85)

        for fec in fec_levels:
            g = col[col["FEC_wt"] == fec][y_col].to_numpy(dtype=float)
            g = g[np.isfinite(g)]
            if len(g) >= 2:
                ax.errorbar(fec, float(np.mean(g)), yerr=float(np.std(g, ddof=1)),
                            fmt="k_", capsize=2, alpha=0.8)
            elif len(g) == 1:
                ax.scatter([fec], [float(g[0])], color="k", s=10)

        ax.set_xticks(list(fec_levels))
        if top_xlim is not None:
            ax.set_xlim(*top_xlim)
        if top_ylim is not None:
            ax.set_ylim(*top_ylim)
        ax.grid(True, alpha=0.2)

        # --- bottom row: representative curves (best per FEC) ---
        axc = axes[1, j]
        axc.set_xlabel("Discharge (mAh/g)")
        if j == 0:
            axc.set_ylabel("Voltage (V)")
        axc.set_ylim(0, 4.5)
        axc.set_xlim(-5, 180)
        axc.grid(True, alpha=0.2)

        for fec in fec_levels:
            cand = col[(col["FEC_wt"] == fec) & np.isfinite(col.get(cap_col))]
            if cand.empty:
                continue
            best = cand.sort_values(by=cap_col, ascending=False).iloc[0]
            cc = best["CellCode"]
            alpha = best["Alpha"]
            paths = groups.get(cc, [])
            if not paths:
                continue
            try:
                Q, V = aggregate_discharge_for_cell(paths, lookup)
            except Exception:
                continue
            axc.plot(Q, V,
                     color=alpha_colors.get(alpha, "tab:gray"),
                     linestyle=get_line_style_for_trial_alpha(alpha),
                     linewidth=2.0,
                     label=f"FEC {fec}: {cc}")

    fig.suptitle(title_prefix, y=0.98)

    # global legend from bottom row
    seen = set()
    handles, labels = [], []
    for j in range(ncols):
        h, l = axes[1, j].get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in seen:
                seen.add(ll)
                handles.append(hh); labels.append(ll)
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.03),
                   ncol=min(max(1, ncols * 4), 12),
                   fontsize="x-small", frameon=False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"fec_shift_{y_col}_byVC.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")



def plot_effect_of_fec_shift_cap_and_midV_with_curves(df, groups, lookup, alpha_colors, out_dir,
                                                     title_prefix="Effect of FEC Shift",
                                                     capacity_voltage=2.5,
                                                     cap_col=None,
                                                     cap_label=None,
                                                     v_col="MidpointDischargeV_V",
                                                     v_label="V @ 50% Q (V)",
                                                     fec_levels=GRID_FEC_LEVELS,
                                                     vc_levels=GRID_VC_LEVELS):
    """
    For each VC panel, plot discharge capacity on the LEFT y-axis and a voltage
    figure-of-merit on the RIGHT y-axis (top row), both vs FEC.

    Bottom row shows representative discharge curves (best by cap_col) for each FEC.

    Notes:
      - Capacity points are filled markers (colored by alpha).
      - Voltage points are the same markers but unfilled (hollow) on the right axis.
      - Black mean±sd is overlaid per x-level for both metrics.
    """
    os.makedirs(out_dir, exist_ok=True)

    if cap_col is None:
        cap_col = f"cap_at_{capacity_voltage:.1f}V"
    if cap_label is None:
        cap_label = f"Cap @{capacity_voltage:.1f} V (mAh/g)"

    # --- global axis limits so every panel is directly comparable ---
    def _padded_limits(vals, frac=0.08, min_pad=0.0):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        span = hi - lo
        if span <= 0:
            span = max(abs(hi), 1.0)
        pad = max(frac * span, min_pad)
        return (lo - pad, hi + pad)

    cap_ylim = _padded_limits(df[cap_col].to_numpy(dtype=float) if cap_col in df.columns else [],
                              frac=0.08, min_pad=1.0)
    v_ylim = _padded_limits(df[v_col].to_numpy(dtype=float) if v_col in df.columns else [],
                            frac=0.08, min_pad=0.01)

    fec_levels = list(fec_levels)
    if len(fec_levels) > 0:
        x0 = float(np.min(fec_levels))
        x1 = float(np.max(fec_levels))
        dx = max(0.10 * (x1 - x0), 0.5)
        top_xlim = (x0 - dx, x1 + dx)
    else:
        top_xlim = None


    ncols = len(vc_levels)
    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(5 * ncols, 9), sharey="row")

    for j, vc in enumerate(vc_levels):
        col = df[df["VC_wt"] == vc].copy()

        # --- top row: dual-axis scatter ---
        ax = axes[0, j]
        axr = ax.twinx()
        ax.set_title(f"VC {vc} wt%")

        if j == 0:
            ax.set_ylabel(cap_label)
        if j == ncols - 1:
            axr.set_ylabel(v_label)

        ax.set_xlabel("FEC (wt%)")
        ax.set_xticks(list(fec_levels))

        # scatter points
        for _, row in col.iterrows():
            a = row.get("Alpha")
            color = alpha_colors.get(a, "tab:gray")
            marker = marker_for_cell_number(row.get("CellNumber"))
            x = row.get("FEC_wt")

            y_cap = row.get(cap_col)
            if pd.notna(y_cap):
                ax.scatter(x, y_cap, color=color, marker=marker, s=55, alpha=0.9)

            y_v = row.get(v_col)
            if pd.notna(y_v):
                # hollow marker for voltage metric
                axr.scatter(x, y_v, facecolors="none", edgecolors=color,
                            marker=marker, s=55, alpha=0.8, linewidths=1.5)

        # overlay mean ± sd for both metrics at each FEC level
        for fec in fec_levels:
            g_cap = col[col["FEC_wt"] == fec][cap_col].to_numpy(dtype=float)
            g_cap = g_cap[np.isfinite(g_cap)]
            if len(g_cap) >= 2:
                ax.errorbar(fec, float(np.mean(g_cap)), yerr=float(np.std(g_cap, ddof=1)),
                            fmt="k_", capsize=2, alpha=0.8)
            elif len(g_cap) == 1:
                ax.scatter([fec], [float(g_cap[0])], color="k", s=10)

            g_v = col[col["FEC_wt"] == fec][v_col].to_numpy(dtype=float)
            g_v = g_v[np.isfinite(g_v)]
            if len(g_v) >= 2:
                axr.errorbar(fec, float(np.mean(g_v)), yerr=float(np.std(g_v, ddof=1)),
                             color="k", linestyle="--", linewidth=1.5,
                             marker="o", markersize=3, capsize=2, alpha=0.65)
            elif len(g_v) == 1:
                axr.scatter([fec], [float(g_v[0])], color="k", s=10, alpha=0.65)

        if top_xlim is not None:
            ax.set_xlim(*top_xlim)
        if cap_ylim is not None:
            ax.set_ylim(*cap_ylim)
        if v_ylim is not None:
            axr.set_ylim(*v_ylim)
        ax.grid(True, alpha=0.2)

        # --- bottom row: representative curves (best per FEC) ---
        axc = axes[1, j]
        axc.set_xlabel("Discharge (mAh/g)")
        if j == 0:
            axc.set_ylabel("Voltage (V)")
        axc.set_ylim(0, 4.5)
        axc.set_xlim(-5, 180)
        axc.grid(True, alpha=0.2)

        for fec in fec_levels:
            cand = col[(col["FEC_wt"] == fec) & np.isfinite(col.get(cap_col))]
            if cand.empty:
                continue
            best = cand.sort_values(by=cap_col, ascending=False).iloc[0]
            cc = best["CellCode"]
            alpha = best["Alpha"]
            paths = groups.get(cc, [])
            if not paths:
                continue
            try:
                Q, V = aggregate_discharge_for_cell(paths, lookup)
            except Exception:
                continue

            axc.plot(Q, V,
                     color=alpha_colors.get(alpha, "tab:gray"),
                     linestyle=get_line_style_for_trial_alpha(alpha),
                     linewidth=2.0,
                     label=f"FEC {fec}: {cc}")

    fig.suptitle(title_prefix, y=0.98)

    # global legend from bottom row
    seen = set()
    handles, labels = [], []
    for j in range(ncols):
        h, l = axes[1, j].get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in seen:
                seen.add(ll)
                handles.append(hh)
                labels.append(ll)
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.03),
                   ncol=min(max(1, ncols * 4), 12),
                   fontsize="x-small", frameon=False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fec_shift_cap_and_midV_byVC.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_effect_of_vc_shift_with_curves(df, groups, lookup, alpha_colors, out_dir,
                                       title_prefix="Effect of VC Shift",
                                       y_col="cap_at_2.5V",
                                       y_label="Metric",
                                       capacity_voltage=2.5,
                                       fec_levels=GRID_FEC_LEVELS,
                                       vc_levels=GRID_VC_LEVELS):
    """
    For each FEC panel, plot y_col vs VC (scatter + mean±sd) on the top row,
    and plot representative discharge curves (best by cap_at_{capacity_voltage}V) on the bottom row.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap_col = f"cap_at_{capacity_voltage:.1f}V"

    ncols = len(fec_levels)
    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(5 * ncols, 9), sharey="row")
    if ncols == 1:
        axes = np.array(axes).reshape(2, 1)

    # --- enforce identical axes across ALL top-row panels ---
    # x-limits based on provided VC levels
    try:
        x0 = float(np.nanmin(list(vc_levels)))
        x1 = float(np.nanmax(list(vc_levels)))
        dx = 0.15 * (x1 - x0) if np.isfinite(x0) and np.isfinite(x1) and (x1 > x0) else 0.5
        top_xlim = (x0 - dx, x1 + dx)
    except Exception:
        top_xlim = None

    # y-limits based on all finite values of the selected metric
    y_all = df.get(y_col)
    if y_all is not None:
        y_all = np.asarray(y_all, dtype=float)
        y_all = y_all[np.isfinite(y_all)]
    else:
        y_all = np.array([], dtype=float)

    if y_all.size:
        y0 = float(np.nanmin(y_all))
        y1 = float(np.nanmax(y_all))
        dy = 0.08 * (y1 - y0) if (y1 > y0) else (0.05 * abs(y0) if y0 != 0 else 0.05)
        top_ylim = (y0 - dy, y1 + dy)
    else:
        top_ylim = None

    for j, fec in enumerate(fec_levels):
        col = df[df["FEC_wt"] == fec].copy()

        # --- top row ---
        ax = axes[0, j]
        ax.set_title(f"FEC {fec} wt%")
        ax.set_xlabel("VC (wt%)")
        if j == 0:
            ax.set_ylabel(y_label)

        for _, r in col.iterrows():
            vc = r.get("VC_wt", np.nan)
            yv = r.get(y_col, np.nan)
            if not (np.isfinite(vc) and np.isfinite(yv)):
                continue
            alpha = r.get("Alpha", "")
            cc = r.get("CellCode", "")
            ax.scatter(vc, yv,
                       color=alpha_colors.get(alpha, "tab:gray"),
                       marker=marker_for_cell_code(cc),
                       s=35, alpha=0.85)

        for vc in vc_levels:
            g = col[col["VC_wt"] == vc][y_col].to_numpy(dtype=float)
            g = g[np.isfinite(g)]
            if len(g) >= 2:
                ax.errorbar(vc, float(np.mean(g)), yerr=float(np.std(g, ddof=1)),
                            fmt="k_", capsize=2, alpha=0.8)
            elif len(g) == 1:
                ax.scatter([vc], [float(g[0])], color="k", s=10)

        ax.set_xticks(list(vc_levels))
        if top_xlim is not None:
            ax.set_xlim(*top_xlim)
        if top_ylim is not None:
            ax.set_ylim(*top_ylim)
        ax.grid(True, alpha=0.2)

        # --- bottom row curves ---
        axc = axes[1, j]
        axc.set_xlabel("Discharge (mAh/g)")
        if j == 0:
            axc.set_ylabel("Voltage (V)")
        axc.set_ylim(0, 4.5)
        axc.set_xlim(-5, 180)
        axc.grid(True, alpha=0.2)

        for vc in vc_levels:
            cand = col[(col["VC_wt"] == vc) & np.isfinite(col.get(cap_col))]
            if cand.empty:
                continue
            best = cand.sort_values(by=cap_col, ascending=False).iloc[0]
            cc = best["CellCode"]
            alpha = best["Alpha"]
            paths = groups.get(cc, [])
            if not paths:
                continue
            try:
                Q, V = aggregate_discharge_for_cell(paths, lookup)
            except Exception:
                continue
            axc.plot(Q, V,
                     color=alpha_colors.get(alpha, "tab:gray"),
                     linestyle=get_line_style_for_trial_alpha(alpha),
                     linewidth=2.0,
                     label=f"VC {vc}: {cc}")

    fig.suptitle(title_prefix, y=0.98)

    # global legend from bottom row
    seen = set()
    handles, labels = [], []
    for j in range(ncols):
        h, l = axes[1, j].get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in seen:
                seen.add(ll)
                handles.append(hh); labels.append(ll)
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.03),
                   ncol=min(max(1, ncols * 4), 12),
                   fontsize="x-small", frameon=False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"vc_shift_{y_col}_byFEC.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")




def plot_effect_of_vc_shift_cap_and_midV_with_curves(df, groups, lookup, alpha_colors, out_dir,
                                                    title_prefix="Effect of VC Shift",
                                                    capacity_voltage=2.5,
                                                    cap_col=None,
                                                    cap_label=None,
                                                    v_col="MidpointDischargeV_V",
                                                    v_label="V @ 50% Q (V)",
                                                    fec_levels=GRID_FEC_LEVELS,
                                                    vc_levels=GRID_VC_LEVELS):
    """
    For each FEC panel, plot discharge capacity on the LEFT y-axis and a voltage
    figure-of-merit on the RIGHT y-axis (top row), both vs VC.

    Bottom row shows representative discharge curves (best by cap_col) for each VC.

    Notes:
      - cap_col defaults to cap_at_{capacity_voltage}V.
      - v_col defaults to MidpointDischargeV_V (computed in compute_cell_discharge_metrics).
      - v points are drawn as OPEN markers to visually separate from capacity points.
    """
    os.makedirs(out_dir, exist_ok=True)
    if cap_col is None:
        cap_col = f"cap_at_{capacity_voltage:.1f}V"
    if cap_label is None:
        cap_label = f"Cap @ {capacity_voltage:.1f} V (mAh/g)"

    # --- global axis limits so every panel is directly comparable ---
    def _padded_limits(vals, frac=0.08, min_pad=0.0):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        span = hi - lo
        if span <= 0:
            span = max(abs(hi), 1.0)
        pad = max(frac * span, min_pad)
        return (lo - pad, hi + pad)

    cap_ylim = _padded_limits(df[cap_col].to_numpy(dtype=float) if cap_col in df.columns else [],
                              frac=0.08, min_pad=1.0)
    v_ylim = _padded_limits(df[v_col].to_numpy(dtype=float) if v_col in df.columns else [],
                            frac=0.08, min_pad=0.01)

    vc_levels = list(vc_levels)
    if len(vc_levels) > 0:
        x0 = float(np.min(vc_levels))
        x1 = float(np.max(vc_levels))
        dx = max(0.10 * (x1 - x0), 0.25)
        top_xlim = (x0 - dx, x1 + dx)
    else:
        top_xlim = None


    ncols = len(fec_levels)
    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(5 * ncols, 9), sharex="row")
    if ncols == 1:
        axes = np.array(axes).reshape(2, 1)

    for j, fec in enumerate(fec_levels):
        col = df[df["FEC_wt"] == fec].copy()

        # --- top row: dual-axis scatter + mean±sd ---
        ax = axes[0, j]
        axr = ax.twinx()
        ax.set_title(f"FEC {fec} wt%")
        ax.set_xlabel("VC (wt%)")
        if j == 0:
            ax.set_ylabel(cap_label)
        if j == (ncols - 1):
            axr.set_ylabel(v_label)

        for _, r in col.iterrows():
            vc = r.get("VC_wt", np.nan)
            cap = r.get(cap_col, np.nan)
            vv = r.get(v_col, np.nan)
            if not np.isfinite(vc):
                continue
            alpha = r.get("Alpha", "")
            cc = r.get("CellCode", "")
            c = alpha_colors.get(alpha, "tab:gray")
            mk = marker_for_cell_code(cc)

            if np.isfinite(cap):
                ax.scatter(vc, cap, color=c, marker=mk, s=35, alpha=0.85)

            if np.isfinite(vv):
                axr.scatter(vc, vv, facecolors="none", edgecolors=c,
                            marker=mk, s=35, alpha=0.9, linewidths=1.5)

        # mean±sd overlays (capacity: solid; voltage: dashed)
        for vc in vc_levels:
            g_cap = col[col["VC_wt"] == vc][cap_col].to_numpy(dtype=float)
            g_cap = g_cap[np.isfinite(g_cap)]
            if len(g_cap) >= 2:
                ax.errorbar(vc, float(np.mean(g_cap)), yerr=float(np.std(g_cap, ddof=1)),
                            fmt="k_", capsize=2, alpha=0.8)
            elif len(g_cap) == 1:
                ax.scatter([vc], [float(g_cap[0])], color="k", s=10)

            g_v = col[col["VC_wt"] == vc][v_col].to_numpy(dtype=float)
            g_v = g_v[np.isfinite(g_v)]
            if len(g_v) >= 2:
                axr.errorbar(vc, float(np.mean(g_v)), yerr=float(np.std(g_v, ddof=1)),
                             color="k", linestyle="--", linewidth=1.5,
                             marker="o", markersize=3, capsize=2, alpha=0.65)
            elif len(g_v) == 1:
                axr.scatter([vc], [float(g_v[0])], color="k", s=10, alpha=0.65)

        ax.set_xticks(list(vc_levels))
        if top_xlim is not None:
            ax.set_xlim(*top_xlim)
        if cap_ylim is not None:
            ax.set_ylim(*cap_ylim)
        if v_ylim is not None:
            axr.set_ylim(*v_ylim)
        ax.grid(True, alpha=0.2)

        # --- bottom row: representative curves (best per VC) ---
        axc = axes[1, j]
        axc.set_xlabel("Discharge (mAh/g)")
        if j == 0:
            axc.set_ylabel("Voltage (V)")
        axc.set_ylim(0, 4.5)
        axc.set_xlim(-5, 180)
        axc.grid(True, alpha=0.2)

        for vc in vc_levels:
            cand = col[(col["VC_wt"] == vc) & np.isfinite(col.get(cap_col))]
            if cand.empty:
                continue
            best = cand.sort_values(by=cap_col, ascending=False).iloc[0]
            cc = best["CellCode"]
            alpha = best["Alpha"]
            paths = groups.get(cc, [])
            if not paths:
                continue
            try:
                Q, V = aggregate_discharge_for_cell(paths, lookup)
            except Exception:
                continue
            axc.plot(Q, V,
                     color=alpha_colors.get(alpha, "tab:gray"),
                     linestyle=get_line_style_for_trial_alpha(alpha),
                     linewidth=2.0,
                     label=f"VC {vc}: {cc}")

    fig.suptitle(title_prefix, y=0.98)

    # global legend from bottom row
    seen = set()
    handles, labels = [], []
    for j in range(ncols):
        h, l = axes[1, j].get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in seen:
                seen.add(ll)
                handles.append(hh)
                labels.append(ll)
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.03),
                   ncol=min(max(1, ncols * 4), 12),
                   fontsize="x-small", frameon=False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "vc_shift_cap_and_midV_byFEC.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

def main():
    files_with_source = find_discharge_files(base_dir, old_directory, temp_tag="-51")
    if not files_with_source:
        print(f"No matching .xlsx files found under `{base_dir}` or `{old_directory}` for -51C")
        return

    lookup = load_lookup_table(lookup_table_path)

    # Group by cell code (use only the path part)
    groups = {}
    for p, src in files_with_source:
        code = get_cell_code(p)
        groups.setdefault(code, []).append(p)

    # Build metadata: electrolyte + total additive per cell
    cell_meta = {}
    for cell_code in groups.keys():
        electrolyte = get_electrolyte_name(cell_code, lookup)
        total_add = get_total_additive(electrolyte)
        cell_meta[cell_code] = {
            "electrolyte": electrolyte,
            "total_additive": total_add,
        }

    # Build global color mapping by alpha (stable across all plots)
    unique_alphas = sorted({get_alpha_prefix(c) for c in groups.keys()})
    cmap = plt.get_cmap("tab20")
    alpha_colors = {a: cmap(i % cmap.N) for i, a in enumerate(unique_alphas)}


    max_additive = max(
        (m["total_additive"] for m in cell_meta.values()),
        default=0.0
    )

    # --- per-cell plots ---
    if MAKE_PER_CELL_PLOTS:
        for cell_code, paths in groups.items():
            meta = cell_meta.get(cell_code, {})
            electrolyte = meta.get("electrolyte", "")
            total_add = meta.get("total_additive", 0.0)
            color = alpha_colors.get(cell_code[:2], "tab:gray")
            lw = compute_linewidth(total_add, max_additive)

            anode, cathode = get_anode_cathode(cell_code, lookup)
            chem_str = ""
            if cathode or anode:
                if cathode and anode:
                    chem_str = f"{cathode}|{anode}"
                elif cathode:
                    chem_str = cathode
                else:
                    chem_str = anode

            title_parts = [cell_code]
            if chem_str:
                title_parts.append(chem_str)
            if electrolyte:
                title_parts.append(electrolyte)
            title_label = " - ".join(title_parts)

            fig, ax = plt.subplots(figsize=(8, 6))
            plotted_any = False
            for p in paths:
                try:
                    x_spec, y_volt, _ = load_discharge_curve(p, lookup=lookup)
                except Exception as e:
                    print(f"Skipping {p}: {e}")
                    continue
                plotted_any = True
                legend_label = os.path.basename(p)
                ax.plot(x_spec, y_volt, label=legend_label, linewidth=lw, color=color)

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
            out_path = os.path.join(plots_dir, f"{cell_code}_discharge.png")
            # fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"Saved {out_path}")

    # ---------- one-figure master grid (FEC x VC) ----------
    grid_out_dir = os.path.join(plots_dir, "FECxVC_master_grid")
    plot_fec_vc_master_grid(
        groups, lookup, grid_out_dir,
        cell_meta=cell_meta,
        alpha_colors=alpha_colors,
        max_additive=max_additive,
        fec_levels=GRID_FEC_LEVELS,
        vc_levels=GRID_VC_LEVELS,
        alpha_select=GRID_ALPHA_SELECT,
        selection=GRID_SELECTION,
        reference_alphas=GRID_REFERENCE_ALPHAS,
        best_voltage=GRID_BEST_VOLTAGE,
        show_markers=GRID_SHOW_MARKERS,
    )


    # ---------- scatter metrics (split by trial) ----------

    capacity_voltage = CAPACITY_VOLTAGE  # voltage for capacity interpolation in metrics/trends
    scatter_out_dir = os.path.join(plots_dir, "Scatter_by_trial")
    df_metrics = compute_cell_discharge_metrics(
        groups,
        lookup,
        cell_meta,
        fec_levels=GRID_FEC_LEVELS,
        vc_levels=GRID_VC_LEVELS,
        capacity_voltage=capacity_voltage,
    )
    os.makedirs(scatter_out_dir, exist_ok=True)
    metrics_csv = os.path.join(scatter_out_dir, "cell_discharge_metrics.csv")
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"Saved metrics table: {metrics_csv}")

    plot_scatter_by_trial(
        df_metrics,
        alpha_colors,
        scatter_out_dir,
        x_col="InitialDischargeV_V",
        y_col="Capacity_at_2.5V_mAh_g",
        y_label="Capacity at 2.5 V (mAh/g)",
        title="Initial discharge potential vs capacity at 2.5 V",
        filename="scatter_initV_vs_capAt2p5_byTrial.png",
    )

    plot_scatter_by_trial(
        df_metrics,
        alpha_colors,
        scatter_out_dir,
        x_col="InitialDischargeV_V",
        y_col="AvgDischargeV_V",
        y_label="Average discharge voltage (V)",
        title="Initial discharge potential vs average discharge voltage",
        filename="scatter_initV_vs_avgV_byTrial.png",
    )

    # ---------- effect/trend plots (FEC and VC shifts) ----------
    if MAKE_EFFECT_TREND_PLOTS:
        effect_dir = os.path.join(plots_dir, "effect_trends_by_trial")
        cap_col = f"cap_at_{capacity_voltage:.1f}V"

        for trial in (1, 2, 3):
            dft = df_metrics[df_metrics["Trial"] == trial].copy()
            if dft.empty:
                continue
            tdir = os.path.join(effect_dir, f"Trial{trial}")
            os.makedirs(tdir, exist_ok=True)

            plot_effect_of_fec_shift_with_curves(
                dft, groups, lookup, alpha_colors, tdir,
                title_prefix=f"Effect of FEC Shift (Trial {trial}) — capacity@{capacity_voltage:.1f}V",
                y_col=cap_col, y_label=f"Cap @ {capacity_voltage:.1f} V (mAh/g)",
                capacity_voltage=capacity_voltage
            )
            plot_effect_of_vc_shift_with_curves(
                dft, groups, lookup, alpha_colors, tdir,
                title_prefix=f"Effect of VC Shift (Trial {trial}) — capacity@{capacity_voltage:.1f}V",
                y_col=cap_col, y_label=f"Cap @ {capacity_voltage:.1f} V (mAh/g)",
                capacity_voltage=capacity_voltage
            )
            plot_effect_of_fec_shift_with_curves(
                dft, groups, lookup, alpha_colors, tdir,
                title_prefix=f"Effect of FEC Shift (Trial {trial}) — avg discharge V",
                y_col="AvgDischargeV_V", y_label="Avg discharge V (V)",
                capacity_voltage=capacity_voltage
            )
            plot_effect_of_vc_shift_with_curves(
                dft, groups, lookup, alpha_colors, tdir,
                title_prefix=f"Effect of VC Shift (Trial {trial}) — avg discharge V",
                y_col="AvgDischargeV_V", y_label="Avg discharge V (V)",
                capacity_voltage=capacity_voltage
            )
    # by-alpha plots (if you want them, uncomment)
    # plot_groups_by_alpha(groups, lookup, plots_dir, cell_meta, electrolyte_colors, max_additive)

    # DTF / DTFV grouped plots with solid(old)/dashed(new)
    #plot_dtf_dtfv_groups(groups, lookup, plots_dir, cell_meta, electrolyte_colors, max_additive)

    # Electrolyte-wise old vs new comparison plots (includes FA/EC where applicable)
    #plot_old_vs_new_comparisons(groups, lookup, plots_dir, cell_meta, electrolyte_colors, max_additive)

    # CSV index summarizing which alphas/cells are in old vs new for each electrolyte
    make_old_new_index(groups, lookup, cell_meta, plots_dir)

    # summary table (unchanged)
    make_dtf_dtfv_summary(groups, plots_dir)


    make_dtf_dtfv_summary(groups, plots_dir)
    make_alpha_best_summary(groups, lookup, cell_meta, plots_dir, target_voltage=2.0)
# python
# python

def add_dual_axis_best_fit_lines(ax, axr, x, y_cap, y_v,
                                xlim=None,
                                use_level_means=True,
                                cap_line_style="--",
                                v_line_style=":",
                                line_color="k",
                                line_width=1.5,
                                alpha=0.75,
                                text_fs=9,
                                cap_text_xy=(0.02, 0.98),
                                v_text_xy=(0.02, 0.90)):
    """
    Add linear best-fit lines + R^2 annotations to a dual-axis (ax, axr) subplot.

    Fits:
      - y_cap vs x on ax (left axis)
      - y_v   vs x on axr (right axis)

    If use_level_means=True, the fit uses mean(y) at each unique x-level (equal weight per level).
    Otherwise, fits all points (levels with more replicates get more weight).

    Returns:
      dict with keys:
        cap_slope, cap_intercept, cap_r2, v_slope, v_intercept, v_r2
    """
    import numpy as _np

    def _fit_line(xi, yi):
        xi = _np.asarray(xi, dtype=float)
        yi = _np.asarray(yi, dtype=float)
        m = _np.isfinite(xi) & _np.isfinite(yi)
        xi = xi[m]
        yi = yi[m]

        if xi.size < 2:
            return None  # not enough points

        if use_level_means:
            # mean y at each x (equal weight per x-level)
            ux = _np.unique(xi)
            if ux.size < 2:
                return None
            mx = []
            my = []
            for u in ux:
                yy = yi[xi == u]
                yy = yy[_np.isfinite(yy)]
                if yy.size:
                    mx.append(u)
                    my.append(float(_np.mean(yy)))
            xi = _np.asarray(mx, dtype=float)
            yi = _np.asarray(my, dtype=float)
            if xi.size < 2:
                return None

        # linear fit
        slope, intercept = _np.polyfit(xi, yi, 1)
        yhat = slope * xi + intercept
        ss_res = float(_np.sum((yi - yhat) ** 2))
        ss_tot = float(_np.sum((yi - float(_np.mean(yi))) ** 2))
        r2 = _np.nan if ss_tot <= 0 else (1.0 - ss_res / ss_tot)

        # x-grid for plotting
        if xlim is not None:
            xg0, xg1 = float(xlim[0]), float(xlim[1])
        else:
            xg0, xg1 = float(_np.min(xi)), float(_np.max(xi))
        xg = _np.linspace(xg0, xg1, 200)
        yg = slope * xg + intercept
        return slope, intercept, r2, xg, yg

    out = {
        "cap_slope": _np.nan, "cap_intercept": _np.nan, "cap_r2": _np.nan,
        "v_slope": _np.nan, "v_intercept": _np.nan, "v_r2": _np.nan
    }

    cap_fit = _fit_line(x, y_cap)
    if cap_fit is not None:
        m, b, r2, xg, yg = cap_fit
        ax.plot(xg, yg, linestyle=cap_line_style, color=line_color,
                linewidth=line_width, alpha=alpha, zorder=1)
        out["cap_slope"], out["cap_intercept"], out["cap_r2"] = m, b, r2
        cap_txt = f"Cap fit: y={m:.3g}x+{b:.3g}\n$R^2$={r2:.3f}" if _np.isfinite(r2) else f"Cap fit: y={m:.3g}x+{b:.3g}\n$R^2$=—"
        ax.text(cap_text_xy[0], cap_text_xy[1], cap_txt,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=text_fs, color=line_color)

    v_fit = _fit_line(x, y_v)
    if v_fit is not None:
        m, b, r2, xg, yg = v_fit
        axr.plot(xg, yg, linestyle=v_line_style, color=line_color,
                 linewidth=line_width, alpha=alpha, zorder=1)
        out["v_slope"], out["v_intercept"], out["v_r2"] = m, b, r2
        v_txt = f"Vmid fit: y={m:.3g}x+{b:.3g}\n$R^2$={r2:.3f}" if _np.isfinite(r2) else f"Vmid fit: y={m:.3g}x+{b:.3g}\n$R^2$=—"
        ax.text(v_text_xy[0], v_text_xy[1], v_txt,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=text_fs, color=line_color)

    return out

def main_show_effect_plots():
    """
    Build groups and metrics, then call the effect-of-VC and effect-of-FEC
    plotting-with-curves functions but prevent saving/closing so the figures
    remain open and are shown via matplotlib.
    """
    # find files (uses existing globals `base_dir`, `old_directory`)
    files_with_source = find_discharge_files(base_dir, old_directory, temp_tag="-51")
    if not files_with_source:
        print(f"No matching .xlsx files found under `base_dir` or `old_directory` for -51C")
        return

    # load lookup table
    lookup = load_lookup_table(lookup_table_path)

    # group by cell code
    groups = {}
    for p, src in files_with_source:
        code = get_cell_code(p)
        groups.setdefault(code, []).append(p)

    # build minimal cell_meta (electrolyte + total_additive)
    cell_meta = {}
    for cell_code in groups.keys():
        electrolyte = get_electrolyte_name(cell_code, lookup)
        total_add = get_total_additive(electrolyte)
        cell_meta[cell_code] = {
            "electrolyte": electrolyte,
            "total_additive": total_add,
        }

    # build alpha color mapping
    unique_alphas = sorted({get_alpha_prefix(c) for c in groups.keys()})
    cmap = plt.get_cmap("tab20")
    alpha_colors = {a: cmap(i % cmap.N) for i, a in enumerate(unique_alphas)}

    # compute metrics
    df_metrics = compute_cell_discharge_metrics(
        groups,
        lookup,
        cell_meta,
        fec_levels=GRID_FEC_LEVELS,
        vc_levels=GRID_VC_LEVELS,
        capacity_voltage=CAPACITY_VOLTAGE,
    )

    # Monkeypatch save/close so plots are not written and not closed
    import matplotlib
    orig_fig_savefig = matplotlib.figure.Figure.savefig
    orig_plt_close = plt.close
    try:
        matplotlib.figure.Figure.savefig = lambda self, *args, **kwargs: None
        plt.close = lambda *args, **kwargs: None

        # Call the plotting functions (they will create figures but not save/close)
        # Use `plots_dir` as out_dir (functions call os.makedirs) but saving is disabled.
        plot_effect_of_vc_shift_with_curves(
            df_metrics,
            groups,
            lookup,
            alpha_colors,
            out_dir=plots_dir,
            title_prefix="Effect of VC Shift (display only)",
            y_col=f"cap_at_{CAPACITY_VOLTAGE:.1f}V",
            y_label=f"Cap @{CAPACITY_VOLTAGE:.1f} V (mAh/g)",
            capacity_voltage=CAPACITY_VOLTAGE,
            fec_levels=GRID_FEC_LEVELS,
            vc_levels=GRID_VC_LEVELS,
        )

        plot_effect_of_fec_shift_with_curves(
            df_metrics,
            groups,
            lookup,
            alpha_colors,
            out_dir=plots_dir,
            title_prefix="Effect of FEC Shift (display only)",
            y_col=f"cap_at_{CAPACITY_VOLTAGE:.1f}V",
            y_label=f"Cap @{CAPACITY_VOLTAGE:.1f} V (mAh/g)",
            capacity_voltage=CAPACITY_VOLTAGE,
            fec_levels=GRID_FEC_LEVELS,
            vc_levels=GRID_VC_LEVELS,
        )

        # show all open figures
        plt.show()

    finally:
        # restore original methods
        matplotlib.figure.Figure.savefig = orig_fig_savefig
        plt.close = orig_plt_close
        print("Finished displaying effect-of-VC and effect-of-FEC plots (no files saved).")


def main_2():
    """
    Minimal interactive entry point:
    - Loads lookup & discharge data
    - Filters to Trial 3
    - Builds combined plots (cap@2.5V vs Midpoint V)
    - Displays in Matplotlib (no file writes)
    """
    import matplotlib.pyplot as plt

    # --- paths ---
    base_dir = r"C:\Users\benja\Downloads\Dilute THF Data\11_25_25\-51C_Repeats"
    old_dir = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\-51C_discharges"
    lookup_path = r"C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx"

    # --- data loading ---
    lookup = load_lookup_table(lookup_path)
    all_files = find_discharge_files(base_dir, old_dir, temp_tag="-51C")

    groups = defaultdict(list)
    for p, _ in all_files:
        cc = get_cell_code(p)
        groups[cc].append(p)

    # --- metadata ---
    cell_meta = {}
    for cc in groups.keys():
        e = get_electrolyte_name(cc, lookup)
        cell_meta[cc] = {
            "electrolyte": e,
            "total_additive": get_total_additive(e)
        }

    alpha_colors = {a: plt.cm.tab20(i % 20)
                    for i, a in enumerate(sorted({cc[:2] for cc in groups.keys()}))}

    # --- metrics ---
    df = compute_cell_discharge_metrics(groups, lookup, cell_meta)
    df = df[df["Trial"] == 3].copy()  # Trial 3 only

    cap_col = "cap_at_2.5V"
    v_col = "MidpointDischargeV_V"

    # --- run plots (interactive only) ---
    print("\n--- VC shift combined plot (Trial 3) ---")
    plot_effect_of_vc_shift_cap_and_midV_with_curves(
        df, groups, lookup, alpha_colors, base_dir,
        title_prefix="Effect of VC Shift (Trial 3)",
        capacity_voltage=2.5,
        cap_col=cap_col,
        v_col=v_col,
        v_label="Midpoint Discharge Voltage (V)"
    )

    print("\n--- FEC shift combined plot (Trial 3) ---")
    plot_effect_of_fec_shift_cap_and_midV_with_curves(
        df, groups, lookup, alpha_colors, base_dir,
        title_prefix="Effect of FEC Shift (Trial 3)",
        capacity_voltage=2.5,
        cap_col=cap_col,
        v_col=v_col,
        v_label="Midpoint Discharge Voltage (V)"
    )

    plt.show()


def main_3():
    """
    Minimal, interactive (no file writes) run:
      - Trial 3 only
      - Makes ONLY the two combined shift figures (VC-shift and FEC-shift) WITH curves
      - Adds linear best-fit lines + R^2 in each top-row panel (cap + Vmid)
      - Shows figures in Matplotlib
    """
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from matplotlib.figure import Figure

    # ---------- local helpers (no edits needed elsewhere) ----------
    def _group_axes_by_position(fig):
        """Group axes that occupy the same panel (twin axes share position)."""
        groups = defaultdict(list)
        for ax in fig.get_axes():
            b = ax.get_position().bounds  # (x0, y0, w, h)
            key = tuple(round(v, 4) for v in b)
            groups[key].append(ax)
        return groups

    def _top_row_keys(fig_groups):
        """Return keys corresponding to the top row (largest y0)."""
        y0s = [k[1] for k in fig_groups.keys()]
        if not y0s:
            return set()
        top_y0 = max(y0s)
        # tolerate tight_layout shifts
        return {k for k in fig_groups.keys() if abs(k[1] - top_y0) < 0.02}

    def _pick_left_right_axes(ax_list):
        """Given [ax_left, ax_right] (order unknown), return (axL, axR)."""
        if len(ax_list) != 2:
            return None, None
        a0, a1 = ax_list
        # twin-y axis usually has ticks on the right
        if str(a0.yaxis.get_ticks_position()).lower() == "right":
            return a1, a0
        if str(a1.yaxis.get_ticks_position()).lower() == "right":
            return a0, a1
        # fallback: keep original
        return a0, a1

    def _annotate_vc_shift_figure(fig, df_use, cap_col, v_col):
        """
        Figure made by plot_effect_of_vc_shift_cap_and_midV_with_curves():
          - each top panel title: "FEC {fec} wt%"
          - x-axis: VC_wt
        """
        groups = _group_axes_by_position(fig)
        top_keys = _top_row_keys(groups)

        for key in sorted(top_keys):
            ax_pair = groups[key]
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue

            title = axL.get_title() or ""
            m = re.search(r"FEC\s*([0-9.]+)", title)
            if not m:
                continue
            fec = float(m.group(1))

            sub = df_use.copy()
            sub["FEC_wt"] = pd.to_numeric(sub["FEC_wt"], errors="coerce")
            sub["VC_wt"] = pd.to_numeric(sub["VC_wt"], errors="coerce")
            sub = sub[sub["FEC_wt"] == fec]

            add_dual_axis_best_fit_lines(
                axL, axR,
                x=sub["VC_wt"].to_numpy(dtype=float),
                y_cap=sub[cap_col].to_numpy(dtype=float) if cap_col in sub.columns else np.array([]),
                y_v=sub[v_col].to_numpy(dtype=float) if v_col in sub.columns else np.array([]),
                xlim=axL.get_xlim(),
                use_level_means=True,  # equal weight per VC level
            )

    def _annotate_fec_shift_figure(fig, df_use, cap_col, v_col):
        """
        Figure made by plot_effect_of_fec_shift_cap_and_midV_with_curves():
          - each top panel title: "VC {vc} wt%"
          - x-axis: FEC_wt
        """
        groups = _group_axes_by_position(fig)
        top_keys = _top_row_keys(groups)

        for key in sorted(top_keys):
            ax_pair = groups[key]
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue

            title = axL.get_title() or ""
            m = re.search(r"VC\s*([0-9.]+)", title)
            if not m:
                continue
            vc = float(m.group(1))

            sub = df_use.copy()
            sub["FEC_wt"] = pd.to_numeric(sub["FEC_wt"], errors="coerce")
            sub["VC_wt"] = pd.to_numeric(sub["VC_wt"], errors="coerce")
            sub = sub[sub["VC_wt"] == vc]

            add_dual_axis_best_fit_lines(
                axL, axR,
                x=sub["FEC_wt"].to_numpy(dtype=float),
                y_cap=sub[cap_col].to_numpy(dtype=float) if cap_col in sub.columns else np.array([]),
                y_v=sub[v_col].to_numpy(dtype=float) if v_col in sub.columns else np.array([]),
                xlim=axL.get_xlim(),
                use_level_means=True,  # equal weight per FEC level
            )

    # ---------- build Trial 3-only dataset ----------
    files_with_source = find_discharge_files(base_dir, old_directory, temp_tag="-51")
    if not files_with_source:
        print(f"No matching .xlsx files found under `{base_dir}` or `{old_directory}` for -51C")
        return

    lookup = load_lookup_table(lookup_table_path)

    groups = {}
    for p, _src in files_with_source:
        code = get_cell_code(p)
        groups.setdefault(code, []).append(p)

    cell_meta = {}
    for cell_code in groups.keys():
        electrolyte = get_electrolyte_name(cell_code, lookup)
        cell_meta[cell_code] = {
            "electrolyte": electrolyte,
            "total_additive": get_total_additive(electrolyte),
        }

    unique_alphas = sorted({get_alpha_prefix(c) for c in groups.keys()})
    cmap = plt.get_cmap("tab20")
    alpha_colors = {a: cmap(i % cmap.N) for i, a in enumerate(unique_alphas)}

    dfm = compute_cell_discharge_metrics(
        groups, lookup, cell_meta,
        fec_levels=GRID_FEC_LEVELS, vc_levels=GRID_VC_LEVELS,
        capacity_voltage=CAPACITY_VOLTAGE
    )
    dfm = dfm[dfm["Trial"] == 3].copy()  # Trial #3 only

    cap_col = f"cap_at_{CAPACITY_VOLTAGE:.1f}V"
    v_col = "MidpointDischargeV_V"

    # ---------- NO-SAVE / NO-CLOSE monkeypatch ----------
    _orig_savefig = Figure.savefig
    _orig_close = plt.close
    Figure.savefig = lambda self, *args, **kwargs: None
    plt.close = lambda *args, **kwargs: None

    try:
        # --- VC shift combined plot ---
        n_before = len(plt.get_fignums())
        plot_effect_of_vc_shift_cap_and_midV_with_curves(
            dfm, groups, lookup, alpha_colors, plots_dir,
            title_prefix="Effect of VC Shift (Trial 3) — Cap@2.5V (left) + Vmid (right)",
            capacity_voltage=CAPACITY_VOLTAGE,
            cap_col=cap_col,
            v_col=v_col,
            v_label="V @ 50% Q (V)"
        )
        fig_vc = plt.figure(plt.get_fignums()[-1]) if len(plt.get_fignums()) > n_before else plt.gcf()
        _annotate_vc_shift_figure(fig_vc, dfm, cap_col, v_col)

        # --- FEC shift combined plot ---
        n_before = len(plt.get_fignums())
        plot_effect_of_fec_shift_cap_and_midV_with_curves(
            dfm, groups, lookup, alpha_colors, plots_dir,
            title_prefix="Effect of FEC Shift (Trial 3) — Cap@2.5V (left) + Vmid (right)",
            capacity_voltage=CAPACITY_VOLTAGE,
            cap_col=cap_col,
            v_col=v_col,
            v_label="V @ 50% Q (V)"
        )
        fig_fec = plt.figure(plt.get_fignums()[-1]) if len(plt.get_fignums()) > n_before else plt.gcf()
        _annotate_fec_shift_figure(fig_fec, dfm, cap_col, v_col)

    finally:
        Figure.savefig = _orig_savefig
        plt.close = _orig_close

    plt.show()


def main_4():
    """
    Quick-iterate styling version:
      - Trial 3 only
      - Combined dual-axis shift plots WITH curves:
          * VC-shift: x=VC (darker color w/ higher VC)
          * FEC-shift: x=FEC (darker color w/ higher FEC)
      - Discharge curves forced SOLID
      - Larger fonts for titles/labels/ticks/legend
      - Legends moved INTO each bottom subplot (global bottom legend removed)
      - No saving/closing; just display
    """
    import re
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    # ---------------- helpers ----------------
    def _get_new_fig_id(before_ids):
        after = set(plt.get_fignums())
        new = sorted(list(after - set(before_ids)))
        return new[-1] if new else None

    def _group_axes_by_position(fig):
        groups = {}
        for ax in fig.get_axes():
            b = ax.get_position().bounds  # (x0, y0, w, h)
            key = tuple(round(v, 4) for v in b)
            groups.setdefault(key, []).append(ax)
        return groups

    def _row_keys(groups, which="top"):
        if not groups:
            return set()
        y0s = [k[1] for k in groups.keys()]
        if which == "top":
            target = max(y0s)
        else:
            target = min(y0s)
        return {k for k in groups.keys() if abs(k[1] - target) < 0.02}

    def _pick_left_right_axes(ax_list):
        if len(ax_list) != 2:
            return None, None
        a0, a1 = ax_list
        if str(a0.yaxis.get_ticks_position()).lower() == "right":
            return a1, a0
        if str(a1.yaxis.get_ticks_position()).lower() == "right":
            return a0, a1
        return a0, a1

    def _nearest_level(x, levels):
        levels = np.asarray(levels, dtype=float)
        return float(levels[np.argmin(np.abs(levels - float(x)))])

    def _dark_palette(levels, cmap_name="Blues", lo=0.35, hi=0.90):
        lv = list(sorted([float(v) for v in levels]))
        cmap = plt.get_cmap(cmap_name)
        if len(lv) == 1:
            return {lv[0]: cmap(hi)}
        vals = np.linspace(lo, hi, len(lv))
        return {lv[i]: cmap(vals[i]) for i in range(len(lv))}

    def _bump_fonts(fig):
        # Enforce bigger tick labels (rcParams gets most, but this guarantees)
        for ax in fig.get_axes():
            ax.tick_params(axis="both", which="both", labelsize=FONT_TICK)

    def _remove_global_fig_legends(fig):
        # plot_* functions use fig.legend() at bottom; remove those
        for leg in list(getattr(fig, "legends", [])):
            try:
                leg.remove()
            except Exception:
                pass

    def _legend_inside(ax):
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(loc="best", frameon=False, fontsize=FONT_LEGEND)

    def _solidify_lines(ax):
        for ln in ax.get_lines():
            # only touch actual discharge curves (ignore mean overlay if any ever appears here)
            ln.set_linestyle("-")

    def _recolor_vc_shift(fig, vc_levels):
        """
        For VC-shift combined plot:
          - Top row: x=VC, color by VC (darker with higher VC) for both y1 and y2 scatters
          - Bottom row: curve labels "VC {vc}: {cc}" -> color by vc, solid lines, legend inside
        """
        vc_color = _dark_palette(vc_levels, cmap_name="Blues", lo=0.35, hi=0.90)

        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")
        bot_keys = _row_keys(groups, "bottom")

        # --- top row recolor (capacity filled + voltage hollow edge) ---
        for key in sorted(top_keys):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue

            # recolor capacity scatters (filled) on left axis
            for coll in list(axL.collections):
                try:
                    off = coll.get_offsets()
                    if off is None or len(off) == 0:
                        continue
                    x = float(off[0, 0])
                    lvl = _nearest_level(x, vc_levels)
                    c = vc_color.get(lvl, "k")
                    coll.set_facecolor(c)
                    coll.set_edgecolor(c)
                except Exception:
                    pass

            # recolor voltage scatters (hollow) on right axis: edgecolor only
            for coll in list(axR.collections):
                try:
                    off = coll.get_offsets()
                    if off is None or len(off) == 0:
                        continue
                    x = float(off[0, 0])
                    lvl = _nearest_level(x, vc_levels)
                    c = vc_color.get(lvl, "k")
                    coll.set_facecolor("none")
                    coll.set_edgecolor(c)
                except Exception:
                    pass

        # --- bottom row recolor curves by VC level ---
        for key in sorted(bot_keys):
            ax_list = groups.get(key, [])
            if len(ax_list) != 1:
                # bottom row usually single axis per panel (no twinx)
                continue
            ax = ax_list[0]
            for ln in ax.get_lines():
                lab = ln.get_label() or ""
                m = re.search(r"VC\s*([0-9.]+)", lab)
                if m:
                    vc = float(m.group(1))
                    lvl = _nearest_level(vc, vc_levels)
                    ln.set_color(vc_color.get(lvl, "k"))
                ln.set_linestyle("-")  # discharge curves solid
            _legend_inside(ax)

    def _recolor_fec_shift(fig, fec_levels):
        """
        For FEC-shift combined plot:
          - Top row: x=FEC, color by FEC (darker with higher FEC) for both y1 and y2 scatters
          - Bottom row: curve labels "FEC {fec}: {cc}" -> color by fec, solid lines, legend inside
        """
        fec_color = _dark_palette(fec_levels, cmap_name="Blues", lo=0.35, hi=0.90)

        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")
        bot_keys = _row_keys(groups, "bottom")

        # --- top row recolor (capacity filled + voltage hollow edge) ---
        for key in sorted(top_keys):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue

            for coll in list(axL.collections):
                try:
                    off = coll.get_offsets()
                    if off is None or len(off) == 0:
                        continue
                    x = float(off[0, 0])
                    lvl = _nearest_level(x, fec_levels)
                    c = fec_color.get(lvl, "k")
                    coll.set_facecolor(c)
                    coll.set_edgecolor(c)
                except Exception:
                    pass

            for coll in list(axR.collections):
                try:
                    off = coll.get_offsets()
                    if off is None or len(off) == 0:
                        continue
                    x = float(off[0, 0])
                    lvl = _nearest_level(x, fec_levels)
                    c = fec_color.get(lvl, "k")
                    coll.set_facecolor("none")
                    coll.set_edgecolor(c)
                except Exception:
                    pass

        # --- bottom row recolor curves by FEC level ---
        for key in sorted(bot_keys):
            ax_list = groups.get(key, [])
            if len(ax_list) != 1:
                continue
            ax = ax_list[0]
            for ln in ax.get_lines():
                lab = ln.get_label() or ""
                m = re.search(r"FEC\s*([0-9.]+)", lab)
                if m:
                    fec = float(m.group(1))
                    lvl = _nearest_level(fec, fec_levels)
                    ln.set_color(fec_color.get(lvl, "k"))
                ln.set_linestyle("-")
            _legend_inside(ax)

    def _annotate_bestfits_vc_shift(fig, df_use, cap_col, v_col):
        # top row panels: title "FEC {fec} wt%" ; x = VC_wt
        groups = _group_axes_by_position(fig)
        for key in sorted(_row_keys(groups, "top")):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue
            title = axL.get_title() or ""
            m = re.search(r"FEC\s*([0-9.]+)", title)
            if not m:
                continue
            fec = float(m.group(1))
            sub = df_use.copy()
            sub["FEC_wt"] = pd.to_numeric(sub["FEC_wt"], errors="coerce")
            sub["VC_wt"] = pd.to_numeric(sub["VC_wt"], errors="coerce")
            sub = sub[sub["FEC_wt"] == fec]
            add_dual_axis_best_fit_lines(
                axL, axR,
                x=sub["VC_wt"].to_numpy(dtype=float),
                y_cap=sub[cap_col].to_numpy(dtype=float),
                y_v=sub[v_col].to_numpy(dtype=float),
                xlim=axL.get_xlim(),
                use_level_means=True,
            )

    def _annotate_bestfits_fec_shift(fig, df_use, cap_col, v_col):
        # top row panels: title "VC {vc} wt%" ; x = FEC_wt
        groups = _group_axes_by_position(fig)
        for key in sorted(_row_keys(groups, "top")):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue
            title = axL.get_title() or ""
            m = re.search(r"VC\s*([0-9.]+)", title)
            if not m:
                continue
            vc = float(m.group(1))
            sub = df_use.copy()
            sub["FEC_wt"] = pd.to_numeric(sub["FEC_wt"], errors="coerce")
            sub["VC_wt"] = pd.to_numeric(sub["VC_wt"], errors="coerce")
            sub = sub[sub["VC_wt"] == vc]
            add_dual_axis_best_fit_lines(
                axL, axR,
                x=sub["FEC_wt"].to_numpy(dtype=float),
                y_cap=sub[cap_col].to_numpy(dtype=float),
                y_v=sub[v_col].to_numpy(dtype=float),
                xlim=axL.get_xlim(),
                use_level_means=True,
            )

    # ---------------- build data (Trial 3 only) ----------------
    files_with_source = find_discharge_files(base_dir, old_directory, temp_tag="-51")
    if not files_with_source:
        print("No matching .xlsx files found for -51C.")
        return

    lookup = load_lookup_table(lookup_table_path)

    groups = {}
    for p, _src in files_with_source:
        code = get_cell_code(p)
        groups.setdefault(code, []).append(p)

    cell_meta = {}
    for cell_code in groups.keys():
        electrolyte = get_electrolyte_name(cell_code, lookup)
        cell_meta[cell_code] = {
            "electrolyte": electrolyte,
            "total_additive": get_total_additive(electrolyte),
        }

    # alpha_colors required by plot funcs (we recolor afterward anyway)
    unique_alphas = sorted({get_alpha_prefix(c) for c in groups.keys()})
    cmap = plt.get_cmap("tab20")
    alpha_colors = {a: cmap(i % cmap.N) for i, a in enumerate(unique_alphas)}

    dfm = compute_cell_discharge_metrics(
        groups, lookup, cell_meta,
        fec_levels=GRID_FEC_LEVELS, vc_levels=GRID_VC_LEVELS,
        capacity_voltage=CAPACITY_VOLTAGE,
    )
    dfm = dfm[dfm["Trial"] == 3].copy()

    cap_col = f"cap_at_{CAPACITY_VOLTAGE:.1f}V"
    v_col = "MidpointDischargeV_V"

    # ---------------- styling knobs ----------------
    FONT_TITLE = 18
    FONT_LABEL = 15
    FONT_TICK = 13
    FONT_LEGEND = 12

    rc = {
        "figure.titlesize": FONT_TITLE,
        "axes.titlesize": FONT_TITLE - 2,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
    }

    # ---------------- no-save / no-close + plotting ----------------
    orig_savefig = Figure.savefig
    orig_close = plt.close
    try:
        Figure.savefig = lambda self, *args, **kwargs: None
        plt.close = lambda *args, **kwargs: None

        with plt.rc_context(rc):
            # --- VC shift combined ---
            before = list(plt.get_fignums())
            plot_effect_of_vc_shift_cap_and_midV_with_curves(
                dfm, groups, lookup, alpha_colors, plots_dir,
                title_prefix="VC Shift (Trial 3) — Cap@2.5V (left) + Vmid (right)",
                capacity_voltage=CAPACITY_VOLTAGE,
                cap_col=cap_col,
                v_col=v_col,
                v_label="V @ 50% Q (V)",
            )
            fid = _get_new_fig_id(before)
            fig_vc = plt.figure(fid) if fid is not None else plt.gcf()
            _remove_global_fig_legends(fig_vc)
            _recolor_vc_shift(fig_vc, GRID_VC_LEVELS)
            _annotate_bestfits_vc_shift(fig_vc, dfm, cap_col, v_col)
            _bump_fonts(fig_vc)

            # --- FEC shift combined ---
            before = list(plt.get_fignums())
            plot_effect_of_fec_shift_cap_and_midV_with_curves(
                dfm, groups, lookup, alpha_colors, plots_dir,
                title_prefix="FEC Shift (Trial 3) — Cap@2.5V (left) + Vmid (right)",
                capacity_voltage=CAPACITY_VOLTAGE,
                cap_col=cap_col,
                v_col=v_col,
                v_label="V @ 50% Q (V)",
            )
            fid = _get_new_fig_id(before)
            fig_fec = plt.figure(fid) if fid is not None else plt.gcf()
            _remove_global_fig_legends(fig_fec)
            _recolor_fec_shift(fig_fec, GRID_FEC_LEVELS)
            _annotate_bestfits_fec_shift(fig_fec, dfm, cap_col, v_col)
            _bump_fonts(fig_fec)

            plt.show()

    finally:
        Figure.savefig = orig_savefig
        plt.close = orig_close

def main_5():
    """
    Trial-3-only, interactive (no save/close) combined shift plots WITH curves, styled for quick iteration:
      - Discharge curves forced SOLID
      - Larger fonts (labels/ticks/legend)
      - Darker color with higher additive level (same 3–4 shades per plot)
      - Legends INSIDE each bottom subplot; legend entries are ONLY additive wt% (no cell codes)
      - NO best-fit lines drawn; best-fit params + R^2 printed to console for each top panel
      - Top-row: y1 label only on LEFTMOST panel; y2 label only on RIGHTMOST panel
      - Top-row: y1 axis styled (blue), y2 axis styled (orange)
    """
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    # ---------------- styling knobs ----------------
    FONT_TITLE = 18
    FONT_LABEL = 16
    FONT_TICK = 13
    FONT_LEGEND = 12

    Y1_COLOR = "tab:blue"    # capacity axis color
    Y2_COLOR = "tab:orange"  # voltage axis color

    rc = {
        "figure.titlesize": FONT_TITLE,
        "axes.titlesize": FONT_TITLE - 2,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
    }

    # ---------------- helpers ----------------
    def _get_new_fig_id(before_ids):
        after = set(plt.get_fignums())
        new = sorted(list(after - set(before_ids)))
        return new[-1] if new else None

    def _group_axes_by_position(fig):
        groups = {}
        for ax in fig.get_axes():
            b = ax.get_position().bounds  # (x0, y0, w, h)
            key = tuple(round(v, 4) for v in b)
            groups.setdefault(key, []).append(ax)
        return groups

    def _row_keys(groups, which="top"):
        if not groups:
            return []
        y0s = [k[1] for k in groups.keys()]
        target = max(y0s) if which == "top" else min(y0s)
        keys = [k for k in groups.keys() if abs(k[1] - target) < 0.02]
        # sort left->right by x0
        return sorted(keys, key=lambda k: k[0])

    def _pick_left_right_axes(ax_list):
        if len(ax_list) != 2:
            return None, None
        a0, a1 = ax_list
        if str(a0.yaxis.get_ticks_position()).lower() == "right":
            return a1, a0
        if str(a1.yaxis.get_ticks_position()).lower() == "right":
            return a0, a1
        return a0, a1

    def _nearest_level(x, levels):
        levels = np.asarray(levels, dtype=float)
        return float(levels[np.argmin(np.abs(levels - float(x)))])

    def _dark_palette(levels, cmap_name="Blues", lo=0.35, hi=0.90):
        lv = list(sorted([float(v) for v in levels]))
        cmap = plt.get_cmap(cmap_name)
        if len(lv) == 1:
            return {lv[0]: cmap(hi)}
        vals = np.linspace(lo, hi, len(lv))
        return {lv[i]: cmap(vals[i]) for i in range(len(lv))}

    def _remove_global_fig_legends(fig):
        for leg in list(getattr(fig, "legends", [])):
            try:
                leg.remove()
            except Exception:
                pass

    def _legend_inside(ax):
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(loc="best", frameon=False, fontsize=FONT_LEGEND)

    def _bump_tick_fonts(fig):
        for ax in fig.get_axes():
            ax.tick_params(axis="both", which="both", labelsize=FONT_TICK)

    def _style_y_axes(axL, axR):
        # left axis styling
        axL.spines["left"].set_color(Y1_COLOR)
        axL.tick_params(axis="y", colors=Y1_COLOR)
        axL.yaxis.label.set_color(Y1_COLOR)
        # right axis styling
        axR.spines["right"].set_color(Y2_COLOR)
        axR.tick_params(axis="y", colors=Y2_COLOR)
        axR.yaxis.label.set_color(Y2_COLOR)

    def _set_end_only_ylabels(fig):
        """
        Top row only:
          - y1 label only on leftmost panel (left axis)
          - y2 label only on rightmost panel (right axis)
        """
        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")
        if not top_keys:
            return

        leftmost = top_keys[0]
        rightmost = top_keys[-1]

        for key in top_keys:
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue

            # keep y1 label only on leftmost
            if key != leftmost:
                axL.set_ylabel("")
            # keep y2 label only on rightmost
            if key != rightmost:
                axR.set_ylabel("")

            _style_y_axes(axL, axR)

    def _fit_stats(x, y, use_level_means=True):
        """
        Returns (slope, intercept, r2). Uses mean(y) per unique x-level if use_level_means=True.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 2:
            return np.nan, np.nan, np.nan

        if use_level_means:
            ux = np.unique(x)
            if ux.size < 2:
                return np.nan, np.nan, np.nan
            xs, ys = [], []
            for u in ux:
                yy = y[x == u]
                yy = yy[np.isfinite(yy)]
                if yy.size:
                    xs.append(u)
                    ys.append(float(np.mean(yy)))
            x = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=float)
            if x.size < 2:
                return np.nan, np.nan, np.nan

        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = np.nan if ss_tot <= 0 else (1.0 - ss_res / ss_tot)
        return float(slope), float(intercept), float(r2)

    def _print_bestfits_vc_shift(fig, df_use, cap_col, v_col):
        # panels keyed by FEC; x=VC
        groups = _group_axes_by_position(fig)
        for key in _row_keys(groups, "top"):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, _axR = _pick_left_right_axes(ax_pair)
            title = axL.get_title() or ""
            m = re.search(r"FEC\s*([0-9.]+)", title)
            if not m:
                continue
            fec = float(m.group(1))

            sub = df_use.copy()
            sub["FEC_wt"] = pd.to_numeric(sub["FEC_wt"], errors="coerce")
            sub["VC_wt"] = pd.to_numeric(sub["VC_wt"], errors="coerce")
            sub = sub[sub["FEC_wt"] == fec]

            s1, b1, r1 = _fit_stats(sub["VC_wt"].to_numpy(float), sub[cap_col].to_numpy(float), use_level_means=True)
            s2, b2, r2 = _fit_stats(sub["VC_wt"].to_numpy(float), sub[v_col].to_numpy(float), use_level_means=True)

            print(f"[VC-shift | Trial 3] FEC={fec:g} wt%")
            print(f"  Cap fit (mean-per-VC): slope={s1:.6g}, intercept={b1:.6g}, R^2={r1:.4f}")
            print(f"  Vmid fit (mean-per-VC): slope={s2:.6g}, intercept={b2:.6g}, R^2={r2:.4f}")

    def _print_bestfits_fec_shift(fig, df_use, cap_col, v_col):
        # panels keyed by VC; x=FEC
        groups = _group_axes_by_position(fig)
        for key in _row_keys(groups, "top"):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, _axR = _pick_left_right_axes(ax_pair)
            title = axL.get_title() or ""
            m = re.search(r"VC\s*([0-9.]+)", title)
            if not m:
                continue
            vc = float(m.group(1))

            sub = df_use.copy()
            sub["FEC_wt"] = pd.to_numeric(sub["FEC_wt"], errors="coerce")
            sub["VC_wt"] = pd.to_numeric(sub["VC_wt"], errors="coerce")
            sub = sub[sub["VC_wt"] == vc]

            s1, b1, r1 = _fit_stats(sub["FEC_wt"].to_numpy(float), sub[cap_col].to_numpy(float), use_level_means=True)
            s2, b2, r2 = _fit_stats(sub["FEC_wt"].to_numpy(float), sub[v_col].to_numpy(float), use_level_means=True)

            print(f"[FEC-shift | Trial 3] VC={vc:g} wt%")
            print(f"  Cap fit (mean-per-FEC): slope={s1:.6g}, intercept={b1:.6g}, R^2={r1:.4f}")
            print(f"  Vmid fit (mean-per-FEC): slope={s2:.6g}, intercept={b2:.6g}, R^2={r2:.4f}")

    def _recolor_and_relabel_bottom_vc_shift(fig, vc_levels):
        """
        Bottom-row curves for VC-shift plot:
          - color by VC (darker with higher VC)
          - solid lines
          - legend labels ONLY: "{vc} wt%"
        """
        vc_color = _dark_palette(vc_levels, cmap_name="Blues", lo=0.35, hi=0.90)
        groups = _group_axes_by_position(fig)
        bot_keys = _row_keys(groups, "bottom")

        for key in bot_keys:
            ax_list = groups.get(key, [])
            if len(ax_list) != 1:
                continue
            ax = ax_list[0]
            seen = set()
            for ln in ax.get_lines():
                lab = ln.get_label() or ""
                m = re.search(r"VC\s*([0-9.]+)", lab)
                if m:
                    vc = float(m.group(1))
                    lvl = _nearest_level(vc, vc_levels)
                    ln.set_color(vc_color.get(lvl, "k"))
                    ln.set_linestyle("-")
                    new_lab = f"{int(lvl)} wt%"
                    if new_lab in seen:
                        ln.set_label("_nolegend_")
                    else:
                        ln.set_label(new_lab)
                        seen.add(new_lab)
                else:
                    ln.set_linestyle("-")
                    ln.set_label("_nolegend_")
            _legend_inside(ax)

    def _recolor_and_relabel_bottom_fec_shift(fig, fec_levels):
        """
        Bottom-row curves for FEC-shift plot:
          - color by FEC (darker with higher FEC)
          - solid lines
          - legend labels ONLY: "{fec} wt%"
        """
        fec_color = _dark_palette(fec_levels, cmap_name="Blues", lo=0.35, hi=0.90)
        groups = _group_axes_by_position(fig)
        bot_keys = _row_keys(groups, "bottom")

        for key in bot_keys:
            ax_list = groups.get(key, [])
            if len(ax_list) != 1:
                continue
            ax = ax_list[0]
            seen = set()
            for ln in ax.get_lines():
                lab = ln.get_label() or ""
                m = re.search(r"FEC\s*([0-9.]+)", lab)
                if m:
                    fec = float(m.group(1))
                    lvl = _nearest_level(fec, fec_levels)
                    ln.set_color(fec_color.get(lvl, "k"))
                    ln.set_linestyle("-")
                    new_lab = f"{int(lvl)} wt%"
                    if new_lab in seen:
                        ln.set_label("_nolegend_")
                    else:
                        ln.set_label(new_lab)
                        seen.add(new_lab)
                else:
                    ln.set_linestyle("-")
                    ln.set_label("_nolegend_")
            _legend_inside(ax)

    def _recolor_top_scatter_by_levels(fig, levels, mode):
        """
        Top-row scatter recolor:
          - left axis (capacity) points: filled
          - right axis (voltage) points: hollow
        mode:
          - "vc": x corresponds to VC_wt levels
          - "fec": x corresponds to FEC_wt levels
        """
        color_map = _dark_palette(levels, cmap_name="Blues", lo=0.35, hi=0.90)
        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")

        for key in top_keys:
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue

            # capacity scatters (filled)
            for coll in list(axL.collections):
                try:
                    off = coll.get_offsets()
                    if off is None or len(off) == 0:
                        continue
                    x = float(off[0, 0])
                    lvl = _nearest_level(x, levels)
                    c = color_map.get(lvl, "k")
                    coll.set_facecolor(c)
                    coll.set_edgecolor(c)
                except Exception:
                    pass

            # voltage scatters (hollow edge)
            for coll in list(axR.collections):
                try:
                    off = coll.get_offsets()
                    if off is None or len(off) == 0:
                        continue
                    x = float(off[0, 0])
                    lvl = _nearest_level(x, levels)
                    c = color_map.get(lvl, "k")
                    coll.set_facecolor("none")
                    coll.set_edgecolor(c)
                except Exception:
                    pass

    # ---------------- build data (Trial 3 only) ----------------
    files_with_source = find_discharge_files(base_dir, old_directory, temp_tag="-51")
    if not files_with_source:
        print("No matching .xlsx files found for -51C.")
        return

    lookup = load_lookup_table(lookup_table_path)

    groups = {}
    for p, _src in files_with_source:
        code = get_cell_code(p)
        groups.setdefault(code, []).append(p)

    cell_meta = {}
    for cell_code in groups.keys():
        electrolyte = get_electrolyte_name(cell_code, lookup)
        cell_meta[cell_code] = {
            "electrolyte": electrolyte,
            "total_additive": get_total_additive(electrolyte),
        }

    # alpha_colors required by plot funcs (we recolor afterward anyway)
    unique_alphas = sorted({get_alpha_prefix(c) for c in groups.keys()})
    cmap = plt.get_cmap("tab20")
    alpha_colors = {a: cmap(i % cmap.N) for i, a in enumerate(unique_alphas)}

    dfm = compute_cell_discharge_metrics(
        groups, lookup, cell_meta,
        fec_levels=GRID_FEC_LEVELS, vc_levels=GRID_VC_LEVELS,
        capacity_voltage=CAPACITY_VOLTAGE,
    )
    dfm = dfm[dfm["Trial"] == 3].copy()

    cap_col = f"cap_at_{CAPACITY_VOLTAGE:.1f}V"
    v_col = "MidpointDischargeV_V"

    # ---------------- no-save / no-close + plotting ----------------
    orig_savefig = Figure.savefig
    orig_close = plt.close
    try:
        Figure.savefig = lambda self, *args, **kwargs: None
        plt.close = lambda *args, **kwargs: None

        with plt.rc_context(rc):
            # --- VC shift combined ---
            before = list(plt.get_fignums())
            plot_effect_of_vc_shift_cap_and_midV_with_curves(
                dfm, groups, lookup, alpha_colors, plots_dir,
                title_prefix="VC Shift (Trial 3) — Cap@2.5V (left) + Vmid (right)",
                capacity_voltage=CAPACITY_VOLTAGE,
                cap_col=cap_col,
                v_col=v_col,
                v_label="V @ 50% Q (V)",
            )
            fid = _get_new_fig_id(before)
            fig_vc = plt.figure(fid) if fid is not None else plt.gcf()

            _remove_global_fig_legends(fig_vc)
            _recolor_top_scatter_by_levels(fig_vc, GRID_VC_LEVELS, mode="vc")
            _recolor_and_relabel_bottom_vc_shift(fig_vc, GRID_VC_LEVELS)
            _set_end_only_ylabels(fig_vc)
            _bump_tick_fonts(fig_vc)

            # print fits (NO lines drawn)
            _print_bestfits_vc_shift(fig_vc, dfm, cap_col, v_col)

            # --- FEC shift combined ---
            before = list(plt.get_fignums())
            plot_effect_of_fec_shift_cap_and_midV_with_curves(
                dfm, groups, lookup, alpha_colors, plots_dir,
                title_prefix="FEC Shift (Trial 3) — Cap@2.5V (left) + Vmid (right)",
                capacity_voltage=CAPACITY_VOLTAGE,
                cap_col=cap_col,
                v_col=v_col,
                v_label="V @ 50% Q (V)",
            )
            fid = _get_new_fig_id(before)
            fig_fec = plt.figure(fid) if fid is not None else plt.gcf()

            _remove_global_fig_legends(fig_fec)
            _recolor_top_scatter_by_levels(fig_fec, GRID_FEC_LEVELS, mode="fec")
            _recolor_and_relabel_bottom_fec_shift(fig_fec, GRID_FEC_LEVELS)
            _set_end_only_ylabels(fig_fec)
            _bump_tick_fonts(fig_fec)

            # print fits (NO lines drawn)
            _print_bestfits_fec_shift(fig_fec, dfm, cap_col, v_col)

            plt.show()

    finally:
        Figure.savefig = orig_savefig
        plt.close = orig_close


def main_6():
    """
    Trial-3-only, interactive (no save/close) combined shift plots WITH curves, styled per latest requests:
      - NO best-fit lines drawn; best-fit + R^2 printed to console for each top panel (cap + Vmid)
      - Top-row y-axis *tick label values*:
          * Y2 (right axis, Vmid) tick labels ONLY on LEFTMOST panel
          * Y1 (left axis, Cap)  tick labels ONLY on RIGHTMOST panel
        …and axis label TEXT swapped the same way:
          * Vmid ylabel only on LEFTMOST panel (right axis)
          * Cap  ylabel only on RIGHTMOST panel (left axis)
      - Top-row point colors match axes hues:
          * Capacity points + errorbars = blues (darker with higher additive level)
          * Voltage points + errorbars = oranges (darker with higher additive level)
      - Std-dev bars recolored to match
      - Discharge curves forced SOLID; legends inside each bottom subplot; legend text only additive wt% (no cell codes)
      - Drop figure suptitle (figure title)
    """
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.container import ErrorbarContainer

    # ---------------- styling knobs ----------------
    FONT_TITLE = 18
    FONT_LABEL = 16
    FONT_TICK = 13
    FONT_LEGEND = 12

    Y1_COLOR = "tab:blue"    # capacity axis
    Y2_COLOR = "tab:orange"  # voltage axis

    rc = {
        "figure.titlesize": FONT_TITLE,
        "axes.titlesize": FONT_TITLE - 2,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
    }

    # ---------------- helpers ----------------
    def _get_new_fig_id(before_ids):
        after = set(plt.get_fignums())
        new = sorted(list(after - set(before_ids)))
        return new[-1] if new else None

    def _group_axes_by_position(fig):
        groups = {}
        for ax in fig.get_axes():
            b = ax.get_position().bounds  # (x0, y0, w, h)
            key = tuple(round(v, 4) for v in b)
            groups.setdefault(key, []).append(ax)
        return groups

    def _row_keys(groups, which="top"):
        if not groups:
            return []
        y0s = [k[1] for k in groups.keys()]
        target = max(y0s) if which == "top" else min(y0s)
        keys = [k for k in groups.keys() if abs(k[1] - target) < 0.02]
        return sorted(keys, key=lambda k: k[0])  # left->right

    def _pick_left_right_axes(ax_list):
        if len(ax_list) != 2:
            return None, None
        a0, a1 = ax_list
        if str(a0.yaxis.get_ticks_position()).lower() == "right":
            return a1, a0
        if str(a1.yaxis.get_ticks_position()).lower() == "right":
            return a0, a1
        return a0, a1

    def _nearest_level(x, levels):
        levels = np.asarray(levels, dtype=float)
        return float(levels[np.argmin(np.abs(levels - float(x)))])

    def _palette(levels, cmap_name, lo=0.35, hi=0.90):
        lv = list(sorted([float(v) for v in levels]))
        cmap = plt.get_cmap(cmap_name)
        if len(lv) == 1:
            return {lv[0]: cmap(hi)}
        vals = np.linspace(lo, hi, len(lv))
        return {lv[i]: cmap(vals[i]) for i in range(len(lv))}

    def _remove_global_fig_legends(fig):
        for leg in list(getattr(fig, "legends", [])):
            try:
                leg.remove()
            except Exception:
                pass

    def _drop_suptitle(fig):
        try:
            if getattr(fig, "_suptitle", None) is not None:
                fig._suptitle.set_text("")
                fig._suptitle.set_visible(False)
        except Exception:
            pass

    def _legend_inside(ax):
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(loc="best", frameon=False, fontsize=FONT_LEGEND)

    def _bump_tick_fonts(fig):
        for ax in fig.get_axes():
            ax.tick_params(axis="both", which="both", labelsize=FONT_TICK)

    def _style_y_axes(axL, axR):
        # left axis styling (capacity)
        axL.spines["left"].set_color(Y1_COLOR)
        axL.tick_params(axis="y", colors=Y1_COLOR)
        axL.yaxis.label.set_color(Y1_COLOR)
        # right axis styling (voltage)
        axR.spines["right"].set_color(Y2_COLOR)
        axR.tick_params(axis="y", colors=Y2_COLOR)
        axR.yaxis.label.set_color(Y2_COLOR)

    def _set_toprow_ticks_and_ylabels_swapped(fig):
        """
        Top row only:
          - Show Y2 (right-axis) tick labels + ylabel ONLY on LEFTMOST panel
          - Show Y1 (left-axis)  tick labels + ylabel ONLY on RIGHTMOST panel
        """
        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")
        if not top_keys:
            return

        leftmost = top_keys[0]
        rightmost = top_keys[-1]

        # Grab canonical ylabel texts from first panel so we can re-apply them
        cap_ylabel = None
        vmid_ylabel = None
        for key in top_keys:
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue
            cap_ylabel = cap_ylabel or (axL.get_ylabel() or "")
            vmid_ylabel = vmid_ylabel or (axR.get_ylabel() or "")
        cap_ylabel = cap_ylabel or "Capacity"
        vmid_ylabel = vmid_ylabel or "Voltage"

        for key in top_keys:
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue

            # Tick label values visibility
            axL.tick_params(labelleft=(key == rightmost))     # Y1 ticks only rightmost
            axR.tick_params(labelright=(key == leftmost))     # Y2 ticks only leftmost

            # Y label TEXT swapped ends
            axL.set_ylabel(cap_ylabel if key == rightmost else "")
            axR.set_ylabel(vmid_ylabel if key == leftmost else "")

            _style_y_axes(axL, axR)

    def _fit_stats(x, y, use_level_means=True):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 2:
            return np.nan, np.nan, np.nan

        if use_level_means:
            ux = np.unique(x)
            if ux.size < 2:
                return np.nan, np.nan, np.nan
            xs, ys = [], []
            for u in ux:
                yy = y[x == u]
                yy = yy[np.isfinite(yy)]
                if yy.size:
                    xs.append(u)
                    ys.append(float(np.mean(yy)))
            x = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=float)
            if x.size < 2:
                return np.nan, np.nan, np.nan

        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = np.nan if ss_tot <= 0 else (1.0 - ss_res / ss_tot)
        return float(slope), float(intercept), float(r2)

    def _print_bestfits_vc_shift(fig, df_use, cap_col, v_col):
        groups = _group_axes_by_position(fig)
        for key in _row_keys(groups, "top"):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, _axR = _pick_left_right_axes(ax_pair)
            title = axL.get_title() or ""
            m = re.search(r"FEC\s*([0-9.]+)", title)
            if not m:
                continue
            fec = float(m.group(1))

            sub = df_use.copy()
            sub["FEC_wt"] = pd.to_numeric(sub["FEC_wt"], errors="coerce")
            sub["VC_wt"] = pd.to_numeric(sub["VC_wt"], errors="coerce")
            sub = sub[sub["FEC_wt"] == fec]

            s1, b1, r1 = _fit_stats(sub["VC_wt"].to_numpy(float), sub[cap_col].to_numpy(float), use_level_means=True)
            s2, b2, r2 = _fit_stats(sub["VC_wt"].to_numpy(float), sub[v_col].to_numpy(float), use_level_means=True)

            print(f"[VC-shift | Trial 3] FEC={fec:g} wt%")
            print(f"  Cap  (mean-per-VC):   slope={s1:.6g}, intercept={b1:.6g}, R^2={r1:.4f}")
            print(f"  Vmid (mean-per-VC):   slope={s2:.6g}, intercept={b2:.6g}, R^2={r2:.4f}")

    def _print_bestfits_fec_shift(fig, df_use, cap_col, v_col):
        groups = _group_axes_by_position(fig)
        for key in _row_keys(groups, "top"):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, _axR = _pick_left_right_axes(ax_pair)
            title = axL.get_title() or ""
            m = re.search(r"VC\s*([0-9.]+)", title)
            if not m:
                continue
            vc = float(m.group(1))

            sub = df_use.copy()
            sub["FEC_wt"] = pd.to_numeric(sub["FEC_wt"], errors="coerce")
            sub["VC_wt"] = pd.to_numeric(sub["VC_wt"], errors="coerce")
            sub = sub[sub["VC_wt"] == vc]

            s1, b1, r1 = _fit_stats(sub["FEC_wt"].to_numpy(float), sub[cap_col].to_numpy(float), use_level_means=True)
            s2, b2, r2 = _fit_stats(sub["FEC_wt"].to_numpy(float), sub[v_col].to_numpy(float), use_level_means=True)

            print(f"[FEC-shift | Trial 3] VC={vc:g} wt%")
            print(f"  Cap  (mean-per-FEC):  slope={s1:.6g}, intercept={b1:.6g}, R^2={r1:.4f}")
            print(f"  Vmid (mean-per-FEC):  slope={s2:.6g}, intercept={b2:.6g}, R^2={r2:.4f}")

    def _recolor_toprow_points_and_errorbars(fig, levels):
        """
        Top row:
          - Capacity points + errorbars: BLUE shades (by x level)
          - Voltage points + errorbars: ORANGE shades (by x level)
        """
        cap_map = _palette(levels, "Blues", lo=0.35, hi=0.90)
        v_map = _palette(levels, "Oranges", lo=0.35, hi=0.90)

        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")

        for key in top_keys:
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue

            # --- recolor raw scatter collections ---
            # left axis: filled blue
            for coll in list(axL.collections):
                try:
                    off = coll.get_offsets()
                    if off is None or len(off) == 0:
                        continue
                    x = float(off[0, 0])
                    lvl = _nearest_level(x, levels)
                    c = cap_map.get(lvl, Y1_COLOR)
                    coll.set_facecolor(c)
                    coll.set_edgecolor(c)
                except Exception:
                    pass

            # right axis: hollow orange edge
            for coll in list(axR.collections):
                try:
                    off = coll.get_offsets()
                    if off is None or len(off) == 0:
                        continue
                    x = float(off[0, 0])
                    lvl = _nearest_level(x, levels)
                    c = v_map.get(lvl, Y2_COLOR)
                    coll.set_facecolor("none")
                    coll.set_edgecolor(c)
                except Exception:
                    pass

            # --- recolor errorbar containers (means + SD bars) ---
            # Capacity errorbars on axL
            for cont in list(getattr(axL, "containers", [])):
                if isinstance(cont, ErrorbarContainer):
                    try:
                        data_line, caplines, barlinecols = cont.lines
                        xdat = data_line.get_xdata()
                        if len(xdat) == 0:
                            continue
                        lvl = _nearest_level(float(xdat[0]), levels)
                        c = cap_map.get(lvl, Y1_COLOR)

                        data_line.set_color(c)
                        data_line.set_markerfacecolor(c)
                        data_line.set_markeredgecolor(c)
                        for cl in caplines:
                            cl.set_color(c)
                        for blc in barlinecols:
                            blc.set_color(c)
                    except Exception:
                        pass

            # Voltage errorbars on axR
            for cont in list(getattr(axR, "containers", [])):
                if isinstance(cont, ErrorbarContainer):
                    try:
                        data_line, caplines, barlinecols = cont.lines
                        xdat = data_line.get_xdata()
                        if len(xdat) == 0:
                            continue
                        lvl = _nearest_level(float(xdat[0]), levels)
                        c = v_map.get(lvl, Y2_COLOR)

                        data_line.set_color(c)
                        # keep hollow marker look
                        data_line.set_markerfacecolor("none")
                        data_line.set_markeredgecolor(c)
                        for cl in caplines:
                            cl.set_color(c)
                        for blc in barlinecols:
                            blc.set_color(c)
                    except Exception:
                        pass

    def _relabel_bottom_legend_vc(fig, vc_levels):
        """
        Bottom row curves for VC-shift plot:
          - solid lines
          - legend labels ONLY: "{vc} wt%"
          (colors can remain whatever was applied previously; this only cleans labels)
        """
        groups = _group_axes_by_position(fig)
        bot_keys = _row_keys(groups, "bottom")
        for key in bot_keys:
            ax_list = groups.get(key, [])
            if len(ax_list) != 1:
                continue
            ax = ax_list[0]
            seen = set()
            for ln in ax.get_lines():
                lab = ln.get_label() or ""
                m = re.search(r"VC\s*([0-9.]+)", lab)
                ln.set_linestyle("-")
                if m:
                    vc = float(m.group(1))
                    lvl = _nearest_level(vc, vc_levels)
                    new_lab = f"{int(lvl)} wt%"
                    if new_lab in seen:
                        ln.set_label("_nolegend_")
                    else:
                        ln.set_label(new_lab)
                        seen.add(new_lab)
                else:
                    ln.set_label("_nolegend_")
            _legend_inside(ax)

    def _relabel_bottom_legend_fec(fig, fec_levels):
        """
        Bottom row curves for FEC-shift plot:
          - solid lines
          - legend labels ONLY: "{fec} wt%"
        """
        groups = _group_axes_by_position(fig)
        bot_keys = _row_keys(groups, "bottom")
        for key in bot_keys:
            ax_list = groups.get(key, [])
            if len(ax_list) != 1:
                continue
            ax = ax_list[0]
            seen = set()
            for ln in ax.get_lines():
                lab = ln.get_label() or ""
                m = re.search(r"FEC\s*([0-9.]+)", lab)
                ln.set_linestyle("-")
                if m:
                    fec = float(m.group(1))
                    lvl = _nearest_level(fec, fec_levels)
                    new_lab = f"{int(lvl)} wt%"
                    if new_lab in seen:
                        ln.set_label("_nolegend_")
                    else:
                        ln.set_label(new_lab)
                        seen.add(new_lab)
                else:
                    ln.set_label("_nolegend_")
            _legend_inside(ax)

    # ---------------- build data (Trial 3 only) ----------------
    files_with_source = find_discharge_files(base_dir, old_directory, temp_tag="-51")
    if not files_with_source:
        print("No matching .xlsx files found for -51C.")
        return

    lookup = load_lookup_table(lookup_table_path)

    groups = {}
    for p, _src in files_with_source:
        code = get_cell_code(p)
        groups.setdefault(code, []).append(p)

    cell_meta = {}
    for cell_code in groups.keys():
        electrolyte = get_electrolyte_name(cell_code, lookup)
        cell_meta[cell_code] = {
            "electrolyte": electrolyte,
            "total_additive": get_total_additive(electrolyte),
        }

    # alpha_colors required by plot funcs (we recolor top row post-hoc anyway)
    unique_alphas = sorted({get_alpha_prefix(c) for c in groups.keys()})
    cmap = plt.get_cmap("tab20")
    alpha_colors = {a: cmap(i % cmap.N) for i, a in enumerate(unique_alphas)}

    dfm = compute_cell_discharge_metrics(
        groups, lookup, cell_meta,
        fec_levels=GRID_FEC_LEVELS, vc_levels=GRID_VC_LEVELS,
        capacity_voltage=CAPACITY_VOLTAGE,
    )
    dfm = dfm[dfm["Trial"] == 3].copy()

    cap_col = f"cap_at_{CAPACITY_VOLTAGE:.1f}V"
    v_col = "MidpointDischargeV_V"

    # ---------------- no-save / no-close + plotting ----------------
    orig_savefig = Figure.savefig
    orig_close = plt.close
    try:
        Figure.savefig = lambda self, *args, **kwargs: None
        plt.close = lambda *args, **kwargs: None

        with plt.rc_context(rc):
            # --- VC shift combined ---
            before = list(plt.get_fignums())
            plot_effect_of_vc_shift_cap_and_midV_with_curves(
                dfm, groups, lookup, alpha_colors, plots_dir,
                title_prefix="",  # don't care; we'll drop suptitle anyway
                capacity_voltage=CAPACITY_VOLTAGE,
                cap_col=cap_col,
                v_col=v_col,
                v_label="V @ 50% Q (V)",
            )
            fid = _get_new_fig_id(before)
            fig_vc = plt.figure(fid) if fid is not None else plt.gcf()

            _remove_global_fig_legends(fig_vc)
            _drop_suptitle(fig_vc)
            _recolor_toprow_points_and_errorbars(fig_vc, GRID_VC_LEVELS)
            _relabel_bottom_legend_vc(fig_vc, GRID_VC_LEVELS)
            _set_toprow_ticks_and_ylabels_swapped(fig_vc)
            _bump_tick_fonts(fig_vc)
            _print_bestfits_vc_shift(fig_vc, dfm, cap_col, v_col)

            # --- FEC shift combined ---
            before = list(plt.get_fignums())
            plot_effect_of_fec_shift_cap_and_midV_with_curves(
                dfm, groups, lookup, alpha_colors, plots_dir,
                title_prefix="",
                capacity_voltage=CAPACITY_VOLTAGE,
                cap_col=cap_col,
                v_col=v_col,
                v_label="V @ 50% Q (V)",
            )
            fid = _get_new_fig_id(before)
            fig_fec = plt.figure(fid) if fid is not None else plt.gcf()

            _remove_global_fig_legends(fig_fec)
            _drop_suptitle(fig_fec)
            _recolor_toprow_points_and_errorbars(fig_fec, GRID_FEC_LEVELS)
            _relabel_bottom_legend_fec(fig_fec, GRID_FEC_LEVELS)
            _set_toprow_ticks_and_ylabels_swapped(fig_fec)
            _bump_tick_fonts(fig_fec)
            _print_bestfits_fec_shift(fig_fec, dfm, cap_col, v_col)

            plt.show()

    finally:
        Figure.savefig = orig_savefig
        plt.close = orig_close

def main_7():
    """
    Update of main_6 per latest tweaks:
      - Bottom row: keep the SAME blue gradient scheme as main_5 (darker with higher additive level)
      - Top row y-axis *tick label values* (SWITCHED):
          * Y1 (left axis, Cap)  tick labels ONLY on LEFTMOST panel
          * Y2 (right axis, Vmid) tick labels ONLY on RIGHTMOST panel
        …and ylabel TEXT the same way:
          * Cap ylabel only LEFTMOST (left axis)
          * Vmid ylabel only RIGHTMOST (right axis)
      - Top row points + errorbars:
          * Cap = Blues (by additive level)
          * Vmid = Oranges (by additive level)
      - Std-dev bars recolored to match
      - Discharge curves solid; legends inside each bottom subplot; legend text only additive wt% (no cell codes)
      - No fit lines; print fit params + R^2 to console
      - Drop figure suptitle
    """
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.container import ErrorbarContainer

    # ---------------- styling knobs ----------------
    FONT_TITLE = 18
    FONT_LABEL = 16
    FONT_TICK = 13
    FONT_LEGEND = 12

    Y1_COLOR = "tab:blue"    # capacity axis
    Y2_COLOR = "tab:orange"  # voltage axis

    # ----------- NEW: subplot letter + contrast knobs -----------
    SUBPLOT_LETTER_FONT = FONT_LABEL
    SUBPLOT_LETTER_X = 0.02
    SUBPLOT_LETTER_Y = 0.98

    # Bottom (V vs Q) curve contrast: spread across more of the colormap
    BOTTOM_BLUE_LO = 0.12
    BOTTOM_BLUE_HI = 0.98
    CURVE_LW = 2.4

    import os

    # ----------- NEW: output + sizing knobs -----------
    SAVE_DIR = r"C:\Users\benja\Downloads\Dilute THF Data\11_25_25\-51C_Repeats\plots_t19\_exports"
    os.makedirs(SAVE_DIR, exist_ok=True)

    SAVE_DPI = 150  # used for screen->inch conversion and for file export
    USE_SCREEN_SIZE = True  # True = use monitor size; False = use FIGSIZE_OVERRIDE
    SCREEN_FRAC = 0.95  # 0.95 fills most of the screen (avoid taskbar)
    FIGSIZE_OVERRIDE = None  # e.g. (18, 10) in inches, if you want manual control

    rc = {
        "figure.titlesize": FONT_TITLE,
        "axes.titlesize": FONT_TITLE - 2,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
    }
    # --- enforce replicate marker mapping for top plots (CellNumber == replicate) ---
    # Replicate 1 = triangle, Replicate 2 = square, Replicate 3 = diamond
    # Enforce replicate marker mapping (make sure all are unique)
    # Rep 1 triangle, Rep 2 square, Rep 3 diamond, Rep 4 circle (or pick another)
    CELLNUM_MARKERS.update({1: "^", 2: "s", 3: "D", 4: "o"})

    # ---------------- helpers ----------------
    def _get_new_fig_id(before_ids):
        after = set(plt.get_fignums())
        new = sorted(list(after - set(before_ids)))
        return new[-1] if new else None

    def _get_screen_figsize_inches(dpi=100, frac=0.95, fallback=(18, 10)):
        """
        Returns (w_in, h_in) sized to your primary display.
        Uses tkinter to query screen pixel dims, then converts to inches using dpi.
        """
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            w_px = root.winfo_screenwidth()
            h_px = root.winfo_screenheight()
            root.destroy()
            return (max(6.0, (w_px * frac) / float(dpi)),
                    max(4.0, (h_px * frac) / float(dpi)))
        except Exception:
            return fallback

    def add_top_left_legends(fig, dfm_trial, rep_loc="lower center", rep_bbox=(0.5, 0.10)):
        """
        Single-call legend builder for the TOP-LEFT top-row panel:
          - Replicate legend (top of the stack) using your existing placement style
          - Metric legend (filled vs hollow) directly BELOW replicate legend
        """
        from matplotlib.lines import Line2D

        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")
        if not top_keys:
            return

        # top-left panel = first key (row_keys sorts left->right)
        key = top_keys[0]
        ax_pair = groups.get(key, [])
        if len(ax_pair) != 2:
            return

        axL, axR = _pick_left_right_axes(ax_pair)
        if axL is None or axR is None:
            return

        # ---------- replicate legend handles ----------
        d = dfm_trial.copy()
        if "CellNumber" not in d.columns:
            d["CellNumber"] = d["CellCode"].apply(get_cell_number)

        present = sorted(set(d["CellNumber"].dropna().astype(int).tolist()))
        rep_marker_map = {1: "^", 2: "s", 3: "D", 4: "o"}  # keep in sync with CELLNUM_MARKERS.update(...)
        present = [r for r in present if r in rep_marker_map]

        rep_handles = [
            Line2D([0], [0],
                   marker=rep_marker_map[r], linestyle="None",
                   color="black", markersize=8, label=f"Rep {r}")
            for r in present
        ]

        # ---------- metric legend handles (filled vs hollow) ----------
        cap_lab = (axL.get_ylabel() or "Cap @2.5V").strip()
        v_lab = (axR.get_ylabel() or "V@50%Q").strip()

        metric_handles = [
            Line2D([0], [0],
                   marker="o", linestyle="None",
                   markerfacecolor="black", markeredgecolor="black",
                   markersize=7, label=cap_lab),
            Line2D([0], [0],
                   marker="o", linestyle="None",
                   markerfacecolor="none", markeredgecolor="black",
                   markersize=7, label=v_lab),
        ]

        # Remove any existing legends on this axis (avoid duplicates)
        try:
            old = axL.get_legend()
            if old is not None:
                old.remove()
        except Exception:
            pass

        # ---------- 1) Replicate legend (top of stack) ----------
        # Keep YOUR replicate placement (same style), but a bit higher so room exists below.
        # You can tweak rep_bbox y if needed.
        rep_leg = None
        if rep_handles:
            rep_leg = axL.legend(
                handles=rep_handles,
                loc=rep_loc,
                bbox_to_anchor=rep_bbox,  # <-- replicate placement anchor
                ncol=2,  # two rows for 3–4 entries
                frameon=False,
                title="Replicate",
                borderaxespad=0.0,
                handletextpad=0.6,
                columnspacing=1.0,
            )

        # ---------- 2) Metric legend (below replicate) ----------
        # Place it just below replicate. Use the same anchor x, smaller y.
        # If you change rep_bbox, metric y should stay a bit smaller.
        metric_bbox = (rep_bbox[0], rep_bbox[1] - 0.075)  # <-- stack below
        metric_leg = axL.legend(
            handles=metric_handles,
            loc=rep_loc,
            bbox_to_anchor=metric_bbox,
            ncol=1,
            frameon=False,
            title=None,
            borderaxespad=0.0,
            handletextpad=0.6,
            columnspacing=0.8,
        )

        # Add replicate legend back on top (so both show)
        if rep_leg is not None:
            axL.add_artist(rep_leg)

    def add_metric_fill_legend_to_top_left(fig):
        """
        Add a small legend INSIDE the TOP-LEFT top-row panel that explains:
          - filled marker = Y1 metric (uses left-axis ylabel text)
          - hollow marker = Y2 metric (uses right-axis ylabel text)

        This keeps any existing legend(s) on that axis by re-adding them.
        """
        from matplotlib.lines import Line2D

        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")
        if not top_keys:
            return

        # top-left panel = first key (row_keys sorts left->right)
        key = top_keys[0]
        ax_pair = groups.get(key, [])
        if len(ax_pair) != 2:
            return

        axL, axR = _pick_left_right_axes(ax_pair)
        if axL is None or axR is None:
            return

        # Use the *actual* axis label text (fallbacks if empty)
        cap_lab = (axL.get_ylabel() or "Cap @2.5V").strip()
        v_lab = (axR.get_ylabel() or "V@50%Q").strip()

        handles = [
            Line2D([0], [0],
                   marker="o", linestyle="None",
                   markerfacecolor="black", markeredgecolor="black",
                   markersize=7, label=cap_lab),
            Line2D([0], [0],
                   marker="o", linestyle="None",
                   markerfacecolor="none", markeredgecolor="black",
                   markersize=7, label=v_lab),
        ]

        # Preserve any existing legend on axL
        leg_existing = axL.get_legend()

        # Slightly left-shifted placement
        metric_leg = axL.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.01, 0.02),  # <-- move left (more negative = more left)
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
            handletextpad=0.6,
            columnspacing=0.8,
        )

        if leg_existing is not None:
            axL.add_artist(leg_existing)

    def _add_subplot_letters(fig, start_letter="a"):
        """
        Adds a), b), c)... to each PANEL:
          - top row panels: label only the LEFT axis of each twin-axes pair
          - bottom row panels: label the single axis
        Order: top row L->R then bottom row L->R
        """
        import string
        letters = string.ascii_lowercase

        groups = _group_axes_by_position(fig)

        panel_axes = []

        # top row panels (use left axis of twin pair)
        for key in _row_keys(groups, "top"):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, _axR = _pick_left_right_axes(ax_pair)
            if axL is not None:
                panel_axes.append(axL)

        # bottom row panels (single axis)
        for key in _row_keys(groups, "bottom"):
            ax_list = groups.get(key, [])
            if len(ax_list) == 1:
                panel_axes.append(ax_list[0])

        if not panel_axes:
            return

        start_idx = letters.index(start_letter)
        for i, ax in enumerate(panel_axes):
            if start_idx + i >= len(letters):
                break  # unlikely you exceed 26 panels here
            lab = f"{letters[start_idx + i]})"
            ax.text(
                SUBPLOT_LETTER_X, SUBPLOT_LETTER_Y, lab,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=SUBPLOT_LETTER_FONT,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.0),
                clip_on=False,
            )

    def _group_axes_by_position(fig):
        groups = {}
        for ax in fig.get_axes():
            b = ax.get_position().bounds  # (x0, y0, w, h)
            key = tuple(round(v, 4) for v in b)
            groups.setdefault(key, []).append(ax)
        return groups

    def _bold_top_panel_titles(fig):
        """Bold the top-row panel titles like 'VC 0 wt%' or 'FEC 1 wt%'."""
        groups = _group_axes_by_position(fig)
        for key in _row_keys(groups, "top"):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, _axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue
            t = axL.get_title() or ""
            if t.strip():
                axL.set_title(t, fontweight="bold")

    def _add_replicate_symbol_legend(fig, dfm_trial):
        from matplotlib.lines import Line2D

        # ensure CellNumber exists
        if "CellNumber" not in dfm_trial.columns:
            dfm_trial = dfm_trial.copy()
            dfm_trial["CellNumber"] = dfm_trial["CellCode"].apply(get_cell_number)

        present_reps = sorted(set(dfm_trial["CellNumber"].dropna().astype(int).tolist()))

        rep_marker_map = {1: "^", 2: "s", 3: "D", 4: "o"}  # keep in sync with update above
        handles = []
        for r in present_reps:
            if r in rep_marker_map:
                handles.append(
                    Line2D([0], [0], marker=rep_marker_map[r], linestyle="None",
                           color="black", markersize=8, label=f"Replicate {r}")
                )

        if not handles:
            return

        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.75, 0.995),
            ncol=min(4, len(handles)),
            frameon=False,
            title="Replicate (marker)",
            fontsize=FONT_LEGEND,
            title_fontsize=FONT_LEGEND,
        )

    def set_all_ticks_inward(fig):
        for ax in fig.get_axes():
            ax.tick_params(axis="both", which="both",
                           direction="in",
                           top=True, right=True)

    def _row_keys(groups, which="top"):
        if not groups:
            return []
        y0s = [k[1] for k in groups.keys()]
        target = max(y0s) if which == "top" else min(y0s)
        keys = [k for k in groups.keys() if abs(k[1] - target) < 0.02]
        return sorted(keys, key=lambda k: k[0])  # left->right

    def _pick_left_right_axes(ax_list):
        if len(ax_list) != 2:
            return None, None
        a0, a1 = ax_list
        if str(a0.yaxis.get_ticks_position()).lower() == "right":
            return a1, a0
        if str(a1.yaxis.get_ticks_position()).lower() == "right":
            return a0, a1
        return a0, a1

    def _nearest_level(x, levels):
        levels = np.asarray(levels, dtype=float)
        return float(levels[np.argmin(np.abs(levels - float(x)))])

    def _palette(levels, cmap_name, lo=0.35, hi=0.90):
        lv = list(sorted([float(v) for v in levels]))
        cmap = plt.get_cmap(cmap_name)
        if len(lv) == 1:
            return {lv[0]: cmap(hi)}
        vals = np.linspace(lo, hi, len(lv))
        return {lv[i]: cmap(vals[i]) for i in range(len(lv))}

    def _remove_global_fig_legends(fig):
        for leg in list(getattr(fig, "legends", [])):
            try:
                leg.remove()
            except Exception:
                pass

    def _drop_suptitle(fig):
        try:
            if getattr(fig, "_suptitle", None) is not None:
                fig._suptitle.set_text("")
                fig._suptitle.set_visible(False)
        except Exception:
            pass

    def add_replicate_marker_legend_to_top_left(fig, dfm_trial):
        """
        Put replicate marker legend onto the TOP-LEFT top-row subplot (the (1,1) panel),
        in TWO ROWS at the BOTTOM of that subplot.
        """
        from matplotlib.lines import Line2D

        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")
        if not top_keys:
            return

        # top-left panel = first key (row_keys sorts left->right)
        key = top_keys[0]
        ax_pair = groups.get(key, [])
        if len(ax_pair) != 2:
            return
        axL, _axR = _pick_left_right_axes(ax_pair)
        if axL is None:
            return

        d = dfm_trial.copy()
        if "CellNumber" not in d.columns:
            d["CellNumber"] = d["CellCode"].apply(get_cell_number)

        present = sorted(set(d["CellNumber"].dropna().astype(int).tolist()))
        rep_marker_map = {1: "^", 2: "s", 3: "D", 4: "o"}  # keep in sync with CELLNUM_MARKERS.update(...)
        present = [r for r in present if r in rep_marker_map]
        if not present:
            return

        handles = [
            Line2D([0], [0], marker=rep_marker_map[r], linestyle="None",
                   color="black", markersize=8, label=f"Rep {r}")
            for r in present
        ]

        # Keep any existing legend
        leg1 = axL.get_legend()

        # Two rows: ncol=2 gives 2 rows for 3–4 entries
        rep_leg = axL.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            frameon=False,
            title="Replicate",
            borderaxespad=0.0,
            handletextpad=0.6,
            columnspacing=1.0,
        )

        if leg1 is not None:
            axL.add_artist(leg1)

    def _legend_inside(ax):
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(loc="best", frameon=False, fontsize=FONT_LEGEND)

    def _bump_tick_fonts(fig):
        for ax in fig.get_axes():
            ax.tick_params(axis="both", which="both", labelsize=FONT_TICK)

    def _style_y_axes(axL, axR):
        # left axis styling (capacity)
        axL.spines["left"].set_color(Y1_COLOR)
        axL.tick_params(axis="y", colors=Y1_COLOR)
        axL.yaxis.label.set_color(Y1_COLOR)
        # right axis styling (voltage)
        axR.spines["right"].set_color(Y2_COLOR)
        axR.tick_params(axis="y", colors=Y2_COLOR)
        axR.yaxis.label.set_color(Y2_COLOR)

    def _set_toprow_ticks_and_ylabels_switched(fig):
        """
        Top row only (SWITCHED vs main_6):
          - Show Y1 (left-axis) tick labels + ylabel ONLY on LEFTMOST panel
          - Show Y2 (right-axis) tick labels + ylabel ONLY on RIGHTMOST panel
        """
        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")
        if not top_keys:
            return

        leftmost = top_keys[0]
        rightmost = top_keys[-1]

        cap_ylabel = None
        vmid_ylabel = None
        for key in top_keys:
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue
            cap_ylabel = cap_ylabel or (axL.get_ylabel() or "")
            vmid_ylabel = vmid_ylabel or (axR.get_ylabel() or "")
        cap_ylabel = cap_ylabel or "Capacity"
        vmid_ylabel = vmid_ylabel or "Voltage"

        for key in top_keys:
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue

            axL.tick_params(labelleft=(key == leftmost))    # Y1 ticks only leftmost
            axR.tick_params(labelright=(key == rightmost))  # Y2 ticks only rightmost

            axL.set_ylabel(cap_ylabel if key == leftmost else "")
            axR.set_ylabel(vmid_ylabel if key == rightmost else "")

            _style_y_axes(axL, axR)

    def _fit_stats(x, y, use_level_means=True):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 2:
            return np.nan, np.nan, np.nan

        if use_level_means:
            ux = np.unique(x)
            if ux.size < 2:
                return np.nan, np.nan, np.nan
            xs, ys = [], []
            for u in ux:
                yy = y[x == u]
                yy = yy[np.isfinite(yy)]
                if yy.size:
                    xs.append(u)
                    ys.append(float(np.mean(yy)))
            x = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=float)
            if x.size < 2:
                return np.nan, np.nan, np.nan

        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = np.nan if ss_tot <= 0 else (1.0 - ss_res / ss_tot)
        return float(slope), float(intercept), float(r2)

    def _print_bestfits_vc_shift(fig, df_use, cap_col, v_col):
        groups = _group_axes_by_position(fig)
        for key in _row_keys(groups, "top"):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, _axR = _pick_left_right_axes(ax_pair)
            title = axL.get_title() or ""
            m = re.search(r"FEC\s*([0-9.]+)", title)
            if not m:
                continue
            fec = float(m.group(1))

            sub = df_use.copy()
            sub["FEC_wt"] = pd.to_numeric(sub["FEC_wt"], errors="coerce")
            sub["VC_wt"] = pd.to_numeric(sub["VC_wt"], errors="coerce")
            sub = sub[sub["FEC_wt"] == fec]

            s1, b1, r1 = _fit_stats(sub["VC_wt"].to_numpy(float), sub[cap_col].to_numpy(float), use_level_means=True)
            s2, b2, r2 = _fit_stats(sub["VC_wt"].to_numpy(float), sub[v_col].to_numpy(float), use_level_means=True)

            print(f"[VC-shift | Trial 3] FEC={fec:g} wt%")
            print(f"  Cap  (mean-per-VC):   slope={s1:.6g}, intercept={b1:.6g}, R^2={r1:.4f}")
            print(f"  Vmid (mean-per-VC):   slope={s2:.6g}, intercept={b2:.6g}, R^2={r2:.4f}")

    def _print_bestfits_fec_shift(fig, df_use, cap_col, v_col):
        groups = _group_axes_by_position(fig)
        for key in _row_keys(groups, "top"):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, _axR = _pick_left_right_axes(ax_pair)
            title = axL.get_title() or ""
            m = re.search(r"VC\s*([0-9.]+)", title)
            if not m:
                continue
            vc = float(m.group(1))

            sub = df_use.copy()
            sub["FEC_wt"] = pd.to_numeric(sub["FEC_wt"], errors="coerce")
            sub["VC_wt"] = pd.to_numeric(sub["VC_wt"], errors="coerce")
            sub = sub[sub["VC_wt"] == vc]

            s1, b1, r1 = _fit_stats(sub["FEC_wt"].to_numpy(float), sub[cap_col].to_numpy(float), use_level_means=True)
            s2, b2, r2 = _fit_stats(sub["FEC_wt"].to_numpy(float), sub[v_col].to_numpy(float), use_level_means=True)

            print(f"[FEC-shift | Trial 3] VC={vc:g} wt%")
            print(f"  Cap  (mean-per-FEC):  slope={s1:.6g}, intercept={b1:.6g}, R^2={r1:.4f}")
            print(f"  Vmid (mean-per-FEC):  slope={s2:.6g}, intercept={b2:.6g}, R^2={r2:.4f}")

    def _recolor_toprow_points_and_errorbars(fig, levels):
        """
        Top row:
          - Capacity points + errorbars: BLUE shades (by x level)
          - Voltage points + errorbars: ORANGE shades (by x level)
        """
        cap_map = _level_color_map(levels)  # categorical per wt%

        v_map = _palette(levels, "Oranges", lo=0.35, hi=0.90)

        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")

        for key in top_keys:
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue

            # raw scatter collections
            for coll in list(axL.collections):  # cap (filled blue)
                try:
                    off = coll.get_offsets()
                    if off is None or len(off) == 0:
                        continue
                    x = float(off[0, 0])
                    lvl = _nearest_level(x, levels)
                    c = cap_map.get(lvl, Y1_COLOR)
                    coll.set_facecolor(c)
                    coll.set_edgecolor(c)
                except Exception:
                    pass

            for coll in list(axR.collections):  # v (hollow orange edge)
                try:
                    off = coll.get_offsets()
                    if off is None or len(off) == 0:
                        continue
                    x = float(off[0, 0])
                    lvl = _nearest_level(x, levels)
                    c = v_map.get(lvl, Y2_COLOR)
                    coll.set_facecolor("none")
                    coll.set_edgecolor(c)
                except Exception:
                    pass

            # errorbar containers (means + SD bars)
            for cont in list(getattr(axL, "containers", [])):
                if isinstance(cont, ErrorbarContainer):
                    try:
                        data_line, caplines, barlinecols = cont.lines
                        xdat = data_line.get_xdata()
                        if len(xdat) == 0:
                            continue
                        lvl = _nearest_level(float(xdat[0]), levels)
                        c = cap_map.get(lvl, Y1_COLOR)

                        data_line.set_color(c)
                        data_line.set_markerfacecolor(c)
                        data_line.set_markeredgecolor(c)
                        for cl in caplines:
                            cl.set_color(c)
                        for blc in barlinecols:
                            blc.set_color(c)
                    except Exception:
                        pass

            for cont in list(getattr(axR, "containers", [])):
                if isinstance(cont, ErrorbarContainer):
                    try:
                        data_line, caplines, barlinecols = cont.lines
                        xdat = data_line.get_xdata()
                        if len(xdat) == 0:
                            continue
                        lvl = _nearest_level(float(xdat[0]), levels)
                        c = v_map.get(lvl, Y2_COLOR)

                        data_line.set_color(c)
                        data_line.set_markerfacecolor("none")
                        data_line.set_markeredgecolor(c)
                        for cl in caplines:
                            cl.set_color(c)
                        for blc in barlinecols:
                            blc.set_color(c)
                    except Exception:
                        pass
    # --- NEW: categorical colors for wt% levels (Y1 + discharge curves) ---
    LEVEL_COLORS_5 = {
        0.0:  "#1f77b4",  # blue
        1.0:  "#2ca02c",  # green
        2.0:  "#9467bd",  # purple
        5.0:  "#d62728",  # red
        10.0: "#8c564b",  # brown
    }

    def _level_color_map(levels):
        """
        Discrete high-contrast mapping for additive wt% levels.
        If a level isn't in LEVEL_COLORS_5 (unlikely), falls back to tab10 cycling.
        """
        lv = list(sorted([float(v) for v in levels]))
        tab = plt.get_cmap("tab10")
        out = {}
        for i, v in enumerate(lv):
            out[v] = LEVEL_COLORS_5.get(float(v), tab(i % 10))
        return out

    def _recolor_and_relabel_bottom_vc_blue(fig, vc_levels):
        """
        Bottom row curves for VC-shift:
          - blue gradient by VC level
          - solid lines
          - legend labels ONLY: "{vc} wt%"
        """
        blue_map = _level_color_map(vc_levels)  # categorical per wt%

        groups = _group_axes_by_position(fig)
        bot_keys = _row_keys(groups, "bottom")

        for key in bot_keys:
            ax_list = groups.get(key, [])
            if len(ax_list) != 1:
                continue
            ax = ax_list[0]
            seen = set()
            for ln in ax.get_lines():
                lab = ln.get_label() or ""
                m = re.search(r"VC\s*([0-9.]+)", lab)
                ln.set_linestyle("-")
                ln.set_linewidth(CURVE_LW)
                if m:
                    vc = float(m.group(1))
                    lvl = _nearest_level(vc, vc_levels)
                    ln.set_color(blue_map.get(lvl, Y1_COLOR))
                    new_lab = f"{int(lvl)} wt%"
                    if new_lab in seen:
                        ln.set_label("_nolegend_")
                    else:
                        ln.set_label(new_lab)
                        seen.add(new_lab)
                else:
                    ln.set_label("_nolegend_")
            _legend_inside(ax)

    def _recolor_and_relabel_bottom_fec_blue(fig, fec_levels):
        """
        Bottom row curves for FEC-shift:
          - blue gradient by FEC level
          - solid lines
          - legend labels ONLY: "{fec} wt%"
        """
        blue_map = _level_color_map(fec_levels)  # categorical per wt%

        groups = _group_axes_by_position(fig)
        bot_keys = _row_keys(groups, "bottom")

        for key in bot_keys:
            ax_list = groups.get(key, [])
            if len(ax_list) != 1:
                continue
            ax = ax_list[0]
            seen = set()
            for ln in ax.get_lines():
                lab = ln.get_label() or ""
                m = re.search(r"FEC\s*([0-9.]+)", lab)
                ln.set_linestyle("-")
                ln.set_linewidth(CURVE_LW)
                if m:
                    fec = float(m.group(1))
                    lvl = _nearest_level(fec, fec_levels)
                    ln.set_color(blue_map.get(lvl, Y1_COLOR))
                    new_lab = f"{int(lvl)} wt%"
                    if new_lab in seen:
                        ln.set_label("_nolegend_")
                    else:
                        ln.set_label(new_lab)
                        seen.add(new_lab)
                else:
                    ln.set_label("_nolegend_")
            _legend_inside(ax)

    # ---------------- build data (Trial 3 only) ----------------
    files_with_source = find_discharge_files(base_dir, old_directory, temp_tag="-51")
    if not files_with_source:
        print("No matching .xlsx files found for -51C.")
        return

    lookup = load_lookup_table(lookup_table_path)

    groups = {}
    for p, _src in files_with_source:
        code = get_cell_code(p)
        groups.setdefault(code, []).append(p)

    cell_meta = {}
    for cell_code in groups.keys():
        electrolyte = get_electrolyte_name(cell_code, lookup)
        cell_meta[cell_code] = {
            "electrolyte": electrolyte,
            "total_additive": get_total_additive(electrolyte),
        }

    # --- COPY/PASTE PATCH: replace your alpha colormap block with this ---

    unique_alphas = sorted({get_alpha_prefix(c) for c in groups.keys()})

    # High-contrast / colorblind-friendly categorical palette (hex)
    high_contrast_colors = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#17becf",  # cyan
        "#7f7f7f",  # gray
    ]

    alpha_colors = {
        a: high_contrast_colors[i % len(high_contrast_colors)]
        for i, a in enumerate(unique_alphas)
    }

    dfm = compute_cell_discharge_metrics(
        groups, lookup, cell_meta,
        fec_levels=GRID_FEC_LEVELS, vc_levels=GRID_VC_LEVELS,
        capacity_voltage=CAPACITY_VOLTAGE,
    )
    dfm = dfm[dfm["Trial"] == 3].copy()
    debug_marker_collisions(dfm, trial=3)
    cap_col = f"cap_at_{CAPACITY_VOLTAGE:.1f}V"
    v_col = "MidpointDischargeV_V"

    # ---------------- no-save / no-close + plotting ----------------
    orig_savefig = Figure.savefig
    orig_close = plt.close
    try:
        Figure.savefig = lambda self, *args, **kwargs: None
        plt.close = lambda *args, **kwargs: None

        with plt.rc_context(rc):
            # --- VC shift combined ---
            before = list(plt.get_fignums())
            plot_effect_of_vc_shift_cap_and_midV_with_curves(
                dfm, groups, lookup, alpha_colors, plots_dir,
                title_prefix="",
                capacity_voltage=CAPACITY_VOLTAGE,
                cap_col=cap_col,
                v_col=v_col,
                v_label="V @ 50% Q (V)",
            )
            fid = _get_new_fig_id(before)
            fig_vc = plt.figure(fid) if fid is not None else plt.gcf()

            _remove_global_fig_legends(fig_vc)
            _drop_suptitle(fig_vc)
            _bold_top_panel_titles(fig_vc)

            _recolor_toprow_points_and_errorbars(fig_vc, GRID_VC_LEVELS)
            _recolor_and_relabel_bottom_vc_blue(fig_vc, GRID_VC_LEVELS)
            _set_toprow_ticks_and_ylabels_switched(fig_vc)
            _bump_tick_fonts(fig_vc)
            _print_bestfits_vc_shift(fig_vc, dfm, cap_col, v_col)
            #add_replicate_marker_legend_to_top_left(fig_vc, dfm[dfm["Trial"] == 3],)
            #add_metric_fill_legend_to_top_left(fig_vc)
            add_top_left_legends(fig_vc, dfm[dfm["Trial"] == 3])
            #add_top_left_legends(fig_fec, dfm[dfm["Trial"] == 3])

            # --- FEC shift combined ---
            before = list(plt.get_fignums())
            plot_effect_of_fec_shift_cap_and_midV_with_curves(
                dfm, groups, lookup, alpha_colors, plots_dir,
                title_prefix="",
                capacity_voltage=CAPACITY_VOLTAGE,
                cap_col=cap_col,
                v_col=v_col,
                v_label="V @ 50% Q (V)",
            )
            fid = _get_new_fig_id(before)
            fig_fec = plt.figure(fid) if fid is not None else plt.gcf()

            _remove_global_fig_legends(fig_fec)
            _drop_suptitle(fig_fec)
            _bold_top_panel_titles(fig_fec)

            _recolor_toprow_points_and_errorbars(fig_fec, GRID_FEC_LEVELS)
            _recolor_and_relabel_bottom_fec_blue(fig_fec, GRID_FEC_LEVELS)
            _set_toprow_ticks_and_ylabels_switched(fig_fec)
            _bump_tick_fonts(fig_fec)
            _print_bestfits_fec_shift(fig_fec, dfm, cap_col, v_col)
            #add_replicate_marker_legend_to_top_left(fig_fec, dfm[dfm["Trial"] == 3])
            #add_metric_fill_legend_to_top_left(fig_fec)
            #add_top_left_legends(fig_vc, dfm[dfm["Trial"] == 3])
            add_top_left_legends(fig_fec, dfm[dfm["Trial"] == 3])
            set_all_ticks_inward(fig_vc)
            set_all_ticks_inward(fig_fec)
            _add_subplot_letters(fig_vc, start_letter="a")
            _add_subplot_letters(fig_fec, start_letter="a")
            # ----------- NEW: resize to screen (or override) and save final figs -----------
            figsize_in = FIGSIZE_OVERRIDE
            if figsize_in is None and USE_SCREEN_SIZE:
                figsize_in = _get_screen_figsize_inches(dpi=SAVE_DPI, frac=SCREEN_FRAC)

            if figsize_in is not None:
                fig_vc.set_size_inches(figsize_in[0], figsize_in[1], forward=True)
                fig_fec.set_size_inches(figsize_in[0], figsize_in[1], forward=True)

            # Use orig_savefig because Figure.savefig is monkeypatched to no-op above
            out_vc = os.path.join(SAVE_DIR, "vc_shift_cap_and_midV_trial3.png")
            out_fec = os.path.join(SAVE_DIR, "fec_shift_cap_and_midV_trial3.png")

            orig_savefig(fig_vc, out_vc, dpi=SAVE_DPI, bbox_inches="tight")
            orig_savefig(fig_fec, out_fec, dpi=SAVE_DPI, bbox_inches="tight")

            print(f"Saved:\n  {out_vc}\n  {out_fec}")

            plt.show()

    finally:
        Figure.savefig = orig_savefig
        plt.close = orig_close


if __name__ == "__main__":
    main_7()



