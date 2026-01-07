import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# --- user settings ---
base_dir = r"C:\Users\benja\Downloads\Dilute THF Data\11_25_25\-51C_Repeats"
old_directory = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\-51C_discharges'
lookup_table_path = r"C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx"
plots_dir = os.path.join(base_dir, "plots_t9")
os.makedirs(plots_dir, exist_ok=True)

# --- FEC x VC master-grid settings ---
# alpha_select: None -> include all alphas; "HU" -> only HU##; "best" -> best-per-bin across all alphas
GRID_ALPHA_SELECT = None
# selection: "all" overlays all replicates; "best" selects best cell per (FEC,VC) bin (still overlays its replicates)
GRID_SELECTION = "best"
# Voltage used to score "best" (mAh/g at this voltage, via interpolation)
GRID_BEST_VOLTAGE = 2.0
# Always include these alpha prefixes as references in every subplot
GRID_REFERENCE_ALPHAS = ("FA", "EC")
# Grid layout
GRID_FEC_LEVELS = (1, 2, 5, 10)
GRID_VC_LEVELS = (0, 1, 2)
# Markers on curves (cell-code markers); set False if you want clean lines
GRID_SHOW_MARKERS = True


REF_CAP_MAH = 4.0
REF_SPEC_MAH = 160.6
CONV_AH_TO_MAHG = 1000.0 * REF_SPEC_MAH / REF_CAP_MAH  # 40150.0
non_16mm_Ref = REF_CAP_MAH * 1.606 / 2


# ---------- helpers ----------
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

            color = electrolyte_colors.get(electrolyte, "tab:gray")
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
                color = electrolyte_colors.get(electrolyte, "tab:gray")
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

        color = electrolyte_colors.get(electrolyte, "tab:gray")

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


# ---------- single-figure master grid ----------
def plot_fec_vc_master_grid(groups, lookup, out_dir, cell_meta, electrolyte_colors, max_additive,
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

    # consistent marker per cell code (across all panels)
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]
    eligible_cells = sorted(groups.keys())
    cell_marker = {cc: markers[i % len(markers)] for i, cc in enumerate(eligible_cells)}

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

                color = electrolyte_colors.get(electrolyte, "tab:gray")
                lw = compute_linewidth(total_add, max_additive)
                marker = cell_marker.get(ref_cc, "o")
                ls = get_line_style_for_alpha(ref_cc[:2])

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

                color = electrolyte_colors.get(electrolyte, "tab:gray")
                lw = compute_linewidth(total_add, max_additive)
                marker = cell_marker.get(cc, "o")
                ls = get_line_style_for_alpha(cc[:2])

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

    # Build global color mapping by electrolyte
    unique_electrolytes = sorted(
        set(m["electrolyte"] for m in cell_meta.values() if m["electrolyte"])
    )
    color_palette = [
        "tab:blue", "tab:orange", "tab:green", "tab:red",
        "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        "tab:olive", "tab:cyan"
    ]
    electrolyte_colors = {
        e: color_palette[i % len(color_palette)]
        for i, e in enumerate(unique_electrolytes)
    }

    max_additive = max(
        (m["total_additive"] for m in cell_meta.values()),
        default=0.0
    )

    # --- per-cell plots ---
    # for cell_code, paths in groups.items():
    #     meta = cell_meta.get(cell_code, {})
    #     electrolyte = meta.get("electrolyte", "")
    #     total_add = meta.get("total_additive", 0.0)
    #     color = electrolyte_colors.get(electrolyte, "tab:gray")
    #     lw = compute_linewidth(total_add, max_additive)
    #
    #     anode, cathode = get_anode_cathode(cell_code, lookup)
    #     chem_str = ""
    #     if cathode or anode:
    #         if cathode and anode:
    #             chem_str = f"{cathode}|{anode}"
    #         elif cathode:
    #             chem_str = cathode
    #         else:
    #             chem_str = anode
    #
    #     title_parts = [cell_code]
    #     if chem_str:
    #         title_parts.append(chem_str)
    #     if electrolyte:
    #         title_parts.append(electrolyte)
    #     title_label = " - ".join(title_parts)
    #
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     plotted_any = False
    #     for p in paths:
    #         try:
    #             x_spec, y_volt, _ = load_discharge_curve(p, lookup=lookup)
    #         except Exception as e:
    #             print(f"Skipping {p}: {e}")
    #             continue
    #         plotted_any = True
    #         legend_label = os.path.basename(p)
    #         ax.plot(x_spec, y_volt, label=legend_label, linewidth=lw, color=color)
    #
    #     if not plotted_any:
    #         plt.close(fig)
    #         continue
    #
    #     ax.set_xlabel("Discharge Specific Capacity (mAh/g)")
    #     ax.set_ylabel("Voltage (V)")
    #     ax.set_title(title_label)
    #     ax.set_ylim(0, 4.5)
    #     ax.set_xlim(-4, 160)
    #     ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    #     handles, labels = ax.get_legend_handles_labels()
    #     if handles:
    #         ax.legend(fontsize="xx-small", ncol=1, loc="best")
    #     plt.tight_layout()
    #     out_path = os.path.join(plots_dir, f"{cell_code}_discharge.png")
    #     # fig.savefig(out_path, dpi=300)
    #     plt.close(fig)
    #     print(f"Saved {out_path}")

    # ---------- one-figure master grid (FEC x VC) ----------
    grid_out_dir = os.path.join(plots_dir, "FECxVC_master_grid")
    plot_fec_vc_master_grid(
        groups, lookup, grid_out_dir,
        cell_meta=cell_meta,
        electrolyte_colors=electrolyte_colors,
        max_additive=max_additive,
        fec_levels=GRID_FEC_LEVELS,
        vc_levels=GRID_VC_LEVELS,
        alpha_select=GRID_ALPHA_SELECT,
        selection=GRID_SELECTION,
        reference_alphas=GRID_REFERENCE_ALPHAS,
        best_voltage=GRID_BEST_VOLTAGE,
        show_markers=GRID_SHOW_MARKERS,
    )

    # by-alpha plots (if you want them, uncomment)
    # plot_groups_by_alpha(groups, lookup, plots_dir, cell_meta, electrolyte_colors, max_additive)

    # DTF / DTFV grouped plots with solid(old)/dashed(new)
    #plot_dtf_dtfv_groups(groups, lookup, plots_dir, cell_meta, electrolyte_colors, max_additive)

    # Electrolyte-wise old vs new comparison plots (includes FA/EC where applicable)
    #plot_old_vs_new_comparisons(groups, lookup, plots_dir, cell_meta, electrolyte_colors, max_additive)

    # CSV index summarizing which alphas/cells are in old vs new for each electrolyte
    #make_old_new_index(groups, lookup, cell_meta, plots_dir)

    # summary table (unchanged)
    #make_dtf_dtfv_summary(groups, plots_dir)


    #make_dtf_dtfv_summary(groups, plots_dir)
    #make_alpha_best_summary(groups, lookup, cell_meta, plots_dir, target_voltage=2.0)



if __name__ == "__main__":
    main()
