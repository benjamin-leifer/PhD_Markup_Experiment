import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# --- user settings ---
base_dir = r"C:\Users\benja\Downloads\Dilute THF Data\11_25_25\-51C_Repeats"
old_directory = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\-51C_discharges'
lookup_table_path = r"C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx"
plots_dir = os.path.join(base_dir, "plots_t5")
os.makedirs(plots_dir, exist_ok=True)

REF_CAP_MAH = 4.0
REF_SPEC_MAH = 160.6
CONV_AH_TO_MAHG = 1000.0 * REF_SPEC_MAH / REF_CAP_MAH  # 40150.0
non_16mm_Ref = REF_CAP_MAH *1.606/2

# ---------- helpers ----------
def find_discharge_files(base_dir, old_directory=None, temp_tag=None):
    """
    Search `base_dir` and optional `old_directory` for discharge .xlsx files.
    Returns a list of (path, source) tuples where source is 'base' or 'old'.
    Deduplicates paths while preserving order (base_dir files first).
    If temp_tag is provided (e.g. "-51"), only files whose normalized path contains
    the tag (or tag+'c') are included.
    """
    dirs_to_search = []
    if base_dir and os.path.isdir(base_dir):
        dirs_to_search.append((base_dir, "base"))
    if old_directory and os.path.isdir(old_directory):
        dirs_to_search.append((old_directory, "old"))

    all_results = []
    seen = set()
    temp_tag_norm = temp_tag.lower() if temp_tag else None

    for root_dir, src in dirs_to_search:
        for r, _, files in os.walk(root_dir):
            for fn in sorted(files):
                if not (fn.lower().endswith(".xlsx") and "dis" in fn.lower()):
                    continue
                p = os.path.join(r, fn)
                norm = os.path.normcase(os.path.normpath(p))
                # temperature filter (case-insensitive)
                if temp_tag_norm:
                    # match common variants like "-51" or "-51c"
                    if not (
                        temp_tag_norm in norm
                        or (temp_tag_norm + "c") in norm
                        or ("-" + temp_tag_norm.lstrip("-")) in norm
                        or ("-" + temp_tag_norm.lstrip("-") + "c") in norm
                    ):
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


# python
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
    Build 'Electrolyte-SampleNumber' for a given cell_code.
    Looks up Electrolyte by alpha / full code and extracts sample number from suffix.
    """
    electrolyte = get_electrolyte_name(cell_code, lookup)
    suffix = cell_code[2:]
    sample_num = suffix.lstrip("0") or suffix or ""

    if electrolyte and sample_num:
        return f"{electrolyte}-{sample_num}"
    elif electrolyte:
        return electrolyte
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


# ---------- aggregation helper ----------
def aggregate_discharge_for_cell(paths):
    x_all = []
    y_all = []
    for p in paths:
        try:
            x_spec, y_volt, _ = load_discharge_curve(p)
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
        #if chem_title:
        #   title_parts.append(chem_title)
        #title_parts.append("DTF/DTFV discharge")

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


# ---------- summary table ----------
def make_dtf_dtfv_summary(groups, out_dir, target_voltage=2.5):
    electrolyte_sets = {
        "DTF_new": ["HU", "HV", "HW", "HX"],
        "DTFV_new": ["IA", "IB", "IC", "ID", "IE", "IF"],
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


# --- main ---
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
    for cell_code, paths in groups.items():
        meta = cell_meta.get(cell_code, {})
        electrolyte = meta.get("electrolyte", "")
        total_add = meta.get("total_additive", 0.0)
        color = electrolyte_colors.get(electrolyte, "tab:gray")
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
                x_spec, y_volt, _ = load_discharge_curve(p,lookup=lookup)
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
        #fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")

    # by-alpha plots with chem info in titles
    #plot_groups_by_alpha(groups, lookup, plots_dir, cell_meta, electrolyte_colors, max_additive)

    # DTF / DTFV grouped plots with chem info in titles
    plot_dtf_dtfv_groups(groups, lookup, plots_dir, cell_meta, electrolyte_colors, max_additive)

    # summary table (unchanged)
    make_dtf_dtfv_summary(groups, plots_dir)


if __name__ == "__main__":
    main()
