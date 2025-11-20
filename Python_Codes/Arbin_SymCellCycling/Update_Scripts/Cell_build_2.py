import os
import zipfile
import io
import csv
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# USER CONFIG
# =============================================================================
ZIP_PATH = r"C:\Users\benja\Downloads\149 Data-20251114T183705Z-1-001.zip"

# Main folder containing the A6 Cell Build .xls/.xlsx files
DATA_ROOT_XLS = r"C:\Users\benja\Downloads\149 Data-20251114T183705Z-1-001\149 Data\149-001 A6 Cell Build"

# RT data is in a separate OLD folder as .xlsx
DATA_ROOT_RT_OLD = r"C:\Users\benja\Downloads\149 Data-20251114T183705Z-1-001\149 Data\149-001 3450 Cell Build\OLD"

# Output folder for plots/tables
OUT_DIR = r"C:\Users\benja\Downloads\149_outputs"

# If CSV capacities are already in Ah, set this to 1.0. If mAh, set to 1000.0
CSV_CAPACITY_DIVISOR = 1000.0

# =============================================================================
# FILE MAP  (by BASENAME; DataSource searches zip then roots)
# =============================================================================
cell_filename_map = {
    # RT Results (xlsx; may live in DATA_ROOT_RT_OLD)
    "AC-5 - RT": "149-001-Cell-AC5.xlsx",
    "AE-2 - RT": "149-001-Cell-AE2.xlsx",
    "AE-3 - RT": "149-001-Cell-BC3.xlsx",
    "BC-3 - RT": "149-001-Cell-BE1.xlsx",
    "BE-1 - RT": "149-001-Cell-CC3.xlsx",
    "BE-2 - RT": "149-001-Cell-CE1.xlsx",
    "CC-3 - RT": "149-001-Cell-CE2.xlsx",
    "CE-1 - RT": "149-001-Cell-AE3.xlsx",
    "CE-2 - RT": "149-001-Cell-BE2.xlsx",

    # -21°C cycling CSVs (in ZIP)
    "AC-5 - -21C": "Test51716_Cell-1_Results-1.csv",
    "AE-2 - -21C": "Test51714_Cell-1_Results-1.csv",
    "AE-3 - -21C": "Test51715_Cell-1_Results-1.csv",
    "BC-3 - -21C": "Test51719_Cell-1_Results-1.csv",
    "BE-1 - -21C": "Test51717_Cell-1_Results-1.csv",
    "BE-2 - -21C": "Test51718_Cell-1_Results-1.csv",
    "CC-3 - -21C": "Test51722_Cell-1_Results-1.csv",
    "CE-1 - -21C": "Test51720_Cell-1_Results-1.csv",
    "CE-2 - -21C": "Test51721_Cell-1_Results-1.csv",

    # Large-format formation XLS files (E1–F2)
    "E1":   "149-001-Cell-E1_Formation.xls",
    "E2":   "149-001-Cell-E2_Formation.xls",
    "F1":   "149-001-Cell-F1_Formation.xls",
    "F2":   "149-001-Cell-F2_Formation.xls"
}


# =============================================================================
# METADATA for styling (base cell name -> info)
# =============================================================================
@dataclass
class CellMeta:
    electrolyte_type: str  # "control" or "experimental"
    fill_mL: float | None  # 2.0 / 2.5 / 3.0 or None for large format


CELL_META = {
    # 2.0 mL fill
    "AC-5": CellMeta("control", 2.0),
    "AE-2": CellMeta("experimental", 2.0),
    "AE-3": CellMeta("experimental", 2.0),

    # 2.5 mL fill
    "BC-3": CellMeta("control", 2.5),
    "BE-1": CellMeta("experimental", 2.5),
    "BE-2": CellMeta("experimental", 2.5),

    # 3.0 mL fill
    "CC-3": CellMeta("control", 3.0),
    "CE-1": CellMeta("experimental", 3.0),
    "CE-2": CellMeta("experimental", 3.0),

    # Large format (fill not specified)
    "E1": CellMeta("experimental", None),
    "E2": CellMeta("experimental", None),
    "F1": CellMeta("control", None),
    "F2": CellMeta("control", None),
}

COLOR_MAP = {
    "control": "navy",
    "experimental": "darkorange"
}

LW_BY_FILL = {
    2.0: 1.5,   # thin
    2.5: 2.5,   # medium
    3.0: 3.5    # thick
}
DEFAULT_LW = 2.5

CYCLE_MARKERS = {
    1: "o",
    2: "s",
    3: "^",
    4: "D",
    5: "*"
}

ALPHA_BY_TEMP = {
    "-21C": 1.0,   # opaque
    "RT": 0.45     # faded
}


# =============================================================================
# DATA SOURCE HANDLER (adapted from your original)
# =============================================================================
def drop_non_monotonic(points, eps=1e-6):
    """Drop points where capacity goes backwards by more than eps."""
    cleaned = []
    last_q = None
    for q, v in points:
        if last_q is None:
            cleaned.append((q, v))
            last_q = q
        else:
            if q >= last_q - eps:
                cleaned.append((q, v))
                if q > last_q:
                    last_q = q
    return cleaned


class DataSource:
    """
    Searches:
      1) ZIP (if provided)
      2) Any number of filesystem roots (in order)
    """
    def __init__(self, zip_path=None, roots=None):
        self.zip_path = zip_path if zip_path and os.path.isfile(zip_path) else None
        self.zf = None
        if self.zip_path:
            try:
                self.zf = zipfile.ZipFile(self.zip_path, 'r')
                print(f"Using zip archive: {self.zip_path}")
            except Exception as e:
                print(f"Could not open zip '{self.zip_path}': {e}")
                self.zf = None
                self.zip_path = None

        self.roots = [r for r in (roots or []) if r and os.path.isdir(r)]
        if self.roots:
            print("Searching filesystem under:")
            for r in self.roots:
                print("  -", r)
        if not self.zf and not self.roots:
            print("WARNING: No valid ZIP or roots. Searching current folder only.")
            self.roots = [os.getcwd()]

    def resolve_pattern(self, filename_pattern):
        """Match by basename in zip first then filesystem roots."""
        bn = os.path.basename(filename_pattern)

        # zip
        if self.zf:
            for name in self.zf.namelist():
                if os.path.basename(name) == bn:
                    return ('zip', name)

        # filesystem roots
        for root in self.roots:
            for r, _, files in os.walk(root):
                for f in files:
                    if f == bn:
                        return ('fs', os.path.join(r, f))

        return (None, None)

    def open_csv(self, kind, path):
        if kind == 'zip':
            f = self.zf.open(path)
            return io.TextIOWrapper(f, encoding='cp1252', errors='ignore')
        if kind == 'fs':
            return open(path, 'r', encoding='cp1252', errors='ignore')
        raise ValueError(f"Unknown CSV source kind: {kind}")

    def open_excel(self, kind, path, **kwargs):
        ext = os.path.splitext(path)[1].lower()
        engine = 'xlrd' if ext == '.xls' else None
        if engine and 'engine' not in kwargs:
            kwargs['engine'] = engine

        if kind == 'zip':
            with self.zf.open(path) as f:
                data = f.read()
            return pd.read_excel(io.BytesIO(data), **kwargs)

        if kind == 'fs':
            return pd.read_excel(path, **kwargs)

        raise ValueError(f"Unknown Excel source kind: {kind}")


# =============================================================================
# HELPERS
# =============================================================================
def _get_col(df, candidates):
    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in cols}

    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]

    for cand in candidates:
        key = cand.lower()
        for low, orig in lower_map.items():
            if key in low:
                return orig

    return None


def parse_cycle_xls(data_source, file_info):
    """
    Parse Arbin .xls/.xlsx for cycles 1–5 from 'record' sheet.

    Logic preserved from your current script:
      - Capacity(Ah) treated as cumulative in time order
      - CCcharge / CCDischarge define phase
      - Drop rest/OCV
      - For cycle 1, drop step index < 4
      - Zero-shift capacity per (cycle, phase)
      - No sorting; keep time order
    """
    (kind, path) = file_info
    cycles = {i: {"charge": [], "discharge": []} for i in range(1, 6)}

    try:
        df = data_source.open_excel(kind, path, sheet_name="record")
    except Exception as e:
        print(f"Error reading Excel file {path}: {e}")
        return cycles

    if df is None or df.empty:
        print(f"'record' sheet empty for {path}")
        return cycles

    cycle_col = _get_col(df, ["Cycle Index", "Cycle", "Cycle_Index"])
    volt_col  = _get_col(df, ["Voltage(V)", "Voltage (V)", "Voltage"])
    curr_col  = _get_col(df, ["Current(A)", "Current (A)", "Current"])
    cap_col   = _get_col(df, ["Capacity(Ah)", "Capacity (Ah)", "Capacity"])

    if cycle_col is None or volt_col is None or cap_col is None:
        print(f"Missing columns in {path}. Columns={list(df.columns)}")
        return cycles

    # Step index and step name
    step_idx_col, step_name_col = None, None
    for col in df.columns:
        name = str(col).strip().lower()
        if "step" in name:
            if pd.api.types.is_numeric_dtype(df[col]) and step_idx_col is None:
                step_idx_col = col
            elif step_name_col is None:
                step_name_col = col
    if step_name_col is None:
        step_name_col = _get_col(df, ["Step Name", "StepName", "Step Type", "Label"])

    REST_TERMS = ("rest", "ocv", "open circuit", "wait", "pause", "idle")
    base_cap = {}

    for _, row in df.iterrows():
        # cycle
        try:
            cycle = int(row[cycle_col])
        except Exception:
            continue
        if cycle < 1 or cycle > 5:
            continue

        # step index
        step_idx = None
        if step_idx_col is not None:
            try:
                step_idx = int(row[step_idx_col])
            except Exception:
                step_idx = None
        if cycle == 1 and step_idx is not None and step_idx < 4:
            continue

        # voltage
        try:
            voltage = float(row[volt_col])
        except Exception:
            continue

        # current
        current = None
        if curr_col is not None:
            try:
                current = float(row[curr_col])
            except Exception:
                current = None

        # capacity absolute
        try:
            Q_abs = float(row[cap_col])
        except Exception:
            continue

        # step name
        step_name = ""
        if step_name_col is not None and not pd.isna(row.get(step_name_col, "")):
            step_name = str(row[step_name_col])
        step_norm = step_name.replace(" ", "").lower()

        # remove rests/ocv
        if step_name and any(term.replace(" ", "") in step_norm for term in REST_TERMS):
            continue

        # classify
        dest = None
        if step_name:
            if "cccharge" in step_norm:
                dest = "charge"
            elif "ccdischarge" in step_norm or "ccdischg" in step_norm:
                dest = "discharge"
            elif "cc" in step_norm:
                if current is not None and abs(current) > 1e-4:
                    dest = "charge" if current > 0 else "discharge"
                else:
                    continue
            else:
                continue
        else:
            if current is None or abs(current) < 1e-4:
                continue
            dest = "charge" if current > 0 else "discharge"

        key = (cycle, dest)
        if key not in base_cap:
            base_cap[key] = Q_abs
        Q_rel = Q_abs - base_cap[key]
        if Q_rel < -1e-6:
            continue

        cycles[cycle][dest].append((Q_rel, voltage))

    # clean monotonicity once at end
    for c in range(1, 6):
        for ph in ("charge", "discharge"):
            cycles[c][ph] = drop_non_monotonic(cycles[c][ph])

    return cycles


def parse_cycle_csv(data_source, file_info):
    """Parse -21°C CSV cycling data. Returns same structure as xls parser."""
    (kind, path) = file_info
    cycles = {i: {"charge": [], "discharge": []} for i in range(1, 6)}

    try:
        with data_source.open_csv(kind, path) as fh:
            reader = csv.reader(fh)

            # Skip to header starting with "Test"
            for row in reader:
                if row and row[0] == "Test":
                    break

            for row in reader:
                if len(row) < 18:
                    continue
                try:
                    cycle = int(float(row[8]))
                except Exception:
                    continue
                if cycle < 1 or cycle > 5:
                    continue

                try:
                    voltage = float(row[14])
                except Exception:
                    continue

                current = float(row[15]) if row[15].strip() != "" else 0.0
                chg_cap = float(row[16]) if row[16].strip() != "" else 0.0
                dis_cap = float(row[17]) if row[17].strip() != "" else 0.0

                if current > 0:
                    cycles[cycle]["charge"].append((chg_cap / CSV_CAPACITY_DIVISOR, voltage))
                elif current < 0:
                    cycles[cycle]["discharge"].append((dis_cap / CSV_CAPACITY_DIVISOR, voltage))
    except Exception as e:
        print(f"Error reading CSV file {path}: {e}")
        return cycles

    # sort by capacity (csv can be slightly unordered)
    for c in range(1, 6):
        for ph in ("charge", "discharge"):
            cycles[c][ph].sort(key=lambda x: x[0])
            cycles[c][ph] = drop_non_monotonic(cycles[c][ph])

    return cycles


def split_cell_key(cell_key: str):
    """
    Returns (base, temp_tag)
    Examples:
        "AC-5 - RT"   -> ("AC-5", "RT")
        "AC-5 - -21C" -> ("AC-5", "-21C")
        "E1"          -> ("E1", None)
    """
    if " - " not in cell_key:
        return cell_key.strip(), None
    base, suff = cell_key.split(" - ", 1)
    return base.strip(), suff.strip()


def avg_voltage(curve):
    """
    Compute average voltage via integral(V dQ) / Q_end.
    curve: list of (Q, V) in order
    """
    if not curve or len(curve) < 2:
        return np.nan, 0.0
    cap = np.array([p[0] for p in curve], dtype=float)
    volt = np.array([p[1] for p in curve], dtype=float)
    dQ = np.diff(cap)
    Vmid = 0.5 * (volt[1:] + volt[:-1])
    E = np.sum(Vmid * dQ)
    Qend = cap[-1]
    if Qend <= 1e-9:
        return np.nan, Qend
    return E / Qend, Qend


# =============================================================================
# LOAD ALL CELLS
# =============================================================================
os.makedirs(OUT_DIR, exist_ok=True)

data_source = DataSource(
    zip_path=ZIP_PATH,
    roots=[DATA_ROOT_XLS, DATA_ROOT_RT_OLD]
)

cell_cycles_data = {}
missing_cells = []

for cell, filename in cell_filename_map.items():
    kind, resolved_path = data_source.resolve_pattern(filename)
    if not resolved_path:
        print(f"WARNING: Could not locate file for cell {cell} (pattern '{filename}').")
        cell_cycles_data[cell] = None
        missing_cells.append(cell)
        continue

    print(f"Cell {cell}: using {kind} -> {resolved_path}")

    if filename.lower().endswith(".csv"):
        cycles = parse_cycle_csv(data_source, (kind, resolved_path))
    else:
        cycles = parse_cycle_xls(data_source, (kind, resolved_path))

    cell_cycles_data[cell] = cycles

    if all(len(cycles[i]["charge"]) == 0 and len(cycles[i]["discharge"]) == 0 for i in range(1, 6)):
        print(f"WARNING: No cycle data parsed for cell {cell}.")

if missing_cells:
    print("\nCells NOT found:", ", ".join(missing_cells))


# =============================================================================
# PLOTTING (styled per your rules)
# =============================================================================
plt.rcParams["figure.figsize"] = (7, 5)
plt.rcParams["font.size"] = 11

for cell_key, cycles in cell_cycles_data.items():
    if not cycles:
        continue

    base, temp_tag = split_cell_key(cell_key)
    meta = CELL_META.get(base, CellMeta("experimental", None))  # safe default

    color = COLOR_MAP[meta.electrolyte_type]
    lw = LW_BY_FILL.get(meta.fill_mL, DEFAULT_LW)
    alpha = ALPHA_BY_TEMP.get(temp_tag, 1.0)

    fig, ax = plt.subplots()

    for cyc in range(1, 6):
        cd = cycles.get(cyc, {"charge": [], "discharge": []})

        marker = CYCLE_MARKERS[cyc]
        # markevery chosen to show a few markers without clutter
        me = max(1, int(len(cd["charge"]) / 12)) if cd["charge"] else None

        if cd["charge"]:
            cap = [p[0] for p in cd["charge"]]
            volt = [p[1] for p in cd["charge"]]
            ax.plot(
                cap, volt,
                linestyle="-",
                color=color,
                linewidth=lw,
                alpha=alpha,
                marker=marker,
                markevery=me,
                markersize=5,
                label=f"Cycle {cyc} (chg)"
            )

        me_d = max(1, int(len(cd["discharge"]) / 12)) if cd["discharge"] else None
        if cd["discharge"]:
            cap = [p[0] for p in cd["discharge"]]
            volt = [p[1] for p in cd["discharge"]]
            ax.plot(
                cap, volt,
                linestyle="--",
                color=color,
                linewidth=lw,
                alpha=alpha,
                marker=marker,
                markevery=me_d,
                markersize=5,
                label=f"Cycle {cyc} (dis)"
            )

    ax.set_title(f"{cell_key}  |  {meta.electrolyte_type.upper()}  |  fill={meta.fill_mL or 'n/a'} mL")
    ax.set_xlabel("Capacity (Ah)" if CSV_CAPACITY_DIVISOR == 1 else "Capacity (Ah)")
    ax.set_ylabel("Voltage (V)")
    ax.set_ylim(2.5, 4.2)
    ax.set_xlim(left=0)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, f"{cell_key.replace(' ', '_').replace('/', '-')}_cycles1-5.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print("Saved plot:", out_png)


# =============================================================================
# SUMMARY TABLES (Avg charge/discharge V per cycle)
# =============================================================================
summary_rows = []

for cell_key, cycles in cell_cycles_data.items():
    if not cycles:
        continue

    base, temp_tag = split_cell_key(cell_key)
    meta = CELL_META.get(base, CellMeta("experimental", None))

    for cyc in range(1, 6):
        cd = cycles.get(cyc, {"charge": [], "discharge": []})

        Vc_avg, Qc = avg_voltage(cd["charge"])
        Vd_avg, Qd = avg_voltage(cd["discharge"])
        CE = (Qd / Qc * 100.0) if Qc and Qc > 1e-9 else np.nan

        summary_rows.append({
            "Cell": cell_key,
            "Base Cell": base,
            "Temp": temp_tag or "n/a",
            "Electrolyte Type": meta.electrolyte_type,
            "Fill (mL)": meta.fill_mL,
            "Cycle": cyc,
            "Charge Capacity (Ah)": Qc,
            "Discharge Capacity (Ah)": Qd,
            "Avg Charge V (V)": Vc_avg,
            "Avg Discharge V (V)": Vd_avg,
            "Coulombic Eff (%)": CE
        })

summary_df = pd.DataFrame(summary_rows)
summary_df["Coulombic Eff (%)"] = summary_df["Coulombic Eff (%)"].round(2)
summary_df["Avg Charge V (V)"] = summary_df["Avg Charge V (V)"].round(3)
summary_df["Avg Discharge V (V)"] = summary_df["Avg Discharge V (V)"].round(3)

out_csv = os.path.join(OUT_DIR, "Cell_Cycle_Summary.csv")
out_xlsx = os.path.join(OUT_DIR, "Cell_Cycle_Summary.xlsx")

summary_df.to_csv(out_csv, index=False)
summary_df.to_excel(out_xlsx, index=False)

print("Saved summary table:", out_csv)
print("Saved summary table:", out_xlsx)


# =============================================================================
# OPTIONAL: quick group comparison plots (cycle 1 only), still styled
# =============================================================================
base_fill_groups = {
    "2.0 mL fill": ["AC-5", "AE-2", "AE-3"],
    "2.5 mL fill": ["BC-3", "BE-1", "BE-2"],
    "3.0 mL fill": ["CC-3", "CE-1", "CE-2"]
}

base_electrolyte_groups = {
    "Control (LiPF6)": ["AC-5", "BC-3", "CC-3", "F1", "F2"],
    "Experimental (LiTFSI)": ["AE-2", "AE-3", "BE-1", "BE-2", "CE-1", "CE-2", "E1", "E2"]
}

suffixes = ["RT", "-21C"]

def plot_group_for_suffix(base_groups, suffix, filename_prefix):
    for group_name, base_cells in base_groups.items():
        keyed_cells = []
        for base in base_cells:
            key = f"{base} - {suffix}"
            if key in cell_cycles_data and cell_cycles_data[key]:
                keyed_cells.append(key)
        if not keyed_cells:
            continue

        fig, ax = plt.subplots()
        for cell_key in keyed_cells:
            cycles = cell_cycles_data[cell_key]
            base, temp_tag = split_cell_key(cell_key)
            meta = CELL_META.get(base, CellMeta("experimental", None))

            color = COLOR_MAP[meta.electrolyte_type]
            lw = LW_BY_FILL.get(meta.fill_mL, DEFAULT_LW)
            alpha = ALPHA_BY_TEMP.get(temp_tag, 1.0)

            c1 = cycles.get(1, {"charge": [], "discharge": []})
            if c1["charge"]:
                ax.plot(
                    [p[0] for p in c1["charge"]],
                    [p[1] for p in c1["charge"]],
                    "-", color=color, lw=lw, alpha=alpha,
                    label=f"{cell_key} chg"
                )
            if c1["discharge"]:
                ax.plot(
                    [p[0] for p in c1["discharge"]],
                    [p[1] for p in c1["discharge"]],
                    "--", color=color, lw=lw, alpha=alpha,
                    label=f"{cell_key} dis"
                )

        ax.set_title(f"{group_name} | Cycle 1 | {suffix}")
        ax.set_xlabel("Capacity (Ah)")
        ax.set_ylabel("Voltage (V)")
        ax.set_ylim(2.5, 4.2)
        ax.set_xlim(left=0)
        ax.grid(True, linestyle=":")
        ax.legend(fontsize=7, ncol=1, loc="best")
        fig.tight_layout()

        out_png = os.path.join(
            OUT_DIR,
            f"{filename_prefix}_{group_name.replace(' ', '_')}_{suffix.replace('-', '')}.png"
        )
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print("Saved group plot:", out_png)

for s in suffixes:
    plot_group_for_suffix(base_fill_groups, s, "Comparison_Fill")
    plot_group_for_suffix(base_electrolyte_groups, s, "Comparison_Electrolyte")
