import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER INPUT
# =========================

CELL_CODES = [

    "JC05",
    "JD04",

    "JE05",
    # "HU03",
]

# Where to search for -51C discharge xlsx files
base_dir = r"C:\Users\benja\Downloads\Dilute THF Data\11_25_25\-51C_Repeats"
old_directory = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\-51C_discharges"

# Lookup table used by your cold grid script (must contain an "Electrolyte" column)
lookup_table_path = r"C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx"

# Output folder
plots_dir = os.path.join(base_dir, "plots_-51C_selected_cells_dis1")
os.makedirs(plots_dir, exist_ok=True)

# Capacity normalization (same reference style as your grid scripts)
REF_CAP_MAH = 4.0
REF_SPEC_MAH_G = 160.6
CONV_AH_TO_MAHG = 1000.0 * REF_SPEC_MAH_G / REF_CAP_MAH


# =========================
# LOOKUP HELPERS
# =========================

def load_lookup_table(path: str) -> dict:
    """
    Loads a lookup table where the first column is treated as the key (cell code or alpha group).
    Must include a column named "Electrolyte" for the labeling behavior you want.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Lookup table not found: {path}")

    df = pd.read_excel(path)
    if df.shape[1] < 2:
        raise ValueError("Lookup table must have at least 2 columns (key + fields).")

    key_col = df.columns[0]
    df[key_col] = df[key_col].astype(str).str.upper()

    lookup = {}
    for _, row in df.iterrows():
        key = str(row[key_col]).strip().upper()
        if not key or key == "NAN":
            continue
        lookup[key] = {c: (row[c] if pd.notna(row[c]) else "") for c in df.columns[1:]}

    return lookup


def get_electrolyte_label(cell_code: str, lookup: dict) -> str:
    """
    Return electrolyte name for a full cell code (e.g., HU03 → DTF14).
    Falls back to alpha group if full code not present (HU, HT, etc.).
    """
    code = cell_code.upper()

    if code in lookup:
        elec = str(lookup[code].get("Electrolyte", "")).strip()
        if elec:
            return elec

    alpha = "".join([c for c in code if c.isalpha()])
    if alpha in lookup:
        elec = str(lookup[alpha].get("Electrolyte", "")).strip()
        if elec:
            return elec

    return cell_code  # final fallback


# =========================
# FILE FINDING + PARSING
# =========================

def get_cell_code_from_filename(path: str) -> str:
    """
    Extract cell code from filename:
      e.g., "BL-LL-HU03_RT_Discharge.xlsx" -> "HU03"
    """
    base = os.path.basename(path)
    root = base.split("_")[0]
    return root.split("-")[-1].upper()


def get_channel_sheet_name(path: str) -> str:
    """
    Arbin exports often have metadata in sheet 0 and channel data in sheet 1.
    """
    xls = pd.ExcelFile(path)
    if len(xls.sheet_names) < 2:
        raise ValueError(f"{path} has no channel sheet (expected >=2 sheets).")
    return xls.sheet_names[1]


def find_51C_discharge_files(base_dir: str, old_directory: str | None = None) -> list[str]:
    """
    Find .xlsx discharge files explicitly marked -51C
    in the filename or immediate parent folder name.
    """
    search_dirs = [d for d in [base_dir, old_directory] if d and os.path.isdir(d)]
    files_out = []
    seen = set()

    for root_dir in search_dirs:
        for r, _, files in os.walk(root_dir):
            parent_lower = os.path.basename(r).lower()

            for fn in sorted(files):
                fn_lower = fn.lower()
                if not (fn_lower.endswith(".xlsx") and "dis" in fn_lower):
                    continue

                # Explicit -51C gate
                if "-51c" not in fn_lower and "-51c" not in parent_lower:
                    continue

                p = os.path.join(r, fn)
                norm = os.path.normcase(os.path.normpath(p))
                if norm in seen:
                    continue

                seen.add(norm)
                files_out.append(p)

    return files_out


def slice_first_discharge(df_dis: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only discharge #1.

    Strategy:
      A) If a cycle column exists, take the minimum cycle present in discharge rows.
      B) Otherwise, detect the first reset in Discharge Capacity (Ah).
    """
    # A) Prefer explicit cycle columns when present
    for cyc_col in ["Cycle Index", "Cycle", "Cycle_Index", "CycleIndex"]:
        if cyc_col in df_dis.columns:
            non_na = df_dis[cyc_col].dropna()
            if not non_na.empty:
                first_cycle = non_na.min()
                return df_dis[df_dis[cyc_col] == first_cycle].copy()

    # B) Reset detection in discharge capacity
    cap = df_dis["Discharge Capacity (Ah)"].to_numpy()
    if len(cap) < 3:
        return df_dis

    max_cap = float(cap.max())
    reset_thresh = max(0.005 * max_cap, 5e-5)  # 0.5% of max or 5e-5 Ah

    reset_idx = None
    start_cap = float(cap[0])

    for i in range(1, len(cap)):
        if (cap[i - 1] - cap[i]) > reset_thresh and cap[i] <= (start_cap + 1e-6):
            reset_idx = i
            break

    if reset_idx is None:
        return df_dis

    return df_dis.iloc[:reset_idx].copy()


def load_discharge1_curve(path: str):
    """
    Load discharge-only rows (Current < 0), then keep discharge #1.
    Returns (Q_mAh_g, V).
    """
    sheet = get_channel_sheet_name(path)
    df = pd.read_excel(path, sheet_name=sheet)

    required = ["Voltage (V)", "Current (A)", "Discharge Capacity (Ah)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")

    df_dis = df[df["Current (A)"] < 0].copy()
    if df_dis.empty:
        raise ValueError(f"{path}: no discharge rows (Current (A) < 0).")

    df_dis = df_dis.dropna(subset=["Voltage (V)", "Discharge Capacity (Ah)"])
    df_dis = slice_first_discharge(df_dis)

    Q = df_dis["Discharge Capacity (Ah)"].to_numpy() * CONV_AH_TO_MAHG
    V = df_dis["Voltage (V)"].to_numpy()

    # Sort by increasing capacity for a clean curve
    order = Q.argsort()
    return Q[order], V[order]


# =========================
# MAIN
# =========================

def main():
    lookup = load_lookup_table(lookup_table_path)

    all_files = find_51C_discharge_files(base_dir, old_directory=old_directory)
    if not all_files:
        print("No -51C discharge .xlsx files found.")
        return

    for cell_code in CELL_CODES:
        cell_code_u = cell_code.upper()

        cell_files = [p for p in all_files if get_cell_code_from_filename(p) == cell_code_u]
        if not cell_files:
            print(f"No -51C discharge files found for {cell_code_u}")
            continue

        electrolyte = get_electrolyte_label(cell_code_u, lookup)

        fig, ax = plt.subplots(figsize=(7, 6))
        plotted = 0

        for p in sorted(cell_files):
            try:
                Q, V = load_discharge1_curve(p)
                ax.plot(Q, V, linewidth=1.8, label=os.path.basename(p))
                plotted += 1
            except Exception as e:
                print(f"Skipping {p}: {e}")

        if plotted == 0:
            print(f"{cell_code_u}: files found but none could be plotted.")
            plt.close(fig)
            continue

        # Title electrolyte only (as requested)
        ax.set_title(f"{electrolyte}  −51°C  Discharge 1")
        ax.set_xlabel("Specific Capacity (mAh/g)")
        ax.set_ylabel("Voltage (V)")
        ax.set_ylim(0, 4.5)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="x-small", loc="best")

        # Save name includes cell code to avoid overwriting if multiple codes share electrolyte
        out_path = os.path.join(plots_dir, f"{cell_code_u}_-51C_discharge1.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
