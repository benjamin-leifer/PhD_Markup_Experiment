import os
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ================== USER SETTINGS ==================
# Root folder to walk; default is current working directory
ROOT_DIR = os.getcwd()
# Example:
ROOT_DIR = r"C:\Users\benja\Downloads\Devcom\2025_11_11"

# If you want specific capacity (mAh/g), put your active mass (in grams) here.
# If you leave this as None, plots will use absolute discharge capacity (Ah).
ACTIVE_MASS_G = None# 12.45 * 2.01/1e6   # e.g., 0.005  -> mAh/g; None -> Ah
# ===================================================

REQUIRED_COLS = {"Voltage (V)", "Current (A)", "Discharge Capacity (Ah)"}
PLOTS_DIR = Path(ROOT_DIR) / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def find_data_sheet(xl: pd.ExcelFile) -> str | None:
    for name in xl.sheet_names:
        if name.lower().startswith("global"):
            continue
        try:
            df = xl.parse(name, nrows=5)
        except Exception:
            continue
        cols = set(map(str, df.columns))
        if REQUIRED_COLS.issubset(cols):
            return name
    return None

def extract_cell_code(fname: str) -> str:
    m = re.search(r"([A-Z]{2}\d{2})", Path(fname).stem)
    return m.group(1) if m else Path(fname).stem

def get_discharge_curve(df: pd.DataFrame) -> pd.DataFrame | None:
    if not REQUIRED_COLS.issubset(df.columns):
        return None
    neg = df[df["Current (A)"] < 0].copy()
    if neg.empty:
        neg = df[df["Discharge Capacity (Ah)"].fillna(0) > 0].copy()
        if neg.empty:
            return None
    if "Cycle Index" in neg.columns:
        neg = neg[neg["Cycle Index"] == neg["Cycle Index"].max()]
    neg = neg.sort_values("Discharge Capacity (Ah)").drop_duplicates(
        subset=["Discharge Capacity (Ah)"]
    )
    return neg[["Discharge Capacity (Ah)", "Voltage (V)"]].dropna()

def plot_single_file(file_path: str):
    fpath = Path(file_path)
    try:
        xl = pd.ExcelFile(fpath)
    except Exception as e:
        print(f"[error] open {fpath}: {e}")
        return
    sheet = find_data_sheet(xl)
    if sheet is None:
        print(f"[skip] no valid sheet in {fpath.name}")
        return
    df = xl.parse(sheet)
    curve = get_discharge_curve(df)
    if curve is None or curve.empty:
        print(f"[skip] no discharge data in {fpath.name}")
        return

    if ACTIVE_MASS_G is not None and ACTIVE_MASS_G > 0:
        x = (curve["Discharge Capacity (Ah)"] * 1000.0) / ACTIVE_MASS_G
        xlabel = "Discharge Capacity (mAh/g)"
        suffix = "_mAh-per-g"
    else:
        x = curve["Discharge Capacity (Ah)"]
        xlabel = "Discharge Capacity (Ah)"
        suffix = "_Ah"
    y = curve["Voltage (V)"]

    cell_code = extract_cell_code(fpath.name)
    title = fpath.stem
    out_png = PLOTS_DIR / f"{fpath.stem}{suffix}.png"
    #out_pdf = PLOTS_DIR / f"{fpath.stem}{suffix}.pdf"

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, linewidth=2, label=cell_code)
    plt.xlabel(xlabel)
    plt.ylabel("Voltage (V)")
    plt.title(title)
    plt.legend(title="Cell Code", loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    #plt.savefig(out_pdf)
    plt.close()
    print(f"[saved] {out_png}")

def main():
    print(f"Walking {ROOT_DIR}")
    for dirpath, _, filenames in os.walk(ROOT_DIR):
        for fname in filenames:
            if fname.lower().endswith(".xlsx") and not fname.startswith("~$"):
                plot_single_file(os.path.join(dirpath, fname))
    print(f"All plots saved in: {PLOTS_DIR}")

if __name__ == "__main__":
    main()