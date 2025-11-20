# plot_raw_points.py
# Plot the extracted RAW pixel points for cyan and green only.
# Files expected in the same folder:
#   - cyan_raw_points_xy_correctedX.csv
#   - green_raw_points_xy_correctedX.csv

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Config (edit paths if your files live elsewhere) ---
CYAN_CSV  = Path(r"C:\Users\benja\Downloads\10_22_2025 Temp\Slide Trials\Data\Digitizer\DT14_Li\cyan_raw_points_xy_correctedX.csv")
GREEN_CSV = Path(r"C:\Users\benja\Downloads\10_22_2025 Temp\Slide Trials\Data\Digitizer\DT14_Li\green_raw_points_xy_correctedX.csv")
X_LIM = (-8, 193.3)
Y_LIM = (2.42, 4.30)
SAVE_PNG = False
PNG_NAME = "raw_points.png"
# --------------------------------------------------------

def load_raw(path: Path):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    xcol = cols.get("capacity_mah_g") or next((cols[k] for k in cols if "cap" in k or "mah" in k), None)
    ycol = cols.get("voltage_v")      or next((cols[k] for k in cols if "volt" in k or k == "v"), None)
    if xcol is None or ycol is None:
        raise ValueError(f"Couldn't find capacity/voltage columns in {path}")
    df = df[[xcol, ycol]].rename(columns={xcol: "capacity_mAh_g", ycol: "voltage_V"})
    return df.dropna()

def main():
    cyan  = load_raw(CYAN_CSV)
    green = load_raw(GREEN_CSV)

    plt.figure(figsize=(8,5))
    # raw points only
    plt.plot(cyan["capacity_mAh_g"],  cyan["voltage_V"],  ".", ms=1, alpha=0.5, label="cyan raw")
    plt.plot(green["capacity_mAh_g"], green["voltage_V"], ".", ms=1, alpha=0.5, label="green raw")

    plt.xlabel("Capacity (mAh/g)")
    plt.ylabel("Voltage (V)")
    plt.xlim(*X_LIM); plt.ylim(*Y_LIM)
    plt.title("Voltage vs. Capacity — RAW pixel points")
    plt.legend(loc="best")
    plt.tight_layout()

    if SAVE_PNG:
        #plt.savefig(PNG_NAME, dpi=300)
        print(f"Saved {PNG_NAME}")
    plt.show()

if __name__ == "__main__":
    main()
