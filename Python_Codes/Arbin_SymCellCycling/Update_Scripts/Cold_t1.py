import os
import pandas as pd
import matplotlib.pyplot as plt

# =======================
# User settings
# =======================

# Root directory that contains all your -51C data subfolders
# (e.g. r"C:\Users\benja\...\Low Temp Li Ion\2025\-51C_discharges")
base_dir = r"C:\Users\benja\Downloads\DRT EIS Stair 1\-51C Discharges"   # <<< CHANGE THIS

# Identify -51C discharge files by name
TEMP_TAG = "-51C"
REQUIRE_DIS_IN_NAME = True   # only files with "dis" in the name (Discharge, C10Dis, etc.)

# Normalization: 4 mAh -> 160.6 mAh/g
REF_CAP_MAH = 4.0
REF_SPEC_MAH_G = 160.6
# Conversion factor from Ah -> mAh/g
# q_spec (mAh/g) = Q_Ah * CONV_AH_TO_MAHG
CONV_AH_TO_MAHG = 1000.0 * REF_SPEC_MAH_G / REF_CAP_MAH  # 40150.0


# =======================
# Helpers
# =======================

def find_discharge_files(root_dir, temp_tag=TEMP_TAG, require_discharge_name=True):
    """
    Walk through root_dir and return a sorted list of .xlsx files that:
      - contain temp_tag (e.g. '-51C') in the filename
      - optionally contain 'dis' in the filename (to pick out discharges)
    """
    matches = []
    temp_tag_lower = temp_tag.lower()
    for r, dirs, files in os.walk(root_dir):
        for fn in files:
            if not fn.lower().endswith(".xlsx"):
                continue
            name_lower = fn.lower()
            if temp_tag_lower in name_lower:
                if (not require_discharge_name) or ("dis" in name_lower):
                    matches.append(os.path.join(r, fn))
    return sorted(matches)


def get_cell_code(path):
    """
    Extract cell code from filenames like:
      BL-LL-HU01_-51C_Discharge_t1_Channel_37_Wb_1.xlsx -> HU01
      BL-LL-DN06_-51C_C10Dis_2025_02_15_214754.xlsx    -> DN06
    """
    base = os.path.basename(path)
    root = base.split("_")[0]  # 'BL-LL-HU01'
    return root.split("-")[-1]  # 'HU01'


def get_channel_sheet_name(path):
    """
    For these Arbin exports:
      sheet 0 = 'Global_Info'
      sheet 1 = 'ChannelXX_1'
    We use the second sheet by default.
    """
    xls = pd.ExcelFile(path)
    if len(xls.sheet_names) < 2:
        raise ValueError(f"{path} has no channel sheet")
    return xls.sheet_names[1]


def load_discharge_curve(path):
    """
    Read the Excel file, extract discharge data, and return:
      specific capacity (mAh/g), voltage (V), label
    """
    cell_code = get_cell_code(path)
    sheet_name = get_channel_sheet_name(path)

    df = pd.read_excel(path, sheet_name=sheet_name)

    required_cols = ["Voltage (V)", "Current (A)", "Discharge Capacity (Ah)"]
    if not all(col in df.columns for col in required_cols):
        raise KeyError(f"{path} missing one of {required_cols}")

    # Keep only discharge portion: negative current
    dis_mask = df["Current (A)"] < 0
    df_dis = df.loc[dis_mask].copy()

    if df_dis.empty:
        raise ValueError(f"{path}: no rows with Current (A) < 0 (no discharge segment?)")

    # Make sure Voltage + capacity are clean and sorted for monotonic x
    df_dis = df_dis.dropna(subset=["Discharge Capacity (Ah)", "Voltage (V)"])
    # Specific capacity in mAh/g, with 4 mAh -> 160.6 mAh/g
    df_dis["Spec Discharge Capacity (mAh/g)"] = (
        df_dis["Discharge Capacity (Ah)"] * CONV_AH_TO_MAHG
    )

    df_dis = df_dis.sort_values("Spec Discharge Capacity (mAh/g)")

    x_spec = df_dis["Spec Discharge Capacity (mAh/g)"].values
    y_volt = df_dis["Voltage (V)"].values

    return x_spec, y_volt, cell_code


# =======================
# Main
# =======================

def main():
    files = find_discharge_files(base_dir)

    if not files:
        print(f"No -51C discharge files found under: {base_dir}")
        return

    print("Found -51C discharge files:")
    for f in files:
        print("  ", f)

    plt.figure(figsize=(8, 6))

    for path in files:
        try:
            x_spec, y_volt, label = load_discharge_curve(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        plt.plot(x_spec, y_volt, label=label, linewidth=3)

    plt.xlabel("Discharge Specific Capacity (mAh/g)")
    plt.ylabel("Voltage (V)")
    plt.title("-51°C Discharge Curves (normalized to 4 mAh = 160.6 mAh/g)")
    #plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    plt.fontsize = 16
    plt.legend(title="Cell", loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
