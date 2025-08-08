import os
import re
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
thickness_cm = 20e-4  # 20 µm
area_cm2 = 1.96       # cm²
R_gas = 8.314         # J/mol·K

# Biologic .mpt column headers (partial)
column_names = [
    "Frequency (Hz)", "Zmod (Ohm)", "Zre (Ohm)", "-Zim (Ohm)", "Phase(Z) (deg)", "Time (s)",
    "Ewe (V)", "I (A)", "|Ewe| (V)", "|I| (A)", "Control/V", "Ns", "|Z| (Ohm)", "Phase(Z) (rad)",
    "Time index", "Corr (A)", "Zreal_corr (Ohm)", "Zimag_corr (Ohm)", "Zfit_corr (Ohm)",
    "Loop_1", "Loop_2", "Loop_3", "Loop_4", "Loop_5",
    "Misc1", "Misc2", "Misc3", "Misc4", "Misc5", "Misc6", "Misc7", "Misc8", "Misc9", "Misc10",
    "Misc11", "Misc12", "Misc13", "Misc14", "Misc15", "Misc16", "Misc17", "Misc18"
]

def extract_metadata(filename):
    temp_match = re.search(r'_(\-?\d+)C_', filename)
    temp_C = int(temp_match.group(1)) if temp_match else None
    cell_match = re.search(r'^[^-]+-[^-]+-([A-Z]{2}\d{2})_', filename)
    cell_code = cell_match.group(1) if cell_match else None
    return temp_C, cell_code

def estimate_Rb(df):
    print("Estimating Rb from the spectrum...")
    print(df.sort_values("Frequency (Hz)", ascending=False)["Zre (Ohm)"].iloc[0])
    return df.sort_values("Frequency (Hz)", ascending=False)["Zre (Ohm)"].iloc[0]

def collect_eis_files_by_temperature(root_dir):
    """
    Traverse all subdirectories under root_dir and collect .mpt files
    located inside folders with names like -20C, 0C, 25C, etc.
    """
    eis_entries = []
    for folder in pathlib.Path(root_dir).rglob("*C"):
        temp_str = folder.name.replace("C", "")
        try:
            temp_C = int(temp_str)
        except ValueError:
            continue
        for mpt_file in folder.glob("*.mpt"):
            eis_entries.append((mpt_file, temp_C))
    return eis_entries

def main(root_dir):
    figures_dir = os.path.join(root_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    eis_files = collect_eis_files_by_temperature(root_dir)
    parsed_data = []

    for mpt_file, temp_C in eis_files:
        if "PEIS" not in mpt_file.name:
            continue

        try:
            df = pd.read_csv(mpt_file, skiprows=65, delimiter='\t', encoding='ISO-8859-1', names=column_names)
            df = df[["Frequency (Hz)", "Zre (Ohm)", "-Zim (Ohm)"]].dropna()

            if df["Zre (Ohm)"].min() < 0 or df.empty:
                print(f"Skipping {mpt_file.name} due to invalid spectrum")
                continue

            Rb = estimate_Rb(df)
            T_K = temp_C + 273.15
            conductivity = thickness_cm / (Rb * area_cm2)

            filename = mpt_file.name
            _, cell_code = extract_metadata(filename)
            if cell_code is None:
                continue

            parsed_data.append((cell_code, temp_C, T_K, conductivity, filename))
        except Exception as e:
            print(f"Failed to process {mpt_file.name}: {e}")

    df_results = pd.DataFrame(parsed_data, columns=["Cell Code", "T_C", "T_K", "Conductivity", "Filename"])
    df_results["1000/T"] = 1000 / df_results["T_K"]
    df_results = df_results.sort_values(["Cell Code", "1000/T"])

    # Save Excel sheet
    excel_path = os.path.join(figures_dir, "ionic_conductivity_results.xlsx")
    df_results.to_excel(excel_path, index=False)
    print(f"Saved Excel sheet to {excel_path}")

    # Plot and save Arrhenius fits for each cell individually
    unique_cells = df_results["Cell Code"].unique()
    colors = plt.cm.tab10.colors

    for idx, cell in enumerate(unique_cells):
        cell_df = df_results[df_results["Cell Code"] == cell]
        valid_df = cell_df[cell_df["Conductivity"] > 0]

        if len(valid_df) < 2:
            continue

        x = valid_df["1000/T"]
        y = valid_df["Conductivity"]
        log_y = np.log(y)

        plt.figure(figsize=(8, 6))
        try:
            popt, _ = curve_fit(lambda x, a, b: a * x + b, x, log_y)
            Ea = -popt[0] * R_gas
            plt.plot(x, y, 'o', color=colors[idx % len(colors)], label=f'{cell} data')
            plt.plot(x, np.exp(popt[0]*x + popt[1]), '--', color=colors[idx % len(colors)],
                     label=f'{cell} fit (Ea={Ea/1000:.2f} kJ/mol)')
        except Exception as e:
            print(f"Fit failed for cell {cell}: {e}")

        plt.xlabel("1000 / T (K⁻¹)")
        plt.ylabel("Ionic Conductivity (S/cm)")
        plt.title(f"Arrhenius Plot: {cell}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, f"arrhenius_plot_{cell}.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved plot to {fig_path}")
if __name__ == "__main__":
    root_dir = r"C:\Users\benja\Downloads\DQ_DV Work\Lab Arbin_DQ_DV_2025_07_15\07\Ionic Conductivity\2025_07_27"
    main(root_dir)
