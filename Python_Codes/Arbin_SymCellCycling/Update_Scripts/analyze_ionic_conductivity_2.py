import os
import re
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
thickness_cm = .002
area_cm2 = 1.96
R_gas = 8.314
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

def estimate_Rb(df, n_hf=25, verbose=True):
    """
    Robustly estimate bulk electrolyte resistance R_b from an EC-Lab PEIS spectrum.
    Expects DataFrame with columns: ["Frequency (Hz)", "Zre (Ohm)", "-Zim (Ohm)"].

    Method:
      1) Sort by frequency descending (HF first).
      2) Build Im = -(-Zim).
      3) Linear fit on top-N HF points: Re = a*Im + b -> R_b = b at Im=0.
      4) One-pass outlier removal by residuals, refit.
      5) Sanity checks; fallback to min(Re) if needed.

    Returns:
      float R_b in Ohms.
    """
    import numpy as np
    import pandas as pd

    # Basic cleanup
    req_cols = ["Frequency (Hz)", "Zre (Ohm)", "-Zim (Ohm)"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"estimate_Rb: missing column '{c}'")

    d = df[req_cols].copy()
    # Force numeric
    for c in req_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()

    # Sort HF -> LF
    d = d.sort_values("Frequency (Hz)", ascending=False).reset_index(drop=True)

    # Build Im with correct sign
    d["Im (Ohm)"] = -d["-Zim (Ohm)"]
    d.rename(columns={"Zre (Ohm)": "Re (Ohm)"}, inplace=True)

    # Guard: need enough points
    if len(d) < 8:
        if verbose:
            print("estimate_Rb: too few points; using min(Re).")
        return float(d["Re (Ohm)"].min())

    # Choose HF window size sensibly
    n = int(np.clip(n_hf, 10, min(60, len(d)//2)))
    hf = d.head(n).copy()

    # If the HF Im spread is tiny, linear extrapolation is unstable
    if np.nanstd(hf["Im (Ohm)"]) < 1e-5:
        if verbose:
            print("estimate_Rb: tiny Im spread at HF; using min(Re).")
        return float(d["Re (Ohm)"].min())

    # First pass fit: Re = a*Im + b
    Im = hf["Im (Ohm)"].to_numpy()
    Re = hf["Re (Ohm)"].to_numpy()
    a1, b1 = np.polyfit(Im, Re, 1)

    # Outlier removal by residuals, keep best 75%
    resid = np.abs(Re - (a1*Im + b1))
    keep = resid <= np.percentile(resid, 75)
    Im2, Re2 = Im[keep], Re[keep]

    # Require enough points after trimming
    if len(Im2) >= 6 and np.nanstd(Im2) >= 1e-6:
        a2, b2 = np.polyfit(Im2, Re2, 1)
        Rb_fit = float(b2)
    else:
        Rb_fit = float(b1)

    # Fallback candidate: min(Re) over entire spectrum
    Rb_min = float(d["Re (Ohm)"].min())

    # Pick a reasonable value
    candidates = [x for x in [Rb_fit, Rb_min] if np.isfinite(x) and 0 < x < 1e6]
    if not candidates:
        if verbose:
            print("estimate_Rb: no valid candidates; returning NaN.")
        return float("nan")

    # Prefer the fit if it isn't wildly off the min(Re)
    # If the fit is >2x away from min(Re), trust min(Re).
    chosen = Rb_min

    if verbose:
        print(f"estimate_Rb: Rb_fit={Rb_fit:.6g} Ω, Rb_min={Rb_min:.6g} Ω -> chosen={chosen:.6g} Ω")

    return float(chosen)


def collect_eis_files_by_temperature(root_dir):
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
                continue

            Rb = estimate_Rb(df)
            T_K = temp_C + 273.15
            conductivity = thickness_cm / (Rb * area_cm2) * 1000  # Convert to mS/cm

            filename = mpt_file.name
            _, cell_code = extract_metadata(filename)
            if cell_code is None or cell_code in ["GG01", "GF01"]:
                continue

            parsed_data.append((cell_code, temp_C, T_K, conductivity, filename))
        except Exception:
            continue

    df_results = pd.DataFrame(parsed_data, columns=["Cell Code", "T_C", "T_K", "Conductivity_mS_cm", "Filename"])
    df_results["1000/T"] = 1000 / df_results["T_K"]
    df_results = df_results.sort_values(["Cell Code", "1000/T"])

    # Save Excel sheet
    excel_path = os.path.join(figures_dir, "ionic_conductivity_results.xlsx")
    df_results.to_excel(excel_path, index=False)

    # Group by cell prefix (first two letters)
    df_results["Group"] = df_results["Cell Code"].str[:2]
    groups = df_results.groupby("Group")

    colors = plt.cm.tab10.colors
    for idx, (group, group_df) in enumerate(groups):
        # Pivot so each cell is a column, index is 1000/T
        pivot = group_df.pivot_table(index="1000/T", columns="Cell Code", values="Conductivity_mS_cm")
        mean_cond = pivot.mean(axis=1)
        std_cond = pivot.std(axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(mean_cond.index, mean_cond, color=colors[idx % len(colors)], label=f"{group} mean")
        plt.fill_between(mean_cond.index, mean_cond - std_cond, mean_cond + std_cond,
                         color=colors[idx % len(colors)], alpha=0.2, label=f"{group} ±1 std")

        plt.xlabel("1000 / T (K⁻¹)")
        plt.ylabel("Ionic Conductivity (mS/cm)")
        plt.title(f"Arrhenius Plot: {group} (mean ± std)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, f"arrhenius_plot_{group}_mean_std.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()


def main_2(root_dir):
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
                continue

            Rb = estimate_Rb(df)
            T_K = temp_C + 273.15
            conductivity = thickness_cm / (Rb * area_cm2) * 1000  # mS/cm

            filename = mpt_file.name
            _, cell_code = extract_metadata(filename)
            if cell_code is None or cell_code in ["GG01", "GF01"]:
                continue

            parsed_data.append((cell_code, temp_C, T_K, conductivity, filename))
        except Exception:
            continue

    df_results = pd.DataFrame(parsed_data, columns=["Cell Code", "T_C", "T_K", "Conductivity_mS_cm", "Filename"])
    df_results["1000/T"] = 1000 / df_results["T_K"]
    df_results = df_results.sort_values(["Cell Code", "1000/T"])
    df_results["Group"] = df_results["Cell Code"].str[:2]
    groups = df_results.groupby("Group")

    colors = plt.cm.tab10.colors
    plt.figure(figsize=(10, 7))
    for idx, (group, group_df) in enumerate(groups):
        pivot = group_df.pivot_table(index="1000/T", columns="Cell Code", values="Conductivity_mS_cm")
        mean_cond = pivot.mean(axis=1)
        std_cond = pivot.std(axis=1)

        plt.plot(mean_cond.index, mean_cond, color=colors[idx % len(colors)], label=f"{group} mean")
        plt.fill_between(mean_cond.index, mean_cond - std_cond, mean_cond + std_cond,
                         color=colors[idx % len(colors)], alpha=0.2)

    plt.xlabel("1000 / T (K⁻¹)")
    plt.ylabel("Ionic Conductivity (mS/cm)")
    plt.title("Arrhenius Plot: All Groups (mean ± std)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig_path = os.path.join(figures_dir, "arrhenius_plot_all_groups_mean_std.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()


def main_2_mean_only_logS(root_dir):
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
                continue
            Rb = estimate_Rb(df)
            T_K = temp_C + 273.15
            conductivity = thickness_cm / (Rb * area_cm2) * 1000  # mS/cm

            filename = mpt_file.name
            _, cell_code = extract_metadata(filename)
            if cell_code is None or cell_code in ["GG01", "GF01"]:
                continue

            parsed_data.append((cell_code, temp_C, T_K, conductivity, filename))
        except Exception:
            continue

    df_results = pd.DataFrame(parsed_data, columns=["Cell Code", "T_C", "T_K", "Conductivity_mS_cm", "Filename"])
    df_results["1000/T"] = 1000 / df_results["T_K"]
    df_results = df_results.sort_values(["Cell Code", "1000/T"])
    df_results["Group"] = df_results["Cell Code"].str[:2]
    groups = df_results.groupby("Group")

    colors = plt.cm.tab10.colors
    plt.figure(figsize=(10, 7))
    for idx, (group, group_df) in enumerate(groups):
        pivot = group_df.pivot_table(index="1000/T", columns="Cell Code", values="Conductivity_mS_cm")
        mean_cond_S_cm = pivot.mean(axis=1) / 1000  # Convert mS/cm to S/cm
        plt.plot(mean_cond_S_cm.index, np.log10(mean_cond_S_cm), color=colors[idx % len(colors)], label=f"{group} mean")

    plt.xlabel("1000 / T (K⁻¹)")
    plt.ylabel("log$_{10}$(Conductivity [S/cm])")
    plt.title("Arrhenius Plot: All Groups (mean only, log scale)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig_path = os.path.join(figures_dir, "arrhenius_plot_all_groups_mean_only_logS.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

def main_2_mean_std_logS(root_dir):
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
                continue
            Rb = estimate_Rb(df)
            T_K = temp_C + 273.15
            conductivity = thickness_cm / (Rb * area_cm2) * 1000  # mS/cm

            filename = mpt_file.name
            _, cell_code = extract_metadata(filename)
            if cell_code is None or cell_code in ["GG01", "GF01"]:
                continue

            parsed_data.append((cell_code, temp_C, T_K, conductivity, filename))
        except Exception:
            continue

    df_results = pd.DataFrame(parsed_data, columns=["Cell Code", "T_C", "T_K", "Conductivity_mS_cm", "Filename"])
    df_results["1000/T"] = 1000 / df_results["T_K"]
    df_results = df_results.sort_values(["Cell Code", "1000/T"])
    df_results["Group"] = df_results["Cell Code"].str[:2]
    groups = df_results.groupby("Group")

    colors = plt.cm.tab10.colors
    plt.figure(figsize=(10, 7))
    for idx, (group, group_df) in enumerate(groups):
        pivot = group_df.pivot_table(index="1000/T", columns="Cell Code", values="Conductivity_mS_cm")
        mean_cond_S_cm = pivot.mean(axis=1) / 1000  # Convert mS/cm to S/cm
        std_cond_S_cm = pivot.std(axis=1) / 1000
        log_mean = np.log10(mean_cond_S_cm)
        log_std = std_cond_S_cm / (mean_cond_S_cm * np.log(10))  # error propagation for log10

        plt.plot(mean_cond_S_cm.index, log_mean, color=colors[idx % len(colors)], label=f"{group} mean")
        plt.fill_between(mean_cond_S_cm.index, log_mean - log_std, log_mean + log_std,
                         color=colors[idx % len(colors)], alpha=0.2)

    plt.xlabel("1000 / T (K⁻¹)")
    plt.ylabel("log$_{10}$(Conductivity [S/cm])")
    plt.title("Arrhenius Plot: All Groups (mean ± std, log scale)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig_path = os.path.join(figures_dir, "arrhenius_plot_all_groups_mean_std_logS.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    root_dir = r"C:\Users\benja\Downloads\DQ_DV Work\Lab Arbin_DQ_DV_2025_07_15\07\Ionic Conductivity\2025_07_27"
    main_2_mean_std_logS(root_dir)
    #main_2(root_dir)