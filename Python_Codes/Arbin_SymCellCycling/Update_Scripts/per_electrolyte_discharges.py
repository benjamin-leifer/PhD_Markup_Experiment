# Per-electrolyte discharge plots in Scratch_t7 style with 090325 color scheme
import os
import re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import itertools
import matplotlib.pyplot as plt

# -------- USER: set these before running locally --------
lookup_table_path = r"C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx"
search_directory   = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\0915--51C Data\Arbin -51C"
snowflake_image_path = r'C:\Users\benja\Downloads\Temp\Data_Work_4_19\Snowflake.png'  # optional image path

out_dir = Path(search_directory) / "per_electrolyte_discharges_out"
out_dir.mkdir(parents=True, exist_ok=True)

# -------- 090325 color convention --------
ELECTROLYTE_COLOR = {
    # Original 090325 mapping
    "DTFV1411": "#0072B2",  # blue
    "DTFV1422": "#E69F00",  # orange
    "DTFV1452": "#009E73",  # green
    "DTFV1425": "#D55E00",  # vermillion
    "DTV1410":  "#CC79A7",  # reddish purple
    "DTV142":   "#56B4E9",  # sky blue
    "DTF1410":  "#F0E442",  # yellow
    "DTF142":   "#000000",  # black
    "MF91":     "#FF0000",  # red

    # New electrolytes you showed in plots
    "DTF14-10": "#9467BD",  # purple
    "DT14":     "#8C564B",  # brown
    "DTF14-1":  "#BCBD22",  # olive
    "DTF14-5":  "#17BECF",  # teal

    "DTFV1412": "#1F77B4",  # another blue
    "DTFV1421": "#FF7F0E",  # orange
    "DTFV14102":"#2CA02C",  # green

    "MFV912":   "#d62728",  # dark red
    "MTF11":    "#9467bd",  # purple
    "MTF14":    "#8c564b",  # brown
    "TF91":     "#e377c2",  # pink
    "DTV14":    "#7f7f7f",  # gray
}


# alpha → electrolyte mapping (extend as needed)
electrolyte_lookup = {
    "AS": "DT14",
    "AT": "DTFV1425",
    "AU": "DT14",
    "FT": "DTFV1422",
    "FU": "MF91",
    "GB": "DTFV1411",
    "GN": "DTFV1452",
    "GO": "DTFV1425",
    "GW": "DTFV1411",
    "GX": "DTFV1422",
    "GV": "DTV1410",
    "GU": "DTV142",
    "GT": "DTF1410",
    "GS": "DTF142",
    "GY": "DT14",
    "GJ": "DTFV1452",
    "GK": "DTFV1425",
    "FR": "DTFV1422",
    "FS": "MF91",
    "GC": "DTFV1422",
    "GD": "DTFV1452",
}

# Set up a cycle of fallback colors (Okabe–Ito + extras)
_fallback_colors = itertools.cycle([
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"
])

def get_electrolyte_color(elec: str):
    """Return color for electrolyte, with fallback if not predefined."""
    if elec not in ELECTROLYTE_COLOR:
        ELECTROLYTE_COLOR[elec] = next(_fallback_colors)
    return ELECTROLYTE_COLOR[elec]
def extract_cell_identifier(filename: str):
    m = re.search(r"([A-Z]{2}\d{2})", filename)
    return m.group(1) if m else None


def process_all_cycles_for_voltage_vs_capacity(file_path: str, dataset_key: str, normalized: bool = False):
    capacities = {
        "LFP": 2.0075 / 1000 / 100,
        "NMC": 3.212 / 1000 / 100,
        "Gr": 3.8544 / 1000 / 100,
        "NEI-16mm": 4.02 / 1000 / 100,
    }
    weights_g = {
        "LFP": 7.09 / 1000 * 1.606 / 1000,
        "NMC": 12.45 / 1000 * 1.606 / 1000,
        "Gr": 6.61 / 1000 * 2.01 / 1000,
        "NEI-16mm": 12.45 / 1000 * 2.01 / 1000,
    }

    if normalized:
        if "LFP" in dataset_key:
            norm_factor = capacities["LFP"]
        elif "NEI-16mm" in dataset_key:
            norm_factor = capacities["NEI-16mm"]
        elif "NMC" in dataset_key:
            norm_factor = capacities["NMC"]
        elif "Gr" in dataset_key:
            norm_factor = capacities["Gr"]
        else:
            raise ValueError("Dataset key does not match known capacities")
    else:
        if "LFP" in dataset_key:
            norm_factor = weights_g["LFP"]
        elif "NEI-16mm" in dataset_key:
            norm_factor = capacities["NEI-16mm"]
        elif "NMC" in dataset_key:
            norm_factor = weights_g["NMC"]
        elif "Gr" in dataset_key:
            norm_factor = weights_g["Gr"]
        else:
            raise ValueError("Dataset key does not match known capacities")

    data = pd.ExcelFile(file_path)
    sheets = [s for s in data.sheet_names if s.lower().startswith("channel")]
    if not sheets:
        raise ValueError(f"No 'Channel*' sheet in {file_path}")
    df = data.parse(sheets[0])
    df = df[df["Current (A)"] != 0]

    cycles_data = []
    for cyc, g in df.groupby("Cycle Index"):
        if len(g) > 4:
            charge = g[g["Current (A)"] > 0].iloc[2:-2]
            discharge = g[g["Current (A)"] < 0].iloc[2:-2]
        else:
            charge = g[g["Current (A)"] > 0]
            discharge = g[g["Current (A)"] < 0]
        cycles_data.append((cyc, charge, discharge))
    return cycles_data, norm_factor

def format_key(key: str) -> str:
    if "(NEI-16mm)" in key:
        key = key.replace("(NEI-16mm)", "")
    key = re.sub(r"\s*\([A-Z]{2}\d{2}\)\s*$", "", key)
    return key.strip()

def temperature_from_filename(filename: str) -> str:
    m = re.search(r"-(\d{2})C", filename)
    return f"-{m.group(1)}°C" if m else "RT"

def alpha_from_cell_identifier(cell_identifier: str) -> str:
    return cell_identifier[:2].upper()

def generate_file_paths_keys(directory: str, lookup_table_path: str):
    """
    Walk `directory`, find Excel files, and build (path, key, cell_identifier, electrolyte)
    using metadata from the lookup spreadsheet (no hard-coded dict).
    - key format matches Scratch_t7: "Anode|Cathode - {Electrolyte} Elyte (XX##)"
    - cell_identifier is the XX## code parsed from the filename.
    """
    import os
    import re
    import pandas as pd

    def extract_cell_identifier(filename: str):
        m = re.search(r"([A-Z]{2}\d{2})", filename)
        return m.group(1) if m else None

    file_paths_keys = []
    lookup_df = pd.read_excel(lookup_table_path)

    for root, _, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith(".xlsx"):
                continue
            full_path = os.path.join(root, file)
            cell_identifier = extract_cell_identifier(file)
            if cell_identifier is None:
                continue

            cell_code = cell_identifier[:2]  # e.g., "GK"
            row = lookup_df[lookup_df["Cell Code"] == cell_code]

            if row.empty:
                anode = cathode = electrolyte = ""
            else:
                row = row.iloc[0]
                # use .get for robustness; fall back to "" if column missing
                anode = row.get("Anode", "")
                cathode = row.get("Cathode", "")
                electrolyte = row.get("Electrolyte", "")

                # Replace NaN with empty strings
                anode = "" if pd.isna(anode) else str(anode)
                cathode = "" if pd.isna(cathode) else str(cathode)
                electrolyte = "" if pd.isna(electrolyte) else str(electrolyte)

            key = f"{anode}|{cathode} - {electrolyte} Elyte ({cell_identifier})"
            file_paths_keys.append((full_path, key, cell_identifier, electrolyte))

    return file_paths_keys

def plot_per_electrolyte_discharge(
    file_tuples,
    normalized: bool = False,
    voltage_cutoff: float = 1.8,
    linewidths = {"DT14": 3, "DTF14": 1, "DTFV": 2, "MF91": 2},
):
    """
    Create one Scratch_t7-style discharge plot per electrolyte, with colors from 090325…,
    grouping by the `Electrolyte` read from the spreadsheet.

    Expects file_tuples to be a list of (full_path, key, cell_identifier, electrolyte).
    """
    import os
    import re
    from pathlib import Path
    import matplotlib.pyplot as plt
    # Marker cycle to differentiate cells
    marker_styles = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H", "+", "x", "|", "_"]

    # Helper functions kept local for portability
    def format_key(key: str) -> str:
        if "(NEI-16mm)" in key:
            key = key.replace("(NEI-16mm)", "")
        key = re.sub(r"\s*\([A-Z]{2}\d{2}\)\s*$", "", key)
        return key.strip()

    def temperature_from_filename(filename: str) -> str:
        m = re.search(r"-(\d{2})C", filename)
        return f"-{m.group(1)}°C" if m else "RT"

    # Use spreadsheet electrolyte to group (fallback "Unknown")
    grouped = {}
    for full_path, key, cell_ident, electrolyte in file_tuples:
        elec = electrolyte if electrolyte else "Unknown"
        grouped.setdefault(elec, []).append((full_path, key, cell_ident))

    for elec, entries in grouped.items():
        color = get_electrolyte_color(elec)
        fig, ax = plt.subplots(figsize=(6, 4))
        # Assign markers per cell within this electrolyte
        unique_cells = sorted({cell_ident for _, _, cell_ident in entries})
        cell_marker_map = {cid: marker_styles[i % len(marker_styles)] for i, cid in enumerate(unique_cells)}

        for (file_path, key, cell_ident) in entries:
            try:
                cycles_data, norm_factor = process_all_cycles_for_voltage_vs_capacity(file_path, key, normalized)
            except Exception as e:
                print(f"[WARN] Skipping {file_path}: {e}")
                continue

            # Family linewidth logic (same as before)
            lw = 2
            fam = None
            if "DT14" in key and "DTF" not in key and "DTFV" not in key:
                fam = "DT14"
            if "DTF14" in key:
                fam = "DTF14"
            if "DTFV" in key:
                fam = "DTFV"
            if "MF91" in key:
                fam = "MF91"
            if fam in linewidths:
                lw = linewidths[fam]

            filename = os.path.basename(file_path)
            temp_txt = temperature_from_filename(filename)

            # Legend includes the cell code: [… (XX##)]
            label_text = f"{format_key(key)} ({temp_txt}) [{cell_ident}]"

            for cycle, _chg, dch in cycles_data:
                if dch.empty:
                    continue
                d = dch[dch["Voltage (V)"] > voltage_cutoff].copy()
                if d.empty:
                    continue

                x = d["Discharge Capacity (Ah)"] / norm_factor
                # Keep the Scratch_t7 capacity scaling quirk
                if norm_factor > 4e-5 and any(s in key for s in ["DT14", "DTF14", "DTFV", "MF91"]):
                    x = x * 1.6

                ax.plot(
                    x, d["Voltage (V)"],
                    linestyle="-", color=color,
                    marker=cell_marker_map[cell_ident], markersize=3,
                    label=label_text
                )

        ax.set_xlabel("Capacity (mAh/g)")
        ax.set_ylabel("Voltage (V)")
        ax.set_ylim(0, 4.5)
        #ax.set_xlim(-4, 160)
        ax.grid(False)
        ax.tick_params(which="both", axis="both", direction="in",
                       bottom=True, left=True, labelbottom=True, labelleft=True)

        # De-duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        uniq = []
        seen = set()
        for h, lab in zip(handles, labels):
            if lab not in seen:
                uniq.append((h, lab))
                seen.add(lab)
        if uniq:
            ax.legend([h for h, _ in uniq], [lab for _, lab in uniq],
                      fontsize="xx-small", ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.05))

        fig.suptitle(f"Discharge Curves — {elec}", y=0.98, fontsize=12)
        fig.tight_layout()

        out_png = out_dir / f"discharge_{elec.replace('/','-')}.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    file_tuples = generate_file_paths_keys(search_directory, lookup_table_path)
    if not file_tuples:
        print("No Excel files found under search_directory.")
        return
    plot_per_electrolyte_discharge(file_tuples, normalized=False, voltage_cutoff=2.5)

if __name__ == "__main__":
    main()
