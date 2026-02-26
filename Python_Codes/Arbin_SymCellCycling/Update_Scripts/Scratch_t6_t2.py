from __future__ import annotations
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog
import matplotlib
import numpy as np
from scipy.signal import savgol_filter      # pip install scipy if needed
from electrolyte_style import clean_elyte_str, style_for_electrolyte, pretty_label



from Python_Codes.Arbin_SymCellCycling.Update_Scripts.Scratch_t6 import DT14_control

# Provide the path to your lookup table Excel file.
lookup_table_path = r'C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx'
# lookup_table_path = filedialog.askopenfilename(title="Select Lookup Table")

search_directory = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\03\Cycle Life Best Survivors'

search_directory = r'C:\Users\benja\Downloads\Temp\Data_Work_4_19\Cycle Life Best Survivors'
search_directory = r'C:\Users\benja\Downloads\Temp\Data_Work_4_19\Cycle Life Best Survivors\Proposal Figs'
#search_directory = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\02'
#search_directory = r'C:\Users\benja\Downloads\Temp\Data_Work_4_19\Cycle Life Best Survivors\Form Experiment'
#search_directory = r'C:\Users\benja\Downloads\Temp\Data_Work_4_19\Cycle Life Best Survivors\Form Experiment\07152025'
#search_directory = r'C:\Users\benja\Downloads\Temp\C_10 Cycling\2025\08\C_10 Sept Update\Brookhaven Cells'
# ==========================
# 1. Set the working directory
# ==========================
os.chdir(search_directory)


# ==========================
# 2. Helper functions
# ==========================
_LOOKUP_CACHE = {}
_CHANNEL_SHEET_CACHE = {}


def load_lookup_df(path):
    abs_path = os.path.abspath(path)
    df = _LOOKUP_CACHE.get(abs_path)
    if df is None:
        df = pd.read_excel(abs_path)
        _LOOKUP_CACHE[abs_path] = df
    return df


def load_channel_sheet(file_path):
    abs_path = os.path.abspath(file_path)
    df = _CHANNEL_SHEET_CACHE.get(abs_path)
    if df is None:
        data = pd.ExcelFile(abs_path)
        sheet_name = next((s for s in data.sheet_names if s.startswith('Channel')), None)
        if sheet_name is None:
            raise ValueError(f"No sheet starting with 'Channel' found in {file_path}")
        df = data.parse(sheet_name)
        _CHANNEL_SHEET_CACHE[abs_path] = df
    return df


def get_norm_factor(dataset_key, normalized, capacities, weights):
    if normalized:
        if 'LFP' in dataset_key:
            return capacities['LFP']
        if 'NMC' in dataset_key:
            return capacities['NMC']
        if '16mm' in dataset_key and '16mm' in capacities:
            return capacities['16mm']
        if 'Gr' in dataset_key:
            return capacities['Gr']
    else:
        if 'LFP' in dataset_key:
            return weights['LFP']
        if 'NMC' in dataset_key:
            return weights['NMC']
        if '16mm' in dataset_key and '16mm' in weights:
            return weights['16mm']
        if 'Gr' in dataset_key:
            return weights['Gr']
    raise ValueError("Dataset key does not match known capacities")

def sanitize_filename(name):
    """Sanitize a string to create a valid filename by replacing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name


def reset_cycle_capacity(df, capacity_column, threshold=1e-6):
    """
    Reset the capacity values in a DataFrame for a given column.
    If the first value is greater than a threshold (i.e. not close to zero),
    subtract that baseline from all entries.
    """
    if df.empty:
        return df
    baseline = df[capacity_column].iloc[0]
    if abs(baseline) > threshold:
        df = df.copy()  # avoid modifying the original DataFrame
        df[capacity_column] = df[capacity_column] - baseline
    return df


# ==========================
# 3. Generate file paths and keys for GITT experiments
# ==========================
def generate_gitt_file_paths_keys(directory, lookup_table_path):
    """
    Walk through the directory (and subdirectories) to find Excel files related to GITT experiments.
    A file is considered a GITT file if its filename contains "GITT" (case insensitive).
    For each file, extract the cell identifier and lookup additional details in the provided lookup table.

    Returns:
      A list of tuples: (full_path, key, cell_code)
    """
    file_paths_keys = []
    lookup_df = load_lookup_df(lookup_table_path)

    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check for Excel files that contain "GITT" in the filename (case insensitive)
            if file.endswith('.xlsx') and 'GITT' in file.upper():
                full_path = os.path.join(root, file)
                cell_identifier = extract_cell_identifier(file)
                if cell_identifier is None:
                    print(f"Could not extract cell identifier from file: {file}")
                    continue
                cell_code = cell_identifier[:2]
                lookup_row = lookup_df[lookup_df['Cell Code'] == cell_code]
                if lookup_row.empty:
                    print(f"Cell code {cell_code} not found in lookup table for file: {file}")
                    continue
                row = lookup_row.iloc[0]
                anode = row['Anode'] if not pd.isna(row['Anode']) else ''
                cathode = row['Cathode'] if not pd.isna(row['Cathode']) else ''
                electrolyte = row['Electrolyte'] if not pd.isna(row['Electrolyte']) else ''
                key = f"{anode}|{cathode} - {electrolyte} Elyte ({cell_identifier})"
                file_paths_keys.append((full_path, key, cell_code))
    return file_paths_keys


# ==========================
# 4. Process all cycles for Voltage vs. Capacity (with capacity reset)
# ==========================
def process_all_cycles_for_voltage_vs_capacity(file_path, dataset_key, normalized=False):
    """
    Loads cycling data from an Excel file, groups it by cycle,
    and for each cycle separates the charge and discharge data.
    For each cycle, transient portions are removed (first two and last two rows)
    and the capacity is reset to start at zero.
    The normalization factor is computed based on the dataset key.
    Returns:
      cycles_data: a list of tuples (cycle_index, charge_group, discharge_group)
      norm_factor: the normalization factor (same for all cycles)
    """
    capacities = {
        'LFP': 2.0075 / 1000 / 100,
        'NMC': 3.212 / 1000 / 100*4.02/3.212,
        'Gr': 3.8544 / 1000 / 100
    }
    weights_g = {
        'LFP': 7.09 / 1000 * 1.606 / 1000,
        'NMC': 12.45 / 1000 * 1.606 / 1000*4.02/3.212,
        'Gr': 6.61 / 1000 * 2.01 / 1000
    }

    norm_factor = get_norm_factor(dataset_key, normalized, capacities, weights_g)
    sheet_data = load_channel_sheet(file_path)
    # Filter out rows where Current equals zero
    filtered_data = sheet_data[sheet_data['Current (A)'] != 0]

    cycles_data = []
    # Group by 'Cycle Index' and process each cycle
    for cycle, group in filtered_data.groupby('Cycle Index'):
        if len(group) > 4:
            charge_group = group[group['Current (A)'] > 0].iloc[2:-2]
            discharge_group = group[group['Current (A)'] < 0].iloc[2:-2]
        else:
            charge_group = group[group['Current (A)'] > 0]
            discharge_group = group[group['Current (A)'] < 0]

        # Reset the capacity columns so that each cycle starts at zero
        if not charge_group.empty and 'Charge Capacity (Ah)' in charge_group.columns:
            charge_group = reset_cycle_capacity(charge_group, 'Charge Capacity (Ah)')
        if not discharge_group.empty and 'Discharge Capacity (Ah)' in discharge_group.columns:
            discharge_group = reset_cycle_capacity(discharge_group, 'Discharge Capacity (Ah)')

        cycles_data.append((cycle, charge_group, discharge_group))
    return cycles_data, norm_factor


# ==========================
# 5. Process all cycles for Voltage vs. Time
# ==========================
def process_all_cycles_for_voltage_vs_time(file_path, dataset_key, normalized=False):
    """
    Loads cycling data from an Excel file, groups it by cycle,
    and for each cycle separates the charge and discharge data.
    Unlike the capacity functions, no trimming is done so that all steps (including rest steps)
    are included. Uses the 'Test Time (s)' column.
    Returns:
      cycles_data: a list of tuples (cycle_index, charge_group, discharge_group)
      norm_factor: the normalization factor (same for all cycles)
    """
    capacities = {
        'LFP': 2.0075 / 1000 / 100,
        'NMC': 3.212 / 1000 / 100*4.02/3.212,
        'Gr': 3.8544 / 1000 / 100
    }
    weights_g = {
        'LFP': 7.09 / 1000 * 1.606 / 1000,
        'NMC': 12.45 / 1000 * 1.606 / 1000*4.02/3.212,
        'Gr': 6.61 / 1000 * 2.01 / 1000
    }

    norm_factor = get_norm_factor(dataset_key, normalized, capacities, weights_g)
    sheet_data = load_channel_sheet(file_path)

    # Do NOT filter out any rows, so that all steps are included.
    cycles_data = []
    for cycle, group in sheet_data.groupby('Cycle Index'):
        # For voltage vs time, we simply separate rows based on Current sign.
        charge_group = group[group['Current (A)'] > 0]
        discharge_group = group[group['Current (A)'] <= 0]
        cycles_data.append((cycle, charge_group, discharge_group))
    return cycles_data, norm_factor


# ==========================
# 6. Helper: Determine color based on dataset key
# ==========================
color_map = {
    'LFP': 'blue',
    'NMC': 'green',
    'Gr': 'red',
    'LTO': 'purple'
}


def get_color(key):
    """Return a color based on which chemistry substring is found in the key."""
    for chem, col in color_map.items():
        if chem in key:
            return col
    return 'black'


# ==========================
# 7. Helper: Extract cell identifier from filename
# ==========================
def extract_cell_identifier(filename):
    match = re.search(r'([A-Z]{2}\d{2})', filename)
    if match:
        return match.group(1)
    else:
        return None


def generate_file_paths_keys(directory, lookup_table_path):
    """
    Walk through the directory (and subdirectories) to find Excel files.
    For each file, extract the cell identifier and lookup additional details in the provided lookup table.
    Returns a list of tuples: (full_path, key, cell_code)
    """
    file_paths_keys = []
    lookup_df = load_lookup_df(lookup_table_path)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx') and ('Rate_Test' in file or 'RateTest' in file):
                full_path = os.path.join(root, file)
                cell_identifier = extract_cell_identifier(file)
                if cell_identifier is None:
                    print(f"Could not extract cell identifier from file: {file}")
                    continue
                cell_code = cell_identifier[:2]
                lookup_row = lookup_df[lookup_df['Cell Code'] == cell_code]
                if lookup_row.empty:
                    print(f"Cell code {cell_code} not found in lookup table for file: {file}")
                    continue
                row = lookup_row.iloc[0]
                anode = row['Anode'] if not pd.isna(row['Anode']) else ''
                cathode = row['Cathode'] if not pd.isna(row['Cathode']) else ''
                electrolyte = row['Electrolyte'] if not pd.isna(row['Electrolyte']) else ''
                key = f"{anode}|{cathode} - {electrolyte} Elyte ({cell_identifier})"
                file_paths_keys.append((full_path, key, cell_code))
    return file_paths_keys


# ==========================
# 8. Process cycle data for Capacity vs Cycle Number (with capacity reset)
# ==========================
def process_cycle_data(file_path, dataset_key, normalized=False):
    """
    Process the Excel file to extract cycle-wise capacity data.
    Resets the capacity data for each cycle so that it starts at zero.
    Returns:
      cycle_numbers, charge_capacities, discharge_capacities, coulombic_efficiency
    """
    capacities = {
        'LFP': 2.0075 / 1000 / 100,
        'NMC': 3.212 / 1000 / 100,
        '16mm': 3.212 / 1000 / 100/4.02*3.212,
        'Gr': 3.8544 / 1000 / 100
    }
    weights_g = {
        'LFP': 7.09 / 1000 * 1.606 / 1000,
        'NMC': 12.45 / 1000 * 1.606 / 1000,
        '16mm': 12.45 / 1000 * 1.606 / 1000/4.02*3.212,
        'Gr': 6.61 / 1000 * 2.01 / 1000
    }

    norm_factor = get_norm_factor(dataset_key, normalized, capacities, weights_g)
    sheet_data = load_channel_sheet(file_path)
    # Filter out rows where Current equals zero
    filtered_data = sheet_data[sheet_data['Current (A)'] != 0]

    cycle_numbers = []
    charge_capacities = []
    discharge_capacities = []
    coulombic_efficiency = []

    # Group by 'Cycle Index' and process each cycle
    for cycle, group in filtered_data.groupby('Cycle Index'):
        charge_data = group[group['Current (A)'] > 0]
        discharge_data = group[group['Current (A)'] < 0]

        # Reset capacities so that they start at zero for each cycle
        if not charge_data.empty and 'Charge Capacity (Ah)' in charge_data.columns:
            charge_data = reset_cycle_capacity(charge_data, 'Charge Capacity (Ah)')
        if not discharge_data.empty and 'Discharge Capacity (Ah)' in discharge_data.columns:
            discharge_data = reset_cycle_capacity(discharge_data, 'Discharge Capacity (Ah)')

        if not charge_data.empty and not discharge_data.empty:
            charge_cap = charge_data['Charge Capacity (Ah)'].max()
            discharge_cap = discharge_data['Discharge Capacity (Ah)'].max()
            if pd.notna(charge_cap) and pd.notna(discharge_cap) and charge_cap != 0:
                cycle_numbers.append(cycle)
                charge_capacities.append(charge_cap / norm_factor)
                discharge_capacities.append(discharge_cap / norm_factor)
                coulombic_efficiency.append((discharge_cap / charge_cap) * 100)
    return cycle_numbers, charge_capacities, discharge_capacities, coulombic_efficiency


# ==========================
# 9. Plot Voltage vs. Capacity for all cycles with distinct colors for each file
# ==========================
def plot_grouped_files(group, normalized=False):
    """
    For each file in the group, process all cycles and plot their charge and discharge curves
    (Voltage vs. Capacity). Each cycle is plotted with a label including its cycle number.
    All cycles for all files in the cell are shown in one figure.
    """
    plt.figure(figsize=(10, 6))
    # One color per file
    cmap = matplotlib.colormaps["tab10"].resampled(len(group))
    for file_idx, (file_path, key, cell_code) in enumerate(group):
        color = cmap(file_idx)
        try:
            cycles_data, norm_factor = process_all_cycles_for_voltage_vs_capacity(file_path, key, normalized)
        except Exception as e:
            print(f"Error processing {file_path} for voltage vs capacity: {e}")
            continue
        # Plot each cycle's charge and discharge data.
        for cycle, charge, discharge in cycles_data:
            # Label includes file key and cycle number
            if cycle in (1, 2, 5, 10, 25, 50):
                if not charge.empty:
                    plt.plot(charge['Charge Capacity (Ah)'] / norm_factor, charge['Voltage (V)'],
                             label=f'{key} Cycle {cycle} (Charge)', linestyle='-', color=color)
                if not discharge.empty:
                    plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor, discharge['Voltage (V)'],
                             label=f'{key} Cycle {cycle} (Discharge)', linestyle='--', color=color)
    plt.xlabel('Capacity (Ah)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Voltage vs Capacity for Cell Code {group[0][2]} (All Cycles)')
    plt.legend(fontsize='small', ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.show()


# ==========================
# 10. Plot Capacity vs Cycle Number (with vertical lines)
# (This function already processes all cycles.)
# ==========================
def plot_capacity_vs_cycle(group, normalized=False):
    """
    For each file in the group, extract cycle-wise capacity data and plot them
    versus cycle number. Vertical dashed lines mark specified cycle numbers.
    """
    plt.figure(figsize=(10, 6))
    colors = matplotlib.colormaps["tab10"].resampled(len(group)).colors
    for i, (file_path, key, cell_code) in enumerate(group):
        try:
            cycles, charge_caps, discharge_caps, _ = process_cycle_data(file_path, key, normalized)
        except Exception as e:
            print(f"Error processing {file_path} for cycle data: {e}")
            continue
        plt.plot(cycles, charge_caps, marker='o', linestyle='-', color=colors[i],
                 label=f'{key} (Charge)')
        plt.plot(cycles, discharge_caps, marker='x', linestyle='--', color=colors[i],
                 label=f'{key} (Discharge)')

    plt.xlabel('Cycle Number')
    if normalized:
        plt.ylabel('Capacity (%)')
        plt.ylim(0, 110)
    else:
        plt.ylabel('Capacity (mAh/g)')
        plt.ylim(0, 300)
    plt.title(f'Capacity vs Cycle Number for Cell Code {group[0][2]}')
    for cycle in [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5]:
        plt.axvline(x=cycle, color='black', linestyle='--')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    save_path = os.getcwd() + '/figures/' + f'{sanitize_filename(group[0][2])}_Capacity_vs_Cycle.png'
    plt.savefig(save_path, dpi=300, )
    plt.close()
    # plt.show()


# ==========================
# 11. Plot Voltage vs. Time for all cycles with markers and limit 5 files per cell
# ==========================
def plot_voltage_vs_time(group, normalized=False):
    """
    For up to 5 files in the group, process all cycles and plot their Voltage vs. Test Time.
    Charging curves use circle markers with a solid line.
    Discharging curves use square markers with a dashed line.
    Each cycle is plotted (with its own label showing the cycle number).
    """
    plt.figure(figsize=(10, 6))
    # Limit number of files to 5 per cell
    cmap = matplotlib.colormaps["tab10"].resampled(min(len(group), 5))
    for file_idx, (file_path, key, cell_code) in enumerate(group[:5]):
        color = cmap(file_idx)
        try:
            cycles_data, _ = process_all_cycles_for_voltage_vs_time(file_path, key, normalized)
        except Exception as e:
            print(f"Error processing {file_path} for voltage vs time: {e}")
            continue
        for cycle, charge, discharge in cycles_data:
            if not charge.empty:
                if 'Test Time (s)' in charge.columns:
                    plt.plot(charge['Test Time (s)'], charge['Voltage (V)'],
                             label=f'{key} Cycle {cycle} (Charge)', marker='o', linestyle='-', color=color)
                else:
                    print(f"'Test Time (s)' column not found in charge data for {file_path}")
            if not discharge.empty:
                if 'Test Time (s)' in discharge.columns:
                    plt.plot(discharge['Test Time (s)'], discharge['Voltage (V)'],
                             label=f'{key} Cycle {cycle} (Discharge)', marker='s', linestyle='--', color=color)
                else:
                    print(f"'Test Time (s)' column not found in discharge data for {file_path}")
    plt.xlabel('Test Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Voltage vs Time for Cell Code {group[0][2]} (All Cycles)')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_gitt_file(file_path, dataset_key, normalized=False):
    """
    Plots a GITT experiment file individually using Voltage vs. Test Time.
    This function assumes the file contains columns 'Test Time (s)' and 'Voltage (V'.
    It does not group by cycle; it simply plots the entire time series.
    """
    try:
        sheet_data = load_channel_sheet(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    if 'Test Time (s)' not in sheet_data.columns or 'Voltage (V)' not in sheet_data.columns:
        print(f"Columns 'Test Time (s)' or 'Voltage (V)' not found in {file_path}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(sheet_data['Test Time (s)'], sheet_data['Voltage (V)'],
             marker='o', linestyle='-', color='blue')
    plt.xlabel('Test Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title(f'GITT Experiment: {dataset_key}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_cells_on_same_plot(file_tuples, normalized=False, x_bounds=(0, 100), save_str='', color_scheme=None):
    """
    Compare multiple cells on one plot:
      - Left y-axis: Capacity (mAh/g) vs. Cycle Number
      - Right y-axis: Coulombic Efficiency (%) vs. Cycle Number
      - Marker shapes, fills, and colors reflect LPV/DT14, Gr/Li, LFP/NMC identities.
      - Optional color_scheme overrides default color logic.

    Args:
        file_tuples (list): List of (file_path, key, cell_code)
        normalized (bool): Normalize capacity if True
        x_bounds (tuple): x-axis limits
        save_str (str): Filename prefix for saving plot
        color_scheme (dict): Optional mapping of cell_code to color
    """
    if not file_tuples:
        raise ValueError("No file tuples provided for comparison.")

    # Flatten if a list of lists was passed (e.g., grouping sets).
    flattened = []
    for item in file_tuples:
        if isinstance(item, (list, tuple)) and len(item) == 3 and isinstance(item[0], str):
            flattened.append(item)
        elif isinstance(item, list):
            for sub in item:
                if isinstance(sub, (list, tuple)) and len(sub) == 3:
                    flattened.append(sub)
        else:
            raise ValueError(f"Invalid entry in file_tuples: {item!r}")

    if not flattened:
        raise ValueError("No valid (file_path, key, cell_code) tuples found for comparison.")

    def electrolyte_label(key):
        if ' - ' in key:
            tail = key.split(' - ', 1)[1]
            return tail.replace(' Elyte', '').split(' (', 1)[0].strip()
        return key

    fig, ax1 = plt.subplots(figsize=(10, 6))
    #ax2 = ax1.twinx()

    # Define C-Rate annotations for specific cycles
    c_rate_labels = {
        2: "Form",
        4: "C/10",
        7: "C/8",
        10: "C/4",
        13: "C/2",
        16: "1C",
        19: "2C"
    }

    # Track annotated cycles
    annotated_cycles = set()

    all_cycles = []
    all_ce = []

    for i, (file_path, key, cell_code) in enumerate(flattened):
        try:
            cycles, charge_caps, discharge_caps, ce = process_cycle_data(file_path, key, normalized)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
        all_cycles.extend(cycles)
        all_ce.extend(ce)

        # === Custom Plot Appearance ===
        is_gr = 'Gr' in key
        is_li = 'Li' in key
        print(f"Key: {key}, is_gr: {is_gr}, is_li: {is_li}")
        is_dt14 = 'DT14' in key
        is_lpv = 'LPV' in key
        is_lfp = 'LFP' in key
        is_nmc = 'NMC' in key

        # Marker shape: square for DT14, circle for LPV
        base_marker = 's' if is_dt14 else 'o'
        ce_marker = '*' if is_dt14 else 'D'

        # Fill style: solid for Gr, open for Li
        facecolor = 'none' if is_li else ''

        # Color logic: use custom scheme if provided
        if color_scheme and cell_code in color_scheme:
            color = color_scheme[cell_code]
        else:
            color = 'blue' if is_lfp else 'black'

            # Fill style: solid for Gr, open for Li
        facecolor = 'none' if is_li else color

        #color = 'blue' if is_lfp else 'black'
        # === Plotting ===
        # Replace 'auto' with a valid color or 'none'
        # ax1.scatter(cycles, charge_caps, marker=base_marker,
        #             facecolors=facecolor, edgecolors=color,
        #             label=f'{format_key(key)}')
        # # ax1.scatter(cycles, discharge_caps, marker=base_marker,
        # #             facecolors=color, edgecolors=color,
        # #             label=f'{key} (Discharge)', linestyle='--')
        #
        # ax2.scatter(cycles, ce, marker=ce_marker,
        #             facecolors=facecolor, edgecolors=color,
        #             label=f'{format_key(key)} (CE)')
        if color_scheme == None:
            ax1.scatter(cycles, charge_caps, marker=base_marker,
                        label=electrolyte_label(key))
            # ax1.scatter(cycles, discharge_caps, marker=base_marker,
            #             facecolors=color, edgecolors=color,
            #             label=f'{key} (Discharge)', linestyle='--')

            #ax2.scatter(cycles, ce, marker=ce_marker,
                        #label='_nolegend_')

        # Add C-Rate annotations (handled after data is plotted)

    # Formatting
    ax1.grid(False)
    #ax2.grid(False)

    for cycle in [4.5, 7.5, 10.5, 13.5, 16.5, 19.5]:
        ax1.axvline(x=cycle, color='black', linestyle='--')

    ax1.set_xlabel('Cycle Number', fontsize=14)
    ax1.set_xlim(x_bounds)
    ax1.set_ylim(0, 220)
    ax1.set_ylabel('Capacity (%)' if normalized else 'Capacity (mAh/g)', fontsize=14)
    #ax2.set_ylabel('Coulombic Efficiency (%)', fontsize=14)
    #ax2.set_ylim(0, 120)

    lines1, labels1 = ax1.get_legend_handles_labels()
    unique = {}
    for line, label in zip(lines1, labels1):
        unique.setdefault(label, line)
    ax1.legend(list(unique.values()), list(unique.keys()), loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fontsize=12, ncol=2)

    plt.title('Capacity and Coulombic Efficiency vs. Cycle Number', fontsize=16)
    # C-rate labels above cycling regimes.
    c_rate_segments = [
        (3, 4, "C/10"),
        (5, 7, "C/8"),
        (8, 10, "C/4"),
        (11, 13, "C/2"),
        (14, 16, "1C"),
        (17, 19, "2C"),
        (20, x_bounds[1], "C/2"),
    ]
    y_top = ax1.get_ylim()[1]
    for start, end, label in c_rate_segments:
        seg_start = max(start, x_bounds[0])
        seg_end = min(end, x_bounds[1])
        if seg_start <= seg_end:
            x = (seg_start + seg_end) / 2
            ax1.text(x, y_top * 0.99, label, fontsize=10, ha='center', va='top', color='black')

    # 2) Mirror those ticks on the top, and move all tick‐marks inward
    ax1.tick_params(
        which='both',  # apply to both major and minor ticks
        axis='both',  # apply on both axes
        direction='in',  # point ticks inward
        bottom=True, top=True,
        left=True, right=True
    )
    ax1.tick_params(axis='both', labelsize=12)
    #ax2.tick_params(which='both', axis='y', direction='in', right=True, labelright=True, labelsize=12)
    #plt.tick_params(which='both', axis='both', direction='in', bottom=True, left=True, top=True, right=True)
    plt.tight_layout()
    if save_str:
        plt.savefig(f'{save_str}_Capacity_and_Coulombic_Efficiency_vs_Cycle_v5.png', dpi=300)
    plt.show()


def get_tuples_by_cell_code(file_paths_keys, target_cell_code):
    """
    Search the list of (full_path, key, cell_code) tuples for a given cell_code.

    Args:
        file_paths_keys (list): A list of (full_path, key, cell_code) tuples.
        target_cell_code (str): The cell code to search for (e.g., 'DN').

    Returns:
        list: A list of all matching (full_path, key, cell_code) tuples.
              If no matches are found, returns an empty list.
    """
    matches = []
    for (full_path, key, cell_code) in file_paths_keys:
        if target_cell_code in key:
            matches.append((full_path, key, cell_code))
    return matches


def get_tuples_by_full_cell_code(file_paths_keys, pattern):
    """
    Filter a list of (full_path, key, cell_identifier) tuples by matching the full cell identifier
    against a given regex pattern. This version will "unwrap" an item if it’s a list containing a single tuple.

    Args:
        file_paths_keys (list): A list of tuples (or lists containing a tuple) in the form (full_path, key, cell_identifier)
        pattern (str): A regular expression pattern to match the full cell identifier.
                       (Uses re.fullmatch, so the entire cell_identifier must match.)

    Returns:
        list: A list of all tuples that match the pattern.
    """
    matches = []
    for index, item in enumerate(file_paths_keys):
        # Unwrap if item is a list with one element
        if isinstance(item, list):
            if len(item) == 1 and isinstance(item[0], tuple) and len(item[0]) == 3:
                tup = item[0]
            else:
                print(f"Skipping invalid tuple at index {index}: {item}")
                continue
        elif isinstance(item, tuple) and len(item) == 3:
            tup = item
        else:
            print(f"Skipping invalid tuple at index {index}: {item}")
            continue

        full_path, key, cell_identifier = tup
        if pattern in key:
            matches.append(tup)
    return matches


import numpy as np  # Ensure numpy is imported


def plot_selected_cycles_charge_and_discharge_vs_voltage(cell_tuple, normalized=False):
    """
    Plot both charge and discharge curves (Voltage vs. Capacity) for selected cycles
    for a given cell. The selected cycles are 1, 4, 7, 10, 13, 16, and 19 with custom labels:
      Cycle 1: Formation
      Cycle 4: C/10
      Cycle 7: C/8
      Cycle 10: C/4
      Cycle 13: C/2
      Cycle 16: 1C
      Cycle 19: 2C

    Charge curves are plotted as solid lines and discharge curves as dashed lines.
    Each cycle is assigned a different color.
    """
    # Unpack the cell tuple: (file_path, key, cell_code)
    file_path, key, cell_code = cell_tuple

    # Mapping from cycle number to custom label
    cycle_labels = {
        1: "Form",
        4: "C/10",
        7: "C/8",
        10: "C/4",
        13: "C/2",
        16: "1C",
        19: "2C"
    }

    # The selected cycle numbers
    selected_cycles = list(cycle_labels.keys())

    # Process the file to extract cycle data and the normalization factor.
    # cycles_data is a list of tuples: (cycle, charge_group, discharge_group)
    cycles_data, norm_factor = process_all_cycles_for_voltage_vs_capacity(file_path, key, normalized)

    # Create a colormap with as many colors as selected cycles (here 7)
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_cycles)))

    plt.figure(figsize=(10, 6))

    # Loop over the processed cycles and plot only the selected ones.
    for cycle, charge, discharge in cycles_data:
        if cycle in selected_cycles:
            # Determine the color index based on the order in selected_cycles
            idx = selected_cycles.index(cycle)
            color = colors[idx]

            # Plot the charge curve (solid line)
            if not charge.empty:
                plt.plot(
                    charge['Charge Capacity (Ah)'] / norm_factor,
                    charge['Voltage (V)'],
                    label=f"{cycle_labels[cycle]} Charge (Cycle {cycle})",
                    linestyle='-',
                    color=color
                )
            # Plot the discharge curve (dashed line)
            if not discharge.empty:
                plt.plot(
                    discharge['Discharge Capacity (Ah)'] / norm_factor,
                    discharge['Voltage (V)'],
                    label=f"{cycle_labels[cycle]} Discharge (Cycle {cycle})",
                    linestyle='--',
                    color=color
                )

    plt.xlabel('Capacity (Ah)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Charge and Discharge Curves for {key} (Selected Cycles)')
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_excel_summary(file_paths_keys, output_path, normalized=False):
    """
    Create an Excel document listing the cell code, specific capacity of cycles 4, 7, 10, 13, 16, and 19,
    and the average coulombic efficiency of all cycles past cycle 20.
    """
    summary_data = []

    for file_path, key, cell_code in file_paths_keys:
        try:
            cycles, charge_caps, discharge_caps, ce = process_cycle_data(file_path, key, normalized)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        # Extract specific capacities for the specified cycles
        specific_capacities = {}
        for cycle, charge_cap in zip(cycles, charge_caps):
            if cycle in [4, 7, 10, 13, 16, 19]:
                specific_capacities[cycle] = charge_cap

        # Calculate average coulombic efficiency for cycles past cycle 20
        ce_past_20 = [eff for cycle, eff in zip(cycles, ce) if cycle > 20]
        avg_ce_past_20 = sum(ce_past_20) / len(ce_past_20) if ce_past_20 else None

        summary_data.append({
            'Cell Code': cell_code,
            'Electrolyte': key.split(' - ')[-1],
            'Formation CE': ce[0] if len(ce) > 0 else None,
            'C/10': specific_capacities.get(4, None),
            'C/8': specific_capacities.get(7, None),
            'C/4': specific_capacities.get(10, None),
            'C/2': specific_capacities.get(13, None),
            '1C': specific_capacities.get(16, None),
            '2C': specific_capacities.get(19, None),

        })

    # Create a DataFrame and save to Excel
    df = pd.DataFrame(summary_data)
    df.to_excel(output_path, index=False)

import matplotlib.pyplot as plt
import matplotlib

def compare_cells_cycle_2(file_tuples, normalized=False):
    """
    Compare multiple cells on one plot for cycle 2:
      - Charge and discharge curves (Voltage vs. Capacity)
    Args:
        file_tuples (list): A list of (full_path, key, cell_code) tuples.
        normalized (bool): Whether to use normalization.
    """
    if not file_tuples:
        raise ValueError("No file tuples provided for comparison.")

    plt.figure(figsize=(10, 6))
    cmap = matplotlib.colormaps["tab10"].resampled(len(file_tuples))

    for i, (file_path, key, cell_code) in enumerate(file_tuples):
        try:
            cycles_data, norm_factor = process_all_cycles_for_voltage_vs_capacity(file_path, key, normalized)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        color = cmap(i)
        for cycle, charge, discharge in cycles_data:
            if cycle == 2:
                if not charge.empty:
                    plt.plot(charge['Charge Capacity (Ah)'] / norm_factor, charge['Voltage (V)'],
                             label=f'{key} Cycle 2 (Charge)', linestyle='-', color=color)
                if not discharge.empty:
                    plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor, discharge['Voltage (V)'],
                             label=f'{key} Cycle 2 (Discharge)', linestyle='--', color=color)

    plt.xlabel('Capacity (Ah)')
    plt.ylabel('Voltage (V)')
    plt.title('Charge and Discharge Curves for Cycle 2')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Example usage
def assign_tol_colors(cell_codes):
    """
    Assign Paul Tol's color palette (bright or nightfall) based on the number of unique cell codes.
    Returns a dict mapping cell_code → hex color.
    """
    tol_bright = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7"
    ]

    tol_nightfall = [
        "#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77",
        "#CC6677", "#882255", "#AA4499", "#661100", "#6699CC", "#888888"
    ]

    josh_colors = [
        '#000000', '#8A2BE2', '#1e90ff', '#32CD32', '#FFD700', '#DC143C'
    ]
    color_dict = {}

    color_dict = {}

    if len(cell_codes) <= len(tol_bright):
        palette = tol_bright
    else:
        palette = tol_nightfall

    if len(cell_codes) <= len(tol_bright):
        palette = josh_colors
    else:
        palette = josh_colors




    for i, code in enumerate(cell_codes):
        color_dict[str(code)] = palette[i % len(palette)]

    return color_dict

def format_key(key):
    """
    Remove (NEI-16mm) from the key and format it for display.
    Remove cell code from end of the key.
    """
    if '(NEI-16mm)' in key:
        key = key.replace('(NEI-16mm)', '')
    key = key[:-6]  # Remove the last 5 characters (cell code)

    return key.strip()
# ---------------------------------------------------------------------------
# DQ / DV TOOLKIT
# ---------------------------------------------------------------------------
def smooth_series(y, mode, window=11, poly=3):
    """Return a smoothed copy of *y*."""
    if mode is None:
        return y
    if mode == "rolling":
        return pd.Series(y).rolling(window, center=True, min_periods=1).mean().to_numpy()
    if mode == "savgol":
        if window % 2 == 0:                   # window must be odd for Savitzky-Golay
            window += 1
        return savgol_filter(y, window, poly, mode='interp')
    raise ValueError("smooth must be None, 'rolling', or 'savgol'")

def compute_dq_dv(df, q_col, v_col, smooth=None):
    """
    Compute dQ/dV given a DataFrame containing *v_col* and *q_col* columns.
    The derivative is returned on the **mid-point** voltage grid.
    """
    if df.empty:
        return np.array([]), np.array([])
    v = df[v_col].to_numpy()
    q = df[q_col].to_numpy()

    # Optional smoothing before differentiation
    q = smooth_series(q, smooth)

    # Central difference
    dq = np.diff(q)
    dv = np.diff(v)
    with np.errstate(divide='ignore', invalid='ignore'):
        dq_dv = dq / dv
    v_mid = (v[:-1] + v[1:]) / 2.0

    # Drop any NaNs or infs introduced by division by zero
    mask = np.isfinite(dq_dv)
    return v_mid[mask], dq_dv[mask]

def interpolate_to_grid(v, y, grid):
    """Interpolate *y(v)* onto a common *grid* (outside range → nan)."""
    return np.interp(grid, v, y, left=np.nan, right=np.nan)

# ---------------------------------------------------------------------------
# Single-cell visualisations
# ---------------------------------------------------------------------------
def _extract_cycle_segment(cycles_data, cycle_num, segment):
    for cy, chg, dchg in cycles_data:
        if cy == cycle_num:
            return chg if segment == 'charge' else dchg
    raise ValueError(f"Cycle {cycle_num} not found")

def plot_dq_dv_cycle(cell_tuple, cycle=1, segment='charge',
                     smooth=None, normalized=False):
    file_path, key, _ = cell_tuple
    cycles_data, norm = process_all_cycles_for_voltage_vs_capacity(file_path, key, normalized)
    df = _extract_cycle_segment(cycles_data, cycle, segment)
    v, y = compute_dq_dv(df, 'Charge Capacity (Ah)' if segment=='charge'
                              else 'Discharge Capacity (Ah)', 'Voltage (V)', smooth)
    plt.figure(figsize=(6,4))
    plt.plot(v, y, lw=1.2)
    plt.xlabel('Voltage (V)'); plt.ylabel('dQ/dV (Ah V⁻¹)')
    plt.title(f'{key} – {segment.capitalize()} Cycle {cycle}')
    plt.grid(True); plt.tight_layout(); plt.show()

def plot_dq_dv_all_cycles(cell_tuple, segment='charge',
                          smooth=None, normalized=False, alpha=0.6):
    file_path, key, _ = cell_tuple
    cycles_data, norm = process_all_cycles_for_voltage_vs_capacity(file_path, key, normalized)
    plt.figure(figsize=(6,4))
    for cy, chg, dchg in cycles_data:
        df = chg if segment=='charge' else dchg
        v, y = compute_dq_dv(df, 'Charge Capacity (Ah)' if segment=='charge'
                                  else 'Discharge Capacity (Ah)', 'Voltage (V)', smooth)
        plt.plot(v, y, label=f'Cy {cy}', alpha=alpha)
    plt.xlabel('Voltage (V)'); plt.ylabel('dQ/dV (Ah V⁻¹)')
    plt.title(f'{key} – {segment.capitalize()} (all cycles)')
    plt.legend(ncol=4, fontsize='x-small'); plt.grid(True); plt.tight_layout(); plt.show()

def plot_dq_dv_difference(cell_tuple, cycle_a, cycle_b, segment='charge',
                           smooth=None, normalized=False):
    file_path, key, _ = cell_tuple
    cycles_data, norm = process_all_cycles_for_voltage_vs_capacity(file_path, key, normalized)
    df_a = _extract_cycle_segment(cycles_data, cycle_a, segment)
    df_b = _extract_cycle_segment(cycles_data, cycle_b, segment)

    # Align on common voltage grid
    v_grid = np.linspace(max(df_a['Voltage (V)'].min(), df_b['Voltage (V)'].min()),
                         min(df_a['Voltage (V)'].max(), df_b['Voltage (V)'].max()),
                         1000)
    v_a, y_a = compute_dq_dv(df_a, 'Charge Capacity (Ah)' if segment=='charge'
                                   else 'Discharge Capacity (Ah)', 'Voltage (V)', smooth)
    v_b, y_b = compute_dq_dv(df_b, 'Charge Capacity (Ah)' if segment=='charge'
                                   else 'Discharge Capacity (Ah)', 'Voltage (V)', smooth)
    y_a_i = interpolate_to_grid(v_a, y_a, v_grid)
    y_b_i = interpolate_to_grid(v_b, y_b, v_grid)

    plt.figure(figsize=(6,4))
    plt.plot(v_grid, y_b_i - y_a_i, 'k-', lw=1.4)
    plt.xlabel('Voltage (V)'); plt.ylabel('Δ(dQ/dV) (Ah V⁻¹)')
    plt.title(f'{key}: Δ[{segment}, Cy {cycle_a} → {cycle_b}]')
    plt.grid(True); plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------------
# Multi-cell aggregation
# ---------------------------------------------------------------------------
def plot_mean_dq_dv(cell_tuples, segment='charge', smooth=None,
                    normalized=False, n_grid=1500):
    if not cell_tuples:
        raise ValueError("No cells supplied")

    # First pass → find global V-range
    v_min = 10; v_max = 0
    for fp, key, _ in cell_tuples:
        cds, _ = process_all_cycles_for_voltage_vs_capacity(fp, key, normalized)
        df = _extract_cycle_segment(cds, 1, segment)       # use cycle 1 for range
        v_min = min(v_min, df['Voltage (V)'].min())
        v_max = max(v_max, df['Voltage (V)'].max())

    grid = np.linspace(v_min, v_max, n_grid)
    stack = []

    for fp, key, _ in cell_tuples:
        cds, _ = process_all_cycles_for_voltage_vs_capacity(fp, key, normalized)
        # choose a representative cycle – here 2 (adjust as needed)
        df = _extract_cycle_segment(cds, 2, segment)
        v, y = compute_dq_dv(df, 'Charge Capacity (Ah)' if segment=='charge'
                                  else 'Discharge Capacity (Ah)', 'Voltage (V)', smooth)
        stack.append(interpolate_to_grid(v, y, grid))

    stack = np.vstack(stack)         # shape: (n_cells, n_grid)
    mean = np.nanmean(stack, axis=0)
    std  = np.nanstd(stack, axis=0)

    plt.figure(figsize=(6,4))
    plt.plot(grid, mean, 'k-', lw=2.0, label='Mean')
    plt.fill_between(grid, mean-std, mean+std, color='gray', alpha=0.3, label='±1 σ')
    plt.xlabel('Voltage (V)'); plt.ylabel('dQ/dV (Ah V⁻¹)')
    plt.title(f'Mean ± 1 σ – {len(cell_tuples)} cells ({segment})')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()



# ==========================
# 12. Main Execution
# ==========================
file_paths_keys = generate_file_paths_keys(os.getcwd(), lookup_table_path)
print('Starting')
#eate_excel_summary(file_paths_keys, 'output_summary_22.xlsx', normalized=False)

print("Generated file_paths_keys:")
for full_path, key, cell_code in file_paths_keys:
    print(f"File: {full_path}\nKey: {key}\nCell Code: {cell_code}\n")

gitt_file_paths_keys = generate_gitt_file_paths_keys(os.getcwd(), lookup_table_path)
for full_path, key, cell_code in gitt_file_paths_keys:
    print(f"File: {full_path}\nKey: {key}\nCell Code: {cell_code}\n")
    # Then call your GITT-specific plotting function, for example:
    #plot_gitt_file(full_path, key, normalized=False)

if not file_paths_keys:
    print("No valid Excel files were found. Please check your directory and filtering criteria.")
else:
    # Group the files by cell code (e.g., 'DN', 'DO', etc.)
    grouped_files = {}
    for full_path, key, cell_code in file_paths_keys:
        grouped_files.setdefault(cell_code, []).append((full_path, key, cell_code))

    # For each cell group, generate the three plots.
    for cell_code, group in grouped_files.items():
        print(f"Plotting Voltage vs Capacity for {len(group)} files for cell code {cell_code} (all cycles)...")
        #plot_grouped_files(group, normalized=False)

        print(f"Plotting Capacity vs Cycle Number for {len(group)} files for cell code {cell_code}...")
        #plot_capacity_vs_cycle(group, normalized=False)
        #plt.savefig(f'Capacity_vs_Cycle_{cell_code}.png')

        # print(
        #     f"Plotting Voltage vs Time for up to {min(len(group), 5)} files for cell code {cell_code} (all cycles)...")
        # plot_voltage_vs_time(group, normalized=False)
# filtered_tuples = get_tuples_by_cell_code(file_paths_keys, r'DQ01')
# print("Filtered tuples:", filtered_tuples)
# compare_cells_on_same_plot(filtered_tuples, normalized=False)

#DT14 Set
# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DN02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DO03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DP01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DQ01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DD03')[0],
# ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)
#
# #DTF14 Set
# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DR02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DS03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DT03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DU02')[0],
# ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)
#
# #MF91 Set
# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DZ03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EA02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EB03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EC01')[0],
# ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)
#
# #DTV14 Set

# files_to_compare = [
#      get_tuples_by_cell_code(file_paths_keys, r'DV01')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'DW02')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'DU02')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'EC01')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'DY01')[0],
#  ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)

# #Best of each set
# python
import sys
def plot_rate_curves_formatted(comparison_set,
                               normalized=False,
                               color_scheme=None,
                               figsize=(10, 6)):
    """
    Publication-style rate plot.

    Features
    --------
    - Uses comparison_set (list of lists of tuples)
    - Cycles: 3,6,9,12,15,18
    - Solid = charge, dashed = discharge
    - Rate labels written once per cycle
    - Labels aligned at top/bottom bands
    - Optional color_scheme (cell_code -> color)
    """

    # ---------- Flatten comparison_set ----------
    flattened = []
    for item in comparison_set:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)

    if not flattened:
        raise ValueError("comparison_set is empty")

    # ---------- Cycle → rate mapping ----------
    cycle_map = {
        3: "C/10",
        6: "C/8",
        9: "C/4",
        12: "C/2",
        15: "1C",
        18: "2C"
    }
    selected_cycles = list(cycle_map.keys())

    # ---------- Colors ----------
    if color_scheme:
        colors = [color_scheme.get(cell_code, 'black')
                  for _, _, cell_code in flattened]
    else:
        cmap = matplotlib.colormaps["tab10"].resampled(len(flattened))
        colors = [cmap(i) for i in range(len(flattened))]

    plt.figure(figsize=figsize)

    # Store x-positions for labels (use first cell as reference)
    label_positions = {}

    # ---------- Plot curves ----------
    for i, (file_path, key, cell_code) in enumerate(flattened):

        try:
            cycles_data, norm_factor = process_all_cycles_for_voltage_vs_capacity(
                file_path, key, normalized
            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        color = colors[i]

        for cycle, charge, discharge in cycles_data:
            if cycle not in selected_cycles:
                continue

            # --- Charge ---
            if not charge.empty:
                x = charge['Charge Capacity (Ah)'] / norm_factor
                y = charge['Voltage (V)']
                plt.plot(x, y, '-', color=color, lw=1.5)

                # Save median x for label placement (first cell only)
                if cycle not in label_positions:
                    label_positions[cycle] = np.nanmedian(x)

            # --- Discharge ---
            if not discharge.empty:
                x = discharge['Discharge Capacity (Ah)'] / norm_factor
                y = discharge['Voltage (V)']
                plt.plot(x, y, '--', color=color, lw=1.5)

    # ---------- Axis formatting ----------
    ax = plt.gca()
    ax.set_xlabel('Capacity (Ah)' if not normalized else 'Capacity (%)', fontsize=14)
    ax.set_ylabel('Voltage (V)', fontsize=14)

    ax.tick_params(which='both', direction='in',
                   top=True, right=True, labelsize=12)

    # Add headroom for labels
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)

    # ---------- Rate labels (clean, once only) ----------
    y_min, y_max = ax.get_ylim()

    y_top = y_max - 0.02 * (y_max - y_min)
    y_bot = y_min + 0.02 * (y_max - y_min)

    for cycle, label in cycle_map.items():
        if cycle in label_positions:
            x_pos = label_positions[cycle]

            # Top (charge)
            ax.text(x_pos, y_top, label,
                    ha='center', va='top',
                    fontsize=11, color='black')

            # Bottom (discharge)
            ax.text(x_pos, y_bot, label,
                    ha='center', va='bottom',
                    fontsize=11, color='black')

    plt.title('Rate Capability', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_rate_curves_from_comparison_set(comparison_set, normalized=False):
    """
    Plot charge/discharge curves for cycles [3,6,9,12,15,18]
    using a comparison_set structure (list of lists of tuples).

    C-rate labels are written directly on the curves:
        3  -> C/10
        6  -> C/8
        9  -> C/4
        12 -> C/2
        15 -> 1C
        18 -> 2C
    """

    # Flatten comparison_set if nested
    flattened = []
    for item in comparison_set:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)

    if not flattened:
        raise ValueError("comparison_set is empty")

    # Cycle → label mapping
    cycle_map = {
        3: "C/10",
        6: "C/8",
        9: "C/4",
        12: "C/2",
        15: "1C",
        18: "2C"
    }
    selected_cycles = list(cycle_map.keys())

    plt.figure(figsize=(10, 6))
    cmap = matplotlib.colormaps["tab10"].resampled(len(flattened))

    for i, (file_path, key, cell_code) in enumerate(flattened):
        try:
            cycles_data, norm_factor = process_all_cycles_for_voltage_vs_capacity(
                file_path, key, normalized
            )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        color = cmap(i)

        for cycle, charge, discharge in cycles_data:
            if cycle not in selected_cycles:
                continue

            label_text = cycle_map[cycle]

            # ---- Charge (solid) ----
            if not charge.empty:
                x = charge['Charge Capacity (Ah)'] / norm_factor
                y = charge['Voltage (V)']
                plt.plot(x, y, linestyle='-', color=color)

                # Place label at top of charge
                idx = np.argmax(y)
                plt.text(
                    x.iloc[idx],
                    y.iloc[idx],
                    label_text,
                    fontsize=10,
                    ha='center',
                    va='bottom',
                    color=color
                )

            # ---- Discharge (dashed) ----
            if not discharge.empty:
                x = discharge['Discharge Capacity (Ah)'] / norm_factor
                y = discharge['Voltage (V)']
                plt.plot(x, y, linestyle='--', color=color)

                # Place label at bottom of discharge
                idx = np.argmin(y)
                plt.text(
                    x.iloc[idx],
                    y.iloc[idx],
                    label_text,
                    fontsize=10,
                    ha='center',
                    va='top',
                    color=color
                )

    plt.xlabel('Capacity (Ah)' if not normalized else 'Capacity (%)', fontsize=14)
    plt.ylabel('Voltage (V)', fontsize=14)
    plt.title('Rate Capability Curves', fontsize=16)

    plt.grid(False)
    plt.tight_layout()
    plt.show()

def first_or_warn(matches, desc=None):
    """Return first item of matches or print a warning and return None."""
    if matches:
        return matches[0]
    msg = f"Warning: no matches found for {desc or 'pattern'}"
    print(msg)
    return None

# Example: replace lines that do get_tuples_by_cell_code(...)[0] with guarded lookups
files_to_compare = []
for code in (r'DN02', r'DR02', r'DT01', r'DU02', r'DV01', r'DW02', r'DX02',
             r'DZ02', r'EA02', r'EB03', r'EC01', r'EG02', r'EH02', r'EI03',
             r'EJ03', r'EL03', r'EM01', r'EN02', r'EO02'):
    t = first_or_warn(get_tuples_by_cell_code(file_paths_keys, code), desc=code)
    if t is not None:
        files_to_compare.append(t)

if not files_to_compare:
    print("No files selected for comparison — check `search_directory` and filename filters.")
    # Optionally exit gracefully
    # sys.exit(0)
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _flatten_comparison_set(comparison_set):
    flattened = []
    for item in comparison_set:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened

def _electrolyte_label(key: str) -> str:
    # Matches the helper you already used in compare_cells_on_same_plot
    if ' - ' in key:
        tail = key.split(' - ', 1)[1]
        return tail.replace(' Elyte', '').split(' (', 1)[0].strip()
    return key

def _ensure_dir(p: str | Path):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _save_fig_and_legend(fig, ax, out_png: Path, out_legend_png: Path,
                         legend_loc='upper center', legend_ncol=2,
                         legend_fontsize=12, legend_bbox=(0.5, -0.15)):
    """
    Save main figure and a separate legend-only figure.
    """
    # --- Main legend (for on-plot figure) ---
    handles, labels = ax.get_legend_handles_labels()

    # de-duplicate labels while preserving order
    unique = {}
    for h, lab in zip(handles, labels):
        if lab not in unique and lab not in ("_nolegend_", ""):
            unique[lab] = h
    handles_u = list(unique.values())
    labels_u = list(unique.keys())

    if handles_u:
        ax.legend(handles_u, labels_u,
                  loc=legend_loc,
                  bbox_to_anchor=legend_bbox,
                  fontsize=legend_fontsize,
                  ncol=legend_ncol,
                  frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- Legend-only figure ---
    if handles_u:
        fig_leg = plt.figure(figsize=(10, 2))
        fig_leg.legend(handles_u, labels_u,
                       loc='center',
                       ncol=legend_ncol,
                       frameon=False,
                       fontsize=legend_fontsize)
        fig_leg.tight_layout()
        fig_leg.savefig(out_legend_png, dpi=300, bbox_inches='tight')
        plt.close(fig_leg)

def _apply_axes_formatting(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)

    # "Josh style" ticks: inward, mirror top/right
    ax.tick_params(which='both',
                   direction='in',
                   bottom=True, top=True,
                   left=True, right=True,
                   labelsize=12)

    ax.grid(False)

# ------------------------------------------------------------
# 1) Representative (dis)charge curves (cycles 3,6,9,12,15,18)
# ------------------------------------------------------------
def plot_representative_rate_curves_export(comparison_set,
                                          out_dir,
                                          save_stem="Representative_Rate_Curves",
                                          normalized=False,
                                          color_scheme=None,
                                          figsize=(10, 6)):
    """
    Plots voltage vs capacity for cycles [3,6,9,12,15,18] for each cell in comparison_set.
    Writes C-rate labels ONCE per cycle at top (charge) and bottom (discharge).
    Exports main + legend-only.
    """
    flattened = _flatten_comparison_set(comparison_set)
    if not flattened:
        raise ValueError("comparison_set is empty")

    out_dir = _ensure_dir(out_dir)

    cycle_map = {3: "C/10", 6: "C/8", 9: "C/4", 12: "C/2", 15: "1C", 18: "2C"}
    selected_cycles = list(cycle_map.keys())

    # Colors
    if color_scheme:
        colors = [color_scheme.get(cell_code, 'black') for _, _, cell_code in flattened]
    else:
        cmap = matplotlib.colormaps["tab10"].resampled(len(flattened))
        colors = [cmap(i) for i in range(len(flattened))]

    fig, ax = plt.subplots(figsize=figsize)

    # Use first cell that has data to define label x positions
    label_positions = {}  # cycle -> x median (capacity)
    plotted_any = False

    for i, (file_path, key, cell_code) in enumerate(flattened):
        try:
            cycles_data, norm_factor = process_all_cycles_for_voltage_vs_capacity(file_path, key, normalized)
        except Exception as e:
            print(f"[Representative curves] Error processing {file_path}: {e}")
            continue

        color = colors[i]
        lab = _electrolyte_label(key)

        for cycle, charge, discharge in cycles_data:
            if cycle not in selected_cycles:
                continue

            # Charge
            if not charge.empty:
                x = charge['Charge Capacity (Ah)'] / norm_factor
                y = charge['Voltage (V)']
                ax.plot(x, y, '-', color=color, lw=1.6, label=lab)
                plotted_any = True

                if cycle not in label_positions:
                    label_positions[cycle] = float(np.nanmedian(x))

            # Discharge
            if not discharge.empty:
                x = discharge['Discharge Capacity (Ah)'] / norm_factor
                y = discharge['Voltage (V)']
                ax.plot(x, y, '--', color=color, lw=1.6)
                plotted_any = True

                if cycle not in label_positions:
                    label_positions[cycle] = float(np.nanmedian(x))

    if not plotted_any:
        raise RuntimeError("No curves were plotted — check that the files contain the requested cycles.")

    _apply_axes_formatting(
        ax,
        xlabel=('Capacity (Ah)' if not normalized else 'Capacity (%)'),
        ylabel='Voltage (V)',
        title='Representative (Dis)charge Curves (Rate Steps)'
    )

    # Add headroom for labels
    y_min, y_max = ax.get_ylim()
    y_rng = y_max - y_min
    ax.set_ylim(y_min - 0.05*y_rng, y_max + 0.08*y_rng)

    # Label band positions
    y_min, y_max = ax.get_ylim()
    y_top = y_max - 0.02*(y_max - y_min)
    y_bot = y_min + 0.02*(y_max - y_min)

    for cycle, rate_lab in cycle_map.items():
        if cycle in label_positions:
            x_pos = label_positions[cycle]
            ax.text(x_pos, y_top, rate_lab, ha='center', va='top', fontsize=11, color='black')
            ax.text(x_pos, y_bot, rate_lab, ha='center', va='bottom', fontsize=11, color='black')

    out_png = out_dir / f"{save_stem}.png"
    out_leg = out_dir / f"{save_stem}_LEGEND.png"
    _save_fig_and_legend(fig, ax, out_png, out_leg, legend_ncol=2, legend_bbox=(0.5, -0.15))


# ------------------------------------------------------------
# 2–4) Capacity vs Cycle exports (full / rate / life-cut)
# ------------------------------------------------------------
def plot_capacity_vs_cycle_exports(comparison_set,
                                  out_dir,
                                  save_prefix="Capacity_vs_Cycle",
                                  normalized=False,
                                  color_scheme=None,
                                  figsize=(10, 6),
                                  use_charge=False):
    """
    Exports:
      A) full cycling (all cycles)
      B) rate-only (1–19)
      C) cycle-life (20–end), where end is cut to the LOWEST max cycle in the set

    By default it plots CHARGE capacity vs cycle (to match your compare_cells_on_same_plot behavior).
    Set use_charge=False to plot discharge instead.
    """
    flattened = _flatten_comparison_set(comparison_set)
    if not flattened:
        raise ValueError("comparison_set is empty")

    out_dir = _ensure_dir(out_dir)

    # Determine "cycle life end" = min(max_cycle) across cells that have cycles >= 20
    maxes = []
    for file_path, key, cell_code in flattened:
        try:
            cycles, charge_caps, discharge_caps, ce = process_cycle_data(file_path, key, normalized)
        except Exception:
            continue
        cyc = np.array(cycles)
        if np.any(cyc >= 20):
            maxes.append(int(np.max(cyc)))
    life_end = min(maxes) if maxes else None
    if life_end is None:
        print("[Capacity plots] No cells had cycles >= 20; cycle-life plot will be skipped.")

    # Colors
    if color_scheme:
        colors = [color_scheme.get(cell_code, 'black') for _, _, cell_code in flattened]
    else:
        cmap = matplotlib.colormaps["tab10"].resampled(len(flattened))
        colors = [cmap(i) for i in range(len(flattened))]

    def _plot_one(xlim, title, save_stem):
        fig, ax = plt.subplots(figsize=figsize)
        plotted_any = False

        for i, (file_path, key, cell_code) in enumerate(flattened):
            try:
                cycles, charge_caps, discharge_caps, ce = process_cycle_data(file_path, key, normalized)
            except Exception as e:
                print(f"[{save_stem}] Error processing {file_path}: {e}")
                continue

            cyc = np.array(cycles)
            y = np.array(charge_caps if use_charge else discharge_caps)

            # Filter to xlim window
            if xlim is not None:
                m = (cyc >= xlim[0]) & (cyc <= xlim[1])
                cyc = cyc[m]
                y = y[m]

            if len(cyc) == 0:
                continue

            color = colors[i]
            lab = _electrolyte_label(key)

            # line + markers (clean but readable)
            #ax.plot(cyc, y, '-', lw=1.6, color=color, label=lab)
            ax.scatter(cyc, y, s=22, label=lab)
            plotted_any = True

        if not plotted_any:
            raise RuntimeError(f"No data plotted for {save_stem} — check bounds and inputs.")

        _apply_axes_formatting(
            ax,
            xlabel="Cycle Number",
            ylabel=("Capacity (%)" if normalized else "Capacity (mAh/g)"),
            title=title
        )

        if xlim is not None:
            ax.set_xlim(xlim)

        # Save
        out_png = out_dir / f"{save_stem}.png"
        out_leg = out_dir / f"{save_stem}_LEGEND.png"
        _save_fig_and_legend(fig, ax, out_png, out_leg, legend_ncol=2, legend_bbox=(0.5, -0.15))

    # A) Full
    _plot_one(
        xlim=None,
        title="Capacity vs Cycle (Full)",
        save_stem=f"{save_prefix}_FULL"
    )

    # B) Rate portion: 1–19
    _plot_one(
        xlim=(1, 19),
        title="Capacity vs Cycle (Rate Portion: 1–19)",
        save_stem=f"{save_prefix}_RATE_1to19"
    )

    # C) Cycle life: 20–life_end (cut to lowest max cycle)
    if life_end is not None and life_end >= 20:
        _plot_one(
            xlim=(20, life_end),
            title=f"Capacity vs Cycle (Cycle Life: 20–{life_end})",
            save_stem=f"{save_prefix}_LIFE_20to{life_end}"
        )


import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ============================
# 1) STYLE ENCODING CONSTANTS
# ============================

# Additive package -> base hue (projector-safe)
BASE_COLOR = {
    "NONE": "#2B2B2B",  # baseline (no additives)
    "F":    "#E68600",  # FEC-only
    "V":    "#2C7FB8",  # VC-only
    "FV":   "#6A3D9A",  # FEC+VC
    "WHITE":"#FFFFFF",
}

# System -> linestyle
SYSTEM_LS = {
    "DT":  "-",    # DME:THF
    "TPT": "--",   # TFSI - DPE:THF
}

# “More wt% = darker/more saturated” blending strength (t=0 -> white, t=1 -> base hue)
WT_TO_T = {1: 0.45, 2: 0.60, 5: 0.82, 10: 1.00}
ALLOWED_WT = {1, 2, 5, 10}

# VC amount -> marker encoding (so VC is obvious without extra colors)
# marker, fillmode, mew, ms
VC_MARKER = {
    0: (None, None, None, None),
    1: ("o", "filled", 0.0, 4.8),
    2: ("s", "open",   1.2, 5.2),
}

# ============================
# 2) COLOR UTILITIES
# ============================

def _hex_to_rgb01(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return r, g, b

def _rgb01_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = rgb
    r = max(0, min(255, int(round(r * 255))))
    g = max(0, min(255, int(round(g * 255))))
    b = max(0, min(255, int(round(b * 255))))
    return f"#{r:02X}{g:02X}{b:02X}"

def blend_hex(base_hex: str, mix_hex: str, t: float) -> str:
    """t=0 -> mix_hex, t=1 -> base_hex"""
    t = max(0.0, min(1.0, float(t)))
    br, bg, bb = _hex_to_rgb01(base_hex)
    mr, mg, mb = _hex_to_rgb01(mix_hex)
    r = mr * (1 - t) + br * t
    g = mg * (1 - t) + bg * t
    b = mb * (1 - t) + bb * t
    return _rgb01_to_hex((r, g, b))

def _t_from_wt(wt: int) -> float:
    if wt in WT_TO_T:
        return WT_TO_T[wt]
    # fallback for odd values (shouldn't happen given your constraint)
    if wt <= 1:
        return WT_TO_T[1]
    if wt >= 10:
        return WT_TO_T[10]
    # interpolate
    if 2 < wt < 5:
        return WT_TO_T[2] + (WT_TO_T[5] - WT_TO_T[2]) * (wt - 2) / 3.0
    if 5 < wt < 10:
        return WT_TO_T[5] + (WT_TO_T[10] - WT_TO_T[5]) * (wt - 5) / 5.0
    return WT_TO_T[2]

# ============================
# 3) PARSING YOUR ELECTROLYTE CODES
# ============================

@dataclass(frozen=True)
class ElectrolyteSpec:
    raw: str
    system: str        # "DT" or "TPT"
    additives: str     # "NONE", "F", "V", "FV"
    ratio: str         # e.g., "14"
    fec_wt: int        # 0 if absent
    vc_wt: int         # 0 if absent

def _clean_code(s: str) -> str:
    s = str(s).strip().upper()

    # Remove " - DATE..." suffixes (space-dash-space)
    s = re.sub(r"\s*-\s*\d.*$", "", s)

    # Remove hyphen-date patterns like "...-7_18_25" (underscore signals a date)
    s = re.sub(r"-(?=\d+_)\d.*$", "", s)

    # Remove spaces/underscores
    s = s.replace(" ", "").replace("_", "")

    # Convert dashed additive notation: DTF14-5 -> DTF145; DTF14-10 -> DTF1410
    s = s.replace("-", "")

    # Tolerate common typo: DFTV -> DTFV
    s = s.replace("DFTV", "DTFV")

    return s

def _split_fv_amounts(nums: str) -> Tuple[int, int]:
    """
    Parse FV amounts from concatenated digits.
    Examples:
      "11"  -> (1,1)
      "52"  -> (5,2)
      "102" -> (10,2)
    """
    if not nums:
        return 0, 0

    # Prefer FEC=10 if it matches
    if nums.startswith("10"):
        fec = 10
        rest = nums[2:]
        if rest.isdigit():
            vc = int(rest)
            return fec, vc

    # Otherwise FEC is first digit (1/2/5), VC is remaining
    fec = int(nums[0])
    rest = nums[1:]
    vc = int(rest) if rest else 0
    return fec, vc

def parse_electrolyte(code: str) -> ElectrolyteSpec:
    """
    Your convention:
      - DT... refers to DME:THF (ratio digits next, e.g. 14)
      - TPT... refers to TFSI-DPE-THF (T = salt convention), then PT solvents
      - Additives are appended after base letters: F, V, or FV
      - After ratio (two digits), numbers encode additive wt%:
          DTF145   -> FEC 5%
          DTFV1452 -> FEC 5%, VC 2%
      - Additive wt% are whole numbers: 1,2,5,10.
    """
    raw = str(code)
    s = _clean_code(raw)

    # Identify system
    if s.startswith("TPT"):
        system = "TPT"
        rest = s[3:]
    elif s.startswith("DT"):
        system = "DT"
        rest = s[2:]
    else:
        raise ValueError(f"Not a DT/TPT electrolyte code after cleaning: '{raw}' -> '{s}'")

    # Additives token
    additives = "NONE"
    if rest.startswith("FV"):
        additives = "FV"
        rest = rest[2:]
    elif rest.startswith("F"):
        additives = "F"
        rest = rest[1:]
    elif rest.startswith("V"):
        additives = "V"
        rest = rest[1:]

    # ratio is next two digits, then optional amount digits
    m = re.match(r"(?P<ratio>\d{2})(?P<num>\d*)$", rest)
    if not m:
        raise ValueError(f"Could not parse ratio/amount from '{raw}' -> '{s}' (rest='{rest}')")

    ratio = m.group("ratio")
    nums = m.group("num") or ""

    fec_wt, vc_wt = 0, 0
    if additives == "F":
        fec_wt = int(nums) if nums else 0
    elif additives == "V":
        vc_wt = int(nums) if nums else 0
    elif additives == "FV":
        fec_wt, vc_wt = _split_fv_amounts(nums)

    return ElectrolyteSpec(raw=raw, system=system, additives=additives, ratio=ratio, fec_wt=fec_wt, vc_wt=vc_wt)

def ratio_to_text(ratio: str) -> str:
    """'14' -> '1:4'"""
    if len(ratio) == 2 and ratio.isdigit():
        return f"{ratio[0]}:{ratio[1]}"
    return ratio

def pretty_label(code: str) -> str:
    """Compact slide-friendly label."""
    spec = parse_electrolyte(code)
    base = f"{spec.system} {ratio_to_text(spec.ratio)}"
    if spec.additives == "NONE":
        return base
    if spec.additives == "F":
        return f"{base} + FEC {spec.fec_wt}%"
    if spec.additives == "V":
        return f"{base} + VC {spec.vc_wt}%"
    return f"{base} + FEC {spec.fec_wt}% + VC {spec.vc_wt}%"

# ============================
# 4) STYLE FOR A CURVE
# ============================

def style_for_electrolyte(
    code: str,
    *,
    lw_base: float = 3.2,
    lw: float = 2.6,
    markevery: Optional[int] = 20,
) -> Dict:
    """
    Returns Matplotlib kwargs for ax.plot(...).
    Encoding:
      - Hue = additive package (NONE/F/V/FV)
      - Shade = wt% (higher = darker)
      - Marker = VC wt% (1 filled circle, 2 open square)
      - Linestyle = DT solid, TPT dashed
    """
    spec = parse_electrolyte(code)

    linestyle = SYSTEM_LS.get(spec.system, "-")

    # Color + shade
    if spec.additives == "NONE":
        color = BASE_COLOR["NONE"]
        linewidth = lw_base
    else:
        hue = BASE_COLOR[spec.additives]
        # Shade by the relevant wt%
        if spec.additives in ("F", "FV"):
            wt = spec.fec_wt
        else:
            wt = spec.vc_wt
        t = _t_from_wt(wt) if wt else _t_from_wt(1)
        color = blend_hex(hue, BASE_COLOR["WHITE"], t=t)
        linewidth = lw

    # VC markers only for V / FV packages
    marker = None
    mfc = None
    mec = None
    mew = None
    ms = None

    if spec.additives in ("V", "FV"):
        marker, fillmode, mew, ms = VC_MARKER.get(spec.vc_wt, (None, None, None, None))
        if marker is not None:
            if fillmode == "filled":
                mfc = color
                mec = color
                mew = 0.0
            else:
                mfc = "none"
                mec = color

    out = dict(
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=ms,
        markerfacecolor=mfc,
        markeredgecolor=mec,
        markeredgewidth=mew,
    )
    if marker is not None and markevery is not None:
        out["markevery"] = markevery

    return {k: v for k, v in out.items() if v is not None}

# ============================
# 5) TWO-BLOCK "ENCODING" LEGEND HANDLES
# ============================

def legend_handles_system_additives() -> List[mlines.Line2D]:
    """Colors = additive package; linestyle = system."""
    h = [
        mlines.Line2D([], [], color=BASE_COLOR["NONE"], lw=3.2, linestyle="-",  label="NONE (baseline)"),
        mlines.Line2D([], [], color=BASE_COLOR["F"],    lw=3.0, linestyle="-",  label="FEC package (F)"),
        mlines.Line2D([], [], color=BASE_COLOR["V"],    lw=3.0, linestyle="-",  label="VC package (V)"),
        mlines.Line2D([], [], color=BASE_COLOR["FV"],   lw=3.0, linestyle="-",  label="FEC+VC package (FV)"),
        mlines.Line2D([], [], color="#4A4A4A", lw=2.6, linestyle=SYSTEM_LS["DT"],  label="DT system (solid)"),
        mlines.Line2D([], [], color="#4A4A4A", lw=2.6, linestyle=SYSTEM_LS["TPT"], label="TPT system (dashed)"),
    ]
    return h

def legend_handles_vc_markers() -> List[mlines.Line2D]:
    """Markers = VC wt%."""
    h = [
        mlines.Line2D([], [], color="#4A4A4A", lw=2.6, linestyle="-", marker=None, label="VC 0% (no marker)"),
        mlines.Line2D([], [], color="#4A4A4A", lw=2.6, linestyle="-",
                      marker="o", markersize=6, markerfacecolor="#4A4A4A",
                      markeredgecolor="#4A4A4A", markeredgewidth=0.0, label="VC 1%"),
        mlines.Line2D([], [], color="#4A4A4A", lw=2.6, linestyle="-",
                      marker="s", markersize=6, markerfacecolor="none",
                      markeredgecolor="#4A4A4A", markeredgewidth=1.2, label="VC 2%"),
    ]
    return h

def add_two_block_legend(
    ax,
    *,
    loc_colors: str = "upper left",
    loc_vc: str = "lower left",
    frameon: bool = False,
    fontsize: int = 10,
):
    """Adds two separate legends inside the axis."""
    leg1 = ax.legend(
        handles=legend_handles_system_additives(),
        loc=loc_colors,
        frameon=frameon,
        fontsize=fontsize,
        title="Encoding",
        title_fontsize=fontsize,
        borderaxespad=0.0,
        handlelength=2.8,
        handletextpad=0.8,
        labelspacing=0.6,
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=legend_handles_vc_markers(),
        loc=loc_vc,
        frameon=frameon,
        fontsize=fontsize,
        title="VC markers",
        title_fontsize=fontsize,
        borderaxespad=0.0,
        handlelength=2.8,
        handletextpad=0.8,
        labelspacing=0.6,
    )
    ax.add_artist(leg2)

    return leg1, leg2

# ============================
# 6) SAVE LEGENDS AS SEPARATE PNGs
# ============================

def save_encoding_legend_png(
    out_png: str,
    *,
    dpi: int = 600,
    fontsize: int = 11,
    transparent: bool = True,
    pad_in: float = 0.02,
) -> None:
    """Standalone PNG with both encoding legends (no curve list)."""
    fig = plt.figure(figsize=(6.5, 2.4))
    fig.patch.set_alpha(0.0 if transparent else 1.0)
    ax = fig.add_subplot(111)
    ax.axis("off")

    leg1 = ax.legend(
        handles=legend_handles_system_additives(),
        loc="upper left",
        frameon=False,
        fontsize=fontsize,
        title="Encoding",
        title_fontsize=fontsize,
        borderaxespad=0.0,
        handlelength=2.8,
        handletextpad=0.8,
        labelspacing=0.6,
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=legend_handles_vc_markers(),
        loc="lower left",
        frameon=False,
        fontsize=fontsize,
        title="VC markers",
        title_fontsize=fontsize,
        borderaxespad=0.0,
        handlelength=2.8,
        handletextpad=0.8,
        labelspacing=0.6,
    )
    ax.add_artist(leg2)

    fig.savefig(out_png, dpi=dpi, transparent=transparent, bbox_inches="tight", pad_inches=pad_in)
    plt.close(fig)

def save_ax_legend_png(
    ax,
    out_png: str,
    *,
    title: Optional[str] = None,
    fontsize: int = 11,
    ncol: int = 1,
    dpi: int = 600,
    transparent: bool = True,
    pad_in: float = 0.02,
) -> None:
    """
    One-axis-per-plot: save EXACT legend handles/labels from the axis as a standalone PNG.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        raise ValueError("Axis has no legend handles/labels. Did you set label=... in ax.plot(...) ?")

    fig_h = max(1.0, 0.32 * (len(labels) / max(1, ncol)))
    fig_w = 6.0 if ncol == 1 else 9.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_alpha(0.0 if transparent else 1.0)

    ax_leg = fig.add_subplot(111)
    ax_leg.axis("off")

    ax_leg.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        frameon=False,
        fontsize=fontsize,
        ncol=ncol,
        title=title,
        title_fontsize=fontsize,
        borderaxespad=0.0,
        handlelength=2.8,
        handletextpad=0.8,
        labelspacing=0.6,
    )

    fig.savefig(out_png, dpi=dpi, transparent=transparent, bbox_inches="tight", pad_inches=pad_in)
    plt.close(fig)




# ------------------------------------------------------------
# One-call driver: makes the whole figure set
# ------------------------------------------------------------
def export_proposal_cycling_figures(comparison_set,
                                   out_dir=r"C:\Users\benja\Downloads\Final Countdown\Proposal Slide Figures - Cycling Plots",
                                   tag="JoshAsk",
                                   normalized=False,
                                   color_scheme=None):
    """
    Exports:
      1) Representative rate curves (V-Q) + legend-only
      2) Capacity vs cycle FULL + legend-only
      3) Capacity vs cycle RATE (1–19) + legend-only
      4) Capacity vs cycle LIFE (20–cut) + legend-only
    """
    out_dir = _ensure_dir(out_dir)

    plot_representative_rate_curves_export(
        comparison_set,
        out_dir=out_dir,
        save_stem=f"{tag}_Representative_Rate_Curves",
        normalized=normalized,
        color_scheme=color_scheme,
        figsize=(10, 6),
    )

    plot_capacity_vs_cycle_exports(
        comparison_set,
        out_dir=out_dir,
        save_prefix=f"{tag}_Capacity_vs_Cycle",
        normalized=normalized,
        color_scheme=color_scheme,
        figsize=(10, 6),
        use_charge=False  # matches your existing capacity-vs-cycle convention
    )

    print(f"Export complete → {out_dir}")
import numpy as np
import matplotlib.pyplot as plt

def plot_capacity_rate_and_life_like_example(
        file_tuples,                     # list of (file_path, key, cell_code) OR nested list (comparison_set)
        normalized=False,
        x_bounds=(1, 200),
        save_path=None,
        color_scheme=None,
        title='Capacity vs. Cycle Number (Rate + Cycle Life)'
    ):
    """
    Capacity vs cycle (DISCHARGE capacity only), formatted like your example:
    - points only
    - vertical dashed lines separating rate steps
    - C-rate labels at the top
    - Labels aligned at top/bottom bands
    - Optional color_scheme (cell_code -> color)
    """

    # ---------- Flatten comparison_set ----------
    flattened = []
    for item in file_tuples:
        if isinstance(item, (list, tuple)) and len(item) == 3 and isinstance(item[0], str):
            flattened.append(item)
        elif isinstance(item, list):
            for sub in item:
                if isinstance(sub, (list, tuple)) and len(sub) == 3:
                    flattened.append(sub)

    if not flattened:
        raise ValueError("No valid (file_path, key, cell_code) tuples found.")

    # DEBUG: show what was flattened so we can see whether LPV_controls/DT14_control/DTF_set made it
    try:
        dbg_list = [(os.path.basename(fp), key, cell_code) for (fp, key, cell_code) in flattened]
        print("[DEBUG] Flattened tuples count:", len(flattened))
        for idx, (fn, key, code) in enumerate(dbg_list):
            print(f"[DEBUG] {idx}: file={fn}, key={key}, cell_code={code}")
    except Exception as e:
        print("[DEBUG] Could not print flattened list:", e)

    def electrolyte_label(key):
        if ' - ' in key:
            tail = key.split(' - ', 1)[1]
            return tail.replace(' Elyte', '').split(' (', 1)[0].strip()
        return key
    fig, ax = plt.subplots(figsize=(10, 6))

    # ---- plot discharge capacity (points only) ----
    for i, (file_path, key, cell_code) in enumerate(flattened):
        print(f"[DEBUG] Processing index {i}: file_path={os.path.basename(file_path)}, key={key}, cell_code={cell_code}")
        cycles, charge_caps, discharge_caps, ce = process_cycle_data(file_path, key, normalized)

        print(f"[DEBUG] Raw cycles found: {len(cycles)}, discharge_caps: {len(discharge_caps)}")

        cyc = np.array(cycles)
        cap = np.array(discharge_caps)  # DISCHARGE capacity

        # apply x bounds
        m = (cyc >= x_bounds[0]) & (cyc <= x_bounds[1])
        cyc = cyc[m]
        cap = cap[m]
        print(f"[DEBUG] After x_bounds {x_bounds}: cycles_in_range={len(cyc)}, caps_in_range={len(cap)}")
        if len(cyc) == 0:
            print(f"[DEBUG] Skipping {os.path.basename(file_path)} ({cell_code}): no cycles in x_bounds {x_bounds}")
            continue

        lab = electrolyte_label(key)

        if color_scheme and cell_code in color_scheme:
            col = color_scheme[cell_code]
            ax.scatter(cyc, cap, s=38, color=col, label=lab)
        else:
            ax.scatter(cyc, cap, s=38, label=lab)

    # ---- vertical dashed lines for rate windows (match your example) ----
    # boundaries at 4.5, 7.5, 10.5, 13.5, 16.5, 19.5
    for x in [4.5, 7.5, 10.5, 13.5, 16.5, 19.5]:
        if x_bounds[0] <= x <= x_bounds[1]:
            ax.axvline(x=x, color='black', linestyle='--', linewidth=2.6)

    # ---- axis formatting (Josh style) ----
    ax.set_xlabel('Cycle Number', fontsize=14)
    ax.set_ylabel('Capacity (%)' if normalized else 'Capacity (mAh/g)', fontsize=24)
    ax.set_xlim(x_bounds)

    # tick formatting like your other plot
    ax.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='both', labelsize=24)
    ax.grid(False)

    # ---- C-rate labels across the top (same segment logic you used) ----
    # rate test is cycles 1–19, then cycle life continues (you can rename last label)
    c_rate_segments = [
        (3, 4, "C/10"),
        (5, 7, "C/8"),
        (8, 10, "C/4"),
        (11, 13, "C/2"),
        (14, 16, "1C"),
        (17, 19, "2C"),
        (20, x_bounds[1], "Cycle life"),  # or "C/2" if that’s what you want
    ]

    y_top = ax.get_ylim()[1]
    for start, end, label in c_rate_segments:
        seg_start = max(start, x_bounds[0])
        seg_end = min(end, x_bounds[1])
        if seg_start <= seg_end:
            x_mid = 0.5 * (seg_start + seg_end)
            ax.text(x_mid, y_top * 0.99, label, fontsize=20, ha='center', va='top', color='black')

    # ---- legend (dedupe) ----
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        unique.setdefault(l, h)
    ax.legend(list(unique.values()), list(unique.keys()),
              loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fontsize=16, ncol=2, frameon=False)

    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# Continue with plotting only if list is non-empty
# if files_to_compare:
#     compare_cells_on_same_plot(files_to_compare, normalized=False)
# files_to_compare = [
#      get_tuples_by_cell_code(file_paths_keys, r'DN02')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'DR02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DT01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DU02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DV01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DW02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DX02')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'DZ02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EA02')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'EB03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EC01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EG02')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'EH02')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'EI03')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'EJ03')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'EL03')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'EM01')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'EN02')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'EO02')[0],
#  ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)

DTFV_set = [get_tuples_by_cell_code(file_paths_keys, r'FC03')[0],
            get_tuples_by_cell_code(file_paths_keys, r'FD03')[0],
        get_tuples_by_cell_code(file_paths_keys, r'FE01')[0],
        get_tuples_by_cell_code(file_paths_keys, r'FF02')[0],
        get_tuples_by_cell_code(file_paths_keys, r'FG03')[0],
            get_tuples_by_cell_code(file_paths_keys, r'ES03')[0],
]
# DTFV_set = [get_tuples_by_cell_code(file_paths_keys, r'FC03')[0],
#             get_tuples_by_cell_code(file_paths_keys, r'FF02')[0],
#             get_tuples_by_cell_code(file_paths_keys, r'FG03')[0],
#         ]

DTF_set = [get_tuples_by_cell_code(file_paths_keys, r'EN02')[0],
           get_tuples_by_cell_code(file_paths_keys, r'DU02')[0],
        get_tuples_by_cell_code(file_paths_keys, r'EO02')[0],
        get_tuples_by_cell_code(file_paths_keys, r'EJ03')[0],
]
# DTF_set = [
#            get_tuples_by_cell_code(file_paths_keys, r'DU02')[0],
#
#         get_tuples_by_cell_code(file_paths_keys, r'EJ03')[0],
# ]
LPV_controls = [get_tuples_by_cell_code(file_paths_keys, r'EU03')[0],
    #get_tuples_by_cell_code(file_paths_keys, r'EV03')[0],
    ]
DT14_control_2 = [get_tuples_by_cell_code(file_paths_keys, r'DQ01')[0],]
TPT_set = [get_tuples_by_cell_code(file_paths_keys, r'IM01')[0],
           get_tuples_by_cell_code(file_paths_keys, r'IN01')[0],
           get_tuples_by_cell_code(file_paths_keys, r'IO01')[0]]
# #tuple_control_gr = [get_tuples_by_cell_code(file_paths_keys, r'EV03')[0],
#     ]
# DT14_control = [get_tuples_by_cell_code(file_paths_keys, r'DP01')[0],]
#
# MF_set = [get_tuples_by_cell_code(file_paths_keys, r'EC01')[0],]
comparison_set = [LPV_controls, DT14_control_2, DTF_set, ]#DTFV_set]
plot_capacity_rate_and_life_like_example(
    comparison_set,
    normalized=False,
    x_bounds=(1, 70),   # set to your full cycling range
    save_path='comp_'+str(comparison_set)       # or a full png path
)

# import pprint as pp
# #pp.pprint(DT14_control)
#form_set_DTF2 = [get_tuples_by_cell_code(file_paths_keys, r'DU02')[0],
#                get_tuples_by_cell_code(file_paths_keys, r'FH02')[0],
#                get_tuples_by_cell_code(file_paths_keys, r'FH05')[0],
#                ]
#form_set_DTV = [get_tuples_by_cell_code(file_paths_keys, r'DY01')[0],
#               get_tuples_by_cell_code(file_paths_keys, r'FI03')[0],
#                get_tuples_by_cell_code(file_paths_keys, r'FI05')[0],
#                ]
# form_set_DTFV = [get_tuples_by_cell_code(file_paths_keys, r'FF02')[0],
#                  get_tuples_by_cell_code(file_paths_keys, r'FJ02')[0],
#                  get_tuples_by_cell_code(file_paths_keys, r'FJ04')[0],
#                  ]
# form_set_mf91 = [get_tuples_by_cell_code(file_paths_keys, r'EC01')[0],
#                  get_tuples_by_cell_code(file_paths_keys, r'FK02')[0],
#                  get_tuples_by_cell_code(file_paths_keys, r'FK05')[0],
#                  get_tuples_by_cell_code(file_paths_keys, r'FM01')[0],
#                  get_tuples_by_cell_code(file_paths_keys, r'FM06')[0],
#                  ]
# Real_comp = [get_tuples_by_cell_code(file_paths_keys, r'EU03')[0],
#              get_tuples_by_cell_code(file_paths_keys, r'FF02')[0],
#              get_tuples_by_cell_code(file_paths_keys, r'EC01')[0],]
# josh_ask = [get_tuples_by_cell_code(file_paths_keys, r'EU03')[0],
#             get_tuples_by_cell_code(file_paths_keys, r'EC01')[0],
#             get_tuples_by_cell_code(file_paths_keys, r'FF02')[0],]



# Forms_0716 =[
#     get_tuples_by_cell_code(file_paths_keys, r'FR03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'FR04')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'FR05')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'FR06')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'FR07')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'FS03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'FS04')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'FS05')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'FS06')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'FS07')[0],
# ]
# One cell → all discharge curves smoothed with Savitzky–Golay
#plot_dq_dv_all_cycles(selected_cell, segment='discharge', smooth='savgol')

# One cell → compare cycle-1 vs cycle-3
#plot_dq_dv_difference(selected_cell, 1, 3, segment='charge', smooth='rolling')

# Many cells → mean trace (+1 σ) of first-cycle charge dQ/dV
#plot_mean_dq_dv(files_to_compare, segment='charge', smooth='savgol')

export_proposal_cycling_figures(
    comparison_set,
    tag="JoshAsk",
    normalized=False,
    color_scheme=None  # or your Tol/Josh mapping dict
)



cycle_str = 'CycleLife_JoshColors'
rate_str = 'Rate_JoshColors'
rate_bounds = (0, 19.5)
cycle_life_bounds = (0, 70)
# Full_set = []
# Full_set.extend(form_set_DTF2)
# Full_set.extend(form_set_DTV)
# Full_set.extend(form_set_DTFV)
# Full_set.extend(form_set_mf91)
# cell_codes= [cell_code for _, _, cell_code in Full_set]
# custom_colors = assign_tol_colors(cell_codes)
# compare_cells_on_same_plot()
#compare_cells_on_same_plot(form_set_DTF2, normalized=False, x_bounds=rate_bounds, save_str=rate_str+'DTF2', color_scheme=None)
#compare_cells_on_same_plot(form_set_DTV, normalized=False, x_bounds=rate_bounds, save_str=rate_str+'DTV', color_scheme=None)
#compare_cells_on_same_plot(form_set_DTFV, normalized=False, x_bounds=rate_bounds, save_str=rate_str+'DTFV', color_scheme=None)
#compare_cells_on_same_plot(form_set_mf91, normalized=False, x_bounds=cycle_life_bounds, save_str=rate_str+'MF91_life', color_scheme=None)
compare_cells_on_same_plot(comparison_set, normalized=False, x_bounds=cycle_life_bounds, save_str=rate_str+'JoshAsk_life', color_scheme=None)
compare_cells_on_same_plot(Real_comp, normalized=False, x_bounds=cycle_life_bounds, save_str=rate_str+'comp_life', color_scheme=None)
compare_cells_on_same_plot(Real_comp, normalized=False, x_bounds=rate_bounds, save_str=rate_str+'comp_rate', color_scheme=None)

DT_Set = []
DT_Set.extend(DT14_control)
DT_Set.extend(LPV_controls)

Full_set = []
Full_set.extend(LPV_controls)
Full_set.extend(DT14_control)
Full_set.extend(DTF_set)
Full_set.extend(DTFV_set)
Full_set.extend(MF_set)
cell_codes= [cell_code for _, _, cell_code in Full_set]
custom_colors = assign_tol_colors(cell_codes)

cycle_str = 'CycleLife_JoshColors'
rate_str = 'Rate_JoshColors'
rate_bounds = (0, 19.5)
cycle_life_bounds = (19.5, 100)

compare_cells_on_same_plot(DT_Set, normalized=False, x_bounds=rate_bounds, save_str=rate_str+'DT', color_scheme=custom_colors)
compare_cells_on_same_plot(DTF_set, normalized=False, x_bounds=rate_bounds, save_str=rate_str+'DTF', color_scheme=custom_colors)
compare_cells_on_same_plot(DTFV_set, normalized=False, x_bounds=rate_bounds, save_str=rate_str+'DTFV', color_scheme=custom_colors)
compare_cells_on_same_plot(Full_set, normalized=False, x_bounds=rate_bounds, save_str=rate_str+'Full', color_scheme=custom_colors)

files_to_compare = []
files_to_compare.extend(DT14_control)
files_to_compare.extend(LPV_controls)
files_to_compare.extend(DTF_set)
cell_codes = [cell_code for _, _, cell_code in files_to_compare]
custom_colors = assign_tol_colors(files_to_compare)
compare_cells_on_same_plot(DT_Set, normalized=False, x_bounds=(0, 19.5), save_str='CycleLife_TolColors', color_scheme=custom_colors)
#pp.pprint(files_to_compare)
# Flatten the list if it contains nested lists


# Now extract cell codes
#cell_codes = [cell_code for cell_path, key, cell_code in files_to_compare]


compare_cells_on_same_plot(files_to_compare, normalized=False, x_bounds=(0, 20), save_str='CycleLife_TolColors', color_scheme=custom_colors)


files_to_compare  = [
    get_tuples_by_cell_code(file_paths_keys, r'EN02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'DU02')[0],
get_tuples_by_cell_code(file_paths_keys, r'EO02')[0],
get_tuples_by_cell_code(file_paths_keys, r'EJ03')[0],
get_tuples_by_cell_code(file_paths_keys, r'ES03')[0],
get_tuples_by_cell_code(file_paths_keys, r'EU03')[0],
    ]
compare_cells_on_same_plot(files_to_compare, normalized=False, x_bounds = (0, 19.4),save_str='DTF14_19_5')
compare_cells_cycle_2(files_to_compare, normalized=False)

files_to_compare  = [
    get_tuples_by_cell_code(file_paths_keys, r'EP03')[0],
    get_tuples_by_cell_code(file_paths_keys, r'ER03')[0],
get_tuples_by_cell_code(file_paths_keys, r'ET01')[0],
get_tuples_by_cell_code(file_paths_keys, r'EC01')[0],
get_tuples_by_cell_code(file_paths_keys, r'EU03')[0],

    ]
compare_cells_on_same_plot(files_to_compare, normalized=False, x_bounds = (0, 19.4),save_str='MF91_19_5')
compare_cells_cycle_2(files_to_compare, normalized=False)

#DT14 Comparison
files_to_compare = [
     get_tuples_by_cell_code(file_paths_keys, r'DN02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'DP01')[0],
 ].append(tuple_controls)
compare_cells_on_same_plot(files_to_compare, normalized=False, x_bounds = (0, 100),save_str='DT14_100')
compare_cells_on_same_plot(files_to_compare, normalized=False, x_bounds = (0, 19.4),save_str='DT14_19_5')
compare_cells_on_same_plot(files_to_compare, normalized=False, x_bounds = (20, 100),save_str='DT14_20-100')

files_to_compare = [
     get_tuples_by_cell_code(file_paths_keys, r'DN02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'DP01')[0],
 ].append(tuple_controls)
compare_cells_on_same_plot(files_to_compare, normalized=False, x_bounds = (0, 100),save_str='DT14_100')
compare_cells_on_same_plot(files_to_compare, normalized=False, x_bounds = (0, 19.4),save_str='DT14_19_5')
compare_cells_on_same_plot(files_to_compare, normalized=False, x_bounds = (20, 100),save_str='DT14_20-100')

#
# #Li|LFP
# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DN01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DR02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DZ03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DV01')[0],
# ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)
#
# #LTO|LFP
# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DO03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DS03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EA02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DW02')[0],
# ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)
# Example usage:
# Assume `file_paths_keys` is the list of tuples you generated for the cell files.
# Here, we select the first tuple as the cell we want to plot.
selected_cell = get_tuples_by_cell_code(file_paths_keys, r'EA02')[0]  # Replace with the tuple for your selected cell.
plot_selected_cycles_charge_and_discharge_vs_voltage(selected_cell, normalized=False)

#
# #Li|NMC
# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DP01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DT03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EB03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DX02')[0],
# ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)
#
# #Gr|NMC
# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DQ01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DU02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EC01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DY01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DD03')[0],
# ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)
#
# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DN06')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DO06')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DP06')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DR06')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DS06')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DT06')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DU06')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DZ06')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EA06')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EB06')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EC06')[0],
# ]
#

# compare_cells_on_same_plot(files_to_compare, normalized=False)

files_to_compare = [
    #get_tuples_by_cell_code(file_paths_keys, r'DR02')[0],
    #get_tuples_by_cell_code(file_paths_keys, r'DS03')[0],
    get_tuples_by_cell_code(file_paths_keys, r'DT03')[0],
    get_tuples_by_cell_code(file_paths_keys, r'DU02')[0],
    #get_tuples_by_cell_code(file_paths_keys, r'EG02')[0],
    #get_tuples_by_cell_code(file_paths_keys, r'EH02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'EI03')[0],
    get_tuples_by_cell_code(file_paths_keys, r'EJ03')[0],
]
compare_cells_on_same_plot(files_to_compare, normalized=False)

files_to_compare = [
    get_tuples_by_cell_code(file_paths_keys, r'DR02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'DS03')[0],
    #get_tuples_by_cell_code(file_paths_keys, r'DT03')[0],
    #get_tuples_by_cell_code(file_paths_keys, r'DU02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'EG02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'EH02')[0],
    #get_tuples_by_cell_code(file_paths_keys, r'EI03')[0],
    #get_tuples_by_cell_code(file_paths_keys, r'EJ03')[0],
]
compare_cells_on_same_plot(files_to_compare, normalized=False)

def plot_discharge_curves_at_minus_51C(file_paths_keys, normalized=False):
    """Plot the -51°C discharge curves for all cells."""
    plt.figure(figsize=(10, 6))
    colors = matplotlib.colormaps["tab10"].resampled(len(file_paths_keys)).colors
    print('-51°C Discharge Curves:')
    print('-----------------------')
    print(file_paths_keys)

    for i, (file_path, key, cell_code) in enumerate(file_paths_keys):
        try:
            cycles_data, norm_factor = process_all_cycles_for_voltage_vs_capacity(file_path, key, normalized)
            print(f"Processing {file_path} for voltage vs capacity...")
            print(f"Dataset key: {key}")
            print(f"Cycles Data: {cycles_data}")
        except Exception as e:
            print(f"Error processing {file_path} for voltage vs capacity: {e}")
            continue

        for cycle, charge, discharge in cycles_data:
            if not discharge.empty and '-51C' in key:
                plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor, discharge['Voltage (V)'],
                         label=f'{key} Cycle {cycle} (Discharge)', linestyle='--', color=colors[i])

    plt.xlabel('Capacity (Ah)')
    plt.ylabel('Voltage (V)')
    plt.title('Discharge Curves at -51°C')
    plt.legend(fontsize='small', ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.show()


def generate_file_paths_keys_low_temp(directory, lookup_table_path):
    """
    Walk through the directory (and subdirectories) to find Excel files that contain '-51C'
    in their filename. For each file, extract the cell identifier and lookup additional details
    in the provided lookup table.
    Returns a list of tuples: (full_path, key, cell_code)
    """
    file_paths_keys = []
    lookup_df = load_lookup_df(lookup_table_path)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx') and '-51C' in file:
                full_path = os.path.join(root, file)
                cell_identifier = extract_cell_identifier(file)
                if cell_identifier is None:
                    print(f"Could not extract cell identifier from file: {file}")
                    continue
                cell_code = cell_identifier[:2]
                lookup_row = lookup_df[lookup_df['Cell Code'] == cell_code]
                if lookup_row.empty:
                    print(f"Cell code {cell_code} not found in lookup table for file: {file}")
                    continue
                row = lookup_row.iloc[0]
                anode = row['Anode'] if not pd.isna(row['Anode']) else ''
                cathode = row['Cathode'] if not pd.isna(row['Cathode']) else ''
                electrolyte = row['Electrolyte'] if not pd.isna(row['Electrolyte']) else ''
                key = f"{anode}|{cathode} - {electrolyte} Elyte ({cell_identifier})"
                file_paths_keys.append((full_path, key, cell_code))
    return file_paths_keys
