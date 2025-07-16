import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog
import matplotlib
import numpy as np
from scipy.signal import savgol_filter      # pip install scipy if needed

#from Python_Codes.Arbin_SymCellCycling.Update_Scripts.Scratch_t6 import files_to_compare

# Provide the path to your lookup table Excel file.
lookup_table_path = r'C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx'
# lookup_table_path = filedialog.askopenfilename(title="Select Lookup Table")

search_directory = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\03\Cycle Life Best Survivors'

#search_directory = r'C:\Users\benja\Downloads\Temp\Data_Work_4_19\Cycle Life Best Survivors'
search_directory = r'C:\Users\benja\Downloads\Temp\Data_Work_4_19\Cycle Life Best Survivors\Form Experiment'
search_directory = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\KRI Arbin\Low Temp Li Ion\2025\07'
# ==========================
# 1. Set the working directory
# ==========================
os.chdir(search_directory)


# ==========================
# 2. Helper functions
# ==========================
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
    lookup_df = pd.read_excel(lookup_table_path)

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

    if normalized:
        if 'LFP' in dataset_key:
            norm_factor = capacities['LFP']
        elif 'NMC' in dataset_key:
            norm_factor = capacities['NMC']
        elif 'Gr' in dataset_key:
            norm_factor = capacities['Gr']
        else:
            raise ValueError("Dataset key does not match known capacities")
    else:
        if 'LFP' in dataset_key:
            norm_factor = weights_g['LFP']
        elif 'NMC' in dataset_key:
            norm_factor = weights_g['NMC']
        elif 'Gr' in dataset_key:
            norm_factor = weights_g['Gr']
        else:
            raise ValueError("Dataset key does not match known capacities")

    data = pd.ExcelFile(file_path)
    data_sheets = [sheet for sheet in data.sheet_names if sheet.startswith('Channel')]
    if not data_sheets:
        raise ValueError(f"No sheet starting with 'Channel' found in {file_path}")
    sheet_data = data.parse(data_sheets[0])
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

    if normalized:
        if 'LFP' in dataset_key:
            norm_factor = capacities['LFP']
        elif 'NMC' in dataset_key:
            norm_factor = capacities['NMC']
        elif 'Gr' in dataset_key:
            norm_factor = capacities['Gr']
        else:
            raise ValueError("Dataset key does not match known capacities")
    else:
        if 'LFP' in dataset_key:
            norm_factor = weights_g['LFP']
        elif 'NMC' in dataset_key:
            norm_factor = weights_g['NMC']
        elif 'Gr' in dataset_key:
            norm_factor = weights_g['Gr']
        else:
            raise ValueError("Dataset key does not match known capacities")

    data = pd.ExcelFile(file_path)
    data_sheets = [sheet for sheet in data.sheet_names if sheet.startswith('Channel')]
    if not data_sheets:
        raise ValueError(f"No sheet starting with 'Channel' found in {file_path}")
    sheet_data = data.parse(data_sheets[0])

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
    lookup_df = pd.read_excel(lookup_table_path)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx'):
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

    if normalized:
        if 'LFP' in dataset_key:
            norm_factor = capacities['LFP']
        elif '16mm' in dataset_key:
            norm_factor = capacities['16mm']
        elif 'NMC' in dataset_key:
            norm_factor = capacities['NMC']
        elif 'Gr' in dataset_key:
            norm_factor = capacities['Gr']
        else:
            raise ValueError("Dataset key does not match known capacities")
    else:
        if 'LFP' in dataset_key:
            norm_factor = weights_g['LFP']
        elif '16mm' in dataset_key:
            norm_factor = capacities['16mm']
        elif 'NMC' in dataset_key:
            norm_factor = weights_g['NMC']
        elif 'Gr' in dataset_key:
            norm_factor = weights_g['Gr']
        else:
            raise ValueError("Dataset key does not match known capacities")

    data = pd.ExcelFile(file_path)
    data_sheets = [sheet for sheet in data.sheet_names if sheet.startswith('Channel')]
    if not data_sheets:
        raise ValueError(f"No sheet starting with 'Channel' found in {file_path}")
    sheet_data = data.parse(data_sheets[0])
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
    This function assumes the file contains columns 'Test Time (s)' and 'Voltage (V)'.
    It does not group by cycle; it simply plots the entire time series.
    """
    try:
        data = pd.ExcelFile(file_path)
        data_sheets = [sheet for sheet in data.sheet_names if sheet.startswith('Channel')]
        if not data_sheets:
            raise ValueError(f"No sheet starting with 'Channel' found in {file_path}")
        sheet_data = data.parse(data_sheets[0])
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

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

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

    for i, (file_path, key, cell_code) in enumerate(file_tuples):
        try:
            cycles, charge_caps, discharge_caps, ce = process_cycle_data(file_path, key, normalized)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

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
                        label=f'{format_key(key)}')
            # ax1.scatter(cycles, discharge_caps, marker=base_marker,
            #             facecolors=color, edgecolors=color,
            #             label=f'{key} (Discharge)', linestyle='--')

            ax2.scatter(cycles, ce, marker=ce_marker,
                        label=f'{format_key(key)} (CE)')

        # Add C-Rate annotations
        if x_bounds[1]<20:
            for cycle, label in c_rate_labels.items():
                if cycle in cycles and cycle not in annotated_cycles:
                    x = cycle-1
                    y = 200
                    ax1.text(x, y + 5, label, fontsize=10, ha='center', color='black')
                    annotated_cycles.add(cycle)

    # Formatting
    ax1.grid(False)
    ax2.grid(False)

    for cycle in [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5]:
        ax1.axvline(x=cycle, color='black', linestyle='--')

    ax1.set_xlabel('Cycle Number')
    ax1.set_xlim(x_bounds)
    ax1.set_ylim(0, 220)
    ax1.set_ylabel('Capacity (%)' if normalized else 'Capacity (mAh/g)')
    ax2.set_ylabel('Coulombic Efficiency (%)')
    ax2.set_ylim(0, 120)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small',
               ncol=2)

    plt.title('Capacity and Coulombic Efficiency vs. Cycle Number')
    # 1) Force specific integer ticks on the bottom x‐axis
    ax1.set_xticks([ 25, 30, 35, 40, 45])

    # 2) Mirror those ticks on the top, and move all tick‐marks inward
    ax1.tick_params(
        which='both',  # apply to both major and minor ticks
        axis='both',  # apply on both axes
        direction='in',  # point ticks inward
        bottom=True, top=True,
        left=True, right=True
    )
    ax2.set_yticks([])  # no ticks
    ax2.set_yticklabels([])  # no tick labels
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
# ---------------------------------------------------------------------------
# Pre‑processing helpers
# ---------------------------------------------------------------------------

def preprocess_df(df, v_col='Voltage (V)', q_col='Charge Capacity (Ah)',
                  dedup_thresh=1e-4,  # volts
                  bin_width=0.002):   # volts
    """Return a cleaned & binned copy of *df* ready for dQ/dV calculation.

    1. Drops consecutive rows whose voltage change is below *dedup_thresh*.
    2. Optionally bins the remaining data into fixed‑width voltage bins and
       averages the capacities inside each bin to suppress quantisation noise.
    """
    if df.empty:
        return df

    # Step 1 – cull near‑duplicate voltages
    mask = df[v_col].diff().abs().fillna(1) >= dedup_thresh
    df = df.loc[mask]

    # Step 2 – fixed‑width voltage bins
    if bin_width is not None and bin_width > 0:
        bins = np.arange(df[v_col].min(), df[v_col].max() + bin_width, bin_width)
        # digitize returns 1‑based bin indices; use them to group
        grouped = df.groupby(np.digitize(df[v_col], bins))
        df = grouped[[v_col, q_col]].mean().dropna()

    return df

# ---------------------------------------------------------------------------
# Smoothing helper
# ---------------------------------------------------------------------------

def smooth_series(y, mode=None, window=11, poly=3):
    """Return a smoothed copy of *y*.

    Parameters
    ----------
    y : 1‑D array‑like
    mode : None | 'rolling' | 'savgol'
    window : int
        Window length for rolling average or Savitzky‑Golay filter.
    poly : int
        Polynomial order for Savitzky‑Golay.
    """
    if mode is None:
        return y
    if mode == "rolling":
        return pd.Series(y).rolling(window, center=True, min_periods=1).mean().to_numpy()
    if mode == "savgol":
        if window % 2 == 0:
            window += 1  # Savitzky‑Golay requires odd window length
        return savgol_filter(y, window, poly, mode='interp')
    raise ValueError("smooth must be None, 'rolling', or 'savgol'")

# ---------------------------------------------------------------------------
# Main derivative routine
# ---------------------------------------------------------------------------

def compute_dq_dv(df, q_col='Charge Capacity (Ah)', v_col='Voltage (V)',
                  smooth=None, preprocess=True, **pre_kw):
    """Compute dQ/dV from a raw cc/voltage dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *v_col* and *q_col* columns.
    q_col, v_col : str
        Column names for capacity and voltage.
    smooth : None | 'rolling' | 'savgol'
        Optional pre‑smoothing of capacity trace.
    preprocess : bool
        If True (default) run :func:`preprocess_df` before differentiating.
    **pre_kw :
        Extra keyword arguments forwarded to :func:`preprocess_df`
        (e.g. dedup_thresh=5e-5, bin_width=0.001).

    Returns
    -------
    v_mid : np.ndarray
        Mid‑point voltage grid.
    dq_dv : np.ndarray
        Incremental capacity values (same length as *v_mid*).
    """
    if preprocess:
        df = preprocess_df(df, v_col=v_col, q_col=q_col, **pre_kw)

    if df.empty:
        return np.array([]), np.array([])

    v = df[v_col].to_numpy()
    q = df[q_col].to_numpy()

    # Optional smoothing BEFORE differentiation
    q = smooth_series(q, smooth)

    # Central difference
    dq = np.diff(q)
    dv = np.diff(v)
    with np.errstate(divide='ignore', invalid='ignore'):
        dq_dv = dq / dv
    v_mid = 0.5 * (v[:-1] + v[1:])

    # Filter invalid points
    mask = np.isfinite(dq_dv)
    return v_mid[mask], dq_dv[mask]

def compute_dq_dv_2(
        df,
        q_col='Charge Capacity (Ah)',
        v_col='Voltage (V)',
        *,
        # ── new pre-smoothing options
        smooth=None,
        smooth_kw=None,          # dict passed to smooth_series
        # ── new post-smoothing (optional)
        post_smooth=None,
        post_kw=None,
        # ── preprocessing forwarded to preprocess_df
        bin_width=0.002,         # V
        dedup_thresh=1e-4,       # V
        preprocess=True,
):
    """Return (voltage_midpoints, dQ/dV) for one half-cycle."""
    smooth_kw = smooth_kw or {}
    post_kw = post_kw or {}

    # 1.  basic cleaning / binning
    if preprocess:
        df = preprocess_df(df, v_col=v_col, q_col=q_col,
                           bin_width=bin_width,
                           dedup_thresh=dedup_thresh)

    v = df[v_col].to_numpy()
    q = df[q_col].to_numpy()

    # 2.  optional pre-smoothing on capacity
    q = smooth_series(q, mode=smooth, **smooth_kw)

    # 3.  derivative
    dq = np.diff(q)
    dv = np.diff(v)
    v_mid = 0.5 * (v[:-1] + v[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        y = dq / dv
    y[~np.isfinite(y)] = np.nan   # clean inf / nan

    # 4.  optional post-smoothing on dQ/dV itself
    y = smooth_series(y, mode=post_smooth, **post_kw)

    return v_mid, y

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
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Multi-cell overlay: discharge capacity (left-y) + CE (right-y)
# ------------------------------------------------------------------
def plot_discharge_and_ce_vs_cycle(
        file_tuples,
        normalized=False,
        x_bounds=None,
        save_str='',
        color_scheme=None,
        marker_cycle_labels=(2, 4, 7, 10, 13, 16, 19)
):
    """
    Overlay *discharge* capacity and Coulombic efficiency (CE) for several cells.

    Parameters
    ----------
    file_tuples : list[(path, key, cell_code)]
        Output of ``get_tuples_by_cell_code(...)`` or similar.
    normalized : bool, default False
        Pass-through to ``process_cycle_data`` (affects capacity units).
    x_bounds : tuple(float, float) | None
        X-axis limits.  ``None`` → auto-scale to max common cycle.
    save_str : str
        If non-empty, figure is saved as ``{save_str}_Discharge_CE.png``.
    color_scheme : dict[str -> str] | None
        Optional mapping *cell_code* → hex colour.
    marker_cycle_labels : iterable[int]
        Cycle numbers where “C-rate” annotations are drawn (left axis only).
    """
    if not file_tuples:
        raise ValueError("file_tuples is empty")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Cycle-rate annotations (adjust as needed)
    c_rate = {2: "Form", 4: "C/10", 7: "C/8", 10: "C/4",
              13: "C/2", 16: "1C", 19: "2C"}
    annotated = set()

    cmap = plt.cm.get_cmap("tab10").resampled(len(file_tuples))

    for i, (fp, key, code) in enumerate(file_tuples):
        try:
            cycles, _, dcap, ce = process_cycle_data(fp, key, normalized)
        except Exception as e:
            print(f"[WARN] {fp} → {e}")
            continue

        color = (color_scheme.get(code)
                 if color_scheme and code in color_scheme else cmap(i))

        # left-axis: discharge capacity
        ax1.scatter(
            cycles, dcap,
            marker='o', s=40,
            facecolors='none', edgecolors=color,
            label=f"{format_key(key)} – Discharge"
        )

        # right-axis: CE
        ax2.scatter(
            cycles, ce,
            marker='D', s=35,
            facecolors=color, edgecolors=color,
            label=f"{format_key(key)} – CE"
        )

        # # annotate chosen cycles once
        # for cy in marker_cycle_labels:
        #     if cy in cycles and cy not in annotated and cy in c_rate:
        #         ax1.text(cy, dcap[cycles.index(cy)]*1.05, c_rate[cy],
        #                 ha='center', va='bottom', fontsize=9)
        #         annotated.add(cy)

    # Aesthetics ----------------------------------------------------
    ax1.set_xlabel("Cycle number")
    ax1.set_ylabel("Discharge capacity"
                   + (" (%)" if normalized else " (mAh g$^{-1}$)"))
    ax2.set_ylabel("Coulombic efficiency (%)")

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(75, 120)
    if x_bounds:
        ax1.set_xlim(*x_bounds)

    # inward ticks all around
    for ax in (ax1, ax2):
        ax.tick_params(which="both", direction="in", top=True, right=True)

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2,
               loc="upper center", bbox_to_anchor=(0.5, -0.12),
               ncol=2, fontsize='small')

    #ax1.grid(alpha=0.3)
    plt.title("Discharge capacity & CE vs. cycle")
    plt.tight_layout()

    if save_str:
        plt.savefig(f"{save_str}_Discharge_CE.png", dpi=300)
    plt.show()

# ------------------------------------------------------------------
# 2. Mean ± 1 σ across cells
# ------------------------------------------------------------------
def plot_mean_discharge_and_ce(
        file_tuples,
        normalized=False,
        x_bounds=None,
        min_cells=2,
        n_cycles=None,
        save_str=''
):
    """
    Plot mean ± 1 σ discharge capacity *and* CE across *file_tuples*.

    Notes
    -----
    * Cycles present in < *min_cells* cells are ignored.
    * If ``n_cycles`` is None, use the smallest max-cycle among cells.
    """
    if not file_tuples:
        raise ValueError("file_tuples is empty")

    # ----------------------------------------------------------------
    # Gather per-cell vectors – pad with NaN to equal length
    # ----------------------------------------------------------------
    cap_stack, ce_stack = [], []
    common_cycles = None

    for fp, key, _ in file_tuples:
        cycles, _, dcap, ce = process_cycle_data(fp, key, normalized)

        if n_cycles is None:
            n_cycles = len(cycles) if n_cycles is None else min(n_cycles, len(cycles))

        # truncate / pad
        idx = np.arange(1, n_cycles+1)
        cap_vec = np.full_like(idx, np.nan, dtype=float)
        ce_vec  = np.full_like(idx, np.nan, dtype=float)

        for c, cap, eff in zip(cycles, dcap, ce):
            if 1 <= c <= n_cycles:
                cap_vec[c-1] = cap
                ce_vec[c-1]  = eff

        cap_stack.append(cap_vec)
        ce_stack.append(ce_vec)

    cap_stack = np.vstack(cap_stack)
    ce_stack  = np.vstack(ce_stack)

    # keep only cycles with ≥ min_cells non-NaN
    valid = (np.isfinite(cap_stack).sum(axis=0) >= min_cells)
    if not valid.any():
        raise RuntimeError("No cycle has data from the required minimum number of cells.")

    cycles = np.arange(1, n_cycles+1)[valid]

    cap_mean = np.nanmean(cap_stack[:, valid], axis=0)
    cap_std  = np.nanstd(cap_stack[:, valid], axis=0)

    ce_mean  = np.nanmean(ce_stack[:, valid], axis=0)
    ce_std   = np.nanstd(ce_stack[:, valid], axis=0)

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # mean ± σ shading – left axis
    ax1.fill_between(cycles, cap_mean-cap_std, cap_mean+cap_std,
                     color='tab:blue', alpha=0.25, label='Capacity ±1σ')
    ax1.plot(cycles, cap_mean, 'o-', color='tab:blue', lw=2,
             label='Mean discharge capacity')

    # CE on right axis
    ax2.fill_between(cycles, ce_mean-ce_std, ce_mean+ce_std,
                     color='tab:red', alpha=0.25, label='CE ±1σ')
    ax2.plot(cycles, ce_mean, 'D--', color='tab:red', lw=2,
             label='Mean CE')

    # labels / limits / ticks
    ax1.set_xlabel("Cycle number")
    if x_bounds:
        ax1.set_xlim(*x_bounds)
    ax1.set_ylabel("Discharge capacity"
                   + (" (%)" if normalized else " (mAh g$^{-1}$)"))
    ax2.set_ylabel("Coulombic efficiency (%)")
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(75, 120)

    ax1.tick_params(which="both", direction="in", top=True, right=True)
    #ax2.tick_params(which="both", direction="in", top=True, right=True)

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper center", bbox_to_anchor=(0.5, -0.12),
               ncol=2, fontsize='small')

    #ax1.grid(alpha=0.3)
    plt.title(f"Mean ± 1 σ over {len(file_tuples)} cells")
    plt.tight_layout()

    if save_str:
        plt.savefig(f"{save_str}_Mean_Discharge_CE.png", dpi=300)
    plt.show()

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
        df = _extract_cycle_segment(cds, 2, segment)       # use cycle 1 for range
        v_min = min(v_min, df['Voltage (V)'].min())
        v_max = max(v_max, df['Voltage (V)'].max())

    grid = np.linspace(v_min, v_max, n_grid)
    stack = []

    for fp, key, _ in cell_tuples:
        cds, _ = process_all_cycles_for_voltage_vs_capacity(fp, key, normalized)
        # choose a representative cycle – here 2 (adjust as needed)
        df = _extract_cycle_segment(cds, 1, segment)
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

# ---------------------------------------------------------------------------
# Multi-cell comparison: same cycle, one panel
# ---------------------------------------------------------------------------
def plot_dq_dv_cells_same_cycle(cell_tuples, cycle=1, segment='charge',
                                smooth=None, normalized=False,
                                labels=None, alpha=0.85):
    """
    Overlay dQ/dV for the *same* cycle of several cells on one plot.

    Parameters
    ----------
    cell_tuples : list[(file_path, key, extra)]
        The same list you pass to plot_mean_dq_dv().
    cycle : int
        Cycle index to plot (default = 1).
    segment : {'charge', 'discharge'}
        Which half-cycle to use.
    smooth : None | 'rolling' | 'savgol'
        Pre-smoothing option (passed to compute_dq_dv).
    normalized : bool
        If True, capacity normalisation inside
        process_all_cycles_for_voltage_vs_capacity() is used.
    labels : list[str] | None
        Legend labels (must match len(cell_tuples)) – defaults to each cell’s key.
    alpha : float
        Line transparency.
    """
    if not cell_tuples:
        raise ValueError("cell_tuples is empty")

    if labels is None:
        labels = [t[1] for t in cell_tuples]
    elif len(labels) != len(cell_tuples):
        raise ValueError("labels must be same length as cell_tuples")

    plt.figure(figsize=(6, 4))
    for (file_path, key, _), label in zip(cell_tuples, labels):
        cycles_data, _ = process_all_cycles_for_voltage_vs_capacity(
            file_path, key, normalized
        )
        df_seg = _extract_cycle_segment(cycles_data, cycle, segment)
        v, y = compute_dq_dv(
            df_seg,
            'Charge Capacity (Ah)' if segment == 'charge'
            else 'Discharge Capacity (Ah)',
            'Voltage (V)',
            smooth,
        )
        plt.plot(v, y, lw=1.2, alpha=alpha, label=label)

    plt.xlabel('Voltage (V)')
    plt.ylabel('dQ/dV (Ah V⁻¹)')
    plt.title(f'dQ/dV – {segment.capitalize()} Cycle {cycle}')
    plt.legend(fontsize='x-small', ncol=3)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_dq_dv_cells_same_cycle_2(
        cell_tuples,
        cycle: int = 1,
        segment: str = "charge",
        *,
        # ───────────── preprocessing ──────────────
        normalized: bool = False,
        bin_width: float = 0.005,     # V   (pre-derivative voltage binning)
        dedup_thresh: float = 5e-4,   # V   (drop ∆V < thresh before binning)
        monotonic: bool = True,       # enforce non-decreasing capacity

        # ───────────── smoothing options ──────────
        smooth: str | None = "savgol",
        smooth_kw: dict | None = None,          # e.g. {"window": 31, "poly": 3}
        post_smooth: str | None = None,         # optional 2nd pass on dQ/dV
        post_kw: dict | None = None,            # e.g. {"window": 7}

        # ───────────── outlier handling ──────────
        clip_sigma: float | None = None,        # e.g. 3 → mask |y| > 3·σ

        # ───────────── plotting tweaks ───────────
        labels: list[str] | None = None,
        alpha: float = 0.9,
        figsize: tuple[int, int] = (6, 4),
        lw: float = 1.2,
):
    """
    Overlay dQ/dV for the same cycle of several cells on one plot, with
    extensive noise-mitigation controls.

    Parameters
    ----------
    cell_tuples : list[(file_path, key, extra)]
        Same format you pass to your other plotting utilities.
    cycle, segment, normalized
        As in the original function.
    bin_width, dedup_thresh
        Passed through to `preprocess_df` via `compute_dq_dv`.
    monotonic : bool
        If True, drops any points that break Q-monotonicity *before*
        the derivative (helps when the logger records out-of-order rows).
    smooth, smooth_kw
        Pre-derivative smoothing (Savitzky–Golay, rolling, …).
    post_smooth, post_kw
        Optional second smoothing on the *dQ/dV* trace itself.
    clip_sigma : float | None
        Masks any dQ/dV points with |y| > clip_sigma × σ(y) (σ from current
        curve).  Set to 3–5 to catch spikes without hurting real peaks.
    labels, alpha, figsize, lw
        Plot cosmetics.
    """
    if not cell_tuples:
        raise ValueError("cell_tuples is empty")

    if labels is None:
        labels = [t[1] for t in cell_tuples]
    elif len(labels) != len(cell_tuples):
        raise ValueError("labels length must match cell_tuples")

    smooth_kw = smooth_kw or {}
    post_kw = post_kw or {}

    plt.figure(figsize=figsize)

    for (file_path, key, _), label in zip(cell_tuples, labels):
        # 1.  load & slice the requested half-cycle
        cycles_data, _ = process_all_cycles_for_voltage_vs_capacity(
            file_path, key, normalized
        )
        df_seg = _extract_cycle_segment(cycles_data, cycle, segment)

        # 2.  optional monotonic filter (rare corrupted rows)
        if monotonic:
            q_col = ("Charge Capacity (Ah)" if segment == "charge"
                     else "Discharge Capacity (Ah)")
            df_seg = df_seg.sort_values("Voltage (V)")
            df_seg = df_seg[df_seg[q_col].diff().fillna(0) >= 0]

        # 3.  derivative
        v, y = compute_dq_dv_2(
            df_seg,
            q_col=("Charge Capacity (Ah)" if segment == "charge"
                   else "Discharge Capacity (Ah)"),
            v_col="Voltage (V)",
            smooth=smooth,
            smooth_kw=smooth_kw,
            post_smooth=post_smooth,
            post_kw=post_kw,
            bin_width=bin_width,
            dedup_thresh=dedup_thresh,
        )

        # 4.  optional spike clipping
        if clip_sigma is not None and y.size:
            σ = np.nanstd(y)
            bad = np.abs(y) > clip_sigma * σ
            y = y.copy()
            y[bad] = np.nan

        plt.plot(v, y, lw=lw, alpha=alpha, label=label)

    plt.xlabel("Voltage (V)")
    plt.ylabel("dQ/dV (Ah V$^{-1}$)")
    plt.title(f"dQ/dV – {segment.capitalize()} Cycle {cycle}")
    plt.grid(True)
    plt.legend(fontsize="x-small", ncol=3)
    plt.tight_layout()
    plt.show()


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
#      #get_tuples_by_cell_code(file_paths_keys, r'EL03')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'EM01')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'EN02')[0],
#      get_tuples_by_cell_code(file_paths_keys, r'EO02')[0],
#  ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)

# DTFV_set = [get_tuples_by_cell_code(file_paths_keys, r'FC03')[0],
#             get_tuples_by_cell_code(file_paths_keys, r'FD03')[0],
#         get_tuples_by_cell_code(file_paths_keys, r'FE01')[0],
#         get_tuples_by_cell_code(file_paths_keys, r'FF02')[0],
#         get_tuples_by_cell_code(file_paths_keys, r'FG03')[0],
#             get_tuples_by_cell_code(file_paths_keys, r'ES03')[0],
# ]
# DTF_set = [get_tuples_by_cell_code(file_paths_keys, r'EN02')[0],
#            get_tuples_by_cell_code(file_paths_keys, r'DU02')[0],
#         get_tuples_by_cell_code(file_paths_keys, r'EO02')[0],
#         get_tuples_by_cell_code(file_paths_keys, r'EJ03')[0],
# ]
# LPV_controls = [get_tuples_by_cell_code(file_paths_keys, r'EV03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EU03')[0],]
#
# tuple_control_gr = [get_tuples_by_cell_code(file_paths_keys, r'EV03')[0],
#     ]
# DT14_control = [get_tuples_by_cell_code(file_paths_keys, r'DP01')[0],]
#
# MF_set = [get_tuples_by_cell_code(file_paths_keys, r'EC01')[0],]
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
#
# selected_cell = get_tuples_by_cell_code(file_paths_keys, r'EC01')[0]
#selected_cell = get_tuples_by_cell_code(file_paths_keys, r'DU02')[0]
# One cell → all discharge curves smoothed with Savitzky–Golay
#plot_dq_dv_all_cycles(selected_cell, segment='discharge', smooth='savgol')

# One cell → compare cycle-1 vs cycle-3
#plot_dq_dv_difference(selected_cell, 1, 3, segment='charge', smooth='rolling')
FU_list = [
    get_tuples_by_cell_code(file_paths_keys, r'FU01')[0],
get_tuples_by_cell_code(file_paths_keys, r'FU02')[0],
get_tuples_by_cell_code(file_paths_keys, r'FU03')[0],
get_tuples_by_cell_code(file_paths_keys, r'FU04')[0],
get_tuples_by_cell_code(file_paths_keys, r'FU05')[0],
]

# LPV_List = [
#     get_tuples_by_cell_code(file_paths_keys, r'EU01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EU02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EU03')[0],
# ]

FT_List = [
    get_tuples_by_cell_code(file_paths_keys, r'FT01')[0],
    get_tuples_by_cell_code(file_paths_keys, r'FT02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'FT03')[0],
    get_tuples_by_cell_code(file_paths_keys, r'FT04')[0],
    get_tuples_by_cell_code(file_paths_keys, r'FT05')[0],
]
FR_C10 = [
    get_tuples_by_cell_code(file_paths_keys, r'FR01')[0],
    get_tuples_by_cell_code(file_paths_keys, r'FR02')[0],
]
FS_C10 = [
    #get_tuples_by_cell_code(file_paths_keys, r'FS01')[0],
    get_tuples_by_cell_code(file_paths_keys, r'FS02')[0],
]

DTFV_C10 = FT_List + FR_C10
MF91_C10 = FU_list + FS_C10

HighFi_set = [
    get_tuples_by_cell_code(file_paths_keys, r'FW02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'FX02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'FY02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'FZ02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'GA02')[0],
]
# Example: all tuples for cell-code pattern 'FU'

# 1. Overlay individual cells
# plot_discharge_and_ce_vs_cycle(
#     MF91_C10,
#     normalized=False,
#     x_bounds=(0, 11.5),          # optional
#     #save_str='FU_overlay'      # optional
# )
#
# # 2. Mean ± 1 σ across the same set
# plot_mean_discharge_and_ce(
#     MF91_C10,
#     normalized=False,
#     x_bounds=(0, 11.5),          # optional
#     n_cycles=45,               # cap at 45 cycles (optional)
#     #save_str='FU_mean'         # optional
# )

# Compare first-cycle charge dQ/dV of four cells
plot_dq_dv_cells_same_cycle(
    HighFi_set,           # your list of (path, key, extra)
    cycle=1,
    segment='charge',
    smooth='savgol',
    labels=['FW02', 'FX02', 'FY02', 'FZ02', 'GA02'],
)

plot_dq_dv_cells_same_cycle_2(
    HighFi_set,
    smooth="savgol",  smooth_kw={"window": 31},
    post_smooth="rolling", post_kw={"window": 7},
    clip_sigma=4
)
# Many cells → mean trace (+1 σ) of first-cycle charge dQ/dV
plot_mean_dq_dv(FU_list, segment='charge', smooth='savgol')


cycle_str = 'CycleLife_JoshColors'
rate_str = 'Rate_JoshColors'
rate_bounds = (0, 19.5)
cycle_life_bounds = (21.5, 45)
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
compare_cells_on_same_plot(josh_ask, normalized=False, x_bounds=cycle_life_bounds, save_str=rate_str+'JoshAsk_life', color_scheme=None)
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
#cell_codes = [cell_code for _, _, cell_code in files_to_compare]


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
    lookup_df = pd.read_excel(lookup_table_path)
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
