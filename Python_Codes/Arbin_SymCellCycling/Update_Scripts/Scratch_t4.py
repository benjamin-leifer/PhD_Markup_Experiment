import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog
import matplotlib


# Provide the path to your lookup table Excel file.
lookup_table_path = r'C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx'
# lookup_table_path = filedialog.askopenfilename(title="Select Lookup Table")

search_directory = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\02'

# ==========================
# 1. Set the working directory
# ==========================
os.chdir(search_directory)


# ==========================
# 2. Helper function to sanitize filenames
# ==========================
def sanitize_filename(name):
    """Sanitize a string to create a valid filename by replacing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name


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
# 3. NEW: Process all cycles for Voltage vs. Capacity
# ==========================
def process_all_cycles_for_voltage_vs_capacity(file_path, dataset_key, normalized=False):
    """
    Loads cycling data from an Excel file, groups it by cycle,
    and for each cycle separates the charge and discharge data.
    For each cycle, transient portions are removed (first two and last two rows)
    and the normalization factor is computed based on the dataset key.
    Returns:
      cycles_data: a list of tuples (cycle_index, charge_group, discharge_group)
      norm_factor: the normalization factor (same for all cycles)
    """
    capacities = {
        'LFP': 2.0075 / 1000 / 100,
        'NMC': 3.212 / 1000 / 100,
        'Gr': 3.8544 / 1000 / 100
    }
    weights_g = {
        'LFP': 7.09 / 1000 * 1.606 / 1000,
        'NMC': 12.45 / 1000 * 1.606 / 1000,
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
        # Remove transients if possible
        if len(group) > 4:
            charge_group = group[group['Current (A)'] > 0].iloc[2:-2]
            discharge_group = group[group['Current (A)'] < 0].iloc[2:-2]
        else:
            charge_group = group[group['Current (A)'] > 0]
            discharge_group = group[group['Current (A)'] < 0]
        cycles_data.append((cycle, charge_group, discharge_group))
    return cycles_data, norm_factor


# ==========================
# 4. NEW: Process all cycles for Voltage vs. Time
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
        'NMC': 3.212 / 1000 / 100,
        'Gr': 3.8544 / 1000 / 100
    }
    weights_g = {
        'LFP': 7.09 / 1000 * 1.606 / 1000,
        'NMC': 12.45 / 1000 * 1.606 / 1000,
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
# 5. Helper: Determine color based on dataset key
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
# 6. Function to generate file paths and keys from directory and lookup table
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
            if file.endswith('.xlsx') and 'Rate_Test' or 'RateTest' in file:
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
# 7. Plot Voltage vs. Capacity for all cycles with distinct colors for each file
# ==========================
def process_cycle_data(file_path, dataset_key, normalized=False):
    """
    Process the Excel file to extract cycle-wise capacity data.
    Returns:
      cycle_numbers, charge_capacities, discharge_capacities, coulombic_efficiency
    """
    capacities = {
        'LFP': 2.0075 / 1000 / 100,
        'NMC': 3.212 / 1000 / 100,
        'Gr': 3.8544 / 1000 / 100
    }
    weights_g = {
        'LFP': 7.09 / 1000 * 1.606 / 1000,
        'NMC': 12.45 / 1000 * 1.606 / 1000,
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

    cycle_numbers = []
    charge_capacities = []
    discharge_capacities = []
    coulombic_efficiency = []

    # Group by 'Cycle Index' and process each cycle
    for cycle, group in filtered_data.groupby('Cycle Index'):
        charge_data = group[group['Current (A)'] > 0]
        discharge_data = group[group['Current (A)'] < 0]
        if not charge_data.empty and not discharge_data.empty:
            charge_cap = charge_data['Charge Capacity (Ah)'].max()
            discharge_cap = discharge_data['Discharge Capacity (Ah)'].max()
            if pd.notna(charge_cap) and pd.notna(discharge_cap) and charge_cap != 0:
                cycle_numbers.append(cycle)
                charge_capacities.append(charge_cap / norm_factor)
                discharge_capacities.append(discharge_cap / norm_factor)
                coulombic_efficiency.append((discharge_cap / charge_cap) * 100)
    return cycle_numbers, charge_capacities, discharge_capacities, coulombic_efficiency

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
            if cycle in (1,2,5,10,25,50):
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
# 8. Plot Capacity vs Cycle Number (with vertical lines)
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
    save_path = os.getcwd()+'/figures/'+f'{sanitize_filename(group[0][2])}_Capacity_vs_Cycle.png'
    plt.savefig(save_path, dpi=300,)
    plt.close()
    #plt.show()


# ==========================
# 9. Plot Voltage vs. Time for all cycles with markers and limit 5 files per cell
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


def compare_cells_on_same_plot(file_tuples, normalized=False):
    """
    Compare multiple cells on one plot:
      - Left y-axis: Capacity (mAh/g) vs. Cycle Number
      - Right y-axis: Coulombic Efficiency (%) vs. Cycle Number
      - No grid lines and vertical dashed lines at specified cycle indices
    Args:
        file_tuples (list): A list of (full_path, key, cell_code) tuples.
        normalized (bool): Whether to use normalization.
    """
    if not file_tuples:
        raise ValueError("No file tuples provided for comparison.")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Use the new colormap API and resample to the number of cells provided
    cmap = matplotlib.colormaps["tab10"].resampled(len(file_tuples))

    for i, tup in enumerate(file_tuples):
        # Ensure each tuple has exactly three elements
        if not (isinstance(tup, tuple) and len(tup) == 3):
            print(f"Skipping invalid tuple at index {i}: {tup}")
            continue

        file_path, key, cell_code = tup
        try:
            cycles, charge_caps, discharge_caps, ce = process_cycle_data(file_path, key, normalized)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        color = cmap(i)
        # Plot capacity on left y-axis
        ax1.scatter(cycles, charge_caps, marker='o', color=color,
                 label=f'{key} (Charge)')
        ax1.scatter(cycles, discharge_caps, marker='x', color=color,
                 label=f'{key} (Discharge)')
        # Plot coulombic efficiency on right y-axis
        ax2.scatter(cycles, ce, marker='d', color=color,
                 label=f'{key} (CE)')

    # Remove grid lines
    ax1.grid(False)
    ax2.grid(False)

    # Add vertical dashed lines at specified cycle numbers
    for cycle in [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5]:
        ax1.axvline(x=cycle, color='black', linestyle='--')

    ax1.set_xlabel('Cycle Number')
    ax1.set_ylim(0, 200)
    ax1.set_xlim(0, 25)
    if normalized:
        ax1.set_ylabel('Capacity (%)')
    else:
        ax1.set_ylabel('Capacity (mAh/g)')
    ax2.set_ylabel('Coulombic Efficiency (%)')
    ax2.set_ylim(0, 120)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize='small')

    plt.title('Capacity and Coulombic Efficiency vs. Cycle Number')
    plt.tight_layout()
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
        if  target_cell_code in key:
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



# ==========================
# 10. Main Execution
# ==========================
file_paths_keys = generate_file_paths_keys(os.getcwd(), lookup_table_path)

print("Generated file_paths_keys:")
for full_path, key, cell_code in file_paths_keys:
    print(f"File: {full_path}\nKey: {key}\nCell Code: {cell_code}\n")

gitt_file_paths_keys = generate_gitt_file_paths_keys(os.getcwd(), lookup_table_path)
for full_path, key, cell_code in gitt_file_paths_keys:
    print(f"File: {full_path}\nKey: {key}\nCell Code: {cell_code}\n")
    # Then call your GITT-specific plotting function, for example:
    plot_gitt_file(full_path, key, normalized=False)

if not file_paths_keys:
    print("No valid Excel files were found. Please check your directory and filtering criteria.")
else:
    # Group the files by cell code (e.g., 'DN', 'DO', etc.)
    grouped_files = {}
    for full_path, key, cell_code in file_paths_keys:
        grouped_files.setdefault(cell_code, []).append((full_path, key, cell_code))

    # For each cell group, generate the three plots.
    for cell_code, group in grouped_files.items():
        # print(f"Plotting Voltage vs Capacity for {len(group)} files for cell code {cell_code} (all cycles)...")
        # plot_grouped_files(group, normalized=False)

        print(f"Plotting Capacity vs Cycle Number for {len(group)} files for cell code {cell_code}...")
        #plot_capacity_vs_cycle(group, normalized=False)
        #plt.savefig(f'Capacity_vs_Cycle_{cell_code}.png')

        # print(
        #     f"Plotting Voltage vs Time for up to {min(len(group), 5)} files for cell code {cell_code} (all cycles)...")
        # plot_voltage_vs_time(group, normalized=False)
#filtered_tuples = get_tuples_by_cell_code(file_paths_keys, r'DQ01')
#print("Filtered tuples:", filtered_tuples)
#compare_cells_on_same_plot(filtered_tuples, normalized=False)

# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DN02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DO03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DP01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DQ01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DD03')[0],
# ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)

files_to_compare = [
    get_tuples_by_cell_code(file_paths_keys, r'DR02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'DS03')[0],
    get_tuples_by_cell_code(file_paths_keys, r'DT03')[0],
    get_tuples_by_cell_code(file_paths_keys, r'DU02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'EG02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'EH02')[0],
    get_tuples_by_cell_code(file_paths_keys, r'EI03')[0],
    get_tuples_by_cell_code(file_paths_keys, r'EJ03')[0],
]
compare_cells_on_same_plot(files_to_compare, normalized=False)

# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DZ03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EA02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EB03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EC01')[0],
# ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)
#
# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DN01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DR02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DZ03')[0],
# ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)
#
# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DO03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DS03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EA02')[0],
# ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)
#
# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DP01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DT03')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EB03')[0],
# ]
# compare_cells_on_same_plot(files_to_compare, normalized=False)
#
# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DQ01')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'DU02')[0],
#     get_tuples_by_cell_code(file_paths_keys, r'EC01')[0],
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
#compare_cells_on_same_plot(files_to_compare, normalized=False)

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

