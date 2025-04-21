import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog
import matplotlib

# Provide the path to your lookup table Excel file.
lookup_table_path = r'C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx'
# lookup_table_path = filedialog.askopenfilename(title="Select Lookup Table")

search_directory = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\03\Cycle Life Best Survivors'

search_directory = r'C:\Users\benja\Downloads\Temp\Data_Work_4_19\Cycle Life Best Survivors'
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
        ax1.scatter(cycles, charge_caps, marker=base_marker,
                    facecolors=facecolor, edgecolors=color,
                    label=f'{key} (Charge)')
        # ax1.scatter(cycles, discharge_caps, marker=base_marker,
        #             facecolors=color, edgecolors=color,
        #             label=f'{key} (Discharge)', linestyle='--')

        ax2.scatter(cycles, ce, marker=ce_marker,
                    facecolors=facecolor, edgecolors=color,
                    label=f'{key} (CE)')

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
    plt.tight_layout()
    if save_str:
        plt.savefig(f'{save_str}_Capacity_and_Coulombic_Efficiency_vs_Cycle.png', dpi=300)
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


    color_dict = {}

    if len(cell_codes) <= len(tol_bright):
        palette = tol_bright
    else:
        palette = tol_nightfall

    for i, code in enumerate(cell_codes):
        color_dict[str(code)] = palette[i % len(palette)]

    return color_dict





# ==========================
# 12. Main Execution
# ==========================
file_paths_keys = generate_file_paths_keys(os.getcwd(), lookup_table_path)
print('Starting')
create_excel_summary(file_paths_keys, 'output_summary_2.xlsx', normalized=False)

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
        # plt.savefig(f'Capacity_vs_Cycle_{cell_code}.png')

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

DTFV_set = [get_tuples_by_cell_code(file_paths_keys, r'FC03')[0],
            get_tuples_by_cell_code(file_paths_keys, r'FD03')[0],
        get_tuples_by_cell_code(file_paths_keys, r'FE01')[0],
        get_tuples_by_cell_code(file_paths_keys, r'FF02')[0],
        get_tuples_by_cell_code(file_paths_keys, r'FG03')[0],
            get_tuples_by_cell_code(file_paths_keys, r'ES03')[0],
]
DTF_set = [get_tuples_by_cell_code(file_paths_keys, r'EN02')[0],
           get_tuples_by_cell_code(file_paths_keys, r'DU02')[0],
        get_tuples_by_cell_code(file_paths_keys, r'EO02')[0],
        get_tuples_by_cell_code(file_paths_keys, r'EJ03')[0],
]
LPV_controls = [get_tuples_by_cell_code(file_paths_keys, r'EV03')[0],
    get_tuples_by_cell_code(file_paths_keys, r'EU03')[0],]

tuple_control_gr = [get_tuples_by_cell_code(file_paths_keys, r'EV03')[0],
    ]
DT14_control = [get_tuples_by_cell_code(file_paths_keys, r'DP01')[0],]
import pprint as pp
#pp.pprint(DT14_control)


DT_Set = []
DT_Set.extend(DT14_control)
DT_Set.extend(LPV_controls)

Full_set = []
Full_set.extend(LPV_controls)
Full_set.extend(DT14_control)
Full_set.extend(DTF_set)
Full_set.extend(DTFV_set)
cell_codes= [cell_code for _, _, cell_code in Full_set]
custom_colors = assign_tol_colors(cell_codes)

compare_cells_on_same_plot(DT_Set, normalized=False, x_bounds=(19.5, 100), save_str='CycleLife_TolColors', color_scheme=custom_colors)
compare_cells_on_same_plot(DTF_set, normalized=False, x_bounds=(19.5, 100), save_str='CycleLife_TolColors', color_scheme=custom_colors)
compare_cells_on_same_plot(DTFV_set, normalized=False, x_bounds=(19.5, 100), save_str='CycleLife_TolColors', color_scheme=custom_colors)
compare_cells_on_same_plot(Full_set, normalized=False, x_bounds=(19.5, 100), save_str='CycleLife_TolColors', color_scheme=custom_colors)

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
