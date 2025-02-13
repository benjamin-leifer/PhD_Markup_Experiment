import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog

# Provide the path to your lookup table Excel file.
lookup_table_path = r'C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx'
# lookup_table_path = filedialog.askopenfilename(title="Select Lookup Table")

search_directory = r'C:\Users\benja\OneDrive\Documents\KRI Arbin 2_12_2025'

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


# ==========================
# 3. Function to process a cycling Excel file (for voltage vs capacity)
# ==========================
def process_and_plot(file_path, dataset_key, normalized=False):
    """
    Loads cycling data from an Excel file, selects the first cycle (after filtering)
    and separates charge and discharge data.
    """
    capacities = {
        'LFP': 2.0075 / 1000 / 100,  # mAh
        'NMC': 3.212 / 1000 / 100,  # mAh
        'Gr': 3.8544 / 1000 / 100  # mAh
    }
    weights_g = {
        'LFP': 7.09 / 1000 * 1.606 / 1000,  # g
        'NMC': 12.45 / 1000 * 1.606 / 1000,  # g
        'Gr': 6.61 / 1000 * 2.01 / 1000  # g
    }

    # Select normalization factor (capacity or weight) based on chemistry
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

    # Load the Excel file and select the sheet that starts with "Channel"
    data = pd.ExcelFile(file_path)
    data_sheets = [sheet for sheet in data.sheet_names if sheet.startswith('Channel')]
    if not data_sheets:
        raise ValueError(f"No sheet starting with 'Channel' found in {file_path}")
    data_sheet = data_sheets[0]
    sheet_data = data.parse(data_sheet)

    # Filter out rows where Current equals zero
    filtered_data = sheet_data[sheet_data['Current (A)'] != 0]

    # Group data by Cycle Index and select the first cycle
    grouped = list(filtered_data.groupby('Cycle Index'))
    first_cycle = grouped[0][1]

    # Remove potential transients by slicing off the first two and last two rows
    charge_group = first_cycle[first_cycle['Current (A)'] > 0].iloc[2:-2]
    discharge_group = first_cycle[first_cycle['Current (A)'] < 0].iloc[2:-2]

    return charge_group, discharge_group, norm_factor


# ==========================
# 4. Helper: Determine color based on dataset key
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
# 5. Function to generate file paths and keys from directory and lookup table
# ==========================
def extract_cell_identifier(filename):
    """
    Extract a substring that matches two letters followed by two digits (e.g., 'DB01')
    from the filename.
    """
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
            if file.endswith('.xlsx') and 'Rate_Test' in file:
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
# 6. Function to plot voltage vs capacity for a given cell identifier group
# ==========================
def plot_grouped_files(group, normalized=False):
    """
    Given a list of (file_path, key, cell_code) tuples (all sharing the same cell code),
    load the cycling data for each file and plot their charge and discharge curves.
    """
    plt.figure(figsize=(10, 6))
    for file_path, key, cell_code in group:
        try:
            charge, discharge, norm_factor = process_and_plot(file_path, key, normalized)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
        color = get_color(key)
        if not charge.empty:
            plt.plot(charge['Charge Capacity (Ah)'] / norm_factor, charge['Voltage (V)'],
                     label=f'{key} (Charge)', linestyle='-', color=color)
        if not discharge.empty:
            plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor, discharge['Voltage (V)'],
                     label=f'{key} (Discharge)', linestyle='--', color=color)
    plt.xlabel('Capacity (Ah)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Voltage vs Capacity for Cell Code {group[0][2]}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    save_name = f"{sanitize_filename(group[0][2])}_Voltage_vs_Capacity.png"
    # plt.savefig(save_name, dpi=300)
    plt.show()


# ==========================
# 7. New function: Process cycle data for capacity vs cycle plot
# ==========================
def process_cycle_data(file_path, dataset_key, normalized=False):
    """
    Process the Excel file to extract cycle-wise capacity data.
    Returns:
      cycle_numbers, charge_capacities, discharge_capacities, coulombic_efficiency
    """
    capacities = {
        'LFP': 2.0075 / 1000 / 100,  # mAh
        'NMC': 3.212 / 1000 / 100,  # mAh
        'Gr': 3.8544 / 1000 / 100  # mAh
    }
    weights_g = {
        'LFP': 7.09 / 1000 * 1.606 / 1000,  # g
        'NMC': 12.45 / 1000 * 1.606 / 1000,  # g
        'Gr': 6.61 / 1000 * 2.01 / 1000  # g
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
    filtered_data = sheet_data[sheet_data['Current (A)'] != 0]

    groups = list(filtered_data.groupby('Cycle Index'))
    cycle_numbers = []
    charge_capacities = []
    discharge_capacities = []
    coulombic_efficiency = []

    for cycle, group in groups:
        # Optionally, remove transients (e.g., group = group.iloc[2:-2] if desired)
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


# ==========================
# 8. New function: Plot capacity (mAh/g) vs Cycle Number Comparison with vertical lines
# ==========================
def plot_capacity_vs_cycle(group, normalized=False):
    """
    Given a list of (file_path, key, cell_code) tuples (all sharing the same cell code),
    this function plots capacity (mAh/g) vs cycle number with vertical dashed lines.
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for i, (file_path, key, cell_code) in enumerate(group):
        try:
            cycles, charge_caps, discharge_caps, _ = process_cycle_data(file_path, key, normalized)
        except Exception as e:
            print(f"Error processing {file_path} for cycle data: {e}")
            continue
        plt.plot(cycles, charge_caps, marker='o', linestyle='-', color=colors[i % len(colors)],
                 label=f'{key} (Charge)')
        plt.plot(cycles, discharge_caps, marker='x', linestyle='--', color=colors[i % len(colors)],
                 label=f'{key} (Discharge)')

    plt.xlabel('Cycle Number')
    if normalized:
        plt.ylabel('Capacity (%)')
        plt.ylim(0, 110)
    else:
        plt.ylabel('Capacity (mAh/g)')
        plt.ylim(0, 200)
    plt.title(f'Capacity vs Cycle Number for Cell Code {group[0][2]}')

    # Add vertical dashed lines at specified cycle numbers
    for cycle in [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5]:
        plt.axvline(x=cycle, color='black', linestyle='--')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================
# 9. Main execution
# ==========================
file_paths_keys = generate_file_paths_keys(os.getcwd(), lookup_table_path)

print("Generated file_paths_keys:")
for full_path, key, cell_code in file_paths_keys:
    print(f"File: {full_path}\nKey: {key}\nCell Code: {cell_code}\n")

if not file_paths_keys:
    print("No valid Excel files were found. Please check your directory and filtering criteria.")
else:
    # Group the files by cell code (e.g., 'DN', 'DO', etc.)
    grouped_files = {}
    for full_path, key, cell_code in file_paths_keys:
        grouped_files.setdefault(cell_code, []).append((full_path, key, cell_code))

    # For each group, first plot the voltage vs capacity curves...
    for cell_code, group in grouped_files.items():
        print(f"Plotting voltage vs capacity for {len(group)} files for cell code {cell_code}...")
        plot_grouped_files(group, normalized=False)

        # ...and then plot the capacity (mAh/g) vs cycle number comparison (with vertical lines).
        print(f"Plotting capacity vs cycle number for {len(group)} files for cell code {cell_code}...")
        plot_capacity_vs_cycle(group, normalized=False)
