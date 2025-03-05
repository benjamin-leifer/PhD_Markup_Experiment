import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Provide the path to your lookup table Excel file.
lookup_table_path = r'C:\Users\benja\OneDrive - Northeastern University\Spring 2025 Cell List.xlsx'
search_directory = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\-51C_discharges'

# ==========================
# 1. Set the working directory
# ==========================
os.chdir(search_directory)

# ==========================
# 2. Helper: Extract cell identifier
# ==========================
def extract_cell_identifier(filename):
    match = re.search(r'([A-Z]{2}\d{2})', filename)
    if match:
        return match.group(1)
    else:
        return None

# ==========================
# 3. Generate file paths and keys from directory and lookup table
# ==========================

def generate_file_paths_keys(directory, lookup_table_path):
    """
    Walk through the directory (and subdirectories) to find Excel files.
    For each file, extract the cell identifier and lookup additional details from the lookup table.
    Returns a list of tuples: (full_path, key, cell_code)
    """
    file_paths_keys = []
    lookup_df = pd.read_excel(lookup_table_path)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx') and ('-51C' in file):
                full_path = os.path.join(root, file)
                if not os.path.exists(full_path):
                    print(f"File does not exist: {full_path}")
                    continue
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
# 4. Process all cycles for Voltage vs. Capacity (for discharge curves)
# ==========================
def process_all_cycles_for_voltage_vs_capacity(file_path, dataset_key, normalized=False):
    """
    Loads cycling data from an Excel file, groups it by cycle,
    and for each cycle separates the charge and discharge data.
    Transients (first two and last two rows) are trimmed when possible.
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

    # Use weights for non-normalized data
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
    # Remove rows where Current equals zero
    filtered_data = sheet_data[sheet_data['Current (A)'] != 0]

    cycles_data = []
    # Group by 'Cycle Index'
    for cycle, group in filtered_data.groupby('Cycle Index'):
        # Trim transients if enough rows are present
        if len(group) > 4:
            charge_group = group[group['Current (A)'] > 0].iloc[2:-2]
            discharge_group = group[group['Current (A)'] < 0].iloc[2:-2]
        else:
            charge_group = group[group['Current (A)'] > 0]
            discharge_group = group[group['Current (A)'] < 0]
        cycles_data.append((cycle, charge_group, discharge_group))
    return cycles_data, norm_factor

# ==========================
# 5. Helper: Get tuples by cell code (using substring match in key)
# ==========================
def get_tuples_by_cell_code(file_paths_keys, target_cell_code):
    """
    Search the list of (full_path, key, cell_code) tuples for a given cell code.
    Returns all matching tuples.
    """
    matches = []
    for (full_path, key, cell_code) in file_paths_keys:
        if target_cell_code in key:
            matches.append((full_path, key, cell_code))
    return matches

# ==========================
# 6. New: Plot only the discharge curves for selected cells
# ==========================
def plot_last_cells_discharge_curves(file_tuples, normalized=False):
    """
    For each file in file_tuples, process the cycling data and plot the discharge curves.
    The x-axis shows capacity (Ah, normalized by the norm factor) and y-axis shows voltage (V).
    """
    plt.figure(figsize=(10, 6))
    cmap = matplotlib.colormaps["tab20"].resampled(len(file_tuples))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'x', 'd', '|', '_', '+', '1', '2', '3', '4']
    for idx, (file_path, key, cell_code) in enumerate(file_tuples):
        color = cmap(idx)
        marker = markers[idx % len(markers)]
        try:
            cycles_data, norm_factor = process_all_cycles_for_voltage_vs_capacity(file_path, key, normalized)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
        for cycle, charge, discharge in cycles_data:
            if not discharge.empty:
                plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor, discharge['Voltage (V)'],
                         label=f'{key} Cycle {cycle}', linestyle='-', color=color, marker=marker)
    plt.xlabel('Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.title('Discharge Curves for Selected Cells')
    plt.gca().set_ylim(0, 4.5)
    plt.gca().set_xlim(0, 130)
    # Only show legend if there are labeled artists
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize='small', ncol=2)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# ==========================
# 7. Main Execution
# ==========================
# Generate the list of file paths and keys from the search directory and lookup table
file_paths_keys = generate_file_paths_keys(os.getcwd(), lookup_table_path)

print("Generated file_paths_keys:")
for full_path, key, cell_code in file_paths_keys:
    print(f"File: {full_path}\nKey: {key}\nCell Code: {cell_code}\n")

# Here you can select the "last cells" (adjust the cell code substrings as needed)
# files_to_compare = [
#     get_tuples_by_cell_code(file_paths_keys, r'DN06'),
#     get_tuples_by_cell_code(file_paths_keys, r'DO06'),
#     get_tuples_by_cell_code(file_paths_keys, r'DP06'),
#     get_tuples_by_cell_code(file_paths_keys, r'DR06'),
#     get_tuples_by_cell_code(file_paths_keys, r'DS06'),
#     get_tuples_by_cell_code(file_paths_keys, r'DT06'),
#     get_tuples_by_cell_code(file_paths_keys, r'DU06'),
#     get_tuples_by_cell_code(file_paths_keys, r'DZ06'),
#     get_tuples_by_cell_code(file_paths_keys, r'EA06'),
#     get_tuples_by_cell_code(file_paths_keys, r'EB06'),
#     get_tuples_by_cell_code(file_paths_keys, r'EC06'),
# ]

files_to_compare = []
#target_codes = ['DN06', 'DO06', 'DP06', 'DR06', 'DS06', 'DT06', 'DU06','DV06','DW06','DX06', 'DY06', 'DZ06', 'EA06', 'EB06', 'EC06']
target_codes = ['DR06', 'DU06', 'EG05', 'EH05','EI04','EJ05', ]
for code in target_codes:
    matches = get_tuples_by_cell_code(file_paths_keys, code)
    if matches:
        files_to_compare.append(matches[0])
    else:
        print(f"No files found for cell code {code}")

if files_to_compare:
    plot_last_cells_discharge_curves(files_to_compare, normalized=False)
else:
    print("No matching files found to plot.")
# Plot the discharge curves for these selected cells
#plot_last_cells_discharge_curves(files_to_compare, normalized=False)
