import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # Or try 'QtAgg' if 'TkAgg' doesn't work
import matplotlib.pyplot as plt
plt.rcParams['toolbar'] = 'toolbar2'  # Enable the toolbar for interactive plots

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
            if file.endswith('.xlsx'):
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
        'Gr': 3.8544 / 1000 / 100,
        'NEI-16mm': 4.02 / 1000 / 100
    }
    weights_g = {
        'LFP': 7.09 / 1000 * 1.606 / 1000,
        'NMC': 12.45 / 1000 * 1.606 / 1000,
        'Gr': 6.61 / 1000 * 2.01 / 1000,
        'NEI-16mm': 12.45 / 1000 * 2.01 / 1000,
    }

    # Use weights for non-normalized data
    if normalized:
        if 'LFP' in dataset_key:
            norm_factor = capacities['LFP']
        elif 'NEI-16mm' in dataset_key:
            norm_factor = capacities['NEI-16mm']
        elif 'NMC' in dataset_key:
            norm_factor = capacities['NMC']
        elif 'Gr' in dataset_key:
            norm_factor = capacities['Gr']
        else:
            raise ValueError("Dataset key does not match known capacities")
    else:
        if 'LFP' in dataset_key:
            norm_factor = weights_g['LFP']
        elif 'NEI-16mm' in dataset_key:
            norm_factor = capacities['NEI-16mm']  # using capacity normalization for NEI-16mm
            print("Using NEI-16mm capacity for normalization")
        elif 'NMC' in dataset_key:
            norm_factor = weights_g['NMC']
            print("Using NMC capacity for normalization")
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
    for cycle, group in filtered_data.groupby('Cycle Index'):
        if len(group) > 4:
            charge_group = group[group['Current (A)'] > 0].iloc[2:-2]
            discharge_group = group[group['Current (A)'] < 0].iloc[2:-2]
        else:
            charge_group = group[group['Current (A)'] > 0]
            discharge_group = group[group['Current (A)'] < 0]
        cycles_data.append((cycle, charge_group, discharge_group))
        print(f'Cell code: {dataset_key}')
        print('Norm Factor is: ' + str(norm_factor))
        print('max discharge capacity is: ' + str(discharge_group['Discharge Capacity (Ah)'].max()))
        print('specific discharge capacity is: ' + str(discharge_group['Discharge Capacity (Ah)'].max() / norm_factor))
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

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

# Example function to add a snowflake image to the plot
def add_snowflake_to_plot(ax, image_path, x, y, zoom=0.1):
    snowflake_img = mpimg.imread(image_path)
    imagebox = OffsetImage(snowflake_img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False, box_alignment=(0.5, 0.5))
    ax.add_artist(ab)

# Add a snowflake image to the plot
snowflake_image_path = r'C:\Users\benja\Downloads\Temp\Data_Work_4_19\Snowflake.png'

def extract_temperature_from_filename(filename):
    match = re.search(r'-(\d{2})C', filename)
    if match:
        return f"-{match.group(1)}°C"
    else:
        return "RT"

def format_key(key):
    """
    Remove (NEI-16mm) from the key and format it for display.
    Remove cell code from end of the key.
    """
    if '(NEI-16mm)' in key:
        key = key.replace('(NEI-16mm)', '')
    key = key[:-6]  # remove "(XX##)"
    return key.strip()

# ==========================
# 6. Plot only the discharge curves for selected cells
# ==========================
def plot_last_cells_discharge_curves(file_tuples, normalized=False, color_dict=None):
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    cmap = matplotlib.colormaps["tab20"].resampled(len(file_tuples))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'x', 'd', '|', '_', '+', '1', '2', '3', '4']

    for idx, (file_path, key, cell_code) in enumerate(file_tuples):
        print(key)
        print(key[-5:-1])
        if color_dict is not None:
            color = color_dict.get(str(key[-5:-1]), cmap(idx))
        else:
            color = cmap(idx)
        marker = markers[idx % len(markers)]
        print(color)
        try:
            cycles_data, norm_factor = process_all_cycles_for_voltage_vs_capacity(file_path, key, normalized)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        for cycle, charge, discharge in cycles_data:
            if discharge.empty:
                continue
            # cut off Voltage below 2.5 V
            discharge = discharge[discharge['Voltage (V)'] > 1.5]

            # decide label text (add temperature when helpful)
            filename = os.path.basename(file_path)
            temperature = extract_temperature_from_filename(filename)
            base_label = format_key(key)
            label_text = f"{base_label} ({temperature})"

            # NEI-16mm capacity-normalized branch (norm_factor ≈ 4.02e-5)
            if norm_factor > 4e-5:
                if 'DT14' in key:
                    plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor * 1.6,
                             discharge['Voltage (V)'], label=base_label, linestyle='-',
                             color='0.5', lw=3)
                elif 'DTF14' in key:
                    plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor * 1.6,
                             discharge['Voltage (V)'], label=base_label, linestyle='-',
                             color=color, lw=1)
                elif ('DTFV' in key) or ('TPTFV' in key):  # <-- PATCH: include TPTFV
                    plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor * 1.6,
                             discharge['Voltage (V)'], label=label_text, linestyle='-',
                             color=color, lw=2)
                elif 'MF91' in key:
                    plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor * 1.6,
                             discharge['Voltage (V)'], label=label_text, linestyle='-',
                             color=color, lw=2)
                elif 'TPTF' in key:  # <-- PATCH: include TPTF (non-V variant)
                    plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor * 1.6,
                             discharge['Voltage (V)'], label=label_text, linestyle='-',
                             color=color, lw=2)
                else:  # safe fallback so nothing gets skipped
                    plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor * 1.6,
                             discharge['Voltage (V)'], label=label_text, linestyle='-',
                             color=color, lw=2)
            else:
                if 'DT14' in key:
                    plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor,
                             discharge['Voltage (V)'], label=base_label, linestyle='-',
                             color='0.5', lw=3)
                elif 'DTF14' in key:
                    plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor,
                             discharge['Voltage (V)'], label=base_label, linestyle='-',
                             color=color, lw=1)
                elif ('DTFV' in key) or ('TPTFV' in key):  # <-- PATCH mirrored
                    plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor,
                             discharge['Voltage (V)'], label=label_text, linestyle='-',
                             color=color, lw=2)
                elif 'MF91' in key:
                    plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor,
                             discharge['Voltage (V)'], label=label_text, linestyle='-',
                             color=color, lw=2)
                elif 'TPTF' in key:  # <-- PATCH mirrored
                    plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor,
                             discharge['Voltage (V)'], label=label_text, linestyle='-',
                             color=color, lw=2)
                else:  # safe fallback
                    plt.plot(discharge['Discharge Capacity (Ah)'] / norm_factor,
                             discharge['Voltage (V)'], label=label_text, linestyle='-',
                             color=color, lw=2)

    plt.xlabel('Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.title('Low-Temp Discharge Curves')  # generic title so new electrolytes fit
    plt.gca().set_ylim(0, 4.5)
    plt.gca().set_xlim(-4, 160)

    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize='xx-small', ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.05))

    try:
        add_snowflake_to_plot(ax, snowflake_image_path, x=80, y=4, zoom=0.01)
    except Exception as e:
        print(f"Snowflake overlay skipped: {e}")

    plt.grid(False)
    plt.tick_params(which='both', axis='both', direction='in', bottom=True, left=True,
                    labelbottom=True, labelleft=True)
    plt.tight_layout()
    plt.savefig('t1.png', dpi=300)
    plt.show()

def assign_tol_colors(cell_codes):
    """
    Assign Paul Tol's color palette (replaced with your fixed palette).
    Returns a dict mapping cell_code → hex color.
    """
    josh_colors = [
        '#000000', '#8A2BE2', '#1e90ff', '#32CD32', '#FFD700', '#DC143C'
    ]
    color_dict = {}
    for i, code in enumerate(cell_codes):
        color_dict[str(code)] = josh_colors[i % len(josh_colors)]
    return color_dict

# ==========================
# 7. Main Execution
# ==========================
file_paths_keys = generate_file_paths_keys(os.getcwd(), lookup_table_path)

print("Generated file_paths_keys:")
for full_path, key, cell_code in file_paths_keys:
    print(f"File: {full_path}\nKey: {key}\nCell Code: {cell_code}\n")

# Choose the targets (includes HI01 / HK02 / HJ02)
target_codes = [ 'FA01',#DT14 Li
                 #'EN04', #DTF14-1 Gr
                 'DU06', #DTF14-2 Gr
                 #'EO05', #DTF14-5 Gr
                 #'EJ05', #DTF14-10 Gr
                 'FC04', #DTFV1411 Gr
                 #'FD04', #DTFV1412 Gr
                 'FE04', #DTFV1421 Gr
                 #'FF05', #DTFV1422 Gr
                 #'FG05', #DTFV1452 Gr
                 #'ES05', #DTFV14102 Gr
                 'EC06', #MF91
                 'HI01', #PTF142
                 'HJ02', #PTFV1411
                 'HK02', #PTFV1421


                 ]

cell_codes = [code for code in target_codes]
custom_colors = assign_tol_colors(cell_codes)

files_to_compare = []
# Preview max discharge per selection (kept from your script)
for code in target_codes:
    matches = get_tuples_by_cell_code(file_paths_keys, code)
    if matches:
        file_path, key, cell_code = matches[0]
        try:
            cycles_data, norm_factor = process_all_cycles_for_voltage_vs_capacity(file_path, key, normalized=False)
            max_discharge_capacity = max(
                discharge[discharge['Voltage (V)'] > 2]['Discharge Capacity (Ah)'].max() / norm_factor * 160.64 / 100
                for _, _, discharge in cycles_data if not discharge.empty
            )
            print(f"Cell Code: {cell_code}, Max Discharge Capacity: {max_discharge_capacity:.4f} mAh/g")
        except Exception as e:
            print(f"Error processing {file_path} for cell code {cell_code}: {e}")
    else:
        print(f"No files found for cell code {code}")

# Build plotting list
for code in target_codes:
    matches = get_tuples_by_cell_code(file_paths_keys, code)
    if matches:
        files_to_compare.append(matches[0])
    else:
        print(f"No files found for cell code {code}")

if files_to_compare:
    plot_last_cells_discharge_curves(files_to_compare, normalized=False, color_dict=custom_colors)
else:
    print("No matching files found to plot.")
