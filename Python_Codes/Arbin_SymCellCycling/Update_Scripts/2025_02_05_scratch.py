import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\01BL-LL-Ds Combined')

def sanitize_filename(name):
    """Sanitize a string to create a valid filename by replacing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name

def process_and_plot(file_path, dataset_key, normalized=False):
    # Define specific capacities
    capacities = {
        'LFP': 2.0075/1000/100,  # mAh
        'NMC': 3.212/1000/100,  # mAh
        'Gr': 3.8544/1000/100  # mAh
    }
    weights_g = {
        'LFP': 7.09/1000*1.606/1000,  # g
        'NMC': 12.45/1000*1.606/1000,  # g
        'Gr': 6.61/1000*2.01/1000  # g
    }

    if normalized:
        if 'LFP' in dataset_key:
            norm_capacity = capacities['LFP']
        elif 'NMC' in dataset_key:
            norm_capacity = capacities['NMC']
        elif 'Gr' in dataset_key:
            norm_capacity = capacities['Gr']
        else:
            raise ValueError("Dataset key does not match known capacities")
    else:
        if 'LFP' in dataset_key:
            norm_capacity = weights_g['LFP']
        elif 'NMC' in dataset_key:
            norm_capacity = weights_g['NMC']
        elif 'Gr' in dataset_key:
            norm_capacity = weights_g['Gr']
        else:
            raise ValueError("Dataset key does not match known capacities")

    # Load the Excel file
    data = pd.ExcelFile(file_path)

    # Identify the relevant data sheet dynamically
    data_sheet = [sheet for sheet in data.sheet_names if sheet.startswith('Channel')][0]
    sheet_data = data.parse(data_sheet)

    # Filter out data where Current = 0
    filtered_data = sheet_data[sheet_data['Current (A)'] != 0]

    # Group data by Cycle Index
    grouped = list(filtered_data.groupby('Cycle Index'))[:-1]  # Drop the last cycle

    # Collect data for plotting
    cycle_numbers = []
    charge_capacities = []
    discharge_capacities = []
    coulombic_efficiency = []

    for cycle, group in grouped:
        charge_capacity = group[group['Current (A)'] > 0]['Charge Capacity (Ah)'].max()
        discharge_capacity = group[group['Current (A)'] < 0]['Discharge Capacity (Ah)'].max()

        if charge_capacity and discharge_capacity:
            cycle_numbers.append(cycle)
            charge_capacities.append(charge_capacity / norm_capacity)
            discharge_capacities.append(discharge_capacity / norm_capacity)
            coulombic_efficiency.append((discharge_capacity / charge_capacity) * 100)

    return cycle_numbers, charge_capacities, discharge_capacities, coulombic_efficiency

def plot_multiple_files(file_paths_keys, normalized=False):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors  # Use a colormap for different colors

    for i, (file_path, dataset_key) in enumerate(file_paths_keys):
        cycle_numbers, charge_capacities, discharge_capacities, coulombic_efficiency = process_and_plot(file_path, dataset_key, normalized)

        ax1.scatter(cycle_numbers, charge_capacities, label=f'{dataset_key} (Charge)', marker='o', color=colors[i % len(colors)])
        ax1.scatter(cycle_numbers, discharge_capacities, label=f'{dataset_key} (Discharge)', marker='x', color=colors[i % len(colors)])

    ax1.set_xlabel('Cycle Number')
    if normalized:
        ax1.set_ylabel('Capacity (%)')
        ax1.set_ylim(0, 110)
    else:
        ax1.set_ylabel('Capacity (mAh/g)')
        ax1.set_ylim(0, 200)
    #ax1.legend()
    #ax1.grid()


    ax2 = ax1.twinx()
    for i, (file_path, dataset_key) in enumerate(file_paths_keys):
        cycle_numbers, charge_capacities, discharge_capacities, coulombic_efficiency = process_and_plot(file_path, dataset_key, normalized)
        ax2.scatter(cycle_numbers, coulombic_efficiency, label=f'{dataset_key} (Coulombic Efficiency)', marker='d', color=colors[(i + len(file_paths_keys)) % len(colors)])

    ax2.set_ylabel('Coulombic Efficiency (%)')
    ax2.set_ylim(0, 120)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower left', bbox_to_anchor=(0.05, 0.075), ncol=1, frameon=False)

    fig.tight_layout()

    plt.show()

# List of filenames and dataset keys
file_paths_keys = [
    ('BL-LL-DB01_RT_RateTest_Channel_46_Wb_1.xlsx', 'Li|LFP - LPV Elyte (DB01)'),
    ('BL-LL-DE02_RT_RateTest_Channel_61_Wb_1.xlsx', 'Gr|LFP - LPV Elyte (DE02)'),
]

# Plot multiple files
#plot_multiple_files(file_paths_keys, normalized=False)

file_paths_keys = [
    ('BL-LL-DA02_RT_RateTest_Channel_44_Wb_1.xlsx', 'Li|NMC - LPV Elyte (DA02)'),
    ('BL-LL-DD03_RT_RateTest_Channel_59_Wb_1.xlsx', 'Gr|NMC - LPV Elyte (DD03)'),
]

# Plot multiple files
#plot_multiple_files(file_paths_keys, normalized=False)

file_paths_keys = [
    ('BL_LL_DJ02_RT_Rate_Test_Channel_44_Wb_1.xlsx', 'Gr|NMC - DTV14 Elyte (DJ02)'),
    ('BL-LL-DD03_RT_RateTest_Channel_59_Wb_1.xlsx', 'Gr|NMC - LPV Elyte (DD03)'),
]

# Plot multiple files
plot_multiple_files(file_paths_keys, normalized=False)