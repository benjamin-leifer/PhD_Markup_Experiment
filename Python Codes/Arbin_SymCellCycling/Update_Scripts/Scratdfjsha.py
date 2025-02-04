import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\01\BL-LL-Ds Combined')

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
    grouped = list(filtered_data.groupby('Cycle Index'))

    # Select the first cycle
    first_cycle = grouped[0][1]

    # Separate charge and discharge data based on Current (A)
    charge_group = first_cycle[(first_cycle['Current (A)'] > 0)].iloc[2:-2]
    discharge_group = first_cycle[(first_cycle['Current (A)'] < 0)].iloc[2:-2]

    return charge_group, discharge_group

def plot_comparison(file_paths_keys, comparison_pairs, normalized=False):
    # Define color map for each cell type
    color_map = {
        'LFP': 'blue',
        'NMC': 'green',
        'Gr': 'red'
    }

    for (file1, key1), (file2, key2) in comparison_pairs:
        charge1, discharge1 = process_and_plot(file1, key1, normalized)
        charge2, discharge2 = process_and_plot(file2, key2, normalized)

        plt.figure(figsize=(10, 6))
        color1 = color_map['Gr'] if 'Gr' in key1 else (color_map['LFP'] if 'LFP' in key1 else color_map['NMC'])
        color2 = color_map['Gr'] if 'Gr' in key2 else (color_map['LFP'] if 'LFP' in key2 else color_map['NMC'])

        if not charge1.empty:
            plt.plot(charge1['Charge Capacity (Ah)'], charge1['Voltage (V)'], label=f'{key1} (Charge)', linestyle='-', color=color1)
        if not discharge1.empty:
            plt.plot(discharge1['Discharge Capacity (Ah)'], discharge1['Voltage (V)'], label=f'{key1} (Discharge)', linestyle='--', color=color1)
        if not charge2.empty:
            plt.plot(charge2['Charge Capacity (Ah)'], charge2['Voltage (V)'], label=f'{key2} (Charge)', linestyle='-', color=color2)
        if not discharge2.empty:
            plt.plot(discharge2['Discharge Capacity (Ah)'], discharge2['Voltage (V)'], label=f'{key2} (Discharge)', linestyle='--', color=color2)

        plt.xlabel('Capacity (Ah)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Voltage vs Capacity for {key1} and {key2}')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{sanitize_filename(key1)}_vs_{sanitize_filename(key2)}_Voltage_vs_Capacity.png', dpi=300)
        plt.show()

# List of filenames and dataset keys
file_paths_keys = [
    ('BL-LL-DB01_RT_RateTest_Channel_46_Wb_1.xlsx', 'Li|LFP - LPV Elyte (DB01)'),
    ('BL-LL-DB02_RT_RateTest_Channel_47_Wb_1.xlsx', 'Li|LFP - LPV Elyte (DB02)'),
    ('BL-LL-DB03_RT_RateTest_Channel_48_Wb_1.xlsx', 'Li|LFP - LPV Elyte (DB03)'),
    ('BL-LL-DC01_RT_RateTest_Channel_49_Wb_1.xlsx', 'Li|Gr - LPV Elyte (DC01)'),
    ('BL-LL-DC02_RT_RateTest_Channel_50_Wb_1.xlsx', 'Li|Gr - LPV Elyte (DC02)'),
    ('BL-LL-DC03_RT_RateTest_Channel_51_Wb_1.xlsx', 'Li|Gr - LPV Elyte (DC03)'),
    ('BL-LL-DA01_RT_RateTest_Channel_43_Wb_1.xlsx', 'Li|NMC - LPV Elyte (DA01)'),
    ('BL-LL-DA02_RT_RateTest_Channel_44_Wb_1.xlsx', 'Li|NMC - LPV Elyte (DA02)'),
    ('BL-LL-DA03_RT_RateTest_Channel_45_Wb_1.xlsx', 'Li|NMC - LPV Elyte (DA03)'),
    ('BL-LL-DD01_RT_RateTest_Channel_57_Wb_1.xlsx', 'Gr|NMC - LPV Elyte (DD01)'),
    ('BL-LL-DD02_RT_RateTest_Channel_58_Wb_1.xlsx', 'Gr|NMC - LPV Elyte (DD02)'),
    ('BL-LL-DD03_RT_RateTest_Channel_59_Wb_1.xlsx', 'Gr|NMC - LPV Elyte (DD03)'),
    ('BL-LL-DE01_RT_RateTest_Channel_60_Wb_1.xlsx', 'Gr|LFP - LPV Elyte (DE01)'),
    ('BL-LL-DE02_RT_RateTest_Channel_61_Wb_1.xlsx', 'Gr|LFP - LPV Elyte (DE02)'),
]

# Define comparison pairs
comparison_pairs = [
    (('BL-LL-DB02_RT_RateTest_Channel_47_Wb_1.xlsx', 'Li|LFP - LPV Elyte (DB02)'), ('BL-LL-DE02_RT_RateTest_Channel_61_Wb_1.xlsx', 'Gr|LFP - LPV Elyte (DE02)')),
    (('BL-LL-DA02_RT_RateTest_Channel_44_Wb_1.xlsx', 'Li|NMC - LPV Elyte (DA02)'), ('BL-LL-DD03_RT_RateTest_Channel_59_Wb_1.xlsx', 'Gr|NMC - LPV Elyte (DD03)'))
]

# Plot comparisons
plot_comparison(file_paths_keys, comparison_pairs)