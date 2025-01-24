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

    if normalized:# Determine normalization capacity based on dataset key
        if 'LFP' in dataset_key:
            norm_capacity = capacities['LFP']
        elif 'NMC' in dataset_key:
            norm_capacity = capacities['NMC']
        elif 'Gr' in dataset_key:
            if 'LFP' in dataset_key:
                norm_capacity = capacities['LFP']
            elif 'NMC' in dataset_key:
                norm_capacity = capacities['NMC']
            else:
                norm_capacity = capacities['Gr']  # Default fallback
        else:
            raise ValueError("Dataset key does not match known capacities")
    else:
        if 'LFP' in dataset_key:
            norm_capacity = weights_g['LFP']
        elif 'NMC' in dataset_key:
            norm_capacity = weights_g['NMC']
        elif 'Gr' in dataset_key:
            if 'LFP' in dataset_key:
                norm_capacity = weights_g['LFP']
            elif 'NMC' in dataset_key:
                norm_capacity = weights_g['NMC']
            else:
                norm_capacity = weights_g['Gr']  # Default fallback
        else:
            raise ValueError("Dataset key does not match known capacities")

    if 'LFP' in dataset_key:
        c_norm_capacity = capacities['LFP']
    elif 'NMC' in dataset_key:
        c_norm_capacity = capacities['NMC']
    elif 'Gr' in dataset_key:
        if 'LFP' in dataset_key:
            c_norm_capacity = capacities['LFP']
        elif 'NMC' in dataset_key:
            c_norm_capacity = capacities['NMC']
        else:
            c_norm_capacity = capacities['Gr']  # Default fallback
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

    # Plot percent capacity vs cycle number with C-rate and coulombic efficiency
    cycle_numbers = []
    charge_capacities = []
    discharge_capacities = []
    c_rates = []
    coulombic_efficiency = []

    for cycle, group in grouped:
        charge_capacity = group[group['Current (A)'] > 0]['Charge Capacity (Ah)'].max()
        discharge_capacity = group[group['Current (A)'] < 0]['Discharge Capacity (Ah)'].max()
        valid_currents = group['Current (A)'].abs().dropna()
        current = valid_currents.mean() if not valid_currents.empty else 0  # Avoid NaN values

        if charge_capacity and discharge_capacity:
            cycle_numbers.append(cycle)
            charge_capacities.append(charge_capacity / norm_capacity )
            discharge_capacities.append(discharge_capacity / norm_capacity )
            if current > 0:
                rounded_c_rate = round(current / c_norm_capacity / 100, 2)
                if rounded_c_rate >= 1:
                    c_rate = f'{int(rounded_c_rate)}C'
                elif rounded_c_rate >= 0.48:
                    c_rate = 'C/2'
                elif rounded_c_rate >= 0.23:
                    c_rate = 'C/4'
                elif rounded_c_rate >= 0.12:
                    c_rate = 'C/8'
                else:
                    c_rate = 'C/10'
            else:
                c_rate = 'C/∞'
            c_rates.append(c_rate)
            coulombic_efficiency.append((discharge_capacity / charge_capacity) * 100)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.scatter(cycle_numbers, charge_capacities, label='Charge Capacity (%)', marker='o')
    ax1.scatter(cycle_numbers, discharge_capacities, label='Discharge Capacity (%)', marker='x')
    ax1.set_xlabel('Cycle Number')
    if normalized:
        ax1.set_ylabel('Capacity (%)')
    else:
        ax1.set_ylabel('Capacity (mAh/g)')
    ax1.legend()
    if normalized:
        ax1.set_ylim(0, 120)
    else:
        ax1.set_ylim(0, 220)
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.scatter(cycle_numbers, coulombic_efficiency, label='Coulombic Efficiency (%)', marker='d', color='purple')
    ax2.set_ylabel('Coulombic Efficiency (%)')
    ax2.set_ylim(0, 120)

    # Add vertical dashed lines and C-rate text
    last_seen_c_rate = None
    plot_height = ax1.get_ylim()[1]
    for i, (cycle, c_rate) in enumerate(zip(cycle_numbers, c_rates)):
        if c_rate != last_seen_c_rate and last_seen_c_rate is not None:
            ax1.axvline(x=cycle - 0.5, color='black', linestyle='--')  # Add vertical dashed line
            #ax1.text(cycle - 1, plot_height / 2, last_seen_c_rate, fontsize=14, ha='right', va='center', color='red')
        last_seen_c_rate = c_rate

        # Final C-rate annotation
        if last_seen_c_rate:
            #ax1.text(max(cycle_numbers[-1] + 0.5, cycle_numbers[-1] + 1), plot_height / 2, last_seen_c_rate, fontsize=14, ha='left', va='center', color='red')
            print(f"Cycle {cycle_numbers[-1]}: {last_seen_c_rate}")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
               frameon=False)

    plt.title(f'Percent Capacity and Coulombic Efficiency vs Cycle Number for {dataset_key}')
    plt.tight_layout()
    plt.savefig(f'{sanitize_filename(dataset_key)}_Capacity_and_Efficiency_vs_Cycle.png', dpi=300)
    plt.show()

    # Select cycles to plot
    index = len(grouped) // 2
    if len(grouped) > 5:
        selected_cycles = [grouped[0], grouped[1], grouped[4], grouped[index], grouped[-1]]
    else:
        selected_cycles = grouped

    # Plot voltage versus capacity for selected cycles
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors  # Use a colormap for different colors

    for i, (cycle, group) in enumerate(selected_cycles):
        # Separate charge and discharge data based on Current (A)
        charge_group = group[(group['Current (A)'] > 0)].iloc[2:-2]
        discharge_group = group[(group['Current (A)'] < 0)].iloc[2:-2]

        color = colors[i % len(colors)]

        if not charge_group.empty:
            plt.plot(charge_group['Charge Capacity (Ah)'] / norm_capacity , charge_group['Voltage (V)'],
                     label=f'Cycle {cycle} (Charge) - {c_rates[i]}', linestyle='-', color=color)
        if not discharge_group.empty:
            plt.plot(discharge_group['Discharge Capacity (Ah)'] / norm_capacity , discharge_group['Voltage (V)'],
                     label=f'Cycle {cycle} (Discharge) - {c_rates[i]}', linestyle='--', color=color)

    if normalized:
        plt.xlabel('Normalized Capacity (%)')
    else:
        plt.xlabel('Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Voltage vs Normalized Capacity for {dataset_key}')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=3, frameon=False)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{sanitize_filename(dataset_key)}_Voltage_vs_Capacity.png', dpi=300)
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

# Process and plot each file with its corresponding key
for file_name, dataset_key in file_paths_keys:
    process_and_plot(file_name, dataset_key, )
