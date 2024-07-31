import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the new working directory path
new_dir_path = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\07\Update_0731\All'

# Change the current working directory
os.chdir(new_dir_path)

# Verify the current working directory
print("Current Working Directory: ", os.getcwd())

# Active mass in grams
active_mass_g = 0.01293303225

# Define file paths for the newly uploaded data
file_paths_new = [
    'BL-LL-AW02_-21C_t1_Channel_37_Wb_1.xlsx',
    'BL-LL-AW02_RT_t1_Channel_44_Wb_1.xlsx',
    'BL-LL-AX02_-21C_t1_Channel_38_Wb_1.xlsx',
    'BL-LL-AX02_RT_t1_Channel_46_Wb_1.xlsx',
    'BL-LL-AY02_-21C_t1_Channel_39_Wb_1.xlsx',
    'BL-LL-AY02_RT_t1_Channel_48_Wb_1.xlsx'
]

# Define legends for the newly uploaded data
legends_new = {
    'BL-LL-AW02_-21C_t1_Channel_37_Wb_1.xlsx': 'Li||NMC, DT14, -21C',
    'BL-LL-AW02_RT_t1_Channel_44_Wb_1.xlsx': 'Li||NMC, DT14, RT',
    'BL-LL-AX02_-21C_t1_Channel_38_Wb_1.xlsx': 'Li||NMC, DTF14, -21C',
    'BL-LL-AX02_RT_t1_Channel_46_Wb_1.xlsx': 'Li||NMC, DTF14, RT',
    'BL-LL-AY02_-21C_t1_Channel_39_Wb_1.xlsx': 'Gr||NMC, DTF14, -21C',
    'BL-LL-AY02_RT_t1_Channel_48_Wb_1.xlsx': 'Gr||NMC, DTF14, RT'
}

# Define function to process and plot all cycles data
def plot_all_cycles(dataframes, active_mass_g):
    fig, ax1 = plt.subplots(1,1,figsize=(4.6, 3.5))

    colors = ['b', 'b', 'g', 'g', 'r', 'r',]

    for idx, (file_path, df) in enumerate(dataframes.items()):
        if idx >= len(colors):
            print(f"Warning: Not enough colors defined for file {file_path}. Skipping this file.")
            continue
        if 'RT' in file_path:
            cycle_data = df[df['Cycle Index'] == 2]
            line_style = '-'
        else:
            cycle_data = df
            line_style = '--'

        # Calculate capacity in mAh/g
        cycle_data['Charge Capacity (mAh/g)'] = (cycle_data['Charge Capacity (Ah)'] * 1000) / active_mass_g
        cycle_data['Discharge Capacity (mAh/g)'] = (cycle_data['Discharge Capacity (Ah)'] * 1000) / active_mass_g

        # Filter data for all cycles
        charge_data = cycle_data[cycle_data['Current (A)'] > 0]
        discharge_data = cycle_data[cycle_data['Current (A)'] < 0]

        # Plot charge and discharge data with same color
        plt.plot(discharge_data['Discharge Capacity (mAh/g)'], discharge_data['Voltage (V)'],
                 label=f'{legends_new[file_path]}', linestyle=line_style, color=colors[idx])

        # Calculate and print the maximum charge and discharge capacities
        max_charge_capacity = charge_data['Charge Capacity (mAh/g)'].max()
        max_discharge_capacity = discharge_data['Discharge Capacity (mAh/g)'].max()
        print(f"Label: {legends_new[file_path]}")
        print(f"Maximum Charge Capacity (mAh/g): {max_charge_capacity}")
        print(f"Maximum Discharge Capacity (mAh/g): {max_discharge_capacity}")
        print()

    # Configure plot
    plt.xlabel('Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage vs. Discharge Capacity (mAh/g)')
    plt.legend()
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
    ax1.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='minor', direction='in', left=True, right=True, length=3)
    plt.show()

# Load the second sheet data from all files into a dictionary
dataframes_minus_21C = {}
for file_path in file_paths_new:
    # Load the second sheet
    sheet_data = pd.read_excel(file_path, sheet_name=1)
    dataframes_minus_21C[file_path] = sheet_data

# Plot data for all cycles at -21°C
plot_all_cycles(dataframes_minus_21C, active_mass_g)