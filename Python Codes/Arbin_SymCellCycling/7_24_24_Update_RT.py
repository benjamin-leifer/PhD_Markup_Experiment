import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the new working directory path
new_dir_path = r'E:\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\07\Update 0724\RT'

# Change the current working directory
os.chdir(new_dir_path)

# Verify the current working directory
print("Current Working Directory: ", os.getcwd())

file_paths = [
    'BL-LL-AR01_Channel_1_Wb_1.xlsx',
    'BL-LL-AS01_Channel_4_Wb_1.xlsx',
    'BL-LL-AT03_Channel_9_Wb_1.xlsx',
    'BL-LL-AU02_Channel_11_Wb_1.xlsx'
]

# Define function to process and plot data by cycle number
def plot_by_cycle_number(dataframes, active_mass_g):
    plt.figure(figsize=(15, 10))

    for file_path, df in dataframes.items():
        # Calculate capacity in mAh/g
        df['Charge Capacity (mAh/g)'] = (df['Charge Capacity (Ah)'] * 1000) / active_mass_g
        df['Discharge Capacity (mAh/g)'] = (df['Discharge Capacity (Ah)'] * 1000) / active_mass_g

        # Get unique cycle numbers
        cycle_numbers = df['Cycle Index'].unique()

        for cycle in cycle_numbers:
            # Filter data for current cycle
            cycle_data = df[df['Cycle Index'] == cycle]

            # Filter data for charge and discharge cycles
            charge_data = cycle_data[cycle_data['Current (A)'] > 0]
            discharge_data = cycle_data[cycle_data['Current (A)'] < 0]

            # Plot charge and discharge data
            plt.plot(charge_data['Charge Capacity (mAh/g)'], charge_data['Voltage (V)'],
                     label=f'Charge Cycle {cycle} {file_path}', linestyle='-')
            plt.plot(discharge_data['Discharge Capacity (mAh/g)'], discharge_data['Voltage (V)'],
                     label=f'Discharge Cycle {cycle} {file_path}', linestyle='--')

    # Configure plot
    plt.xlabel('Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage vs. Capacity (mAh/g) by Cycle Number for Multiple Batteries')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.show()


# File paths for the provided data
file_paths = [
    'BL-LL-AR01_Channel_1_Wb_1.xlsx',
    'BL-LL-AS01_Channel_4_Wb_1.xlsx',
    'BL-LL-AT03_Channel_9_Wb_1.xlsx',
    'BL-LL-AU02_Channel_11_Wb_1.xlsx'
]

# Active mass in grams
active_mass_g = 0.01293303225

# Define function to process and plot data by cycle number
def plot_cycle_2(dataframes, active_mass_g):
    fig, ax1 = plt.subplots(1,1,figsize=(4.6, 3.5))

    legends = {
        'BL-LL-AR01_Channel_1_Wb_1.xlsx': 'Gr||NMC - DTF14',
        'BL-LL-AS01_Channel_4_Wb_1.xlsx': 'Gr||NMC - DT14',
        'BL-LL-AT03_Channel_9_Wb_1.xlsx': 'Li||NMC - DTF14',
        'BL-LL-AU02_Channel_11_Wb_1.xlsx': 'Li||NMC - DT14'
    }

    for file_path, df in dataframes.items():
        # Calculate capacity in mAh/g
        df['Charge Capacity (mAh/g)'] = (df['Charge Capacity (Ah)'] * 1000) / active_mass_g
        df['Discharge Capacity (mAh/g)'] = (df['Discharge Capacity (Ah)'] * 1000) / active_mass_g

        # Filter data for cycle 2
        cycle_data = df[df['Cycle Index'] == 2]

        # Filter data for charge and discharge cycles
        charge_data = cycle_data[cycle_data['Current (A)'] > 0]
        discharge_data = cycle_data[cycle_data['Current (A)'] < 0]

        # Plot charge and discharge data
        plt.plot(charge_data['Charge Capacity (mAh/g)'], charge_data['Voltage (V)'],
                 label=f'Charge {legends[file_path]}', linestyle='-')
        plt.plot(discharge_data['Discharge Capacity (mAh/g)'], discharge_data['Voltage (V)'],
                 label=f'Discharge {legends[file_path]}', linestyle='--')

    # Configure plot
    plt.xlabel('Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage vs. Capacity (mAh/g) for Cycle 2')
    plt.legend(loc='best', fontsize='small')
    ax1.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='minor', direction='in', left=True, right=True, length=3)
    plt.show()


# Load the second sheet data from all files into a dictionary
dataframes = {}
for file_path in file_paths:
    # Load the second sheet
    sheet_data = pd.read_excel(file_path, sheet_name=1)
    dataframes[file_path] = sheet_data

# Plot data for cycle 2
#plot_cycle_2(dataframes, active_mass_g)


# Define function to process and plot data by cycle number with consistent colors
def plot_cycle_2_with_colors(dataframes, active_mass_g):
    fig, ax1 = plt.subplots(1,1,figsize=(4.6, 3.5))

    legends = {
        'BL-LL-AR01_Channel_1_Wb_1.xlsx': 'Gr||NMC - DTF14',
        'BL-LL-AS01_Channel_4_Wb_1.xlsx': 'Gr||NMC - DT14',
        'BL-LL-AT03_Channel_9_Wb_1.xlsx': 'Li||NMC - DTF14',
        'BL-LL-AU02_Channel_11_Wb_1.xlsx': 'Li||NMC - DT14'
    }

    colors = ['b', 'g', 'r', 'c']

    for idx, (file_path, df) in enumerate(dataframes.items()):
        # Calculate capacity in mAh/g
        df['Charge Capacity (mAh/g)'] = (df['Charge Capacity (Ah)'] * 1000) / active_mass_g
        df['Discharge Capacity (mAh/g)'] = (df['Discharge Capacity (Ah)'] * 1000) / active_mass_g

        # Filter data for cycle 2
        cycle_data = df[df['Cycle Index'] == 2]

        # Filter data for charge and discharge cycles
        charge_data = cycle_data[cycle_data['Current (A)'] > 0]
        discharge_data = cycle_data[cycle_data['Current (A)'] < 0]

        # Plot charge and discharge data with same color
        plt.plot(charge_data['Charge Capacity (mAh/g)'], charge_data['Voltage (V)'],
                 label=f'Charge {legends[file_path]}', linestyle='-', color=colors[idx])
        plt.plot(discharge_data['Discharge Capacity (mAh/g)'], discharge_data['Voltage (V)'],
                 label=f'Discharge {legends[file_path]}', linestyle='--', color=colors[idx])

    # Configure plot
    plt.xlabel('Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage vs. Capacity (mAh/g)')
    #plt.legend(loc='best', fontsize='small')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    ax1.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='minor', direction='in', left=True, right=True, length=3)
    plt.show()


# Plot data for cycle 2 with consistent colors
plot_cycle_2_with_colors(dataframes, active_mass_g)

