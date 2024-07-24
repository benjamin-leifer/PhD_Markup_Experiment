import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the new working directory path
new_dir_path = r'E:\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\07\Update 0724\-21C Trial 1'

# Change the current working directory
os.chdir(new_dir_path)

# Verify the current working directory
print("Current Working Directory: ", os.getcwd())
# Active mass in grams
active_mass_g = 0.01293303225
# Define file paths for the newly uploaded data at -21°C
file_paths_minus_21C = [
    'BL-LL-AR01_-21C_1Cycle_Channel_37_Wb_1.xlsx',
    'BL-LL-AS01_-21C_1Cycle_Channel_38_Wb_1.xlsx',
    #'BL-LL-AT03_-21C_1Cycle_Channel_39_Wb_1.xlsx',
    #'BL-LL-AU02_-21C_1Cycle_Channel_40_Wb_1.xlsx',

]

# Define legends for the newly uploaded data
legends_minus_21C = {
    'BL-LL-AR01_-21C_1Cycle_Channel_37_Wb_1.xlsx': 'Gr||NMC - DTF14',
    'BL-LL-AS01_-21C_1Cycle_Channel_38_Wb_1.xlsx': 'Gr||NMC - DT14',
    #'BL-LL-AT03_-21C_1Cycle_Channel_39_Wb_1.xlsx': 'Li||NMC - DTF14',
    #'BL-LL-AU02_-21C_1Cycle_Channel_40_Wb_1.xlsx': 'Li||NMC - DT14'
}


# Define function to process and plot all cycles data
def plot_all_cycles(dataframes, active_mass_g):
    fig, ax1 = plt.subplots(1,1,figsize=(4.6, 3.5))

    colors = ['b', 'g', 'r', 'c']

    for idx, (file_path, df) in enumerate(dataframes.items()):
        # Calculate capacity in mAh/g
        df['Charge Capacity (mAh/g)'] = (df['Charge Capacity (Ah)'] * 1000) / active_mass_g
        df['Discharge Capacity (mAh/g)'] = (df['Discharge Capacity (Ah)'] * 1000) / active_mass_g

        # Filter data for all cycles
        charge_data = df[df['Current (A)'] > 0]
        discharge_data = df[df['Current (A)'] < 0]

        # Plot charge and discharge data with same color
        plt.plot(charge_data['Charge Capacity (mAh/g)'], charge_data['Voltage (V)'],
                 label=f'Charge {legends_minus_21C[file_path]}', linestyle='-', color=colors[idx])
        plt.plot(discharge_data['Discharge Capacity (mAh/g)'], discharge_data['Voltage (V)'],
                 label=f'Discharge {legends_minus_21C[file_path]}', linestyle='--', color=colors[idx])

    # Configure plot
    plt.xlabel('Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage vs. Capacity (mAh/g) at -21°C')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    ax1.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='minor', direction='in', left=True, right=True, length=3)
    plt.show()


# Load the second sheet data from all files into a dictionary
dataframes_minus_21C = {}
for file_path in file_paths_minus_21C:
    # Load the second sheet
    sheet_data = pd.read_excel(file_path, sheet_name=1)
    dataframes_minus_21C[file_path] = sheet_data

# Plot data for all cycles at -21°C
plot_all_cycles(dataframes_minus_21C, active_mass_g)
