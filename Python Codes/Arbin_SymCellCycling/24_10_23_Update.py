import Arbin_SymCell
import os
import shutil
import tkinter as tk
import tkinter.filedialog as filedialog
import pandas as pd
import matplotlib.pyplot as plt
# Active mass in grams
active_mass_g = 0.01293303225

root = tk.Tk()
root.withdraw()  # to hide the main window

folder_selected = filedialog.askdirectory()  # open the dialog to choose directory
os.chdir(folder_selected)
cwd = os.getcwd()

# Get the tab20 colormap
cmap = plt.get_cmap('tab20')

# Generate a list of 20 colors
colors = [cmap(i) for i in range(20)]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# Print the list of colors
print(colors)
# Define the root directory to search for Excel files
def move_to_dir():
    # Define the current working directory
    cwd = os.getcwd()

    # Define the destination directory
    dest_dir = 'All cycling files'

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Walk through the directory tree
    for subdir, _, files in os.walk(cwd):
        for file in files:
            # Check if the file is an Excel file
            if file.endswith('.xlsx') or file.endswith('.xls'):
                # Construct full file path
                file_path = os.path.join(subdir, file)
                # Move the file to the destination directory
                shutil.copy(file_path, dest_dir)

    print("All Excel files have been moved to the 'All cycling files' folder.")
"""
# Define the destination directory
dest_dir = 'All cycling files'

# Create the destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Walk through the directory tree
for subdir, _, files in os.walk(cwd):
    for file in files:
        # Check if the file is an Excel file
        if file.endswith('.xlsx') or file.endswith('.xls'):
            # Construct full file path
            file_path = os.path.join(subdir, file)
            # Move the file to the destination directory
            shutil.copy(file_path, dest_dir)

print("All Excel files have been moved to the 'All cycling files' folder.")
"""

legends = {
    #'BL-LL-BS01_RT_RateTest_Channel_1_Wb_1.xlsx': 'Gr||NMC - DTF14',
    'BL-LL-BS02_RT_RateTest_Channel_2_Wb_1.xlsx': 'Gr_NMC _ DTF14',
    #'BL-LL-BS03_RT_RateTest_Channel_3_Wb_1.xlsx': 'Gr||NMC - DTF14',
    #'BL-LL-BT01_RT_RateTest_Channel_4_Wb_1.xlsx': 'Gr||NMC - MF91',
    'BL-LL-BT02_RT_RateTest_Channel_5_Wb_1.xlsx': 'Gr_NMC _ MF91',
    #'BL-LL-BT03_RT_RateTest_Channel_6_Wb_1.xlsx': 'Gr||NMC - MF91',
    #'BL-LL-BU01_RT_RateTest_Channel_7_Wb_1.xlsx': 'Gr||NMC - LP',
    #'BL-LL-BU02_RT_RateTest_Channel_8_Wb_1.xlsx': 'Gr||NMC - LP',
    'BL-LL-BU03_RT_RateTest_Channel_9_Wb_1.xlsx': 'Gr_NMC _ LP',
    'BL-LL-BV01_RT_RateTest_Channel_10_Wb_1.xlsx': 'HC_NMC _ DTF14',
    #'BL-LL-BV02_RT_RateTest_Channel_11_Wb_1.xlsx': 'HC||NMC - DTF14',
    #'BL-LL-BV03_RT_RateTest_Channel_12_Wb_1.xlsx': 'HC||NMC - DTF14',
    #'BL-LL-BW01_RT_RateTest_Channel_13_Wb_1.xlsx': 'HC||NMC - MF91',
    'BL-LL-BW02_RT_RateTest_Channel_14_Wb_1.xlsx': 'HC_NMC _ MF91',
    #'BL-LL-BW03_RT_RateTest_Channel_15_Wb_1.xlsx': 'HC||NMC - MF91',
    #'BL-LL-BX01_RT_RateTest_Channel_16_Wb_1.xlsx': 'HC||NMC - LP',
    #'BL-LL-BX02_RT_RateTest_Channel_17_Wb_1.xlsx': 'HC||NMC - LP',
    'BL-LL-BX03_RT_RateTest_Channel_18_Wb_1.xlsx': 'HC_NMC _ LP',
}

#: 'Li||NMC - DTF14 - 51C'

# Define function to process and plot all cycles data
def plot_all_cycles(dataframes, active_mass_g):
    fig, ax1 = plt.subplots(1,1,figsize=(4.6, 3.5))

    #colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    for idx, (file_path, df) in enumerate(dataframes.items()):
        # Calculate capacity in mAh/g
        df['Charge Capacity (mAh/g)'] = (df['Charge Capacity (Ah)'] * 1000) / active_mass_g
        df['Discharge Capacity (mAh/g)'] = (df['Discharge Capacity (Ah)'] * 1000) / active_mass_g

        # Filter data for all cycles
        #charge_data = df[df['Current (A)'] > 0]
        if 'RT' in file_path:
            cycle_data = df[df['Cycle Index'] == 2]
        else:
            cycle_data = df[df['Cycle Index'] == 1]
        discharge_data = cycle_data[cycle_data['Current (A)'] < 0]
        charge_data = cycle_data[cycle_data['Current (A)'] > 0]

        # Plot charge and discharge data with same color
        plt.plot(charge_data['Charge Capacity (mAh/g)'], charge_data['Voltage (V)'],
                label=f'Charge {legends[file_path]}', linestyle='-', color=colors[idx])
        plt.plot(discharge_data['Discharge Capacity (mAh/g)'], discharge_data['Voltage (V)'],
                 label=f'Discharge {legends[file_path]}', linestyle='--', color=colors[idx])

    # Configure plot
    plt.xlabel('Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage vs. Capacity (mAh/g) - RT, Cycle 2, calendared')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)
    ax1.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='minor', direction='in', left=True, right=True, length=3)
    plt.show()



legends_keys = list(legends.keys())
# Load the second sheet data from all files into a dictionary
dataframes_minus_21C = {}
for file_path in legends_keys:
    # Load the second sheet
    sheet_data = pd.read_excel(file_path, sheet_name=1)
    dataframes_minus_21C[file_path] = sheet_data

# Plot data for all cycles at -21°C
plot_all_cycles(dataframes_minus_21C, active_mass_g)
for file_path in legends_keys:
    print(file_path)
    print(legends[file_path])

cells = []
i = 0
for cell in legends_keys:
    label_tag = Arbin_SymCell.find_bl_ll_xx00(cell)
    try:
        cells.append(Arbin_SymCell.arbin_import_Sym_Cell(cell, name=legends[cell], mass=2/155/1000,
                                           theoretical_cap=155, color=colors[i], shape='o'))
    except FileNotFoundError as e:
        print('File not found')
    i += 1
print(cells)

for cell in cells:
    cell.plot_voltage_vs_time()
    plt.savefig(cell.name + '_voltage_vs_time.png', dpi=500, bbox_inches='tight')
    cell.plot_voltage_vs_capacity(clean_filter=False)
    plt.savefig(cell.name + '_voltage_vs_capacity.png', dpi=500, bbox_inches='tight')
    plt.clf()
            #cell.get_max_capacity_per_cycle()
            #cell.get_min_capacity_per_cycle()
            #cell.get_coulombic_efficiency()
    cell.plot_capacity_and_ce_vs_cycle()
    plt.savefig(cell.name + '_coulombic_efficiency.png', dpi=500, bbox_inches='tight')
#move_to_dir()





