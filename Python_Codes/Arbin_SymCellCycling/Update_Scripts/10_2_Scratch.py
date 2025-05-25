import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the new working directory path
new_dir_path = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\09\BM01'

# Change the current working directory
os.chdir(new_dir_path)

# Verify the current working directory
print("Current Working Directory: ", os.getcwd())

import os
import shutil
"""
# Define the new working directory path
import os
import shutil

# Define the new working directory path


# Change the current working directory
os.chdir(new_dir_path)

# Verify the current working directory
print("Current Working Directory: ", os.getcwd())

# Collect all Excel files with BL02 and BM01 in their filenames from all subdirectories
bl02_files = []
bm01_files = []

for root, dirs, files in os.walk(new_dir_path):
    for file in files:
        if file.endswith('.xlsx'):
            if 'BL02' in file:
                bl02_files.append(os.path.join(root, file))
            elif 'BM01' in file:
                bm01_files.append(os.path.join(root, file))

# Create new directories if they don't exist
os.makedirs('BL02', exist_ok=True)
os.makedirs('BM01', exist_ok=True)

# Move the collected files to their respective directories
for file in bl02_files:
    shutil.move(file, os.path.join('BL02', os.path.basename(file)))

for file in bm01_files:
    shutil.move(file, os.path.join('BM01', os.path.basename(file)))

print("Files moved successfully.")

# Active mass in grams
active_mass_g = 0.01293303225
# Define file paths for the newly uploaded data at -21°C
file_paths_minus_21C = [
    'BL-LL-AZ04_RT_t1_Channel_45_Wb_1.xlsx',
    'BL-LL-BA02_RT_t1_Channel_47_Wb_1.xlsx',
    'BL-LL-BB04_RT_t1_Channel_55_Wb_1.xlsx',
    'BL-LL-_RT_t1_BC03_Channel_3_Wb_1.xlsx',
    'BL-LL-_RT_t1_BD03_Channel_6_Wb_1.xlsx',
    'BL-LL-_RT_t1_BE01_Channel_8_Wb_1.xlsx',
    'BL-LL-BF03_RT_t1_Channel_15_Wb_1.xlsx',
    'BL-LL-BG04_RT_t1_Channel_20_Wb_1.xlsx',

]

# Define legends for the newly uploaded data
legends_minus_21C = {
    'BL-LL-AZ04_RT_t1_Channel_45_Wb_1.xlsx': 'Li||NMC - DT14 - RT',
    'BL-LL-BA02_RT_t1_Channel_47_Wb_1.xlsx': 'Li||NMC - DTF14 - RT',
    'BL-LL-BB04_RT_t1_Channel_55_Wb_1.xlsx': 'Gr||NMC - DTF14 - RT',
    'BL-LL-_RT_t1_BC03_Channel_3_Wb_1.xlsx': 'HC||NMC - DT14 - RT',
    'BL-LL-_RT_t1_BD03_Channel_6_Wb_1.xlsx': 'HC||NMC - DTF14 - RT',
    'BL-LL-_RT_t1_BE01_Channel_8_Wb_1.xlsx':'HC||NMC - MF91 - RT',
    'BL-LL-BF03_RT_t1_Channel_15_Wb_1.xlsx':'Li||NMC - MF91 - RT',
    'BL-LL-BG04_RT_t1_Channel_20_Wb_1.xlsx':'Gr||NMC - MF91 - RT',

}
#: 'Li||NMC - DTF14 - 51C'

# Define function to process and plot all cycles data
def plot_all_cycles(dataframes, active_mass_g):
    fig, ax1 = plt.subplots(1,1,figsize=(4.6*1.5, 3.5*1.5))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange']

    for idx, (file_path, df) in enumerate(dataframes.items()):
        # Calculate capacity in mAh/g
        df['Charge Capacity (mAh/g)'] = (df['Charge Capacity (Ah)'] * 1000) / active_mass_g
        df['Discharge Capacity (mAh/g)'] = (df['Discharge Capacity (Ah)'] * 1000) / active_mass_g

        # Filter data for all cycles
        charge_data = df[df['Current (A)'] > 0]
        if 'RT' in file_path:
            cycle_data = df[df['Cycle Index'] == 2]
        else:
            cycle_data = df[df['Cycle Index'] == 1]
        discharge_data = cycle_data[cycle_data['Current (A)'] < 0]

        # Plot charge and discharge data with same color
        #plt.plot(charge_data['Charge Capacity (mAh/g)'], charge_data['Voltage (V)'],
        #        label=f'Charge {legends_minus_21C[file_path]}', linestyle='-', color=colors[idx])
        plt.plot(discharge_data['Discharge Capacity (mAh/g)'], discharge_data['Voltage (V)'],
                 label=f'Discharge {legends_minus_21C[file_path]}', linestyle='--', color=colors[idx])

    # Configure plot
    plt.xlabel('Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage vs. Capacity (mAh/g)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)
    ax1.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='minor', direction='in', left=True, right=True, length=3)
    plt.tight_layout()
    plt.show()



# Load the second sheet data from all files into a dictionary
dataframes_minus_21C = {}
for file_path in file_paths_minus_21C:
    # Load the second sheet
    sheet_data = pd.read_excel(file_path, sheet_name=1)
    dataframes_minus_21C[file_path] = sheet_data

# Plot data for all cycles at -21°C
plot_all_cycles(dataframes_minus_21C, active_mass_g)
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Paths to the uploaded Excel files
file_paths = [
    "BL-LL-BM01_-51C_C50_Channel_41_Wb_1.xlsx",
    "BL-LL-BM01_-51C_C50_cyc2_Channel_41_Wb_1.xlsx",
    "BL-LL-BM01_-51C_C50_Dis_1_Channel_41_Wb_1.xlsx",
    "BL-LL-BM01_Charge_1_Channel_56_Wb_1.xlsx",
    "BL-LL-BM01_RT_Charge_Channel_62_Wb_1.xlsx",
    "BL-LL-BM01_RT_Form_Channel_56_Wb_1.xlsx",
    "BL-LL-BM01_-21C_C-50_Channel_41_Wb_1.xlsx",
    "BL-LL-BM01_-21C_cycles2-3_Channel_41_Wb_1.xlsx",
    "BL-LL-BM01_-32C_C50_Channel_41_Wb_1.xlsx"
]

# Define the path for the combined RT_Form file
combined_form_file_path = "BL-LL-BM01_RT_Form_Channel_56_Wb_1_Combined.xlsx"

# Read the second sheet from the combined form file
combined_form_df = pd.read_excel(combined_form_file_path, sheet_name=1)

# Filter to keep only non-zero current rows
combined_form_df = combined_form_df[combined_form_df['Current (A)'] != 0]

# Calculate the total charge capacity (maximum value) across all cycles for the Combined RT_Form
combined_form_total_capacity = combined_form_df[combined_form_df['Current (A)'] > 0]['Charge Capacity (Ah)'].max()


# Updated function to extract and relabel temperature and C-rate from file names
def extract_and_relabel_temp_and_c_rate(file_name):
    # Use regex to find temperature (e.g., -51C, -21C) and C-rate (e.g., C50)
    temp_match = re.search(r'-\d{1,2}C', file_name)
    c_rate_match = re.search(r'C-?\d+', file_name)

    temp = temp_match.group() if temp_match else "Unknown Temp"

    # Handle relabeling of C-rates
    if c_rate_match:
        c_rate = c_rate_match.group()
        if "C50" in c_rate or "C-50" in c_rate:
            c_rate = "C/50"
        else:
            c_rate = c_rate.replace("C", "C/")
    else:
        c_rate = "C/10"  # Assume unknown C-rate is C/10

    return f"{temp}, {c_rate}"


# Initialize a new plot for normalized discharge curves with updated labels
fig, ax1 = plt.subplots(1,1,figsize=(4.6*1.5, 3.5*1.5))

# Variable to track the count of -51C samples plotted
count_51C = 0

# Loop through each file path except the combined file
for file_path in file_paths:
    # Read the second sheet of each file
    df = pd.read_excel(file_path, sheet_name=1)

    # Convert Date_Time to datetime format
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])

    # Filter rows where Current is not zero
    df = df[df['Current (A)'] != 0]

    # Calculate discharge capacity as a percentage of the total combined charge capacity
    combined_form_total_capacity = 0.044
    df['Normalized Discharge Capacity (%)'] = (df['Discharge Capacity (Ah)'] / combined_form_total_capacity) * 100

    # Extract the file name and create a label based on temperature and relabeled C-rate
    file_name = os.path.basename(file_path).replace('.xlsx', '')
    label = extract_and_relabel_temp_and_c_rate(file_name)

    # Skip unknown temperature samples
    if "Unknown Temp" in label:
        continue

    # Filter to only discharge data (where current is negative)
    discharge_data = df[df['Current (A)'] < 0]

    # Identify the -51C samples and exclude the first two
    if "-51C" in label:
        count_51C += 1
        # Skip the first two -51C samples
        if count_51C <= 2:
            continue

    # Plot normalized discharge capacity vs voltage for the current file with temperature and relabeled C-rate
    plt.plot(discharge_data['Normalized Discharge Capacity (%)'], discharge_data['Voltage (V)'], label=label)

plt.xlabel('Normalized Discharge Capacity (%)')
plt.ylabel('Voltage (V)')
plt.title('Normalized Discharge Capacity vs Voltage For Pouch Cell BM01')
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Paths to the uploaded Excel files
file_paths = [
    "BL-LL-BM01_-51C_C50_Channel_41_Wb_1.xlsx",
    "BL-LL-BM01_-51C_C50_cyc2_Channel_41_Wb_1.xlsx",
    "BL-LL-BM01_-51C_C50_Dis_1_Channel_41_Wb_1.xlsx",
    "BL-LL-BM01_Charge_1_Channel_56_Wb_1.xlsx",
    "BL-LL-BM01_RT_Charge_Channel_62_Wb_1.xlsx",
    "BL-LL-BM01_RT_Form_Channel_56_Wb_1.xlsx",
    "BL-LL-BM01_-21C_C-50_Channel_41_Wb_1.xlsx",
    "BL-LL-BM01_-21C_cycles2-3_Channel_41_Wb_1.xlsx",
    "BL-LL-BM01_-32C_C50_Channel_41_Wb_1.xlsx"
]

# Define the path for the combined RT_Form file
combined_form_file_path = "BL-LL-BM01_RT_Form_Channel_56_Wb_1_Combined.xlsx"

# Read the second sheet from the combined form file
combined_form_df = pd.read_excel(combined_form_file_path, sheet_name=1)

# Filter to keep only non-zero current rows
combined_form_df = combined_form_df[combined_form_df['Current (A)'] != 0]

# Calculate the total charge capacity (maximum value) across all cycles for the Combined RT_Form
combined_form_total_capacity = combined_form_df[combined_form_df['Current (A)'] > 0]['Charge Capacity (Ah)'].max()

# Updated function to extract and relabel temperature and C-rate from file names
def extract_and_relabel_temp_and_c_rate(file_name):
    # Use regex to find temperature (e.g., -51C, -21C) and C-rate (e.g., C50)
    temp_match = re.search(r'-\d{1,2}C', file_name)
    c_rate_match = re.search(r'C-?\d+', file_name)

    temp = temp_match.group() if temp_match else "Unknown Temp"

    # Handle relabeling of C-rates
    if c_rate_match:
        c_rate = c_rate_match.group()
        if "C50" in c_rate or "C-50" in c_rate:
            c_rate = "C/50"
        else:
            c_rate = c_rate.replace("C", "C/")
    else:
        c_rate = "C/10"  # Assume unknown C-rate is C/10

    return f"{temp}, {c_rate}"

# Initialize a new plot for normalized discharge curves with updated labels
fig, ax1 = plt.subplots(1,1,figsize=(4.6*1.5, 3.5*1.5))

# Variable to track the count of -51C samples plotted
count_51C = 0

# Loop through each file path except the combined file
for file_path in file_paths:
    # Read the second sheet of each file
    df = pd.read_excel(file_path, sheet_name=1)

    # Convert Date_Time to datetime format
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])

    # Filter rows where Current is not zero
    df = df[df['Current (A)'] != 0]

    # Calculate discharge capacity as a percentage of the total combined charge capacity
    combined_form_total_capacity = 0.044
    df['Normalized Discharge Capacity (%)'] = (df['Discharge Capacity (Ah)'] / combined_form_total_capacity) * 100

    # Extract the file name and create a label based on temperature and relabeled C-rate
    file_name = os.path.basename(file_path).replace('.xlsx', '')
    label = extract_and_relabel_temp_and_c_rate(file_name)

    # Skip unknown temperature samples
    if "Unknown Temp" in label:
        continue

    # Filter to only discharge data (where current is negative)
    discharge_data = df[df['Current (A)'] < 0]

    # Identify the -51C samples and exclude the first two
    if "-51C" in label:
        count_51C += 1
        # Skip the first two -51C samples
        if count_51C <= 2:
            continue

    # Plot normalized discharge capacity vs voltage for the current file with temperature and relabeled C-rate
    plt.plot(discharge_data['Normalized Discharge Capacity (%)'], discharge_data['Voltage (V)'], label=label)

# Add the formation cycle discharge curve
formation_cycle_file_path = "BL-LL-BM01_RT_Form_Channel_56_Wb_1.xlsx"
formation_cycle_df = pd.read_excel(formation_cycle_file_path, sheet_name=1)
formation_cycle_df = formation_cycle_df[formation_cycle_df['Current (A)'] != 0]
formation_cycle_df['Normalized Discharge Capacity (%)'] = (formation_cycle_df['Discharge Capacity (Ah)'] / combined_form_total_capacity) * 100
formation_discharge_data = formation_cycle_df[formation_cycle_df['Current (A)'] < 0]
plt.plot(formation_discharge_data['Normalized Discharge Capacity (%)'], formation_discharge_data['Voltage (V)'], label='Formation Cycle Discharge', linestyle='--', color='black')

plt.xlabel('Normalized Discharge Capacity (%)')
plt.ylabel('Voltage (V)')
plt.title('Normalized Discharge Capacity vs Voltage For Pouch Cell BM01')
plt.legend()
plt.tight_layout()
plt.show()
