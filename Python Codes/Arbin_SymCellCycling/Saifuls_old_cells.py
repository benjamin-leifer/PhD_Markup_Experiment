import pandas as pd
import matplotlib.pyplot as plt
import os

import os
import shutil

def copy_cm_ck_files(src_dir):
    # Create the destination directory if it doesn't exist
    dest_dir = os.path.join(src_dir, 'CM and CK')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Walk through all folders in the source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if 'CM' in file or 'CK' in file:
                # Construct full file path
                file_path = os.path.join(root, file)
                # Copy the file to the destination directory
                shutil.copy(file_path, dest_dir)
                print(f"Copied: {file_path} to {dest_dir}")




os.chdir(r'C:\Users\leifer.be\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\11')
src_directory = os.getcwd()

def load_and_group_data(src_dir):
    # Define the destination directory
    dest_dir = os.path.join(src_dir, 'CM and CK')

    # Initialize a dictionary to store dataframes
    dataframes = {}

    # Walk through the destination directory
    for root, dirs, files in os.walk(dest_dir):
        for file in files:
            if file.endswith('.xlsx') and ('CM' in file or 'CK' in file):
                # Extract the cell ID from the file name
                cell_id = file.split('_')[0]
                # Construct full file path
                file_path = os.path.join(root, file)
                # Load the Excel file into a DataFrame
                df = pd.read_excel(file_path, sheet_name=1)
                # Append the DataFrame to the dictionary
                if cell_id not in dataframes:
                    dataframes[cell_id] = []
                dataframes[cell_id].append((file, df))

    # Define the custom order
    custom_order = ['Form', 'cyc2', 'dist-1st']

    # Concatenate all DataFrames for each cell ID into a single DataFrame in the specified order
    for cell_id in dataframes:
        # Sort the list of DataFrames by the custom order
        dataframes[cell_id].sort(
            key=lambda x: next((custom_order.index(order) for order in custom_order if order in x[0]),
                               len(custom_order)))

        # Initialize a variable to keep track of the cumulative max test time
        cumulative_max_test_time = 0

        # Update the Test Time (s) column and concatenate the DataFrames
        updated_dfs = []
        for _, df in dataframes[cell_id]:
            df['Test Time (s)'] += cumulative_max_test_time
            cumulative_max_test_time = df['Test Time (s)'].max()
            updated_dfs.append(df)

        dataframes[cell_id] = pd.concat(updated_dfs, ignore_index=True)

    return dataframes


# Example usage

cell_dataframes = load_and_group_data(src_directory)

# Print the keys (cell IDs) and the first few rows of each DataFrame
for cell_id, df in cell_dataframes.items():
    print(f"Cell ID: {cell_id}")
    print(df.head())

import matplotlib.pyplot as plt

# Iterate over each cell ID and its corresponding DataFrame
for cell_id, df in cell_dataframes.items():
    plt.figure(figsize=(10, 6))
    plt.plot(df['Test Time (s)']/3600, df['Voltage (V)'], label=f'Cell ID: {cell_id}')
    plt.xlabel('Time (hr)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Time vs Voltage for {cell_id}')
    plt.legend()
    plt.grid(True)
    plt.show()
"""
# File paths for CK datasets
# Creating the lists of CK and CM file names based on previously uploaded files
ck_file_names = [
    'BL-LL-CK05_RT_Form_Channel_41_Wb_1.xlsx',
    'BL-LL-CK01_RT_Form_Channel_37_Wb_1.xlsx',
    'BL-LL-CK02_RT_Form_Channel_38_Wb_1.xlsx',
    'BL-LL-CK03_RT_Form_Channel_39_Wb_1.xlsx',
    'BL-LL-CK04_RT_Form_Channel_40_Wb_1.xlsx'
]

cm_file_names = [
    'BL-LL-CM03_RT_Form_Channel_37_Wb_1.xlsx',
    'BL-LL-CM04_RT_Form_Channel_38_Wb_1.xlsx',
    'BL-LL-CM05_RT_Form_Channel_39_Wb_1.xlsx',
    'BL-LL-CM06_RT_Form_Channel_40_Wb_1.xlsx',
    'BL-LL-CM07_RT_Form_Channel_41_Wb_1.xlsx'
]


# Load data for CK files
uploaded_data = {}
for file_path in ck_file_names:
    xls = pd.ExcelFile(file_path)
    sheet_name = xls.sheet_names[1]  # Assuming data is in the second sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    uploaded_data[sheet_name] = df

# Load data for CM files
new_uploaded_data = {}
for file_path in cm_file_names:
    xls = pd.ExcelFile(file_path)
    sheet_name = xls.sheet_names[1]  # Assuming data is in the second sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    new_uploaded_data[sheet_name] = df

# Active mass (adjust as needed)
active_mass = 0.01293303225

# Plotting
plt.figure(figsize=(7, 5))

# Plot CK datasets with "DT14" in the legend
for sheet_name, df in uploaded_data.items():
    discharge_data_cycle1 = df[(df['Current (A)'] > 0) & (df['Cycle Index'] == 1)]
    if not discharge_data_cycle1.empty:
        discharge_data_cycle1['Specific Capacity (mAh/g)'] = (discharge_data_cycle1['Charge Capacity (Ah)'] * 1000) / active_mass
        plt.plot(discharge_data_cycle1['Specific Capacity (mAh/g)'], discharge_data_cycle1['Voltage (V)'],
                 label=f'{sheet_name} (DT14)', linestyle='--', markersize=3)

# Plot CM datasets with "DTF14" in the legend
for sheet_name, df in new_uploaded_data.items():
    discharge_data_cycle1 = df[(df['Current (A)'] > 0) & (df['Cycle Index'] == 1)]
    if not discharge_data_cycle1.empty:
        discharge_data_cycle1['Specific Capacity (mAh/g)'] = (discharge_data_cycle1['Charge Capacity (Ah)'] * 1000) / active_mass
        plt.plot(discharge_data_cycle1['Specific Capacity (mAh/g)'], discharge_data_cycle1['Voltage (V)'],
                 label=f'{sheet_name} (DTF14)', linestyle='--', markersize=3)

# Adding titles and labels
plt.title('Specific Capacity vs. Voltage Curves for Discharge Cycle 1 (DT14 vs. DTF14)')
plt.xlabel('Specific Capacity (mAh/g)')
plt.ylabel('Voltage (V)')
plt.legend()
#plt.grid(True)
plt.show()

# Plot comparison for each channel
channels = ['Channel37', 'Channel38', 'Channel39', 'Channel40', 'Channel41']
print(uploaded_data.keys())

for channel in channels:
    plt.figure(figsize=(7, 5))

    # Plot CK datasets for the current channel
    for sheet_name, df in uploaded_data.items():
        if channel in sheet_name:
            discharge_data_cycle1 = df[(df['Current (A)'] > 0) & (df['Cycle Index'] == 1)]
            if not discharge_data_cycle1.empty:
                discharge_data_cycle1['Specific Capacity (mAh/g)'] = (discharge_data_cycle1['Charge Capacity (Ah)'] * 1000) / active_mass
                plt.plot(discharge_data_cycle1['Specific Capacity (mAh/g)'], discharge_data_cycle1['Voltage (V)'],
                         label=f'{sheet_name} (DT14)', linestyle='--', markersize=3)

    # Plot CM datasets for the current channel
    for sheet_name, df in new_uploaded_data.items():
        if channel in sheet_name:
            discharge_data_cycle1 = df[(df['Current (A)'] > 0) & (df['Cycle Index'] == 1)]
            if not discharge_data_cycle1.empty:
                discharge_data_cycle1['Specific Capacity (mAh/g)'] = (discharge_data_cycle1['Charge Capacity (Ah)'] * 1000) / active_mass
                plt.plot(discharge_data_cycle1['Specific Capacity (mAh/g)'], discharge_data_cycle1['Voltage (V)'],
                         label=f'{sheet_name} (DTF14)', linestyle='--', markersize=3)

    # Adding titles and labels
    plt.title(f'Specific Capacity vs. Voltage Curves for Discharge Cycle 1 ({channel})')
    plt.xlabel('Specific Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.show()
"""