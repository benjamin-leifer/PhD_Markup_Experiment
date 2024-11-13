import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(r'C:\Users\leifer.be\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\11\CK')
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
