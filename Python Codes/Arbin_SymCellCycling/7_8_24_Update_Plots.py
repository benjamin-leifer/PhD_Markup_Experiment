import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the new working directory path
new_dir_path = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Low Temp Li ion\Cell Data\Arbin Data\2024\07\Li_low_temp\Data-files\Cell-DY'

# Change the current working directory
os.chdir(new_dir_path)

# Verify the current working directory
print("Current Working Directory: ", os.getcwd())
# Load the CSV files
file_path_discharge_minus_21 = 'YA_DY_Li_Discharge_at_-21.CSV'
file_path_discharge_minus_32 = 'YA_DY_Li_Discharge_at_-32.CSV'
file_path_discharge_minus_50 = 'YA_DY_Li_Discharge_at_-50.CSV'
file_path_formation_new = 'YA_DY_Li_formation.CSV'

df_discharge_minus_21 = pd.read_csv(file_path_discharge_minus_21)
df_discharge_minus_32 = pd.read_csv(file_path_discharge_minus_32)
df_discharge_minus_50 = pd.read_csv(file_path_discharge_minus_50)
df_formation_new = pd.read_csv(file_path_formation_new)

# Active mass for specific capacity calculation
active_mass = 0.01293303225

# Filter out rows where the current is zero
df_discharge_minus_21_filtered = df_discharge_minus_21[df_discharge_minus_21['Current (A)'] != 0]
df_discharge_minus_32_filtered = df_discharge_minus_32[df_discharge_minus_32['Current (A)'] != 0]
df_discharge_minus_50_filtered = df_discharge_minus_50[df_discharge_minus_50['Current (A)'] != 0]
df_formation_new_filtered = df_formation_new[df_formation_new['Current (A)'] != 0]

# Convert capacities to specific capacities (mAh/g)
df_discharge_minus_21_filtered['Specific Discharge Capacity (mAh/g)'] = df_discharge_minus_21_filtered['Discharge Capacity (Ah)'] * 1000 / active_mass
df_discharge_minus_32_filtered['Specific Discharge Capacity (mAh/g)'] = df_discharge_minus_32_filtered['Discharge Capacity (Ah)'] * 1000 / active_mass
df_discharge_minus_50_filtered['Specific Discharge Capacity (mAh/g)'] = df_discharge_minus_50_filtered['Discharge Capacity (Ah)'] * 1000 / active_mass
df_formation_new_filtered['Specific Discharge Capacity (mAh/g)'] = df_formation_new_filtered['Discharge Capacity (Ah)'] * 1000 / active_mass

# Split the data into discharge profiles
discharge_split_discharge_minus_21 = df_discharge_minus_21_filtered[df_discharge_minus_21_filtered['Current (A)'] < 0]
discharge_split_discharge_minus_32 = df_discharge_minus_32_filtered[df_discharge_minus_32['Current (A)'] < 0]
discharge_split_discharge_minus_50 = df_discharge_minus_50_filtered[df_discharge_minus_50['Current (A)'] < 0]

# Select the second cycle of formation data
formation_second_cycle_discharge = df_formation_new_filtered[(df_formation_new_filtered['Cycle Index'] == 2) & (df_formation_new_filtered['Current (A)'] < 0)]

# Recalculate unique cycles for each dataset
unique_cycles_discharge_minus_21 = df_discharge_minus_21_filtered['Cycle Index'].unique()
unique_cycles_discharge_minus_32 = df_discharge_minus_32_filtered['Cycle Index'].unique()
unique_cycles_discharge_minus_50 = df_discharge_minus_50_filtered['Cycle Index'].unique()

# Plot all the discharge capacities on a single plot, with the temperatures as labels
fig, ax1 = plt.subplots(1,1,figsize=(4.6, 3.5))

# Discharge at -21°C
for cycle in unique_cycles_discharge_minus_21:
    cycle_discharge = discharge_split_discharge_minus_21[discharge_split_discharge_minus_21['Cycle Index'] == cycle]
    plt.plot(cycle_discharge['Specific Discharge Capacity (mAh/g)'], cycle_discharge['Voltage (V)'], label=f'-21°C Cycle',)

# Discharge at -32°C
for cycle in unique_cycles_discharge_minus_32:
    cycle_discharge = discharge_split_discharge_minus_32[discharge_split_discharge_minus_32['Cycle Index'] == cycle]
    plt.plot(cycle_discharge['Specific Discharge Capacity (mAh/g)'], cycle_discharge['Voltage (V)'], label=f'-32°C Cycle',)

# Discharge at -50°C
for cycle in unique_cycles_discharge_minus_50:
    cycle_discharge = discharge_split_discharge_minus_50[discharge_split_discharge_minus_50['Cycle Index'] == cycle]
    plt.plot(cycle_discharge['Specific Discharge Capacity (mAh/g)'], cycle_discharge['Voltage (V)'], label=f'-50°C Cycle')

# Discharge from the second cycle of formation
plt.plot(formation_second_cycle_discharge['Specific Discharge Capacity (mAh/g)'], formation_second_cycle_discharge['Voltage (V)'], label='Room Temperature',)

plt.xlabel('Specific Discharge Capacity (mAh/g)')
plt.ylabel('Voltage (V)')
plt.title('Voltage vs. Specific Discharge Capacity')
#plt.title('Voltage vs. Capacity for all Pouch Cells')
ax1.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
ax1.tick_params(which='minor', direction='in', left=True, right=True, length=3)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend(loc='best')  # Add this line to show the legend
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
file_paths = {
    'AK': [
        'BL-LL-AK01_Channel_51_Wb_1.CSV',
        'BL-LL-AK01--20C_Rate_Channel_38_Wb_1.CSV'
    ],
    'AO': [
        'BL-LL-AO01_Channel_62_Wb_1.CSV',
        #'BL-LL-AO01-51C_Channel_37_Wb_1.CSV',
        'BL-LL-AO01--32C_Rate_Test_Channel_37_Wb_1.CSV'
    ],
    'AM': [
        'BL-LL-AM04_Channel_55_Wb_1.CSV',
        #'BL-LL-AM04--51C_Rate_Channel_38_Wb_1.CSV',
        'BL-LL-AM04--32C_Rate_test_Channel_38_Wb_1.CSV'
    ]
}

# Active mass for specific capacity calculation
active_mass = 0.01293303225

# Extract the second cycle of each dataset, split by groups
grouped_second_cycle_dfs = {'AK': [], 'AO': [], 'AM': []}

for group, paths in file_paths.items():
    for path in paths:
        df = pd.read_csv(path)
        # Filter out rows where the current is zero
        df_filtered = df[df['Current (A)'] != 0]

        # Extract the second cycle
        df_second_cycle = df_filtered[df_filtered['Cycle Index'] == 2]

        # Convert capacities to specific capacities (mAh/g)
        df_second_cycle['Specific Discharge Capacity (mAh/g)'] = df_second_cycle[
                                                                     'Discharge Capacity (Ah)'] * 1000 / active_mass

        # Split into discharge profile
        df_second_cycle_discharge = df_second_cycle[df_second_cycle['Current (A)'] < 0]

        grouped_second_cycle_dfs[group].append((path, df_second_cycle_discharge))

# Plot the second cycle discharge profiles for each group
for group, dfs in grouped_second_cycle_dfs.items():
    fig, ax1 = plt.subplots(1,1,figsize=(4.6, 3.5))
    for path, df in dfs:
        label = path.split('/')[-1].split('_')[0]  # Use a part of the filename as the label
        plt.plot(df['Specific Discharge Capacity (mAh/g)'], df['Voltage (V)'], label=label)

    plt.xlabel('Specific Discharge Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Voltage vs. Specific Discharge Capacity - {group}')
    ax1.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='minor', direction='in', left=True, right=True, length=3)
    plt.legend()
    fig.tight_layout()
    plt.show()
