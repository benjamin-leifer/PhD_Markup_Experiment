import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
file_1_path = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\09\BL-LL-BM01_Charge_1_2024_09_12_172429\BL-LL-BM01_Charge_1_Channel_56_Wb_1.xlsx'
file_2_path = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\09\BL-LL-BM01_-21C_C-50_2024_09_13_094020\BL-LL-BM01_-21C_C-50_Channel_41_Wb_1.xlsx'
file_3_path = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\09\BL-LL-BM01_RT_Form_2024_09_12_121458\BL-LL-BM01_RT_Form_Channel_56_Wb_1.xlsx'

# Load the second sheet from all files
data_file_1 = pd.read_excel(file_1_path, sheet_name=1)
data_file_2 = pd.read_excel(file_2_path, sheet_name=1)
data_file_3 = pd.read_excel(file_3_path, sheet_name=1)

# Filter out rows with zero current
filtered_data_file_1 = data_file_1[data_file_1['Current (A)'] != 0]
filtered_data_file_2 = data_file_2[data_file_2['Current (A)'] != 0]
filtered_data_file_3 = data_file_3[data_file_3['Current (A)'] != 0]

# Adjust the charge capacity from file 1 by adding the max charge capacity of file 3
charge_capacity_offset_file_3 = filtered_data_file_3['Charge Capacity (Ah)'].max()
filtered_data_file_1_adjusted = filtered_data_file_1.copy()
filtered_data_file_1_adjusted['Charge Capacity (Ah)'] = filtered_data_file_1_adjusted['Charge Capacity (Ah)'] + charge_capacity_offset_file_3

# Concatenate the adjusted charge data from file 3 and file 1
adjusted_charge_data_offset = pd.concat([filtered_data_file_3[['Charge Capacity (Ah)', 'Voltage (V)']],
                                         filtered_data_file_1_adjusted[['Charge Capacity (Ah)', 'Voltage (V)']]])

# Adjust the charge capacities to start from 0
adjusted_charge_data_offset['Charge Capacity (Ah)'] = adjusted_charge_data_offset['Charge Capacity (Ah)'] - adjusted_charge_data_offset['Charge Capacity (Ah)'].min()

# Adjust the discharge capacity from file 2 by subtracting the minimum to start at 0
charge_capacity_offset_total = adjusted_charge_data_offset['Charge Capacity (Ah)'].max()
filtered_data_file_2_adjusted = filtered_data_file_2.copy()
filtered_data_file_2_adjusted['Discharge Capacity (Ah)'] = filtered_data_file_2_adjusted['Discharge Capacity (Ah)'] + charge_capacity_offset_total
filtered_data_file_2_adjusted['Discharge Capacity (Ah)'] = filtered_data_file_2_adjusted['Discharge Capacity (Ah)'] - filtered_data_file_2_adjusted['Discharge Capacity (Ah)'].min()

# Plot the charge and discharge curves
fig, ax1 = plt.subplots(1,1,figsize=(4.6, 3.5))
plt.plot(adjusted_charge_data_offset['Charge Capacity (Ah)'], adjusted_charge_data_offset['Voltage (V)'],
         label='Charge 1 (RT, C/10)', color='red')
plt.plot(filtered_data_file_2_adjusted['Discharge Capacity (Ah)'], filtered_data_file_2_adjusted['Voltage (V)'],
         label='Discharge 1 (-21C, C/50)', color='blue')
plt.xlabel('Capacity (Ah)')
plt.ylabel('Voltage (V)')
plt.title('Voltage vs. Capacity (Ah)')
#plt.legend(loc='best')
plt.legend( loc='upper center', bbox_to_anchor=(0.4, 1), ncol=1)
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
ax1.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
ax1.tick_params(which='minor', direction='in', left=True, right=True, length=3)
plt.show()

# Calculate the Coulombic efficiency
total_charge_capacity = adjusted_charge_data_offset['Charge Capacity (Ah)'].max()
total_discharge_capacity = filtered_data_file_2_adjusted['Discharge Capacity (Ah)'].max()
coulombic_efficiency = (total_discharge_capacity / total_charge_capacity) * 100

# Output the Coulombic efficiency
print(f'Coulombic Efficiency: {coulombic_efficiency:.2f}%')
print(f'Total Charge Capacity: {total_charge_capacity:.5f} Ah')
print(f'Total Discharge Capacity: {total_discharge_capacity:.5f} Ah')

