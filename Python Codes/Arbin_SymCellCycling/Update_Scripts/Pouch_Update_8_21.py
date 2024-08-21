import pandas as pd
import matplotlib.pyplot as plt

# Load the first dataset (Uncompressed)
file_path_1 = r'E:\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Results\Low Temp Li Ion\2024\07\0718\BL-LL-AV01_-21C_discharge_02_GCPL_C05.mpt'
data_skip_74 = pd.read_csv(file_path_1, delimiter='\t', skiprows=74, encoding='ISO-8859-1')

# Load the second dataset (Compressed)
file_path_2 = r'E:\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\08\BL-LL-AV01_-21C_Dis_t1_2024_08_06_104108/BL-LL-AV01_-21C_Dis_t1_Channel_41_Wb_1.xlsx'
data_new_cleaned = pd.read_excel(file_path_2, sheet_name=1)

# Filter the datasets where the current is negative (discharge phase)
data_uncompressed_negative = data_skip_74[data_skip_74['<I>/mA'] < 0]
data_compressed_negative = data_new_cleaned[data_new_cleaned['Current (A)'] < 0]

# Plotting the data from both sources on the same plot
fig, ax1 = plt.subplots(1,1,figsize=(4.6, 3.5))

# Plot from the first dataset (Uncompressed)
plt.plot(data_uncompressed_negative['(Q-Qo)/mA.h'].abs(), data_uncompressed_negative['Ewe/V'], label='Uncompressed')

# Plot from the second dataset (Compressed)
plt.plot(data_compressed_negative['Discharge Capacity (Ah)'].abs() * 1000, data_compressed_negative['Voltage (V)'], label='Compressed')

plt.xlabel('Absolute Capacity (mAh)')
plt.ylabel('Voltage (V)')
plt.title('Voltage vs. Discharge Capacity')
plt.legend()
plt.tight_layout()
#plt.grid(True)
plt.show()
