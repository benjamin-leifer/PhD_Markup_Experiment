import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Constants
d = 20e-6  # Thickness of the cell in meters (20 µm)
S = 0.000201  # Electrode area in square meters (adjust as necessary)
os.chdir(r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Results\Low Temp Li Ion\2025\01\Blocking Cells\MPT')
print(os.getcwd())

# Load the data (adjust the file name and columns as needed)
# The data should include 'Temperature (K)', 'Resistance (Ohms)', and 'Cell'
file_path = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Results\Low Temp Li Ion\2025\01\Blocking Cells\MPT\2025_01_15_Blocking_Experiment.xlsx"
data = pd.read_excel(file_path)

# Split the dataset into 4 datasets by 'Cell'
cell_groups = data.groupby('Cell')
datasets = {cell: cell_groups.get_group(cell) for cell in cell_groups.groups}

map = {
    'CR01': 'DT14',
    'CS01': 'DTF14',
    'CT01': 'LP',
    'CU01': 'LPV'
}
# Plot Conductivity vs 1000/T for each dataset
plt.figure(figsize=(4.6, 3.5))
for cell, dataset in datasets.items():
    # Extract temperature and resistance
    temperature = dataset['Temperature (K)']
    resistance = dataset['Resistance (Ohms)']

    # Calculate conductivity (S/m)
    conductivity = d / (resistance * S)

    # Calculate 1000/T
    inv_temp = 1000 / temperature

    # Plot Conductivity vs 1000/T
    plt.plot(inv_temp, conductivity, marker='o', linestyle='-', label=map.get(cell, cell))

plt.xlabel("1000/T (K$^{-1}$)")
plt.ylabel("Conductivity (S/m)")
plt.yscale('log')
plt.title("Ionic Conductivity vs 1000/T for Different Cells")
plt.grid(True)
plt.legend()
plt.show()

# Optional: Save the processed data to a new CSV file for each cell
for cell, dataset in datasets.items():
    temperature = dataset['Temperature (K)']
    resistance = dataset['Resistance (Ohms)']
    conductivity = d / (resistance * S)
    inv_temp = 1000 / temperature
    output_data = pd.DataFrame({
        'Temperature (K)': temperature,
        'Resistance (Ohms)': resistance,
        'Conductivity (S/m)': conductivity,
        '1000/T (K^-1)': inv_temp
    })
    output_data.to_csv(f"processed_eis_data_{cell}.csv", index=False)
    print(f"Processed data saved to 'processed_eis_data_{cell}.csv'")