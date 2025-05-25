import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to handle dynamic header row extraction and file loading
def dynamic_load_ec_lab(file_path):
    try:
        # Open the file to determine the header rows dynamically
        with open(file_path, 'r', encoding='cp1252') as readfile:
            header_rows = readfile.readlines(18)
            header_content = str(header_rows).split()
            header_num = int(header_content[7]) - 4  # Adjust based on header structure

        # Load the file with the dynamic header row count
        data = pd.read_csv(file_path, header=header_num, sep='\t', engine='python', encoding='cp1252')
        return data
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

os.chdir(r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Results\Low Temp Li Ion\2025\01\LSV')
print(os.getcwd())

# File paths for processing
file_paths = [
    'BL-LL-CO03_LSV_40C_C02.mpt',
    'BL-LL-CP01_LSV_RT_C05.mpt',
    'BL-LL-CP02_LSV_-21C_C01.mpt',
    'BL-LL-CP03_LSV_40C_C03.mpt',
    'BL-LL-CQ01_LSV_RT_C05.mpt',
    'BL-LL-CQ02_LSV_-21C_C01.mpt',
    'BL-LL-CQ03_LSV_40C_C04.mpt',
    'BL-LL-CO01_LSV_RT_C05.mpt',
    'BL-LL-CO02_LSV_-21C_C01.mpt',
    'BL-LL-CN01_LSV_RT_t2_C05.mpt',
    'BL-LL-CN02_LSV_-21C_C01.mpt',
    'BL-LL-CN03_LSV_40C_C01.mpt'
]

# Parse all files
parsed_data = []
for path in file_paths:
    parsed_file = dynamic_load_ec_lab(path)
    if parsed_file is not None:
        series_name = path.split('/')[-1].split('_')[0]  # Extract series name
        parsed_file['series'] = series_name
        parsed_data.append(parsed_file)

# Combine all data into a single DataFrame
all_data = pd.concat(parsed_data, ignore_index=True)

# Mapping suffix numbers to temperature labels
temperature_mapping = {
    '01': 'RT',
    '02': '-21C',
    '03': '40C'
}

# Mapping prefixes to electrolyte labels
electrolyte_mapping = {
    'BL-LL-CN0': 'DT14',
    'BL-LL-CO0': 'DTF14',
    'BL-LL-CP0': 'LP',
    'BL-LL-CQ0': 'LPV'
}

# Grouping by prefix (e.g., BL-LL-CO, BL-LL-CN, etc.) and plotting
prefix_groups = all_data['series'].str.slice(0, 9).unique()  # Extract first 9 characters as prefix

for prefix in prefix_groups:
    subset = all_data[all_data['series'].str.startswith(prefix)]
    electrolyte_label = electrolyte_mapping.get(prefix, prefix)  # Map prefix to electrolyte
    plt.figure(figsize=(4.6, 3.5))
    for series in subset['series'].unique():
        # Extract temperature suffix (last 2 characters before underscore) and map to label
        temp_suffix = series[-2:]
        temp_label = temperature_mapping.get(temp_suffix, temp_suffix)  # Default to suffix if not mapped
        series_data = subset[subset['series'] == series]
        plt.plot(series_data['Ewe/V'], series_data['<I>/mA'], label=temp_label)
    plt.title(electrolyte_label)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (mA)')
    plt.legend()
    plt.grid(True)
    plt.show()
