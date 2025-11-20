import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 14,           # default font size
    'axes.labelsize': 18,      # x/y label size
    'axes.titlesize': 18,      # title size
    'xtick.labelsize': 16,     # x tick label size
    'ytick.labelsize': 16,     # y tick label size
    'legend.fontsize': 14
})
import matplotlib.patheffects as pe
import os

# Define the new working directory path
new_dir_path = r'C:\Users\benja\Downloads\10_22_2025 Temp\Slide Trials\Data\Update 0724\RT'

# Change the current working directory
os.chdir(new_dir_path)

# Verify the current working directory
print("Current Working Directory: ", os.getcwd())

file_paths = [
    #'BL-LL-AR01_Channel_1_Wb_1.xlsx',  # (DTF Gr sample – still excluded below if re-enabled)
    'BL-LL-AS01_Channel_4_Wb_1.xlsx',    # Gr||NMC – DT14
    'BL-LL-AT03_Channel_9_Wb_1.xlsx',    # Li||NMC – DTF14 (will be excluded)
    'BL-LL-AU02_Channel_11_Wb_1.xlsx'    # Li||NMC – DT14
]

# --------------------
# Helpers (unchanged)
# --------------------
def plot_by_cycle_number(dataframes, active_mass_g):
    plt.figure(figsize=(15, 10))
    for file_path, df in dataframes.items():
        df['Charge Capacity (mAh/g)'] = (df['Charge Capacity (Ah)'] * 1000) / active_mass_g
        df['Discharge Capacity (mAh/g)'] = (df['Discharge Capacity (Ah)'] * 1000) / active_mass_g

        cycle_numbers = df['Cycle Index'].unique()
        for cycle in cycle_numbers:
            cycle_data = df[df['Cycle Index'] == cycle]
            charge_data = cycle_data[cycle_data['Current (A)'] > 0]
            discharge_data = cycle_data[cycle_data['Current (A)'] < 0]

            plt.plot(charge_data['Charge Capacity (mAh/g)'], charge_data['Voltage (V)'],
                     label=f'Charge Cycle {cycle} {file_path}', linestyle='-')
            plt.plot(discharge_data['Discharge Capacity (mAh/g)'], discharge_data['Voltage (V)'],
                     label=f'Discharge Cycle {cycle} {file_path}', linestyle='--')

    plt.xlabel('Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage vs. Capacity (mAh/g) by Cycle Number for Multiple Batteries')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.show()


# Active mass in grams
active_mass_g = 0.01293303225

# -------------------------------
# Main plot with "reddish" scheme
# -------------------------------
import matplotlib.patheffects as pe

def plot_cycle_2_with_reds(dataframes, active_mass_g):
    """
    Excludes any DTF samples and plots only DT14:
      - Li||NMC – DT14: deep red, thicker line (no markers)
      - Gr||NMC – DT14: light red, thinner line (no markers)
    Charge = solid; Discharge = dashed.
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(4.6*1.5, 3.5*1.5))

    legends = {
        #'BL-LL-AR01_Channel_1_Wb_1.xlsx': 'Gr||NMC - DTF14',
        'BL-LL-AS01_Channel_4_Wb_1.xlsx': 'Gr||NMC - DT14',
        'BL-LL-AT03_Channel_9_Wb_1.xlsx': 'Li||NMC - DTF14',
        'BL-LL-AU02_Channel_11_Wb_1.xlsx': 'Li||NMC - DT14'
    }

    # Reddish palette
    RED_DEEP  = "#A50F15"  # Li||NMC – DT14
    RED_LIGHT = "#FB6A4A"  # Gr||NMC – DT14

    style_map = {
        'Li||NMC - DT14': dict(color=RED_DEEP,  lw=5,
                               path_effects=[pe.Stroke(linewidth=6, foreground='white'), pe.Normal()]),
        'Gr||NMC - DT14': dict(color=RED_LIGHT, lw=4,
                               path_effects=None),
    }

    for file_path, df in dataframes.items():
        label = legends[file_path]

        # Exclude any DTF samples
        if "DTF" in label.upper():
            continue

        df = df.copy()
        df['Charge Capacity (mAh/g)']    = (df['Charge Capacity (Ah)'] * 1000) / active_mass_g
        df['Discharge Capacity (mAh/g)'] = (df['Discharge Capacity (Ah)'] * 1000) / active_mass_g

        # Cycle 2
        cycle_data = df[df['Cycle Index'] == 2]
        charge_data    = cycle_data[cycle_data['Current (A)'] > 0]
        discharge_data = cycle_data[cycle_data['Current (A)'] < 0]

        s = style_map[label]

        # Charge (solid)
        ax1.plot(charge_data['Charge Capacity (mAh/g)'], charge_data['Voltage (V)'],
                 linestyle='-', label=f'{label[:-6]}', color=s['color'], lw=s['lw'],
                 path_effects=s['path_effects'])

        # Discharge (dashed)
        ax1.plot(discharge_data['Discharge Capacity (mAh/g)'], discharge_data['Voltage (V)'],
                 linestyle='--', color=s['color'], lw=s['lw'],
                 path_effects=s['path_effects'])

    ax1.set_xlabel('Capacity (mAh/g)', fontsize='16', fontweight='bold')
    ax1.set_ylabel('Voltage (V)', fontsize='16', fontweight='bold')
    #ax1.set_title('Voltage vs. Capacity (mAh/g) – Cycle 2 (DT14 only)')
    #ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True, shadow=True, fontsize='small')
    ax1.legend(loc='best',fontsize='14')
    ax1.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='minor', direction='in', left=True, right=True, length=3)
    #ax1.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()



# --------------------
# Load the data (sheet 2)
# --------------------
dataframes = {}
for file_path in file_paths:
    sheet_data = pd.read_excel(file_path, sheet_name=1)
    dataframes[file_path] = sheet_data

# --------------------
# Plot: DT14 only, reddish scheme
# --------------------
plot_cycle_2_with_reds(dataframes, active_mass_g)
