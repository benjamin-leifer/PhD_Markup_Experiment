import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the new working directory path
new_dir_path = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\10\From Eric'

# Change the current working directory
os.chdir(new_dir_path)

# Verify the current working directory
print("Current Working Directory: ", os.getcwd())
# Active mass in grams
active_mass_g = 0.01293303225
# Define file paths for the newly uploaded data at -21°C
file_paths_minus_21C = [
    #'BL-LL-CD03_-51C_Dis_2_Channel_41_Wb_1.xlsx',
    #'BL-LL-CD03_RT_Form_Channel_29_Wb_1.xlsx',
    'BL-LL-CD03_-51C_Dis_Channel_41_Wb_1.xlsx',
    'BL-LL-CD07_-51C_Dis_ext_Channel_40_Wb_1.xlsx',
    'BL-LL-CE04_-51C_Dis_ext_Channel_40_Wb_1.xlsx',
    'BL-LL-CF03_-51C_Dis_ext_Channel_41_Wb_1.xlsx',
    #'BL-LL-BR03_RT_Form_2_Channel_49_Wb_1.xlsx',
    #'BL-LL-AZ04_-21C_t1_Channel_37_Wb_1.xlsx',
    #'BL-LL-AZ04_-32_t1_Channel_37_Wb_1.xlsx',
    #'BL-LL-BR03_-21C_Dis2_Channel_40_Wb_1.xlsx',
    #'BL-LL-BA02_RT_t1_Channel_47_Wb_1.xlsx',
    #'BL-LL-BA02_-21C_t1_Channel_38_Wb_1.xlsx',
    #'BL-LL-BA02_-32_t1_Channel_38_Wb_1.xlsx',
    #'BL-LL-BR03_-32C_Dis1_Channel_41_Wb_1.xlsx',
    #'BL-LL-BB04_RT_t1_Channel_55_Wb_1.xlsx',
    #'BL-LL-BB04_-21C_t1_Channel_39_Wb_1.xlsx',
    #'BL-LL-BB04_-3C_t1_Channel_39_Wb_1.xlsx',
    #'BL-LL-BR03_-51C_Dis1_Channel_41_Wb_1.xlsx',

]

# Define legends for the newly uploaded data
legends_minus_21C = {
    #'BL-LL-BR03_RT_Form_2_Channel_49_Wb_1.xlsx': 'Li||NMC - DTF14 - RT',
    #'BL-LL-CD03_-51C_Dis_2_Channel_41_Wb_1.xlsx': 'Gr||NMC - DTF14 - -51C',
    #'BL-LL-CD03_RT_Form_Channel_29_Wb_1.xlsx': 'Gr||NMC - DTF14 - RT',
    'BL-LL-CD03_-51C_Dis_Channel_41_Wb_1.xlsx': 'Gr||NMC - DTF14 - -51C',
    'BL-LL-CD07_-51C_Dis_ext_Channel_40_Wb_1.xlsx': 'Gr||NMC - DTF14 - -51C',
    'BL-LL-CE04_-51C_Dis_ext_Channel_40_Wb_1.xlsx': 'Li||NMC - DTF14 - -51C',
    'BL-LL-CF03_-51C_Dis_ext_Channel_41_Wb_1.xlsx': 'Li||NMC - DT14 - -51C',

    # 'BL-LL-AZ04_-21C_t1_Channel_37_Wb_1.xlsx',
    # 'BL-LL-AZ04_-32_t1_Channel_37_Wb_1.xlsx',
    #'BL-LL-BR03_-21C_Dis2_Channel_40_Wb_1.xlsx': 'Li||NMC - DTF14 - -21C',
    # 'BL-LL-BA02_RT_t1_Channel_47_Wb_1.xlsx',
    # 'BL-LL-BA02_-21C_t1_Channel_38_Wb_1.xlsx',
    # 'BL-LL-BA02_-32_t1_Channel_38_Wb_1.xlsx',
    #'BL-LL-BR03_-32C_Dis1_Channel_41_Wb_1.xlsx': 'Li||NMC - DTF14 - -32C',
    # 'BL-LL-BB04_RT_t1_Channel_55_Wb_1.xlsx',
    # 'BL-LL-BB04_-21C_t1_Channel_39_Wb_1.xlsx',
    # 'BL-LL-BB04_-3C_t1_Channel_39_Wb_1.xlsx',
    #'BL-LL-BR03_-51C_Dis1_Channel_41_Wb_1.xlsx': 'Li||NMC - DTF14 - -51C',
}
#: 'Li||NMC - DTF14 - 51C'

# Define function to process and plot all cycles data
def plot_all_cycles(dataframes, active_mass_g):
    fig, ax1 = plt.subplots(1,1,figsize=(4.6, 3.5))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for idx, (file_path, df) in enumerate(dataframes.items()):
        # Calculate capacity in mAh/g
        df['Charge Capacity (mAh/g)'] = (df['Charge Capacity (Ah)'] * 1000) / active_mass_g
        df['Discharge Capacity (mAh/g)'] = (df['Discharge Capacity (Ah)'] * 1000) / active_mass_g

        # Filter data for all cycles
        charge_data = df[df['Current (A)'] > 0]
        if 'RT' in file_path:
            cycle_data = df[df['Cycle Index'] == 1]
        else:
            cycle_data = df[df['Cycle Index'] == 1]
        discharge_data = cycle_data[cycle_data['Current (A)'] < 0]

        # Plot charge and discharge data with same color
        #plt.plot(charge_data['Charge Capacity (mAh/g)'], charge_data['Voltage (V)'],
        #        label=f'Charge {legends_minus_21C[file_path]}', linestyle='-', color=colors[idx])
        plt.plot(discharge_data['Discharge Capacity (mAh/g)'], discharge_data['Voltage (V)'],
                 label=f'{legends_minus_21C[file_path]}', linestyle='--', color=colors[idx])

    # Configure plot
    plt.xlabel('Capacity (mAh/g)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage vs. Capacity (mAh/g)')
    plt.legend(loc='upper right')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    ax1.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='minor', direction='in', left=True, right=True, length=3)
    #plt.tight_layout()
    plt.show()


def plot_all_cycles_normalized(dataframes, active_mass_g):
    fig, ax1 = plt.subplots(1,1,figsize=(4.6, 3.5))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for idx, (file_path, df) in enumerate(dataframes.items()):
        # Calculate capacity in mAh/g
        df['Charge Capacity (mAh/g)'] = (df['Charge Capacity (Ah)'] * 1000) / active_mass_g
        df['Discharge Capacity (mAh/g)'] = (df['Discharge Capacity (Ah)'] * 1000) / active_mass_g

        # Filter data for all cycles
        charge_data = df[df['Current (A)'] > 0]
        if 'RT' in file_path:
            cycle_data = df[df['Cycle Index'] == 1]
        else:
            cycle_data = df[df['Cycle Index'] == 1]
        discharge_data = cycle_data[cycle_data['Current (A)'] < 0]

        # Plot charge and discharge data with same color
        #plt.plot(charge_data['Charge Capacity (mAh/g)'], charge_data['Voltage (V)'],
        #        label=f'Charge {legends_minus_21C[file_path]}', linestyle='-', color=colors[idx])
        plt.plot(discharge_data['Discharge Capacity (mAh/g)']/1.462, discharge_data['Voltage (V)'],
                 label=f'{legends_minus_21C[file_path]}', linestyle='--', color=colors[idx])

    # Configure plot
    plt.xlabel('Normalized Capacity (%)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage vs. Normalized Capacity (%)')
    plt.legend(loc = 'upper right')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    ax1.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax1.tick_params(which='minor', direction='in', left=True, right=True, length=3)
    #plt.tight_layout()
    plt.show()


# Load the second sheet data from all files into a dictionary
dataframes_minus_21C = {}
for file_path in file_paths_minus_21C:
    # Load the second sheet
    sheet_data = pd.read_excel(file_path, sheet_name=1)
    dataframes_minus_21C[file_path] = sheet_data

# Plot data for all cycles at -21°C
plot_all_cycles(dataframes_minus_21C, active_mass_g)
plot_all_cycles_normalized(dataframes_minus_21C, active_mass_g)
