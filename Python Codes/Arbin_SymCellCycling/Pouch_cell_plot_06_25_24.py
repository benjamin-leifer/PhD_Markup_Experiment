import Arbin_SymCell as arb

import tkinter as tk
from tkinter import filedialog
import os
import shutil
import matplotlib.pyplot as plt

root = tk.Tk()
root.withdraw()  # to hide the main window

folder_selected = filedialog.askdirectory()  # open the dialog to choose directory
os.chdir(folder_selected)  # change the current working directory to the selected folder

# Get a list of all files in the directory
files = os.listdir()
print(files)
cells = []
colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'w', 'r', 'b', 'g', 'y', 'c', 'm', 'k', 'w']
# Iterate over each file
"""
for i, file in enumerate(files):
    # Create a label tag from the first 8 characters of the filename
    label_tag = file[:10]


    # Create a new Tkinter window
    root = tk.Tk()

    # Hide the main window
    root.withdraw()

    # Prompt the user to enter text
    user_input = tk.simpledialog.askstring(title="Input", prompt="Please enter your text:")

    # Add the user input to the label_tag string
    label_tag += user_input

    # Now label_tag contains the original label_tag string plus the user's input
    print(label_tag)
    print(label_tag)
    # Create an arbin_import_Sym_Cell object for each file
    #cells.append(arbin_import_Sym_Cell(file, name=label_tag, mass=0.01430645161/1000,
    #                               theoretical_cap=155, color=colors[i], shape='o'))

    cells.append(arbin_import_Sym_Cell(file, name=label_tag, mass=0.01293303225/1000,
                               theoretical_cap=155, color=colors[i], shape='o'))
"""
# Get a list of all files in the directory and its subdirectories
csv_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if 'Wb' in file and file.endswith('.CSV'):
            csv_files.append(os.path.join(root, file))
            # Create a label tag from the first 8 characters of the filename
            label_tag = file[:10]

            # Create a new Tkinter window
            # root = tk.Tk()

            # Hide the main window
            # root.withdraw()

            # Prompt the user to enter text
            # user_input = tk.simpledialog.askstring(title="Input", prompt="Please enter your text:")

            # Add the user input to the label_tag string
            # label_tag += user_input

            # Now label_tag contains the original label_tag string plus the user's input
            # print(label_tag)
            # print(label_tag)
            # Create an arbin_import_Sym_Cell object for each file
            # cells.append(arbin_import_Sym_Cell(file, name=label_tag, mass=0.01430645161/1000,
            #                               theoretical_cap=155, color=colors[i], shape='o'))

            # cells.append(arbin_import_Sym_Cell(file, name=label_tag, mass=0.01293303225 / 1000,
            #                                 theoretical_cap=155, color='black', shape='o'))

print('csv files are: ')
print(csv_files)
for i, cell in enumerate(csv_files):
    label_tag = cell[2:12]
    try:
        cells.append(arb.arbin_import_Sym_Cell(cell, name=label_tag, mass=0.01293303225 / 1000,
                                           theoretical_cap=155, color=colors[i], shape='o'))
    except FileNotFoundError as e:
        print('File not found')
print(cells)
# Create a new directory
from datetime import datetime

# Get the current date and time
now = datetime.now()

# Convert the current date and time to a string
current_date_time = now.strftime("%Y-%m-%d %H_%M_%S")

# print(current_date_time)
new_dir_name = 'new_directory ' + current_date_time
os.mkdir(new_dir_name)

# Change the current working directory to the new directory
os.chdir(new_dir_name)

for cell in cells:
    if cell.cycles < 10:
        cell.plot_voltage_vs_capacity_absolute()
        plt.savefig(cell.get_filename() + '_voltage_vs_capacity.png', dpi=500, bbox_inches='tight')
        plt.clf()

        # cell.get_max_capacity_per_cycle()
        # cell.get_min_capacity_per_cycle()
        # cell.get_coulombic_efficiency()

    # plt.savefig(cell.get_filename() + '_voltage_vs_time.png', dpi=500, bbox_inches='tight')
    # plt.show()
    # cell.get_max_capacity_per_cycle()
    # cell.get_min_capacity_per_cycle()
    # plt.show()
    # cell.get_coulombic_efficiency()
    # plt.show()
    """
    try:
        #cell.plot_voltage_vs_capacity()
        cell.get_max_capacity_per_cycle()
        cell.get_min_capacity_per_cycle()
        cell.get_coulombic_efficiency()
        plt.show()
    except:
        pass
    """
    #plt.show()
#fig2, (ax12) = plt.subplots(1,1,figsize=(2.78,2.78))
fig, ax1 = plt.subplots(1,1,figsize=(4.6, 3.5))
ax1.set_xlabel('Capacity (mAh)')
ax1.set_ylabel('Voltage (V)')
for cell in cells:
    print(cell.name+':')
    print('Charge Capacity: '+str(cell.data['Charge Capacity (Ah)'].max()*1000))
    print('Discharge Capacity: '+str(cell.data['Discharge Capacity (Ah)'].max()*1000))
    grouped_data = cell.data.groupby('Cycle Index')
    for name, group in grouped_data:
        # Filter the data where the amperage is positive (for charging)
        filtered_group_charge = group[group['Current (A)'] > 0]
        filtered_group_dis = group[group['Current (A)'] < 0]

        charge_cap = filtered_group_charge['Charge Capacity (Ah)']
        charge_volt = filtered_group_charge['Voltage (V)']
        dis_cap = filtered_group_dis['Discharge Capacity (Ah)']
        dis_volt = filtered_group_dis['Voltage (V)']
    #ax1.plot(charge_cap * 1000, charge_volt, color=cell.color, linestyle='dashed',
    #        label=cell.name + ' Charge Cycle ' + str(cell.name))
    ax1.plot(dis_cap * 1000, dis_volt, color=cell.color, linestyle='solid',
             label='Discharge Cycle ' + str(cell.name))

plt.title('Voltage vs. Capacity for all Pouch Cells')
ax1.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
ax1.tick_params(which='minor', direction='in', left=True, right=True, length=3)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend()  # Add this line to show the legend
plt.savefig('Voltage_vs_Capacity.png', dpi=500, bbox_inches='tight')
plt.show()