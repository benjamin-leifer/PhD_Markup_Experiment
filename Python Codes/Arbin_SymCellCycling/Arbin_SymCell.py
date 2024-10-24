import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from fractions import Fraction

# Create a custom legend handler
class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle([x0, y0], width, height, facecolor='red',
                                   edgecolor='black', hatch='xx', lw=3,
                                   transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch

class Cell_Cycle():

    def __init__(self, data, mass = 0.00133, theoretical_cap = 175, color = 'r', shape = 'o'):
        self.data = data
        self.mass = mass
        self.color = color
        self.shape = shape
        self.theoretical_cap = theoretical_cap
        self.C_Rate = None
        self.C_Rate_str = None
        self.charge_capacity = None
        self.discharge_capacity = None
        self.cycle_index = self.data['Cycle Index'].max()
        self.ingest_data()

    def lowest_c_rate_str(num, den):
        fraction = Fraction(num, den)
        str_frac = str(fraction)
        c_rate = str_frac.split('/')
        print(c_rate)
        C_Rate = 'C/' + c_rate[1]
        print(C_Rate)
        return C_Rate

    def ingest_data(self):
        self.get_c_rate()
        self.get_charge_capacity()
        self.get_discharge_capacity()
        self.get_ce()

    def get_c_rate(self):
        C_Rate = round(np.average(self.data['Current (A)'] / (self.mass*self.theoretical_cap)), 3)
        print('C Rate: ', C_Rate)
        print(C_Rate)
        print(type(C_Rate))
        self.C_Rate = C_Rate
        #self.C_Rate_str = self.lowest_c_rate_str(num = C_Rate, den= 1)
        return C_Rate

    def get_charge_capacity(self):
        charge_capacity = self.data['Charge Capacity (Ah)'].max()
        print('Charge Capacity: ', charge_capacity)
        self.charge_capacity = charge_capacity
        return charge_capacity

    def get_discharge_capacity(self):
        discharge_capacity = self.data['Discharge Capacity (Ah)'].max()
        print('Discharge Capacity: ', discharge_capacity)
        self.discharge_capacity = discharge_capacity
        return discharge_capacity

    def get_ce(self):
        try:
            ce = self.discharge_capacity/self.charge_capacity
        except ZeroDivisionError:
            ce = 0
        print('Coulombic Efficiency: ', ce)
        print('Cycle Index: ', self.cycle_index)
        return ce




class arbin_import_Sym_Cell():
    def __init__(self, path, name='cell', mass=.00133, theoretical_cap=175, num_cycles=None, color='r', shape='o'):
        self.path = path
        self.data = self.read_data()
        self.name = name
        self.mass = mass
        self.color = color
        self.shape = shape
        self.cycle_Num_list = ['1:C/10', '2:C/10', '3:C/10', '4:C/5', '5:C/5', '6:C/5', '7:C/2', '8:C/2', '9:C/2']
        self.theoretical_cap = theoretical_cap
        self.cycles = self.data['Cycle Index'].max()
        if not num_cycles:
            self.num_cycles = min(self.data['Cycle Index'].max(), 22)  # Limit to 22 cycles
        self.cycles_objs = []
        self.instantiate_cycle_list()
        matplotlib_colors = [
            'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w',  # basic colors
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',  # tab colors
            'xkcd:sky blue', 'xkcd:seafoam green', 'xkcd:hot pink', 'xkcd:lime green', 'xkcd:lavender', 'xkcd:bright orange', 'xkcd:light brown', 'xkcd:pale green', 'xkcd:dark purple', 'xkcd:mauve',  # xkcd colors
        ]
        self.color_list = matplotlib_colors
        print('List of all cycles is:')
        print(self.cycles_objs)

    def instantiate_cycle_list(self):
        grouped_data = self.data.groupby('Cycle Index')
        self.cycles_objs = [Cell_Cycle(group, mass=self.mass, theoretical_cap=self.theoretical_cap, color=self.color, shape=self.shape) for name, group in list(grouped_data)[:22]]



    def read_data(self):

        if self.get_filetype() == '.csv':
            data = pd.read_csv(self.path, header=0, engine='python')
        elif self.get_filetype() == '.CSV':
            data = pd.read_csv(self.path, header=0, engine='python')
            print(data.head())
        elif self.get_filetype() == '.xlsx':
            data = pd.read_excel(self.path, header=0, sheet_name=1)
        print(data.head())
        print(data.keys())
        return data

    def plot_voltage_vs_time(self):
        fig, ax1 = plt.subplots()

        color = self.color
        ax1.set_xlabel('Test Time (s)')
        ax1.set_ylabel('Voltage (V)', color=color)
        ax1.plot(self.data['Test Time (s)']/3600, self.data['Voltage (V)'], label=self.name)
        ax1.tick_params(axis='y', labelcolor=color)



        ax1.tick_params(which='both', axis='both', direction='in', bottom=True, top=True, left=True, right=False)
        ax1.set_xlabel('Time (hr)')
        ax1.set_ylabel('Voltage (V)')
        plt.title('Voltage vs. Time for %s' % self.name)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        # plt.show()

    def plot_voltage_and_current_vs_time(self):
        fig, ax1 = plt.subplots()

        color = self.color
        ax1.set_xlabel('Test Time (s)')
        ax1.set_ylabel('Voltage (V)', color=color)
        ax1.plot(self.data['Test Time (s)'], self.data['Voltage (V)'], label=self.name)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Current (A)', color=color)  # we already handled the x-label with ax1
        ax2.plot(self.data['Test Time (s)'], self.data['Current (A)'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        ax1.tick_params(which='both', axis='both', direction='in', bottom=True, top=True, left=True, right=False)
        ax2.tick_params(which='both', axis='both', direction='in', bottom=False, top=False, left=False, right=True)

        plt.title('Voltage and Current vs. Time for %s' % self.name)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        #plt.show()

    def plot_voltage_vs_capacity(self, cycles='All', clean_filter=False):
        fig, ax1 = plt.subplots()

        # Group the data by cycle
        grouped_data = self.data.groupby('Cycle Index')

        if clean_filter:
            # Group the data by cycle
            cleaned_data = pd.DataFrame()
            for name, group in grouped_data:
                if (group['Charge Capacity (Ah)'].max()/self.mass < 50 or
                        group['Discharge Capacity (Ah)'].max()/self.mass < 50):
                    pass
                else:
                    cleaned_data = pd.concat([cleaned_data, group])
            try:
                grouped_data = cleaned_data.groupby('Cycle Index')
            except:
                print('No cycles with capacity greater than 50 mAh/g')
                return


        if cycles == 'All':
            i=0
            for name, group in grouped_data:
                # Filter the data where the amperage is positive (for charging)
                filtered_group_charge = group[group['Current (A)'] > 0]
                filtered_group_dis = group[group['Current (A)'] < 0]

                charge_cap = filtered_group_charge['Charge Capacity (Ah)']
                charge_volt = filtered_group_charge['Voltage (V)']
                dis_cap = filtered_group_dis['Discharge Capacity (Ah)']
                dis_volt = filtered_group_dis['Voltage (V)']

                color = self.color
                ax1.set_xlabel('Capacity (mAh/g)')
                ax1.set_ylabel('Voltage (V)', color=color)
                #ax1.plot(charge_cap/self.mass, charge_volt, color=self.color_list[i], linestyle='dashed', label=self.name + ' Charge Cycle ' + str(name))
                ax1.plot(charge_cap/self.mass, charge_volt, linestyle='dashed', label=self.name + ' Charge Cycle ' + str(name))
                #ax1.plot(dis_cap/self.mass, dis_volt, color=self.color_list[i], linestyle='dotted', label=self.name + ' Discharge Cycle ' + str(name))
                ax1.plot(dis_cap/self.mass, dis_volt, linestyle='dotted', label=self.name + ' Discharge Cycle ' + str(name))
                i=i+1
        else:
            i=0
            for cycle in cycles:
                group = grouped_data.get_group(cycle)
                filtered_group_charge = group[group['Current (A)'] > 0]
                filtered_group_dis = group[group['Current (A)'] < 0]

                charge_cap = filtered_group_charge['Charge Capacity (Ah)']
                charge_volt = filtered_group_charge['Voltage (V)']
                dis_cap = filtered_group_dis['Discharge Capacity (Ah)']
                dis_volt = filtered_group_dis['Voltage (V)']

                color = self.color
                ax1.set_xlabel('Capacity (mAh/g)')
                ax1.set_ylabel('Voltage (V)', color=color)
                ax1.plot(charge_cap/self.mass, charge_volt, color=self.color_list[i], linestyle='dashed', label=self.name + ' Charge Cycle ' + str(cycle))
                ax1.plot(dis_cap/self.mass, dis_volt, color=self.color_list[i], linestyle='dotted', label=self.name + ' Discharge Cycle ' + str(cycle))
                i=i+1

        ax1.tick_params(axis='y', labelcolor=color)
        plt.title('Voltage vs. Capacity for %s' % self.name)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2) # Add this line to show the legend

        """
        fig, ax1 = plt.subplots()


        discharge_cap = self.data['Discharge Capacity (Ah)']
        dis_volt = self.data['Voltage (V)']

        charge_cap = self.data['Charge Capacity (Ah)']
        charge_volt = self.data['Voltage (V)']

        color = self.color
        ax1.set_xlabel('Capacity (mAh/g)')
        ax1.set_ylabel('Voltage (V)', color=color)
        ax1.plot(self.data['Discharge Capacity (Ah)']/self.mass, self.data['Voltage (V)'], color=color, label=self.name)
        ax1.plot(self.data['Charge Capacity (Ah)']/self.mass, self.data['Voltage (V)'], color=color, label=self.name)
        ax1.tick_params(axis='y', labelcolor=color)

        #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        #color = 'tab:blue'
        #ax2.set_ylabel('Current (A)', color=color)  # we already handled the x-label with ax1
        #ax2.plot(self.data[], self.data['Current (A)'], color=color)
        #ax2.tick_params(axis='y', labelcolor=color)

        ax1.tick_params(which='both', axis='both', direction='in', bottom=True, top=True, left=True, right=False)
        #ax2.tick_params(which='both', axis='both', direction='in', bottom=False, top=False, left=False, right=True)

        plt.title('Voltage vs. Capacity for %s' % self.name)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        """

    def get_overpotentials(self):
        self.overpotentials = []

        self.cycles = self.data.groupby(['Cycle Index'])

        i = 0
        for voltage in self.data['Voltage (V)']:
            self.overpotentials.append(voltage - self.data['Voltage (V)'][i])
            i = i + 1
            if i == self.break_point:
                break
        print(self.overpotentials)
        plt.scatter(self.cycle_Num_list[0:len(self.overpotentials)], self.overpotentials, c=self.color, marker=self.shape, label=self.name+' Overpotentials')
        #plt.show()


        pass

    def plot_voltage_vs_capacity_absolute(self, cycles='All', clean_filter=False):
        fig, ax1 = plt.subplots()

        # Group the data by cycle
        grouped_data = self.data.groupby('Cycle Index')

        if clean_filter:
            # Group the data by cycle
            cleaned_data = pd.DataFrame()
            for name, group in grouped_data:
                if (group['Charge Capacity (Ah)'].max() / self.mass < 50 or
                        group['Discharge Capacity (Ah)'].max() / self.mass < 50):
                    pass
                else:
                    cleaned_data = pd.concat([cleaned_data, group])
            try:
                grouped_data = cleaned_data.groupby('Cycle Index')
            except:
                print('No cycles with capacity greater than 50 mAh/g')
                return

        if cycles == 'All':
            for name, group in grouped_data:
                # Filter the data where the amperage is positive (for charging)
                filtered_group_charge = group[group['Current (A)'] > 0]
                filtered_group_dis = group[group['Current (A)'] < 0]

                charge_cap = filtered_group_charge['Charge Capacity (Ah)']
                charge_volt = filtered_group_charge['Voltage (V)']
                dis_cap = filtered_group_dis['Discharge Capacity (Ah)']
                dis_volt = filtered_group_dis['Voltage (V)']

                color = self.color
                ax1.set_xlabel('Capacity (mAh)')
                ax1.set_ylabel('Voltage (V)', color=color)
                ax1.plot(charge_cap*1000, charge_volt, linestyle='dashed',
                         label=self.name + ' Charge Cycle ' + str(name))
                ax1.plot(dis_cap*1000, dis_volt, linestyle='dotted',
                         label=self.name + ' Discharge Cycle ' + str(name))
        else:
            for cycle in cycles:
                group = grouped_data.get_group(cycle)
                filtered_group_charge = group[group['Current (A)'] > 0]
                filtered_group_dis = group[group['Current (A)'] < 0]

                charge_cap = filtered_group_charge['Charge Capacity (Ah)']
                charge_volt = filtered_group_charge['Voltage (V)']
                dis_cap = filtered_group_dis['Discharge Capacity (Ah)']
                dis_volt = filtered_group_dis['Voltage (V)']

                color = self.color
                ax1.set_xlabel('Capacity (mAh)')
                ax1.set_ylabel('Voltage (V)', color=color)
                ax1.plot(charge_cap*1000, charge_volt, linestyle='dashed',
                         label=self.name + ' Charge Cycle ' + str(cycle))
                ax1.plot(dis_cap*1000, dis_volt, color=color, linestyle='dotted',
                         label=self.name + ' Discharge Cycle ' + str(cycle))

        ax1.tick_params(axis='y', labelcolor=color)
        plt.title('Voltage vs. Capacity for %s' % self.name)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend()  # Add this line to show the legend


    def get_max_capacity_per_cycle(self):
        #split data into cycles
        #find max capacity for each cycle
        #return max capacity for each cycle
        cycles = self.data.groupby(['Cycle Index'])
        self.max_cap = cycles['Discharge Capacity (Ah)'].max()
        i =0
        cycles = []
        self.max_capacity = []
        for mAh in self.max_cap:
            print(i)
            cycles.append(i)
            print(mAh)
            self.max_capacity.append(mAh/self.mass)
            i = i + 1
            if i==self.num_cyles:
                break
        print('Charge Capacity')
        print(self.max_cap)
        print(self.max_cap.keys())
        plt.scatter(self.cycle_Num_list[0:len(self.max_capacity)], self.max_capacity, c=self.color, marker='*' ,label=self.name+ ' Charge Capacity')
        #plt.show()

    def get_min_capacity_per_cycle(self):
        # split data into cycles
        # find max capacity for each cycle
        # return max capacity for each cycle
        cycles = self.data.groupby(['Cycle Index'])
        self.max_dis_cap = cycles['Charge Capacity (Ah)'].max()
        i = 0
        cycles = []
        self.max_dis_capacity = []
        for mAh in self.max_dis_cap:
            print(i)
            cycles.append(i)
            print(mAh)
            self.max_dis_capacity.append(mAh/self.mass)
            i = i + 1
            if i == self.num_cyles:
                break
        print('Discharge Capacity')
        print(self.max_dis_cap)
        print(self.max_dis_cap.keys())
        plt.scatter(self.cycle_Num_list[0:len(self.max_dis_capacity)], self.max_dis_capacity, c=self.color, marker=self.shape, label=self.name+ ' Discharge Capacity')
        #plt.show()
    def get_coulombic_efficiency(self):
        self.ce = []
        i = 0
        for mAh in self.max_dis_cap:
            try:
                self.ce.append(self.max_capacity[i]/self.max_dis_capacity[i])
            except ZeroDivisionError:
                print('divied by zero error')
                self.ce.append(0)
            i = i + 1
            #if i == self.break_point: break
        print(self.ce)
        plt.scatter(self.cycle_Num_list[0:len(self.ce)], self.ce, c=self.color, marker=self.shape, label=self.name+' Coulombic Efficiency')
        #plt.show()

    def plot_capacity_and_ce_vs_cycle(self):

        # Create a new figure and an axis for the left y-axis
        fig, ax1 = plt.subplots()

        cycles = self.data.groupby(['Cycle Index'])
        charge_cap = []
        dis_cap = []
        cycle_num = []
        for i, (cycle_index, cycle_data) in enumerate(cycles):
            cycle_num.append(i+1)
            charge_cap.append(cycle_data['Charge Capacity (Ah)'].max()/self.mass)
            dis_cap.append(cycle_data['Discharge Capacity (Ah)'].max()/self.mass)

        line1, = ax1.plot(cycle_num, charge_cap, 'o', label='Charge Capacity')
        line2, = ax1.plot(cycle_num, charge_cap, '.', label='Discharge Capacity')
        # Create a second axis for the right y-axis that shares the same x-axis
        ax2 = ax1.twinx()

        coulombic_efficiency = []
        for (i, j) in zip(charge_cap, dis_cap):
            try:
                coulombic_efficiency.append(j/i*100)
            except ZeroDivisionError:
                coulombic_efficiency.append(0)

        #coulombic_efficiency = [i / j for i, j in zip(charge_cap, dis_cap)]

        # Plot the Coulombic efficiency on the right y-axis
        line3, = ax2.plot(cycle_num, coulombic_efficiency, '*', color='g', label='Coulombic Efficiency')

        # Set the labels for the x-axis and both y-axes
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Capacity (mAh/g)', color='black')
        ax2.set_ylabel('Coulombic Efficiency', color='g')
        ax2.set_ylim(0, 120)
        first_legend = plt.legend(handles=[line1, line2, line3],
                                  loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
        ax = plt.gca().add_artist(first_legend)
        #plt.legend(handles=[ax2], loc='upper right')
        #ax2.legend(loc='lower left')
        plt.title('Capacity and Coulombic Efficiency vs. Cycle Number for %s' % self.name)

        plt.tight_layout()

    def get_data(self):
        return self.data

    def get_path(self):
        return self.path

    def get_filename(self):
        return os.path.basename(self.path)

    def get_filetype(self):
        return os.path.splitext(self.path)[1]

    def get_filesize(self):
        return os.path.getsize(self.path)

    def find_max_voltage(self):
        return self.data['Voltage (V)'].max()

    def find_min_voltage(self):
        return self.data['Voltage (V)'].min()

    def find_max_current(self):
        return self.data['Current (A)'].max()

    def find_min_current(self):
        return self.data['Current (A)'].min()

    def get_full_charges(self):
        cycles = self.data.groupby(['Cycle Index'])
        print(cycles.head())
import re

def find_bl_ll_xx00(input_string):
    # Define the regex pattern for BL-LL-XX00
    pattern = r'\w{2}\d{2}'

    # Search for the pattern in the input string
    match = re.search(pattern, input_string)

    # Return the matched pattern if found, else return None
    return match.group(0) if match else None

# Example usage
example_string = "This is a test string with BL-LL-AX02 in it."
result = find_bl_ll_xx00(example_string)
print(result)  # Output: BL-LL-AX02

if __name__ == '__main__':
    """
    root = tk.Tk()
    root.withdraw()
    file_path_1 = filedialog.askopenfilename()
    
    root = tk.Tk()
    root.withdraw()
    file_path_2 = filedialog.askopenfilename()
    """
    import tkinter as tk
    from tkinter import filedialog
    import os
    import shutil

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
            #if 'Wb' in file and file.endswith('.CSV') or 'Wb' in file and file.endswith('.xslx'):
            if file.endswith('.xlsx') or file.endswith('.CSV'):
                csv_files.append(os.path.join(root, file))
                print(csv_files)
                # Create a label tag from the first 8 characters of the filename
                label_tag = find_bl_ll_xx00(file)
                print('Label Tag is: ')
                print(label_tag)

                # Create a new Tkinter window
                #root = tk.Tk()

                # Hide the main window
                #root.withdraw()

                # Prompt the user to enter text
                #user_input = tk.simpledialog.askstring(title="Input", prompt="Please enter your text:")

                # Add the user input to the label_tag string
                #label_tag += user_input

                # Now label_tag contains the original label_tag string plus the user's input
                #print(label_tag)
                #print(label_tag)
                # Create an arbin_import_Sym_Cell object for each file
                # cells.append(arbin_import_Sym_Cell(file, name=label_tag, mass=0.01430645161/1000,
                #                               theoretical_cap=155, color=colors[i], shape='o'))

                #cells.append(arbin_import_Sym_Cell(file, name=label_tag, mass=0.01293303225 / 1000,
                #                                 theoretical_cap=155, color='black', shape='o'))

    print('csv files are: ')
    print(csv_files)
    for cell in csv_files:
        label_tag = find_bl_ll_xx00(cell)
        try:
            cells.append(arbin_import_Sym_Cell(cell, name=label_tag, mass=40/155/1000,
                                           theoretical_cap=155, color='black', shape='o'))
        except FileNotFoundError as e:
            print('File not found')
    print(cells)
    # Create a new directory
    from datetime import datetime

    # Get the current date and time
    now = datetime.now()

    # Convert the current date and time to a string
    current_date_time = now.strftime("%Y-%m-%d %H_%M_%S")

    #print(current_date_time)
    new_dir_name = 'new_directory '+current_date_time
    os.mkdir(new_dir_name)

    # Change the current working directory to the new directory
    os.chdir(new_dir_name)

    for cell in cells:
        if cell.cycles < 1000:

            cell.plot_voltage_vs_time()
            plt.savefig(cell.name + '_voltage_vs_time.png', dpi=500, bbox_inches='tight')
            cell.plot_voltage_vs_capacity(clean_filter=False)
            plt.savefig(cell.name + '_voltage_vs_capacity.png', dpi=500, bbox_inches='tight')
            plt.clf()
            #cell.get_max_capacity_per_cycle()
            #cell.get_min_capacity_per_cycle()
            #cell.get_coulombic_efficiency()
            cell.plot_capacity_and_ce_vs_cycle()
            plt.savefig(cell.name + '_coulombic_efficiency.png', dpi=500, bbox_inches='tight')
        #plt.savefig(cell.get_filename() + '_voltage_vs_time.png', dpi=500, bbox_inches='tight')
        #plt.show()
        #cell.get_max_capacity_per_cycle()
        #cell.get_min_capacity_per_cycle()
        #plt.show()
        #cell.get_coulombic_efficiency()
        #plt.show()
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

    def plot_combined_capacity_and_ce(cells):
        fig, ax1 = plt.subplots()

        for cell in cells:
            cycles = cell.data.groupby(['Cycle Index'])
            charge_cap = []
            dis_cap = []
            cycle_num = []
            for i, (cycle_index, cycle_data) in enumerate(cycles):
                if i >= 40:
                    break
                cycle_num.append(i + 1)
                charge_cap.append(cycle_data['Charge Capacity (Ah)'].max() / cell.mass)
                dis_cap.append(cycle_data['Discharge Capacity (Ah)'].max() / cell.mass)

            coulombic_efficiency = []
            for (i, j) in zip(charge_cap, dis_cap):
                try:
                    coulombic_efficiency.append(j / i * 100)
                except ZeroDivisionError:
                    coulombic_efficiency.append(0)

            #ax1.plot(cycle_num, charge_cap, 'o', label=f'{cell.name} Charge Capacity')
            ax1.plot(cycle_num, dis_cap, '.', label=f'{cell.name} Discharge Capacity')
            # Define cycle rates and their ranges
        cycle_rates = [
                (4, 'C/10'),
                (3, 'C/8'),
                (3, 'C/4'),
                (3, 'C/2'),
                (3, '1C'),
                (3, '2C'),
                (20, 'C/10')  # Remaining cycles
            ]

        # Add vertical lines for each cycle rate change
        cycle_start = 0
        for cycles, rate in cycle_rates:
            cycle_start += cycles
            ax1.axvline(x=cycle_start, color='k', linestyle='--')
            ax1.text(cycle_start-2.5, ax1.get_ylim()[1]/1.05 , rate, rotation=0, verticalalignment='bottom')

        #ax2 = ax1.twinx()
            #ax2.plot(cycle_num, coulombic_efficiency, '*', label=f'{cell.name} Coulombic Efficiency', color='g')

        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Capacity (mAh/g)', color='black')
        #ax2.set_ylabel('Coulombic Efficiency (%)', color='g')
        #ax2.set_ylim(0, 120)
        plt.title('Discharge Capacity vs. Cycle Number')
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
        plt.tight_layout()
        plt.show()

    # Assuming `cells` is a list of `Cell_Cycle` objects
    #plot_combined_capacity_and_ce(cells)
    """
    root = tk.Tk()
    root.withdraw()
    file_list = []
    for i in range(1):
        file_path = filedialog.askopenfilename()
        file_list.append(file_path)
    print(file_list)

    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'w', 'r', 'b', 'g', 'y', 'c', 'm', 'k', 'w']
    data = []
    for i, file in enumerate(file_list):
        data.append(arbin_import(file_list[i], name='Cell0'+str(4), mass=0.004113, theoretical_cap=160, color=colors[4], shape='o'))
    #Capacity 1 -> 0.004113
    #Capacity 2 -> 0.003339

    
    data.append(arbin_import(
        'All Cycling Results/BL_20230401_01_2023_04_01_163122/BL_20230401_01_Channel_51_Wb_1.CSV',
        name='Cell01 - Calendared',
        mass=0.003866,
        color=colors[1],
        shape='s'))
    data.append(arbin_import(
        'All Cycling Results/BL_20230401_02_2023_04_01_163137/BL_20230401_02_Channel_52_Wb_1.CSV',
        name='Cell02 - Calendared',
        mass=0.003866,
        color=colors[2],
        shape='s'))
    

    for datum in data:
        datum.get_max_capacity_per_cycle()
        datum.get_min_capacity_per_cycle()
        datum.get_coulombic_efficiency()
    #data.get_max_capacity_per_cycle()
    plt.legend()
    plt.xlabel('Cycle Number:C-Rate')
    #plt.ylabel('Capacity (Charge and Discharge, mAh/g)')
    #plt.title('Capacity vs. Cycle Number - NMC622 2 Spacer')
    plt.ylabel('Coulombic Efficiency')
    plt.title('Coulombic Efficiency vs. Cycle Number - NMC622 2 Spacer')
    plt.xticks(fontsize=10)
    plt.legend(loc='lower right')
    #plt.savefig('NMC622 2_spacer_calendared_ch_dc.png', dpi=500, bbox_inches='tight')
    plt.show()
    """