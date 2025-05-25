import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog

class arbin_import():

    def __init__(self, path, name='cell', mass=.003, theoretical_cap = 175, num_cycles = None, color = 'r', shape = 'o'):
        self.path = path
        self.data = self.read_data()
        self.name = name
        self.mass = mass
        self.color = color
        self.shape = shape
        self.cycle_Num_list = ['1:C/10', '2:C/10', '3:C/10', '4:C/5', '5:C/5', '6:C/5', '7:C/2', '8:C/2', '9:C/2',]
        self.theoretical_cap = theoretical_cap
        if not num_cycles:
            self.num_cyles = self.data['Cycle Index'].max()



    def read_data(self):
        data = pd.read_csv(self.path, header=0, engine='python')
        print(data.head())
        print(data.keys())
        return data

    def plot_voltage_vs_time(self):
        plt.plot(self.data['Test Time (s)'], self.data['Voltage (V)'], 'r', label='Voltage (V)')
        plt.legend()
        plt.xlabel('Test Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Voltage vs. Time')
        #self.data.plot(x='Test Time (s)', y='Voltage (V)')
        plt.show()
    def get_ocv_and_max_voltage(self):
        self.ocv = self.data['Voltage (V)'].iloc[0]
        self.max_voltage = self.data['Voltage (V)'].max()
        print(self.ocv)
        print(self.max_voltage)
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
            self.max_capacity.append(mAh/self.mass*self.theoretical_cap)
            i = i + 1
            if i==self.num_cyles:
                break

        print(self.max_cap)
        print(self.max_cap.keys())
        #plt.scatter(self.cycle_Num_list[0:len(self.max_capacity)], self.max_capacity, c=self.color, marker='*' ,label=self.name+ ' Charge Capacity')
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
            self.max_dis_capacity.append(mAh / self.mass * self.theoretical_cap)
            i = i + 1
            if i == self.num_cyles:
                break

        print(self.max_dis_cap)
        print(self.max_dis_cap.keys())
        #plt.scatter(self.cycle_Num_list[0:len(self.max_dis_capacity)], self.max_dis_capacity, c=self.color, marker=self.shape, label=self.name+ ' Discharge Capacity')
        #plt.show()
    def get_coulombic_efficiency(self):
        self.ce = []
        i = 0
        for mAh in self.max_dis_cap:
            self.ce.append(self.max_capacity[i]/self.max_dis_capacity[i])
            i = i + 1
            if i == self.break_point:
                break
        print(self.ce)
        plt.scatter(self.cycle_Num_list[0:len(self.ce)], self.ce, c=self.color, marker=self.shape, label=self.name+' Coulombic Efficiency')
        #plt.show()

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

if __name__ == '__main__':


    root = tk.Tk()
    root.withdraw()
    file_list = []
    for i in range(2):
        file_path = filedialog.askopenfilename()
        file_list.append(file_path)
    print(file_list)

    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'w', 'r', 'b', 'g', 'y', 'c', 'm', 'k', 'w']
    data = []
    for i, file in enumerate(file_list):
        data.append(arbin_import(file_list[i], name='Cell0'+str(4), mass=0.004113, theoretical_cap=160, color=colors[4], shape='o'))
    #Capacity 1 -> 0.004113
    #Capacity 2 -> 0.003339

    """
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
    """
    for datum in data:
        print(datum.get_filename())
        datum.get_ocv_and_max_voltage()
    """
    #for datum in data:
        #datum.get_max_capacity_per_cycle()
        #datum.get_min_capacity_per_cycle()
        #datum.get_coulombic_efficiency()
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