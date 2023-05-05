import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog

class arbin_import():

    def __init__(self, path, name='cell', mass=.003, color = 'r', shape = 'o'):
        self.path = path
        self.data = self.read_data()
        self.name = name
        self.mass = mass
        self.color = color
        self.shape = shape
        self.cycle_Num_list = ['1:C/10', '2:C/10', '3:C/10', '4:C/5', '5:C/5', '6:C/5', '7:C/2', '8:C/2', '9:C/2',]


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
            self.max_capacity.append(mAh/self.mass*175)
            i = i + 1
            if i==7:
                break

        print(self.max_cap)
        print(self.max_cap.keys())
        plt.scatter(self.cycle_Num_list[0:len(self.max_capacity)], self.max_capacity, c=self.color, marker='*' ,label=self.name)
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
            self.max_dis_capacity.append(mAh / self.mass * 175)
            i = i + 1
            if i == 7:
                break

        print(self.max_dis_cap)
        print(self.max_dis_cap.keys())
        plt.scatter(self.cycle_Num_list[0:len(self.max_dis_capacity)], self.max_dis_capacity, c=self.color, marker=self.shape, label=self.name)
        #plt.show()
    def get_coulombic_efficiency(self):
        self.ce = []
        i = 1
        for mAh in self.max_dis_cap:
            self.ce.append(self.max_capacity[i]/self.max_dis_capacity[i])
            i = i + 1
            if i == 7:
                break
        print(self.ce)
        plt.scatter(self.cycle_Num_list[0:len(self.ce)], self.ce, c=self.color, marker=self.shape, label=self.name)
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

if __name__ == '__main__':

    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    """
    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'w', 'r', 'b', 'g', 'y', 'c', 'm', 'k', 'w']
    data = []
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
        #datum.get_coulombic_efficiency()
    #data.get_max_capacity_per_cycle()
    plt.legend()
    plt.xlabel('Cycle Number:C-Rate')
    plt.ylabel('Capacity (Charge and Discharge, mAh/g)')
    plt.title('Capacity vs. Cycle Number')
    plt.xticks(fontsize=10)
    plt.savefig('NMC622 2_spacer_calendared_ch_dc.png', dpi=500, bbox_inches='tight')
    plt.show()
