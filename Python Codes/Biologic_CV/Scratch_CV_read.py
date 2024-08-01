import numpy as np
import pylab
import glob
import pandas as pd
import scipy.optimize as optimize
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
# from Functions import readMPTData
from Functions import plotNyquist
from Functions import getmin
from Functions import plotLSV
import tkinter as tk
from tkinter import filedialog
import Arbin_SymCell

def readMPTData_CV(filename):
    """
This function reads a biologic .mpt data file

    Parameters
    ----------
    filename : Name including .mpt of a biologic file to read

    Returns
    -------
    data : dataframe of mpt data

    """
    # def readMPTData(filename, header_num):

    #     data = pd.read_csv(filename, header= header_num, sep='\t',
    #                       engine= 'python', encoding='cp1252')

    # Open the file
    with open(filename, 'r', encoding='cp1252') as readfile:
        header_rows = readfile.readlines(18)
        header_rows = str(header_rows).split()
        df = pd.DataFrame(header_rows)
        header_num = df.iloc[7]
        header_num = int(header_num)

        # print(header_num)
        header_num = header_num - 4
    data = pd.read_csv(filename, header=header_num, sep='\t',
                       engine='python', encoding='cp1252')

    print(data.head(5))
    data.to_csv(filename+'_CV_data.csv')
    return data

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()  # to hide the main window

    file_path = filedialog.askopenfilename()  # open the dialog to choose directory
    #readMPTData_CV(file_path)
    #readMPTData_CV(file_path)

    import pandas as pd
    import numpy as np

    # Load the CSV file
    #file_path = 'path_to_csv_file.csv'

    cv_data = pd.read_csv(file_path)


    # Function to calculate capacities by cycle step
    def calculate_capacities_by_step(cv_data):
        cycle_steps = cv_data['cycle number'].unique()

        charge_capacities_by_step = {}
        discharge_capacities_by_step = {}

        for step in cycle_steps:
            step_data = cv_data[cv_data['cycle number'] == step]
            time_step = step_data['time/s'] / 3600
            current_step = step_data['<I>/mA']

            charge_current_step = np.where(current_step > 0, current_step, 0)
            discharge_current_step = np.where(current_step < 0, current_step, 0)

            charge_capacity_step = np.trapz(charge_current_step, time_step)
            discharge_capacity_step = np.trapz(discharge_current_step, time_step)

            charge_capacities_by_step[step] = abs(charge_capacity_step)
            discharge_capacities_by_step[step] = abs(discharge_capacity_step)

        return charge_capacities_by_step, discharge_capacities_by_step


    # Calculate capacities by cycle step
    charge_capacities_by_step, discharge_capacities_by_step = calculate_capacities_by_step(cv_data)

    # Convert the results to a dataframe for better readability
    capacities_by_step = pd.DataFrame({
        'Cycle Step': list(charge_capacities_by_step.keys()),
        'Charge Capacity (mA.h)': list(charge_capacities_by_step.values()),
        'Discharge Capacity (mA.h)': list(discharge_capacities_by_step.values())
    })

    # Display the results
    print("Capacities by Cycle Step:\n", capacities_by_step)
    #calculate mAh/g
    active_mass = 0.0007
    capacities_by_step['Charge Capacity (mAh/g)'] = capacities_by_step['Charge Capacity (mA.h)'] / active_mass
    capacities_by_step['Discharge Capacity (mAh/g)'] = capacities_by_step['Discharge Capacity (mA.h)'] / active_mass
    capacities_by_step['Mass Calc'] = capacities_by_step['Discharge Capacity (mA.h)']/140

    # Set the option to display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)


    print("Capacities by Cycle Step (mAh/g):\n", capacities_by_step)


