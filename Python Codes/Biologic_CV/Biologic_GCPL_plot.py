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


# from Functions import plotNyquist_calcRohm
# np.set_printoptions(suppress=True)

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
    return data

def readMPTData_CV2(filename):
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

        print(header_num)
        header_num = header_num - 6
    data = pd.read_csv(filename, header=header_num, sep='\t',
                       engine='python', encoding='cp1252')

    print(data.head(5))
    return data

def readMPTData_CV3(filename):
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

        print(header_num)
        header_num = header_num - 5
    data = pd.read_csv(filename, header=header_num, sep='\t',
                       engine='python', encoding='cp1252')

    print(data.head(5))
    return data

def plotNyquist_calcRohm(data, i, label, offset):
    """
    Nyquist Plot of cycle i

    Parameters
    ----------
    data : dataframe
    i : # of cycle
    label : label on plot
    offset : y offset in Nyquist plot

    Returns
    -------
    xfirst : R ohmic

    """
    df0 = data.loc[data['cycle number'] == i]
    # print(df0)
    df = df0.loc[df0['-Im(Z)/Ohm'] >= 0]

    mValueIndex = df[['-Im(Z)/Ohm']].idxmin()
    # print(mValueIndex)

    m_mum = data.iloc[mValueIndex]
    # print(m_mum)
    # print ('minimum is at', m_mum)

    firstline = pd.DataFrame(m_mum)

    # firstline = df.iloc[mValueIndex:]

    yf = firstline['-Im(Z)/Ohm']
    xf = firstline['Re(Z)/Ohm']
    xf = float(xf)
    yf = float(yf)
    yf = yf + offset
    print('minimum is at ', xf, yf)

    # df = df0
    # print(df)
    Real = df['Re(Z)/Ohm']
    Imag = df['-Im(Z)/Ohm']

    # freq= df['freq/Hz']
    # print(freq)
    # offset = 0
    offset = offset

    Imag = Imag + offset

    # axD.plot(Real,Imag+(i*0.01), '-o',markersize = 4, label = 'Discharge step #'+str(i) )
    plt.plot(Real, Imag + (i * 0.01), '-o', markersize=4, label=label)
    # axD.plot(Real,Imag+(i*0.01), '-o', label = label)

    # dflargefreq = df.loc[df['freq/Hz'] >= 900 ]
    # largefreq = dflargefreq.loc[dflargefreq['freq/Hz'] <= 1300 ]
    # yl = largefreq['-Im(Z)/Ohm']
    # xl = largefreq['Re(Z)/Ohm']
    # yl = yl + offset
    # plt.plot(xl,yl, 'ko',markersize = 4, label= '')

    # dfsmallfreq = df.loc[df['freq/Hz'] <= 1.2 ]
    # smallfreq = dfsmallfreq.loc[dfsmallfreq['freq/Hz'] >= .85 ]
    # ys = smallfreq['-Im(Z)/Ohm']
    # xs = smallfreq['Re(Z)/Ohm']
    # ys = ys + offset
    # plt.plot(xs,ys, 'ko',markersize = 4, label= '')

    # smallestfreq = df.loc[df['freq/Hz'] <= .0105 ]
    # ysest = smallestfreq['-Im(Z)/Ohm']
    # xsest = smallestfreq['Re(Z)/Ohm']
    # ysest = ysest + offset
    # plt.plot(xsest,ysest, 'ko',markersize = 4, label= '')

    plt.plot(xf, yf, 'ko', markersize=3, label='')
    ## xfirst='{:.3f}'.format(xf)
    xfirst = round(xf, 3)
    # # print(xfirst)
    # if label == 'cell C3':
    #     plt.annotate(xfirst, (xf+15000, yf+50), color='b', fontsize=9)
    # else:
    # plt.annotate(xfirst, (xf-10, yf+100), color='k', fontsize=9)
    # # axD.annotate(xfirst, (xf-10, yf+30), color='b', fontsize=9)

    # return xfirst

def add_CV_data_charge(data):
    """
    This function adds a column to the dataframe that indicates the direction of the CV cycle

    Parameters
    ----------
    data : dataframe of mpt data

    Returns
    -------
    data : dataframe of mpt data with added column

    """
    data['charge'] = 'discharge'
    data.loc[data['<I>/mA'] > 0, 'charge'] = 'charge'
    return data

def add_CV_data_direction(data):
    """
    This function adds a column to the dataframe that indicates the direction of the CV cycle

    Parameters
    ----------
    data : dataframe of mpt data

    Returns
    -------
    data : dataframe of mpt data with added column

    """
    data['direction'] = 'forward'
    data.loc[data['Ewe/V'] < data['Ewe/V'].shift(), 'direction'] = 'reverse'
    return data

def add_CV_redox_step(data):
    """
    This function adds a column to the dataframe that indicates the redox step of the CV cycle

    Parameters
    ----------
    data : dataframe of mpt data

    Returns
    -------
    data : dataframe of mpt data with added column

    """
    redox_step = [0]
    current_state = 'charge'
    for i, row in enumerate(data.iterrows()):
        #print(current_state)
        if row[1]['charge'] != current_state:
            print('change in charge step')
            redox_step.append(redox_step[-1]+1)
            current_state = row[1]['charge']
        else:
            redox_step.append(redox_step[-1])
    data['redox step'] = redox_step[1:]
    return data

def calc_CV_capacity_sweep(data):
    """
    This function calculates the capacity of a CV sweep

    Parameters
    ----------
    data : dataframe of mpt data

    Returns
    -------
    capacity : capacity of the CV sweep

    """
    capacity = data['<I>/mA'].sum()
    return capacity

def cumulative_current_of_step(data,active_mass=0.00015):
    """
    This function calculates the cumulative current of each charge step in the dataframe

    Parameters
    ----------
    data : dataframe of mpt data

    Returns
    -------
    data : dataframe of mpt data with added column

    """
    data['integral'] = data['<I>/mA']*data['time/s'].diff()
    data['cumulative current of Step (mAh)'] = data.groupby(['charge', 'redox step'])['integral'].cumsum()/(60*60)
    data['cumulative current of Step (mAh/g)'] = data['cumulative current of Step (mAh)']/active_mass
    return data

# ===========
# MAIN PROGRAM
# ===========
#fig3 = plt.figure()
#axD = fig3.add_subplot(111)

active_mass = 0.006

# plt.rcParams['axes.linewidth'] = 2
# plt.rcParams['lines.linewidth'] = 2
##plt.rcParams.update({'font.size': 22})
# plt.rcParams['font.size'] = 15
#
# mpl.rc_file_defaults()

"""
axD.set_prop_cycle(
    color=['indianred', 'mediumpurple', 'tab:blue', 'lightsalmon', 'gold', 'limegreen', 'seagreen', 'turquoise',
           'tab:blue', 'mediumpurple', 'orchid', 'pink'])
"""
# axD.set_prop_cycle(color = ['mediumpurple','indianred', 'lightsalmon', 'gold',  'limegreen','seagreen', 'turquoise', 'tab:blue',  'mediumpurple',  'orchid', 'pink'])
# axD.set_prop_cycle(color = [ 'khaki', 'limegreen','seagreen', 'turquoise', 'tab:blue',  'mediumpurple',  'orchid', 'pink'])
# axD.set_prop_cycle(color = [ 'khaki, 'turquoise', 'tab:blue',  'mediumpurple',  'orchid', 'pink'])




#
# axD.annotate('1000 Hz', (140, 120), fontsize=9)
# axD.annotate('1 Hz', (75, 20), fontsize=9)
# axD.annotate('0.1 Hz', (210, 1650), fontsize=9)
"""

axD.set_ylabel("'Ewe/V'", fontweight='bold')
axD.set_xlabel("mAh/g", fontweight='bold')
label = 'Ch/0.5M Zn-TFSI GF/Zn - 012824-01'
title = 'Cycling Performance for '+label+' at 10 mA/g'
axD.set_title(title, fontweight='bold')
#axD.set_xlim(0, 1)
#axD.set_ylim(-0.25, 0.25)


 #upper right
#axD.legend(frameon=True, bbox_to_anchor=(.65, 0.2), loc='best', ncol=1, borderaxespad=0, fontsize=10)



axD.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
axD.tick_params(which='minor', direction='in', left=True, right=True, length=3)
"""
# **** Get data
label = 'Ch/GPE/Zn 022524-03 @ 50C'
for i in range(1):
    root = tk.Tk()
    root.withdraw()
    file_path_1 = filedialog.askopenfilename(filetypes=[("MPT files", "*.mpt")])
    #file_path_1 = filedialog.askopenfilename()
    file = file_path_1
    if i !=2:
        data = readMPTData_CV3(file)
    else:
        data = readMPTData_CV(file)
    #label = 'CV Sweep Step #'+str(i+1)
    #label = 'Ch/0.5M Zn-TFSI GF/Zn - 012824-02'
    print(label)
    num_cycles = data['cycle number'].max()
    i_max = data['<I>/mA'][2:].max()
    print(str(i_max)+'imax')
    i_min = data['<I>/mA'][2:].min()
    print(str(i_min)+'imin')
    print(num_cycles)
    """
    add_CV_data_charge(data)
    add_CV_data_direction(data)
    add_CV_redox_step(data)
    cumulative_current_of_step(data)
    print(data.head(5))
    #plt.plot(data['Ewe/V'], data['<I>/mA'], '*-', markersize=2, label=label, color='blue')
    for name, group in data.groupby(['charge', 'redox step']):
        plt.plot(abs(group['cumulative current of Step (mAh/g)']), group['Ewe/V'], '-o', markersize=2, label=label+str(name))
    """
import tkinter as tk
from tkinter import simpledialog

def get_dataset_label():
    root = tk.Tk()
    # Hide the main window
    root.withdraw()
    # Create a simple dialog
    dataset_label = simpledialog.askstring(title="Dataset Label",
                                           prompt="Enter the label of the dataset:")
    return dataset_label

# Use the function
label = get_dataset_label()
print(f"The label of the dataset is: {label}")
"""
root = tk.Tk()
root.withdraw()
file_path_1 = filedialog.askopenfilename()
file = file_path_1
data = readMPTData_CV(file)
label = 'EMD/GPE/Zn 0920-02'
num_cycles = data['cycle number'].max()
#plt.plot((data['time/s']-data['time/s'][0])[2:]/3600, data['I/mA'][2:], '-', markersize=2,)
"""
"""
#dataset = data[data['cycle number'] == 1.0]
#data_1stcycl_01 = data[data['cycle number'] == 1.0]

#file = '0805_02_EIS_CV_0.5MTFSI_ZnSym_InsideGB_0hr_RT_-2.5--2.5V_t1_02_CV_C01.mpt'
#data = readMPTData_CV(file)
#label = 'GPE 0805-02'
#data_1stcycl_02 = data[data['cycle number'] == 1.0]
# **** Plotting



# fig3, (axD, ax) = plt.subplots(1,2,figsize=(5,5))


# x = np.array([0,100,250])
# axD.xaxis.set_ticks()
# axD.xaxis.set_ticklabels(x)

#axD.set_aspect('equal', adjustable='box')
# ax.set_aspect('equal', adjustable='box')
"""

color = 'green'
# Set tick parameters
plt.rc('xtick', direction='in', top=True)
plt.rc('ytick', direction='in', right=True)
plt.rc('xtick.minor', visible=True, size=3)
plt.rc('ytick.minor', visible=True, size=3)
# Set grid style
plt.rc('grid', linestyle='-', linewidth='0.5', color='black')
# Increase the font size for the axes
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels

#plt.plot(data_1stcycl_02['Ewe/V'], data_1stcycl_02['<I>/mA'], '-o',markersize = 4, label = label)
cycles_to_plot = range(1, int(num_cycles)+1)
# Create a figure with two subplots side by side
#fig, (ax_combined, axD) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



# Create a figure
fig = plt.figure(figsize=(15, 5))

# Define the gridspec
gs = gridspec.GridSpec(1, 10)  # 1 row, 3 columns

# Create the subplots
ax_combined = plt.subplot(gs[0, 0:6])  # First row, all columns except the last one
axD= plt.subplot(gs[0, 7:])  # First row, last column
# Set aspect ratio for axD
#axD.set_aspect('equal')

"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Create a figure for axD
fig1 = plt.figure(figsize=(7.5, 5))
axD = fig1.add_subplot(111)

# Create a figure for ax_combined
fig2 = plt.figure(figsize=(7.5, 5))
ax_combined = fig2.add_subplot(111)
"""
# Continue with your plotting code...
# Continue with your plotting code...
grouped = data[data['cycle number'].isin(cycles_to_plot)].groupby('cycle number')
max_discharge_per_cycle = grouped['Q discharge/mA.h'].max()/active_mass
max_charge_per_cycle = grouped['Q charge/mA.h'].max()/active_mass
# Calculate the bounds for max_charge_per_cycle
lower_bound_charge = max_charge_per_cycle.min() * 0.8
upper_bound_charge = max_charge_per_cycle.max() * 1.5

# Use the between() function to get a mask of the rows where max_charge_per_cycle is between lower_bound_charge and upper_bound_charge
mask_charge = max_charge_per_cycle.between(lower_bound_charge, upper_bound_charge)

# Use the mask to get a new Series with only the rows where max_charge_per_cycle is between lower_bound_charge and upper_bound_charge
filtered_max_charge_per_cycle = max_charge_per_cycle[mask_charge]

# Calculate the bounds for max_discharge_per_cycle
lower_bound_discharge = max_discharge_per_cycle.min() * 0.8
upper_bound_discharge = max_discharge_per_cycle.max() * 1.5

# Use the between() function to get a mask of the rows where max_discharge_per_cycle is between lower_bound_discharge and upper_bound_discharge
mask_discharge = max_discharge_per_cycle.between(lower_bound_discharge, upper_bound_discharge)

# Use the mask to get a new Series with only the rows where max_discharge_per_cycle is between lower_bound_discharge and upper_bound_discharge
filtered_max_discharge_per_cycle = max_discharge_per_cycle[mask_discharge]

"""
filtered_max_discharge_per_cycle.iloc[1:-1].plot(label='Discharge', symbol='o')
filtered_max_charge_per_cycle.iloc[1:-1].plot(label='Charge', symbol='x')


filtered_max_discharge_per_cycle.plot(kind='scatter', x=filtered_max_discharge_per_cycle.index,
                                      y=filtered_max_discharge_per_cycle.values, label='Discharge', color=color, marker='o')
filtered_max_charge_per_cycle.plot(kind='scatter', x=filtered_max_charge_per_cycle.index,
                                      y=filtered_max_charge_per_cycle.values, label='Charge', color=color, marker='x')
"""
axD.plot(filtered_max_discharge_per_cycle.index, filtered_max_discharge_per_cycle.values,'o',  label='Discharge', color=color)
axD.plot(filtered_max_charge_per_cycle.index, filtered_max_charge_per_cycle.values,'x', label='Charge', color='blue')
axD.set_xlabel('Cycle Number (#)')
axD.set_ylabel('Capacity (mAh/g)')
#axD.set_title('Capacity vs. Cycle Number ' +label)
axD.set_ylim([0, 1.5*max_charge_per_cycle.max()])
axD.legend(frameon=True, borderaxespad=0, fontsize=10,)# loc='lower center')
# Add major grid lines
axD.grid(which='major', linestyle='-', linewidth='0.5', color='black')

# Add minor grid lines
axD.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

# Customize the minor grid
axD.minorticks_on()


import matplotlib.pyplot as plt
marker_types = ['-', '--', '-.', ':', (0, (5,2)), (0, (3,2,1,2)), (0, (3,2,1,2,1,2)),
                (0, (3,2,1,2,1,2,1,2)), (0, (3,2,1,2,1,2,1,2,1,2))]
plt.rc('lines', linewidth=1.5)
color = 'green'
# List of cycles to plot
cycles_to_plot = [1, 2, 10, 20, 30, num_cycles-1]

# Filter the dataframe for the specific cycles
filtered_data = data[data['cycle number'].isin(cycles_to_plot)]

# Exclude rows where Q discharge is 0 for discharge and Q charge is 0 for charge
filtered_data_discharge = filtered_data[filtered_data['Q discharge/mA.h'] != 0]
filtered_data_charge = filtered_data[filtered_data['Q charge/mA.h'] != 0]
# Exclude rows where current is 0
filtered_data_discharge = filtered_data_discharge[filtered_data_discharge['<I>/mA'] != 0]
filtered_data_charge = filtered_data_charge[filtered_data_charge['<I>/mA'] != 0]
# Group the filtered dataframe by the cycle number
grouped_discharge = filtered_data_discharge.groupby('cycle number')
grouped_charge = filtered_data_charge.groupby('cycle number')





# Plot the charge and discharge curves for each cycle on separate plots
for cycle, group in grouped_discharge:
    #discharge = group['Q discharge/mA.h']/active_mass
    discharge = (group['time/s']-group['time/s'].iloc[0])/3600
    discharge_V = group['Ewe/V']
    #ax_discharge.plot(discharge, discharge_V, label=f'Discharge Cycle {cycle}')
    #ax_combined.plot(discharge, discharge_V, color=color, linestyle=marker_types[cycles_to_plot.index(cycle)], label=f'Discharge Cycle {cycle}')
    ax_combined.plot(discharge, discharge_V, color=color, linestyle=marker_types[cycles_to_plot.index(cycle)])

for cycle, group in grouped_charge:
    charge = group['Q charge/mA.h']/active_mass
    charge_V = group['Ewe/V']
    #ax_charge.plot(charge, charge_V, label=f'Charge Cycle {cycle}')
    #ax_combined.plot(charge, charge_V, color=color, linestyle=marker_types[cycles_to_plot.index(cycle)], label=f'Charge Cycle {cycle}')
    ax_combined.plot(charge, charge_V, color=color, linestyle=marker_types[cycles_to_plot.index(cycle)],
                     label=f'Cycle {cycle}')

"""
for cycle, group in grouped_discharge:
    discharge = group['Q discharge/mA.h']/active_mass
    discharge_V = group['Ewe/V']
    charge = group['Q charge/mA.h'] / active_mass
    charge_V = group['Ewe/V']
    ax_combined.plot(discharge, discharge_V, label=f'Discharge Cycle {cycle}')
    ax_combined.plot(charge, charge_V, label=f'Charge Cycle {cycle}')


ax_discharge.set_ylabel('Voltage vs. Zn/Zn2+ (V)')
ax_discharge.set_xlabel('Capacity (mAh/g)')
ax_discharge.legend()
ax_discharge.set_title('Discharge Curves for Multiple Cycles of '+label)

ax_charge.set_ylabel('Voltage vs. Zn/Zn2+ (V)')
ax_charge.set_xlabel('Capacity (mAh/g)')
ax_charge.legend()
ax_charge.set_title('Charge Curves for Multiple Cycles of '+label)
"""


# Plot on axD
# ... (your code to plot on axD)

# Plot on ax_combined
# ... (your code to plot on ax_combined)

# Show the plot
#upper_limit = 7
#ax_combined.set_xlim([0, upper_limit])
#axD.set_ylim([0, upper_limit])
#axD.set_aspect('equal')
#ax_combined.set_ylabel('Voltage vs. Zn/Zn2+ (V)', fontsize=10)
#ax_combined.set_xlabel('Capacity (mAh/g)')
# Increase the size and make the x and y labels bold
axD.set_xlabel('Cycle Number (#)', fontsize=14, labelpad=10, weight='bold')
axD.set_ylabel('Capacity (mAh/g)', fontsize=14, labelpad=10, weight='bold')

ax_combined.set_ylabel('Voltage vs. Zn/Zn2+ (V)', fontsize=14, labelpad=10, weight='bold')
ax_combined.set_xlabel('Capacity (mAh/g)', fontsize=14, labelpad=10, weight='bold')

ax_combined.legend(frameon=True, bbox_to_anchor=(1.05, 0.5) , loc='upper center', borderaxespad=0, fontsize=10)
#ax_combined.set_title('Charge and Discharge Curves for Multiple Cycles of '+label)
# Add major grid lines
ax_combined.grid(which='major', linestyle='-', linewidth='0.5', color='black')

# Add minor grid lines
ax_combined.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

# Customize the minor grid
ax_combined.minorticks_on()

# Increase the font size for the legend
#axD.legend(frameon=True, borderaxespad=0, fontsize=20) # fontsize of the legend in axD
#ax_combined.legend(frameon=True, bbox_to_anchor=(1.05, 0.5) , loc='upper center', borderaxespad=0, fontsize=20) # fontsize of the legend in ax_combined
#axD.set_title('Capacity vs. Cycle Number ' +label, fontsize=16, fontweight='bold')
#ax_combined.set_title('Charge and Discharge Curves for Multiple Cycles of '+label, fontsize=16, fontweight = 'bold')
fig.suptitle('Cycling Performance of '+label, fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show()

"""
# Create a list of all the lines and labels
lines = axD.get_lines() + ax_combined.get_lines()
labels = [line.get_label() for line in lines]

# Create a legend in the middle of the figure
fig.legend(lines, labels, loc='center')
"""

if __name__ == '__main__':
    pass