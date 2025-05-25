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

        # print(header_num)
        header_num = header_num - 5
    data = pd.read_csv(filename, header=header_num, sep='\t',
                       engine='python', encoding='cp1252')

    print(data.head(5))
    return data


# ===========
# MAIN PROGRAM
# ===========
fig3 = plt.figure()
axD = fig3.add_subplot(111)

# plt.rcParams['axes.linewidth'] = 2
# plt.rcParams['lines.linewidth'] = 2
##plt.rcParams.update({'font.size': 22})
# plt.rcParams['font.size'] = 15
#
# mpl.rc_file_defaults()

axD.set_prop_cycle(
    color=['indianred', 'mediumpurple', 'tab:blue', 'lightsalmon', 'gold', 'limegreen', 'seagreen', 'turquoise',
           'tab:blue', 'mediumpurple', 'orchid', 'pink'])
# axD.set_prop_cycle(color = ['mediumpurple','indianred', 'lightsalmon', 'gold',  'limegreen','seagreen', 'turquoise', 'tab:blue',  'mediumpurple',  'orchid', 'pink'])
# axD.set_prop_cycle(color = [ 'khaki', 'limegreen','seagreen', 'turquoise', 'tab:blue',  'mediumpurple',  'orchid', 'pink'])
# axD.set_prop_cycle(color = [ 'khaki, 'turquoise', 'tab:blue',  'mediumpurple',  'orchid', 'pink'])




#
# axD.annotate('1000 Hz', (140, 120), fontsize=9)
# axD.annotate('1 Hz', (75, 20), fontsize=9)
# axD.annotate('0.1 Hz', (210, 1650), fontsize=9)


axD.set_xlabel("Time(hr)", fontweight='bold')
axD.set_ylabel("Voltage (V) vs. Zn/Zn2+", fontweight='bold')
axD.set_title('Discharge for EMD/GPE/Zn Cell @ 0.5 uA', fontweight='bold')

 #upper right
#axD.legend(frameon=True, bbox_to_anchor=(.65, 0.2), loc='best', ncol=1, borderaxespad=0, fontsize=10)



axD.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
axD.tick_params(which='minor', direction='in', left=True, right=True, length=3)
# **** Get data
"""
for i in range(1):
    root = tk.Tk()
    root.withdraw()
    file_path_1 = filedialog.askopenfilename()
    file = file_path_1
    if i !=2:
        data = readMPTData_CV(file)
    else:
        data = readMPTData_CV2(file)
    #label = 'CV Sweep Step #'+str(i+1)
    label = 'EMD/GPE/Zn 1101-03'
    print(label)
    num_cycles = data['cycle number'].max()
    i_max = data['<I>/mA'][2:].max()
    print(str(i_max)+'imax')
    i_min = data['<I>/mA'][2:].min()
    print(str(i_min)+'imin')
    print(num_cycles)
    plt.plot(data['Ewe/V'], data['<I>/mA'], '-', markersize=2, label=label)
"""
# **** Get data
root = tk.Tk()
root.withdraw()
file_path_1 = filedialog.askopenfilename()
file = file_path_1
data = readMPTData_CV(file)
label = 'EMD/GPE/Zn Discharge'
plt.plot((data['time/s']-data['time/s'][0])/3600, data['Ewe/V'], '-', markersize=2, label=label)

"""
root = tk.Tk()
root.withdraw()
file_path_1 = filedialog.askopenfilename()
file = file_path_1
data = readMPTData_CV(file)
label = 'EMD/0.5M Zn-TFSI GF/Zn 1212-02 Discharge'
plt.plot((data['time/s']-data['time/s'][0])/3600, data['Ewe/V'], '-', markersize=2, label=label)

root = tk.Tk()
root.withdraw()
file_path_1 = filedialog.askopenfilename()
file = file_path_1
data = readMPTData_CV(file)
label = 'EMD/GPE/Zn 0920-02'
num_cycles = data['cycle number'].max()
#plt.plot((data['time/s']-data['time/s'][0])[2:]/3600, data['I/mA'][2:], '-', markersize=2,)
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
for cycle in range(int(num_cycles)):
    dataset = data[data['cycle number'] == cycle + 1.0]
    plt.plot(dataset['Ewe/V'], dataset['<I>/mA'], '-o',markersize = 4, label = label+' Cycle #'+str(cycle+1),)
"""
#plt.plot(data_1stcycl_02['Ewe/V'], data_1stcycl_02['<I>/mA'], '-o',markersize = 4, label = label)

axD.legend(frameon=True, borderaxespad=0, fontsize=10, bbox_to_anchor=(1.2, 0.5), loc='center')
plt.tight_layout()
plt.show()


if __name__ == '__main__':
    pass
