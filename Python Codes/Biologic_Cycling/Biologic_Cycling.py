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
        header_num = header_num -5
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

def setmatplotlib():
    axD.set_xlabel("Time (hr)", fontweight='bold')
    axD.set_ylabel("Ewe/V", fontweight='bold')
    axD.set_title('Galvanostatic Cycling of 0.5M Zn-TFSI GPE - 0825-01', fontweight='bold')
    axD.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    axD.tick_params(which='minor', direction='in', left=True, right=True, length=3)

# ===========
# MAIN PROGRAM
# ===========

# **** Get data
root = tk.Tk()
root.withdraw()
file_path_1 = filedialog.askopenfilename()
#file = 'C:\Users\benja\OneDrive - Northeastern University\Northeastern\Gallaway Group\PhD Markup Experiment\Python Codes\Alyssa_Old_Files\Functions.py\EIS Results\LL GPE\2023\0523\FullCell_EMD_ZnAnode_NoCalendar_EIS_RT_C01.mpt'  # 'MIT cell C3_01_PEIS_C03.mpt'
file = file_path_1
data = readMPTData_CV(file)
label = '0.05 mA/cm^2'
print(label)
print(data.info())

#dataset = data[data['cycle number'] == 1.0]
#data_1stcycl_01 = data[data['cycle number'] == 1.0]
cycling_start_time = data['time/s'].min()
time = (data['time/s']-cycling_start_time)/3600
# file = '0805_02_EIS_CV_0.5MTFSI_ZnSym_InsideGB_0hr_RT_-2.5--2.5V_t1_02_CV_C01.mpt'
# data = readMPTData_CV(file)
# label = 'GPE 0805-02'
#data_1stcycl_02 = data[data['cycle number'] == 1.0]
root = tk.Tk()
root.withdraw()
file_path_2 = filedialog.askopenfilename()
#file = 'C:\Users\benja\OneDrive - Northeastern University\Northeastern\Gallaway Group\PhD Markup Experiment\Python Codes\Alyssa_Old_Files\Functions.py\EIS Results\LL GPE\2023\0523\FullCell_EMD_ZnAnode_NoCalendar_EIS_RT_C01.mpt'  # 'MIT cell C3_01_PEIS_C03.mpt'
file2 = file_path_2
data2 = readMPTData_CV(file2)
label2 = '0.1 mA/cm^2'

#dataset = data[data['cycle number'] == 1.0]
#data_1stcycl_01 = data[data['cycle number'] == 1.0]
time2 = (data2['time/s']-cycling_start_time)/3600

root = tk.Tk()
root.withdraw()
file_path_3 = filedialog.askopenfilename()
#file = 'C:\Users\benja\OneDrive - Northeastern University\Northeastern\Gallaway Group\PhD Markup Experiment\Python Codes\Alyssa_Old_Files\Functions.py\EIS Results\LL GPE\2023\0523\FullCell_EMD_ZnAnode_NoCalendar_EIS_RT_C01.mpt'  # 'MIT cell C3_01_PEIS_C03.mpt'
file3 = file_path_3
data3 = readMPTData_CV2(file3)
label3 = '0.2 mA/cm^2'
#dataset = data[data['cycle number'] == 1.0]
#data_1stcycl_01 = data[data['cycle number'] == 1.0]
time3 = (data3['time/s']-cycling_start_time)/3600


root = tk.Tk()
root.withdraw()
file_path_4 = filedialog.askopenfilename()
#file = 'C:\Users\benja\OneDrive - Northeastern University\Northeastern\Gallaway Group\PhD Markup Experiment\Python Codes\Alyssa_Old_Files\Functions.py\EIS Results\LL GPE\2023\0523\FullCell_EMD_ZnAnode_NoCalendar_EIS_RT_C01.mpt'  # 'MIT cell C3_01_PEIS_C03.mpt'
file4= file_path_4
data4 = readMPTData_CV(file4)
print(data4.info())
label4 = '0.1 mA/cm^2'
#dataset = data[data['cycle number'] == 1.0]
#data_1stcycl_01 = data[data['cycle number'] == 1.0]
time4 = (data4['time/s']-cycling_start_time)/3600
# **** Plotting



# fig3, (axD, ax) = plt.subplots(1,2,figsize=(5,5))
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


colorlist = ['indianred', 'mediumpurple', 'tab:blue', 'lightsalmon', 'gold', 'limegreen', 'seagreen', 'turquoise',
           'tab:blue', 'mediumpurple', 'orchid', 'pink']

#
# axD.annotate('1000 Hz', (140, 120), fontsize=9)
# axD.annotate('1 Hz', (75, 20), fontsize=9)
# axD.annotate('0.1 Hz', (210, 1650), fontsize=9)


axD.set_xlabel("Time (hr)", fontweight='bold')
axD.set_ylabel("Ewe/V", fontweight='bold')
axD.set_title('Galvanostatic Cycling of 0.5M Zn-TFSI GPE - 0825-01', fontweight='bold')

 #upper right
#axD.legend(frameon=True, bbox_to_anchor=(.65, 0.2), loc='best', ncol=1, borderaxespad=0, fontsize=10)



axD.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
axD.tick_params(which='minor', direction='in', left=True, right=True, length=3)

# x = np.array([0,100,250])
# axD.xaxis.set_ticks()
# axD.xaxis.set_ticklabels(x)

#axD.set_aspect('equal', adjustable='box')
# ax.set_aspect('equal', adjustable='box')
"""
for cycle in range(int(num_cycles)):
    dataset = data[data['cycle number'] == cycle + 1.0]
    plt.plot(dataset['Ewe/V'], dataset['<I>/mA'], '-o',markersize = 4, label = label+' Cycle #'+str(cycle+1),)
#plt.plot(data_1stcycl_02['Ewe/V'], data_1stcycl_02['<I>/mA'], '-o',markersize = 4, label = label)
"""
"""
plt.plot(time, data['Ewe/V'], '-', markersize=4, label=label)
plt.plot(time2, data2['Ewe/V'], '-', markersize=4, label=label2)
plt.plot(time3, data3['Ewe/V'], '-', markersize=4, label=label3)
plt.plot(time4, data4['Ewe/V'], '-', markersize=4,label = label4)
"""
i=0
# setmatplotlib()
# plt.plot(time.tail(1000), data['Ewe/V'].tail(1000), '-', markersize=4, label=label, color = colorlist[i])
# axD.legend(frameon=True, borderaxespad=0, fontsize=10, bbox_to_anchor=(1.2, 0.5), loc='center')
# plt.tight_layout()
# plt.show()
i=1+i
# setmatplotlib()
# plt.plot(time2.tail(1000), data2['Ewe/V'].tail(1000), '-', markersize=4, label=label2, color = colorlist[i])
# axD.legend(frameon=True, borderaxespad=0, fontsize=10, bbox_to_anchor=(1.2, 0.5), loc='center')
# plt.tight_layout()
# plt.show()
i=1+i
# setmatplotlib()
# plt.plot(time3.tail(1000), data3['Ewe/V'].tail(1000), '-', markersize=4, label=label3, color = colorlist[i])
# axD.legend(frameon=True, borderaxespad=0, fontsize=10, bbox_to_anchor=(1.2, 0.5), loc='center')
# plt.tight_layout()
# plt.show()
i=1+i
setmatplotlib()
plt.plot(time4.tail(1000), data4['Ewe/V'].tail(1000), '-', markersize=4,label = label4, color = colorlist[i])
axD.legend(frameon=True, borderaxespad=0, fontsize=10, bbox_to_anchor=(1.2, 0.5), loc='center')
plt.tight_layout()
plt.show()



if __name__ == '__main__':
    pass