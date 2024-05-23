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
    fig3 = plt.figure()
    axD = fig3.add_subplot(111)
    axD.set_xlabel("Time (hr)", fontweight='bold',fontsize=40)
    axD.set_ylabel("Voltage (V)", fontweight='bold', fontsize=40)
    #axD.set_title('Galvanostatic Cycling of 0.5M Zn-TFSI GPE - 0825-01', fontweight='bold')
    axD.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True, labelsize=30)
    axD.tick_params(which='minor', direction='in', left=True, right=True, length=3)

def get_cycles(data: pd.DataFrame, cycle1: int, cycle2: int) -> pd.DataFrame:
    """
    This function takes a DataFrame and two cycle numbers, and returns a new DataFrame containing only the rows
    from the input DataFrame that correspond to the given cycle numbers.

    Parameters:
    data (pd.DataFrame): The input DataFrame.
    cycle1 (int): The first cycle number.
    cycle2 (int): The second cycle number.

    Returns:
    pd.DataFrame: A DataFrame containing only the rows from the input DataFrame that correspond to the given cycle numbers.
    """
    return data[(data['cycle number'] == cycle1) | (data['cycle number'] == cycle2)]
# ===========
# MAIN PROGRAM
# ===========

# **** Get data
out_1 = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Manuscript\Zn-Ion Fiber Work\Resources\ECS Talk\Zn-Sym Data\Sym Cycling\t2\out_1.csv'
out_2 = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Manuscript\Zn-Ion Fiber Work\Resources\ECS Talk\Zn-Sym Data\Sym Cycling\t2\out_2.csv'
out_3 = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Manuscript\Zn-Ion Fiber Work\Resources\ECS Talk\Zn-Sym Data\Sym Cycling\t2\out_3.csv'
out_4 = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Manuscript\Zn-Ion Fiber Work\Resources\ECS Talk\Zn-Sym Data\Sym Cycling\t2\out_4.csv'

"""
root = tk.Tk()
root.withdraw()
file_path_1 = filedialog.askopenfilename()
file = file_path_1
data = readMPTData_CV(file)
"""
file = out_1
data = pd.read_csv(file)

label = '0.05 mA/cm^2'
print(label)
print(data.info())
select_data_1 = get_cycles(data, 10, 11)
select_time_1 = (select_data_1['time/s'] - data['time/s'].min()) / 3600

#dataset = data[data['cycle number'] == 1.0]
#data_1stcycl_01 = data[data['cycle number'] == 1.0]
cycling_start_time = data['time/s'].min()
time = (data['time/s']-cycling_start_time)/3600
# file = '0805_02_EIS_CV_0.5MTFSI_ZnSym_InsideGB_0hr_RT_-2.5--2.5V_t1_02_CV_C01.mpt'
# data = readMPTData_CV(file)
# label = 'GPE 0805-02'
#data_1stcycl_02 = data[data['cycle number'] == 1.0]

"""
root = tk.Tk()
root.withdraw()
file_path_2 = filedialog.askopenfilename()
file2 = file_path_2
data2 = readMPTData_CV(file2)
"""
file2 = out_2
data2 = pd.read_csv(file2)
label2 = '0.1 mA/cm^2'

#dataset = data[data['cycle number'] == 1.0]
#data_1stcycl_01 = data[data['cycle number'] == 1.0]
time2 = (data2['time/s']-cycling_start_time)/3600
select_data_2 = get_cycles(data2, 10, 11)
select_time_2 = (select_data_2['time/s'] - data
['time/s'].min()) / 3600
"""
root = tk.Tk()
root.withdraw()
file_path_3 = filedialog.askopenfilename()
file3 = file_path_3
data3 = readMPTData_CV2(file3)
"""
file3 = out_3
data3 = pd.read_csv(file3)
label3 = '0.2 mA/cm^2'
#dataset = data[data['cycle number'] == 1.0]
#data_1stcycl_01 = data[data['cycle number'] == 1.0]
time3 = (data3['time/s']-cycling_start_time)/3600
select_data_3 = get_cycles(data3, 10, 11)
select_time_3 = (select_data_3['time/s'] - select_data_1['time/s'].min()) / 3600
"""
root = tk.Tk()
root.withdraw()
file_path_4 = filedialog.askopenfilename()
file4= file_path_4
data4 = readMPTData_CV(file4)
print(data4.info())
"""
file4 = out_4
data4 = pd.read_csv(file4)
label4 = '0.1 mA/cm^2'
#dataset = data[data['cycle number'] == 1.0]
#data_1stcycl_01 = data[data['cycle number'] == 1.0]
time4 = (data4['time/s']-cycling_start_time)/3600
select_data_4 = get_cycles(data4, 10, 11)
select_time_4 = (select_data_4['time/s'] - select_data_1['time/s'].min()) / 3600

out_1 = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Manuscript\Zn-Ion Fiber Work\Resources\ECS Talk\Zn-Sym Data\Sym Cycling\t2\out_1.csv'
out_2 = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Manuscript\Zn-Ion Fiber Work\Resources\ECS Talk\Zn-Sym Data\Sym Cycling\t2\out_2.csv'
out_3 = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Manuscript\Zn-Ion Fiber Work\Resources\ECS Talk\Zn-Sym Data\Sym Cycling\t2\out_3.csv'
out_4 = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Manuscript\Zn-Ion Fiber Work\Resources\ECS Talk\Zn-Sym Data\Sym Cycling\t2\out_4.csv'

#data.to_csv(out_1, index=False)
#data2.to_csv(out_2, index=False)
#data3.to_csv(out_3, index=False)
#data4.to_csv(out_4, index=False)

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


#axD.set_xlabel("Time (hr)", fontweight='bold')
#axD.set_ylabel("Ewe/V", fontweight='bold')
#axD.set_title('Galvanostatic Cycling of 0.5M Zn-TFSI GPE - 0825-01', fontweight='bold')

 #upper right
#axD.legend(frameon=True, bbox_to_anchor=(.65, 0.2), loc='best', ncol=1, borderaxespad=0, fontsize=10)

axD.set_xlabel("Time (hr)", fontweight='bold',fontsize=24)
axD.set_ylabel("Voltage (V)", fontweight='bold', fontsize=24)

axD.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True, labelsize=20)
axD.tick_params(which='minor', direction='in', left=True, right=True, length=3, labelsize=20)

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

plt.plot(time, data['Ewe/V'], '-', markersize=4, label=label, color = colorlist[0])
plt.plot(time2, data2['Ewe/V'], '-', markersize=4, label=label2, color = colorlist[1])
plt.plot(time3, data3['Ewe/V'], '-', markersize=4, label=label3, color = colorlist[2])
plt.plot(time4, data4['Ewe/V'], '-', markersize=4,label = label4, color = colorlist[0])
#plt.show()
i=0
setmatplotlib()
plt.plot(select_time_1, select_data_1['Ewe/V'], '-', markersize=4, label=label, color = colorlist[i])
setmatplotlib()
#axD.legend(frameon=True, borderaxespad=0, fontsize=10, bbox_to_anchor=(1.2, 0.5), loc='center')

#plt.show()
i=1+i
setmatplotlib()
plt.plot(select_time_2, select_data_2['Ewe/V'], '-', markersize=4, label=label, color = colorlist[i])
setmatplotlib()
#axD.legend(frameon=True, borderaxespad=0, fontsize=10, bbox_to_anchor=(1.2, 0.5), loc='center')

#plt.show()
i=1+i
setmatplotlib()
plt.plot(select_time_3, select_data_3['Ewe/V'], '-', markersize=4, label=label, color = colorlist[i])
setmatplotlib()
#axD.legend(frameon=True, borderaxespad=0, fontsize=10, bbox_to_anchor=(1.2, 0.5), loc='center')

#plt.show()
i=1+i
setmatplotlib()
plt.plot(select_time_4, select_data_4['Ewe/V'], '-', markersize=4, label=label, color = colorlist[0])
setmatplotlib()
#axD.legend(frameon=True, borderaxespad=0, fontsize=10, bbox_to_anchor=(1.2, 0.5), loc='center')

#plt.show()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Create a figure with a custom layout
fig = plt.figure(figsize=(10, 20))

# Define the layout
gs = gridspec.GridSpec(2, 4, height_ratios=[3, 1])

# Create the subplots
axs = [plt.subplot(gs[0, :])]  # First row spans entire width
axs += [plt.subplot(gs[1, i]) for i in range(4)]  # Second row contains four subplots

# Plot the data on the first subplot
axs[0].plot(time, data['Ewe/V'], '-', markersize=4, label=label, color=colorlist[0])
axs[0].plot(time2, data2['Ewe/V'], '-', markersize=4, label=label2, color=colorlist[1])
axs[0].plot(time3, data3['Ewe/V'], '-', markersize=4, label=label3, color=colorlist[2])
axs[0].plot(time4, data4['Ewe/V'], '-', markersize=4, label=label4, color=colorlist[3])
axs[0].set_xlabel("Time (hr)", fontweight='bold', fontsize=24)
axs[0].set_ylabel("Voltage (V)", fontweight='bold', fontsize=24)

# Plot the data on the remaining subplots
axs[1].plot(select_time_1, select_data_1['Ewe/V'], '-', markersize=4, label=label, color=colorlist[0])
axs[2].plot(select_time_2, select_data_2['Ewe/V'], '-', markersize=4, label=label, color=colorlist[1])
axs[3].plot(select_time_3, select_data_3['Ewe/V'], '-', markersize=4, label=label, color=colorlist[2])
axs[4].plot(select_time_4, select_data_4['Ewe/V'], '-', markersize=4, label=label, color=colorlist[3])

# Set the labels for the remaining subplots
for i in range(1, 5):
    axs[i].set_xlabel('Time (hr)', fontweight='bold', fontsize=24)
    axs[i].set_ylabel('Voltage (V)', fontweight='bold', fontsize=24)

# For the first subplot
axs[0].tick_params(axis='both', labelsize=18)

# For the remaining subplots
for i in range(1, 5):
    axs[i].tick_params(axis='both', labelsize=18)
x_position = [.05, 0.15, 0.25, 0.5, 0.5, 0.25, 0.65, 0.85]
y_position = [0.5, 0.5, 0.5, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1]
font_size = 18

axs[0].text(x_position[0], y_position[0], '$0.5 mAh/cm^2$', fontsize=font_size, fontweight='bold', transform=axs[0].transAxes)
axs[0].text(x_position[1], y_position[1], '$1 mAh/cm^2$', fontsize=font_size, fontweight='bold', transform=axs[0].transAxes)
axs[0].text(x_position[2], y_position[2], '$2 mAh/cm^2$', fontsize=font_size, fontweight='bold', transform=axs[0].transAxes)
axs[0].text(x_position[3], y_position[3], '$0.5 mAh/cm^2$', fontsize=font_size, fontweight='bold', transform=axs[0].transAxes)

axs[1].text(x_position[4], y_position[4], '$0.5 mAh/cm^2$', fontsize=font_size, fontweight='bold', transform=axs[1].transAxes)
axs[2].text(x_position[4], y_position[4], '$1 mAh/cm^2$', fontsize=font_size, fontweight='bold', transform=axs[2].transAxes)
axs[3].text(x_position[4], y_position[4], '$2 mAh/cm^2$', fontsize=font_size, fontweight='bold', transform=axs[3].transAxes)
axs[4].text(x_position[4], y_position[4], '$0.5 mAh/cm^2$', fontsize=font_size, fontweight='bold', transform=axs[4].transAxes)
# Display the plot
plt.tight_layout()
#plt.show()


tt = pd.concat([time, time2, time3, time4], axis=1)
EE = pd.concat([data['Ewe/V'], data2['Ewe/V'], data3['Ewe/V'], data4['Ewe/V']], axis=1)

fig1, (ax11) = plt.subplots(1,1,figsize=(8.34,2.78))


#ax12 = ax11.twinx()

ax11.set_xlabel("Time (hours)")
ax11.set_ylabel("Potential (V)")
#ax12.set_ylabel("Current (mA)")


ax11.tick_params(which='both', axis='both', direction='in', bottom=True, top=True, left=True, right=False)
#ax12.tick_params(which='both', axis='both', direction='in', bottom=False, top=False, left=False, right=True)




ax11.set_ylim((-.15,.15))
#ax12.set_ylim((-0.35,0.35))

fig1.tight_layout()




# Plot the data ================================

color = "Teal"
ax11.plot(time, data['Ewe/V'], color=colorlist[0], ls='-', lw=2.0, label='Potential')
ax11.plot(time2, data2['Ewe/V'], color=colorlist[1], ls='-', lw=2.0, label='Potential')
ax11.plot(time3, data3['Ewe/V'], color=colorlist[2], ls='-', lw=2.0, label='Potential')
ax11.plot(time4, data4['Ewe/V'], color=colorlist[0], ls='-', lw=2.0, label='Potential')

color = 'm'
#ax12.plot(tt,ii,color = color, ls='-', lw=1.0, label='Current')


#fig1.legend(bbox_to_anchor=(0.45, 0.4), fontsize = 'small', frameon=False)

fig1.tight_layout()



# Save the figure ================================

output_name = 'plo'
output_path = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Manuscript\Zn-Ion Fiber Work\Resources\ECS Talk\Zn-Sym Data\Sym Cycling\t2\fig_1.png'
plt.savefig(output_path, dpi=400)
plt.show()

fig2, (ax12) = plt.subplots(1,1,figsize=(2.6,2.1))
ax12.set_xlabel("Time (hours)")
ax12.set_ylabel("Potential (V)")
#ax12.set_ylabel("Current (mA)")


ax12.tick_params(which='both', axis='both', direction='in', bottom=True, top=True, left=True, right=False)
#ax12.tick_params(which='both', axis='both', direction='in', bottom=False, top=False, left=False, right=True)




#ax11.set_ylim((1.8,4.2))
ax12.set_ylim((-0.1,0.1))

fig2.tight_layout()




# Plot the data ================================

color = "Teal"
ax12.plot(select_time_4, select_data_4['Ewe/V'], color=colorlist[0], ls='-', lw=2.0, label='Potential')
fig2.tight_layout()

color = 'm'
#ax12.plot(tt,ii,color = color, ls='-', lw=1.0, label='Current')


#fig1.legend(bbox_to_anchor=(0.45, 0.4), fontsize = 'small', frameon=False)





# Save the figure ================================

output_name = 'plo'
output_path = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Manuscript\Zn-Ion Fiber Work\Resources\ECS Talk\Zn-Sym Data\Sym Cycling\t2\fig_5.png'
plt.savefig(output_path, dpi=400)
plt.show()

if __name__ == '__main__':
    pass