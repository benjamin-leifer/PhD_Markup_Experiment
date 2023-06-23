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


# from Functions import plotNyquist_calcRohm
# np.set_printoptions(suppress=True)

def readMPTData(filename):
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


# ===========
# MAIN PROGRAM
# ===========

# **** Get data

#file = 'C:\Users\benja\OneDrive - Northeastern University\Northeastern\Gallaway Group\PhD Markup Experiment\Python Codes\Alyssa_Old_Files\Functions.py\EIS Results\LL GPE\2023\0523\FullCell_EMD_ZnAnode_NoCalendar_EIS_RT_C01.mpt'  # 'MIT cell C3_01_PEIS_C03.mpt'
file = '../../FullCell_EMD_ZnAnode_NoCalendar_EIS_RT_C01.mpt'
data = readMPTData(file)
labels = 'No Calendar'

#file2 = 'C:\Users\benja\OneDrive - Northeastern University\Northeastern\Gallaway Group\PhD Markup Experiment\Python Codes\Alyssa_Old_Files\Functions.py\EIS Results\LL GPE\2023\0523\FullCell_EMD_ZnAnode_Calendar_30C_EIS_RT_C01.mpt'  # 'MIT cell B6_01_PEIS_C03.mpt' #B6 filename
file2 = 'FullCell_EMD_ZnAnode_Calendar_30C_EIS_RT_C01.mpt'
data2 = readMPTData(file2)
label2 = '30C Calendar'

#file3 = 'C:\Users\benja\OneDrive - Northeastern University\Northeastern\Gallaway Group\PhD Markup Experiment\Python Codes\Alyssa_Old_Files\Functions.py\EIS Results\LL GPE\2023\0523\FullCell_EMD_ZnAnode_Calendar_60C_EIS_RT_C01.mpt'  # 'MIT cell B3_01_PEIS_C04.mpt' #B3 filename
file3 = 'FullCell_EMD_ZnAnode_Calendar_60C_EIS_RT_C01.mpt'
data3 = readMPTData(file3)
label3 = '60C Calendar'

file4 = 'FullCell_EMD_ZnAnode_Calendar_90C_EIS_RT_C01.mpt'
data4 = readMPTData(file4)
label4 = '90C Calendar'

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


# plotNyquist(data, 1, 'cell C1', 0, 157, 300)
# plotNyquist(data, 1, 'cell C2', 0, 70, 300)
plotNyquist_calcRohm(data, 1, labels, 0)
plotNyquist_calcRohm(data2, 1, label2, 2000)
plotNyquist_calcRohm(data3, 1, label3, 4000)
plotNyquist_calcRohm(data4, 1, label4, 6000)

#
# axD.annotate('1000 Hz', (140, 120), fontsize=9)
# axD.annotate('1 Hz', (75, 20), fontsize=9)
# axD.annotate('0.1 Hz', (210, 1650), fontsize=9)


axD.set_xlabel("z' / ohms", fontweight='bold')
axD.set_ylabel("-z'' / ohms", fontweight='bold')
axD.set_title('Nyquist Plot for EMD/0.5M Zn-TSFI GPE/Zn Blocking Cell Experiments', fontweight='bold')

axD.legend(loc='best')  #upper right
#axD.legend(frameon=True, bbox_to_anchor=(.65, 0.2), loc='best', ncol=1, borderaxespad=0, fontsize=10)

#axD.set_ylim(-10, 400)
# axD.set_xlim(-10, 150)
# axD.set_xlim(-10, 50)
# axD.set_ylim(-10, 50)
# axD.set_ylim(-10, 100)
# axD.set_xlim(-10, 250)
#axD.set_xlim(-10, 400)

axD.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
axD.tick_params(which='minor', direction='in', left=True, right=True, length=3)

# x = np.array([0,100,250])
# axD.xaxis.set_ticks()
# axD.xaxis.set_ticklabels(x)

axD.set_aspect('equal', adjustable='box')
# ax.set_aspect('equal', adjustable='box')

#plt.savefig('MIT EIS.png', dpi=500, bbox_inches='tight')
# plt.savefig('MIT C2.png', dpi=500, bbox_inches = 'tight')
# plt.savefig('MIT C5.png', dpi=500, bbox_inches = 'tight')
# plt.savefig(str(labels)+'.png', dpi=500, bbox_inches = 'tight')
label_list = [labels, label2, label3, label4]
Rct_list = []
for i, datum in enumerate([data, data2, data3, data4]):
    print('for ' + label_list[i] + ' Rct = ')
    Rct_list.append(plotNyquist(datum, 1, label_list[i], 2000*i, 2000, 10000))

for i,rct in enumerate(Rct_list):
    print('Rct for ' + label_list[i] + ' = ' + str(rct))
