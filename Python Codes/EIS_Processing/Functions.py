#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:00:30 2022

@author: alyssastavola
"""
import numpy as np
import pylab
import glob
import pandas as pd
import scipy.optimize as optimize
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import statistics
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

##do I want to have it read active mass or capacity too
def readArbinData(filename):
    """
This function reads an Arbin data file

    Parameters
    ----------
    filename : Name including .csv of Arbin file to read

    Returns
    -------
    data : dataframe of arbin data

    """
    data = pd.read_csv(filename, sep=',', delimiter=None, 
                    header= 0,index_col=None, usecols=None, engine= 'python')
    print(data.head(1))
    return data
    
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
    with open(filename,  'r', encoding='cp1252') as readfile:
            header_rows = readfile.readlines(18)
            header_rows = str(header_rows).split()
            df = pd.DataFrame(header_rows)
            header_num = df.iloc[7]
            header_num = int(header_num)

            # print(header_num)
            header_num= header_num-5 #-4 #-5
    data = pd.read_csv(filename, header= header_num, sep='\t',
                      engine= 'python', encoding='cp1252') 
    
    # print(data.head(5))
    return data

def plotNyquist(data, i,label, offset, lowerlimit, upperlimit):
    """
    Nyquist Plot of cycle i
    
    Parameters
    ----------
    data : dataframe 
    i : # of cycle
    label : label on plot
    offset : y offset in Nyquist plot
    lowerlimit : lower limit for minimum function 
    upperlimit : upper limit for minimum function 
        DESCRIPTION.

    Returns
    -------
    Rct : charge transfer resistance

    """
    df0 = data.loc[data['cycle number'] == i]
    #print(df0)
    df = df0.loc[df0['-Im(Z)/Ohm'] >= 0]
    
    if label == 'cell C1':
        df=df.iloc[1:]
    
    firstline = df.iloc[0]

    yf = firstline['-Im(Z)/Ohm']
    xf = firstline['Re(Z)/Ohm']
    yf = yf + offset
    print('first point is at ', xf, yf )
    
    #df = df0
    #print(df)
    Real = df['Re(Z)/Ohm']
    Imag = df['-Im(Z)/Ohm']

    #freq= df['freq/Hz']
    #print(freq)
    #offset = 0
    offset = offset
    
    Imag = Imag + offset

    # axD.plot(Real,Imag+(i*0.01), '-o',markersize = 4, label = 'Discharge step #'+str(i) )  
    plt.plot(Real,Imag+(i*0.01), 'o',markersize = 4, label = label ) 
    #axD.plot(Real,Imag+(i*0.01), '-o', label = label) 
             
    dflargefreq = df.loc[df['freq/Hz'] >= 900 ]
    largefreq = dflargefreq.loc[dflargefreq['freq/Hz'] <= 1300 ]
    yl = largefreq['-Im(Z)/Ohm']
    xl = largefreq['Re(Z)/Ohm']
    yl = yl + offset
    plt.plot(xl,yl, 'ko',markersize = 4, label= '') 
     
    dfsmallfreq = df.loc[df['freq/Hz'] <= 1.2 ]
    smallfreq = dfsmallfreq.loc[dfsmallfreq['freq/Hz'] >= .85 ]
    ys = smallfreq['-Im(Z)/Ohm']
    xs = smallfreq['Re(Z)/Ohm']
    ys = ys + offset
    plt.plot(xs,ys, 'ko',markersize = 4, label= '')
     
    smallestfreq = df.loc[df['freq/Hz'] <= .0105 ]
    ysest = smallestfreq['-Im(Z)/Ohm']
    xsest = smallestfreq['Re(Z)/Ohm']
    ysest = ysest + offset
    plt.plot(xsest,ysest, 'ko',markersize = 4, label= '')
    
    plt.plot(xf,yf, 'bo',markersize = 3, label= '') 
    # xfirst='{:.3f}'.format(xf)
    xfirst = round(xf,3)
    # print(xfirst)
    plt.annotate(xfirst, (xf-10, yf+50), color='b', fontsize=9)
    # axD.annotate(xfirst, (xf-10, yf+30), color='b', fontsize=9)
    
    Rmin, Imin = getmin (data, '-Im(Z)/Ohm', 'Re(Z)/Ohm', lowerlimit, upperlimit)
    # axD.annotate(Rmin, (Rmin+10, Imin), color='g', fontsize=9)
    plt.annotate(Rmin, (Rmin+20, Imin-10), color='g', fontsize=9)
    plt.plot(Rmin,Imin, 'go',markersize = 3, label= '') 
    print('min point is at ', Rmin, Imin )
    
    
    Rct = Rmin - xfirst
    Rct = round(Rct,3)
    print(Rct)
    plt.annotate('Rct is ' + str(Rct), (10, 200), color='k', fontsize=9)
    # axD.annotate('Rct is ' + str(Rct), (10, 80), color='k', fontsize=9)
    # axD.annotate('Rct is ' + str(Rct), (Rmin+10, Imin+100), color='k', fontsize=9)
    # axD.annotate('Rct is ' + str(Rct), (Rmin+10, Imin+150), color='k', fontsize=9)
    return Rct
    
def plotNyquist_calcRohm(data, i,label, offset):
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
    #print(df0)
    df = df0.loc[df0['-Im(Z)/Ohm'] >= 0]
    
    mValueIndex = df[['-Im(Z)/Ohm']].idxmin()
    # print(mValueIndex)
    
    m_mum = data.iloc[mValueIndex]
    # print(m_mum)
    # print ('minimum is at', m_mum)
    
    firstline= pd.DataFrame(m_mum)

    # firstline = df.iloc[mValueIndex:]

    yf = firstline['-Im(Z)/Ohm']
    xf = firstline['Re(Z)/Ohm']
    xf = float(xf)
    yf = float(yf)
    yf = yf + offset
    print('minimum is at ', xf, yf )
    
    #df = df0
    #print(df)
    Real = df['Re(Z)/Ohm']
    Imag = df['-Im(Z)/Ohm']

    #freq= df['freq/Hz']
    #print(freq)
    #offset = 0
    offset = offset
    
    Imag = Imag + offset

    # axD.plot(Real,Imag+(i*0.01), '-o',markersize = 4, label = 'Discharge step #'+str(i) )  
    plt.plot(Real,Imag+(i*0.01), 'o',markersize = 4, label = label ) 
    #axD.plot(Real,Imag+(i*0.01), '-o', label = label) 
             
    dflargefreq = df.loc[df['freq/Hz'] >= 900 ]
    largefreq = dflargefreq.loc[dflargefreq['freq/Hz'] <= 1300 ]
    yl = largefreq['-Im(Z)/Ohm']
    xl = largefreq['Re(Z)/Ohm']
    yl = yl + offset
    plt.plot(xl,yl, 'ko',markersize = 4, label= '') 
     
    dfsmallfreq = df.loc[df['freq/Hz'] <= 1.2 ]
    smallfreq = dfsmallfreq.loc[dfsmallfreq['freq/Hz'] >= .85 ]
    ys = smallfreq['-Im(Z)/Ohm']
    xs = smallfreq['Re(Z)/Ohm']
    ys = ys + offset
    plt.plot(xs,ys, 'ko',markersize = 4, label= '')
     
    smallestfreq = df.loc[df['freq/Hz'] <= .0105 ]
    ysest = smallestfreq['-Im(Z)/Ohm']
    xsest = smallestfreq['Re(Z)/Ohm']
    ysest = ysest + offset
    plt.plot(xsest,ysest, 'ko',markersize = 4, label= '')
    
    plt.plot(xf,yf, 'bo',markersize = 3, label= '') 
    # xfirst='{:.3f}'.format(xf)
    xfirst = round(xf,3)
    # print(xfirst)
    if label == 'cell C3':
        plt.annotate(xfirst, (xf+15000, yf+50), color='b', fontsize=9)
    else: 
        plt.annotate(xfirst, (xf-10, yf+50), color='b', fontsize=9)
    # axD.annotate(xfirst, (xf-10, yf+30), color='b', fontsize=9)

    return xfirst

def plotLSV(data, label):
    
    df = data
    
    x = df['Ewe/V']
    y = df['<I>/mA']
    
    plt.plot(x,y, '-',linewidth = 2, label = label ) 

def plotCV(data, label, cycle_num, line_style):

    df = data.loc[data['cycle number'] == cycle_num]
    
    x = df['Ewe/V']
    y = df['<I>/mA']
    
    plt.plot(x,y, linewidth = 2, label = label, linestyle =line_style ) 
    
def getmin(data,findingmof, limitsof, lowerlimit,upperlimit): 
    #finds minimum value of Nyquist plot where data has real in 2nd column and 
    #Imag in third column
    
    dfi = data.loc[data[limitsof] > lowerlimit]   
    df = dfi.loc[dfi[limitsof] < upperlimit]   
#    print(df.columns)
    
    mValueIndex = df[[findingmof]].idxmin(axis=0)
    # print(mValueIndex)
    
    # m_mum = data.iloc[mValueIndex]
    # print(m_mum)
    # print ('minimum is at', m_mum)

    real = data.iloc[mValueIndex,1]
    real = float(real)
    real = round(real,3)
    #print (real)
    
    imag = data.iloc[mValueIndex,2]
    imag = float(imag)
    imag = round(imag,3)
    # print(imag)
    return (real,imag) 
#    print(df.columns)
    
    mValueIndex = df[[findingmof]].idxmin(axis=0)
    # print(mValueIndex)
    
    # m_mum = data.iloc[mValueIndex]
    # print(m_mum)
    # print ('minimum is at', m_mum)

    real = data.iloc[mValueIndex,1]
    real = float(real)
    real = round(real,3)
    #print (real)
    
    imag = data.iloc[mValueIndex,2]
    imag = float(imag)
    imag = round(imag,3)
    # print(imag)
    return (real,imag)

def plotcyclesold(data, w, x, keyword): 
    

    n = x
    w = w
    data = data
    
    df1 = data.loc[data['Cycle_Index'] == n]
    
    df = df1.loc[df1['Step_Index'] >= 2]
    # print(df)
    

    dfc = df.loc[df['Current(A)'] > 0] #charge
    df1 =dfc.iloc[-1] #this is the final value at the end of charge

        
    dfci = dfc.iloc[1]  #this is the initial value at the start of charge
    init = dfci['Test_Time(s)']
    final = df1['Test_Time(s)']
    timeelapsed= final - init
    
    dfd = df.loc[df['Current(A)'] < 0]  #discharge
    df3= dfd.iloc[-1]
    dfi = dfd.iloc[1]  #this is the initial value at the start of discharge

#charge
    Ia = df1['Current(A)']
    Cycle = dfc['Cycle_Index']
    V = dfc['Voltage(V)']
    IA = dfc['Current(A)']
    T = dfc['Test_Time(s)']
    
    Tfirst = dfci['Test_Time(s)']
    T = T - Tfirst
    Th = T/3600


#Calculate the capacity of the battery 
    I= IA*1000

    #C = T * abs(IA) / 3600 * 1000  #(mAh)
    
    C = T * abs(Ia) / 3600 * 1000 / w  #(mAh/g) 
   
#discharge
    IAd = dfd['Current(A)']
    IAdis = dfci['Current(A)']
    Cycled = dfd['Cycle_Index']
    Vd = dfd['Voltage(V)']
    IAd = df3['Current(A)']
    Td = df3['Test_Time(s)'] 
    Ti = dfi['Test_Time(s)']
    Tdis = (Ti- Td) / 3600
    timed = dfd['Test_Time(s)']
    timed = timed - Ti

    Id= IAd*1000
    
    Cd = timed * abs(IAdis) / 3600 * 1000 / w  #(mAh/g)  
    #print (Cd)
    
    plt.plot(C,V, label = 'Cycle ' + str(n) + ' charge' + keyword,linewidth=2)
    plt.plot(Cd,Vd, label = '_nolegend_', linestyle = '--',linewidth=2) 

def plotcycles(data, w, x, keyword): 
    n = x
    w = w
    data = data
    
    df1 = data.loc[data['Cycle Index'] == n]
    
    df = df1.loc[df1['Step Index'] >= 2]
    # print(df)
    

    dfc = df.loc[df['Current (A)'] > 0] #charge
    df1 =dfc.iloc[-1] #this is the final value at the end of charge

        
    dfci = dfc.iloc[1]  #this is the initial value at the start of charge
    init = dfci['Test Time (s)']
    final = df1['Test Time (s)']
    timeelapsed= final - init
    
    dfd = df.loc[df['Current (A)'] < 0]  #discharge
    df3= dfd.iloc[-1]
    dfi = dfd.iloc[1]  #this is the initial value at the start of discharge


#charge
    Ia = df1['Current (A)']
    Cycle = dfc['Cycle Index']
    V = dfc['Voltage (V)']
    IA = dfc['Current (A)']
    T = dfc['Test Time (s)']
    
    Tfirst = dfci['Test Time (s)']
    T = T - Tfirst
    Th = T/3600


#Calculate the capacity of the battery 
    I= IA*1000

    #C = T * abs(IA) / 3600 * 1000  #(mAh)
    
    C = T * abs(Ia) / 3600 * 1000 / w  #(mAh/g) 
   
#discharge
    IAd = dfd['Current (A)']
    IAdis = dfci['Current (A)']
    Cycled = dfd['Cycle Index']
    Vd = dfd['Voltage (V)']
    IAd = df3['Current (A)']
    Td = df3['Test Time (s)'] 
    Ti = dfi['Test Time (s)']
    Tdis = (Ti- Td) / 3600
    timed = dfd['Test Time (s)']
    timed = timed - Ti

    Id= IAd*1000
    
    Cd = timed * abs(IAdis) / 3600 * 1000 / w  #(mAh/g)  
    #print (Cd)
    
    plt.plot(C,V, label = 'Cycle ' + str(n) + ' charge' + keyword,linewidth=2)
    plt.plot(Cd,Vd, label = '_nolegend_', linestyle = '--',linewidth=2)

def plotMACCORcycles(data, w, x, keyword): 
    n = x
    w = w
    data = data
    
    df = data.loc[data['Cycle_Index	'] == n]
    
    dfc = df.loc[df['Step_Index'] == 2] #    dfc = df.loc[df['Current(A)'] > 0] #charge
    dfd = df.loc[df['Step_Index'] == 3] #    dfd = df.loc[df['Current(A)'] < 0]  #discharge
    df1 =dfc.iloc[-1] #this is the final value at the end of charge
       
    dfci = dfc.iloc[1]  #this is the initial value at the start of charge
    init = dfci['Test_Time(s)']
    final = df1['Test_Time(s)']
    timeelapsed= final - init
    
    df3= dfd.iloc[-1]
    dfi = dfd.iloc[1]  #this is the initial value at the start of discharge
    
    Ia = df1['Current(A)'] #charge
    Cycle = dfc['Cycle_Index	']
    V = dfc['Voltage(V)']
    IA = dfc['Current(A)']
    T = dfc['Test_Time(s)']
    
    Tfirst = dfci['Test_Time(s)']
    T = T - Tfirst
    Th = T/3600
    I= IA*1000 #Calculate the capacity of the battery 
    #C = T * abs(IA) / 3600 * 1000  #(mAh)
    C = T * abs(Ia) / 3600 * 1000 / w  #(mAh/g) 
   
    IAd = dfd['Current(A)'] #discharge
    IAdis = dfci['Current(A)']
    Cycled = dfd['Cycle_Index	']
    Vd = dfd['Voltage(V)']
    IAd = df3['Current(A)']
    Td = df3['Test_Time(s)'] 
    Ti = dfi['Test_Time(s)']
    Tdis = (Ti- Td) / 3600
    timed = dfd['Test_Time(s)']
    timed = timed - Ti
    Id= IAd*1000
    Cd = timed * abs(IAdis) / 3600 * 1000 / w  #(mAh/g)  
    #print (Cd)
    # ax0.plot(C,V, label = 'Cycle ' + str(n) + ' charge' + keyword,linewidth=2)
    ax0.plot(Cd,Vd, label = '_nolegend_', linestyle = '--',linewidth=2)

# def coulombic_efficiency(filedata,w, num_cycles, color, keyword, mark, fill):
    """
    This function gets the coulombic efficiency of each cycle of the battery
    = Coulombs on discharge/ coulombs on charge *100 
    1 mAh = 3.6C
          
    C is capacity of charge cycle
    Cd is capacity of discharge cycle
    
    Parameters
    ----------
    filedata : data
    w : weight of the active mass in the cell 
    color : color for the plot
    keyword : label for the plot
    mark : marker for the plot
    fill : fill color for the plot
    """
    for n in range(1,num_cycles):    
        data = filedata
        w = w
    
        df = data.loc[data['Cycle_Index'] == n]
        dfa = df.loc[df['Current(A)'] > 0] #charge
        df1 =dfa.iloc[-1] #this is the final value at the end of charge
        #ccap = df1['Charge_Capacity(Ah)']
        
        dfci = dfa.iloc[1]  #this is the initial value at the start of charge
        init = dfci['Test_Time(s)']
        final = df1['Test_Time(s)']
        timeelapsed= final - init
        
        dfd = df.loc[df['Current(A)'] < 0]  #discharge
        df3= dfd.iloc[-1]
        dfi = dfd.iloc[1]  #this is the initial value at the start of discharge
        
        IA = df1['Current(A)']
        
        IAd = df3['Current(A)']
        Td = df3['Test_Time(s)']
    
        Ti = dfi['Test_Time(s)']
        
        I= IA*1000
        Id = IAd*1000
        
        Cd = (Td-Ti) * abs(Id) / 3600 
        C = timeelapsed * abs(I) / 3600 
        # C = ccap *1000 
        CE = ((Cd *3.6) / (C * 3.6))  *100
        Cw = C / w
        Cdw = Cd/ w  
        if n == 1:
            ax2.plot(n,CE, color,  marker = mark, markerfacecolor= fill, linewidth=2,label = 'Coulombic Efficiency' + keyword, markersize = 7, linestyle = 'None')
            ax.plot(n,Cw, color,  marker = 'o', markerfacecolor= color, linewidth=2,label = 'Charge Capacity' + keyword, markersize = 8, linestyle = 'None')
            ax.plot(n,Cdw, color,  marker = 'o', markerfacecolor= 'None', linewidth=2,label = 'Discharge Capacity' + keyword, markersize = 8, linestyle = 'None')
            print ('Cd of cycle', n, '=', Cd)
            print ('Cc of cycle', n, '=', C)
        else:
            ax2.plot(n,CE, color,  marker = mark, markerfacecolor= fill, linewidth=2, markersize = 7, linestyle = 'None')
            # print ('Cd of cycle', n, '=', Cd)
            # print ('Cc of cycle', n, '=', C)
            ax.plot(n,Cdw, color,  marker = 'o', markerfacecolor= color, linewidth=2, markersize = 8, linestyle = 'None')
            ax.plot(n,Cdw, color,  marker = 'o', markerfacecolor= 'None', linewidth=2, markersize = 8, linestyle = 'None')
            print ('CE (%) of cycle', n, '=', CE)
            

# def cap(filedata,w, color, keyword, mark, fill):
##        """
##        This function gets the coulombic efficiency of each cycle of the battery
##        
##        = Coulombs on discharge/ coulombs on charge *100 
##    
##        1 mAh = 3.6C
##   
##        C is capacity of charge cycle
##        Cd is capacity of discharge cycle
#        """
        
        data = filedata
        
        df = data.loc[data['Cycle_Index'] == n]
        dfa = df.loc[df['Current(A)'] > 0] #charge
        df1 =dfa.iloc[-1] #this is the final value at the end of charge
        #ccap = df1['Charge_Capacity(Ah)']
        
        dfci = dfa.iloc[1]  #this is the initial value at the start of charge
        init = dfci['Test_Time(s)']
        final = df1['Test_Time(s)']
        timeelapsed= final - init


        dfd = df.loc[df['Current(A)'] < 0]  #discharge
        df3= dfd.iloc[-1]
        dfi = dfd.iloc[1]  #this is the initial value at the start of discharge

        
        IA = df1['Current(A)']
        
        IAd = df3['Current(A)']
        Td = df3['Test_Time(s)']
        
        Ti = dfi['Test_Time(s)']
        

        I= IA*1000
        Id = IAd*1000
        
        Cd = (Td-Ti) * abs(Id) / 3600 
        C = timeelapsed * abs(I) / 3600 
        # C = ccap *1000 

        CE = ((Cd *3.6) / (C * 3.6))  *100
#        
#        print ('Cd of cycle', n, '=', Cd)
#        print ('C of cycle', n, '=', C)

        
        Cw = C / w
        Cdw = Cd/ w

#        
        if n == 1:
#            ax2.plot(n,CE, color,  marker = mark, markerfacecolor= fill, linewidth=2,label = keyword, markersize = 7, linestyle = 'None')
            ax.plot(n,Cw, color,  marker = 'o', markerfacecolor= color, linewidth=2,label = 'Charge Capacity' + keyword, markersize = 8, linestyle = 'None')
            ax.plot(n,Cdw, color,  marker = 'o', markerfacecolor= 'None', linewidth=2,label = 'Discharge Capacity' + keyword, markersize = 8, linestyle = 'None')
#            print ('Cd of cycle', n, '=', Cd)
#            print ('Cc of cycle', n, '=', C)
        else:
#            ax2.plot(n,CE, color,  marker = mark, markerfacecolor= fill, linewidth=2, markersize = 7, linestyle = 'None')
            ax.plot(n,Cdw, color,  marker = 'o', markerfacecolor= color, linewidth=2, markersize = 8, linestyle = 'None')
            ax.plot(n,Cdw, color,  marker = 'o', markerfacecolor= 'None', linewidth=2, markersize = 8, linestyle = 'None')

#            print ('Cd of cycle', n, '=', Cd)
#            print ('Cc of cycle', n, '=', C)
#        print ('CE (%) of cycle', n, '=', CE)
        return dfa



# def endvoltage(filedata, color, keyword, mark, fill):
    """
    This function gets the end voltage of each discharge cycle of the battery

    """
    data = filedata
    df = data.loc[data['Cycle_Index	'] == n]
    dfd = df.loc[df['Current(A)'] < 0]  #discharge
    df3= dfd.iloc[-1]
    #dfi = dfd.iloc[1]  #this is the initial value at the start of discharge
        
    Vfd = df3['Voltage(V)'] #discharge end voltage
    #print(Vfd)
    
    #print (Vfd)
#    if n == 1:
#        ax.plot(n,Vfd, color, marker = mark, markerfacecolor= fill,  linewidth=2,label = keyword, markersize = 8)
#    else:
#        ax.plot(n,Vfd, color, marker = mark, markerfacecolor= fill, linewidth=2, markersize = 8)
# 
    if n == 1:
        ax2.plot(n,Vfd, color, marker = mark, markerfacecolor= fill,  linewidth=2,label = keyword, markersize = 7, linestyle = 'None')
    else:
        ax2.plot(n,Vfd, color, marker = mark, markerfacecolor= fill, linewidth=2, markersize = 7, linestyle = 'None')
   
# def energyefficiency(filedata, color, keyword, mark, fill):
#        """
#        This function gets the energy efficiency of each cycle of the battery
#        
#   
#        C is capacity of charge cycle
#        Cd is capacity of discharge cycle
#        """
        
 
        data = filedata
        
        df = data.loc[data['Cycle_Index	'] == n]
        dfa = df.loc[df['Current(A)'] > 0] #charge
        df1 =dfa.iloc[-1] #this is the final value at the end of charge
        ccap = df1['Charge_Capacity(Ah)']
        
        dfci = dfa.iloc[1]  #this is the initial value at the start of charge
        init = dfci['Test_Time(s)']
        final = df1['Test_Time(s)']
        timeelapsed= final - init


        dfd = df.loc[df['Current(A)'] < 0]  #discharge
        df3= dfd.iloc[-1]
        dfi = dfd.iloc[1]  #this is the initial value at the start of discharge
 
        #Cdf = df3['Discharge_Capacity(Ah)'] #this is the capacity at the end of the discharge cycle
        
        #ccapi = dfi['Charge_Capacity(Ah)']   
        #print ('ccapi', ccapi)
        
        #IA = df2['Current(A)']
        #T = df2['Test_Time(s)']
        
        IAd = df3['Current(A)']
        Td = df3['Test_Time(s)']
        
        Ti = dfi['Test_Time(s)']
        

        #I= IA*1000
        Id = IAd*1000
        
        AllchargeV= dfa['Voltage(V)']
        AlldischargeV= dfd['Voltage(V)']
        
        AvgchargeV= statistics.mean(AllchargeV)
        AvgdischargeV = statistics.mean(AlldischargeV)
        
        VE = AvgdischargeV/ AvgchargeV * 100   
        
        Cd = (Td-Ti) * abs(Id) / 3600 
        C = timeelapsed * abs(Id) / 3600 
        # C = ccap *1000 
#        print ('Cd of cycle', n, '=', Cd)
#        print ('C of cycle', n, '=', C)
        
        CE = (Cd *3.6) / (C * 3.6)  *100
        
        EE = VE * CE /100
        
        
        if n == 1:
            ax2.plot(n,EE, color, marker = mark, markerfacecolor= 'None', linewidth=2,label = keyword, markersize = 7, linestyle = 'None')
        else:
            ax2.plot(n,EE, color, marker = mark, markerfacecolor= 'None', linewidth=2, markersize = 7, linestyle = 'None')  #print ('EE (%) of cycle', n, '=', EE)  


