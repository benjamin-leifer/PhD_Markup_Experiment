#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 09:07:55 2022

@author: alyssastavola
"""

import numpy as np
import PIL
import pylab
import os
import colorsys
import math
import scipy.optimize
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.colors import colorConverter
from PIL import Image
import matplotlib
import pandas as pd
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

# default font as Arial #normally DejaVu Sans
matplotlib.rcParams['font.sans-serif'] = "Arial"  # "DejaVu Sans" #
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"


def plotXRD(dfX, weighting, offset):
    Xx = dfX['two theta']
    Xi = dfX['y_obs']
    Xb = dfX['y_bkg']
    Xi = Xi - Xb
    Xi = Xi * weighting

    sax2.plot(Xx, Xi + offset, color='k', linewidth=1, label='ADXRD observed, bkg corrected')


def plotSTEM(df, weighting):
    Tx = df['two theta']
    Ty = df['I']
    # Ty = Ty + 90
    Ty = Ty * weighting

    sax2.stem(Tx, Ty, linefmt='-g', markerfmt='none', basefmt='none', label='NMC622', bottom=0)


# MnO2Dec19.csv
df = pd.read_csv('NMC622_NEI_New1_2_23_23.csv', header=0, sep=',',
                 parse_dates=False, engine='python', names=['two theta', 'y_obs', 'y_calc', 'y_bkg', 'diff'])
# print(df.head(5))

df2 = pd.read_csv('NMC622.xy', header=0, sep='      ',
                  parse_dates=False, engine='python', names = ['two theta', 'I', 'I2'])
# print(df2.head(5))


fig, sax2 = plt.subplots(1, 1, figsize=(6, 4))

# sax2.stem(LSx, LSi, linefmt = '-g', markerfmt = 'none', basefmt = 'none', label = 'Li\N{SUBSCRIPT TWO}S', bottom = y)
# sax2.stem(LIx, LIi, linefmt = '-r',  markerfmt = 'none', basefmt = 'none', label = 'LiI', bottom = y)
plotXRD(df, 1, 100)
#plotSTEM(df2, 5)

plt.yticks([])
# plt.xticks(np.arange(10, 90, step=10))
sax2.yaxis.set_visible(False)
sax2.set_xlabel('2 theta (degrees)')
# adder = adder + stack_amount1
sax2.legend(frameon=False, bbox_to_anchor=(1.02, .15), loc=3, ncol=1, borderaxespad=0, fontsize=14)
sax2.tick_params(direction='in', bottom=True, top=True, left=False, right=False, length=5)
sax2.set_xlim([10, 80])
# sax2.set_xlim([20,50])
sax2.xaxis.set_minor_locator(MultipleLocator(5))
sax2.tick_params(which='minor', direction='in', bottom=True, top=True, left=False, right=False, length=3)
plt.legend(bbox_to_anchor=(1.05, 1.15), loc='upper right', frameon=False, fontsize=14)
#plt.show()

plt.savefig('NCM622_New xrd v2.png', dpi=500, bbox_inches='tight')