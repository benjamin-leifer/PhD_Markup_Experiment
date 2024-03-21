# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:11:36 2022

@author: Eric Zimmerer
"""

import h5py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import tkinter as tk
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from tkinter.filedialog import askopenfilename

aaa = []

root = Tk()
root.withdraw()
root.update()
tk.messagebox.showwarning(title="Import h5 file", message="Select H5 files from Raman")
Files = askopenfilenames()
root.destroy()

directory = os.path.dirname(os.getcwd())
#directory = directory + '/Raman Figures'
os.chdir(directory)

# Files = os.listdir()
extractedh5 = []
scan_times = []
scan_numbers = []
for count, filename in enumerate(Files):
    h5data = {}
    h5dataclean = {}
    h5 = h5py.File(filename, 'r')
    h5data['file name'] = filename
    # convert h5 file into a dictionary of h5 groups
    firstlevelkeys = list(h5.keys())
    print(firstlevelkeys)
    for key1 in firstlevelkeys:
        h5data[key1] = h5[key1]
        print(list(h5data[key1].keys()))
        secondlevelkeys = list(h5data[key1].keys())
        for key2 in secondlevelkeys:
            h5data[key2] = h5data[key1][key2]
    # extract data from h5 files
    Addendum = h5data['Addendum']['History']['Part-0']
    FileInfo = h5data['FileInfo']['MetaData']
    Mosaic = {}
    for Image in h5data['Mosaic'].keys():
        Mosaic[str(Image)] = h5data['Mosaic'][Image]

    Regions = h5data['Regions']['Region102']
    break

if __name__ == '__main__':
    print('Eric_HD5_Import_Start.py executed')
