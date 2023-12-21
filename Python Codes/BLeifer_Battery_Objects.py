import numpy as np
import pylab
import glob
import pandas as pd
import scipy.optimize as optimize
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
# from Functions import readMPTData
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import pydantic as pyd
from pydantic import BaseModel, Field, dataclass, validator
from typing import List, Optional, Union, Dict, Any
from pymongo import MongoClient


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


class Battery_Component(BaseModel):
    """
    A class used to represent a battery component

    ...
    """
    name: str
    client: Optional[MongoClient] = lambda: MongoClient("localhost", 27017)
    data: Optional[pd.DataFrame] = None
    file_path: Optional[str] = None
    parents: Optional[list] = None
    children: Optional[list] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize()

    def tkinter_file_selection(self):
        root = tk.Tk()
        root.withdraw()
        self.file_path = filedialog.askopenfilename()
        self.data = self.process_data(self.file_path)
    def initialize(self):

        print('%s initialized' % self.name)

    def process_data(self, file_path):
        if ('.mpt' in file_path):
            self.data = readMPTData_CV(file_path)
        return self.data


class Current_Collector(Battery_Component):
    type: str
    thickness: float
    conductivity: Optional[float]
    density: Optional[float]


class Cathode(Battery_Component):
    current_collector: Current_Collector
    active_material: str
    active_material_fraction: float
    binder: Optional[str]
    binder_fraction: Optional[float]
    conductive_additive: Optional[str]
    conductive_additive_fraction: Optional[float]
    specific_area: Optional[float]
    density: Optional[float]
    thickness: Optional[float]
    porosity: Optional[float]
    tortuosity: Optional[float]
    conductivity: Optional[float]


class Anode(Battery_Component):
    current_collector: Current_Collector
    active_material: str
    active_material_fraction: float
    binder: Optional[str]
    binder_fraction: Optional[float]
    conductive_additive: Optional[str]
    conductive_additive_fraction: Optional[float]
    specific_area: Optional[float]
    density: Optional[float]
    thickness: Optional[float]
    porosity: Optional[float]
    tortuosity: Optional[float]
    conductivity: Optional[float]


class Separator(Battery_Component):
    type: str
    thickness: float
    porosity: Optional[float]
    tortuosity: Optional[float]
    conductivity: Optional[float]
    density: Optional[float]


class Electrolyte(Battery_Component):
    type: str
    salt: str
    concentration: Optional[float]
    viscosity: Optional[float]
    density: Optional[float]


class Battery(Battery_Component):
    cathode: Cathode
    anode: Anode
    separator: Separator
    electrolyte: Electrolyte
    specific_heat: Optional[float]
    thermal_conductivity: Optional[float]
    electrochemical_reaction_stoichiometry: Optional[float]
    electrochemical_reaction_modulus: Optional[float]
    electrochemical_reaction_order: Optional[float]
    electrochemical_reaction_prefactor: Optional[float]
    electrochemical_reaction_activation_energy: Optional[float]
    electrochemical_reaction_pre_exponential_factor: Optional[float]
    electrochemical_reaction_transfer_coefficient: Optional[float]
    electrochemical_reaction_exchange_current_density: Optional[float]
    electrochemical_reaction_exchange_current_density_units: Optional[str]
    electrochemical_reaction_exchange_current_density_units_conversion_factor: Optional[float]
    electrochemical_reaction_exchange_current_density_units_SI: Optional[str]
    electrochemical_reaction_exchange_current_density_units_SI_conversion_factor: Optional[float]
    electrochemical_reaction_exchange_current_density_units_non_SI: Optional[str]


if __name__ == '__main__':
    # **** Get data
    print('hello')
