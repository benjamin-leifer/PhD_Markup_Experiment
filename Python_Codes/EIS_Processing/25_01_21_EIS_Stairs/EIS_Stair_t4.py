import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from impedance.models.circuits import CustomCircuit
import pandas as pd
from tkinter import Tk, filedialog

# Constants for column mapping
VOLTAGE_COLUMN_ALTERNATIVES = ["Ewe/V", "Voltage", "Ewe (V)", "1.1"]
CAPACITY_COLUMN_ALTERNATIVES = ["Capacity/mA.h", "Capacity (mA.h)", "Q/mA.h", "1.2"]

def select_files(title="Select Files", filetypes=(("MPT files", "*.mpt"), ("All files", "*.*"))):
    root = Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    return list(files)

def select_directory(title="Select Directory"):
    root = Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title=title)
    return directory

def rename_columns(data, file_path):
    """
    Dynamically renames columns based on known alternatives.
    """
    # Attempt to find the correct voltage and capacity columns
    voltage_col = next((col for col in VOLTAGE_COLUMN_ALTERNATIVES if col in data.columns), None)
    capacity_col = next((col for col in CAPACITY_COLUMN_ALTERNATIVES if col in data.columns), None)

    if voltage_col and capacity_col:
        # Rename columns to standardized names
        data = data.rename(columns={voltage_col: "Voltage (V)", capacity_col: "Capacity (mA.h)"})
        return data
    else:
        raise ValueError(
            f"Error: Missing required columns in {file_path}. Available columns: {list(data.columns)}"
        )

def extract_voltage_capacity_from_file(file_path):
    """
    Extracts voltage vs capacity data dynamically.
    Returns None if valid columns cannot be inferred.
    """
    try:
        header_lines = 50
        data = pd.read_csv(
            file_path,
            skiprows=header_lines,
            delimiter="\t",
            engine="python",
            encoding="cp1252",
            on_bad_lines="skip"
        )
        data.columns = [col.strip() for col in data.columns]
        print(f"Columns in {file_path}: {data.columns}")  # Debugging step

        # Rename columns dynamically
        data = rename_columns(data, file_path)
        voltage_capacity_data = data[["Voltage (V)", "Capacity (mA.h)"]].dropna()
        return voltage_capacity_data
    except Exception as e:
        print(f"Error processing voltage file {file_path}: {e}")
        return None

def read_eis_data(file_path):
    """
    Reads EIS data from a file and returns a DataFrame. Logs an error if the file is invalid.
    """
    try:
        with open(file_path, 'r', encoding='cp1252') as file:
            lines = file.readlines()
            header_lines = 0
            for line in lines:
                if "Nb header lines" in line:
                    header_lines = int(line.split(":")[1].strip())
                    break
        data = pd.read_csv(
            file_path,
            skiprows=header_lines,
            delimiter="\t",
            engine="python",
            encoding="cp1252",
            on_bad_lines="skip"
        )
        data.columns = [col.strip() for col in data.columns]

        # Rename EIS columns if necessary
        if len(data.columns) >= 3:
            data.columns = [
                "Frequency (Hz)", "Re(Z) (Ohm)", "-Im(Z) (Ohm)", "|Z| (Ohm)",
                "Phase (deg)", "Time (s)", "Ewe (V)", "I (mA)",
                "Cs (µF)", "Cp (µF)", "Cycle Number", "I Range",
                "|Ewe| (V)", "|I| (A)", "Re(Y) (Ohm⁻¹)", "Im(Y) (Ohm⁻¹)",
                "|Y| (Ohm⁻¹)", "Phase(Y)"
            ][:len(data.columns)]
        return data
    except Exception as e:
        print(f"Error reading EIS file {file_path}: {e}")
        return None

# Main Script
print("Select Voltage Data Files...")
voltage_files = select_files("Select Voltage Data Files")
print("Select EIS Data Files...")
eis_files = select_files("Select EIS Data Files")
print("Select Save Directory...")
save_dir = select_directory("Select Save Directory")

if len(voltage_files) != len(eis_files):
    raise ValueError("The number of voltage files must match the number of EIS files!")

# Load voltage datasets
voltage_datasets = []
for file in voltage_files:
    voltage_data = extract_voltage_capacity_from_file(file)
    if voltage_data is not None:
        voltage_datasets.append(voltage_data)
    else:
        print(f"Skipped voltage file: {file}")

# Load EIS datasets
eis_datasets = []
for file in eis_files:
    eis_data = read_eis_data(file)
    if eis_data is not None:
        eis_datasets.append(eis_data)
    else:
        print(f"Skipped EIS file: {file}")

# Ensure datasets are valid
if not voltage_datasets or not eis_datasets:
    raise ValueError("Failed to load valid datasets. Check your input files for proper formatting.")

print(f"Loaded {len(voltage_datasets)} voltage datasets and {len(eis_datasets)} EIS datasets.")
