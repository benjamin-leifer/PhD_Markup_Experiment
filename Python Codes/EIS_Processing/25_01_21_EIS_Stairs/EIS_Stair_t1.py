import numpy as np
import matplotlib.pyplot as plt
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
import pandas as pd

def extract_voltage_per_cycle(eis_data, cycle_num):
    """
    Extract the working voltage at which the EIS spectrum is taken for a given cycle.

    Parameters:
        eis_data (pd.DataFrame): The full EIS dataset.
        cycle_num (int): The cycle number for which the voltage is extracted.

    Returns:
        float: The voltage for the given cycle.
    """
    # Filter data for the specific cycle
    cycle_data = eis_data[eis_data["Cycle Number"] == cycle_num]

    # Ensure the voltage column exists and extract its unique value
    if "Ewe (V)" in cycle_data.columns:
        # Voltage should be the same for all rows within a cycle
        voltage = cycle_data["Ewe (V)"].unique()

        if len(voltage) == 1:
            return voltage[0]  # Return the single voltage value
        elif len(voltage) > 1:
            print(f"Warning: Multiple voltage values detected for Cycle {cycle_num}. Using the first value.")
            return voltage[0]
        else:
            print(f"Error: No voltage data found for Cycle {cycle_num}.")
            return None
    else:
        print(f"Error: 'Ewe (V)' column is missing in the dataset.")
        return None


def read_eis_data(file_path):
    """
    Reads an EIS data file (.mpt) and extracts impedance data into a pandas DataFrame.
    """
    try:
        # Detect header lines
        with open(file_path, 'r', encoding='cp1252') as file:
            lines = file.readlines()
            header_lines = 0
            for line in lines:
                if "Nb header lines" in line:
                    header_lines = int(line.split(":")[1].strip())
                    break

        # Load the data
        data = pd.read_csv(
            file_path,
            skiprows=header_lines,
            delimiter='\t',
            engine='python',
            encoding='cp1252'
        )

        # Clean column names
        data.columns = [col.strip() for col in data.columns]

        # Rename columns if sufficient data is present
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
        print(f"Error reading EIS data: {e}")
        return None

def fit_eis_cycle(eis_data, cycle_num=1, exc_start=3, plot=True):
    if eis_data is not None and "Cycle Number" in eis_data.columns:
        # Filter Cycle 1 data
        cycle_data = eis_data[eis_data["Cycle Number"] == cycle_num]

        # Extract working voltage
        voltage = extract_voltage_per_cycle(eis_data, cycle_num)
        if voltage is None:
            print(f"Skipping Cycle {cycle_num} due to missing voltage.")
            return pd.DataFrame()  # Return an empty DataFrame for invalid cycles

        # Validate the critical columns for real and imaginary impedance
        if "Re(Z) (Ohm)" in cycle_data and "-Im(Z) (Ohm)" in cycle_data:
            # Extract frequency, real, and imaginary impedance
            frequencies = cycle_data["Frequency (Hz)"].values[exc_start:-1]
            Z_real_exp = cycle_data["Re(Z) (Ohm)"].values[exc_start:-1]
            Z_imag_exp = -cycle_data["-Im(Z) (Ohm)"].values[exc_start:-1]

            # Ensure extracted data is numeric
            Z_real_exp = np.asarray(Z_real_exp, dtype=float)
            Z_imag_exp = np.asarray(Z_imag_exp, dtype=float)

            # Combine real and imaginary impedance
            Z_exp = Z_real_exp + 1j * Z_imag_exp

            print(f"Extracted data for Cycle {cycle_num}: {len(frequencies)} points.")

            # Define the equivalent circuit model
            circuit = 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4-W)'

            # Initial guesses for parameters
            initial_guess = [10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300, 10]

            # Create and fit the circuit
            circuit_model = CustomCircuit(circuit, initial_guess=initial_guess)
            circuit_model.fit(frequencies, Z_exp)

            # Extract parameter names and values
            param_names = circuit_model.get_param_names()
            param_values = circuit_model.parameters_

            # Ensure both are flat lists of the same length
            if isinstance(param_names[0], list):
                param_names = [item for sublist in param_names for item in sublist]
            if isinstance(param_values[0], list):
                param_values = [item for sublist in param_values for item in sublist]

            # Create a dictionary of parameter names and values
            fit_results = {'Cycle': cycle_num}
            for name, value in zip(param_names, param_values):
                fit_results[name] = value

            fit_results['Voltage (V)'] = voltage
            # Optionally plot the data
            if plot:
                fig, ax = plt.subplots(figsize=(8, 6))  # Ensure `ax` is a valid Axes object

                # Plot experimental data
                plot_nyquist(Z_exp, fmt='o', label="Experimental Data", ax=ax)  # Correct argument order

                # Plot fitted data
                plot_nyquist(circuit_model.predict(frequencies), fmt='-', label="Fitted Circuit", ax=ax)

                plt.legend()
                plt.grid()
                plt.show()

            return pd.DataFrame([fit_results])  # Return a single-row DataFrame

        else:
            print("Critical columns missing for real/imaginary impedance.")
    else:
        print("Failed to load or filter EIS data.")
    return pd.DataFrame()  # Return an empty DataFrame in case of errors

def extract_voltage_capacity(file_path):
    """
    Extracts voltage vs capacity data from an .mpt file.

    Parameters:
        file_path (str): Path to the .mpt file.

    Returns:
        pd.DataFrame: DataFrame with 'Voltage (V)' and 'Capacity (mAh)' columns.
    """
    try:
        # Detect header lines
        with open(file_path, 'r', encoding='cp1252') as file:
            lines = file.readlines()
            header_lines = 0
            for line in lines:
                if "Nb header lines" in line:
                    header_lines = int(line.split(":")[1].strip())
                    break

        # Load the data
        data = pd.read_csv(
            file_path,
            skiprows=header_lines,
            delimiter='\t',
            engine='python',
            encoding='cp1252'
        )

        # Clean column names
        data.columns = [col.strip() for col in data.columns]

        # Identify relevant columns
        if "Ewe (V)" in data.columns and "Capacity (mAh)" in data.columns:
            voltage_capacity_data = data[["Ewe (V)", "Capacity (mAh)"]].dropna()
            voltage_capacity_data.columns = ["Voltage (V)", "Capacity (mAh)"]  # Standardize column names
            return voltage_capacity_data

        else:
            print("Error: Relevant columns ('Ewe (V)' and 'Capacity (mAh)') not found in the dataset.")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()

# File path to EIS data
file_path = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Results\Low Temp Li Ion\2025\01\EIS Stairs\CV01_Cycle_1\BL-LL-CV01_EIS_STAIR_RT_05_CP_C03_01_PEIS_C04.mpt"

# File paths
voltage_capacity_file = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Results\Low Temp Li Ion\2025\01\EIS Stairs\CV02_Cycle_1\BL-LL-CV02_EIS_STAIR_RT_02_CP_C02.mpt"
eis_file = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Results\Low Temp Li Ion\2025\01\EIS Stairs\CV01_Cycle_1\BL-LL-CV01_EIS_STAIR_RT_05_CP_C03_01_PEIS_C04.mpt"

# Read EIS data

# Read the EIS data
eis_data = read_eis_data(file_path)
print(eis_data)
# Fix to avoid deprecation
# Fix to avoid deprecation and handle the fitted parameters
fits_list = []  # Use a list to collect fitted parameters from all cycles

for cycle in range(1, int(eis_data["Cycle Number"].max())):
    # Collect fitted results for each cycle
    fit_result = fit_eis_cycle(eis_data, cycle_num=cycle, exc_start=3, plot=False)
    fits_list.append(fit_result)

# Combine all fitted results into a single DataFrame
fits = pd.concat(fits_list, ignore_index=True)

# Print the final DataFrame of fitted parameters
print(fits)
# Save the fitted parameters to a CSV file
csv_save_path = "fitted_parameters for CV01.csv"
fits.to_csv(csv_save_path, index=False)
