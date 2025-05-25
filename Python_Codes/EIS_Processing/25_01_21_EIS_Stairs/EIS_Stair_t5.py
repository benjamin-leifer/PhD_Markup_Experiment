import os
import numpy as np
import pandas as pd
from impedance.models.circuits import CustomCircuit

# Define the circuit
CIRCUIT = 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4)-W'
DEFAULT_INITIAL_GUESS = [10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300, 10]
OUTPUT_FILE = "ecm_parameters.csv"

BACKUP_CIRCUIT = 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4)'
BACKUP_DEFAULT_INITIAL_GUESS = [10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300]


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

def validate_data(frequencies, Z_real, Z_imag):
    """
    Validate data arrays, ensuring no NaN/Inf values and sufficient range.
    """
    valid_indices = ~(
            (np.isnan(frequencies) | np.isnan(Z_real) | np.isnan(Z_imag)) |
            (np.isinf(frequencies) | np.isinf(Z_real) | np.isinf(Z_imag)) |
            (frequencies <= 0)
    )

    return frequencies[valid_indices], Z_real[valid_indices], Z_imag[valid_indices]

def get_initial_guess(file_name, cycle_number, output_file):
    """
    Retrieves the initial guess for a fit from existing output or returns default.
    """
    if os.path.exists(output_file):
        try:
            # Read the output CSV file
            previous_results = pd.read_csv(output_file)

            # Find rows matching the current file and cycle number
            matching_row = previous_results[
                (previous_results["Filename"] == file_name) &
                (previous_results["Cycle Number"] == cycle_number)
                ]

            # If a matching row exists, return its parameters
            if not matching_row.empty:
                param_names = [col for col in previous_results.columns if col not in ["Filename", "Cycle Number", "Mean Residual (%)"]]
                return matching_row.iloc[0][param_names].values.tolist()

        except Exception as e:
            print(f"Error reading previous results for {file_name}, Cycle {cycle_number}: {e}")

    # Return the default initial guess if no prior result is found
    return DEFAULT_INITIAL_GUESS

def fit_cycle(cycle_data, file_name, cycle_number):
    try:
        frequencies = cycle_data["Frequency (Hz)"].values[3:]
        Z_real = cycle_data["Re(Z) (Ohm)"].values[3:]
        Z_imag = -cycle_data["-Im(Z) (Ohm)"].values[3:]
        Z_exp = Z_real + 1j * Z_imag

        # Validate and clean data
        frequencies, Z_real, Z_imag = validate_data(frequencies, Z_real, Z_imag)
        if len(frequencies) == 0:
            print(f"Cycle {cycle_number} in file {file_name} has insufficient valid data. Skipping.")
            return

        # Attempt fitting with initial guess
        initial_guess = get_initial_guess(file_name, cycle_number, OUTPUT_FILE)
        print(f"Initial guess for {file_name}, Cycle {cycle_number}: {initial_guess}")

        circuit_model = CustomCircuit(CIRCUIT, initial_guess=initial_guess)
        best_fit_parameters = None

        try:
            circuit_model.fit(frequencies, Z_exp, method='trf', maxfev=1000)
            print(f"Fit results for {file_name}, Cycle {cycle_number}: {circuit_model.parameters_}")
            best_fit_parameters = circuit_model.parameters_
        except Exception as e:
            print(f"Fitting failed for {file_name}, Cycle {cycle_number}: {e}")
            backup_circuit_model = CustomCircuit(BACKUP_CIRCUIT, initial_guess=BACKUP_DEFAULT_INITIAL_GUESS)
            try:
                backup_circuit_model.fit(frequencies, Z_exp, method='trf', maxfev=1000)
                best_fit_parameters = backup_circuit_model.parameters_
            except Exception as e:
                print(f"Backup fitting failed for {file_name}, Cycle {cycle_number}: {e}")

        if best_fit_parameters is not None:
            Z_fit = circuit_model.predict(frequencies)
            residuals = np.mean((np.abs(Z_exp) - np.abs(Z_fit)) / np.abs(Z_exp) * 100)
            print(f"Mean Residual for {file_name}, Cycle {cycle_number}: {residuals:.2f}%")
            print(f"Best fit parameters: {best_fit_parameters}")
            print(f"Parameter names{circuit_model.get_param_names()[0]}")
            # Save results incrementally
            results = {
                "Filename": file_name,
                "Cycle Number": cycle_number,
                **{param: val for param, val in zip(circuit_model.get_param_names()[0], tuple(best_fit_parameters))},
                "Mean Residual (%)": residuals,
            }
            save_to_csv(results, OUTPUT_FILE)
        else:
            print(f"No fit results for {file_name}, Cycle {cycle_number}. Skipping.")

    except Exception as e:
        print(f"Error fitting cycle {cycle_number} in file {file_name}: {e}")

def save_to_csv(results, output_file):
    """Saves or updates results in the CSV file."""
    try:
        results_df = pd.DataFrame([results])
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            mask = (
                    (existing_df["Filename"] == results["Filename"]) &
                    (existing_df["Cycle Number"] == results["Cycle Number"])
            )
            if mask.any():
                existing_df.loc[mask, :] = results_df.iloc[0]
            else:
                existing_df = pd.concat([existing_df, results_df], ignore_index=True)
        else:
            existing_df = results_df

        existing_df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")

def process_file(file_path):
    """Processes a single EIS file."""
    print(f"Processing {file_path}...")
    try:
        data = read_eis_data(file_path)
        if data is None or "Cycle Number" not in data.columns:
            print(f"File {file_path} is missing required columns. Skipping.")
            return

        for cycle_number, cycle_data in data.groupby("Cycle Number"):
            fit_cycle(cycle_data, os.path.basename(file_path), cycle_number)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def traverse_directory(directory):
    """Finds and processes MPT files in the directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if "PEIS" in file.upper() and file.endswith(".mpt"):
                process_file(os.path.join(root, file))


if __name__ == "__main__":
    from tkinter import filedialog

    directory = filedialog.askdirectory(title="Select Directory to Process")
    if not directory:
        print("No directory selected. Exiting.")
    else:
        print(f"Processing directory: {directory}")
        traverse_directory(directory)
        print(f"Fitting results saved to {OUTPUT_FILE}.")
