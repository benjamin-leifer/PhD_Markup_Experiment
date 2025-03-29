import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from impedance.models.circuits import CustomCircuit
import pandas as pd
from tkinter import Tk, filedialog
from concurrent.futures import ProcessPoolExecutor

def select_files(title="Select Files", filetypes=(("MPT files", "*.mpt"), ("All files", "*.*"))):
    """
    Opens a file dialog to select multiple files.

    Parameters:
        title (str): Title of the file dialog.
        filetypes (tuple): Allowed file types.

    Returns:
        list: List of selected file paths.
    """
    root = Tk()
    root.withdraw()  # Hide the root window
    files = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    return list(files)

def select_directory(title="Select Directory"):
    """
    Opens a directory selection dialog.

    Parameters:
        title (str): Title of the dialog.

    Returns:
        str: Path of the selected directory.
    """
    root = Tk()
    root.withdraw()  # Hide the root window
    directory = filedialog.askdirectory(title=title)
    return directory

def fit_cycle(cycle_data, circuit, exc_start=4):
    """
    Fits EIS data for a single cycle with the specified equivalent circuit model.

    Parameters:
        cycle_data (pd.DataFrame): EIS data for a single cycle.
        circuit (str): Equivalent circuit string for fitting.
        exc_start (int): Index to start extracting data from.

    Returns:
        dict: Fitted results, including frequencies, Z_exp, Z_fit, and residuals.
    """
    frequencies = cycle_data["Frequency (Hz)"].values[exc_start:-1]
    Z_real = cycle_data["Re(Z) (Ohm)"].values[exc_start:-1]
    Z_imag = -cycle_data["-Im(Z) (Ohm)"].values[exc_start:-1]
    Z_exp = Z_real + 1j * Z_imag

    initial_guess = [10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300, 10]
    circuit_model = CustomCircuit(circuit, initial_guess=initial_guess)
    circuit_model.fit(frequencies, Z_exp)
    Z_fit = circuit_model.predict(frequencies)
    residuals_percent = (np.abs(Z_exp) - np.abs(Z_fit)) / np.abs(Z_exp) * 100

    fitted_params = circuit_model.parameters_
    return {
        "frequencies": frequencies,
        "Z_exp": Z_exp,
        "Z_fit": Z_fit,
        "residuals_percent": residuals_percent,
        "parameters": fitted_params
    }

def generate_summary(fitting_results):
    """
    Generates a summary DataFrame of fitted parameters and residuals.

    Parameters:
        fitting_results (dict): Dictionary of fitted results for all datasets and cycles.

    Returns:
        pd.DataFrame: Summary DataFrame with fitted parameters and residuals.
    """
    summary = []
    for dataset_idx, dataset_results in fitting_results.items():
        for cycle_num, result in dataset_results.items():
            params = result["parameters"]
            residuals = np.mean(result["residuals_percent"])
            summary.append({
                "Dataset": dataset_idx + 1,
                "Cycle": cycle_num,
                "R1 (Ohm)": params[0],
                "R2 (Ohm)": params[1],
                "CPE2_0 (Ohm⁻¹sᵃ)": params[2],
                "Residuals (%)": residuals
            })

    return pd.DataFrame(summary)

def extract_voltage_capacity_from_file(file_path):
    """
    Extracts voltage vs capacity data from the provided .mpt file.

    Parameters:
        file_path (str): Path to the .mpt file.

    Returns:
        pd.DataFrame: DataFrame with 'Voltage (V)' and 'Capacity (mA.h)' columns.
    """

    try:
        # Define header lines to skip
        header_lines = 50

        # Load the data section
        data = pd.read_csv(
            file_path,
            skiprows=header_lines,
            delimiter="\t",
            engine="python",
            encoding="cp1252",
            on_bad_lines="skip"
        )
        # Inside extract_voltage_capacity_from_file
        if len(data.columns) == 1:
            raise ValueError(f"File {file_path} has unexpected formatting with only 1 column.")
            return pd.DataFrame()
        # Clean column names
        data.columns = [col.strip() for col in data.columns]
        print("Parsed Columns:", data.columns)  # Debugging step

        # Handle cases where columns are not parsed correctly
        if "Ewe/V" not in data.columns or "Capacity/mA.h" not in data.columns:
            print("Error: Expected columns not found. Manually assigning column names.")
            data.columns = [
                "mode", "ox/red", "error", "control changes", "Ns changes", "counter inc.",
                "Ns", "time/s", "control/mA", "Ewe/V", "I/mA", "dQ/C", "(Q-Qo)/C", "half cycle",
                "Q charge/discharge/mA.h", "Energy charge/W.h", "Energy discharge/W.h",
                "Capacitance charge/µF", "Capacitance discharge/µF", "Q discharge/mA.h",
                "Q charge/mA.h", "Capacity/mA.h", "Efficiency/%", "cycle number", "P/W"
            ]
            print("Manually assigned columns:", data.columns)

        # Ensure the required columns exist
        if "Ewe/V" in data.columns and "Capacity/mA.h" in data.columns:
            voltage_capacity_data = data[["Ewe/V", "Capacity/mA.h"]].dropna()
            voltage_capacity_data.columns = ["Voltage (V)", "Capacity (mA.h)"]  # Standardize column names
            print("Extracted Voltage vs Capacity Data:")
            print(voltage_capacity_data.head())
            return voltage_capacity_data
        else:
            print("Error: Required columns 'Ewe/V' and 'Capacity/mA.h' not found.")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()

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
            delimiter="\t",
            engine="python",
            encoding="cp1252",
            on_bad_lines="skip"  # Skip problematic lines
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

def fit_eis_data_parallel(eis_data, circuit, exc_start=4):
    """
    Fits EIS data for all cycles in parallel.

    Parameters:
        eis_data (pd.DataFrame): EIS data containing all cycles.
        circuit (str): Equivalent circuit string for fitting.
        exc_start (int): Index to start extracting data from.

    Returns:
        dict: Dictionary of fitted results for each cycle.
    """
    fitting_results = {}

    # Group EIS data by cycle
    cycles = [(cycle_num, cycle_data) for cycle_num, cycle_data in eis_data.groupby("Cycle Number")]

    # Fit each cycle in parallel
    with ProcessPoolExecutor() as executor:
        results = executor.map(lambda args: (args[0], fit_cycle(args[1], circuit, exc_start)), cycles)

    # Store results in a dictionary
    for cycle_num, result in results:
        fitting_results[cycle_num] = result

    return fitting_results

def plot_voltage_capacity(ax, voltage_capacity_data, eis_voltages, cycle_colors):
    """
    Plots Voltage vs Capacity data and adds markers for EIS spectra.

    Parameters:
        ax (matplotlib.axes.Axes): Axis to plot on.
        voltage_capacity_data (pd.DataFrame): Voltage vs Capacity data.
        eis_voltages (list): Voltages at which EIS spectra were recorded.
        cycle_colors (list): Colors corresponding to each cycle.
    """
    # Plot the Voltage vs Capacity curve
    ax.plot(voltage_capacity_data["Capacity (mA.h)"], voltage_capacity_data["Voltage (V)"],
            label="Voltage vs Capacity", color="black")

    # Add markers for EIS spectra
    for idx, voltage in enumerate(eis_voltages):
        # Find the closest capacity for the given voltage
        closest_row = voltage_capacity_data.iloc[(voltage_capacity_data["Voltage (V)"] - voltage).abs().idxmin()]
        capacity = closest_row["Capacity (mA.h)"]

        # Plot the marker
        ax.scatter(capacity, voltage, color=cycle_colors[idx], label=f"Cycle {idx + 1} EIS Point")

    ax.set_xlabel("Capacity (mA.h)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltage vs Capacity with EIS Markers")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True)

def plot_3d_nyquist_with_fit(ax, frequencies, Z_exp, Z_fit, voltage):
    """
    Plots 3D Nyquist data with experimental and fitted impedance.

    Parameters:
        ax (mpl_toolkits.mplot3d.Axes3D): 3D Axes for the Nyquist plot.
        frequencies (np.array): Frequency data.
        Z_exp (np.array): Experimental impedance data.
        Z_fit (np.array): Fitted impedance data.
        voltage (float): Voltage at which the spectrum was taken.
    """
    # Split real and imaginary parts for both experimental and fitted data
    Z_real_exp, Z_imag_exp = Z_exp.real, -Z_exp.imag
    Z_real_fit, Z_imag_fit = Z_fit.real, -Z_fit.imag

    # Plot experimental data
    ax.scatter(Z_real_exp, Z_imag_exp, voltage, color="red", label="Experimental Data", s=20)

    # Plot fitted data
    ax.plot(Z_real_fit, Z_imag_fit, zs=voltage, zdir='z', color="blue", label="Fitted Circuit")

    ax.set_xlabel("Re(Z) (Ohm)")
    ax.set_ylabel("-Im(Z) (Ohm)")
    ax.set_zlabel("Voltage (V)")
    ax.legend()
    ax.grid(True)

def plot_combined_3d_nyquist_with_fits(ax, eis_data, circuit,exc_start=4):
    """
    Plots a combined 3D Nyquist plot with experimental and fitted data for all cycles.

    Parameters:
        ax (mpl_toolkits.mplot3d.Axes3D): 3D Axes for the Nyquist plot.
        eis_data (pd.DataFrame): EIS data containing all cycles.
        circuit (str): Equivalent circuit string for fitting.
    """
    cycle_colors = plt.cm.viridis(np.linspace(0, 1, len(eis_data["Cycle Number"].unique())))

    # Loop through each cycle
    for idx, (cycle_num, cycle_data) in enumerate(eis_data.groupby("Cycle Number")):
        frequencies = cycle_data["Frequency (Hz)"].values[exc_start:-1]
        Z_real = cycle_data["Re(Z) (Ohm)"].values[exc_start:-1]
        Z_imag = -cycle_data["-Im(Z) (Ohm)"].values[exc_start:-1]
        Z_exp = Z_real + 1j * Z_imag

        # Fit the circuit model
        initial_guess = [10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300, 10]
        #initial_guess = [10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300, 10, 0.01, 0.1, 0.8]
        circuit_model = CustomCircuit(circuit, initial_guess=initial_guess)
        circuit_model.fit(frequencies, Z_exp)
        Z_fit = circuit_model.predict(frequencies)
        residuals_percent = (abs(Z_exp) - abs(Z_fit ) )/ abs(Z_exp) *100

        # Extract voltage for this cycle
        voltage = cycle_data["Ewe (V)"].iloc[0]

        # Plot experimental data
        ax.scatter(voltage, Z_real,zs=-Z_imag, color=cycle_colors[idx], label=f"Cycle {int(cycle_num)} (Exp)", s=20)

        # Plot fitted data
        ax.plot(voltage, Z_fit.real, zs=-Z_fit.imag,  color=cycle_colors[idx], linestyle='-', label=f"Cycle {int(cycle_num)} (Fit)")

    # Label axes
    ax.set_ylabel("Re(Z) (Ohm)")
    ax.set_zlabel("Im(Z) (Ohm)")
    ax.set_xlabel("Voltage (V)")
    ax.set_title("3D Nyquist Plot with Fits")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True)

def plot_residuals(ax, frequencies, Z_exp, Z_fit, color = 'blue', legend = ''):
    """
    Plots percentage residuals between experimental and fitted impedance data.

    Parameters:
        ax (matplotlib.axes.Axes): Axis to plot on.
        frequencies (np.array): Frequency data (plotted on log scale).
        Z_exp (np.array): Experimental impedance data.
        Z_fit (np.array): Fitted impedance data.
    """
    # Calculate magnitudes of impedance
    magnitude_exp = np.abs(Z_exp)
    magnitude_fit = np.abs(Z_fit)

    # Calculate percentage residuals
    residuals_percent = (magnitude_exp - magnitude_fit) / magnitude_exp * 100

    # Plot the residuals
    #ax.semilogx(frequencies, residuals_percent, color=color, label="Residuals (%): "+legend)
    ax.semilogy(frequencies, abs(residuals_percent), color=cycle_colors[idx],
                 label=f"Cycle {int(cycle_num)} Residuals (log scale)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Residuals (%)")
    ax.set_title("Residuals vs Frequency")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True, which="both", linestyle="--")

def plot_residuals_loglog(ax, frequencies, Z_exp, Z_fit, color='blue', legend=''):
    """
    Plots percentage residuals on a log-log scale between experimental and fitted impedance data.

    Parameters:
        ax (matplotlib.axes.Axes): Axis to plot on.
        frequencies (np.array): Frequency data.
        Z_exp (np.array): Experimental impedance data.
        Z_fit (np.array): Fitted impedance data.
        color (str): Line color for the residuals plot.
        legend (str): Legend entry for the residuals.
    """
    # Calculate magnitudes of impedance
    magnitude_exp = np.abs(Z_exp)
    magnitude_fit = np.abs(Z_fit)

    # Calculate percentage residuals
    residuals_percent = (magnitude_exp - magnitude_fit) / magnitude_exp * 100

    # Plot residuals on a log-log scale
    ax.loglog(frequencies, np.abs(residuals_percent), color=color, label=legend)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Residuals (%)")
    ax.set_title("Log-Log Plot of Residuals vs Frequency")
    ax.legend()
    ax.grid(True, which="both", linestyle="--")

def fit_eis_data(eis_data, circuit, exc_start=4):
    """
    Fits EIS data with the specified equivalent circuit model.

    Parameters:
        eis_data (pd.DataFrame): EIS data containing all cycles.
        circuit (str): Equivalent circuit string for fitting.
        exc_start (int): Index to start extracting data from.

    Returns:
        dict: Dictionary of fitting results for each cycle.
    """
    fitting_results = {}

    # Loop through each cycle
    for idx, (cycle_num, cycle_data) in enumerate(eis_data.groupby("Cycle Number")):
        frequencies = cycle_data["Frequency (Hz)"].values[exc_start:-1]
        Z_real = cycle_data["Re(Z) (Ohm)"].values[exc_start:-1]
        Z_imag = -cycle_data["-Im(Z) (Ohm)"].values[exc_start:-1]
        Z_exp = Z_real + 1j * Z_imag

        # Initial guess for the circuit model
        initial_guess = [10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300, 10]
        circuit_model = CustomCircuit(circuit, initial_guess=initial_guess)

        try:
            # Fit the circuit model
            print(f"Fitting Cycle {cycle_num}...")
            try:
                circuit_model.fit(frequencies, Z_exp, maxfev=1000)
            except RuntimeError:
                print(f"Warning: Fitting failed for Cycle {cycle_num}. Skipping this cycle.")
                continue
            Z_fit = circuit_model.predict(frequencies)

            # Calculate residuals
            residuals_percent = (np.abs(Z_exp) - np.abs(Z_fit)) / np.abs(Z_exp) * 100

            # Store the results for reuse
            fitting_results[cycle_num] = {
                "frequencies": frequencies,
                "Z_exp": Z_exp,
                "Z_fit": Z_fit,
                "residuals_percent": residuals_percent
            }

        except RuntimeWarning as e:
            print(f"Warning: Fitting failed for Cycle {cycle_num}. Skipping this cycle.")
            continue

    return fitting_results

def plot_voltage_capacity_multi(ax, voltage_datasets, eis_voltages, cycle_colors):
    """
    Plots multiple Voltage vs Capacity datasets and adds markers for EIS spectra.

    Parameters:
        ax (matplotlib.axes.Axes): Axis to plot on.
        voltage_datasets (list of pd.DataFrame): List of voltage vs capacity datasets.
        eis_voltages (list of list): Voltages for EIS spectra for each dataset.
        cycle_colors (list): Colors corresponding to each dataset.
    """
    for i, (voltage_data, eis_voltages_set) in enumerate(zip(voltage_datasets, eis_voltages)):
        # Plot voltage vs capacity curve
        ax.plot(voltage_data["Capacity (mA.h)"], voltage_data["Voltage (V)"],
                label=f"Voltage Data {i+1}", color=cycle_colors[i])

        # Add markers for EIS spectra
        for idx, voltage in enumerate(eis_voltages_set):
            closest_row = voltage_data.iloc[(voltage_data["Voltage (V)"] - voltage).abs().idxmin()]
            capacity = closest_row["Capacity (mA.h)"]
            ax.scatter(capacity, voltage, color=cycle_colors[idx], label=f"Data {i+1} Cycle {idx + 1}")

    ax.set_xlabel("Capacity (mA.h)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltage vs Capacity with EIS Markers")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True)

def plot_combined_3d_nyquist_with_fits_multi(ax, eis_datasets, circuit, exc_start=4):
    """
    Plots combined 3D Nyquist plots for multiple datasets with experimental and fitted data.

    Parameters:
        ax (mpl_toolkits.mplot3d.Axes3D): 3D Axes for the Nyquist plot.
        eis_datasets (list of pd.DataFrame): List of EIS datasets.
        circuit (str): Equivalent circuit string for fitting.
        exc_start (int): Index to start extracting data from.
    """
    cycle_colors = plt.cm.viridis(np.linspace(0, 1, sum(len(eis["Cycle Number"].unique()) for eis in eis_datasets)))

    color_idx = 0
    for dataset_idx, eis_data in enumerate(eis_datasets):
        for cycle_num, cycle_data in eis_data.groupby("Cycle Number"):
            frequencies = cycle_data["Frequency (Hz)"].values[exc_start:-1]
            Z_real = cycle_data["Re(Z) (Ohm)"].values[exc_start:-1]
            Z_imag = -cycle_data["-Im(Z) (Ohm)"].values[exc_start:-1]
            Z_exp = Z_real + 1j * Z_imag

            # Fit the circuit model
            initial_guess = [10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300, 10]
            circuit_model = CustomCircuit(circuit, initial_guess=initial_guess)
            circuit_model.fit(frequencies, Z_exp)
            Z_fit = circuit_model.predict(frequencies)

            # Plot experimental and fitted data
            voltage = cycle_data["Ewe (V)"].iloc[0]
            ax.scatter(Z_real, Z_imag, zs=voltage, color=cycle_colors[color_idx], label=f"Data {dataset_idx + 1} Cycle {int(cycle_num)} Exp", s=20)
            ax.plot(Z_fit.real, Z_fit.imag, zs=voltage, color=cycle_colors[color_idx], linestyle='--', label=f"Data {dataset_idx + 1} Cycle {int(cycle_num)} Fit")
            color_idx += 1

    ax.set_xlabel("Re(Z) (Ohm)")
    ax.set_ylabel("-Im(Z) (Ohm)")
    ax.set_zlabel("Voltage (V)")
    ax.set_title("Combined 3D Nyquist Plot")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True)

# Step 1: Select files once
print("Select ONE Voltage Data File...")
voltage_file = select_files("Select Voltage Data File")
if not voltage_file:
    raise ValueError("No voltage file selected.")
voltage_file = voltage_file[0]  # Only use first

print("Select ONE EIS Data File...")
eis_file = select_files("Select EIS Data File")
if not eis_file:
    raise ValueError("No EIS file selected.")
eis_file = eis_file[0]  # Only use first

print("Select Save Directory...")
save_dir = select_directory("Select Save Directory")

# Step 2: Load data
voltage_data = extract_voltage_capacity_from_file(voltage_file)
eis_data = read_eis_data(eis_file)

if voltage_data.empty or eis_data is None or eis_data.empty:
    raise ValueError("Failed to load one or both files.")

# Step 3: Process EIS voltages
eis_voltages = [cycle_data["Ewe (V)"].iloc[0] for _, cycle_data in eis_data.groupby("Cycle Number")]

# Step 4: Fit EIS
circuit = 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4)-W'
fitting_results = fit_eis_data(eis_data, circuit)


# Step 5: Plot
fig = plt.figure(figsize=(20, 10))

# Subplot (a): Voltage vs Capacity
ax1 = fig.add_subplot(1, 2, 1)
cycle_colors = plt.cm.viridis(np.linspace(0, 1, len(eis_voltages)))
plot_voltage_capacity(ax1, voltage_data, eis_voltages, cycle_colors)

# Subplot (b): 3D Nyquist Plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_combined_3d_nyquist_with_fits(ax2, eis_data, circuit)

# Save everything
output_path = f"{save_dir}/voltage_eis_plot.jpg"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()
print(f"Figure saved at {output_path}")

summary_df = generate_summary({0: fitting_results})
summary_path = f"{save_dir}/fitting_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Summary saved at {summary_path}")
