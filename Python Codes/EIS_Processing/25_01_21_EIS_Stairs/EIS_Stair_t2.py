import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from impedance.models.circuits import CustomCircuit
import pandas as pd

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
            delimiter="\t",  # Tab-separated file
            engine="python",
            encoding="cp1252"
        )

        # Clean and print column names
        data.columns = [col.strip() for col in data.columns]
        print("Parsed Columns:", data.columns)  # Debugging step

        # Handle cases where columns are not parsed correctly
        if "Ewe/V" not in data.columns or "Capacity/mA.h" not in data.columns:
            print("Error: Expected columns not found. Manually assigning column names.")

            # Manually define column names based on the last header line
            data.columns = [
                "mode", "ox/red", "error", "control changes", "Ns changes", "counter inc.",
                "Ns", "time/s", "control/mA", "Ewe/V", "I/mA", "dQ/C", "(Q-Qo)/C", "half cycle",
                "Q charge/discharge/mA.h", "Energy charge/W.h", "Energy discharge/W.h",
                "Capacitance charge/µF", "Capacitance discharge/µF", "Q discharge/mA.h",
                "Q charge/mA.h", "Capacity/mA.h", "Efficiency/%", "cycle number", "P/W"
            ]
            print("Manually assigned columns:", data.columns)

        # Check again for required columns
        if "Ewe/V" in data.columns and "Capacity/mA.h" in data.columns:
            voltage_capacity_data = data[["Ewe/V", "Capacity/mA.h"]].dropna()
            voltage_capacity_data.columns = ["Voltage (V)", "Capacity (mA.h)"]  # Standardize column names
            return voltage_capacity_data
        else:
            print("Error: Required columns 'Ewe/V' and 'Capacity/mA.h' not found even after manual assignment.")
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

def plot_voltage_capacity(ax, voltage_capacity_data):
    """
    Plots Voltage vs Capacity data.

    Parameters:
        ax (matplotlib.axes.Axes): Axis to plot on.
        voltage_capacity_data (pd.DataFrame): Voltage vs Capacity data.
    """
    ax.plot(voltage_capacity_data["Capacity (mA.h)"], voltage_capacity_data["Voltage (V)"],
            label="Voltage vs Capacity", color="black")
    ax.set_xlabel("Capacity (mA.h)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltage vs Capacity")
    ax.legend()
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

# File paths
voltage_capacity_file = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Results\Low Temp Li Ion\2025\01\EIS Stairs\CV02_Cycle_1\BL-LL-CV02_EIS_STAIR_RT_02_CP_C02.mpt"
eis_file = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Results\Low Temp Li Ion\2025\01\EIS Stairs\CV01_Cycle_1\BL-LL-CV01_EIS_STAIR_RT_05_CP_C03_01_PEIS_C04.mpt"

# Read Voltage vs Capacity Data
voltage_capacity_data = extract_voltage_capacity_from_file(voltage_capacity_file)

# Read EIS Data
eis_data = read_eis_data(eis_file)

# Subplots (b, c, d): 3D Nyquist Plots with Fitted Circuit
cycle_colors = plt.cm.viridis(np.linspace(0, 1, len(eis_data["Cycle Number"].unique())))

# Define the circuit model
#circuit = 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4-W)'
circuit = 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4)-W'
#circuit = 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4)-T'

# Create the combined figure
fig = plt.figure(figsize=(20, 10))

# Subplot (a): Voltage vs Capacity
ax1 = fig.add_subplot(2, 2, 1)  # Two rows, two columns, subplot 1
plot_voltage_capacity(ax1, voltage_capacity_data)

# Subplot (b): Combined 3D Nyquist Plot
ax2 = fig.add_subplot(2, 2, 2, projection='3d')  # Two rows, two columns, subplot 2
plot_combined_3d_nyquist_with_fits(ax2, eis_data, circuit)

# Subplot (c): Residuals Plot
ax3 = fig.add_subplot(2, 1, 2)  # Two rows, one column, subplot 3
for idx, (cycle_num, cycle_data) in enumerate(eis_data.groupby("Cycle Number")):
    frequencies = cycle_data["Frequency (Hz)"].values
    Z_real = cycle_data["Re(Z) (Ohm)"].values
    Z_imag = -cycle_data["-Im(Z) (Ohm)"].values
    Z_exp = Z_real + 1j * Z_imag

    # Fit the circuit model
    initial_guess = [10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300, 10]
    #initial_guess = [10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300, 10, 0.01, 0.1, 0.8]
    circuit_model = CustomCircuit(circuit, initial_guess=initial_guess)
    circuit_model.fit(frequencies, Z_exp)
    Z_fit = circuit_model.predict(frequencies)

    # Plot residuals for each cycle
    plot_residuals_loglog(ax3, frequencies, Z_exp, Z_fit, color=cycle_colors[idx], legend=f"Cycle {int(cycle_num)}")

plt.tight_layout()
plt.show()
plt.savefig('EIS_Stair_t2.png', dpi=300)

