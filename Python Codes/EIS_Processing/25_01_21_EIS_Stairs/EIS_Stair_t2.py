import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from impedance.models.circuits import CustomCircuit
import pandas as pd
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


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

def plot_residuals_loglog(ax, frequencies, Z_exp, Z_fit, color='blue', label_text=''):
    """
    Plots percentage residuals on a log-log scale between experimental and fitted impedance data.

    Parameters:
        ax (matplotlib.axes.Axes): Axis to plot on.
        frequencies (np.array): Frequency data.
        Z_exp (np.array): Experimental impedance data.
        Z_fit (np.array): Fitted impedance data.
        color (str): Line color for the residuals plot.
        label_text (str): Label for the legend.
    """
    # Calculate magnitudes of impedance
    magnitude_exp = np.abs(Z_exp)
    magnitude_fit = np.abs(Z_fit)
    # Calculate percentage residuals
    residuals_percent = np.abs(magnitude_exp - magnitude_fit) / magnitude_exp * 100

    # Plot residuals on a log-log scale
    ax.loglog(frequencies, residuals_percent, color=color, label=label_text)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Residuals (%)")
    ax.set_title("Log-Log Plot of Residuals vs Frequency")
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
            circuit_model.fit(frequencies, Z_exp)
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

def interactive_fit(cycle_data, circuit, initial_guess):
    """
    Performs a fit for a single cycle and prints the resulting parameters.
    The user can then decide to adjust the initial guess if desired.
    """
    frequencies = cycle_data["Frequency (Hz)"].values[4:-1]
    Z_real = cycle_data["Re(Z) (Ohm)"].values[4:-1]
    Z_imag = -cycle_data["-Im(Z) (Ohm)"].values[4:-1]
    Z_exp = Z_real + 1j * Z_imag

    circuit_model = CustomCircuit(circuit, initial_guess=initial_guess)
    try:
        circuit_model.fit(frequencies, Z_exp)
        Z_fit = circuit_model.predict(frequencies)
        print("Fitted Parameters:", circuit_model.parameters_)
        return frequencies, Z_exp, Z_fit
    except Exception as e:
        print("Fit failed:", e)
        return frequencies, Z_exp, None

def update(val):
    # Update the initial guess from slider values (you would do this for all sliders)
    new_initial_guess = [
        slider_p1.val,
        initial_guess[1], initial_guess[2], initial_guess[3],
        initial_guess[4], initial_guess[5], initial_guess[6],
        initial_guess[7], initial_guess[8], initial_guess[9],
        initial_guess[10]
    ]
    circuit_model = CustomCircuit(circuit, initial_guess=new_initial_guess)
    circuit_model.fit(frequencies, Z_exp)
    Z_fit = circuit_model.predict(frequencies)
    l_fit.set_ydata(np.abs(Z_fit))
    fig.canvas.draw_idle()


def interactive_nyquist_adjustment(eis_data, circuit, initial_guess, bounds, exc_start=4, cycle=None, param_names=None):
    """
    Launch an interactive Nyquist plot with sliders for live adjustment of circuit parameters.
    Optionally, filter the data to a single cycle. In addition, for parameters representing
    resistances (assumed here to be at indices 0, 1, 4, and 9), red dots are drawn on the x-axis
    (Re(Z)) to indicate their lower and upper bounds. A button labeled "Auto Fit" is provided to
    perform an automated fit, update slider values, and update the bound markers.

    The slider labels are pulled directly from the circuit model if `param_names` is not provided.

    Parameters:
        eis_data (pd.DataFrame): EIS data containing at least the following columns:
            - "Frequency (Hz)"
            - "Re(Z) (Ohm)"
            - "-Im(Z) (Ohm)"
        circuit (str): The equivalent circuit string
            (e.g., 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4-W)').
        initial_guess (list): A list of initial parameter guesses. For a multi-parameter element
            (e.g., the Warburg element), supply a tuple or list of numbers.
        bounds (list): A list of bounds corresponding to each parameter. For a multi-parameter element,
            supply a tuple (or list) of (min, max) pairs.
        exc_start (int): Index to start using data (default: 4).
        cycle (optional): If provided (e.g. an integer), filter the EIS data to include only rows
            where the "Cycle Number" column equals this value.
        param_names (optional): A list of parameter names (or iterables of names for multi-parameter
            elements) to label the sliders. If not provided, the function pulls these directly from the
            circuit model.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    from impedance.models.circuits import CustomCircuit
    from collections.abc import Iterable

    print("Launching interactive Nyquist plot with live parameter adjustment...")

    # --- Optionally filter the data by cycle ---
    if cycle is not None:
        if "Cycle Number" in eis_data.columns:
            eis_data = eis_data[eis_data["Cycle Number"] == cycle]
        else:
            print("Cycle column not found in the EIS data; using entire dataset.")

    # --- Check that the top-level lengths of initial_guess and bounds match ---
    if len(initial_guess) != len(bounds):
        raise ValueError("The length of initial_guess and bounds must be the same. "
                         f"Got {len(initial_guess)} and {len(bounds)} respectively.")

    # --- Extract frequency and impedance data from the EIS dataset ---
    frequencies = eis_data["Frequency (Hz)"].values[exc_start:-1]
    Z_real = eis_data["Re(Z) (Ohm)"].values[exc_start:-1]
    # Adjust the sign for the imaginary part as needed:
    Z_imag = -eis_data["-Im(Z) (Ohm)"].values[exc_start:-1]
    Z_exp = Z_real + 1j * Z_imag

    # --- Set up the Matplotlib figure and adjust bottom margin for sliders and button ---
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.1, bottom=0.55)  # increased bottom margin for sliders and button

    # --- Perform the initial circuit fit ---
    circuit_model = CustomCircuit(circuit, initial_guess=initial_guess)
    circuit_model.fit(frequencies, Z_exp)
    Z_fit = circuit_model.predict(frequencies)

    # --- Pull parameter names from the circuit model if not provided ---
    if param_names is None:
        # Assume the circuit model has a flat list attribute 'parameter_names_'
        flat_names = circuit_model.get_param_names()[0]
        print(flat_names)
        structured_names = []
        idx = 0
        for item in initial_guess:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                n = len(item)
                structured_names.append(tuple(flat_names[idx:idx + n]))
                idx += n
            else:
                structured_names.append(flat_names[idx])
                idx += 1
        param_names = structured_names

    # --- Plot the experimental data and the fitted Nyquist curve ---
    # Nyquist plot: x-axis is Re(Z), y-axis is -Im(Z)
    l_exp = ax.scatter(Z_exp.real, -Z_exp.imag, label="Experimental", color='black')
    l_fit, = ax.plot(Z_fit.real, -Z_fit.imag, label="Fitted", color='red')
    ax.set_xlabel("Re(Z) (Ohm)")
    ax.set_ylabel("-Im(Z) (Ohm)")
    ax.set_title("Nyquist Plot with Live Parameter Adjustment")
    ax.legend()

    # --- Create sliders ---
    # Build a list "slider_info" that holds tuples of (param_index, sub_index, slider).
    slider_info = []
    slider_height = 0.03
    slider_spacing = 0.005
    bottom_start_sliders = 0.05  # vertical position (in figure fraction) for the first slider
    current_slider_index = 0

    for i, (param, bnd) in enumerate(zip(initial_guess, bounds)):
        # Use the derived parameter names.
        label_base = param_names[i]
        if isinstance(param, Iterable) and not isinstance(param, (str, bytes)):
            # Multi-parameter element (e.g., Warburg)
            for j, (sub_val, sub_bound) in enumerate(zip(param, bnd)):
                ax_slider = plt.axes(
                    [0.1, bottom_start_sliders + current_slider_index * (slider_height + slider_spacing),
                     0.8, slider_height], facecolor='lightgoldenrodyellow')
                # If label_base is iterable, use its j-th element; otherwise append sub-index.
                if isinstance(label_base, Iterable) and not isinstance(label_base, (str, bytes)):
                    label = f"{label_base[j]}"
                else:
                    label = f"{label_base} ({j + 1})"
                s = Slider(ax_slider, label, float(sub_bound[0]), float(sub_bound[1]),
                           valinit=float(sub_val), valfmt="%.3g")
                slider_info.append((i, j, s))
                current_slider_index += 1
        else:
            # Scalar parameter.
            ax_slider = plt.axes([0.1, bottom_start_sliders + current_slider_index * (slider_height + slider_spacing),
                                  0.8, slider_height], facecolor='lightgoldenrodyellow')
            s = Slider(ax_slider, f"{label_base}", float(bnd[0]), float(bnd[1]),
                       valinit=float(param), valfmt="%.3g")
            slider_info.append((i, None, s))
            current_slider_index += 1

    # --- Function to update the red-bound markers for resistance parameters ---
    # (Assuming resistances are at indices 0, 1, 4, and 9.)
    bound_markers = []  # list to hold the marker artists

    def update_bounds_markers():
        nonlocal bound_markers
        # Remove previous markers if they exist.
        for marker in bound_markers:
            marker.remove()
        bound_markers.clear()
        # Choose a y coordinate (for example, near the bottom of the current y-axis range).
        y_min, y_max = ax.get_ylim()
        y_coord = y_min + 0.05 * (y_max - y_min)
        resistance_indices = [0, 1, 4, 9]
        for i in resistance_indices:
            # Get the bound values for parameter i.
            lb, ub = bounds[i]
            # Plot red dots at these x positions.
            m1, = ax.plot(lb, y_coord, 'ro', markersize=8)
            m2, = ax.plot(ub, y_coord, 'ro', markersize=8)
            bound_markers.extend([m1, m2])
        fig.canvas.draw_idle()

    # Draw the initial bound markers.
    update_bounds_markers()

    # --- Define the slider update function ---
    def update(val):
        # Build a new initial_guess that mirrors the structure of the original.
        new_initial_guess = [None] * len(initial_guess)
        temp_dict = {}  # key: parameter index, value: list of (sub_index, value) or a single value
        for (i, sub_idx, s) in slider_info:
            if sub_idx is None:
                temp_dict[i] = s.val
            else:
                if i not in temp_dict:
                    temp_dict[i] = []
                temp_dict[i].append((sub_idx, s.val))
        for i in range(len(initial_guess)):
            if isinstance(initial_guess[i], Iterable) and not isinstance(initial_guess[i], (str, bytes)):
                sub_vals = sorted(temp_dict[i], key=lambda x: x[0])
                new_initial_guess[i] = tuple(val for (_, val) in sub_vals)
            else:
                new_initial_guess[i] = temp_dict[i]
        # Re-fit the circuit model with the updated parameters.
        model = CustomCircuit(circuit, initial_guess=new_initial_guess)
        try:
            print('Re-fitting with:', new_initial_guess)
            #model.fit(frequencies, Z_exp)
        except Exception as e:
            print("Fit failed:", e)
            return
        Z_fit_new = model.predict(frequencies)
        l_fit.set_xdata(Z_fit_new.real)
        l_fit.set_ydata(-Z_fit_new.imag)
        fig.canvas.draw_idle()

    # Connect each slider to the update function.
    for (_, _, s) in slider_info:
        s.on_changed(update)

    # --- Add an "Auto Fit" button that re-computes the fit and updates slider values ---
    ax_button = plt.axes([0.8, 0.01, 0.15, 0.04])  # adjust the position as needed
    auto_button = Button(ax_button, "Auto Fit")

    def auto_fit(event):
        # Build the current initial_guess from the slider values.
        new_initial_guess = [None] * len(initial_guess)
        temp_dict = {}
        for (i, sub_idx, s) in slider_info:
            if sub_idx is None:
                temp_dict[i] = s.val
            else:
                if i not in temp_dict:
                    temp_dict[i] = []
                temp_dict[i].append((sub_idx, s.val))
        for i in range(len(initial_guess)):
            if isinstance(initial_guess[i], Iterable) and not isinstance(initial_guess[i], (str, bytes)):
                sub_vals = sorted(temp_dict[i], key=lambda x: x[0])
                new_initial_guess[i] = tuple(val for (_, val) in sub_vals)
            else:
                new_initial_guess[i] = temp_dict[i]
        # Run an automated fit using the current slider values as starting points.
        auto_model = CustomCircuit(circuit, initial_guess=new_initial_guess)
        try:
            auto_model.fit(frequencies, Z_exp)
        except Exception as e:
            print("Automated fit failed:", e)
            return
        fitted_params = auto_model.parameters_
        print("Automated fit parameters:", fitted_params)
        # Update each slider with the fitted parameter values.
        for (i, sub_idx, s) in slider_info:
            if sub_idx is None:
                s.set_val(fitted_params[i])
            else:
                s.set_val(fitted_params[i][sub_idx])
        # Update the fitted curve on the Nyquist plot.
        Z_fit_new = auto_model.predict(frequencies)
        l_fit.set_xdata(Z_fit_new.real)
        l_fit.set_ydata(-Z_fit_new.imag)
        fig.canvas.draw_idle()
        # Also update the bound markers for the resistance parameters.
        update_bounds_markers()

    auto_button.on_clicked(auto_fit)

    plt.show()


"""
# Select voltage data files
voltage_files = select_files(title="Select Voltage Data Files")

# Select EIS data files
eis_files = select_files(title="Select EIS Data Files")

if len(voltage_files) != len(eis_files):
    raise ValueError("The number of voltage files must match the number of EIS files!")
"""


# File paths
voltage_capacity_file = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Results\Low Temp Li Ion\2025\01\EIS Stairs\CV01\BL-LL-CV01_EIS_STAIR_RT_05_CP_C03_02_CP_C04.mpt"
eis_file = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Results\Low Temp Li Ion\2025\01\EIS Stairs\CV01\BL-LL-CV01_EIS_STAIR_RT_05_CP_C03_01_PEIS_C04.mpt"

# Read Voltage vs Capacity Data
voltage_capacity_data = extract_voltage_capacity_from_file(voltage_capacity_file)

# Read EIS Data
eis_data = read_eis_data(eis_file)

# Subplots (b, c, d): 3D Nyquist Plots with Fitted Circuit
cycle_colors = plt.cm.viridis(np.linspace(0, 1, len(eis_data["Cycle Number"].unique())))
#cycle_colors = plt.cm.viridis(np.linspace(0, 1, 12))
# Define the circuit model
circuit = 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4-W)'
#circuit = 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4)-W'
#circuit = 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4)-T'
#fitting_results = fit_eis_data(eis_data, circuit)




# ======= USAGE EXAMPLE =======
# The following code demonstrates how to use the interactive_nyquist_adjustment() function.
# Make sure to have your EIS dataset (as a pandas DataFrame) with the required columns.
#
# Example:
# import pandas as pd
# eis_cycle_data = pd.read_csv("your_eis_data_file.csv", delimiter='\t')
# circuit_str = 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4-W)'
# initial_guess = [10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300]
# bounds = [
#     (0, 500),     # Bounds for R1
#     (0, 500),     # Bounds for R2
#     (1e-7, 1e-3), # Bounds for CPE2
#     (0, 1),       # Bounds for n2
#     (0, 500),     # Bounds for R3
#     (1e-7, 1e-3), # Bounds for CPE3
#     (0, 1),       # Bounds for n3
#     (1e-7, 1e-3), # Bounds for CPE4
#     (0, 1),       # Bounds for n4
#     (0, 500)      # Bounds for R4
# ]
#
# interactive_nyquist_adjustment(eis_cycle_data, circuit_str, initial_guess, bounds)



fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(left=0.25, bottom=0.35)

# Initial data (assuming you have your cycle_data and frequencies, etc.)
frequencies = eis_data["Frequency (Hz)"].values[4:-1]
Z_real = eis_data["Re(Z) (Ohm)"].values[4:-1]
Z_imag = -eis_data["-Im(Z) (Ohm)"].values[4:-1]
Z_exp = Z_real + 1j * Z_imag
initial_guess = [10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300, 10]
bounds = [
    (0, 500),     # R1
    (0, 500),     # R2
    (1e-7, 1e-3), # CPE2
    (0, 1),       # n2
    (0, 500),     # R3
    (1e-7, 1e-3), # CPE3
    (0, 1),       # n3
    (1e-7, 1e-3), # CPE4
    (0, 1),       # n4
    (0, 500),     # R4
    (0, 50),    # Warburg parameter 1 bound
]

interactive_nyquist_adjustment(eis_data, circuit, initial_guess, bounds, cycle=1)

# Do an initial fit
circuit_model = CustomCircuit(circuit, initial_guess=initial_guess)
circuit_model.fit(frequencies, Z_exp)
Z_fit = circuit_model.predict(frequencies)

# Plot experimental data and initial fit
l_exp = ax.scatter(frequencies, np.abs(Z_exp), label="Experimental", color='black')
l_fit, = ax.plot(frequencies, np.abs(Z_fit), label="Fitted", color='red')
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|Z| (Ohm)")
ax.legend()

axcolor = 'lightgoldenrodyellow'
ax_p1 = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
slider_p1 = Slider(ax=ax_p1, label='R1', valmin=0, valmax=100, valinit=initial_guess[0])
slider_p1.on_changed(update)
plt.show()

# Create the combined figure
fig = plt.figure(figsize=(20, 10))

# Subplot (a): Voltage vs Capacity
ax1 = fig.add_subplot(2, 2, 1)  # Two rows, two columns, subplot 1
# Extract EIS voltages
eis_voltages = [eis_data[eis_data["Cycle Number"] == cycle]["Ewe (V)"].iloc[0]
                for cycle in eis_data["Cycle Number"].unique()]

# Plot Voltage vs Capacity with markers for EIS spectra
plot_voltage_capacity(ax1, voltage_capacity_data, eis_voltages, cycle_colors)

# Subplot (b): Combined 3D Nyquist Plot
ax2 = fig.add_subplot(2, 2, 2, projection='3d')  # Two rows, two columns, subplot 2
plot_combined_3d_nyquist_with_fits(ax2, eis_data, circuit)

# Subplot (c): Residuals Plot using stored fitting results
ax3 = fig.add_subplot(2, 1, 2)  # Two rows, one column, subplot 3

for idx, (cycle_num, fit_result) in enumerate(fitting_results.items()):
    frequencies = fit_result["frequencies"]
    Z_exp = fit_result["Z_exp"]
    Z_fit = fit_result["Z_fit"]
    # Plot residuals using the updated plot_residuals_loglog function
    plot_residuals_loglog(ax3, frequencies, Z_exp, Z_fit, color=cycle_colors[idx],
                          label_text=f"Cycle {int(cycle_num)}")

ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

plt.tight_layout()
#plt.savefig('EIS_Stair_t2.png', dpi=300)
plt.show()


