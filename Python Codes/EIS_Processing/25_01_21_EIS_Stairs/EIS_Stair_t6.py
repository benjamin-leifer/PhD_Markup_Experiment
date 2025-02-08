import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from impedance.models.circuits import CustomCircuit
import pandas as pd
from tkinter import Tk, filedialog
from matplotlib.widgets import RangeSlider, Button, TextBox


# --- Utility functions ---

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


def extract_voltage_capacity_from_file(file_path):
    try:
        header_lines = 50
        data = pd.read_csv(file_path, skiprows=header_lines, delimiter="\t",
                           engine="python", encoding="cp1252")
        data.columns = [col.strip() for col in data.columns]
        print("Parsed Columns:", data.columns)
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
        if "Ewe/V" in data.columns and "Capacity/mA.h" in data.columns:
            voltage_capacity_data = data[["Ewe/V", "Capacity/mA.h"]].dropna()
            voltage_capacity_data.columns = ["Voltage (V)", "Capacity (mA.h)"]
            return voltage_capacity_data
        else:
            print("Error: Required columns not found even after manual assignment.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()


def read_eis_data(file_path):
    try:
        with open(file_path, 'r', encoding='cp1252') as file:
            lines = file.readlines()
            header_lines = 0
            for line in lines:
                if "Nb header lines" in line:
                    header_lines = int(line.split(":")[1].strip())
                    break
        data = pd.read_csv(file_path, skiprows=header_lines, delimiter='\t',
                           engine='python', encoding='cp1252')
        data.columns = [col.strip() for col in data.columns]
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


# --- Main interactive function ---

def interactive_nyquist_adjustment(eis_data, circuit, initial_guess, bounds, exc_start=4, cycle=None, param_names=None):
    """
    Launch an interactive Nyquist plot with controls to update parameter bounds.

    For each parameter, a single RangeSlider (with two handles) sets its lower and upper bounds.
    A TextBox next to each RangeSlider shows the "initial guess" (defaulting to the mean of the bounds)
    and does not auto-update when the bounds change.

    An "Auto Fit" button performs a full fit (using the current textbox values as the initial guess and
    the current slider values as the bounds) and updates the text boxes. A "Save" button saves the current
    fitting parameters and bounds to a CSV file.

    If param_names is not provided, parameter names are pulled from the circuit model’s get_param_names()
    method and restructured to match the structure of initial_guess.

    For resistance parameters (assumed at indices 0, 1, 4, and 9):
      - The RangeSlider labels are color coded.
      - The markers (dots) in the plot are placed additively along the x-axis, meaning that each
        resistance’s markers are offset by the cumulative sum of the previous resistance values.
    Note: In this usage, the Warburg parameter is a scalar (e.g., 10) with bounds (e.g., (0,50)).
    """
    from matplotlib.widgets import RangeSlider, Button, TextBox
    from impedance.models.circuits import CustomCircuit
    from collections.abc import Iterable

    print("Launching interactive Nyquist plot with bound controls...")

    # Optionally filter by cycle.
    if cycle is not None:
        if "Cycle Number" in eis_data.columns:
            eis_data = eis_data[eis_data["Cycle Number"] == cycle]
        else:
            print("Cycle column not found; using entire dataset.")

    if len(initial_guess) != len(bounds):
        raise ValueError("The length of initial_guess and bounds must be the same.")

    # Extract frequency and impedance data.
    frequencies = eis_data["Frequency (Hz)"].values[exc_start:-1]
    Z_real = eis_data["Re(Z) (Ohm)"].values[exc_start:-1]
    Z_imag = -eis_data["-Im(Z) (Ohm)"].values[exc_start:-1]
    Z_exp = Z_real + 1j * Z_imag

    # Perform an initial fit (to get parameter names and baseline prediction).
    circuit_model = CustomCircuit(circuit, initial_guess=initial_guess)
    circuit_model.fit(frequencies, Z_exp)
    Z_fit = circuit_model.predict(frequencies)

    # If param_names not provided, pull them from the model.
    if param_names is None:
        flat_names = circuit_model.get_param_names()[0]
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

    # Create a figure with a fixed size (not full screen).
    fig, ax = plt.subplots(figsize=(10, 8))
    # (No full-screen toggle.)

    # Compute control entries: one per parameter (or sub-parameter).
    # Each entry: (i, sub, lb, ub, mean, label)
    control_entries = []
    for i, (param, bnd) in enumerate(zip(initial_guess, bounds)):
        if isinstance(param, Iterable) and not isinstance(param, (str, bytes)):
            for j, _ in enumerate(param):
                lb = bnd[j][0]
                ub = bnd[j][1]
                mean_val = (lb + ub) / 2.0
                if isinstance(param_names[i], Iterable):
                    label = param_names[i][j]
                else:
                    label = f"{param_names[i]} ({j + 1})"
                control_entries.append((i, j, lb, ub, mean_val, label))
        else:
            lb = bnd[0]
            ub = bnd[1]
            mean_val = (lb + ub) / 2.0
            label = param_names[i]
            control_entries.append((i, None, lb, ub, mean_val, label))
    num_controls = len(control_entries)
    control_height = 0.06
    bottom_margin = 0.05 + num_controls * control_height
    plt.subplots_adjust(left=0.1, bottom=bottom_margin)

    # Plot the initial Nyquist curve (remove plot labels).
    l_exp = ax.scatter(Z_exp.real, -Z_exp.imag, color='black')
    l_fit, = ax.plot(Z_fit.real, -Z_fit.imag, color='red')
    ax.set_xlabel("Re(Z) (Ohm)")
    ax.set_ylabel("-Im(Z) (Ohm)")
    ax.set_title("Nyquist Plot with Bound Controls")
    # Do not call ax.legend() to remove plot labels.

    # Create controls: one RangeSlider and one TextBox per control entry.
    controls = {}  # key: (i, sub) -> {"range_slider": ..., "textbox": ...}
    # Adjust horizontal positions to avoid overlap:
    for k, (i, sub, lb, ub, mean_val, label) in enumerate(control_entries):
        y = 0.05 + (num_controls - 1 - k) * control_height
        ax_range = plt.axes([0.05, y, 0.35, control_height * 0.8])  # Wider slider
        # Create a RangeSlider with two handles.
        range_slider = RangeSlider(ax_range, label, valmin=lb, valmax=ub, valinit=(lb, ub))
        ax_text = plt.axes([0.42, y, 0.2, control_height * 0.8])  # Shifted TextBox
        textbox = TextBox(ax_text, "Init", initial=str(mean_val))
        controls[(i, sub)] = {"range_slider": range_slider, "textbox": textbox}
        # For resistance parameters (i in [0,1,4,9] and scalar), color-code the slider label.
        res_colors = {0: 'red', 1: 'green', 4: 'blue', 9: 'orange'}
        if sub is None and i in res_colors:
            range_slider.label.set_color(res_colors[i])

        # When the RangeSlider changes, update the prediction and bound markers.
        def make_range_callback(key):
            def range_update(val):
                update_bounds_markers()
                update_prediction()

            return range_update

        controls[(i, sub)]["range_slider"].on_changed(make_range_callback((i, sub)))

        # When the TextBox is submitted, update the prediction.
        def make_text_callback(key):
            def text_update(text):
                update_prediction()

            return text_update

        controls[(i, sub)]["textbox"].on_submit(make_text_callback((i, sub)))

    # Function to read the current initial guess from the textboxes.
    def get_current_initial_guess():
        new_guess = {}
        for (i, sub), ctrl in controls.items():
            try:
                val = float(ctrl["textbox"].text)
            except:
                rng = ctrl["range_slider"].val
                val = (rng[0] + rng[1]) / 2.0
            if i in new_guess:
                new_guess[i].append((sub, val))
            else:
                if sub is None:
                    new_guess[i] = val
                else:
                    new_guess[i] = [(sub, val)]
        final_guess = []
        for i in range(len(initial_guess)):
            if i in new_guess:
                if isinstance(new_guess[i], list):
                    sorted_vals = [x[1] for x in sorted(new_guess[i], key=lambda x: x[0])]
                    final_guess.append(tuple(sorted_vals))
                else:
                    final_guess.append(new_guess[i])
            else:
                final_guess.append(initial_guess[i])
        return final_guess

    # Function to read the current bounds from the RangeSliders.
    def get_current_bounds():
        new_bounds = {}
        for (i, sub), ctrl in controls.items():
            rng = ctrl["range_slider"].val
            if i in new_bounds:
                new_bounds[i].append((rng[0], rng[1]))
            else:
                if sub is None:
                    new_bounds[i] = (rng[0], rng[1])
                else:
                    new_bounds[i] = [(rng[0], rng[1])]
        final_bounds = []
        for i in range(len(bounds)):
            if i in new_bounds:
                if isinstance(new_bounds[i], list):
                    final_bounds.append(tuple(new_bounds[i]))
                else:
                    final_bounds.append(new_bounds[i])
            else:
                final_bounds.append(bounds[i])
        return final_bounds

    # Update prediction using current textbox values.
    def update_prediction():
        current_guess = get_current_initial_guess()
        model = CustomCircuit(circuit, initial_guess=current_guess)
        Z_fit_new = model.predict(frequencies)
        l_fit.set_xdata(Z_fit_new.real)
        l_fit.set_ydata(-Z_fit_new.imag)
        fig.canvas.draw_idle()

    # Update bound markers for resistance parameters with additive placement.
    bound_markers = []

    def update_bounds_markers():
        nonlocal bound_markers
        # Remove previous markers.
        for marker in bound_markers:
            marker.remove()
        bound_markers.clear()
        y_min, y_max = ax.get_ylim()
        y_coord = y_min + 0.05 * (y_max - y_min)
        # Resistance parameters: assume indices 0, 1, 4, 9.
        res_indices = [0, 1, 4, 9]
        res_controls = []
        for (i, sub), ctrl in controls.items():
            if sub is None and i in res_indices:
                res_controls.append((i, ctrl))
        # Sort in order of parameter index.
        res_controls.sort(key=lambda x: x[0])
        # Define colors for each resistance.
        res_colors = {0: 'red', 1: 'green', 4: 'blue', 9: 'orange'}
        cumulative = 0.0
        for (i, ctrl) in res_controls:
            rng = ctrl["range_slider"].val  # (lb, ub)
            try:
                current_val = float(ctrl["textbox"].text)
            except:
                current_val = (rng[0] + rng[1]) / 2.0
            lower_marker = cumulative + rng[0]
            upper_marker = cumulative + rng[1]
            c = res_colors.get(i, 'black')
            m1, = ax.plot(lower_marker, y_coord, marker='o', color=c, markersize=10)
            m2, = ax.plot(upper_marker, y_coord, marker='o', color=c, markersize=10)
            bound_markers.extend([m1, m2])
            cumulative += current_val
        fig.canvas.draw_idle()

    update_prediction()
    update_bounds_markers()

    # "Auto Fit" button.
    ax_auto = plt.axes([0.8, 0.01, 0.15, 0.04])
    auto_button = Button(ax_auto, "Auto Fit")

    def auto_fit(event):
        current_guess = get_current_initial_guess()
        current_bounds = get_current_bounds()
        auto_model = CustomCircuit(circuit, initial_guess=current_guess, bounds=current_bounds)
        try:
            auto_model.fit(frequencies, Z_exp)
        except Exception as e:
            print("Automated fit failed:", e)
            return
        fitted_params = auto_model.parameters_
        print("Automated fit parameters:", fitted_params)
        for (i, sub), ctrl in controls.items():
            if sub is None:
                ctrl["textbox"].set_val(str(fitted_params[i]))
            else:
                ctrl["textbox"].set_val(str(fitted_params[i][sub]))
        update_prediction()
        update_bounds_markers()

    auto_button.on_clicked(auto_fit)

    # "Save" button.
    ax_save = plt.axes([0.6, 0.01, 0.15, 0.04])
    save_button = Button(ax_save, "Save")

    def save_fit(event):
        current_guess = get_current_initial_guess()
        current_bounds = get_current_bounds()
        data_rows = []
        for i in range(len(initial_guess)):
            if isinstance(initial_guess[i], Iterable) and not isinstance(initial_guess[i], (str, bytes)):
                for j, val in enumerate(current_guess[i]):
                    name = param_names[i][j] if (isinstance(param_names[i], Iterable) and not isinstance(param_names[i],
                                                                                                         (str,
                                                                                                          bytes))) else f"{param_names[i]} ({j + 1})"
                    lb_val, ub_val = current_bounds[i][j]
                    data_rows.append(
                        {"Parameter": name, "Initial Guess": val, "Lower Bound": lb_val, "Upper Bound": ub_val})
            else:
                name = param_names[i]
                lb_val, ub_val = current_bounds[i]
                data_rows.append({"Parameter": name, "Initial Guess": current_guess[i], "Lower Bound": lb_val,
                                  "Upper Bound": ub_val})
        df_save = pd.DataFrame(data_rows)
        root = Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            df_save.to_csv(file_path, index=False)
            print(f"Saved fitting parameters and bounds to {file_path}")

    save_button.on_clicked(save_fit)

    plt.show()


# --------------------------
# ===== Usage Example ======
#
# Example usage:
#
# import pandas as pd
#
# eis_data = read_eis_data("path_to_your_eis_file.mpt")
# circuit_str = 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4-W)'
#
# # initial_guess: 10 scalars and then a scalar for the Warburg parameter.
# initial_guess = [10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300, 10]
#
# # bounds: for the Warburg parameter, supply a tuple, e.g., (0,50).
# bounds = [
#     (0, 500),     # R1
#     (0, 500),     # R2
#     (1e-7, 1e-3), # CPE2
#     (0, 1),       # n2
#     (0, 500),     # R3
#     (1e-7, 1e-3), # CPE3
#     (0, 1),       # n3
#     (1e-7, 1e-3), # CPE4
#     (0, 1),       # n4
#     (0, 500),     # R4
#     (0, 50)       # Warburg
# ]
#
# interactive_nyquist_adjustment(eis_data, circuit_str, initial_guess, bounds, cycle=1)
# --------------------------

if __name__ == "__main__":
    import pandas as pd

    eis_data = read_eis_data(r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Biologic\Leifer\Results\Low Temp Li Ion\2025\01\EIS Stairs\CV01\BL-LL-CV01_EIS_STAIR_RT_05_CP_C03_01_PEIS_C04.mpt")
    circuit_str = 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4-W)'

    # initial_guess: 10 scalars and then a tuple for the Warburg element.
    initial_guess = [10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300, 10]

    # bounds: for the Warburg element, supply a tuple of bounds for its two parameters.
    bounds = [
        (0, 30),     # R1
        (0, 30),     # R2
        (1e-7, 1e-3), # CPE2
        (0, 1),       # n2
        (0, 50),     # R3
        (1e-7, 1e-3), # CPE3
        (0, 1),       # n3
        (1e-7, 1e-3), # CPE4
        (0, 1),       # n4
        (0, 100),     # R4
        (0, 1000),  # Warburg element bounds (two parameters)
    ]

    # No need to supply param_names explicitly; they are pulled from the circuit model.

    interactive_nyquist_adjustment(eis_data, circuit_str, initial_guess, bounds, cycle=1)
    #--------------------------
