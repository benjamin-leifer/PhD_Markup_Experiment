import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from impedance.models.circuits import CustomCircuit
import pandas as pd
from tkinter import Tk, filedialog
import os

# --- User-configurable settings ---
font_size = 16      # Controls all fonts (axis labels, legend, titles)
marker_size = 120    # Controls scatter marker size

# --- GUI Utilities ---
def select_files(title="Select Files", filetypes=(("MPT files", "*.mpt"), ("All files", "*.*"))):
    root = Tk()
    root.withdraw()
    return list(filedialog.askopenfilenames(title=title, filetypes=filetypes))

def select_directory(title="Select Directory"):
    root = Tk()
    root.withdraw()
    return filedialog.askdirectory(title=title)

# --- Data Loaders ---
def extract_voltage_capacity_from_file(file_path):
    try:
        header_lines = 50
        data = pd.read_csv(file_path, skiprows=header_lines, delimiter="\t", engine="python", encoding="cp1252", on_bad_lines="skip")
        if len(data.columns) == 1:
            raise ValueError(f"File {file_path} has only one column (bad formatting).")
        data.columns = [col.strip() for col in data.columns]

        if "Ewe/V" not in data.columns or "Capacity/mA.h" not in data.columns:
            data.columns = [
                "mode", "ox/red", "error", "control changes", "Ns changes", "counter inc.",
                "Ns", "time/s", "control/mA", "Ewe/V", "I/mA", "dQ/C", "(Q-Qo)/C", "half cycle",
                "Q charge/discharge/mA.h", "Energy charge/W.h", "Energy discharge/W.h",
                "Capacitance charge/µF", "Capacitance discharge/µF", "Q discharge/mA.h",
                "Q charge/mA.h", "Capacity/mA.h", "Efficiency/%", "cycle number", "P/W"
            ]

        vc_data = data[["Ewe/V", "Capacity/mA.h"]].dropna()
        vc_data.columns = ["Voltage (V)", "Capacity (mA.h)"]
        return vc_data
    except Exception as e:
        print(f"Error reading voltage file: {e}")
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

        data = pd.read_csv(file_path, skiprows=header_lines, delimiter="\t", engine="python", encoding="cp1252", on_bad_lines="skip")
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
        print(f"Error reading EIS file: {e}")
        return None

# --- Fitting + Summary ---
def fit_eis_data(eis_data, circuit, exc_start=4):
    results = {}
    for idx, (cycle_num, cycle_data) in enumerate(eis_data.groupby("Cycle Number")):
        freqs = cycle_data["Frequency (Hz)"].values[exc_start:-1]
        Z_real = cycle_data["Re(Z) (Ohm)"].values[exc_start:-1]
        Z_imag = -cycle_data["-Im(Z) (Ohm)"].values[exc_start:-1]
        Z_exp = Z_real + 1j * Z_imag
        circuit_model = CustomCircuit(circuit, initial_guess=[10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300, 10])
        try:
            print(f"Fitting spectrum {cycle_num}...")
            circuit_model.fit(freqs, Z_exp, maxfev=1000)
            Z_fit = circuit_model.predict(freqs)
            residuals_percent = (np.abs(Z_exp) - np.abs(Z_fit)) / np.abs(Z_exp) * 100
            results[cycle_num] = {
                "frequencies": freqs,
                "Z_exp": Z_exp,
                "Z_fit": Z_fit,
                "residuals_percent": residuals_percent,
                "parameters": circuit_model.parameters_
            }
        except Exception as e:
            print(f"Fit failed for spectrum {cycle_num}: {e}")
    return results

def generate_summary(fitting_results):
    summary = []
    for cycle_num, result in fitting_results.items():
        params = result["parameters"]
        residuals = np.mean(result["residuals_percent"])
        summary.append({
            "Spectrum": cycle_num,
            "R1 (Ohm)": params[0],
            "R2 (Ohm)": params[1],
            "CPE2_0 (Ohm⁻¹sᵃ)": params[2],
            "Residuals (%)": residuals
        })
    return pd.DataFrame(summary)

# --- Plotting ---
def plot_voltage_capacity(ax, voltage_data, eis_voltages, colors):
    ax.plot(voltage_data["Capacity (mA.h)"], voltage_data["Voltage (V)"], color="black", label="Voltage vs Capacity", linewidth=2)
    for idx, voltage in enumerate(eis_voltages):
        closest = voltage_data.iloc[(voltage_data["Voltage (V)"] - voltage).abs().idxmin()]
        ax.scatter(closest["Capacity (mA.h)"], voltage, color=colors[idx], label=f"Spectrum {idx+1} EIS", s=marker_size)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=font_size-2)
    ax.set_xlabel("Capacity (mA.h)", fontsize=font_size)
    ax.set_ylabel("Voltage (V)", fontsize=font_size)
    ax.set_title("Voltage vs Capacity with EIS Markers", fontsize=font_size+2)
    ax.tick_params(labelsize=font_size)
    ax.grid(True)

def plot_3d_nyquist(ax, eis_data, circuit, colors, exc_start=4):
    for idx, (cycle_num, cycle_data) in enumerate(eis_data.groupby("Cycle Number")):
        freqs = cycle_data["Frequency (Hz)"].values[exc_start:-1]
        Z_real = cycle_data["Re(Z) (Ohm)"].values[exc_start:-1]
        Z_imag = -cycle_data["-Im(Z) (Ohm)"].values[exc_start:-1]
        Z_exp = Z_real + 1j * Z_imag
        circuit_model = CustomCircuit(circuit, initial_guess=[10, 100, 1e-5, 0.9, 200, 1e-5, 0.9, 1e-5, 0.9, 300, 10])
        circuit_model.fit(freqs, Z_exp)
        Z_fit = circuit_model.predict(freqs)
        voltage = cycle_data["Ewe (V)"].iloc[0]
        ax.scatter(voltage, Z_real, zs=-Z_imag, zdir='z', color=colors[idx], s=marker_size, label=f"Spectrum {int(cycle_num)} Exp")
        ax.plot(voltage, Z_fit.real, zs=-Z_fit.imag, zdir='z', color=colors[idx], linestyle='--', label=f"Spectrum {int(cycle_num)} Fit")
    ax.set_ylabel("Re(Z) (Ohm)", fontsize=font_size)
    ax.set_zlabel("-Im(Z) (Ohm)", fontsize=font_size)
    ax.set_xlabel("Voltage (V)", fontsize=font_size)
    ax.set_title("3D Nyquist Plot with Fits", fontsize=font_size+2)
    ax.tick_params(labelsize=font_size)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=font_size-2)
    ax.grid(True)
    ax.view_init(elev=30, azim=-60)

# --- Main Script ---
print("Select ONE Voltage File:")
voltage_file = select_files("Select Voltage File")[0]
print("Select ONE EIS File:")
eis_file = select_files("Select EIS File")[0]
print("Select Save Directory:")
save_dir = select_directory()

voltage_data = extract_voltage_capacity_from_file(voltage_file)
eis_data = read_eis_data(eis_file)

if voltage_data.empty or eis_data is None or eis_data.empty:
    raise ValueError("Failed to load one or both files.")

eis_voltages = [cycle_data["Ewe (V)"].iloc[0] for _, cycle_data in eis_data.groupby("Cycle Number")]
colors = plt.cm.viridis(np.linspace(0, 1, len(eis_voltages)))
circuit = 'R1-p(R2,CPE2)-p(R3,CPE3)-p(CPE4,R4)-W'
fitting_results = fit_eis_data(eis_data, circuit)

basename = os.path.splitext(os.path.basename(eis_file))[0]
fig = plt.figure(figsize=(18, 8))

# Subplot 1: Voltage vs Capacity
ax1 = fig.add_subplot(1, 2, 1)
plot_voltage_capacity(ax1, voltage_data, eis_voltages, colors)

# Subplot 2: 3D Nyquist
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_3d_nyquist(ax2, eis_data, circuit, colors)

# Save figure
plt.tight_layout()
output_path = os.path.join(save_dir, f"{basename}_eis_plot.jpg")
plt.savefig(output_path, dpi=300)
plt.show()
print(f"Saved plot to {output_path}")

# Save summary
summary_df = generate_summary(fitting_results)
summary_path = os.path.join(save_dir, f"{basename}_fitting_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary to {summary_path}")
