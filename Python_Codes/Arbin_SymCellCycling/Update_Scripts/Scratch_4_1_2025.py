import pandas as pd
import matplotlib.pyplot as plt

# File paths
formcharge_path = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\04\BL-LL-FA01_RT_FormCharge_2025_03_29_124048\BL-LL-FA01_RT_FormCharge_Channel_56_Wb_1.xlsx"
discharge_path = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\04\BL-LL-FA01_-51C_Discharge_2025_03_31_091710\BL-LL-FA01_-51C_Discharge_Channel_37_Wb_1.xlsx"

# Constants
active_mass_mg = 12.45 * 2.01
active_mass_g = active_mass_mg / 1000  # convert to grams

RT_color = 'black'
LT_color = 'blue'

# Load relevant data
formcharge_df = pd.read_excel(formcharge_path, sheet_name="Channel56_1", usecols=["Voltage (V)", "Discharge Capacity (Ah)", "Current (A)"])
discharge_df = pd.read_excel(discharge_path, sheet_name="Channel37_1", usecols=["Voltage (V)", "Discharge Capacity (Ah)", "Current (A)"])

# Remove zero-current and NaNs
formcharge_filtered = formcharge_df[(formcharge_df["Current (A)"] != 0)].dropna()
discharge_filtered = discharge_df[(discharge_df["Current (A)"] != 0)].dropna()

# Convert to specific capacity
formcharge_filtered["Specific Capacity (mAh/g)"] = (formcharge_filtered["Discharge Capacity (Ah)"] * 1000) / active_mass_g
discharge_filtered["Specific Capacity (mAh/g)"] = (discharge_filtered["Discharge Capacity (Ah)"] * 1000) / active_mass_g

# --- Trimming RT formation curve based on voltage features ---
# Start: max voltage where capacity < 1 mAh/g
start_index = formcharge_filtered[formcharge_filtered["Specific Capacity (mAh/g)"] < 1]["Voltage (V)"].idxmax()

# End: minimum voltage point
end_index = formcharge_filtered["Voltage (V)"].idxmin()

formcharge_final = formcharge_filtered.loc[start_index:end_index]
discharge_final = discharge_filtered.iloc[2:-2]  # remove first and last 2 points

# --- Plot 1: Voltage vs Specific Capacity ---
plt.figure(figsize=(10, 6))
plt.plot(formcharge_final["Specific Capacity (mAh/g)"], formcharge_final["Voltage (V)"], label="RT Formation", color=RT_color)
plt.plot(discharge_final["Specific Capacity (mAh/g)"], discharge_final["Voltage (V)"], label="-51°C Discharge", color=LT_color)
plt.xlabel("Specific Discharge Capacity (mAh/g)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs. Specific Discharge Capacity Li|DT14 Half Cell")
plt.ylim(bottom=2.5)
plt.legend()
#plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Normalized by RT max capacity ---
max_capacity_rt = formcharge_final["Specific Capacity (mAh/g)"].max()

formcharge_norm = formcharge_final.copy()
discharge_norm = discharge_final.copy()

formcharge_norm["Normalized Capacity"] = formcharge_norm["Specific Capacity (mAh/g)"] / max_capacity_rt
discharge_norm["Normalized Capacity"] = discharge_norm["Specific Capacity (mAh/g)"] / max_capacity_rt

plt.figure(figsize=(10, 6))
plt.plot(formcharge_norm["Normalized Capacity"], formcharge_norm["Voltage (V)"], label="RT Formation (normalized)", color=RT_color)
plt.plot(discharge_norm["Normalized Capacity"], discharge_norm["Voltage (V)"], label="-51°C Discharge (normalized)", color=LT_color)
plt.xlabel("Normalized Specific Discharge Capacity")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs. Normalized Discharge Capacity Li|DT14 Half Cell")
plt.ylim(bottom=2.5)
plt.legend()
#plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Output summary values
print(f"Max capacity RT: {max_capacity_rt:.2f} mAh/g")
closest_to_2_5V = discharge_final.iloc[(discharge_final["Voltage (V)"] - 2.5).abs().argmin()]
print(f"Discharge capacity at 2.5V (-51°C): {closest_to_2_5V['Specific Capacity (mAh/g)']:.2f} mAh/g")
