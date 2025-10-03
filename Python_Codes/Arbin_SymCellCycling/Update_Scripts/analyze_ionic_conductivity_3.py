import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ================================
# Manual Input Section
# ================================
# Enter temperature in Celsius and Rb in Ohms
# Example:
T_C = [-40, -20, 0, 20,40, 60]
Rb_Ohm = [8650, 2000, 295, 112, 85, 84]  # example numbers

# Geometry constants
thickness_cm = .96   # cm
area_cm2 =   1.2679      # cm^2

# ================================
# Processing
# ================================
T_K = np.array(T_C) + 273.15
conductivity_S_cm = thickness_cm / (np.array(Rb_Ohm) * area_cm2)  # S/cm
conductivity_mS_cm = conductivity_S_cm * 1000

df = pd.DataFrame({
    "T_C": T_C,
    "T_K": T_K,
    "Rb_Ohm": Rb_Ohm,
    "Conductivity_mS_cm": conductivity_mS_cm,
    "1000/T": 1000 / T_K,
    "log10_sigma_S_cm": np.log10(conductivity_S_cm)
})

print(df)

# ================================
# Plot
# ================================
plt.figure(figsize=(6,5))
plt.plot(df["1000/T"], df["log10_sigma_S_cm"],
         marker='o', linestyle='-', linewidth=2, markersize=8)
plt.xlabel("1000 / T (K$^{-1}$)", fontsize=14)
plt.ylabel("log$_{10}$(Conductivity [S/cm])", fontsize=14)
plt.title("Arrhenius Plot (manual Rb input)", fontsize=16)
#plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6, direction='in')
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()
