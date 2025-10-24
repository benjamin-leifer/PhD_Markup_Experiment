import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ================================
# Manual Input Section
# ================================
# Enter temperature in Celsius and Rb in Ohms
# Example:
T_C = [-40, -20, 0, 20,40, 60]
Rb_LP30 = [8650, 2000, 295, 112, 85, 84]  # example numbers
Rb_HP = [2700, 680, 290, 163, 111, 83]  # example numbers
Rb_HM = [1030, 700, 293, 134, 112, 87]  # example numbers
Rb_HS = [1580, 280, 222, 143, 101, 79]  # example numbers

# Geometry constants
thickness_cm = .96   # cm
area_cm2 =   1.2679      # cm^2

# ================================
# Processing
# ================================
def process(Rb, T_C):

    T_K = np.array(T_C) + 273.15
    conductivity_S_cm = thickness_cm / (np.array(Rb) * area_cm2)  # S/cm
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
    return df
# ================================
# Plot
# ================================
#Process All:
df_all = pd.DataFrame()
for Rb_Ohm in [Rb_LP30, Rb_HP, Rb_HM, Rb_HS]:
    df_all.add(process(Rb_Ohm, T_C))
    
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
