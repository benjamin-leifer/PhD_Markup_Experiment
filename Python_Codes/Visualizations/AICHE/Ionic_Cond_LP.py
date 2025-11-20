# python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Manual input
T_C = [-40, -20, 0, 20, 40, 60]
Rb_LP30 = [8650, 2000, 295, 112, 85, 84]
Rb_HP = [2700, 680, 290, 163, 111, 83]
Rb_HM = [1030, 700, 293, 134, 112, 87]
Rb_HS = [1580, 280, 222, 143, 101, 79]

thickness_cm = 0.96
area_cm2 = 1.2679

def process(Rb, T_C, label=None):
    """
    Compute temperatures, resistances and conductivities.
    Adds both true conductivity (S/cm) and mS/cm and keeps log10 for reference.
    """
    T_K = np.array(T_C) + 273.15
    Rb_arr = np.array(Rb, dtype=float)
    conductivity_S_cm = thickness_cm / (Rb_arr * area_cm2)   # S/cm
    conductivity_mS_cm = conductivity_S_cm * 1000.0
    df = pd.DataFrame({
        "T_C": T_C,
        "T_K": T_K,
        "Rb_Ohm": Rb_arr,
        "Conductivity_S_cm": conductivity_S_cm,
        "Conductivity_mS_cm": conductivity_mS_cm,
        "1000/T": 1000.0 / T_K,
        "log10_sigma_S_cm": np.log10(conductivity_S_cm),
        "label": label
    })
    return df

datasets = [
    (Rb_LP30, "LP"),
    #(Rb_HP, "DTFV1422"),
    (Rb_HM, "MF91"),
    (Rb_HS, "DTFV1411"),
]

# collect all dataframes if needed
df_list = [process(Rb, T_C, label) for Rb, label in datasets]
df_all = pd.concat(df_list, ignore_index=True)

# plotting: conductivity (regular values) on a log y-axis, no smoothing
plt.figure(figsize=(6, 5))
ax = plt.gca()

for df in df_list:
    x = df["1000/T"].values
    y = df["Conductivity_S_cm"].values

    # plot original values with markers and connecting lines
    ax.plot(x, y,
            marker='o', linestyle='-', linewidth=2, markersize=6,
            label=df["label"].iloc[0])

# bottom x and left y labels (larger)
ax.set_xlabel(r"1000 / T (K$^{-1}$)", fontsize=20)
ax.set_ylabel("Conductivity [S/cm]", fontsize=20)

# set log scale on y
ax.set_yscale('log')

# formatter to show plain decimal numbers (not scientific notation)
def plain_decimal(y, pos):
    # avoid negative/zero log issues; y should be > 0 on log axis
    if y == 0:
        return "0"
    s = f"{y:.10f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s

ax.yaxis.set_major_formatter(FuncFormatter(plain_decimal))

# create top x-axis that shows temperature in °C
def bottom_to_celsius(x):
    # x is 1000/T (K^-1) -> T_K = 1000 / x -> T_C = T_K - 273.15
    return 1000.0 / x - 273.15

def celsius_to_bottom(x):
    # inverse: given T_C -> x = 1000 / (T_K)
    return 1000.0 / (x + 273.15)

secax = ax.secondary_xaxis('top', functions=(bottom_to_celsius, celsius_to_bottom))
secax.set_xlabel("Temperature (°C)", fontsize=20)

# larger tick labels
ax.tick_params(axis='both', which='major', labelsize=16)
secax.tick_params(axis='x', which='major', labelsize=16)

# legend and layout; no title
ax.legend(fontsize=14)
ax.invert_xaxis()
plt.tight_layout()
plt.show()