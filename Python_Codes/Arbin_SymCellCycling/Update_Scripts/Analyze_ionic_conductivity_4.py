
# python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 14,           # default font size
    'axes.labelsize': 18,      # x/y label size
    'axes.titlesize': 18,      # title size
    'xtick.labelsize': 16,     # x tick label size
    'ytick.labelsize': 16,     # y tick label size
    'legend.fontsize': 14
})
from matplotlib.ticker import FuncFormatter

# Manual input
T_C = [-40, -20, 0, 20, 40, 60]
Rb_LP30 = [8650, 2000, 295, 112, 85, 84]
Rb_HP = [2700, 680, 290, 163, 111, 83]
Rb_HM = [1030, 700, 293, 134, 112, 87]
Rb_HS = [1580, 280, 222, 143, 101, 79]
Rb_HT = [2100, 480, 250, 145, 100, 76]

thickness_cm = 0.96
area_cm2 = 1.2679

def process(Rb, T_C, label=None):
    T_K = np.array(T_C) + 273.15
    Rb_arr = np.array(Rb, dtype=float)
    conductivity_S_cm = thickness_cm / (Rb_arr * area_cm2)
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
    (Rb_HT, "DT14"),
    (Rb_HS, "DTFV1411"),
    (Rb_HP, "DTFV1422"),
    (Rb_HM, "MF91"),


]

df_list = [process(Rb, T_C, label) for Rb, label in datasets]
df_all = pd.concat(df_list, ignore_index=True)

plt.figure(figsize=(6, 5))
ax = plt.gca()

# formatter to show plain decimal numbers (avoid scientific notation on log axis)
def plain_decimal(y, pos):
    if y == 0:
        return "0"
    s = f"{y:.10f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s

for df in df_list:
    label = df["label"].iloc[0] if "label" in df.columns else ""
    label_str = str(label).strip()
    color = 'red' if label_str.lower() == "dtfv1411" else None

    # plot actual conductivity values (mS/cm) on a log y-axis
    ax.plot(df["1000/T"], df["Conductivity_mS_cm"],
            marker='o', linestyle='-', linewidth=2, markersize=6,
            label=label_str, color=color)

ax.set_xlabel(r"1000 / T (K$^{-1}$)", fontsize=20)
ax.set_ylabel("Conductivity [mS/cm]", fontsize=20)

ax.set_yscale('log')
ax.yaxis.set_major_formatter(FuncFormatter(plain_decimal))

def bottom_to_celsius(x):
    return 1000.0 / x - 273.15

def celsius_to_bottom(x):
    return 1000.0 / (x + 273.15)

secax = ax.secondary_xaxis('top', functions=(bottom_to_celsius, celsius_to_bottom))
secax.set_xlabel("Temperature (°C)", fontsize=20)

ax.tick_params(axis='both', which='major', labelsize=16)
secax.tick_params(axis='x', which='major', labelsize=16)

ax.legend(fontsize=14)
ax.invert_xaxis()
plt.tight_layout()
plt.show()