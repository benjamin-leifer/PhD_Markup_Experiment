
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Manual input
T_C = [-40, -20, 0, 20, 40, 60]
Rb_LP30 = [8650, 2000, 295, 112, 85, 84]
Rb_HP = [2700, 680, 290, 163, 111, 83]
Rb_HM = [1030, 700, 293, 134, 112, 87]
Rb_HS = [1580, 280, 222, 143, 101, 79]

thickness_cm = 0.96
area_cm2 = 1.2679

def process(Rb, T_C, label=None):
    T_K = np.array(T_C) + 273.15
    Rb_arr = np.array(Rb)
    conductivity_S_cm = thickness_cm / (Rb_arr * area_cm2)
    conductivity_mS_cm = conductivity_S_cm * 1000
    df = pd.DataFrame({
        "T_C": T_C,
        "T_K": T_K,
        "Rb_Ohm": Rb_arr,
        "Conductivity_mS_cm": conductivity_mS_cm,
        "1000/T": 1000.0 / T_K,
        "log10_sigma_S_cm": np.log10(conductivity_S_cm),
        "label": label
    })
    return df

datasets = [
    (Rb_LP30, "LP30"),
    (Rb_HP, "DTFV1422"),
    (Rb_HM, "MF91"),
    (Rb_HS, "DTFV1411"),
]

# collect all dataframes if needed
df_list = [process(Rb, T_C, label) for Rb, label in datasets]
df_all = pd.concat(df_list, ignore_index=True)

# plot all on same axes
plt.figure(figsize=(6, 5))
for df in df_list:
    plt.plot(df["1000/T"], df["log10_sigma_S_cm"],
             marker='o', linestyle='-', linewidth=2, markersize=6, label=df["label"].iloc[0])

plt.xlabel("1000 / T (K$^{-1}$)", fontsize=14)
plt.ylabel("log$_{10}$(Conductivity [S/cm])", fontsize=14)
plt.title("Arrhenius Plot (manual Rb input)", fontsize=16)
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()