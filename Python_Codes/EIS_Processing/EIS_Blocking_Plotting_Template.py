import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# EIS_Blocking_Results_7_18.py

Temp_array = np.array([25, 40, 60])
inv_Temp_array = 1000/(Temp_array+273.15)

in_GB = np.array([1.16, 1.93, 2.79])*1e-3
out_GB = np.array([1.33, 1.59, 3.03])*1e-3
out_GB_24 = np.array([0.46, 0.60, 0.97])*1e-3

fig, ax1 = plt.subplots()

ax1.set_xlabel('1000/T (K$^{-1}$)')
ax1.set_ylabel('Ionic Conductivity (S/cm)')
plt.yscale('log')
ax1.plot(inv_Temp_array, in_GB, '-o', label='In Glovebox', color='red', markersize=10)
ax1.plot(inv_Temp_array, out_GB, '-o', label='Out of Glovebox', color='blue', markersize=10)
ax1.plot(inv_Temp_array, out_GB_24, '-*', label='Out of Glovebox 24 Hours', color='blue', markersize=10)

ax1.tick_params(which='both', axis='both', direction='in', bottom=True, top=True, left=True, right=True)
#ax12.tick_params(which='both', axis='both', direction='in', bottom=False, top=False, left=False, right=True)


#plt.grid()
plt.legend()
plt.title('Ionic Conductivity vs. Temperature (Inside Glovebox vs. Outside Glovebox Trial 1')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

if __name__ == '__main__':
    pass

