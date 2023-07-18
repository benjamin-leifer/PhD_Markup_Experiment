import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TGA_DSC_Names = ['Time (min)', 'Temperature (C)', 'Weight (mg)', 'Heat Flow (mW)', 'Temperature Difference (°C)', 'Heat Flow Difference (µV)','Sample Purge Flow (mL/min)']

df = pd.read_table('Zn-TFSI 07-08-23.txt', sep='\t', header=45, names=TGA_DSC_Names, encoding="utf-16")
df['Weight Percent (%)'] = df['Weight (mg)']/df['Weight (mg)'].max()*100
TGA_DSC_Names.append('Weight Percent (%)')
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel(TGA_DSC_Names[1])
ax1.set_ylabel(TGA_DSC_Names[7], color=color)
ax1.plot(df[TGA_DSC_Names[1]], df[TGA_DSC_Names[7]], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(TGA_DSC_Names[3], color=color)  # we already handled the x-label with ax1
ax2.plot(df[TGA_DSC_Names[1]], df[TGA_DSC_Names[3]], color=color)
ax2.tick_params(axis='y', labelcolor=color)
sample_name = '0.5M Zn-TFSI GPE (07-08-23)'
plt.title('Weight Percent and Heat Flow vs. Temperature for %s' % sample_name)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

if __name__ == '__main__':
    pass