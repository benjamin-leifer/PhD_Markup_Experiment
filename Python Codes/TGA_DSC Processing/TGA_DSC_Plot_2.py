import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TGA_DSC_Names = ['Time (min)', 'Temperature (C)', 'Weight (mg)', 'Heat Flow (mW)', 'Temperature Difference (°C)', 'Heat Flow Difference (µV)','Sample Purge Flow (mL/min)']
sample_name = 'Northeastern Zn-TFSI GPE Sample'
df = pd.read_table('Zn-TFSI 07-08-23.txt', sep='\t', header=45, names=TGA_DSC_Names, encoding="utf-16")
df['Weight Percent (%)'] = df['Weight (mg)']/df['Weight (mg)'].max()*100
TGA_DSC_Names.append('Weight Percent (%)')
df['Differential Heat Flow (mW/C)'] = df['Heat Flow (mW)'].diff()
TGA_DSC_Names.append('Differential Heat Flow (mW/C)')


TGA_DSC_Names = ['Time (min)', 'Temperature (C)', 'Weight (mg)', 'Heat Flow (mW)', 'Temperature Difference (°C)', 'Heat Flow Difference (µV)','Sample Purge Flow (mL/min)']
sample_name1 = 'Sample A in Ar'
df1 = pd.read_table('Sample_A_Ar.txt', sep='\t', header=45, names=TGA_DSC_Names, encoding="utf-16")
df1['Weight Percent (%)'] = df1['Weight (mg)']/df1['Weight (mg)'].max()*100
TGA_DSC_Names.append('Weight Percent (%)')
df1['Differential Heat Flow (mW/C)'] = df1['Heat Flow (mW)'].diff()
TGA_DSC_Names.append('Differential Heat Flow (mW/C)')


TGA_DSC_Names = ['Time (min)', 'Temperature (C)', 'Weight (mg)', 'Heat Flow (mW)', 'Temperature Difference (°C)', 'Heat Flow Difference (µV)','Sample Purge Flow (mL/min)']
sample_name2 = 'Sample B in Ar'
df2 = pd.read_table('Sample_B_Ar.txt', sep='\t', header=45, names=TGA_DSC_Names, encoding="utf-16")
df2['Weight Percent (%)'] = df2['Weight (mg)']/df2['Weight (mg)'].max()*100
TGA_DSC_Names.append('Weight Percent (%)')
df2['Differential Heat Flow (mW/C)'] = df2['Heat Flow (mW)'].diff()
TGA_DSC_Names.append('Differential Heat Flow (mW/C)')



fig, ax1 = plt.subplots()

color = 'tab:purple'
ax1.set_xlabel(TGA_DSC_Names[1])
ax1.set_ylabel(TGA_DSC_Names[7], color=color)
ax1_line1, = ax1.plot(df[TGA_DSC_Names[1]], df[TGA_DSC_Names[7]], label = sample_name)
ax1_line2, = ax1.plot(df1[TGA_DSC_Names[1]], df1[TGA_DSC_Names[7]], label = sample_name1)
ax1_line3, = ax1.plot(df2[TGA_DSC_Names[1]], df2[TGA_DSC_Names[7]], label = sample_name2)
#label=[sample_name, sample_name1, sample_name2])
ax1.tick_params(axis='y', labelcolor=color)



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(TGA_DSC_Names[3], color=color)  # we already handled the x-label with ax1
#ax2.plot(df[TGA_DSC_Names[1]], df[TGA_DSC_Names[3]], df1[TGA_DSC_Names[1]], df1[TGA_DSC_Names[3]], df2[TGA_DSC_Names[1]], df2[TGA_DSC_Names[3]])
ax2_line1, = ax2.plot(df[TGA_DSC_Names[1]], df[TGA_DSC_Names[3]], label = sample_name, linestyle='dashed')
ax2_line2, = ax2.plot(df1[TGA_DSC_Names[1]], df1[TGA_DSC_Names[3]], label = sample_name1, linestyle='dashed')
ax2_line3, = ax2.plot(df2[TGA_DSC_Names[1]], df2[TGA_DSC_Names[3]], label = sample_name2, linestyle='dashed')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Weight Percent and Heat Flow vs. Temperature for GPEs')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend(handles = [ax1_line1, ax1_line2, ax1_line3], loc='center right')
plt.show()

if __name__ == '__main__':
    pass