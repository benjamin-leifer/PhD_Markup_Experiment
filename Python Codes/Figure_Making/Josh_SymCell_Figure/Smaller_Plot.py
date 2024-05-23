#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:35:04 2021

@author: joshua
"""

import numpy as np
import matplotlib.pyplot as plt
import os


#os.chdir('/Users/joshua/Documents/Python/Python examples/Making a plot - Gallaway Group')
out_1 = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Manuscript\Zn-Ion Fiber Work\Resources\ECS Talk\Zn-Sym Data\Sym Cycling\t2\out_1.csv'
out_2 = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Manuscript\Zn-Ion Fiber Work\Resources\ECS Talk\Zn-Sym Data\Sym Cycling\t2\out_2.csv'
out_3 = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Manuscript\Zn-Ion Fiber Work\Resources\ECS Talk\Zn-Sym Data\Sym Cycling\t2\out_3.csv'
out_4 = r'C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Manuscript\Zn-Ion Fiber Work\Resources\ECS Talk\Zn-Sym Data\Sym Cycling\t2\out_4.csv'


# Get the data ================================

cycle_file = out_1

cycle_data = np.genfromtxt(cycle_file, delimiter=",")
#cycle_data = np.transpose(cycle_data)

# Assuming 'data' is your numpy array
#data = np.array([...])  # replace with your actual data

# Get the indices where the value at row index 25 is either 10 or 11

indices = np.where((cycle_data[26, :] == 10.0) | (cycle_data[26, :] == 11.0))

# Use these indices to get the filtered data
filtered_data = cycle_data[:, indices]

cycle_data_2 = filtered_data
cycle_data_2 = np.transpose(filtered_data)

tt = cycle_data[7]
ii = cycle_data[1]*1000
EE = cycle_data[11]





# Make the figure ================================

#fig1, (ax11) = plt.subplots(1,1,figsize=(4.5,2.78))
#fig1, (ax11) = plt.subplots(1,1,figsize=(9,3))
fig1, (ax11) = plt.subplots(1,1,figsize=(2.6,2.1))


#ax12 = ax11.twinx()

ax11.set_xlabel("Time (hours)")
ax11.set_ylabel("Potential (V)")
#ax12.set_ylabel("Current (mA)")


ax11.tick_params(which='both', axis='both', direction='in', bottom=True, top=True, left=True, right=False)
#ax12.tick_params(which='both', axis='both', direction='in', bottom=False, top=False, left=False, right=True)




#ax11.set_ylim((1.8,4.2))
#ax12.set_ylim((-0.35,0.35))

fig1.tight_layout()




# Plot the data ================================

color = "Teal"
ax11.plot(tt,EE,color = color, ls='-', lw=2.0, label='Potential')

color = 'm'
#ax12.plot(tt,ii,color = color, ls='-', lw=1.0, label='Current')


#fig1.legend(bbox_to_anchor=(0.45, 0.4), fontsize = 'small', frameon=False)





# Save the figure ================================

output_name = 'plo'
plt.show()
#plt.savefig('/Users/joshua/' + output_name + '.png', dpi=400)




