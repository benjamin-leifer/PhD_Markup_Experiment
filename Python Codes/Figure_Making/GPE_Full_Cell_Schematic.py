import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create a new figure
fig, ax = plt.subplots(figsize=(10, 6))

# Define the text size and calculate the layer thickness
text_size = 24
layer_thickness = text_size * 1.25 / 24

# Adjust the Zn thickness to fit the entire label
zn_thickness = text_size / 24

# Define the original thicknesses
steel_thickness_orig = 0.5
gpe_thickness_orig = 0.7

# Calculate the layer thickness for Zn and Stainless Steel, and twice the original for GPE
steel_thickness = layer_thickness
gpe_thickness = layer_thickness * 2

# Calculate the total height of the layers
total_height = 2 * steel_thickness + 2 * zn_thickness + gpe_thickness

# Draw the stainless steel layers
steel_color = 'grey'
ax.add_patch(patches.Rectangle((0, total_height - steel_thickness), 10, steel_thickness, color=steel_color))  # Top layer
ax.add_patch(patches.Rectangle((0, 0), 10, steel_thickness, color=steel_color))  # Bottom layer

# Draw the Zn layer
zn_color = 'lightgrey'
ax.add_patch(patches.Rectangle((0, total_height - steel_thickness - zn_thickness), 10, zn_thickness, color=zn_color))  # Top Zn layer

# Draw the gel polymer electrolyte layer
electrolyte_color = 'green'
ax.add_patch(patches.Rectangle((0, steel_thickness + zn_thickness), 10, gpe_thickness, color=electrolyte_color))

# Draw the cathode layer
cathode_color = 'black'
ax.add_patch(patches.Rectangle((0, steel_thickness), 10, zn_thickness, color=cathode_color))  # Cathode layer

# Add transparent square behind the "Gel Polymer Electrolyte" text
bg_color = 'white'
bg_alpha = 0.5
ax.add_patch(patches.Rectangle((1.5, steel_thickness + zn_thickness + gpe_thickness / 2 - 0.2), 7, 0.6, color=bg_color, alpha=bg_alpha))  # Gel Polymer Electrolyte

# Increase text size by 2x for labels on the figure
figure_text_size = 24
text_offset_x = 5

# Add text labels on the figure
plt.text(text_offset_x, total_height - steel_thickness / 2, 'Stainless Steel', ha='center', va='center', fontsize=figure_text_size)
plt.text(text_offset_x, total_height - steel_thickness - zn_thickness / 2, 'Zn', ha='center', va='center', fontsize=figure_text_size)
plt.text(text_offset_x, steel_thickness + zn_thickness + gpe_thickness / 2, 'Gel Polymer Electrolyte', ha='center', va='center', fontsize=figure_text_size)
plt.text(text_offset_x, steel_thickness + zn_thickness / 2, 'Cathode', ha='center', va='center', fontsize=figure_text_size, color='white')
plt.text(text_offset_x, steel_thickness / 2, 'Stainless Steel', ha='center', va='center', fontsize=figure_text_size)

# Add yellow dots to represent ions, ensuring they don't overlap the Zn or Cathode layers
num_dots = 100
dot_color = 'yellow'
for i in range(num_dots):
    x = np.random.uniform(0, 10)
    y = np.random.uniform(steel_thickness + zn_thickness, steel_thickness + zn_thickness + gpe_thickness)
    ax.add_patch(patches.Circle((x, y), 0.05, color=dot_color))

# Add thickness labels to the sides with the original thicknesses and text size 16
side_text_size = 16
offset = -1.2
plt.text(offset, total_height - steel_thickness / 2, '0.5 mm', ha='center', va='center', fontsize=side_text_size)
plt.text(offset, total_height - steel_thickness - zn_thickness / 2, '0.075 mm', ha='center', va='center', fontsize=side_text_size)
plt.text(offset, steel_thickness + zn_thickness + gpe_thickness / 2, '~0.7 mm', ha='center', va='center', fontsize=side_text_size)
plt.text(offset, steel_thickness + zn_thickness / 2, '0.075 mm', ha='center', va='center', fontsize=side_text_size)
plt.text(offset, steel_thickness / 2, '0.5 mm', ha='center', va='center', fontsize=side_text_size)

# Set plot limits and hide axes
plt.xlim(-2, 11)
plt.ylim(0, total_height + 0.5)
ax.axis('off')

# Show the plot
plt.show()
