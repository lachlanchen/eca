import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# Define the P and L matrices for plotting
P_2d = np.array([[0.7693, -0.6585],
                 [0.6389,  0.7526]])
P_3d = np.array([[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]])
P_3d_3c = np.array([[0.9624,  0.2672, -0.0355],
                    [-0.2283,  0.7315, -0.6441],
                    [-0.1473,  0.6273,  0.7641]])

L_2d = np.array([[1., 0.],
                 [0., 1.]])
L_3d = np.array([[1., 0.],
                 [0., 1.],
                 [0., 0.]])
L_3d_3c = np.array([[0., 1., 0.],
                    [1., 0., 0.],
                    [0., 0., 1.]])

# Function to center a smaller matrix within a 3x3 grid
def center_matrix_in_grid(small_matrix, grid_size):
    small_rows, small_cols = small_matrix.shape
    grid = np.zeros((grid_size, grid_size))
    start_row = (grid_size - small_rows) // 2
    start_col = (grid_size - small_cols) // 2
    grid[start_row:start_row+small_rows, start_col:start_col+small_cols] = small_matrix
    return grid

# Adjust L matrices to be displayed as 3x3 grids, centered
L_2d_padded = center_matrix_in_grid(L_2d, 3)
L_3d_padded = center_matrix_in_grid(L_3d, 3)

# Adjusted masks to align with the exact shapes of each L matrix
mask_2d_corrected = np.ones((3, 3), dtype=bool)
mask_2d_corrected[:2, :2] = False  # Unmasking the central 2x2 section

mask_3d_corrected = np.ones((3, 3), dtype=bool)
mask_3d_corrected[:3, :2] = False  # Unmasking the central 3x2 section

# Creating the figure for aligned visuals
fig = plt.figure(figsize=(12, 8))
# yellow_bg = mcolors.to_rgba('yellow', alpha=0.1)

# Adjusted 2D Dataset P matrix vectors with yellow background
ax1 = fig.add_subplot(2, 3, 1)
# ax1.set_facecolor("lightyellow")
for i in range(P_2d.shape[1]):
    ax1.arrow(0, 0, P_2d[0, i], P_2d[1, i], head_width=0.05, head_length=0.05, fc='blue', ec='blue', linewidth=2)
ax1.set_xlim(-1, 1)
ax1.set_ylim(-0.5, 1)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_position([0.05, 0.55, 0.3, 0.3])

# 3D (2 classes) P matrix vectors with 3D background (axes), no grid
# ax2 = fig.add_subplot(2, 3, 2, projection='3d', facecolor="lightyellow")
ax2 = fig.add_subplot(2, 3, 2, projection='3d', facecolor="white")
for i in range(2):  # Only plotting the first two vectors
    ax2.quiver(0, 0, 0, P_3d[0, i], P_3d[1, i], P_3d[2, i], color='blue', arrow_length_ratio=0.1, linewidth=2)
ax2.set_xlim(-0.5, 0.6)
ax2.set_ylim(-0.5, 0.6)
ax2.set_zlim(-0.5, 0.6)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
ax2.grid(False)
ax2.set_box_aspect([1, 1, 1])

# 3D (3 classes) P matrix vectors with 3D background (axes), no grid
# ax3 = fig.add_subplot(2, 3, 3, projection='3d', facecolor="lightyellow")
ax3 = fig.add_subplot(2, 3, 3, projection='3d', facecolor="white")
for i in range(P_3d_3c.shape[1]):
    ax3.quiver(0, 0, 0, P_3d_3c[0, i], P_3d_3c[1, i], P_3d_3c[2, i], color='blue', arrow_length_ratio=0.1, linewidth=2)
ax3.set_xlim(-0.6, 0.6)
ax3.set_ylim(-0.6, 0.6)
ax3.set_zlim(-0.6, 0.6)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_zticks([])
ax3.grid(False)
ax3.set_box_aspect([1, 1, 1])

# L matrix for 2D Dataset, centered in 3x3 grid with masking for empty cells
ax4 = fig.add_subplot(2, 3, 4)
sns.heatmap(L_2d_padded, annot=True, mask=mask_2d_corrected, cmap='Blues', cbar=False, square=True, 
            ax=ax4, annot_kws={"size": 20}, linewidths=0.5, linecolor='white', fmt="g")
ax4.set_xticks([])
ax4.set_yticks([])

# L matrix for 3D (2 classes) Dataset, centered with masking for empty cells
ax5 = fig.add_subplot(2, 3, 5)
sns.heatmap(L_3d_padded, annot=True, mask=mask_3d_corrected, cmap='Blues', cbar=False, square=True, 
            ax=ax5, annot_kws={"size": 20}, linewidths=0.5, linecolor='white', fmt="g")
ax5.set_xticks([])
ax5.set_yticks([])

# L matrix for 3D (3 classes) Dataset, full display
ax6 = fig.add_subplot(2, 3, 6)
sns.heatmap(L_3d_3c, annot=True, cmap='Blues', cbar=False, square=True, 
            # ax=ax6, annot_kws={"size": 20}, linewidths=0.5, linecolor='gray', fmt="g")
            ax=ax6, annot_kws={"size": 20}, linewidths=0.5, linecolor='white', fmt="g")
ax6.set_xticks([])
ax6.set_yticks([])

# Adjust layout for clarity
plt.tight_layout()
plt.show()

