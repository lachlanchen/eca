import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--disable-yellow-bg", action="store_true", help="Disable yellow background for 3D plots")
args = parser.parse_args()

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

# Define background color
yellow_bg = mcolors.to_rgba('yellow', alpha=0.1) if not args.disable_yellow_bg else None

fig = plt.figure(figsize=(12, 8))

# 2D Dataset P matrix vectors with yellow background and enhanced vectors
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_facecolor(yellow_bg)
for i in range(P_2d.shape[1]):
    ax1.arrow(0, 0, P_2d[0, i], P_2d[1, i],
              head_width=0.05, head_length=0.05,
              fc='blue', ec='blue', linewidth=2)
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_aspect('equal')
ax1.axis('off')

# 3D (2 classes) P matrix vectors with optional yellow background
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
if yellow_bg:
    ax2.set_facecolor(yellow_bg)
for i in range(2):
    ax2.quiver(0, 0, 0, P_3d[0, i], P_3d[1, i], P_3d[2, i],
               color='blue', arrow_length_ratio=0.1, linewidth=2)
ax2.set_xlim(-0.5, 1)
ax2.set_ylim(-0.5, 1)
ax2.set_zlim(-0.5, 1)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
ax2.grid(False)
ax2.set_box_aspect([1, 1, 1])

# 3D (3 classes) P matrix vectors with optional yellow background
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
if yellow_bg:
    ax3.set_facecolor(yellow_bg)
for i in range(P_3d_3c.shape[1]):
    ax3.quiver(0, 0, 0, P_3d_3c[0, i], P_3d_3c[1, i], P_3d_3c[2, i],
               color='blue', arrow_length_ratio=0.1, linewidth=2)
ax3.set_xlim(-1, 1)
ax3.set_ylim(-1, 1)
ax3.set_zlim(-1, 1)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_zticks([])
ax3.grid(False)
ax3.set_box_aspect([1, 1, 1])

# Plotting L matrices with adjusted sizes and no border for the 3x3 L matrix
# 2D Dataset L matrix heatmap as 3x3 grid
ax4 = fig.add_subplot(2, 3, 4)
sns.heatmap(L_2d_padded, annot=True, cmap='Blues', fmt=".1f", cbar=False,
            ax=ax4, annot_kws={"size": 20}, linewidths=0.5, linecolor='white', square=True)
ax4.set_xticks([])
ax4.set_yticks([])

# 3D (2 classes) Dataset L matrix heatmap as 3x3 grid
ax5 = fig.add_subplot(2, 3, 5)
sns.heatmap(L_3d_padded, annot=True, cmap='Blues', fmt=".1f", cbar=False,
            ax=ax5, annot_kws={"size": 20}, linewidths=0.5, linecolor='white', square=True)
ax5.set_xticks([])
ax5.set_yticks([])

# 3D (3 classes) Dataset L matrix heatmap as 3x3 grid, with no border
ax6 = fig.add_subplot(2, 3, 6)
sns.heatmap(L_3d_3c, annot=True, cmap='Blues', fmt=".1f", cbar=False,
            ax=ax6, annot_kws={"size": 20}, linewidths=0, linecolor='white', square=True)
ax6.set_xticks([])
ax6.set_yticks([])

plt.tight_layout()
plt.show()

