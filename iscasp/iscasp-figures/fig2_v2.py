import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Function to create a centered matrix within a 3x3 grid
def center_matrix(matrix, size=3, shift_half_cell=False):
    centered = np.zeros((size, size))
    m_rows, m_cols = matrix.shape
    start_row = (size - m_rows) // 2
    start_col = (size - m_cols) // 2
    if shift_half_cell:
        start_col += 0.5  # Shift half cell horizontally
    # Since indices must be integers, we need to handle the half-cell shift
    if start_col % 1 != 0:
        start_col_int = int(np.floor(start_col))
        centered[start_row:start_row + m_rows, start_col_int:start_col_int + m_cols + 1] = 0.5
        centered[start_row:start_row + m_rows, start_col_int + 1:start_col_int + m_cols + 1] = matrix
    else:
        start_col_int = int(start_col)
        centered[start_row:start_row + m_rows, start_col_int:start_col_int + m_cols] = matrix
    return centered

# Creating the figure for aligned visuals
fig = plt.figure(figsize=(12, 8))

# Adjusted 2D Dataset P matrix vectors
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_facecolor('#fff5e6')  # Light yellow background
for i in range(P_2d.shape[1]):
    ax1.arrow(0, 0, P_2d[0, i], P_2d[1, i], head_width=0.03, head_length=0.05,
              fc='blue', ec='blue', linewidth=2, length_includes_head=True)
ax1.set_xlim(-1, 1)
ax1.set_ylim(-0.5, 1)
ax1.set_aspect('equal')
ax1.axis('off')
# Offset to lower position for alignment with 3D plots
ax1.set_position([0.05, 0.55, 0.27, 0.35])

# 3D (2 classes) P matrix vectors, plotting only the first two vectors
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.set_facecolor('#fff5e6')  # Light yellow background
for i in range(2):  # Only plotting the first two vectors
    ax2.quiver(0, 0, 0, P_3d[0, i], P_3d[1, i], P_3d[2, i],
               color='blue', arrow_length_ratio=0.1, linewidth=2)
ax2.set_xlim(-0.5, 0.6)
ax2.set_ylim(-0.5, 0.6)
ax2.set_zlim(-0.5, 0.6)
ax2.axis('off')

# 3D (3 classes) P matrix vectors, plotting all vectors
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.set_facecolor('#fff5e6')  # Light yellow background
for i in range(P_3d_3c.shape[1]):
    ax3.quiver(0, 0, 0, P_3d_3c[0, i], P_3d_3c[1, i], P_3d_3c[2, i],
               color='blue', arrow_length_ratio=0.1, linewidth=2)
ax3.set_xlim(-0.6, 0.6)
ax3.set_ylim(-0.6, 0.6)
ax3.set_zlim(-0.6, 0.6)
ax3.axis('off')

# Prepare L matrices as 3x3 grids, centering the original matrices
L_2d_centered = center_matrix(L_2d)
L_3d_centered = center_matrix(L_3d, shift_half_cell=True)
L_3d_3c_centered = L_3d_3c  # Already 3x3, no need to center

# Plotting L matrices with consistent formatting
# 2D Dataset L matrix heatmap
ax4 = fig.add_subplot(2, 3, 4)
sns.heatmap(L_2d_centered, annot=True, cmap='Blues', fmt=".1f", cbar=False,
            ax=ax4, annot_kws={"size": 16}, linewidths=0.5, linecolor='gray', square=True)
ax4.set_xticks([])
ax4.set_yticks([])

# 3D (2 classes) Dataset L matrix heatmap
ax5 = fig.add_subplot(2, 3, 5)
sns.heatmap(L_3d_centered, annot=True, cmap='Blues', fmt=".1f", cbar=False,
            ax=ax5, annot_kws={"size": 16}, linewidths=0.5, linecolor='gray', square=True)
ax5.set_xticks([])
ax5.set_yticks([])

# 3D (3 classes) Dataset L matrix heatmap
ax6 = fig.add_subplot(2, 3, 6)
sns.heatmap(L_3d_3c_centered, annot=True, cmap='Blues', fmt=".1f", cbar=False,
            ax=ax6, annot_kws={"size": 16}, linewidths=0.5, linecolor='gray', square=True)
ax6.set_xticks([])
ax6.set_yticks([])

# Adjust layout for clarity
plt.tight_layout()
plt.show()

