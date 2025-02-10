import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

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

# Creating the figure for aligned visuals
fig = plt.figure(figsize=(12, 8))

# Adjusted 2D Dataset P matrix vectors
ax1 = fig.add_subplot(2, 3, 1)
for i in range(P_2d.shape[1]):
    ax1.arrow(0, 0, P_2d[0, i], P_2d[1, i], head_width=0.05, head_length=0.05, fc='blue', ec='blue')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-0.5, 1)
ax1.set_aspect('equal')
ax1.axis('off')
# Offset to lower position for alignment with 3D plots
ax1.set_position([0.05, 0.55, 0.3, 0.3])

# 3D (2 classes) P matrix vectors, plotting only the first two vectors, with adjusted zoom-out
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
for i in range(2):  # Only plotting the first two vectors
    ax2.quiver(0, 0, 0, P_3d[0, i], P_3d[1, i], P_3d[2, i], color='blue', arrow_length_ratio=0.1)
ax2.set_xlim(-0.5, 0.6)
ax2.set_ylim(-0.5, 0.6)
ax2.set_zlim(-0.5, 0.6)
ax2.axis('off')

# 3D (3 classes) P matrix vectors, plotting all vectors, with adjusted zoom-out
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
for i in range(P_3d_3c.shape[1]):
    ax3.quiver(0, 0, 0, P_3d_3c[0, i], P_3d_3c[1, i], P_3d_3c[2, i], color='blue', arrow_length_ratio=0.1)
ax3.set_xlim(-0.6, 0.6)
ax3.set_ylim(-0.6, 0.6)
ax3.set_zlim(-0.6, 0.6)
ax3.axis('off')

# Plotting L matrices below with integer/float formatting without trailing zeros
# 2D Dataset L matrix heatmap
ax4 = fig.add_subplot(2, 3, 4)
sns.heatmap(L_2d, annot=True, cmap='Blues', fmt="g", cbar=False, 
            ax=ax4, annot_kws={"size": 20}, linewidths=0.5, linecolor='gray')
ax4.set_xticks([])
ax4.set_yticks([])

# 3D (2 classes) Dataset L matrix heatmap
ax5 = fig.add_subplot(2, 3, 5)
sns.heatmap(L_3d, annot=True, cmap='Blues', fmt="g", cbar=False, 
            ax=ax5, annot_kws={"size": 20}, linewidths=0.5, linecolor='gray')
ax5.set_xticks([])
ax5.set_yticks([])

# 3D (3 classes) Dataset L matrix heatmap
ax6 = fig.add_subplot(2, 3, 6)
sns.heatmap(L_3d_3c, annot=True, cmap='Blues', fmt="g", cbar=False, 
            ax=ax6, annot_kws={"size": 20}, linewidths=0.5, linecolor='gray')
ax6.set_xticks([])
ax6.set_yticks([])

# Adjust layout for clarity
plt.tight_layout()
plt.show()
