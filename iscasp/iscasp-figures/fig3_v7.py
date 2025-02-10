import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Original data
y = iris.target  # Class labels

# Define the transformation matrix P (from your ECA model)
P = np.array([[-0.4772,  0.4370,  0.4973, -0.1016],
              [-0.7707,  0.4478, -0.4387, -0.0193],
              [-0.3080, -0.5219,  0.7055,  0.2904],
              [ 0.2888,  0.5798, -0.2498,  0.9513]])

# Define the L_matrix with 4x4 format and masked last column
L_matrix = np.array([[np.nan, 1., 1., 0, np.nan],
                     [np.nan, 1., 0, 0, np.nan],
                     [np.nan, 0, 1., 1., np.nan],
                     [np.nan, 0, 0, 1., np.nan]])

L_matrix = np.array([[np.nan, 1., 1., 0],
                     [np.nan, 1., 0, 0],
                     [np.nan, 0, 1., 1.],
                     [np.nan, 0, 0, 1.]])

# Mask for the heatmap to hide certain cells
mask = np.isnan(L_matrix)

# Transform the data using P (ECA) and PCA
eca_transformed = X @ P
pca = PCA(n_components=3)
pca_transformed = pca.fit_transform(X)

# Set up the 1x5 figure layout
fig = plt.figure(figsize=(30, 6))

# Set font sizes for consistency
title_fontsize = 18
axis_label_fontsize = 16

# PCA Result (All Classes)
ax0 = fig.add_subplot(151, projection='3d')
scatter = ax0.scatter(pca_transformed[:, 0], pca_transformed[:, 1], pca_transformed[:, 2], c=y, cmap='cividis', s=50)
ax0.set_title('PCA (All Classes)', fontsize=title_fontsize, pad=20)
ax0.set_xlabel('PC 1', fontsize=axis_label_fontsize, labelpad=15)
ax0.set_ylabel('PC 2', fontsize=axis_label_fontsize, labelpad=15)
ax0.set_zlabel('PC 3', fontsize=axis_label_fontsize, labelpad=25)
ax0.tick_params(axis='both', which='both', labelsize=0, length=0)
ax0.grid(False)

# ECA Class 1 vs Class 2
ax1 = fig.add_subplot(152, projection='3d')
indices = np.where((y == 0) | (y == 1))[0]
X_plot = eca_transformed[indices][:, [0, 1, 2]]
y_plot = y[indices]
scatter = ax1.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], c=y_plot, cmap='cividis', s=50)
ax1.set_title('ECA Class 1 vs Class 2', fontsize=title_fontsize, pad=20)
ax1.set_xlabel('Eigenfeature 1', fontsize=axis_label_fontsize, labelpad=15)
ax1.set_ylabel('Eigenfeature 2', fontsize=axis_label_fontsize, labelpad=15)
ax1.set_zlabel('Eigenfeature 3', fontsize=axis_label_fontsize, labelpad=25)
ax1.tick_params(axis='both', which='both', labelsize=0, length=0)
ax1.grid(False)

# ECA Class 2 vs Class 3
ax2 = fig.add_subplot(153, projection='3d')
indices = np.where((y == 1) | (y == 2))[0]
X_plot = eca_transformed[indices][:, [0, 2, 3]]
y_plot = y[indices]
scatter = ax2.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], c=y_plot, cmap='cividis', s=50)
ax2.set_title('ECA Class 2 vs Class 3', fontsize=title_fontsize, pad=20)
ax2.set_xlabel('Eigenfeature 1', fontsize=axis_label_fontsize, labelpad=15)
ax2.set_ylabel('Eigenfeature 3', fontsize=axis_label_fontsize, labelpad=15)
ax2.set_zlabel('Eigenfeature 4', fontsize=axis_label_fontsize, labelpad=25)
ax2.tick_params(axis='both', which='both', labelsize=0, length=0)
ax2.grid(False)

# ECA Class 1 vs Class 3
ax3 = fig.add_subplot(154, projection='3d')
indices = np.where((y == 0) | (y == 2))[0]
X_plot = eca_transformed[indices][:, [0, 1, 3]]
y_plot = y[indices]
scatter = ax3.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], c=y_plot, cmap='cividis', s=50)
ax3.set_title('ECA Class 1 vs Class 3', fontsize=title_fontsize, pad=20)
ax3.set_xlabel('Eigenfeature 1', fontsize=axis_label_fontsize, labelpad=15)
ax3.set_ylabel('Eigenfeature 2', fontsize=axis_label_fontsize, labelpad=15)
ax3.set_zlabel('Eigenfeature 4', fontsize=axis_label_fontsize, labelpad=25)
ax3.tick_params(axis='both', which='both', labelsize=0, length=0)
ax3.grid(False)

# L_matrix heatmap with formatting to hide zero entries and mask last column
ax4 = fig.add_subplot(155)
sns.heatmap(L_matrix, annot=True, mask=mask, cmap='Blues', fmt=".0f", cbar=False,
            ax=ax4, annot_kws={"size": 20}, linewidths=0, linecolor='white', square=True)
ax4.set_xticks([])
ax4.set_yticks([])

# Adjust spacing between plots for a balanced appearance
plt.subplots_adjust(left=0.05, right=0.95, wspace=0.50)
plt.show()


