import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Original data
y = iris.target  # Class labels

# Define the transformation matrix P (from your ECA model)
P = np.array([[-0.4772,  0.4370,  0.4973, -0.1016],
              [-0.7707,  0.4478, -0.4387, -0.0193],
              [-0.3080, -0.5219,  0.7055,  0.2904],
              [ 0.2888,  0.5798, -0.2498,  0.9513]])

# Define the mapping matrix L (from your ECA model)
L = np.array([[1., 1., 0.],
              [1., 0., 0.],
              [0., 1., 1.],
              [0., 0., 1.]])

# Transform the data using P
Psi = X @ P

# Prepare the figure with three subplots in a row
fig = plt.figure(figsize=(18, 5))

# Class 1 vs Class 2
ax1 = fig.add_subplot(131, projection='3d')
indices = np.where((y == 0) | (y == 1))[0]
X_plot = Psi[indices][:, [0, 1, 2]]  # Use dimensions associated with classes 1 and 2
y_plot = y[indices]
scatter = ax1.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], c=y_plot, cmap='viridis', edgecolor='k')
ax1.set_title('Class 1 vs Class 2')
ax1.set_xlabel('Eigenfeature 1')
ax1.set_ylabel('Eigenfeature 2')
ax1.set_zlabel('Eigenfeature 3')

# Class 2 vs Class 3
ax2 = fig.add_subplot(132, projection='3d')
indices = np.where((y == 1) | (y == 2))[0]
X_plot = Psi[indices][:, [0, 2, 3]]  # Excluding the 2nd dimension (1-indexed)
y_plot = y[indices]
scatter = ax2.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], c=y_plot, cmap='viridis', edgecolor='k')
ax2.set_title('Class 2 vs Class 3')
ax2.set_xlabel('Eigenfeature 1')
ax2.set_ylabel('Eigenfeature 3')
ax2.set_zlabel('Eigenfeature 4')

# Class 1 vs Class 3
ax3 = fig.add_subplot(133, projection='3d')
indices = np.where((y == 0) | (y == 2))[0]
X_plot = Psi[indices][:, [0, 1, 3]]  # Use relevant dimensions
y_plot = y[indices]
scatter = ax3.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], c=y_plot, cmap='viridis', edgecolor='k')
ax3.set_title('Class 1 vs Class 3')
ax3.set_xlabel('Eigenfeature 1')
ax3.set_ylabel('Eigenfeature 2')
ax3.set_zlabel('Eigenfeature 4')

plt.tight_layout()
plt.show()

