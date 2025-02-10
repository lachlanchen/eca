import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

###############################################################################
# 1. Data Generation Functions
###############################################################################

def generate_2d_dataset(n_samples=1500, random_state=42, width=1.0):
    """
    Generate a 2D synthetic dataset with two classes roughly on lines y=x and y=-x.
    """
    np.random.seed(random_state)
    n_samples_per_class = n_samples // 2

    # Class 1: points around y = x
    t1 = np.random.normal(0, width, n_samples_per_class)
    x1 = t1
    y1 = t1 + np.random.normal(0, 0.5 * width, n_samples_per_class)
    X_class1 = np.vstack((x1, y1)).T

    # Class 2: points around y = -x
    t2 = np.random.normal(0, width, n_samples_per_class)
    x2 = t2
    y2 = -t2 + np.random.normal(0, 0.5 * width, n_samples_per_class)
    X_class2 = np.vstack((x2, y2)).T

    # Combine and label
    X = np.vstack((X_class1, X_class2))
    y = np.array([0] * n_samples_per_class + [1] * n_samples_per_class)

    return X, y

def generate_3d_dataset(n_samples=1500, random_state=42, width=1.0):
    """
    Generate a 3D synthetic dataset with two classes, each distributed along different axes.
    """
    np.random.seed(random_state)
    n_samples_per_class = n_samples // 2

    # Class 1: along x-axis (with some noise)
    x1 = np.random.normal(0, width, n_samples_per_class)
    y1 = np.random.normal(0, 0.5 * width, n_samples_per_class)
    z1 = np.random.normal(0, 0.5 * width, n_samples_per_class)
    X_class1 = np.vstack((x1, y1, z1)).T

    # Class 2: along y-axis (with some noise)
    x2 = np.random.normal(0, 0.5 * width, n_samples_per_class)
    y2 = np.random.normal(0, width, n_samples_per_class)
    z2 = np.random.normal(0, 0.5 * width, n_samples_per_class)
    X_class2 = np.vstack((x2, y2, z2)).T

    # Combine and label
    X = np.vstack((X_class1, X_class2))
    y = np.array([0] * n_samples_per_class + [1] * n_samples_per_class)

    return X, y

def generate_orthogonal_vectors():
    """
    Generate three mutually orthogonal vectors using a Gram-Schmidt-like process.
    """
    np.random.seed(42)
    v1 = np.random.rand(3)
    v1 /= np.linalg.norm(v1)

    v2 = np.random.rand(3)
    v2 -= v1 * np.dot(v2, v1)
    v2 /= np.linalg.norm(v2)

    v3 = np.random.rand(3)
    v3 -= (v1 * np.dot(v3, v1) + v2 * np.dot(v3, v2))
    v3 /= np.linalg.norm(v3)

    return v1, v2, v3

def generate_3d_3c_dataset(n_samples=3000, random_state=42, width=1.0):
    """
    Generate a 3D synthetic dataset with three classes along three orthogonal directions.
    """
    np.random.seed(random_state)
    n_samples_per_class = n_samples // 3

    # Get three orthogonal vectors
    v1, v2, v3 = generate_orthogonal_vectors()

    # Class 1 along v1
    t1 = np.random.normal(0, width, n_samples_per_class)
    X_class1 = np.outer(t1, v1) + np.random.normal(0, 0.3 * width, (n_samples_per_class, 3))

    # Class 2 along v2
    t2 = np.random.normal(0, width, n_samples_per_class)
    X_class2 = np.outer(t2, v2) + np.random.normal(0, 0.3 * width, (n_samples_per_class, 3))

    # Class 3 along v3
    t3 = np.random.normal(0, width, n_samples_per_class)
    X_class3 = np.outer(t3, v3) + np.random.normal(0, 0.3 * width, (n_samples_per_class, 3))

    # Combine and label
    X = np.vstack((X_class1, X_class2, X_class3))
    y = np.array([0] * n_samples_per_class + [1] * n_samples_per_class + [2] * n_samples_per_class)

    return X, y

###############################################################################
# 2. Generate the Datasets
###############################################################################

X_2d, y_2d = generate_2d_dataset()
X_3d, y_3d = generate_3d_dataset()
X_3d_3c, y_3d_3c = generate_3d_3c_dataset()

###############################################################################
# 3. Plotting with "Ball-Like" Markers (Circles)
###############################################################################

# Create a figure with three subplots side by side
fig = plt.figure(figsize=(15, 5))

# ---------------------------
# Subplot 1: 2D Dataset
# ---------------------------
ax1 = fig.add_subplot(131)
sns.scatterplot(
    x=X_2d[:, 0],
    y=X_2d[:, 1],
    hue=y_2d,
    palette='Set1',
    edgecolor='k',
    s=60,
    marker='o',
    ax=ax1,
    legend='brief'
)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_frame_on(False)  # 2D axes support set_frame_on
ax1.legend(title="Class", loc="upper right", markerscale=1.5, fontsize=12, title_fontsize=14)

# ---------------------------
# Subplot 2: 3D Dataset (2 classes)
# ---------------------------
ax2 = fig.add_subplot(132, projection='3d')
scatter_2d = ax2.scatter(
    X_3d[:, 0],
    X_3d[:, 1],
    X_3d[:, 2],
    c=y_3d,
    cmap='Set1',
    edgecolor='k',
    s=60,
    marker='o'
)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])

# In 3D, set_frame_on(False) is not supported, so omit it.
# Instead, we can hide the 3D panes if desired:
ax2.xaxis.pane.set_visible(False)
ax2.yaxis.pane.set_visible(False)
ax2.zaxis.pane.set_visible(False)

ax2.dist = 8  # Zoom factor for 3D
legend2d = ax2.legend(
    *scatter_2d.legend_elements(),
    title="Class",
    loc="upper right",
    markerscale=1.5,
    fontsize=12,
    title_fontsize=14
)

# ---------------------------
# Subplot 3: 3D Dataset (3 classes)
# ---------------------------
ax3 = fig.add_subplot(133, projection='3d')
scatter_3d = ax3.scatter(
    X_3d_3c[:, 0],
    X_3d_3c[:, 1],
    X_3d_3c[:, 2],
    c=y_3d_3c,
    cmap='Set1',
    edgecolor='k',
    s=60,
    marker='o'
)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_zticks([])

# Hide 3D panes here as well
ax3.xaxis.pane.set_visible(False)
ax3.yaxis.pane.set_visible(False)
ax3.zaxis.pane.set_visible(False)

ax3.dist = 8  # Keep zoom factor consistent
legend3d = ax3.legend(
    *scatter_3d.legend_elements(),
    title="Class",
    loc="upper right",
    markerscale=1.5,
    fontsize=12,
    title_fontsize=14
)

plt.subplots_adjust(wspace=0.1, hspace=0)
plt.show()

