import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

###############################################################################
# 1. Helper Functions to Generate Synthetic Datasets
###############################################################################

def generate_orthogonal_vectors():
    np.random.seed(42)
    v1 = np.random.rand(3);  v1 /= np.linalg.norm(v1)
    v2 = np.random.rand(3);  v2 -= v1 * np.dot(v2, v1);  v2 /= np.linalg.norm(v2)
    v3 = np.random.rand(3);  v3 -= (v1 * np.dot(v3, v1) + v2 * np.dot(v3, v2));  v3 /= np.linalg.norm(v3)
    return v1, v2, v3

def generate_2d_dataset(n_samples=1500, random_state=42, width=1.0):
    np.random.seed(random_state)
    n_samples_per_class = n_samples // 2

    # Class 1: around y = x
    t1 = np.random.normal(0, width, n_samples_per_class)
    x1 = t1
    y1 = t1 + np.random.normal(0, 0.5 * width, n_samples_per_class)
    X_class1 = np.vstack((x1, y1)).T

    # Class 2: around y = -x
    t2 = np.random.normal(0, width, n_samples_per_class)
    x2 = t2
    y2 = -t2 + np.random.normal(0, 0.5 * width, n_samples_per_class)
    X_class2 = np.vstack((x2, y2)).T

    X = np.vstack((X_class1, X_class2))
    y = np.array([0]*n_samples_per_class + [1]*n_samples_per_class)
    return X.astype(np.float32), y.astype(np.int64)

def generate_3d_dataset(n_samples=1500, random_state=42, width=1.0):
    np.random.seed(random_state)
    n_samples_per_class = n_samples // 2

    # Class 1: along x-axis
    x1 = np.random.normal(0, width, n_samples_per_class)
    y1 = np.random.normal(0, 0.5*width, n_samples_per_class)
    z1 = np.random.normal(0, 0.5*width, n_samples_per_class)
    X_class1 = np.vstack((x1, y1, z1)).T

    # Class 2: along y-axis
    x2 = np.random.normal(0, 0.5*width, n_samples_per_class)
    y2 = np.random.normal(0, width, n_samples_per_class)
    z2 = np.random.normal(0, 0.5*width, n_samples_per_class)
    X_class2 = np.vstack((x2, y2, z2)).T

    X = np.vstack((X_class1, X_class2))
    y = np.array([0]*n_samples_per_class + [1]*n_samples_per_class)
    return X.astype(np.float32), y.astype(np.int64)

def generate_3d_3c_dataset(n_samples=3000, random_state=42, width=1.0):
    np.random.seed(random_state)
    n_samples_per_class = n_samples // 3
    v1, v2, v3 = generate_orthogonal_vectors()

    # Class 1: along v1
    t1 = np.random.normal(0, width, n_samples_per_class)
    X_class1 = np.outer(t1, v1) + np.random.normal(0, 0.3*width, (n_samples_per_class, 3))
    # Class 2: along v2
    t2 = np.random.normal(0, width, n_samples_per_class)
    X_class2 = np.outer(t2, v2) + np.random.normal(0, 0.3*width, (n_samples_per_class, 3))
    # Class 3: along v3
    t3 = np.random.normal(0, width, n_samples_per_class)
    X_class3 = np.outer(t3, v3) + np.random.normal(0, 0.3*width, (n_samples_per_class, 3))

    X = np.vstack((X_class1, X_class2, X_class3))
    y = np.array([0]*n_samples_per_class + [1]*n_samples_per_class + [2]*n_samples_per_class)
    return X.astype(np.float32), y.astype(np.int64)

###############################################################################
# 2. Generate the Three Synthetic Datasets
###############################################################################

X_2d, y_2d = generate_2d_dataset()
X_3d, y_3d = generate_3d_dataset()
X_3d_3c, y_3d_3c = generate_3d_3c_dataset()

###############################################################################
# 3. Plotting (No Titles, No Tick Labels, No Grids)
###############################################################################

fig = plt.figure(figsize=(15, 5))

# --------------------------------------------------------------------
# Left: 2D Synthetic Dataset (ball-like markers in 2D)
# --------------------------------------------------------------------
ax1 = fig.add_subplot(131)
scatter_2d = sns.scatterplot(
    x=X_2d[:, 0],
    y=X_2d[:, 1],
    hue=y_2d,
    palette='Set1',
    s=60,
    marker='o',
    ax=ax1,
    legend='brief'
)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_frame_on(False)
ax1.legend(title="Class", loc="upper right", markerscale=1.5, fontsize=12, title_fontsize=14)

# --------------------------------------------------------------------
# Middle: 3D Synthetic Dataset (two classes) with "ball" markers
# --------------------------------------------------------------------
ax2 = fig.add_subplot(132, projection='3d')
p2 = ax2.scatter(
    X_3d[:, 0],
    X_3d[:, 1],
    X_3d[:, 2],
    c=y_3d,
    cmap='Set1',
    s=60,
    marker='o'
)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])

# Hide 3D panes (optional):
ax2.xaxis.pane.set_visible(False)
ax2.yaxis.pane.set_visible(False)
ax2.zaxis.pane.set_visible(False)

ax2.dist = 7.5  # Slightly closer or further to match 2D scale visually
leg2 = ax2.legend(
    *p2.legend_elements(),
    title="Class",
    loc="upper right",
    markerscale=1.5,
    fontsize=12,
    title_fontsize=14
)

# --------------------------------------------------------------------
# Right: 3D Synthetic Dataset (three classes) with "ball" markers
# --------------------------------------------------------------------
ax3 = fig.add_subplot(133, projection='3d')
p3 = ax3.scatter(
    X_3d_3c[:, 0],
    X_3d_3c[:, 1],
    X_3d_3c[:, 2],
    c=y_3d_3c,
    cmap='Set1',
    s=60,
    marker='o'
)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_zticks([])

ax3.xaxis.pane.set_visible(False)
ax3.yaxis.pane.set_visible(False)
ax3.zaxis.pane.set_visible(False)

ax3.dist = 7.5  # Adjust zoom for a visually similar size
leg3 = ax3.legend(
    *p3.legend_elements(),
    title="Class",
    loc="upper right",
    markerscale=1.5,
    fontsize=12,
    title_fontsize=14
)

# Reduce spacing between subplots
plt.subplots_adjust(wspace=0.1, hspace=0)
plt.show()

