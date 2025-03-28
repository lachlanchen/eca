# Eigen Analysis

A Python package for Eigencomponent Analysis (ECA) and Unsupervised Eigencomponent Analysis (UECA) for classification and clustering tasks.

## Installation

```bash
# Install from PyPI
pip install eigen-analysis

# Install from source
git clone https://github.com/lachlanchen/eigen_analysis.git
cd eigen_analysis
pip install .
```

## Requirements

- numpy >= 1.18.0
- torch >= 1.7.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0
- scipy >= 1.6.0

## Usage

### Classification with ECA

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, adjusted_rand_score
from eigen_analysis import ECA

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train ECA model for classification
eca = ECA(num_clusters=3, learning_rate=0.01, num_epochs=1000)
eca.fit(X, y)

# Evaluate on training data
y_pred = eca.predict(X)
training_accuracy = accuracy_score(y, y_pred)
print(f"Training accuracy: {training_accuracy:.4f}")

# Generate some test data
np.random.seed(42)
test_indices = np.random.choice(len(X), size=int(0.2*len(X)), replace=False)
X_test = X[test_indices]
y_test = y[test_indices]

# Evaluate on test data
y_test_pred = eca.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Visualize results
from eigen_analysis.visualization import visualize_clustering_results

# Create visualization with output directory
fig = visualize_clustering_results(
    X, y, y_pred, 
    eca.loss_history_, 
    eca.transform(X),
    eca.num_epochs,
    eca.model_,
    (eca.L_numpy_ > 0.5).astype(float),
    eca.L_numpy_,
    eca.P_numpy_,
    "Iris",
    output_dir="eca_classification_results"
)

# Calculate clustering performance
ari_score = adjusted_rand_score(y, y_pred)
print(f"Clustering performance (ARI Score): {ari_score:.4f}")
```

### Clustering with UECA

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from eigen_analysis import UECA

# Generate synthetic data with 3 clusters
X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)

# Train UECA model for clustering
ueca = UECA(num_clusters=3, learning_rate=0.01, num_epochs=3000)
ueca.fit(X, y_true)  # y_true used only for evaluation, not for training

# Calculate ARI
ari_score = adjusted_rand_score(y_true, ueca.labels_)
print(f"Adjusted Rand Index: {ari_score:.4f}")

# Visualize clustering results
from eigen_analysis.visualization import visualize_clustering_results

# Create visualization with output directory
fig = visualize_clustering_results(
    X, 
    y_true, 
    ueca.remapped_labels_, 
    ueca.loss_history_, 
    ueca.transform(X),
    ueca.num_epochs,
    ueca.model_,
    ueca.L_hard_numpy_,
    ueca.L_numpy_,
    ueca.P_numpy_,
    "Custom Dataset",
    output_dir="eca_clustering_results"
)

# Print clustering performance
print(f"Clustering performance (ARI Score): {ari_score:.4f}")
```

### Convenience Function

The package also provides a unified API through the `eigencomponent_analysis` function:

```python
from eigen_analysis import eigencomponent_analysis

# For classification (supervised)
model = eigencomponent_analysis(X, y, num_clusters=3)

# For clustering (unsupervised)
model = eigencomponent_analysis(X, num_clusters=3)
```

## Visualization

The package includes visualization tools through the `visualize_clustering_results` function in the `eigen_analysis.visualization` module. This function accepts the following parameters:

```python
visualize_clustering_results(
    X,                        # Input data
    y_true,                   # True class labels
    remapped_predictions,     # Predicted labels after remapping
    loss_history,             # Training loss history
    psi_final,                # Final projections
    num_epochs,               # Number of training epochs
    model,                    # Trained model
    L_matrix,                 # Binary mapping matrix
    L_soft,                   # Soft mapping matrix
    P_matrix,                 # Transformation matrix P
    dataset_name,             # Name of the dataset
    feature_names=None,       # Names of the features (optional)
    class_names=None,         # Names of the classes (optional)
    output_dir=None,          # Directory to save visualizations (optional)
    invert_features=None,     # Features to invert (optional)
    feature_signs=None,       # Signs for features (optional)
    hide_3d_ticks=False       # Hide ticks on 3D plot (optional)
)
```

For MNIST visualization, a specialized function is provided:

```python
from eigen_analysis.visualization import visualize_mnist_eigenfeatures

visualize_mnist_eigenfeatures(
    model,                    # Trained ECA model
    output_dir='mnist_visualizations',  # Output directory
    font_scale_dist=1.2,      # Font scaling for distribution charts
    font_scale_heatmap=1.0    # Font scaling for heatmaps
)
```

## Model Parameters

### ECA Model (Supervised)

- `num_clusters`: Number of classes
- `learning_rate`: Learning rate for optimizer (default: 0.01)
- `num_epochs`: Number of training epochs (default: 10000)
- `temp`: Temperature parameter for sigmoid (default: 10.0)
- `random_state`: Random seed for reproducibility
- `device`: Device to use ('cpu' or 'cuda')

### UECA Model (Unsupervised)

- `num_clusters`: Number of clusters
- `learning_rate`: Learning rate for optimizer (default: 0.01)
- `num_epochs`: Number of training epochs (default: 3000)
- `random_state`: Random seed for reproducibility
- `device`: Device to use ('cpu' or 'cuda')

## Complete Demo Script

The package includes a demo script that demonstrates both classification and clustering:

```python
#!/usr/bin/env python3
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, accuracy_score
from eigen_analysis import ECA, UECA

def run_classification_demo():
    print("\n==== ECA Classification Demo ====")
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Train ECA model for classification
    print("Training ECA model...")
    eca = ECA(num_clusters=3, learning_rate=0.01, num_epochs=10000)
    eca.fit(X, y)
    
    # Evaluate on training data
    y_pred = eca.predict(X)
    training_accuracy = accuracy_score(y, y_pred)
    print(f"Training accuracy: {training_accuracy:.4f}")
    
    # Generate some test data
    np.random.seed(42)
    test_indices = np.random.choice(len(X), size=int(0.2*len(X)), replace=False)
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # Evaluate on test data
    y_test_pred = eca.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Visualize results
    print("Visualizing results...\n")
    from eigen_analysis.visualization import visualize_clustering_results
    
    # Create visualization with output directory
    fig = visualize_clustering_results(
        X, y, y_pred, 
        eca.loss_history_, 
        eca.transform(X),
        eca.num_epochs,
        eca.model_,
        (eca.L_numpy_ > 0.5).astype(float),
        eca.L_numpy_,
        eca.P_numpy_,
        "Iris",
        output_dir="eca_classification_results"
    )
    
    # Calculate clustering performance
    ari_score = adjusted_rand_score(y, y_pred)
    print(f"Clustering performance (ARI Score): {ari_score:.4f}")
    print("Classification demo completed!")
    
    return eca

def run_clustering_demo():
    print("\n==== ECA Clustering Demo ====")
    
    # Generate synthetic data with 3 clusters
    X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)
    
    # Train UECA model for clustering
    print("Training ECA model for clustering...")
    ueca = UECA(num_clusters=3, learning_rate=0.01, num_epochs=3000)
    ueca.fit(X, y_true)  # y_true used only for evaluation, not for training
    
    # Calculate ARI
    ari_score = adjusted_rand_score(y_true, ueca.labels_)
    print(f"Adjusted Rand Index: {ari_score:.4f}")
    
    # Visualize clustering results
    print("Visualizing clustering results...\n")
    from eigen_analysis.visualization import visualize_clustering_results
    
    # Create visualization with output directory
    fig = visualize_clustering_results(
        X, 
        y_true, 
        ueca.remapped_labels_, 
        ueca.loss_history_, 
        ueca.transform(X),
        ueca.num_epochs,
        ueca.model_,
        ueca.L_hard_numpy_,
        ueca.L_numpy_,
        ueca.P_numpy_,
        "Custom Dataset",
        output_dir="eca_clustering_results"
    )
    
    # Print clustering performance
    print(f"Clustering performance (ARI Score): {ari_score:.4f}")
    print("Clustering demo completed!")
    
    return ueca

if __name__ == "__main__":
    # Run demos
    eca_model = run_classification_demo()
    ueca_model = run_clustering_demo()
    
    print("\nDemos completed successfully!")
    print("Results saved in the 'eca_classification_results' and 'eca_clustering_results' directories.")
```

## Citation

If you use this package in your research, please cite:

```bibtex
@inproceedings{chen2025eigen,
  title={Eigen-Component Analysis: {A} Quantum Theory-Inspired Linear Model},
  author={Chen, Rongzhou and Zhao, Yaping and Liu, Hanghang and Xu, Haohan and Ma, Shaohua and Lam, Edmund Y.},
  booktitle={2025 IEEE International Symposium on Circuits and Systems (ISCAS)},
  pages={},
  year={2025},
  publisher={IEEE},
  doi={},
}
```

## License

MIT