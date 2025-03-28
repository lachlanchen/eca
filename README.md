# Eigen Analysis Package Guide

This package provides a scikit-learn compatible implementation of Eigencomponent Analysis (ECA) for both classification and clustering tasks. The package maintains your original implementation's integrity while providing an easier-to-use interface.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/lachlanchen/eca.git
cd eca/eigen_analysis

# Install in development mode
pip install -e .

# Or install directly
pip install .
```

### From PyPI (after publishing)

```bash
pip install eigen-analysis
```

## Usage

### Unified API

The package provides a unified API that automatically selects between supervised (classification) and unsupervised (clustering) modes based on whether labels are provided:

```python
import numpy as np
from eigen_analysis import ECA

# Load your data
X = ...  # Your feature matrix
y = ...  # Your labels (optional)

# For classification (supervised learning)
eca = ECA(num_clusters=3)  # Specify the number of classes
eca.fit(X, y)  # Provide labels

# For clustering (unsupervised learning)
eca = ECA(num_clusters=3)  # Number of clusters must be specified
eca.fit(X)  # No labels provided
```

You can also use the convenience function that automatically determines the mode:

```python
from eigen_analysis import eigencomponent_analysis

# For classification (supervised)
model = eigencomponent_analysis(X, y, num_clusters=3)

# For clustering (unsupervised)
model = eigencomponent_analysis(X, num_clusters=3)
```

### Classification Example

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from eigen_analysis import ECA

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and train model
eca = ECA(num_clusters=3, num_epochs=1000)
eca.fit(X_train, y_train)

# Make predictions
y_pred = eca.predict(X_test)

# Evaluate model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Visualize results
eca.visualize(X_test, y_test, output_dir='results')
```

### Clustering Example

```python
import numpy as np
from sklearn import datasets
from eigen_analysis import ECA

# Load data
iris = datasets.load_iris()
X = iris.data

# Create and train model for clustering
eca = ECA(num_clusters=3, num_epochs=1000)
eca.fit(X)  # No labels provided

# Get cluster assignments
clusters = eca.predict(X)

# Visualize clustering results
eca.visualize(X, output_dir='clustering_results')

# If ground truth labels are available, you can evaluate the clustering
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(iris.target, clusters)
print(f"Adjusted Rand Index: {ari:.4f}")
```

## Key Features

1. **Scikit-learn Compatibility**: The models implement the scikit-learn Estimator API with `fit`, `transform`, and `predict` methods.

2. **Unified Interface**: A consistent API for both classification and clustering tasks.

3. **Visualization Tools**: Built-in visualization methods that highlight:
   - Eigenfeatures
   - Feature-to-class mappings (L matrix)
   - Projected data clusters
   - Training loss curves

4. **MNIST Visualization**: Special visualization capabilities for MNIST dataset.

5. **Model Inspection**: Easy access to important model components like the P and L matrices.

## Advanced Usage

### Accessing Model Components

After training, you can access the model's internal components:

```python
# Train the model
eca = ECA(num_clusters=3)
eca.fit(X, y)

# Access the P matrix (eigenfeatures)
P_matrix = eca.P_numpy_

# Access the L matrix (mapping matrix)
L_matrix = eca.L_numpy_

# Access binary version of L matrix
L_binary = (eca.L_numpy_ > 0.5).astype(float)
```

### Custom Visualization

You can customize the visualization by providing feature and class names:

```python
eca.visualize(
    X=X_test,
    y=y_test,
    feature_names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
    class_names=["Setosa", "Versicolor", "Virginica"],
    output_dir='custom_visualization'
)
```

### Working with MNIST

For MNIST visualization, there's a specialized approach:

```python
from torchvision import datasets, transforms
import torch
import numpy as np

# Load MNIST
mnist_train = datasets.MNIST('../data', train=True, download=True)
X_train = mnist_train.data.reshape(-1, 784).float() / 255.0
y_train = mnist_train.targets

# Train ECA model
eca = ECA(num_clusters=10, num_epochs=1000)
eca.fit(X_train, y_train)

# Visualize MNIST eigenfeatures
eca.visualize(output_dir='mnist_results')
```

## Model Parameters

### ECA and UECA Models

- `num_clusters`: Number of clusters/classes to find
- `learning_rate`: Learning rate for Adam optimizer (default: 0.01)
- `num_epochs`: Number of training epochs (default: 3000)
- `temp`: Temperature parameter for sigmoid (ECA only, default: 10.0)
- `random_state`: Random seed for reproducibility
- `device`: Device to use ('cpu' or 'cuda')

## Notes on Implementation

This package maintains the key mathematical aspects of your original implementation while providing a more user-friendly interface:

1. The antisymmetric matrix A is constructed as `A = A_raw - A_raw.t() + torch.diag(D)`
2. The transformation matrix P is computed using `P = torch.matrix_exp(A)`
3. The mapping matrix L is binarized using the straight-through estimator (STE)
4. The core design of L and P matrices is preserved

For clustering, the cosine similarity-based loss function is used as in your original implementation.
