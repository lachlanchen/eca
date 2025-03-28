# Eigen Analysis

A package for Eigencomponent Analysis (ECA) and Unsupervised Eigencomponent Analysis (UECA) for classification and clustering tasks.

## Installation

```bash
# Install from PyPI
pip install eigen-analysis

# Install from source
git clone https://github.com/yourusername/eigen_analysis.git
cd eigen_analysis
pip install .
```

## Usage

### Classification with ECA

```python
import numpy as np
from eigen_analysis import ECA

# Sample data
X = np.random.randn(100, 10)  # 100 samples, 10 features
y = np.random.randint(0, 3, 100)  # 3 classes

# Fit the ECA model
eca = ECA(num_clusters=3)
eca.fit(X, y)

# Transform data
X_transformed = eca.transform(X)

# Predict classes
y_pred = eca.predict(X)

# Visualize results
eca.visualize()
```

### Clustering with UECA

```python
import numpy as np
from eigen_analysis import ECA

# Sample data
X = np.random.randn(100, 10)  # 100 samples, 10 features

# Fit the ECA model for clustering
eca = ECA(num_clusters=3)
eca.fit(X)  # No labels provided - performs clustering

# Get cluster assignments
clusters = eca.predict(X)

# Visualize clustering results
eca.visualize()
```

## License

MIT
