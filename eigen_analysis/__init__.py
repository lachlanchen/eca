"""
Eigen Analysis
-------------

A package for Eigencomponent Analysis (ECA) and Unsupervised Eigencomponent
Analysis (UECA) for classification and clustering tasks.

The package provides a scikit-learn compatible API for working with ECA models.
"""

__version__ = "0.1.0"

from .eca import ECA
from .ueca import UECA

# Unified API that checks for y to determine method
def eigencomponent_analysis(X, y=None, num_clusters=None, learning_rate=0.01, 
                           num_epochs=None, random_state=None, device=None):
    """
    Factory function to create an ECA or UECA model based on whether y is provided.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.
    y : array-like of shape (n_samples,), optional
        The target values. If provided, a supervised ECA model is created.
        If None, an unsupervised UECA model is created for clustering.
    num_clusters : int, optional
        Number of clusters/classes. If None and y is provided, it is determined
        from y. If None and y is None, an error is raised.
    learning_rate : float, default=0.01
        Learning rate for optimizer.
    num_epochs : int, optional
        Number of training epochs. If None, defaults to 10000 for classification
        and 3000 for clustering.
    random_state : int, optional
        Random seed for reproducibility.
    device : str, optional
        Device to use for computation ('cpu' or 'cuda').
        
    Returns
    -------
    model : ECA or UECA
        The fitted model.
    """
    if y is not None:
        # Supervised mode (classification)
        # Default to 10000 epochs for classification
        if num_epochs is None:
            num_epochs = 10000
            
        model = ECA(
            num_clusters=num_clusters,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            random_state=random_state,
            device=device
        )
    else:
        # Unsupervised mode (clustering)
        # Default to 3000 epochs for clustering
        if num_epochs is None:
            num_epochs = 3000
            
        model = UECA(
            num_clusters=num_clusters,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            random_state=random_state,
            device=device
        )
    
    # Fit the model and return
    model.fit(X, y)
    return model

__all__ = ['ECA', 'UECA', 'eigencomponent_analysis']
