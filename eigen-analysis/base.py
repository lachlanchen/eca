import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin


class BaseEigenAnalysis(BaseEstimator, TransformerMixin):
    """Base class for Eigencomponent Analysis models.
    
    This serves as a foundation for both supervised and unsupervised variants.
    """
    
    def __init__(self, num_clusters=None, learning_rate=0.01, num_epochs=3000,
                 random_state=None, device=None):
        self.num_clusters = num_clusters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.random_state = random_state
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted_ = False
        
    def _validate_input(self, X, y=None):
        """Validate input data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,), optional
            The target values.
            
        Returns
        -------
        X_tensor : torch.Tensor
            The validated input data as a torch Tensor.
        y_tensor : torch.Tensor or None
            The validated target values as a torch Tensor, if provided.
        """
        # Convert to numpy if not already
        if isinstance(X, torch.Tensor):
            X_numpy = X.detach().cpu().numpy()
        else:
            X_numpy = np.asarray(X)
            
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            
        # Convert to torch tensor
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32, device=self.device)
        
        if y is not None:
            if isinstance(y, torch.Tensor):
                y_numpy = y.detach().cpu().numpy()
            else:
                y_numpy = np.asarray(y)
                
            y_tensor = torch.tensor(y_numpy, dtype=torch.long, device=self.device)
            
            # If num_clusters is not specified, determine from y
            if self.num_clusters is None:
                self.num_clusters = len(np.unique(y_numpy))
        else:
            y_tensor = None
            
            # Ensure num_clusters is provided if y is not
            if self.num_clusters is None:
                raise ValueError("num_clusters must be specified when y is not provided")
                
        return X_tensor, y_tensor
        
    def fit(self, X, y=None):
        """Fit the model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values for supervised learning.
            
        Returns
        -------
        self : object
            Returns self.
        """
        raise NotImplementedError("Subclasses must implement fit method")
        
    def transform(self, X):
        """Transform X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed array.
        """
        raise NotImplementedError("Subclasses must implement transform method")
        
    def fit_transform(self, X, y=None):
        """Fit the model and transform X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values for supervised learning.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed array.
        """
        return self.fit(X, y).transform(X)
        
    def predict(self, X):
        """Predict class labels or cluster assignments for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class label or cluster index for each data sample.
        """
        raise NotImplementedError("Subclasses must implement predict method")
        
    def visualize(self, X=None, y=None):
        """Visualize the model results.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            The input data.
        y : array-like of shape (n_samples,), optional
            The target values.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The visualization figure.
        """
        raise NotImplementedError("Subclasses must implement visualize method")
