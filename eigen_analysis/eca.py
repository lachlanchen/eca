import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from .base import BaseEigenAnalysis
from .utils import cosine_similarity_matrix, compute_loss


class ECAModel(nn.Module):
    """Neural network model for Eigencomponent Analysis.
    
    This implements the core ECA architecture with trainable parameters
    for the antisymmetric matrix A and mapping matrix L.
    
    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    num_clusters : int
        Number of clusters/classes.
    temp : float, default=10.0
        Temperature parameter for sigmoid.
    """
    
    def __init__(self, input_dim, num_clusters, temp=10.0):
        super(ECAModel, self).__init__()
        # A_raw is a trainable parameter of size [input_dim, input_dim]
        self.A_raw = nn.Parameter(torch.zeros(input_dim, input_dim))
        # D is a trainable diagonal matrix
        self.D = nn.Parameter(torch.zeros(input_dim))
        # L_raw is a trainable parameter of size [input_dim, num_clusters]
        self.L_raw = nn.Parameter(torch.zeros(input_dim, num_clusters))

        self.num_parameters = input_dim * (input_dim + 1) / 2 + input_dim + input_dim * num_clusters
        self.temp = temp
    
    @property
    def P(self):
        """Compute the transformation matrix P.
        
        Returns
        -------
        P_norm : torch.Tensor
            Normalized transformation matrix P.
        """
        # Compute antisymmetric part
        A_skew = self.A_raw - self.A_raw.t()
        # Add diagonal matrix D for feature scaling
        self.A = A_skew + torch.diag(self.D)
        # Compute transformation P
        P = torch.matrix_exp(self.A)
        # Normalize columns of P to have unit norm
        # Normaization removed to keep the feature-specific scaling
        P_norm = P  # / torch.norm(P, dim=0, keepdim=True).detach()
        
        return P_norm
    
    @property
    def L(self):
        """Compute the mapping matrix L.
        
        Returns
        -------
        L : torch.Tensor
            Binarized mapping matrix L.
        """
        # Compute L using sigmoid
        L_soft = torch.sigmoid(self.temp * self.L_raw)
        # Apply STE to binarize L
        L_hard = (L_soft >= 0.5).float()
        L = (L_hard - L_soft).detach() + L_soft
        
        return L
        
    def forward(self, X):
        """Forward pass through the model.
        
        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, input_dim)
            Input data.
            
        Returns
        -------
        class_scores : torch.Tensor of shape (n_samples, num_clusters)
            Class scores for each sample.
        P_norm : torch.Tensor of shape (input_dim, input_dim)
            Normalized transformation matrix P.
        L : torch.Tensor of shape (input_dim, num_clusters)
            Binarized mapping matrix L.
        A : torch.Tensor of shape (input_dim, input_dim)
            Antisymmetric matrix A.
        """
        
        P = self.P
        L = self.L

        # Transform input (already unit normalized)
        psi = X @ P  # Shape: [n_samples, input_dim]
        prob = psi ** 2  # Element-wise square
        # Compute class scores
        class_scores = prob @ L  # Shape: [n_samples, num_clusters]
        return class_scores, P, L, self.A

class ECA(BaseEigenAnalysis):
    """Eigencomponent Analysis for classification and clustering.
    
    Parameters
    ----------
    num_clusters : int, optional
        Number of clusters/classes. If None, determined from the target values
        during fit.
    learning_rate : float, default=0.01
        Learning rate for optimizer.
    num_epochs : int, default=10000
        Number of training epochs.
    temp : float, default=10.0
        Temperature parameter for sigmoid.
    random_state : int, optional
        Random seed for reproducibility.
    device : str, optional
        Device to use for computation ('cpu' or 'cuda').
    """
    
    def __init__(self, num_clusters=None, learning_rate=0.01, num_epochs=1000, 
                 temp=10.0, random_state=None, device=None):
        super().__init__(
            num_clusters=num_clusters,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            random_state=random_state,
            device=device
        )
        self.temp = temp
        self.is_supervised_ = False
        
    def fit(self, X, y=None):
        """Fit the ECA model.
        
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
        X_tensor, y_tensor = self._validate_input(X, y)
        self.is_supervised_ = y_tensor is not None
        
        # Unit normalize the input data
        X_tensor = self._normalize_data(X_tensor)
        
        # Store dimensions
        n_samples, n_features = X_tensor.shape
        
        # Initialize model
        self.model_ = ECAModel(
            input_dim=n_features,
            num_clusters=self.num_clusters,
            temp=self.temp
        ).to(self.device)
        
        # Initialize optimizer
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        # Storage for loss history
        self.loss_history_ = []
        
        # Training loop
        self.model_.train()
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            class_scores, P, L, A = self.model_(X_tensor)
            
            # Compute loss based on the scenario
            if self.is_supervised_:
                # For classification: use cross-entropy loss
                loss = nn.CrossEntropyLoss()(class_scores, y_tensor)
            else:
                # For clustering: use the cosine similarity loss from UECA
                # Normalize probabilities
                probs = torch.softmax(class_scores, dim=1)
                # Compute cosine similarity matrix
                S_cosine = cosine_similarity_matrix(X_tensor)
                # Compute loss
                loss = compute_loss(S_cosine, probs)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Record loss
            self.loss_history_.append(loss.item())
            
            # Optional: print progress
            if (epoch + 1) % 500 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
        
        # Extract trained components
        with torch.no_grad():
            # Get final model parameters
            _, self.P_, self.L_, self.A_ = self.model_(X_tensor)
            
            # Convert to numpy for storage
            self.P_numpy_ = self.P_.cpu().numpy()
            self.L_numpy_ = self.L_.cpu().numpy()
            self.A_numpy_ = self.A_.cpu().numpy()
            
            # For supervised learning, compute and store accuracy
            if self.is_supervised_:
                pred_scores, _, _, _ = self.model_(X_tensor)
                _, pred_classes = torch.max(pred_scores, dim=1)
                self.train_accuracy_ = accuracy_score(
                    y_tensor.cpu().numpy(), 
                    pred_classes.cpu().numpy()
                )
                print(f"Training accuracy: {self.train_accuracy_:.4f}")
        
        self.is_fitted_ = True
        return self
    
    def _normalize_data(self, X):
        """Normalize data to have unit norm along the sample axis.
        
        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        X_norm : torch.Tensor of shape (n_samples, n_features)
            The normalized data.
        """
        X_norm = X / torch.norm(X, dim=1, keepdim=True)
        return X_norm
        
    def transform(self, X):
        """Transform X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_clusters)
            Transformed array.
        """
        if not self.is_fitted_:
            raise ValueError("ECA model is not fitted yet. Call 'fit' first.")
            
        # Convert input to tensor
        if isinstance(X, torch.Tensor):
            X_tensor = X.to(self.device)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            
        # Normalize input data
        X_tensor = self._normalize_data(X_tensor)
            
        # Transform through the model
        with torch.no_grad():
            self.model_.eval()
            class_scores, _, _, _ = self.model_(X_tensor)
            
        return class_scores.cpu().numpy()
        
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
        if not self.is_fitted_:
            raise ValueError("ECA model is not fitted yet. Call 'fit' first.")
            
        # Get transformed data
        transformed = self.transform(X)
        
        # Get predicted class/cluster
        return np.argmax(transformed, axis=1)
        
    def predict_proba(self, X):
        """Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_clusters)
            Predicted probability for each class/cluster.
        """
        if not self.is_fitted_:
            raise ValueError("ECA model is not fitted yet. Call 'fit' first.")
            
        # Get transformed data
        transformed = self.transform(X)
        
        # Apply softmax to get probabilities
        exp_scores = np.exp(transformed)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
