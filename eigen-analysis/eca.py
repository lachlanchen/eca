import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from .base import BaseEigenAnalysis


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
        Temperature parameter for softmax.
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
        # Compute antisymmetric part
        A_skew = self.A_raw - self.A_raw.t()
        # Add diagonal matrix D
        A = A_skew + torch.diag(self.D)
        # Compute transformation P
        P = torch.matrix_exp(A)
        # Normalize columns of P to have unit norm
        P_norm = P / torch.norm(P, dim=0, keepdim=True).detach()
        # Transform input
        psi = X @ P_norm  # Shape: [n_samples, input_dim]
        psi_sq = psi ** 2  # Element-wise square
        # Compute L using sigmoid
        L = torch.sigmoid(self.temp * self.L_raw)
        # Apply STE to binarize L
        L_hard = (L >= 0.5).float()
        L = (L_hard - L).detach() + L
        # Compute class scores
        class_scores = psi_sq @ L  # Shape: [n_samples, num_clusters]
        return class_scores, P_norm, L, A


class ECA(BaseEigenAnalysis):
    """Eigencomponent Analysis for classification and clustering.
    
    Parameters
    ----------
    num_clusters : int, optional
        Number of clusters/classes. If None, determined from the target values
        during fit.
    learning_rate : float, default=0.01
        Learning rate for optimizer.
    num_epochs : int, default=3000
        Number of training epochs.
    temp : float, default=10.0
        Temperature parameter for sigmoid.
    random_state : int, optional
        Random seed for reproducibility.
    device : str, optional
        Device to use for computation ('cpu' or 'cuda').
    """
    
    def __init__(self, num_clusters=None, learning_rate=0.01, num_epochs=3000, 
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
            class_scores, P_norm, L, A = self.model_(X_tensor)
            
            # Compute loss based on the scenario
            if self.is_supervised_:
                # For classification: use cross-entropy loss
                loss = nn.CrossEntropyLoss()(class_scores, y_tensor)
            else:
                # For clustering: use the cosine similarity loss from UECA
                from .utils import cosine_similarity_matrix, compute_loss
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
        return np.exp(transformed) / np.sum(np.exp(transformed), axis=1, keepdims=True)
        
    def visualize(self, X=None, y=None, feature_names=None, class_names=None, 
                  visualize_eigenfeatures=True, visualize_distribution=True,
                  output_dir=None):
        """Visualize the ECA model results.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            The input data.
        y : array-like of shape (n_samples,), optional
            The target values.
        feature_names : list of str, optional
            Names of the features.
        class_names : list of str, optional
            Names of the classes/clusters.
        visualize_eigenfeatures : bool, default=True
            Whether to visualize eigenfeatures.
        visualize_distribution : bool, default=True
            Whether to visualize eigenfeature distribution.
        output_dir : str, optional
            Directory to save the visualizations.
            
        Returns
        -------
        figs : list of matplotlib.figure.Figure
            The visualization figures.
        """
        if not self.is_fitted_:
            raise ValueError("ECA model is not fitted yet. Call 'fit' first.")
            
        # Import visualization module
        from .visualization import (
            visualize_eigenfeatures, 
            visualize_clustering_results,
            visualize_mnist_eigenfeatures
        )
        import matplotlib.pyplot as plt
        import os
        
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Figures to return
        figs = []
        
        # Determine if this is MNIST based on input dimension
        is_mnist = False
        if hasattr(self.model_, 'A_raw') and self.model_.A_raw.shape[0] == 784:
            is_mnist = True
            
        # If MNIST, use the specialized visualization
        if is_mnist and visualize_eigenfeatures:
            print("Detected MNIST dataset. Using specialized visualization.")
            fig = visualize_mnist_eigenfeatures(
                self.model_, 
                output_dir=output_dir or "mnist_visualizations"
            )
            figs.append(fig)
            
        # Otherwise, use general clustering visualization
        elif X is not None:
            # Convert input to numpy if needed
            if isinstance(X, torch.Tensor):
                X_numpy = X.cpu().numpy()
            else:
                X_numpy = np.asarray(X)
                
            # Get predictions if y is not provided
            if y is None:
                y_pred = self.predict(X)
            else:
                # Convert y to numpy if needed
                if isinstance(y, torch.Tensor):
                    y_numpy = y.cpu().numpy()
                else:
                    y_numpy = np.asarray(y)
                    
                # Predictions are just the true labels for visualization
                y_pred = y_numpy
                
            # Use Unsupervised clustering visualization
            fig = visualize_clustering_results(
                X_numpy, 
                y_numpy if y is not None else y_pred, 
                y_pred, 
                self.loss_history_, 
                self.transform(X),
                self.num_epochs,
                self.model_,
                (self.L_numpy_ > 0.5).astype(float),
                self.L_numpy_,
                self.P_numpy_,
                "Custom Dataset",
                feature_names=feature_names,
                class_names=class_names,
                output_dir=output_dir
            )
            figs.append(fig)
            
        # Visualize loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_epochs+1), self.loss_history_)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.tight_layout()
        loss_fig = plt.gcf()
        figs.append(loss_fig)
        
        if output_dir:
            loss_fig.savefig(os.path.join(output_dir, 'loss_curve.png'))
            
        return figs
