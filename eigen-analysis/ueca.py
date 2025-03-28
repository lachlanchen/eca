import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import adjusted_rand_score

from .base import BaseEigenAnalysis
from .utils import cosine_similarity_matrix, compute_loss, find_best_cluster_mapping, remap_predictions


class UnsupervisedECAModel(nn.Module):
    """Neural network model for Unsupervised Eigencomponent Analysis.
    
    This implements the standard UECA algorithm with trainable parameters
    for the antisymmetric matrix A and mapping matrix L.
    
    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    num_clusters : int
        Number of clusters.
    """
    
    def __init__(self, input_dim, num_clusters):
        super(UnsupervisedECAModel, self).__init__()
        # Initialize A_raw and D for antisymmetric matrix A
        self.A_raw = nn.Parameter(torch.randn(input_dim, input_dim))
        self.D = nn.Parameter(torch.ones(input_dim))
        # Initialize mapping matrix L_raw
        self.L_raw = nn.Parameter(torch.randn(input_dim, num_clusters))
        # Calculate number of trainable parameters
        self.num_parameters = sum(p.numel() for p in self.parameters())
    
    @property
    def P(self):
        """Compute the transformation matrix P.
        
        Returns
        -------
        P_norm : torch.Tensor
            Normalized transformation matrix P.
        """
        # Construct antisymmetric matrix A
        A = self.A_raw - self.A_raw.t() + torch.diag(self.D)
        # Compute P = e^A
        P = torch.matrix_exp(A)
        # Unit normalize P with denominator gradient detached
        P_norm = P  # / P.norm(dim=0, keepdim=True).detach()
        return P_norm
    
    @property
    def L(self):
        """Compute the mapping matrix L.
        
        Returns
        -------
        L_norm : torch.Tensor
            Normalized mapping matrix L.
        """
        # Apply sigmoid to L_raw
        L_soft = torch.sigmoid(1 * self.L_raw)
        # Binarize L_soft using threshold 0.5
        L_hard = (L_soft > 0.5).float()
        # Use straight-through estimator
        L_ste = (L_hard - L_soft).detach() + L_soft
        # Use soft L for training stability
        L_norm = L_soft
        return L_norm
        
    def forward(self, X):
        """Forward pass through the model.
        
        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, input_dim)
            Input data.
            
        Returns
        -------
        probs : torch.Tensor of shape (n_samples, num_clusters)
            Probability distribution for each sample.
        prob : torch.Tensor of shape (n_samples, num_clusters)
            Raw cluster scores.
        P : torch.Tensor of shape (input_dim, input_dim)
            Transformation matrix P.
        proj : torch.Tensor of shape (n_samples, num_clusters)
            Projected data.
        """
        # Get P and L
        P = self.P
        L = self.L
        # Compute projection
        proj = X @ (P @ L)
        # Use raw projection for probabilities
        prob = proj
        # Apply softmax to get probabilities
        probs = torch.softmax(prob, dim=1)
        return probs, prob, P, proj


class UECA(BaseEigenAnalysis):
    """Unsupervised Eigencomponent Analysis for clustering.
    
    Parameters
    ----------
    num_clusters : int
        Number of clusters.
    learning_rate : float, default=0.01
        Learning rate for optimizer.
    num_epochs : int, default=3000
        Number of training epochs.
    random_state : int, optional
        Random seed for reproducibility.
    device : str, optional
        Device to use for computation ('cpu' or 'cuda').
    """
    
    def __init__(self, num_clusters, learning_rate=0.01, num_epochs=3000, 
                 random_state=None, device=None):
        super().__init__(
            num_clusters=num_clusters,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            random_state=random_state,
            device=device
        )
        
    def fit(self, X, y=None):
        """Fit the UECA model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values. If provided, used only for evaluation.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X_tensor, y_tensor = self._validate_input(X, y)
        
        # Store dimensions
        n_samples, n_features = X_tensor.shape
        
        # Initialize model
        self.model_ = UnsupervisedECAModel(
            input_dim=n_features,
            num_clusters=self.num_clusters
        ).to(self.device)
        
        # Initialize optimizer
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        # Compute cosine similarity matrix S
        S_cosine = cosine_similarity_matrix(X_tensor)
        
        # Storage for loss history
        self.loss_history_ = []
        
        # Training loop
        self.model_.train()
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            probs, prob, P, proj = self.model_(X_tensor)
            
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
            # Get final model outputs
            final_probs, final_prob, P_final, final_proj = self.model_(X_tensor)
            
            # Get the L matrix
            L_final = self.model_.L
            L_hard = (L_final > 0.5).float()  # Binary version for visualization
            
            # After training, assign labels
            _, predicted_labels = torch.max(final_probs, dim=1)
            self.labels_ = predicted_labels.cpu().numpy()
            
            # Store model components
            self.P_ = P_final
            self.L_ = L_final
            self.L_hard_ = L_hard
            self.proj_ = final_proj
            
            # Convert to numpy for storage
            self.P_numpy_ = self.P_.cpu().numpy()
            self.L_numpy_ = self.L_.cpu().numpy()
            self.L_hard_numpy_ = self.L_hard_.cpu().numpy()
            self.proj_numpy_ = self.proj_.cpu().numpy()
            
            # If true labels are provided, compute ARI and find mapping
            if y_tensor is not None:
                y_true = y_tensor.cpu().numpy()
                # Calculate ARI score
                self.ari_ = adjusted_rand_score(y_true, self.labels_)
                print(f"Adjusted Rand Index: {self.ari_:.4f}")
                
                # Find the best mapping between clusters and true classes
                self.cluster_mapping_ = find_best_cluster_mapping(
                    y_true, self.labels_, self.num_clusters
                )
                
                # Remap predictions according to mapping
                self.remapped_labels_ = remap_predictions(
                    self.labels_, self.cluster_mapping_
                )
                
                # Calculate accuracy after remapping
                self.accuracy_ = np.mean(self.remapped_labels_ == y_true)
                print(f"Accuracy after mapping: {self.accuracy_:.4f}")
        
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
            raise ValueError("UECA model is not fitted yet. Call 'fit' first.")
            
        # Convert input to tensor
        if isinstance(X, torch.Tensor):
            X_tensor = X.to(self.device)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            
        # Transform through the model
        with torch.no_grad():
            self.model_.eval()
            _, _, _, proj = self.model_(X_tensor)
            
        return proj.cpu().numpy()
        
    def predict(self, X):
        """Predict cluster assignments for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted cluster index for each data sample.
        """
        if not self.is_fitted_:
            raise ValueError("UECA model is not fitted yet. Call 'fit' first.")
            
        # Convert input to tensor
        if isinstance(X, torch.Tensor):
            X_tensor = X.to(self.device)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            
        # Predict through the model
        with torch.no_grad():
            self.model_.eval()
            probs, _, _, _ = self.model_(X_tensor)
            _, predicted_labels = torch.max(probs, dim=1)
            
        return predicted_labels.cpu().numpy()
        
    def predict_proba(self, X):
        """Predict cluster probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_clusters)
            Predicted probability for each cluster.
        """
        if not self.is_fitted_:
            raise ValueError("UECA model is not fitted yet. Call 'fit' first.")
            
        # Convert input to tensor
        if isinstance(X, torch.Tensor):
            X_tensor = X.to(self.device)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            
        # Predict through the model
        with torch.no_grad():
            self.model_.eval()
            probs, _, _, _ = self.model_(X_tensor)
            
        return probs.cpu().numpy()
        
    def visualize(self, X=None, y=None, feature_names=None, class_names=None, output_dir=None):
        """Visualize the UECA clustering results.
        
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
        output_dir : str, optional
            Directory to save the visualizations.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The visualization figure.
        """
        if not self.is_fitted_:
            raise ValueError("UECA model is not fitted yet. Call 'fit' first.")
            
        # Import visualization module
        from .visualization import visualize_clustering_results
        import matplotlib.pyplot as plt
        import os
        
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Use X if provided, otherwise use the training data
        if X is None and hasattr(self, 'X_'):
            X = self.X_
            
        # Convert input to numpy if needed
        if X is not None:
            if isinstance(X, torch.Tensor):
                X_numpy = X.cpu().numpy()
            else:
                X_numpy = np.asarray(X)
                
            # Use y if provided, otherwise use the labels_
            if y is None and hasattr(self, 'labels_'):
                y_numpy = self.labels_
            elif y is not None:
                # Convert y to numpy if needed
                if isinstance(y, torch.Tensor):
                    y_numpy = y.cpu().numpy()
                else:
                    y_numpy = np.asarray(y)
            else:
                # No labels available
                y_numpy = None
                
            # Predict if we have X
            if X is not None:
                y_pred = self.predict(X)
            else:
                y_pred = None
                
            # If we have true labels, use remapped predictions if available
            if y_numpy is not None and hasattr(self, 'remapped_labels_'):
                remapped = self.remapped_labels_
            else:
                remapped = y_pred
                
            # Visualize clustering results
            fig = visualize_clustering_results(
                X_numpy, 
                y_numpy if y_numpy is not None else remapped, 
                remapped if remapped is not None else y_pred, 
                self.loss_history_, 
                self.transform(X) if X is not None else self.proj_numpy_,
                self.num_epochs,
                self.model_,
                self.L_hard_numpy_,
                self.L_numpy_,
                self.P_numpy_,
                "Custom Dataset",
                feature_names=feature_names,
                class_names=class_names,
                output_dir=output_dir
            )
            
            return fig
        else:
            print("No data available for visualization. Please provide X.")
            return None
