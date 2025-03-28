import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


def cosine_similarity_matrix(X):
    """Compute the cosine similarity matrix for a batch of vectors.
    
    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features)
        The input data.
        
    Returns
    -------
    S : torch.Tensor of shape (n_samples, n_samples)
        The cosine similarity matrix.
    """
    X_norm = X / X.norm(dim=1, keepdim=True)
    S = torch.mm(X_norm, X_norm.t())
    return S


def find_best_cluster_mapping(y_true, cluster_assignments, num_classes):
    """Find the optimal mapping between cluster assignments and true classes.
    
    Uses the Hungarian algorithm to maximize matching.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    cluster_assignments : array-like of shape (n_samples,)
        Predicted cluster assignments.
    num_classes : int
        Number of classes/clusters.
        
    Returns
    -------
    mapping : dict
        A dictionary mapping from cluster_id to true_class_id.
    """
    # Create confusion matrix
    conf_mat = confusion_matrix(y_true, cluster_assignments, labels=range(num_classes))
    
    # Use the Hungarian algorithm to find the optimal assignment
    # Note: Hungarian algorithm minimizes cost, so we negate the matrix
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    
    # Create the mapping dictionary
    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    
    return mapping


def remap_predictions(predictions, mapping):
    """Remap predictions according to the given mapping.
    
    Parameters
    ----------
    predictions : array-like of shape (n_samples,)
        The original predictions.
    mapping : dict
        A dictionary mapping from original_id to new_id.
        
    Returns
    -------
    remapped : ndarray of shape (n_samples,)
        The remapped predictions.
    """
    return np.array([mapping.get(pred, pred) for pred in predictions])


def normalize_data(X):
    """Normalize data to have unit norm along the sample axis.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.
        
    Returns
    -------
    X_norm : ndarray of shape (n_samples, n_features)
        The normalized data.
    """
    if isinstance(X, torch.Tensor):
        norms = torch.norm(X, dim=1, keepdim=True)
        return X / norms
    else:
        X_np = np.asarray(X)
        norms = np.linalg.norm(X_np, axis=1, keepdims=True)
        return X_np / norms


def compute_loss(S_cosine, probs):
    """Compute the loss function based on cosine similarity.
    
    Parameters
    ----------
    S_cosine : torch.Tensor of shape (n_samples, n_samples)
        The cosine similarity matrix.
    probs : torch.Tensor of shape (n_samples, n_clusters)
        The probability distribution for each sample.
        
    Returns
    -------
    loss : torch.Tensor
        The computed loss value.
    """
    # Compute similarity between probabilities
    probs_similarity = torch.mm(probs, probs.t())
    # Loss: (1 - S_cosine) * probs_similarity
    loss_matrix = (1 - S_cosine) * probs_similarity
    loss = loss_matrix.sum()
    return loss
