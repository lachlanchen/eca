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
    
    # Standardize features
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    
    # Train ECA model for classification
    print("Training ECA model...")
    eca = ECA(num_clusters=3, learning_rate=0.001, num_epochs=10000)
    eca.fit(X, y)
    
    # Evaluate on training data
    y_pred = eca.predict(X)
    training_accuracy = accuracy_score(y, y_pred)
    print(f"Training accuracy: {training_accuracy:.4f}")
    
    # Generate some test data
    np.random.seed(23)
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
    
    # Standardize features
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    
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
