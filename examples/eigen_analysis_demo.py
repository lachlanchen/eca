#!/usr/bin/env python3
"""
Eigen Analysis Demo
------------------

This script demonstrates how to use the eigen_analysis package for
both classification and clustering tasks on the Iris dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import the ECA model - assumes you've installed the package with pip install .
import eigen_analysis


def load_iris_data():
    """Load the Iris dataset and preprocess it."""
    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    # Normalize data
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    return X_norm, y, iris.feature_names, iris.target_names


def eca_classification_demo(X, y, feature_names, target_names):
    """Demo of ECA for classification."""
    print("\n==== ECA Classification Demo ====")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create and fit the ECA model
    print("Training ECA model...")
    eca = eigen_analysis.ECA(num_clusters=len(np.unique(y)), num_epochs=1000, random_state=42)
    eca.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = eca.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Visualize results
    print("Visualizing results...")
    figs = eca.visualize(
        X=X_test,
        y=y_test,
        feature_names=feature_names,
        class_names=target_names,
        output_dir='eca_classification_results'
    )
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Original Features (1-2)")
    
    plt.subplot(2, 2, 2)
    plt.scatter(X_test[:, 2], X_test[:, 3], c=y_test, cmap='viridis')
    plt.xlabel(feature_names[2])
    plt.ylabel(feature_names[3])
    plt.title("Original Features (3-4)")
    
    # Transform data
    X_transformed = eca.transform(X_test)
    
    plt.subplot(2, 2, 3)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_test, cmap='viridis')
    plt.xlabel("ECA Feature 1")
    plt.ylabel("ECA Feature 2")
    plt.title("ECA Transformed Features (1-2)")
    
    if X_transformed.shape[1] > 2:
        plt.subplot(2, 2, 4)
        plt.scatter(X_transformed[:, 0], X_transformed[:, 2], c=y_test, cmap='viridis')
        plt.xlabel("ECA Feature 1")
        plt.ylabel("ECA Feature 3")
        plt.title("ECA Transformed Features (1-3)")
    
    plt.tight_layout()
    plt.savefig('eca_classification_results/transformed_features.png')
    print("Classification demo completed!")
    
    return eca


def eca_clustering_demo(X, y, feature_names, target_names):
    """Demo of ECA for clustering."""
    print("\n==== ECA Clustering Demo ====")
    
    # Create and fit the ECA model for clustering (unsupervised)
    print("Training ECA model for clustering...")
    
    # Using the unified API automatically chooses UECA for clustering
    ueca = eigen_analysis.eigencomponent_analysis(
        X, num_clusters=len(np.unique(y)), num_epochs=1000, random_state=42
    )
    
    # Evaluate cluster quality
    y_pred = ueca.predict(X)
    ari = adjusted_rand_score(y, y_pred)
    print(f"Adjusted Rand Index: {ari:.4f}")
    
    # Visualize results
    print("Visualizing clustering results...")
    fig = ueca.visualize(
        X=X,
        y=y,  # Provide ground truth only for evaluation, not used in training
        feature_names=feature_names,
        class_names=target_names,
        output_dir='eca_clustering_results'
    )
    
    print("Clustering demo completed!")
    
    return ueca


def main():
    """Run both classification and clustering demos."""
    # Load data
    X, y, feature_names, target_names = load_iris_data()
    
    # Run demos
    eca_model = eca_classification_demo(X, y, feature_names, target_names)
    ueca_model = eca_clustering_demo(X, y, feature_names, target_names)
    
    print("\nDemos completed successfully!")
    print("Results saved in the 'eca_classification_results' and 'eca_clustering_results' directories.")


if __name__ == "__main__":
    main()
