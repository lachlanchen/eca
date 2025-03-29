import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import argparse

# Set random seeds for reproducibility
import random
random.seed(23)
np.random.seed(23)
torch.manual_seed(23)

# Function to compute cosine similarity matrix
def cosine_similarity_matrix(X):
    X_norm = X / X.norm(dim=1, keepdim=True)
    S = torch.mm(X_norm, X_norm.t())
    return S

# Model definition with standard algorithm
class UnsupervisedEigenComponentAnalysis(nn.Module):
    def __init__(self, input_dim, num_clusters):
        super(UnsupervisedEigenComponentAnalysis, self).__init__()
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        # Initialize A_raw and D for antisymmetric matrix A
        self.A_raw = nn.Parameter(torch.randn(input_dim, input_dim))
        self.D = nn.Parameter(torch.ones(input_dim))
        # Initialize mapping matrix L_raw
        self.L_raw = nn.Parameter(torch.randn(input_dim, num_clusters))
        # Calculate number of trainable parameters
        self.num_parameters = sum(p.numel() for p in self.parameters())
        
    @property
    def P(self):
        # Construct antisymmetric matrix A
        A = self.A_raw - self.A_raw.t() + torch.diag(self.D)
        # Compute P = e^A
        P = torch.matrix_exp(A)
        # Unit normalize P with denominator gradient detached
        P_norm = P  # / P.norm(dim=0, keepdim=True).detach()
        return P_norm
    
    @property
    def L(self):
        # Apply sigmoid to L_raw
        L_soft = torch.sigmoid(1*self.L_raw)
        # Binarize L_soft using threshold 0.5
        L_hard = (L_soft > 0.5).float()
        # Use straight-through estimator
        L_ste = (L_hard - L_soft).detach() + L_soft
        # Unit normalize L along M-dimension (input_dim)
        # L_norm = L_ste
        L_norm = L_soft
        return L_norm
        
    def forward(self, X):
        # Unit normalize input data
        X_norm = X  # / X.norm(dim=1, keepdim=True)
        # Get P and L
        P = self.P
        L = self.L
        # Compute projection
        proj = X_norm @ (P @ L)
        # prob = proj ** 2
        # prob = torch.abs(proj)
        prob = proj
        # Apply softmax to get probabilities
        probs = torch.softmax(prob, dim=1)
        return probs, prob, P, proj

# Loss function based on cosine similarity
def compute_loss(S_cosine, probs):
    # Compute similarity between probabilities
    probs_similarity = torch.mm(probs, probs.t())
    # Loss: (1 - S_cosine) * probs_similarity
    loss_matrix = (1 - S_cosine) * probs_similarity
    loss = loss_matrix.sum()
    return loss

# Find optimal mapping between clusters and true classes
def find_best_cluster_mapping(y_true, cluster_assignments, num_classes):
    """
    Find the optimal mapping between cluster assignments and true classes
    using the Hungarian algorithm to maximize matching.
    Returns a dictionary mapping from cluster_id to true_class_id
    """
    # Create confusion matrix
    conf_mat = confusion_matrix(y_true, cluster_assignments)
    
    # Use the Hungarian algorithm to find the optimal assignment
    # Note: Hungarian algorithm minimizes cost, so we negate the matrix
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    
    # Create the mapping dictionary
    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    
    return mapping

# Training function with loss history tracking
def train_model(X, num_clusters, true_labels, num_epochs=3000, learning_rate=0.01):
    input_dim = X.shape[1]
    model = UnsupervisedEigenComponentAnalysis(input_dim, num_clusters)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    X_tensor = torch.from_numpy(X).float()
    
    # Compute cosine similarity matrix S
    S_cosine = cosine_similarity_matrix(X_tensor)
    
    # Track loss history
    loss_history = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        probs, prob, P, proj = model(X_tensor)
        loss = compute_loss(S_cosine, probs)
        loss.backward()
        optimizer.step()
        
        # Record loss
        loss_history.append(loss.item())
        
        # Optional: print progress every 500 epochs
        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Get final probabilities
    with torch.no_grad():
        final_probs, final_prob, P_final, final_proj = model(X_tensor)
        # Get the L matrix (binary form for visualization)
        L_final = model.L
        L_hard = (L_final > 0.5).float()  # Binary version for visualization
    
    # After training, assign labels
    _, predicted_labels = torch.max(final_probs, dim=1)
    predicted_labels = predicted_labels.numpy()
    
    # Find the best mapping between clusters and true classes
    cluster_to_class_mapping = find_best_cluster_mapping(true_labels, predicted_labels, num_clusters)
    
    # Remap the predicted labels according to our mapping
    remapped_predictions = np.array([cluster_to_class_mapping[cluster] for cluster in predicted_labels])
    
    # Convert proj to numpy for visualization
    final_proj_np = final_proj.numpy()
    
    # Calculate ARI score
    ari = adjusted_rand_score(true_labels, predicted_labels)
    
    # return remapped_predictions, ari, model, loss_history, final_proj_np, cluster_to_class_mapping, L_hard.numpy()
    return remapped_predictions, ari, model, loss_history, final_proj_np, cluster_to_class_mapping, L_hard.numpy(), L_final.numpy(), P_final.numpy()

# Enhanced visualization function with optimized 3D visualization and reordered plots
def visualize_clustering_results(X, y_true, remapped_predictions, loss_history, psi_final, num_epochs, 
                               model, L_matrix, L_soft, P_matrix, dataset_name="Dataset", 
                               invert_features=[0], feature_signs=[1, 1, -1], hide_3d_ticks=False):
    """
    Visualize clustering results with:
    - Loss curve on the left
    - 3D projection in the middle (using L_soft as weights for eigenvectors)
    - Heatmap of L matrix on the right
    - Using TCAS-II paper style
    
    Parameters:
    - invert_features: list of indices (0-based) for eigenfeatures to invert
    - feature_signs: list of signs (1 or -1) for each eigenfeature
    - hide_3d_ticks: if True, hide the tick labels on the 3D plot
    """
    # Determine the number of unique classes
    num_classes = len(np.unique(y_true))
    
    # Define feature names and class names for Iris dataset
    if dataset_name.lower() == "iris":
        # feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        feature_names = ["Eigenfeature 1", "Eigenfeature 2", "Eigenfeature 3", "Eigenfeature 4"]
        class_names = ["Setosa", "Versicolor", "Virginica"]
    else:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Calculate ARI score using original predictions
    ari_score = adjusted_rand_score(y_true, remapped_predictions)
    
    # Create a high-quality figure with subplots - IEEE TCAS-II style
    plt.rcParams.update({
        # Use a system-safe font to avoid warnings
        'font.family': 'DejaVu Sans',
        'mathtext.fontset': 'dejavuserif',
        'font.size': 14,
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'lines.linewidth': 2.0,
        'axes.linewidth': 1.5
    })
    
    # Modified GridSpec to reorder the plots
    fig = plt.figure(figsize=(16, 6))  # Wider to accommodate all plots
    gs = GridSpec(1, 3, width_ratios=[1, 1.5, 1], figure=fig)
    
    # Plot loss curve with IEEE style (left)
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(range(1, num_epochs+1), loss_history, color='#1f77b4', linewidth=2.0)
    # ax_loss.set_title('(a) Training Loss', fontsize=15, loc='left', fontweight='bold')
    ax_loss.set_xlabel('Epoch', fontsize=14)
    ax_loss.set_ylabel('Loss Value', fontsize=14)
    ax_loss.set_xscale('log')
    
    # TCAS-II style: clean borders
    ax_loss.spines['top'].set_visible(False)
    ax_loss.spines['right'].set_visible(False)
    ax_loss.spines['bottom'].set_linewidth(1.5)
    ax_loss.spines['left'].set_linewidth(1.5)
    
    # Set tick parameters for better visibility
    ax_loss.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    
    # Set background color to light gray for better contrast in print
    ax_loss.set_facecolor('#f8f8f8')
    
    # Add labels for key points in the loss curve
    min_loss_epoch = np.argmin(loss_history) + 1
    min_loss_value = loss_history[min_loss_epoch-1]
    ax_loss.scatter(min_loss_epoch, min_loss_value, color='red', s=80, zorder=5)
    ax_loss.annotate(f'Min Loss: {min_loss_value:.4f}',
                xy=(min_loss_epoch, min_loss_value),
                xytext=(min_loss_epoch*1.5, min_loss_value*0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=10),
                fontsize=12, fontweight='bold')
    
    # Define the marker shapes for each true class
    class_markers = ['o', '^', 's']  # circle, triangle, square
    
    # Define colors for correct predictions - IEEE-friendly colors with better contrast
    correct_colors = ['#1b9e77', '#7570b3', '#d95f02']  # Dark teal for circles, purple for triangles, orange for squares
    
    # Define color for incorrect predictions - less saturated red for better print quality
    incorrect_color = '#e41a1c'  # crimson red for all incorrect predictions
    
    # We'll use the psi_final which is already the projection of X onto P @ L
    projected_data = psi_final.copy()
    
    # Unit normalize the projection
    norms = np.linalg.norm(projected_data, axis=1, keepdims=True)
    projected_data = projected_data / norms
    
    # Apply feature signs (invert specified eigenfeatures)
    for i, sign in enumerate(feature_signs):
        if i < projected_data.shape[1]:  # Make sure we don't exceed dimensions
            projected_data[:, i] *= sign
    
    # 3D visualization (middle)
    ax_cluster = fig.add_subplot(gs[0, 1], projection='3d')
    
    # Iterate through each true class
    for true_class in range(num_classes):
        # Get indices where true class is this class
        true_class_idx = np.where(y_true == true_class)[0]
        
        # For each true class, separate correct and incorrect predictions
        correct_idx = true_class_idx[remapped_predictions[true_class_idx] == true_class]
        incorrect_idx = true_class_idx[remapped_predictions[true_class_idx] != true_class]
        
        # Plot correct predictions with the appropriate color
        if len(correct_idx) > 0:
            ax_cluster.scatter(
                projected_data[correct_idx, 0], 
                projected_data[correct_idx, 1], 
                projected_data[correct_idx, 2] if projected_data.shape[1] > 2 else np.zeros(len(correct_idx)),
                marker=class_markers[true_class], 
                color=correct_colors[true_class], 
                s=50,  # Further increased size for better visibility
                # alpha=0.7,  # Slightly transparent
                alpha=0.3,  # Slightly transparent
                # edgecolor='black',  # Add edge
                linewidth=1.2,  # Thicker edge
                label=f'{class_names[true_class]} (Correct)'
            )
        
        # Plot incorrect predictions in red
        if len(incorrect_idx) > 0:
            ax_cluster.scatter(
                projected_data[incorrect_idx, 0], 
                projected_data[incorrect_idx, 1], 
                projected_data[incorrect_idx, 2] if projected_data.shape[1] > 2 else np.zeros(len(incorrect_idx)),
                marker=class_markers[true_class], 
                color=incorrect_color, 
                s=50,  # Further increased size for better visibility
                # alpha=0.7,  # Slightly transparent
                alpha=1,  # Slightly transparent
                edgecolor='black',  # Add edge
                linewidth=1.2,  # Thicker edge
                label=f'{class_names[true_class]} (Incorrect)'
            )
    
    # ax_cluster.set_title('(b) ECA Clustering Results', fontsize=15, loc='left', fontweight='bold')
    ax_cluster.set_xlabel('Eigenfeature 1', fontsize=14)
    ax_cluster.set_ylabel('Eigenfeature 2', fontsize=14)
    ax_cluster.set_zlabel('Eigenfeature 3', fontsize=14)
    ax_cluster.view_init(30, 45)  # Set viewing angle
    
    # Explicitly remove all grids while setting subtle background
    ax_cluster.grid(False)
    # Also need to hide all grid lines in 3D
    ax_cluster._axis3don = False
    # But keep the axes and labels visible
    ax_cluster.set_axis_on()
    
    # Set tick parameters for better visibility
    ax_cluster.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    
    # Set subtle light grey background for better contrast
    background_color = '#f2f2f2'
    ax_cluster.set_facecolor(background_color)
    
    # Make tick labels "invisible" by setting their color to match the background if requested
    if hide_3d_ticks:
        ax_cluster.tick_params(axis='x', colors=background_color)
        ax_cluster.tick_params(axis='y', colors=background_color) 
        ax_cluster.tick_params(axis='z', colors=background_color)
    ax_cluster.set_facecolor('#f2f2f2')
    
    # Set pane properties (the "walls" of the 3D box)
    ax_cluster.xaxis.pane.fill = False
    ax_cluster.yaxis.pane.fill = False
    ax_cluster.zaxis.pane.fill = False
    ax_cluster.xaxis.pane.set_edgecolor('darkgray')
    ax_cluster.yaxis.pane.set_edgecolor('darkgray')
    ax_cluster.zaxis.pane.set_edgecolor('darkgray')
    
    # Thicker pane edges
    ax_cluster.xaxis.pane.set_linewidth(1.5)
    ax_cluster.yaxis.pane.set_linewidth(1.5)
    ax_cluster.zaxis.pane.set_linewidth(1.5)
    
    # Add ARI score in the legend as title
    handles, labels = ax_cluster.get_legend_handles_labels()
    
    # Reorganize labels to put correct predictions first, incorrect at the end
    correct_labels = [label for label in labels if "Correct" in label]
    incorrect_labels = [label for label in labels if "Incorrect" in label]
    sorted_labels = correct_labels + incorrect_labels
    
    # Get corresponding handles in the same order
    sorted_handles = [handles[labels.index(label)] for label in sorted_labels]
    
    # Create a legend with ARI score in title
    # legend = ax_cluster.legend(sorted_handles, sorted_labels, loc='upper right', fontsize=10, 
    legend = ax_cluster.legend(sorted_handles, sorted_labels, loc='upper center', fontsize=10, 
               framealpha=0.9, edgecolor='gray', fancybox=False, title=f"ARI Score: {ari_score:.4f}")
    
    # Adjust legend title properties
    plt.setp(legend.get_title(), fontsize=12, fontweight='bold')
    
    # L Matrix Heatmap (right)
    ax_heatmap = fig.add_subplot(gs[0, 2])
    
    # Create the heatmap for L matrix
    im = ax_heatmap.imshow(L_soft, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Add text annotations with 0 and 1 based on the L matrix values
    for i in range(L_soft.shape[0]):
        for j in range(L_soft.shape[1]):
            # Determine text color based on the cell value for better contrast
            text_color = 'white' if L_soft[i, j] > 0.5 else 'black'
            # Display 0 or 1 based on thresholded value
            # text_value = '1' if L_matrix[i, j] > 0.5 else '0'
            text_value = f"{L_soft[i, j]:.2f}"
            ax_heatmap.text(j, i, text_value, ha="center", va="center", color=text_color, fontsize=10)
    
    # Set tick labels using Iris feature names and class names
    ax_heatmap.set_xticks(np.arange(len(class_names)))
    ax_heatmap.set_yticks(np.arange(len(feature_names)))
    # ax_heatmap.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
    ax_heatmap.set_xticklabels(class_names, rotation=30, ha="right", rotation_mode="anchor")
    ax_heatmap.set_yticklabels(feature_names)
    
    # Set title and labels in TCAS-II style
    # ax_heatmap.set_title('(c) L Matrix Mapping', fontsize=15, loc='left', fontweight='bold')
    ax_heatmap.set_xlabel('Clusters', fontsize=14)
    ax_heatmap.set_ylabel('Features', fontsize=14)
    
    # TCAS-II style: clean borders
    ax_heatmap.spines['top'].set_visible(False)
    ax_heatmap.spines['right'].set_visible(False)
    ax_heatmap.spines['bottom'].set_linewidth(1.5)
    ax_heatmap.spines['left'].set_linewidth(1.5)
    
    # Set tick parameters for better visibility
    ax_heatmap.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # Save high-resolution figure for publication with IEEE-compliant settings
    output_filename = f'eca_{dataset_name.lower()}_visualization_3d_optimized'
    # PDF with CMYK color space for publication
    plt.savefig(f'{output_filename}.pdf', format='pdf', dpi=600, bbox_inches='tight', 
                transparent=False, metadata={'Creator': 'ECA Visualization'})
    # High-quality PNG for presentations and web
    plt.savefig(f'{output_filename}.png', format='png', dpi=600, bbox_inches='tight', transparent=False)
    # Also save an EPS version which is often required for IEEE
    plt.savefig(f'{output_filename}.eps', format='eps', dpi=600, bbox_inches='tight')
    
    print(f"\nVisualization saved as '{output_filename}.pdf' and '{output_filename}.png'")
    print(f"Clustering performance (ARI Score): {ari_score:.4f}")
    print(f"Number of parameters in ECA model: {model.num_parameters:.0f}")
    
    return fig

# Example usage with Iris dataset
def main():
    from sklearn import datasets
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Unsupervised Eigencomponent Analysis Visualization')
    parser.add_argument('--hide-3d-ticks', action='store_true', 
                        help='Hide tick labels on the 3D cluster visualization')
    args = parser.parse_args()
    
    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y_true = iris.target
    
    # Normalize data
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # Define parameters
    num_clusters = len(np.unique(y_true))
    num_epochs = 3000
    learning_rate = 0.01
    
    print(f"Training ECA model on Iris dataset...")
    print(f"Number of samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_clusters}")
    
    # Train the model and get remapped predictions
    remapped_predictions, ari, model, loss_history, psi_final, cluster_mapping, L_hard, L_soft, P_matrix = train_model(
        X_norm, num_clusters, y_true, num_epochs, learning_rate
    )
    print("Binary L Matrix:")
    print(L_hard)
    print("Soft L Matrix:")
    print(L_soft)
    
    # Print mapping information
    print(f"Training complete. ARI Score: {ari:.4f}")
    print(f"Cluster to Class Mapping: {cluster_mapping}")
    
    # Set eigenfeature signs for better visualization
    # You can adjust these signs to improve the visualization
    feature_signs = [1, 1, -1]  # Invert the 3rd eigenfeature
    
    # Visualize results with optimized 3D visualization
    visualize_clustering_results(
        X_norm, y_true, remapped_predictions, loss_history, psi_final, 
        num_epochs, model, L_hard, L_soft, P_matrix, "Iris", 
        feature_signs=feature_signs,
        hide_3d_ticks=args.hide_3d_ticks
    )
    
    plt.show()

if __name__ == "__main__":
    main()
