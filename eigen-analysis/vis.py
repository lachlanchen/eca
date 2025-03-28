import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import adjusted_rand_score
import os
import seaborn as sns

def visualize_clustering_results(X, y_true, remapped_predictions, loss_history, psi_final, num_epochs, 
                               model, L_matrix, L_soft, P_matrix, dataset_name="Dataset", 
                               feature_signs=None, hide_3d_ticks=False, 
                               save_path=None, show_plot=True):
    """
    Visualize clustering results.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
        
    y_true : array-like of shape (n_samples,)
        True labels.
        
    remapped_predictions : array-like of shape (n_samples,)
        Predicted labels (remapped to match true labels).
        
    loss_history : list
        Training loss history.
        
    psi_final : array-like of shape (n_samples, n_clusters)
        Final projection of data.
        
    num_epochs : int
        Number of training epochs.
        
    model : EigenComponentModel
        Trained model.
        
    L_matrix : array-like
        Binary L matrix.
        
    L_soft : array-like
        Soft L matrix.
        
    P_matrix : array-like
        Transformation matrix P.
        
    dataset_name : str, default="Dataset"
        Name of the dataset.
        
    feature_signs : list, default=None
        Signs for eigenfeatures for visualization.
        
    hide_3d_ticks : bool, default=False
        Whether to hide 3D ticks.
        
    save_path : str, default=None
        Path to save figures. If None, figures are saved in the current directory.
        
    show_plot : bool, default=True
        Whether to display the plot.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    """
    # Determine the number of unique classes
    num_classes = len(np.unique(y_true))
    
    # Default feature signs if not provided
    if feature_signs is None:
        feature_signs = [1] * min(3, X.shape[1])  # Default all positive

    # Define feature names and class names
    if dataset_name.lower() == "iris":
        feature_names = ["Eigenfeature 1", "Eigenfeature 2", "Eigenfeature 3", "Eigenfeature 4"]
        class_names = ["Setosa", "Versicolor", "Virginica"]
    else:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Calculate ARI score
    ari_score = adjusted_rand_score(y_true, remapped_predictions)
    
    # Create a high-quality figure with subplots
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'mathtext.fontset': 'dejavuserif',
        'font.size': 14,
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'lines.linewidth': 2.0,
        'axes.linewidth': 1.5
    })
    
    # Modified GridSpec to reorder the plots
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, width_ratios=[1, 1.5, 1], figure=fig)
    
    # Plot loss curve (left)
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(range(1, num_epochs+1), loss_history, color='#1f77b4', linewidth=2.0)
    ax_loss.set_xlabel('Epoch', fontsize=14)
    ax_loss.set_ylabel('Loss Value', fontsize=14)
    ax_loss.set_xscale('log')
    
    # Style loss plot
    ax_loss.spines['top'].set_visible(False)
    ax_loss.spines['right'].set_visible(False)
    ax_loss.spines['bottom'].set_linewidth(1.5)
    ax_loss.spines['left'].set_linewidth(1.5)
    ax_loss.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax_loss.set_facecolor('#f8f8f8')
    
    # Add labels for min loss
    min_loss_epoch = np.argmin(loss_history) + 1
    min_loss_value = loss_history[min_loss_epoch-1]
    ax_loss.scatter(min_loss_epoch, min_loss_value, color='red', s=80, zorder=5)
    ax_loss.annotate(f'Min Loss: {min_loss_value:.4f}',
                xy=(min_loss_epoch, min_loss_value),
                xytext=(min_loss_epoch*1.5, min_loss_value*0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=10),
                fontsize=12, fontweight='bold')
    
    # Define markers and colors
    class_markers = ['o', '^', 's', 'D', 'p', '*', 'h', 'v', '>', '<']  # More markers for more classes
    correct_colors = ['#1b9e77', '#7570b3', '#d95f02', '#e7298a', '#66a61e', 
                      '#e6ab02', '#a6761d', '#666666', '#377eb8', '#984ea3']
    incorrect_color = '#e41a1c'
    
    # Process projection data
    projected_data = psi_final.copy()
    norms = np.linalg.norm(projected_data, axis=1, keepdims=True)
    projected_data = projected_data / norms
    
    # Apply feature signs
    for i, sign in enumerate(feature_signs):
        if i < projected_data.shape[1]:
            projected_data[:, i] *= sign
    
    # 3D visualization (middle)
    ax_cluster = fig.add_subplot(gs[0, 1], projection='3d')
    
    # Iterate through each true class
    for true_class in range(num_classes):
        # Get indices for true class
        true_class_idx = np.where(y_true == true_class)[0]
        
        # Separate correct and incorrect predictions
        correct_idx = true_class_idx[remapped_predictions[true_class_idx] == true_class]
        incorrect_idx = true_class_idx[remapped_predictions[true_class_idx] != true_class]
        
        # Plot correct predictions
        if len(correct_idx) > 0:
            ax_cluster.scatter(
                projected_data[correct_idx, 0], 
                projected_data[correct_idx, 1], 
                projected_data[correct_idx, 2] if projected_data.shape[1] > 2 else np.zeros(len(correct_idx)),
                marker=class_markers[min(true_class, len(class_markers)-1)], 
                color=correct_colors[min(true_class, len(correct_colors)-1)], 
                s=50, alpha=0.3, linewidth=1.2,
                label=f'{class_names[min(true_class, len(class_names)-1)]} (Correct)'
            )
        
        # Plot incorrect predictions
        if len(incorrect_idx) > 0:
            ax_cluster.scatter(
                projected_data[incorrect_idx, 0], 
                projected_data[incorrect_idx, 1], 
                projected_data[incorrect_idx, 2] if projected_data.shape[1] > 2 else np.zeros(len(incorrect_idx)),
                marker=class_markers[min(true_class, len(class_markers)-1)], 
                color=incorrect_color, 
                s=50, alpha=1, edgecolor='black', linewidth=1.2,
                label=f'{class_names[min(true_class, len(class_names)-1)]} (Incorrect)'
            )
    
    # Style 3D plot
    ax_cluster.set_xlabel('Eigenfeature 1', fontsize=14)
    ax_cluster.set_ylabel('Eigenfeature 2', fontsize=14)
    ax_cluster.set_zlabel('Eigenfeature 3', fontsize=14)
    ax_cluster.view_init(30, 45)
    ax_cluster.grid(False)
    ax_cluster._axis3don = False
    ax_cluster.set_axis_on()
    ax_cluster.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    
    # Set background and pane properties
    background_color = '#f2f2f2'
    ax_cluster.set_facecolor(background_color)
    
    if hide_3d_ticks:
        ax_cluster.tick_params(axis='x', colors=background_color)
        ax_cluster.tick_params(axis='y', colors=background_color) 
        ax_cluster.tick_params(axis='z', colors=background_color)
    
    ax_cluster.xaxis.pane.fill = False
    ax_cluster.yaxis.pane.fill = False
    ax_cluster.zaxis.pane.fill = False
    ax_cluster.xaxis.pane.set_edgecolor('darkgray')
    ax_cluster.yaxis.pane.set_edgecolor('darkgray')
    ax_cluster.zaxis.pane.set_edgecolor('darkgray')
    ax_cluster.xaxis.pane.set_linewidth(1.5)
    ax_cluster.yaxis.pane.set_linewidth(1.5)
    ax_cluster.zaxis.pane.set_linewidth(1.5)
    
    # Create legend with ARI score
    handles, labels = ax_cluster.get_legend_handles_labels()
    
    # Reorganize labels - correct first, then incorrect
    correct_labels = [label for label in labels if "Correct" in label]
    incorrect_labels = [label for label in labels if "Incorrect" in label]
    sorted_labels = []
    for label in correct_labels:
        if label not in sorted_labels:
            sorted_labels.append(label)
    for label in incorrect_labels:
        if label not in sorted_labels:
            sorted_labels.append(label)
    
    # Get corresponding handles
    sorted_handles = [handles[labels.index(label)] for label in sorted_labels]
    
    # Create a legend with ARI score
    legend = ax_cluster.legend(sorted_handles, sorted_labels, loc='upper center', fontsize=10, 
               framealpha=0.9, edgecolor='gray', fancybox=False, title=f"ARI Score: {ari_score:.4f}")
    
    # Adjust legend title
    plt.setp(legend.get_title(), fontsize=12, fontweight='bold')
    
    # L Matrix Heatmap (right)
    ax_heatmap = fig.add_subplot(gs[0, 2])
    
    # Create the heatmap for L matrix
    im = ax_heatmap.imshow(L_soft, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(L_soft.shape[0]):
        for j in range(L_soft.shape[1]):
            text_color = 'white' if L_soft[i, j] > 0.5 else 'black'
            text_value = f"{L_soft[i, j]:.2f}"
            ax_heatmap.text(j, i, text_value, ha="center", va="center", color=text_color, fontsize=10)
    
    # Set tick labels
    ax_heatmap.set_xticks(np.arange(min(num_classes, L_soft.shape[1])))
    ax_heatmap.set_yticks(np.arange(min(len(feature_names), L_soft.shape[0])))
    ax_heatmap.set_xticklabels(class_names[:min(num_classes, L_soft.shape[1])], rotation=30, ha="right", rotation_mode="anchor")
    ax_heatmap.set_yticklabels(feature_names[:min(len(feature_names), L_soft.shape[0])])
    
    # Style heatmap
    ax_heatmap.set_xlabel('Clusters', fontsize=14)
    ax_heatmap.set_ylabel('Features', fontsize=14)
    ax_heatmap.spines['top'].set_visible(False)
    ax_heatmap.spines['right'].set_visible(False)
    ax_heatmap.spines['bottom'].set_linewidth(1.5)
    ax_heatmap.spines['left'].set_linewidth(1.5)
    ax_heatmap.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # Save figure if requested
    if save_path is None:
        save_path = os.getcwd()
    else:
        os.makedirs(save_path, exist_ok=True)
    
    output_filename = os.path.join(save_path, f'eca_{dataset_name.lower()}_visualization')
    
    plt.savefig(f'{output_filename}.png', format='png', dpi=300, bbox_inches='tight', transparent=False)
    
    print(f"\nVisualization saved as '{output_filename}.png'")
    print(f"Clustering performance (ARI Score): {ari_score:.4f}")
    print(f"Number of parameters in ECA model: {model.num_parameters:.0f}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def visualize_mnist_eigenfeatures(model, output_dir=None):
    """
    Visualize the eigenfeatures and L matrix for MNIST dataset.
    
    Parameters
    ----------
    model : EigenComponentAnalysis
        Trained model.
        
    output_dir : str, default=None
        Directory to save visualizations.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract P and L matrices
    P_np = model.P_matrix_
    L_np = model.L_matrix_
    
    # Save the L matrix as a heatmap
    plt.figure(figsize=(20, 4), facecolor='#f5f5f5')
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')
    
    # Use a blue colormap
    cmap = plt.cm.Blues
    
    plt.imshow(L_np.T, cmap=cmap, aspect='auto')
    plt.colorbar()
    
    plt.ylabel('Classes', color='black', fontsize=16)
    plt.xlabel('Eigenfeatures', color='black', fontsize=16)
    
    # Style the plot
    plt.tick_params(axis='x', colors='black')
    plt.tick_params(axis='y', colors='black')
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('gray')
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'L_matrix_heatmap.png'), dpi=300, facecolor='#f5f5f5')
    plt.close()
    
    # Only process if we have image data (MNIST-like)
    if P_np.shape[0] in [784, 28*28]:  # MNIST dimensions
        # Create directories for eigenfeatures
        # Loop through classes in the L matrix
        for k in range(L_np.shape[1]):
            class_dir = os.path.join(output_dir, f'class_{k}')
            pure_dir = os.path.join(class_dir, 'pure')
            shared_dir = os.path.join(class_dir, 'shared')
            os.makedirs(pure_dir, exist_ok=True)
            os.makedirs(shared_dir, exist_ok=True)
        
        # Process each eigenfeature
        eigen_count_per_class = [0] * L_np.shape[1]
        pure_eigen_count_per_class = [0] * L_np.shape[1]
        shared_eigen_count_per_class = [0] * L_np.shape[1]
        
        for j in range(P_np.shape[1]):
            # Get the eigenfeature
            eigenfeature = P_np[:, j]
            
            # Reshape to 28x28 for visualization
            eigenfeature_img = eigenfeature.reshape(28, 28)
            
            # Normalize to [0, 1] range
            min_val = eigenfeature_img.min()
            max_val = eigenfeature_img.max()
            if max_val > min_val:
                eigenfeature_img = (eigenfeature_img - min_val) / (max_val - min_val)
            
            # Determine if pure or shared
            assigned_classes = []
            for k in range(L_np.shape[1]):
                if L_np[j, k] >= 0.5:
                    assigned_classes.append(k)
            
            is_pure = len(assigned_classes) == 1
            
            # Save to appropriate folders
            for k in assigned_classes:
                folder_type = 'pure' if is_pure else 'shared'
                
                plt.figure(figsize=(3, 3), facecolor='#f5f5f5')
                ax = plt.gca()
                ax.set_facecolor('#f5f5f5')
                plt.imshow(eigenfeature_img, cmap='Blues')
                plt.axis('off')
                plt.tight_layout()
                
                save_path = os.path.join(
                    output_dir, 
                    f'class_{k}/{folder_type}/eigenfeature_{j}.png'
                )
                plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='#f5f5f5')
                plt.close()
                
                # Update counters
                eigen_count_per_class[k] += 1
                if is_pure:
                    pure_eigen_count_per_class[k] += 1
                else:
                    shared_eigen_count_per_class[k] += 1
        
        # Print summary
        print("\nEigenfeature Distribution Summary:")
        print("================================")
        for k in range(L_np.shape[1]):
            print(f"Class {k}: {eigen_count_per_class[k]} total eigenfeatures")
            print(f"  - Pure: {pure_eigen_count_per_class[k]}")
            print(f"  - Shared: {shared_eigen_count_per_class[k]}")
        
        # Generate bar chart
        plt.figure(figsize=(12, 7), facecolor='none')
        ax = plt.gca()
        ax.set_facecolor('none')
        
        x = np.arange(L_np.shape[1])
        width = 0.8
        
        plt.bar(x, shared_eigen_count_per_class, width, label='Shared Eigenfeatures', color='#3a86ff')
        plt.bar(x, pure_eigen_count_per_class, width, bottom=shared_eigen_count_per_class, 
                label='Pure Eigenfeatures', color='#ff006e')
        
        plt.xlabel('Class', fontsize=16, color='black')
        plt.ylabel('Number of Eigenfeatures', fontsize=16, color='black')
        plt.xticks(range(L_np.shape[1]), color='black')
        plt.yticks(color='black')
        plt.grid(axis='y', linestyle='--', alpha=0.2)
        plt.legend(facecolor='none', edgecolor='#dddddd', framealpha=0.7)
        
        # Add value labels on bars
        for i in range(L_np.shape[1]):
            if shared_eigen_count_per_class[i] > 0:
                plt.text(i, shared_eigen_count_per_class[i]/2, str(shared_eigen_count_per_class[i]), 
                         ha='center', va='center', color='white', fontweight='bold')
            
            if pure_eigen_count_per_class[i] > 0:
                plt.text(i, shared_eigen_count_per_class[i] + pure_eigen_count_per_class[i]/2, 
                         str(pure_eigen_count_per_class[i]), ha='center', va='center', 
                         color='white', fontweight='bold')
        
        # Style plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#dddddd')
            spine.set_linewidth(0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eigenfeature_distribution.png'), 
                    dpi=200, transparent=True)
        plt.close()
        
        print(f"\nVisualization complete! Results saved to {output_dir}")
    else:
        print("Input dimensions don't match MNIST (28x28). Skipping eigenfeature visualization.")
