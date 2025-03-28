import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec
from sklearn.metrics import adjusted_rand_score


def visualize_clustering_results(X, y_true, remapped_predictions, loss_history, psi_final, num_epochs, 
                               model, L_matrix, L_soft, P_matrix, dataset_name="Dataset", 
                               feature_names=None, class_names=None, output_dir=None,
                               invert_features=None, feature_signs=None, hide_3d_ticks=False):
    """
    Visualize clustering results with:
    - Loss curve on the left
    - 3D projection in the middle (using L_soft as weights for eigenvectors)
    - Heatmap of L matrix on the right
    - Using TCAS-II paper style
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.
    y_true : array-like of shape (n_samples,)
        True class labels or cluster assignments.
    remapped_predictions : array-like of shape (n_samples,)
        Predicted class labels after optimal remapping.
    loss_history : list
        Training loss history.
    psi_final : array-like of shape (n_samples, n_clusters)
        Final projections of the input data.
    num_epochs : int
        Number of training epochs.
    model : torch.nn.Module
        The trained model.
    L_matrix : array-like of shape (n_features, n_clusters)
        Binary mapping matrix.
    L_soft : array-like of shape (n_features, n_clusters)
        Soft mapping matrix.
    P_matrix : array-like of shape (n_features, n_features)
        Transformation matrix P.
    dataset_name : str, default="Dataset"
        Name of the dataset.
    feature_names : list of str, optional
        Names of the features.
    class_names : list of str, optional
        Names of the classes/clusters.
    output_dir : str, optional
        Directory to save the visualizations.
    invert_features : list of int, optional
        List of indices (0-based) for eigenfeatures to invert.
    feature_signs : list of int, optional
        List of signs (1 or -1) for each eigenfeature.
    hide_3d_ticks : bool, default=False
        If True, hide the tick labels on the 3D plot.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The visualization figure.
    """
    # Determine the number of unique classes
    num_classes = len(np.unique(y_true))
    
    # Set default feature_signs if not provided
    if feature_signs is None:
        feature_signs = [1, 1, -1] if psi_final.shape[1] >= 3 else [1] * psi_final.shape[1]
    
    # Set default invert_features if not provided
    if invert_features is None:
        invert_features = []
    
    # Define feature names and class names if not provided
    if feature_names is None:
        if dataset_name.lower() == "iris":
            feature_names = ["Eigenfeature 1", "Eigenfeature 2", "Eigenfeature 3", "Eigenfeature 4"]
        else:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    
    if class_names is None:
        if dataset_name.lower() == "iris":
            class_names = ["Setosa", "Versicolor", "Virginica"]
        else:
            class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Calculate ARI score
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
    class_markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']  # Various markers
    # Extend if needed
    while len(class_markers) < num_classes:
        class_markers.extend(class_markers)
    class_markers = class_markers[:num_classes]
    
    # Define colors for correct predictions - IEEE-friendly colors with better contrast
    correct_colors = ['#1b9e77', '#7570b3', '#d95f02', '#e7298a', '#66a61e', 
                      '#e6ab02', '#a6761d', '#666666', '#1f78b4', '#b2df8a']
    # Extend if needed
    while len(correct_colors) < num_classes:
        correct_colors.extend(correct_colors)
    correct_colors = correct_colors[:num_classes]
    
    # Define color for incorrect predictions - less saturated red for better print quality
    incorrect_color = '#e41a1c'  # crimson red for all incorrect predictions
    
    # We'll use the psi_final which is already the projection of X onto P @ L
    projected_data = psi_final.copy()
    
    # Unit normalize the projection
    norms = np.linalg.norm(projected_data, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    projected_data = projected_data / norms
    
    # Apply feature signs (invert specified eigenfeatures)
    for i, sign in enumerate(feature_signs):
        if i < projected_data.shape[1]:  # Make sure we don't exceed dimensions
            projected_data[:, i] *= sign
    
    # 3D visualization (middle)
    # If we have at least 3 dimensions, use 3D plot
    if projected_data.shape[1] >= 3:
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
                    projected_data[correct_idx, 2],
                    marker=class_markers[true_class], 
                    color=correct_colors[true_class], 
                    s=50,  # Size for better visibility
                    alpha=0.3,  # Slightly transparent
                    linewidth=1.2,  # Thicker edge
                    label=f'{class_names[true_class]} (Correct)'
                )
            
            # Plot incorrect predictions in red
            if len(incorrect_idx) > 0:
                ax_cluster.scatter(
                    projected_data[incorrect_idx, 0], 
                    projected_data[incorrect_idx, 1], 
                    projected_data[incorrect_idx, 2],
                    marker=class_markers[true_class], 
                    color=incorrect_color, 
                    s=50,  # Size for better visibility
                    alpha=1,  # Fully opaque
                    edgecolor='black',  # Add edge
                    linewidth=1.2,  # Thicker edge
                    label=f'{class_names[true_class]} (Incorrect)'
                )
        
        ax_cluster.set_xlabel('Eigenfeature 1', fontsize=14)
        ax_cluster.set_ylabel('Eigenfeature 2', fontsize=14)
        ax_cluster.set_zlabel('Eigenfeature 3', fontsize=14)
        ax_cluster.view_init(30, 45)  # Set viewing angle
        
    else:
        # 2D visualization if we have fewer than 3 dimensions
        ax_cluster = fig.add_subplot(gs[0, 1])
        
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
                    projected_data[correct_idx, 1] if projected_data.shape[1] > 1 else np.zeros(len(correct_idx)),
                    marker=class_markers[true_class], 
                    color=correct_colors[true_class], 
                    s=50,  # Size for better visibility
                    alpha=0.3,  # Slightly transparent
                    linewidth=1.2,  # Thicker edge
                    label=f'{class_names[true_class]} (Correct)'
                )
            
            # Plot incorrect predictions in red
            if len(incorrect_idx) > 0:
                ax_cluster.scatter(
                    projected_data[incorrect_idx, 0], 
                    projected_data[incorrect_idx, 1] if projected_data.shape[1] > 1 else np.zeros(len(incorrect_idx)),
                    marker=class_markers[true_class], 
                    color=incorrect_color, 
                    s=50,  # Size for better visibility
                    alpha=1,  # Fully opaque
                    edgecolor='black',  # Add edge
                    linewidth=1.2,  # Thicker edge
                    label=f'{class_names[true_class]} (Incorrect)'
                )
        
        ax_cluster.set_xlabel('Eigenfeature 1', fontsize=14)
        if projected_data.shape[1] > 1:
            ax_cluster.set_ylabel('Eigenfeature 2', fontsize=14)
        else:
            ax_cluster.set_ylabel('Value', fontsize=14)
    
    # Common formatting for both 2D and 3D
    if hasattr(ax_cluster, 'grid'):
        ax_cluster.grid(False)
    if hasattr(ax_cluster, '_axis3don'):
        ax_cluster._axis3don = False
        ax_cluster.set_axis_on()
    
    # Set tick parameters
    ax_cluster.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    
    # Set background color
    background_color = '#f2f2f2'
    ax_cluster.set_facecolor(background_color)
    
    # Make tick labels "invisible" if requested (3D only)
    if hide_3d_ticks and hasattr(ax_cluster, 'tick_params') and projected_data.shape[1] >= 3:
        ax_cluster.tick_params(axis='x', colors=background_color)
        ax_cluster.tick_params(axis='y', colors=background_color) 
        ax_cluster.tick_params(axis='z', colors=background_color)
    
    # Set 3D pane properties if available
    if hasattr(ax_cluster, 'xaxis') and hasattr(ax_cluster.xaxis, 'pane'):
        ax_cluster.xaxis.pane.fill = False
        ax_cluster.yaxis.pane.fill = False
        ax_cluster.zaxis.pane.fill = False
        ax_cluster.xaxis.pane.set_edgecolor('darkgray')
        ax_cluster.yaxis.pane.set_edgecolor('darkgray')
        ax_cluster.zaxis.pane.set_edgecolor('darkgray')
        ax_cluster.xaxis.pane.set_linewidth(1.5)
        ax_cluster.yaxis.pane.set_linewidth(1.5)
        ax_cluster.zaxis.pane.set_linewidth(1.5)
    
    # Add ARI score in the legend
    handles, labels = ax_cluster.get_legend_handles_labels()
    
    # Reorganize labels to put correct predictions first, incorrect at the end
    correct_labels = [label for label in labels if "Correct" in label]
    incorrect_labels = [label for label in labels if "Incorrect" in label]
    sorted_labels = []
    
    # Add only unique labels
    for label in correct_labels:
        if label not in sorted_labels:
            sorted_labels.append(label)
    for label in incorrect_labels:
        if label not in sorted_labels:
            sorted_labels.append(label)
    
    # Get corresponding handles in the same order
    sorted_handles = []
    for label in sorted_labels:
        if label in labels:
            sorted_handles.append(handles[labels.index(label)])
    
    # Create a legend with ARI score in title
    legend = ax_cluster.legend(sorted_handles, sorted_labels, loc='upper center', fontsize=10, 
               framealpha=0.9, edgecolor='gray', fancybox=False, title=f"ARI Score: {ari_score:.4f}")
    
    # Adjust legend title properties
    plt.setp(legend.get_title(), fontsize=12, fontweight='bold')
    
    # L Matrix Heatmap (right)
    ax_heatmap = fig.add_subplot(gs[0, 2])
    
    # Create the heatmap for L matrix
    im = ax_heatmap.imshow(L_soft, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Add text annotations with values based on the L matrix values
    for i in range(L_soft.shape[0]):
        for j in range(L_soft.shape[1]):
            # Determine text color based on the cell value for better contrast
            text_color = 'white' if L_soft[i, j] > 0.5 else 'black'
            # Display the value
            text_value = f"{L_soft[i, j]:.2f}"
            ax_heatmap.text(j, i, text_value, ha="center", va="center", color=text_color, fontsize=10)
    
    # Set tick labels using feature names and class names
    num_features_to_show = min(10, L_soft.shape[0])  # Limit to avoid overcrowding
    feature_indices = np.linspace(0, L_soft.shape[0]-1, num_features_to_show, dtype=int)
    
    ax_heatmap.set_xticks(np.arange(len(class_names)))
    ax_heatmap.set_yticks(feature_indices)
    ax_heatmap.set_xticklabels(class_names, rotation=30, ha="right", rotation_mode="anchor")
    
    # Use abbreviated feature names for y-axis if too many features
    if L_soft.shape[0] > num_features_to_show:
        # Use a subset of feature names
        feature_names_to_show = [feature_names[i] if i < len(feature_names) else f"Feature {i}" 
                              for i in feature_indices]
        ax_heatmap.set_yticklabels(feature_names_to_show)
    else:
        # Use all feature names if we have them
        features_to_show = feature_names[:L_soft.shape[0]] if feature_names else [f"Feature {i}" for i in range(L_soft.shape[0])]
        ax_heatmap.set_yticklabels(features_to_show)
    
    # Set labels in TCAS-II style
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
    
    # Save the figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f'eca_{dataset_name.lower()}_visualization')
        
        # Save in multiple formats for compatibility
        plt.savefig(f'{output_filename}.pdf', format='pdf', dpi=600, bbox_inches='tight')
        plt.savefig(f'{output_filename}.png', format='png', dpi=600, bbox_inches='tight')
        plt.savefig(f'{output_filename}.eps', format='eps', dpi=600, bbox_inches='tight')
        
        print(f"\nVisualization saved as '{output_filename}.pdf' and '{output_filename}.png'")
    
    print(f"Clustering performance (ARI Score): {ari_score:.4f}")
    print(f"Number of parameters in ECA model: {model.num_parameters:.0f}")
    
    return fig


def visualize_mnist_eigenfeatures(model, output_dir='mnist_visualizations', font_scale_dist=1.2, font_scale_heatmap=1.0):
    """
    Visualize the eigenfeatures and L matrix of the trained ECA model for MNIST.
    
    Parameters
    ----------
    model : ECAModel or torch.nn.Module
        The trained ECA model.
    output_dir : str, default='mnist_visualizations'
        Directory to save the visualizations.
    font_scale_dist : float, default=1.2
        Font size scaling factor for the distribution figure.
    font_scale_heatmap : float, default=1.0
        Font size scaling factor for L matrix heatmap.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure for the stacked bar chart.
    """
    # Create output directory
    os.makedirs(os.path.expanduser(output_dir), exist_ok=True)
    
    # Extract the L matrix and P matrix (eigenfeatures)
    with torch.no_grad():
        # Create a dummy input to get the matrices
        M = model.A_raw.shape[0]  # Number of features (pixels in MNIST)
        L = model.L_raw.shape[1]  # Number of classes in MNIST (digits 0-9)
        dummy_input = torch.ones(1, M, device=model.A_raw.device)
        _, P_norm, L_matrix, _ = model(dummy_input)
        
        # Convert to numpy for easier handling
        P_np = P_norm.cpu().detach().numpy()
        L_np = L_matrix.cpu().detach().numpy()
    
    # Save the L matrix as a heatmap with light gray background
    plt.rcParams.update({'font.size': 14 * font_scale_heatmap})
    
    plt.figure(figsize=(20, 4), facecolor='#f5f5f5')  # Light gray background
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')  # Light gray plot area
    
    # Use a custom colormap for good contrast
    cmap = plt.cm.Blues  # Blue colormap from light to dark
    
    sns.heatmap(L_np.T, cmap=cmap, annot=False, linewidths=0, cbar=True)
    
    # Set title and labels with black text for readability
    plt.ylabel('Classes (Digits 0-9)', color='black', fontsize=16 * font_scale_heatmap)
    plt.xlabel('Eigenfeatures', color='black', fontsize=16 * font_scale_heatmap)
    
    # Make tick labels black
    plt.tick_params(axis='x', colors='black', labelsize=14 * font_scale_heatmap)
    plt.tick_params(axis='y', colors='black', labelsize=14 * font_scale_heatmap)
    
    # Add thin border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('gray')
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.expanduser(output_dir), 'mnist_L_heatmap.png'), 
                dpi=300, facecolor='#f5f5f5')
    plt.close()
    
    plt.rcdefaults()  # Reset rcParams for subsequent plots
    
    # Create directories for each class with subdirectories for pure and shared eigenfeatures
    for k in range(10):
        class_dir = os.path.join(os.path.expanduser(output_dir), f'class_{k}')
        pure_dir = os.path.join(class_dir, 'pure')
        shared_dir = os.path.join(class_dir, 'shared')
        os.makedirs(pure_dir, exist_ok=True)
        os.makedirs(shared_dir, exist_ok=True)
    
    # Process each eigenfeature
    eigen_count_per_class = [0] * 10  # Track total eigenfeatures per class
    pure_eigen_count_per_class = [0] * 10  # Track pure eigenfeatures per class
    shared_eigen_count_per_class = [0] * 10  # Track shared eigenfeatures per class
    
    for j in range(M):
        # Get the eigenfeature (column j of P)
        eigenfeature = P_np[:, j]
        
        # Reshape to 28x28 for visualization
        eigenfeature_img = eigenfeature.reshape(28, 28)
        
        # Normalize to [0, 1] range for visualization
        min_val = eigenfeature_img.min()
        max_val = eigenfeature_img.max()
        if max_val > min_val:  # Avoid division by zero
            eigenfeature_img = (eigenfeature_img - min_val) / (max_val - min_val)
        
        # Use the normalized image directly
        binary_img = eigenfeature_img
        
        # First, determine if this is a pure or shared eigenfeature
        assigned_classes = [k for k in range(10) if L_np[j, k] >= 0.5]
        is_pure = len(assigned_classes) == 1
        
        # Now save the eigenfeature to the appropriate folders
        for k in assigned_classes:
            # Determine the subdirectory (pure or shared)
            folder_type = 'pure' if is_pure else 'shared'
            
            # Create figure with light gray background
            plt.figure(figsize=(3, 3), facecolor='#f5f5f5')
            ax = plt.gca()
            ax.set_facecolor('#f5f5f5')
            plt.imshow(binary_img, cmap='Blues')  # Use Blues colormap instead of binary
            plt.axis('off')
            plt.tight_layout()
            
            save_path = os.path.join(
                os.path.expanduser(output_dir), 
                f'class_{k}/{folder_type}/eigenfeature_{j}.png'
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='#f5f5f5')
            plt.close()
            
            # Update counters
            eigen_count_per_class[k] += 1
            if is_pure:
                pure_eigen_count_per_class[k] += 1
                print(f"Saved pure eigenfeature {j} to class {k}")
            else:
                shared_eigen_count_per_class[k] += 1
                print(f"Saved shared eigenfeature {j} to class {k} (shared with {len(assigned_classes)-1} other classes)")
    
    # Print summary statistics
    print("\nEigenfeature Distribution Summary:")
    print("================================")
    for k in range(10):
        print(f"Class {k}: {eigen_count_per_class[k]} total eigenfeatures")
        print(f"  - Pure: {pure_eigen_count_per_class[k]}")
        print(f"  - Shared: {shared_eigen_count_per_class[k]}")
    print(f"Total eigenfeatures assigned: {sum(eigen_count_per_class)}")
    
    # ---------------- Distribution Figure with Explicit Font Sizes ----------------
    # Define explicit font sizes based on the scaling factor.
    label_size = 16 * font_scale_dist
    tick_size = 14 * font_scale_dist
    legend_size = 14 * font_scale_dist
    legend_title_size = 14 * font_scale_dist

    # Create the figure and axes explicitly
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='none')
    ax.set_facecolor('none')
    
    x = np.arange(10)
    width = 0.8
    
    # Plot stacked bar chart with more vibrant colors
    ax.bar(x, shared_eigen_count_per_class, width, label='Shared Eigenfeatures', color='#3a86ff')
    ax.bar(x, pure_eigen_count_per_class, width, bottom=shared_eigen_count_per_class, 
           label='Pure Eigenfeatures', color='#ff006e')
    
    # Add labels and styling with explicit font sizes
    ax.set_xlabel('Digit Class', fontsize=label_size, color='black')
    ax.set_ylabel('Number of Eigenfeatures', fontsize=label_size, color='black')
    
    # Set tick labels explicitly
    ax.tick_params(axis='x', colors='black', labelsize=tick_size)
    ax.tick_params(axis='y', colors='black', labelsize=tick_size)
    
    # Add grid for y-axis
    ax.grid(axis='y', linestyle='--', alpha=0.2)
    
    # Create legend with explicit font properties
    leg = ax.legend(facecolor='none', edgecolor='#dddddd', framealpha=0.7, prop={'size': legend_size})
    leg.get_title().set_fontsize(legend_title_size)
    
    # Add value labels on the bars
    for i in range(10):
        # Shared count label (bottom blue section)
        if shared_eigen_count_per_class[i] > 0:
            ax.text(i, shared_eigen_count_per_class[i] / 2, str(shared_eigen_count_per_class[i]), 
                     ha='center', va='center', color='white', fontweight='bold', fontsize=tick_size)
        
        # Pure count label (top pink section)
        if pure_eigen_count_per_class[i] > 0:
            ax.text(i, shared_eigen_count_per_class[i] + pure_eigen_count_per_class[i] / 2, 
                     str(pure_eigen_count_per_class[i]), ha='center', va='center', 
                     color='white', fontweight='bold', fontsize=tick_size)
    
    # Add thin border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#dddddd')
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.expanduser(output_dir), 'eigenfeature_distribution.png'), 
                dpi=200, transparent=True)
    plt.close()
    
    print(f"\nVisualization complete! Results saved to {output_dir}")
    
    return fig
