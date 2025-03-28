#!/usr/bin/env python3
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch.nn as nn
import argparse

# Define the ECA model (same as in your original code)
class ECAModel(nn.Module):
    def __init__(self, M, L):
        super(ECAModel, self).__init__()
        # A_raw is a trainable parameter of size [M, M]
        self.A_raw = nn.Parameter(torch.zeros(M, M))
        # D is a trainable diagonal matrix
        self.D = nn.Parameter(torch.zeros(M))
        # L_raw is a trainable parameter of size [M, L]
        self.L_raw = nn.Parameter(torch.zeros(M, L))

        self.num_parameters = M * (M + 1) / 2 + M + M * L

        self.temp = 10.0

    def forward(self, X):
        # Compute antisymmetric part
        A_skew = self.A_raw - self.A_raw.t()
        # Add diagonal matrix D
        A = A_skew + torch.diag(self.D)
        # Compute transformation P
        P = torch.matrix_exp(A)
        # Normalize columns of P to have unit norm
        P_norm = P / torch.norm(P, dim=0, keepdim=True).detach()
        # Transform input
        psi = X @ P  # Shape: [N, M]
        psi_sq = psi ** 2  # Element-wise square
        # Compute L using sigmoid
        L = torch.sigmoid(self.temp * self.L_raw)
        # Apply STE to binarize L (if needed)
        L_hard = (L >= 0.5).float()
        L = (L_hard - L).detach() + L
        # Compute class scores
        class_scores = psi_sq @ L  # Shape: [N, L]
        return class_scores, P_norm, L, A

def visualize_mnist_eigenfeatures(font_scale_dist=1.2, font_scale_heatmap=1.0):
    """
    Visualize the eigenfeatures and L matrix of the trained ECA model for MNIST.
    
    Parameters:
    -----------
    font_scale_dist : float
        Font size scaling factor for the distribution figure. Default: 1.2
    font_scale_heatmap : float
        Font size scaling factor for L matrix heatmap. Default: 1.0
    """
    # Model parameters for MNIST
    M = 28 * 28  # Number of features (pixels in MNIST)
    L = 10       # Number of classes in MNIST (digits 0-9)
    
    # Create model instance
    model = ECAModel(M, L)
    
    # Load the best trained model
    model_path = '~/ProjectsLFS/eca/best_model_mnist.pth'
    if not os.path.exists(os.path.expanduser(model_path)):
        print(f"Model file '{model_path}' not found. Please train the model first.")
        return
    
    model.load_state_dict(torch.load(os.path.expanduser(model_path)))
    model.eval()
    
    # Extract the L matrix and P matrix (eigenfeatures)
    with torch.no_grad():
        dummy_input = torch.ones(1, M)
        _, P_norm, L_matrix, _ = model(dummy_input)
        P_np = P_norm.detach().numpy()
        L_np = L_matrix.detach().numpy()
    
    # Create output directory
    output_dir = '~/ProjectsLFS/eca/mnist_visualizations'
    os.makedirs(os.path.expanduser(output_dir), exist_ok=True)
    
    # ---------------- Heatmap for L Matrix ----------------
    # Set font size for heatmap using rcParams (only for this figure)
    plt.rcParams.update({'font.size': 14 * font_scale_heatmap})
    
    plt.figure(figsize=(20, 4), facecolor='#f5f5f5')  # Light gray background
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')
    
    cmap = plt.cm.Blues  # Blue colormap from light to dark
    sns.heatmap(L_np.T, cmap=cmap, annot=False, linewidths=0, cbar=True)
    
    plt.ylabel('Classes (Digits 0-9)', color='black', fontsize=16 * font_scale_heatmap)
    plt.xlabel('Eigenfeatures', color='black', fontsize=16 * font_scale_heatmap)
    plt.tick_params(axis='x', colors='black', labelsize=14 * font_scale_heatmap)
    plt.tick_params(axis='y', colors='black', labelsize=14 * font_scale_heatmap)
    
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=14 * font_scale_heatmap)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('gray')
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.expanduser(output_dir), 'mnist_L_heatmap.png'),
                # dpi=300, facecolor='#f5f5f5', transparent=True)
                dpi=300, transparent=True)
    plt.close()
    
    plt.rcdefaults()  # Reset rcParams for subsequent plots
    
    # ---------------- Save Eigenfeature Images ----------------
    for k in range(10):
        class_dir = os.path.join(os.path.expanduser(output_dir), f'class_{k}')
        pure_dir = os.path.join(class_dir, 'pure')
        shared_dir = os.path.join(class_dir, 'shared')
        os.makedirs(pure_dir, exist_ok=True)
        os.makedirs(shared_dir, exist_ok=True)
    
    eigen_count_per_class = [0] * 10
    pure_eigen_count_per_class = [0] * 10
    shared_eigen_count_per_class = [0] * 10
    
    for j in range(M):
        eigenfeature = P_np[:, j]
        eigenfeature_img = eigenfeature.reshape(28, 28)
        
        # Normalize to [0, 1]
        min_val = eigenfeature_img.min()
        max_val = eigenfeature_img.max()
        if max_val > min_val:
            eigenfeature_img = (eigenfeature_img - min_val) / (max_val - min_val)
        
        # Here we keep the grayscale image as is (you can binarize if needed)
        binary_img = eigenfeature_img
        
        assigned_classes = [k for k in range(10) if L_np[j, k] >= 0.5]
        is_pure = len(assigned_classes) == 1
        
        for k in assigned_classes:
            folder_type = 'pure' if is_pure else 'shared'
            plt.figure(figsize=(3, 3), facecolor='#f5f5f5')
            ax = plt.gca()
            ax.set_facecolor('#f5f5f5')
            plt.imshow(binary_img, cmap='Blues')
            plt.axis('off')
            plt.tight_layout()
            
            save_path = os.path.join(
                os.path.expanduser(output_dir), 
                f'class_{k}/{folder_type}/eigenfeature_{j}.png'
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='#f5f5f5')
            plt.close()
            
            eigen_count_per_class[k] += 1
            if is_pure:
                pure_eigen_count_per_class[k] += 1
                print(f"Saved pure eigenfeature {j} to class {k}")
            else:
                shared_eigen_count_per_class[k] += 1
                print(f"Saved shared eigenfeature {j} to class {k} (shared with {len(assigned_classes)-1} other classes)")
    
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
    
    # Plot stacked bar chart
    ax.bar(x, shared_eigen_count_per_class, width, label='Shared Eigenfeatures', color='#3a86ff')
    ax.bar(x, pure_eigen_count_per_class, width, bottom=shared_eigen_count_per_class, 
           label='Pure Eigenfeatures', color='#ff006e')
    
    # Set axis labels with explicit font sizes
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
        if shared_eigen_count_per_class[i] > 0:
            ax.text(i, shared_eigen_count_per_class[i] / 2, str(shared_eigen_count_per_class[i]),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=tick_size)
        if pure_eigen_count_per_class[i] > 0:
            ax.text(i, shared_eigen_count_per_class[i] + pure_eigen_count_per_class[i] / 2,
                    str(pure_eigen_count_per_class[i]), ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=tick_size)
    
    # Add thin borders to axes
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#dddddd')
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.expanduser(output_dir), 'eigenfeature_distribution.png'),
                dpi=200, transparent=True)
    plt.close()
    
    print(f"\nVisualization complete! Results saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize MNIST eigenfeatures with configurable font sizes')
    parser.add_argument('--font-scale-dist', type=float, default=1.2,
                        help='Font scaling factor for distribution figure (default: 1.2)')
    parser.add_argument('--font-scale-heatmap', type=float, default=1.0,
                        help='Font scaling factor for L matrix heatmap (default: 1.0)')
    args = parser.parse_args()
    
    visualize_mnist_eigenfeatures(
        font_scale_dist=args.font_scale_dist, 
        font_scale_heatmap=args.font_scale_heatmap
    )

