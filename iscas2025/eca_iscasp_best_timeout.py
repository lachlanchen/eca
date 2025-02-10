import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_sklearn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA_sklearn
from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import TensorDataset, DataLoader  # Corrected Import
import os  # Added for file operations
import multiprocessing  # For implementing the timeout
import time

def generate_orthogonal_vectors():
    # Generate three random vectors
    np.random.seed(42)
    v1 = np.random.rand(3)
    v1 = v1 / np.linalg.norm(v1)

    # Generate v2 and make it orthogonal to v1
    v2 = np.random.rand(3)
    v2 = v2 - np.dot(v2, v1) * v1
    v2 = v2 / np.linalg.norm(v2)

    # Generate v3 and make it orthogonal to both v1 and v2
    v3 = np.random.rand(3)
    v3 = v3 - np.dot(v3, v1) * v1 - np.dot(v3, v2) * v2
    v3 = v3 / np.linalg.norm(v3)

    # Verify orthogonality
    assert np.allclose(np.dot(v1, v2), 0), "v1 and v2 are not orthogonal"
    assert np.allclose(np.dot(v1, v3), 0), "v1 and v3 are not orthogonal"
    assert np.allclose(np.dot(v2, v3), 0), "v2 and v3 are not orthogonal"

    return v1, v2, v3

def generate_2d_dataset(n_samples=1500, random_state=42, width=1.0):
    np.random.seed(random_state)
    n_samples_per_class = n_samples // 2

    # Class 1: y = x line with Gaussian t
    t1 = np.random.normal(0, width, n_samples_per_class)
    x1 = t1
    y1 = t1 + np.random.normal(0, 0.5 * width, n_samples_per_class)
    X_class1 = np.vstack((x1, y1)).T

    # Class 2: y = -x line with Gaussian t
    t2 = np.random.normal(0, width, n_samples_per_class)
    x2 = t2
    y2 = -t2 + np.random.normal(0, 0.5 * width, n_samples_per_class)
    X_class2 = np.vstack((x2, y2)).T

    # Combine the data
    X = np.vstack((X_class1, X_class2))
    y = np.array([0] * n_samples_per_class + [1] * n_samples_per_class)

    return X.astype(np.float32), y.astype(np.int64)

def generate_3d_dataset(n_samples=1500, random_state=42, width=1.0):
    np.random.seed(random_state)
    n_samples_per_class = n_samples // 2

    # Class 1: Along the x-axis
    x1 = np.random.normal(0, width, n_samples_per_class)
    y1 = np.random.normal(0, 0.5 * width, n_samples_per_class)
    z1 = np.random.normal(0, 0.5 * width, n_samples_per_class)
    X_class1 = np.vstack((x1, y1, z1)).T

    # Class 2: Along the y-axis
    x2 = np.random.normal(0, 0.5 * width, n_samples_per_class)
    y2 = np.random.normal(0, width, n_samples_per_class)
    z2 = np.random.normal(0, 0.5 * width, n_samples_per_class)
    X_class2 = np.vstack((x2, y2, z2)).T

    # Combine the data
    X = np.vstack((X_class1, X_class2))
    y = np.array([0] * n_samples_per_class + [1] * n_samples_per_class)

    return X.astype(np.float32), y.astype(np.int64)

def generate_3d_3c_dataset(n_samples=3000, random_state=42, width=1.0):
    np.random.seed(random_state)
    n_samples_per_class = n_samples // 3

    # Generate three orthogonal vectors not aligned with the standard axes
    v1, v2, v3 = generate_orthogonal_vectors()

    # Class 1: Along v1
    t1 = np.random.normal(0, width, n_samples_per_class)
    X_class1 = np.outer(t1, v1) + np.random.normal(0, 0.3 * width, (n_samples_per_class, 3))

    # Class 2: Along v2
    t2 = np.random.normal(0, width, n_samples_per_class)
    X_class2 = np.outer(t2, v2) + np.random.normal(0, 0.3 * width, (n_samples_per_class, 3))

    # Class 3: Along v3
    t3 = np.random.normal(0, width, n_samples_per_class)
    X_class3 = np.outer(t3, v3) + np.random.normal(0, 0.3 * width, (n_samples_per_class, 3))

    # Combine the data
    X = np.vstack((X_class1, X_class2, X_class3))
    y = np.array([0] * n_samples_per_class + [1] * n_samples_per_class + [2] * n_samples_per_class)

    return X.astype(np.float32), y.astype(np.int64)

def generate_iris_dataset():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X.astype(np.float32), y.astype(np.int64)

def generate_digits_dataset():
    digits = load_digits()
    X = digits.data / 255
    print("X.max: ", X.max())
    print("X.min: ", X.min())
    y = digits.target
    return X.astype(np.float32), y.astype(np.int64)

def generate_mnist_dataset(downsample_rate=1000):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)

    # Downsample by taking every 'downsample_rate'th sample
    train_indices = list(range(0, len(mnist_train), downsample_rate))
    test_indices = list(range(0, len(mnist_test), downsample_rate))

    X_train = mnist_train.data[train_indices].numpy().reshape(-1, 28*28).astype(np.float32)
    y_train = mnist_train.targets[train_indices].numpy().astype(np.int64)
    X_test = mnist_test.data[test_indices].numpy().reshape(-1, 28*28).astype(np.float32)
    y_test = mnist_test.targets[test_indices].numpy().astype(np.int64)

    # Combine train and test for consistency
    X = np.vstack((X_train, X_test)) / 255
    y = np.hstack((y_train, y_test))

    return X.astype(np.float32), y.astype(np.int64)

def generate_fashion_mnist_dataset(downsample_rate=1000):
    transform = transforms.Compose([transforms.ToTensor()])
    fm_train = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fm_test = FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Downsample by taking every 'downsample_rate'th sample
    train_indices = list(range(0, len(fm_train), downsample_rate))
    test_indices = list(range(0, len(fm_test), downsample_rate))

    X_train = fm_train.data[train_indices].numpy().reshape(-1, 28*28).astype(np.float32)
    y_train = fm_train.targets[train_indices].numpy().astype(np.int64)
    X_test = fm_test.data[test_indices].numpy().reshape(-1, 28*28).astype(np.float32)
    y_test = fm_test.targets[test_indices].numpy().astype(np.int64)

    print("X_train.max(): ", X_train.max())
    print("X_train.min(): ", X_train.min())

    # Combine train and test for consistency
    X = np.vstack((X_train, X_test)) / 255
    y = np.hstack((y_train, y_test))

    return X.astype(np.float32), y.astype(np.int64)

def preprocess_data(X_train_orig, X_test_orig, for_eca=True):
    # Centralize data by subtracting the mean of the training data
    mean_train = np.mean(X_train_orig, axis=0)
    X_train_centralized = X_train_orig  # - mean_train
    X_test_centralized = X_test_orig  # - mean_train

    if for_eca:
        # For ECA: Normalize each sample vector to have unit norm after centralization
        norms_train = np.linalg.norm(X_train_centralized, axis=1, keepdims=True)
        X_train_processed = X_train_centralized / norms_train

        norms_test = np.linalg.norm(X_test_centralized, axis=1, keepdims=True)
        X_test_processed = X_test_centralized / norms_test
    else:
        # For other methods: Use centralized data without further preprocessing
        X_train_processed = X_train_centralized
        X_test_processed = X_test_centralized

    return X_train_processed, X_test_processed

# Define the ECA model using nn.Module
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
        # Normalize columns of P to have unit norm (optional, currently commented out)
        P_norm = P  / torch.norm(P, dim=0, keepdim=True).detach()
        # P_norm = P  # Uncomment the above line if normalization is desired
        # Transform input
        # psi = X @ P_norm  # Shape: [N, M]
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

# Timeout functions added here
def run_with_timeout(func, args=(), kwargs={}, timeout_duration=60, default=None):
    """Runs a function with a timeout. If the function takes longer than
    timeout_duration seconds, it will be terminated."""
    import multiprocessing

    class InterruptableProcess(multiprocessing.Process):
        def __init__(self, func, args, kwargs):
            multiprocessing.Process.__init__(self)
            self.func = func
            self.args = args
            self.kwargs = kwargs
            self.result_queue = multiprocessing.Queue()

        def run(self):
            try:
                result = self.func(*self.args, **self.kwargs)
                self.result_queue.put(result)
            except Exception as e:
                self.result_queue.put(e)

    it = InterruptableProcess(func, args, kwargs)
    it.start()
    it.join(timeout_duration)
    if it.is_alive():
        it.terminate()
        it.join()
        return default
    else:
        result = it.result_queue.get()
        if isinstance(result, Exception):
            return default
        else:
            return result

def run_classifier_with_timeout(classifier_name, X_train, y_train, X_test, y_test, timeout_duration=60):
    def train_and_evaluate():
        if classifier_name == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000, multi_class='auto')
        elif classifier_name == 'LDA':
            model = LDA_sklearn()
        elif classifier_name == 'QDA':
            model = QDA_sklearn()
        elif classifier_name == 'SVM':
            model = SVC(kernel='linear')
        elif classifier_name == 'Kernel SVM':
            model = SVC(kernel='rbf')
        else:
            return None, None
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        params = None
        if classifier_name == 'Logistic Regression':
            params = model.coef_.size + model.intercept_.size
        elif classifier_name == 'LDA':
            params = model.coef_.size + model.intercept_.size
        elif classifier_name == 'QDA':
            n_features = X_train.shape[1]
            params = len(model.priors_) + model.means_.size + len(model.classes_) * n_features * n_features
        elif classifier_name == 'SVM':
            params = model.coef_.size + model.intercept_.size
        elif classifier_name == 'Kernel SVM':
            params = len(model.support_vectors_)
        return acc * 100, params  # Return accuracy in percentage and parameters

    result = run_with_timeout(train_and_evaluate, timeout_duration=timeout_duration)
    if result is not None:
        acc, params = result
    else:
        acc, params = None, None
    return acc, params  # None if timed out

def run_experiment(dataset_name, width=1.0, downsample_rate=1000, timeout_duration=60):
    print(f"\nRunning experiment on '{dataset_name}' dataset...\n")

    # Set learning rate and number of epochs based on dataset
    lr = 0.001
    # downsample_rate = 1000
    num_epochs = 1000
    if dataset_name == '2d':
        X, y = generate_2d_dataset(width=width)
        # num_epochs = 480
    elif dataset_name == '3d':
        X, y = generate_3d_dataset(width=width)
        # num_epochs = 120
    elif dataset_name == '3d_3c':
        X, y = generate_3d_3c_dataset(width=width)
        lr = 0.01
        # num_epochs = 960
    elif dataset_name == 'iris':
        X, y = generate_iris_dataset()
        lr = 0.01
        # num_epochs = 480
    elif dataset_name == 'digits':
        X, y = generate_digits_dataset()
        lr = 0.01
        # num_epochs = 240
    elif dataset_name == 'mnist':
        lr = 0.01
        # num_epochs = 1000
        X, y = generate_mnist_dataset(downsample_rate=downsample_rate)
    elif dataset_name == 'fashion_mnist':
        lr = 0.01
        # num_epochs = 240
        X, y = generate_fashion_mnist_dataset(downsample_rate=downsample_rate)
    else:
        print(f"Dataset '{dataset_name}' is not recognized.")
        return

    # Plot the dataset if it's 2D or 3D synthetic data
    if dataset_name == '2d':
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='Set1', edgecolor='k', s=50)
        plt.title(f'2D Synthetic Dataset (Width={width})')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend(title='Class')
        plt.savefig(f'{dataset_name}_dataset.png', dpi=300)
        plt.close()
    elif dataset_name in ['3d', '3d_3c']:
        from mpl_toolkits.mplot3d import Axes3D  # Ensure this import is within the block
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='Set1', edgecolor='k', s=20)
        ax.set_title(f'{dataset_name.upper()} Synthetic Dataset (Width={width})')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$x_3$')
        legend = ax.legend(*scatter.legend_elements(), title="Class", loc="upper right")
        ax.add_artist(legend)
        plt.savefig(f'{dataset_name}_dataset.png', dpi=300)
        plt.close()

    # Split data into training and testing sets
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Preprocess data for ECA
    X_train_eca, X_test_eca = preprocess_data(X_train_orig, X_test_orig, for_eca=True)
    # Convert to torch tensors
    X_train_tensor = torch.from_numpy(X_train_eca)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test_eca)
    y_test_tensor = torch.from_numpy(y_test)

    # Number of features and classes
    M = X_train_tensor.shape[1]
    L = len(np.unique(y))

    # Initialize model
    model = ECAModel(M, L)

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Define regularization parameter
    reg_lambda = 0.01  # Regularization strength

    # Training loop parameters
    batch_size = len(X_train_tensor)
    train_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0  # Initialize best validation accuracy
    best_model_path = f'best_model_{dataset_name}.pth'  # Define model save path

    # Create DataLoader for batch training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # Forward pass
            class_scores, _, L_matrix, _ = model(X_batch)  # Capture L_matrix

            ce_loss = criterion(class_scores, y_batch)

            # Compute L2 regularization term for L_matrix
            reg_term = reg_lambda * torch.norm(L_matrix, p=2)

            # Total loss = cross-entropy loss + regularization term
            total_loss = ce_loss + reg_term

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item() * X_batch.size(0)

        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            # Validation forward pass
            val_scores, _, _, _ = model(X_test_tensor)
            # Predictions
            val_predicted = val_scores.argmax(dim=1)
            # Compute validation accuracy
            val_accuracy = (val_predicted == y_test_tensor).float().mean()

            val_accuracies.append(val_accuracy.item())

        # Check if current validation accuracy is the best
        if val_accuracy.item() > best_val_accuracy:
            best_val_accuracy = val_accuracy.item()
            torch.save(model.state_dict(), best_model_path)  # Save the best model

        # Print progress every (num_epochs // 10) epochs
        if (epoch + 1) % max(1, num_epochs // 10) == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, '
                  f'Val Acc: {val_accuracy.item()*100:.2f}%')

    # After training, load the best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print(f"Best model file '{best_model_path}' not found. Using the last trained model.")

    # After loading the best model, get P_norm, L, and A
    model.eval()
    with torch.no_grad():
        _, P_norm, L_matrix, A = model(X_train_tensor)
        A_raw = model.A_raw
        L_raw = model.L_raw

    print("P: ", P_norm)
    print("L_matrix: ", L_matrix)

    # Save e^{A} (P_norm) as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(P_norm.detach().numpy(), cmap='viridis', annot=True)
    plt.title(f'Heatmap of Normalized $e^{{A}}$ (Transformation Matrix P) for {dataset_name}')
    plt.xlabel('Features')
    plt.ylabel('Transformed Features')
    plt.savefig(f'{dataset_name}_eA_heatmap.png', dpi=300)
    plt.close()

    # Save L as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(L_matrix.detach().numpy(), cmap='viridis', annot=True)
    plt.title(f'Heatmap of Mapping Matrix L for {dataset_name}')
    plt.xlabel('Classes')
    plt.ylabel('Eigenfeatures')
    plt.savefig(f'{dataset_name}_L_heatmap.png', dpi=300)
    plt.close()

    # Print the number of parameters
    num_parameters = model.num_parameters  # sum(p.numel() for p in model.parameters())
    print(f'\nNumber of parameters in ECA model: {num_parameters}')

    # Evaluate ECA on test set using the best model
    with torch.no_grad():
        # Get the raw class scores (logits)
        test_scores, _, _, _ = model(X_test_tensor)
        # Use argmax on the raw scores to get the predicted labels
        test_predicted_labels = test_scores.argmax(dim=1)
        # Compute test accuracy
        test_accuracy = (test_predicted_labels == y_test_tensor).float().mean()
        test_confusion_matrix = confusion_matrix(y_test_tensor.numpy(), test_predicted_labels.numpy())
        print(f'\nECA Test Accuracy: {test_accuracy.item()*100:.2f}%')
        print(f'ECA Confusion Matrix:\n{test_confusion_matrix}')

    # Now, run other classifiers using centralized data
    X_train_other, X_test_other = preprocess_data(X_train_orig, X_test_orig, for_eca=False)

    # Initialize lists to collect results
    model_names = ['Logistic Regression', 'LDA', 'QDA', 'SVM', 'Kernel SVM']
    accuracies = []
    parameters = []

    for clf_name in model_names:
        print(f"\nTraining {clf_name} with timeout {timeout_duration} seconds...")
        acc, params = run_classifier_with_timeout(clf_name, X_train_other, y_train, X_test_other, y_test, timeout_duration=timeout_duration)
        if acc is not None:
            print(f'{clf_name} Test Accuracy: {acc:.2f}%')
            print(f'Number of parameters in {clf_name}: {params}')
            accuracies.append(acc / 100)
            parameters.append(params)
        else:
            print(f'{clf_name} did not finish in {timeout_duration} seconds.')
            accuracies.append(None)
            parameters.append(None)

    # Append ECA results
    model_names.append('ECA')
    accuracies.append(test_accuracy.item())
    parameters.append(num_parameters)

    # Collect results into a table
    results = pd.DataFrame({
        'Model': model_names,
        'Accuracy (%)': [acc * 100 if acc is not None else None for acc in accuracies],
        'Parameters': parameters
    })

    print(f'\nSummary of Results ({dataset_name}):')
    print(results.to_string(index=False))
    print('\n' + '='*60 + '\n')

def main():
    parser = argparse.ArgumentParser(description='Run ECA and other classifiers on synthetic datasets.')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset to use: 2d, 3d, 3d_3c, iris, digits, mnist, or fashion_mnist')
    parser.add_argument('-w', '--width', type=float, default=1.0, help='Width (standard deviation) of the data distributions')
    parser.add_argument('-ds', '--downsample_rate', type=int, default=100, help='Downsample rate for MNIST and FashionMNIST datasets')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout duration in seconds for each classifier.')
    args = parser.parse_args()

    # Available datasets
    dataset_list = [
        '2d',
        '3d',
        '3d_3c',
        'iris',
        'digits',
        'mnist',
        'fashion_mnist'
    ]

    if args.dataset:
        if args.dataset.lower() in dataset_list:
            run_experiment(args.dataset.lower(), width=args.width, downsample_rate=args.downsample_rate, timeout_duration=args.timeout)
        else:
            print(f"Dataset '{args.dataset}' is not recognized.")
            print(f"Available datasets: {', '.join(dataset_list)}")
    else:
        for dataset_name in dataset_list:
            run_experiment(dataset_name, width=args.width, downsample_rate=args.downsample_rate, timeout_duration=args.timeout)

if __name__ == '__main__':
    main()
