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
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm
import multiprocessing
import time

def generate_mnist_dataset(downsample_fraction=1.0):
    # Use only the training data
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_full = MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Convert data to numpy arrays
    X_full = mnist_full.data.numpy().reshape(-1, 28*28).astype(np.float32) / 255
    y_full = mnist_full.targets.numpy().astype(np.int64)

    # Debugging prints
    print("X_full.max: ", X_full.max())
    print("X_full.min: ", X_full.min())
    
    # Split into train and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
    
    # Determine the number of training samples after downsampling
    n_train_samples = int(len(X_train_full) * downsample_fraction)
    
    # Downsample the training data
    if downsample_fraction < 1.0:
        # Randomly select n_train_samples indices
        np.random.seed(42)
        train_indices = np.random.choice(len(X_train_full), n_train_samples, replace=False)
        X_train = X_train_full[train_indices]
        y_train = y_train_full[train_indices]
    else:
        X_train = X_train_full
        y_train = y_train_full
    
    return X_train, X_test, y_train, y_test

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
        return class_scores, P, L, A

def run_with_timeout(func, args=(), kwargs={}, timeout_duration=60, default=None):
    """Runs a function with a timeout. If the function takes longer than
    timeout_duration seconds, it will be terminated."""
    import multiprocessing

    class InterruptableThread(multiprocessing.Process):
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

    it = InterruptableThread(func, args, kwargs)
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

def run_classifier(classifier_name, X_train, y_train, X_test, y_test, timeout_duration=60):
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
            return None
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc * 100  # Return accuracy in percentage

    acc = run_with_timeout(train_and_evaluate, timeout_duration=timeout_duration)
    return acc  # None if timed out

def run_mnist_experiment(downsample_fractions, timeout_duration=60):
    # Initialize dictionaries to store accuracies
    accuracies = {
        'Logistic Regression': [],
        'LDA': [],
        'QDA': [],
        'SVM': [],
        'Kernel SVM': [],
        'ECA': []
    }
    inv_downsample_rates = []

    for downsample_fraction in downsample_fractions:
        print(f"\nRunning experiment with downsample fraction: {downsample_fraction}")

        # Prepare data
        X_train_orig, X_test_orig, y_train, y_test = generate_mnist_dataset(
            downsample_fraction=downsample_fraction)

        # Preprocess data for ECA
        X_train_eca, X_test_eca = preprocess_data(
            X_train_orig, X_test_orig, for_eca=True)
        # Convert to torch tensors
        X_train_tensor = torch.from_numpy(X_train_eca)
        y_train_tensor = torch.from_numpy(y_train)
        X_test_tensor = torch.from_numpy(X_test_eca)
        y_test_tensor = torch.from_numpy(y_test)

        # Number of features and classes
        M = X_train_tensor.shape[1]
        L = len(np.unique(y_train))

        # Initialize model
        model = ECAModel(M, L)

        # Define optimizer
        lr = 0.01
        optimizer = Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Define regularization parameter
        reg_lambda = 0.01  # Regularization strength

        # Training loop parameters
        num_epochs = 1000
        batch_size = len(X_train_tensor)
        train_losses = []
        val_accuracies = []
        best_val_accuracy = 0.0  # Initialize best validation accuracy
        best_model_path = f'best_model_mnist_{downsample_fraction}.pth'  # Define model save path

        # Create DataLoader for batch training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in tqdm(range(num_epochs)):
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

        # After training, load the best model
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
        else:
            print(f"Best model file '{best_model_path}' not found. Using the last trained model.")

        # Evaluate ECA on test set using the best model
        model.eval()
        with torch.no_grad():
            # Get the raw class scores (logits)
            test_scores, _, _, _ = model(X_test_tensor)
            # Use argmax on the raw scores to get the predicted labels
            test_predicted_labels = test_scores.argmax(dim=1)
            # Compute test accuracy
            test_accuracy = (test_predicted_labels == y_test_tensor).float().mean()
            print(f'ECA Test Accuracy: {test_accuracy.item()*100:.2f}%')
            accuracies['ECA'].append(test_accuracy.item()*100)

        # Now, run other classifiers using centralized data
        X_train_other, X_test_other = preprocess_data(
            X_train_orig, X_test_orig, for_eca=False)

        # List of classifiers to run with timeout
        classifiers = ['Logistic Regression', 'LDA', 'QDA', 'SVM', 'Kernel SVM']

        for clf_name in classifiers:
            print(f"\nTraining {clf_name} with timeout {timeout_duration} seconds...")
            acc = run_classifier(clf_name, X_train_other, y_train, X_test_other, y_test, timeout_duration=timeout_duration)
            if acc is not None:
                print(f'{clf_name} Test Accuracy: {acc:.2f}%')
                accuracies[clf_name].append(acc)
            else:
                print(f'{clf_name} did not finish in {timeout_duration} seconds.')
                accuracies[clf_name].append(None)  # None indicates timeout

        # Record inverse downsample rate
        inv_downsample_rate = 1.0 / downsample_fraction
        inv_downsample_rates.append(inv_downsample_rate)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    for model_name, acc_list in accuracies.items():
        plt.plot(inv_downsample_rates, acc_list, marker='o', label=model_name)

    plt.xscale('log')
    plt.xlabel('Inverse Downsample Fraction (log scale)')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracies vs. Inverse Downsample Fraction on MNIST')
    plt.legend()
    plt.grid(True)
    plt.savefig('mnist_downsampled_accuracies.png', dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Run ECA and other classifiers on MNIST with varying downsample rates.')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout duration in seconds for each classifier.')
    args = parser.parse_args()

    # Downsample fractions to test (from smaller to larger datasets)
    downsample_fractions = [0.001, 0.0025, 0.005, 0.01, 0.025,
                            0.05, 0.1, 0.25, 0.5, 1]

    run_mnist_experiment(downsample_fractions, timeout_duration=args.timeout)

if __name__ == '__main__':
    main()
