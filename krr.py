# Implementation of Kernel Ridge Regression
# Last modified 21 October 2026

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist


def rbf_kernel(X1, X2, gamma=1.0):
    """Computes the radial basis function Gaussian kernel between matrices."""
    sq_dists = cdist(X1, X2, metric="sqeuclidean")
    return np.exp(-gamma * sq_dists)


class KernelRidgeRegression:
    def __init__(self, alpha=1.0, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.X_train = None
        self.dual_coef_ = None

    def fit(self, X, y):
        self.X_train = X
        n_samples = X.shape[0]
        K = rbf_kernel(X, X, self.gamma)
        K.flat[:: n_samples + 1] += self.alpha

        self.dual_coef_ = scipy.linalg.solve(K, y, assume_a="pos")
        return self

    def predict(self, X):
        if self.dual_coef_ is None:
            raise ValueError("Model has not been fitted yet!")
        K_test = rbf_kernel(X, self.X_train, self.gamma)
        return K_test @ self.dual_coef_


if __name__ == "__main__":
    np.random.seed(16)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(len(X))

    krr = KernelRidgeRegression(alpha=1.0, gamma=5.0)
    krr.fit(X, y)

    X_fine = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_pred = krr.predict(X_fine)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(X, y, color="blue", label="Training Data", s=20, zorder=3)
    ax.plot(X_fine, y_pred, color="red", linewidth=2, label="KRR Fit (RBF Kernel)")
    ax.plot(
        X_fine, np.sin(X_fine), color="green", linestyle="--", label="True Function"
    )
    ax.set_title("Kernel Ridge Regression with RBF Kernel")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(
        krr.alpha * np.eye(len(X)) + rbf_kernel(X, X, krr.gamma),
        cmap="viridis",
        interpolation="nearest",
    )
    plt.colorbar(label="Kernel Value")
    plt.title("Learned Kernel Matrix (Regularised)")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.show()
