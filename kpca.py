# Kernel PCA as a denoising algorithm for digits
# Last modified 21 October 2026
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
from krr import KernelRidgeRegression, rbf_kernel
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class KernelPCA:
    def __init__(self, n_components=20, gamma=0.1):
        self.n_components = n_components
        self.gamma = gamma

    def fit(self, X):
        self.X_train = X.copy()
        K = rbf_kernel(X, X, self.gamma)

        # Centre the implicit feature vectors using kernel means.
        self.kernel_column_mean_ = K.mean(axis=0)
        self.kernel_mean_ = K.mean()
        K_centered = (
            K
            - K.mean(axis=1, keepdims=True)
            - self.kernel_column_mean_
            + self.kernel_mean_
        )

        # Keep the largest positive eigenvalues and normalise their eigenvectors.
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        tolerance = max(1e-10, np.finfo(float).eps * len(X) * eigenvalues[-1])
        indices = np.flatnonzero(eigenvalues > tolerance)[::-1][: self.n_components]
        if not len(indices):
            raise ValueError("Training kernel has no positive eigenvalues")
        self.eigenvalues_ = eigenvalues[indices]
        self.alphas_ = eigenvectors[:, indices] / np.sqrt(self.eigenvalues_)
        return self

    def transform(self, X):
        # Centre new similarities relative to the training feature-space mean.
        K = rbf_kernel(X, self.X_train, self.gamma)
        K_centered = (
            K
            - K.mean(axis=1, keepdims=True)
            - self.kernel_column_mean_
            + self.kernel_mean_
        )
        return K_centered @ self.alphas_


if __name__ == "__main__":
    data_dir = Path(__file__).with_name(".data")
    data_dir.mkdir(exist_ok=True)
    datasets = []
    for filename in ("usps.bz2", "usps.t.bz2"):
        path = data_dir / filename
        if not path.exists():
            url = (
                "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/"
                + filename
            )
            with urlopen(url, timeout=60) as response:
                data = response.read()
            path.write_bytes(data)
        pixels, labels = load_svmlight_file(str(path), n_features=256, zero_based=False)  # type: ignore
        pixels = pixels.toarray()  # type: ignore
        if not np.isfinite(pixels).all() or pixels.min() < -1 or pixels.max() > 1:
            raise ValueError("USPS pixels must be finite and in [-1, 1]")
        datasets.append(((pixels + 1) / 2, labels))

    # shortcut: subset training for the dense eigensolve; use an approximate kernel
    # method when scaling the experiment to the full training dataset.
    X_train, _ = train_test_split(
        datasets[0][0], train_size=1500, stratify=datasets[0][1], random_state=16
    )
    X_test = datasets[1][0]
    rng = np.random.default_rng(16)
    X_test_noisy = X_test + rng.normal(0, 0.2, X_test.shape)

    kpca = KernelPCA(n_components=20, gamma=0.01).fit(X_train)
    Z_train = kpca.transform(X_train)

    # Learn the approximate inverse using only clean training images.
    decoder = KernelRidgeRegression(alpha=0.01, gamma=1.0).fit(Z_train, X_train)
    X_denoised = decoder.predict(kpca.transform(X_test_noisy))

    pca = PCA(n_components=20, svd_solver="full").fit(X_train)
    X_linear = pca.inverse_transform(pca.transform(X_test_noisy))
    linear_mse = np.mean((X_linear - X_test) ** 2)

    noisy_mse = np.mean((X_test_noisy - X_test) ** 2)
    denoised_mse = np.mean((X_denoised - X_test) ** 2)
    print(f"Noisy MSE:    {noisy_mse:.6f}")
    print(f"Linear PCA MSE: {linear_mse:.6f}")
    print(f"Denoised MSE: {denoised_mse:.6f}")

    fig, axes = plt.subplots(4, 10, figsize=(13, 6.5))
    for row, (images, label) in enumerate(
        [
            (X_test, "Clean"),
            (X_test_noisy, "Noisy"),
            (X_linear, "Linear PCA"),
            (X_denoised, "Kernel PCA"),
        ]
    ):
        for col in range(10):
            ax = axes[row, col]
            ax.imshow(images[col].reshape(16, 16), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(label)
    fig.suptitle(
        f"USPS denoising — 20 components per method\n"
        f"Test MSE: noisy {noisy_mse:.4f} | linear PCA {linear_mse:.4f} "
        f"| kernel PCA {denoised_mse:.4f}"
    )
    fig.tight_layout()
    output = Path(__file__).with_name("usps_denoising.png")
    fig.savefig(output, dpi=180)
    print(f"Saved {output}")
    plt.show()
