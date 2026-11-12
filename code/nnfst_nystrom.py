"""Scalable nNFST-inspired Nyström/regularized near-null approximation.

NOT the full-kernel, exact-null nNFST published by Nguyen & Le (2025).
This method preserves mutual-rNN per-class components, component-labelled
nearest-base classification, and a kernel map, but replaces exact RKHS
null-space by generalized minimum-within-scatter directions of a Nyström
feature map. Publication label: nNFST-Nyström (independent approximation).

Requires nnfst_paper.py, the previously installed research implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import euclidean_distances

from nnfst_paper import NNFSTPaper


@dataclass
class NNFSTNystrom:
    r: int = 5
    gamma: float = 0.1
    kernel: str = "l05_exponential_kernel"
    n_landmarks: int = 384
    projection_cap: int = 0  # 0 = all possible up to Q-1 and feature-map rank
    eig_tol: float = 1e-9
    regularization: float = 1e-6
    batch_size: int = 256
    seed: int = 42

    def __post_init__(self):
        if self.r < 1 or self.n_landmarks < 2 or self.batch_size < 1:
            raise ValueError("r >= 1, n_landmarks >= 2 and batch_size >= 1 are required")
        if not (self.eig_tol > 0 and self.regularization > 0):
            raise ValueError("eig_tol and regularization must be positive")
        self._paper = NNFSTPaper(r=self.r, gamma=self.gamma, kernel=self.kernel)

    def _select_landmarks(self, X, y):
        """Select landmarks from TRAIN ONLY, including each class when possible."""
        rng = np.random.default_rng(self.seed)
        n = len(y)
        target = min(n, self.n_landmarks)
        labels, encoded = np.unique(y, return_inverse=True)
        if len(labels) > target:
            raise ValueError("n_landmarks must be >= number of classes")
        selected = []
        for k in range(len(labels)):
            candidates = np.flatnonzero(encoded == k)
            selected.append(int(rng.choice(candidates)))
        used = set(selected)
        remaining = np.array([i for i in range(n) if i not in used], dtype=int)
        selected += rng.choice(remaining, size=target - len(selected), replace=False).tolist()
        return X[np.array(selected, dtype=int)]

    def _features(self, X):
        parts = []
        for begin in range(0, len(X), self.batch_size):
            K = self._paper._kernel(X[begin:begin + self.batch_size], self.landmarks_)
            parts.append(K @ self.whitening_basis_)
        return np.concatenate(parts, axis=0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        if X.ndim != 2 or X.shape[0] != y.size or y.size < 4:
            raise ValueError("Invalid X/y shapes")
        if not np.isfinite(X).all():
            raise ValueError("Nonfinite input; inspect preprocessing before training")
        if np.unique(y).size < 2:
            raise ValueError("At least two training classes are needed")

        # Components are defined on the ENTIRE training set (not landmarks).
        comp, parent = self._paper._build_component_labels(X, y)
        q = len(parent)
        if q < 2:
            raise ValueError("No discriminative components")
        counts = np.bincount(comp, minlength=q)

        self.landmarks_ = self._select_landmarks(X, y)
        W = self._paper._kernel(self.landmarks_, self.landmarks_)
        W = (W + W.T) * .5
        eig, basis = eigh(W, check_finite=False, overwrite_a=True)
        largest = max(float(eig[-1]), 0.0)
        keep = eig > max(1e-12, largest * self.eig_tol)
        if not np.any(keep):
            raise ValueError("Nyström landmark kernel has zero positive rank")
        self.whitening_basis_ = basis[:, keep] / np.sqrt(eig[keep])[None, :]
        phi = self._features(X)  # all training rows, kernel features only
        self.feature_mean_ = phi.mean(axis=0)
        Z = phi - self.feature_mean_[None, :]
        del phi

        # Orthogonal projection onto component-wise means, computed O(NM).
        sums = np.zeros((q, Z.shape[1]), dtype=np.float64)
        np.add.at(sums, comp, Z)
        group_means = sums / counts[:, None]
        residuals = Z - group_means[comp]
        within = residuals.T @ residuals
        total = Z.T @ Z
        within = (within + within.T) * .5
        total = (total + total.T) * .5
        del residuals

        # In finite-rank Nyström space, exact within-scatter nullspace generally
        # DOES NOT exist on full training data. The smallest generalized
        # eigenvalues provide near-null discriminant directions instead.
        reg = self.regularization * max(float(np.trace(total)) / total.shape[0], 1e-12)
        b = total + reg * np.eye(total.shape[0])
        ew, V = eigh(within, b, check_finite=False)
        dims = min(q - 1, V.shape[1])
        if self.projection_cap:
            dims = min(dims, self.projection_cap)
        if dims < 1:
            raise ValueError("Projection dimension is zero")
        directions = V[:, :dims]
        # Normalize to unit pooled train variance per feature for distances.
        proj_train = Z @ directions
        scales = np.sqrt(np.maximum(np.mean(proj_train ** 2, axis=0), 1e-12))
        directions = directions / scales[None, :]
        self.directions_ = directions
        self.base_points_ = group_means @ directions
        self.component_class_ = parent
        self.n_components_ = q
        self.n_projection_dims_ = dims
        self.feature_rank_ = int(self.whitening_basis_.shape[1])
        self.n_samples_fit_ = len(y)
        self.approximation_error_ = float(np.mean(ew[:dims]))
        self.method_ = "NNFST_NYSTROM_REGULARIZED_NEAR_NULL"
        return self

    def transform(self, X):
        if not hasattr(self, "directions_"):
            raise ValueError("Call fit first")
        X = np.asarray(X, dtype=np.float64)
        if not np.isfinite(X).all():
            raise ValueError("Nonfinite test input")
        result = []
        for begin in range(0, len(X), self.batch_size):
            batch = X[begin:begin + self.batch_size]
            mapped = self._paper._kernel(batch, self.landmarks_) @ self.whitening_basis_
            result.append((mapped - self.feature_mean_) @ self.directions_)
        return np.concatenate(result, axis=0)

    def predict(self, X):
        if not hasattr(self, "base_points_"):
            raise ValueError("Call fit first")
        pred = []
        # Batch so distance matrix never becomes N_test x N_components.
        for start in range(0, len(X), self.batch_size):
            P = self.transform(X[start:start + self.batch_size])
            dist = euclidean_distances(P, self.base_points_, squared=True)
            pred.append(self.component_class_[np.argmin(dist, axis=1)])
        return np.concatenate(pred)
