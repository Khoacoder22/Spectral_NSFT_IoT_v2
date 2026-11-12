
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.neighbors import NearestNeighbors


@dataclasses.dataclass
class NNFSTPaper:
    # Hyperparameters to tune
    r: int = 5
    gamma: float = 0.01

    # Default kernel for the competitor experiment
    kernel: str = "l05_exponential_kernel"

    # Numerical and resource settings
    relative_tol: float = 1e-9
    max_eigensize: int = 1200
    batch_size: int = 256

    # None = closed-set classification
    novelty_threshold: Optional[float] = None

    def __post_init__(self):
        if self.r < 1:
            raise ValueError("r must be >= 1")

        if self.gamma <= 0:
            raise ValueError("gamma must be > 0")

        if self.kernel not in (
            "rbf",
            "l05_exponential_kernel",
            "linear",
        ):
            raise ValueError("Unsupported kernel")

        if self.relative_tol <= 0:
            raise ValueError("relative_tol must be positive")

        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def _kernel(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:

        if self.kernel == "rbf":
            return rbf_kernel(
                X,
                Y,
                gamma=self.gamma,
            )

        if self.kernel == "linear":
            return X @ Y.T

        # L05 exponential kernel
        # K(x,y) = exp(-gamma * sqrt(||x-y||_2))
        #
        # Must be checked against the project's
        # cpai kernel implementation.

        dist = euclidean_distances(X, Y)

        np.sqrt(dist, out=dist)
        np.multiply(dist, -self.gamma, out=dist)
        np.exp(dist, out=dist)

        return dist

    @staticmethod
    def _scc_from_mutual_knn(
        X: np.ndarray,
        r: int,
    ) -> np.ndarray:

        n = len(X)

        if n < 2:
            return np.zeros(n, dtype=np.int32)

        k = min(r, n - 1)

        knn = NearestNeighbors(
            n_neighbors=k + 1,
            metric="euclidean",
            algorithm="auto",
        )

        knn.fit(X)

        _, raw_neighbors = knn.kneighbors(X)

        neighbors = []

        for i, row in enumerate(raw_neighbors):
            q = [
                int(j)
                for j in row
                if j != i
            ][:k]

            if len(q) < k:
                raise ValueError(
                    "Insufficient non-self neighbors"
                )

            neighbors.append(set(q))

        rows = []
        cols = []

        for i, candidates in enumerate(neighbors):
            for j in candidates:
                if i < j and i in neighbors[j]:
                    rows.extend((i, j))
                    cols.extend((j, i))

        graph = csr_matrix(
            (
                np.ones(len(rows), dtype=np.int8),
                (rows, cols),
            ),
            shape=(n, n),
        )

        _, labels = connected_components(
            graph,
            directed=False,
        )

        return labels.astype(
            np.int32,
            copy=False,
        )

    def _build_component_labels(self, X, y):

        label_by_row = np.empty(
            len(y),
            dtype=np.int32,
        )

        original_class_for_component = []
        offset = 0

        for label in np.unique(y):

            idx = np.where(y == label)[0]

            local = self._scc_from_mutual_knn(
                X[idx],
                self.r,
            )

            label_by_row[idx] = local + offset

            num_groups = int(local.max()) + 1

            original_class_for_component.extend(
                [label] * num_groups
            )

            offset += num_groups

        return (
            label_by_row,
            np.asarray(original_class_for_component),
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "NNFSTPaper":

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()

        if (
            X.ndim != 2
            or len(X) != len(y)
            or len(X) < 3
        ):
            raise ValueError(
                "Invalid training data"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Input X contains NaN or Infinity"
            )

        if len(np.unique(y)) < 2:
            raise ValueError(
                "At least two classes required"
            )

        n = len(X)

        # Safety guard for full-kernel computation
        if self.max_eigensize and n > self.max_eigensize:

            gb = (n * n * 8) / (1024 ** 3)

            raise MemoryError(
                f"Full kernel n={n} requires "
                f"at least {gb:.2f} GiB "
                "for one dense matrix. "
                f"Limit={self.max_eigensize}"
            )

        # ---------------------------------------
        # Step 1: Create SCC components
        # ---------------------------------------

        component_id, parents = (
            self._build_component_labels(X, y)
        )

        q = len(parents)

        if q < 2:
            raise ValueError(
                "Less than two SCC components"
            )

        counts = np.bincount(
            component_id,
            minlength=q,
        )

        # Sparse normalized membership matrix
        group = csr_matrix(
            (
                1 / np.sqrt(counts[component_id]),
                (np.arange(n), component_id),
            ),
            shape=(n, q),
            dtype=np.float64,
        )

        # ---------------------------------------
        # Step 2: Kernel matrix
        # ---------------------------------------

        K = self._kernel(X, X)

        K = (K + K.T) * 0.5

        km = K.mean(axis=0)
        grand = float(km.mean())

        Kc = (
            K
            - km[None, :]
            - km[:, None]
            + grand
        )

        Kc = (Kc + Kc.T) * 0.5

        # ---------------------------------------
        # Step 3: Kernel eigendecomposition
        # ---------------------------------------

        w, U = np.linalg.eigh(Kc)

        lambda_tol = max(
            1e-12,
            self.relative_tol * max(float(w[-1]), 0),
        )

        valid = w > lambda_tol

        if not valid.any():
            raise ValueError(
                "Centered kernel has zero rank"
            )

        w = w[valid]
        U = U[:, valid]

        Z = U * np.sqrt(w)[None, :]

        # ---------------------------------------
        # Step 4: Within-SCC null space
        # ---------------------------------------

        E = Z - group @ (group.T @ Z)

        C = E.T @ E
        C = (C + C.T) / 2

        ew, V = np.linalg.eigh(C)

        within_tol = max(
            1e-12,
            self.relative_tol * max(float(ew[-1]), 0),
        )

        null_ids = np.flatnonzero(
            ew <= within_tol
        )

        if null_ids.size == 0:
            raise ValueError(
                "No null directions found"
            )

        null_vectors = V[:, null_ids]

        # ---------------------------------------
        # Step 5: Remove zero-variance directions
        # ---------------------------------------

        total_variation = np.sum(
            (Z @ null_vectors) ** 2,
            axis=0,
        )

        keep = total_variation > max(
            1e-12,
            self.relative_tol * float(
                np.max(total_variation)
            ),
        )

        null_vectors = null_vectors[:, keep]

        if null_vectors.shape[1] == 0:
            raise ValueError(
                "No valid projection directions"
            )

        if null_vectors.shape[1] > q - 1:

            ranks = np.argsort(
                total_variation[keep]
            )[::-1][:q - 1]

            null_vectors = null_vectors[:, ranks]

        # ---------------------------------------
        # Step 6: Projection coefficients
        # ---------------------------------------

        coefficients = (
            U / np.sqrt(w)[None, :]
        ) @ null_vectors

        train_projected = Z @ null_vectors

        # ---------------------------------------
        # Step 7: SCC base points
        # ---------------------------------------

        bases = (
            np.asarray(
                group.T @ train_projected
            )
            / np.sqrt(counts)[:, None]
        )

        within_resid = (
            train_projected
            - group @ (
                group.T @ train_projected
            )
        )

        within_err = float(
            np.linalg.norm(within_resid)
            / max(
                np.linalg.norm(train_projected),
                1e-12,
            )
        )

        # ---------------------------------------
        # Save fitted model
        # ---------------------------------------

        self.X_fit_ = X

        self.kernel_column_mean_ = km
        self.kernel_grand_mean_ = grand

        self.coefficients_ = coefficients

        self.component_labels_ = component_id
        self.component_class_ = parents

        self.base_points_ = bases

        self.n_components_ = q

        self.n_projection_dims_ = (
            coefficients.shape[1]
        )

        self.within_null_relative_error_ = (
            within_err
        )

        self.n_samples_fit_ = n

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:

        if not hasattr(self, "coefficients_"):
            raise ValueError(
                "Call fit() first"
            )

        X = np.atleast_2d(
            np.asarray(X, dtype=np.float64)
        )

        result = np.empty(
            (
                len(X),
                self.n_projection_dims_,
            ),
            dtype=np.float64,
        )

        for i in range(
            0,
            len(X),
            self.batch_size,
        ):

            slice_ = slice(
                i,
                min(i + self.batch_size, len(X)),
            )

            k = self._kernel(
                X[slice_],
                self.X_fit_,
            )

            rowmean = k.mean(axis=1)

            k -= rowmean[:, None]
            k -= self.kernel_column_mean_[None, :]
            k += self.kernel_grand_mean_

            result[slice_] = (
                k @ self.coefficients_
            )

        return result

    def predict_with_novelty(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None,
    ):

        xproj = self.transform(X)

        distances = euclidean_distances(
            xproj,
            self.base_points_,
        )

        arg = distances.argmin(axis=1)

        novelty = distances[
            np.arange(len(arg)),
            arg,
        ]

        predictions = (
            self.component_class_[arg].copy()
        )

        if threshold is None:
            threshold = self.novelty_threshold

        if threshold is not None:

            predictions = np.asarray(
                predictions,
                dtype=object,
            )

            predictions[
                novelty > threshold
            ] = "__NOVEL__"

        return predictions, novelty

    def predict(self, X: np.ndarray):
        return self.predict_with_novelty(X)[0]

    def predict_novelty_score(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        return self.predict_with_novelty(X)[1]
