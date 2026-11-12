import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import matthews_corrcoef
from nnfst_nystrom import NNFSTNystrom


def test_full_train_and_prediction():
    X, y = make_blobs(n_samples=240, n_features=5, centers=3, cluster_std=.35, random_state=9)
    m = NNFSTNystrom(r=4, n_landmarks=55, gamma=.5, seed=9)
    m.fit(X, y)
    pred = m.predict(X)
    assert m.n_samples_fit_ == len(X)
    assert m.feature_rank_ <= 55
    assert m.n_projection_dims_ >= 1
    assert np.isfinite(m.base_points_).all()
    assert matthews_corrcoef(y, pred) > .65


def test_projection_and_determinism():
    X, y = make_blobs(n_samples=100, n_features=4, centers=2, random_state=2)
    a = NNFSTNystrom(r=5, gamma=.4, n_landmarks=32, seed=42)
    b = NNFSTNystrom(r=5, gamma=.4, n_landmarks=32, seed=42)
    a.fit(X, y)
    b.fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))


def test_no_landmark_test_leakage():
    X, y = make_blobs(n_samples=90, n_features=3, centers=3, random_state=10)
    model = NNFSTNystrom(r=4, gamma=.2, n_landmarks=27, seed=5)
    model.fit(X[:60], y[:60])
    assert model.n_samples_fit_ == 60
    assert len(model.landmarks_) == 27
    # Test split must not contribute landmarks.
    landmarks = {tuple(row) for row in model.landmarks_}
    assert landmarks.issubset({tuple(row) for row in X[:60]})
