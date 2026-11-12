import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import matthews_corrcoef
from nnfst_paper import NNFSTPaper


def test_components_separate_clusters_and_class():
    X = np.r_[np.random.RandomState(11).normal(-3, .05, (15, 2)),
              np.random.RandomState(12).normal(3, .05, (15, 2))]
    scc = NNFSTPaper._scc_from_mutual_knn(X, 3)
    assert len(np.unique(scc)) >= 2
    assert set(scc[:15]).isdisjoint(set(scc[15:]))


def test_fit_predict_and_novelty_scores():
    X, y = make_blobs(n_samples=135, centers=[[-4, -4], [0, 4], [5, -2]],
                      cluster_std=.65, random_state=19)
    for kernel in ['rbf', 'l05_exponential_kernel']:
        m = NNFSTPaper(r=5, gamma=.8, kernel=kernel, max_eigensize=150)
        m.fit(X, y)
        pred, scores = m.predict_with_novelty(X)
        assert len(pred) == len(X)
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0)
        assert m.n_projection_dims_ >= 1
        assert m.n_components_ >= 3
        assert m.within_null_relative_error_ < 1e-4
        assert matthews_corrcoef(y, pred) >= .9


def test_guard_against_accidental_large_matrix():
    X = np.random.default_rng(42).normal(size=(52, 3))
    y = np.repeat([0, 1], 26)
    try:
        NNFSTPaper(max_eigensize=30).fit(X, y)
        assert False, 'Expected MemoryError'
    except MemoryError as e:
        assert 'PILOT' in str(e)
