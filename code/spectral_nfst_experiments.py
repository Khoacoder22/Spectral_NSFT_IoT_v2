"""Shared experiment functions for Spectral NFST.

This module contains orchestration only.  The Spectral NFST mathematics remains
in ``cpai.models.SpectralNFST``.  Two public functions deliberately separate the
two research questions:

``run_spectral_nfst_classification``
    Closed-set multiclass classification. Every class may appear in training.

``run_spectral_nfst_multinovelty``
    Multiclass novelty detection. One or more selected classes are removed from
    training and represented by label -1 in the mixed known/unknown test set.

The novelty function accepts an externally selected threshold or calibrates
it on held-out known training samples. It never tunes the threshold on test
labels, avoiding the test leakage present in the legacy notebook.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

from cpai import SpectralNFST
from cpai.datasets import load_dataset
from cpai.metrics import evaluate
from cpai.paths import RESULTS_DIR
from cpai.preprocessing import _remove_outliers_lof, _scaler
from cpai.gaussian_elim import form_independent
from train_evaluate import append_row, run_one


# Ten registered CSV variants. ToN-IoT and IoTID20 each have two variants.
DATASET_VARIANTS = (
    ("BoT_IoT", 1000),
    ("CIC_IoT2023", 1000),
    ("ToN_IoT", 1000),
    ("ToN_IoT", 2000),
    ("UNSW_NB15", 1000),
    ("IoTID20", 1000),
    ("IoTID20", 2000),
    ("N_BaIoT", 1000),
    ("Edge_IIoTset", 1000),
    ("5G_NIDD", 1000),
)


def _sample_per_class(
    df: pd.DataFrame, samples_per_class: int | None, seed: int
) -> pd.DataFrame:
    """Return a deterministic class-balanced cap without changing source data.

    ``None`` keeps every row. Otherwise, each class contributes at most the
    requested number of rows. This helper is only a memory/runtime control for
    smoke tests; it does not modify the model algorithm.
    """
    if samples_per_class is None:
        return df
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2")
    sampled = [
        group.sample(n=min(samples_per_class, len(group)), random_state=seed)
        for _, group in df.groupby("Label", sort=True)
    ]
    return pd.concat(sampled, ignore_index=True)


def _prepare_multinovelty_data(
    df: pd.DataFrame,
    dataset: str,
    unknown_labels: Iterable,
    poly: int,
    kernel: str,
    scaler: str,
    seed: int,
):
    """Split known training data again to create an isolated calibration set."""
    unknown_labels = np.asarray(list(unknown_labels), dtype=object)
    if unknown_labels.size == 0:
        raise ValueError("At least one unknown label is required")

    data = df.to_numpy()
    X = data[:, :-1].astype(np.float64)
    y_raw = data[:, -1]

    all_labels = np.unique(y_raw)
    missing = [label for label in unknown_labels if label not in all_labels]
    if missing:
        raise ValueError(f"Unknown labels are not present in {dataset}: {missing}")
    if len(unknown_labels) >= len(all_labels):
        raise ValueError("At least one known class must remain for training")

    # Keep the final test split untouched by model fitting and calibration.
    X_train_all, X_test, y_train_all, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, stratify=y_raw, random_state=seed
    )

    # Unknown classes are excluded from training and calibration.
    known_mask = ~np.isin(y_train_all, unknown_labels)
    X_known = X_train_all[known_mask]
    y_known = y_train_all[known_mask]

    # Reserve known samples for threshold calibration.
    X_train, X_cal, y_train_raw, _ = train_test_split(
        X_known,
        y_known,
        test_size=0.2,
        stratify=y_known,
        random_state=seed,
    )

    X_train = X_train.copy()
    X_cal = X_cal.copy()
    X_test = X_test.copy()

    # Clean non-finite values; fit the imputer on training data only.
    for part in (X_train, X_cal, X_test):
        part[~np.isfinite(part)] = np.nan

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_cal = imputer.transform(X_cal)
    X_test = imputer.transform(X_test)

    # Fit feature expansion on training data and reuse it for calibration/test.
    if poly > 1:
        poly_transform = PolynomialFeatures(poly, interaction_only=True)
        X_train = poly_transform.fit_transform(X_train)
        X_cal = poly_transform.transform(X_cal)
        X_test = poly_transform.transform(X_test)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    known_lookup = {
        label: index for index, label in enumerate(encoder.classes_)
    }
    y_test = np.asarray(
        [known_lookup.get(label, -1) for label in y_test_raw], dtype=int
    )

    fitted_scaler = _scaler(scaler, random_state=seed)
    fitted_scaler.fit(X_train)

    def transform(part):
        return np.nan_to_num(
            fitted_scaler.transform(part),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    X_train = transform(X_train)
    X_cal = transform(X_cal)
    X_test = transform(X_test)

    X_train, y_train = _remove_outliers_lof(X_train, y_train)
    if poly != -1:
        X_train, y_train = form_independent(X_train, y_train, kernel)

    return X_train, y_train, X_cal, X_test, y_test, encoder


def _novelty_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, anomaly_scores: np.ndarray
) -> dict:
    """Compute open-set multiclass and binary novelty metrics.

    The continuous nearest-manifold distance is used for AUROC/AUPRC. NDR is
    novelty recall at the supplied operating threshold. FPR_N is the proportion
    of known test samples incorrectly rejected as novelty.
    """
    y_pred = np.asarray(y_pred, dtype=int)
    open_set = evaluate(y_true, y_pred)
    true_novel = y_true == -1
    pred_novel = y_pred == -1
    if not np.any(true_novel) or np.all(true_novel):
        raise ValueError("The test set must contain both known and unknown samples")

    ndr = recall_score(true_novel, pred_novel, zero_division=0) * 100.0
    known_mask = ~true_novel
    fpr_n = float(np.mean(pred_novel[known_mask]) * 100.0)
    auc_n = roc_auc_score(true_novel.astype(int), anomaly_scores) * 100.0
    auprc_n = average_precision_score(true_novel.astype(int), anomaly_scores) * 100.0
    return {
        "MCC": open_set["MCC"],
        "ACC": open_set["ACC"],
        "TPR Macro": open_set["TPR Macro"],
        "FPR": open_set["FPR"],
        "PPV Macro": open_set["PPV Macro"],
        "F1 Macro": open_set["F1 Macro"],
        "NDR": ndr,
        "FPR_N": fpr_n,
        "AUC_N": auc_n,
        "AUPRC_N": auprc_n,
        "CFS Matrix": open_set["CFS Matrix"],
    }


def _append_novelty_row(row: dict, output: Path) -> None:
    """Append one novelty result while keeping unknown labels JSON-readable."""
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(row)
    serializable["Unknown labels"] = json.dumps(row["Unknown labels"], ensure_ascii=False)
    serializable["CFS Matrix"] = np.array2string(row["CFS Matrix"])
    pd.DataFrame([serializable]).to_csv(
        output, mode="a", header=not output.exists(), index=False
    )


def run_spectral_nfst_classification(
    *,
    kernel: str = "rbf",
    scaler: str = "QuantileTransformer",
    poly: int = -1,
    components_per_class: int = 2,
    samples_per_class: int | None = None,
    seed: int = 42,
    output: Path | None = None,
    variants=None,
) -> list[dict]:
    """Run closed-set Spectral NFST classification on dataset variants."""
    if kernel == "none":
        raise ValueError("SpectralNFST requires a kernel; use rbf (recommended)")
    output = output or RESULTS_DIR / "spectral_nfst" / "classification.csv"
    variants = DATASET_VARIANTS if variants is None else tuple(variants)
    results: list[dict] = []

    for index, (dataset, limit) in enumerate(variants, start=1):
        variant = f"{dataset}_{limit}"
        print(f"[{index:02d}/{len(variants):02d}] {variant}: classification", flush=True)
        try:
            row = run_one(
                dataset=dataset,
                model_name="SpectralNFST",
                kernel=kernel,
                scaler=scaler,
                poly=poly,
                seed=seed,
                components_per_class=components_per_class,
                limit=limit,
                samples_per_class=samples_per_class,
            )
            append_row(row, output)
            results.append({"_status": "ok", **row})

            train_t = row.get("Train(s)", row.get("Training time", 0.0))
            test_t = row.get("Test(s)", row.get("Test time", 0.0))

            print(
                f"          MCC={row['MCC']:.4f} ACC={row['ACC']:.2f} "
                f"F1={row['F1 Macro']:.2f} Train={train_t:.2f}s Test={test_t:.4f}s",
                flush=True,
            )
        except Exception as exc:
            results.append({
                "_status": "error", "Data Type": variant,
                "_error": f"{type(exc).__name__}: {exc}",
            })
            print(f"          FAILED: {type(exc).__name__}: {exc}", flush=True)
    return results


def run_spectral_nfst_multinovelty(
    *,
    dataset: str,
    unknown_labels: Iterable,
    threshold: float | None = None,
    target_fpr: float = 0.01,
    limit: int | None = None,
    kernel: str = "rbf",
    scaler: str = "QuantileTransformer",
    poly: int = -1,
    components_per_class: int = 2,
    samples_per_class: int | None = None,
    seed: int = 42,
    output: Path | None = None,
) -> dict:
    """Run novelty detection with a fixed or calibration-derived threshold."""
    if kernel == "none":
        raise ValueError("SpectralNFST requires a kernel; use rbf (recommended)")
    if threshold is not None and threshold < 0:
        raise ValueError("threshold must be non-negative")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be between 0 and 1")

    df, resolved_limit = load_dataset(dataset, limit=limit)
    df = _sample_per_class(df, samples_per_class, seed)
    unknown_labels = list(unknown_labels)

    X_train, y_train, X_cal, X_test, y_test, encoder = (
        _prepare_multinovelty_data(
            df, dataset, unknown_labels, poly, kernel, scaler, seed
        )
    )

    model = SpectralNFST(
        n_components=components_per_class,
        kernel=kernel,
        novelty_threshold=threshold,
        novelty_label=-1,
    )

    started = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - started

    if threshold is None:
        cal_scores = model.anomaly_score(X_cal)
        threshold = float(np.quantile(cal_scores, 1.0 - target_fpr))

    started = time.perf_counter()
    y_pred, anomaly_scores = model.predict_with_novelty(
        X_test, threshold=threshold
    )
    test_time = time.perf_counter() - started

    metrics = _novelty_metrics(y_test, y_pred, anomaly_scores)

    row = {
        "Data Type": f"{dataset}_{resolved_limit}",
        "Task": "SpectralNFST-MultiNovelty",
        "Unknown labels": [str(label) for label in unknown_labels],
        "Known labels": [str(label) for label in encoder.classes_],
        "Threshold": threshold,
        "Target FPR": target_fpr,
        "Components per class": components_per_class,
        "Kernel": kernel,
        "SCALER": scaler,
        "Poly": poly,
        "Seed": seed,
        "N Train": len(y_train),
        "N Test": len(y_test),
        "Training time": training_time,
        "Test time": test_time,
        **metrics,
    }

    if output is not None:
        _append_novelty_row(row, output)
    return row


def dataset_class_labels(dataset: str, limit: int | None = None) -> list:
    """List cleaned original labels so CLI users can choose unknown classes."""
    df, _ = load_dataset(dataset, limit=limit)
    return np.unique(df["Label"].to_numpy()).tolist()