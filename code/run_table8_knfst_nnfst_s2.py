from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from cpai.datasets import load_dataset
from cpai.kernels import compute_kernel
from cpai.models import _learn_knfst
from nnfst_paper import NNFSTPaper
from spectral_nfst_experiments import _prepare_multinovelty_data


DATASETS = [
    "BoT_IoT",
    "IoTID20",
    "ToN_IoT",
    "N_BaIoT",
    "CIC_IoT2023",
    "Edge_IIoTset",
    "5G_NIDD",
    "UNSW_NB15",
]

SEEDS = [42, 43, 44]
TARGET_FPR = 0.05
KNFST_GAMMAS = [0.001, 0.01, 0.1]
KNFST_TUNING_LIMIT = 800

BEST_PARAMS_FILE = (
    ROOT / "results" / "nnfst_competitor16_reimpl" / "best_params.csv"
)
OUTPUT_FILE = ROOT / "results" / "table8_knfst_nnfst_s2" / "all_runs.csv"
SUMMARY_FILE = ROOT / "results" / "table8_knfst_nnfst_s2" / "best_by_dataset.csv"

BENIGN_LABELS = {
    "0Normal",
    "Normal",
    "normal",
    "Benign",
    "BENIGN",
    "BenignTraffic",
    "Normal_Normal",
}

NOVEL_LABEL = -1


def fit_knfst(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str,
    gamma: float,
) -> dict:
    gram = compute_kernel(X, kernel=kernel, gamma=gamma)
    projection, centroids = _learn_knfst(gram, y)

    return {
        "X_train": X,
        "y_classes": np.unique(y),
        "projection": projection,
        "centroids": centroids,
        "kernel": kernel,
        "gamma": gamma,
    }


def predict_knfst(
    model: dict,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cross_kernel = compute_kernel(
        model["X_train"],
        X,
        kernel=model["kernel"],
        gamma=model["gamma"],
    )

    projected = (cross_kernel.T @ model["projection"]).real

    distances = np.linalg.norm(
        projected[:, None, :] - model["centroids"][None, :, :],
        axis=2,
    )

    nearest = distances.argmin(axis=1)
    predicted_labels = model["y_classes"][nearest]
    novelty_scores = distances.min(axis=1)

    return predicted_labels.astype(int), novelty_scores


def choose_knfst_gamma(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    kernel: str,
    seed: int,
) -> float:
    """Select KNFST gamma using known validation labels."""
    if len(y_train) > KNFST_TUNING_LIMIT:
        tune_indices, _ = train_test_split(
            np.arange(len(y_train)),
            train_size=KNFST_TUNING_LIMIT,
            stratify=y_train,
            random_state=seed,
        )
        X_tune = X_train[tune_indices]
        y_tune = y_train[tune_indices]
    else:
        X_tune = X_train
        y_tune = y_train

    best_gamma = KNFST_GAMMAS[0]
    best_mcc = -np.inf

    for gamma in KNFST_GAMMAS:
        candidate = fit_knfst(X_tune, y_tune, kernel, gamma)
        validation_pred, _ = predict_knfst(candidate, X_validation)
        score = matthews_corrcoef(y_validation, validation_pred)

        if score > best_mcc:
            best_mcc = score
            best_gamma = gamma

    return best_gamma


def metrics_at_calibrated_threshold(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    test_known_predictions: np.ndarray,
    calibration_scores: np.ndarray,
) -> tuple[float, float, float]:
    threshold = float(
        np.quantile(calibration_scores, 1.0 - TARGET_FPR)
    )

    predictions = test_known_predictions.astype(int).copy()
    predictions[test_scores > threshold] = NOVEL_LABEL

    is_novel = (y_test == NOVEL_LABEL).astype(int)

    auroc = float(roc_auc_score(is_novel, test_scores) * 100.0)
    os_f1 = float(
        f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
        * 100.0
    )

    return auroc, os_f1, threshold


def append_row(row: dict) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([row]).to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not OUTPUT_FILE.exists(),
        index=False,
        encoding="utf-8",
    )


def read_completed_runs() -> set[tuple[str, str, str, int]]:
    if not OUTPUT_FILE.exists():
        return set()

    old = pd.read_csv(OUTPUT_FILE)

    return {
        (
            str(row["Model"]),
            str(row["Data Type"]),
            str(row["Unknown label"]),
            int(row["Seed"]),
        )
        for _, row in old.iterrows()
    }


def summarize_best_observed() -> None:
    if not OUTPUT_FILE.exists():
        return

    runs = pd.read_csv(OUTPUT_FILE)
    summary = []

    for dataset in [f"{name}_1000" for name in DATASETS]:
        for model in ["KNFST", "nNFST"]:
            group = runs[
                (runs["Data Type"] == dataset)
                & (runs["Model"] == model)
            ]

            if group.empty:
                continue

            best_auc = group.loc[group["AUROC"].idxmax()]
            best_f1 = group.loc[group["OS-F1"].idxmax()]

            summary.append(
                {
                    "Data Type": dataset,
                    "Model": model,
                    "AUROC": round(float(best_auc["AUROC"]), 2),
                    "AUROC Unknown label": best_auc["Unknown label"],
                    "AUROC Seed": int(best_auc["Seed"]),
                    "OS-F1": round(float(best_f1["OS-F1"]), 2),
                    "OS-F1 Unknown label": best_f1["Unknown label"],
                    "OS-F1 Seed": int(best_f1["Seed"]),
                    "Runs": len(group),
                }
            )

    result = pd.DataFrame(summary)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    print("\nBest observed AUROC and OS-F1:")
    print(result.to_string(index=False))
    print(f"\nSummary CSV: {SUMMARY_FILE}")


def known_validation_labels(
    df: pd.DataFrame,
    unknown_label: str,
    seed: int,
) -> np.ndarray:
    """Recreate the known calibration labels used by the shared data split."""
    y_raw = df.to_numpy()[:, -1]

    # Match the first split in _prepare_multinovelty_data.
    y_train_all, _ = train_test_split(
        y_raw,
        test_size=0.2,
        stratify=y_raw,
        random_state=seed,
    )

    # Exclude the withheld attack from training and calibration.
    y_known = y_train_all[~np.isin(y_train_all, [unknown_label])]

    # Match the known train/calibration split.
    y_train_raw, y_cal_raw = train_test_split(
        y_known,
        test_size=0.2,
        stratify=y_known,
        random_state=seed,
    )

    encoder = LabelEncoder()
    encoder.fit(y_train_raw)

    return encoder.transform(y_cal_raw).astype(int)


def main() -> None:
    if not BEST_PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"Missing nNFST validation settings: {BEST_PARAMS_FILE}"
        )

    best_params = pd.read_csv(BEST_PARAMS_FILE)

    settings = {
        str(row["Data Type"]): row
        for _, row in best_params.iterrows()
    }

    completed = read_completed_runs()
    total_runs = 0

    for dataset in DATASETS:
        data_type = f"{dataset}_1000"

        if data_type not in settings:
            raise ValueError(
                f"No validation-selected settings for {data_type}"
            )

        params = settings[data_type]

        kernel = str(params["Kernel"])
        scaler = str(params["SCALER"])
        poly = int(params["Poly"])
        n_nf_r = int(params["r"])
        n_nf_gamma = float(params["Gamma"])
        relative_tol = float(params.get("Relative_Tol", 1e-9))

        df, _ = load_dataset(dataset, limit=1000)

        labels = sorted(df["Label"].astype(str).unique().tolist())
        attack_labels = [
            label for label in labels
            if label not in BENIGN_LABELS
        ]

        for unknown_label in attack_labels:
            for seed in SEEDS:
                todo = [
                    model_name
                    for model_name in ["KNFST", "nNFST"]
                    if (
                        model_name,
                        data_type,
                        unknown_label,
                        seed,
                    )
                    not in completed
                ]

                if not todo:
                    continue

                print(
                    f"\n{data_type}, unknown={unknown_label}, seed={seed}",
                    flush=True,
                )

                (
                    X_train,
                    y_train,
                    X_cal,
                    X_test,
                    y_test,
                    _,
                ) = _prepare_multinovelty_data(
                    df=df,
                    dataset=data_type,
                    unknown_labels=[unknown_label],
                    poly=poly,
                    kernel=kernel,
                    scaler=scaler,
                    seed=seed,
                )

                if "KNFST" in todo:
                    y_cal = known_validation_labels(
                        df=df,
                        unknown_label=unknown_label,
                        seed=seed,
                    )

                    gamma = choose_knfst_gamma(
                        X_train=X_train,
                        y_train=y_train,
                        X_validation=X_cal,
                        y_validation=y_cal,
                        kernel=kernel,
                        seed=seed,
                    )

                    knfst = fit_knfst(
                        X_train,
                        y_train,
                        kernel,
                        gamma,
                    )

                    cal_pred, cal_scores = predict_knfst(knfst, X_cal)
                    test_pred, test_scores = predict_knfst(knfst, X_test)

                    auroc, os_f1, threshold = (
                        metrics_at_calibrated_threshold(
                            y_test=y_test,
                            test_scores=test_scores,
                            test_known_predictions=test_pred,
                            calibration_scores=cal_scores,
                        )
                    )

                    row = {
                        "Data Type": data_type,
                        "Model": "KNFST",
                        "Unknown label": unknown_label,
                        "Seed": seed,
                        "Kernel": kernel,
                        "Gamma": gamma,
                        "Scaler": scaler,
                        "Poly": poly,
                        "Target FPR": TARGET_FPR,
                        "Threshold": threshold,
                        "AUROC": auroc,
                        "OS-F1": os_f1,
                    }

                    append_row(row)
                    completed.add(
                        ("KNFST", data_type, unknown_label, seed)
                    )

                    print(
                        f"KNFST: AUROC={auroc:.2f} "
                        f"OS-F1={os_f1:.2f}",
                        flush=True,
                    )

                if "nNFST" in todo:
                    model = NNFSTPaper(
                        r=n_nf_r,
                        gamma=n_nf_gamma,
                        kernel=kernel,
                        relative_tol=relative_tol,
                        max_eigensize=0,
                    )

                    model.fit(X_train, y_train)

                    cal_scores = model.predict_novelty_score(X_cal)
                    threshold = float(
                        np.quantile(cal_scores, 1.0 - TARGET_FPR)
                    )

                    raw_pred, test_scores = model.predict_with_novelty(
                        X_test,
                        threshold=threshold,
                    )

                    test_pred = np.array(
                        [
                            NOVEL_LABEL
                            if str(value) == "__NOVEL__"
                            else int(value)
                            for value in raw_pred
                        ],
                        dtype=int,
                    )

                    is_novel = (y_test == NOVEL_LABEL).astype(int)

                    auroc = float(
                        roc_auc_score(is_novel, test_scores) * 100.0
                    )
                    os_f1 = float(
                        f1_score(
                            y_test,
                            test_pred,
                            average="macro",
                            zero_division=0,
                        )
                        * 100.0
                    )

                    row = {
                        "Data Type": data_type,
                        "Model": "nNFST",
                        "Unknown label": unknown_label,
                        "Seed": seed,
                        "Kernel": kernel,
                        "Gamma": n_nf_gamma,
                        "Scaler": scaler,
                        "Poly": poly,
                        "r": n_nf_r,
                        "Target FPR": TARGET_FPR,
                        "Threshold": threshold,
                        "AUROC": auroc,
                        "OS-F1": os_f1,
                    }

                    append_row(row)
                    completed.add(
                        ("nNFST", data_type, unknown_label, seed)
                    )

                    print(
                        f"nNFST: AUROC={auroc:.2f} "
                        f"OS-F1={os_f1:.2f}",
                        flush=True,
                    )

                total_runs += len(todo)

    print(f"\nNew model runs completed: {total_runs}")
    print(f"Per-run results: {OUTPUT_FILE}")
    summarize_best_observed()


if __name__ == "__main__":
    main()