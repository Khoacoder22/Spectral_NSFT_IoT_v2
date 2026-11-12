from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

# Allow running as: py code/tune_spectral_nfst.py ...
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpai import SpectralNFST
from cpai.datasets import DATASETS, load_dataset
from cpai.metrics import evaluate
from cpai.preprocessing import preprocess


# ============================================================
# Fixed experiment settings
# ============================================================

KERNEL = "l05_exponential_kernel"

SCALERS = (
    "QuantileTransformer",
    "StandardScaler",
    "RobustScaler",
)

FEATURES = (-1, 0)

GAMMAS = (
    ("heuristic", "heuristic"),
    ("0.001", 0.001),
    ("0.01", 0.01),
    ("0.1", 0.1),
    ("1", 1.0),
    ("10", 10.0),
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# Helpers
# ============================================================

def normalize_text(value) -> str:
    """Normalize CSV values for resume/cache keys."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def make_completed_key(
    feature,
    gamma,
    q,
    seed,
    scaler,
) -> tuple[str, str, str, str, str]:
    return (
        normalize_text(feature),
        normalize_text(gamma),
        normalize_text(q),
        normalize_text(seed),
        normalize_text(scaler),
    )


def load_completed(
    output: Path,
    error_output: Path,
) -> set[tuple[str, str, str, str, str]]:
    """
    Load completed configurations from success and error CSVs.

    Full key includes Feature + Gamma + Q + Seed + Scaler so a CSV cannot
    accidentally cause another Q/seed/scaler job to be skipped.
    """
    completed: set[tuple[str, str, str, str, str]] = set()

    for path in (output, error_output):
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as exc:
            print(
                f"[WARNING] Cannot read cache file {path}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        required = {"Feature", "Gamma", "Q", "Seed", "Scaler"}
        if not required.issubset(df.columns):
            print(
                f"[WARNING] Cache file has old/incomplete schema and will not "
                f"be used for skipping: {path}"
            )
            continue

        for _, row in df.iterrows():
            completed.add(
                make_completed_key(
                    row["Feature"],
                    row["Gamma"],
                    row["Q"],
                    row["Seed"],
                    row["Scaler"],
                )
            )

    return completed


def append_csv_row(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def print_summary(output: Path) -> None:
    if not output.exists():
        return

    results = pd.read_csv(output)

    if results.empty:
        return

    sort_cols = [
        col for col in ("MCC", "F1 Macro", "ACC")
        if col in results.columns
    ]
    if sort_cols:
        results = results.sort_values(
            sort_cols,
            ascending=[False] * len(sort_cols),
        )

    print(f"\nSaved: {output}")

    display_columns = [
        col
        for col in (
            "Seed",
            "Scaler",
            "Q",
            "Feature",
            "Gamma",
            "MCC",
            "F1 Macro",
            "ACC",
            "FPR",
            "Training time",
            "Test time",
        )
        if col in results.columns
    ]

    print(results[display_columns].to_string(index=False))


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune SpectralNFST Feature x Gamma for one "
            "dataset/scaler/Q/seed configuration."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS,
    )

    parser.add_argument(
        "--limit",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--scaler",
        default="QuantileTransformer",
        choices=SCALERS,
    )

    parser.add_argument(
        "--q",
        type=int,
        default=2,
    )

    # IMPORTANT: seed is now configurable.
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help="0 = full data; otherwise sample N rows per class.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("limit must be at least 1")

    if args.q < 1:
        parser.error("q must be at least 1")

    if args.samples_per_class == 1 or args.samples_per_class < 0:
        parser.error(
            "samples-per-class must be 0 (full data) or at least 2"
        )

    seed = args.seed

    sample_tag = (
        "full"
        if args.samples_per_class == 0
        else f"sample{args.samples_per_class}"
    )

    output = args.output or (
        RESULTS_DIR
        / "spectral_nfst"
        / (
            f"gamma_{args.dataset}_{args.limit}_"
            f"{sample_tag}_{args.scaler}_Q{args.q}_seed{seed}.csv"
        )
    )

    if not output.is_absolute():
        output = PROJECT_ROOT / output

    error_output = output.with_name(
        f"{output.stem}_errors.csv"
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed(output, error_output)

    print("=" * 72)
    print("SPECTRAL NFST TUNING")
    print("=" * 72)
    print(f"Dataset : {args.dataset}_{args.limit}")
    print(f"Kernel  : {KERNEL}")
    print(f"Scaler  : {args.scaler}")
    print(f"Q       : {args.q}")
    print(f"Seed    : {seed}")
    print(f"Output  : {output}")
    print("=" * 72)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    df, _ = load_dataset(
        args.dataset,
        limit=args.limit,
    )

    if args.samples_per_class:
        sampled_groups = []

        for _, group in df.groupby("Label", sort=True):
            sampled_groups.append(
                group.sample(
                    n=min(
                        args.samples_per_class,
                        len(group),
                    ),
                    random_state=seed,
                )
            )

        df = pd.concat(
            sampled_groups,
            ignore_index=True,
        )

    total = len(GAMMAS) * len(FEATURES)
    position = 0

    # --------------------------------------------------------
    # Feature x Gamma grid
    # --------------------------------------------------------
    for feature in FEATURES:
        # Preprocess once for this feature.
        X_train, y_train, X_test, y_test, _ = preprocess(
            df,
            args.dataset,
            poly=feature,
            kernel=KERNEL,
            scaler=args.scaler,
            seed=seed,
        )

        for gamma_label, gamma in GAMMAS:
            position += 1

            key = make_completed_key(
                feature,
                gamma_label,
                args.q,
                seed,
                args.scaler,
            )

            if key in completed:
                print(
                    f"[{position}/{total}] SKIP "
                    f"feature={feature}, gamma={gamma_label}"
                )
                continue

            print(
                f"[{position}/{total}] RUN  "
                f"feature={feature}, gamma={gamma_label}"
            )

            try:
                model = SpectralNFST(
                    n_components=args.q,
                    kernel=KERNEL,
                    gamma=gamma,
                )

                # --------------------
                # Training time
                # --------------------
                started = time.perf_counter()

                # Suppress model debug output that can break some Windows
                # consoles, but still measure the real fit time.
                with redirect_stdout(io.StringIO()):
                    model.fit(X_train, y_train)

                train_time = time.perf_counter() - started

                # --------------------
                # Test / prediction time
                # --------------------
                started = time.perf_counter()
                prediction = model.predict(X_test)
                test_time = time.perf_counter() - started

                metrics = evaluate(
                    y_test,
                    prediction,
                )

            except Exception as exc:
                error_row = {
                    "Dataset": f"{args.dataset}_{args.limit}",
                    "Kernel": KERNEL,
                    "Gamma": gamma_label,
                    "Feature": feature,
                    "Scaler": args.scaler,
                    "Q": args.q,
                    "Seed": seed,
                    "Error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }

                append_csv_row(
                    error_output,
                    error_row,
                )

                completed.add(key)

                print(
                    f"    FAILED: {error_row['Error']}; "
                    f"continuing"
                )
                continue

            row = {
                "Dataset": f"{args.dataset}_{args.limit}",
                "Kernel": KERNEL,
                "Gamma": gamma_label,
                "Gamma used": getattr(
                    model,
                    "_gamma_used",
                    gamma,
                ),
                "Feature": feature,
                "Scaler": args.scaler,
                "Q": args.q,
                "Seed": seed,
                "MCC": metrics["MCC"],
                "ACC": metrics["ACC"],
                "F1 Macro": metrics["F1 Macro"],
                "TPR Macro": metrics["TPR Macro"],
                "FPR": metrics["FPR"],
                "PPV Macro": metrics["PPV Macro"],
                "Training time": train_time,
                "Test time": test_time,
            }

            append_csv_row(
                output,
                row,
            )

            completed.add(key)

            # Show metrics + times immediately in terminal.
            print(
                f"    MCC={row['MCC']:.6f}, "
                f"F1={row['F1 Macro']:.4f}, "
                f"ACC={row['ACC']:.4f}, "
                f"FPR={row['FPR']:.4f}, "
                f"Train={train_time:.4f}s, "
                f"Test={test_time:.6f}s"
            )

    print_summary(output)

    if error_output.exists():
        try:
            errors = pd.read_csv(error_output)
            if not errors.empty:
                print(
                    f"\nFailed configurations: "
                    f"{error_output}"
                )
        except Exception:
            print(
                f"\nFailed configurations: "
                f"{error_output}"
            )


if __name__ == "__main__":
    main()
