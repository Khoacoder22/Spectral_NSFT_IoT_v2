from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

THIS_FILE = Path(__file__).resolve()

PROJECT_ROOT = None
for candidate in [THIS_FILE.parent, *THIS_FILE.parents]:
    if (candidate / "code" / "tune_spectral_nfst.py").exists():
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Cannot locate project root containing "
        "code/tune_spectral_nfst.py"
    )

TUNE_SCRIPT = (
    PROJECT_ROOT
    / "code"
    / "tune_spectral_nfst.py"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "spectral_nfst_result"
)

LOG_FILE = RESULT_ROOT / "run.log"


# ============================================================
# Full grid
# ============================================================

DATASETS = [
    ("5G_NIDD", 1000),
    ("BoT_IoT", 1000),
    ("CIC_IoT2023", 1000),
    ("Edge_IIoTset", 1000),
    ("IoTID20", 1000),
    ("IoTID20", 2000),
    ("N_BaIoT", 1000),
    ("ToN_IoT", 1000),
    ("ToN_IoT", 2000),
    ("UNSW_NB15", 1000),
]

SCALERS = [
    "QuantileTransformer",
    "StandardScaler",
    "RobustScaler",
]

QS = [2, 3, 4, 5]
SEEDS = [42, 43, 44]


# ============================================================
# Helpers
# ============================================================

def log_print(message: str = "") -> None:
    print(message)

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as log:
        log.write(message + "\n")


def run_and_tee(
    command: list[str],
) -> int:
    """
    Run tune_spectral_nfst.py and mirror its stdout/stderr
    to terminal + run.log.
    """
    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as log:

        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(
                line,
                end="",
            )

            log.write(line)
            log.flush()

        return process.wait()


def print_job_results(
    output: Path,
) -> None:
    """
    Print current CSV result table, including Training time
    and Test time.
    """
    if not output.exists():
        log_print(
            f"[WARNING] Output CSV not found: {output}"
        )
        return

    try:
        df = pd.read_csv(output)
    except Exception as exc:
        log_print(
            f"[WARNING] Cannot read {output.name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    if df.empty:
        log_print(
            f"[WARNING] Empty CSV: {output.name}"
        )
        return

    sort_cols = [
        col
        for col in ("MCC", "F1 Macro", "ACC")
        if col in df.columns
    ]

    if sort_cols:
        df = df.sort_values(
            sort_cols,
            ascending=[False] * len(sort_cols),
        )

    columns = [
        col
        for col in (
            "Q",
            "Seed",
            "Feature",
            "Gamma",
            "MCC",
            "F1 Macro",
            "ACC",
            "FPR",
            "Training time",
            "Test time",
        )
        if col in df.columns
    ]

    display_df = df[columns].copy()

    formats = {
        "MCC": "{:.6f}",
        "F1 Macro": "{:.4f}",
        "ACC": "{:.4f}",
        "FPR": "{:.4f}",
        "Training time": "{:.4f}",
        "Test time": "{:.6f}",
    }

    for col, fmt in formats.items():
        if col not in display_df.columns:
            continue

        values = pd.to_numeric(
            display_df[col],
            errors="coerce",
        )

        display_df[col] = values.map(
            lambda value: (
                fmt.format(value)
                if pd.notna(value)
                else ""
            )
        )

    log_print("")
    log_print(
        f"RESULT SUMMARY: {output.name}"
    )
    log_print("-" * 120)

    table_text = display_df.to_string(
        index=False,
    )

    print(table_text)

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as log:
        log.write(
            table_text + "\n"
        )

    best = df.iloc[0]

    parts = []

    for col in (
        "Q",
        "Seed",
        "Feature",
        "Gamma",
        "MCC",
        "F1 Macro",
        "ACC",
        "FPR",
        "Training time",
        "Test time",
    ):
        if col in df.columns:
            parts.append(
                f"{col}={best[col]}"
            )

    log_print("")
    log_print("BEST IN THIS JOB:")
    log_print(
        "  " + ", ".join(parts)
    )
    log_print("-" * 120)


# ============================================================
# Main
# ============================================================

def main() -> None:
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TUNE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Missing script: {TUNE_SCRIPT}"
        )

    total_jobs = (
        len(DATASETS)
        * len(SCALERS)
        * len(SEEDS)
        * len(QS)
        - len(DATASETS)
    )

    job_index = 0

    log_print("=" * 72)
    log_print("SPECTRAL NFST - L05 FULL GRID")
    log_print("=" * 72)

    log_print(
        f"Project root : {PROJECT_ROOT}"
    )
    log_print(
        f"Tune script  : {TUNE_SCRIPT}"
    )
    log_print(
        f"Result folder: {RESULT_ROOT}"
    )
    log_print(
        f"Total jobs   : {total_jobs}"
    )

    log_print("Kernel: l05_exponential_kernel")
    log_print(
        f"Scalers      : {', '.join(SCALERS)}"
    )
    log_print(
        f"Q            : {QS}"
    )
    log_print(
        f"Seeds        : {SEEDS}"
    )
    log_print("Feature: -1, 0")
    log_print("Gamma : heuristic, 0.001, 0.01, 0.1, 1, 10")

    log_print("=" * 72)

    for dataset, limit in DATASETS:

        dataset_tag = (
            f"{dataset}_{limit}"
        )

        for seed in SEEDS:
            for scaler in SCALERS:
                #  ignore q1 and q2 for seed 42 to avoid duplicate results
                for q in QS:
                    if seed == 42 and q == 2 and scaler == "QuantileTransformer":
                        continue

                    job_index += 1

                    output = (
                        RESULT_ROOT
                        / (
                            f"{dataset_tag}_"
                            f"{scaler}_"
                            f"Q{q}_"
                            f"seed{seed}.csv"
                        )
                    )

                    log_print("")
                    log_print("=" * 72)

                    log_print(
                        f"[{job_index}/{total_jobs}]"
                    )
                    log_print(
                        f"Dataset = {dataset_tag}"
                    )
                    log_print(
                        "Kernel  = l05_exponential_kernel"
                    )
                    log_print(
                        f"Scaler  = {scaler}"
                    )
                    log_print(
                        f"Q       = {q}"
                    )
                    log_print(
                        f"Seed    = {seed}"
                    )
                    log_print(
                        f"Output  = {output}"
                    )

                    log_print("=" * 72)

                    command = [
                        sys.executable,
                        str(TUNE_SCRIPT),
                        "--dataset",
                        dataset,
                        "--limit",
                        str(limit),
                        "--scaler",
                        scaler,
                        "--q",
                        str(q),
                        "--seed",
                        str(seed),
                        "--output",
                        str(output),
                    ]

                    wall_start = (
                        time.perf_counter()
                    )

                    exit_code = run_and_tee(
                        command
                    )

                    wall_time = (
                        time.perf_counter()
                        - wall_start
                    )

                    log_print("")
                    log_print(
                        f"JOB WALL TIME: "
                        f"{wall_time:.2f}s "
                        f"({wall_time / 60:.2f} min)"
                    )

                    print_job_results(
                        output
                    )

                    if exit_code != 0:
                        log_print(
                            "[WARNING] Job exited "
                            f"with code {exit_code}: "
                            f"{dataset_tag}, "
                            f"scaler={scaler}, "
                            f"Q={q}, "
                            f"seed={seed}"
                        )

                        log_print(
                            "Continuing with "
                            "the next job..."
                        )

    log_print("")
    log_print("=" * 72)
    log_print(
        "FULL GRID FINISHED"
    )
    log_print(
        f"Results: {RESULT_ROOT}"
    )
    log_print(
        f"Log    : {LOG_FILE}"
    )
    log_print("=" * 72)


if __name__ == "__main__":
    main()
