from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import csv
import os
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "code" / "run_spectral_nfst_novelty.py"
WORKERS = 2
SEEDS = (42, 43, 44)

DATASETS = (
    ("5G_NIDD", 1000, "QuantileTransformer", "0.01", 2, -1),
    ("BoT_IoT", 1000, "QuantileTransformer", "0.01", 2, -1),
    ("CIC_IoT2023", 1000, "QuantileTransformer", "0.1", 3, -1),
    ("Edge_IIoTset", 1000, "QuantileTransformer", "0.001", 5, -1),
    ("IoTID20", 1000, "QuantileTransformer", "0.01", 2, -1),
    ("IoTID20", 2000, "QuantileTransformer", "0.001", 5, -1),
    ("N_BaIoT", 1000, "QuantileTransformer", "0.001", 2, -1),
    ("ToN_IoT", 1000, "StandardScaler", "0.01", 3, 0),
    ("ToN_IoT", 2000, "QuantileTransformer", "0.001", 2, -1),
    ("UNSW_NB15", 1000, "QuantileTransformer", "0.1", 4, 0),
)

env = os.environ.copy()
for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    env[name] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = ROOT / "results" / f"table8_spectral_parallel_{stamp}"
runs_dir = out_dir / "individual_runs"
logs_dir = out_dir / "logs"
runs_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)


def get_attack_labels(name, limit):
    command = [
        sys.executable,
        str(CLI),
        "--dataset",
        name,
        "--limit",
        str(limit),
        "--list-classes",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not read labels for {name}_{limit}:\n{result.stderr}"
        )

    labels = []
    for line in result.stdout.splitlines():
        match = re.match(r"^\[\d+\]\s+(.+)$", line.strip())
        if match:
            label = match.group(1).strip()
            if not re.search(r"normal|benign", label, flags=re.IGNORECASE):
                labels.append(label)
    return labels


def run_one(task):
    command = [
        sys.executable,
        str(CLI),
        "--dataset",
        task["name"],
        "--limit",
        str(task["limit"]),
        "--unknown-labels",
        task["label"],
        "--target-fpr",
        "0.05",
        "--kernel",
        "l05_exponential_kernel",
        "--gamma",
        task["gamma"],
        "--scaler",
        task["scaler"],
        "--components-per-class",
        str(task["q"]),
        "--poly",
        str(task["poly"]),
        "--seed",
        str(task["seed"]),
        "--output",
        str(task["output"]),
    ]

    with task["log"].open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return task, result.returncode


def main():
    if not CLI.exists():
        raise FileNotFoundError(f"CLI script not found: {CLI}")

    tasks = []
    for name, limit, scaler, gamma, q, poly in DATASETS:
        variant = f"{name}_{limit}"
        print(f"Reading labels for {variant}...", flush=True)

        for label in get_attack_labels(name, limit):
            for seed in SEEDS:
                run_id = f"{len(tasks) + 1:05d}"
                tasks.append(
                    {
                        "name": name,
                        "limit": limit,
                        "variant": variant,
                        "label": label,
                        "seed": seed,
                        "scaler": scaler,
                        "gamma": gamma,
                        "q": q,
                        "poly": poly,
                        "output": runs_dir / f"run_{run_id}.csv",
                        "log": logs_dir / f"run_{run_id}.log",
                    }
                )

    print(
        f"Total runs: {len(tasks)}. Running up to {WORKERS} at a time.",
        flush=True,
    )

    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            task, code = future.result()
            status = "OK" if code == 0 else "FAILED"
            print(
                f"[{completed}/{len(tasks)}] {status}: "
                f"{task['variant']} {task['label']} seed={task['seed']}",
                flush=True,
            )
            if code != 0:
                failures.append(task)

    if failures:
        print("Some runs failed. Check these log files:")
        for task in failures:
            print(task["log"])
        raise SystemExit(1)

    summary = []
    for name, limit, *_ in DATASETS:
        variant = f"{name}_{limit}"
        variant_tasks = [task for task in tasks if task["variant"] == variant]
        rows = []

        for task in variant_tasks:
            with task["output"].open(
                "r", newline="", encoding="utf-8-sig"
            ) as file:
                rows.extend(csv.DictReader(file))

        if not rows:
            continue

        combined_path = out_dir / f"{variant}.csv"
        with combined_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        means_by_seed = []
        for seed in SEEDS:
            scores = [
                float(row["AUC_N"])
                for row in rows
                if int(row["Seed"]) == seed
            ]
            if scores:
                means_by_seed.append(statistics.mean(scores))

        mean_auc = statistics.mean(means_by_seed)
        sd_auc = (
            statistics.stdev(means_by_seed)
            if len(means_by_seed) > 1
            else 0.0
        )
        summary.append(
            {
                "Dataset": variant,
                "Runs": len(rows),
                "AUROC_Mean": round(mean_auc, 2),
                "AUROC_SD": round(sd_auc, 2),
            }
        )

    summary_path = out_dir / "AUROC_summary.csv"
    with summary_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Dataset", "Runs", "AUROC_Mean", "AUROC_SD"],
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nFinished. AUROC summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()