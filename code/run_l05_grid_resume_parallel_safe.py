#!/usr/bin/env python3
"""Continue the existing SpectralNFST full grid without changing the model.

Run this script from the project root after stopping the old runner.
Examples:
    py code/run_l05_grid_resume_parallel_safe.py --dry-run
    py code/run_l05_grid_resume_parallel_safe.py --workers 1 --blas-threads 4 --max-jobs 1
    py code/run_l05_grid_resume_parallel_safe.py --workers 2 --blas-threads 2

The original code/tune_spectral_nfst.py and cpai/SpectralNFST are NOT modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (p for p in (THIS_FILE.parent, *THIS_FILE.parents)
     if (p / "code" / "tune_spectral_nfst.py").is_file()),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Cannot find code/tune_spectral_nfst.py in this project")

TUNE_SCRIPT = PROJECT_ROOT / "code" / "tune_spectral_nfst.py"
RESULT_DIR = PROJECT_ROOT / "results" / "spectral_nfst_result"
LOG_DIR = RESULT_DIR / "job_logs"
ERROR_ARCHIVE_DIR = LOG_DIR / "old_error_csv"

DATASETS = (
    ("5G_NIDD", 1000), ("BoT_IoT", 1000), ("CIC_IoT2023", 1000),
    ("Edge_IIoTset", 1000), ("IoTID20", 1000), ("IoTID20", 2000),
    ("N_BaIoT", 1000), ("ToN_IoT", 1000), ("ToN_IoT", 2000),
    ("UNSW_NB15", 1000),
)
SCALERS = ("QuantileTransformer", "StandardScaler", "RobustScaler")
QS = (2, 3, 4, 5)  # Q=1 is excluded for ALL seeds.
SEEDS = (42, 43, 44)
GAMMAS = ( "0.001", "0.01", "0.1", "1", "10")
FEATURES = (-1, 0)
PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


@dataclass(frozen=True)
class Job:
    dataset: str
    limit: int
    scaler: str
    q: int
    seed: int

    @property
    def dataset_tag(self) -> str:
        return f"{self.dataset}_{self.limit}"

    @property
    def stem(self) -> str:
        return f"{self.dataset_tag}_{self.scaler}_Q{self.q}_seed{self.seed}"

    @property
    def output(self) -> Path:
        return RESULT_DIR / f"{self.stem}.csv"

    @property
    def error_output(self) -> Path:
        return RESULT_DIR / f"{self.stem}_errors.csv"


def planned_jobs() -> list[Job]:
    jobs = []
    for dataset, limit in DATASETS:
        for seed in SEEDS:
            for scaler in SCALERS:
                for q in QS:
                    # These runs were already done separately in results/spectral_nfst/.
                    if seed == 42 and q == 2 and scaler == "QuantileTransformer":
                        continue
                    jobs.append(Job(dataset, limit, scaler, q, seed))
    return jobs


def normalized_gamma(value: str) -> str:
    value = str(value).strip()
    if value == "heuristic":
        return value
    try:
        return format(float(value), ".12g")
    except ValueError:
        return value


EXPECTED = {(str(f), normalized_gamma(g)) for f in FEATURES for g in GAMMAS}


def inspect_job(job: Job) -> tuple[bool, str]:
    """Check a successful CSV contains exactly 12 distinct valid experiments."""
    if not job.output.is_file():
        return False, "missing CSV"
    try:
        df = pd.read_csv(job.output, dtype=str)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return False, f"unreadable CSV: {exc}"
    cols = {"Dataset", "Scaler", "Seed", "Q", "Feature", "Gamma", "MCC"}
    if not cols.issubset(df.columns):
        return False, "CSV columns missing"
    if df.empty:
        return False, "empty CSV"
    if not (df["Dataset"].eq(job.dataset_tag).all()
            and df["Scaler"].eq(job.scaler).all()
            and df["Seed"].eq(str(job.seed)).all()
            and df["Q"].eq(str(job.q)).all()):
        return False, "CSV has mismatched dataset/scaler/Q/seed"
    mcc = pd.to_numeric(df["MCC"], errors="coerce")
    if not mcc.notna().all():
        return False, "MCC missing or invalid"
    actual = [(str(f).strip(), normalized_gamma(g))
              for f, g in zip(df["Feature"], df["Gamma"])]
    if len(actual) != len(set(actual)):
        return False, "duplicate successful rows; inspect manually"
    if set(actual) != EXPECTED or len(actual) != 12:
        return False, f"{len(actual)}/12 successful configurations"
    return True, "complete"


def archive_previous_errors(job: Job) -> None:
    """Preserve failed CSV entries but retry them; tuning otherwise marks errors SKIP."""
    if not job.error_output.is_file():
        return
    ERROR_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = ERROR_ARCHIVE_DIR / f"{job.stem}_errors_{suffix}.csv"
    job.error_output.rename(destination)
    log(f"[ARCHIVE OLD ERRORS] {destination.name}")


def run_one(job: Job, blas_threads: int | None) -> tuple[Job, int, float, bool]:
    env = os.environ.copy()
    if blas_threads is not None:
        # Limits numerical library threads, not the SpectralNFST algorithm.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[name] = str(blas_threads)

    command = [
        sys.executable, str(TUNE_SCRIPT),
        "--dataset", job.dataset,
        "--limit", str(job.limit),
        "--scaler", job.scaler,
        "--q", str(job.q),
        "--seed", str(job.seed),
        "--output", str(job.output),
    ]
    started = time.perf_counter()
    outfile = LOG_DIR / f"{job.stem}.log"
    try:
        # Existing successful rows remain intact and tune_spectral_nfst.py
        # will SKIP them; only failed configurations are retried.
        archive_previous_errors(job)
        log(f"[START] {job.stem}")
        with outfile.open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== RUN {datetime.now().isoformat(timespec='seconds')} ===\n")
            stream.flush()
            process = subprocess.run(
                command, cwd=PROJECT_ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, check=False,
            )
            exit_code = process.returncode
    except (OSError, ValueError) as exc:
        log(f"[ERROR] {job.stem}: {exc}")
        exit_code = 1
    elapsed = time.perf_counter() - started
    complete, why = inspect_job(job)
    log(f"[DONE] {job.stem} | exit={exit_code} | complete={complete} "
        f"| {elapsed / 60:.1f} min | {why}")
    return job, exit_code, elapsed, complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only count completed/pending jobs; run nothing")
    parser.add_argument("--workers", type=int, default=1,
                        help="Simultaneous jobs (1 recommended for 16 GB RAM)")
    parser.add_argument("--blas-threads", type=int, default=None,
                        help="Numerical library threads/job; omit to preserve defaults")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="Run only first N pending jobs; useful for a short test")
    args = parser.parse_args()
    if args.workers not in (1, 2, 3, 4):
        parser.error("--workers must be 1, 2, 3, or 4")
    if args.blas_threads is not None and args.blas_threads < 1:
        parser.error("--blas-threads must be >= 1")
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("--max-jobs must be >= 1")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = planned_jobs()
    assert len(jobs) == 350, f"Unexpected number of planned jobs: {len(jobs)}"

    pending = []
    for job in jobs:
        complete, _ = inspect_job(job)
        if not complete:
            pending.append(job)
    log(f"Planned  : {len(jobs)} jobs")
    log(f"Completed: {len(jobs) - len(pending)} fully completed jobs")
    log(f"Pending  : {len(pending)} jobs")
    log(f"Workers  : {args.workers}, BLAS threads/job: {args.blas_threads or 'system default'}")
    log(f"CSV dir  : {RESULT_DIR}")
    if args.dry_run:
        for job in pending[:10]:
            _, why = inspect_job(job)
            log(f"  PENDING: {job.stem} ({why})")
        if len(pending) > 10:
            log(f"  ...and {len(pending) - 10} more pending jobs")
        return 0
    if not pending:
        log("All jobs complete.")
        return 0

    selected = pending[:args.max_jobs] if args.max_jobs is not None else pending
    log(f"Selected : {len(selected)} jobs to run now")
    log("IMPORTANT: Stop the original full-grid script before using this runner.")
    iterator = iter(selected)
    failures = []
    finished = 0
    fatal = False
    start = time.perf_counter()

    # Keep at most 'workers' active; do not queue all jobs. On an abnormal
    # Python process exit, stop scheduling new jobs instead of failing 100 in a row.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        active = {}
        for _ in range(min(args.workers, len(selected))):
            job = next(iterator)
            active[executor.submit(run_one, job, args.blas_threads)] = job
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    _, exit_code, _, complete = future.result()
                except Exception as exc:
                    log(f"[UNEXPECTED] {job.stem}: {exc}")
                    exit_code, complete = 1, False
                finished += 1
                if exit_code != 0 or not complete:
                    failures.append(job)
                if exit_code != 0:
                    fatal = True
                log(f"PROGRESS: {finished}/{len(selected)} selected jobs")
                if not fatal:
                    next_job = next(iterator, None)
                    if next_job is not None:
                        active[executor.submit(run_one, next_job, args.blas_threads)] = next_job

    completed_now = sum(inspect_job(job)[0] for job in jobs)
    log(f"Elapsed: {(time.perf_counter() - start) / 3600:.2f} hours")
    log(f"Fully completed: {completed_now}/350 jobs")
    if fatal:
        log("STOPPED: a job process failed; check its job_logs/*.log before resuming.")
    if failures:
        log(f"Incomplete or failed jobs this run: {len(failures)}")
        for job in failures[:20]:
            log(f"  {job.stem}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
