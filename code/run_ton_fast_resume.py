#!/usr/bin/env python3
"""Resume ToN_IoT_2000 without NEW heuristic runs.

Existing CSVs and error logs are preserved. Previously completed 11/12 or
12/12 jobs are accepted as-is. Remaining jobs need ten successful numeric-gamma
experiments: feature(-1,0) x gamma(0.001, 0.01, 0.1, 1, 10).

Place in project code/ alongside run_l05_grid_resume_parallel_safe.py and
code/tune_spectral_nfst.py.

Usage:
  py -u code/resume_ton_skip_heuristic_v2.py --dry-run
  py -u code/resume_ton_skip_heuristic_v2.py --workers 1 --blas-threads 4
  py -u code/resume_ton_skip_heuristic_v2.py --workers 2 --blas-threads 2
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import pandas as pd

THIS = Path(__file__).resolve()
CODE_DIR = THIS.parent
sys.path.insert(0, str(CODE_DIR))
import run_l05_grid_resume_parallel_safe as base

DATASET = "ToN_IoT_2000"
# Preserve the ORIGINAL six-gamma grid for reading old CSVs, regardless of
# whether a user has changed GAMMAS in the original tuner or base runner.
ALL_GAMMAS = ("heuristic", "0.001", "0.01", "0.1", "1", "10")
FEATURES = (-1, 0)
EXPECTED_NUMERIC = {
    (str(feature), base.normalized_gamma(gamma))
    for feature in FEATURES
    for gamma in ALL_GAMMAS if gamma != "heuristic"
}
VALID_ALL = EXPECTED_NUMERIC | {(str(f), "heuristic") for f in FEATURES}

REQUIRED_COLUMNS = {"Dataset", "Scaler", "Seed", "Q", "Feature", "Gamma", "MCC"}


def _pair(feature, gamma):
    text = str(gamma).strip()
    if text.lower() == "heuristic":
        return (str(feature).strip(), "heuristic")
    return (str(feature).strip(), base.normalized_gamma(text))


def _rows(path: Path, required: set[str]):
    """Read CSV conservatively; never write or repair an existing file."""
    if not path.exists():
        return [], None
    try:
        df = pd.read_csv(path, dtype=str)
        if not required.issubset(df.columns):
            return [], f"missing columns: {sorted(required - set(df.columns))}"
        return df.to_dict("records"), None
    except Exception as exc:
        return [], f"cannot read {path.name}: {exc}"


def _inspect_fields(job, row):
    return (str(row["Dataset"]).strip() == job.dataset_tag
            and str(row["Scaler"]).strip() == job.scaler
            and str(row["Seed"]).strip() == str(job.seed)
            and str(row["Q"]).strip() == str(job.q))


def inspect(job):
    """Evaluate UNIQUE feature/gamma successes; keep legacy CSVs unchanged.

    Repeated rows are never counted twice. Conflicting MCC values for the same
    feature/gamma are flagged for scientific review (not silently accepted).
    """
    successes, error = _rows(job.output, REQUIRED_COLUMNS)
    if error:
        return {"state": "invalid", "reason": error,
                "success": set(), "failed": set(), "todo": set(),
                "duplicates": {}, "conflicts": {}}

    seen_mcc = {}
    duplicates = Counter()
    conflicts = {}
    unexpected = set()
    for row in successes:
        if not _inspect_fields(job, row):
            return {"state": "invalid", "reason": "CSV contains mismatched dataset/scaler/Q/seed", 
                    "success": set(), "failed": set(), "todo": set(),
                    "duplicates": dict(duplicates), "conflicts": conflicts}
        key = _pair(row["Feature"], row["Gamma"])
        if key not in VALID_ALL:
            unexpected.add(key)
            continue
        try:
            mcc = float(row["MCC"])
            if not math.isfinite(mcc):
                raise ValueError("nonfinite MCC")
        except (ValueError, TypeError):
            return {"state": "invalid", "reason": f"invalid MCC for {key}",
                    "success": set(), "failed": set(), "todo": set(),
                    "duplicates": dict(duplicates), "conflicts": conflicts}
        if key in seen_mcc:
            duplicates[key] += 1
            if not math.isclose(seen_mcc[key], mcc, rel_tol=1e-9, abs_tol=1e-9):
                conflicts[key] = (seen_mcc[key], mcc)
        else:
            seen_mcc[key] = mcc

    seen = set(seen_mcc)
    if unexpected:
        return {"state": "invalid", "reason": f"unexpected feature-gamma: {sorted(unexpected)}",
                "success": seen, "failed": set(), "todo": set(),
                "duplicates": dict(duplicates), "conflicts": conflicts}
    if conflicts:
        return {"state": "invalid", "reason": f"conflicting duplicate MCC: {conflicts}",
                "success": seen, "failed": set(), "todo": set(),
                "duplicates": dict(duplicates), "conflicts": conflicts}

    meta = {"success": seen, "duplicates": dict(duplicates), "conflicts": conflicts}

    # Preserve the original criterion: 11 DISTINCT successful configs are accepted.
    if len(seen) >= 11:
        return {"state": "old_accepted", "reason": f"{len(seen)}/12 distinct successes", 
                **meta, "failed": set(), "todo": set()}
    if EXPECTED_NUMERIC <= seen:
        return {"state": "new_accepted", "reason": "10/10 numeric configs successful", 
                **meta, "failed": set(), "todo": set()}

    # Do not retry old numerical failures automatically; preserve their evidence.
    failed = set()
    errfiles = [job.error_output]
    if base.ERROR_ARCHIVE_DIR.exists():
        errfiles.extend(sorted(base.ERROR_ARCHIVE_DIR.glob(f"{job.stem}_errors_*.csv")))
    for path in errfiles:
        rows, err = _rows(path, {"Feature", "Gamma", "Error"})
        if err:
            return {"state": "invalid", "reason": err,
                    **meta, "failed": failed, "todo": set()}
        for row in rows:
            pair = _pair(row["Feature"], row["Gamma"])
            if pair in EXPECTED_NUMERIC:
                failed.add(pair)

    todo = EXPECTED_NUMERIC - seen - failed
    reason = (f"{len(seen)} distinct/12 legacy configs; numeric success="
              f"{len(seen & EXPECTED_NUMERIC)}/10; known numeric failures="
              f"{len(failed - seen)}; numeric remaining={len(todo)}")
    state = "pending" if todo else "known_numeric_failure"
    return {"state": state, "reason": reason, **meta,
            "failed": failed, "todo": todo}


def child_tune(tune_args: list[str]) -> int:
    """Import existing tuning script and change GAMMAS *in memory only*."""
    path = base.TUNE_SCRIPT
    spec = importlib.util.spec_from_file_location("_ton_original_tuner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load tuner: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "GAMMAS") or not hasattr(mod, "main"):
        raise RuntimeError("Original tuning script must define GAMMAS and main()")
    original = tuple(mod.GAMMAS)
    chosen = tuple(
        (label, gamma) for label, gamma in original
        if base.normalized_gamma(label) != "heuristic"
    )
    if len(chosen) != 5 or len(original) not in (5, 6):
        raise RuntimeError(f"Unexpected GAMMAS configuration: {original!r}")
    mod.GAMMAS = chosen
    sys.argv = [str(path), *tune_args]
    print("[NO HEURISTIC] Running only 5 numeric gammas x 2 features", flush=True)
    outcome = mod.main()
    return int(outcome or 0)


def run_one(job, blas_threads: int):
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = str(blas_threads)
    cmd = [
        sys.executable, "-u", str(THIS), "--child-tune",
        "--dataset", job.dataset, "--limit", str(job.limit),
        "--scaler", job.scaler, "--q", str(job.q),
        "--seed", str(job.seed), "--output", str(job.output),
    ]
    base.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = base.LOG_DIR / f"{job.stem}.log"
    print(f"[START] {job.stem}", flush=True)
    t0 = time.monotonic()
    with logfile.open("a", encoding="utf-8") as stream:
        stream.write("\n=== RESUME WITHOUT HEURISTIC ===\n")
        stream.flush()
        result = subprocess.run(cmd, cwd=base.PROJECT_ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    seconds = time.monotonic() - t0
    after = inspect(job)
    print(f"[DONE] {job.stem} | exit={result.returncode} | "
          f"state={after['state']} | {seconds/60:.1f} min | {after['reason']}", flush=True)
    return result.returncode, after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--blas-threads", type=int, default=4)
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()
    if args.blas_threads < 1 or (args.max_jobs is not None and args.max_jobs < 1):
        parser.error("threads and max-jobs must be positive")

    jobs = [j for j in base.planned_jobs() if j.dataset_tag == DATASET]
    states = [(j, inspect(j)) for j in jobs]
    counts = Counter(info["state"] for _, info in states)
    remaining = sum(len(info["todo"]) for _, info in states)
    duplicate_jobs = sum(bool(info["duplicates"]) for _, info in states)
    duplicate_rows = sum(sum(info["duplicates"].values()) for _, info in states)
    print(f"Dataset: {DATASET} | jobs: {len(jobs)}", flush=True)
    print(f"Legacy accepted 11/12+: {counts['old_accepted']} | numeric accepted 10/10: {counts['new_accepted']}", flush=True)
    print(f"Pending jobs: {counts['pending']} | remaining numeric fits: {remaining}", flush=True)
    print(f"Known nonheuristic failures: {counts['known_numeric_failure']} | invalid CSVs: {counts['invalid']}", flush=True)
    print(f"Duplicate legacy CSVs: {duplicate_jobs} jobs / {duplicate_rows} extra rows (not counted twice)", flush=True)
    # Estimated computational time, not a guarantee; excludes CPU contention and I/O overhead.
    print(f"Approximate remaining time at 20-30 min/fit, {args.workers} worker(s): "
          f"{remaining*20/60/args.workers:.1f}-{remaining*30/60/args.workers:.1f} hours (optimistic)", flush=True)
    for job, info in states:
        if info["state"] in ("pending", "known_numeric_failure", "invalid"):
            print(f"  {job.stem} : {info['state']} : {info['reason']}", flush=True)
        if info["duplicates"]:
            print(f"  [DUPLICATES] {job.stem}: {info['duplicates']}", flush=True)
    if args.dry_run:
        return 0
    if counts["invalid"]:
        print("[STOP] Invalid CSVs detected. Repair/inspect them before running to avoid duplicates.", flush=True)
        return 2

    todo_jobs = [job for job, info in states if info["state"] == "pending"]
    if args.max_jobs is not None:
        todo_jobs = todo_jobs[:args.max_jobs]
    if not todo_jobs:
        print("No runnable numeric-gamma work left.", flush=True)
        return 0
    print("Do not run another runner on these CSVs concurrently.", flush=True)
    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(todo_jobs)
        active = {}
        for _ in range(min(args.workers, len(todo_jobs))):
            job = next(iterator)
            active[pool.submit(run_one, job, args.blas_threads)] = job
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    code, state = future.result()
                except Exception as exc:
                    print(f"[CRASH] {job.stem}: {exc}", flush=True)
                    errors += 1
                    code = 1
                if code != 0:
                    errors += 1
                # Do not launch further work after a fatal child process error.
                if errors == 0:
                    nextjob = next(iterator, None)
                    if nextjob is not None:
                        active[pool.submit(run_one, nextjob, args.blas_threads)] = nextjob
    updated = [inspect(j) for j in jobs]
    print("\nFinal counts:", dict(Counter(s["state"] for s in updated)), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--child-tune":
        raise SystemExit(child_tune(sys.argv[2:]))
    raise SystemExit(main())
