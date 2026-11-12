"""All-training-data Nyström approximate nNFST runner, separated from original baselines.

Example:
    python -u code/run_nnfst_fulltrain_approx.py --dataset N_BaIoT --seed 42
    python -u code/run_nnfst_fulltrain_approx.py --all --seeds 42 43 44

Every run fits on ALL rows returned by the EXISTING cpai.preprocess() train
split; uses only train data to choose Nyström landmarks. Never call this the
original full-kernel nNFST implementation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from nnfst_nystrom import NNFSTNystrom
from run_nnfst_paper import DATASETS, load_prepared

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "nnfst_fulltrain_approx"


def filename(s):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s))


def config_key(a):
    return (f"{a.dataset}_{a.limit}" if a.dataset else "all")


def run(ds, limit, seed, a):
    dataset = f"{ds}_{limit}"
    directory = OUTPUT / dataset
    directory.mkdir(parents=True, exist_ok=True)
    signature = (f"s{seed}_r{a.r}_g{a.gamma:g}_m{a.landmarks}_"
                 f"{a.kernel}_{a.scaler}_p{a.poly}_proj{a.projection_cap}")
    target = directory / (filename(signature) + ".csv")
    if target.exists() and not a.force:
        old = pd.read_csv(target)
        if len(old) == 1 and old.iloc[0].get("Method") == "NNFST_NYSTROM_REGULARIZED_NEAR_NULL":
            print(f"[SKIP] {dataset} seed={seed}: {target.name}", flush=True)
            return True
        raise ValueError(f"Invalid cached file: {target}")
    print(f"[START] {dataset} seed={seed} full_training=True landmarks={a.landmarks}", flush=True)
    try:
        Xtr, ytr, Xte, yte = load_prepared(ds, limit, a.scaler, a.poly, seed)
        if len(ytr) < 2:
            raise ValueError("Training split is empty")
        model = NNFSTNystrom(
            r=a.r, gamma=a.gamma, kernel=a.kernel,
            n_landmarks=a.landmarks, projection_cap=a.projection_cap,
            batch_size=a.batch_size, seed=seed,
        )
        print(f"[DATA] train_used={len(ytr)} test={len(yte)} "
              f"classes_train={len(np.unique(ytr))} features={Xtr.shape[1]}", flush=True)
        start = time.perf_counter()
        model.fit(Xtr, ytr)
        train_s = time.perf_counter() - start
        start = time.perf_counter()
        predictions = model.predict(Xte)
        test_s = time.perf_counter() - start
        row = {
            "Seed": seed,
            "Input Model": "nNFST_NystromApprox",
            "Data Type": dataset,
            "Poly": a.poly,
            "Kernel": a.kernel,
            "SCALER": a.scaler,
            "Model": "nNFST_NystromApprox",
            "MCC": float(matthews_corrcoef(yte, predictions)),
            "ACC": 100 * float(accuracy_score(yte, predictions)),
            "F1 Macro": 100 * float(f1_score(yte, predictions, average="macro", zero_division=0)),
            "Training time": train_s,
            "Test time": test_s,
            "Method": model.method_,
            "Algorithm_Source": "independent_approximation_not_author_code",
            "Protocol": "closed_set",
            "Run_Type": "FULL_TRAIN_APPROX_KERNEL",
            "N_Train_Original": int(len(ytr)),
            "N_Train_Used": int(model.n_samples_fit_),
            "N_Test": int(len(yte)),
            "N_Classes_Train": int(np.unique(ytr).size),
            "Feature_Count": int(Xtr.shape[1]),
            "r": a.r,
            "Gamma": a.gamma,
            "N_Landmarks": int(len(model.landmarks_)),
            "Nystrom_Rank": model.feature_rank_,
            "Q_SCC": model.n_components_,
            "Projection_Dim": model.n_projection_dims_,
            "NearNull_GenEigenvalue_Mean": model.approximation_error_,
        }
        if row["N_Train_Used"] != row["N_Train_Original"]:
            raise AssertionError("Some training samples were lost")
        tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
        pd.DataFrame([row]).to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, target)
        errpath = target.with_suffix(".error.txt")
        if errpath.exists():
            errpath.unlink()
        print(f"[DONE] {dataset} seed={seed} "
              f"MCC={row['MCC']:.6f} ACC={row['ACC']:.3f}% "
              f"rNN_SCC={model.n_components_} map_rank={model.feature_rank_} "
              f"training={train_s:.1f}s test={test_s:.1f}s", flush=True)
        return True
    except Exception as exc:
        path = target.with_suffix(".error.txt")
        path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[FAILED] {dataset} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
        print(f"         Error log: {path}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    scope = ap.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all", action="store_true")
    scope.add_argument("--dataset", type=str, help="e.g. N_BaIoT")
    ap.add_argument("--limit", type=int, default=1000, choices=[1000, 2000])
    ap.add_argument("--seed", type=int, default=42, help="For --dataset only")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--kernel", default="l05_exponential_kernel",
                    choices=["rbf", "l05_exponential_kernel", "linear"])
    ap.add_argument("--scaler", default="QuantileTransformer",
                    choices=["QuantileTransformer", "StandardScaler", "RobustScaler"])
    ap.add_argument("--poly", type=int, default=-1)
    ap.add_argument("--landmarks", type=int, default=384,
                    help="Nyström basis size; training uses ALL rows regardless")
    ap.add_argument("--projection-cap", type=int, default=0,
                    help="0 uses up to Q-1 discriminant directions")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    jobs = [(d, l, s) for (d, l) in DATASETS for s in a.seeds] if a.all else [(a.dataset, a.limit, a.seed)]
    print("MODE: ALL TRAINING ROWS / Nyström approximate kernel; NOT exact paper implementation.", flush=True)
    print(f"Jobs: {len(jobs)} | landmarks={a.landmarks} | seed(s)={[z[2] for z in jobs[:3]]}", flush=True)
    if a.dry_run:
        for d, l, s in jobs:
            print(f"[PLAN] {d}_{l} seed={s}")
        return
    all_good = True
    for d, l, s in jobs:
        good = run(d, l, s, a)
        all_good &= good
    frames = []
    for path in OUTPUT.glob("*/*.csv"):
        try:
            f = pd.read_csv(path, encoding="utf-8-sig")
            if len(f) == 1 and f.iloc[0].get("Method") == "NNFST_NYSTROM_REGULARIZED_NEAR_NULL":
                frames.append(f)
        except Exception:
            pass
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.sort_values(["Data Type", "Seed", "N_Landmarks", "Gamma"], inplace=True)
        tmp = OUTPUT / f"all_runs.csv.tmp.{os.getpid()}"
        combined.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, OUTPUT / "all_runs.csv")
        print(f"Saved {len(combined)} run rows (may include multiple parameter versions): {OUTPUT / 'all_runs.csv'}")
    print(f"[FINISHED] {sum(1 for x in jobs)} attempted / {len(jobs)} requested; all_success={all_good}")
    if not all_good:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
