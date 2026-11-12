"""Run paper-method nNFST reimplementation using EXISTING Spectral-NFST loaders.

Place this and nnfst_paper.py under PROJECT_ROOT/code, then invoke:
    python -u code/run_nnfst_paper.py --dataset N_BaIoT --limit 1000 --seeds 42 --max-train 400
    python -u code/run_nnfst_paper.py --all --max-train 600 --r 5 --gamma 0.1

Caution: --max-train N (N>0) runs a SUBSAMPLED PILOT, not comparable to the
existing full-training 15 competitor baselines. Full 34,000-row kernel
reproduction requires high RAM and prohibitive O(N^3) eigendecomposition.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from nnfst_paper import NNFSTPaper

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'results' / 'nnfst_paper_reimplementation'
DATASETS = [
    ('5G_NIDD', 1000), ('BoT_IoT', 1000), ('CIC_IoT2023', 1000),
    ('Edge_IIoTset', 1000), ('IoTID20', 1000), ('IoTID20', 2000),
    ('N_BaIoT', 1000), ('ToN_IoT', 1000), ('ToN_IoT', 2000),
    ('UNSW_NB15', 1000),
]


def safe_filename(text):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(text))


def sample_training(X, y, limit, seed):
    """Stratified fit-only subsampling. Never reduces the test set."""
    if limit <= 0 or limit >= len(y):
        return X, y
    c = len(np.unique(y))
    if limit < 2 * c:
        raise ValueError(f'--max-train={limit} is too small for {c} classes')
    X_small, _, y_small, _ = train_test_split(
        X, y, train_size=limit, random_state=seed, stratify=y,
    )
    if len(np.unique(y_small)) != c:
        raise ValueError('Subsampling lost at least one class')
    return X_small, y_small


def load_prepared(dataset: str, limit: int, scaler: str, poly: int, seed: int):
    # Import from user's existing project, so preprocessing is not invented here.
    from cpai.datasets import load_dataset
    from cpai.preprocessing import preprocess
    df, _ = load_dataset(dataset, limit=limit)
    result = preprocess(
        df, dataset,
        poly=poly,
        kernel='l05_exponential_kernel',  # existing project preprocessing switch
        scaler=scaler,
        seed=seed,
    )
    if len(result) < 4:
        raise ValueError('Unexpected preprocessing return: need Xtrain,ytrain,Xtest,ytest')
    Xtr, ytr, Xte, yte = result[:4]
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    ytr = np.asarray(ytr).ravel()
    yte = np.asarray(yte).ravel()
    if not np.isfinite(Xtr).all() or not np.isfinite(Xte).all():
        raise ValueError('Preprocessing produced NaN or Infinity values')
    return Xtr, ytr, Xte, yte


def single_run(dataset: str, limit: int, seed: int, args):
    name = f'{dataset}_{limit}'
    dataset_dir = OUTPUT_DIR / safe_filename(name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Per-job permanent CSV; no parallel process should ever write this job.
    task_key = (
        f'seed{seed}_r{args.r}_gamma{args.gamma:g}_{args.kernel}_'
        f'{args.scaler}_feature{args.poly}_maxtrain{args.max_train}'
    )
    result_file = dataset_dir / f'{safe_filename(task_key)}.csv'
    error_file = dataset_dir / f'{safe_filename(task_key)}.error.txt'
    if result_file.exists() and not args.force:
        existing = pd.read_csv(result_file)
        if len(existing) == 1 and pd.notna(existing.iloc[0].get('MCC')):
            print(f'[SKIP] {name} seed={seed} (existing valid CSV)', flush=True)
            return True
        raise ValueError(f'Invalid cached CSV: {result_file}')
    print(f'\n[START] {name} seed={seed} r={args.r} gamma={args.gamma} '
          f'kernel={args.kernel} max_train={args.max_train}', flush=True)
    try:
        Xtr, ytr, Xte, yte = load_prepared(dataset, limit, args.scaler, args.poly, seed)
        original_size = len(ytr)
        Xsmall, ysmall = sample_training(Xtr, ytr, args.max_train, seed)
        effective_size = len(ysmall)
        run_type = 'PILOT_SUBSAMPLED' if effective_size < original_size else 'FULL_TRAIN'
        model = NNFSTPaper(
            r=args.r, gamma=args.gamma, kernel=args.kernel,
            relative_tol=args.relative_tol, max_eigensize=args.max_eigensize,
            batch_size=args.batch_size,
        )
        print(f'[DATA] train_original={original_size} train_fit={effective_size} '
              f'test={len(yte)} classes={len(np.unique(ysmall))} '
              f'features={Xsmall.shape[1]} RUN_TYPE={run_type}', flush=True)
        t0 = time.perf_counter()
        model.fit(Xsmall, ysmall)
        fit_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred = model.predict(Xte)
        pred_time = time.perf_counter() - t1
        row = {
            'Seed': seed, 'Input Model': 'nNFST_paper_reimplementation',
            'Data Type': name, 'Poly': args.poly, 'Kernel': args.kernel,
            'SCALER': args.scaler, 'Model': 'nNFST_paper_reimplementation',
            'MCC': float(matthews_corrcoef(yte, pred)),
            'ACC': 100 * float(accuracy_score(yte, pred)),
            'F1 Macro': 100 * float(f1_score(yte, pred, average='macro', zero_division=0)),
            'Training time': float(fit_time), 'Test time': float(pred_time),
            'r': args.r, 'Gamma': args.gamma,
            'N_Train_Original': original_size, 'N_Train_Used': effective_size,
            'N_Test': len(yte), 'Run_Type': run_type,
            'Q_SCC': int(model.n_components_),
            'Projection_Dim': int(model.n_projection_dims_),
            'Within_Null_Error': float(model.within_null_relative_error_),
            'Algorithm_Source': 'independent_reimplementation_not_official',
            'Protocol': 'closed_set',
        }
        # Atomic job write so crashes cannot leave truncated success files.
        tmp = result_file.with_name(result_file.name + f'.tmp.{os.getpid()}')
        pd.DataFrame([row]).to_csv(tmp, index=False, encoding='utf-8-sig')
        os.replace(tmp, result_file)
        if error_file.exists():
            error_file.unlink()
        print(f'[DONE] {name} seed={seed}: MCC={row["MCC"]:.6f} '
              f'ACC={row["ACC"]:.3f}% Q_SCC={model.n_components_} '
              f'projection_dim={model.n_projection_dims_} train={fit_time:.2f}s '
              f'predict={pred_time:.2f}s', flush=True)
        return True
    except Exception as exc:
        error_file.write_text(f'{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}', encoding='utf-8')
        print(f'[FAILED] {name} seed={seed} {type(exc).__name__}: {exc}', flush=True)
        print('          Details:', error_file, flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dataset', help='Dataset key, e.g. N_BaIoT or ToN_IoT')
    mode.add_argument('--all', action='store_true')
    parser.add_argument('--limit', type=int, default=1000, choices=[1000, 2000])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--r', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--kernel', default='l05_exponential_kernel',
                        choices=['rbf', 'l05_exponential_kernel', 'linear'])
    parser.add_argument('--scaler', default='QuantileTransformer',
                        choices=['QuantileTransformer', 'StandardScaler', 'RobustScaler'])
    parser.add_argument('--poly', type=int, default=-1)
    parser.add_argument('--max-train', type=int, default=600,
                        help='Max fit rows only. 0 = no subsampling. >0 = PILOT.')
    parser.add_argument('--max-eigensize', type=int, default=1200,
                        help='Safety cutoff for full kernel eigen decomposition. 0=unlimited, dangerous.')
    parser.add_argument('--relative-tol', type=float, default=1e-9)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--force', action='store_true', help='Rerun an existing job, overwrites only that job CSV')
    args = parser.parse_args()
    chosen = DATASETS if args.all else [(args.dataset, args.limit)]
    if args.all:
        print('Running ALL 10 datasets x seeds. Existing successful jobs are skipped.', flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status = []
    for ds, size in chosen:
        for seed in args.seeds:
            okay = single_run(ds, size, seed, args)
            status.append((ds, size, seed, okay))
    all_csv = list(OUTPUT_DIR.glob('*/*.csv'))
    runs = []
    for path in all_csv:
        try:
            df = pd.read_csv(path)
            if len(df) == 1 and 'MCC' in df.columns:
                runs.append(df)
        except Exception:
            continue
    if runs:
        summary = pd.concat(runs, ignore_index=True)
        # Multiple parameter variations may coexist (that's deliberate).
        summary = summary.sort_values(['Data Type', 'Seed', 'r', 'Gamma'])
        outfile = OUTPUT_DIR / 'all_runs.csv'
        temp = OUTPUT_DIR / f'all_runs.csv.tmp.{os.getpid()}'
        summary.to_csv(temp, index=False, encoding='utf-8-sig')
        os.replace(temp, outfile)
        print(f'\nSaved all_runs.csv ({len(summary)} rows): {outfile}', flush=True)
    done = sum(okay for _, _, _, okay in status)
    print(f'FINISHED: {done}/{len(status)} selected runs successful/skipped. '
          f'FAILED: {len(status)-done}. Results: {OUTPUT_DIR}', flush=True)
    if done != len(status):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
