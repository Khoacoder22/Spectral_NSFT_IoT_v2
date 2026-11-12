"""Multiclass Novelty Detection (MND) class-holdout research protocol for nNFST.

This is a validation-only threshold protocol (avoid unknown/test data leakage),
NOT a verified duplication of the paper's threshold hyperparameter selection.
The train/test feature engineering here is independent from the existing
project preprocessing, since holding out the novel class MUST happen before
fitting encoders/scalers; don't compare it directly to the 450 closed-set runs.

Example:
    python -u code/run_nnfst_mnd.py --dataset N_BaIoT --limit 1000 --list-classes
    python -u code/run_nnfst_mnd.py --dataset N_BaIoT --limit 1000 --unknown-class udp --seed 42 --max-train 500
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler, RobustScaler

from nnfst_paper import NNFSTPaper

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results' / 'nnfst_paper_reimplementation' / 'mnd'
UNKNOWN = '__NOVEL__'


def preprocess_with_known_train_only(train: pd.DataFrame, calibration: pd.DataFrame,
                                     test: pd.DataFrame, scaler_name: str):
    # Label/target metadata must NOT enter features. The project's canonical
    # feature-selection rules should replace this conservative heuristic if known.
    forbidden = re.compile(r'(^|_)(label|attack|category|class|target)(_|$)', re.I)
    ignored = [c for c in train if forbidden.search(str(c)) or str(c).lower() in
               ('id', 'flow_id', 'timestamp', 'src_ip', 'dst_ip', 'ip', 'unnamed: 0')]
    keep = [c for c in train.columns if c not in ignored]
    if not keep:
        raise ValueError('No feature columns after label/ID protection')
    train, calibration, test = train[keep], calibration[keep], test[keep]
    numerical = train.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    categorical = [c for c in keep if c not in numerical]
    if not numerical and not categorical:
        raise ValueError('No feature columns')
    if scaler_name == 'QuantileTransformer':
        scaler = QuantileTransformer(n_quantiles=min(1000, len(train)), random_state=42)
    elif scaler_name == 'RobustScaler':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    steps = []
    if numerical:
        steps.append(('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', scaler),
        ]), numerical))
    if categorical:
        # Ordinal encoding is a pragmatic choice; the paper does NOT specify
        # this transformation. Confirm original dataset encoding for replication.
        steps.append(('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ]), categorical))
    processor = ColumnTransformer(steps, sparse_threshold=0)
    Xtr = np.asarray(processor.fit_transform(train), dtype=float)
    Xval = np.asarray(processor.transform(calibration), dtype=float)
    Xte = np.asarray(processor.transform(test), dtype=float)
    print(f'[PREPROCESS] features={Xtr.shape[1]} numeric={len(numerical)} '
          f'categorical={len(categorical)} ignored={len(ignored)}', flush=True)
    return Xtr, Xval, Xte


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--list-classes', action='store_true')
    parser.add_argument('--unknown-class', help='The EXACT Label value to hold out of fitting')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--r', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=.1)
    parser.add_argument('--kernel', default='l05_exponential_kernel',
                        choices=['rbf', 'l05_exponential_kernel', 'linear'])
    parser.add_argument('--scaler', default='QuantileTransformer',
                        choices=['QuantileTransformer', 'StandardScaler', 'RobustScaler'])
    parser.add_argument('--max-train', type=int, default=600)
    parser.add_argument('--max-eigensize', type=int, default=1200)
    parser.add_argument('--known-acceptance', type=float, default=0.95)
    args = parser.parse_args()
    if not 0 < args.known_acceptance < 1:
        raise ValueError('known-acceptance must be between 0 and 1')
    from cpai.datasets import load_dataset
    df, _ = load_dataset(args.dataset, limit=args.limit)
    if 'Label' not in df.columns:
        raise ValueError('Expected normalized Label column from cpai.datasets')
    y = df['Label'].astype(str)
    if args.list_classes:
        print(y.value_counts().to_string())
        return
    if args.unknown_class is None:
        parser.error('Provide --unknown-class or use --list-classes first')
    if args.unknown_class not in set(y):
        raise ValueError(f'Unknown class {args.unknown_class!r}; run --list-classes')
    known = df[y != args.unknown_class].copy()
    novel = df[y == args.unknown_class].copy()
    if known['Label'].nunique() < 2:
        raise ValueError('MND holdout leaves <2 known classes; use a MULTICLASS dataset')
    xknown = known.drop(columns=['Label'])
    yknown = known['Label'].astype(str).to_numpy()
    xtr_all, xknown_test, ytr_all, yknown_test = train_test_split(
        xknown, yknown, test_size=0.2, stratify=yknown, random_state=args.seed,
    )
    # Known-only calibration split. Never use the unknown class or test labels
    # for preprocessing, threshold calibration, or parameter selection.
    xfit, xcal, yfit, _ = train_test_split(
        xtr_all, ytr_all, test_size=0.15, stratify=ytr_all, random_state=args.seed,
    )
    original_train_size = len(yfit)
    if args.max_train > 0 and len(yfit) > args.max_train:
        if args.max_train < 2 * len(np.unique(yfit)):
            raise ValueError('max-train too small to retain all known classes')
        xfit, _, yfit, _ = train_test_split(
            xfit, yfit, train_size=args.max_train,
            stratify=yfit, random_state=args.seed,
        )
    xtest = pd.concat([xknown_test, novel.drop(columns=['Label'])], ignore_index=True)
    ytest = np.r_[yknown_test, np.full(len(novel), UNKNOWN)]
    Xfit, Xcal, Xtest = preprocess_with_known_train_only(xfit, xcal, xtest, args.scaler)
    model = NNFSTPaper(r=args.r, gamma=args.gamma, kernel=args.kernel,
                       max_eigensize=args.max_eigensize)
    print(f'[DATA] FIT={len(yfit)}/{original_train_size} KNOWN_VAL={len(Xcal)} '
          f'KNOWN_TEST={len(yknown_test)} NOVEL_TEST={len(novel)}', flush=True)
    t0 = time.perf_counter()
    model.fit(Xfit, yfit)
    train_time = time.perf_counter() - t0
    cal_scores = model.predict_novelty_score(Xcal)
    q95 = float(np.quantile(cal_scores, args.known_acceptance))
    # Paper bounds candidate thresholds by minimum pairwise base-point distance.
    bases = model.base_points_
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(bases, bases)
    distances[np.diag_indices_from(distances)] = np.inf
    upper = float(np.min(distances))
    threshold = min(q95, upper) if np.isfinite(upper) else q95
    start = time.perf_counter()
    pred, score = model.predict_with_novelty(Xtest, threshold=threshold)
    predict_time = time.perf_counter() - start
    novel_mask = ytest == UNKNOWN
    rec = {
        'Seed': args.seed, 'Data Type': f'{args.dataset}_{args.limit}',
        'Model': 'nNFST_paper_reimplementation',
        'Protocol': 'MND_holdout_knownonly_validation95',
        'Held_Out_Class': args.unknown_class,
        'MCC': float(matthews_corrcoef(ytest, pred)),
        'ACC': 100 * float(accuracy_score(ytest, pred)),
        'F1 Macro': 100 * float(f1_score(ytest, pred, average='macro', zero_division=0)),
        'Novelty_Recall': float(np.mean(pred[novel_mask] == UNKNOWN)),
        'Known_Accuracy': float(np.mean(pred[~novel_mask] == ytest[~novel_mask])),
        'Threshold': threshold, 'Known_Validation_Quantile': args.known_acceptance,
        'N_Train_Original': original_train_size,
        'N_Train_Used': len(yfit), 'N_Test': len(ytest),
        'Run_Type': 'PILOT_SUBSAMPLED' if len(yfit)<original_train_size else 'FULL_TRAIN',
        'r': args.r, 'Gamma': args.gamma, 'Kernel': args.kernel, 'Scaler': args.scaler,
        'Q_SCC': model.n_components_, 'Projection_Dim': model.n_projection_dims_,
        'Training time': train_time, 'Test time': predict_time,
        'Algorithm_Source': 'independent_reimplementation_not_official',
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[^A-Za-z0-9_.-]', '_', f'{args.dataset}_{args.limit}_{args.unknown_class}_{args.seed}_r{args.r}_g{args.gamma}_{args.kernel}_max{args.max_train}')
    out = RESULTS / (name + '.csv')
    if out.exists():
        raise FileExistsError(f'{out} exists; refusing to overwrite')
    tmp = out.with_name(out.name + '.tmp')
    pd.DataFrame([rec]).to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, out)
    print('[DONE]', {k: v for k,v in rec.items() if k in ('Data Type','Held_Out_Class','MCC','Novelty_Recall','Known_Accuracy','Threshold','Run_Type')})
    print('Saved:', out)


if __name__ == '__main__':
    main()
