#!/usr/bin/env python3
"""Validation tuning + full-train evaluation of independent nNFST reimplementation.

Uses existing PROJECT/code/nnfst_paper.py and PROJECT/code/run_nnfst_paper.py.
Does not modify existing results for SpectralNFST or 15 competitors.

IMPORTANT: nNFST code supplied earlier is NOT verified against original authors.
IMPORTANT: validation split is created after project's preprocess() has been fitted
on the *outer* train set. If preprocess learns any scaling statistics, inner
validation leaks into its fitted transform. Fix upstream pipeline for a strictly
nested, publication-grade evaluation. See README.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from nnfst_paper import NNFSTPaper
from run_nnfst_paper import DATASETS, load_prepared, sample_training

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'nnfst_competitor16_reimpl'
TUNE = OUT / 'tuning_jobs'
FINAL = OUT / 'fulltrain_jobs'
SEEDS = (42, 43, 44)
MODEL = 'nNFST_Reimplementation'
VALID_FIELDS = ('Data Type', 'Seed', 'r', 'Gamma', 'Kernel', 'SCALER', 'Poly')


def canonical_tag(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def save_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.temp.{os.getpid()}')
    pd.DataFrame([row]).to_csv(temp, index=False, encoding='utf-8-sig')
    os.replace(temp, path)


def error_record(path: Path, exc: Exception):
    path.with_suffix('.error.txt').write_text(
        f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}', encoding='utf-8'
    )


def completed_row(path: Path):
    if not path.exists():
        return None
    data = pd.read_csv(path, encoding='utf-8-sig')
    if len(data) != 1 or not np.isfinite(float(data.iloc[0]['MCC'])):
        raise RuntimeError('Invalid success cache: ' + str(path))
    return data.iloc[0].to_dict()


def datasets_selected(names: list[str] | None):
    all_map = {f'{x}_{lim}': (x, lim) for x, lim in DATASETS}
    if names is None or names == ['all']:
        return list(all_map.items())
    invalid = set(names) - set(all_map)
    if invalid:
        raise ValueError('Unknown datasets: ' + ', '.join(sorted(invalid)))
    return [(name, all_map[name]) for name in names]


def prepare_inner(Xtr, ytr, seed, validation_fraction, max_fit):
    classes, counts = np.unique(ytr, return_counts=True)
    if len(classes) < 2 or min(counts) < 2:
        raise ValueError('Inner validation requires >=2 rows in every class')
    if int(round(len(ytr)*validation_fraction)) < len(classes):
        raise ValueError('Validation split too small for stratification')
    xi, xv, yi, yv = train_test_split(
        Xtr, ytr, test_size=validation_fraction,
        stratify=ytr, random_state=seed + 101_101,
    )
    # Only subsample *inside* the inner train fold; all validation rows remain unseen.
    xf, yf = sample_training(xi, yi, max_fit, seed + 303_303)
    return xf, yf, xv, yv, len(xi)


def scan_one_row_dir(path: Path) -> pd.DataFrame:
    records=[]
    if not path.exists():
        return pd.DataFrame()
    for p in sorted(path.glob('*/*.csv')):
        try:
            row = completed_row(p)
        except Exception as exc:
            print(f'[WARN] Skipping invalid {p}: {exc}', flush=True)
            continue
        if row is not None:
            records.append(row)
    return pd.DataFrame(records)


def tune(args):
    chosen = datasets_selected(args.datasets)
    TUNE.mkdir(parents=True, exist_ok=True)
    n_fail = 0
    for name, (dataset, limit) in chosen:
        print(f'\n===== TUNING {name} / holdout validation only =====', flush=True)
        for scaler in args.scalers:
            for poly in args.features:
                for seed in SEEDS:
                    print(f'[LOAD] {name} seed={seed} scaler={scaler} feature={poly}', flush=True)
                    try:
                        Xtr, ytr, _Xtest, _ytest = load_prepared(dataset, limit, scaler, poly, seed)
                        xf, yf, xv, yv, inner_original = prepare_inner(
                            Xtr, ytr, seed, args.validation_fraction, args.tune_max_train
                        )
                    except Exception as exc:
                        n_fail += len(args.r_values) * len(args.gamma_values)
                        # A feature value may be unsupported by cpai.preprocessing;
                        # continue safely without claiming it was evaluated.
                        path = TUNE / name / f'load_seed{seed}_feature{poly}_{scaler}.error.txt'
                        path.write_text(f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}', encoding='utf-8')
                        print(f'[UNSUPPORTED OR INVALID FEATURE] {name} seed={seed} feature={poly} scaler={scaler}: {exc}', flush=True)
                        continue
                    for r in args.r_values:
                        for gamma in args.gamma_values:
                            cfg = {
                                'dataset': name, 'seed': seed, 'r': r, 'gamma': gamma,
                                'scaler': scaler, 'poly': poly, 'kernel': args.kernel,
                                'val_frac': args.validation_fraction,
                                'tune_max_train': args.tune_max_train,
                                'relative_tol': args.relative_tol,
                            }
                            path = TUNE / name / f'{canonical_tag(cfg)}.csv'
                            if completed_row(path) is not None:
                                print(f'[SKIP] {name} seed={seed} r={r} gamma={gamma:g} scaler={scaler} feature={poly}', flush=True)
                                continue
                            try:
                                clf = NNFSTPaper(
                                    r=int(r), gamma=float(gamma), kernel=args.kernel,
                                    relative_tol=args.relative_tol,
                                    max_eigensize=args.max_eigensize,
                                )
                                t0=time.perf_counter()
                                clf.fit(xf, yf)
                                fit_sec=time.perf_counter()-t0
                                pred=clf.predict(xv)
                                row={
                                    'Data Type': name, 'Seed':seed,'r':r,'Gamma':gamma,
                                    'Kernel':args.kernel,'SCALER':scaler,'Poly':poly,
                                    'MCC':float(matthews_corrcoef(yv,pred)),
                                    'F1 Macro':float(f1_score(yv,pred,average='macro',zero_division=0)*100),
                                    'ACC':float(accuracy_score(yv,pred)*100),
                                    'Train_On_Inner':int(inner_original),
                                    'Train_Used_Tuning':len(yf),
                                    'Validation_Rows':len(yv),
                                    'Outer_Train_Rows':len(ytr),
                                    'Fit_Seconds':fit_sec,
                                    'Validation_Fraction':args.validation_fraction,
                                    'Tuning_Limit':args.tune_max_train,
                                    'Relative_Tol':args.relative_tol,
                                    'Selection_Set':'INNER_VALIDATION_NOT_TEST',
                                }
                                save_row(path,row)
                                print(f'[VAL] {name} seed={seed} r={r} gamma={gamma:g} MCC={row["MCC"]:.6f} (train {len(yf)}, validation {len(yv)})',flush=True)
                            except Exception as exc:
                                n_fail+=1
                                error_record(path, exc)
                                print(f'[FAILED] {name} seed={seed} r={r} gamma={gamma:g}: {exc}',flush=True)

    rows=scan_one_row_dir(TUNE)
    if rows.empty:
        raise RuntimeError('No successful tuning rows')
    rows.to_csv(OUT/'validation_runs.csv',index=False,encoding='utf-8-sig')
    group_cols=['Data Type','r','Gamma','Kernel','SCALER','Poly','Validation_Fraction','Tuning_Limit','Relative_Tol']
    stats=(rows.groupby(group_cols,dropna=False)['MCC']
           .agg(MCC_Validation_Mean='mean',MCC_Validation_Std='std',Seed_Count='size')
           .reset_index())
    stats=stats[stats.Seed_Count==len(SEEDS)].copy()
    stats.sort_values(['Data Type','MCC_Validation_Mean','MCC_Validation_Std','r','Gamma'],
                      ascending=[True,False,True,True,True], inplace=True)
    stats.to_csv(OUT/'validation_grid_summary.csv',index=False,encoding='utf-8-sig')
    best=stats.drop_duplicates('Data Type',keep='first').copy()
    best.to_csv(OUT/'best_params.csv',index=False,encoding='utf-8-sig')
    for name, _ in chosen:
        selected=best[best['Data Type'].eq(name)]
        if selected.empty:
            print(f'[INCOMPLETE] {name}: no hyperparameter candidate has 3/3 successful seeds',flush=True)
        else:
            p=selected.iloc[0]
            print(f'[BEST] {name}: r={int(p.r)} gamma={p.Gamma:g} scaler={p.SCALER} feature={int(p.Poly)} validation_MCC={p.MCC_Validation_Mean:.6f}', flush=True)
    print(f'\nTuning jobs failed: {n_fail}. Output: {OUT}',flush=True)
    print('IMPORTANT: preprocessing runs before inner validation split; upstream scaler leakage is possible.',flush=True)


def evaluate(args):
    chosen=datasets_selected(args.datasets)
    best_file=OUT/'best_params.csv'
    if not best_file.exists():
        raise FileNotFoundError('Run tune first: missing '+str(best_file))
    best=pd.read_csv(best_file,encoding='utf-8-sig')
    failures=0
    for name,(dataset,limit) in chosen:
        b=best[best['Data Type'].eq(name)]
        if len(b)!=1:
            print(f'[NO_BEST] {name}: missing or ambiguous params in best_params.csv',flush=True)
            failures+=3
            continue
        p=b.iloc[0]
        r=int(p['r']); gamma=float(p['Gamma'])
        scaler=str(p['SCALER']); poly=int(p['Poly']); kernel=str(p['Kernel'])
        for seed in SEEDS:
            cfg={'dataset':name,'seed':seed,'r':r,'gamma':gamma,
                 'scaler':scaler,'poly':poly,'kernel':kernel,
                 'relative_tol':float(p['Relative_Tol'])}
            path=FINAL/name/f'{canonical_tag(cfg)}.csv'
            if completed_row(path) is not None:
                print(f'[SKIP FULL] {name} seed={seed}',flush=True)
                continue
            try:
                Xtr,ytr,Xte,yte=load_prepared(dataset,limit,scaler,poly,seed)
                print(f'[FULL] {name} seed={seed} n_train={len(ytr)} n_test={len(yte)} classes={len(np.unique(ytr))} r={r} gamma={gamma:g}',flush=True)
                if args.max_eigensize and len(ytr)>args.max_eigensize:
                    raise MemoryError(f'FULL kernel requires n={len(ytr)} > safety max-eigensize={args.max_eigensize}. No subsampled fallback; use a suitable high-memory machine.')
                clf=NNFSTPaper(r=r,gamma=gamma,kernel=kernel,
                               relative_tol=float(p['Relative_Tol']),
                               max_eigensize=args.max_eigensize)
                t0=time.perf_counter(); clf.fit(Xtr,ytr); tr_sec=time.perf_counter()-t0
                t1=time.perf_counter(); prediction=clf.predict(Xte); te_sec=time.perf_counter()-t1
                result={'Data Type':name,'Seed':seed,'r':r,'Gamma':gamma,
                        'Kernel':kernel,'SCALER':scaler,'Poly':poly,
                        'Model':MODEL, 'Input Model':MODEL,
                        'MCC':float(matthews_corrcoef(yte,prediction)),
                        'ACC':float(accuracy_score(yte,prediction)*100),
                        'F1 Macro':float(f1_score(yte,prediction,average='macro',zero_division=0)*100),
                        'Training time':tr_sec,'Test time':te_sec,
                        'N_Train_Original':len(ytr),'N_Train_Used':len(ytr),'N_Test':len(yte),
                        'Train_Classes':len(np.unique(ytr)), 'Test_Classes':len(np.unique(yte)),
                        'Run_Type':'FULL_TRAIN','Algorithm_Source':'independent_reimplementation_not_official',
                        'Q_SCC':clf.n_components_,'Projection_Dim':clf.n_projection_dims_,
                        'Within_Null_Error':clf.within_null_relative_error_,
                        'Params_Chosen_By':'INNER_VALIDATION',
                        'Validation_MCC_Mean':float(p['MCC_Validation_Mean']),
                        'Protocol':'closed_set',
                        'Relative_Tol':float(p['Relative_Tol'])}
                save_row(path,result)
                print(f'[DONE] {name} seed={seed} test_MCC={result["MCC"]:.6f} train_time={tr_sec:.1f}s',flush=True)
            except Exception as exc:
                failures+=1
                error_record(path,exc)
                print(f'[FAILED FULL] {name} seed={seed}: {exc}',flush=True)
    rows=scan_one_row_dir(FINAL)
    if not rows.empty:
        rows.to_csv(OUT/'all_fulltrain_versions.csv',index=False,encoding='utf-8-sig')
    print(f'Full evaluation failures in this invocation: {failures}. Output: {OUT}',flush=True)
    print('Do not put unverified results in the official 15-competitor directory.',flush=True)


def report(_args):
    bf=OUT/'best_params.csv'
    if not bf.exists():
        raise FileNotFoundError('Missing best_params.csv: run tune first')
    best=pd.read_csv(bf,encoding='utf-8-sig')
    if len(best)!=len(DATASETS):
        raise ValueError('Need best hyperparameters for all 10 datasets before official 17-model report')
    full=scan_one_row_dir(FINAL)
    if full.empty:
        raise ValueError('No FULL training evaluation rows')
    selection=[]
    for ds, _lim in [(f'{d}_{lim}',lim) for d,lim in DATASETS]:
        b=best[best['Data Type'].eq(ds)]
        if len(b)!=1:
            raise ValueError(f'Missing best params for {ds}')
        p=b.iloc[0]
        records=full[(full['Data Type']==ds) & (full['r'].astype(int)==int(p.r)) &
                     (np.isclose(full['Gamma'].astype(float),float(p.Gamma),atol=0,rtol=1e-12)) &
                     (full['Kernel']==p.Kernel) & (full['SCALER']==p.SCALER) &
                     (full['Poly'].astype(int)==int(p.Poly)) &
                     (full['Run_Type']=='FULL_TRAIN')].copy()
        if len(records)!=3 or set(records.Seed.astype(int))!=set(SEEDS):
            raise ValueError(f'{ds}: expected 3 FULL_TRAIN rows for selected config, got {len(records)}')
        if not (records['N_Train_Original'].astype(int)==records['N_Train_Used'].astype(int)).all():
            raise ValueError(f'{ds}: training size reduced; cannot report official benchmark')
        selection.append(records)
    selected=pd.concat(selection,ignore_index=True)
    if len(selected)!=30:
        raise ValueError('Need exactly 30 selected full-train evaluations')
    selected.to_csv(OUT/'nNFST_Reimplementation_30runs.csv',index=False,encoding='utf-8-sig')
    metrics = {
        'MCC': 'MCC',
        'F1 Macro': 'F1',
        'ACC': 'ACC',
        'Training time': 'Train_s',
        'Test time': 'Test_s',
    }
    records = []
    for ds, rows in selected.groupby('Data Type'):
        rec = {'Dataset': ds, 'Model': MODEL, 'Seed_Count': 3}
        for source, target in metrics.items():
            values = pd.to_numeric(rows[source], errors='raise')
            rec[target + '_Mean'] = float(values.mean())
            rec[target + '_Std'] = float(values.std(ddof=1))
        records.append(rec)
    summary = pd.DataFrame(records)
    summary.to_csv(OUT/'nnfst_mean_std.csv',index=False,encoding='utf-8-sig')
    prev=ROOT/'results'/'spectral_nfst_analysis'/'competitor_comparison'/'final_leaderboard_with_best_params.csv'
    if not prev.exists():
        raise FileNotFoundError(f'Missing 16-model leaderboard: {prev}')
    original=pd.read_csv(prev,encoding='utf-8-sig')
    original=original[original.Model.ne(MODEL)].copy()
    if len(original)!=160 or original.groupby('Dataset')['Model'].nunique().ne(16).any():
        raise ValueError('Original ranking should have 160 rows: 16 models across 10 datasets')
    if set(original.Dataset)!=set(summary.Dataset):
        raise ValueError('Dataset mismatch between 15 competitors + SpectralNFST and nNFST')
    merged=pd.concat([original,summary],ignore_index=True,sort=False)
    merged['Rank_By_MCC']=merged.groupby('Dataset')['MCC_Mean'].rank(method='min',ascending=False).astype(int)
    merged.sort_values(['Dataset','Rank_By_MCC','Model'],inplace=True)
    if merged.groupby('Dataset')['Model'].nunique().ne(17).any():
        raise ValueError('Not exactly 17 models per dataset')
    merged.to_csv(OUT/'leaderboard_17_models.csv',index=False,encoding='utf-8-sig')
    wide=merged.pivot(index='Dataset',columns='Model',values='MCC_Mean')
    diff=(wide['SpectralNFST_Tuned']-wide[MODEL]).astype(float)
    wilcox=1.0 if (diff==0).all() else float(wilcoxon(diff.to_numpy(),alternative='two-sided').pvalue)
    comp=summary.copy()
    comp['SpectralNFST_MCC_Mean']=comp.Dataset.map(wide['SpectralNFST_Tuned'])
    comp['Spectral_minus_nNFST']=comp['SpectralNFST_MCC_Mean']-comp.MCC_Mean
    comp['nNFST_Rank_17']=comp.Dataset.map(merged[merged.Model.eq(MODEL)].set_index('Dataset').Rank_By_MCC)
    comp['Spectral_Wins']=comp.Spectral_minus_nNFST.gt(0)
    comp.to_csv(OUT/'spectral_vs_nnfst_reimpl.csv',index=False,encoding='utf-8-sig')
    print('\n=== SELECTED nNFST REIMPLEMENTATION VS SPECTRALNFST ===')
    print(comp[['Dataset','MCC_Mean','MCC_Std','SpectralNFST_MCC_Mean','Spectral_minus_nNFST','nNFST_Rank_17']].to_string(index=False,float_format=lambda a:f'{a:.6f}'))
    print(f'\nWilcoxon two-sided p-value (10 paired dataset means): {wilcox:.6f}')
    print(f'SpectralNFST win rate: {int(diff.gt(0).sum())}/10')
    print(f'Saved report files to: {OUT}')
    print('IMPORTANT: nNFST is independent, unverified reimplementation, not official implementation.')
    print('IMPORTANT: check preprocessing/labels/splits across models; tuning currently has possible inner-validation scaler leakage.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest='command',required=True)
    p=modes.add_parser('tune',help='Choose feature/r/gamma via held-out inner VALIDATION MCC, not TEST MCC')
    p.add_argument('--datasets',nargs='+',default=['all'])
    p.add_argument('--r-values',type=int,nargs='+',default=[5,10,15,20,30])
    p.add_argument('--gamma-values',type=float,nargs='+',default=[0.001,0.01,0.1])
    p.add_argument('--scalers',nargs='+',default=['QuantileTransformer'])
    p.add_argument('--features',nargs='+',type=int,default=[-1,0,1,2], help='Exactly four values, passed through as preprocess(poly=feature)')
    p.add_argument('--kernel',choices=['l05_exponential_kernel'],default='l05_exponential_kernel',help='Fixed to L05 for comparison')
    p.add_argument('--tune-max-train',type=int,default=800,help='Only for inner-tuning fitting. 0=full inner fold; final refit always FULL.')
    p.add_argument('--validation-fraction',type=float,default=0.2)
    p.add_argument('--relative-tol',type=float,default=1e-9)
    p.add_argument('--max-eigensize',type=int,default=1200)
    z=modes.add_parser('plan',help='Show 4-feature/r/gamma grid size; performs no training')
    z.add_argument('--datasets',nargs='+',default=['all'])
    z.add_argument('--features',nargs='+',type=int,default=[-1,0,1,2])
    z.add_argument('--r-values',nargs='+',type=int,default=[5,10,15,20,30])
    z.add_argument('--gamma-values',nargs='+',type=float,default=[0.001,0.01,0.1])
    z.add_argument('--scalers',nargs='+',default=['QuantileTransformer'])
    q=modes.add_parser('evaluate',help='Use selected params and FULL outer training set to score each test split once')
    q.add_argument('--datasets',nargs='+',default=['all'])
    q.add_argument('--max-eigensize',type=int,default=3000,help='Safety stop; do NOT set 14000 on 16GiB PC casually')
    modes.add_parser('report',help='Report only when all 10*3 FULL runs exist; never modifies old results')
    args=parser.parse_args()
    if args.command in ('tune','plan'):
        if len(args.features)!=4 or len(set(args.features))!=4:
            parser.error('Provide exactly 4 DISTINCT feature values with --features')
    if args.command=='plan':
        names=datasets_selected(args.datasets)
        configs=len(args.features)*len(args.r_values)*len(args.gamma_values)*len(args.scalers)
        print(f'Datasets: {len(names)}, seeds: {len(SEEDS)}, features: {args.features}, r: {args.r_values}, gamma: {args.gamma_values}, scalers: {args.scalers}')
        print(f'Validation candidates per dataset: {configs}; TOTAL candidate fits: {len(names)*len(SEEDS)*configs}. Kernel fixed: l05_exponential_kernel')
    elif args.command=='tune':
        if not 0<args.validation_fraction<0.5: parser.error('validation fraction must be between 0 and 0.5')
        if args.tune_max_train and args.tune_max_train < 2*34: parser.error('tune-max-train too small for 34-class case')
        print(f'[GRID] {len(datasets_selected(args.datasets))} datasets x 3 seeds x {len(args.features)} features x {len(args.r_values)} r x {len(args.gamma_values)} gammas x {len(args.scalers)} scalers')
        tune(args)
    elif args.command=='evaluate':evaluate(args)
    else:report(args)


if __name__=='__main__':
    main()
