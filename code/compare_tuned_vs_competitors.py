#!/usr/bin/env python3
"""Compare previously evaluated SpectralNFST-tuned with existing competitor CSVs.

Does NOT train models, change originals, invent missing results, or assert fair splits.

Run from Spectral-NFST root:
    py -u code/compare_tuned_vs_competitors.py
    py -u code/compare_tuned_vs_competitors.py --root D:/Research-new/Spectral-NFST

Requires:
  results/spectral_nfst_analysis/best_config_per_dataset.csv
  results/spectral_nfst_analysis/all_numeric_results.csv
  results/knfst_comparison/models/<Model>.csv

Writes to results/spectral_nfst_analysis/competitor_comparison/.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

DATASETS = [
    '5G_NIDD_1000', 'BoT_IoT_1000', 'CIC_IoT2023_1000',
    'Edge_IIoTset_1000', 'IoTID20_1000', 'IoTID20_2000',
    'N_BaIoT_1000', 'ToN_IoT_1000', 'ToN_IoT_2000', 'UNSW_NB15_1000'
]
SEEDS = [42, 43, 44]
COMPETITORS = [
    'HHH', 'FTTransformer', 'KNN', 'HHHv2', 'TabNet', 'LDA', 'SGD',
    'SAINT', 'GauNB', 'LGBM', 'MLP', 'NuSVC', 'NC', 'RNC', 'KNFST'
]
METRICS = ['MCC', 'F1 Macro', 'ACC', 'FPR', 'Training time', 'Test time']


def normalize_gamma(x):
    try:
        y = float(str(x).strip())
        if math.isfinite(y):
            return f'{y:g}'
    except (TypeError, ValueError):
        pass
    return str(x).strip().lower()


def as_float(x):
    try:
        z = float(str(x).strip())
        return z if math.isfinite(z) else float('nan')
    except (ValueError, TypeError):
        return float('nan')


def read_csv(path):
    return pd.read_csv(path, dtype=str, encoding='utf-8-sig')


def norm_dataset(x):
    return {d.casefold(): d for d in DATASETS}.get(str(x).strip().casefold())


def seed_value(x):
    try:
        val = float(str(x).strip())
        if val.is_integer() and int(val) in SEEDS:
            return int(val)
    except (TypeError, ValueError):
        pass
    return None


def collect_spectral(base, issues):
    folder = base / 'results' / 'spectral_nfst_analysis'
    best_path = folder / 'best_config_per_dataset.csv'
    all_path = folder / 'all_numeric_results.csv'
    if not best_path.is_file() or not all_path.is_file():
        raise FileNotFoundError(
            'Run py code/analyze_spectral_nfst_best.py first; missing '
            f'{best_path if not best_path.is_file() else all_path}'
        )
    best, all_rows = read_csv(best_path), read_csv(all_path)
    required = {'Dataset', 'Scaler', 'Q', 'Feature', 'Gamma'}
    if not required.issubset(best.columns):
        raise ValueError(f'best_config_per_dataset.csv missing {required-set(best.columns)}')
    required_all = required | {'Seed', 'MCC'}
    if not required_all.issubset(all_rows.columns):
        raise ValueError(f'all_numeric_results.csv missing {required_all-set(all_rows.columns)}')

    best = best.copy()
    all_rows = all_rows.copy()
    for df in (best, all_rows):
        df['Gamma'] = df['Gamma'].map(normalize_gamma)
        df['Dataset'] = df['Dataset'].map(norm_dataset)
        df['Q'] = pd.to_numeric(df['Q'], errors='coerce')
        df['Feature'] = pd.to_numeric(df['Feature'], errors='coerce')
        df['Scaler'] = df['Scaler'].astype(str).str.strip()
    all_rows['Seed'] = all_rows['Seed'].map(seed_value)
    for m in METRICS:
        if m not in all_rows.columns:
            all_rows[m] = float('nan')
        all_rows[m] = pd.to_numeric(all_rows[m], errors='coerce')

    chosen = []
    for dataset in DATASETS:
        b = best[best['Dataset'].eq(dataset)]
        if len(b) != 1:
            issues.append(f'SPECTRAL_BEST_CONFIG {dataset}: expected 1 row, found {len(b)}')
            continue
        r = b.iloc[0]
        selection = (all_rows['Dataset'].eq(dataset) &
                     all_rows['Scaler'].eq(r['Scaler']) &
                     all_rows['Q'].eq(r['Q']) &
                     all_rows['Feature'].eq(r['Feature']) &
                     all_rows['Gamma'].eq(r['Gamma']))
        sub = all_rows[selection].copy()
        for seed in SEEDS:
            sr = sub[sub['Seed'].eq(seed)]
            if len(sr) != 1:
                issues.append(f'SPECTRAL_RUN {dataset} seed{seed}: expected 1 row, found {len(sr)}')
                continue
            v = sr.iloc[0]
            if not math.isfinite(v['MCC']):
                issues.append(f'SPECTRAL_METRIC {dataset} seed{seed}: invalid MCC')
                continue
            d = {'Dataset': dataset, 'Seed': seed, 'Model': 'SpectralNFST_Tuned',
                 'Scaler': r['Scaler'], 'Q': int(r['Q']), 'Feature': int(r['Feature']),
                 'Gamma': r['Gamma']}
            d.update({m: v[m] for m in METRICS})
            chosen.append(d)
    return pd.DataFrame(chosen)


def collect_competitors(base, issues):
    folder = base / 'results' / 'knfst_comparison' / 'models'
    records = []
    for model in COMPETITORS:
        path = folder / (model + '.csv')
        if not path.exists():
            issues.append(f'FILE_MISSING {path}')
            continue
        try:
            df = read_csv(path)
        except Exception as exc:
            issues.append(f'CANNOT_READ {path.name}: {exc}')
            continue
        dataset_col = next((c for c in ('Data Type', 'Dataset', 'DataType') if c in df.columns), None)
        seed_col = next((c for c in ('Seed', 'seed', 'SEED') if c in df.columns), None)
        if dataset_col is None or seed_col is None or 'MCC' not in df.columns:
            issues.append(f'COLUMNS_MISSING {path.name}: need Data Type/Dataset, Seed, MCC')
            continue
        for row_index, row in df.iterrows():
            dataset = norm_dataset(row.get(dataset_col))
            seed = seed_value(row.get(seed_col))
            if dataset is None or seed is None:
                continue
            mcc = as_float(row.get('MCC'))
            if not math.isfinite(mcc) or not -1 <= mcc <= 1:
                issues.append(f'INVALID_MCC {path.name} line {row_index+2}')
                continue
            rec = {'Dataset': dataset, 'Seed': seed, 'Model': model}
            for col in METRICS:
                rec[col] = as_float(row.get(col, float('nan')))
            records.append(rec)
    if not records:
        return pd.DataFrame(columns=['Dataset', 'Seed', 'Model'] + METRICS)
    result = pd.DataFrame(records)
    # Reject ambiguous duplicate runs. Never silently choose the best/recent row.
    dups = result.duplicated(['Dataset', 'Seed', 'Model'], keep=False)
    if dups.any():
        for key, group in result[dups].groupby(['Dataset', 'Seed', 'Model']):
            issues.append(f'DUPLICATE_COMPETITOR_ROW {key}: {len(group)} rows; excluded')
        result = result[~dups].copy()
    return result


def summarize_by_model(data):
    out = []
    for (ds, model), grp in data.groupby(['Dataset', 'Model'], sort=True):
        present = set(grp['Seed'].tolist())
        full = present == set(SEEDS) and len(grp) == len(SEEDS)
        row = {'Dataset': ds, 'Model': model, 'Seeds_Observed': ','.join(str(x) for x in sorted(present)),
               'Seed_Count': len(present), 'Ready_3_Seeds': full}
        for m, alias in [('MCC','MCC'), ('F1 Macro','F1'), ('ACC','ACC'),
                         ('FPR','FPR'), ('Training time','Train_s'), ('Test time','Test_s')]:
            vec = pd.to_numeric(grp[m], errors='coerce')
            row[alias+'_Mean'] = float(vec.mean()) if vec.notna().any() else float('nan')
            row[alias+'_Std'] = float(vec.std(ddof=1)) if vec.notna().sum()>1 else float('nan')
        out.append(row)
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    base = args.root.resolve()
    target = base / 'results' / 'spectral_nfst_analysis' / 'competitor_comparison'
    target.mkdir(parents=True, exist_ok=True)
    issues = []

    spectral = collect_spectral(base, issues)
    competitors = collect_competitors(base, issues)
    all_runs = pd.concat([spectral, competitors], ignore_index=True)
    all_runs.to_csv(target/'all_available_runs.csv', index=False, encoding='utf-8-sig')
    spectral.to_csv(target/'spectral_tuned_30_runs.csv', index=False, encoding='utf-8-sig')

    expected_models = ['SpectralNFST_Tuned'] + COMPETITORS
    missing = []
    coverage = []
    for ds in DATASETS:
        for model in expected_models:
            view = all_runs[all_runs['Dataset'].eq(ds) & all_runs['Model'].eq(model)]
            seen = sorted({int(x) for x in view['Seed'].tolist()})
            absent = [s for s in SEEDS if s not in seen]
            coverage.append({'Dataset':ds, 'Model':model, 'Seeds_Found':','.join(map(str, seen)),
                             'Found':len(seen), 'Expected':3, 'Missing':len(absent)})
            for seed in absent:
                missing.append({'Dataset':ds, 'Model':model, 'Seed':seed})
    pd.DataFrame(coverage).to_csv(target/'coverage_by_model_dataset.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(missing, columns=['Dataset','Model','Seed']).to_csv(
        target/'missing_competitor_runs.csv', index=False, encoding='utf-8-sig')

    aggregated = summarize_by_model(all_runs)
    aggregated.to_csv(target/'performance_by_dataset_model.csv', index=False, encoding='utf-8-sig')
    ranked = aggregated[aggregated['Ready_3_Seeds']].copy()
    if not ranked.empty:
        ranked = ranked.sort_values(['Dataset','MCC_Mean','MCC_Std','F1_Mean'],
                                    ascending=[True,False,True,False])
        ranked['Rank_Within_Available_Models'] = ranked.groupby('Dataset').cumcount() + 1
    ranked.to_csv(target/'ranking_only_complete_3seed.csv', index=False, encoding='utf-8-sig')

    pairs = []
    for ds in DATASETS:
        s = spectral[spectral['Dataset'].eq(ds)]
        if len(s) != 3 or set(s['Seed']) != set(SEEDS):
            continue
        s = s.set_index('Seed')
        for model in COMPETITORS:
            b = competitors[competitors['Dataset'].eq(ds) & competitors['Model'].eq(model)]
            if len(b) != 3 or set(b['Seed']) != set(SEEDS):
                continue
            b = b.set_index('Seed')
            diffs = [float(s.loc[seed,'MCC'] - b.loc[seed,'MCC']) for seed in SEEDS]
            pairs.append({'Dataset': ds, 'Competitor': model, 'Seeds': '42,43,44',
                          'Spectral_MCC_Mean': float(s['MCC'].mean()),
                          'Competitor_MCC_Mean': float(b['MCC'].mean()),
                          'MCC_Difference_Mean': sum(diffs)/3,
                          'Spectral_Better_Seeds': sum(1 for d in diffs if d>0),
                          'Competitor_Better_Seeds': sum(1 for d in diffs if d<0)})
    pd.DataFrame(pairs, columns=['Dataset','Competitor','Seeds','Spectral_MCC_Mean',
        'Competitor_MCC_Mean','MCC_Difference_Mean','Spectral_Better_Seeds',
        'Competitor_Better_Seeds']).to_csv(
        target/'spectral_vs_competitor_paired.csv', index=False, encoding='utf-8-sig')
    (target/'validation_warnings.txt').write_text('\n'.join(issues) if issues else 'No issues detected.\n',
                                                   encoding='utf-8')

    print('\n=== COVERAGE FOR FAIR 10-DATASET / 3-SEED COMPARISON ===')
    cov = pd.DataFrame(coverage)
    print(cov.groupby('Model', sort=False)['Found'].sum().to_string())
    print(f'\nSpectralNFST tuned runs: {len(spectral)}/30')
    print(f'Competitor results available: {len(competitors)}/{len(COMPETITORS)*30}')
    print(f'Missing dataset-model-seed rows: {len(missing)}')
    print(f'Complete paired dataset/model comparisons: {len(pairs)}')
    print(f'Validation warnings: {len(issues)}')
    print(f'Reports: {target}')
    print('\nIMPORTANT: CSV aggregation alone cannot verify that all models used identical splits,')
    print('same train/test data, and an independent validation strategy. Verify before publishing.')


if __name__ == '__main__':
    main()
