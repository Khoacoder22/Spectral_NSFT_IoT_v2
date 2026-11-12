#!/usr/bin/env python3
"""Post-evaluation comparison: SpectralNFST + 15 baselines + nNFST reimplementation.

Requires the 30 full-training nNFST runs and 16 original models already
assembled by nnfst_competitor16_pipeline.py report. Does not train, tune,
or overwrite original model CSVs. Produces exploratory paired Wilcoxon tests.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SPECTRAL = 'SpectralNFST_Tuned'
NNFST = 'nNFST_Reimplementation'
EXPECTED_DATASETS = 10
EXPECTED_MODELS = 17


def holm_correct(pvalues: list[float]) -> list[float]:
    """Holm's familywise error correction; monotone adjusted p-values."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def build_reports(leaderboard: pd.DataFrame, outdir: Path):
    required = {'Dataset', 'Model', 'MCC_Mean', 'MCC_Std'}
    missing = required.difference(leaderboard.columns)
    if missing:
        raise ValueError('Leaderboard missing fields: ' + ', '.join(sorted(missing)))
    src = leaderboard.copy()
    src['Dataset'] = src['Dataset'].astype(str).str.strip()
    src['Model'] = src['Model'].astype(str).str.strip()
    for col in ('MCC_Mean', 'MCC_Std'):
        src[col] = pd.to_numeric(src[col], errors='raise')
    if not np.isfinite(src['MCC_Mean']).all():
        raise ValueError('NaN or Inf MCC: cannot report ranking')
    if src.duplicated(['Dataset','Model']).any():
        raise ValueError('Duplicate dataset/model in leaderboard')
    ds = sorted(src['Dataset'].unique())
    models = sorted(src['Model'].unique())
    if len(ds) != EXPECTED_DATASETS or len(models) != EXPECTED_MODELS:
        raise ValueError(f'Expected 10 datasets and 17 models, got {len(ds)} datasets and {len(models)} models')
    if SPECTRAL not in models or NNFST not in models:
        raise ValueError('SpectralNFST or nNFST missing')
    counts = src.groupby('Dataset')['Model'].nunique()
    if not (counts == EXPECTED_MODELS).all():
        raise ValueError('Each dataset must have exactly 17 models')

    src['MCC_Rank'] = src.groupby('Dataset')['MCC_Mean'].rank(
        ascending=False, method='min').astype(int)
    src = src.sort_values(['Dataset','MCC_Rank','Model'])
    outdir.mkdir(parents=True, exist_ok=True)
    src.to_csv(outdir/'mcc_leaderboard_17.csv', index=False, encoding='utf-8-sig')

    def pretty(r):
        return f'{r.MCC_Mean:.6f} ± {r.MCC_Std:.6f}'
    printable = src.copy()
    printable['MCC_Mean±Std'] = printable.apply(pretty, axis=1)
    matrix = printable.pivot(index='Dataset',columns='Model',values='MCC_Mean±Std')
    matrix.to_csv(outdir/'mcc_mean_std_matrix_17.csv',encoding='utf-8-sig')

    ranks = src.pivot(index='Dataset',columns='Model',values='MCC_Rank')
    ranks.to_csv(outdir/'mcc_ranks_by_dataset_17.csv',encoding='utf-8-sig')
    rank_summary = (src.groupby('Model')
                    .agg(Mean_Rank=('MCC_Rank','mean'),
                         Median_Rank=('MCC_Rank','median'),
                         Number_of_Firsts=('MCC_Rank',lambda s:int((s==1).sum())),
                         Mean_MCC_Across_Datasets=('MCC_Mean','mean'))
                    .reset_index().sort_values(['Mean_Rank','Model']))
    rank_summary.insert(0,'Overall_Rank',np.arange(1,len(rank_summary)+1))
    rank_summary.to_csv(outdir/'model_average_rank_17.csv',index=False,encoding='utf-8-sig')

    wide = src.pivot(index='Dataset',columns='Model',values='MCC_Mean').sort_index()
    comparisons=[]
    for model in sorted(m for m in models if m != SPECTRAL):
        diff = (wide[SPECTRAL]-wide[model]).to_numpy(dtype=float)
        win = int(np.sum(diff > 1e-12))
        loss = int(np.sum(diff < -1e-12))
        tie = len(diff)-win-loss
        if np.all(np.abs(diff)<1e-12):
            p=1.0
        else:
            p=float(wilcoxon(diff,alternative='two-sided',zero_method='wilcox').pvalue)
        comparisons.append({
            'Comparator':model,
            'N_Datasets':len(ds),
            'Spectral_Wins':win,
            'Spectral_Losses':loss,
            'Ties':tie,
            'Spectral_Win_Rate_Percent':round(win/len(ds)*100,2),
            'Mean_MCC_Difference_Spectral_minus_Comparator':float(np.mean(diff)),
            'Median_MCC_Difference':float(np.median(diff)),
            'Wilcoxon_p_two_sided':p,
        })
    tests = pd.DataFrame(comparisons)
    tests['Wilcoxon_p_Holm_16_comparisons']=holm_correct(tests['Wilcoxon_p_two_sided'].tolist())
    tests['Significant_Holm_0_05']=tests['Wilcoxon_p_Holm_16_comparisons']<0.05
    tests = tests.sort_values('Wilcoxon_p_two_sided')
    tests.to_csv(outdir/'spectral_vs_each_16_winrate_wilcoxon.csv',index=False,encoding='utf-8-sig')

    pair = src[src['Model'].isin([SPECTRAL,NNFST])]
    pair = pair.pivot(index='Dataset',columns='Model',values='MCC_Mean').reset_index()
    pair['MCC_Gap_Spectral_minus_nNFST']=pair[SPECTRAL]-pair[NNFST]
    pair.to_csv(outdir/'spectral_vs_nnfst_10datasets.csv',index=False,encoding='utf-8-sig')
    print(f'COMPLETED: {len(ds)} datasets × {len(models)} models')
    print('\nTop models by mean rank:')
    print(rank_summary[['Model','Mean_Rank','Number_of_Firsts']].head(17).to_string(index=False))
    print('\nSpectralNFST vs nNFST:')
    print(tests[tests.Comparator.eq(NNFST)].to_string(index=False))
    print('\nResults saved to:', outdir)
    print('Note: Wilcoxon is exploratory; ensure comparable evaluation protocols and investigate correlated dataset variants.')
    return src, tests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    result=(args.root/'results'/'nnfst_competitor16_reimpl')
    path=result/'leaderboard_17_models.csv'
    if not path.is_file():
        raise SystemExit('Missing leaderboard_17_models.csv; first run: python code/nnfst_competitor16_pipeline.py report')
    build_reports(pd.read_csv(path,encoding='utf-8-sig'),result)


if __name__=='__main__':
    main()
