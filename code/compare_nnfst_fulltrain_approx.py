"""Optional 17-model ranking including nNFST-Nyström approximate full-train baseline.

Inputs are the 16-model CSV produced by compare_best_config_final.py and
30 selected nNFST_NystromApprox runs. Never touches original reports/CSVs.

Run from project root:
    python -u code/compare_nnfst_fulltrain_approx.py
"""
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
COMPARE = ROOT / 'results' / 'spectral_nfst_analysis' / 'competitor_comparison'
APPROX = ROOT / 'results' / 'nnfst_fulltrain_approx' / 'all_runs.csv'
NAME = 'nNFST_NystromApprox'
SEEDS = {42, 43, 44}


def main():
    base = pd.read_csv(COMPARE / 'final_leaderboard_with_best_params.csv', encoding='utf-8-sig')
    runs = pd.read_csv(APPROX, encoding='utf-8-sig')
    required = {'Data Type', 'Seed', 'MCC', 'Run_Type', 'N_Train_Original',
                'N_Train_Used', 'r', 'Gamma', 'N_Landmarks', 'Kernel', 'SCALER', 'Poly'}
    if not required.issubset(runs.columns):
        raise ValueError(f'Missing nNFST fields: {required - set(runs.columns)}')
    selected = runs[(runs['r'] == 5) & (runs['Gamma'] == 0.1) &
                    (runs['N_Landmarks'] == 384) &
                    (runs['Kernel'] == 'l05_exponential_kernel') &
                    (runs['SCALER'] == 'QuantileTransformer') &
                    (runs['Poly'] == -1)].copy()
    if len(selected) != 30 or selected.duplicated(['Data Type', 'Seed']).any():
        raise ValueError(f'Expected exactly 30 unique selected nNFST runs, found {len(selected)}')
    if (not selected['Run_Type'].eq('FULL_TRAIN_APPROX_KERNEL').all()
            or not selected['N_Train_Original'].eq(selected['N_Train_Used']).all()):
        raise ValueError('Some nNFST rows do not use the full training split')
    datasets = sorted(base['Dataset'].unique())
    if len(datasets) != 10:
        raise ValueError('Expected 10 datasets in existing leaderboard')
    if set(selected['Data Type']) != set(datasets):
        raise ValueError('Dataset list differs between existing benchmark and nNFST runs')
    if any(set(g['Seed'].astype(int)) != SEEDS for _, g in selected.groupby('Data Type')):
        raise ValueError('Need seeds 42, 43, 44 for every dataset')

    ranked_approx = selected.groupby('Data Type', sort=True)['MCC'].agg(['mean', 'std']).reset_index()
    ranked_approx.rename(columns={'Data Type': 'Dataset', 'mean': 'MCC_Mean',
                                  'std': 'MCC_Std'}, inplace=True)
    ranked_approx['Model'] = NAME
    # Avoid changing the original 16-model leaderboard or competitor CSVs.
    ranked = pd.concat([base, ranked_approx], ignore_index=True, sort=False)
    ranked['MCC_Mean'] = pd.to_numeric(ranked['MCC_Mean'], errors='raise')
    ranked['MCC_Std'] = pd.to_numeric(ranked['MCC_Std'], errors='raise')
    counts = ranked.groupby('Dataset')['Model'].nunique()
    if not counts.eq(17).all():
        raise ValueError('All datasets must have exactly 17 unique models')
    ranked['Rank_By_MCC'] = ranked.groupby('Dataset')['MCC_Mean'].rank(
        method='min', ascending=False
    ).astype(int)
    ranked.sort_values(['Dataset', 'Rank_By_MCC', 'Model'], inplace=True)

    wide = ranked.pivot(index='Dataset', columns='Model', values='MCC_Mean')
    difference = wide['SpectralNFST_Tuned'] - wide[NAME]
    if (difference == 0).all():
        p = 1.0
    else:
        p = float(wilcoxon(difference.to_numpy(), alternative='two-sided').pvalue)
    result = ranked_approx.copy()
    result['SpectralNFST_MCC_Mean'] = result['Dataset'].map(wide['SpectralNFST_Tuned'])
    result['Spectral_minus_Approx'] = result['SpectralNFST_MCC_Mean'] - result['MCC_Mean']
    result['Winner_By_Mean_MCC'] = result['Spectral_minus_Approx'].map(
        lambda d: 'SpectralNFST_Tuned' if d > 0 else (NAME if d < 0 else 'Tie')
    )
    result['nNFST_Rank_17'] = result['Dataset'].map(
        ranked[ranked['Model'].eq(NAME)].set_index('Dataset')['Rank_By_MCC']
    )
    out = APPROX.parent
    result.to_csv(out/'spectral_vs_nnfst_nystrom_approx.csv', index=False, encoding='utf-8-sig')
    ranked.to_csv(out/'leaderboard_17_models_including_approx.csv', index=False, encoding='utf-8-sig')
    print('\n=== NNFST FULL-TRAIN NYSTROM APPROXIMATION VS SPECTRALNFST ===')
    print(result[['Dataset','MCC_Mean','MCC_Std','SpectralNFST_MCC_Mean',
                  'Spectral_minus_Approx','Winner_By_Mean_MCC','nNFST_Rank_17']].to_string(
                      index=False, float_format=lambda v: f'{v:.6f}'))
    print(f'\nWilcoxon two-sided p (10 paired dataset MCC means): {p:.6f}')
    print(f'SpectralNFST win rate: {(difference > 0).sum()}/10')
    print(f'Saved: {out}')
    print('IMPORTANT: model is nNFST-Nyström/regularized near-null approximation, NOT exact nNFST.')
    print('Statistical significance is exploratory; overlapping dataset variants are not independent.')


if __name__ == '__main__':
    main()
