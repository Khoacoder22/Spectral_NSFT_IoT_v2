import pandas as pd
from scipy.stats import wilcoxon

INPUT = r"results\wilcoxon_closed_set_data.csv"

OUR_MODEL = "SpectralNFST-Tuned"

COMPETITORS = [
    "KNFST",
    "HHH",
    "FTTransformer",
    "KNN",
    "HHHv2",
    "TabNet",
    "LDA",
    "SGD",
    "SAINT",
    "GauNB",
    "LGBM",
    "MLP",
    "NuSVC",
    "NC",
    "RNC"
]

DATASETS = [
    "5G_NIDD_1000",
    "BoT_IoT_1000",
    "CIC_IoT2023_1000",
    "N_BaIoT_1000",
    "ToN_IoT_1000",
    "UNSW_NB15_1000",
    "Edge_IIoTset_1000",
    "IoTID20_1000",
    "IoTID20_2000",
    "ToN_IoT_2000"
]

df = pd.read_csv(INPUT)

# ------------------------------------------------------------
# Pivot:
# rows = datasets
# columns = models
# values = mean Macro-F1
# ------------------------------------------------------------

table = df.pivot_table(
    index="Dataset",
    columns="Model",
    values="MacroF1",
    aggfunc="mean"
)

print()
print("=" * 80)
print("Wilcoxon Signed-Rank Test — Closed-set Macro-F1")
print("SpectralNFST-Tuned vs 15 Competitors")
print("=" * 80)

print(
    f"{'Opponent':<18}"
    f"{'N':>6}"
    f"{'Win Rate':>12}"
    f"{'Wins':>8}"
    f"{'Losses':>8}"
    f"{'Ties':>8}"
    f"{'p-value':>14}"
)

print("-" * 80)

results = []

for competitor in COMPETITORS:

    our_values = []
    opponent_values = []

    for dataset in DATASETS:

        if dataset not in table.index:
            continue

        if OUR_MODEL not in table.columns:
            continue

        if competitor not in table.columns:
            continue

        our = table.loc[dataset, OUR_MODEL]
        opponent = table.loc[dataset, competitor]

        if pd.isna(our) or pd.isna(opponent):
            continue

        our_values.append(float(our))
        opponent_values.append(float(opponent))

    n = len(our_values)

    wins = sum(
        a > b
        for a, b in zip(our_values, opponent_values)
    )

    losses = sum(
        a < b
        for a, b in zip(our_values, opponent_values)
    )

    ties = sum(
        a == b
        for a, b in zip(our_values, opponent_values)
    )

    win_rate = wins / n if n > 0 else float("nan")

    # Differences
    differences = [
        a - b
        for a, b in zip(our_values, opponent_values)
        if a != b
    ]

    if len(differences) == 0:
        p_value = float("nan")
    else:
        try:
            test = wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto"
            )

            p_value = test.pvalue

        except Exception:
            p_value = float("nan")

    results.append({
        "Opponent": competitor,
        "N": n,
        "Win Rate": win_rate,
        "Wins": wins,
        "Losses": losses,
        "Ties": ties,
        "p-value": p_value
    })

    print(
        f"{competitor:<18}"
        f"{n:>6}"
        f"{win_rate:>11.1%}"
        f"{wins:>8}"
        f"{losses:>8}"
        f"{ties:>8}"
        f"{p_value:>14.6f}"
    )

print()
print("=" * 80)
print("Interpretation")
print("=" * 80)
print("Win Rate = proportion of datasets where SpectralNFST-Tuned has higher Macro-F1.")
print("p < 0.05 = statistically significant difference.")
print("Wilcoxon test is paired across datasets.")
print()

# Save result
pd.DataFrame(results).to_csv(
    "results/wilcoxon_closed_set_results.csv",
    index=False
)

print("Saved:")
print("results/wilcoxon_closed_set_results.csv")
