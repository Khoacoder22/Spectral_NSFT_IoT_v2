#!/usr/bin/env python3
"""Join per-dataset best SpectralNFST hyperparameters with 3-seed competitor ranks.

Run from Spectral-NFST project root:
    py -u code/analyze_spectral_nfst_best.py
    py -u code/compare_tuned_vs_competitors.py
    py -u code/compare_best_config_final.py

Reads existing analysis results. No model training and no edits to source CSVs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MODEL = "SpectralNFST_Tuned"
COMPETITORS = [
    "KNFST", "HHH", "HHHv2", "FTTransformer", "KNN", "TabNet",
    "LDA", "SGD", "SAINT", "GauNB", "LGBM", "MLP",
    "NuSVC", "NC", "RNC",
]
DATASETS = [
    "5G_NIDD_1000", "BoT_IoT_1000", "CIC_IoT2023_1000",
    "Edge_IIoTset_1000", "IoTID20_1000", "IoTID20_2000",
    "N_BaIoT_1000", "ToN_IoT_1000", "ToN_IoT_2000", "UNSW_NB15_1000",
]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {path}\nRun the two analysis scripts listed above first.")
    return pd.read_csv(path, encoding="utf-8-sig")


def need_cols(df: pd.DataFrame, cols: list[str], file_name: str) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"{file_name} is missing columns: {missing}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = ap.parse_args()
    root = args.root.resolve()

    base = root / "results" / "spectral_nfst_analysis"
    cmp_dir = base / "competitor_comparison"
    best_file = base / "best_config_per_dataset.csv"
    rank_file = cmp_dir / "ranking_only_complete_3seed.csv"
    missing_file = cmp_dir / "missing_competitor_runs.csv"
    conflicts_file = base / "duplicate_key_conflicts.csv"
    best = load_csv(best_file)
    ranked = load_csv(rank_file)
    missing = load_csv(missing_file)
    conflicts = load_csv(conflicts_file)

    need_cols(best, ["Dataset", "Scaler", "Q", "Feature", "Gamma", "MCC_Mean", "MCC_Std"], best_file.name)
    need_cols(ranked, ["Dataset", "Model", "MCC_Mean", "MCC_Std", "Seed_Count"], rank_file.name)
    need_cols(missing, ["Dataset", "Model", "Seed"], missing_file.name)

    if best["Dataset"].duplicated().any():
        raise ValueError("best_config_per_dataset.csv contains multiple best rows per dataset")

    wanted = [MODEL] + COMPETITORS
    ranked = ranked[ranked["Model"].isin(wanted) & ranked["Dataset"].isin(DATASETS)].copy()
    ranked = ranked[ranked["Seed_Count"].eq(3)].copy()
    if ranked.duplicated(["Dataset", "Model"]).any():
        raise ValueError("ranking_only_complete_3seed.csv has duplicate dataset/model rows")

    best_cols = ["Dataset", "Scaler", "Q", "Feature", "Gamma", "MCC_Mean", "MCC_Std"]
    if "F1_Mean" in best.columns:
        best_cols.append("F1_Mean")
    best_small = best[best_cols].copy().rename(columns={
        "Scaler": "Best_Scaler", "Q": "Best_Q", "Feature": "Best_Feature",
        "Gamma": "Best_Gamma", "MCC_Mean": "Best_Spectral_MCC_Mean",
        "MCC_Std": "Best_Spectral_MCC_Std", "F1_Mean": "Best_Spectral_F1_Mean",
    })
    best_small.insert(1, "Best_Kernel", "l05_exponential_kernel")
    best_small["Dataset"] = best_small["Dataset"].astype(str).str.strip()
    ranked["Dataset"] = ranked["Dataset"].astype(str).str.strip()

    for c in ("MCC_Mean", "MCC_Std", "F1_Mean", "ACC_Mean", "FPR_Mean"):
        if c in ranked.columns:
            ranked[c] = pd.to_numeric(ranked[c], errors="coerce")
    ranked = ranked.sort_values(
        ["Dataset", "MCC_Mean", "MCC_Std", "Model"],
        ascending=[True, False, True, True], kind="stable",
    )
    ranked["Rank_By_MCC"] = ranked.groupby("Dataset").cumcount() + 1

    leaderboard = ranked.merge(best_small, on="Dataset", how="left", validate="many_to_one")
    if leaderboard["Best_Scaler"].isna().any():
        raise ValueError("Some ranking datasets have no SpectralNFST best hyperparameters")
    leaderboard["Dataset"] = pd.Categorical(leaderboard["Dataset"], categories=DATASETS, ordered=True)
    leaderboard = leaderboard.sort_values(["Dataset", "Rank_By_MCC"])

    summaries = []
    for ds in DATASETS:
        one_best = best_small[best_small["Dataset"].eq(ds)]
        subset = ranked[ranked["Dataset"].eq(ds)]
        if len(one_best) != 1:
            print(f"[WARNING] {ds}: best configuration is missing")
            continue
        b = one_best.iloc[0]
        s = subset[subset["Model"].eq(MODEL)]
        rivals = subset[subset["Model"].isin(COMPETITORS)].sort_values("MCC_Mean", ascending=False)
        valid_rivals = rivals[rivals["MCC_Mean"].notna()]
        row = b.to_dict()
        row.update({
            "Competitors_3Seeds": len(valid_rivals),
            "Competitors_Expected": len(COMPETITORS),
            "Comparison_Status": "COMPLETE" if len(valid_rivals) == len(COMPETITORS) else "PARTIAL",
            "Spectral_Rank": int(s.iloc[0]["Rank_By_MCC"]) if len(s) == 1 else pd.NA,
            "Best_Competitor": valid_rivals.iloc[0]["Model"] if not valid_rivals.empty else pd.NA,
            "Best_Competitor_MCC_Mean": float(valid_rivals.iloc[0]["MCC_Mean"]) if not valid_rivals.empty else float("nan"),
            "Best_Competitor_MCC_Std": float(valid_rivals.iloc[0]["MCC_Std"]) if not valid_rivals.empty else float("nan"),
            "Spectral_Wins_Against": int((valid_rivals["MCC_Mean"] < b["Best_Spectral_MCC_Mean"]).sum()),
            "Spectral_Loses_Against": int((valid_rivals["MCC_Mean"] > b["Best_Spectral_MCC_Mean"]).sum()),
            "Spectral_Ties_Against": int((valid_rivals["MCC_Mean"] == b["Best_Spectral_MCC_Mean"]).sum()),
        })
        row["MCC_Gap_vs_Best_Competitor"] = (
            b["Best_Spectral_MCC_Mean"] - row["Best_Competitor_MCC_Mean"]
            if not valid_rivals.empty else float("nan")
        )
        summaries.append(row)

    summary = pd.DataFrame(summaries)
    if summary.empty:
        raise RuntimeError("No per-dataset comparison could be created")
    summary["Dataset"] = pd.Categorical(summary["Dataset"], categories=DATASETS, ordered=True)
    summary = summary.sort_values("Dataset")

    head_to_head = leaderboard[leaderboard["Model"].isin(COMPETITORS)].copy()
    head_to_head["MCC_Gap_Spectral_minus_Competitor"] = (
        head_to_head["Best_Spectral_MCC_Mean"] - head_to_head["MCC_Mean"]
    )

    cmp_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = cmp_dir / "final_leaderboard_with_best_params.csv"
    summary_path = cmp_dir / "final_spectral_vs_best_competitor.csv"
    head_to_head_path = cmp_dir / "final_spectral_vs_each_competitor.csv"
    leaderboard.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    head_to_head.to_csv(head_to_head_path, index=False, encoding="utf-8-sig")

    view_cols = ["Dataset", "Best_Scaler", "Best_Q", "Best_Feature", "Best_Gamma",
                 "Best_Spectral_MCC_Mean", "Best_Competitor", "Best_Competitor_MCC_Mean",
                 "MCC_Gap_vs_Best_Competitor", "Spectral_Rank", "Competitors_3Seeds"]
    print("\n=== SPECTRALNFST BEST CONFIG VS BEST COMPETITOR (3 SEEDS) ===")
    print(summary[view_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nNote: positive MCC gap means SpectralNFST has higher observed mean MCC.")
    print(f"Missing competitor/dataset/seed rows (coverage script): {len(missing)}")
    print(f"Conflicting SpectralNFST duplicate configuration keys: {len(conflicts)}")
    if len(missing):
        print("WARNING: Some comparisons are partial. Run only missing competitor trials and regenerate reports.")
    if len(conflicts):
        print("WARNING: Resolve duplicate MCC conflicts before treating these tables as final paper results.")
    print("IMPORTANT: Confirm identical sample splits/preprocessing and test-set independence separately.")
    print(f"\nSaved:\n  {summary_path}\n  {leaderboard_path}\n  {head_to_head_path}")


if __name__ == "__main__":
    main()
