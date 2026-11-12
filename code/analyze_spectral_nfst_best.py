#!/usr/bin/env python3
"""Select robust SpectralNFST configurations from full-data result CSVs.

Run from project root:
    py code/analyze_spectral_nfst_best.py

Inputs (read only):
    results/spectral_nfst_result/<dataset>_<scaler>_Q<q>_seed<seed>.csv
    results/spectral_nfst/gamma_<dataset>_full_[Q<q>_]seed<seed>.csv

The ranking uses numeric gamma values and *identical* (scaler, Q, feature,
gamma) settings with all three seeds 42, 43, 44. Q1 with one seed is
reported separately, not compared against three-seed settings.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd

DATASETS = (
    "5G_NIDD_1000",
    "BoT_IoT_1000",
    "CIC_IoT2023_1000",
    "Edge_IIoTset_1000",
    "IoTID20_1000",
    "IoTID20_2000",
    "N_BaIoT_1000",
    "ToN_IoT_1000",
    "ToN_IoT_2000",
    "UNSW_NB15_1000",
)
SCALERS = ("QuantileTransformer", "StandardScaler", "RobustScaler")
NUMERIC_GAMMAS = ("0.001", "0.01", "0.1", "1", "10")
SEEDS = (42, 43, 44)
Q_GRID = (2, 3, 4, 5)
FEATURES = (-1, 0)
KERNEL = "l05_exponential_kernel"
MAIN_RE = re.compile(
    r"^(?P<dataset>.+_\d+)_(?P<scaler>QuantileTransformer|StandardScaler|RobustScaler)"
    r"_Q(?P<q>\d+)_seed(?P<seed>\d+)$"
)
GAMMA_RE = re.compile(
    r"^gamma_(?P<dataset>.+_\d+)_full_(?:Q(?P<q>\d+)_)?seed(?P<seed>\d+)$"
)
KEY = ["Dataset", "Scaler", "Q", "Feature", "Gamma", "Seed"]
METRICS = ["MCC", "F1 Macro", "ACC", "FPR", "Training time", "Test time"]


def gamma_norm(value: object) -> str | None:
    s = str(value).strip().lower()
    if s == "heuristic":
        return None
    try:
        n = float(s)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(n):
        return None
    result = f"{n:g}"
    return result if result in NUMERIC_GAMMAS else None


def int_field(value: object, fallback: int) -> int | None:
    if pd.isna(value):
        return fallback
    try:
        x = float(str(value).strip())
        return int(x) if math.isfinite(x) and x.is_integer() else None
    except (TypeError, ValueError):
        return None


def float_field(value: object) -> float:
    try:
        x = float(str(value).strip())
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def read_rows(path: Path, metadata: dict, kind: str, warnings: list[str]) -> list[dict]:
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except Exception as exc:
        warnings.append(f"CANNOT_READ {path.name}: {exc}")
        return []
    required = {"Feature", "Gamma", "MCC"}
    if not required.issubset(df.columns):
        warnings.append(f"MISSING_COLUMNS {path.name}: {sorted(required - set(df.columns))}")
        return []

    rows = []
    for row_no, (_, item) in enumerate(df.iterrows(), start=2):
        dataset = str(item.get("Dataset", metadata["dataset"])).strip()
        scaler = str(item.get("Scaler", metadata["scaler"])).strip()
        kernel = str(item.get("Kernel", KERNEL)).strip()
        q = int_field(item.get("Q", metadata["q"]), metadata["q"])
        seed = int_field(item.get("Seed", metadata["seed"]), metadata["seed"])
        feature = int_field(item.get("Feature"), 999)
        gamma = gamma_norm(item.get("Gamma"))
        if gamma is None:
            continue  # Intentionally exclude heuristic.
        mcc = float_field(item.get("MCC"))
        if (
            dataset != metadata["dataset"] or scaler != metadata["scaler"]
            or q != metadata["q"] or seed != metadata["seed"]
            or kernel != KERNEL or feature not in FEATURES
            or not math.isfinite(mcc) or not (-1.0 <= mcc <= 1.0)
        ):
            warnings.append(f"INVALID_ROW {path.name}:{row_no} - incompatible metadata or MCC")
            continue
        rec = {
            "Dataset": dataset, "Scaler": scaler, "Q": q,
            "Feature": feature, "Gamma": gamma, "Seed": seed,
            "Kernel": kernel, "MCC": mcc,
            "Source": str(path.relative_to(path.parents[2])),
            "_priority": 2 if kind == "main" else 1,
        }
        for col in METRICS[1:]:
            rec[col] = float_field(item.get(col, "nan"))
        rows.append(rec)
    return rows


def collect(root: Path):
    base = root / "results"
    primary = base / "spectral_nfst_result"
    legacy = base / "spectral_nfst"
    records, warnings = [], []
    for folder, regex, kind in [(primary, MAIN_RE, "main"), (legacy, GAMMA_RE, "gamma")]:
        if not folder.is_dir():
            warnings.append(f"MISSING_FOLDER {folder}")
            continue
        for path in sorted(folder.glob("*.csv")):
            match = regex.fullmatch(path.stem)
            if not match:
                continue  # Ignore *_errors.csv, sample100, ranking, logs.
            m = match.groupdict()
            if m["dataset"] not in DATASETS:
                continue
            metadata = {
                "dataset": m["dataset"],
                "scaler": m.get("scaler") or "QuantileTransformer",
                "q": int(m.get("q") or "2"),
                "seed": int(m["seed"]),
            }
            records.extend(read_rows(path, metadata, kind, warnings))
    return records, warnings


def analyze(root: Path) -> dict[str, Path]:
    records, warnings = collect(root)
    if not records:
        raise RuntimeError("Không tìm thấy CSV kết quả hợp lệ. Kiểm tra thư mục results/ và tên file.")
    out = root / "results" / "spectral_nfst_analysis"
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.DataFrame(records)
    # If same key exists in both result trees, keep the main grid's row.
    # NEVER silently dismiss conflicting MCC values; report them separately.
    raw = raw.sort_values(KEY + ["_priority", "Source"], kind="stable")
    duplicates = raw[raw.duplicated(KEY, keep=False)].copy()
    conflicts = []
    for key, grp in duplicates.groupby(KEY, sort=False):
        mm = grp["MCC"].astype(float)
        diff = float(mm.max() - mm.min())
        if diff > 1e-7:
            conflicts.append({**dict(zip(KEY, key)),
                              "MCC_Min": float(mm.min()), "MCC_Max": float(mm.max()),
                              "MCC_Difference": diff,
                              "Sources": " | ".join(grp["Source"].astype(str))})
    distinct = raw.drop_duplicates(KEY, keep="last").drop(columns=["_priority"])
    distinct = distinct.sort_values(["Dataset", "Scaler", "Q", "Seed", "Feature", "Gamma"])
    unique_path = out / "all_numeric_results.csv"
    distinct.to_csv(unique_path, index=False, encoding="utf-8-sig")

    # Every candidate must have at least one run; only all-three-seed candidates
    # are eligible for the 'best' ranking, to avoid seed selection bias.
    group_key = ["Dataset", "Scaler", "Q", "Feature", "Gamma"]
    candidates = distinct.groupby(group_key, dropna=False).agg(
        Seeds=("Seed", lambda x: ",".join(map(str, sorted(set(x))))),
        Seed_Count=("Seed", "nunique"),
        MCC_Mean=("MCC", "mean"),
        MCC_Std=("MCC", lambda x: float(x.std(ddof=1)) if len(x) > 1 else float("nan")),
        MCC_Min=("MCC", "min"),
        F1_Mean=("F1 Macro", "mean"),
        ACC_Mean=("ACC", "mean"),
        FPR_Mean=("FPR", "mean"),
        Training_Time_Mean_s=("Training time", "mean"),
        Test_Time_Mean_s=("Test time", "mean"),
    ).reset_index()
    candidates["All_3_Seeds"] = candidates["Seeds"].eq("42,43,44")
    candidates = candidates.sort_values(
        ["Dataset", "All_3_Seeds", "MCC_Mean", "MCC_Std", "F1_Mean"],
        ascending=[True, False, False, True, False], kind="stable",
    )
    candidate_path = out / "config_comparison_all.csv"
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    valid = candidates[candidates["All_3_Seeds"]].copy()
    # Higher MCC_mean first; tie breaker lower MCC_std, higher F1_mean.
    valid = valid.sort_values(
        ["Dataset", "MCC_Mean", "MCC_Std", "F1_Mean", "FPR_Mean", "Q"],
        ascending=[True, False, True, False, True, True], kind="stable",
    )
    valid["Rank_Per_Dataset"] = valid.groupby("Dataset").cumcount() + 1
    ranked_path = out / "ranking_three_seeds.csv"
    valid.to_csv(ranked_path, index=False, encoding="utf-8-sig")
    best = valid[valid["Rank_Per_Dataset"] == 1].copy()
    best_path = out / "best_config_per_dataset.csv"
    best.to_csv(best_path, index=False, encoding="utf-8-sig")

    # Q1 cases typically exist only for seed42; report but don't rank them
    # against configurations assessed on all 3 seeds.
    q1 = distinct[distinct["Q"] == 1].copy()
    q1_path = out / "q1_reference_not_three_seed.csv"
    q1.to_csv(q1_path, index=False, encoding="utf-8-sig")

    cov_rows = []
    for dataset in DATASETS:
        d = distinct[distinct["Dataset"] == dataset]
        core = d[d["Q"].isin(Q_GRID) & d["Seed"].isin(SEEDS)]
        cand = candidates[candidates["Dataset"] == dataset]
        cov_rows.append({
            "Dataset": dataset,
            "Numeric_Config_Rows_Q2_to_Q5": len(core),
            "Expected_Numeric_Config_Rows_Q2_to_Q5": len(SCALERS) * len(Q_GRID) * len(SEEDS) * len(FEATURES) * len(NUMERIC_GAMMAS),
            "Missing_Numeric_Config_Rows": len(SCALERS) * len(Q_GRID) * len(SEEDS) * len(FEATURES) * len(NUMERIC_GAMMAS) - len(core),
            "Candidates_All_3_Seeds": int(cand["All_3_Seeds"].sum()),
            "Candidate_Configs_Incomplete_Seeds": int((~cand["All_3_Seeds"]).sum()),
            "Q1_Reference_Rows": len(d[d["Q"] == 1]),
        })
    coverage_path = out / "dataset_coverage.csv"
    pd.DataFrame(cov_rows).to_csv(coverage_path, index=False, encoding="utf-8-sig")

    conflict_path = out / "duplicate_key_conflicts.csv"
    pd.DataFrame(conflicts, columns=KEY + ["MCC_Min", "MCC_Max", "MCC_Difference", "Sources"]).to_csv(
        conflict_path, index=False, encoding="utf-8-sig"
    )
    warnings_path = out / "import_warnings.txt"
    warnings_path.write_text("\n".join(warnings) if warnings else "No import warnings.\n", encoding="utf-8")

    print("\n=== BEST SPECTRAL NFST PER DATASET (MCC MEAN, 3 SEEDS) ===")
    if best.empty:
        print("No configurations have all seeds 42, 43, 44. Check CSV coverage.")
    else:
        cols = ["Dataset", "Scaler", "Q", "Feature", "Gamma", "MCC_Mean", "MCC_Std", "F1_Mean", "Seed_Count"]
        print(best[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n=== DATASET COVERAGE (Q2, Q3, Q4, Q5) ===")
    print(pd.DataFrame(cov_rows)[["Dataset", "Missing_Numeric_Config_Rows", "Candidates_All_3_Seeds"]].to_string(index=False))
    print(f"\nDuplicate keys with conflicting MCC: {len(conflicts)}")
    print(f"Import warnings: {len(warnings)}")
    print(f"Reports saved to: {out}")
    if conflicts:
        print("WARNING: Conflicting duplicate results exist. Review duplicate_key_conflicts.csv before publishing.")
    return {"best": best_path, "all": unique_path, "candidates": candidate_path,
            "ranking": ranked_path, "coverage": coverage_path, "q1": q1_path,
            "conflicts": conflict_path, "warnings": warnings_path}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                   help="Spectral-NFST project root (defaults to parent of code/)")
    args = p.parse_args()
    analyze(args.root.resolve())


if __name__ == "__main__":
    main()
