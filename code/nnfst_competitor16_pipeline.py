#!/usr/bin/env python3
"""Tune and evaluate an independent nNFST implementation as a competitor.

Luồng chương trình:
1. Chia dữ liệu thành outer-train và outer-test.
2. Chia outer-train thành inner-train và validation để chọn tham số.
3. Fit lại nNFST trên toàn bộ outer-train với tham số đã chọn.
4. Đánh giá trên outer-test.
5. Tổng hợp nNFST cùng 16 mô hình hiện có thành bảng 17 mô hình.

Đây là bản cài đặt độc lập, chưa được xác nhận bằng mã chính thức của tác giả.
Kết quả được lưu riêng, không ghi đè CSV của SpectralNFST hay 15 đối thủ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

from cpai.gaussian_elim import form_independent
from cpai.preprocessing import _remove_outliers_lof, _scaler
from nnfst_paper import NNFSTPaper
from run_nnfst_paper import DATASETS, load_prepared, sample_training

# ---------------------------------------------------------------------------
# 1. Cấu hình chung và thư mục lưu kết quả
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'nnfst_competitor16_reimpl'
TUNE = OUT / 'tuning_jobs'
FINAL = OUT / 'fulltrain_jobs'
SEEDS = (42, 43, 44)
MODEL = 'nNFST_Reimplementation'
# Tăng mã này nếu thay kernel hoặc thuật toán để không dùng nhầm cache cũ.
IMPLEMENTATION_ID = 'project_l05_nested_train_preprocess_v2'
VALID_FIELDS = ('Data Type', 'Seed', 'r', 'Gamma', 'Kernel', 'SCALER', 'Poly')


# ---------------------------------------------------------------------------
# 2. Hàm phụ trợ: cache, lưu kết quả, và chọn dataset
# ---------------------------------------------------------------------------
def canonical_tag(config: dict) -> str:
    """Tạo mã cache ổn định từ cấu hình thí nghiệm."""
    config = {**config, 'implementation_id': IMPLEMENTATION_ID}
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def save_row(path: Path, row: dict):
    """Lưu một dòng kết quả theo cách an toàn, tránh file CSV bị ghi dở."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.temp.{os.getpid()}')
    pd.DataFrame([row]).to_csv(temp, index=False, encoding='utf-8-sig')
    os.replace(temp, path)


def error_record(path: Path, exc: Exception):
    """Ghi lỗi chi tiết vào file cạnh job để tiện xem lại."""
    path.with_suffix('.error.txt').write_text(
        f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}', encoding='utf-8'
    )


def completed_row(path: Path):
    """Trả về kết quả cache nếu file tồn tại và MCC hợp lệ."""
    if not path.exists():
        return None
    data = pd.read_csv(path, encoding='utf-8-sig')
    if len(data) != 1 or not np.isfinite(float(data.iloc[0]['MCC'])):
        raise RuntimeError('Invalid success cache: ' + str(path))
    return data.iloc[0].to_dict()


def datasets_selected(names: list[str] | None):
    """Kiểm tra danh sách dataset yêu cầu và ghép với giới hạn dữ liệu."""
    all_map = {f'{x}_{lim}': (x, lim) for x, lim in DATASETS}
    if names is None or names == ['all']:
        return list(all_map.items())
    invalid = set(names) - set(all_map)
    if invalid:
        raise ValueError('Unknown datasets: ' + ', '.join(sorted(invalid)))
    return [(name, all_map[name]) for name in names]


def load_nested_inner(dataset, limit, scaler, poly, seed,
                      validation_fraction, max_fit):
    """Tạo validation trong outer-train và chỉ fit tiền xử lý trên inner-train.

    Validation được tách trước khi fit imputer/scaler. Nhờ vậy, các thống kê
    chuẩn hoá không lấy thông tin từ validation khi chọn tham số.
    """
    from cpai.datasets import load_dataset

    # Đọc dữ liệu thô: cột cuối là nhãn, các cột trước là đặc trưng.
    df, _ = load_dataset(dataset, limit=limit)
    data = df.to_numpy()
    X = data[:, :-1].astype(np.float64)
    y_raw = data[:, -1]

    # Bước 1: tạo outer split 80/20, khớp với phép chia của pipeline dự án.
    outer_train_idx, _outer_test_idx = train_test_split(
        np.arange(len(y_raw)),
        test_size=0.2,
        stratify=y_raw,
        random_state=seed,
    )
    # Bước 2: chỉ tách inner validation từ outer-train, không dùng outer-test.
    y_outer = y_raw[outer_train_idx]
    inner_train_idx, val_idx = train_test_split(
        np.arange(len(outer_train_idx)),
        test_size=validation_fraction,
        stratify=y_outer,
        random_state=seed + 101_101,
    )
    fit_rows = outer_train_idx[inner_train_idx]
    val_rows = outer_train_idx[val_idx]
    X_fit_raw = X[fit_rows].copy()
    X_val_raw = X[val_rows].copy()
    y_fit_raw = y_raw[fit_rows]
    y_val_raw = y_raw[val_rows]

    # Bước 3: với IoTID20, học giá trị điền thiếu chỉ từ inner-train.
    if dataset == 'IoTID20':
        X_fit_raw[np.isinf(X_fit_raw)] = np.nan
        X_val_raw[np.isinf(X_val_raw)] = np.nan
        imputer = SimpleImputer(strategy='mean')
        X_fit_raw = imputer.fit_transform(X_fit_raw)
        X_val_raw = imputer.transform(X_val_raw)

    # Bước 4: mở rộng đặc trưng nếu poly > 1; không fit trên validation.
    if poly > 1:
        polynomial = PolynomialFeatures(poly, interaction_only=True)
        X_fit_raw = polynomial.fit_transform(X_fit_raw)
        X_val_raw = polynomial.transform(X_val_raw)

    # Bước 5: mã hoá nhãn và fit scaler trên inner-train, rồi transform validation.
    order = np.argsort(y_fit_raw)
    X_fit_raw, y_fit_raw = X_fit_raw[order], y_fit_raw[order]
    encoder = LabelEncoder()
    y_fit = encoder.fit_transform(y_fit_raw)
    y_val = encoder.transform(y_val_raw)

    transformer = _scaler(scaler, random_state=seed)
    X_fit = np.nan_to_num(transformer.fit_transform(X_fit_raw), nan=0.0)
    X_val = np.nan_to_num(transformer.transform(X_val_raw), nan=0.0)

    # Bước 6: lọc ngoại lệ và chọn mẫu độc lập chỉ trong phần train.
    X_fit, y_fit = _remove_outliers_lof(X_fit, y_fit)
    if poly != -1:
        X_fit, y_fit = form_independent(X_fit, y_fit, 'l05_exponential_kernel')

    # Bước 7: giới hạn số mẫu train để tune nhanh; giữ nguyên validation.
    X_fit, y_fit = sample_training(X_fit, y_fit, max_fit, seed + 303_303)
    return X_fit, y_fit, X_val, y_val, len(inner_train_idx), len(outer_train_idx)


# ---------------------------------------------------------------------------
# 3. Đọc các job đã hoàn thành để có thể tiếp tục sau khi bị ngắt
# ---------------------------------------------------------------------------
def scan_one_row_dir(path: Path) -> pd.DataFrame:
    """Đọc các CSV hợp lệ trong thư mục job."""
    records = []
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


# ---------------------------------------------------------------------------
# 4. TUNE: chọn tham số bằng điểm validation
# ---------------------------------------------------------------------------
def tune(args):
    """Tìm cấu hình tốt nhất theo MCC trên inner-validation.

    Mỗi cấu hình được chạy với ba seed. Kết quả validation được dùng để chọn
    feature, r và gamma; outer-test chưa được mở ở bước này.
    """
    chosen = datasets_selected(args.datasets)
    TUNE.mkdir(parents=True, exist_ok=True)
    n_fail = 0

    # Bước 1: tạo inner-train/validation riêng cho từng dataset, scaler, feature, seed.
    for name, (dataset, limit) in chosen:
        print(f'\n===== TUNING {name} / holdout validation only =====', flush=True)
        for scaler in args.scalers:
            for poly in args.features:
                for seed in SEEDS:
                    print(f'[LOAD] {name} seed={seed} scaler={scaler} feature={poly}', flush=True)
                    try:
                        xf, yf, xv, yv, inner_original, outer_train_rows = load_nested_inner(
                            dataset, limit, scaler, poly, seed,
                            args.validation_fraction, args.tune_max_train,
                        )
                    except Exception as exc:
                        n_fail += len(args.r_values) * len(args.gamma_values)
                        # Ghi lại cấu hình lỗi; không tính cấu hình này là đã đánh giá.
                        path = TUNE / name / f'load_seed{seed}_feature{poly}_{scaler}.error.txt'
                        path.write_text(
                            f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}',
                            encoding='utf-8',
                        )
                        print(
                            f'[UNSUPPORTED OR INVALID FEATURE] {name} '
                            f'seed={seed} feature={poly} scaler={scaler}: {exc}',
                            flush=True,
                        )
                        continue
                    # Bước 2: thử mọi tổ hợp r và gamma trên cùng validation fold.
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
                                print(
                                    f'[SKIP] {name} seed={seed} r={r} '
                                    f'gamma={gamma:g} scaler={scaler} feature={poly}',
                                    flush=True,
                                )
                                continue
                            try:
                                # Bước 3: fit trên inner-train, rồi đo MCC trên validation.
                                clf = NNFSTPaper(
                                    r=int(r),
                                    gamma=float(gamma),
                                    kernel=args.kernel,
                                    relative_tol=args.relative_tol,
                                    max_eigensize=args.max_eigensize,
                                )
                                t0 = time.perf_counter()
                                clf.fit(xf, yf)
                                fit_sec = time.perf_counter() - t0
                                pred = clf.predict(xv)
                                row = {
                                    'Data Type': name,
                                    'Seed': seed,
                                    'r': r,
                                    'Gamma': gamma,
                                    'Kernel': args.kernel,
                                    'SCALER': scaler,
                                    'Poly': poly,
                                    'Kernel_Implementation': IMPLEMENTATION_ID,
                                    'MCC': float(matthews_corrcoef(yv, pred)),
                                    'F1 Macro': float(
                                        f1_score(
                                            yv,
                                            pred,
                                            average='macro',
                                            zero_division=0,
                                        ) * 100
                                    ),
                                    'ACC': float(accuracy_score(yv, pred) * 100),
                                    'Train_On_Inner': int(inner_original),
                                    'Train_Used_Tuning': len(yf),
                                    'Validation_Rows': len(yv),
                                    'Outer_Train_Rows': outer_train_rows,
                                    'Fit_Seconds': fit_sec,
                                    'Validation_Fraction': args.validation_fraction,
                                    'Tuning_Limit': args.tune_max_train,
                                    'Relative_Tol': args.relative_tol,
                                    'Selection_Set': 'INNER_VALIDATION_NOT_TEST',
                                }
                                save_row(path, row)
                                print(
                                    f'[VAL] {name} seed={seed} r={r} '
                                    f'gamma={gamma:g} MCC={row["MCC"]:.6f} '
                                    f'(train {len(yf)}, validation {len(yv)})',
                                    flush=True,
                                )
                            except Exception as exc:
                                n_fail += 1
                                error_record(path, exc)
                                print(
                                    f'[FAILED] {name} seed={seed} r={r} '
                                    f'gamma={gamma:g}: {exc}',
                                    flush=True,
                                )

    # Bước 4: ghép các job hợp lệ; cache cũ của implementation khác sẽ bị bỏ qua.
    rows = scan_one_row_dir(TUNE)
    rows = rows[
        rows.get(
            'Kernel_Implementation',
            pd.Series(index=rows.index, dtype=object),
        ).eq(IMPLEMENTATION_ID)
    ]
    if rows.empty:
        raise RuntimeError('No successful tuning rows')
    rows.to_csv(OUT / 'validation_runs.csv', index=False, encoding='utf-8-sig')
    group_cols = [
        'Data Type', 'r', 'Gamma', 'Kernel', 'SCALER', 'Poly',
        'Validation_Fraction', 'Tuning_Limit', 'Relative_Tol',
    ]
    stats = (
        rows.groupby(group_cols, dropna=False)['MCC']
        .agg(
            MCC_Validation_Mean='mean',
            MCC_Validation_Std='std',
            Seed_Count='size',
        )
        .reset_index()
    )

    # Bước 5: chỉ xếp hạng cấu hình có đủ kết quả ở cả ba seed.
    stats = stats[stats.Seed_Count == len(SEEDS)].copy()
    stats.sort_values(
        ['Data Type', 'MCC_Validation_Mean', 'MCC_Validation_Std', 'r', 'Gamma'],
        ascending=[True, False, True, True, True],
        inplace=True,
    )
    stats.to_csv(OUT / 'validation_grid_summary.csv', index=False, encoding='utf-8-sig')

    # MCC trung bình cao nhất là tiêu chí chính; độ lệch chuẩn và tham số nhỏ hơn
    # được dùng để phá hòa theo thứ tự sắp xếp ở trên.
    best = stats.drop_duplicates('Data Type', keep='first').copy()
    best.to_csv(OUT / 'best_params.csv', index=False, encoding='utf-8-sig')
    for name, _ in chosen:
        selected = best[best['Data Type'].eq(name)]
        if selected.empty:
            print(
                f'[INCOMPLETE] {name}: no hyperparameter candidate has '
                '3/3 successful seeds',
                flush=True,
            )
        else:
            p = selected.iloc[0]
            print(
                f'[BEST] {name}: r={int(p.r)} gamma={p.Gamma:g} '
                f'scaler={p.SCALER} feature={int(p.Poly)} '
                f'validation_MCC={p.MCC_Validation_Mean:.6f}',
                flush=True,
            )
    print(f'\nTuning jobs failed: {n_fail}. Output: {OUT}', flush=True)
    print(
        'Preprocessing for each validation run was fitted on its inner-training fold only.',
        flush=True,
    )


# ---------------------------------------------------------------------------
# 5. EVALUATE: fit lại bằng toàn bộ outer-train và chấm outer-test
# ---------------------------------------------------------------------------
def evaluate(args):
    """Đánh giá cấu hình đã chọn trên outer-test, sau khi fit lại trên outer-train."""
    chosen = datasets_selected(args.datasets)
    best_file = OUT / 'best_params.csv'
    if not best_file.exists():
        raise FileNotFoundError('Run tune first: missing '+str(best_file))
    best = pd.read_csv(best_file, encoding='utf-8-sig')
    failures = 0

    # Bước 1: lấy tham số đã chọn từ validation; không tune lại trên test.
    for name, (dataset, limit) in chosen:
        b = best[best['Data Type'].eq(name)]
        if len(b) != 1:
            print(
                f'[NO_BEST] {name}: missing or ambiguous params in best_params.csv',
                flush=True,
            )
            failures += 3
            continue
        p = b.iloc[0]
        r = int(p['r'])
        gamma = float(p['Gamma'])
        scaler = str(p['SCALER'])
        poly = int(p['Poly'])
        kernel = str(p['Kernel'])
        for seed in SEEDS:
            cfg = {
                'dataset': name,
                'seed': seed,
                'r': r,
                'gamma': gamma,
                'scaler': scaler,
                'poly': poly,
                'kernel': kernel,
                'relative_tol': float(p['Relative_Tol']),
            }
            path = FINAL / name / f'{canonical_tag(cfg)}.csv'
            if completed_row(path) is not None:
                print(f'[SKIP FULL] {name} seed={seed}', flush=True)
                continue
            try:
                # Bước 2: chuẩn bị outer-train/test bằng preprocessing của pipeline.
                Xtr, ytr, Xte, yte = load_prepared(
                    dataset, limit, scaler, poly, seed
                )
                print(
                    f'[FULL] {name} seed={seed} n_train={len(ytr)} '
                    f'n_test={len(yte)} classes={len(np.unique(ytr))} '
                    f'r={r} gamma={gamma:g}',
                    flush=True,
                )
                if args.max_eigensize and len(ytr) > args.max_eigensize:
                    raise MemoryError(
                        f'FULL kernel requires n={len(ytr)} > safety '
                        f'max-eigensize={args.max_eigensize}. No subsampled '
                        'fallback; use a suitable high-memory machine.'
                    )
                # Bước 3: fit lại trên toàn bộ outer-train rồi dự đoán outer-test.
                clf = NNFSTPaper(
                    r=r,
                    gamma=gamma,
                    kernel=kernel,
                    relative_tol=float(p['Relative_Tol']),
                    max_eigensize=args.max_eigensize,
                )
                t0 = time.perf_counter()
                clf.fit(Xtr, ytr)
                tr_sec = time.perf_counter() - t0

                t1 = time.perf_counter()
                prediction = clf.predict(Xte)
                te_sec = time.perf_counter() - t1

                # Bước 4: lưu chỉ số đánh giá, kích thước dữ liệu và thông tin cấu hình.
                result = {
                    'Data Type': name,
                    'Seed': seed,
                    'r': r,
                    'Gamma': gamma,
                    'Kernel': kernel,
                    'SCALER': scaler,
                    'Poly': poly,
                    'Kernel_Implementation': IMPLEMENTATION_ID,
                    'Model': MODEL,
                    'Input Model': MODEL,
                    'MCC': float(matthews_corrcoef(yte, prediction)),
                    'ACC': float(accuracy_score(yte, prediction) * 100),
                    'F1 Macro': float(
                        f1_score(yte, prediction, average='macro', zero_division=0) * 100
                    ),
                    'Training time': tr_sec,
                    'Test time': te_sec,
                    'N_Train_Original': len(ytr),
                    'N_Train_Used': len(ytr),
                    'N_Test': len(yte),
                    'Train_Classes': len(np.unique(ytr)),
                    'Test_Classes': len(np.unique(yte)),
                    'Run_Type': 'FULL_TRAIN',
                    'Algorithm_Source': 'independent_reimplementation_not_official',
                    'Q_SCC': clf.n_components_,
                    'Projection_Dim': clf.n_projection_dims_,
                    'Within_Null_Error': clf.within_null_relative_error_,
                    'Params_Chosen_By': 'INNER_VALIDATION',
                    'Validation_MCC_Mean': float(p['MCC_Validation_Mean']),
                    'Protocol': 'closed_set',
                    'Relative_Tol': float(p['Relative_Tol']),
                }
                save_row(path, result)
                print(
                    f'[DONE] {name} seed={seed} '
                    f'test_MCC={result["MCC"]:.6f} '
                    f'train_time={tr_sec:.1f}s',
                    flush=True,
                )
            except Exception as exc:
                failures += 1
                error_record(path, exc)
                print(f'[FAILED FULL] {name} seed={seed}: {exc}', flush=True)
    # Bước 5: tổng hợp các kết quả full-train đã hoàn thành.
    rows = scan_one_row_dir(FINAL)
    if not rows.empty:
        rows.to_csv(
            OUT / 'all_fulltrain_versions.csv',
            index=False,
            encoding='utf-8-sig',
        )
    print(
        f'Full evaluation failures in this invocation: {failures}. Output: {OUT}',
        flush=True,
    )
    print(
        'Do not put unverified results in the official 15-competitor directory.',
        flush=True,
    )


# ---------------------------------------------------------------------------
# 6. REPORT: tổng hợp nNFST và xếp hạng chung 17 mô hình
# ---------------------------------------------------------------------------
def report(_args):
    """Tạo bảng 17 mô hình và so sánh nNFST với SpectralNFST."""
    bf = OUT / 'best_params.csv'
    if not bf.exists():
        raise FileNotFoundError('Missing best_params.csv: run tune first')
    best = pd.read_csv(bf, encoding='utf-8-sig')
    if len(best) != len(DATASETS):
        raise ValueError(
            'Need best hyperparameters for all 10 datasets before official '
            '17-model report'
        )
    # Bước 1: kiểm tra đủ tham số và kết quả full-train cho 10 dataset x 3 seed.
    full = scan_one_row_dir(FINAL)
    full = full[
        full.get(
            'Kernel_Implementation',
            pd.Series(index=full.index, dtype=object),
        ).eq(IMPLEMENTATION_ID)
    ]
    if full.empty:
        raise ValueError('No FULL training evaluation rows')
    selection = []
    for dataset, limit in DATASETS:
        ds = f'{dataset}_{limit}'
        b = best[best['Data Type'].eq(ds)]
        if len(b) != 1:
            raise ValueError(f'Missing best params for {ds}')
        p = b.iloc[0]
        records = full[
            (full['Data Type'] == ds)
            & (full['r'].astype(int) == int(p.r))
            & np.isclose(
                full['Gamma'].astype(float),
                float(p.Gamma),
                atol=0,
                rtol=1e-12,
            )
            & (full['Kernel'] == p.Kernel)
            & (full['SCALER'] == p.SCALER)
            & (full['Poly'].astype(int) == int(p.Poly))
            & (full['Run_Type'] == 'FULL_TRAIN')
        ].copy()
        if len(records) != 3 or set(records.Seed.astype(int)) != set(SEEDS):
            raise ValueError(
                f'{ds}: expected 3 FULL_TRAIN rows for selected config, '
                f'got {len(records)}'
            )
        if not (
            records['N_Train_Original'].astype(int)
            == records['N_Train_Used'].astype(int)
        ).all():
            raise ValueError(f'{ds}: training size reduced; cannot report official benchmark')
        selection.append(records)
    # Bước 2: lưu đúng 30 lần chạy của cấu hình đã chọn.
    selected = pd.concat(selection, ignore_index=True)
    if len(selected) != 30:
        raise ValueError('Need exactly 30 selected full-train evaluations')
    selected.to_csv(
        OUT / 'nNFST_Reimplementation_30runs.csv',
        index=False,
        encoding='utf-8-sig',
    )
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
    summary.to_csv(
        OUT / 'nnfst_mean_std.csv',
        index=False,
        encoding='utf-8-sig',
    )
    prev = (
        ROOT / 'results' / 'spectral_nfst_analysis' / 'competitor_comparison'
        / 'final_leaderboard_with_best_params.csv'
    )
    if not prev.exists():
        raise FileNotFoundError(f'Missing 16-model leaderboard: {prev}')
    original = pd.read_csv(prev, encoding='utf-8-sig')
    original = original[original.Model.ne(MODEL)].copy()
    if (
        len(original) != 160
        or original.groupby('Dataset')['Model'].nunique().ne(16).any()
    ):
        raise ValueError('Original ranking should have 160 rows: 16 models across 10 datasets')
    if set(original.Dataset) != set(summary.Dataset):
        raise ValueError('Dataset mismatch between 15 competitors + SpectralNFST and nNFST')
    # Bước 3: ghép 16 mô hình có sẵn với nNFST, rồi xếp hạng MCC từng dataset.
    merged = pd.concat([original, summary], ignore_index=True, sort=False)
    merged['Rank_By_MCC'] = (
        merged.groupby('Dataset')['MCC_Mean']
        .rank(method='min', ascending=False)
        .astype(int)
    )
    merged.sort_values(['Dataset', 'Rank_By_MCC', 'Model'], inplace=True)
    if merged.groupby('Dataset')['Model'].nunique().ne(17).any():
        raise ValueError('Not exactly 17 models per dataset')
    merged.to_csv(
        OUT / 'leaderboard_17_models.csv',
        index=False,
        encoding='utf-8-sig',
    )

    # Bước 4: ghép MCC của SpectralNFST và nNFST theo từng dataset để so sánh.
    wide = merged.pivot(index='Dataset', columns='Model', values='MCC_Mean')
    diff = (wide['SpectralNFST_Tuned'] - wide[MODEL]).astype(float)
    wilcox = (
        1.0
        if (diff == 0).all()
        else float(wilcoxon(diff.to_numpy(), alternative='two-sided').pvalue)
    )
    comp = summary.copy()
    comp['SpectralNFST_MCC_Mean'] = comp.Dataset.map(wide['SpectralNFST_Tuned'])
    comp['Spectral_minus_nNFST'] = (
        comp['SpectralNFST_MCC_Mean'] - comp.MCC_Mean
    )
    comp['nNFST_Rank_17'] = comp.Dataset.map(
        merged[merged.Model.eq(MODEL)]
        .set_index('Dataset')
        .Rank_By_MCC
    )
    comp['Spectral_Wins'] = comp.Spectral_minus_nNFST.gt(0)
    comp.to_csv(
        OUT / 'spectral_vs_nnfst_reimpl.csv',
        index=False,
        encoding='utf-8-sig',
    )
    print('\n=== SELECTED nNFST REIMPLEMENTATION VS SPECTRALNFST ===')
    columns = [
        'Dataset', 'MCC_Mean', 'MCC_Std', 'SpectralNFST_MCC_Mean',
        'Spectral_minus_nNFST', 'nNFST_Rank_17',
    ]
    print(
        comp[columns].to_string(
            index=False,
            float_format=lambda value: f'{value:.6f}',
        )
    )
    print(f'\nWilcoxon two-sided p-value (10 paired dataset means): {wilcox:.6f}')
    print(f'SpectralNFST win rate: {int(diff.gt(0).sum())}/10')
    print(f'Saved report files to: {OUT}')
    print(
        'IMPORTANT: nNFST is independent, unverified reimplementation, '
        'not official implementation.'
    )
    print(
        'Preprocessing is fitted on inner-train rows during tuning; '
        'final test rows are used only for evaluation.'
    )


# ---------------------------------------------------------------------------
# 7. CLI: đọc lệnh người dùng và gọi đúng giai đoạn
# ---------------------------------------------------------------------------
def main():
    """Đọc lệnh terminal và chuyển sang đúng giai đoạn của pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='command', required=True)

    # Lệnh tune: thử các cấu hình và chọn tham số bằng MCC trên validation.
    tune_parser = modes.add_parser(
        'tune',
        help='Choose feature/r/gamma via held-out inner VALIDATION MCC, not TEST MCC',
    )
    tune_parser.add_argument('--datasets', nargs='+', default=['all'])
    tune_parser.add_argument(
        '--r-values', type=int, nargs='+', default=[5, 10, 15, 20, 30]
    )
    tune_parser.add_argument(
        '--gamma-values', type=float, nargs='+', default=[0.001, 0.01, 0.1]
    )
    tune_parser.add_argument(
        '--scalers', nargs='+', default=['QuantileTransformer']
    )
    tune_parser.add_argument(
        '--features',
        nargs='+',
        type=int,
        default=[-1, 0, 1, 2],
        help='Exactly four values, passed through as preprocess(poly=feature)',
    )
    tune_parser.add_argument(
        '--kernel',
        choices=['l05_exponential_kernel'],
        default='l05_exponential_kernel',
        help='Fixed to L05 for comparison',
    )
    tune_parser.add_argument(
        '--tune-max-train',
        type=int,
        default=800,
        help='Only for inner-tuning fitting. 0=full inner fold; final refit always FULL.',
    )
    tune_parser.add_argument('--validation-fraction', type=float, default=0.2)
    tune_parser.add_argument('--relative-tol', type=float, default=1e-9)
    tune_parser.add_argument('--max-eigensize', type=int, default=1200)

    # Lệnh plan: tính số lần fit dự kiến, không huấn luyện mô hình.
    plan_parser = modes.add_parser(
        'plan',
        help='Show 4-feature/r/gamma grid size; performs no training',
    )
    plan_parser.add_argument('--datasets', nargs='+', default=['all'])
    plan_parser.add_argument('--features', nargs='+', type=int, default=[-1, 0, 1, 2])
    plan_parser.add_argument(
        '--r-values', nargs='+', type=int, default=[5, 10, 15, 20, 30]
    )
    plan_parser.add_argument(
        '--gamma-values', nargs='+', type=float, default=[0.001, 0.01, 0.1]
    )
    plan_parser.add_argument('--scalers', nargs='+', default=['QuantileTransformer'])

    # Lệnh evaluate: fit cấu hình đã chọn trên outer-train và chấm outer-test.
    evaluate_parser = modes.add_parser(
        'evaluate',
        help='Use selected params and FULL outer training set to score each test split once',
    )
    evaluate_parser.add_argument('--datasets', nargs='+', default=['all'])
    evaluate_parser.add_argument(
        '--max-eigensize',
        type=int,
        default=3000,
        help='Safety stop; do NOT set 14000 on 16GiB PC casually',
    )

    # Lệnh report: tổng hợp kết quả khi đã đủ 10 dataset x 3 seed.
    modes.add_parser(
        'report',
        help='Report only when all 10*3 FULL runs exist; never modifies old results',
    )

    args = parser.parse_args()

    # Kiểm tra lưới feature trước khi bắt đầu công việc tốn thời gian.
    if args.command in ('tune', 'plan'):
        if len(args.features) != 4 or len(set(args.features)) != 4:
            parser.error('Provide exactly 4 DISTINCT feature values with --features')

    # Chuyển sang đúng bước theo subcommand người dùng đã nhập.
    if args.command == 'plan':
        names = datasets_selected(args.datasets)
        configs_per_dataset = (
            len(args.features)
            * len(args.r_values)
            * len(args.gamma_values)
            * len(args.scalers)
        )
        total_fits = len(names) * len(SEEDS) * configs_per_dataset
        print(
            f'Datasets: {len(names)}, seeds: {len(SEEDS)}, '
            f'features: {args.features}, r: {args.r_values}, '
            f'gamma: {args.gamma_values}, scalers: {args.scalers}'
        )
        print(
            f'Validation candidates per dataset: {configs_per_dataset}; '
            f'TOTAL candidate fits: {total_fits}. '
            'Kernel fixed: l05_exponential_kernel'
        )
    elif args.command == 'tune':
        if not 0 < args.validation_fraction < 0.5:
            parser.error('validation fraction must be between 0 and 0.5')
        if args.tune_max_train and args.tune_max_train < 2 * 34:
            parser.error('tune-max-train too small for 34-class case')
        dataset_count = len(datasets_selected(args.datasets))
        print(
            f'[GRID] {dataset_count} datasets x 3 seeds x '
            f'{len(args.features)} features x {len(args.r_values)} r x '
            f'{len(args.gamma_values)} gammas x {len(args.scalers)} scalers'
        )
        tune(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        report(args)


if __name__ == '__main__':
    main()
