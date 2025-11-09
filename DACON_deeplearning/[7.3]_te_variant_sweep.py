# -*- coding: utf-8 -*-
"""
[7.3]_te_variant_sweep.py
- Target Encoding/FE 설정 스윕으로 5-Fold CV 성능 비교
- 스윕 파라미터:
  * TE smoothing: [5, 10, 20, 50]
  * TE columns set: 여러 프리셋
  * Quantile bins: [5, 10, 20]
  * Winsorize: [(0.01,0.99), (0.02,0.98)]
- 산출물: [7.3]_te_sweep_results.csv
"""

import os, time, json, itertools
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

DATA_PATH = "./"
TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")
TARGET = "support_needs"
ID_COL = "ID"
N_CLASSES = 3

# CatBoost 파라미터 (7.2와 동일)
TUNED_CB = {
    'iterations': 1807,
    'depth': 7,
    'learning_rate': 0.011007442573869236,
    'l2_leaf_reg': 9.960037311199006,
    'colsample_bylevel': 0.8239914444396733,
    'random_strength': 0.15658132267797456,
    'bagging_temperature': 0.2528316917680684,
    'auto_class_weights': 'Balanced',
    'verbose': 0
}

# ---------- 공통 FE ----------
def base_add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6
    df["tenure_per_contract"] = df["tenure"] / (df["contract_length"] + eps)
    df["delay_ratio"]         = df["payment_interval"] / (df["contract_length"] + eps)
    df["idle_ratio"]          = df["after_interaction"] / (df["tenure"] + eps)
    df["late_flag"]           = (df["payment_interval"] > 0).astype(int)
    if 'frequent' in df.columns:
        df['total_usage_score'] = df['tenure'] * df['frequent']
    bins   = [0, 20, 30, 40, 50, 60, 100]
    labels = ['10s', '20s', '30s', '40s', '50s', '60+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    df['risk_score'] = df['payment_interval'] + df['after_interaction']
    return df

def _winsorize_series(s: pd.Series, lower=0.01, upper=0.99):
    ql = s.quantile(lower)
    qu = s.quantile(upper)
    return s.clip(lower=ql, upper=qu)

def _quantile_rank_bins(s: pd.Series, bins: int):
    # 0..bins-1 정수
    return np.minimum((s.rank(method="average", pct=True) * bins).astype(int), bins-1)

def prepare_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["contract_length"] = out["contract_length"].astype(str)
    out["age_group"] = out["age_group"].astype(str)
    out["gender_subscription"] = (out["gender"].astype(str) + "_" +
                                  out["subscription_type"].astype(str))
    return out

def winsor_block(df: pd.DataFrame, lo=0.01, hi=0.99):
    out = df.copy()
    for c in ["payment_interval", "after_interaction", "tenure"]:
        if c in out.columns:
            out[c] = _winsorize_series(out[c], lo, hi)
    return out

def build_interactions_and_bins(df: pd.DataFrame, qbins: int):
    out = df.copy()
    if set(["tenure", "payment_interval"]).issubset(out.columns):
        out["tenure_x_payint"] = out["tenure"] * out["payment_interval"]
    if set(["tenure", "after_interaction"]).issubset(out.columns):
        out["tenure_x_after"] = out["tenure"] * out["after_interaction"]
    if "frequent" in out.columns:
        out["freq_x_after"] = out["frequent"] * out["after_interaction"]
    if set(["delay_ratio", "risk_score"]).issubset(out.columns):
        out["delay_x_risk"] = out["delay_ratio"] * out["risk_score"]
    if "risk_score" in out.columns:
        out["risk_log1p"] = np.log1p(out["risk_score"].clip(lower=0))
    if "idle_ratio" in out.columns:
        out["idle_sq"] = out["idle_ratio"] ** 2
    for c in ["tenure", "payment_interval", "after_interaction", "tenure_per_contract"]:
        if c in out.columns:
            out[f"{c}_q{qbins}"] = _quantile_rank_bins(out[c], qbins)
    return out

def prepare_cat_features(df: pd.DataFrame) -> List[str]:
    return [c for c in ["gender", "subscription_type", "age_group", "contract_length", "gender_subscription"]
            if c in df.columns and df[c].dtype == object]

# ---------- Multiclass CV-TE ----------
def cv_target_encode_multiclass(train_df: pd.DataFrame,
                                y: pd.Series,
                                te_cols: List[str],
                                n_splits=5,
                                seed=42,
                                smoothing=20.0):
    K = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = train_df.copy()
    prior = np.bincount(y, minlength=K) / len(y)

    for col in te_cols:
        for k in range(K):
            out[f"TE_{col}_c{k}"] = np.nan

    for tr_idx, va_idx in skf.split(train_df, y):
        X_tr, y_tr = train_df.iloc[tr_idx], y.iloc[tr_idx]
        X_va = train_df.iloc[va_idx]
        # fold map
        fmap = {}
        for col in te_cols:
            d = {}
            grp = pd.concat([X_tr[col].astype(str), y_tr], axis=1).groupby(col)
            for cat, g in grp:
                cnt = np.bincount(g[TARGET].to_numpy(), minlength=K)
                prob = (cnt + smoothing / K) / (cnt.sum() + smoothing)
                d[str(cat)] = prob
            fmap[col] = d
        # apply
        for col in te_cols:
            arr = np.vstack([fmap[col].get(str(v), prior) for v in X_va[col].astype(str).values])
            for k in range(K):
                out.loc[X_va.index, f"TE_{col}_c{k}"] = arr[:, k]

    for col in te_cols:
        for k in range(K):
            out[f"TE_{col}_c{k}"] = out[f"TE_{col}_c{k}"].fillna(prior[k])

    return out

# ---------- 실험 유틸 ----------
def run_cv(X_df: pd.DataFrame, y: pd.Series, cat_features: List[str]) -> Tuple[float, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, va_idx in skf.split(X_df, y):
        m = CatBoostClassifier(**TUNED_CB, cat_features=cat_features)
        m.fit(X_df.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X_df.iloc[va_idx], y.iloc[va_idx])],
              early_stopping_rounds=100, verbose=0)
        p = m.predict(X_df.iloc[va_idx], prediction_type='Class')
        p = np.asarray(p).reshape(-1).astype(int)
        scores.append(f1_score(y.iloc[va_idx], p, average="macro"))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))

def main():
    t0 = time.time()
    print("--- [7.3] TE/FE Variant Sweep 시작 ---")

    train = pd.read_csv(TRAIN_PATH)
    X_raw = train.drop([ID_COL, TARGET], axis=1)
    y = train[TARGET].astype(int)

    # 스윕 그리드
    te_smooth_list = [5, 10, 20, 50]
    te_sets = [
        ["subscription_type", "age_group", "gender", "contract_length", "gender_subscription"],
        ["subscription_type", "age_group", "gender"],
        ["subscription_type", "gender_subscription"],
        ["subscription_type", "age_group"],
    ]
    qbins_list = [5, 10, 20]
    winsor_list = [(0.01, 0.99), (0.02, 0.98)]

    results = []
    exp_id = 0

    for smoothing, te_cols, qbins, (lo, hi) in itertools.product(te_smooth_list, te_sets, qbins_list, winsor_list):
        exp_id += 1
        # 1) 기본 FE
        X = base_add_features(X_raw)
        # 2) 타입/조합
        X = prepare_types(X)
        # 3) 윈저
        X = winsor_block(X, lo=lo, hi=hi)
        # 4) 교차항/분위
        X = build_interactions_and_bins(X, qbins=qbins)
        # 5) CV-TE
        te_cols_use = [c for c in te_cols if c in X.columns]
        X_te = cv_target_encode_multiclass(X, y, te_cols=te_cols_use, n_splits=5, seed=42, smoothing=smoothing)
        # 6) Cat features
        cat_features = prepare_cat_features(X_te)
        # 7) CV
        mean_f1, std_f1 = run_cv(X_te, y, cat_features)

        results.append({
            "exp_id": exp_id,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
            "te_smoothing": smoothing,
            "te_cols": "|".join(te_cols_use),
            "qbins": qbins,
            "winsor": f"{lo}-{hi}"
        })
        print(f"[{exp_id:03d}] F1={mean_f1:.5f} (std={std_f1:.5f}) | smooth={smoothing}, te={te_cols_use}, qbins={qbins}, winsor=({lo},{hi})")

    df = pd.DataFrame(results).sort_values("mean_macro_f1", ascending=False)
    out_csv = "[7.3]_te_sweep_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nTop 10:")
    print(df.head(10).to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()


# (venv) jh@jh0223:~/DLDACON$ python3 "[7.3]_te_variant_sweep.py"
# --- [7.3] TE/FE Variant Sweep 시작 ---
# [001] F1=0.49411 (std=0.00277) | smooth=5, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=5, winsor=(0.01,0.99)
# [002] F1=0.49411 (std=0.00277) | smooth=5, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=5, winsor=(0.02,0.98)
# [003] F1=0.49442 (std=0.00278) | smooth=5, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=10, winsor=(0.01,0.99)
# [004] F1=0.49442 (std=0.00278) | smooth=5, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=10, winsor=(0.02,0.98)
# [005] F1=0.49441 (std=0.00312) | smooth=5, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=20, winsor=(0.01,0.99)
# [006] F1=0.49441 (std=0.00312) | smooth=5, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=20, winsor=(0.02,0.98)
# [007] F1=0.49413 (std=0.00294) | smooth=5, te=['subscription_type', 'age_group', 'gender'], qbins=5, winsor=(0.01,0.99)
# [008] F1=0.49413 (std=0.00294) | smooth=5, te=['subscription_type', 'age_group', 'gender'], qbins=5, winsor=(0.02,0.98)
# [009] F1=0.49385 (std=0.00284) | smooth=5, te=['subscription_type', 'age_group', 'gender'], qbins=10, winsor=(0.01,0.99)
# [010] F1=0.49385 (std=0.00284) | smooth=5, te=['subscription_type', 'age_group', 'gender'], qbins=10, winsor=(0.02,0.98)
# [011] F1=0.49445 (std=0.00261) | smooth=5, te=['subscription_type', 'age_group', 'gender'], qbins=20, winsor=(0.01,0.99)
# [012] F1=0.49445 (std=0.00261) | smooth=5, te=['subscription_type', 'age_group', 'gender'], qbins=20, winsor=(0.02,0.98)
# [013] F1=0.49452 (std=0.00294) | smooth=5, te=['subscription_type', 'gender_subscription'], qbins=5, winsor=(0.01,0.99)
# [014] F1=0.49452 (std=0.00294) | smooth=5, te=['subscription_type', 'gender_subscription'], qbins=5, winsor=(0.02,0.98)
# [015] F1=0.49404 (std=0.00326) | smooth=5, te=['subscription_type', 'gender_subscription'], qbins=10, winsor=(0.01,0.99)
# [016] F1=0.49404 (std=0.00326) | smooth=5, te=['subscription_type', 'gender_subscription'], qbins=10, winsor=(0.02,0.98)
# [017] F1=0.49514 (std=0.00224) | smooth=5, te=['subscription_type', 'gender_subscription'], qbins=20, winsor=(0.01,0.99)
# [018] F1=0.49514 (std=0.00224) | smooth=5, te=['subscription_type', 'gender_subscription'], qbins=20, winsor=(0.02,0.98)
# [019] F1=0.49444 (std=0.00290) | smooth=5, te=['subscription_type', 'age_group'], qbins=5, winsor=(0.01,0.99)
# [020] F1=0.49444 (std=0.00290) | smooth=5, te=['subscription_type', 'age_group'], qbins=5, winsor=(0.02,0.98)
# [021] F1=0.49421 (std=0.00273) | smooth=5, te=['subscription_type', 'age_group'], qbins=10, winsor=(0.01,0.99)
# [022] F1=0.49421 (std=0.00273) | smooth=5, te=['subscription_type', 'age_group'], qbins=10, winsor=(0.02,0.98)
# [023] F1=0.49440 (std=0.00270) | smooth=5, te=['subscription_type', 'age_group'], qbins=20, winsor=(0.01,0.99)
# [024] F1=0.49440 (std=0.00270) | smooth=5, te=['subscription_type', 'age_group'], qbins=20, winsor=(0.02,0.98)
# [025] F1=0.49422 (std=0.00276) | smooth=10, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=5, winsor=(0.01,0.99)
# [026] F1=0.49422 (std=0.00276) | smooth=10, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=5, winsor=(0.02,0.98)
# [027] F1=0.49422 (std=0.00333) | smooth=10, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=10, winsor=(0.01,0.99)
# [028] F1=0.49422 (std=0.00333) | smooth=10, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=10, winsor=(0.02,0.98)
# [029] F1=0.49448 (std=0.00267) | smooth=10, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=20, winsor=(0.01,0.99)
# [030] F1=0.49448 (std=0.00267) | smooth=10, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=20, winsor=(0.02,0.98)
# [031] F1=0.49417 (std=0.00273) | smooth=10, te=['subscription_type', 'age_group', 'gender'], qbins=5, winsor=(0.01,0.99)
# [032] F1=0.49417 (std=0.00273) | smooth=10, te=['subscription_type', 'age_group', 'gender'], qbins=5, winsor=(0.02,0.98)
# [033] F1=0.49424 (std=0.00289) | smooth=10, te=['subscription_type', 'age_group', 'gender'], qbins=10, winsor=(0.01,0.99)
# [034] F1=0.49424 (std=0.00289) | smooth=10, te=['subscription_type', 'age_group', 'gender'], qbins=10, winsor=(0.02,0.98)
# [035] F1=0.49465 (std=0.00291) | smooth=10, te=['subscription_type', 'age_group', 'gender'], qbins=20, winsor=(0.01,0.99)
# [036] F1=0.49465 (std=0.00291) | smooth=10, te=['subscription_type', 'age_group', 'gender'], qbins=20, winsor=(0.02,0.98)
# [037] F1=0.49452 (std=0.00294) | smooth=10, te=['subscription_type', 'gender_subscription'], qbins=5, winsor=(0.01,0.99)
# [038] F1=0.49452 (std=0.00294) | smooth=10, te=['subscription_type', 'gender_subscription'], qbins=5, winsor=(0.02,0.98)
# [039] F1=0.49404 (std=0.00326) | smooth=10, te=['subscription_type', 'gender_subscription'], qbins=10, winsor=(0.01,0.99)
# [040] F1=0.49404 (std=0.00326) | smooth=10, te=['subscription_type', 'gender_subscription'], qbins=10, winsor=(0.02,0.98)
# [041] F1=0.49514 (std=0.00224) | smooth=10, te=['subscription_type', 'gender_subscription'], qbins=20, winsor=(0.01,0.99)
# [042] F1=0.49514 (std=0.00224) | smooth=10, te=['subscription_type', 'gender_subscription'], qbins=20, winsor=(0.02,0.98)
# [043] F1=0.49433 (std=0.00298) | smooth=10, te=['subscription_type', 'age_group'], qbins=5, winsor=(0.01,0.99)
# [044] F1=0.49433 (std=0.00298) | smooth=10, te=['subscription_type', 'age_group'], qbins=5, winsor=(0.02,0.98)
# [045] F1=0.49454 (std=0.00258) | smooth=10, te=['subscription_type', 'age_group'], qbins=10, winsor=(0.01,0.99)
# [046] F1=0.49454 (std=0.00258) | smooth=10, te=['subscription_type', 'age_group'], qbins=10, winsor=(0.02,0.98)
# [047] F1=0.49422 (std=0.00296) | smooth=10, te=['subscription_type', 'age_group'], qbins=20, winsor=(0.01,0.99)
# [048] F1=0.49422 (std=0.00296) | smooth=10, te=['subscription_type', 'age_group'], qbins=20, winsor=(0.02,0.98)
# [049] F1=0.49462 (std=0.00268) | smooth=20, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=5, winsor=(0.01,0.99)
# [050] F1=0.49462 (std=0.00268) | smooth=20, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=5, winsor=(0.02,0.98)
# [051] F1=0.49433 (std=0.00317) | smooth=20, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=10, winsor=(0.01,0.99)
# [052] F1=0.49433 (std=0.00317) | smooth=20, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=10, winsor=(0.02,0.98)
# [053] F1=0.49444 (std=0.00281) | smooth=20, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=20, winsor=(0.01,0.99)
# [054] F1=0.49444 (std=0.00281) | smooth=20, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=20, winsor=(0.02,0.98)
# [055] F1=0.49416 (std=0.00273) | smooth=20, te=['subscription_type', 'age_group', 'gender'], qbins=5, winsor=(0.01,0.99)
# [056] F1=0.49416 (std=0.00273) | smooth=20, te=['subscription_type', 'age_group', 'gender'], qbins=5, winsor=(0.02,0.98)
# [057] F1=0.49427 (std=0.00290) | smooth=20, te=['subscription_type', 'age_group', 'gender'], qbins=10, winsor=(0.01,0.99)
# [058] F1=0.49427 (std=0.00290) | smooth=20, te=['subscription_type', 'age_group', 'gender'], qbins=10, winsor=(0.02,0.98)
# [059] F1=0.49470 (std=0.00291) | smooth=20, te=['subscription_type', 'age_group', 'gender'], qbins=20, winsor=(0.01,0.99)
# [060] F1=0.49470 (std=0.00291) | smooth=20, te=['subscription_type', 'age_group', 'gender'], qbins=20, winsor=(0.02,0.98)
# [061] F1=0.49452 (std=0.00294) | smooth=20, te=['subscription_type', 'gender_subscription'], qbins=5, winsor=(0.01,0.99)
# [062] F1=0.49452 (std=0.00294) | smooth=20, te=['subscription_type', 'gender_subscription'], qbins=5, winsor=(0.02,0.98)
# [063] F1=0.49404 (std=0.00326) | smooth=20, te=['subscription_type', 'gender_subscription'], qbins=10, winsor=(0.01,0.99)
# [064] F1=0.49404 (std=0.00326) | smooth=20, te=['subscription_type', 'gender_subscription'], qbins=10, winsor=(0.02,0.98)
# [065] F1=0.49514 (std=0.00224) | smooth=20, te=['subscription_type', 'gender_subscription'], qbins=20, winsor=(0.01,0.99)
# [066] F1=0.49514 (std=0.00224) | smooth=20, te=['subscription_type', 'gender_subscription'], qbins=20, winsor=(0.02,0.98)
# [067] F1=0.49427 (std=0.00296) | smooth=20, te=['subscription_type', 'age_group'], qbins=5, winsor=(0.01,0.99)
# [068] F1=0.49427 (std=0.00296) | smooth=20, te=['subscription_type', 'age_group'], qbins=5, winsor=(0.02,0.98)
# [069] F1=0.49453 (std=0.00258) | smooth=20, te=['subscription_type', 'age_group'], qbins=10, winsor=(0.01,0.99)
# [070] F1=0.49453 (std=0.00258) | smooth=20, te=['subscription_type', 'age_group'], qbins=10, winsor=(0.02,0.98)
# [071] F1=0.49429 (std=0.00296) | smooth=20, te=['subscription_type', 'age_group'], qbins=20, winsor=(0.01,0.99)
# [072] F1=0.49429 (std=0.00296) | smooth=20, te=['subscription_type', 'age_group'], qbins=20, winsor=(0.02,0.98)
# [073] F1=0.49417 (std=0.00300) | smooth=50, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=5, winsor=(0.01,0.99)
# [074] F1=0.49417 (std=0.00300) | smooth=50, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=5, winsor=(0.02,0.98)
# [075] F1=0.49410 (std=0.00290) | smooth=50, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=10, winsor=(0.01,0.99)
# [076] F1=0.49410 (std=0.00290) | smooth=50, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=10, winsor=(0.02,0.98)
# [077] F1=0.49424 (std=0.00298) | smooth=50, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=20, winsor=(0.01,0.99)
# [078] F1=0.49424 (std=0.00298) | smooth=50, te=['subscription_type', 'age_group', 'gender', 'contract_length', 'gender_subscription'], qbins=20, winsor=(0.02,0.98)
# [079] F1=0.49428 (std=0.00303) | smooth=50, te=['subscription_type', 'age_group', 'gender'], qbins=5, winsor=(0.01,0.99)
# [080] F1=0.49428 (std=0.00303) | smooth=50, te=['subscription_type', 'age_group', 'gender'], qbins=5, winsor=(0.02,0.98)
# [081] F1=0.49382 (std=0.00284) | smooth=50, te=['subscription_type', 'age_group', 'gender'], qbins=10, winsor=(0.01,0.99)
# [082] F1=0.49382 (std=0.00284) | smooth=50, te=['subscription_type', 'age_group', 'gender'], qbins=10, winsor=(0.02,0.98)
# [083] F1=0.49435 (std=0.00307) | smooth=50, te=['subscription_type', 'age_group', 'gender'], qbins=20, winsor=(0.01,0.99)
# [084] F1=0.49435 (std=0.00307) | smooth=50, te=['subscription_type', 'age_group', 'gender'], qbins=20, winsor=(0.02,0.98)
# [085] F1=0.49436 (std=0.00244) | smooth=50, te=['subscription_type', 'gender_subscription'], qbins=5, winsor=(0.01,0.99)
# [086] F1=0.49436 (std=0.00244) | smooth=50, te=['subscription_type', 'gender_subscription'], qbins=5, winsor=(0.02,0.98)
# [087] F1=0.49436 (std=0.00294) | smooth=50, te=['subscription_type', 'gender_subscription'], qbins=10, winsor=(0.01,0.99)
# [088] F1=0.49436 (std=0.00294) | smooth=50, te=['subscription_type', 'gender_subscription'], qbins=10, winsor=(0.02,0.98)
# [089] F1=0.49446 (std=0.00255) | smooth=50, te=['subscription_type', 'gender_subscription'], qbins=20, winsor=(0.01,0.99)
# [090] F1=0.49446 (std=0.00255) | smooth=50, te=['subscription_type', 'gender_subscription'], qbins=20, winsor=(0.02,0.98)
# [091] F1=0.49438 (std=0.00345) | smooth=50, te=['subscription_type', 'age_group'], qbins=5, winsor=(0.01,0.99)
# [092] F1=0.49438 (std=0.00345) | smooth=50, te=['subscription_type', 'age_group'], qbins=5, winsor=(0.02,0.98)
# [093] F1=0.49457 (std=0.00276) | smooth=50, te=['subscription_type', 'age_group'], qbins=10, winsor=(0.01,0.99)
# [094] F1=0.49457 (std=0.00276) | smooth=50, te=['subscription_type', 'age_group'], qbins=10, winsor=(0.02,0.98)
# [095] F1=0.49435 (std=0.00287) | smooth=50, te=['subscription_type', 'age_group'], qbins=20, winsor=(0.01,0.99)
# [096] F1=0.49435 (std=0.00287) | smooth=50, te=['subscription_type', 'age_group'], qbins=20, winsor=(0.02,0.98)
#
# Top 10:
#  exp_id  mean_macro_f1  std_macro_f1  te_smoothing                               te_cols  qbins    winsor
#      18       0.495138      0.002238             5 subscription_type|gender_subscription     20 0.02-0.98
#      17       0.495138      0.002238             5 subscription_type|gender_subscription     20 0.01-0.99
#      41       0.495138      0.002238            10 subscription_type|gender_subscription     20 0.01-0.99
#      42       0.495138      0.002238            10 subscription_type|gender_subscription     20 0.02-0.98
#      65       0.495138      0.002238            20 subscription_type|gender_subscription     20 0.01-0.99
#      66       0.495138      0.002238            20 subscription_type|gender_subscription     20 0.02-0.98
#      59       0.494699      0.002913            20    subscription_type|age_group|gender     20 0.01-0.99
#      60       0.494699      0.002913            20    subscription_type|age_group|gender     20 0.02-0.98
#      35       0.494646      0.002910            10    subscription_type|age_group|gender     20 0.01-0.99
#      36       0.494646      0.002910            10    subscription_type|age_group|gender     20 0.02-0.98
#
# Saved: [7.3]_te_sweep_results.csv
# 완료! 총 소요시간: 10138.5초