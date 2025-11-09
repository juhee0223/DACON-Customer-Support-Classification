# -*- coding: utf-8 -*-
"""
[9.2]_mlp_v2_oof_and_test.py
- V2 FE(고정) + Scaler + MLPClassifier 5-Fold
- 산출물:
  * [9.2]_cv_scores_mlp.csv
  * [9.2]_oof_probs_mlp.npy
  * [9.2]_oof_preds_mlp.npy
  * [9.2]_test_probs_mlp.npy
  * [9.2]_submission_mlp.csv
"""

import os, time, json
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ===== 공통 설정 (7.3 고정) =====
DATA_PATH = "./"
TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")
TEST_PATH  = os.path.join(DATA_PATH, "test.csv")
TARGET = "support_needs"
ID_COL = "ID"
N_CLASSES = 3

TE_COLS   = ["subscription_type", "gender_subscription"]
QBINS     = 20
WINSOR_LO = 0.02
WINSOR_HI = 0.98
TE_SMOOTH = 10.0
N_SPLITS  = 5
SEED      = 42

# ===== V2 FE 유틸 =====
def base_add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6
    df["tenure_per_contract"] = df["tenure"] / (df["contract_length"] + eps)
    df["delay_ratio"]         = df["payment_interval"] / (df["contract_length"] + eps)
    df["idle_ratio"]          = df["after_interaction"] / (df["tenure"] + eps)
    df["late_flag"]           = (df["payment_interval"] > 0).astype(int)
    if 'frequent' in df.columns:
        df['total_usage_score'] = df['tenure'] * df['frequent']
    # age bin
    bins   = [0, 20, 30, 40, 50, 60, 100]
    labels = ['10s', '20s', '30s', '40s', '50s', '60+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    df['risk_score'] = df['payment_interval'] + df['after_interaction']
    return df

def prepare_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["contract_length"] = out["contract_length"].astype(str)
    out["age_group"] = out["age_group"].astype(str)
    out["gender_subscription"] = (out["gender"].astype(str) + "_" +
                                  out["subscription_type"].astype(str))
    return out

def _winsorize_series(s: pd.Series, lower=0.01, upper=0.99):
    ql = s.quantile(lower)
    qu = s.quantile(upper)
    return s.clip(lower=ql, upper=qu)

def winsor_block(df: pd.DataFrame, lo=0.01, hi=0.99):
    out = df.copy()
    for c in ["payment_interval", "after_interaction", "tenure"]:
        if c in out.columns:
            out[c] = _winsorize_series(out[c], lo, hi)
    return out

def _quantile_rank_bins(s: pd.Series, bins: int):
    return np.minimum((s.rank(method="average", pct=True) * bins).astype(int), bins-1)

def build_interactions_and_bins(df: pd.DataFrame, qbins: int):
    out = df.copy()
    # 교차/비선형 일부
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
    # 분위 랭크
    for c in ["tenure", "payment_interval", "after_interaction", "tenure_per_contract"]:
        if c in out.columns:
            out[f"{c}_q{qbins}"] = _quantile_rank_bins(out[c], qbins)
    return out

# ---- Multiclass OOF Target Encoding + FullMap ----
def cv_target_encode_multiclass(train_df: pd.DataFrame, y: pd.Series, te_cols, n_splits=5, seed=42, smoothing=10.0):
    K = N_CLASSES
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = train_df.copy()
    prior = np.bincount(y, minlength=K) / len(y)

    for col in te_cols:
        for k in range(K):
            out[f"TE_{col}_c{k}"] = np.nan

    for tr, va in skf.split(train_df, y):
        Xtr, ytr = train_df.iloc[tr], y.iloc[tr]
        Xva = train_df.iloc[va]

        fmap = {}
        for col in te_cols:
            d = {}
            grp = pd.concat([Xtr[col].astype(str), ytr], axis=1).groupby(col)
            for cat, g in grp:
                cnt = np.bincount(g[TARGET].to_numpy(), minlength=K)
                prob = (cnt + smoothing / K) / (cnt.sum() + smoothing)
                d[str(cat)] = prob
            fmap[col] = d

        for col in te_cols:
            arr = np.vstack([fmap[col].get(str(v), prior) for v in Xva[col].astype(str).values])
            for k in range(K):
                out.loc[Xva.index, f"TE_{col}_c{k}"] = arr[:, k]

    for col in te_cols:
        for k in range(K):
            out[f"TE_{col}_c{k}"] = out[f"TE_{col}_c{k}"].fillna(prior[k])

    # full map (test 적용용)
    full_map = {col: {} for col in te_cols}
    for col in te_cols:
        grp = pd.concat([train_df[col].astype(str), y], axis=1).groupby(col)
        for cat, g in grp:
            cnt = np.bincount(g[TARGET].to_numpy(), minlength=K)
            prob = (cnt + smoothing / K) / (cnt.sum() + smoothing)
            full_map[col][str(cat)] = prob

    return out, full_map, prior

def apply_te_from_map(df: pd.DataFrame, te_cols, full_map, prior):
    out = df.copy()
    for col in te_cols:
        arr = np.vstack([full_map[col].get(str(v), prior) for v in out[col].astype(str).values])
        for k in range(len(prior)):
            out[f"TE_{col}_c{k}"] = arr[:, k]
    return out

# ===== 메인 =====
def main():
    t0 = time.time()
    print("--- [9.2] MLP V2 OOF & TEST 시작 ---")

    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    X_raw = train.drop([ID_COL, TARGET], axis=1).copy()
    y     = train[TARGET].astype(int).copy()
    Xte_raw = test.drop([ID_COL], axis=1).copy()
    test_ids = test[ID_COL].values

    # ---- FE: train ----
    X = base_add_features(X_raw)
    X = prepare_types(X)
    X = winsor_block(X, WINSOR_LO, WINSOR_HI)
    X = build_interactions_and_bins(X, QBINS)
    te_cols = [c for c in TE_COLS if c in X.columns]
    X_te, full_map, prior = cv_target_encode_multiclass(X, y, te_cols, n_splits=N_SPLITS, seed=SEED, smoothing=TE_SMOOTH)

    # ---- FE: test ----
    Xtest = base_add_features(Xte_raw)
    Xtest = prepare_types(Xtest)
    Xtest = winsor_block(Xtest, WINSOR_LO, WINSOR_HI)
    Xtest = build_interactions_and_bins(Xtest, QBINS)
    Xtest = apply_te_from_map(Xtest, te_cols, full_map, prior)

    # ---- 수치형 컬럼만 선택 (문자열 카테고리 제외) ----
    num_cols = [c for c in X_te.columns if X_te[c].dtype != object]
    Xn  = X_te[num_cols].copy()
    Xtn = Xtest[num_cols].copy()

    # ---- CV ----
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_probs = np.zeros((len(y), N_CLASSES), dtype=float)
    oof_preds = np.full(len(y), -1, dtype=int)
    test_probs = np.zeros((len(Xtn), N_CLASSES), dtype=float)
    rows = []

    for fold, (tr, va) in enumerate(skf.split(Xn, y), start=1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xn.iloc[tr])
        Xva = scaler.transform(Xn.iloc[va])
        Xte = scaler.transform(Xtn)

        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=256,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=SEED + fold,  # fold별 고정
            verbose=False
        )
        clf.fit(Xtr, y.iloc[tr])

        pv = clf.predict_proba(Xva)
        oof_probs[va] = pv
        pred = pv.argmax(axis=1).astype(int)
        oof_preds[va] = pred
        f1 = f1_score(y.iloc[va], pred, average="macro")
        rows.append({"fold": fold, "macro_f1": f1})
        test_probs += clf.predict_proba(Xte) / N_SPLITS
        print(f"[Fold {fold}] Macro F1={f1:.5f}")

    mean_f1 = float(np.mean([r["macro_f1"] for r in rows]))
    std_f1  = float(np.std([r["macro_f1"] for r in rows], ddof=1))
    print(f"\nCV Macro F1: {mean_f1:.5f} ± {std_f1:.5f}")

    # ---- 저장 ----
    np.save("[9.2]_oof_probs_mlp.npy",  oof_probs)
    np.save("[9.2]_oof_preds_mlp.npy",  oof_preds)
    np.save("[9.2]_test_probs_mlp.npy", test_probs)
    pd.DataFrame(rows + [{"fold":"mean","macro_f1":mean_f1},{"fold":"std","macro_f1":std_f1}]).to_csv("[9.2]_cv_scores_mlp.csv", index=False)

    sub = pd.DataFrame({ID_COL: test_ids, TARGET: test_probs.argmax(axis=1).astype(int)})
    sub.to_csv("[9.2]_submission_mlp.csv", index=False)

    print("Saved: [9.2]_oof_probs_mlp.npy, [9.2]_test_probs_mlp.npy, [9.2]_submission_mlp.csv, [9.2]_cv_scores_mlp.csv")
    print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    from sklearn.model_selection import StratifiedKFold
    main()
