# -*- coding: utf-8 -*-
"""
[9.3]_catboost_ova_seed_ensemble.py
- V2 FE 그대로 + CatBoost OVA(One-vs-All)로 5-seed 시드 앙상블
- 산출물:
  * [9.3]_cv_scores_seed_ensemble_ova.csv
  * [9.3]_oof_probs_seed_ens_ova.npy
  * [9.3]_oof_preds_seed_ens_ova.npy
  * [9.3]_test_probs_seed_ens_ova.npy
  * [9.3]_submission_seed_ens_ova.csv
  * [9.3]_submission_seed_ens_ova_bias.csv  (옵션: [8.2]_best_bias.npy 있을 때)
"""

import os, time, json
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

# ===== 공통 경로/설정 =====
DATA_PATH = "./"
TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")
TEST_PATH  = os.path.join(DATA_PATH, "test.csv")
TARGET = "support_needs"
ID_COL = "ID"
N_CLASSES = 3

# V2 FE 구성(7.x 고정)
TE_COLS   = ["subscription_type", "gender_subscription"]
QBINS     = 20
WINSOR_LO = 0.02
WINSOR_HI = 0.98
TE_SMOOTH = 10.0

N_SPLITS  = 5
CV_SEED   = 42
SEED_LIST = [42, 52, 62, 72, 82]

# CatBoost OVA 기본값(안정형)
CB_BASE = {
    "iterations": 1600,
    "depth": 7,
    "learning_rate": 0.015,
    "l2_leaf_reg": 10.0,
    "colsample_bylevel": 0.8,
    "random_strength": 0.15,
    "bagging_temperature": 0.25,
    "auto_class_weights": "Balanced",
    "loss_function": "MultiClassOneVsAll",
    "verbose": 0,
}

# ============ FE 유틸 ============
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

def prepare_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["contract_length"] = out["contract_length"].astype(str)
    out["age_group"] = out["age_group"].astype(str)
    out["gender_subscription"] = (out["gender"].astype(str) + "_" +
                                  out["subscription_type"].astype(str))
    return out

def _winsorize_series(s: pd.Series, lower=0.01, upper=0.99):
    ql = s.quantile(lower); qu = s.quantile(upper)
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

# ---- Multiclass Target Encoding (OOF + Full) ----
def cv_target_encode_multiclass(train_df: pd.DataFrame,
                                y: pd.Series,
                                te_cols: List[str],
                                n_splits=N_SPLITS,
                                seed=CV_SEED,
                                smoothing=TE_SMOOTH):
    K = N_CLASSES
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = train_df.copy()
    prior = np.bincount(y, minlength=K) / len(y)
    for col in te_cols:
        for k in range(K):
            out[f"TE_{col}_c{k}"] = np.nan

    for tr_idx, va_idx in skf.split(train_df, y):
        X_tr, y_tr = train_df.iloc[tr_idx], y.iloc[tr_idx]
        X_va = train_df.iloc[va_idx]
        fmap = {}
        for col in te_cols:
            d = {}
            grp = pd.concat([X_tr[col].astype(str), y_tr], axis=1).groupby(col)
            for cat, g in grp:
                cnt = np.bincount(g[TARGET].to_numpy(), minlength=K)
                prob = (cnt + smoothing / K) / (cnt.sum() + smoothing)
                d[str(cat)] = prob
            fmap[col] = d
        for col in te_cols:
            arr = np.vstack([fmap[col].get(str(v), prior) for v in X_va[col].astype(str).values])
            for k in range(K):
                out.loc[X_va.index, f"TE_{col}_c{k}"] = arr[:, k]

    for col in te_cols:
        for k in range(K):
            out[f"TE_{col}_c{k}"] = out[f"TE_{col}_c{k}"].fillna(prior[k])

    full_map: Dict[str, Dict[str, np.ndarray]] = {col: {} for col in te_cols}
    for col in te_cols:
        grp = pd.concat([train_df[col].astype(str), y], axis=1).groupby(col)
        for cat, g in grp:
            cnt = np.bincount(g[TARGET].to_numpy(), minlength=K)
            prob = (cnt + smoothing / K) / (cnt.sum() + smoothing)
            full_map[col][str(cat)] = prob
    return out, full_map, prior

def apply_te_from_map(df: pd.DataFrame, te_cols: List[str],
                      full_map: Dict[str, Dict[str, np.ndarray]],
                      prior: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    for col in te_cols:
        arr = np.vstack([full_map[col].get(str(v), prior) for v in out[col].astype(str).values])
        for k in range(len(prior)):
            out[f"TE_{col}_c{k}"] = arr[:, k]
    return out

def apply_bias_to_probs(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    eps = 1e-15
    logp = np.log(np.clip(probs, eps, 1.0))
    z = logp + bias.reshape(1, -1)
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=1, keepdims=True)

# ============ 메인 ============
def main():
    t0 = time.time()
    print("--- [9.3] CatBoost OVA Seed Ensemble 시작 ---")

    # 데이터 로드
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    X_raw = train.drop([ID_COL, TARGET], axis=1).copy()
    y = train[TARGET].astype(int).copy()
    Xte_raw = test.drop([ID_COL], axis=1).copy()
    test_ids = test[ID_COL].values

    # FE: train
    X = base_add_features(X_raw)
    X = prepare_types(X)
    X = winsor_block(X, WINSOR_LO, WINSOR_HI)
    X = build_interactions_and_bins(X, QBINS)
    te_cols = [c for c in TE_COLS if c in X.columns]
    X_te, full_map, prior = cv_target_encode_multiclass(X, y, te_cols, n_splits=N_SPLITS, seed=CV_SEED, smoothing=TE_SMOOTH)
    cat_features = prepare_cat_features(X_te)

    # FE: test
    Xtest = base_add_features(Xte_raw)
    Xtest = prepare_types(Xtest)
    Xtest = winsor_block(Xtest, WINSOR_LO, WINSOR_HI)
    Xtest = build_interactions_and_bins(Xtest, QBINS)
    Xtest = apply_te_from_map(Xtest, te_cols, full_map, prior)

    # 공용 CV
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)

    avg_oof_probs  = np.zeros((len(y), N_CLASSES), dtype=float)
    avg_test_probs = np.zeros((len(Xtest), N_CLASSES), dtype=float)
    all_rows = []

    for seed in SEED_LIST:
        print(f"\n[Seed {seed}]")
        oof_probs = np.zeros((len(y), N_CLASSES), dtype=float)
        test_probs = np.zeros((len(Xtest), N_CLASSES), dtype=float)
        fold_scores = []

        for fold, (tr, va) in enumerate(skf.split(X_te, y), start=1):
            params = dict(CB_BASE, random_seed=seed)
            model = CatBoostClassifier(**params, cat_features=cat_features)
            model.fit(
                X_te.iloc[tr], y.iloc[tr],
                eval_set=[(X_te.iloc[va], y.iloc[va])],
                early_stopping_rounds=150,
                verbose=0
            )

            pv = model.predict_proba(X_te.iloc[va])
            oof_probs[va] = pv
            pred = pv.argmax(axis=1)
            f1 = f1_score(y.iloc[va], pred, average="macro")
            fold_scores.append(f1)

            test_probs += model.predict_proba(Xtest) / N_SPLITS

        mean_f1, std_f1 = float(np.mean(fold_scores)), float(np.std(fold_scores, ddof=1))
        print(f"Seed {seed} CV Macro F1: {mean_f1:.5f} ± {std_f1:.5f}")
        all_rows.append({"seed": seed, "mean_macro_f1": mean_f1, "std_macro_f1": std_f1})

        avg_oof_probs  += oof_probs / len(SEED_LIST)
        avg_test_probs += test_probs / len(SEED_LIST)

    oof_preds = avg_oof_probs.argmax(axis=1).astype(int)
    overall_f1 = f1_score(y, oof_preds, average="macro")

    # 저장
    np.save("[9.3]_oof_probs_seed_ens_ova.npy",  avg_oof_probs)
    np.save("[9.3]_oof_preds_seed_ens_ova.npy",  oof_preds)
    np.save("[9.3]_test_probs_seed_ens_ova.npy", avg_test_probs)
    pd.DataFrame(all_rows + [{"seed": "avg_oof", "mean_macro_f1": overall_f1, "std_macro_f1": np.nan}]
                ).to_csv("[9.3]_cv_scores_seed_ensemble_ova.csv", index=False)

    # 기본 제출
    sub = pd.DataFrame({ID_COL: test_ids, TARGET: avg_test_probs.argmax(axis=1).astype(int)})
    sub.to_csv("[9.3]_submission_seed_ens_ova.csv", index=False)

    # (옵션) 8.2 bias 적용 제출
    bias_path = "[8.2]_best_bias.npy"
    if os.path.exists(bias_path):
        try:
            best_bias = np.load(bias_path)
            test_probs_bias = apply_bias_to_probs(avg_test_probs, best_bias)
            sub_bias = pd.DataFrame({ID_COL: test_ids, TARGET: test_probs_bias.argmax(axis=1).astype(int)})
            sub_bias.to_csv("[9.3]_submission_seed_ens_ova_bias.csv", index=False)
            print(f"Bias 적용 제출 저장: [9.3]_submission_seed_ens_ova_bias.csv (bias={np.round(best_bias,4)})")
        except Exception as e:
            print(f"Bias 적용 실패(무시): {e}")

    # 요약
    print("\n===== OVA Seed Ensemble Summary =====")
    for r in all_rows:
        print(f"seed={r['seed']} -> CV Macro F1: {r['mean_macro_f1']:.5f} ± {r['std_macro_f1']:.5f}")
    print(f"\nAvg-OOF argmax Macro F1 (OVA seed-ens): {overall_f1:.5f}")
    print("Saved:",
          "[9.3]_cv_scores_seed_ensemble_ova.csv,",
          "[9.3]_oof_probs_seed_ens_ova.npy,",
          "[9.3]_oof_preds_seed_ens_ova.npy,",
          "[9.3]_test_probs_seed_ens_ova.npy,",
          "[9.3]_submission_seed_ens_ova.csv",
          "(+ bias 제출이 있으면 [9.3]_submission_seed_ens_ova_bias.csv)")
    print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()
