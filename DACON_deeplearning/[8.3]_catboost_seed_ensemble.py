# -*- coding: utf-8 -*-
"""
[8.3]_catboost_seed_ensemble.py
- 7.3에서 고정한 V2 FE 설정 + CatBoost를 시드 앙상블(5 seeds)로 학습
- 동일한 5-Fold 분할을 사용하여 OOF/TEST 확률을 seed 평균
- (옵션) [8.2]_best_bias.npy가 존재하면 평균 TEST 확률에 bias 적용한 제출 추가 생성

산출물:
  * [8.3]_cv_scores_seed_ensemble.csv
  * [8.3]_oof_probs_seed_ens.npy        (평균 OOF 확률)
  * [8.3]_oof_preds_seed_ens.npy        (평균 확률 argmax)
  * [8.3]_test_probs_seed_ens.npy       (평균 TEST 확률)
  * [8.3]_submission_seed_ens.csv       (평균 확률 argmax 제출)
  * [8.3]_submission_seed_ens_bias.csv  (옵션) bias 적용 제출(8.2 bias 존재 시)

실행:
  python3 "[8.3]_catboost_seed_ensemble.py"
"""

import os, json, time
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

# =================== 고정 세팅 ===================
DATA_PATH = "./"
TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")
TEST_PATH  = os.path.join(DATA_PATH, "test.csv")
TARGET = "support_needs"
ID_COL = "ID"
N_CLASSES = 3

# CatBoost (이전 튜닝값)
CB_BASE = {
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

# V2 FE 설정 (7.3 Top)
TE_COLS   = ["subscription_type", "gender_subscription"]
QBINS     = 20
WINSOR_LO = 0.02
WINSOR_HI = 0.98
TE_SMOOTH = 10.0

# CV/Seed
N_SPLITS = 5
CV_SEED  = 42            # CV 분할 고정 (OOF 일관성)
SEED_LIST = [42, 52, 62, 72, 82]  # 모델 랜덤시드 앙상블

# =================== 공통 FE 함수 ===================
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

# ----- Multiclass Target Encoding (OOF + FullMap) -----
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

    # OOF init
    for col in te_cols:
        for k in range(K):
            out[f"TE_{col}_c{k}"] = np.nan

    # fold maps & apply
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

    # fill remain with prior
    for col in te_cols:
        for k in range(K):
            out[f"TE_{col}_c{k}"] = out[f"TE_{col}_c{k}"].fillna(prior[k])

    # full map for test
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

# ----- Bias 적용 유틸 (8.2) -----
def apply_bias_to_probs(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    eps = 1e-15
    logp = np.log(np.clip(probs, eps, 1.0))
    logit_adj = logp + bias.reshape(1, -1)
    m = logit_adj.max(axis=1, keepdims=True)
    expv = np.exp(logit_adj - m)
    return expv / expv.sum(axis=1, keepdims=True)

# =================== 메인 ===================
def main():
    t0 = time.time()
    print("--- [8.3] CatBoost Seed Ensemble 시작 ---")

    # 데이터
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    X_raw = train.drop([ID_COL, TARGET], axis=1).copy()
    y = train[TARGET].astype(int).copy()
    Xte_raw = test.drop([ID_COL], axis=1).copy()
    test_ids = test[ID_COL].values

    # ===== FE: train/test 공통 =====
    # train
    X = base_add_features(X_raw)
    X = prepare_types(X)
    X = winsor_block(X, WINSOR_LO, WINSOR_HI)
    X = build_interactions_and_bins(X, QBINS)
    te_cols = [c for c in TE_COLS if c in X.columns]
    X_te, full_map, prior = cv_target_encode_multiclass(X, y, te_cols, n_splits=N_SPLITS, seed=CV_SEED, smoothing=TE_SMOOTH)
    cat_features = prepare_cat_features(X_te)

    # test
    Xtest = base_add_features(Xte_raw)
    Xtest = prepare_types(Xtest)
    Xtest = winsor_block(Xtest, WINSOR_LO, WINSOR_HI)
    Xtest = build_interactions_and_bins(Xtest, QBINS)
    Xtest = apply_te_from_map(Xtest, te_cols, full_map, prior)

    # 공용 CV 분할(고정)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)

    # 누적 컨테이너(평균용)
    avg_oof_probs  = np.zeros((len(y), N_CLASSES), dtype=float)
    avg_test_probs = np.zeros((len(Xtest), N_CLASSES), dtype=float)
    all_rows = []

    # ===== 시드 루프 =====
    for seed in SEED_LIST:
        print(f"\n[Seed {seed}]")
        # 한 시드의 OOF/TEST 확률
        oof_probs = np.zeros((len(y), N_CLASSES), dtype=float)
        test_probs = np.zeros((len(Xtest), N_CLASSES), dtype=float)
        fold_scores = []

        for fold, (tr, va) in enumerate(skf.split(X_te, y), start=1):
            params = dict(CB_BASE, random_seed=seed)
            model = CatBoostClassifier(**params, cat_features=cat_features)
            model.fit(
                X_te.iloc[tr], y.iloc[tr],
                eval_set=[(X_te.iloc[va], y.iloc[va])],
                early_stopping_rounds=100, verbose=0
            )

            pv = model.predict_proba(X_te.iloc[va])
            oof_probs[va] = pv
            pred = pv.argmax(axis=1)
            f1 = f1_score(y.iloc[va], pred, average="macro")
            fold_scores.append(f1)

            test_probs += model.predict_proba(Xtest) / N_SPLITS

        # 시드별 성능 로그
        mean_f1, std_f1 = float(np.mean(fold_scores)), float(np.std(fold_scores, ddof=1))
        print(f"Seed {seed} CV Macro F1: {mean_f1:.5f} ± {std_f1:.5f}")
        all_rows.append({"seed": seed, "mean_macro_f1": mean_f1, "std_macro_f1": std_f1})

        # 평균에 누적
        avg_oof_probs  += oof_probs / len(SEED_LIST)
        avg_test_probs += test_probs / len(SEED_LIST)

    # ===== 최종 집계 & 저장 =====
    oof_preds = avg_oof_probs.argmax(axis=1).astype(int)
    overall_f1 = f1_score(y, oof_preds, average="macro")

    # 저장
    np.save("[8.3]_oof_probs_seed_ens.npy",  avg_oof_probs)
    np.save("[8.3]_oof_preds_seed_ens.npy",  oof_preds)
    np.save("[8.3]_test_probs_seed_ens.npy", avg_test_probs)
    pd.DataFrame(all_rows + [{"seed": "avg_oof", "mean_macro_f1": overall_f1, "std_macro_f1": np.nan}]
                ).to_csv("[8.3]_cv_scores_seed_ensemble.csv", index=False)

    # 기본 제출 (bias 미적용)
    sub = pd.DataFrame({ID_COL: test_ids, TARGET: avg_test_probs.argmax(axis=1).astype(int)})
    sub.to_csv("[8.3]_submission_seed_ens.csv", index=False)

    # (옵션) 8.2 bias 적용 추가 제출
    bias_path = "[8.2]_best_bias.npy"
    if os.path.exists(bias_path):
        try:
            best_bias = np.load(bias_path)
            test_probs_bias = apply_bias_to_probs(avg_test_probs, best_bias)
            sub_bias = pd.DataFrame({ID_COL: test_ids, TARGET: test_probs_bias.argmax(axis=1).astype(int)})
            sub_bias.to_csv("[8.3]_submission_seed_ens_bias.csv", index=False)
            print(f"Bias 적용 제출 저장: [8.3]_submission_seed_ens_bias.csv (bias={best_bias})")
        except Exception as e:
            print(f"Bias 적용 실패(무시하고 진행): {e}")

    # 콘솔 요약
    print("\n===== Seed Ensemble Summary =====")
    for r in all_rows:
        print(f"seed={r['seed']} -> CV Macro F1: {r['mean_macro_f1']:.5f} ± {r['std_macro_f1']:.5f}")
    print(f"\nAvg-OOF argmax Macro F1 (seed-ens): {overall_f1:.5f}")
    print("Saved:",
          "[8.3]_cv_scores_seed_ensemble.csv,",
          "[8.3]_oof_probs_seed_ens.npy,",
          "[8.3]_oof_preds_seed_ens.npy,",
          "[8.3]_test_probs_seed_ens.npy,",
          "[8.3]_submission_seed_ens.csv",
          "(+ bias 제출이 있으면 [8.3]_submission_seed_ens_bias.csv)")
    print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()
