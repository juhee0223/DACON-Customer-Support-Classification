# -*- coding: utf-8 -*-
"""
[8.1]_imbalance_ablation_catboost.py
- 클래스 불균형 전략 4종 5-Fold 공정 비교:
  1) Baseline: auto_class_weights="Balanced"
  2) Manual Class Weights: class_weights=[1/p_k]
  3) SMOTENC: fold 내부 오버샘플
  4) Logit Adjustment: Baseline 확률에 prior 보정(τ ∈ {0.5,1.0,1.5})
- FE는 7.3에서 고른 V2(하드코딩) 사용: TE_COLS=["subscription_type","gender_subscription"], QBINS=20, winsor(0.02,0.98), TE_SMOOTH=10.0
- 산출물:
  * [8.1]_cv_results.csv
  * [8.1]_oof_probs_<strategy>.npy
  * [8.1]_test_probs_<strategy>.npy
  * [8.1]_submission_<strategy>.csv
"""

import os, json, time, math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

# SMOTENC
try:
    from imblearn.over_sampling import SMOTENC
    _HAS_IMBLEARN = True
except Exception:
    _HAS_IMBLEARN = False

# =================== 공통 설정 ===================
DATA_PATH = "./"
TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")
TEST_PATH  = os.path.join(DATA_PATH, "test.csv")

TARGET = "support_needs"
ID_COL = "ID"
N_CLASSES = 3

# CatBoost 튜닝(기존과 동일)
CB_PARAMS_BASE = {
    'iterations': 1807,
    'depth': 7,
    'learning_rate': 0.011007442573869236,
    'l2_leaf_reg': 9.960037311199006,
    'colsample_bylevel': 0.8239914444396733,
    'random_strength': 0.15658132267797456,
    'bagging_temperature': 0.2528316917680684,
    'verbose': 0
}

# V2 FE (하드코딩)
TE_COLS = ["subscription_type", "gender_subscription"]
QBINS = 20
WINSOR_LO, WINSOR_HI = 0.02, 0.98
TE_SMOOTH = 10.0
N_SPLITS = 5
SEED = 42

# =================== 공통 FE 함수 (7.2/7.4와 합치) ===================
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

# ---- Multiclass OOF Target Encoding + FullMap ----
def cv_target_encode_multiclass(train_df: pd.DataFrame, y: pd.Series, te_cols: List[str],
                                n_splits=N_SPLITS, seed=SEED, smoothing=TE_SMOOTH):
    K = N_CLASSES
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = train_df.copy()
    prior = np.bincount(y, minlength=K) / len(y)

    # init OOF cols
    for col in te_cols:
        for k in range(K):
            out[f"TE_{col}_c{k}"] = np.nan

    # fold maps and apply
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

    # fill remaining with prior
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
                      full_map: Dict[str, Dict[str, np.ndarray]], prior: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    for col in te_cols:
        arr = np.vstack([full_map[col].get(str(v), prior) for v in out[col].astype(str).values])
        for k in range(len(prior)):
            out[f"TE_{col}_c{k}"] = arr[:, k]
    return out

# ---- Logit Adjustment ----
def logit_adjustment(probs: np.ndarray, prior: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """
    probs: (n, K) softmax 확률
    prior: (K,) train class prior
    p_adj ∝ p_k * exp(-tau * log prior_k)  (동치: logit에 -tau*log(prior) 더하기)
    """
    eps = 1e-15
    prior = np.clip(prior, eps, 1.0)
    log_prior = np.log(prior)
    logit = np.log(np.clip(probs, eps, 1.0))
    logit_adj = logit - tau * log_prior
    # re-normalize
    m = logit_adj.max(axis=1, keepdims=True)
    expv = np.exp(logit_adj - m)
    return expv / expv.sum(axis=1, keepdims=True)

# ---- 유틸 ----
def get_class_weights_from_y(y: pd.Series) -> List[float]:
    cnt = np.bincount(y, minlength=N_CLASSES).astype(float)
    p = cnt / cnt.sum()
    return (1.0 / np.clip(p, 1e-9, 1.0)).tolist(), p

def dataframe_cat_indices(df: pd.DataFrame, cat_cols: List[str]) -> List[int]:
    return [df.columns.get_loc(c) for c in cat_cols if c in df.columns]

# =================== 메인 ===================
def main():
    t0 = time.time()
    print("--- [8.1] Imbalance Ablation 시작 ---")

    # Load
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    X_raw = train.drop([ID_COL, TARGET], axis=1).copy()
    y = train[TARGET].astype(int).copy()
    Xte_raw = test.drop([ID_COL], axis=1).copy()
    test_ids = test[ID_COL].values

    # ======= FE: train/test 공통 V2 =======
    # train
    X = base_add_features(X_raw)
    X = prepare_types(X)
    X = winsor_block(X, WINSOR_LO, WINSOR_HI)
    X = build_interactions_and_bins(X, QBINS)
    te_cols = [c for c in TE_COLS if c in X.columns]
    X_te, full_map, prior = cv_target_encode_multiclass(X, y, te_cols, N_SPLITS, SEED, TE_SMOOTH)
    cat_features = prepare_cat_features(X_te)
    cat_idx = dataframe_cat_indices(X_te, cat_features)

    # test
    Xtest = base_add_features(Xte_raw)
    Xtest = prepare_types(Xtest)
    Xtest = winsor_block(Xtest, WINSOR_LO, WINSOR_HI)
    Xtest = build_interactions_and_bins(Xtest, QBINS)
    Xtest = apply_te_from_map(Xtest, te_cols, full_map, prior)

    # 공용 CV 분할
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # ================= 전략 1: Baseline (auto_class_weights) =================
    print("\n[Strategy] baseline_auto_weight")
    oof_probs_baseline = np.zeros((len(y), N_CLASSES))
    test_probs_baseline = np.zeros((len(Xtest), N_CLASSES))
    f1s_baseline, fold_rows = [], []

    for fold, (tr, va) in enumerate(skf.split(X_te, y), start=1):
        params = dict(CB_PARAMS_BASE, auto_class_weights="Balanced")
        model = CatBoostClassifier(**params, cat_features=cat_features)
        model.fit(X_te.iloc[tr], y.iloc[tr],
                  eval_set=[(X_te.iloc[va], y.iloc[va])],
                  early_stopping_rounds=100, verbose=0)
        pv = model.predict_proba(X_te.iloc[va])
        oof_probs_baseline[va] = pv
        pred = pv.argmax(axis=1)
        f1 = f1_score(y.iloc[va], pred, average="macro")
        f1s_baseline.append(f1)
        test_probs_baseline += model.predict_proba(Xtest) / N_SPLITS
        fold_rows.append({"strategy":"baseline_auto_weight", "fold":fold, "macro_f1":f1})

    # ================= 전략 2: Manual class_weights =================
    print("\n[Strategy] manual_class_weights")
    cw_list, class_prior = get_class_weights_from_y(y)
    oof_probs_cw = np.zeros((len(y), N_CLASSES))
    test_probs_cw = np.zeros((len(Xtest), N_CLASSES))
    f1s_cw = []

    for fold, (tr, va) in enumerate(skf.split(X_te, y), start=1):
        params = dict(CB_PARAMS_BASE, class_weights=cw_list)  # auto 사용 안 함
        model = CatBoostClassifier(**params, cat_features=cat_features)
        model.fit(X_te.iloc[tr], y.iloc[tr],
                  eval_set=[(X_te.iloc[va], y.iloc[va])],
                  early_stopping_rounds=100, verbose=0)
        pv = model.predict_proba(X_te.iloc[va])
        oof_probs_cw[va] = pv
        pred = pv.argmax(axis=1)
        f1 = f1_score(y.iloc[va], pred, average="macro")
        f1s_cw.append(f1)
        test_probs_cw += model.predict_proba(Xtest) / N_SPLITS
        fold_rows.append({"strategy":"manual_class_weights", "fold":fold, "macro_f1":f1})

    # ================= 전략 3: SMOTENC (fold 내부) =================
    print("\n[Strategy] smotenc")
    if not _HAS_IMBLEARN:
        print("WARNING: imbalanced-learn 미설치 → SMOTENC 스킵됩니다. `pip install imbalanced-learn` 후 재시도하세요.")
        f1s_smote = []
        oof_probs_smote = np.zeros((len(y), N_CLASSES))
        test_probs_smote = np.zeros((len(Xtest), N_CLASSES))
    else:
        oof_probs_smote = np.zeros((len(y), N_CLASSES))
        test_probs_smote = np.zeros((len(Xtest), N_CLASSES))
        f1s_smote = []
        smote = SMOTENC(categorical_features=cat_idx, random_state=SEED)
        for fold, (tr, va) in enumerate(skf.split(X_te, y), start=1):
            Xtr_df, ytr = X_te.iloc[tr].copy(), y.iloc[tr].copy()
            Xva_df, yva = X_te.iloc[va].copy(), y.iloc[va].copy()

            # SMOTENC (DataFrame→np.array로 갔다가 다시 복원)
            X_res, y_res = smote.fit_resample(Xtr_df.values, ytr.values)
            X_res_df = pd.DataFrame(X_res, columns=Xtr_df.columns)
            # 문자열 카테고리 열은 원래 dtype(object)로 되돌림
            for c in cat_features:
                if c in X_res_df.columns:
                    X_res_df[c] = X_res_df[c].astype(str)

            params = dict(CB_PARAMS_BASE)  # 가중치 사용 안 함 (이미 재샘플링)
            model = CatBoostClassifier(**params, cat_features=cat_features)
            model.fit(X_res_df, y_res,
                      eval_set=[(Xva_df, yva)],
                      early_stopping_rounds=100, verbose=0)
            pv = model.predict_proba(Xva_df)
            oof_probs_smote[va] = pv
            pred = pv.argmax(axis=1)
            f1 = f1_score(yva, pred, average="macro")
            f1s_smote.append(f1)
            test_probs_smote += model.predict_proba(Xtest) / N_SPLITS
            fold_rows.append({"strategy":"smotenc", "fold":fold, "macro_f1":f1})

    # ================= 전략 4: Logit Adjustment (Baseline 확률 보정) =================
    print("\n[Strategy] logit_adjustment_from_baseline")
    taus = [0.5, 1.0, 1.5]
    # OOF baseline 확률로 τ 스윕(한 개 τ를 전체에 고정)
    best_tau, best_mean = None, -1.0
    for tau in taus:
        oof_adj = logit_adjustment(oof_probs_baseline, prior=class_prior, tau=tau) if 'class_prior' in locals() \
                  else logit_adjustment(oof_probs_baseline, prior=np.bincount(y, minlength=N_CLASSES)/len(y), tau=tau)
        preds = oof_adj.argmax(axis=1)
        # fold별로 점수 집계(동일 분할이므로 간단 평균)
        f1 = f1_score(y, preds, average="macro")
        if f1 > best_mean:
            best_mean, best_tau = f1, tau
    print(f"Best tau = {best_tau} (OOF macro F1 ≈ {best_mean:.5f})")

    # 최종 tau로 test 보정
    prior_use = class_prior if 'class_prior' in locals() else (np.bincount(y, minlength=N_CLASSES)/len(y))
    test_probs_la = logit_adjustment(test_probs_baseline, prior=prior_use, tau=best_tau)

    # ================= 결과 집계/저장 =================
    # 평균/표준편차
    stats = []
    def add_stats(name, scores):
        if len(scores) > 0:
            stats.append({"strategy":name, "mean_macro_f1":float(np.mean(scores)), "std_macro_f1":float(np.std(scores, ddof=1))})

    add_stats("baseline_auto_weight", f1s_baseline)
    add_stats("manual_class_weights", f1s_cw)
    add_stats("smotenc", f1s_smote)

    # 저장: OOF/TEST 확률 & 제출
    np.save("[8.1]_oof_probs_baseline.npy", oof_probs_baseline)
    np.save("[8.1]_test_probs_baseline.npy", test_probs_baseline)
    pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: test_probs_baseline.argmax(axis=1).astype(int)}).to_csv("[8.1]_submission_baseline.csv", index=False)

    np.save("[8.1]_oof_probs_classweights.npy", oof_probs_cw)
    np.save("[8.1]_test_probs_classweights.npy", test_probs_cw)
    pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: test_probs_cw.argmax(axis=1).astype(int)}).to_csv("[8.1]_submission_classweights.csv", index=False)

    np.save("[8.1]_oof_probs_smotenc.npy", locals().get("oof_probs_smote", np.zeros_like(oof_probs_baseline)))
    np.save("[8.1]_test_probs_smotenc.npy", locals().get("test_probs_smote", np.zeros_like(test_probs_baseline)))
    pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: locals().get("test_probs_smote", test_probs_baseline).argmax(axis=1).astype(int)}).to_csv("[8.1]_submission_smotenc.csv", index=False)

    # Logit Adjustment submission
    np.save("[8.1]_test_probs_logit_adjust.npy", test_probs_la)
    pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: test_probs_la.argmax(axis=1).astype(int)}).to_csv("[8.1]_submission_logit_adjust.csv", index=False)

    # CV fold 로그 저장
    cv_log = pd.DataFrame(fold_rows)
    # 요약 붙이기
    for s in stats:
        cv_log = pd.concat([cv_log, pd.DataFrame([{"strategy": s["strategy"], "fold": "mean", "macro_f1": s["mean_macro_f1"]},
                                                  {"strategy": s["strategy"], "fold": "std",  "macro_f1": s["std_macro_f1"]}])],
                           ignore_index=True)
    cv_log.to_csv("[8.1]_cv_results.csv", index=False)

    # 콘솔 요약
    print("\n===== CV Summary =====")
    for s in stats:
        print(f"{s['strategy']}: {s['mean_macro_f1']:.5f} ± {s['std_macro_f1']:.5f}")
    print(f"LogitAdjustment(best tau={best_tau}) OOF approx: {best_mean:.5f} (fold별 표는 baseline과 동일 분할)")

    print("\nSaved: [8.1]_cv_results.csv and submissions:")
    print(" - [8.1]_submission_baseline.csv")
    print(" - [8.1]_submission_classweights.csv")
    print(" - [8.1]_submission_smotenc.csv")
    print(" - [8.1]_submission_logit_adjust.csv")
    print(f"\n완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()
