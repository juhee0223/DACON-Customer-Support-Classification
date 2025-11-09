# # -*- coding: utf-8 -*-
# """
# [7.2]_feature_eval_catboost.py
# - Baseline Feature vs. V2 Feature 공정 비교 (동일 5-Fold 분할)
# - CatBoost(튜닝 파라미터)로 교차검증 수행
# - V2는 '훈련 폴드에서 학습한 멀티클래스 Target Encoding(확률형, Laplace smoothing)'을
#   검증 폴드에만 적용(OOF 방식)하여 누수 방지
# - 산출물:
#   * [7.2]_feature_eval_report.md
#   * [7.2]_oof_preds_baseline.npy
#   * [7.2]_oof_preds_v2.npy
#   * [7.2]_fold_scores.csv
#
# 실행:
#   python3 "[7.2]_feature_eval_catboost.py"
# """
#
# import os, json, time
# import numpy as np
# import pandas as pd
# from typing import List, Dict
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import f1_score
# from catboost import CatBoostClassifier
#
# # --------------------- 공통 설정 ---------------------
# DATA_PATH = "./"
# TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")
# TEST_PATH  = os.path.join(DATA_PATH, "test.csv")
#
# TARGET = "support_needs"
# ID_COL = "ID"
# N_CLASSES = 3
#
# TUNED_CB = {
#     'iterations': 1807,
#     'depth': 7,
#     'learning_rate': 0.011007442573869236,
#     'l2_leaf_reg': 9.960037311199006,
#     'colsample_bylevel': 0.8239914444396733,
#     'random_strength': 0.15658132267797456,
#     'bagging_temperature': 0.2528316917680684,
#     'auto_class_weights': 'Balanced',
#     'verbose': 0
# }
#
# # --------------------- Baseline FE ---------------------
# def base_add_features(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     eps = 1e-6
#     df["tenure_per_contract"] = df["tenure"] / (df["contract_length"] + eps)
#     df["delay_ratio"]         = df["payment_interval"] / (df["contract_length"] + eps)
#     df["idle_ratio"]          = df["after_interaction"] / (df["tenure"] + eps)
#     df["late_flag"]           = (df["payment_interval"] > 0).astype(int)
#     if 'frequent' in df.columns:
#         df['total_usage_score'] = df['tenure'] * df['frequent']
#     bins   = [0, 20, 30, 40, 50, 60, 100]
#     labels = ['10s', '20s', '30s', '40s', '50s', '60+']
#     df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
#     df['risk_score'] = df['payment_interval'] + df['after_interaction']
#     return df
#
# def prepare_cat_features(df: pd.DataFrame) -> List[str]:
#     cats = []
#     for c in ["gender", "subscription_type", "age_group", "contract_length", "gender_subscription"]:
#         if c in df.columns and df[c].dtype == object:
#             cats.append(c)
#     return cats
#
# # --------------------- V2 FE(TE 포함) ---------------------
# def _winsorize_series(s: pd.Series, lower=0.01, upper=0.99):
#     ql = s.quantile(lower)
#     qu = s.quantile(upper)
#     return s.clip(lower=ql, upper=qu)
#
# def _quantile_rank_0to9(s: pd.Series):
#     return (s.rank(method="average", pct=True) - 1e-9).clip(0, 0.999999).mul(10).astype(int)
#
# def v2_prepare_types(df: pd.DataFrame) -> pd.DataFrame:
#     out = df.copy()
#     out["contract_length"] = out["contract_length"].astype(str)
#     out["age_group"] = out["age_group"].astype(str)
#     out["gender_subscription"] = (out["gender"].astype(str) + "_" +
#                                   out["subscription_type"].astype(str))
#     return out
#
# def v2_winsorize_block(df: pd.DataFrame) -> pd.DataFrame:
#     out = df.copy()
#     for c in ["payment_interval", "after_interaction", "tenure"]:
#         if c in out.columns:
#             out[c] = _winsorize_series(out[c], 0.01, 0.99)
#     return out
#
# def v2_build_interactions(df: pd.DataFrame) -> pd.DataFrame:
#     out = df.copy()
#     if set(["tenure", "payment_interval"]).issubset(out.columns):
#         out["tenure_x_payint"] = out["tenure"] * out["payment_interval"]
#     if set(["tenure", "after_interaction"]).issubset(out.columns):
#         out["tenure_x_after"] = out["tenure"] * out["after_interaction"]
#     if "frequent" in out.columns:
#         out["freq_x_after"] = out["frequent"] * out["after_interaction"]
#     if set(["delay_ratio", "risk_score"]).issubset(out.columns):
#         out["delay_x_risk"] = out["delay_ratio"] * out["risk_score"]
#     if "risk_score" in out.columns:
#         out["risk_log1p"] = np.log1p(out["risk_score"].clip(lower=0))
#     if "idle_ratio" in out.columns:
#         out["idle_sq"] = out["idle_ratio"] ** 2
#     for c in ["tenure", "payment_interval", "after_interaction", "tenure_per_contract"]:
#         if c in out.columns:
#             out[c + "_q10"] = _quantile_rank_0to9(out[c])
#     return out
#
# def cv_target_encode_multiclass(train_df: pd.DataFrame,
#                                 y: pd.Series,
#                                 te_cols: List[str],
#                                 n_splits=5,
#                                 seed=42,
#                                 smoothing=20.0):
#     K = N_CLASSES
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     out = train_df.copy()
#     prior = np.bincount(y, minlength=K) / len(y)
#
#     full_map: Dict[str, Dict[str, np.ndarray]] = {col: {} for col in te_cols}
#     for col in te_cols:
#         grp = pd.concat([train_df[col].astype(str), y], axis=1).groupby(col)
#         for cat, g in grp:
#             cnt = np.bincount(g[TARGET].to_numpy(), minlength=K)
#             prob = (cnt + smoothing / K) / (cnt.sum() + smoothing)
#             full_map[col][str(cat)] = prob
#
#     for col in te_cols:
#         for k in range(K):
#             out[f"TE_{col}_c{k}"] = np.nan
#
#     for tr_idx, va_idx in skf.split(train_df, y):
#         X_tr, y_tr = train_df.iloc[tr_idx], y.iloc[tr_idx]
#         X_va = train_df.iloc[va_idx]
#
#         fold_map: Dict[str, Dict[str, np.ndarray]] = {}
#         for col in te_cols:
#             d = {}
#             grp = pd.concat([X_tr[col].astype(str), y_tr], axis=1).groupby(col)
#             for cat, g in grp:
#                 cnt = np.bincount(g[TARGET].to_numpy(), minlength=K)
#                 prob = (cnt + smoothing / K) / (cnt.sum() + smoothing)
#                 d[str(cat)] = prob
#             fold_map[col] = d
#
#         for col in te_cols:
#             arr = np.vstack([fold_map[col].get(str(v), prior) for v in X_va[col].astype(str).values])
#             for k in range(K):
#                 out.loc[X_va.index, f"TE_{col}_c{k}"] = arr[:, k]
#
#     for col in te_cols:
#         for k in range(K):
#             out[f"TE_{col}_c{k}"] = out[f"TE_{col}_c{k}"].fillna(prior[k])
#
#     return out, full_map, prior
#
# # --------------------- 메인 ---------------------
# def main():
#     t0 = time.time()
#     print("--- [7.2] Feature Eval (CatBoost) 시작 ---")
#
#     train = pd.read_csv(TRAIN_PATH)
#     X = train.drop([ID_COL, TARGET], axis=1).copy()
#     y = train[TARGET].astype(int).copy()
#
#     # Baseline
#     X_base = base_add_features(X)
#     X_base["contract_length"] = X_base["contract_length"].astype(str)
#     X_base["age_group"] = X_base["age_group"].astype(str)
#     cat_features_base = prepare_cat_features(X_base)
#
#     # V2(TE 제외 프레임 만들기)
#     X_v2 = base_add_features(X)
#     X_v2 = v2_prepare_types(X_v2)
#     X_v2 = v2_winsorize_block(X_v2)
#     X_v2 = v2_build_interactions(X_v2)
#     te_cols = [c for c in ["subscription_type", "gender_subscription"] if c in X_v2.columns]
#
#
#     # OOF-TE 생성
#     X_v2_te, full_map, prior = cv_target_encode_multiclass(
#         train_df=X_v2, y=y, te_cols=te_cols, n_splits=5, seed=42, smoothing=20.0
#     )
#     cat_features_v2 = prepare_cat_features(X_v2_te)
#
#     # 동일 5-Fold CV
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
#     oof_pred_base = np.full(len(y), -1, dtype=int)
#     oof_pred_v2   = np.full(len(y), -1, dtype=int)
#     fold_rows, f1s_base, f1s_v2 = [], [], []
#
#     for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
#         # Baseline
#         model_b = CatBoostClassifier(**TUNED_CB, cat_features=cat_features_base)
#         model_b.fit(
#             X_base.iloc[tr_idx], y.iloc[tr_idx],
#             eval_set=[(X_base.iloc[va_idx], y.iloc[va_idx])],
#             early_stopping_rounds=100, verbose=0
#         )
#         pred_b = model_b.predict(X_base.iloc[va_idx], prediction_type='Class')
#         pred_b = np.asarray(pred_b).reshape(-1).astype(int)  # ← 확실한 1D 변환
#         oof_pred_base[va_idx] = pred_b
#         f1_b = f1_score(y.iloc[va_idx], pred_b, average="macro")
#         f1s_base.append(f1_b)
#
#         # V2
#         model_v = CatBoostClassifier(**TUNED_CB, cat_features=cat_features_v2)
#         model_v.fit(
#             X_v2_te.iloc[tr_idx], y.iloc[tr_idx],
#             eval_set=[(X_v2_te.iloc[va_idx], y.iloc[va_idx])],
#             early_stopping_rounds=100, verbose=0
#         )
#         pred_v = model_v.predict(X_v2_te.iloc[va_idx], prediction_type='Class')
#         pred_v = np.asarray(pred_v).reshape(-1).astype(int)  # ← 확실한 1D 변환
#         oof_pred_v2[va_idx] = pred_v
#         f1_v = f1_score(y.iloc[va_idx], pred_v, average="macro")
#         f1s_v2.append(f1_v)
#
#         fold_rows.append({
#             "fold": fold,
#             "baseline_macro_f1": f1_b,
#             "v2_macro_f1": f1_v,
#             "gain": f1_v - f1_b
#         })
#         print(f"[Fold {fold}] Baseline={f1_b:.5f} | V2={f1_v:.5f} | Gain={f1_v - f1_b:+.5f}")
#
#     # 집계/저장
#     mean_b, std_b = np.mean(f1s_base), np.std(f1s_base, ddof=1)
#     mean_v, std_v = np.mean(f1s_v2),   np.std(f1s_v2, ddof=1)
#     gain = mean_v - mean_b
#
#     np.save("[7.2]_oof_preds_baseline_mod.npy", oof_pred_base)
#     np.save("[7.2]_oof_preds_v2_mod.npy",       oof_pred_v2)
#     pd.DataFrame(fold_rows).to_csv("[7.2]_fold_scores_mod.csv", index=False)
#
#     lines = []
#     lines.append("# [7.2] Feature Eval Report (CatBoost)\n")
#     lines.append("## CV 결과 (5-Fold, 동일 분할)")
#     lines.append(f"- Baseline  : mean={mean_b:.5f}, std={std_b:.5f}")
#     lines.append(f"- V2 (OOF-TE): mean={mean_v:.5f}, std={std_v:.5f}")
#     lines.append(f"- **Gain**   : **{gain:+.5f}**\n")
#     lines.append("## 설정 요약")
#     lines.append("- CatBoost 파라미터:")
#     lines.append("```json\n" + json.dumps(TUNED_CB, indent=2) + "\n```")
#     lines.append("- TE 대상 열: " + ", ".join(te_cols))
#     lines.append("- Smoothing: 20.0 (Laplace)")
#     lines.append("- early_stopping_rounds=100")
#     with open("[7.2]_feature_eval_report.md", "w", encoding="utf-8") as f:
#         f.write("\n".join(lines))
#
#     print("\n===== 결과 요약 =====")
#     print(f"Baseline 5-Fold Macro F1: {mean_b:.5f} ± {std_b:.5f}")
#     print(f"V2      5-Fold Macro F1: {mean_v:.5f} ± {std_v:.5f}")
#     print(f"Gain                     : {gain:+.5f}")
#     print("Saved: [7.2]_feature_eval_report.md, [7.2]_oof_preds_*.npy, [7.2]_fold_scores.csv")
#     print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")
#
# if __name__ == "__main__":
#     main()
#
#
# # (venv) jh@jh0223:~/DLDACON$ python3 \[7.2\]_feature_eval_catboost.py
# # --- [7.2] Feature Eval (CatBoost) 시작 ---
# # [Fold 1] Baseline=0.49769 | V2=0.49858 | Gain=+0.00089
# # [Fold 2] Baseline=0.49408 | V2=0.49514 | Gain=+0.00107
# # [Fold 3] Baseline=0.49103 | V2=0.49015 | Gain=-0.00088
# # [Fold 4] Baseline=0.49531 | V2=0.49524 | Gain=-0.00007
# # [Fold 5] Baseline=0.49491 | V2=0.49256 | Gain=-0.00235
# #
# # ===== 결과 요약 =====
# # Baseline 5-Fold Macro F1: 0.49460 ± 0.00241
# # V2      5-Fold Macro F1: 0.49433 ± 0.00317
# # Gain                     : -0.00027
# # Saved: [7.2]_feature_eval_report.md, [7.2]_oof_preds_*.npy, [7.2]_fold_scores.csv
# # 완료! 총 소요시간: 175.7초

# -*- coding: utf-8 -*-
"""
[7.2]_feature_eval_catboost.py (Hard-coded best FE settings)
- Baseline Feature vs. V2 Feature 공정 비교 (동일 5-Fold 분할)
- V2: 고정 설정
    * TE 대상: ["subscription_type", "gender_subscription"]
    * Quantile bins: 20
    * Winsor: (0.02, 0.98)
    * TE smoothing: 10.0
- 산출물:
  * [7.2]_feature_eval_report.md
  * [7.2]_oof_preds_baseline.npy
  * [7.2]_oof_preds_v2.npy
  * [7.2]_fold_scores.csv
"""

import os, json, time
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

# =================== 고정 설정 ===================
DATA_PATH = "./"
TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")
TARGET = "support_needs"
ID_COL = "ID"
N_CLASSES = 3

# CatBoost (이전 튜닝값)
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

# 하드코딩된 V2 FE 설정
TE_COLS = ["subscription_type", "gender_subscription"]
QBINS = 20
WINSOR_LO, WINSOR_HI = 0.02, 0.98
TE_SMOOTH = 10.0
N_SPLITS = 5
SEED = 42

# =================== 공통 FE (Baseline) ===================
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

def prepare_cat_features(df: pd.DataFrame) -> List[str]:
    cats = []
    for c in ["gender", "subscription_type", "age_group", "contract_length", "gender_subscription"]:
        if c in df.columns and df[c].dtype == object:
            cats.append(c)
    return cats

# =================== V2 FE (하드코딩) ===================
def _winsorize_series(s: pd.Series, lower=0.01, upper=0.99):
    ql = s.quantile(lower)
    qu = s.quantile(upper)
    return s.clip(lower=ql, upper=qu)

def _quantile_rank_bins(s: pd.Series, bins: int):
    # 0..bins-1
    return np.minimum((s.rank(method="average", pct=True) * bins).astype(int), bins-1)

def v2_prepare_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["contract_length"] = out["contract_length"].astype(str)
    out["age_group"] = out["age_group"].astype(str)
    out["gender_subscription"] = (out["gender"].astype(str) + "_" +
                                  out["subscription_type"].astype(str))
    return out

def v2_winsorize_block(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["payment_interval", "after_interaction", "tenure"]:
        if c in out.columns:
            out[c] = _winsorize_series(out[c], WINSOR_LO, WINSOR_HI)
    return out

def v2_build_interactions_and_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 교차항
    if set(["tenure", "payment_interval"]).issubset(out.columns):
        out["tenure_x_payint"] = out["tenure"] * out["payment_interval"]
    if set(["tenure", "after_interaction"]).issubset(out.columns):
        out["tenure_x_after"] = out["tenure"] * out["after_interaction"]
    if "frequent" in out.columns:
        out["freq_x_after"] = out["frequent"] * out["after_interaction"]
    if set(["delay_ratio", "risk_score"]).issubset(out.columns):
        out["delay_x_risk"] = out["delay_ratio"] * out["risk_score"]
    # 비선형
    if "risk_score" in out.columns:
        out["risk_log1p"] = np.log1p(out["risk_score"].clip(lower=0))
    if "idle_ratio" in out.columns:
        out["idle_sq"] = out["idle_ratio"] ** 2
    # 분위 랭크(bins=20 고정)
    for c in ["tenure", "payment_interval", "after_interaction", "tenure_per_contract"]:
        if c in out.columns:
            out[f"{c}_q{QBINS}"] = _quantile_rank_bins(out[c], QBINS)
    return out

def cv_target_encode_multiclass(train_df: pd.DataFrame,
                                y: pd.Series,
                                te_cols: List[str],
                                n_splits=N_SPLITS,
                                seed=SEED,
                                smoothing=TE_SMOOTH):
    """
    멀티클래스 OOF-TE 생성:
    - 각 te_col마다 K개 확률 열(TE_{col}_c0..)
    - fold별 train으로 카테고리 분포 추정 후 val에만 적용
    """
    K = N_CLASSES
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = train_df.copy()
    prior = np.bincount(y, minlength=K) / len(y)

    # 초기화
    for col in te_cols:
        for k in range(K):
            out[f"TE_{col}_c{k}"] = np.nan

    for tr_idx, va_idx in skf.split(train_df, y):
        X_tr, y_tr = train_df.iloc[tr_idx], y.iloc[tr_idx]
        X_va = train_df.iloc[va_idx]

        # fold map
        fmap: Dict[str, Dict[str, np.ndarray]] = {}
        for col in te_cols:
            d = {}
            grp = pd.concat([X_tr[col].astype(str), y_tr], axis=1).groupby(col)
            for cat, g in grp:
                cnt = np.bincount(g[TARGET].to_numpy(), minlength=K)
                prob = (cnt + smoothing / K) / (cnt.sum() + smoothing)
                d[str(cat)] = prob
            fmap[col] = d

        # val 적용
        for col in te_cols:
            arr = np.vstack([fmap[col].get(str(v), prior) for v in X_va[col].astype(str).values])
            for k in range(K):
                out.loc[X_va.index, f"TE_{col}_c{k}"] = arr[:, k]

    # 남은 결측(prior로 채움)
    for col in te_cols:
        for k in range(K):
            out[f"TE_{col}_c{k}"] = out[f"TE_{col}_c{k}"].fillna(prior[k])

    return out

# =================== 메인 ===================
def main():
    t0 = time.time()
    print("--- [7.2] Feature Eval (CatBoost, hard-coded FE) 시작 ---")

    train = pd.read_csv(TRAIN_PATH)
    X = train.drop([ID_COL, TARGET], axis=1).copy()
    y = train[TARGET].astype(int).copy()

    # ---------- Baseline ----------
    X_base = base_add_features(X)
    X_base["contract_length"] = X_base["contract_length"].astype(str)
    X_base["age_group"] = X_base["age_group"].astype(str)
    cat_features_base = prepare_cat_features(X_base)

    # ---------- V2 (고정 설정) ----------
    X_v2 = base_add_features(X)
    X_v2 = v2_prepare_types(X_v2)
    X_v2 = v2_winsorize_block(X_v2)                # winsor=(0.02,0.98)
    X_v2 = v2_build_interactions_and_bins(X_v2)    # qbins=20
    te_cols = [c for c in TE_COLS if c in X_v2.columns]

    # OOF-TE (smoothing=10.0)
    X_v2_te = cv_target_encode_multiclass(
        train_df=X_v2, y=y, te_cols=te_cols,
        n_splits=N_SPLITS, seed=SEED, smoothing=TE_SMOOTH
    )
    cat_features_v2 = prepare_cat_features(X_v2_te)

    # ---------- 동일 5-Fold CV ----------
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_pred_base = np.full(len(y), -1, dtype=int)
    oof_pred_v2   = np.full(len(y), -1, dtype=int)
    fold_rows, f1s_base, f1s_v2 = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        # Baseline
        m_b = CatBoostClassifier(**TUNED_CB, cat_features=cat_features_base)
        m_b.fit(
            X_base.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X_base.iloc[va_idx], y.iloc[va_idx])],
            early_stopping_rounds=100, verbose=0
        )
        pred_b = m_b.predict(X_base.iloc[va_idx], prediction_type="Class")
        pred_b = np.asarray(pred_b).reshape(-1).astype(int)
        oof_pred_base[va_idx] = pred_b
        f1_b = f1_score(y.iloc[va_idx], pred_b, average="macro")
        f1s_base.append(f1_b)

        # V2
        m_v = CatBoostClassifier(**TUNED_CB, cat_features=cat_features_v2)
        m_v.fit(
            X_v2_te.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X_v2_te.iloc[va_idx], y.iloc[va_idx])],
            early_stopping_rounds=100, verbose=0
        )
        pred_v = m_v.predict(X_v2_te.iloc[va_idx], prediction_type="Class")
        pred_v = np.asarray(pred_v).reshape(-1).astype(int)
        oof_pred_v2[va_idx] = pred_v
        f1_v = f1_score(y.iloc[va_idx], pred_v, average="macro")
        f1s_v2.append(f1_v)

        fold_rows.append({
            "fold": fold,
            "baseline_macro_f1": f1_b,
            "v2_macro_f1": f1_v,
            "gain": f1_v - f1_b
        })
        print(f"[Fold {fold}] Baseline={f1_b:.5f} | V2={f1_v:.5f} | Gain={f1_v - f1_b:+.5f}")

    # ---------- 집계/저장 ----------
    mean_b, std_b = float(np.mean(f1s_base)), float(np.std(f1s_base, ddof=1))
    mean_v, std_v = float(np.mean(f1s_v2)),   float(np.std(f1s_v2, ddof=1))
    gain = mean_v - mean_b

    np.save("[7.2]_oof_preds_baseline_mod.npy", oof_pred_base)
    np.save("[7.2]_oof_preds_v2_mod.npy",       oof_pred_v2)
    pd.DataFrame(fold_rows).to_csv("[7.2]_fold_scores_mod.csv", index=False)

    lines = []
    lines.append("# [7.2] Feature Eval Report (CatBoost, hard-coded FE)\n")
    lines.append("## CV 결과 (5-Fold, 동일 분할)")
    lines.append(f"- Baseline  : mean={mean_b:.5f}, std={std_b:.5f}")
    lines.append(f"- V2 (hard) : mean={mean_v:.5f}, std={std_v:.5f}")
    lines.append(f"- **Gain**  : **{gain:+.5f}**\n")
    lines.append("## V2 고정 설정")
    lines.append(f"- te_cols = {TE_COLS}")
    lines.append(f"- qbins    = {QBINS}")
    lines.append(f"- winsor   = ({WINSOR_LO}, {WINSOR_HI})")
    lines.append(f"- smoothing= {TE_SMOOTH}")
    lines.append("\n## CatBoost 파라미터")
    lines.append("```json\n" + json.dumps(TUNED_CB, indent=2) + "\n```")

    with open("[7.2]_feature_eval_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n===== 결과 요약 =====")
    print(f"Baseline 5-Fold Macro F1: {mean_b:.5f} ± {std_b:.5f}")
    print(f"V2(hard) 5-Fold Macro F1: {mean_v:.5f} ± {std_v:.5f}")
    print(f"Gain                    : {gain:+.5f}")
    print("Saved: [7.2]_feature_eval_report.md, [7.2]_oof_preds_*.npy, [7.2]_fold_scores.csv")
    print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()

# (venv) jh@jh0223:~/DLDACON$ python3 \[7.2\]_feature_eval_catboost.py
# --- [7.2] Feature Eval (CatBoost, hard-coded FE) 시작 ---
# [Fold 1] Baseline=0.49769 | V2=0.49783 | Gain=+0.00014
# [Fold 2] Baseline=0.49408 | V2=0.49456 | Gain=+0.00049
# [Fold 3] Baseline=0.49103 | V2=0.49188 | Gain=+0.00084
# [Fold 4] Baseline=0.49531 | V2=0.49647 | Gain=+0.00116
# [Fold 5] Baseline=0.49491 | V2=0.49495 | Gain=+0.00004
#
# ===== 결과 요약 =====
# Baseline 5-Fold Macro F1: 0.49460 ± 0.00241
# V2(hard) 5-Fold Macro F1: 0.49514 ± 0.00224
# Gain                    : +0.00053
# Saved: [7.2]_feature_eval_report.md, [7.2]_oof_preds_*.npy, [7.2]_fold_scores.csv
# 완료! 총 소요시간: 177.3초
