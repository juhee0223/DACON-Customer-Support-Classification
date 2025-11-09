# -*- coding: utf-8 -*-
"""
[9.6]_tune_catboost_optuna_plus.py
- "하나의 스크립트" 안에서 점수 올리기에 집중:
  1) V2 FE (winsor, 교차항/분위 bin, gender_subscription)
  2) Multiclass OOF Target Encoding (누수 방지)
  3) 5-Fold CV 기반 Optuna (OOF Macro F1 최대화)
  4) CatBoost 탐색폭 확장 (MultiClass vs OneVsAll, bootstrap 등)
  5) 최종: (A) 전체데이터 재학습 제출 + (B) 5-Fold 앙상블 제출 + (C) bias 적용 제출(옵션)

산출물:
  * [9.6]_optuna_best_params.json
  * [9.6]_cv_log.csv                          (trial별 기록)
  * [9.6]_submission_fullfit.csv              (전체 데이터 한 모델)
  * [9.6]_submission_cv_ens.csv               (5-Fold 앙상블)
  * [9.6]_submission_cv_ens_bias.csv          (옵션: [8.2]_best_bias.npy 있을 때)
"""

import os, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna

from typing import List, Dict, Tuple
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

# ===== 경로/설정 =====
SEED = 42
DATA_PATH = "./"
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV  = os.path.join(DATA_PATH, "test.csv")
SUB_CSV   = os.path.join(DATA_PATH, "sample_submission.csv")
BIAS_PATH = os.path.join(DATA_PATH, "[8.2]_best_bias.npy")  # 있으면 자동 적용

TARGET = "support_needs"
ID_COL = "ID"
N_CLASSES = 3

# V2 FE 하이퍼
QBINS     = 20
WINSOR_LO = 0.02
WINSOR_HI = 0.98
TE_COLS   = ["subscription_type", "gender_subscription"]
TE_SMOOTH = 10.0

N_SPLITS  = 5
CV_SEED   = 42

# ===== V2 Feature Engineering =====
def base_add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6
    df["tenure_per_contract"] = df["tenure"] / (df["contract_length"] + eps)
    df["delay_ratio"]         = df["payment_interval"] / (df["contract_length"] + eps)
    df["idle_ratio"]          = df["after_interaction"] / (df["tenure"] + eps)
    df["late_flag"]           = (df["payment_interval"] > 0).astype(int)
    if "frequent" in df.columns:
        df["total_usage_score"] = df["tenure"] * df["frequent"]
    # age bin
    bins = [0,20,30,40,50,60,100]
    labels = ['10s','20s','30s','40s','50s','60+']
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    # cross features
    df["risk_score"] = df["payment_interval"] + df["after_interaction"]
    return df

def prepare_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["contract_length"] = out["contract_length"].astype(str)
    out["age_group"] = out["age_group"].astype(str)
    out["gender_subscription"] = out["gender"].astype(str) + "_" + out["subscription_type"].astype(str)
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
    for c in ["tenure","payment_interval","after_interaction","tenure_per_contract"]:
        if c in out.columns:
            out[f"{c}_q{qbins}"] = _quantile_rank_bins(out[c], qbins)
    return out

def prepare_cat_features(df: pd.DataFrame) -> List[str]:
    return [c for c in ["gender","subscription_type","age_group","contract_length","gender_subscription"]
            if c in df.columns and df[c].dtype == object]

# ===== Multiclass OOF Target Encoding (누수 방지) =====
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

    # full map (test 적용용)
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

# ===== Bias 적용 (옵션) =====
def apply_bias_to_probs(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    eps = 1e-15
    logp = np.log(np.clip(probs, eps, 1.0))
    z = logp + bias.reshape(1, -1)
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=1, keepdims=True)

# ===== Optuna + CV 파이프라인 =====
def build_cv_frames(train: pd.DataFrame, test: pd.DataFrame):
    # FE 공통
    X_raw = train.drop([ID_COL, TARGET], axis=1).copy()
    y = train[TARGET].astype(int).copy()
    Xte_raw = test.drop([ID_COL], axis=1).copy()
    test_ids = test[ID_COL].values

    X = base_add_features(X_raw)
    X = prepare_types(X)
    X = winsor_block(X, WINSOR_LO, WINSOR_HI)
    X = build_interactions_and_bins(X, QBINS)

    Xtest = base_add_features(Xte_raw)
    Xtest = prepare_types(Xtest)
    Xtest = winsor_block(Xtest, WINSOR_LO, WINSOR_HI)
    Xtest = build_interactions_and_bins(Xtest, QBINS)

    te_cols = [c for c in TE_COLS if c in X.columns]
    X_te, full_map, prior = cv_target_encode_multiclass(X, y, te_cols, n_splits=N_SPLITS, seed=CV_SEED, smoothing=TE_SMOOTH)
    Xtest = apply_te_from_map(Xtest, te_cols, full_map, prior)
    cat_features = prepare_cat_features(X_te)
    return X_te, y, Xtest, test_ids, cat_features

def objective_builder(X_te, y, cat_features):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)

    def objective(trial: optuna.trial.Trial) -> float:
        # 탐색 공간
        params = {
            "iterations": trial.suggest_int("iterations", 900, 2200),
            "depth": trial.suggest_int("depth", 6, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 12.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.05, 2.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 0.6),
            "auto_class_weights": "Balanced",
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
            "loss_function": trial.suggest_categorical("loss_function", ["MultiClass", "MultiClassOneVsAll"]),
            "verbose": 0,
            "random_seed": SEED,
        }
        if params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)

        fold_scores = []
        for tr, va in skf.split(X_te, y):
            model = CatBoostClassifier(**params, cat_features=cat_features)
            model.fit(
                X_te.iloc[tr], y.iloc[tr],
                eval_set=[(X_te.iloc[va], y.iloc[va])],
                early_stopping_rounds=150,
                verbose=0
            )
            pv = model.predict_proba(X_te.iloc[va])
            pred = pv.argmax(axis=1)
            f1 = f1_score(y.iloc[va], pred, average="macro")
            fold_scores.append(f1)

        score = float(np.mean(fold_scores))
        trial.set_user_attr("cv_mean", score)
        return score

    return objective

def train_cv_ensemble(X_te, y, Xtest, cat_features, best_params) -> Tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    oof_probs = np.zeros((len(y), N_CLASSES), dtype=float)
    test_probs = np.zeros((len(Xtest), N_CLASSES), dtype=float)

    for fold, (tr, va) in enumerate(skf.split(X_te, y), start=1):
        model = CatBoostClassifier(**best_params, cat_features=cat_features, verbose=0)
        model.fit(
            X_te.iloc[tr], y.iloc[tr],
            eval_set=[(X_te.iloc[va], y.iloc[va])],
            early_stopping_rounds=150,
            verbose=0
        )
        pv = model.predict_proba(X_te.iloc[va])
        oof_probs[va] = pv
        test_probs += model.predict_proba(Xtest) / N_SPLITS
        f1 = f1_score(y.iloc[va], pv.argmax(axis=1), average="macro")
        print(f"[Fold {fold}] Macro F1={f1:.5f}")

    overall_f1 = f1_score(y, oof_probs.argmax(axis=1), average="macro")
    print(f"\nOOF Macro F1 (cv-ens argmax): {overall_f1:.5f}")
    return oof_probs, test_probs

def main():
    t0 = time.time()
    print("--- [9.6] Optuna + V2 FE + TE + CV Ensemble 시작 ---")

    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    sub   = pd.read_csv(SUB_CSV)

    # ===== FE + TE (공통) =====
    X_te, y, Xtest, test_ids, cat_features = build_cv_frames(train, test)

    # ===== Optuna (CV 기반) =====
    study = optuna.create_study(direction="maximize")
    objective = objective_builder(X_te, y, cat_features)
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_score  = study.best_value
    print("\n===== Optuna 완료 =====")
    print(f"Best OOF Macro F1: {best_score:.5f}")
    print("Best Params:", best_params)

    # 로그 저장
    with open("[9.6]_optuna_best_params.json", "w", encoding="utf-8") as f:
        json.dump({"best_score": best_score, "best_params": best_params}, f, ensure_ascii=False, indent=2)
    # trial 로그 간단 저장
    rows = []
    for t in study.trials:
        rows.append({"number": t.number, "value": t.value, **t.params})
    pd.DataFrame(rows).to_csv("[9.6]_cv_log.csv", index=False)

    # ===== (A) 전체 데이터 한 모델로 재학습 (빠른 제출) =====
    full_model = CatBoostClassifier(**best_params, cat_features=cat_features, verbose=100)
    full_model.fit(X_te, y)
    test_probs_full = full_model.predict_proba(Xtest)
    sub_full = pd.DataFrame({ID_COL: test_ids, TARGET: test_probs_full.argmax(axis=1).astype(int)})
    sub_full.to_csv("[9.6]_submission_fullfit.csv", index=False)
    print("Saved: [9.6]_submission_fullfit.csv")

    # ===== (B) 5-Fold CV 앙상블 제출 =====
    print("\n--- CV Ensemble 학습 ---")
    oof_probs, test_probs = train_cv_ensemble(X_te, y, Xtest, cat_features, best_params)
    sub_cv = pd.DataFrame({ID_COL: test_ids, TARGET: test_probs.argmax(axis=1).astype(int)})
    sub_cv.to_csv("[9.6]_submission_cv_ens.csv", index=False)
    print("Saved: [9.6]_submission_cv_ens.csv")

    # ===== (C) (옵션) 8.2 bias 자동 적용 제출 =====
    if os.path.exists(BIAS_PATH):
        try:
            bias = np.load(BIAS_PATH)
            test_probs_bias = apply_bias_to_probs(test_probs, bias)
            sub_bias = pd.DataFrame({ID_COL: test_ids, TARGET: test_probs_bias.argmax(axis=1).astype(int)})
            sub_bias.to_csv("[9.6]_submission_cv_ens_bias.csv", index=False)
            print(f"Saved: [9.6]_submission_cv_ens_bias.csv (bias={np.round(bias,4)})")
        except Exception as e:
            print("Bias 적용 실패(무시):", e)

    print(f"\n완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()
