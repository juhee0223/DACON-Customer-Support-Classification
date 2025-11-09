# -*- coding: utf-8 -*-
"""
[10.7]_catboost_oversample.py
- 소수 클래스 단순 오버샘플링 후 CatBoost 학습
- 파라미터: Optuna에서 찾은 best params 사용
- OOF/TEST probs 저장 + bias 튜닝
"""

import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from sklearn.utils import resample

TARGET, ID_COL = "support_needs", "ID"
N_CLASSES = 3

# ===== Optuna Best Params =====
BEST_PARAMS = dict(
    iterations=1460,
    depth=8,
    learning_rate=0.01011375157729347,
    l2_leaf_reg=2.9134559843727033,
    colsample_bylevel=0.8914955767108875,
    random_strength=0.10543719325919827,
    bagging_temperature=0.1804730161382898,
    auto_class_weights="Balanced",
    loss_function="MultiClass",
    verbose=0,
)

N_SPLITS = 5
SEED = 42
EARLY_STOP = 100

# ===== bias utils =====
def _preds_with_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    eps = 1e-15
    z = np.log(np.clip(probs, eps, 1.0)) + bias.reshape(1,-1)
    return z.argmax(axis=1)

def tune_global_bias(oof_probs, y, step=0.2, min_step=0.02, rng=(-3,3), max_passes=7):
    bias = np.zeros(oof_probs.shape[1])
    best = f1_score(y, _preds_with_bias(oof_probs, bias), average="macro")
    cur_step = step
    for _ in range(max_passes):
        improved = False
        for k in range(len(bias)):
            base = bias[k]; best_k = base; best_f1 = best
            for delta in (-cur_step, 0, +cur_step):
                cand = np.clip(base+delta, rng[0], rng[1])
                b_try = bias.copy(); b_try[k] = cand
                f1 = f1_score(y, _preds_with_bias(oof_probs, b_try), average="macro")
                if f1 > best_f1 + 1e-9:
                    best_f1, best_k = f1, cand
            if best_k != base:
                bias[k] = best_k; best = best_f1; improved = True
        if not improved:
            cur_step *= 0.5
            if cur_step < min_step:
                break
    return bias, best

def apply_bias(probs: np.ndarray, bias: np.ndarray):
    eps = 1e-15
    z = np.log(np.clip(probs, eps, 1.0)) + bias.reshape(1,-1)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# ===== main =====
def main():
    t0 = time.time()
    print("--- [10.7] CatBoost Oversample 시작 ---")

    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")
    sub   = pd.read_csv("sample_submission.csv")

    X = train.drop([ID_COL, TARGET], axis=1).copy()
    y = train[TARGET].astype(int).copy()
    Xtest = test.drop([ID_COL], axis=1).copy()
    test_ids = test[ID_COL].values

    # cat features
    cat_cols = []
    for c in ["gender","subscription_type","contract_length"]:
        if c in X.columns:
            X[c] = X[c].astype(str)
            Xtest[c] = Xtest[c].astype(str)
            cat_cols.append(c)

    # ===== Oversampling =====
    train_balanced = []
    for cls in np.unique(y):
        X_cls = X[y==cls]
        y_cls = y[y==cls]
        if len(y_cls) < y.value_counts().max():
            # 소수 클래스는 2배로 복제
            X_up, y_up = resample(X_cls, y_cls,
                                  replace=True,
                                  n_samples=len(y_cls)*2,
                                  random_state=SEED)
            train_balanced.append(pd.concat([X_cls, X_up]))
            y_bal = np.concatenate([y_cls, y_up])
            train_balanced.append(pd.DataFrame({"y": y_bal}))
        else:
            train_balanced.append(X_cls)
            train_balanced.append(pd.DataFrame({"y": y_cls}))
    # concat
    X_bal = pd.concat(train_balanced[::2], axis=0).reset_index(drop=True)
    y_bal = pd.concat(train_balanced[1::2], axis=0)["y"].values
    print("Oversampled dataset:", X_bal.shape, y_bal.shape)

    # ===== CV =====
    oof = np.zeros((len(y_bal), N_CLASSES))
    tet = np.zeros((len(Xtest), N_CLASSES))
    cv_scores = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold,(tr,va) in enumerate(skf.split(X_bal,y_bal), start=1):
        model = CatBoostClassifier(**BEST_PARAMS, cat_features=cat_cols, random_seed=SEED+fold)
        model.fit(X_bal.iloc[tr], y_bal[tr],
                  eval_set=[(X_bal.iloc[va], y_bal[va])],
                  early_stopping_rounds=EARLY_STOP, verbose=0)
        pv = model.predict_proba(X_bal.iloc[va])
        oof[va] = pv
        tet += model.predict_proba(Xtest) / N_SPLITS
        f1 = f1_score(y_bal[va], pv.argmax(1), average="macro")
        cv_scores.append(f1)
        print(f"Fold {fold}: Macro F1 = {f1:.5f}")

    # CV 평균
    mean_cv = np.mean(cv_scores); std_cv = np.std(cv_scores, ddof=1)
    print(f"\nOOF Macro F1 (argmax): {f1_score(y_bal, oof.argmax(1), average='macro'):.5f}")
    print(f"CV 평균: {mean_cv:.5f} ± {std_cv:.5f}")

    # 저장
    np.save("[10.7]_oof_probs.npy", oof)
    np.save("[10.7]_test_probs.npy", tet)
    pd.DataFrame({"cv_macro_f1":cv_scores}).to_csv("[10.7]_cv_scores.csv", index=False)

    # 기본 제출
    base_pred = tet.argmax(1)
    pd.DataFrame({ID_COL:test_ids, TARGET:base_pred}).to_csv("[10.7]_submission_oversample.csv", index=False)

    # bias 튜닝 후 제출
    best_bias, f1_bias = tune_global_bias(oof, y_bal)
    print(f"OOF + bias Macro F1: {f1_bias:.5f} bias={best_bias}")
    test_probs_bias = apply_bias(tet, best_bias)
    bias_pred = test_probs_bias.argmax(1)
    pd.DataFrame({ID_COL:test_ids, TARGET:bias_pred}).to_csv("[10.7]_submission_oversample_bias.csv", index=False)

    print("\nSaved: [10.7]_submission_oversample.csv, [10.7]_submission_oversample_bias.csv, [10.7]_cv_scores.csv")
    print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__=="__main__":
    main()
