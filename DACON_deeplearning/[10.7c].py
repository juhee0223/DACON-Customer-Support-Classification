# -*- coding: utf-8 -*-
"""
[10.7c]_oversample_1.5x.py
- 소수 클래스만 "원본의 +50%" 만큼만 추가(= 1.5배 오버샘플)
- Optuna best CatBoost 파라미터 사용
- OOF/TEST 확률 저장, 기본 제출 + bias 제출 모두 생성
"""

import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils import resample
from catboost import CatBoostClassifier

TARGET, ID_COL = "support_needs", "ID"
N_CLASSES = 3

# Optuna Best Params (네가 준 것)
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

# ----- bias utils -----
def _preds_with_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    eps = 1e-15
    z = np.log(np.clip(probs, eps, 1.0)) + bias.reshape(1, -1)
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
                cand = np.clip(base + delta, rng[0], rng[1])
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
    z = np.log(np.clip(probs, eps, 1.0)) + bias.reshape(1, -1)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# ----- oversample(1.5x) -----
def oversample_one_point_five(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    """각 클래스별로 현재 샘플 수의 0.5배만큼 '추가' 복제 → 총 1.5배가 되도록."""
    rng = np.random.RandomState(seed)
    frames_X = []
    frames_y = []
    vc = y.value_counts().to_dict()
    for cls in sorted(vc.keys()):
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        n = len(y_cls)
        n_extra = int(round(n * 0.5))  # 추가 샘플 수 (= 0.5x)
        if n_extra > 0:
            X_extra, y_extra = resample(
                X_cls, y_cls, replace=True, n_samples=n_extra, random_state=seed + cls
            )
            X_aug = pd.concat([X_cls, X_extra], axis=0)
            y_aug = pd.concat([y_cls, y_extra], axis=0)
        else:
            X_aug, y_aug = X_cls, y_cls
        frames_X.append(X_aug)
        frames_y.append(y_aug)
    X_ov = pd.concat(frames_X, axis=0).reset_index(drop=True)
    y_ov = pd.concat(frames_y, axis=0).reset_index(drop=True).astype(int).values
    return X_ov, y_ov

def main():
    t0 = time.time()
    print("--- [10.7c] CatBoost 1.5x Oversample 시작 ---")

    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")
    sub   = pd.read_csv("sample_submission.csv")

    X = train.drop([ID_COL, TARGET], axis=1).copy()
    y = train[TARGET].astype(int).copy()
    Xtest = test.drop([ID_COL], axis=1).copy()
    test_ids = test[ID_COL].values

    # CatBoost 범주형 처리
    cat_cols = []
    for c in ["gender", "subscription_type", "contract_length"]:
        if c in X.columns:
            X[c] = X[c].astype(str)
            Xtest[c] = Xtest[c].astype(str)
            cat_cols.append(c)

    # 1.5x 오버샘플링
    X_bal, y_bal = oversample_one_point_five(X, y, seed=SEED)
    print("Oversampled(1.5x) dataset:", X_bal.shape, y_bal.shape)

    # CV
    oof = np.zeros((len(y_bal), N_CLASSES))
    tet = np.zeros((len(Xtest), N_CLASSES))
    cv_scores = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (tr, va) in enumerate(skf.split(X_bal, y_bal), start=1):
        model = CatBoostClassifier(**BEST_PARAMS, cat_features=cat_cols, random_seed=SEED + fold)
        model.fit(
            X_bal.iloc[tr], y_bal[tr],
            eval_set=[(X_bal.iloc[va], y_bal[va])],
            early_stopping_rounds=EARLY_STOP,
            verbose=0
        )
        pv = model.predict_proba(X_bal.iloc[va])
        oof[va] = pv
        tet += model.predict_proba(Xtest) / N_SPLITS
        f1 = f1_score(y_bal[va], pv.argmax(1), average="macro")
        cv_scores.append(f1)
        print(f"Fold {fold}: Macro F1 = {f1:.5f}")

    # 요약
    mean_cv = float(np.mean(cv_scores)); std_cv = float(np.std(cv_scores, ddof=1))
    oof_f1 = f1_score(y_bal, oof.argmax(1), average="macro")
    print(f"\nOOF Macro F1 (argmax): {oof_f1:.5f}")
    print(f"CV 평균: {mean_cv:.5f} ± {std_cv:.5f}")

    # 저장
    np.save("[10.7c]_oof_probs.npy", oof)
    np.save("[10.7c]_test_probs.npy", tet)
    pd.DataFrame({"cv_macro_f1": cv_scores}).to_csv("[10.7c]_cv_scores.csv", index=False)

    # 제출: bias 없이(권장)
    base_pred = tet.argmax(1).astype(int)
    pd.DataFrame({ID_COL: test_ids, TARGET: base_pred}).to_csv("[10.7c]_submission_oversample1p5.csv", index=False)

    # 참고용: bias 버전도 생성 (LB에선 비추지만 옵션 제공)
    best_bias, f1_bias = tune_global_bias(oof, y_bal)
    print(f"OOF + bias Macro F1: {f1_bias:.5f} bias={np.round(best_bias,4)}")
    test_probs_bias = apply_bias(tet, best_bias)
    bias_pred = test_probs_bias.argmax(1).astype(int)
    pd.DataFrame({ID_COL: test_ids, TARGET: bias_pred}).to_csv("[10.7c]_submission_oversample1p5_bias.csv", index=False)

    print("\nSaved:")
    print(" - [10.7c]_submission_oversample1p5.csv")
    print(" - [10.7c]_submission_oversample1p5_bias.csv")
    print(" - [10.7c]_cv_scores.csv, [10.7c]_oof_probs.npy, [10.7c]_test_probs.npy")
    print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()
