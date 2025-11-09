# -*- coding: utf-8 -*-
"""
[10.7d]_oversample_sweep.py
- 단순 복제 오버샘플(카테고리 문자열 유지) 강도 미세조정
- 소수 클래스별 배수 grid를 CV로 탐색 → 최적 조합으로 full-train 후 제출
- 주의: 오버샘플 시 auto_class_weights는 끈다(중복 보정 방지)
"""

import os, time
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils import resample
from catboost import CatBoostClassifier

TARGET, ID_COL = "support_needs", "ID"
N_CLASSES = 3

# Optuna best (네 기록)
BEST_PARAMS_FULL = dict(
    iterations=1460,
    depth=8,
    learning_rate=0.01011375157729347,
    l2_leaf_reg=2.9134559843727033,
    colsample_bylevel=0.8914955767108875,
    random_strength=0.10543719325919827,
    bagging_temperature=0.1804730161382898,
    loss_function="MultiClass",
    verbose=0,
)
# 빠른 스윕용(가벼운 조합) — early stop이 있으니 iterations=800 정도로 충분
BEST_PARAMS_SWEEP = dict(BEST_PARAMS_FULL, iterations=800)

N_SPLITS = 5
SEED = 42
EARLY_STOP = 100

# ---- 유틸: 오버샘플(복제) ----
def oversample_by_factors(
    X: pd.DataFrame, y: pd.Series, factors: Dict[int, float], seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    factors[class] = 배수 (1.0이면 그대로, 1.5면 50% 추가 복제)
    """
    frames_X, frames_y = [], []
    for cls in sorted(np.unique(y)):
        X_c = X[y == cls]
        y_c = y[y == cls]
        mult = float(factors.get(cls, 1.0))
        if mult <= 1.0:
            X_aug, y_aug = X_c, y_c
        else:
            n_extra = int(round(len(y_c) * (mult - 1.0)))
            X_extra, y_extra = resample(
                X_c, y_c, replace=True, n_samples=n_extra, random_state=seed + cls
            )
            X_aug = pd.concat([X_c, X_extra], axis=0)
            y_aug = pd.concat([y_c, y_extra], axis=0)
        frames_X.append(X_aug)
        frames_y.append(y_aug)
    X_ov = pd.concat(frames_X, axis=0).reset_index(drop=True)
    y_ov = pd.concat(frames_y, axis=0).reset_index(drop=True).astype(int).values
    return X_ov, y_ov

def prepare_data():
    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")
    sub   = pd.read_csv("sample_submission.csv")

    X = train.drop([ID_COL, TARGET], axis=1).copy()
    y = train[TARGET].astype(int).copy()
    Xtest = test.drop([ID_COL], axis=1).copy()
    test_ids = test[ID_COL].values

    # CatBoost 범주형: 문자열 유지(일관성 중요)
    cat_cols: List[str] = []
    for c in ["gender", "subscription_type", "contract_length"]:
        if c in X.columns:
            X[c] = X[c].astype(str)
            Xtest[c] = Xtest[c].astype(str)
            cat_cols.append(c)

    return X, y, Xtest, test_ids, cat_cols

def cv_score_catboost(X, y, Xtest, cat_cols, params, seed=42, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), N_CLASSES))
    tet = np.zeros((len(Xtest), N_CLASSES))
    scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        model = CatBoostClassifier(**params, cat_features=cat_cols, random_seed=seed+fold, auto_class_weights=None)  # 오버샘플 시 꺼둠
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[va], y[va])],
            early_stopping_rounds=EARLY_STOP,
            verbose=0
        )
        pv = model.predict_proba(X.iloc[va])
        oof[va] = pv
        tet += model.predict_proba(Xtest) / n_splits
        scores.append(f1_score(y[va], pv.argmax(1), average="macro"))
    return float(np.mean(scores)), float(np.std(scores, ddof=1)), oof, tet

def main():
    t0 = time.time()
    print("--- [10.7d] Oversample Sweep 시작 ---")
    X, y, Xtest, test_ids, cat_cols = prepare_data()
    cls_counts = y.value_counts().sort_index().to_dict()
    print("원본 클래스 분포:", cls_counts)

    # 소수 클래스 식별(가장 많은 클래스 제외)
    maj_cls = max(cls_counts, key=lambda k: cls_counts[k])
    min_classes = [c for c in cls_counts if c != maj_cls]
    print("다수 클래스:", maj_cls, "| 소수 클래스:", min_classes)

    # 스윕 그리드: 각 소수 클래스에 대해 {1.2, 1.35, 1.5, 1.7} 배
    grid_vals = [1.2, 1.35, 1.5, 1.7]
    combos = []
    if len(min_classes) == 2:
        a, b = min_classes
        for fa in grid_vals:
            for fb in grid_vals:
                combos.append({maj_cls:1.0, a:fa, b:fb})
    else:
        # 혹시 3클래스가 아닌 케이스 대비
        for f in grid_vals:
            combos.append({maj_cls:1.0, min_classes[0]:f})

    # 1) 빠른 스윕 (iterations=800)
    results = []
    best = (-1.0, None, None, None)  # (cv, factors, oof, tet)
    for i, factors in enumerate(combos, 1):
        X_bal, y_bal = oversample_by_factors(X, y, factors, seed=SEED)
        cv_mean, cv_std, oof, tet = cv_score_catboost(X_bal, y_bal, Xtest, cat_cols, BEST_PARAMS_SWEEP, seed=SEED, n_splits=N_SPLITS)
        results.append({"factors": factors, "cv_macro_f1": cv_mean, "cv_std": cv_std, "n_train": len(y_bal)})
        print(f"[{i:02d}/{len(combos)}] factors={factors} | CV={cv_mean:.5f} ± {cv_std:.5f} | n={len(y_bal)}")
        if cv_mean > best[0]:
            best = (cv_mean, factors, oof, tet)

    df_res = pd.DataFrame(results)
    df_res.sort_values("cv_macro_f1", ascending=False, inplace=True)
    df_res.to_csv("[10.7d]_oversample_sweep_cv.csv", index=False)

    best_cv, best_factors, _, _ = best
    print(f"\n>>> 스윕 베스트: CV={best_cv:.5f}, factors={best_factors}")

    # 2) 베스트 factors로 풀 파라미터 재학습 + 제출 산출
    print("\n--- Full 파라미터 재학습 ---")
    X_best, y_best = oversample_by_factors(X, y, best_factors, seed=SEED)
    cv_mean, cv_std, oof, tet = cv_score_catboost(X_best, y_best, Xtest, cat_cols, BEST_PARAMS_FULL, seed=SEED, n_splits=N_SPLITS)
    print(f"[Full] CV={cv_mean:.5f} ± {cv_std:.5f}")

    # 저장 및 제출 (bias 없이 권장)
    np.save("[10.7d]_oof_probs.npy", oof)
    np.save("[10.7d]_test_probs.npy", tet)
    df_res.to_csv("[10.7d]_oversample_sweep_cv.csv", index=False)

    sub = pd.DataFrame({ID_COL: test_ids, TARGET: tet.argmax(1).astype(int)})
    sub.to_csv("[10.7d]_submission_oversample_sweep.csv", index=False)

    # 참고: bias 버전도 생성(옵션)
    # LB에서 역효과 날 수 있으니 제출 컷 아낄 때만 사용
    # from sklearn.metrics import f1_score
    # def _preds(probs, bias):
    #     z = np.log(np.clip(probs,1e-15,1))+bias.reshape(1,-1)
    #     return z.argmax(1)
    # bias = np.zeros(N_CLASSES); best_b = bias.copy(); best_f1 = f1_score(y_best, _preds(oof, bias), average="macro")
    # for _ in range(6):
    #   imp=False
    #   for k in range(N_CLASSES):
    #     for d in (-0.2,0,+0.2):
    #       b=bias.copy(); b[k]=np.clip(b[k]+d,-3,3)
    #       f=f1_score(y_best, _preds(oof,b), average="macro")
    #       if f>best_f1: best_f1=f; best_b=b; imp=True
    #   bias=best_b;
    # tet_b = np.exp(np.log(np.clip(tet,1e-15,1))+bias.reshape(1,-1)); tet_b/=tet_b.sum(1,keepdims=True)
    # pd.DataFrame({ID_COL:test_ids, TARGET: tet_b.argmax(1)}).to_csv("[10.7d]_submission_oversample_sweep_bias.csv", index=False)

    print("\nSaved:")
    print(" - [10.7d]_submission_oversample_sweep.csv")
    print(" - [10.7d]_oversample_sweep_cv.csv")
    print(" - [10.7d]_oof_probs.npy, [10.7d]_test_probs.npy")
    print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()
