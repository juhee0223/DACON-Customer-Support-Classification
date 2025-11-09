# -*- coding: utf-8 -*-
"""
[10.1]_catboost_minimal_baseline.py
- "베이직" 회귀: 최소 전처리 + CatBoost 기본 모델
- 5-Fold Stratified CV, 가벼운 Seed Ensemble(기본 3개)
- 출력:
  * [10.1]_cv_scores.csv
  * [10.1]_oof_probs.npy / [10.1]_test_probs.npy
  * [10.1]_submission_minimal.csv
"""

import os, time
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

# ===== 경로/컬럼 =====
DATA_PATH = "./"
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV  = os.path.join(DATA_PATH, "test.csv")
SUB_CSV   = os.path.join(DATA_PATH, "sample_submission.csv")

TARGET = "support_needs"
ID_COL = "ID"
N_CLASSES = 3

# ===== 실행 설정 =====
N_SPLITS = 5
SEEDS = [42, 102, 202]   # 완전 베이직으로 하려면 [42]로 줄이기
EARLY_STOP = 150

# CatBoost 베이직 파라미터(안정형)
CB_BASE = dict(
    iterations=1500,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=3.0,
    random_strength=0.1,
    bagging_temperature=0.2,
    colsample_bylevel=0.9,
    loss_function="MultiClass",
    auto_class_weights="Balanced",  # Macro F1에 보통 안정적
    verbose=0,
)

# ===== 최소 전처리 =====
def prepare_data(train: pd.DataFrame, test: pd.DataFrame):
    X = train.drop([ID_COL, TARGET], axis=1).copy()
    y = train[TARGET].astype(int).copy()
    Xtest = test.drop([ID_COL], axis=1).copy()
    test_ids = test[ID_COL].values

    # CatBoost용 범주형 지정: 문자열로 캐스팅만 (베이직)
    cat_cols: List[str] = []
    for c in ["gender", "subscription_type", "contract_length"]:
        if c in X.columns:
            X[c] = X[c].astype(str)
            Xtest[c] = Xtest[c].astype(str)
            cat_cols.append(c)

    # 선택: age를 구간화하지 않고 그대로 둠(진짜 베이직). 필요하면 아래 한 줄만 켜기.
    # if "age" in X.columns:
    #     bins = [0, 20, 30, 40, 50, 60, 200]
    #     labels = ['10s','20s','30s','40s','50s','60+']
    #     X["age_group"] = pd.cut(X["age"], bins=bins, labels=labels, right=False).astype(str)
    #     Xtest["age_group"] = pd.cut(Xtest["age"], bins=bins, labels=labels, right=False).astype(str)
    #     cat_cols.append("age_group")

    return X, y, Xtest, test_ids, cat_cols

def main():
    t0 = time.time()
    print("--- [10.1] CatBoost Minimal Baseline 시작 ---")

    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    sub   = pd.read_csv(SUB_CSV)

    X, y, Xtest, test_ids, cat_cols = prepare_data(train, test)
    print(f"데이터: train={X.shape}, test={Xtest.shape}, cat_cols={cat_cols}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Seed 앙상블 누적
    oof_avg = np.zeros((len(y), N_CLASSES), dtype=float)
    test_avg = np.zeros((len(Xtest), N_CLASSES), dtype=float)
    rows = []

    for seed in SEEDS:
        print(f"\n[Seed {seed}]")
        params = dict(CB_BASE, random_seed=seed)
        oof = np.zeros_like(oof_avg)
        tet = np.zeros_like(test_avg)
        fold_scores = []

        for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
            model = CatBoostClassifier(**params, cat_features=cat_cols)
            model.fit(
                X.iloc[tr], y.iloc[tr],
                eval_set=[(X.iloc[va], y.iloc[va])],
                early_stopping_rounds=EARLY_STOP,
                verbose=0
            )
            pv = model.predict_proba(X.iloc[va])
            oof[va] = pv
            tet += model.predict_proba(Xtest) / N_SPLITS

            f1 = f1_score(y.iloc[va], pv.argmax(axis=1), average="macro")
            fold_scores.append(f1)
            print(f"  Fold {fold}: Macro F1 = {f1:.5f}")

        mean_f1 = float(np.mean(fold_scores)); std_f1 = float(np.std(fold_scores, ddof=1))
        print(f"[Seed {seed}] CV Macro F1: {mean_f1:.5f} ± {std_f1:.5f}")

        # 누적 평균
        oof_avg  += oof  / len(SEEDS)
        test_avg += tet  / len(SEEDS)
        rows.append({"seed": seed, "cv_macro_f1": mean_f1, "cv_std": std_f1})

    # 최종 OOF 성능
    oof_pred = oof_avg.argmax(axis=1)
    overall_f1 = f1_score(y, oof_pred, average="macro")
    rows.append({"seed": "avg", "cv_macro_f1": overall_f1, "cv_std": np.nan})
    print(f"\nAvg-OOF argmax Macro F1: {overall_f1:.5f}")

    # 저장
    np.save("[10.1]_oof_probs.npy", oof_avg)
    np.save("[10.1]_test_probs.npy", test_avg)
    pd.DataFrame(rows).to_csv("[10.1]_cv_scores.csv", index=False)

    # 제출
    submit = pd.DataFrame({ID_COL: test_ids, TARGET: test_avg.argmax(axis=1).astype(int)})
    submit.to_csv("[10.1]_submission_minimal.csv", index=False)
    print("Saved: [10.1]_submission_minimal.csv, [10.1]_cv_scores.csv, [10.1]_oof_probs.npy, [10.1]_test_probs.npy")
    print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()
