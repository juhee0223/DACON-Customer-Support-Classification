# -*- coding: utf-8 -*-
"""
[10.2]_catboost_minimal_plus.py
- 베이직 baseline에 최소한만 추가해서 0.5 돌파 확률을 올리기:
  1) age_group (가벼운 카테고리 파생 1개)
  2) auto_class_weights: "Balanced" vs "SqrtBalanced"
  3) loss_function: "MultiClass" vs "MultiClassOneVsAll"
  4) 글로벌 bias 튜닝(OOF Macro F1 최대화) 후 test에 적용

산출물:
  * [10.2]_cv_scores.csv
  * [10.2]_oof_probs.npy / [10.2]_test_probs.npy
  * [10.2]_submission_minplus.csv
  * [10.2]_submission_minplus_bias.csv
"""

import os, time
import numpy as np
import pandas as pd
from typing import List, Tuple
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
SEEDS = [42, 52, 62]      # 베이직 앙상블
EARLY_STOP = 150

# 두 가지 손실 + 두 가지 class weight를 시도 (총 4조합 × SEEDS)
LOSS_LIST = ["MultiClass", "MultiClassOneVsAll"]
ACW_LIST  = ["Balanced", "SqrtBalanced"]

# CatBoost 기본 파라미터(안정형, 베이직)
CB_BASE = dict(
    iterations=1500,        # 약간 늘림
    depth=8,
    learning_rate=0.02,     # 안정적이고 빠른 수렴
    l2_leaf_reg=3.0,
    random_strength=0.1,
    bagging_temperature=0.2,
    colsample_bylevel=0.9,
    verbose=0,
)

# ===== 최소 전처리 (+ age_group 1개만 추가) =====
def prepare_data(train: pd.DataFrame, test: pd.DataFrame):
    X = train.drop([ID_COL, TARGET], axis=1).copy()
    y = train[TARGET].astype(int).copy()
    Xtest = test.drop([ID_COL], axis=1).copy()
    test_ids = test[ID_COL].values

    # 베이직: 문자열 캐스팅
    cat_cols: List[str] = []
    for c in ["gender", "subscription_type", "contract_length"]:
        if c in X.columns:
            X[c] = X[c].astype(str)
            Xtest[c] = Xtest[c].astype(str)
            cat_cols.append(c)

    # 가벼운 파생(딱 1개): age_group
    if "age" in X.columns:
        bins = [0, 20, 30, 40, 50, 60, 200]
        labels = ['10s','20s','30s','40s','50s','60+']
        X["age_group"] = pd.cut(X["age"], bins=bins, labels=labels, right=False).astype(str)
        Xtest["age_group"] = pd.cut(Xtest["age"], bins=bins, labels=labels, right=False).astype(str)
        cat_cols.append("age_group")

    return X, y, Xtest, test_ids, cat_cols

# ===== 글로벌 bias 튜닝 (OOF Macro F1 최대화) =====
def _preds_with_bias_block(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    eps = 1e-15
    z = np.log(np.clip(probs, eps, 1.0)) + bias.reshape(1, -1)
    return z.argmax(axis=1).astype(int)

def tune_global_bias(oof_probs: np.ndarray, y: np.ndarray,
                     init_bias=None, step=0.2, min_step=0.02,
                     bias_range=(-2.0, 2.0), max_passes=7) -> Tuple[np.ndarray, float]:
    if init_bias is None:
        bias = np.zeros(oof_probs.shape[1], dtype=float)
    else:
        bias = np.array(init_bias, dtype=float).copy()
    best = f1_score(y, _preds_with_bias_block(oof_probs, bias), average="macro")
    cur_step = step
    for _ in range(max_passes):
        improved = False
        for k in range(len(bias)):
            base = bias[k]
            best_k = base
            best_k_f1 = best
            for delta in (-cur_step, 0.0, +cur_step):
                cand = float(np.clip(base + delta, bias_range[0], bias_range[1]))
                b_try = bias.copy()
                b_try[k] = cand
                f1 = f1_score(y, _preds_with_bias_block(oof_probs, b_try), average="macro")
                if f1 > best_k_f1 + 1e-9:
                    best_k_f1 = f1
                    best_k = cand
            if best_k != base:
                bias[k] = best_k
                best = best_k_f1
                improved = True
        if not improved:
            cur_step *= 0.5
            if cur_step < min_step:
                break
    return bias, best

def apply_bias_to_probs(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    eps = 1e-15
    z = np.log(np.clip(probs, eps, 1.0)) + bias.reshape(1, -1)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# ===== 메인 =====
def main():
    t0 = time.time()
    print("--- [10.2] CatBoost Minimal+ 시작 ---")

    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    sub   = pd.read_csv(SUB_CSV)

    X, y, Xtest, test_ids, cat_cols = prepare_data(train, test)
    print(f"데이터: train={X.shape}, test={Xtest.shape}, cat_cols={cat_cols}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # 조합: LOSS × ACW × SEED
    results = []
    best_oof = None
    best_test = None
    best_name = None
    best_cv = -1.0

    # 모든 조합 중 Macro F1 최고 모델 하나만 채택 (베이직한 선택)
    for loss in LOSS_LIST:
        for acw in ACW_LIST:
            for seed in SEEDS:
                tag = f"{loss}_{acw}_s{seed}"
                print(f"\n[{tag}]")
                params = dict(CB_BASE, loss_function=loss, auto_class_weights=acw, random_seed=seed)

                oof = np.zeros((len(y), N_CLASSES), dtype=float)
                tet = np.zeros((len(Xtest), N_CLASSES), dtype=float)
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

                mean_f1 = float(np.mean(fold_scores))
                std_f1  = float(np.std(fold_scores, ddof=1))
                results.append({"loss": loss, "acw": acw, "seed": seed,
                                "cv_macro_f1": mean_f1, "cv_std": std_f1})
                print(f"[{tag}] CV Macro F1: {mean_f1:.5f} ± {std_f1:.5f}")

                if mean_f1 > best_cv:
                    best_cv = mean_f1
                    best_oof = oof.copy()
                    best_test = tet.copy()
                    best_name = tag

    # 요약 저장
    pd.DataFrame(results).to_csv("[10.2]_cv_scores.csv", index=False)

    # 베스트 조합 기준 기본 제출
    base_pred = best_test.argmax(axis=1).astype(int)
    pd.DataFrame({ID_COL: test_ids, TARGET: base_pred}).to_csv("[10.2]_submission_minplus.csv", index=False)

    # ===== 글로벌 bias 튜닝 후 제출 =====
    best_bias, oof_f1_bias = tune_global_bias(best_oof, y.values, init_bias=None)
    test_probs_bias = apply_bias_to_probs(best_test, best_bias)
    bias_pred = test_probs_bias.argmax(axis=1).astype(int)
    pd.DataFrame({ID_COL: test_ids, TARGET: bias_pred}).to_csv("[10.2]_submission_minplus_bias.csv", index=False)

    # OOF 보고
    oof_f1_argmax = f1_score(y, best_oof.argmax(axis=1), average="macro")
    print("\n===== Summary =====")
    print(f"Best combo  : {best_name}")
    print(f"OOF argmax  : {oof_f1_argmax:.5f}")
    print(f"OOF + bias  : {oof_f1_bias:.5f}")
    print("Saved: [10.2]_submission_minplus.csv, [10.2]_submission_minplus_bias.csv, [10.2]_cv_scores.csv")
    print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()
