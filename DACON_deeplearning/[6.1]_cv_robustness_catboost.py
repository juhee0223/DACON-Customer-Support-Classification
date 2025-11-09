# # -*- coding: utf-8 -*-
# """
# [6.1] CV Robustness for CatBoost
# - Seed sweep + RepeatedStratifiedKFold 로 Macro F1의 평균/표준편차 산출
# - 결과물:
#   1) [6.1]_cv_robustness_catboost.csv  (seed, repeat, fold, macro_f1)
#   2) [6.1]_cv_robustness_report.md     (요약 리포트)
# """
#
# (venv) jh@jh0223:~/DLDACON$ python3 \[6.1\]_cv_robustness_catboost.py
# --- [6.1] CV Robustness (CatBoost) 시작 ---
#
# [Seed 42] RepeatedStratifiedKFold 5x2 실행...
#   - rep 0 fold 1: macro_f1=0.49824
#   - rep 0 fold 2: macro_f1=0.49451
#   - rep 0 fold 3: macro_f1=0.48820
#   - rep 0 fold 4: macro_f1=0.49473
#   - rep 0 fold 5: macro_f1=0.49449
#   - rep 1 fold 1: macro_f1=0.48867
#   - rep 1 fold 2: macro_f1=0.49175
#   - rep 1 fold 3: macro_f1=0.49561
#   - rep 1 fold 4: macro_f1=0.49896
#   - rep 1 fold 5: macro_f1=0.49999
#
# [Seed 2025] RepeatedStratifiedKFold 5x2 실행...
#   - rep 0 fold 1: macro_f1=0.49235
#   - rep 0 fold 2: macro_f1=0.48804
#   - rep 0 fold 3: macro_f1=0.48912
#   - rep 0 fold 4: macro_f1=0.50361
#   - rep 0 fold 5: macro_f1=0.49819
#   - rep 1 fold 1: macro_f1=0.49540
#   - rep 1 fold 2: macro_f1=0.49679
#   - rep 1 fold 3: macro_f1=0.49396
#   - rep 1 fold 4: macro_f1=0.49496
#   - rep 1 fold 5: macro_f1=0.49634
#
# [Seed 777] RepeatedStratifiedKFold 5x2 실행...
#   - rep 0 fold 1: macro_f1=0.49676
#   - rep 0 fold 2: macro_f1=0.49099
#   - rep 0 fold 3: macro_f1=0.50094
#   - rep 0 fold 4: macro_f1=0.49247
#   - rep 0 fold 5: macro_f1=0.49249
#   - rep 1 fold 1: macro_f1=0.50117
#   - rep 1 fold 2: macro_f1=0.49349
#   - rep 1 fold 3: macro_f1=0.49635
#   - rep 1 fold 4: macro_f1=0.48753
#   - rep 1 fold 5: macro_f1=0.49467
#
# CV 상세결과 저장: [6.1]_cv_robustness_catboost.csv
# 요약 리포트 저장: [6.1]_cv_robustness_report.md
#
# 완료! 총 소요시간: 434.7초

import os, time, json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

# ========= 공통 설정 =========
SEEDS = [42, 2025, 777]        # 필요시 늘려서 재실험 가능
N_SPLITS = 5
N_REPEATS = 2                   # 5x2 = 10 folds per seed
DATA_PATH = "./"
TARGET = "support_needs"
ID_COL = "ID"

# ========= Feature Engineering (기존 규칙 유지) =========
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6
    df["tenure_per_contract"] = df["tenure"] / (df["contract_length"] + eps)
    df["delay_ratio"]         = df["payment_interval"] / (df["contract_length"] + eps)
    df["idle_ratio"]          = df["after_interaction"] / (df["tenure"] + eps)
    df["late_flag"]           = (df["payment_interval"] > 0).astype(int)
    if 'frequent' in df.columns:
        df['total_usage_score'] = df['tenure'] * df['frequent']
    # age bin (기존과 동일)
    bins   = [0, 20, 30, 40, 50, 60, 100]
    labels = ['10s', '20s', '30s', '40s', '50s', '60+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    # 간단 리스크 지표
    df['risk_score'] = df['payment_interval'] + df['after_interaction']
    return df

def load_and_prepare():
    train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    test  = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))  # test는 피처 변환 스키마 확인용
    X = train.drop([ID_COL, TARGET], axis=1).copy()
    y = train[TARGET].copy()
    X_test = test.drop([ID_COL], axis=1).copy()

    X = add_features(X)
    X_test = add_features(X_test)

    # CatBoost용 카테고리 캐스팅
    X['contract_length'] = X['contract_length'].astype(str)
    X_test['contract_length'] = X_test['contract_length'].astype(str)
    X['age_group'] = X['age_group'].astype(str)
    X_test['age_group'] = X_test['age_group'].astype(str)

    cat_features = ["gender", "subscription_type", "age_group", "contract_length"]
    return X, y, X_test, cat_features

def main():
    start = time.time()
    print("--- [6.1] CV Robustness (CatBoost) 시작 ---")
    X, y, X_test, cat_features = load_and_prepare()

    # [3.2]에서 찾은 최적 파라미터 기반 (동일 세팅 + 보수적 조기종료)
    tuned_params = {
        'iterations': 1807,
        'depth': 7,
        'learning_rate': 0.011007442573869236,
        'l2_leaf_reg': 9.960037311199006,
        'colsample_bylevel': 0.8239914444396733,
        'random_strength': 0.15658132267797456,
        'bagging_temperature': 0.2528316917680684
    }

    rows = []
    for seed in SEEDS:
        print(f"\n[Seed {seed}] RepeatedStratifiedKFold {N_SPLITS}x{N_REPEATS} 실행...")
        rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
        rep_idx = -1
        prev_test_idx = None
        for i, (tr_idx, va_idx) in enumerate(rskf.split(X, y), start=1):
            # 반복(rep) 번호 추적용(대략적인 표시)
            if i % N_SPLITS == 1:
                rep_idx += 1

            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

            model = CatBoostClassifier(
                **tuned_params,
                random_state=seed,
                verbose=0,
                auto_class_weights='Balanced',
                cat_features=cat_features
            )
            # 검증 기반 조기 종료(과적합 방지 & 시간 단축)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=100, verbose=0)

            pred = model.predict(X_va)
            f1 = f1_score(y_va, pred, average='macro')
            rows.append({
                "seed": seed,
                "repeat": rep_idx,
                "fold": (i - rep_idx * N_SPLITS),  # 1~N_SPLITS
                "macro_f1": f1
            })
            print(f"  - rep {rep_idx} fold {i - rep_idx * N_SPLITS}: macro_f1={f1:.5f}")

    df = pd.DataFrame(rows)
    out_csv = "[6.1]_cv_robustness_catboost.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nCV 상세결과 저장: {out_csv}")

    # 요약 리포트
    overall_mean = df["macro_f1"].mean()
    overall_std  = df["macro_f1"].std(ddof=1)
    by_seed = df.groupby("seed")["macro_f1"].agg(["mean", "std", "count"]).reset_index()

    # 95% CI (정규근사)
    n = len(df)
    ci95 = 1.96 * (overall_std / np.sqrt(n)) if n > 1 else np.nan

    report_lines = []
    report_lines.append("# [6.1] CV Robustness Report (CatBoost)\n")
    report_lines.append(f"- 총 결과 수: {n}")
    report_lines.append(f"- 전체 평균 Macro F1: **{overall_mean:.5f}**")
    report_lines.append(f"- 전체 표준편차: **{overall_std:.5f}**")
    report_lines.append(f"- 95% 신뢰구간(정규근사): **±{ci95:.5f}**\n")
    report_lines.append("## Seed별 통계\n")
    report_lines.append(by_seed.to_markdown(index=False))
    report_lines.append("\n## 사용 파라미터(CatBoost)\n")
    report_lines.append("```json\n" + json.dumps(tuned_params, indent=2) + "\n```")
    report_lines.append("\n## 실험 설정\n")
    report_lines.append(f"- SEEDS={SEEDS}, N_SPLITS={N_SPLITS}, N_REPEATS={N_REPEATS}")
    report_lines.append(f"- early_stopping_rounds=100, auto_class_weights='Balanced'")

    out_md = "[6.1]_cv_robustness_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"요약 리포트 저장: {out_md}")

    print(f"\n완료! 총 소요시간: {time.time()-start:.1f}초")

if __name__ == "__main__":
    main()
