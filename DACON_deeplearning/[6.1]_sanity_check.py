# -*- coding: utf-8 -*-
"""
[6.1] Sanity Check for Dataset
- 누락/중복/상관과다/분포 드리프트(Train vs Test) 점검
- 결과물:
  1) [6.1]_sanity_summary.md
  2) [6.1]_missing_values.csv
  3) [6.1]_high_corr_pairs.csv
  4) [6.1]_train_test_ks_numeric.csv
  5) [6.1]_train_test_chi2_categorical.csv
"""
# (venv) jh@jh0223:~/DLDACON$ python3 \[6.1\]_sanity_check.py
# --- [6.1] Sanity Check 시작 ---
# 리포트 저장 완료: [6.1]_sanity_summary.md
# 완료! 총 소요시간: 0.2초

import os, time
import numpy as np
import pandas as pd

DATA_PATH = "./"
TARGET = "support_needs"
ID_COL = "ID"
HIGH_CORR_TH = 0.98  # 상관과다 임계치

# (선택) scipy가 있으면 KS/chi2 사용, 없으면 간단 지표로 대체
try:
    from scipy.stats import ks_2samp, chi2_contingency
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def add_features(df: pd.DataFrame) -> pd.DataFrame:
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

def main():
    start = time.time()
    print("--- [6.1] Sanity Check 시작 ---")
    train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    test  = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

    # 분리
    X = train.drop([ID_COL, TARGET], axis=1).copy()
    y = train[TARGET].copy()
    X_test = test.drop([ID_COL], axis=1).copy()

    # 동일 Feature Engineering
    X = add_features(X)
    X_test = add_features(X_test)

    # 기본 정보
    n_train, p = X.shape
    n_test = X_test.shape[0]
    label_dist = y.value_counts().sort_index()
    label_ratio = (label_dist / len(y)).round(5)

    # 결측치
    miss_train = X.isnull().sum().to_frame("train_missing")
    miss_test  = X_test.isnull().sum().to_frame("test_missing")
    missing_df = miss_train.join(miss_test, how="outer").sort_values(by=["train_missing","test_missing"], ascending=False)
    missing_df.to_csv("[6.1]_missing_values.csv")

    # 중복
    dup_rows = X.duplicated().sum()
    dup_ids  = train[ID_COL].duplicated().sum()

    # 상수 컬럼
    nunique = X.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()

    # 고상관 쌍(수치형)
    num_cols = X.select_dtypes(include=[np.number]).columns
    high_corr_pairs = []
    if len(num_cols) >= 2:
        corr = X[num_cols].corr().abs()
        for i, ci in enumerate(num_cols):
            for j, cj in enumerate(num_cols):
                if j <= i:
                    continue
                v = corr.loc[ci, cj]
                if v >= HIGH_CORR_TH:
                    high_corr_pairs.append((ci, cj, float(v)))
    high_corr_df = pd.DataFrame(high_corr_pairs, columns=["col_a","col_b","abs_corr"])
    high_corr_df.to_csv("[6.1]_high_corr_pairs.csv", index=False)

    # 분포 드리프트: Train vs Test
    # 수치형: KS, 범주형: chi2
    ks_rows = []
    for c in num_cols:
        a = X[c].dropna().values
        b = X_test[c].dropna().values
        if _HAS_SCIPY and len(a) > 0 and len(b) > 0:
            stat, pval = ks_2samp(a, b)
            ks_rows.append({"feature": c, "ks_stat": float(stat), "p_value": float(pval)})
        else:
            # 간단 대체(평균 차 비율)
            ma, mb = np.mean(a), np.mean(b)
            denom = (np.abs(ma) + np.abs(mb) + 1e-9)
            ks_rows.append({"feature": c, "ks_stat": float(np.abs(ma-mb)/denom), "p_value": np.nan})
    pd.DataFrame(ks_rows).sort_values("ks_stat", ascending=False).to_csv("[6.1]_train_test_ks_numeric.csv", index=False)

    # 범주형: 문자열/카테고리/불리언
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    chi_rows = []
    for c in cat_cols:
        tr = X[c].astype(str).fillna("NaN").value_counts()
        te = X_test[c].astype(str).fillna("NaN").value_counts()
        cats = sorted(set(tr.index).union(set(te.index)))
        table = np.array([[tr.get(k,0) for k in cats],
                          [te.get(k,0) for k in cats]])
        if _HAS_SCIPY:
            chi2, p, dof, _ = chi2_contingency(table, correction=False)
            chi_rows.append({"feature": c, "chi2": float(chi2), "p_value": float(p), "dof": int(dof)})
        else:
            # 간단 지표: L1 distance of normalized hist
            trp = table[0]/max(table[0].sum(),1)
            tep = table[1]/max(table[1].sum(),1)
            l1 = float(np.abs(trp - tep).sum())
            chi_rows.append({"feature": c, "chi2": np.nan, "p_value": np.nan, "l1_dist": l1})
    pd.DataFrame(chi_rows).sort_values(by=[k for k in ["chi2","l1_dist"] if k in (chi_rows[0].keys() if chi_rows else [])], ascending=False)\
        .to_csv("[6.1]_train_test_chi2_categorical.csv", index=False)

    # 요약 리포트
    lines = []
    lines.append("# [6.1] Sanity Check Summary\n")
    lines.append("## 데이터 개요")
    lines.append(f"- Train: {n_train} rows, {p} columns")
    lines.append(f"- Test:  {n_test} rows\n")
    lines.append("## 라벨 분포 (train)")
    lines.append("```\n" + str(label_dist.to_string()) + "\n```")
    lines.append("## 라벨 비율")
    lines.append("```\n" + str(label_ratio.to_string()) + "\n```")
    lines.append("\n## 결측치 요약 (상세는 [6.1]_missing_values.csv 확인)")
    top_missing = missing_df.head(10)
    lines.append(top_missing.to_markdown())
    lines.append("\n## 중복/상수/고상관")
    lines.append(f"- 중복 행 수(train): {dup_rows}")
    lines.append(f"- 중복 ID 수(train): {dup_ids}")
    lines.append(f"- 상수 컬럼 수(train): {len(constant_cols)} → {constant_cols}")
    lines.append(f"- 고상관(≥{HIGH_CORR_TH}) 쌍 수: {len(high_corr_df)} (상세 CSV 참고)")
    lines.append("\n## 분포 드리프트(Train vs Test)")
    lines.append("- 수치형: [6.1]_train_test_ks_numeric.csv (KS 통계량 상위 정렬)")
    lines.append("- 범주형: [6.1]_train_test_chi2_categorical.csv (chi2 or L1 상위 정렬)")
    out_md = "[6.1]_sanity_summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"리포트 저장 완료: {out_md}")
    print(f"완료! 총 소요시간: {time.time()-start:.1f}초")

if __name__ == "__main__":
    main()
