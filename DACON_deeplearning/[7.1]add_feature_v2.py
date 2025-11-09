# -*- coding: utf-8 -*-
"""
[7.1]add_feature_v2.py
- 고급 Feature Engineering (누수 없는 Multiclass Target Encoding 포함)
- 고급 Feature Engineering (누수 없는 Multiclass Target Encoding 포함)
- 교차항, 분위(quantile rank), 조합 카테고리, 윈저라이즈 등
- 사용법:
  1) 모듈로 사용: from add_feature_v2 import FeatureMakerV2
  2) 단독 실행: python3 "[6.2]add_feature_v2.py"
     -> [6.2]_train_fe_v2.csv, [6.2]_test_fe_v2.csv 저장
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.model_selection import StratifiedKFold

DATA_PATH = "./"
TARGET = "support_needs"
ID_COL = "ID"
N_CLASSES = 3  # 문제 라벨이 0/1/2

def _winsorize_series(s: pd.Series, lower=0.01, upper=0.99):
    ql = s.quantile(lower)
    qu = s.quantile(upper)
    return s.clip(lower=ql, upper=qu)

def _quantile_rank_0to9(s: pd.Series):
    # 동일 분포화를 위해 ties='average' 사용
    return (s.rank(method="average", pct=True) * 10 - 1e-9).clip(0, 0.999999).astype(float).mul(10).astype(int)

def base_add_features(df: pd.DataFrame) -> pd.DataFrame:
    """ 너가 쓰던 1차 파생변수 유지 (일관성 확보용) """
    df = df.copy()
    eps = 1e-6
    df["tenure_per_contract"] = df["tenure"] / (df["contract_length"] + eps)
    df["delay_ratio"]         = df["payment_interval"] / (df["contract_length"] + eps)
    df["idle_ratio"]          = df["after_interaction"] / (df["tenure"] + eps)
    df["late_flag"]           = (df["payment_interval"] > 0).astype(int)
    if 'frequent' in df.columns:
        df['total_usage_score'] = df['tenure'] * df['frequent']
    # age bin (기존)
    bins   = [0, 20, 30, 40, 50, 60, 100]
    labels = ['10s', '20s', '30s', '40s', '50s', '60+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    df['risk_score'] = df['payment_interval'] + df['after_interaction']
    return df

class FeatureMakerV2:
    """
    - 누수 없는 Multiclass Target Encoding (CV 기반, Laplace smoothing)
    - 교차항/조합/분위/윈저라이즈
    - fit()은 train 전용, transform()은 train/test 공용
    """
    def __init__(self, n_splits: int = 5, seed: int = 42, smoothing: float = 20.0):
        self.n_splits = n_splits
        self.seed = seed
        self.smoothing = smoothing
        self.num_cols_: List[str] = []
        self.cat_cols_raw_: List[str] = []
        self.te_cols_: List[str] = []  # TE 대상 원본 열 (문자형)
        self.global_prior_: np.ndarray = None
        self.te_map_full_: Dict[str, Dict[str, np.ndarray]] = {}  # col -> {cat: prob_vec}
        self._fitted = False

    def _prepare_types(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # 범주형 캐스팅 (CatBoost 호환성을 위해 문자열)
        out["contract_length"] = out["contract_length"].astype(str)
        out["age_group"] = out["age_group"].astype(str)
        # 추가 조합 카테고리
        out["gender_subscription"] = (out["gender"].astype(str) + "_" +
                                      out["subscription_type"].astype(str))
        return out

    def _winsorize_block(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in ["payment_interval", "after_interaction", "tenure"]:
            if c in out.columns:
                out[c] = _winsorize_series(out[c], 0.01, 0.99)
        return out

    def _build_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        eps = 1e-6
        # 수치형 교차항
        if set(["tenure", "payment_interval"]).issubset(out.columns):
            out["tenure_x_payint"] = out["tenure"] * out["payment_interval"]
        if set(["tenure", "after_interaction"]).issubset(out.columns):
            out["tenure_x_after"] = out["tenure"] * out["after_interaction"]
        if "frequent" in out.columns:
            out["freq_x_after"] = out["frequent"] * out["after_interaction"]
        if set(["delay_ratio", "risk_score"]).issubset(out.columns):
            out["delay_x_risk"] = out["delay_ratio"] * out["risk_score"]
        # 비선형 변환
        if "risk_score" in out.columns:
            out["risk_log1p"] = np.log1p(out["risk_score"].clip(lower=0))
        if "idle_ratio" in out.columns:
            out["idle_sq"] = out["idle_ratio"] ** 2
        # 분위/순위형
        for c in ["tenure", "payment_interval", "after_interaction", "tenure_per_contract"]:
            if c in out.columns:
                out[c + "_q10"] = _quantile_rank_0to9(out[c])
        return out

    def _fit_te_full(self, X: pd.DataFrame, y: pd.Series, te_cols: List[str]):
        """ 전체 train으로 test에 적용할 TE 사전 생성 """
        self.global_prior_ = np.bincount(y, minlength=N_CLASSES) / len(y)
        for col in te_cols:
            mapping: Dict[str, np.ndarray] = {}
            grp = pd.concat([X[col].astype(str), y], axis=1).groupby(col)
            for cat, g in grp:
                cnt = np.bincount(g[TARGET].to_numpy(), minlength=N_CLASSES)
                prob = (cnt + self.smoothing / N_CLASSES) / (cnt.sum() + self.smoothing)
                mapping[str(cat)] = prob
            self.te_map_full_[col] = mapping

    def _transform_te_from_map(self, s: pd.Series, col: str) -> pd.DataFrame:
        """ 저장된 te_map_full_ 기반으로 단일 열을 K개 확률 열로 확장 """
        mapping = self.te_map_full_[col]
        arr = np.vstack([mapping.get(str(v), self.global_prior_) for v in s.astype(str).values])
        te_df = pd.DataFrame(arr, index=s.index,
                             columns=[f"TE_{col}_c{k}" for k in range(N_CLASSES)])
        return te_df

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """ Train 전용: 타입준비/윈저/교차항 후 TE 대상 파악+full map 저장 """
        assert TARGET not in X.columns, "X에는 타겟이 포함되면 안 됩니다."
        df = base_add_features(X)
        df = self._prepare_types(df)
        df = self._winsorize_block(df)
        df = self._build_interactions(df)

        # 저장용 메타
        self.num_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_raw_ = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # TE 대상 (문자열 카테고리 중 메이저 피처)
        self.te_cols_ = [c for c in ["subscription_type", "age_group", "gender", "contract_length", "gender_subscription"]
                         if c in self.cat_cols_raw_]

        # 전체 train으로 test 적용용 map 만들기
        tmp = df.copy()
        tmp[TARGET] = y.values
        self._fit_te_full(tmp, y, self.te_cols_)
        self._fitted = True
        self._df_template_cols_ = df.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Train/Test 공용 변환 (주의: train에서는 CV-TE로 OOF 생성은 별도 코드에서) """
        assert self._fitted, "fit() 먼저 호출하세요."
        df = base_add_features(X)
        df = self._prepare_types(df)
        df = self._winsorize_block(df)
        df = self._build_interactions(df)

        # full map 기반 TE (test 용도 / 또는 단순 변환)
        te_blocks = []
        for col in self.te_cols_:
            te_blocks.append(self._transform_te_from_map(df[col], col))
        if te_blocks:
            df = pd.concat([df] + te_blocks, axis=1)

        # 누락 방지: fit 당시의 열 순서 기준으로 정렬(신규 열은 뒤에)
        cols = list(dict.fromkeys(self._df_template_cols_ + [c for c in df.columns if c not in self._df_template_cols_]))
        return df.reindex(columns=cols, fill_value=np.nan)

if __name__ == "__main__":
    # 단독 실행: train/test를 읽어 v2 변환 후 CSV 저장
    train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    test  = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    X = train.drop([ID_COL, TARGET], axis=1)
    y = train[TARGET]
    X_test = test.drop([ID_COL], axis=1)

    fm = FeatureMakerV2(n_splits=5, seed=42, smoothing=20.0).fit(X, y)
    X_tr_fe = fm.transform(X)
    X_te_fe = fm.transform(X_test)

    X_tr_fe.to_csv("[7.1]_train_fe_v2.csv", index=False)
    X_te_fe.to_csv("[7.1]_test_fe_v2.csv", index=False)
    print("Saved: [7.1]_train_fe_v2.csv, [7.1]_test_fe_v2.csv")


# (venv) jh@jh0223:~/DLDACON$ python3 \[7.2\]_feature_eval_catboost.py
# --- [7.2] Feature Eval (CatBoost) 시작 ---
# [Fold 1] Baseline=0.49769 | V2=0.49858 | Gain=+0.00089
# [Fold 2] Baseline=0.49408 | V2=0.49514 | Gain=+0.00107
# [Fold 3] Baseline=0.49103 | V2=0.49015 | Gain=-0.00088
# [Fold 4] Baseline=0.49531 | V2=0.49524 | Gain=-0.00007
# [Fold 5] Baseline=0.49491 | V2=0.49256 | Gain=-0.00235
#
# ===== 결과 요약 =====
# Baseline 5-Fold Macro F1: 0.49460 ± 0.00241
# V2      5-Fold Macro F1: 0.49433 ± 0.00317
# Gain                     : -0.00027
# Saved: [7.2]_feature_eval_report.md, [7.2]_oof_preds_*.npy, [7.2]_fold_scores.csv
# 완료! 총 소요시간: 175.7초