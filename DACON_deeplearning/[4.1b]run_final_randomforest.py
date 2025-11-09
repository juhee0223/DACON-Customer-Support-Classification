import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- 이 스크립트는 CV 없이, 전체 데이터로 최종 모델을 학습하고 예측 확률만 저장합니다. ---

# 0) 설정
SEED = 42

# 1) 데이터 로드 및 Feature Engineering (위 4.1a 코드와 동일한 부분을 여기에 포함)
# (간결성을 위해 코드 생략, 4.1a의 1번 항목 코드를 그대로 복사)
data_path = "./"
train_path = os.path.join(data_path, "train.csv")
test_path  = os.path.join(data_path, "test.csv")
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)
TARGET = "support_needs"; ID_COL = "ID"
X = train_df.drop([ID_COL, TARGET], axis=1).copy(); y = train_df[TARGET].copy()
X_test = test_df.drop([ID_COL], axis=1).copy()
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); eps = 1e-6; df["tenure_per_contract"] = df["tenure"] / (df["contract_length"] + eps); df["delay_ratio"] = df["payment_interval"] / (df["contract_length"] + eps); df["idle_ratio"] = df["after_interaction"] / (df["tenure"] + eps); df["late_flag"] = (df["payment_interval"] > 0).astype(int)
    if 'frequent' in df.columns: df['total_usage_score'] = df['tenure'] * df['frequent']
    bins = [0, 20, 30, 40, 50, 60, 100]; labels = ['10s', '20s', '30s', '40s', '50s', '60+']; df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    df['risk_score'] = df['payment_interval'] + df['after_interaction']
    return df
X = add_features(X); X_test = add_features(X_test)

# 2) 전처리 (RandomForest 방식)
cat_cols = ["gender", "subscription_type", "age_group", "contract_length"]
X['contract_length'] = X['contract_length'].astype(str); X_test['contract_length'] = X_test['contract_length'].astype(str)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
preprocess = ColumnTransformer(transformers=[("cat", ohe, cat_cols)], remainder="passthrough")

# 3) 최종 모델 정의 (알려진 최적 파라미터 사용)
best_rf_params = {'n_estimators': 310, 'max_depth': 8, 'min_samples_split': 4, 'min_samples_leaf': 4, 'max_features': None, 'random_state': SEED, 'class_weight': "balanced", 'n_jobs': -1}
clf_rf_tuned = RandomForestClassifier(**best_rf_params)
pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf_rf_tuned)])

# 4) 전체 데이터로 학습
print("--- [4.1b] Tuned RandomForest 최종 학습 및 확률 저장 시작 ---")
final_model = pipe.fit(X, y)

# 5) 예측 확률 저장
test_pred_proba = final_model.predict_proba(X_test)
np.save('[4.1b]_randomforest_tuned_proba.npy', test_pred_proba)
print("RandomForest 예측 확률 저장 완료: [4.1b]_randomforest_tuned_proba.npy")