import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# --- 이 스크립트는 CV 없이, 전체 데이터로 최종 모델을 학습하고 예측 확률만 저장합니다. ---

# 0) 설정
SEED = 42
np.random.seed(SEED)

# 1) 데이터 로드 및 Feature Engineering
# (이전 스크립트들과 동일한 데이터 로드 및 add_features 함수를 여기에 포함)
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
X['contract_length'] = X['contract_length'].astype(str); X_test['contract_length'] = X_test['contract_length'].astype(str)
X['age_group'] = X['age_group'].astype(str); X_test['age_group'] = X_test['age_group'].astype(str)
cat_features = ["gender", "subscription_type", "age_group", "contract_length"]

# 2) 최종 모델 정의 (Optuna로 찾은 최적 파라미터 사용)
best_params_catboost = {'iterations': 1807, 'depth': 7, 'learning_rate': 0.011007442573869236, 'l2_leaf_reg': 9.960037311199006, 'colsample_bylevel': 0.8239914444396733, 'random_strength': 0.15658132267797456, 'bagging_temperature': 0.2528316917680684}

final_model = CatBoostClassifier(**best_params_catboost, random_state=SEED, verbose=100,
                                 cat_features=cat_features, auto_class_weights='Balanced')

# 3) 전체 데이터로 학습
print("--- [4.1a] Tuned CatBoost 최종 학습 및 확률 저장 시작 ---")
final_model.fit(X, y)

# 4) 예측 확률 저장
test_pred_proba = final_model.predict_proba(X_test)
np.save('[4.1a]_catboost_tuned_proba.npy', test_pred_proba)
print("CatBoost 예측 확률 저장 완료: [4.1a]_catboost_tuned_proba.npy")