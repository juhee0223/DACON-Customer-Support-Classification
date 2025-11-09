import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

# 0) 설정
SEED = 42
np.random.seed(SEED)

# 1) 데이터 로드 및 Feature Engineering (이전과 동일)
data_path = "./"
train_path = os.path.join(data_path, "train.csv")
test_path  = os.path.join(data_path, "test.csv")
sub_path   = os.path.join(data_path, "sample_submission.csv")
print("--- [5.1] Stacking 앙상블 시작 ---")
train_df = pd.read_csv(train_path); test_df = pd.read_csv(test_path); sub_df = pd.read_csv(sub_path)
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

# --- ★★★ 스태킹을 위한 모델별 전처리 및 정의 ★★★ ---

# 2-1) RandomForest 모델 및 전처리 파이프라인 정의
cat_features_rf = ["gender", "subscription_type", "age_group", "contract_length"]
X_rf = X.copy(); X_test_rf = X_test.copy()
X_rf['contract_length'] = X_rf['contract_length'].astype(str); X_test_rf['contract_length'] = X_test_rf['contract_length'].astype(str)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
preprocess_rf = ColumnTransformer(transformers=[("cat", ohe, cat_features_rf)], remainder="passthrough")
best_rf_params = {'n_estimators': 310, 'max_depth': 8, 'min_samples_split': 4, 'min_samples_leaf': 4, 'max_features': None, 'random_state': SEED, 'class_weight': "balanced", 'n_jobs': -1}
model_rf = Pipeline(steps=[("prep", preprocess_rf), ("clf", RandomForestClassifier(**best_rf_params))])

# 2-2) CatBoost 모델 정의
cat_features_cb = ["gender", "subscription_type", "age_group", "contract_length"]
X_cb = X.copy(); X_test_cb = X_test.copy()
X_cb['contract_length'] = X_cb['contract_length'].astype(str); X_test_cb['contract_length'] = X_test_cb['contract_length'].astype(str)
X_cb['age_group'] = X_cb['age_group'].astype(str); X_test_cb['age_group'] = X_test_cb['age_group'].astype(str)
best_cat_params = {'iterations': 1807, 'depth': 7, 'learning_rate': 0.011007442573869236, 'l2_leaf_reg': 9.960037311199006, 'colsample_bylevel': 0.8239914444396733, 'random_strength': 0.15658132267797456, 'bagging_temperature': 0.2528316917680684}
model_cat = CatBoostClassifier(**best_cat_params, random_state=SEED, verbose=0, cat_features=cat_features_cb, auto_class_weights='Balanced')

# 3) Stacking Classifier 정의
# 1차 모델(Base Models) 목록
estimators = [
    ('randomforest', model_rf),
    ('catboost', model_cat)
]
# 2차 모델(Meta-Model)
meta_model = LogisticRegression(random_state=SEED, n_jobs=-1)

# Stacking Classifier 조립
# cv=5: 1차 모델들이 예측값을 만들 때 사용할 내부 교차 검증 폴드 수
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

# 4) 교차 검증으로 스태킹 모델 성능 측정
print("\n스태킹 앙상블 모델의 교차 검증을 시작합니다...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
# StackingClassifier 자체가 하나의 모델처럼 동작하므로, 그대로 CV를 수행하면 됨
# 단, StackingClassifier 내부에서도 CV가 돌아가므로 시간이 매우 오래 걸릴 수 있음!
# 여기서는 CV 없이 바로 최종 모델을 학습하고 제출
# (만약 CV 점수가 꼭 필요하다면 아래 final_model.fit 부분을 for loop으로 변경)

# 5) 전체 데이터로 최종 모델 학습 및 제출 파일 생성
print("\n전체 데이터로 최종 스태킹 모델을 학습합니다... (시간이 오래 소요될 수 있습니다)")
final_model = stacking_clf.fit(X, y)
print("최종 모델 학습 완료.")

print("\n제출 파일을 생성합니다...")
test_pred = final_model.predict(X_test)
submit = sub_df.copy()
submit[TARGET] = test_pred
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"./[5.1]_submission_stacking_ensemble_{timestamp}.csv"
submit.to_csv(out_path, index=False)
print(f"스태킹 앙상블 제출 파일 저장 완료: {out_path}")