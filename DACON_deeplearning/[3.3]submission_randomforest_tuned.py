import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

# 0) 설정
SEED = 42
np.random.seed(SEED)

# 1) 데이터 로드 및 Feature Engineering (이전과 동일)
data_path = "./"
train_path = os.path.join(data_path, "train.csv")
test_path  = os.path.join(data_path, "test.csv")
sub_path   = os.path.join(data_path, "sample_submission.csv")

print("--- [3.3] Tuned RandomForest 성능 측정 시작 ---")
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)
sub_df   = pd.read_csv(sub_path)
TARGET = "support_needs"; ID_COL = "ID"
X = train_df.drop([ID_COL, TARGET], axis=1).copy(); y = train_df[TARGET].copy()
X_test = test_df.drop([ID_COL], axis=1).copy()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); eps = 1e-6
    df["tenure_per_contract"] = df["tenure"] / (df["contract_length"] + eps)
    df["delay_ratio"] = df["payment_interval"] / (df["contract_length"] + eps)
    df["idle_ratio"] = df["after_interaction"] / (df["tenure"] + eps)
    df["late_flag"] = (df["payment_interval"] > 0).astype(int)
    if 'frequent' in df.columns: df['total_usage_score'] = df['tenure'] * df['frequent']
    bins = [0, 20, 30, 40, 50, 60, 100]; labels = ['10s', '20s', '30s', '40s', '50s', '60+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    df['risk_score'] = df['payment_interval'] + df['after_interaction']
    return df
X = add_features(X)
X_test = add_features(X_test)

# 3) 전처리 (RandomForest 방식)
cat_cols = ["gender", "subscription_type", "age_group", "contract_length"]
X['contract_length'] = X['contract_length'].astype(str)
X_test['contract_length'] = X_test['contract_length'].astype(str)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
preprocess = ColumnTransformer(
    transformers=[("cat", ohe, cat_cols)],
    remainder="passthrough"
)

# 4) 모델 정의 (우리가 찾은 최적 파라미터 직접 사용)
best_rf_params = {
    'n_estimators': 310, 'max_depth': 8, 'min_samples_split': 4,
    'min_samples_leaf': 4, 'max_features': None, 'random_state': SEED,
    'class_weight': "balanced", 'n_jobs': -1
}
clf_rf_tuned = RandomForestClassifier(**best_rf_params)
pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf_rf_tuned)])

# 5) 교차 검증으로 최종 성능 측정
print("\n교차 검증을 시작합니다...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
f1_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    pipe_clone = pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    val_preds = pipe_clone.predict(X.iloc[val_idx])
    fold_f1 = f1_score(y.iloc[val_idx], val_preds, average="macro")
    f1_scores.append(fold_f1)
    print(f"Fold {fold+1} Macro F1 Score: {fold_f1:.5f}")

print("\n===== [결과물 3.3] Tuned RandomForest CV 결과 =====")
print(f"평균 Macro F1 Score: {np.mean(f1_scores):.5f}")
print("===================================================")

# 6) 전체 데이터로 학습 및 제출 파일 생성
print("\n전체 데이터로 최종 모델을 학습합니다...")
final_model = pipe.fit(X, y)
print("최종 모델 학습 완료.")
test_pred = final_model.predict(X_test)
submit = sub_df.copy()
submit[TARGET] = test_pred
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"./[3.3]_submission_randomforest_tuned_{timestamp}.csv"
submit.to_csv(out_path, index=False)
print(f"\n제출 파일 저장 완료: {out_path}")

# (venv) jh@jh0223:~/DLDACON$ python3 \[3.3\]submission_randomforest_tuned.py
# --- [3.3] Tuned RandomForest 성능 측정 시작 ---
#
# 교차 검증을 시작합니다...
# Fold 1 Macro F1 Score: 0.50003
# Fold 2 Macro F1 Score: 0.49271
# Fold 3 Macro F1 Score: 0.48911
# Fold 4 Macro F1 Score: 0.49732
# Fold 5 Macro F1 Score: 0.49175

# ===== [결과물 3.3] Tuned RandomForest CV 결과 =====
# 평균 Macro F1 Score: 0.49419
# ===================================================
#
# 전체 데이터로 최종 모델을 학습합니다...
# 최종 모델 학습 완료.
#
# 제출 파일 저장 완료: ./[3.3]_submission_randomforest_tuned_20250918_141750.csv