import warnings

warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from datetime import datetime

# 0) 설정
SEED = 42
np.random.seed(SEED)

# 1) 데이터 로드
data_path = "./"
train_path = os.path.join(data_path, "train.csv")
test_path = os.path.join(data_path, "test.csv")
sub_path = os.path.join(data_path, "sample_submission.csv")

print("--- [3.1] CatBoost 베이스라인 성능 측정 시작 ---")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sub_df = pd.read_csv(sub_path)

TARGET = "support_needs"
ID_COL = "ID"

X = train_df.drop([ID_COL, TARGET], axis=1).copy()
y = train_df[TARGET].copy()
X_test = test_df.drop([ID_COL], axis=1).copy()


# 2) Feature Engineering
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6
    df["tenure_per_contract"] = df["tenure"] / (df["contract_length"] + eps)
    df["delay_ratio"] = df["payment_interval"] / (df["contract_length"] + eps)
    df["idle_ratio"] = df["after_interaction"] / (df["tenure"] + eps)
    df["late_flag"] = (df["payment_interval"] > 0).astype(int)
    if 'frequent' in df.columns:
        df['total_usage_score'] = df['tenure'] * df['frequent']
    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ['10s', '20s', '30s', '40s', '50s', '60+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    df['risk_score'] = df['payment_interval'] + df['after_interaction']
    return df


print("Feature Engineering을 적용합니다...")
X = add_features(X)
X_test = add_features(X_test)

# 3) 전처리
X['contract_length'] = X['contract_length'].astype(str)
X_test['contract_length'] = X_test['contract_length'].astype(str)
X['age_group'] = X['age_group'].astype(str)
X_test['age_group'] = X_test['age_group'].astype(str)
cat_features = ["gender", "subscription_type", "age_group", "contract_length"]

# 4) 교차 검증으로 베이스라인 성능 측정
print("\n교차 검증을 시작합니다...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # 기본 파라미터 CatBoost 모델
    model = CatBoostClassifier(
        random_state=SEED,
        verbose=0,
        auto_class_weights='Balanced',
        cat_features=cat_features
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    fold_f1 = f1_score(y_val, preds, average='macro')
    f1_scores.append(fold_f1)
    print(f"Fold {fold + 1} Macro F1 Score: {fold_f1:.5f}")

print("\n===== [결과물 3.1] CatBoost 베이스라인 CV 결과 =====")
print(f"평균 Macro F1 Score: {np.mean(f1_scores):.5f}")
print("================================================")

# 5) 전체 데이터로 학습 및 제출 파일 생성 (베이스라인)
print("\n전체 데이터로 최종 모델을 학습합니다...")
final_model = CatBoostClassifier(
    random_state=SEED,
    verbose=100,
    auto_class_weights='Balanced',
    cat_features=cat_features
)
final_model.fit(X, y)

print("\n제출 파일을 생성합니다...")
test_pred = final_model.predict(X_test)
submit = sub_df.copy()
submit[TARGET] = test_pred

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"./[3.1]_submission_catboost_baseline_{timestamp}.csv"
submit.to_csv(out_path, index=False)
print(f"베이스라인 제출 파일 저장 완료: {out_path}")

# (venv) jh@jh0223:~/DLDACON$ python3 \[3.1\]run_catboost_baseline.py
# --- [3.1] CatBoost 베이스라인 성능 측정 시작 ---
# Feature Engineering을 적용합니다...
#
# 교차 검증을 시작합니다...
# Fold 1 Macro F1 Score: 0.48556
# Fold 2 Macro F1 Score: 0.48850
# Fold 3 Macro F1 Score: 0.48330
# Fold 4 Macro F1 Score: 0.47966
# Fold 5 Macro F1 Score: 0.47401
#
# ===== [결과물 3.1] CatBoost 베이스라인 CV 결과 =====
# 평균 Macro F1 Score: 0.48221
# ================================================
#
# 전체 데이터로 최종 모델을 학습합니다...
# Learning rate set to 0.09425
# 0:      learn: 1.0849042        total: 26.1ms   remaining: 26s
# 100:    learn: 0.9802982        total: 1.84s    remaining: 16.4s
# 200:    learn: 0.9597388        total: 3.92s    remaining: 15.6s
# 300:    learn: 0.9430648        total: 6.02s    remaining: 14s
# 400:    learn: 0.9270344        total: 9.25s    remaining: 13.8s
# 500:    learn: 0.9114862        total: 16.7s    remaining: 16.6s
# 600:    learn: 0.8971315        total: 19.7s    remaining: 13.1s
# 700:    learn: 0.8834205        total: 21.5s    remaining: 9.19s
# 800:    learn: 0.8701206        total: 23.5s    remaining: 5.84s
# 900:    learn: 0.8579006        total: 25.3s    remaining: 2.78s
# 999:    learn: 0.8454839        total: 27.1s    remaining: 0us
#
# 제출 파일을 생성합니다...
# 베이스라인 제출 파일 저장 완료: ./[3.1]_submission_catboost_baseline_20250918_134818.csv