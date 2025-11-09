# 1. 라이브러리 임포트
import pandas as pd
from catboost import CatBoostClassifier
import os
import pandas as pd

# 경로 설정
data_path = "./"
train_path = os.path.join(data_path, "train.csv")
test_path  = os.path.join(data_path, "test.csv")
sub_path   = os.path.join(data_path, "sample_submission.csv")

# 데이터 로드
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)
sub_df   = pd.read_csv(sub_path)

# 타겟 및 ID 칼럼 정의
TARGET = "support_needs"
ID_COL = "ID"

# 학습 데이터와 테스트 데이터 분리
X = train_df.drop([ID_COL, TARGET], axis=1).copy()
y = train_df[TARGET].copy()
X_test = test_df.drop([ID_COL], axis=1).copy()

print("데이터 로드 및 분할 완료.")
print(f"학습 데이터 형태: {X.shape}")
print(f"테스트 데이터 형태: {X_test.shape}")

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6
    # 비율 특성
    df["tenure_per_contract"] = df["tenure"] / (df["contract_length"] + eps)
    df["delay_ratio"]         = df["payment_interval"] / (df["contract_length"] + eps)
    df["idle_ratio"]          = df["after_interaction"] / (df["tenure"] + eps)
    # 플래그 특성
    df["late_flag"]           = (df["payment_interval"] > 0).astype(int)
    # 상호작용 특성
    if 'frequent' in df.columns:
        df['total_usage_score'] = df['tenure'] * df['frequent']
    df['risk_score'] = df['payment_interval'] + df['after_interaction']
    # 구간화 특성
    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ['10s', '20s', '30s', '40s', '50s', '60+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    return df

# 4. 함수 호출하여 데이터 변환
print("Feature Engineering을 적용합니다...")
X = add_features(X)
X_test = add_features(X_test)
print("Feature Engineering 완료.")

#결과 확인
# 데이터 로드 및 분할 완료.
# 학습 데이터 형태: (30858, 8)
# 테스트 데이터 형태: (13225, 8)
# Feature Engineering을 적용합니다...
# Feature Engineering 완료.
