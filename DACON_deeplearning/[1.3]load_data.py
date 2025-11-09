# (venv) jh@jh0223:~/DLDACON$ python3 \[1.3\]load_data.py
# 데이터 로드 및 분할 완료.
# 학습 데이터 형태: (30858, 8)
# 테스트 데이터 형태: (13225, 8)

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