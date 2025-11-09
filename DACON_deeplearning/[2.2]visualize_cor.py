# [코드 1.3]의 데이터 로드 부분을 먼저 실행했다고 가정
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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


# 상관관계 분석은 숫자형 변수들만 가능
numerical_features = X.select_dtypes(include=np.number)
corr_matrix = numerical_features.corr()

# 히트맵 그리기
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Original Feature Correlation Heatmap', fontsize=16)

# 이미지 파일로 저장
output_filename = 'correlation_heatmap_original.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"상관관계 히트맵이 '{output_filename}' 파일로 저장되었습니다.")