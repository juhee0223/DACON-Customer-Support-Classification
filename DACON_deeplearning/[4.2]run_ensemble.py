import numpy as np
import pandas as pd
from datetime import datetime
import os

print("--- [4.2] 앙상블 시작 ---")
data_path = "./"
sub_path = os.path.join(data_path, "sample_submission.csv")

# 1. 저장된 예측 확률 파일 로드
print("저장된 예측 확률 파일을 로드합니다...")
cat_proba = np.load('[4.1a]_catboost_tuned_proba.npy')
rf_proba = np.load('[4.1b]_randomforest_tuned_proba.npy')

# 2. 가중 평균 계산 (CatBoost 6 : RF 4)
# CV 점수가 더 높았던 CatBoost에 더 높은 가중치를 부여합니다.
print("예측 확률을 가중 평균합니다 (CatBoost: 60%, RF: 40%)...")
ensemble_proba = (cat_proba * 0.6) + (rf_proba * 0.4)

# 3. 최종 클래스 예측
final_pred = np.argmax(ensemble_proba, axis=1)

# 4. 제출 파일 생성
print("최종 제출 파일을 생성합니다...")
sub_df = pd.read_csv(sub_path)
sub_df['support_needs'] = final_pred

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"./[4.2]_submission_ensemble_{timestamp}.csv"
sub_df.to_csv(out_path, index=False)

print(f"최종 앙상블 제출 파일 저장 완료: {out_path}")