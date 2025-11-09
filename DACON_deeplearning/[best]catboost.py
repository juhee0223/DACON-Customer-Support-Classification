# 파일 이름: tune_catboost_optuna.py

import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from datetime import datetime
import optuna  # Optuna 라이브러리 임포트
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score

# 0) 설정
SEED = 45
np.random.seed(SEED)

# 경로
data_path = "./"
train_path = os.path.join(data_path, "train.csv")
test_path = os.path.join(data_path, "test.csv")
sub_path = os.path.join(data_path, "sample_submission.csv")

# 1) 데이터 로드
print("--- Optuna CatBoost 튜닝 시작 ---")
print("데이터 로드를 시작합니다...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sub_df = pd.read_csv(sub_path)

TARGET = "support_needs"
ID_COL = "ID"

X = train_df.drop([ID_COL, TARGET], axis=1).copy()
y = train_df[TARGET].copy()
X_test = test_df.drop([ID_COL], axis=1).copy()


# 2) Feature Engineering (가장 좋았던 Feature Set 사용)
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

# 3) 전처리 (CatBoost 방식)
X['contract_length'] = X['contract_length'].astype(str)
X_test['contract_length'] = X_test['contract_length'].astype(str)
X['age_group'] = X['age_group'].astype(str)
X_test['age_group'] = X_test['age_group'].astype(str)
cat_features = ["gender", "subscription_type", "age_group", "contract_length"]

# --- ★★★ 빠른 튜닝을 위해 데이터의 50%만 샘플링 ★★★ ---
X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.5, random_state=SEED, stratify=y)
print(f"빠른 튜닝을 위해 데이터를 샘플링합니다. 학습 데이터 크기: {len(X_sample)}")

# 4) Optuna Objective 함수 정의
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def objective(trial):
    # CatBoost 하이퍼파라미터 탐색 공간 정의
    params = {
        'iterations': trial.suggest_int('iterations', 800, 2500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_state': SEED,
        'verbose': 0,
        'auto_class_weights': 'Balanced',
    }

    model = CatBoostClassifier(**params, cat_features=cat_features)

    f1_list = []
    for train_idx, val_idx in skf.split(X_sample, y_sample):
        X_train, y_train = X_sample.iloc[train_idx], y_sample.iloc[train_idx]
        X_val, y_val = X_sample.iloc[val_idx], y_sample.iloc[val_idx]

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
        preds = model.predict(X_val)
        f1_list.append(f1_score(y_val, preds, average='macro'))

    return np.mean(f1_list)


# 5) Optuna 튜닝 실행
print("\nOptuna를 사용한 하이퍼파라미터 튜닝을 시작합니다...")
study = optuna.create_study(direction='maximize')
# n_trials: 총 시도 횟수. 30~50회 정도면 대부분 좋은 결과를 찾습니다.
study.optimize(objective, n_trials=50)

print("\n===== Optuna 튜닝 완료 =====")
best_params = study.best_params
print(f"최고 점수 (Macro F1): {study.best_value:.5f}")
print("찾아낸 최적의 파라미터:")
print(best_params)
print("=" * 30)

# 6) 전체 데이터로 최종 모델 학습 및 제출
print("\n전체 데이터로 최종 모델 학습을 시작합니다...")
final_model = CatBoostClassifier(**best_params, random_state=SEED, verbose=100,
                                 cat_features=cat_features, auto_class_weights='Balanced')
final_model.fit(X, y)
print("최종 모델 학습 완료.")

print("\n테스트 데이터 예측 및 제출 파일 생성을 시작합니다...")
test_pred = final_model.predict(X_test)
submit = sub_df.copy()
submit[TARGET] = test_pred

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"./submission_catboost_optuna_90percent_{timestamp}.csv"
submit.to_csv(out_path, index=False)

print(f"제출 파일 저장 완료: {out_path}")

# ===== Optuna 튜닝 완료 ===== seed 42, 50%
#
# 최고 점수 (Macro F1): 0.50249
#
# 찾아낸 최적의 파라미터:
#
# {'iterations': 1460, 'depth': 8, 'learning_rate': 0.01011375157729347, 'l2_leaf_reg': 2.9134559843727033, 'colsample_bylevel': 0.8914955767108875, 'random_strength': 0.10543719325919827, 'bagging_temperature': 0.1804730161382898}
#
# ==============================
#
#
# 전체 데이터로 최종 모델 학습을 시작합니다...
#
# 0:      learn: 1.0967854        total: 30.4ms   remaining: 44.4s
# 100:    learn: 1.0046277        total: 2.52s    remaining: 33.9s
# 200:    learn: 0.9828460        total: 5.11s    remaining: 32s
# 300:    learn: 0.9727740        total: 7.67s    remaining: 29.5s
# 400:    learn: 0.9666311        total: 10.2s    remaining: 26.9s
# 500:    learn: 0.9625547        total: 12.6s    remaining: 24.1s
# 600:    learn: 0.9590959        total: 15s      remaining: 21.4s
# 700:    learn: 0.9560608        total: 17.4s    remaining: 18.8s
# 800:    learn: 0.9530655        total: 19.7s    remaining: 16.2s
# 900:    learn: 0.9499584        total: 22.1s    remaining: 13.7s
# 1000:   learn: 0.9466495        total: 24.6s    remaining: 11.3s
# 1100:   learn: 0.9431048        total: 27s      remaining: 8.81s
# 1200:   learn: 0.9396662        total: 29.5s    remaining: 6.36s
# 1300:   learn: 0.9364365        total: 32s      remaining: 3.91s
# 1400:   learn: 0.9326745        total: 34.5s    remaining: 1.45s
# 1459:   learn: 0.9306904        total: 36s      remaining: 0us
#
# 최종 모델 학습 완료.
# 테스트 데이터 예측 및 제출 파일 생성을 시작합니다...
# 제출 파일 저장 완료: ./submission_catboost_tuned_20250914_145548.csv



# ===== Optuna 튜닝 완료 ===== 80퍼, seed 12
# 최고 점수 (Macro F1): 0.49310
# 찾아낸 최적의 파라미터:
# {'iterations': 1920, 'depth': 8, 'learning_rate': 0.01024844230058963, 'l2_leaf_reg': 2.6043668437766123, 'colsample_bylevel': 0.677670069446338, 'random_strength': 0.10626849714518394, 'bagging_temperature': 0.818499748120263}
# ==============================
#
# 전체 데이터로 최종 모델 학습을 시작합니다...
# 0:      learn: 1.0965703        total: 31ms     remaining: 59.4s
# 100:    learn: 1.0041390        total: 2.35s    remaining: 42.4s
# 200:    learn: 0.9811528        total: 4.71s    remaining: 40.3s
# 300:    learn: 0.9708244        total: 6.96s    remaining: 37.5s
# 400:    learn: 0.9639943        total: 9.23s    remaining: 35s
# 500:    learn: 0.9585332        total: 11.5s    remaining: 32.7s
# 600:    learn: 0.9537866        total: 13.7s    remaining: 30.1s
# 700:    learn: 0.9491829        total: 16s      remaining: 27.8s
# 800:    learn: 0.9443386        total: 18.3s    remaining: 25.6s
# 900:    learn: 0.9396671        total: 20.7s    remaining: 23.4s
# 1000:   learn: 0.9348377        total: 23s      remaining: 21.1s
# 1100:   learn: 0.9300672        total: 25.3s    remaining: 18.8s
# 1200:   learn: 0.9254411        total: 27.7s    remaining: 16.6s
# 1300:   learn: 0.9205231        total: 30s      remaining: 14.3s
# 1400:   learn: 0.9157342        total: 32.3s    remaining: 12s
# 1500:   learn: 0.9109467        total: 34.7s    remaining: 9.69s
# 1600:   learn: 0.9063307        total: 37s      remaining: 7.38s
# 1700:   learn: 0.9018234        total: 39.5s    remaining: 5.08s
# 1800:   learn: 0.8972458        total: 41.9s    remaining: 2.77s
# 1900:   learn: 0.8924015        total: 46.9s    remaining: 468ms
# 1919:   learn: 0.8915827        total: 47.6s    remaining: 0us
# 최종 모델 학습 완료.
#
# 테스트 데이터 예측 및 제출 파일 생성을 시작합니다...
# 제출 파일 저장 완료: ./submission_catboost_optuna_90percent_20250914_152432.csv

# ===== Optuna 튜닝 완료 =====
# 최고 점수 (Macro F1): 0.49646
# 찾아낸 최적의 파라미터:
# {'iterations': 2347, 'depth': 8, 'learning_rate': 0.03472616265209295, 'l2_leaf_reg': 5.699103335617795, 'colsample_bylevel': 0.8609855298323066, 'random_strength': 0.15804123353294222, 'bagging_temperature': 0.3893541408606065}
# ==============================
#
# 전체 데이터로 최종 모델 학습을 시작합니다...
# 0:      learn: 1.0922116        total: 30.4ms   remaining: 1m 11s
# 100:    learn: 0.9726959        total: 2.48s    remaining: 55.2s
# 200:    learn: 0.9592598        total: 4.88s    remaining: 52.1s
# 300:    learn: 0.9490318        total: 7.25s    remaining: 49.3s
# 400:    learn: 0.9377763        total: 9.76s    remaining: 47.3s
# 500:    learn: 0.9262178        total: 12.3s    remaining: 45.2s
# 600:    learn: 0.9163807        total: 14.8s    remaining: 42.9s
# 700:    learn: 0.9061409        total: 17.3s    remaining: 40.6s
# 800:    learn: 0.8966258        total: 19.8s    remaining: 38.2s
# 900:    learn: 0.8869863        total: 22.3s    remaining: 35.8s
# 1000:   learn: 0.8771661        total: 24.9s    remaining: 33.5s
# 1100:   learn: 0.8669895        total: 27.5s    remaining: 31.1s
# 1200:   learn: 0.8573809        total: 30s      remaining: 28.6s
# 1300:   learn: 0.8475538        total: 32.6s    remaining: 26.2s
# 1400:   learn: 0.8382229        total: 35.1s    remaining: 23.7s
# 1500:   learn: 0.8286604        total: 37.7s    remaining: 21.2s
# 1600:   learn: 0.8189363        total: 40.3s    remaining: 18.8s
# 1700:   learn: 0.8100350        total: 42.9s    remaining: 16.3s
# 1800:   learn: 0.8005465        total: 45.6s    remaining: 13.8s
# 1900:   learn: 0.7924410        total: 53.4s    remaining: 12.5s
# 2000:   learn: 0.7832216        total: 57.3s    remaining: 9.91s
# 2100:   learn: 0.7746001        total: 59.9s    remaining: 7.01s
# 2200:   learn: 0.7660300        total: 1m 2s    remaining: 4.17s
# 2300:   learn: 0.7577615        total: 1m 5s    remaining: 1.31s
# 2346:   learn: 0.7536992        total: 1m 6s    remaining: 0us
# 최종 모델 학습 완료.
#
# 테스트 데이터 예측 및 제출 파일 생성을 시작합니다...
# 제출 파일 저장 완료: ./submission_catboost_optuna_90percent_20250914_154022.csv

# ===== Optuna 튜닝 완료 ===== 42, 60%??
# 최고 점수 (Macro F1): 0.50250
# 찾아낸 최적의 파라미터:
# {'iterations': 2000, 'depth': 8, 'learning_rate': 0.022017976609618187, 'l2_leaf_reg': 4.7946371135678945, 'colsample_bylevel': 0.5303402768146288, 'random_strength': 0.18767339847825132, 'bagging_temperature': 0.4697532671010844}
# ==============================
#
# 전체 데이터로 최종 모델 학습을 시작합니다...
# 0:      learn: 1.0946787        total: 26.3ms   remaining: 52.6s
# 100:    learn: 0.9824185        total: 2.13s    remaining: 40.1s
# 200:    learn: 0.9681336        total: 4.2s     remaining: 37.6s
# 300:    learn: 0.9602225        total: 6.27s    remaining: 35.4s
# 400:    learn: 0.9528166        total: 8.36s    remaining: 33.3s
# 500:    learn: 0.9449100        total: 10.4s    remaining: 31.2s
# 600:    learn: 0.9374427        total: 12.5s    remaining: 29.1s
# 700:    learn: 0.9299071        total: 14.6s    remaining: 27.1s
# 800:    learn: 0.9225673        total: 16.9s    remaining: 25.2s
# 900:    learn: 0.9157501        total: 23.7s    remaining: 28.9s
# 1000:   learn: 0.9082329        total: 27.3s    remaining: 27.2s
# 1100:   learn: 0.9007477        total: 29.5s    remaining: 24s
# 1200:   learn: 0.8938687        total: 31.7s    remaining: 21.1s
# 1300:   learn: 0.8868329        total: 34s      remaining: 18.3s
# 1400:   learn: 0.8796016        total: 36.2s    remaining: 15.5s
# 1500:   learn: 0.8724790        total: 38.4s    remaining: 12.8s
# 1600:   learn: 0.8656677        total: 40.5s    remaining: 10.1s
# 1700:   learn: 0.8589388        total: 42.8s    remaining: 7.51s
# 1800:   learn: 0.8525616        total: 44.9s    remaining: 4.96s
# 1900:   learn: 0.8458091        total: 47.2s    remaining: 2.46s
# 1999:   learn: 0.8396863        total: 49.5s    remaining: 0us
# 최종 모델 학습 완료.
#
# 테스트 데이터 예측 및 제출 파일 생성을 시작합니다...
# 제출 파일 저장 완료: ./submission_catboost_optuna_90percent_20250914_162048.csv


# ===== Optuna 튜닝 완료 ===== 42, 0.4
# 최고 점수 (Macro F1): 0.50063
# 찾아낸 최적의 파라미터:
# {'iterations': 1827, 'depth': 8, 'learning_rate': 0.01292921006155721, 'l2_leaf_reg': 3.3448170336398997, 'colsample_bylevel': 0.773316038147292, 'random_strength': 0.19452261669379048, 'bagging_temperature': 0.5369284474261624}
# ==============================
#
# 전체 데이터로 최종 모델 학습을 시작합니다...
# 0:      learn: 1.0961356        total: 26.3ms   remaining: 48s
# 100:    learn: 0.9967168        total: 2.45s    remaining: 41.8s
# 200:    learn: 0.9775711        total: 4.79s    remaining: 38.7s
# 300:    learn: 0.9681783        total: 7.13s    remaining: 36.2s
# 400:    learn: 0.9613954        total: 9.51s    remaining: 33.8s
# 500:    learn: 0.9556330        total: 11.9s    remaining: 31.5s
# 600:    learn: 0.9506449        total: 14.2s    remaining: 29s
# 700:    learn: 0.9456312        total: 16.6s    remaining: 26.7s
# 800:    learn: 0.9404443        total: 19s      remaining: 24.4s
# 900:    learn: 0.9351510        total: 21.7s    remaining: 22.3s
# 1000:   learn: 0.9294374        total: 24.2s    remaining: 19.9s
# 1100:   learn: 0.9234146        total: 26.7s    remaining: 17.6s
# 1200:   learn: 0.9186678        total: 34.8s    remaining: 18.1s
# 1300:   learn: 0.9136586        total: 37.3s    remaining: 15.1s
# 1400:   learn: 0.9087858        total: 39.7s    remaining: 12.1s
# 1500:   learn: 0.9035085        total: 42.1s    remaining: 9.15s
# 1600:   learn: 0.8985800        total: 44.9s    remaining: 6.33s
# 1700:   learn: 0.8933332        total: 47.4s    remaining: 3.51s
# 1800:   learn: 0.8882496        total: 50.3s    remaining: 726ms
# 1826:   learn: 0.8870065        total: 51.2s    remaining: 0us
# 최종 모델 학습 완료.
#
# 테스트 데이터 예측 및 제출 파일 생성을 시작합니다...
# 제출 파일 저장 완료: ./submission_catboost_optuna_90percent_20250914_164627.csv