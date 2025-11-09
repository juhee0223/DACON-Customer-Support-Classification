# 위 코드를 [3.2]_tune_catboost_optuna.py로 저장하고 실행하면,
# Optuna가 30번의 시도를 통해 최적의 파라미터를 탐색하는 과정이 실시간으로 보일 것입니다.
#
# 모든 과정이 끝나면, 콘솔에 **[결과물 3.2]**가 출력됩니다.
# 여기서 얻은 **최고 점수 (Macro F1)**가 우리의 베이스라인 점수
# **0.48221**을 얼마나 뛰어넘었는지 확인하는 것이 이번 단계의 핵심입니다.

import warnings

warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from datetime import datetime
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score

# 0) 설정
SEED = 42
np.random.seed(SEED)

# 1) 데이터 로드 및 Feature Engineering (이전과 동일)
data_path = "./"
train_path = os.path.join(data_path, "train.csv")
test_path = os.path.join(data_path, "test.csv")
sub_path = os.path.join(data_path, "sample_submission.csv")

print("--- [3.2] Optuna CatBoost 튜닝 시작 ---")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sub_df = pd.read_csv(sub_path)
TARGET = "support_needs";
ID_COL = "ID"
X = train_df.drop([ID_COL, TARGET], axis=1).copy();
y = train_df[TARGET].copy()
X_test = test_df.drop([ID_COL], axis=1).copy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy();
    eps = 1e-6
    df["tenure_per_contract"] = df["tenure"] / (df["contract_length"] + eps)
    df["delay_ratio"] = df["payment_interval"] / (df["contract_length"] + eps)
    df["idle_ratio"] = df["after_interaction"] / (df["tenure"] + eps)
    df["late_flag"] = (df["payment_interval"] > 0).astype(int)
    if 'frequent' in df.columns: df['total_usage_score'] = df['tenure'] * df['frequent']
    bins = [0, 20, 30, 40, 50, 60, 100];
    labels = ['10s', '20s', '30s', '40s', '50s', '60+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    df['risk_score'] = df['payment_interval'] + df['after_interaction']
    return df


X = add_features(X)
X_test = add_features(X_test)

# 3) 전처리
X['contract_length'] = X['contract_length'].astype(str)
X_test['contract_length'] = X_test['contract_length'].astype(str)
X['age_group'] = X['age_group'].astype(str)
X_test['age_group'] = X_test['age_group'].astype(str)
cat_features = ["gender", "subscription_type", "age_group", "contract_length"]

# 빠른 튜닝을 위해 데이터의 50%만 샘플링
X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.5, random_state=SEED, stratify=y)
print(f"빠른 튜닝을 위해 데이터를 샘플링합니다. 학습 데이터 크기: {len(X_sample)}")

# 4) Optuna Objective 함수 정의
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def objective(trial):
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
study.optimize(objective, n_trials=30)  # n_trials: 총 시도 횟수

# 6) 튜닝 결과 출력
print("\n===== [결과물 3.2] Optuna 튜닝 완료 =====")
best_params = study.best_params
print(f"최고 점수 (Macro F1): {study.best_value:.5f}")
print("찾아낸 최적의 파라미터:")
print(best_params)
print("=" * 50)

# 7) 전체 데이터로 최종 모델 학습 및 제출
print("\n전체 데이터로 최종 모델 학습을 시작합니다...")
final_model = CatBoostClassifier(**best_params, random_state=SEED, verbose=100,
                                 cat_features=cat_features, auto_class_weights='Balanced')
final_model.fit(X, y)

print("\n제출 파일을 생성합니다...")
test_pred = final_model.predict(X_test)
submit = sub_df.copy();
submit[TARGET] = test_pred
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"./[3.2]_submission_catboost_tuned_{timestamp}.csv"
submit.to_csv(out_path, index=False)
print(f"튜닝된 모델의 제출 파일 저장 완료: {out_path}")

# (venv) jh@jh0223:~/DLDACON$ python3 \[3.2\]tune_catboost_optuna.py
# --- [3.2] Optuna CatBoost 튜닝 시작 ---
# 빠른 튜닝을 위해 데이터를 샘플링합니다. 학습 데이터 크기: 15429
#
# Optuna를 사용한 하이퍼파라미터 튜닝을 시작합니다...
# [I 2025-09-18 13:57:05,305] A new study created in memory with name: no-name-114bd4b2-315f-4c61-8697-2fb3451e14be
# [I 2025-09-18 13:57:51,045] Trial 0 finished with value: 0.50030503317578 and parameters: {'iterations': 1313, 'depth': 10, 'learning_rate': 0.0273177445349186, 'l2_leaf_reg': 7.39655521479796, 'colsample_bylevel': 0.5506811008612942, 'random_strength': 0.8646523367144177, 'bagging_temperature': 0.03216304431659567}. Best is trial 0 with value: 0.50030503317578.
# [I 2025-09-18 13:58:05,503] Trial 1 finished with value: 0.4980147381943034 and parameters: {'iterations': 2447, 'depth': 4, 'learning_rate': 0.03608867334605293, 'l2_leaf_reg': 5.7994484005292986, 'colsample_bylevel': 0.684141469460887, 'random_strength': 3.583819588427091, 'bagging_temperature': 0.04837678157286951}. Best is trial 0 with value: 0.50030503317578.
# [I 2025-09-18 13:58:26,564] Trial 2 finished with value: 0.4984754278035858 and parameters: {'iterations': 2441, 'depth': 9, 'learning_rate': 0.03209777790965327, 'l2_leaf_reg': 2.6939355599211208, 'colsample_bylevel': 0.9426434766430931, 'random_strength': 0.6308656240103806, 'bagging_temperature': 0.07841692883994011}. Best is trial 0 with value: 0.50030503317578.
# [I 2025-09-18 13:59:21,743] Trial 3 finished with value: 0.5003967191353489 and parameters: {'iterations': 1748, 'depth': 9, 'learning_rate': 0.010425059289547121, 'l2_leaf_reg': 3.460724617544609, 'colsample_bylevel': 0.9439995305020125, 'random_strength': 0.8959544571516189, 'bagging_temperature': 0.32643717376487935}. Best is trial 3 with value: 0.5003967191353489.
# [I 2025-09-18 14:00:03,766] Trial 4 finished with value: 0.49815717485638533 and parameters: {'iterations': 1282, 'depth': 8, 'learning_rate': 0.01803747123508693, 'l2_leaf_reg': 2.65713924222419, 'colsample_bylevel': 0.954983566155747, 'random_strength': 7.4596221908160025, 'bagging_temperature': 0.6467673139310449}. Best is trial 3 with value: 0.5003967191353489.
# [I 2025-09-18 14:00:24,508] Trial 5 finished with value: 0.4972997351598313 and parameters: {'iterations': 1188, 'depth': 9, 'learning_rate': 0.03570765237933341, 'l2_leaf_reg': 3.681855610303627, 'colsample_bylevel': 0.7358579193912687, 'random_strength': 1.6699596605427645, 'bagging_temperature': 0.8502932321814086}. Best is trial 3 with value: 0.5003967191353489.
# [I 2025-09-18 14:00:36,096] Trial 6 finished with value: 0.49914129674867735 and parameters: {'iterations': 1286, 'depth': 6, 'learning_rate': 0.06876111195723938, 'l2_leaf_reg': 5.026676047414477, 'colsample_bylevel': 0.9812331319766632, 'random_strength': 9.682706930700103, 'bagging_temperature': 0.05685458814680133}. Best is trial 3 with value: 0.5003967191353489.
# [I 2025-09-18 14:01:05,823] Trial 7 finished with value: 0.49576472575954417 and parameters: {'iterations': 1806, 'depth': 10, 'learning_rate': 0.03615834332293168, 'l2_leaf_reg': 4.885473917509314, 'colsample_bylevel': 0.7634021632549224, 'random_strength': 0.39094373643705194, 'bagging_temperature': 0.8195174193548235}. Best is trial 3 with value: 0.5003967191353489.
# [I 2025-09-18 14:01:40,367] Trial 8 finished with value: 0.4980145111217646 and parameters: {'iterations': 1460, 'depth': 9, 'learning_rate': 0.03425481895053704, 'l2_leaf_reg': 6.203486589154315, 'colsample_bylevel': 0.9601411385664683, 'random_strength': 6.2263697443002215, 'bagging_temperature': 0.5995646737238144}. Best is trial 3 with value: 0.5003967191353489.
# [I 2025-09-18 14:01:55,288] Trial 9 finished with value: 0.5013723515038699 and parameters: {'iterations': 1139, 'depth': 6, 'learning_rate': 0.02762090660349932, 'l2_leaf_reg': 9.844781654932666, 'colsample_bylevel': 0.636242667675528, 'random_strength': 0.10233111176642483, 'bagging_temperature': 0.8972666762400124}. Best is trial 9 with value: 0.5013723515038699.
# [I 2025-09-18 14:02:08,882] Trial 10 finished with value: 0.49914353417359597 and parameters: {'iterations': 834, 'depth': 6, 'learning_rate': 0.09817633956972284, 'l2_leaf_reg': 9.979063245006417, 'colsample_bylevel': 0.5277107928824921, 'random_strength': 0.12870159365884726, 'bagging_temperature': 0.9876333545403143}. Best is trial 9 with value: 0.5013723515038699.
# [I 2025-09-18 14:03:15,501] Trial 11 finished with value: 0.5019223288513696 and parameters: {'iterations': 1807, 'depth': 7, 'learning_rate': 0.011007442573869236, 'l2_leaf_reg': 9.960037311199006, 'colsample_bylevel': 0.8239914444396733, 'random_strength': 0.15658132267797456, 'bagging_temperature': 0.2528316917680684}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:03:59,847] Trial 12 finished with value: 0.4995863813324132 and parameters: {'iterations': 2034, 'depth': 6, 'learning_rate': 0.011784038525027435, 'l2_leaf_reg': 9.818538565194402, 'colsample_bylevel': 0.8339067418818698, 'random_strength': 0.10057888975353751, 'bagging_temperature': 0.3360276340197232}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:04:15,262] Trial 13 finished with value: 0.4985765452966744 and parameters: {'iterations': 2032, 'depth': 4, 'learning_rate': 0.016579149965873917, 'l2_leaf_reg': 8.371365248438423, 'colsample_bylevel': 0.6675180214913278, 'random_strength': 0.2229407976700504, 'bagging_temperature': 0.3139134294409466}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:04:34,660] Trial 14 finished with value: 0.5009713106383323 and parameters: {'iterations': 913, 'depth': 7, 'learning_rate': 0.018931646874518065, 'l2_leaf_reg': 8.366312991425485, 'colsample_bylevel': 0.6035636299265099, 'random_strength': 0.23592821144936438, 'bagging_temperature': 0.21598554329570335}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:04:40,680] Trial 15 finished with value: 0.49892557721236874 and parameters: {'iterations': 1547, 'depth': 5, 'learning_rate': 0.05386297183587029, 'l2_leaf_reg': 8.719830723471063, 'colsample_bylevel': 0.8547490025935333, 'random_strength': 0.21362119396791351, 'bagging_temperature': 0.4920086455452827}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:05:00,401] Trial 16 finished with value: 0.5001199185171445 and parameters: {'iterations': 1997, 'depth': 7, 'learning_rate': 0.023152830355178757, 'l2_leaf_reg': 7.132273255880593, 'colsample_bylevel': 0.8390997239474063, 'random_strength': 0.4459130531358541, 'bagging_temperature': 0.46177928482723773}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:05:05,684] Trial 17 finished with value: 0.4990928365373164 and parameters: {'iterations': 1111, 'depth': 5, 'learning_rate': 0.05400792931082833, 'l2_leaf_reg': 1.1749538388606595, 'colsample_bylevel': 0.6151151572529235, 'random_strength': 0.14870038877348835, 'bagging_temperature': 0.6664872947969602}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:05:46,624] Trial 18 finished with value: 0.49847532933351324 and parameters: {'iterations': 1038, 'depth': 8, 'learning_rate': 0.013004822016570614, 'l2_leaf_reg': 8.990565028321019, 'colsample_bylevel': 0.7651122353643544, 'random_strength': 1.5364188038089905, 'bagging_temperature': 0.9908914843964006}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:06:21,346] Trial 19 finished with value: 0.499420000127205 and parameters: {'iterations': 1604, 'depth': 5, 'learning_rate': 0.014673872434103017, 'l2_leaf_reg': 7.570241363981035, 'colsample_bylevel': 0.8742791500370907, 'random_strength': 0.3277542903864024, 'bagging_temperature': 0.18649029439964365}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:07:05,284] Trial 20 finished with value: 0.5013560869762724 and parameters: {'iterations': 2294, 'depth': 8, 'learning_rate': 0.022768195740607868, 'l2_leaf_reg': 9.61603269216798, 'colsample_bylevel': 0.702940845099064, 'random_strength': 0.1449609202496863, 'bagging_temperature': 0.7631114137234986}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:07:50,563] Trial 21 finished with value: 0.5015939294987157 and parameters: {'iterations': 2252, 'depth': 8, 'learning_rate': 0.022921154630579223, 'l2_leaf_reg': 9.331223857538111, 'colsample_bylevel': 0.7045012950574308, 'random_strength': 0.15510893295071854, 'bagging_temperature': 0.8032019068285838}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:08:25,779] Trial 22 finished with value: 0.5005363907616481 and parameters: {'iterations': 2242, 'depth': 7, 'learning_rate': 0.02557776109335563, 'l2_leaf_reg': 9.162545156643167, 'colsample_bylevel': 0.6305132618082245, 'random_strength': 0.10994326749429928, 'bagging_temperature': 0.8757726931167322}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:08:52,659] Trial 23 finished with value: 0.5008290125687862 and parameters: {'iterations': 1916, 'depth': 7, 'learning_rate': 0.044857757625314236, 'l2_leaf_reg': 8.299474603769415, 'colsample_bylevel': 0.7983780689812746, 'random_strength': 0.1802366037560636, 'bagging_temperature': 0.7444376753744021}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:09:27,582] Trial 24 finished with value: 0.5002266610224015 and parameters: {'iterations': 2240, 'depth': 6, 'learning_rate': 0.0206754432406021, 'l2_leaf_reg': 6.651043188586048, 'colsample_bylevel': 0.582313007276312, 'random_strength': 0.3053747644515389, 'bagging_temperature': 0.902389755822089}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:10:21,295] Trial 25 finished with value: 0.5002082330440396 and parameters: {'iterations': 1414, 'depth': 8, 'learning_rate': 0.015229659724569989, 'l2_leaf_reg': 9.309454309713386, 'colsample_bylevel': 0.7202321823859027, 'random_strength': 0.5204944132105459, 'bagging_temperature': 0.5525504550838833}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:10:52,665] Trial 26 finished with value: 0.5006995333828683 and parameters: {'iterations': 1681, 'depth': 7, 'learning_rate': 0.028200117238441808, 'l2_leaf_reg': 7.790467246843398, 'colsample_bylevel': 0.6439071537399416, 'random_strength': 0.1015652402608955, 'bagging_temperature': 0.7173151035318104}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:11:46,867] Trial 27 finished with value: 0.5001716993736071 and parameters: {'iterations': 1850, 'depth': 6, 'learning_rate': 0.010691245897730873, 'l2_leaf_reg': 9.851244651010148, 'colsample_bylevel': 0.6627021982616429, 'random_strength': 0.25893055860482267, 'bagging_temperature': 0.912865034793714}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:13:00,996] Trial 28 finished with value: 0.5018942603263586 and parameters: {'iterations': 2100, 'depth': 8, 'learning_rate': 0.01328969101912971, 'l2_leaf_reg': 9.05039660764281, 'colsample_bylevel': 0.890589255575101, 'random_strength': 0.1807578246473189, 'bagging_temperature': 0.17990543801124181}. Best is trial 11 with value: 0.5019223288513696.
# [I 2025-09-18 14:14:39,577] Trial 29 finished with value: 0.5001196125508466 and parameters: {'iterations': 2110, 'depth': 10, 'learning_rate': 0.01263125345294833, 'l2_leaf_reg': 7.03582124637947, 'colsample_bylevel': 0.8991090206359614, 'random_strength': 0.6489984920446735, 'bagging_temperature': 0.40603817261653874}. Best is trial 11 with value: 0.5019223288513696.
#
# ===== [결과물 3.2] Optuna 튜닝 완료 =====
# 최고 점수 (Macro F1): 0.50192
# 찾아낸 최적의 파라미터:
# {'iterations': 1807, 'depth': 7, 'learning_rate': 0.011007442573869236, 'l2_leaf_reg': 9.960037311199006, 'colsample_bylevel': 0.8239914444396733, 'random_strength': 0.15658132267797456, 'bagging_temperature': 0.2528316917680684}
# ==================================================
#
# 전체 데이터로 최종 모델 학습을 시작합니다...
# 0:      learn: 1.0966955        total: 24.3ms   remaining: 44s
# 100:    learn: 1.0087074        total: 2.08s    remaining: 35.1s
# 200:    learn: 0.9900222        total: 4.16s    remaining: 33.2s
# 300:    learn: 0.9826629        total: 6.22s    remaining: 31.1s
# 400:    learn: 0.9784352        total: 8.31s    remaining: 29.1s
# 500:    learn: 0.9757790        total: 10.2s    remaining: 26.6s
# 600:    learn: 0.9735402        total: 12.2s    remaining: 24.4s
# 700:    learn: 0.9716318        total: 14.1s    remaining: 22.2s
# 800:    learn: 0.9699348        total: 16s      remaining: 20.1s
# 900:    learn: 0.9683506        total: 17.9s    remaining: 18s
# 1000:   learn: 0.9670776        total: 19.7s    remaining: 15.9s
# 1100:   learn: 0.9656733        total: 21.6s    remaining: 13.8s
# 1200:   learn: 0.9641391        total: 23.5s    remaining: 11.8s
# 1300:   learn: 0.9626524        total: 25.4s    remaining: 9.87s
# 1400:   learn: 0.9609790        total: 27.4s    remaining: 7.93s
# 1500:   learn: 0.9593559        total: 29.3s    remaining: 5.98s
# 1600:   learn: 0.9579010        total: 31.3s    remaining: 4.03s
# 1700:   learn: 0.9563785        total: 33.3s    remaining: 2.07s
# 1800:   learn: 0.9548236        total: 35.2s    remaining: 117ms
# 1806:   learn: 0.9547277        total: 35.3s    remaining: 0us
#
# 제출 파일을 생성합니다...
# 튜닝된 모델의 제출 파일 저장 완료: ./[3.2]_submission_catboost_tuned_20250918_141515.csv