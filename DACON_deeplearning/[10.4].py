# [10.4]_catboost_seed_ensemble.py
import os, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

TARGET, ID_COL = "support_needs", "ID"
N_CLASSES = 3

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
sub   = pd.read_csv("sample_submission.csv")

X = train.drop([ID_COL, TARGET], axis=1).copy()
y = train[TARGET].astype(int).copy()
Xtest = test.drop([ID_COL], axis=1).copy()
test_ids = test[ID_COL].values

# CatBoost categorical cols
cat_cols = []
for c in ["gender","subscription_type","contract_length"]:
    if c in X.columns:
        X[c] = X[c].astype(str)
        Xtest[c] = Xtest[c].astype(str)
        cat_cols.append(c)

# Optuna best 파라미터
BEST_PARAMS = dict(
    iterations=1460,
    depth=8,
    learning_rate=0.01011375157729347,
    l2_leaf_reg=2.9134559843727033,
    colsample_bylevel=0.8914955767108875,
    random_strength=0.10543719325919827,
    bagging_temperature=0.1804730161382898,
    auto_class_weights="Balanced",
    verbose=0,
)

SEEDS = [42,52,62,72,82,92,102]  # seed 확장
N_SPLITS = 5

oof_total = np.zeros((len(y), N_CLASSES))
test_total = np.zeros((len(Xtest), N_CLASSES))

for seed in SEEDS:
    print(f"\n[Seed {seed}]")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros_like(oof_total)
    tet = np.zeros_like(test_total)
    scores = []
    for fold,(tr,va) in enumerate(skf.split(X,y), start=1):
        model = CatBoostClassifier(**BEST_PARAMS, random_seed=seed, cat_features=cat_cols)
        model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], early_stopping_rounds=100, verbose=0)
        pv = model.predict_proba(X.iloc[va])
        oof[va] = pv
        tet += model.predict_proba(Xtest)/N_SPLITS
        f1 = f1_score(y.iloc[va], pv.argmax(1), average="macro")
        scores.append(f1)
        print(f"  Fold {fold}: {f1:.5f}")
    mean = np.mean(scores)
    print(f"[Seed {seed}] CV Macro F1 = {mean:.5f}")
    oof_total += oof / len(SEEDS)
    test_total += tet / len(SEEDS)

# 최종 OOF 성능
f1_final = f1_score(y, oof_total.argmax(1), average="macro")
print("\nEnsemble OOF Macro F1:", f1_final)

# 저장
np.save("[10.4]_oof_probs.npy", oof_total)
np.save("[10.4]_test_probs.npy", test_total)
pd.DataFrame({ID_COL: test_ids, TARGET: test_total.argmax(1)}).to_csv("[10.4]_submission_seedens.csv", index=False)
