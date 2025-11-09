# [10.3]_bias_gridsearch.py
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

TARGET, ID_COL = "support_needs", "ID"

# probs 불러오기 (예: 10.2에서 나온 test_probs, oof_probs)
oof = np.load("[10.1]_oof_probs.npy")
test = np.load("[10.1]_test_probs.npy")
y = pd.read_csv("train.csv")[TARGET].values
test_ids = pd.read_csv("test.csv")[ID_COL].values

def apply_bias(probs, bias):
    import numpy as np
    eps = 1e-15
    z = np.log(np.clip(probs, eps, 1.0)) + bias.reshape(1,-1)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

best = (-1,None)
for b0 in np.linspace(-1,1,21):
    for b1 in np.linspace(-1,1,21):
        for b2 in np.linspace(-1,1,21):
            bias = np.array([b0,b1,b2])
            f1 = f1_score(y, apply_bias(oof, bias).argmax(axis=1), average="macro")
            if f1 > best[0]:
                best = (f1, bias)

print("Best OOF Macro F1:", best[0], "bias:", best[1])

# 최적 bias 적용 후 제출 생성
test_probs_bias = apply_bias(test, best[1])
pred = test_probs_bias.argmax(axis=1)
pd.DataFrame({ID_COL: test_ids, TARGET: pred}).to_csv("[10.3]_submission_biasgrid.csv", index=False)
