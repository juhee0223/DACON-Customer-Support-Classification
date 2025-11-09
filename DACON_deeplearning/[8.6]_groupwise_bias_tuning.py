# -*- coding: utf-8 -*-
"""
[8.6]_groupwise_bias_tuning.py  (fixed)
- OOF/Test 확률에 대해 '그룹별' logit bias를 최적화하여 Macro F1 향상 시도
- 기본 입력: [8.4]_oof_probs_blend.npy, [8.4]_test_probs_blend.npy
- 그룹 키: subscription_type, age_group(나이 bin), gender_subscription
- 출력:
  * [8.6]_submission_groupbias.csv
  * [8.6]_group_bias_map.json
  * [8.6]_groupbias_report.md
"""

import os, json, time
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import f1_score

# ====== 경로/설정 ======
DATA_PATH = "./"
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV  = os.path.join(DATA_PATH, "test.csv")
# 기본은 8.4 결과 사용 (필요시 바꿔도 됨)
OOF_NPY  = os.path.join(DATA_PATH, "[8.4]_oof_probs_blend.npy")
TEST_NPY = os.path.join(DATA_PATH, "[8.4]_test_probs_blend.npy")
# 8.2 글로벌 bias가 있으면 초기화에 사용(선택)
GLOBAL_BIAS_NPY = os.path.join(DATA_PATH, "[8.2]_best_bias.npy")

TARGET = "support_needs"
ID_COL = "ID"
N_CLASSES = 3

# 그룹 최소 표본 수 (작으면 글로벌 bias로 백오프)
MIN_GROUP_SAMPLES = 200

# 좌표 탐색 하이퍼파라미터
BIAS_INIT_STEP = 0.2
BIAS_MIN_STEP  = 0.02
BIAS_RANGE     = (-2.0, 2.0)
MAX_PASSES     = 7

# ====== 유틸 ======
def make_age_group(age: pd.Series) -> pd.Categorical:
    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ['10s','20s','30s','40s','50s','60+']
    return pd.cut(age, bins=bins, labels=labels, right=False)

def build_group_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_group"] = make_age_group(out["age"])
    out["gender_subscription"] = out["gender"].astype(str) + "_" + out["subscription_type"].astype(str)
    for c in ["subscription_type","age_group","gender_subscription"]:
        out[c] = out[c].astype(str)
    return out

def logits_with_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    eps = 1e-15
    logp = np.log(np.clip(probs, eps, 1.0))
    return logp + bias.reshape(1, -1)

def preds_with_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return logits_with_bias(probs, bias).argmax(axis=1).astype(int)

def coordinate_descent_bias(oof_probs: np.ndarray, y: np.ndarray,
                            b0: np.ndarray,
                            step=BIAS_INIT_STEP, min_step=BIAS_MIN_STEP,
                            bias_range=BIAS_RANGE, max_passes=MAX_PASSES) -> Tuple[np.ndarray, float]:
    bias = b0.astype(float).copy()
    best = f1_score(y, preds_with_bias(oof_probs, bias), average="macro")
    for _ in range(max_passes):
        improved = False
        for k in range(len(bias)):
            base = bias[k]
            best_k = base
            best_k_f1 = best
            for delta in (-step, 0.0, +step):
                cand = np.clip(base + delta, bias_range[0], bias_range[1])
                b_try = bias.copy()
                b_try[k] = cand
                f1 = f1_score(y, preds_with_bias(oof_probs, b_try), average="macro")
                if f1 > best_k_f1 + 1e-8:
                    best_k_f1 = f1
                    best_k = cand
            if best_k != base:
                bias[k] = best_k
                best = best_k_f1
                improved = True
        if not improved:
            step *= 0.5
            if step < min_step:
                break
    return bias, best

def main():
    t0 = time.time()
    print("--- [8.6] Group-wise Bias Tuning 시작 ---")

    # 데이터/확률 로드
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    y = train[TARGET].astype(int).values
    oof_probs  = np.load(OOF_NPY)
    test_probs = np.load(TEST_NPY)
    assert oof_probs.shape[0] == len(y), "OOF와 train 길이 불일치"
    K = oof_probs.shape[1]

    # 그룹 키 생성
    train_g = build_group_cols(train)
    test_g  = build_group_cols(test)

    # 글로벌 bias 초기값
    if os.path.exists(GLOBAL_BIAS_NPY):
        b_global = np.load(GLOBAL_BIAS_NPY)
        if b_global.shape[0] != K:
            b_global = np.zeros(K, dtype=float)
    else:
        b_global = np.zeros(K, dtype=float)

    # 베이스 OOF/MacroF1
    base_pred = preds_with_bias(oof_probs, np.zeros(K))
    base_f1 = f1_score(y, base_pred, average="macro")
    print(f"Base OOF Macro F1 (no group bias): {base_f1:.5f}")

    group_cols = ["subscription_type", "age_group", "gender_subscription"]

    report_lines = []
    report_lines.append("# [8.6] Group-wise Bias Tuning Report\n")
    report_lines.append(f"- Base OOF Macro F1 (no bias): {base_f1:.5f}\n")

    # 최종 선택용 베스트 기록
    best_oof_f1 = base_f1
    best_gcol = None
    best_test_probs = test_probs.copy()
    bias_map: Dict[str, Dict[str, list]] = {}

    # ===== 그룹 컬럼 루프 =====
    for gcol in group_cols:
        print(f"\n=== 그룹 컬럼: {gcol} ===")
        bias_map[gcol] = {}

        # 그룹별 bias 학습 (표본수 부족 시 글로벌 백오프)
        for gv, idx in train_g.groupby(gcol).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            if len(idx) < MIN_GROUP_SAMPLES:
                bias_map[gcol][gv] = b_global.tolist()
                continue
            b0 = b_global.copy()
            b_star, _ = coordinate_descent_bias(oof_probs[idx], y[idx], b0=b0)
            bias_map[gcol][gv] = b_star.tolist()

        # === train OOF에 동일 규칙 적용해서 OOF F1 측정 ===
        oof_adj = np.zeros_like(oof_probs)
        for i in range(len(train)):
            gv = str(train_g.iloc[i][gcol])
            b = np.array(bias_map[gcol].get(gv, b_global), dtype=float)
            eps = 1e-15
            logp = np.log(np.clip(oof_probs[i], eps, 1.0))
            z = logp + b
            m = z.max()
            e = np.exp(z - m)
            oof_adj[i] = e / e.sum()

        f1_after = f1_score(y, oof_adj.argmax(axis=1), average="macro")
        gain = f1_after - base_f1
        print(f"[{gcol}] OOF Macro F1: {f1_after:.5f} (gain {gain:+.5f})")
        report_lines.append(f"- {gcol}: OOF Macro F1={f1_after:.5f} (gain {gain:+.5f})")

        # === test에도 적용해 제출 후보 확률 생성 ===
        probs_adj_test = np.zeros_like(test_probs)
        for i in range(len(test)):
            gv = str(test_g.iloc[i][gcol])
            b = np.array(bias_map[gcol].get(gv, b_global), dtype=float)
            eps = 1e-15
            logp = np.log(np.clip(test_probs[i], eps, 1.0))
            z = logp + b
            m = z.max()
            e = np.exp(z - m)
            probs_adj_test[i] = e / e.sum()

        # 가장 높은 OOF F1을 주는 그룹 규칙 채택
        if f1_after > best_oof_f1 + 1e-9:
            best_oof_f1 = f1_after
            best_gcol = gcol
            best_test_probs = probs_adj_test.copy()

    # 채택 결과 요약
    if best_gcol is None:
        best_gcol = "none (no gain over base)"
        final_preds = test_probs.argmax(axis=1).astype(int)
    else:
        final_preds = best_test_probs.argmax(axis=1).astype(int)

    report_lines.append(f"\n## 채택된 그룹 규칙: {best_gcol}")
    report_lines.append(f"- 최종 OOF Macro F1: **{best_oof_f1:.5f}**")

    # 제출/저장
    sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: final_preds})
    sub.to_csv("[8.6]_submission_groupbias.csv", index=False)

    with open("[8.6]_group_bias_map.json", "w", encoding="utf-8") as f:
        json.dump(bias_map, f, ensure_ascii=False, indent=2)
    with open("[8.6]_groupbias_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nSaved: [8.6]_submission_groupbias.csv, [8.6]_group_bias_map.json, [8.6]_groupbias_report.md")
    print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()
