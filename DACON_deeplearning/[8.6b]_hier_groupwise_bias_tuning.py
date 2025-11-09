# -*- coding: utf-8 -*-
"""
[8.6b]_hier_groupwise_bias_tuning.py  (fixed)
- 8.4(블렌드) 확률을 입력으로 받아 단일/조합 그룹별 bias를 학습(OOF)하고
  정규화(수축) + 백오프로 과적합 억제 → OOF가 가장 좋은 규칙으로 test 제출 생성.
- JSON 저장 시 tuple 키를 문자열로 변환하여 오류 방지.

출력:
  * [8.6b]_submission_groupbias_hier.csv
  * [8.6b]_hier_bias_map.json
  * [8.6b]_hier_groupbias_report.md
"""

import os, json, time
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score

# ====== 경로/설정 ======
DATA_PATH = "./"
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV  = os.path.join(DATA_PATH, "test.csv")

# 기본 입력 확률 (필요시 바꿔도 됨)
OOF_NPY  = os.path.join(DATA_PATH, "[8.4]_oof_probs_blend.npy")
TEST_NPY = os.path.join(DATA_PATH, "[8.4]_test_probs_blend.npy")
GLOBAL_BIAS_NPY = os.path.join(DATA_PATH, "[8.2]_best_bias.npy")  # 있으면 초기화에 사용

TARGET = "support_needs"
ID_COL = "ID"
N_CLASSES = 3

# 그룹 후보 키
BASE_KEYS = ["subscription_type", "age_group", "contract_length", "gender_subscription"]
# 단일/조합 최소 표본 수
MIN_CNT_SINGLE = 200
MIN_CNT_PAIR   = 400
# 정규화(수축) 강도 λ (클수록 글로벌로 더 수축)
REG_LAMBDA_SINGLE = 600.0
REG_LAMBDA_PAIR   = 800.0

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
    out["contract_length"] = out["contract_length"].astype(str)
    out["gender_subscription"] = out["gender"].astype(str) + "_" + out["subscription_type"].astype(str)
    for c in BASE_KEYS:
        out[c] = out[c].astype(str)
    return out

def _key_str_from_row(row: pd.Series, key_cols: List[str]) -> str:
    # "colA=foo|colB=bar" 형태로 문자열 키 생성
    return "|".join([f"{k}={str(row[k])}" for k in key_cols])

def _key_str_from_tuple(gv, key_cols: List[str]) -> str:
    # 그룹핑 결과(gv)가 tuple일 수도 있고 단일값일 수도 있으므로 안전 처리
    if isinstance(gv, tuple):
        return "|".join([f"{k}={str(v)}" for k, v in zip(key_cols, gv)])
    else:
        return f"{key_cols[0]}={str(gv)}"

def logits_with_bias_row(probs_row: np.ndarray, bias: np.ndarray) -> np.ndarray:
    eps = 1e-15
    z = np.log(np.clip(probs_row, eps, 1.0)) + bias
    m = z.max()
    e = np.exp(z - m)
    return e / e.sum()

def preds_with_bias_block(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    eps = 1e-15
    return (np.log(np.clip(probs, eps, 1.0)) + bias.reshape(1, -1)).argmax(axis=1).astype(int)

def coordinate_descent_bias(oof_probs: np.ndarray, y: np.ndarray,
                            b0: np.ndarray,
                            step=BIAS_INIT_STEP, min_step=BIAS_MIN_STEP,
                            bias_range=BIAS_RANGE, max_passes=MAX_PASSES) -> Tuple[np.ndarray, float]:
    bias = b0.astype(float).copy()
    best = f1_score(y, preds_with_bias_block(oof_probs, bias), average="macro")
    for _ in range(max_passes):
        improved = False
        for k in range(len(bias)):
            base = bias[k]
            best_k = base
            best_k_f1 = best
            for delta in (-step, 0.0, +step):
                cand = float(np.clip(base + delta, bias_range[0], bias_range[1]))
                b_try = bias.copy()
                b_try[k] = cand
                f1 = f1_score(y, preds_with_bias_block(oof_probs, b_try), average="macro")
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

def shrink_bias(b_star: np.ndarray, b_global: np.ndarray, n: int, lam: float) -> np.ndarray:
    w = float(n) / (n + lam)
    return w * b_star + (1.0 - w) * b_global

def fit_bias_map_for_key(train_g: pd.DataFrame, oof_probs: np.ndarray, y: np.ndarray,
                         key_cols: List[str], b_global: np.ndarray,
                         min_cnt: int, lam: float):
    """
    key_cols: ["subscription_type"] 또는 ["gender_subscription","contract_length"] 등
    반환: (bias_map_str, oof_adj, f1)
      - bias_map_str: { "colA=foo|colB=bar": [b0,b1,b2], ... }
    """
    # 그룹 인덱스 (문자열 키)
    keys_train_str = train_g[key_cols].agg(lambda r: _key_str_from_row(r, key_cols), axis=1)

    # 그룹별 bias 학습
    bias_map: Dict[str, List[float]] = {}
    for gv, idx in keys_train_str.groupby(keys_train_str).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        n = len(idx)
        if n < min_cnt:
            continue
        b0 = b_global.copy()
        b_star, _ = coordinate_descent_bias(oof_probs[idx], y[idx], b0=b0)
        b_reg = shrink_bias(b_star, b_global, n=n, lam=lam)
        bias_map[str(gv)] = b_reg.tolist()

    # OOF 적용
    oof_adj = np.zeros_like(oof_probs)
    for i in range(len(train_g)):
        gv = str(keys_train_str.iloc[i])
        b = np.array(bias_map.get(gv, b_global), dtype=float)
        oof_adj[i] = logits_with_bias_row(oof_probs[i], b)
    f1 = f1_score(y, oof_adj.argmax(axis=1), average="macro")
    return bias_map, oof_adj, f1

def main():
    t0 = time.time()
    print("--- [8.6b] Hierarchical Group-wise Bias Tuning 시작 ---")

    # 데이터/확률 로드
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    y = train[TARGET].astype(int).values
    oof_probs  = np.load(OOF_NPY)
    test_probs = np.load(TEST_NPY)
    assert oof_probs.shape[0] == len(y), "OOF와 train 길이 불일치"
    K = oof_probs.shape[1]

    # 그룹 컬럼 구성
    train_g = build_group_cols(train)
    test_g  = build_group_cols(test)

    # 글로벌 bias 초기값
    if os.path.exists(GLOBAL_BIAS_NPY):
        b_global = np.load(GLOBAL_BIAS_NPY)
        if b_global.shape[0] != K:
            b_global = np.zeros(K, dtype=float)
    else:
        b_global = np.zeros(K, dtype=float)

    base_f1 = f1_score(y, oof_probs.argmax(axis=1), average="macro")
    print(f"Base OOF Macro F1 (no bias): {base_f1:.5f}")

    # 후보 키 세트(단일 + 조합 2개)
    key_sets: List[List[str]] = [[k] for k in BASE_KEYS]
    key_sets += [list(cmb) for cmb in combinations(BASE_KEYS, 2)]

    best = {"keys": None, "f1": base_f1, "bias_map": {}, "oof_adj": oof_probs}

    report = []
    report.append("# [8.6b] Hierarchical Group-wise Bias Tuning Report\n")
    report.append(f"- Base OOF Macro F1 (no bias): {base_f1:.5f}\n")
    report.append("## Candidates\n")

    # 각 후보 키셋 학습/평가
    for ks in key_sets:
        if len(ks) == 1:
            min_cnt = MIN_CNT_SINGLE
            lam = REG_LAMBDA_SINGLE
        else:
            min_cnt = MIN_CNT_PAIR
            lam = REG_LAMBDA_PAIR

        bm, oof_adj, f1 = fit_bias_map_for_key(
            train_g, oof_probs, y, ks, b_global, min_cnt=min_cnt, lam=lam
        )
        gain = f1 - base_f1
        report.append(f"- {ks}: OOF F1={f1:.5f} (gain {gain:+.5f}, groups={len(bm)})")
        print(f"[{'+'.join(ks)}] OOF Macro F1: {f1:.5f} (gain {gain:+.5f})  groups={len(bm)}")

        if f1 > best["f1"] + 1e-9:
            best.update({"keys": ks, "f1": f1, "bias_map": bm, "oof_adj": oof_adj})

    # ===== test에 적용 (백오프: 선택된 조합 → 단일 → 글로벌) =====
    # 단일 백오프용 단일키 bias도 준비
    single_maps: Dict[str, Dict[str, List[float]]] = {}
    for k in BASE_KEYS:
        bm_single, _, _ = fit_bias_map_for_key(
            train_g, oof_probs, y, [k], b_global, min_cnt=MIN_CNT_SINGLE, lam=REG_LAMBDA_SINGLE
        )
        single_maps[k] = bm_single  # 문자열 키 사용

    def row_bias_for_test(i: int) -> np.ndarray:
        # 1) 선택된 조합 키
        if best["keys"] is not None and len(best["keys"]) >= 1:
            ks = best["keys"]
            gv_comb = _key_str_from_row(test_g.iloc[i], ks)
            if gv_comb in best["bias_map"]:
                return np.array(best["bias_map"][gv_comb], dtype=float)
        # 2) 단일 백오프 순차 탐색
        for k in BASE_KEYS:
            gv1 = _key_str_from_row(test_g.iloc[i], [k])
            bm1 = single_maps.get(k, {})
            if gv1 in bm1:
                return np.array(bm1[gv1], dtype=float)
        # 3) 글로벌
        return b_global.astype(float)

    test_adj = np.zeros_like(test_probs)
    for i in range(len(test)):
        b = row_bias_for_test(i)
        test_adj[i] = logits_with_bias_row(test_probs[i], b)

    # 저장/출력
    pred = test_adj.argmax(axis=1).astype(int)
    sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: pred})
    sub.to_csv("[8.6b]_submission_groupbias_hier.csv", index=False)

    with open("[8.6b]_hier_bias_map.json", "w", encoding="utf-8") as f:
        json.dump({"chosen_keys": best["keys"], "bias_map": best["bias_map"]},
                  f, ensure_ascii=False, indent=2)

    report.append(f"\n## Selected keys: {best['keys']}")
    report.append(f"- Final OOF Macro F1: **{best['f1']:.5f}**")
    with open("[8.6b]_hier_groupbias_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\nSaved: [8.6b]_submission_groupbias_hier.csv, [8.6b]_hier_bias_map.json, [8.6b]_hier_groupbias_report.md")
    print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()
