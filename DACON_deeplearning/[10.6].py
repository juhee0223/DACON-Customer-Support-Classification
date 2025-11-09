# -*- coding: utf-8 -*-
"""
[10.6]_bias_gridsearch_expanded.py
- 3-class 로짓 bias(b0,b1,b2) 를 OOF Macro F1 기준으로 최적화 (그리드 + 로컬 탐색)
- 입력: [10.5]_oof_probs.npy / [10.5]_test_probs.npy (없으면 10.2 경로로 자동 폴백)
- 출력:
  * [10.6]_best_bias.npy
  * [10.6]_submission_biasgrid_expanded.csv        (best bias 적용)
  * [10.6]_biasgrid_report.md                      (요약)
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TARGET, ID_COL = "support_needs", "ID"

# ========= 파일 경로 (10.5 우선, 없으면 10.2 폴백) =========
CANDIDATES = [
    ("[10.5]_oof_probs.npy",  "[10.5]_test_probs.npy",  "[10.6]_from_10_5"),
    ("[10.2]_oof_probs.npy",  "[10.2]_test_probs.npy",  "[10.6]_from_10_2"),
]

def first_existing_paths():
    for oof_p, tst_p, tag in CANDIDATES:
        if os.path.exists(oof_p) and os.path.exists(tst_p):
            return oof_p, tst_p, tag
    raise FileNotFoundError("OOF/TEST 확률 파일을 찾을 수 없습니다. 10.5 또는 10.2를 먼저 실행해 주세요.")

def apply_bias_probs(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """logit(biased) = log(p) + bias; softmax 재정규화"""
    eps = 1e-15
    z = np.log(np.clip(probs, eps, 1.0)) + bias.reshape(1, -1)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def eval_f1(y_true: np.ndarray, probs: np.ndarray, bias: np.ndarray) -> float:
    pred = apply_bias_probs(probs, bias).argmax(axis=1)
    return f1_score(y_true, pred, average="macro")

def coarse_grid_search(y, oof_probs,
                       lo=-2.0, hi=2.0, step=0.1, top_k=25):
    """조밀 그리드(-2~2, 0.1 간격 = 41^3=68,921 평가) 후 상위 top_k 반환"""
    best_list = []  # (f1, [b0,b1,b2])
    grid = np.arange(lo, hi + 1e-9, step)
    for b0 in grid:
        for b1 in grid:
            for b2 in grid:
                bias = np.array([b0, b1, b2], dtype=float)
                f1 = eval_f1(y, oof_probs, bias)
                # 상위 top_k 유지
                if len(best_list) < top_k:
                    best_list.append((f1, bias))
                    best_list.sort(key=lambda x: x[0], reverse=True)
                else:
                    if f1 > best_list[-1][0]:
                        best_list[-1] = (f1, bias)
                        best_list.sort(key=lambda x: x[0], reverse=True)
    return best_list  # 내림차순

def local_refine(y, oof_probs, bias_init,
                 start_step=0.1, min_step=0.01, max_iter=50, bound=(-2.5, 2.5)):
    """좌표 상승 기반 로컬 미세탐색: step 줄여가며 각 좌표를 -/0/+로 움직여 F1 향상 탐색"""
    bias = bias_init.astype(float).copy()
    best = eval_f1(y, oof_probs, bias)
    step = start_step
    it = 0
    while step >= min_step and it < max_iter:
        it += 1
        improved = False
        for k in range(3):
            base = bias[k]
            cand_vals = [np.clip(base + d, bound[0], bound[1]) for d in (-step, 0.0, +step)]
            best_k = base
            best_k_f1 = best
            for cand in cand_vals:
                b_try = bias.copy()
                b_try[k] = cand
                f1 = eval_f1(y, oof_probs, b_try)
                if f1 > best_k_f1 + 1e-12:
                    best_k_f1 = f1
                    best_k = cand
            if best_k != base:
                bias[k] = best_k
                best = best_k_f1
                improved = True
        if not improved:
            step *= 0.5
    return bias, best

def main():
    oof_path, tst_path, tag = first_existing_paths()
    print(f"--- [10.6] Bias GridSearch Expanded 시작 ({tag}) ---")
    oof = np.load(oof_path)
    test = np.load(tst_path)
    train = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    y = train[TARGET].astype(int).values
    test_ids = test_df[ID_COL].values

    # 1) Coarse grid (조밀) → 상위 후보 선정
    top = coarse_grid_search(y, oof, lo=-2.0, hi=2.0, step=0.1, top_k=25)
    print(f"Coarse Top-5:")
    for i, (f1, b) in enumerate(top[:5], 1):
        print(f"  {i}. F1={f1:.6f}, bias={np.round(b,3)}")

    # 2) 각 후보 로컬 미세탐색 → 최종 best 선정
    best_f1 = -1.0
    best_bias = None
    for rank, (f1, b) in enumerate(top, 1):
        b_ref, f1_ref = local_refine(y, oof, b, start_step=0.1, min_step=0.01, max_iter=60, bound=(-3, 3))
        if f1_ref > best_f1:
            best_f1 = f1_ref
            best_bias = b_ref

    print(f"\nBest OOF Macro F1: {best_f1:.6f}")
    print(f"Best bias: {np.round(best_bias, 6)}")

    # 3) 최적 bias로 test 적용 & 저장
    test_probs_bias = apply_bias_probs(test, best_bias)
    pred = test_probs_bias.argmax(axis=1).astype(int)
    out_csv = "[10.6]_submission_biasgrid_expanded.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET: pred}).to_csv(out_csv, index=False)

    # 저장물
    np.save("[10.6]_best_bias.npy", best_bias)

    # 리포트
    lines = []
    lines.append("# [10.6] Bias GridSearch Expanded Report\n")
    lines.append(f"- Source: {tag}")
    lines.append(f"- Best OOF Macro F1: **{best_f1:.6f}**")
    lines.append(f"- Best bias: `{np.round(best_bias, 6).tolist()}`")
    lines.append(f"- Submission: {out_csv}")
    with open("[10.6]_biasgrid_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved: {out_csv}, [10.6]_best_bias.npy, [10.6]_biasgrid_report.md")
    print("완료!")

if __name__ == "__main__":
    main()
