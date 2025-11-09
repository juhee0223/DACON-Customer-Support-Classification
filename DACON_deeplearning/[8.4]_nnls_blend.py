# -*- coding: utf-8 -*-
"""
[8.4]_nnls_blend.py
- OOF/TEST 확률을 불러와 비음수 가중치 블렌딩
- 초기값: NNLS(비음수 최소제곱, 합=1로 정규화)
- 미세탐색: OOF Macro F1을 직접 최대화하는 그리드/좌표탐색
- (옵션) [8.2]_best_bias.npy가 있으면 bias 적용 제출도 추가 생성

입출력:
  입력(있으면 자동 사용, 최소 2개 필요):
    [8.1]_oof_probs_baseline.npy         / [8.1]_test_probs_baseline.npy
    [8.1]_oof_probs_classweights.npy     / [8.1]_test_probs_classweights.npy
    [8.3]_oof_probs_seed_ens.npy         / [8.3]_test_probs_seed_ens.npy
  출력:
    [8.4]_blend_report.md
    [8.4]_oof_probs_blend.npy
    [8.4]_test_probs_blend.npy
    [8.4]_submission_blend.csv
    [8.4]_submission_blend_bias.csv  (옵션, 8.2 bias가 있을 때)
"""

import os, time, itertools
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import f1_score

DATA_PATH = "./"
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV  = os.path.join(DATA_PATH, "test.csv")
TARGET = "support_needs"
ID_COL = "ID"

# 사용할 모델 후보(파일 prefix, 표시명)
# CANDIDATES = [
#     ("[8.1]_oof_probs_baseline.npy",     "[8.1]_test_probs_baseline.npy",     "baseline"),
#     ("[8.1]_oof_probs_classweights.npy", "[8.1]_test_probs_classweights.npy", "classweights"),
#     ("[8.3]_oof_probs_seed_ens.npy",     "[8.3]_test_probs_seed_ens.npy",     "seed_ens"),
#     ("[9.1]_oof_probs_lgbm.npy",         "[9.1]_test_probs_lgbm.npy",         "lgbm"),  # ← 추가
# ]
CANDIDATES = [
    ("[8.1]_oof_probs_baseline.npy",     "[8.1]_test_probs_baseline.npy",     "baseline"),
    ("[8.1]_oof_probs_classweights.npy", "[8.1]_test_probs_classweights.npy", "classweights"),
    ("[8.3]_oof_probs_seed_ens.npy",     "[8.3]_test_probs_seed_ens.npy",     "seed_ens"),
    ("[9.1]_oof_probs_lgbm.npy",         "[9.1]_test_probs_lgbm.npy",         "lgbm"),
    ("[9.2]_oof_probs_mlp.npy",          "[9.2]_test_probs_mlp.npy",          "mlp"),   # ← 추가
]



# 그리드 탐색 해상도(0.0~1.0), 3개 모델 기준 0.02면 조합 최대 ~1276개
GRID_STEP = 0.02

# 좌표 탐색 파라미터(미세 조정)
REFINE_INIT_STEP = 0.05
REFINE_MIN_STEP  = 0.005
REFINE_MAX_PASSES = 10

def softmax_normalize(p: np.ndarray) -> np.ndarray:
    # 확률 합이 1이 되도록 안전 재정규화
    s = p.sum(axis=1, keepdims=True)
    s = np.clip(s, 1e-15, None)
    return p / s

def blend_probs(probs_list: List[np.ndarray], w: np.ndarray) -> np.ndarray:
    # w는 (M,) 합=1, 비음수
    out = np.zeros_like(probs_list[0], dtype=float)
    for pi, wi in zip(probs_list, w):
        out += wi * pi
    return softmax_normalize(out)

def macro_f1_from_probs(y_true: np.ndarray, probs: np.ndarray) -> float:
    preds = probs.argmax(axis=1)
    return float(f1_score(y_true, preds, average="macro"))

def nnls_init(probs_list: List[np.ndarray], y: np.ndarray) -> np.ndarray:
    """
    NNLS로 초기 가중치 추정: 각 클래스에 대해 독립적으로 원-핫을 근사,
    클래스별 해를 평균내어 하나의 w로 만듦. (합=1로 정규화)
    SciPy 없이 간단/안정하게: 비음수 최소제곱을 좌표강하식으로 간략 근사.
    """
    M = len(probs_list)
    K = probs_list[0].shape[1]
    # 타겟 원-핫
    Y = np.eye(K, dtype=float)[y]  # (N, K)
    # 설계행렬 X: (N, M) - 각 모델의 "해당 클래스 확률"을 열로 구성해야 하는데
    # 클래스별로 X_k = [p1_k, p2_k, ... pM_k]
    # NNLS: min ||X_k w - Y_k||^2, w>=0
    # 간이해: 비음수 제약 + 합=1를 전제로, w를 1/M로 시작하고 좌표강하
    w = np.full(M, 1.0 / M, dtype=float)

    def proj_simplex(v: np.ndarray) -> np.ndarray:
        # 비음수 & 합=1 프로젝션 (Wang & Carreira-Perpinan 2013식)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        wproj = np.maximum(v - theta, 0)
        return wproj

    # 간단한 반복 최적화: 클래스별 gradient를 평균해 가중치 업데이트
    lr = 5.0  # 학습률(조정된 크기)
    for _ in range(100):
        grad = np.zeros_like(w)
        for k in range(K):
            Xk = np.column_stack([p[:, k] for p in probs_list])  # (N,M)
            yk = Y[:, k]  # (N,)
            # grad = 2 X^T (X w - y)
            grad += (Xk.T @ (Xk @ w - yk)) / K
        w_new = w - lr * grad
        w_new = proj_simplex(w_new)
        # 수렴 체크
        if np.linalg.norm(w_new - w) < 1e-6:
            w = w_new
            break
        w = w_new
    return w

def grid_search_f1(probs_list: List[np.ndarray], y: np.ndarray, init_w: np.ndarray) -> np.ndarray:
    M = len(probs_list)
    if M == 2:
        # w2 = 1 - w1
        best_w = init_w.copy()
        best_f1 = -1.0
        for a in np.arange(0.0, 1.0 + 1e-9, GRID_STEP):
            w = np.array([a, 1.0 - a])
            f1 = macro_f1_from_probs(y, blend_probs(probs_list, w))
            if f1 > best_f1:
                best_f1, best_w = f1, w
        return best_w
    elif M == 3:
        best_w = init_w.copy()
        best_f1 = macro_f1_from_probs(y, blend_probs(probs_list, init_w))
        for a in np.arange(0.0, 1.0 + 1e-9, GRID_STEP):
            for b in np.arange(0.0, 1.0 - a + 1e-9, GRID_STEP):
                c = 1.0 - a - b
                if c < -1e-12: 
                    continue
                w = np.array([a, b, c])
                f1 = macro_f1_from_probs(y, blend_probs(probs_list, w))
                if f1 > best_f1:
                    best_f1, best_w = f1, w
        return best_w
    else:
        # 모델이 더 많다면, 그리드는 조합폭이 급증 → NNLS 결과 사용
        return init_w

def refine_by_coordinate_ascent(probs_list: List[np.ndarray], y: np.ndarray, w0: np.ndarray) -> np.ndarray:
    w = w0.copy()
    step = REFINE_INIT_STEP
    best = macro_f1_from_probs(y, blend_probs(probs_list, w))
    for _ in range(REFINE_MAX_PASSES):
        improved = False
        for i in range(len(w)):
            for delta in (+step, -step):
                w_try = w.copy()
                w_try[i] = np.clip(w_try[i] + delta, 0.0, 1.0)
                # 나머지 축으로 정규화
                if w_try.sum() == 0:
                    continue
                w_try = w_try / w_try.sum()
                f1 = macro_f1_from_probs(y, blend_probs(probs_list, w_try))
                if f1 > best + 1e-8:
                    w, best = w_try, f1
                    improved = True
        if not improved:
            step *= 0.5
            if step < REFINE_MIN_STEP:
                break
    return w

def apply_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    eps = 1e-15
    logp = np.log(np.clip(probs, eps, 1.0))
    z = logp + bias.reshape(1, -1)
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=1, keepdims=True)

def main():
    t0 = time.time()
    print("--- [8.4] NNLS/그리드 블렌딩 시작 ---")

    # 데이터 로드
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    y = train[TARGET].astype(int).values
    test_ids = test[ID_COL].values

    # 후보 수집
    names, oof_list, test_list = [], [], []
    for oof_path, test_path, name in CANDIDATES:
        if os.path.exists(oof_path) and os.path.exists(test_path):
            oof = np.load(oof_path)
            tet = np.load(test_path)
            oof_list.append(oof)
            test_list.append(tet)
            names.append(name)
            print(f" - Loaded: {name} ({oof.shape})")
        else:
            print(f" - Skip (missing): {name}")

    if len(oof_list) < 2:
        raise RuntimeError("블렌딩할 모델 확률이 2개 미만입니다. 8.1/8.3 스크립트를 먼저 실행해 주세요.")

    # 초기/단독 성능 표시
    base_rows = []
    for nm, oof in zip(names, oof_list):
        f1 = macro_f1_from_probs(y, oof)
        base_rows.append((nm, f1))
    base_rows.sort(key=lambda x: x[1], reverse=True)
    print("\n단일 모델 OOF 성능:")
    for nm, f1 in base_rows:
        print(f" * {nm:12s}: {f1:.5f}")

    # NNLS 초기값
    w0 = nnls_init(oof_list, y)
    print(f"\nNNLS init w: {dict(zip(names, [round(x,4) for x in w0]))} | OOF F1 = {macro_f1_from_probs(y, blend_probs(oof_list, w0)):.5f}")

    # F1 그리드 탐색
    w1 = grid_search_f1(oof_list, y, w0)
    print(f"Grid best w: {dict(zip(names, [round(x,4) for x in w1]))} | OOF F1 = {macro_f1_from_probs(y, blend_probs(oof_list, w1)):.5f}")

    # 좌표탐색 미세 조정
    w2 = refine_by_coordinate_ascent(oof_list, y, w1)
    oof_blend = blend_probs(oof_list, w2)
    test_blend = blend_probs(test_list, w2)
    best_f1 = macro_f1_from_probs(y, oof_blend)

    print(f"Refine best w: {dict(zip(names, [round(x,4) for x in w2]))} | OOF F1 = {best_f1:.5f}")

    # 저장
    np.save("[8.4]_oof_probs_blend.npy",  oof_blend)
    np.save("[8.4]_test_probs_blend.npy", test_blend)

    # 제출(기본)
    sub = pd.DataFrame({ID_COL: test_ids, TARGET: test_blend.argmax(axis=1).astype(int)})
    sub.to_csv("[8.4]_submission_blend.csv", index=False)

    # (옵션) bias 적용 제출
    if os.path.exists("[8.2]_best_bias.npy"):
        bias = np.load("[8.2]_best_bias.npy")
        test_bias = apply_bias(test_blend, bias)
        subb = pd.DataFrame({ID_COL: test_ids, TARGET: test_bias.argmax(axis=1).astype(int)})
        subb.to_csv("[8.4]_submission_blend_bias.csv", index=False)
        bias_info = f"bias={np.round(bias, 4).tolist()}"
    else:
        bias_info = "None"

    # 리포트
    lines = []
    lines.append("# [8.4] NNLS + Grid Blending Report\n")
    lines.append("## 사용 모델")
    for nm in names:
        lines.append(f"- {nm}")
    lines.append("\n## 단일 모델 OOF 성능")
    for nm, f1 in base_rows:
        lines.append(f"- {nm}: {f1:.5f}")
    lines.append("\n## 최종 가중치")
    for nm, wi in zip(names, w2):
        lines.append(f"- {nm}: {wi:.6f}")
    lines.append(f"\n## 최종 OOF Macro F1: **{best_f1:.5f}**")
    lines.append("\n## 설정")
    lines.append(f"- GRID_STEP={GRID_STEP}, REFINE_INIT_STEP={REFINE_INIT_STEP}, REFINE_MIN_STEP={REFINE_MIN_STEP}, REFINE_MAX_PASSES={REFINE_MAX_PASSES}")
    lines.append(f"- Bias file: {bias_info}")
    with open("[8.4]_blend_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nSaved:")
    print(" - [8.4]_oof_probs_blend.npy")
    print(" - [8.4]_test_probs_blend.npy")
    print(" - [8.4]_submission_blend.csv")
    if os.path.exists("[8.2]_best_bias.npy"):
        print(" - [8.4]_submission_blend_bias.csv")
    print(" - [8.4]_blend_report.md")
    print(f"\n완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()
