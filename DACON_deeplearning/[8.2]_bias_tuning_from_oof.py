# -*- coding: utf-8 -*-
"""
[8.2]_bias_tuning_from_oof.py
- 입력: [8.1]에서 저장한 baseline OOF/Test 확률
    * [8.1]_oof_probs_baseline.npy  (shape: [n_train, K])
    * [8.1]_test_probs_baseline.npy (shape: [n_test,  K])
    * train.csv (정답 y)
    * test.csv  (ID)
- 목적: OOF 기준으로 클래스별 logit bias b_k 최적화 → Macro F1 최대화
- 출력:
    * [8.2]_bias_tuned_submission.csv
    * [8.2]_bias_tuned_report.md
    * [8.2]_best_bias.npy  (길이 K)
"""

import os, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DATA_PATH = "./"
TRAIN = os.path.join(DATA_PATH, "train.csv")
TEST  = os.path.join(DATA_PATH, "test.csv")
OOF_NPY   = os.path.join(DATA_PATH, "[8.1]_oof_probs_baseline.npy")
TEST_NPY  = os.path.join(DATA_PATH, "[8.1]_test_probs_baseline.npy")

TARGET = "support_needs"
ID_COL = "ID"

# 좌표 탐색 하이퍼파라미터(안전/경량 설정)
BIAS_RANGE = (-2.0, 2.0)   # 각 클래스 bias 검색 범위
INIT_STEP  = 0.2           # 초기 스텝(로그릿 단위)
MIN_STEP   = 0.02          # 최소 스텝(수렴 기준)
MAX_PASSES = 7             # 좌표 순환 횟수(클래스별로 반복)

def logits_with_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """ log(p) + b (각 행에 동일 bias 벡터를 broadcast) """
    eps = 1e-15
    logp = np.log(np.clip(probs, eps, 1.0))
    return logp + bias.reshape(1, -1)

def preds_from_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return logits_with_bias(probs, bias).argmax(axis=1).astype(int)

def coordinate_descent_bias(oof_probs: np.ndarray, y: np.ndarray,
                            n_classes: int,
                            bias_range=BIAS_RANGE, init_step=INIT_STEP,
                            min_step=MIN_STEP, max_passes=MAX_PASSES):
    """
    좌표 탐색으로 클래스별 bias 최적화.
    - 각 클래스 k에 대해 bias[k]를 ±step 범위에서 개선되는 지점으로 이동
    - 한 pass가 끝날 때 step을 줄이며, 개선이 없을 때까지 반복
    """
    bias = np.zeros(n_classes, dtype=float)
    step = init_step
    best_f1 = f1_score(y, preds_from_bias(oof_probs, bias), average="macro")

    for p in range(max_passes):
        improved = False
        for k in range(n_classes):
            base = bias[k]
            best_k = base
            best_k_f1 = best_f1

            # 세 방향 탐색: -step, 0, +step (범위 클램프)
            for delta in [-step, 0.0, step]:
                cand = np.clip(base + delta, bias_range[0], bias_range[1])
                bias_try = bias.copy()
                bias_try[k] = cand
                f1 = f1_score(y, preds_from_bias(oof_probs, bias_try), average="macro")
                if f1 > best_k_f1 + 1e-8:
                    best_k_f1 = f1
                    best_k = cand

            if best_k != base:
                bias[k] = best_k
                best_f1 = best_k_f1
                improved = True

        # 개선이 없으면 스텝 감소
        if not improved:
            step *= 0.5
            if step < min_step:
                break

    return bias, best_f1

def main():
    t0 = time.time()
    print("--- [8.2] Bias tuning (from OOF) 시작 ---")

    # 데이터 로드
    train = pd.read_csv(TRAIN)
    test  = pd.read_csv(TEST)
    y = train[TARGET].astype(int).values
    oof_probs  = np.load(OOF_NPY)
    test_probs = np.load(TEST_NPY)

    n_train, K = oof_probs.shape
    assert K == len(np.unique(y)), f"K(={K})와 라벨 클래스 수가 불일치할 수 있습니다."

    # 초기 성능
    base_preds = oof_probs.argmax(axis=1).astype(int)
    base_f1 = f1_score(y, base_preds, average="macro")
    print(f"Base OOF Macro F1 (argmax): {base_f1:.5f}")

    # 좌표 탐색으로 bias 최적화
    best_bias, best_f1 = coordinate_descent_bias(oof_probs, y, K)
    print(f"Best OOF Macro F1(after bias): {best_f1:.5f}")
    print(f"Best bias: {best_bias}")

    # 테스트에 적용 → 제출
    test_pred = preds_from_bias(test_probs, best_bias)
    sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: test_pred})
    sub.to_csv("[8.2]_bias_tuned_submission.csv", index=False)

    # 저장 & 리포트
    np.save("[8.2]_best_bias.npy", best_bias)
    lines = []
    lines.append("# [8.2] Bias tuning Report\n")
    lines.append(f"- Base OOF Macro F1: {base_f1:.5f}")
    lines.append(f"- Bias-tuned OOF Macro F1: **{best_f1:.5f}**")
    lines.append(f"- Best bias vector: {best_bias.tolist()}")
    lines.append("\n## 설정")
    cfg = {
        "bias_range": list(BIAS_RANGE),
        "init_step": INIT_STEP,
        "min_step": MIN_STEP,
        "max_passes": MAX_PASSES
    }
    lines.append("```json\n" + json.dumps(cfg, indent=2) + "\n```")
    with open("[8.2]_bias_tuned_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved: [8.2]_bias_tuned_submission.csv, [8.2]_best_bias.npy, [8.2]_bias_tuned_report.md")
    print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()
