# -*- coding: utf-8 -*-
"""
[9.7]_catboost_local_autoensemble.py
- 설치 없이 현재 환경에서 '경량 AutoML' 효과를 내는 파이프라인
- V2 FE + OOF Target Encoding(누수방지) + CatBoost 다변화(Seed, Line-Search, OVA/MC) + 자동 블렌딩
- 산출물:
  * [9.7]_oof_probs_<tag>.npy / [9.7]_test_probs_<tag>.npy (각 후보)
  * [9.7]_cv_scores_candidates.csv
  * [9.7]_submission_<tag>.csv (각 후보, 선택)
  * [9.7]_submission_blend.csv / [9.7]_submission_blend_bias.csv (최종)
  * [9.7]_blend_report.md
"""

import os, time, json, itertools, math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

# ============ 경로/데이터 설정 ============
DATA_PATH = "./"
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV  = os.path.join(DATA_PATH, "test.csv")
SUB_CSV   = os.path.join(DATA_PATH, "sample_submission.csv")
BIAS_NPY  = os.path.join(DATA_PATH, "[8.2]_best_bias.npy")  # 있으면 bias 제출도 생성

TARGET = "support_needs"
ID_COL = "ID"
N_CLASSES = 3

# ============ 실행 옵션(리소스 조절 포인트) ============
N_SPLITS = 5
CV_SEED  = 42

# 후보 모델 구성 (총 8개 내외 권장: 과도하면 오래 걸림)
SEEDS_MC = [42, 52]   # MultiClass용 seed
SEEDS_OVA= [42, 52]   # OneVsAll용 seed
# Best 파라미터(네가 준 Optuna 결과) - 여기를 베이스로 변형
BEST_BASE = dict(
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
# 라인서치(미세 주변) - MultiClass에만 2개 추가 (총 MC=base+2, OVA=base -> seed 2개씩 => 8개)
LINESEARCH_LR_MULTS = [0.8, 1.2]   # 0.8×lr, 1.2×lr
LINESEARCH_IT_MULTS = [1.0]        # iterations는 고정 (원하면 [0.9,1.0,1.1])

# 블렌딩 그리드 간격(0.05이면 충분히 촘촘)
BLEND_STEP = 0.05

# ============ V2 FE + TE 유틸 ============
QBINS     = 20
WINSOR_LO = 0.02
WINSOR_HI = 0.98
TE_COLS   = ["subscription_type","gender_subscription"]
TE_SMOOTH = 10.0

def base_add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6
    df["tenure_per_contract"] = df["tenure"] / (df["contract_length"] + eps)
    df["delay_ratio"]         = df["payment_interval"] / (df["contract_length"] + eps)
    df["idle_ratio"]          = df["after_interaction"] / (df["tenure"] + eps)
    df["late_flag"]           = (df["payment_interval"] > 0).astype(int)
    if 'frequent' in df.columns:
        df['total_usage_score'] = df['tenure'] * df['frequent']
    # age bin
    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ['10s', '20s', '30s', '40s', '50s', '60+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    df['risk_score'] = df['payment_interval'] + df['after_interaction']
    return df

def prepare_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["contract_length"] = out["contract_length"].astype(str)
    out["age_group"] = out["age_group"].astype(str)
    out["gender_subscription"] = out["gender"].astype(str) + "_" + out["subscription_type"].astype(str)
    return out

def _winsorize_series(s: pd.Series, lower=0.01, upper=0.99):
    ql = s.quantile(lower); qu = s.quantile(upper)
    return s.clip(lower=ql, upper=qu)

def winsor_block(df: pd.DataFrame, lo=WINSOR_LO, hi=WINSOR_HI):
    out = df.copy()
    for c in ["payment_interval", "after_interaction", "tenure"]:
        if c in out.columns:
            out[c] = _winsorize_series(out[c], lo, hi)
    return out

def _quantile_rank_bins(s: pd.Series, bins: int):
    return np.minimum((s.rank(method="average", pct=True) * bins).astype(int), bins-1)

def build_interactions_and_bins(df: pd.DataFrame, qbins: int):
    out = df.copy()
    if set(["tenure", "payment_interval"]).issubset(out.columns):
        out["tenure_x_payint"] = out["tenure"] * out["payment_interval"]
    if set(["tenure", "after_interaction"]).issubset(out.columns):
        out["tenure_x_after"] = out["tenure"] * out["after_interaction"]
    if "frequent" in out.columns:
        out["freq_x_after"] = out["frequent"] * out["after_interaction"]
    if set(["delay_ratio", "risk_score"]).issubset(out.columns):
        out["delay_x_risk"] = out["delay_ratio"] * out["risk_score"]
    if "risk_score" in out.columns:
        out["risk_log1p"] = np.log1p(out["risk_score"].clip(lower=0))
    if "idle_ratio" in out.columns:
        out["idle_sq"] = out["idle_ratio"] ** 2
    for c in ["tenure", "payment_interval", "after_interaction", "tenure_per_contract"]:
        if c in out.columns:
            out[f"{c}_q{qbins}"] = _quantile_rank_bins(out[c], qbins)
    return out

def prepare_cat_features(df: pd.DataFrame) -> List[str]:
    return [c for c in ["gender", "subscription_type", "age_group", "contract_length", "gender_subscription"]
            if c in df.columns and df[c].dtype == object]

def cv_target_encode_multiclass(train_df: pd.DataFrame, y: pd.Series, te_cols, n_splits=N_SPLITS, seed=CV_SEED, smoothing=TE_SMOOTH):
    K = N_CLASSES
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = train_df.copy()
    prior = np.bincount(y, minlength=K) / len(y)
    for col in te_cols:
        for k in range(K):
            out[f"TE_{col}_c{k}"] = np.nan
    for tr, va in skf.split(train_df, y):
        Xtr, ytr = train_df.iloc[tr], y.iloc[tr]
        Xva = train_df.iloc[va]
        fmap = {}
        for col in te_cols:
            d = {}
            grp = pd.concat([Xtr[col].astype(str), ytr], axis=1).groupby(col)
            for cat, g in grp:
                cnt = np.bincount(g[TARGET].to_numpy(), minlength=K)
                prob = (cnt + smoothing / K) / (cnt.sum() + smoothing)
                d[str(cat)] = prob
            fmap[col] = d
        for col in te_cols:
            arr = np.vstack([fmap[col].get(str(v), prior) for v in Xva[col].astype(str).values])
            for k in range(K):
                out.loc[Xva.index, f"TE_{col}_c{k}"] = arr[:, k]
    for col in te_cols:
        for k in range(K):
            out[f"TE_{col}_c{k}"] = out[f"TE_{col}_c{k}"].fillna(prior[k])
    # full map
    full_map: Dict[str, Dict[str, np.ndarray]] = {col: {} for col in te_cols}
    for col in te_cols:
        grp = pd.concat([train_df[col].astype(str), y], axis=1).groupby(col)
        for cat, g in grp:
            cnt = np.bincount(g[TARGET].to_numpy(), minlength=K)
            prob = (cnt + smoothing / K) / (cnt.sum() + smoothing)
            full_map[col][str(cat)] = prob
    return out, full_map, prior

def apply_te_from_map(df: pd.DataFrame, te_cols, full_map, prior):
    out = df.copy()
    for col in te_cols:
        arr = np.vstack([full_map[col].get(str(v), prior) for v in out[col].astype(str).values])
        for k in range(len(prior)):
            out[f"TE_{col}_c{k}"] = arr[:, k]
    return out

def apply_bias_to_probs(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    eps = 1e-15
    logp = np.log(np.clip(probs, eps, 1.0))
    z = logp + bias.reshape(1, -1)
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=1, keepdims=True)

# ============ 학습/평가 유틸 ============
def train_cv_one(X: pd.DataFrame, y: pd.Series, Xtest: pd.DataFrame, cat_features: List[str], params: dict, tag: str, n_splits=N_SPLITS, seed=CV_SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), N_CLASSES), dtype=float)
    tet = np.zeros((len(Xtest), N_CLASSES), dtype=float)
    scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        model = CatBoostClassifier(**params, cat_features=cat_features)
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[va], y.iloc[va])],
            early_stopping_rounds=150,
            verbose=0
        )
        pv = model.predict_proba(X.iloc[va])
        oof[va] = pv
        tet += model.predict_proba(Xtest) / n_splits
        f1 = f1_score(y.iloc[va], pv.argmax(axis=1), average="macro")
        scores.append(f1)
        print(f"[{tag}] Fold {fold} Macro F1={f1:.5f}  it≈{model.tree_count_}")
    mean_f1 = float(np.mean(scores)); std_f1 = float(np.std(scores, ddof=1))
    print(f"[{tag}] CV Macro F1: {mean_f1:.5f} ± {std_f1:.5f}")
    return oof, tet, mean_f1

def simplex_weight_grid(n_models: int, step: float):
    # n_models 차원의 단순체 그리드 (합=1, 각 w>=0)
    ints = int(1/step)
    if n_models == 2:
        for i in range(ints+1):
            yield np.array([i*step, 1 - i*step])
    elif n_models == 3:
        for i in range(ints+1):
            for j in range(ints+1 - i):
                k = ints - i - j
                yield np.array([i, j, k])*step
    else:
        # n>=4는 반복폭 커짐 → random sampling + 정규화 (가볍게 400 샘플)
        rng = np.random.default_rng(42)
        for _ in range(400):
            v = rng.random(n_models)
            v = v / v.sum()
            yield v

def blend_search(oofs: List[np.ndarray], y: np.ndarray, labels: List[str], step=0.05):
    # Macro F1 기준 최적 가중치 탐색
    m = len(oofs)
    P = np.stack(oofs, axis=0)  # (m, n, K)
    best = (-1.0, None)
    w_best = None
    for w in simplex_weight_grid(m, step):
        mix = np.tensordot(w, P, axes=(0,0))  # (n,K)
        pred = mix.argmax(axis=1)
        f1 = f1_score(y, pred, average="macro")
        if f1 > best[0]:
            best = (f1, w)
            w_best = w
    # 소폭 개선 위해 top-weights 근방 미세탐색
    # (여기선 생략하거나 step 더 작게 다시 한 번 돌릴 수도 있음)
    return best[0], w_best

# ============ 메인 ============
def main():
    t0 = time.time()
    print("--- [9.7] CatBoost Local Auto-Ensemble 시작 ---")

    # 데이터 로드
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    sub   = pd.read_csv(SUB_CSV)
    y = train[TARGET].astype(int)
    test_ids = test[ID_COL].values

    # 공통 FE + TE
    X_raw = train.drop([ID_COL, TARGET], axis=1).copy()
    Xte_raw = test.drop([ID_COL], axis=1).copy()

    X = base_add_features(X_raw);   X = prepare_types(X);   X = winsor_block(X);   X = build_interactions_and_bins(X, QBINS)
    Xte= base_add_features(Xte_raw);Xte= prepare_types(Xte);Xte= winsor_block(Xte);Xte= build_interactions_and_bins(Xte, QBINS)

    te_cols = [c for c in TE_COLS if c in X.columns]
    X_te, full_map, prior = cv_target_encode_multiclass(X, y, te_cols, n_splits=N_SPLITS, seed=CV_SEED, smoothing=TE_SMOOTH)
    Xtest = apply_te_from_map(Xte, te_cols, full_map, prior)
    cat_features = prepare_cat_features(X_te)

    # 후보 파라미터 생성
    candidates: List[Tuple[str, dict]] = []

    # (A) MultiClass: base + line-search 두 개 (seed별 2개) → 태그: mc_s{seed}_base, mc_s{seed}_lr08, mc_s{seed}_lr12
    for sd in SEEDS_MC:
        p = BEST_BASE.copy()
        p["random_seed"] = sd
        p["loss_function"] = "MultiClass"
        candidates.append((f"mc_s{sd}_base", p.copy()))
        for lm in LINESEARCH_LR_MULTS:
            q = p.copy()
            q["learning_rate"] = max(1e-4, p["learning_rate"] * lm)
            # iterations 라인서치가 켜져 있으면 여기도 반영
            if len(LINESEARCH_IT_MULTS) == 1 and LINESEARCH_IT_MULTS[0] == 1.0:
                pass
            else:
                q["iterations"] = int(max(200, round(p["iterations"] * LINESEARCH_IT_MULTS[0])))
            candidates.append((f"mc_s{sd}_lr{str(lm).replace('.','')}", q))

    # (B) OneVsAll: base만 (seed별) → 태그: ova_s{seed}
    for sd in SEEDS_OVA:
        q = BEST_BASE.copy()
        q["random_seed"] = sd
        q["loss_function"] = "MultiClassOneVsAll"
        candidates.append((f"ova_s{sd}", q))

    # 학습 루프
    oof_list, test_list, labels, rows = [], [], [], []
    for tag, params in candidates:
        print(f"\n=== Train Candidate: {tag} ===")
        oof, tet, cvf1 = train_cv_one(X_te, y, Xtest, cat_features, params, tag=tag, n_splits=N_SPLITS, seed=CV_SEED)
        np.save(f"[9.7]_oof_probs_{tag}.npy",  oof)
        np.save(f"[9.7]_test_probs_{tag}.npy", tet)
        # 개별 제출(원하면 제출)
        pred = tet.argmax(axis=1).astype(int)
        pd.DataFrame({ID_COL: test_ids, TARGET: pred}).to_csv(f"[9.7]_submission_{tag}.csv", index=False)

        oof_list.append(oof); test_list.append(tet); labels.append(tag)
        rows.append({"tag": tag, "cv_macro_f1": cvf1})

    pd.DataFrame(rows).to_csv("[9.7]_cv_scores_candidates.csv", index=False)
    print("\n=== 후보 CV 요약 ===")
    for r in rows:
        print(f"{r['tag']:>16s} : {r['cv_macro_f1']:.5f}")

    # 블렌딩 (단계 1: 전체 후보로, 안되면 상위 4개만 재시도)
    print("\n--- 블렌딩 탐색 (Macro F1 기준) ---")
    best_f1, best_w = blend_search(oof_list, y.values, labels, step=BLEND_STEP)
    if best_w is None:
        # 방어적: 상위 4개로 축소해서 재시도
        top_idx = np.argsort([-r["cv_macro_f1"] for r in rows])[:4]
        oof_s = [oof_list[i] for i in top_idx]
        test_s= [test_list[i] for i in top_idx]
        labels_s=[labels[i] for i in top_idx]
        print("초기 탐색 실패 → 상위 4개로 재탐색")
        best_f1, best_w = blend_search(oof_s, y.values, labels_s, step=BLEND_STEP)
        chosen = labels_s
        oofs   = oof_s
        tests  = test_s
    else:
        chosen = labels
        oofs   = oof_list
        tests  = test_list

    # 최종 블렌드 확률 산출
    W = np.array(best_w, dtype=float)
    blend_oof  = np.tensordot(W, np.stack(oofs, axis=0), axes=(0,0))   # (n,K)
    blend_test = np.tensordot(W, np.stack(tests, axis=0), axes=(0,0))  # (m,K)
    blend_pred = blend_oof.argmax(axis=1)
    blend_f1   = f1_score(y, blend_pred, average="macro")

    # 제출 저장
    sub_blend = pd.DataFrame({ID_COL: test_ids, TARGET: blend_test.argmax(axis=1).astype(int)})
    sub_blend.to_csv("[9.7]_submission_blend.csv", index=False)

    # bias 제출(있으면)
    if os.path.exists(BIAS_NPY):
        try:
            bias = np.load(BIAS_NPY)
            blend_test_bias = apply_bias_to_probs(blend_test, bias)
            pd.DataFrame({ID_COL: test_ids, TARGET: blend_test_bias.argmax(axis=1).astype(int)}
                        ).to_csv("[9.7]_submission_blend_bias.csv", index=False)
            bias_str = f"(bias={np.round(bias,4)})"
        except Exception as e:
            bias_str = f"(bias 적용 실패: {e})"
    else:
        bias_str = "(no bias file)"

    # 리포트
    lines = []
    lines.append("# [9.7] Local Auto-Ensemble Blend Report\n")
    lines.append("## Candidate CV Macro F1\n")
    for r in rows:
        lines.append(f"- {r['tag']}: {r['cv_macro_f1']:.5f}")
    lines.append("\n## Blending Result")
    lines.append(f"- Chosen models: {', '.join(chosen)}")
    wtable = ", ".join([f"{lbl}={float(w):.3f}" for lbl, w in zip(chosen, W)])
    lines.append(f"- Weights: {wtable}")
    lines.append(f"- Blend OOF Macro F1: **{blend_f1:.5f}**")
    lines.append(f"- Submissions: [9.7]_submission_blend.csv, [9.7]_submission_blend_bias.csv {bias_str}")
    with open("[9.7]_blend_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n===== 최종 블렌딩 요약 =====")
    print("선택 모델 :", ", ".join(chosen))
    print("가중치   :", wtable)
    print(f"Blend OOF Macro F1: {blend_f1:.5f}")
    print("Saved: [9.7]_submission_blend.csv, [9.7]_submission_blend_bias.csv,", "[9.7]_blend_report.md")
    print(f"완료! 총 소요시간: {time.time()-t0:.1f}초")

if __name__ == "__main__":
    main()
