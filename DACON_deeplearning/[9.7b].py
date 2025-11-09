# -*- coding: utf-8 -*-
"""
[9.7b]_catboost_local_autoensemble_plus.py
- V2 FE + OOF Target Encoding
- CatBoost 후보 다양화(학습량 보존 스케줄 × depth × 규제 × bootstrap × 가중치 × 손실)
- 5-Fold CV로 각 후보 OOF/TEST 확률 산출 후, OOF Macro F1 기준 블렌딩 가중치 탐색
- 산출물:
  * [9.7b]_oof_probs_<tag>.npy / [9.7b]_test_probs_<tag>.npy
  * [9.7b]_cv_scores_candidates.csv
  * [9.7b]_submission_<tag>.csv (각 후보)
  * [9.7b]_submission_blend.csv / [9.7b]_submission_blend_bias.csv
  * [9.7b]_blend_report.md
"""

import os, time, json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

# ===== 경로/설정 =====
DATA_PATH = "./"
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV  = os.path.join(DATA_PATH, "test.csv")
SUB_CSV   = os.path.join(DATA_PATH, "sample_submission.csv")
BIAS_NPY  = os.path.join(DATA_PATH, "[8.2]_best_bias.npy")  # 있으면 bias 제출도 생성

TARGET, ID_COL = "support_needs", "ID"
N_CLASSES = 3

N_SPLITS = 5
CV_SEED  = 42

# ===== Optuna 베이스 (네가 준 값) =====
BASE = dict(
    iterations=1460,
    depth=8,
    learning_rate=0.01011375157729347,
    l2_leaf_reg=2.9134559843727033,
    colsample_bylevel=0.8914955767108875,
    random_strength=0.10543719325919827,
    bagging_temperature=0.1804730161382898,
)

# ===== 후보 구성 (폭↑, 수는 10~12개 선) =====
SEEDS = [42]              # seed 다양화 원하면 [42,52] 정도로
LOSS_LIST = ["MultiClass", "MultiClassOneVsAll"]
LR_IT_SCHEDULES = [       # it*lr ≈ 상수 유지
    (0.0080, 1840),
    (BASE["learning_rate"], BASE["iterations"]),
    (0.0120, 1230),
]
DEPTH_LIST = [7, 8, 9]
L2_LIST = [2.9, 6.0]
BOOTSTRAPS = [("Bayesian", None), ("Bernoulli", 0.8)]
CLASS_WEIGHTS = ["Balanced", "SqrtBalanced"]  # CatBoost 지원

# 블렌딩 탐색 간격(촘촘)
BLEND_STEP = 0.02

# ===== V2 FE + TE =====
QBINS=20; WINSOR_LO=0.02; WINSOR_HI=0.98
TE_COLS=["subscription_type","gender_subscription"]; TE_SMOOTH=10.0

def base_add_features(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy(); eps=1e-6
    df["tenure_per_contract"]=df["tenure"]/(df["contract_length"]+eps)
    df["delay_ratio"]=df["payment_interval"]/(df["contract_length"]+eps)
    df["idle_ratio"]=df["after_interaction"]/(df["tenure"]+eps)
    df["late_flag"]=(df["payment_interval"]>0).astype(int)
    if "frequent" in df.columns: df["total_usage_score"]=df["tenure"]*df["frequent"]
    bins=[0,20,30,40,50,60,100]; labels=['10s','20s','30s','40s','50s','60+']
    df["age_group"]=pd.cut(df["age"], bins=bins, labels=labels, right=False)
    df["risk_score"]=df["payment_interval"]+df["after_interaction"]
    return df

def prepare_types(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    out["contract_length"]=out["contract_length"].astype(str)
    out["age_group"]=out["age_group"].astype(str)
    out["gender_subscription"]=out["gender"].astype(str)+"_"+out["subscription_type"].astype(str)
    return out

def _winsorize(s, lo=0.01, hi=0.99):
    ql=s.quantile(lo); qu=s.quantile(hi); return s.clip(lower=ql, upper=qu)
def winsor_block(df, lo=WINSOR_LO, hi=WINSOR_HI):
    out=df.copy()
    for c in ["payment_interval","after_interaction","tenure"]:
        if c in out.columns: out[c]=_winsorize(out[c], lo, hi)
    return out

def _qrank_bins(s, bins:int):
    return np.minimum((s.rank(method="average", pct=True)*bins).astype(int), bins-1)
def build_interactions_and_bins(df: pd.DataFrame, qbins:int):
    out=df.copy()
    if {"tenure","payment_interval"}.issubset(out.columns):
        out["tenure_x_payint"]=out["tenure"]*out["payment_interval"]
    if {"tenure","after_interaction"}.issubset(out.columns):
        out["tenure_x_after"]=out["tenure"]*out["after_interaction"]
    if "frequent" in out.columns:
        out["freq_x_after"]=out["frequent"]*out["after_interaction"]
    if {"delay_ratio","risk_score"}.issubset(out.columns):
        out["delay_x_risk"]=out["delay_ratio"]*out["risk_score"]
    if "risk_score" in out.columns: out["risk_log1p"]=np.log1p(out["risk_score"].clip(lower=0))
    if "idle_ratio" in out.columns: out["idle_sq"]=out["idle_ratio"]**2
    for c in ["tenure","payment_interval","after_interaction","tenure_per_contract"]:
        if c in out.columns: out[f"{c}_q{qbins}"]=_qrank_bins(out[c], qbins)
    return out

def prepare_cat_features(df: pd.DataFrame)->List[str]:
    return [c for c in ["gender","subscription_type","age_group","contract_length","gender_subscription"]
            if c in df.columns and df[c].dtype==object]

def cv_target_encode_multiclass(train_df: pd.DataFrame, y: pd.Series, te_cols, n_splits=5, seed=42, smoothing=10.0):
    K=3; skf=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out=train_df.copy()
    prior=np.bincount(y, minlength=K)/len(y)
    for col in te_cols:
        for k in range(K): out[f"TE_{col}_c{k}"]=np.nan
    for tr,va in skf.split(train_df, y):
        Xtr,ytr=train_df.iloc[tr], y.iloc[tr]; Xva=train_df.iloc[va]
        fmap={}
        for col in te_cols:
            d={}
            grp=pd.concat([Xtr[col].astype(str), ytr], axis=1).groupby(col)
            for cat,g in grp:
                cnt=np.bincount(g[TARGET].to_numpy(), minlength=K)
                prob=(cnt+smoothing/K)/(cnt.sum()+smoothing)
                d[str(cat)]=prob
            fmap[col]=d
        for col in te_cols:
            arr=np.vstack([fmap[col].get(str(v), prior) for v in Xva[col].astype(str).values])
            for k in range(K): out.loc[Xva.index, f"TE_{col}_c{k}"]=arr[:,k]
    for col in te_cols:
        for k in range(K): out[f"TE_{col}_c{k}"]=out[f"TE_{col}_c{k}"].fillna(prior[k])
    # full map
    full_map={col:{} for col in te_cols}
    for col in te_cols:
        grp=pd.concat([train_df[col].astype(str), y], axis=1).groupby(col)
        for cat,g in grp:
            cnt=np.bincount(g[TARGET].to_numpy(), minlength=K)
            prob=(cnt+smoothing/K)/(cnt.sum()+smoothing)
            full_map[col][str(cat)]=prob
    return out, full_map, prior

def apply_te_from_map(df: pd.DataFrame, te_cols, full_map, prior):
    out=df.copy()
    for col in te_cols:
        arr=np.vstack([full_map[col].get(str(v), prior) for v in out[col].astype(str).values])
        for k in range(len(prior)): out[f"TE_{col}_c{k}"]=arr[:,k]
    return out

def apply_bias_to_probs(probs: np.ndarray, bias: np.ndarray)->np.ndarray:
    eps=1e-15
    z=np.log(np.clip(probs, eps, 1.0))+bias.reshape(1,-1)
    m=z.max(axis=1, keepdims=True)
    e=np.exp(z-m); return e/e.sum(axis=1, keepdims=True)

# ===== 학습/평가 =====
def train_cv_one(X, y, Xtest, cat_features, params, tag, n_splits=N_SPLITS, seed=CV_SEED):
    skf=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof=np.zeros((len(y), N_CLASSES)); tet=np.zeros((len(Xtest), N_CLASSES))
    scores=[]
    for fold,(tr,va) in enumerate(skf.split(X,y), start=1):
        model=CatBoostClassifier(**params, cat_features=cat_features)
        model.fit(X.iloc[tr], y.iloc[tr],
                  eval_set=[(X.iloc[va], y.iloc[va])],
                  early_stopping_rounds=200, verbose=0)
        pv=model.predict_proba(X.iloc[va]); oof[va]=pv; tet+=model.predict_proba(Xtest)/n_splits
        f1=f1_score(y.iloc[va], pv.argmax(axis=1), average="macro")
        scores.append(f1)
        print(f"[{tag}] Fold {fold} Macro F1={f1:.5f}  it≈{model.tree_count_}")
    mean_f1=float(np.mean(scores)); std_f1=float(np.std(scores, ddof=1))
    print(f"[{tag}] CV Macro F1: {mean_f1:.5f} ± {std_f1:.5f}")
    return oof, tet, mean_f1

def simplex_weight_grid(n_models:int, step:float):
    ints=int(1/step)
    if n_models==2:
        for i in range(ints+1):
            yield np.array([i*step, 1-i*step])
    elif n_models==3:
        for i in range(ints+1):
            for j in range(ints+1-i):
                k=ints-i-j
                yield np.array([i,j,k])*step
    else:
        rng=np.random.default_rng(42)
        for _ in range(800):  # 표본수 ↑ (이전보다 촘촘)
            v=rng.random(n_models); v/=v.sum(); yield v

def blend_search(oofs: List[np.ndarray], y: np.ndarray, labels: List[str], step=0.02):
    P=np.stack(oofs, axis=0)
    best_f1=-1.0; best_w=None
    for w in simplex_weight_grid(len(oofs), step):
        mix=np.tensordot(w, P, axes=(0,0))
        f1=f1_score(y, mix.argmax(axis=1), average="macro")
        if f1>best_f1:
            best_f1=f1; best_w=w
    return best_f1, best_w

# ===== 메인 =====
def main():
    t0=time.time()
    print("--- [9.7b] Local Auto-Ensemble PLUS 시작 ---")

    train=pd.read_csv(TRAIN_CSV); test=pd.read_csv(TEST_CSV); sub=pd.read_csv(SUB_CSV)
    y=train[TARGET].astype(int); test_ids=test[ID_COL].values

    # FE + TE
    Xr=train.drop([ID_COL,TARGET], axis=1); Xte_r=test.drop([ID_COL], axis=1)
    X=build_interactions_and_bins(winsor_block(prepare_types(base_add_features(Xr))), QBINS)
    Xte=build_interactions_and_bins(winsor_block(prepare_types(base_add_features(Xte_r))), QBINS)
    te_cols=[c for c in TE_COLS if c in X.columns]
    X_te, full_map, prior=cv_target_encode_multiclass(X, y, te_cols, n_splits=N_SPLITS, seed=CV_SEED, smoothing=TE_SMOOTH)
    Xtest=apply_te_from_map(Xte, te_cols, full_map, prior)
    cat_features=prepare_cat_features(X_te)

    # 후보 생성
    candidates=[]
    for seed in SEEDS:
        for loss in LOSS_LIST:
            for lr, it in LR_IT_SCHEDULES:
                for depth in DEPTH_LIST:
                    for l2 in L2_LIST:
                        for bs_type, subs in BOOTSTRAPS:
                            for acw in CLASS_WEIGHTS:
                                p = dict(
                                    iterations=int(it), depth=depth, learning_rate=float(lr),
                                    l2_leaf_reg=float(l2),
                                    colsample_bylevel=BASE["colsample_bylevel"],
                                    random_strength=BASE["random_strength"],
                                    bagging_temperature=BASE["bagging_temperature"],
                                    auto_class_weights=acw,
                                    loss_function=loss,
                                    bootstrap_type=bs_type,
                                    random_seed=seed,
                                    verbose=0,
                                )
                                if bs_type=="Bernoulli":
                                    p["subsample"]=subs
                                # 후보 수를 너무 늘리지 않도록 필터링(손실별 일정 개수만)
                                # - MultiClass는 depth in {7,8}, OVA는 depth in {8,9} 만 사용
                                if loss=="MultiClass" and depth not in (7,8): continue
                                if loss=="MultiClassOneVsAll" and depth not in (8,9): continue
                                tag=f"{'mc' if loss=='MultiClass' else 'ova'}_s{seed}_d{depth}_lr{str(lr).replace('.','')}_it{it}_l2{l2}_{bs_type[:3].lower()}_{acw.lower()}"
                                candidates.append((tag, p))

    # 너무 많으면 상위 K개만(임계치)
    MAX_CAND = 12
    if len(candidates)>MAX_CAND:
        candidates = candidates[:MAX_CAND]

    # 학습
    rows=[]; labels=[]; oofs=[]; tests=[]
    for tag, params in candidates:
        print(f"\n=== Train Candidate: {tag} ===")
        oof, tet, cvf1 = train_cv_one(X_te, y, Xtest, cat_features, params, tag)
        np.save(f"[9.7b]_oof_probs_{tag}.npy",  oof)
        np.save(f"[9.7b]_test_probs_{tag}.npy", tet)
        pd.DataFrame({ID_COL:test_ids, TARGET: tet.argmax(axis=1).astype(int)}).to_csv(f"[9.7b]_submission_{tag}.csv", index=False)
        rows.append({"tag": tag, "cv_macro_f1": cvf1})
        labels.append(tag); oofs.append(oof); tests.append(tet)

    pd.DataFrame(rows).to_csv("[9.7b]_cv_scores_candidates.csv", index=False)
    print("\n=== 후보 CV 요약 ===")
    for r in rows: print(f"{r['tag']:>30s} : {r['cv_macro_f1']:.5f}")

    # 블렌딩
    print("\n--- 블렌딩 탐색 ---")
    best_f1, best_w = blend_search(oofs, y.values, labels, step=BLEND_STEP)
    W = np.array(best_w, dtype=float)
    blend_oof  = np.tensordot(W, np.stack(oofs, axis=0), axes=(0,0))
    blend_test = np.tensordot(W, np.stack(tests, axis=0), axes=(0,0))
    blend_pred = blend_oof.argmax(axis=1)
    blend_oof_f1 = f1_score(y, blend_pred, average="macro")

    sub_blend = pd.DataFrame({ID_COL:test_ids, TARGET: blend_test.argmax(axis=1).astype(int)})
    sub_blend.to_csv("[9.7b]_submission_blend.csv", index=False)

    bias_str="(no bias)"
    if os.path.exists(BIAS_NPY):
        try:
            bias=np.load(BIAS_NPY)
            blend_test_b = apply_bias_to_probs(blend_test, bias)
            pd.DataFrame({ID_COL:test_ids, TARGET: blend_test_b.argmax(axis=1).astype(int)}
                        ).to_csv("[9.7b]_submission_blend_bias.csv", index=False)
            bias_str=f"(bias={np.round(bias,4)})"
        except Exception as e:
            bias_str=f"(bias 실패: {e})"

    # 리포트
    wstr=", ".join([f"{lb}={float(w):.3f}" for lb,w in zip(labels, W)])
    lines=[
        "# [9.7b] Blend Report",
        "## Candidate CV Macro F1",
    ]+[f"- {r['tag']}: {r['cv_macro_f1']:.5f}" for r in rows]+[
        "\n## Blend",
        f"- Weights: {wstr}",
        f"- Blend OOF Macro F1: **{blend_oof_f1:.5f}**",
        f"- Submissions: [9.7b]_submission_blend.csv, [9.7b]_submission_blend_bias.csv {bias_str}"
    ]
    with open("[9.7b]_blend_report.md","w",encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n===== 최종 요약 =====")
    print("가중치 :", wstr)
    print(f"Blend OOF Macro F1: {blend_oof_f1:.5f}")
    print("Saved: [9.7b]_submission_blend.csv (+ _bias.csv), [9.7b]_blend_report.md")
    print(f"완료! 총 소요시간: {round(time.time()-t0,1)}s")

if __name__=="__main__":
    main()
