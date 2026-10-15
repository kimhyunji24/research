"""라운드별 실패 예측 프로브 평가 (설계 재검토 문서 5, 6절).

입력: collect/rollout.py 출력 폴더 (episodes.jsonl, steps.jsonl, hidden.npy[N, L, H]).
평가 단위: 라운드 r에 살아 있는 episode 1개당 1행 -> step 번호가 상수라 길이 교락이 없다.
라벨: episode 성공=0, 실패=1 (실패를 양성으로 둔다).
PCA/표준화는 항상 Pipeline 안에서 fold별로 학습한다(누수 방지).
"""
import json, os
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WINDOW = 6  # rollout.py의 history 길이와 동일


def load_run(path):
    eps = pd.read_json(os.path.join(path, "episodes.jsonl"), lines=True)
    steps = pd.read_json(os.path.join(path, "steps.jsonl"), lines=True)
    hid = np.load(os.path.join(path, "hidden.npy"))  # (N, L, H), float16
    steps = steps.merge(eps[["episode", "task_type", "success"]], on="episode")
    steps["y"] = (~steps["success"]).astype(int)
    steps = steps.sort_values(["episode", "step"]).reset_index(drop=True)
    # 직전 WINDOW step 안에서 같은 action을 이미 골랐던 횟수 (현재 step 이전 정보만 사용)
    rep = []
    for _, g in steps.groupby("episode", sort=False):
        acts = g["action"].tolist()
        rep += [sum(a == b for b in acts[max(0, i - WINDOW):i]) for i, a in enumerate(acts)]
    steps["repeat"] = rep
    return steps, hid


def baseline_features(df, rung):
    """baseline 사다리 1~3단. (4단 진행도, 5단 TF-IDF는 추출기 구현 후 추가)"""
    cols = [df[["step"]].to_numpy(float)]
    if rung >= 2: cols.append(pd.get_dummies(df["task_type"]).to_numpy(float))
    if rung >= 3: cols.append(df[["repeat", "p_chosen", "n_cand"]].to_numpy(float))
    return np.hstack(cols)


def make_model(kind="linear", pca_k=None, C=0.01, seed=0):
    steps = [StandardScaler()]
    if pca_k: steps.append(PCA(n_components=pca_k, random_state=seed))
    if kind == "linear":
        steps.append(LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
    else:
        steps.append(MLPClassifier((64,), alpha=1e-2, early_stopping=True, max_iter=300, random_state=seed))
    return make_pipeline(*steps)


def crossfit_scores(X, y, groups, model_fn, n_splits=5):
    """task 단위 GroupKFold cross-fitting. 모든 행이 한 번씩 평가에 쓰인다."""
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2: return None
        m = model_fn().fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    return s


def boot_ci(y, s, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) == 2: out.append(roc_auc_score(y[i], s[i]))
    return tuple(np.percentile(out, [2.5, 97.5]))


def round_eval(df, X, r, model_fn, n_splits=5, shuffle_labels=False, seed=0):
    """라운드 r에서의 AUROC/AUPRC. X는 df 행과 정렬된 (len(df), d) 배열."""
    m = (df["step"] == r).to_numpy()
    y, g, Xr = df.loc[m, "y"].to_numpy(), df.loc[m, "task_id"].to_numpy(), X[m]
    if y.sum() < n_splits or (1 - y).sum() < n_splits: return None
    if shuffle_labels:  # episode 단위 무작위 라벨 대조군
        y = np.random.default_rng(seed).permutation(y)
    s = crossfit_scores(Xr, y, g, model_fn, n_splits)
    if s is None: return None
    lo, hi = boot_ci(y, s, seed=seed)
    return dict(round=r, n=len(y), n_fail=int(y.sum()), base_rate=float(y.mean()),
                auroc=roc_auc_score(y, s), ci_lo=lo, ci_hi=hi, auprc=average_precision_score(y, s))
