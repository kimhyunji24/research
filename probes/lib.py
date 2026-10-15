"""라운드별 실패 예측 프로브 평가 (설계 재검토 문서 5, 6절, docs/analysis_plan_2026-10-15.md).

입력: collect/rollout.py 출력 폴더 (episodes.jsonl, steps.jsonl, hidden.npy[N, L, H]).
평가 단위: 라운드 r(0-based step)에 살아 있는 episode 1개당 1행 -> step 번호가 상수라 길이 교락이 없다.
라벨: 성공=0, 실패=1 (실패를 양성으로 둔다).
표준화/PCA/TF-IDF는 항상 Pipeline 안에서 fold별로 학습한다(누수 방지).
분할 단위는 장면(scene)이고, 선형 모델의 C는 바깥 train fold 안에서 nested로 고른다.
"""
import json, os
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import PROGRESS_COLS, add_text_columns, progress_features

WINDOW = 6  # rollout.py의 history 길이와 동일
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


# ---------------------------------------------------------------- 데이터
def scene_of(task_id):
    """게임 경로 `.../<task_type>-<obj>-<mrecep>-<recep>-<scene>/trial_.../game.tw-pddl`에서 장면 번호."""
    return task_id.replace("\\", "/").split("/")[-3].split("-")[-1]


def load_run(path, hidden_path=None):
    """hidden_path: 다른 reader가 뽑은 hidden.npy(collect/extract_reader.py). 행 순서는 steps의 hid_idx와 같다."""
    eps = pd.read_json(os.path.join(path, "episodes.jsonl"), lines=True, dtype={"task_id": str})
    steps = pd.read_json(os.path.join(path, "steps.jsonl"), lines=True, dtype={"task_id": str})
    hid = np.load(hidden_path or os.path.join(path, "hidden.npy"), mmap_mode="r")  # (N, L, H), float16
    steps = steps.merge(eps[["episode", "task_type", "success"]], on="episode")
    steps["y"] = (~steps["success"]).astype(int)
    steps = steps.sort_values(["episode", "step"]).reset_index(drop=True)
    steps["scene"] = steps["task_id"].map(scene_of)
    # 직전 WINDOW step 안에서 같은 action을 이미 골랐던 횟수 (현재 선택한 action 포함 -> baseline에 유리한 쪽)
    rep = []
    for _, g in steps.groupby("episode", sort=False):
        acts = g["action"].tolist()
        rep += [sum(a == b for b in acts[max(0, i - WINDOW):i]) for i, a in enumerate(acts)]
    steps["repeat"] = rep
    steps = add_text_columns(steps)
    steps = pd.concat([steps, progress_features(steps)], axis=1)
    return steps, hid


# ---------------------------------------------------------------- baseline 사다리
def baseline_columns(df, rung):
    cols = ["step"]
    if rung >= 2: cols += [c for c in df.columns if c.startswith("tt_")]
    if rung >= 3: cols += ["repeat", "p_chosen", "n_cand"]
    if rung >= 4: cols += PROGRESS_COLS
    return cols


def baseline_frame(df, rung):
    """rung 1~5의 입력 표. 5단은 'text' 열(프롬프트 user 메시지)을 포함한다."""
    d = df.copy()
    if rung >= 2:
        d = pd.concat([d, pd.get_dummies(d["task_type"], prefix="tt").astype(float)], axis=1)
    X = d[baseline_columns(d, rung)].astype(float)
    if rung >= 5:
        if "text" not in df.columns: raise ValueError("5단은 steps.jsonl에 prompt가 있어야 한다(main_3b 이후 수집분)")
        X["text"] = df["text"].to_numpy()
    return X


def make_baseline(rung, cols, C):
    num = [c for c in cols if c != "text"]
    tr = [("num", StandardScaler(), num)]
    if rung >= 5:
        tr.append(("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000), "text"))
    return make_pipeline(ColumnTransformer(tr), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))


# ---------------------------------------------------------------- 프로브
def make_probe(kind="linear", pca_k=None, C=0.01, seed=0):
    steps = [StandardScaler()]
    if pca_k: steps.append(PCA(n_components=pca_k, random_state=seed))
    if kind == "linear":
        steps.append(LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
    else:
        steps.append(MLPClassifier((64,), alpha=1e-2, early_stopping=True, max_iter=300, random_state=seed))
    return make_pipeline(*steps)


# ---------------------------------------------------------------- 교차적합
def _take(X, idx):
    return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]


def _select_C(X, y, groups, make, Cs, inner_splits, seed):
    scores = {C: [] for C in Cs}
    for tr, va in StratifiedGroupKFold(inner_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2: continue
        for C in Cs:
            m = make(C).fit(_take(X, tr), y[tr])
            scores[C].append(roc_auc_score(y[va], m.predict_proba(_take(X, va))[:, 1]))
    return max(Cs, key=lambda c: np.mean(scores[c]) if scores[c] else -1.0)


def crossfit(X, y, groups, make, Cs=C_GRID, n_splits=5, n_repeats=2, inner_splits=3, seed=0):
    """장면 단위 층화 교차적합. 반환: (행별 점수(반복 평균), 선택된 C 목록, PCA explained variance 비율 목록).
    Cs=None이면 C를 고르지 않고 make(None)을 쓴다."""
    s = np.zeros(len(y)); chosen, evr = [], []
    for rep in range(n_repeats):
        for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + rep).split(np.zeros(len(y)), y, groups):
            if len(np.unique(y[tr])) < 2: raise ValueError("train fold에 한 클래스만 있다")
            C = None
            if Cs is not None:
                C = _select_C(_take(X, tr), y[tr], groups[tr], make, Cs, inner_splits, seed + rep); chosen.append(C)
            m = make(C).fit(_take(X, tr), y[tr])
            s[te] += m.predict_proba(_take(X, te))[:, 1] / n_repeats
            pca = m.named_steps.get("pca")
            if pca is not None: evr.append(float(pca.explained_variance_ratio_.sum()))
    return s, chosen, evr


def cluster_boot_idx(groups, n_boot=1000, seed=0):
    """장면 단위 cluster bootstrap 인덱스 목록(모든 방법이 같은 재표집을 공유 -> 짝지은 차이 계산 가능)과
    episode 단위 bootstrap 인덱스 목록."""
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(ug))]
    rng = np.random.default_rng(seed)
    cl = [np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))]) for _ in range(n_boot)]
    ep = [rng.integers(0, len(groups), len(groups)) for _ in range(n_boot)]
    return cl, ep


def _auc_boot(y, s, idx_list):
    out = []
    for i in idx_list:
        if len(np.unique(y[i])) == 2: out.append(roc_auc_score(y[i], s[i]))
    return np.array(out)


def ci(y, s, idx_list):
    a = _auc_boot(y, s, idx_list)
    return tuple(np.percentile(a, [2.5, 97.5])) if len(a) else (np.nan, np.nan)


def paired_diff(y, s_a, s_b, idx_list):
    """AUROC(a) - AUROC(b)의 점추정과 bootstrap 95% CI."""
    d = []
    for i in idx_list:
        if len(np.unique(y[i])) == 2: d.append(roc_auc_score(y[i], s_a[i]) - roc_auc_score(y[i], s_b[i]))
    est = roc_auc_score(y, s_a) - roc_auc_score(y, s_b)
    return est, *np.percentile(d, [2.5, 97.5])


def _one_perm(X, y, groups, make, Cs, n_splits, n_repeats, seed):
    yp = np.random.default_rng(seed).permutation(y)
    s, _, _ = crossfit(X, yp, groups, make, Cs, n_splits, n_repeats, seed=seed)
    return roc_auc_score(yp, s)


def round_eval(df, X, r, make, Cs=C_GRID, n_splits=5, n_repeats=2, n_perm=100, boot=None, seed=0,
               group_col="scene", n_jobs=-1):
    """라운드 r(0-based step)의 AUROC/AUPRC, CI, 순열 대조군. X는 df 행과 정렬된 (len(df), d) 배열 또는 표.
    반환 dict. '_y', '_s', '_groups'는 짝지은 비교용이다."""
    m = (df["step"] == r).to_numpy()
    idx = np.where(m)[0]
    y, g = df.loc[m, "y"].to_numpy(), df.loc[m, group_col].to_numpy()
    if y.sum() < n_splits or (1 - y).sum() < n_splits: return None
    Xr = _take(X, idx)
    try:
        s, chosen, evr = crossfit(Xr, y, g, make, Cs, n_splits, n_repeats, seed=seed)
    except ValueError as e:  # 예: PCA k가 fold 학습 표본 수 이상
        return dict(step=r, n=len(y), n_fail=int(y.sum()), base_rate=float(y.mean()), auroc=np.nan, note=str(e)[:80])
    boot = boot or cluster_boot_idx(g, 1000, seed)
    lo, hi = ci(y, s, boot[0]); elo, ehi = ci(y, s, boot[1])
    res = dict(step=r, n=len(y), n_fail=int(y.sum()), base_rate=round(float(y.mean()), 3), auroc=roc_auc_score(y, s),
               ci_lo=lo, ci_hi=hi, ci_ep_lo=elo, ci_ep_hi=ehi, auprc=average_precision_score(y, s),
               C_median=float(np.median(chosen)) if chosen else np.nan,
               pca_evr=float(np.mean(evr)) if evr else np.nan)
    if n_perm:
        null = np.array(Parallel(n_jobs=n_jobs)(delayed(_one_perm)(Xr, y, g, make, Cs, n_splits, n_repeats, seed + 1 + i)
                                                for i in range(n_perm)))
        res.update(null_mean=float(null.mean()), null_lo=float(np.percentile(null, 2.5)),
                   null_hi=float(np.percentile(null, 97.5)), p_perm=float((1 + (null >= res["auroc"]).sum()) / (1 + n_perm)))
    res.update(_y=y, _s=s, _groups=g)
    return res
