"""프로브 강화 점검(탐색, docs/probe_strength_plan_2026-10-20.md). 1회 실행.
사용: python scripts/2026-10-20/probe_strength.py --out results/probe_strength_3b.csv [--n_jobs 10] [--smoke]"""
import argparse, itertools, os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "probes"))
from lib import C_GRID, _take, baseline_columns, baseline_frame, ci, cluster_boot_idx, load_run, make_baseline

WIDE_C = tuple(10.0 ** k for k in range(-6, 4))


def lr(C): return LogisticRegression(C=C, max_iter=2000, class_weight="balanced")


def builders(num5, hid_cols):
    def r5hid(p):
        tr = [("num", StandardScaler(), num5 + hid_cols),
              ("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000), "text")]
        return make_pipeline(ColumnTransformer(tr), lr(p["C"]))
    return {
        "base": ([{"C": c} for c in C_GRID], lambda p: make_pipeline(StandardScaler(), lr(p["C"]))),
        "wideC": ([{"C": c} for c in WIDE_C], lambda p: make_pipeline(StandardScaler(), lr(p["C"]))),
        "pca": ([{"k": k, "C": c} for k, c in itertools.product((8, 32, 64, 128), C_GRID)],
                lambda p: make_pipeline(StandardScaler(), PCA(n_components=p["k"], random_state=0), lr(p["C"]))),
        "concat": ([{"C": c} for c in C_GRID], lambda p: make_pipeline(StandardScaler(), lr(p["C"]))),
        "mlp": ([{"h": h, "a": a} for h, a in itertools.product((32, 128), (1e-3, 1e-1))],
                lambda p: make_pipeline(StandardScaler(), MLPClassifier((p["h"],), alpha=p["a"], early_stopping=True,
                                                                        max_iter=300, random_state=0))),
        "r5+hid": ([{"C": c} for c in C_GRID], r5hid),
        "baseline5": ([{"C": c} for c in C_GRID], lambda p: make_baseline(5, num5 + ["text"], p["C"])),
    }


def _fold(X, y, g, grid, build, tr, te, rep):
    best, best_s = None, -1.0
    for p in grid:
        if "k" in p and p["k"] >= int(len(tr) * 2 / 3) - 1: continue  # PCA 차원이 inner 학습 표본 수 이상이면 건너뜀
        sc = []
        for itr, iva in StratifiedGroupKFold(3, shuffle=True, random_state=rep).split(np.zeros(len(tr)), y[tr], g[tr]):
            a, b = tr[itr], tr[iva]
            if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2: continue
            sc.append(roc_auc_score(y[b], build(p).fit(_take(X, a), y[a]).predict_proba(_take(X, b))[:, 1]))
        if sc and np.mean(sc) > best_s: best, best_s = p, float(np.mean(sc))
    m = build(best).fit(_take(X, tr), y[tr])
    return te, m.predict_proba(_take(X, te))[:, 1], best


def crossfit_grid(X, y, g, grid, build, n_jobs):
    jobs = [(tr, te, rep) for rep in (0, 1) for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, g)]
    out = Parallel(n_jobs=n_jobs)(delayed(_fold)(X, y, g, grid, build, tr, te, rep) for tr, te, rep in jobs)
    s = np.zeros(len(y))
    for te, p, _ in out: s[te] += p / 2
    return s, [b for _, _, b in out]


ap = argparse.ArgumentParser()
ap.add_argument("--path", default="data/main_3b/Qwen2.5-3B-Instruct")
ap.add_argument("--rounds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--n_jobs", type=int, default=10)
ap.add_argument("--out", required=True)
ap.add_argument("--smoke", action="store_true", help="라벨을 섞어 step 0에서만 모든 변형이 돌아가는지 확인(결과 아님)")
a = ap.parse_args()
if not a.smoke and os.path.exists(a.out): raise SystemExit(f"출력 파일이 이미 있다: {a.out}")
df, hid = load_run(a.path)
df = df[df["episode"] < 250].reset_index(drop=True)
X5 = baseline_frame(df, 5); num5 = baseline_columns(X5, 5)
hid_cols = [f"h{i}" for i in range(hid.shape[2])]
rows, diffs, scores = [], [], {}
for r in (a.rounds[:1] if a.smoke else a.rounds):
    m = (df["step"] == r).to_numpy(); idx = np.where(m)[0]
    y = df.loc[m, "y"].to_numpy(); g = df.loc[m, "scene"].to_numpy()
    if a.smoke: y = np.random.default_rng(0).permutation(y)
    H = np.asarray(hid[df["hid_idx"].to_numpy()[idx]], dtype=np.float32)  # (n, 4, d)
    Xs = {"base": H[:, 2], "wideC": H[:, 2], "pca": H[:, 2], "concat": H.reshape(len(idx), -1), "mlp": H[:, 2],
          "r5+hid": pd.concat([X5.iloc[idx].reset_index(drop=True), pd.DataFrame(H[:, 2], columns=hid_cols)], axis=1),
          "baseline5": X5.iloc[idx].reset_index(drop=True)}
    B = builders(num5, hid_cols)
    for name, (grid, build) in B.items():
        t0 = time.time(); s, chosen = crossfit_grid(Xs[name], y, g, grid, build, a.n_jobs); scores[(r, name)] = s
        lo, hi = ci(y, s, cluster_boot_idx(g, 1000, 0)[0])
        rows.append(dict(step=r, method=name, n=len(y), n_fail=int(y.sum()), auroc=roc_auc_score(y, s), ci_lo=lo, ci_hi=hi,
                         chosen=";".join(sorted({str(c) for c in chosen}))))
        print(f"  step {r} {name}: auroc={rows[-1]['auroc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    boot = cluster_boot_idx(g, 1000, 0)[0]
    for name in ("base", "wideC", "pca", "concat", "mlp", "r5+hid"):
        sa, sb = scores[(r, name)], scores[(r, "baseline5")]
        dd = np.array([roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]) for i in boot if len(np.unique(y[i])) == 2])
        diffs.append(dict(step=r, method=name, vs="baseline5", diff=roc_auc_score(y, sa) - roc_auc_score(y, sb),
                          ci_lo=np.percentile(dd, 2.5), ci_hi=np.percentile(dd, 97.5),
                          p_boot=float(min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean())))))
out = pd.DataFrame(rows); d = pd.DataFrame(diffs)
order = d["p_boot"].sort_values().index; k, run = len(order), 0.0
for j, i in enumerate(order):
    run = max(run, min(1.0, (k - j) * d.loc[i, "p_boot"])); d.loc[i, "p_holm"] = run
d["verdict"] = np.where(d.ci_lo > 0, "superior", np.where(d.ci_hi < 0, "inferior", "inconclusive"))
pd.set_option("display.width", 220)
print(out.round(3).to_string(index=False)); print(); print(d.round(3).to_string(index=False))
if a.smoke: sys.exit(0)
out.to_csv(a.out, index=False); d.to_csv(a.out.replace(".csv", "_diff.csv"), index=False)
