"""B6 analysis (docs/review_b_plan_2026-10-21.md): action-token probes A50 (layer 18) and A75 (layer 27) against R5 and
P75. Same cross-fitting and scene bootstrap as review_b.py; P75 and R5 scores are read from results/review_b_scores.csv
(f = 1.0), so every comparison uses the same scores as B1. One run.

    .venv/bin/python scripts/2026-10-21/review_b6.py [--n_jobs 24] [--smoke]
Writes results/review_b6_action.csv and results/review_b6_action_diff.csv; stops if they exist.
"""
import argparse, datetime, os, subprocess, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "probes"))
from lib import C_GRID, cluster_boot_idx, load_run  # noqa: E402


def lin(C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))


def _fold(X, y, g, tr, te, rep):  # same as review_b._fold with f = 1
    best, best_s = None, -1.0
    for C in C_GRID:
        sc = []
        for itr, iva in StratifiedGroupKFold(3, shuffle=True, random_state=rep).split(np.zeros(len(tr)), y[tr], g[tr]):
            a, b = tr[itr], tr[iva]
            if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2: continue
            sc.append(roc_auc_score(y[b], lin(C).fit(X[a], y[a]).predict_proba(X[b])[:, 1]))
        if sc and np.mean(sc) > best_s: best, best_s = C, float(np.mean(sc))
    return te, lin(best).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]


def crossfit(X, y, g, n_jobs):
    js = [(tr, te, rep) for rep in (0, 1) for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, g)]
    s = np.zeros(len(y))
    for te, p in Parallel(n_jobs=n_jobs)(delayed(_fold)(X, y, g, tr, te, rep) for tr, te, rep in js): s[te] += p / 2
    return s


def summarize(Y, A, B, boot, steps):
    est = np.array([roc_auc_score(Y[r], A[r]) - (roc_auc_score(Y[r], B[r]) if B else 0) for r in steps])
    bt = np.array([[roc_auc_score(Y[r][i], A[r][i]) - (roc_auc_score(Y[r][i], B[r][i]) if B else 0) for r in steps] for i in boot])
    out = []
    for j, lab in list(enumerate(steps)) + [(None, "mean")]:
        e = est.mean() if j is None else est[j]; b = bt.mean(1) if j is None else bt[:, j]
        row = dict(step=lab, est=e, ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5), n_boot=len(b))
        if B: row["p_boot"] = float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))
        out.append(row)
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--path", default="data/main_3b/Qwen2.5-3B-Instruct")
ap.add_argument("--act", default="data/main_3b/action_token/Qwen2.5-3B-Instruct")
ap.add_argument("--n_jobs", type=int, default=24)
ap.add_argument("--smoke", action="store_true", help="shuffled labels, step 0, no files written")
a = ap.parse_args()
OUT = ["results/review_b6_action.csv", "results/review_b6_action_diff.csv"]
if not a.smoke:
    for p in OUT:
        if os.path.exists(p): raise SystemExit(f"output exists: {p}")
print("start", datetime.datetime.now().isoformat(timespec="seconds"), "HEAD",
      subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
      "smoke" if a.smoke else "", flush=True)
T0 = time.time()
df, _ = load_run(a.path)
df = df[(df["episode"] < 250) & (df["step"] <= 2)].reset_index(drop=True)
rows = pd.read_csv(os.path.join(a.act, "rows.csv")); Hact = np.load(os.path.join(a.act, "hidden.npy")).astype(np.float32)
assert (rows[["episode", "step", "hid_idx"]].to_numpy() == df[["episode", "step", "hid_idx"]].to_numpy()).all()
sc = pd.read_csv("results/review_b_scores.csv")
STEPS = [0] if a.smoke else [0, 1, 2]
Y, S = {}, {}
for r in STEPS:
    m = (df["step"] == r).to_numpy(); idx = np.where(m)[0]
    y = df.loc[m, "y"].to_numpy(); g = df.loc[m, "scene"].to_numpy(); ep = df.loc[m, "episode"].to_numpy()
    if a.smoke: y = np.random.default_rng(0).permutation(y)
    Y[r] = y
    for meth in ("P75", "R5"):
        t = sc[(sc.step == r) & (sc.method == f"{meth}_f1.0_d0")].set_index("episode").loc[ep]
        assert (t.y.to_numpy() == df.loc[m, "y"].to_numpy()).all()
        S[(r, meth)] = t.score.to_numpy()
    for meth, j in (("A50", 0), ("A75", 1)):
        S[(r, meth)] = crossfit(Hact[idx, j], y, g, a.n_jobs)
    print(f"step {r} crossfit done {time.time() - T0:.0f}s", flush=True)
BOOT = cluster_boot_idx(df.loc[df.step == STEPS[0], "scene"].to_numpy(), 1000, 0)[0]
res = [dict(method=m, **row) for m in ("A50", "A75", "P75", "R5")
       for row in summarize(Y, {r: S[(r, m)] for r in STEPS}, None, BOOT, STEPS)]
dif = [dict(comparison=f"{x}-{z}", **row) for x, z in (("A50", "R5"), ("A75", "R5"), ("A75", "P75"))
       for row in summarize(Y, {r: S[(r, x)] for r in STEPS}, {r: S[(r, z)] for r in STEPS}, BOOT, STEPS)]
res, dif = pd.DataFrame(res), pd.DataFrame(dif)
order = dif["p_boot"].sort_values(kind="stable").index; k, run = len(order), 0.0
for jj, i in enumerate(order):
    run = max(run, min(1.0, (k - jj) * dif.loc[i, "p_boot"])); dif.loc[i, "p_holm"] = run
dif["verdict"] = np.where(dif.ci_lo > 0, "superior", np.where(dif.ci_hi < 0, "inferior", "inconclusive"))
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False)); print(); print(dif.round(3).to_string(index=False))
if not a.smoke:
    res.to_csv(OUT[0], index=False); dif.to_csv(OUT[1], index=False)
print("end", datetime.datetime.now().isoformat(timespec="seconds"), f"{time.time() - T0:.0f}s", flush=True)
