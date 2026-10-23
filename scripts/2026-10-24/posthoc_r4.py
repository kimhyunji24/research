"""Post-hoc exploration, written AFTER seeing the sweep1500 results (docs/collect_ext1000_plan_2026-10-21.md, change history 2026-10-24).
Cluster-bootstrap CIs for P75 - R4 and R5 - R4 (and, as a check, P75 - R5). Same cross-fit and bootstrap procedure as
scripts/2026-10-23/sweep1500.py (same folds, seeds, draws, C grid, 1,000 scene-cluster bootstrap replicates, seed 0); only methods
P75, R4, R5 and fractions f in {1/24, 1/3, 1} (about 50, 400, 1,200 training episodes). The sweep1500 scores were not saved, so the
cross-fit is recomputed (deterministic); results/posthoc_r4_check.csv compares the recomputed values with sweep1500_lc/diff.csv.
Evaluation sets: all 1,500 episodes and the new 1,000 only. Every value is exploratory and uncorrected.

    .venv/bin/python scripts/2026-10-24/posthoc_r4.py [--n_jobs 24]
Writes results/posthoc_r4_{diff,contrast,check}.csv; stops if any exists.
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
from lib import C_GRID, _take, baseline_columns, baseline_frame, cluster_boot_idx, load_run, make_baseline  # noqa: E402

FRACS, N_DRAW = (1 / 24, 1 / 3, 1.0), 5
N_MAIN = 500
OUT = {k: f"results/posthoc_r4_{k}.csv" for k in ("diff", "contrast", "check")}
RUNS = ("data/main_3b/Qwen2.5-3B-Instruct", "data/ext1000_3b/Qwen2.5-3B-Instruct")
COMPS = (("P75", "R4"), ("R5", "R4"), ("P75", "R5"))  # P75-R5 is the check against sweep1500_diff.csv


def lin(C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))


def _fold(X, y, g, make, tr, te, rep, k, f, draw):  # identical to sweep1500._fold
    if f < 1:
        rng = np.random.default_rng([draw, rep, k])
        tr = np.sort(np.concatenate([rng.choice(tr[y[tr] == c], max(3, int(round(f * (y[tr] == c).sum()))), replace=False)
                                     for c in (0, 1)]))
    best, best_s = None, -1.0
    for C in C_GRID:
        sc = []
        for itr, iva in StratifiedGroupKFold(3, shuffle=True, random_state=rep).split(np.zeros(len(tr)), y[tr], g[tr]):
            a, b = tr[itr], tr[iva]
            if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2: continue
            sc.append(roc_auc_score(y[b], make(C).fit(_take(X, a), y[a]).predict_proba(_take(X, b))[:, 1]))
        if sc and np.mean(sc) > best_s: best, best_s = C, float(np.mean(sc))
    return te, len(tr), make(best).fit(_take(X, tr), y[tr]).predict_proba(_take(X, te))[:, 1]


def run_many(specs, n_jobs):  # identical to sweep1500.run_many
    flat = [(key, X, y, g, make, tr, te, rep, k, f, draw) for key, (X, y, g, make, f, draw) in specs.items()
            for rep in (0, 1) for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, g))]
    out = Parallel(n_jobs=n_jobs)(delayed(_fold)(X, y, g, make, tr, te, rep, k, f, draw)
                                  for key, X, y, g, make, tr, te, rep, k, f, draw in flat)
    S, NT = {key: np.zeros(len(specs[key][1])) for key in specs}, {key: [] for key in specs}
    for (key, *_), (te, ntr, p) in zip(flat, out): S[key][te] += p / 2; NT[key].append(ntr)
    return S, NT


def val(y, ss, i=None):  # identical to sweep1500.val
    if i is not None:
        y = y[i]
        if len(np.unique(y)) < 2: return np.nan
    return float(np.mean([roc_auc_score(y, s if i is None else s[i]) for s in ss]))


def bt_matrix(Y, terms, boot, steps):  # identical to sweep1500.bt_matrix
    est = np.array([sum(sg * val(Y[r], T[r]) for sg, T in terms) for r in steps])
    bt = np.array([[sum(sg * val(Y[r], T[r], i) for sg, T in terms) for r in steps] for i in boot])
    return est, bt


def rows_from(est, bt, steps):  # same as sweep1500.rows_from, two-sided, no Holm
    out = []
    for j, lab in list(enumerate(steps)) + [(None, "mean")]:
        e = est.mean() if j is None else est[j]
        b = np.nanmean(bt, 1) if j is None else bt[:, j]
        b = b[~np.isnan(b)]
        out.append(dict(step=lab, est=e, ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5), n_boot=len(b),
                        p_boot=float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))))
    return out


def load_all():  # identical to sweep1500.load_all
    dfs, hs = [], []
    for k, path in enumerate(RUNS):
        d, hid = load_run(path)
        d = d[d["step"] <= 2].reset_index(drop=True)
        hs.append(np.asarray(hid[d["hid_idx"].to_numpy()], dtype=np.float16))
        d["src"] = ("main_3b", "ext1000_3b")[k]
        if k == 1: d["episode"] = d["episode"] + N_MAIN
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    assert df["task_id"].nunique() == df["episode"].nunique() == 1500, "games are not unique across the two runs"
    df["hid_idx"] = np.arange(len(df))
    return df, np.concatenate(hs)


ap = argparse.ArgumentParser()
ap.add_argument("--n_jobs", type=int, default=24)
a = ap.parse_args()
for p in OUT.values():
    if os.path.exists(p): raise SystemExit(f"output exists: {p}")
print("start", datetime.datetime.now().isoformat(timespec="seconds"), "HEAD",
      subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(), flush=True)
T0 = time.time()
df, hid = load_all()
assert len(df) == 4500, len(df)
X4, X5 = baseline_frame(df, 4), baseline_frame(df, 5)
num4, num5 = baseline_columns(X4, 4), baseline_columns(X5, 5)
STEPS = [0, 1, 2]
Y, G, EP, D = {}, {}, {}, {}
for r in STEPS:
    m = (df["step"] == r).to_numpy(); idx = np.where(m)[0]
    EP[r] = df.loc[m, "episode"].to_numpy(); G[r] = df.loc[m, "scene"].to_numpy(); Y[r] = df.loc[m, "y"].to_numpy()
    H = np.asarray(hid[df["hid_idx"].to_numpy()[idx]], dtype=np.float32)
    D[r] = {"P75": (H[:, 2], lin),
            "R4": (X4.iloc[idx].reset_index(drop=True), lambda C: make_baseline(4, num4, C)),
            "R5": (X5.iloc[idx].reset_index(drop=True), lambda C: make_baseline(5, num5 + ["text"], C))}
for r in STEPS: assert (EP[r] == EP[STEPS[0]]).all() and len(EP[r]) == 1500
print("rows per step", {r: len(Y[r]) for r in STEPS}, "scenes", len(np.unique(G[STEPS[0]])), flush=True)
METH = ["P75", "R4", "R5"]
specs = {(r, mth, f, d): (D[r][mth][0], Y[r], G[r], D[r][mth][1], f, d)
         for r in STEPS for mth in METH for f in FRACS for d in (range(N_DRAW) if f < 1 else [0])}
t0 = time.time(); S, NT = run_many(specs, a.n_jobs)
print(f"crossfit: {len(specs)} runs, {time.time() - t0:.0f}s", flush=True)


def sc(mth, f, r, idx=None):
    ss = [S[(r, mth, f, d)] for d in (range(N_DRAW) if f < 1 else [0])]
    return ss if idx is None else [s[idx] for s in ss]


# check against sweep1500 (AUROC per method, step mean; P75-R5 difference, step mean)
lc, dif0 = pd.read_csv("results/sweep1500_lc.csv"), pd.read_csv("results/sweep1500_diff.csv")
chk = []
for f in FRACS:
    for mth in METH:
        mine = float(np.mean([val(Y[r], sc(mth, f, r)) for r in STEPS]))
        ref = lc[(lc.method == mth) & np.isclose(lc.frac, f) & (lc.step.astype(str) == "mean")]["est"].iloc[0]
        chk.append(dict(what=f"AUROC {mth}", frac=f, recomputed=mine, sweep1500=ref, abs_diff=abs(mine - ref)))
    mine = float(np.mean([val(Y[r], sc("P75", f, r)) - val(Y[r], sc("R5", f, r)) for r in STEPS]))
    ref = dif0[(dif0.comparison == "P75-R5") & np.isclose(dif0.frac, f) & (dif0.step.astype(str) == "mean")]["est"].iloc[0]
    chk.append(dict(what="P75-R5", frac=f, recomputed=mine, sweep1500=ref, abs_diff=abs(mine - ref)))
chk = pd.DataFrame(chk)
print("check: max abs diff vs sweep1500 =", float(chk.abs_diff.max()), flush=True)

SETS = {"all_1500": np.arange(len(Y[0])), "new_1000": np.where(EP[0] >= N_MAIN)[0]}
diff, con = [], []
for name, idx in SETS.items():
    Yh = {r: Y[r][idx] for r in STEPS}
    Bh = cluster_boot_idx(G[0][idx], 1000, 0)[0]
    for A, B in COMPS:
        for f in FRACS:
            est, bt = bt_matrix(Yh, [(1, {r: sc(A, f, r, idx) for r in STEPS}), (-1, {r: sc(B, f, r, idx) for r in STEPS})], Bh, STEPS)
            diff += [dict(eval_set=name, comparison=f"{A}-{B}", frac=f, n_train_mean=np.mean(NT[(0, "P75", f, 0)]), **row)
                     for row in rows_from(est, bt, STEPS)]
        for lo in (FRACS[1], FRACS[0]):
            terms = [(1, {r: sc(A, 1.0, r, idx) for r in STEPS}), (-1, {r: sc(B, 1.0, r, idx) for r in STEPS}),
                     (-1, {r: sc(A, lo, r, idx) for r in STEPS}), (1, {r: sc(B, lo, r, idx) for r in STEPS})]
            est, bt = bt_matrix(Yh, terms, Bh, STEPS)
            con += [dict(eval_set=name, contrast=f"({A}-{B})[f=1] - ({A}-{B})[f={lo:.4g}]", **row) for row in rows_from(est, bt, STEPS)]
    print("done", name, f"{time.time() - T0:.0f}s", flush=True)
diff, con = pd.DataFrame(diff), pd.DataFrame(con)

pd.set_option("display.width", 220)
for nm, t in (("check", chk), ("differences (mean step)", diff[diff.step.astype(str) == "mean"]),
              ("contrasts (mean step)", con[con.step.astype(str) == "mean"])):
    print(f"\n== {nm}"); print(t.round(3).to_string(index=False))
for k, t in (("diff", diff), ("contrast", con), ("check", chk)): t.to_csv(OUT[k], index=False)
print("end", datetime.datetime.now().isoformat(timespec="seconds"), f"{time.time() - T0:.0f}s", flush=True)
