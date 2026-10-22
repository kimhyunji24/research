"""Training-size sweep on 1,500 episodes = main_3b (500) + ext1000_3b (1,000) (exploratory; docs/collect_ext1000_plan_2026-10-21.md,
analysis section, with the 2026-10-23 additions). Same procedure as scripts/2026-10-21/sweep500.py: evaluation folds are fixed
(scene-grouped StratifiedGroupKFold, 5 folds x 2 repeats); only the share f of each outer training fold used for training changes
(about 50-1,200 episodes). One run.

    .venv/bin/python scripts/2026-10-23/sweep1500.py [--n_jobs 24] [--smoke]
Writes results/sweep1500_{lc,diff,contrast,half,margin}.csv; stops if any exists. --smoke: shuffled labels, step 0, small grid,
no files.
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

FRACS, N_DRAW = (1 / 24, 1 / 12, 1 / 6, 1 / 3, 2 / 3, 1.0), 5
MARGINS = (0.02, 0.05, 0.10)
N_MAIN = 500
OUT = {k: f"results/sweep1500_{k}.csv" for k in ("lc", "diff", "contrast", "half", "margin")}
RUNS = ("data/main_3b/Qwen2.5-3B-Instruct", "data/ext1000_3b/Qwen2.5-3B-Instruct")


def lin(C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))


def _fold(X, y, g, make, tr, te, rep, k, f, draw):  # same as sweep500._fold
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


def run_many(specs, n_jobs):
    flat = [(key, X, y, g, make, tr, te, rep, k, f, draw) for key, (X, y, g, make, f, draw) in specs.items()
            for rep in (0, 1) for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, g))]
    out = Parallel(n_jobs=n_jobs)(delayed(_fold)(X, y, g, make, tr, te, rep, k, f, draw)
                                  for key, X, y, g, make, tr, te, rep, k, f, draw in flat)
    S, NT = {key: np.zeros(len(specs[key][1])) for key in specs}, {key: [] for key in specs}
    for (key, *_), (te, ntr, p) in zip(flat, out): S[key][te] += p / 2; NT[key].append(ntr)
    return S, NT


def val(y, ss, i=None):
    if i is not None:
        y = y[i]
        if len(np.unique(y)) < 2: return np.nan
    return float(np.mean([roc_auc_score(y, s if i is None else s[i]) for s in ss]))


def bt_matrix(Y, terms, boot, steps):
    """terms: list of (sign, {step: [scores]}). Returns point (steps,) and bootstrap (n_boot, steps) of the signed sum."""
    est = np.array([sum(sg * val(Y[r], T[r]) for sg, T in terms) for r in steps])
    bt = np.array([[sum(sg * val(Y[r], T[r], i) for sg, T in terms) for r in steps] for i in boot])
    return est, bt


def rows_from(est, bt, steps, two_sided):
    out = []
    for j, lab in list(enumerate(steps)) + [(None, "mean")]:
        e = est.mean() if j is None else est[j]
        b = np.nanmean(bt, 1) if j is None else bt[:, j]
        b = b[~np.isnan(b)]
        row = dict(step=lab, est=e, ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5), n_boot=len(b))
        if two_sided: row["p_boot"] = float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))
        out.append(row)
    return out


def holm(d):
    d = d.copy(); order = d["p_boot"].sort_values(kind="stable").index; k, run = len(order), 0.0
    for j, i in enumerate(order):
        run = max(run, min(1.0, (k - j) * d.loc[i, "p_boot"])); d.loc[i, "p_holm"] = run
    d["verdict_ci"] = np.where(d.ci_lo > 0, "above0", np.where(d.ci_hi < 0, "below0", "includes0"))
    return d


def load_all():
    """main_3b (episodes 0-499) + ext1000_3b (episodes 500-1499), steps 0-2 only. Returns the step table and hidden (n, 4, 2048)."""
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
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if not a.smoke:
    for p in OUT.values():
        if os.path.exists(p): raise SystemExit(f"output exists: {p}")
print("start", datetime.datetime.now().isoformat(timespec="seconds"), "HEAD",
      subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
      "smoke" if a.smoke else "", flush=True)
T0 = time.time()
df, hid = load_all()
assert len(df) == 4500, len(df)
X4, X5 = baseline_frame(df, 4), baseline_frame(df, 5)
num4, num5 = baseline_columns(X4, 4), baseline_columns(X5, 5)
STEPS = [0] if a.smoke else [0, 1, 2]
fracs, n_draw = ((1 / 3, 1.0), 1) if a.smoke else (FRACS, N_DRAW)
Y, G, EP, D = {}, {}, {}, {}
for r in STEPS:
    m = (df["step"] == r).to_numpy(); idx = np.where(m)[0]
    EP[r] = df.loc[m, "episode"].to_numpy(); G[r] = df.loc[m, "scene"].to_numpy(); Y[r] = df.loc[m, "y"].to_numpy()
    if a.smoke: Y[r] = np.random.default_rng(0).permutation(Y[r])
    H = np.asarray(hid[df["hid_idx"].to_numpy()[idx]], dtype=np.float32)
    D[r] = {"P75": (H[:, 2], lin), "P50": (H[:, 1], lin),
            "R4": (X4.iloc[idx].reset_index(drop=True), lambda C: make_baseline(4, num4, C)),
            "R5": (X5.iloc[idx].reset_index(drop=True), lambda C: make_baseline(5, num5 + ["text"], C))}
for r in STEPS: assert (EP[r] == EP[STEPS[0]]).all() and len(EP[r]) == 1500
print("rows per step", {r: len(Y[r]) for r in STEPS}, "scenes", len(np.unique(G[STEPS[0]])), flush=True)
METH = ["P75", "P50", "R4", "R5"]
specs = {(r, mth, f, d): (D[r][mth][0], Y[r], G[r], D[r][mth][1], f, d)
         for r in STEPS for mth in METH for f in fracs for d in (range(n_draw) if f < 1 else [0])}
t0 = time.time(); S, NT = run_many(specs, a.n_jobs)
print(f"crossfit: {len(specs)} runs, {time.time() - t0:.0f}s", flush=True)


def sc(mth, f, r, idx=None):
    ss = [S[(r, mth, f, d)] for d in (range(n_draw) if f < 1 else [0])]
    return ss if idx is None else [s[idx] for s in ss]


BOOT = cluster_boot_idx(G[STEPS[0]], 1000, 0)[0]
lc, dif, con = [], [], []
for f in fracs:
    ntr = np.mean(NT[(STEPS[0], "P75", f, 0)])
    for mth in METH:
        est, bt = bt_matrix(Y, [(1, {r: sc(mth, f, r) for r in STEPS})], BOOT, STEPS)
        lc += [dict(frac=f, n_train_mean=ntr, method=mth, **row) for row in rows_from(est, bt, STEPS, False)]
    for A in ("P75", "P50"):
        est, bt = bt_matrix(Y, [(1, {r: sc(A, f, r) for r in STEPS}), (-1, {r: sc("R5", f, r) for r in STEPS})], BOOT, STEPS)
        dif += [dict(frac=f, n_train_mean=ntr, comparison=f"{A}-R5", **row) for row in rows_from(est, bt, STEPS, True)]
fmin, fmid = min(fracs), 1 / 3 if not a.smoke else fracs[0]  # fmid: about 400 training episodes (the range already seen)
for A in ("P75", "P50"):
    for lo, tag in ((fmid, "primary"), (fmin, "secondary")):
        terms = [(1, {r: sc(A, 1.0, r) for r in STEPS}), (-1, {r: sc("R5", 1.0, r) for r in STEPS}),
                 (-1, {r: sc(A, lo, r) for r in STEPS}), (1, {r: sc("R5", lo, r) for r in STEPS})]
        est, bt = bt_matrix(Y, terms, BOOT, STEPS)
        con += [dict(contrast=f"({A}-R5)[f=1] - ({A}-R5)[f={lo:.4g}]", role=tag if A == "P75" else "secondary", **row)
                for row in rows_from(est, bt, STEPS, True)]
lc, dif, con = pd.DataFrame(lc), holm(pd.DataFrame(dif)), pd.DataFrame(con)

# evaluation subsets: the 1,000 new episodes only (free of the depth selection made on the first 250 main_3b games) and, for contrast, main_3b's 500
half = []
for name, sel in (("new_1000", EP[STEPS[0]] >= N_MAIN), ("main_500", EP[STEPS[0]] < N_MAIN)):
    idx = np.where(sel)[0]
    Yh = {r: Y[r][idx] for r in STEPS}
    Bh = cluster_boot_idx(G[STEPS[0]][idx], 1000, 0)[0]
    for f in fracs:
        for A in ("P75", "P50"):
            est, bt = bt_matrix(Yh, [(1, {r: sc(A, f, r, idx) for r in STEPS}), (-1, {r: sc("R5", f, r, idx) for r in STEPS})], Bh, STEPS)
            half += [dict(eval_set=name, frac=f, n_train_mean=np.mean(NT[(STEPS[0], "P75", f, 0)]), comparison=f"{A}-R5", **row)
                     for row in rows_from(est, bt, STEPS, True)]
half = pd.DataFrame(half)

# equivalence-margin table (no margin is pre-registered; none is chosen): is the 95% CI inside (-d, +d), and is the lower limit above -d?
mg = []
tabs = [(dif.assign(eval_set="all_1500"), "all_1500")] + [(half[half.eval_set == s], s) for s in ("new_1000", "main_500")]
for t, name in tabs:
    for _, row in t.iterrows():
        for dlt in MARGINS:
            mg.append(dict(eval_set=name, frac=row["frac"], comparison=row["comparison"], step=row["step"], est=row["est"],
                           ci_lo=row["ci_lo"], ci_hi=row["ci_hi"], margin=dlt,
                           ci_inside_margin=bool(row["ci_lo"] > -dlt and row["ci_hi"] < dlt), lower_above_minus_margin=bool(row["ci_lo"] > -dlt)))
mg = pd.DataFrame(mg)

pd.set_option("display.width", 220)
for name, t in (("learning curve", lc), ("differences", dif), ("contrast", con), ("by evaluation set", half)):
    print(f"\n== {name}"); print(t.round(3).to_string(index=False))
if not a.smoke:
    for k, t in (("lc", lc), ("diff", dif), ("contrast", con), ("half", half), ("margin", mg)): t.to_csv(OUT[k], index=False)
print("end", datetime.datetime.now().isoformat(timespec="seconds"), f"{time.time() - T0:.0f}s", flush=True)
