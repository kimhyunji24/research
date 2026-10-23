"""Position exploration analysis (docs/position_plan_2026-10-24.md), post hoc: written AFTER the sweep1500 results had been read.
Probes on the five extracted positions (data/positions_3b, scripts/2026-10-24/extract_positions.py) compared with P75 (prompt last
token, layer 27), R5 and R4. Same cross-fit and bootstrap as scripts/2026-10-23/sweep1500.py and scripts/2026-10-24/posthoc_r4.py
(StratifiedGroupKFold by scene 5-fold x 2 repeats, nested C, 1,000 scene-cluster bootstrap replicates, seed 0, step mean), training
fraction f = 1 only (about 1,200 episodes). Layer 27 positions are the comparison family (position - P75, Holm over the 5 positions);
layer 18 positions are descriptive (AUROC only). Evaluation sets: all 1,500 episodes and the new 1,000 only.

    .venv/bin/python scripts/2026-10-24/positions_analysis.py --smoke      (labels shuffled by episode; writes /tmp/pos_smoke/, not a result)
    .venv/bin/python scripts/2026-10-24/positions_analysis.py              (real labels; run once, after approval)
Writes results/positions_{lc,diff,check}.csv; stops if any exists.
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

N_MAIN = 500
RUNS = ("data/main_3b/Qwen2.5-3B-Instruct", "data/ext1000_3b/Qwen2.5-3B-Instruct")
POS_DIR = "data/positions_3b"
POS = ("C1", "C2", "C3", "C4", "C5")  # c1_assistant, c2_obs_last, c3_obs_mean, c4_hist_mean, c5_prompt_mean
STEPS = [0, 1, 2]


def lin(C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))


def _fold(X, y, g, make, tr, te, rep):  # sweep1500._fold with f = 1
    best, best_s = None, -1.0
    for C in C_GRID:
        sc = []
        for itr, iva in StratifiedGroupKFold(3, shuffle=True, random_state=rep).split(np.zeros(len(tr)), y[tr], g[tr]):
            a, b = tr[itr], tr[iva]
            if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2: continue
            sc.append(roc_auc_score(y[b], make(C).fit(_take(X, a), y[a]).predict_proba(_take(X, b))[:, 1]))
        if sc and np.mean(sc) > best_s: best, best_s = C, float(np.mean(sc))
    return te, make(best).fit(_take(X, tr), y[tr]).predict_proba(_take(X, te))[:, 1]


def run_many(specs, n_jobs):  # sweep1500.run_many with f = 1
    flat = [(key, X, y, g, make, tr, te, rep) for key, (X, y, g, make) in specs.items()
            for rep in (0, 1) for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, g)]
    out = Parallel(n_jobs=n_jobs)(delayed(_fold)(X, y, g, make, tr, te, rep) for key, X, y, g, make, tr, te, rep in flat)
    S = {key: np.zeros(len(specs[key][1])) for key in specs}
    for (key, *_), (te, p) in zip(flat, out): S[key][te] += p / 2
    return S


def val(y, s, i=None):
    if i is not None:
        y, s = y[i], s[i]
        if len(np.unique(y)) < 2: return np.nan
    return float(roc_auc_score(y, s))


def bt_matrix(Y, terms, boot):
    est = np.array([sum(sg * val(Y[r], T[r]) for sg, T in terms) for r in STEPS])
    bt = np.array([[sum(sg * val(Y[r], T[r], i) for sg, T in terms) for r in STEPS] for i in boot])
    return est, bt


def rows_from(est, bt):  # two-sided bootstrap p, as in sweep1500/posthoc_r4
    out = []
    for j, lab in list(enumerate(STEPS)) + [(None, "mean")]:
        e = est.mean() if j is None else est[j]
        b = np.nanmean(bt, 1) if j is None else bt[:, j]
        b = b[~np.isnan(b)]
        out.append(dict(step=lab, est=e, ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5), n_boot=len(b),
                        p_boot=float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))))
    return out


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def load_all():  # sweep1500.load_all, plus the original (src, episode, step) key and the position hidden states
    dfs, hs = [], []
    for k, path in enumerate(RUNS):
        d, hid = load_run(path)
        d = d[d["step"] <= 2].reset_index(drop=True)
        hs.append(np.asarray(hid[d["hid_idx"].to_numpy()], dtype=np.float16))
        d["src"] = ("main_3b", "ext1000_3b")[k]
        d["ep_orig"] = d["episode"]
        if k == 1: d["episode"] = d["episode"] + N_MAIN
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    assert df["task_id"].nunique() == df["episode"].nunique() == 1500, "games are not unique across the two runs"
    pr = pd.read_csv(os.path.join(POS_DIR, "rows.csv"))
    pr["pos_row"] = np.arange(len(pr))
    m = df.merge(pr[["run", "episode", "step", "hid_idx", "pos_row"]].rename(columns={"run": "src", "episode": "ep_orig", "hid_idx": "hid_pos"}),
                 on=["src", "ep_orig", "step"], how="left", validate="one_to_one")
    assert m["pos_row"].notna().all() and (m["hid_pos"] == m["hid_idx"]).all(), "position rows do not match the runs"
    P = np.load(os.path.join(POS_DIR, "hidden.npy"), mmap_mode="r")[m["pos_row"].to_numpy().astype(int)]
    m["hid_idx"] = np.arange(len(m))
    return m, np.concatenate(hs), np.asarray(P, dtype=np.float16)


ap = argparse.ArgumentParser()
ap.add_argument("--n_jobs", type=int, default=24)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--null", type=int, help="permutation null: N label shuffles (seeds 1..N), step-mean point estimates only")
a = ap.parse_args()
if a.null:  # plan change history 2026-10-24: permutation null before the real-label run
    OUT_NULL = "results/positions_null.csv"
    if os.path.exists(OUT_NULL): raise SystemExit(f"output exists: {OUT_NULL}")
    print("start", datetime.datetime.now().isoformat(timespec="seconds"), "HEAD",
          subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
          f"NULL ({a.null} label shuffles, not a result)", flush=True)
    T0 = time.time()
    df, hid, PH = load_all()
    X4, X5 = baseline_frame(df, 4), baseline_frame(df, 5)
    num4, num5 = baseline_columns(X4, 4), baseline_columns(X5, 5)
    ey = df.groupby("episode")["y"].first()
    G, EP, D = {}, {}, {}
    for r in STEPS:
        m = (df["step"] == r).to_numpy(); idx = np.where(m)[0]
        EP[r] = df.loc[m, "episode"].to_numpy(); G[r] = df.loc[m, "scene"].to_numpy()
        H = np.asarray(hid[df["hid_idx"].to_numpy()[idx]], dtype=np.float32)
        D[r] = {"P75": (H[:, 2], lin),
                "R4": (X4.iloc[idx].reset_index(drop=True), lambda C: make_baseline(4, num4, C)),
                "R5": (X5.iloc[idx].reset_index(drop=True), lambda C: make_baseline(5, num5 + ["text"], C))}
        for j, p in enumerate(POS): D[r][f"{p}_L27"] = (np.asarray(PH[idx, j, 1], dtype=np.float32), lin)
    SETS = {"all_1500": np.arange(1500), "new_1000": np.where(EP[0] >= N_MAIN)[0]}
    out = []
    for seed in range(1, a.null + 1):
        perm = pd.Series(np.random.default_rng(seed).permutation(ey.to_numpy()), index=ey.index)
        Y = {r: perm.loc[EP[r]].to_numpy().astype(int) for r in STEPS}
        S = run_many({(r, mth): (D[r][mth][0], Y[r], G[r], D[r][mth][1]) for r in STEPS for mth in D[0]}, a.n_jobs)
        for name, idx in SETS.items():
            auc = {mth: float(np.mean([val(Y[r][idx], S[(r, mth)][idx]) for r in STEPS])) for mth in D[0]}
            for p in POS:
                for B in ("P75", "R5", "R4"):
                    out.append(dict(seed=seed, eval_set=name, comparison=f"{p}_L27-{B}", est=auc[f"{p}_L27"] - auc[B]))
        print(f"  seed {seed}/{a.null} done, {time.time() - T0:.0f}s", flush=True)
    pd.DataFrame(out).to_csv(OUT_NULL, index=False)
    print("end", datetime.datetime.now().isoformat(timespec="seconds"), f"{time.time() - T0:.0f}s", flush=True)
    raise SystemExit(0)
OUT ={k: (f"/tmp/pos_smoke/positions_{k}.csv" if a.smoke else f"results/positions_{k}.csv") for k in ("lc", "diff", "check")}
if a.smoke: os.makedirs("/tmp/pos_smoke", exist_ok=True)
for p in OUT.values():
    if os.path.exists(p): raise SystemExit(f"output exists: {p}")
print("start", datetime.datetime.now().isoformat(timespec="seconds"), "HEAD",
      subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
      "SMOKE (labels shuffled, not a result)" if a.smoke else "REAL LABELS", flush=True)
T0 = time.time()
df, hid, PH = load_all()
assert len(df) == 4500 and PH.shape == (4500, 5, 2, 2048), (len(df), PH.shape)
if a.smoke:  # shuffle the failure label across episodes (same permutation at every step)
    ey = df.groupby("episode")["y"].first()
    perm = pd.Series(np.random.default_rng(12345).permutation(ey.to_numpy()), index=ey.index)
    df["y"] = df["episode"].map(perm).astype(int)
X4, X5 = baseline_frame(df, 4), baseline_frame(df, 5)
num4, num5 = baseline_columns(X4, 4), baseline_columns(X5, 5)
Y, G, EP, D = {}, {}, {}, {}
for r in STEPS:
    m = (df["step"] == r).to_numpy(); idx = np.where(m)[0]
    EP[r] = df.loc[m, "episode"].to_numpy(); G[r] = df.loc[m, "scene"].to_numpy(); Y[r] = df.loc[m, "y"].to_numpy()
    H = np.asarray(hid[df["hid_idx"].to_numpy()[idx]], dtype=np.float32)
    D[r] = {"P75": (H[:, 2], lin),
            "R4": (X4.iloc[idx].reset_index(drop=True), lambda C: make_baseline(4, num4, C)),
            "R5": (X5.iloc[idx].reset_index(drop=True), lambda C: make_baseline(5, num5 + ["text"], C))}
    for j, p in enumerate(POS):
        D[r][f"{p}_L27"] = (np.asarray(PH[idx, j, 1], dtype=np.float32), lin)
        D[r][f"{p}_L18"] = (np.asarray(PH[idx, j, 0], dtype=np.float32), lin)
for r in STEPS: assert (EP[r] == EP[STEPS[0]]).all() and len(EP[r]) == 1500
METH = list(D[0])
print("rows per step", {r: len(Y[r]) for r in STEPS}, "scenes", len(np.unique(G[0])), "methods", METH, flush=True)
specs = {(r, mth): (D[r][mth][0], Y[r], G[r], D[r][mth][1]) for r in STEPS for mth in METH}
t0 = time.time(); S = run_many(specs, a.n_jobs)
print(f"crossfit: {len(specs)} runs, {time.time() - t0:.0f}s", flush=True)

chk = []
if not a.smoke:  # recomputed P75/R4/R5 at f = 1 against sweep1500_lc.csv (step mean)
    lc0 = pd.read_csv("results/sweep1500_lc.csv")
    for mth in ("P75", "R4", "R5"):
        mine = float(np.mean([val(Y[r], S[(r, mth)]) for r in STEPS]))
        ref = lc0[(lc0.method == mth) & np.isclose(lc0.frac, 1.0) & (lc0.step.astype(str) == "mean")]["est"].iloc[0]
        chk.append(dict(what=f"AUROC {mth} f=1", recomputed=mine, sweep1500=ref, abs_diff=abs(mine - ref)))
    print("check: max abs diff vs sweep1500 =", max(c["abs_diff"] for c in chk), flush=True)

SETS = {"all_1500": np.arange(len(Y[0])), "new_1000": np.where(EP[0] >= N_MAIN)[0]}
lc, diff = [], []
for name, idx in SETS.items():
    Yh = {r: Y[r][idx] for r in STEPS}
    Bh = cluster_boot_idx(G[0][idx], 1000, 0)[0]
    for mth in METH:
        est, bt = bt_matrix(Yh, [(1, {r: S[(r, mth)][idx] for r in STEPS})], Bh)
        lc += [dict(eval_set=name, method=mth, **row) for row in rows_from(est, bt)]
    for p in POS:
        for B in ("P75", "R5", "R4"):
            A = f"{p}_L27"
            est, bt = bt_matrix(Yh, [(1, {r: S[(r, A)][idx] for r in STEPS}), (-1, {r: S[(r, B)][idx] for r in STEPS})], Bh)
            diff += [dict(eval_set=name, comparison=f"{A}-{B}", **row) for row in rows_from(est, bt)]
    print("done", name, f"{time.time() - T0:.0f}s", flush=True)
lc, diff = pd.DataFrame(lc), pd.DataFrame(diff)
diff["family"] = np.where(diff.comparison.str.endswith("-P75"), "position-P75", "secondary")
diff["p_holm"] = np.nan
for name in SETS:  # Holm over the five position - P75 comparisons, step mean, per evaluation set
    k = (diff.eval_set == name) & (diff.family == "position-P75") & (diff.step.astype(str) == "mean")
    diff.loc[k, "p_holm"] = holm(diff.loc[k, "p_boot"].to_numpy())
if not a.smoke:  # permutation null rule (plan change history 2026-10-24): the real difference must exceed all null values
    nul = pd.read_csv("results/positions_null.csv")
    agg = nul.groupby(["eval_set", "comparison"]).est.agg(null_n="count", null_mean="mean", null_max="max").reset_index()
    diff = diff.merge(agg, on=["eval_set", "comparison"], how="left")
    diff.loc[diff.step.astype(str) != "mean", ["null_n", "null_mean", "null_max"]] = np.nan
    diff["above_all_null"] = np.where(diff.step.astype(str) == "mean", diff.est > diff.null_max, np.nan)
chk = pd.DataFrame(chk)

if a.smoke:  # pipeline check only: shapes, CI sanity, and how far shuffled-label AUROCs are from 0.5
    mean = lc[lc.step.astype(str) == "mean"]
    print("\n== SMOKE (shuffled labels, not a result)")
    print("lc rows", len(lc), "diff rows", len(diff), "NaN est", int(lc.est.isna().sum() + diff.est.isna().sum()),
          "| CI ordered", bool(((lc.ci_lo <= lc.est) & (lc.est <= lc.ci_hi)).all() and ((diff.ci_lo <= diff.est) & (diff.est <= diff.ci_hi)).all()))
    print("max |AUROC - 0.5| over methods (step mean):", round(float((mean.est - 0.5).abs().max()), 3),
          "| share of diff CIs (step mean) containing 0:",
          round(float(((diff.step.astype(str) == "mean") & (diff.ci_lo <= 0) & (diff.ci_hi >= 0)).sum() / (diff.step.astype(str) == "mean").sum()), 3),
          "| Holm rows", int(diff.p_holm.notna().sum()))
else:
    pd.set_option("display.width", 220)
    for nm, t in (("check", chk), ("AUROC (mean step)", lc[lc.step.astype(str) == "mean"]),
                  ("differences (mean step)", diff[diff.step.astype(str) == "mean"])):
        print(f"\n== {nm}"); print(t.round(3).to_string(index=False))
for k, t in (("lc", lc), ("diff", diff), ("check", chk)): t.to_csv(OUT[k], index=False)
print("end", datetime.datetime.now().isoformat(timespec="seconds"), f"{time.time() - T0:.0f}s", flush=True)
