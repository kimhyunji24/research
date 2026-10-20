"""선행 방법식 baseline 비교(탐색, docs/prior_methods_plan_2026-10-20.md). 1회 실행.
사용: python scripts/2026-10-20/prior_methods.py --out results/prior_methods_3b.csv [--check_only]"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "probes"))
from lib import (C_GRID, baseline_columns, baseline_frame, cluster_boot_idx, load_run, make_baseline, make_probe,
                 paired_diff, round_eval)
from baselines import user_content

ERR = ("nothing happens", "can't", "cannot")


def current_obs(prompt):
    obs = re.findall(r"Observation: (.*)", user_content(prompt))
    return obs[-1].lower() if obs else ""


def prior_features(df):
    """행마다 고른 action의 로그확률·토큰 수, 후보 분포 특징. 직전 step 값은 episode 안에서 shift(step 0은 0)."""
    lp, mean_lp, ntok, ent, margin = [], [], [], [], []
    for r in df.itertuples():
        i = r.cands.index(r.action)
        s = np.asarray(r.cand_logp, float); n = np.asarray(r.cand_ntok, float)
        p = np.exp(s - s.max()); p /= p.sum()
        lp.append(s[i]); mean_lp.append(s[i] / n[i]); ntok.append(n[i])
        ent.append(float(-(p * np.log(p + 1e-300)).sum()))
        top = np.sort(s)[::-1]; margin.append(float(top[0] - top[1]) if len(top) > 1 else 0.0)
    f = pd.DataFrame(dict(chosen_logp=lp, chosen_mean_lp=mean_lp, chosen_ntok=ntok, cand_entropy=ent, top12_margin=margin),
                     index=df.index)
    f["prefix_len"] = df["prompt_tokens"].astype(float)
    f["err_count"] = df["prompt"].map(lambda p: sum(current_obs(p).count(k) for k in ERR)).astype(float)
    g = df["episode"]
    for c in ("chosen_logp", "chosen_mean_lp", "chosen_ntok"):
        f[c + "_prev"] = f.groupby(g)[c].shift(1).fillna(0.0)
    return f


BEHAV = ["chosen_logp", "chosen_mean_lp", "chosen_ntok", "chosen_logp_prev", "chosen_mean_lp_prev", "chosen_ntok_prev", "prefix_len", "err_count"]
LOGP = ["cand_entropy", "top12_margin", "p_chosen", "n_cand", "chosen_mean_lp", "chosen_mean_lp_prev"]

ap = argparse.ArgumentParser()
ap.add_argument("--path", default="data/main_3b/Qwen2.5-3B-Instruct")
ap.add_argument("--layer", type=int, default=2)
ap.add_argument("--rounds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--n_perm", type=int, default=100)
ap.add_argument("--n_jobs", type=int, default=24)
ap.add_argument("--out", required=True)
ap.add_argument("--check_only", action="store_true", help="특징만 만들고 분포를 출력(라벨 미사용)")
a = ap.parse_args()

df, hid = load_run(a.path)
df = df[df["episode"] < 250].reset_index(drop=True)
F = prior_features(df)
df = pd.concat([df, F], axis=1)
if a.check_only:
    print(f"rows={len(df)} episodes={df.episode.nunique()}")
    print(df.loc[df.step.isin(a.rounds), BEHAV + ["cand_entropy", "top12_margin", "p_chosen", "n_cand"]].describe().T.round(3).to_string())
    print("err_count>0 rows:", int((df.err_count > 0).sum()), "| NaN:", int(df[BEHAV + LOGP].isna().sum().sum()))
    sys.exit(0)
if os.path.exists(a.out): raise SystemExit(f"출력 파일이 이미 있다: {a.out}")
H = np.asarray(hid[df["hid_idx"].to_numpy(), a.layer], dtype=np.float32)
d2 = baseline_frame(df, 2)
tt = [c for c in d2.columns if c.startswith("tt_")]
frames = {"B_behav": df[BEHAV].astype(float), "B_logp": df[LOGP].astype(float),
          "B_behav_task": pd.concat([df[BEHAV].astype(float), d2[tt]], axis=1)}
rows, scores = [], {}
for r in a.rounds:
    sc = df.loc[df.step == r, "scene"].to_numpy(); boot = cluster_boot_idx(sc, 1000, 0)
    common = dict(n_splits=5, n_repeats=2, boot=boot, seed=0, n_jobs=a.n_jobs)
    def run(name, X, make):
        res = round_eval(df, X, r, make, Cs=C_GRID, n_perm=a.n_perm, **common)
        scores[(r, name)] = (res.pop("_y"), res.pop("_s"), res.pop("_groups")); rows.append(dict(res, method=name))
        print(f"  step {r} {name}: auroc={res['auroc']:.3f}", flush=True)
    for name, X in frames.items():
        cols = list(X.columns)
        run(name, X, lambda C, cols=cols: make_baseline(4, cols, C))
    X5 = baseline_frame(df, 5); c5 = baseline_columns(X5, 5) + ["text"]
    run("baseline5", X5, lambda C: make_baseline(5, c5, C))
    run("probe_L75_linear", H, lambda C: make_probe("linear", None, C, 0))
out = pd.DataFrame(rows)
diffs = []
for r in a.rounds:
    for a_, b_ in (("probe_L75_linear", "B_behav"), ("probe_L75_linear", "B_logp"), ("B_behav_task", "B_behav"),
                   ("probe_L75_linear", "baseline5"), ("B_behav", "baseline5")):
        y, s, g = scores[(r, a_)]; _, sb, _ = scores[(r, b_)]
        est, lo, hi = paired_diff(y, s, sb, cluster_boot_idx(g, 1000, 0)[0])
        diffs.append(dict(step=r, a=a_, b=b_, diff=est, ci_lo=lo, ci_hi=hi,
                          verdict="superior" if lo > 0 else "inferior" if hi < 0 else "inconclusive"))
d = pd.DataFrame(diffs)
out.to_csv(a.out, index=False); d.to_csv(a.out.replace(".csv", "_diff.csv"), index=False)
pd.set_option("display.width", 220)
cols = ["step", "method", "n", "n_fail", "auroc", "ci_lo", "ci_hi", "p_perm", "C_median"]
print(out[[c for c in cols if c in out.columns]].round(3).to_string(index=False)); print(); print(d.round(3).to_string(index=False))
