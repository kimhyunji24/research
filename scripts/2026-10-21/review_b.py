"""3차 리뷰 B 추가 분석(탐색, docs/review_b_plan_2026-10-21.md): B1 learning curve, B2 낮은 오탐 구간 지표,
B4 게임 단위 대 장면 단위 분할, B5 과제 수준 CI 포함률, B7 신경망 텍스트 인코더 baseline. 1회 실행.
main_3b 앞 250 episode, step 0-2만 쓴다(홀드아웃·Qwen3-4B·ReAct·게임 내 데이터 안 씀).

    .venv/bin/python scripts/2026-10-21/review_b.py --emb data/main_3b/text_embed/bge-base-en-v1.5 [--n_jobs 20] [--smoke]
출력: results/review_b_{lc,lc_diff,lowfpr,lowfpr_diff,split,coverage,textenc,textenc_diff}.csv, review_b_scores.csv.
--smoke: 라벨을 섞고 step 0, 작은 설정으로 파이프라인만 확인(결과 아님, 파일 안 씀).
"""
import argparse, datetime, os, subprocess, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "probes"))
from lib import C_GRID, _take, baseline_columns, baseline_frame, cluster_boot_idx, load_run, make_baseline  # noqa: E402

OUT = ["lc", "lc_diff", "lowfpr", "lowfpr_diff", "split", "coverage", "textenc", "textenc_diff", "scores"]
FRACS, N_DRAW, PERM_SEEDS = (0.25, 0.5, 0.75, 1.0), 5, range(1001, 1021)


def lin(C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))


def _fold(X, y, g, make, tr, te, rep, k, f, draw):
    """probe_strength.py의 _fold와 같은 절차(inner 3-fold, random_state=rep). f < 1이면 바깥 학습 fold에서 라벨 층화로 뽑는다."""
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
    return te, make(best).fit(_take(X, tr), y[tr]).predict_proba(_take(X, te))[:, 1]


def jobs(y, g):
    return [(tr, te, rep, k) for rep in (0, 1)
            for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, g))]


def run_many(specs, n_jobs):
    """specs: key -> (X, y, g, make, f, draw). 모든 fold를 한 번에 병렬로 돌리고 key별 점수(반복 평균)를 돌려준다."""
    flat = [(key, X, y, g, make, tr, te, rep, k, f, draw) for key, (X, y, g, make, f, draw) in specs.items()
            for tr, te, rep, k in jobs(y, g)]
    out = Parallel(n_jobs=n_jobs)(delayed(_fold)(X, y, g, make, tr, te, rep, k, f, draw)
                                  for key, X, y, g, make, tr, te, rep, k, f, draw in flat)
    S = {key: np.zeros(len(specs[key][1])) for key in specs}
    for (key, *_), (te, p) in zip(flat, out): S[key][te] += p / 2
    return S


def tpr_at(t):
    def m(y, s):
        fpr, tpr, _ = roc_curve(y, s); return float(tpr[fpr <= t].max())
    return m


METRICS = {"auroc": roc_auc_score, "tpr_fpr05": tpr_at(0.05), "tpr_fpr10": tpr_at(0.10),
           "pauroc10": lambda y, s: roc_auc_score(y, s, max_fpr=0.1), "auprc": average_precision_score}


def val(metric, y, ss, i=None):
    """ss: 점수 배열 목록(뽑기). 뽑기마다 지표를 내고 평균한다."""
    if i is not None:
        y = y[i]
        if len(np.unique(y)) < 2: return np.nan
    return float(np.mean([metric(y, s if i is None else s[i]) for s in ss]))


def summarize(metric, Y, A, B, boot, steps):
    """A(- B) 지표의 step별·step 평균 점추정과 장면 bootstrap 95% CI, 양측 bootstrap p(B가 있을 때)."""
    est = np.array([val(metric, Y[r], A[r]) - (val(metric, Y[r], B[r]) if B else 0) for r in steps])
    bt = np.array([[val(metric, Y[r], A[r], i) - (val(metric, Y[r], B[r], i) if B else 0) for r in steps] for i in boot])
    out = []
    for j, lab in list(enumerate(steps)) + [(None, "mean")]:
        e = est.mean() if j is None else est[j]
        b = np.nanmean(bt, 1) if j is None else bt[:, j]
        b = b[~np.isnan(b)]
        row = dict(step=lab, est=e, ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5), n_boot=len(b))
        if B: row["p_boot"] = float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))
        out.append(row)
    return out


def holm(d):
    d = d.copy(); order = d["p_boot"].sort_values(kind="stable").index; k, run = len(order), 0.0
    for j, i in enumerate(order):
        run = max(run, min(1.0, (k - j) * d.loc[i, "p_boot"])); d.loc[i, "p_holm"] = run
    d["verdict"] = np.where(d.ci_lo > 0, "superior", np.where(d.ci_hi < 0, "inferior", "inconclusive"))
    return d


ap = argparse.ArgumentParser()
ap.add_argument("--path", default="data/main_3b/Qwen2.5-3B-Instruct")
ap.add_argument("--emb", required=True)
ap.add_argument("--n_jobs", type=int, default=20)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
paths = {k: f"results/review_b_{k}.csv" for k in OUT}
if not a.smoke:
    for p in paths.values():
        if os.path.exists(p): raise SystemExit(f"출력 파일이 이미 있다: {p}")
print("start", datetime.datetime.now().isoformat(timespec="seconds"), "HEAD",
      subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
      "smoke" if a.smoke else "", flush=True)
T0 = time.time()
df, hid = load_run(a.path)
df = df[(df["episode"] < 250) & (df["step"] <= 2)].reset_index(drop=True)
assert len(df) == 750, len(df)
emb_rows = pd.read_csv(os.path.join(a.emb, "rows.csv")); emb = np.load(os.path.join(a.emb, "emb.npy"))
assert (emb_rows[["episode", "step", "hid_idx"]].to_numpy() == df[["episode", "step", "hid_idx"]].to_numpy()).all()
X4, X5 = baseline_frame(df, 4), baseline_frame(df, 5)
num4, num5 = baseline_columns(X4, 4), baseline_columns(X5, 5)
STEPS = [0] if a.smoke else [0, 1, 2]
fracs, n_draw, perm_seeds = ((0.5, 1.0), 1, range(1001, 1003)) if a.smoke else (FRACS, N_DRAW, PERM_SEEDS)
Y, G, GT, D, EP = {}, {}, {}, {}, {}
for r in STEPS:
    m = (df["step"] == r).to_numpy(); idx = np.where(m)[0]
    EP[r] = df.loc[m, "episode"].to_numpy()
    Y[r] = df.loc[m, "y"].to_numpy(); G[r] = df.loc[m, "scene"].to_numpy(); GT[r] = df.loc[m, "task_id"].to_numpy()
    if a.smoke: Y[r] = np.random.default_rng(0).permutation(Y[r])
    H = np.asarray(hid[df["hid_idx"].to_numpy()[idx]], dtype=np.float32)
    E = emb[idx]
    D[r] = {"P75": (H[:, 2], lin), "P50": (H[:, 1], lin),
            "R4": (X4.iloc[idx].reset_index(drop=True), lambda C: make_baseline(4, num4, C)),
            "R5": (X5.iloc[idx].reset_index(drop=True), lambda C: make_baseline(5, num5 + ["text"], C)),
            "E": (E, lin), "R5e": (np.hstack([X4.iloc[idx][num4].to_numpy(float), E]), lin)}
for r in STEPS: assert (EP[r] == EP[STEPS[0]]).all()  # 세 step 모두 같은 250 episode, 같은 순서 -> 같은 장면 재표집 공유
BOOT = cluster_boot_idx(G[STEPS[0]], 1000, 0)[0]
LC = ["P75", "P50", "R4", "R5"]

# ---------------------------------------------------------------- B1·B2·B4·B7 교차적합
specs = {}
for r in STEPS:
    for meth in LC:
        X, make = D[r][meth]
        for f in fracs:
            for draw in (range(n_draw) if f < 1 else [0]):
                specs[("lc", r, meth, f, draw)] = (X, Y[r], G[r], make, f, draw)
    for meth in ("P75", "R5"):
        X, make = D[r][meth]; specs[("game", r, meth)] = (X, Y[r], GT[r], make, 1.0, 0)
    for meth in ("E", "R5e"):
        X, make = D[r][meth]; specs[("lc", r, meth, 1.0, 0)] = (X, Y[r], G[r], make, 1.0, 0)
t0 = time.time(); S = run_many(specs, a.n_jobs)
print(f"crossfit B1/B4/B7: {len(specs)} runs, {time.time() - t0:.0f}s", flush=True)


def sc(meth, f, r): return [S[("lc", r, meth, f, d)] for d in (range(n_draw) if f < 1 else [0])]


# B1
lc, lcd = [], []
for f in fracs:
    for meth in LC:
        for row in summarize(roc_auc_score, Y, {r: sc(meth, f, r) for r in STEPS}, None, BOOT, STEPS):
            lc.append(dict(frac=f, method=meth, n_draw=len(sc(meth, f, STEPS[0])), **row))
    for A in ("P75", "P50"):
        for row in summarize(roc_auc_score, Y, {r: sc(A, f, r) for r in STEPS}, {r: sc("R5", f, r) for r in STEPS}, BOOT, STEPS):
            lcd.append(dict(frac=f, comparison=f"{A}-R5", **row))
lc, lcd = pd.DataFrame(lc), holm(pd.DataFrame(lcd))
# B2
lf, lfd = [], []
for mname, metric in METRICS.items():
    for meth in LC:
        for row in summarize(metric, Y, {r: sc(meth, 1.0, r) for r in STEPS}, None, BOOT, STEPS):
            lf.append(dict(metric=mname, method=meth, **row))
    for A in ("P75", "P50"):
        for row in summarize(metric, Y, {r: sc(A, 1.0, r) for r in STEPS}, {r: sc("R5", 1.0, r) for r in STEPS}, BOOT, STEPS):
            lfd.append(dict(metric=mname, comparison=f"{A}-R5", **row))
lf, lfd = pd.DataFrame(lf), holm(pd.DataFrame(lfd))
# B4: 게임 단위 - 장면 단위(같은 방법), 장면 bootstrap
sp = []
for meth in ("P75", "R5"):
    for kind, B in (("game_split", None), ("game_minus_scene", {r: sc(meth, 1.0, r) for r in STEPS})):
        for row in summarize(roc_auc_score, Y, {r: [S[("game", r, meth)]] for r in STEPS}, B, BOOT, STEPS):
            sp.append(dict(method=meth, quantity=kind, **row))
sp = pd.DataFrame(sp)
sp = pd.concat([sp[sp.quantity == "game_split"], holm(sp[sp.quantity == "game_minus_scene"])])
# B7
te, ted = [], []
for meth in ("E", "R5e"):
    for row in summarize(roc_auc_score, Y, {r: sc(meth, 1.0, r) for r in STEPS}, None, BOOT, STEPS):
        te.append(dict(method=meth, **row))
for A, B_ in (("R5e", "R5"), ("P75", "R5e"), ("P75", "E")):
    for row in summarize(roc_auc_score, Y, {r: sc(A, 1.0, r) for r in STEPS}, {r: sc(B_, 1.0, r) for r in STEPS}, BOOT, STEPS):
        ted.append(dict(comparison=f"{A}-{B_}", **row))
te, ted = pd.DataFrame(te), holm(pd.DataFrame(ted))

# ---------------------------------------------------------------- B5 라벨 섞은 데이터의 CI 포함률
t0 = time.time(); pspecs, YP = {}, {}
for seed in perm_seeds:
    for r in STEPS:
        YP[(seed, r)] = np.random.default_rng(seed).permutation(Y[r])
        for meth in ("P75", "R5"):
            X, make = D[r][meth]; pspecs[(seed, r, meth)] = (X, YP[(seed, r)], G[r], make, 1.0, 0)
PS = run_many(pspecs, a.n_jobs)
print(f"crossfit B5: {len(pspecs)} runs, {time.time() - t0:.0f}s", flush=True)
cov = []
for seed in perm_seeds:
    for r in STEPS:
        y = YP[(seed, r)]
        for q, A, B_ in (("P75", "P75", None), ("R5", "R5", None), ("P75-R5", "P75", "R5")):
            row = summarize(roc_auc_score, {r: y}, {r: [PS[(seed, r, A)]]}, {r: [PS[(seed, r, B_)]]} if B_ else None, BOOT, [r])[0]
            null = 0.0 if B_ else 0.5
            cov.append(dict(seed=seed, step=r, quantity=q, est=row["est"], ci_lo=row["ci_lo"], ci_hi=row["ci_hi"],
                            covers=int(row["ci_lo"] <= null <= row["ci_hi"])))
cov = pd.DataFrame(cov)


def wilson(k, n, z=1.959964):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


cs = []
for q, gq in cov.groupby("quantity", sort=False):
    k, n = int(gq.covers.sum()), len(gq); lo, hi = wilson(k, n)
    cs.append(dict(seed="all", step="all", quantity=q, est=gq.est.mean(), ci_lo=np.nan, ci_hi=np.nan, covers=k, n=n,
                   coverage=k / n, wilson_lo=lo, wilson_hi=hi))
cov = pd.concat([cov, pd.DataFrame(cs)], ignore_index=True)

pd.set_option("display.width", 250)
for name, t in (("B1 learning curve", lc), ("B1 diff", lcd), ("B2 low-FPR", lf), ("B2 diff", lfd), ("B4 split", sp),
                ("B5 coverage (summary)", cov[cov.seed == "all"]), ("B7 text encoder", te), ("B7 diff", ted)):
    print(f"\n== {name}"); print(t.round(3).to_string(index=False))
if not a.smoke:
    p75 = lc[(lc.method == "P75") & (lc.frac == 1.0) & (lc.step != "mean")].est.round(3).tolist()
    print("\nreproduction P75 f=1:", p75, "expected [0.666, 0.721, 0.795]", "OK" if p75 == [0.666, 0.721, 0.795] else "MISMATCH")
    for k, t in (("lc", lc), ("lc_diff", lcd), ("lowfpr", lf), ("lowfpr_diff", lfd), ("split", sp), ("coverage", cov),
                 ("textenc", te), ("textenc_diff", ted)):
        t.to_csv(paths[k], index=False)
    srows = [dict(step=r, episode=e, y=yy, method=f"{meth}_f{f}_d{d}", score=s) for (kind, r, meth, *rest), v in S.items()
             for f, d in [rest if kind == "lc" else ("game", 0)] for e, yy, s in zip(EP[r], Y[r], v)]
    pd.DataFrame(srows).to_csv(paths["scores"], index=False)
print("end", datetime.datetime.now().isoformat(timespec="seconds"), f"{time.time() - T0:.0f}s", flush=True)
