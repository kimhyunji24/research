"""탐색적 추가 분석 (새 수집 없음): reader 간 짝지은 차이(H2), 후반 라운드와 조기 중단 지표, 시간축 프로브(H4 방향).

사용: python probes/extra_analysis.py data/main_3b/Qwen2.5-3B-Instruct --out results/extra_main_3b
모든 점수는 run_probes.py와 같은 절차(장면 단위 층화 교차적합 x2, nested C)로 만든다. 순열 대조군은 만들지 않는다(탐색적).
"""
import argparse, os, sys, itertools
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (C_GRID, baseline_columns, baseline_frame, ci, cluster_boot_idx, crossfit, load_run, make_baseline,
                 make_probe, paired_diff)

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--readers_dir", help="기본: <path 상위>/readers")
ap.add_argument("--out", required=True)
ap.add_argument("--late_max_step", type=int, default=10)
ap.add_argument("--temporal_steps", type=int, nargs="+", default=[1, 2, 3, 5])
ap.add_argument("--n_boot", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

df, hid3 = load_run(a.path)
rdir = a.readers_dir or os.path.join(os.path.dirname(a.path.rstrip("/\\")), "readers")
paths = {"3B": os.path.join(a.path, "hidden.npy")}
for name, d in [("0.5B", "Qwen2.5-0.5B-Instruct"), ("1.5B", "Qwen2.5-1.5B-Instruct"), ("7B-4bit", "Qwen2.5-7B-Instruct-4bit")]:
    p = os.path.join(rdir, d, "hidden.npy")
    if os.path.exists(p): paths[name] = p
print("readers:", list(paths), flush=True)


def H_of(name, layer):
    h = np.load(paths[name], mmap_mode="r")
    return np.asarray(h[df["hid_idx"].to_numpy(), layer]).astype("float32")


def scores_baseline(r, rung=5):
    m = (df["step"] == r).to_numpy(); idx = np.where(m)[0]
    X = baseline_frame(df, rung); cols = baseline_columns(X, rung) + (["text"] if rung >= 5 else [])
    y, g = df.loc[m, "y"].to_numpy(), df.loc[m, "scene"].to_numpy()
    s, _, _ = crossfit(X.iloc[idx], y, g, lambda C: make_baseline(rung, cols, C), C_GRID, 5, 2, seed=a.seed)
    return s, y, g


def scores_probe(X, r, kind="linear"):
    m = (df["step"] == r).to_numpy(); idx = np.where(m)[0]
    y, g = df.loc[m, "y"].to_numpy(), df.loc[m, "scene"].to_numpy()
    s, _, _ = crossfit(X[idx], y, g, lambda C: make_probe(kind, None, C, a.seed), C_GRID if kind == "linear" else None,
                       5, 2, seed=a.seed)
    return s, y, g


def tpr_at_fpr(y, s, fpr=0.10):
    thr = np.percentile(s[y == 0], 100 * (1 - fpr))
    return float((s[y == 1] > thr).mean())


# ---------------------------------------------------------------- A. reader 간 짝지은 차이 (linear, 3 라운드, layer 50/75%)
rows = []
for r in [0, 1, 2]:
    boot = cluster_boot_idx(df.loc[df["step"] == r, "scene"].to_numpy(), a.n_boot, a.seed)
    for layer in [1, 2]:
        sc = {}
        for name in paths:
            s, y, g = scores_probe(H_of(name, layer), r); sc[name] = s
        sb, _, _ = scores_baseline(r)
        for name in paths:
            rows.append(dict(part="A", step=r, layer_pct={1: 50, 2: 75}[layer], a=name, b="baseline5",
                             auroc_a=roc_auc_score(y, sc[name]), auroc_b=roc_auc_score(y, sb),
                             **dict(zip(["diff", "lo", "hi"], paired_diff(y, sc[name], sb, boot[0])))))
        for n1, n2 in itertools.combinations(sc, 2):
            rows.append(dict(part="A", step=r, layer_pct={1: 50, 2: 75}[layer], a=n1, b=n2,
                             auroc_a=roc_auc_score(y, sc[n1]), auroc_b=roc_auc_score(y, sc[n2]),
                             **dict(zip(["diff", "lo", "hi"], paired_diff(y, sc[n1], sc[n2], boot[0])))))
        print(f"A step {r} layer {layer} done", flush=True)
pd.DataFrame(rows).to_csv(a.out + "_A_reader_pairs.csv", index=False)

# ---------------------------------------------------------------- B. 후반 라운드와 조기 중단 지표 (3B reader, 75% linear, baseline 4/5)
H3 = H_of("3B", 2)
rows = []
for r in range(0, a.late_max_step + 1):
    m = (df["step"] == r).to_numpy()
    if m.sum() == 0: continue
    y = df.loc[m, "y"].to_numpy()
    if min(y.sum(), (1 - y).sum()) < 10: print(f"B step {r}: 소수 클래스 10 미만(n={m.sum()}, 실패 {y.sum()}), 건너뜀"); continue
    boot = cluster_boot_idx(df.loc[m, "scene"].to_numpy(), a.n_boot, a.seed)
    sp, y, g = scores_probe(H3, r); s5, _, _ = scores_baseline(r, 5); s4, _, _ = scores_baseline(r, 4)
    ep = df.loc[m, "episode"].to_numpy(); steps_final = df.groupby("episode")["step"].max().add(1)
    rem = float(np.mean(steps_final.loc[ep].to_numpy()[y == 1] - r))
    for name, s in [("probe_3B_L75_linear", sp), ("baseline5", s5), ("baseline4", s4)]:
        rows.append(dict(part="B", step=r, method=name, n=int(m.sum()), n_fail=int(y.sum()), base_rate=float(y.mean()),
                         auroc=roc_auc_score(y, s), ci_lo=ci(y, s, boot[0])[0], ci_hi=ci(y, s, boot[0])[1],
                         auprc=average_precision_score(y, s), tpr_at_10fpr=tpr_at_fpr(y, s), mean_remaining_steps_of_failures=rem))
    d5 = paired_diff(y, sp, s5, boot[0])
    rows.append(dict(part="B", step=r, method="probe_minus_baseline5", n=int(m.sum()), n_fail=int(y.sum()),
                     auroc=d5[0], ci_lo=d5[1], ci_hi=d5[2]))
    print(f"B step {r} n={m.sum()} fail={int(y.sum())} probe={roc_auc_score(y, sp):.3f} b5={roc_auc_score(y, s5):.3f}", flush=True)
pd.DataFrame(rows).to_csv(a.out + "_B_late_rounds.csv", index=False)

# ---------------------------------------------------------------- C. 시간축 프로브 (3B reader, 75% layer)
ep_codes = df["episode"].to_numpy()
cum = np.zeros_like(H3)
for e in np.unique(ep_codes):
    ix = np.where(ep_codes == e)[0]
    cum[ix] = np.cumsum(H3[ix], axis=0) / np.arange(1, len(ix) + 1)[:, None]
rows = []
variants = {"current": H3, "cummean": cum, "current+cummean": np.hstack([H3, cum])}
for r in a.temporal_steps:
    m = (df["step"] == r).to_numpy()
    if m.sum() == 0 or min(df.loc[m, "y"].sum(), (1 - df.loc[m, "y"]).sum()) < 10: continue
    boot = cluster_boot_idx(df.loc[m, "scene"].to_numpy(), a.n_boot, a.seed)
    s5, y, g = scores_baseline(r, 5)
    for name, X in variants.items():
        s, _, _ = scores_probe(X, r)
        rows.append(dict(part="C", step=r, variant=name, n=int(m.sum()), n_fail=int(y.sum()), auroc=roc_auc_score(y, s),
                         auroc_baseline5=roc_auc_score(y, s5), **dict(zip(["diff", "lo", "hi"], paired_diff(y, s, s5, boot[0])))))
    print(f"C step {r} done", flush=True)
pd.DataFrame(rows).to_csv(a.out + "_C_temporal.csv", index=False)
print("saved", a.out + "_{A,B,C}_*.csv")
