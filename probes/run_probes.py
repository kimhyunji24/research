"""사용:
  python probes/run_probes.py data/main_3b/Qwen2.5-3B-Instruct --baselines_only --rounds 0 1 2 3 --out results/baselines_main_3b.csv
  python probes/run_probes.py data/main_3b/Qwen2.5-3B-Instruct --layer 1 --rounds 0 1 2 3 --out results/probes_main_3b.csv

--rounds는 0-based step이다(0 = 첫 번째 action). "라운드 1"의 기준은 미결정(docs/analysis_plan_2026-10-15.md).
출력: baseline 1~5단, 프로브(linear/MLP) x PCA k 스윕, 순열 대조군, 장면 단위 cluster bootstrap CI,
프로브와 가장 높은 baseline의 짝지은 AUROC 차이.
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (baseline_columns, baseline_frame, cluster_boot_idx, load_run, make_baseline, make_probe,
                 paired_diff, round_eval)

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--layer", type=int, default=1, help="relative_layers 인덱스 (1=50%)")
ap.add_argument("--layers", type=int, nargs="+", help="여러 layer를 한 번에(탐색적). 지정하면 --layer는 무시")
ap.add_argument("--hidden", help="다른 reader의 hidden.npy 경로(collect/extract_reader.py 출력)")
ap.add_argument("--rounds", type=int, nargs="+", default=[0, 1, 2], help="0-based step")
ap.add_argument("--pca", type=int, nargs="*", default=[0], help="0=PCA 없음(전체 차원). 스윕 예: 0 8 32 64 128")
ap.add_argument("--kinds", nargs="+", default=["linear"], choices=["linear", "mlp"])
ap.add_argument("--rungs", type=int, nargs="+", default=[1, 2, 3, 4, 5])
ap.add_argument("--baselines_only", action="store_true")
ap.add_argument("--n_perm", type=int, default=100)
ap.add_argument("--n_boot", type=int, default=1000)
ap.add_argument("--n_repeats", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_jobs", type=int, default=-1)
ap.add_argument("--out", default=None)
a = ap.parse_args()

df, hid = load_run(a.path, a.hidden)
layers = a.layers if a.layers else [a.layer]
print(f"episodes={df['episode'].nunique()} steps={len(df)} scenes={df['scene'].nunique()} "
      f"success={(df.groupby('episode')['y'].first() == 0).sum()} prompt_saved={'text' in df.columns}")
if "obj_parsed" in df.columns and "text" in df.columns:
    print(f"목표 물체 파싱 성공 episode 비율: {df.groupby('episode')['obj_parsed'].first().mean():.3f}")
rungs = [r for r in a.rungs if r < 5 or "text" in df.columns]
if len(rungs) < len(a.rungs): print("prompt가 없어 5단은 건너뛴다")

Hs = {}
if not a.baselines_only:
    Hs = {l: np.asarray(hid[df["hid_idx"].to_numpy(), l]).astype("float32") for l in layers}

rows, scores = [], {}
for r in a.rounds:
    m = (df["step"] == r).to_numpy()
    if m.sum() == 0: continue
    boot = cluster_boot_idx(df.loc[m, "scene"].to_numpy(), a.n_boot, a.seed)
    common = dict(n_splits=5, n_repeats=a.n_repeats, boot=boot, seed=a.seed, n_jobs=a.n_jobs)

    def run(method, X, make, Cs, n_perm):
        res = round_eval(df, X, r, make, Cs=Cs, n_perm=n_perm, **common)
        if res is None: return
        scores[(r, method)] = (res.pop("_y"), res.pop("_s"), res.pop("_groups"))
        rows.append(dict(res, method=method)); print(f"  step {r} {method}: auroc={res['auroc']:.3f}", flush=True)

    from lib import C_GRID
    for rung in rungs:
        Xb = baseline_frame(df, rung); cols = [c for c in baseline_columns(Xb, rung)]
        run(f"baseline{rung}", Xb, lambda C, rung=rung, cols=cols: make_baseline(rung, cols + (["text"] if rung >= 5 else []), C),
            C_GRID, a.n_perm)
    if not a.baselines_only:
        for layer in layers:
            for kind in a.kinds:
                for k in a.pca:
                    name = f"{kind}_L{layer}_pca{k or 'full'}" if a.layers else f"{kind}_pca{k or 'full'}"
                    run(name, Hs[layer], lambda C, kind=kind, k=k: make_probe(kind, k or None, C, a.seed),
                        C_GRID if kind == "linear" else None, a.n_perm if kind == "linear" else min(a.n_perm, 20))

out = pd.DataFrame(rows)
cols = ["step", "method", "n", "n_fail", "base_rate", "auroc", "ci_lo", "ci_hi", "ci_ep_lo", "ci_ep_hi", "auprc",
        "null_mean", "null_lo", "null_hi", "p_perm", "C_median", "pca_evr", "note"]
out = out[[c for c in cols if c in out.columns]]
print(out.round(3).to_string(index=False))

diffs = []
top = max(rungs) if rungs else None
for (r, method) in scores:
    if method.startswith("baseline") or top is None or (r, f"baseline{top}") not in scores: continue
    y, s, g = scores[(r, method)]; _, sb, _ = scores[(r, f"baseline{top}")]
    boot = cluster_boot_idx(g, a.n_boot, a.seed)
    est, lo, hi = paired_diff(y, s, sb, boot[0])
    diffs.append(dict(step=r, method=method, vs=f"baseline{top}", auroc_diff=est, ci_lo=lo, ci_hi=hi))
if diffs:
    print("\n프로브 - baseline 짝지은 AUROC 차이 (장면 단위 cluster bootstrap 95% CI)")
    print(pd.DataFrame(diffs).round(3).to_string(index=False))
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    if diffs: pd.DataFrame(diffs).to_csv(a.out.replace(".csv", "_diff.csv"), index=False)
