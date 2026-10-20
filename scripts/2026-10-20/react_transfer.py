"""탐색: v2 hidden으로 학습한 프로브를 ReAct 궤적에 적용(docs/prompt_robustness_plan_2026-10-17.md '탐색: 전이 점검 설계').
장면 누수 방지: ReAct 행의 장면 5-fold(seed 0·1)마다 시험 장면에 없는 v2 행으로만 학습한다. 확증 근거로 쓰지 않는다.
사용: python scripts/2026-10-20/react_transfer.py --out results/react_transfer_qwen25_3b.csv"""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "probes"))
from lib import (C_GRID, _select_C, baseline_columns, baseline_frame, cluster_boot_idx, crossfit, load_run,
                 make_baseline, make_probe, paired_diff, ci)

ap = argparse.ArgumentParser()
ap.add_argument("--v2", default="data/main_3b/Qwen2.5-3B-Instruct")
ap.add_argument("--react", default="data/react_qwen25_3b/Qwen2.5-3B-Instruct")
ap.add_argument("--layer", type=int, default=2)
ap.add_argument("--rounds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--out", required=True)
a = ap.parse_args()
if os.path.exists(a.out): raise SystemExit(f"출력 파일이 이미 있다: {a.out}")

dv, hv = load_run(a.v2); dr, hr = load_run(a.react)
dv = dv[dv["episode"] < 250].reset_index(drop=True)
assert set(dv["task_id"]) == set(dr["task_id"]) and dr["episode"].nunique() == 250, "게임 집합이 다르다"
Hv = np.asarray(hv[dv["hid_idx"].to_numpy(), a.layer], dtype=np.float32)
Hr = np.asarray(hr[dr["hid_idx"].to_numpy(), a.layer], dtype=np.float32)
print(f"v2 rows={len(dv)} react rows={len(dr)} layer={a.layer} hidden={Hr.shape[1]}", flush=True)

rows, diffs = [], []
for r in a.rounds:
    mv = (dv["step"] == r).to_numpy(); mr = (dr["step"] == r).to_numpy()
    Xv, yv, gv = Hv[mv], dv.loc[mv, "y"].to_numpy(), dv.loc[mv, "scene"].to_numpy()
    Xr, yr, gr = Hr[mr], dr.loc[mr, "y"].to_numpy(), dr.loc[mr, "scene"].to_numpy()
    s = {"transfer_A": np.zeros(len(yr)), "transfer_B": np.zeros(len(yr))}; Cs = []
    for rep in (0, 1):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(yr)), yr, gr):
            keep = ~np.isin(gv, gr[te])                         # 시험 장면에 속한 v2 행은 학습에서 뺀다
            C = _select_C(Xv[keep], yv[keep], gv[keep], lambda C: make_probe("linear", None, C, 0), C_GRID, 3, rep); Cs.append(C)
            m = make_probe("linear", None, C, 0).fit(Xv[keep], yv[keep])
            s["transfer_A"][te] += m.predict_proba(Xr[te])[:, 1] / 2
            mu_r, sd_r = Xr[tr].mean(0), Xr[tr].std(0) + 1e-6       # 라벨 없이 ReAct 분포를 v2 학습 행 척도로 맞춤
            mu_v, sd_v = Xv[keep].mean(0), Xv[keep].std(0) + 1e-6
            s["transfer_B"][te] += m.predict_proba((Xr[te] - mu_r) / sd_r * sd_v + mu_v)[:, 1] / 2
    s["react_probe"], _, _ = crossfit(Xr, yr, gr, lambda C: make_probe("linear", None, C, 0), C_GRID, 5, 2, seed=0)
    Xb = baseline_frame(dr, 5).iloc[np.where(mr)[0]]; cols = baseline_columns(Xb, 5) + ["text"]
    s["react_baseline5"], _, _ = crossfit(Xb, yr, gr, lambda C: make_baseline(5, cols, C), C_GRID, 5, 2, seed=0)
    boot = cluster_boot_idx(gr, 1000, 0)[0]
    for k, v in s.items():
        lo, hi = ci(yr, v, boot)
        rows.append(dict(step=r, method=k, n=len(yr), n_fail=int(yr.sum()), auroc=roc_auc_score(yr, v), ci_lo=lo, ci_hi=hi,
                         C_median=float(np.median(Cs)) if k.startswith("transfer") else np.nan))
    for k in ("transfer_A", "transfer_B"):
        for vs in ("react_probe", "react_baseline5"):
            est, lo, hi = paired_diff(yr, s[k], s[vs], boot)
            diffs.append(dict(step=r, method=k, vs=vs, auroc_diff=est, ci_lo=lo, ci_hi=hi))
    print(f"  step {r} done", flush=True)

out = pd.DataFrame(rows); d = pd.DataFrame(diffs)
out.to_csv(a.out, index=False); d.to_csv(a.out.replace(".csv", "_diff.csv"), index=False)
pd.set_option("display.width", 200)
print(out.round(4).to_string(index=False)); print(); print(d.round(4).to_string(index=False))
