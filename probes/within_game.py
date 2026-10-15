"""게임 내(within-game) 평가: 같은 게임을 샘플링으로 여러 번 돌린 rollout들에서 실패를 미리 맞히는가.
(docs/multi_rollout_plan_2026-10-15.md '주 분석')

사용: python probes/within_game.py data/multi_rollout_pilot/Qwen2.5-3B-Instruct data/multi_rollout_main/Qwen2.5-3B-Instruct \
        --layer 2 --steps 1 2 3 --out results/within_game
점수는 run_probes.py와 같은 절차(장면 단위 층화 교차적합, nested C)로 모든 행에서 만들고,
평가는 성공과 실패가 모두 나온 게임의 (실패, 성공) 쌍으로 한다. 지표는 쌍 수 가중 AUROC.
CI는 게임 단위 bootstrap, 대조군은 게임 안에서 라벨을 섞는 순열(점수는 고정).
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import C_GRID, baseline_columns, baseline_frame, crossfit, load_run, make_baseline, make_probe


def game_stats(y, s, g):
    """게임별 (실패>성공 쌍 수 + 0.5 x 동점 쌍 수, 쌍 수). 성공과 실패가 모두 있는 게임만."""
    out = {}
    for gid in np.unique(g):
        m = g == gid
        pos, neg = s[m][y[m] == 1], s[m][y[m] == 0]
        if len(pos) and len(neg):
            out[gid] = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum(), len(pos) * len(neg))
    return out


def pair_auc(stats, keys=None):
    keys = list(stats) if keys is None else keys
    num = sum(stats[k][0] for k in keys); den = sum(stats[k][1] for k in keys)
    return num / den if den else np.nan


def run(a):
    dfs, Hs = [], []
    for i, d in enumerate(a.dirs):
        df, hid = load_run(d)
        df["run"] = i; df["episode"] = df["episode"] + 10000 * i
        if a.methods != ["baseline2"]:
            Hs.append(np.asarray(hid[df["hid_idx"].to_numpy(), a.layer]).astype("float32"))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    H = np.vstack(Hs) if Hs else None
    print(f"runs={len(dfs)} episodes={df['episode'].nunique()} games={df['task_id'].nunique()} rows={len(df)}", flush=True)
    rng = np.random.default_rng(a.seed)
    rows, diffs = [], []
    for r in a.steps:
        m = (df["step"] == r).to_numpy(); idx = np.where(m)[0]
        y = df.loc[m, "y"].to_numpy(); g_game = df.loc[m, "task_id"].to_numpy(); g_scene = df.loc[m, "scene"].to_numpy()
        scores = {}
        for meth in a.methods:
            if meth.startswith("baseline"):
                rung = int(meth[len("baseline"):]); Xb = baseline_frame(df, rung).iloc[idx]
                cols = baseline_columns(Xb, rung) + (["text"] if rung >= 5 else [])
                s, _, _ = crossfit(Xb, y, g_scene, lambda C, rung=rung, cols=cols: make_baseline(rung, cols, C), C_GRID, 5, 2, seed=a.seed)
            else:
                s, _, _ = crossfit(H[idx], y, g_scene, lambda C: make_probe("linear", None, C, a.seed), C_GRID, 5, 2, seed=a.seed)
            scores[meth] = s
        st = {k: game_stats(y, s, g_game) for k, s in scores.items()}
        mixed = sorted(next(iter(st.values())))
        if not mixed: print(f"step {r}: 갈리는 게임 없음"); continue
        boots = [rng.choice(len(mixed), len(mixed)) for _ in range(a.n_boot)]
        for meth in a.methods:
            S = st[meth]; est = pair_auc(S, mixed)
            ga = np.mean([S[k][0] / S[k][1] for k in mixed])
            bs = [pair_auc(S, [mixed[i] for i in b]) for b in boots]
            # 게임 안에서 라벨을 섞은 순열 대조군(점수 고정)
            nulls = []
            for _ in range(a.n_perm):
                yp = y.copy()
                for k in mixed:
                    mm = g_game == k; yp[mm] = rng.permutation(yp[mm])
                nulls.append(pair_auc(game_stats(yp, scores[meth], g_game), mixed))
            rows.append(dict(step=r, method=meth, n_rows=int(m.sum()), n_games_mixed=len(mixed),
                             n_pairs=int(sum(S[k][1] for k in mixed)), pair_auroc=est, game_avg_auroc=ga,
                             ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                             null_mean=np.mean(nulls), p_perm=(1 + sum(n >= est for n in nulls)) / (1 + a.n_perm)))
            print(f"  step {r} {meth}: pair_auroc={est:.3f} games={len(mixed)}", flush=True)
        if "probe" in a.methods:
            for base in [b for b in a.methods if b.startswith("baseline") and b != "baseline2"]:
                d = [pair_auc(st["probe"], [mixed[i] for i in b]) - pair_auc(st[base], [mixed[i] for i in b]) for b in boots]
                diffs.append(dict(step=r, a="probe", b=base, diff=pair_auc(st["probe"], mixed) - pair_auc(st[base], mixed),
                                  ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5)))
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    if diffs: print("\n프로브 - baseline (쌍 가중 AUROC, 게임 bootstrap 95% CI)\n" + pd.DataFrame(diffs).round(3).to_string(index=False))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        out.to_csv(a.out + ".csv", index=False)
        if diffs: pd.DataFrame(diffs).to_csv(a.out + "_diff.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--methods", nargs="+", default=["baseline2", "baseline4", "baseline5", "probe"])
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    run(ap.parse_args())
