"""게임 내 평가의 보조 분석 (docs/multi_rollout_plan_2026-10-15.md '보조 분석과 해석 규칙', 2026-10-17 사전 기록).

보조 1 (동점 제외): step s의 (실패, 성공) 평가 쌍 중 두 rollout의 step s 프롬프트가 다른 쌍만으로 쌍 가중 AUROC.
보조 2 (시점 맞춘 baseline): baseline 4t·5t = 4·5단에서 현재 step의 repeat, p_chosen을 직전 step 값으로 바꾼 것(n_cand 유지).
점수는 주 분석(within_game.py)과 같은 교차적합(장면 단위, nested C, seed). CI는 게임 단위 bootstrap, 대조군은 게임 안 라벨 순열.
우수 판정용 p는 게임 bootstrap 차이 분포의 양측 p, 6개 비교(baseline 2종 x step 3개)에 Holm.

사용: python probes/within_game_supp.py <run dirs...> --layer 2 --steps 1 2 3 --out results/within_game_supp
점검용: --shuffle_seed k 는 episode 라벨을 전체에서 섞는다(무신호 대조, 결과 해석용이 아님).
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import C_GRID, baseline_columns, baseline_frame, crossfit, load_run, make_baseline, make_probe

MARGIN = 0.05


def pair_table(y, g, prompt):
    """mixed 게임의 모든 (실패 i, 성공 j) 쌍. 반환: game 배열, i, j, tie(프롬프트 동일)."""
    G, I, J, T = [], [], [], []
    for gid in np.unique(g):
        idx = np.where(g == gid)[0]
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        for i in pos:
            for j in neg:
                G.append(gid); I.append(i); J.append(j); T.append(prompt[i] == prompt[j])
    return np.array(G), np.array(I, dtype=int), np.array(J, dtype=int), np.array(T, dtype=bool)


def pair_auc(s, I, J, w):
    """쌍 가중 AUROC: 실패 점수 > 성공 점수 비율(동점 0.5). w는 쌍별 가중치(bootstrap 횟수, 0이면 제외)."""
    if w.sum() == 0: return np.nan
    c = (s[I] > s[J]) + 0.5 * (s[I] == s[J])
    return float((c * w).sum() / w.sum())


def game_weights(G, games_sample, keep):
    """게임 bootstrap 표본의 게임별 등장 횟수를 쌍 가중치로 바꾼다. keep: 사용할 쌍 마스크."""
    cnt = pd.Series(games_sample).value_counts()
    return pd.Series(G).map(cnt).fillna(0).to_numpy() * keep


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def add_prev_step_features(df):
    """repeat, p_chosen을 같은 episode의 직전 step 값으로 바꾼 열(repeat_prev, p_chosen_prev)."""
    df = df.sort_values(["episode", "step"]).copy()
    df["repeat_prev"] = df.groupby("episode")["repeat"].shift(1)
    df["p_chosen_prev"] = df.groupby("episode")["p_chosen"].shift(1)
    return df


def baseline_t_frame(df, idx, rung):
    """4t/5t: rung 4/5의 입력(전체 df로 만든 것과 같은 열)에서 repeat, p_chosen을 직전 step 값으로 바꾼다."""
    X = baseline_frame(df, rung).iloc[idx].copy()
    X["repeat"] = df["repeat_prev"].to_numpy()[idx]; X["p_chosen"] = df["p_chosen_prev"].to_numpy()[idx]
    if X[["repeat", "p_chosen"]].isna().any().any():
        raise ValueError("직전 step 값이 없는 행이 있다(step 0은 평가하지 않는다)")
    return X


def score_methods(df, H, r, seed):
    m = (df["step"] == r).to_numpy(); idx = np.where(m)[0]
    y = df.loc[m, "y"].to_numpy(); g_scene = df.loc[m, "scene"].to_numpy()
    scores = {}
    for name, rung, timed in [("baseline4", 4, False), ("baseline5", 5, False), ("baseline4t", 4, True), ("baseline5t", 5, True)]:
        Xb = baseline_t_frame(df, idx, rung) if timed else baseline_frame(df, rung).iloc[idx]
        cols = baseline_columns(Xb, rung) + (["text"] if rung >= 5 else [])
        scores[name], _, _ = crossfit(Xb, y, g_scene, lambda C, rung=rung, cols=cols: make_baseline(rung, cols, C), C_GRID, 5, 2, seed=seed)
    scores["probe"], _, _ = crossfit(H[idx], y, g_scene, lambda C: make_probe("linear", None, C, seed), C_GRID, 5, 2, seed=seed)
    return idx, scores


def run(a):
    dfs, Hs = [], []
    for i, d in enumerate(a.dirs):
        df, hid = load_run(d)
        df["run"] = i; df["episode"] = df["episode"] + 10000 * i
        Hs.append(np.asarray(hid[df["hid_idx"].to_numpy(), a.layer]).astype("float32"))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True); H = np.vstack(Hs)
    if a.shuffle_seed is not None:  # 점검용 무신호 대조: episode 라벨을 전체에서 섞는다
        ep = df.groupby("episode")["y"].first()
        perm = pd.Series(np.random.default_rng(a.shuffle_seed).permutation(ep.to_numpy()), index=ep.index)
        df["y"] = df["episode"].map(perm).astype(int)
    df = add_prev_step_features(df).loc[df.index]  # 정렬해서 계산한 뒤 원래 행 순서로 되돌린다(H와 행 대응 유지)
    print(f"runs={len(dfs)} episodes={df['episode'].nunique()} games={df['task_id'].nunique()} rows={len(df)}"
          + (f" [SHUFFLED labels seed={a.shuffle_seed}]" if a.shuffle_seed is not None else ""), flush=True)
    rng = np.random.default_rng(a.seed)
    rows, diffs = [], []
    for r in a.steps:
        idx, scores = score_methods(df, H, r, a.seed)
        y = df["y"].to_numpy()[idx]; g = df["task_id"].to_numpy()[idx]; pr = df["prompt"].to_numpy()[idx]
        G, I, J, T = pair_table(y, g, pr)
        games = np.unique(G)
        if not len(games): print(f"step {r}: 갈리는 게임 없음"); continue
        # 시점 맞춘 baseline이 동점 쌍에서 실제로 같은 입력인지 확인(공정성 전제)
        for name in ("probe", "baseline4t", "baseline5t"):
            neq = int((scores[name][I[T]] != scores[name][J[T]]).sum())
            print(f"  step {r} {name}: 동점 쌍 {T.sum()}개 중 점수가 다른 쌍 {neq}개", flush=True)
        boots = [rng.choice(games, len(games)) for _ in range(a.n_boot)]
        perms = []  # 게임 안 라벨 순열(점수 고정). 모든 방법·쌍 집합이 같은 순열을 공유한다
        for _ in range(a.n_perm):
            yp = y.copy()
            for k in games:
                mm = g == k; yp[mm] = rng.permutation(yp[mm])
            perms.append(pair_table(yp, g, pr)[1:])
        sets = {"all": np.ones(len(T), bool), "nontied": ~T}
        for pset, keep in sets.items():
            W0 = keep.astype(float)
            Wb = [game_weights(G, b, keep) for b in boots]
            for meth, s in scores.items():
                est = pair_auc(s, I, J, W0)
                bs = np.array([pair_auc(s, I, J, w) for w in Wb])
                nulls = np.array([pair_auc(s, Ip, Jp, (np.ones(len(Tp), bool) if pset == "all" else ~Tp).astype(float))
                                  for Ip, Jp, Tp in perms])
                rows.append(dict(step=r, pairs=pset, method=meth, n_games=len(games), n_pairs=int(keep.sum()),
                                 tie_frac=float(T.mean()), pair_auroc=est, ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5),
                                 null_mean=float(np.nanmean(nulls)), p_perm=(1 + np.sum(nulls >= est)) / (1 + a.n_perm)))
            for fam, bases in [("supp1_orig", ["baseline4", "baseline5"]), ("supp2_timed", ["baseline4t", "baseline5t"])]:
                # supp1_orig_all은 주 분석과 같은 비교다(보고는 주 분석 값으로 하고, 여기서는 null 보정 점검용으로만 낸다)
                for base in bases:
                    d = np.array([pair_auc(scores["probe"], I, J, w) - pair_auc(scores[base], I, J, w) for w in Wb])
                    d = d[~np.isnan(d)]  # 표본에 해당 쌍이 하나도 없는 bootstrap은 제외
                    est = pair_auc(scores["probe"], I, J, W0) - pair_auc(scores[base], I, J, W0)
                    p_two = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
                    diffs.append(dict(family=f"{fam}_{pset}", step=r, b=base, diff=est, ci_lo=np.percentile(d, 2.5),
                                      ci_hi=np.percentile(d, 97.5), p_boot=p_two))
    out = pd.DataFrame(rows); dd = pd.DataFrame(diffs)
    if len(dd):
        dd["p_holm"] = np.nan
        for fam, sub in dd.groupby("family"):
            dd.loc[sub.index, "p_holm"] = holm(sub["p_boot"].to_numpy())
        dd["verdict"] = [verdict(r) for r in dd.itertuples()]
    print(out.round(3).to_string(index=False))
    if len(dd): print("\n프로브 - baseline (게임 bootstrap 95% CI, Holm은 family 안 6개)\n" + dd.round(3).to_string(index=False))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        out.to_csv(a.out + ".csv", index=False)
        if len(dd): dd.to_csv(a.out + "_diff.csv", index=False)


def verdict(r):
    """사전 기록한 해석 규칙. 우수는 CI 하한 > 0 이고 Holm p < 0.05."""
    if r.ci_lo > 0 and r.p_holm < 0.05: return "우수"
    if r.ci_hi < 0: return "열등"
    if r.ci_lo > -MARGIN and r.ci_hi < MARGIN: return "동등"
    if r.ci_lo > -MARGIN: return "비열등(약함)"
    return "결론 없음"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle_seed", type=int, default=None)
    ap.add_argument("--out")
    run(ap.parse_args())
