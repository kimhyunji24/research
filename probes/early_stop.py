"""조기 중단 평가 최소판 (docs/early_stop_plan_2026-10-17.md, 사전 기록, 1회 실행).

actor Qwen2.5-3B가 자기 hidden(75% layer)으로 감시한다. 게임 = main_3b 500게임, 100 step 궤적
(30 step 안 성공은 main_3b, 나머지는 horizon100). 라벨 y100 = 100 step 안 실패.
중단 규칙: step t 행동 직전 점수 >= τ이면 멈추고 실패로 처리. 비용 = 실행한 step 수.
사용: python probes/early_stop.py --out results/early_stop_3b [--n_jobs 10]
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LAYER = 2                                              # 75% layer (relative_layers 0.25/0.5/0.75/1.0 중 세 번째)
LOSS_MAIN = np.round(np.arange(0, 0.10001, 0.005), 4)  # 주 지표: 손실 0~10%, 0.5%p 간격(21점)
LOSS_FULL = np.round(np.arange(0, 1.00001, 0.005), 4)  # 보조: 전체 손실 범위
CAP_TAUS = np.arange(1, 100, dtype=float)              # cap c = 1~99 (c = 100은 안 멈춤 = +inf)
LOOP_TAUS = np.arange(1, 7, dtype=float)               # 루프 k = 1~6


# ---------------------------------------------------------------- 곡선 (결정적, 학습 없음)
class Curve:
    """방법 하나의 (τ -> 총 비용, 잃은 성공 수)를 게임 가중치로 계산한다.
    games: DataFrame(game, y(1=실패), full(안 멈출 때 비용 = 성공 step 또는 100)), rows: 게임별 step 순 점수 배열 목록."""

    def __init__(self, scores_by_game, full, y):
        v, g, dc, dl = [], [], [], []
        for gi, sc in enumerate(scores_by_game):
            rec, pm = [], -np.inf
            for t, x in enumerate(sc):           # prefix max가 오르는 시점만 중단 후보(점수 >= τ인 첫 step)
                if x > pm: rec.append((x, t)); pm = x
            nxt = full[gi]
            for k, (x, t) in enumerate(reversed(rec)):  # τ가 x 이하로 내려가면 중단 시점이 t로 당겨진다
                v.append(x); g.append(gi); dc.append(t - nxt); dl.append(1.0 if (k == 0 and y[gi] == 0) else 0.0); nxt = t
        o = np.argsort(v, kind="stable")
        self.v, self.g = np.asarray(v, float)[o], np.asarray(g)[o]
        self.dc, self.dl = np.asarray(dc, float)[o], np.asarray(dl, float)[o]
        self.full, self.y = np.asarray(full, float), np.asarray(y)

    def at(self, taus, w=None):
        """taus(점수 >= τ이면 중단)마다 (절감률, 손실률). +inf는 안 멈춤."""
        w = np.ones(len(self.full)) if w is None else w
        base = float((w * self.full).sum()); nsucc = float((w * (self.y == 0)).sum())
        we = w[self.g]
        sc = np.concatenate([np.cumsum((we * self.dc)[::-1])[::-1], [0.0]])   # suffix sum: v >= τ인 사건 합
        sl = np.concatenate([np.cumsum((we * self.dl)[::-1])[::-1], [0.0]])
        k = np.searchsorted(self.v, np.asarray(taus, float), side="left")
        cost, lost = base + sc[k], sl[k]
        return 1.0 - cost / base, (lost / nsucc if nsucc > 0 else np.zeros_like(lost))

    def taus_all(self):
        return np.concatenate([np.unique(self.v), [np.inf]])


def partial_area(sav, loss, grid):
    """손실 허용 ℓ마다 손실 <= ℓ인 τ 중 최대 절감률의 평균. τ 목록에 +inf(절감 0, 손실 0)가 있어 항상 정의된다."""
    return float(np.mean([sav[loss <= l + 1e-12].max() for l in grid]))


def metric(curve, taus, w=None, grid=LOSS_MAIN):
    s, l = curve.at(taus, w)
    return partial_area(s, l, grid)


def oracle_metric(full, y, w=None):
    """실패 게임은 step 0에서 중단, 성공 게임은 안 멈춤: 손실 0, 절감 = 실패 게임 비용 전부."""
    w = np.ones(len(full)) if w is None else w
    return float(1.0 - (w * full * (y == 0)).sum() / (w * full).sum())


def scene_weights(scenes, n_boot, seed=0):
    """장면 단위 cluster bootstrap의 게임별 가중치(재표집 횟수). 모든 방법이 같은 재표집을 공유한다."""
    ug, inv = np.unique(scenes, return_inverse=True)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        out.append(cnt[inv].astype(float))
    return out


def fold_tau_eval(curve, taus, scenes, y, target=0.05, seed=0):
    """보조: 게임 단위 장면 5-fold. 나머지 fold로 손실 <= target 중 절감 최대 τ를 고르고 남은 fold에 적용해 합산."""
    n = len(y); cost = lost = base = nsucc = 0.0
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y, scenes):
        wtr = np.zeros(n); wtr[tr] = 1; wte = np.zeros(n); wte[te] = 1
        s, l = curve.at(taus, wtr)
        ok = np.where(l <= target + 1e-12)[0]; tau = taus[ok[np.argmax(s[ok])]]
        st, lt = curve.at([tau], wte)
        b = float((wte * curve.full).sum()); ns = float((wte * (curve.y == 0)).sum())
        cost += b * (1 - st[0]); base += b; lost += lt[0] * ns; nsucc += ns
    return 1 - cost / base, lost / nsucc


# ---------------------------------------------------------------- 데이터
def assemble(main_dir, h_dir):
    from lib import load_run
    sm, Hm = load_run(main_dir); sh, Hh = load_run(h_dir)
    em = pd.read_json(os.path.join(main_dir, "episodes.jsonl"), lines=True, dtype={"task_id": str})
    eh = pd.read_json(os.path.join(h_dir, "episodes.jsonl"), lines=True, dtype={"task_id": str})
    assert len(em) == 500 and em["task_id"].is_unique and eh["task_id"].is_unique
    fail30 = em[~em["success"]]
    assert set(eh["task_id"]) == set(fail30["task_id"]) and len(eh) == len(fail30), "horizon100 게임이 main_3b 실패 게임과 다르다"
    main_ep = dict(zip(em["task_id"], em["episode"]))
    for d in (sm, sh):  # 4t·5t·루프용 직전 step 값. step 0은 0
        d["repeat_prev"] = d.groupby("episode")["repeat"].shift(1).fillna(0)
        d["p_chosen_prev"] = d.groupby("episode")["p_chosen"].shift(1).fillna(0)
    a = sm[sm["success"]].copy(); a["game"] = a["episode"]
    b = sh.copy(); b["game"] = b["task_id"].map(main_ep)
    # hidden: step 0~29는 main_3b, 30~99는 horizon100 (계획). 0~29 action이 같은지 확인
    key = {(r.episode, r.step): (r.hid_idx, r.action) for r in sm[["episode", "step", "hid_idx", "action"]].itertuples()}
    src_m, idx, mism = [], [], 0
    for r in b[["game", "step", "hid_idx", "action"]].itertuples():
        if r.step < 30:
            hi, act = key[(r.game, r.step)]; mism += act != r.action; src_m.append(True); idx.append(hi)
        else:
            src_m.append(False); idx.append(r.hid_idx)
    assert mism == 0, f"horizon100 처음 30 step action이 main_3b와 {mism}곳 다르다"
    src_m, idx = np.array(src_m), np.array(idx)
    Ha = np.asarray(Hm[np.sort(a["hid_idx"].to_numpy())][:, LAYER], dtype=np.float32)
    Ha = Ha[np.argsort(np.argsort(a["hid_idx"].to_numpy()))]
    Hb = np.zeros((len(b), Hm.shape[2]), np.float32)
    for H_, m_ in ((Hm, src_m), (Hh, ~src_m)):
        ii = idx[m_]; o = np.argsort(ii)
        Hb[np.flatnonzero(m_)[o]] = np.asarray(H_[ii[o]][:, LAYER], dtype=np.float32)
    df = pd.concat([a, b], ignore_index=True)
    H = np.concatenate([Ha, Hb])
    o = np.lexsort((df["step"].to_numpy(), df["game"].to_numpy())); df = df.iloc[o].reset_index(drop=True); H = H[o]
    ep = pd.concat([em[em["success"]][["episode", "steps"]].assign(y=0).rename(columns={"episode": "game"}),
                    eh.assign(game=eh["task_id"].map(main_ep), y=(~eh["success"]).astype(int))[["game", "steps", "y"]]])
    games = ep.sort_values("game").reset_index(drop=True)
    games["full"] = np.where(games["y"] == 1, 100, games["steps"])
    games["scene"] = games["game"].map(dict(zip(df["game"], df["scene"])))
    n_rows = df.groupby("game").size().reindex(games["game"]).to_numpy()
    assert (n_rows == games["full"]).all(), "게임별 행 수가 성공 step(또는 100)과 다르다"
    assert (df.groupby("game")["y"].first().reindex(games["game"]).to_numpy() == games["y"].to_numpy()).all()
    return df, H, games


# ---------------------------------------------------------------- 학습 규칙 (교차적합, lib.crossfit과 같은 분할·seed를 fold 병렬로)
def _fold(X, y, g, make, tr, te, rep):
    from lib import C_GRID, _select_C, _take
    C = _select_C(_take(X, tr), y[tr], g[tr], make, C_GRID, 3, rep)
    m = make(C).fit(_take(X, tr), y[tr])
    return te, m.predict_proba(_take(X, te))[:, 1], C


def crossfit_par(X, y, g, make, n_jobs):
    jobs = [(tr, te, rep) for rep in (0, 1)
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, g)]
    out = Parallel(n_jobs=n_jobs)(delayed(_fold)(X, y, g, make, tr, te, rep) for tr, te, rep in jobs)
    s = np.zeros(len(y))
    for te, p, _ in out: s[te] += p / 2
    return s, [c for _, _, c in out]


def learned_scores(df, H, n_jobs):
    from lib import baseline_columns, baseline_frame, make_baseline, make_probe
    y = df["y"].to_numpy(); g = df["scene"].to_numpy(); sc, Cs = {}, {}
    for name, rung in (("baseline4t", 4), ("baseline5t", 5)):
        X = baseline_frame(df, rung)
        X["repeat"] = df["repeat_prev"].to_numpy(); X["p_chosen"] = df["p_chosen_prev"].to_numpy()
        cols = baseline_columns(X, rung) + (["text"] if rung >= 5 else [])
        t0 = time.time()
        sc[name], Cs[name] = crossfit_par(X, y, g, lambda C, rung=rung, cols=cols: make_baseline(rung, cols, C), n_jobs)
        print(f"  {name}: {time.time() - t0:.0f}s, C={Cs[name]}", flush=True)
    t0 = time.time()
    Xp = np.hstack([H, df[["step"]].to_numpy(np.float32)])
    sc["probe"], Cs["probe"] = crossfit_par(Xp, y, g, lambda C: make_probe("linear", None, C, 0), n_jobs)
    print(f"  probe: {time.time() - t0:.0f}s, C={Cs['probe']}", flush=True)
    return sc, Cs


# ---------------------------------------------------------------- 평가
def evaluate(df, games, row_scores, n_boot=1000, seed=0):
    gidx = {g: i for i, g in enumerate(games["game"])}
    order = df["game"].map(gidx).to_numpy()
    split = np.r_[0, np.flatnonzero(np.diff(order)) + 1, len(order)]
    full, y, scenes = games["full"].to_numpy(float), games["y"].to_numpy(), games["scene"].to_numpy()
    curves, taus = {}, {}
    for name, s in row_scores.items():
        curves[name] = Curve([s[split[i]:split[i + 1]] for i in range(len(games))], full, y)
        taus[name] = (np.r_[CAP_TAUS, np.inf] if name == "cap" else np.r_[LOOP_TAUS, np.inf] if name == "loop"
                      else curves[name].taus_all())
    W = scene_weights(scenes, n_boot, seed)
    seg = {"all": np.ones(len(y)), "explore_0_249": (games["game"] < 250).to_numpy(float),
           "confirm_250_499": (games["game"] >= 250).to_numpy(float)}
    rows, boot = [], {}
    for sname, sw in seg.items():
        for grid_name, grid in (("loss0_10", LOSS_MAIN), ("loss0_100", LOSS_FULL)):
            if sname != "all" and grid_name == "loss0_100": continue
            pt = {m: metric(curves[m], taus[m], sw, grid) for m in curves}; pt["oracle"] = oracle_metric(full, y, sw)
            bs = {m: np.array([metric(curves[m], taus[m], w * sw, grid) for w in W]) for m in curves}
            bs["oracle"] = np.array([oracle_metric(full, y, w * sw) for w in W])
            boot[(sname, grid_name)] = (pt, bs)
            for m in pt:
                rows.append(dict(segment=sname, area=grid_name, method=m, value=pt[m],
                                 ci_lo=np.quantile(bs[m], .025), ci_hi=np.quantile(bs[m], .975),
                                 ratio_to_oracle=pt[m] / pt["oracle"] if pt["oracle"] > 0 else np.nan))
    res = pd.DataFrame(rows)
    drows = []
    for (sname, grid_name), (pt, bs) in boot.items():
        best_pt = max(pt["cap"], pt["loop"]); best_bs = np.maximum(bs["cap"], bs["loop"])
        comps = [("probe - max(cap,loop)", pt["probe"] - best_pt, bs["probe"] - best_bs, "main" if (sname, grid_name) == ("all", "loss0_10") else "supp"),
                 ("probe - baseline5t", pt["probe"] - pt["baseline5t"], bs["probe"] - bs["baseline5t"], "supp"),
                 ("probe - baseline4t", pt["probe"] - pt["baseline4t"], bs["probe"] - bs["baseline4t"], "supp")]
        for name, d, db, role in comps:
            lo, hi = np.quantile(db, [.025, .975])
            drows.append(dict(segment=sname, area=grid_name, comparison=name, role=role, diff=d, ci_lo=lo, ci_hi=hi,
                              verdict="superior" if lo > 0 else "inferior" if hi < 0 else "inconclusive",
                              best_rule_point="cap" if pt["cap"] >= pt["loop"] else "loop"))
    diff = pd.DataFrame(drows)
    frows = []
    for m in curves:
        s, l = fold_tau_eval(curves[m], taus[m], scenes, y)
        frows.append(dict(method=m, target_loss=0.05, test_saving=s, test_loss=l))
    fold = pd.DataFrame(frows)
    crows = []
    for m in curves:
        s, l = curves[m].at(taus[m])
        crows += [dict(method=m, tau=t, saving=a, loss=b) for t, a, b in zip(taus[m], s, l)]
    return res, diff, fold, pd.DataFrame(crows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="data/main_3b/Qwen2.5-3B-Instruct")
    ap.add_argument("--horizon", default="data/horizon100/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_jobs", type=int, default=10)
    a = ap.parse_args()
    outs = [a.out + s for s in (".csv", "_diff.csv", "_fold.csv", "_curves.csv")]
    for f in outs:
        if os.path.exists(f): raise SystemExit(f"출력 파일이 이미 있다: {f}")
    t0 = time.time()
    df, H, games = assemble(a.main, a.horizon)
    print(f"games={len(games)} rows={len(df)} scenes={games['scene'].nunique()} hidden={H.shape} "
          f"success_within_100={(games['y'] == 0).sum()} base_steps={games['full'].sum():.0f}", flush=True)
    sc, Cs = learned_scores(df, H, a.n_jobs)
    scores = {"cap": df["step"].to_numpy(float), "loop": df["repeat_prev"].to_numpy(float), **sc}
    res, diff, fold, curves = evaluate(df, games, scores, a.n_boot)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for d, f in zip((res, diff, fold, curves), outs): d.to_csv(f, index=False)
    sd = "data/early_stop_3b"; os.makedirs(sd, exist_ok=True)
    np.savez(os.path.join(sd, "row_scores.npz"), game=df["game"].to_numpy(), step=df["step"].to_numpy(), y=df["y"].to_numpy(), **scores)
    pd.set_option("display.width", 220)
    print("\n주 지표(손실 0~10% 평균 최대 절감률) 등\n" + res.round(4).to_string(index=False))
    print("\n비교(장면 cluster bootstrap 95% CI)\n" + diff.round(4).to_string(index=False))
    print("\n보조: fold 안 τ(손실 5%)\n" + fold.round(4).to_string(index=False))
    print(f"\n총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
