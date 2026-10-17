"""within_game_supp 단위 점검(합성 데이터): 쌍 표, 쌍 AUROC, 동점 항등식, bootstrap 가중, Holm, 주 분석 함수와 일치."""
import sys, numpy as np
sys.path.insert(0, "probes")
import within_game_supp as S
import within_game as W

rng = np.random.default_rng(0)
# 게임 30개 x 8 rollout, 게임마다 프롬프트 2~3종(동점 유발)
g = np.repeat(np.arange(30), 8).astype(str)
y = rng.integers(0, 2, len(g))
pr = np.array([f"{gg}-{rng.integers(0, 3)}" for gg in g])
s_tied = np.array([hash(p) % 1000 / 1000 for p in pr])  # 프롬프트가 같으면 점수 같음(프로브 상황)
s_free = rng.random(len(g))                               # 동점 없는 점수(baseline 상황)

G, I, J, T = S.pair_table(y, g, pr)
# 1) 주 분석 함수와 같은 값
st = W.game_stats(y, s_free, g)
assert abs(S.pair_auc(s_free, I, J, np.ones(len(I))) - W.pair_auc(st)) < 1e-12
st2 = W.game_stats(y, s_tied, g)
assert abs(S.pair_auc(s_tied, I, J, np.ones(len(I))) - W.pair_auc(st2)) < 1e-12
# 2) 동점 항등식: all = 0.5 t + (1-t) nontied  (동점 쌍에서 점수가 같을 때)
t = T.mean()
a_all = S.pair_auc(s_tied, I, J, np.ones(len(I))); a_nt = S.pair_auc(s_tied, I, J, (~T).astype(float))
assert abs(a_all - (0.5 * t + (1 - t) * a_nt)) < 1e-12, (a_all, t, a_nt)
# 3) bootstrap 가중 = 주 분석의 키 목록 방식
games = np.unique(G); b = rng.choice(games, len(games))
w = S.game_weights(G, b, np.ones(len(I), bool))
assert abs(S.pair_auc(s_free, I, J, w) - W.pair_auc(st, list(b))) < 1e-12
# 4) Holm
assert np.allclose(S.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
# 5) 직전 step 특징
import pandas as pd
df = pd.DataFrame(dict(episode=[1, 1, 1, 2, 2], step=[0, 1, 2, 0, 1], repeat=[0, 1, 2, 5, 6], p_chosen=[.1, .2, .3, .5, .6]))
d2 = S.add_prev_step_features(df.sample(frac=1, random_state=0)).loc[df.sample(frac=1, random_state=0).index]
d2 = d2.sort_values(["episode", "step"])
assert d2.repeat_prev.tolist()[1:3] == [0, 1] and np.isnan(d2.repeat_prev.iloc[0]) and d2.p_chosen_prev.iloc[4] == .5
print(f"all unit checks passed (tie frac {t:.2f}, pairs {len(I)})")
