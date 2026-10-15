"""within_game.py의 쌍 AUROC 함수를 합성 데이터로 점검한다(실제 데이터 아님)."""
import sys, numpy as np
sys.path.insert(0, "probes")
from within_game import game_stats, pair_auc
rng = np.random.default_rng(0)
G, K = 40, 8
g = np.repeat(np.arange(G), K)
y = np.concatenate([(rng.random(K) < 0.5).astype(int) for _ in range(G)])
# 1) 점수가 라벨과 무관 -> 0.5 근처
s0 = rng.normal(size=len(y)); print("무신호:", round(pair_auc(game_stats(y, s0, g)), 3))
# 2) 게임별 오프셋만 큰 점수(게임 난이도) -> 게임 내 AUROC 0.5 근처여야 함, 전체 AUROC는 높음
off = rng.normal(size=G) * 5; s1 = off[g] + 0.01 * rng.normal(size=len(y))
print("게임 오프셋만:", round(pair_auc(game_stats(y, s1, g)), 3))
# 3) 게임 내 신호
s2 = s1 + 1.5 * y; print("게임 내 신호:", round(pair_auc(game_stats(y, s2, g)), 3))
# 4) 완전 동점 -> 0.5
print("동점:", pair_auc(game_stats(y, np.zeros(len(y)), g)))
