import sys, numpy as np
sys.path.insert(0, "probes")
from within_game import game_stats, pair_auc
G, K = 40, 8
vals = []
for seed in range(300):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(G), K)
    y = np.concatenate([(rng.random(K) < 0.5).astype(int) for _ in range(G)])
    s = rng.normal(size=G)[g] * 5 + 0.01 * rng.normal(size=len(y))
    vals.append(pair_auc(game_stats(y, s, g)))
print("게임 오프셋만, 300회 반복: 평균", round(float(np.mean(vals)), 3), "sd", round(float(np.std(vals)), 3))
