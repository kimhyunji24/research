"""합성 데이터 점검. 실제 결과가 아니라 코드와 가정을 확인하기 위한 것이다.

1. 순열 대조군 보정: 신호가 없는 데이터에서 (a) 층화하지 않은 GroupKFold로 풀링한 AUROC와
   (b) lib.crossfit(StratifiedGroupKFold)의 AUROC가 0.5 근처인지 본다.
2. 신호 검출: 분산이 작은 방향에 약한 신호가 있을 때 PCA k에 따른 AUROC와 explained variance.
3. 장면 누수: 같은 장면 안에서만 라벨이 같은 데이터에서 게임 단위 분할이 부풀리는지 본다.
"""
import os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import crossfit, make_probe

rng = np.random.default_rng(0)
N, D, S, REP = 250, 256, 86, 60
scene = rng.integers(0, S, N)
y0 = (rng.random(N) < 0.78).astype(int)
X0 = rng.normal(size=(N, D))

def old_style(X, y, g, C=0.01):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        s[te] = make_probe("linear", None, C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(y, s)

def new_style(X, y, g):
    s, _, _ = crossfit(X, y, g, lambda C: make_probe("linear", None, C), n_repeats=1)
    return roc_auc_score(y, s)

old = [old_style(X0, rng.permutation(y0), scene) for _ in range(REP)]
new = [new_style(X0, rng.permutation(y0), scene) for _ in range(REP)]
print(f"[1] 무신호, 라벨 순열 {REP}회: 층화 없는 GroupKFold 평균 {np.mean(old):.3f} (sd {np.std(old):.3f}), "
      f"StratifiedGroupKFold+nested C 평균 {np.mean(new):.3f} (sd {np.std(new):.3f})")

big = rng.normal(size=(N, D)) * np.r_[np.full(10, 8.0), np.ones(D - 10)]
sig = np.zeros((N, D)); sig[:, 100] = (y0 - .5) * 0.9
X = big + sig
print("[2] 분산 작은 방향의 신호:")
for k in (None, 4, 16, 64, 128):
    s, _, evr = crossfit(X, y0, scene, lambda C, k=k: make_probe("linear", k, C), n_repeats=1)
    print(f"    pca={k or 'full'}: auroc={roc_auc_score(y0, s):.3f} evr={np.mean(evr) if evr else float('nan'):.3f}")

# 3. 장면 누수: 라벨이 장면에 의해 결정되고, 특징은 장면 정보만 담는다(진짜 신호 없음)
scene_lab = (rng.random(S) < 0.5).astype(int)[scene]
emb = rng.normal(size=(S, D))[scene] + 0.3 * rng.normal(size=(N, D))
game = np.arange(N)  # 게임 단위 분할 = 장면이 train/test에 걸침
a_game = new_style(emb, scene_lab, game)
a_scene = new_style(emb, scene_lab, scene)
print(f"[3] 장면 정보만 있는 특징: 게임 단위 분할 AUROC {a_game:.3f} (누수), 장면 단위 분할 AUROC {a_scene:.3f} (0.5 근처여야 함)")
