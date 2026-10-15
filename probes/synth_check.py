"""합성 데이터 점검: 신호가 (a) 소수 방향에 있을 때, (b) 분산 작은 방향에 있을 때 PCA k가 어떻게 달라지는지,
그리고 무작위 라벨이 0.5 근처인지 확인한다. 실제 결과가 아니라 코드/가정 검증용이다."""
import numpy as np, pandas as pd
from lib import make_model, round_eval

rng = np.random.default_rng(0)
N, D = 250, 1536
y = (rng.random(N) < 0.8).astype(int)
big = rng.normal(size=(N, D)) * np.r_[np.full(10, 8.0), np.ones(D - 10)]  # 분산 큰 소수 방향(길이/문맥류)
sig = np.zeros((N, D)); sig[:, 100] = (y - .5) * 0.6                    # 분산이 작은 곳에 있는 약한 신호
X = big + sig
df = pd.DataFrame(dict(step=1, y=y, task_id=rng.integers(0, 120, N)))
for k in (None, 4, 16, 64, 128):
    r = round_eval(df, X, 1, lambda: make_model("linear", pca_k=k))
    sh = round_eval(df, X, 1, lambda: make_model("linear", pca_k=k), shuffle_labels=True)
    print(f"pca={k or 'full'}: auroc={r['auroc']:.3f} [{r['ci_lo']:.2f},{r['ci_hi']:.2f}]  shuffled={sh['auroc']:.3f}")
