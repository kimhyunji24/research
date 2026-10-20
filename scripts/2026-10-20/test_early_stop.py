"""probes/early_stop.py 곡선·지표 점검(합성 데이터만, 실제 라벨 미사용).
1) 손으로 계산한 작은 예 2) 학습 경로(모든 τ)로 넣은 'step 번호' 점수 = cap 경로 3) 라벨과 무관한 점수는 cap과 비슷한가
4) 라벨 점수는 오라클에 가까운가 5) bootstrap 가중치·fold τ 동작."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "probes"))
from early_stop import (CAP_TAUS, LOSS_MAIN, Curve, fold_tau_eval, metric, oracle_metric, partial_area, scene_weights)

# 1) 손 계산: 게임 A 실패(100행), 게임 B 성공 s=3(3행)
a = np.zeros(100); a[5] = 2.0; a[10] = 3.0     # A의 prefix max 기록: (0,t0) (2,t5) (3,t10)
b = np.array([0.0, 1.0, 0.5])                 # B 기록: (0,t0) (1,t1)
c = Curve([a, b], full=[100, 3], y=np.array([1, 0]))
sav, loss = c.at([np.inf, 3.0, 2.0, 1.0, 0.5, 0.0])
# τ=inf: 103/103 / τ=3: A t10 -> 10+3=13 / τ=2: A t5 -> 5+3=8 / τ=1: A t5, B t1(손실) -> 6 / τ=.5: 같음 / τ=0: A0,B0 -> 0
exp_cost = np.array([103, 13, 8, 6, 6, 0]); exp_loss = np.array([0, 0, 0, 1, 1, 1])
assert np.allclose(1 - sav, exp_cost / 103), (sav, exp_cost)
assert np.allclose(loss, exp_loss), loss
print("1) hand example OK")

# 합성 게임: 장면 60개, 게임 600개, 성공 30%(성공 step 1~100 균등), 실패는 100행
rng = np.random.default_rng(0)
G = 600; scenes = rng.integers(0, 60, G); y = (rng.random(G) > 0.3).astype(int)
full = np.where(y == 1, 100, rng.integers(1, 101, G)).astype(float)
steps = [np.arange(int(f)) for f in full]

# 2) step 번호 점수: 모든 고유값 τ로 훑어도(학습 경로) cap 경로와 주 지표가 같다(차이는 τ=0 = 손실 100%뿐)
cs = Curve([s.astype(float) for s in steps], full, y)
m_cap = metric(cs, np.r_[CAP_TAUS, np.inf]); m_all = metric(cs, cs.taus_all())
assert abs(m_cap - m_all) < 1e-12, (m_cap, m_all)
print(f"2) step score via all-τ path == cap path: {m_cap:.4f}")

# 3) 라벨과 무관한 점수: (a) step + 작은 잡음 (b) 순수 잡음 (c) 게임마다 상수 잡음
def m_of(scores):
    cv = Curve(scores, full, y); return metric(cv, cv.taus_all())
ma = m_of([s + rng.normal(0, 0.1, len(s)) for s in steps])
mb = m_of([rng.random(len(s)) for s in steps])
mc = m_of([np.full(len(s), rng.random()) for s in steps])
print(f"3) label-independent: step+noise {ma:.4f} (cap {m_cap:.4f}), pure noise {mb:.4f}, per-game const {mc:.4f}, oracle {oracle_metric(full, y):.4f}")
assert abs(ma - m_cap) < 0.02, "step+잡음이 cap과 크게 다르다"
assert mb <= m_cap + 0.02 and mc <= m_cap + 0.02, "라벨과 무관한 점수가 cap보다 눈에 띄게 좋다"

# 4) 라벨 점수(실패면 1): 오라클과 같다
md = m_of([np.full(len(s), float(yy)) for s, yy in zip(steps, y)])
assert abs(md - oracle_metric(full, y)) < 1e-12, (md, oracle_metric(full, y))
print(f"4) label score == oracle: {md:.4f}")

# 5) bootstrap 가중치: 장면 수 보존, 가중치 1이면 원래 값. 라벨 무관 점수 - cap 차이의 CI가 0을 넘지 않는다
W = scene_weights(scenes, 200, 0)
assert all(abs(w.sum() - G) < G for w in W) and len(W) == 200
cb = Curve([rng.random(len(s)) for s in steps], full, y)
d = np.array([metric(cb, cb.taus_all(), w) - metric(cs, np.r_[CAP_TAUS, np.inf], w) for w in W])
lo, hi = np.quantile(d, [.025, .975])
print(f"5) pure noise - cap: CI [{lo:.4f}, {hi:.4f}] (우수 판정이 나오면 안 됨)")
assert not lo > 0
s5, l5 = fold_tau_eval(cs, np.r_[CAP_TAUS, np.inf], scenes, y)
print(f"   fold τ(5%) on cap: test saving {s5:.4f}, test loss {l5:.4f}")
assert partial_area(np.array([0.0, 0.5]), np.array([0.0, 0.2]), LOSS_MAIN) == 0.0
print("ALL OK")
