"""ReAct vs v2 성공률, 같은 250게임 게임 단위 짝지은 비교 (docs/prompt_robustness_plan_2026-10-17.md 보조, 판정 대상 아님).
방법(2026-10-20, 결과 보기 전 고정): task_id로 짝짓기(v2 = data/main_3b episode 0~249).
성공률과 Wilson 95% CI, 차이(ReAct - v2)와 게임 단위 bootstrap 95% CI(10,000회, seed 0), 정확 McNemar 양측 p(불일치 쌍 이항검정).
과제 유형별 표는 기술 통계만.
사용: python success_react_vs_v2.py <out_prefix>"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest

out = sys.argv[1]
def load(p):
    return pd.DataFrame([json.loads(l) for l in open(p)])
r = load("data/react_qwen25_3b/Qwen2.5-3B-Instruct/episodes.jsonl")
v = load("data/main_3b/Qwen2.5-3B-Instruct/episodes.jsonl")
v = v[v["episode"] < 250]
games = json.load(open("results/games_main3b_first250.json"))  # task_id와 같은 형식(.../game.tw-pddl). 2026-10-20 첫 실행은 여기서 접미사를 잘못 떼어 집합 점검에서 멈춤(결과 계산 전)
assert len(r) == 250 and len(v) == 250
assert r["task_id"].is_unique and v["task_id"].is_unique
assert set(r["task_id"]) == set(v["task_id"]) == set(games), "game sets differ, stop"
m = v[["task_id", "task_type", "success"]].merge(r[["task_id", "success"]], on="task_id", suffixes=("_v2", "_react"))
assert len(m) == 250
a, b = m["success_v2"].astype(int).values, m["success_react"].astype(int).values

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

rng = np.random.default_rng(0)
idx = rng.integers(0, 250, size=(10000, 250))
boot = b[idx].mean(1) - a[idx].mean(1)
n10 = int(((a == 1) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
p_mc = binomtest(n01, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
rows = []
for name, x in (("v2", a), ("react", b)):
    lo, hi = wilson(x.sum(), 250); rows.append(dict(prompt=name, n=250, success=int(x.sum()), rate=x.mean(), ci_lo=lo, ci_hi=hi))
s = pd.DataFrame(rows)
d = pd.DataFrame([dict(diff_react_minus_v2=b.mean() - a.mean(), ci_lo=np.quantile(boot, 0.025), ci_hi=np.quantile(boot, 0.975),
                       both_success=int(((a == 1) & (b == 1)).sum()), v2_only=n10, react_only=n01, both_fail=int(((a == 0) & (b == 0)).sum()),
                       p_mcnemar_exact=p_mc)])
t = m.groupby("task_type").agg(n=("task_id", "size"), v2=("success_v2", "sum"), react=("success_react", "sum")).reset_index()
s.to_csv(out + ".csv", index=False); d.to_csv(out + "_paired.csv", index=False); t.to_csv(out + "_by_type.csv", index=False)
pd.set_option("display.width", 200)
print(s.round(4).to_string(index=False)); print(); print(d.round(4).to_string(index=False)); print(); print(t.to_string(index=False))
