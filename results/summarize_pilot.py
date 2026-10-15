import json, sys, collections
import numpy as np
from math import sqrt
d = sys.argv[1]
E = [json.loads(l) for l in open(d + "/episodes.jsonl")]
S = [json.loads(l) for l in open(d + "/steps.jsonl")]
meta = {m["task_id"]: m for m in json.load(open(sys.argv[2]))}
def wilson(k, n, z=1.96):
    p = k / n; dd = 1 + z * z / n; c = (p + z * z / (2 * n)) / dd; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return round(c - h, 3), round(c + h, 3)
print("episodes", len(E), "games", len({e["task_id"] for e in E}), "T", {e.get("sample_temperature") for e in E},
      "rollout_idx range", min(e["rollout_idx"] for e in E), max(e["rollout_idx"] for e in E))
k = sum(e["success"] for e in E); print(f"전체 성공 {k}/{len(E)} = {k/len(E):.3f}  CI {wilson(k, len(E))}")
by = collections.defaultdict(list)
for e in E: by[e["task_id"]].append(e)
rows = []
for g, es in by.items():
    n = len(es); s = sum(x["success"] for x in es)
    rows.append((g, meta[g]["greedy_success"], n, s, meta[g]["task_type"]))
mixed = [r for r in rows if 0 < r[3] < r[2]]
print(f"갈리는 게임(성공과 실패가 모두 나옴): {len(mixed)}/{len(rows)} = {len(mixed)/len(rows):.2f}")
print(f"  모두 성공 {sum(r[3]==r[2] for r in rows)}, 모두 실패 {sum(r[3]==0 for r in rows)}")
for gs in (True, False):
    sub = [r for r in rows if r[1] == gs]
    print(f"탐욕 {'성공' if gs else '실패'} 게임 {len(sub)}개: 샘플링 성공률 {sum(r[3] for r in sub)}/{sum(r[2] for r in sub)}"
          f" = {sum(r[3] for r in sub)/sum(r[2] for r in sub):.3f}, 갈리는 게임 {sum(0<r[3]<r[2] for r in sub)}")
print("게임별 성공 횟수(탐욕결과, 8회 중):", sorted([(int(r[1]), r[3]) for r in rows]))
# 같은 게임에서 rollout들이 서로 다른 경로인가
div = []
acts = collections.defaultdict(lambda: collections.defaultdict(list))
for s in S: acts[(s["task_id"], s["episode"])]["a"].append(s["action"])
for g in by:
    paths = [tuple(acts[(g, e["episode"])]["a"]) for e in by[g]]
    div.append(len(set(paths)) / len(paths))
print(f"게임 내 서로 다른 경로 비율(평균): {np.mean(div):.2f}")
first = collections.Counter()
for g in by:
    for e in by[g]: first[acts[(g, e["episode"])]["a"][0] == "go to " + "x"] += 0
steps = [e["steps"] for e in E]; print("steps 평균", round(np.mean(steps), 1), "성공 episode steps", sorted(e["steps"] for e in E if e["success"])[:20])
print("판정 규칙(갈리는 게임 >= 6/20 = 30%):", "충족" if len(mixed) >= 6 else "미충족")
