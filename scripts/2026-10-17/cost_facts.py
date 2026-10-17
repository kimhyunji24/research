"""비용 단위 비교용(탐색 구간 0~249만): step별 actor 토큰(프롬프트 + 후보 토큰 합)과 성공/실패 차이, 반복(루프) 비율, maxstep50 결과."""
import json, numpy as np, collections
d = "data/main_3b/Qwen2.5-3B-Instruct"
succ = {e["episode"]: e["success"] for e in map(json.loads, open(f"{d}/episodes.jsonl")) if e["episode"] < 250}
by = collections.defaultdict(lambda: collections.defaultdict(list)); loop = collections.defaultdict(list)
last = {}
for l in open(f"{d}/steps.jsonl"):
    r = json.loads(l)
    if r["episode"] >= 250: continue
    k = "S" if succ[r["episode"]] else "F"
    t = r["prompt_tokens"] + sum(r["cand_ntok"])           # prefill + 후보 점수화 토큰
    b = min(r["step"] // 5, 5)
    by[k][b].append(t)
    acts = last.setdefault(r["episode"], []); loop[k].append(r["action"] in acts[-6:]); acts.append(r["action"])
for k in "SF":
    print(k, "tokens/step by step bucket(0-4,5-9,..):", [int(np.mean(by[k][b])) if by[k][b] else None for b in range(6)],
          "| repeat-in-window rate", round(np.mean(loop[k]), 3))
allF = [t for b in by["F"].values() for t in b]; allS = [t for b in by["S"].values() for t in b]
print("mean tokens/step: success", int(np.mean(allS)), "fail", int(np.mean(allF)))
tot_steps = sum(len(v) for k in "SF" for v in by[k].values()); tot_tok = sum(sum(v) for k in "SF" for v in by[k].values())
f_tok = sum(sum(v) for v in by["F"].values())
print(f"failure share: steps {sum(len(v) for v in by['F'].values())/tot_steps:.3f}, tokens {f_tok/tot_tok:.3f}")
m = [json.loads(l) for l in open("data/maxstep50/Qwen2.5-3B-Instruct/episodes.jsonl")]
print("maxstep50: n", len(m), "success", sum(e["success"] for e in m), "success steps", sorted(e["steps"] for e in m if e["success"]))
