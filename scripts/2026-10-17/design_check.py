"""판정 규칙의 전제 점검(점수·AUROC는 계산하지 않음).
- 같은 게임 rollout 사이 프롬프트 동일 비율(동일하면 hidden·baseline 특징이 같아 쌍이 동점 -> AUROC가 0.5로 끌려감)
- step 1~3 생존(성공 episode가 step 3 전에 끝나면 행이 빠짐)
- 갈리는 게임 수, 장면 수(교차적합·bootstrap의 실제 단위 수)
"""
import collections, itertools, json
runs = ["data/multi_rollout_pilot/Qwen2.5-3B-Instruct", "data/multi_rollout_main_r2/Qwen2.5-3B-Instruct"]
eps, prompts = [], {}
for i, d in enumerate(runs):
    for l in open(f"{d}/episodes.jsonl"):
        e = json.loads(l); e["key"] = (i, e["episode"]); eps.append(e)
    for l in open(f"{d}/steps.jsonl"):
        r = json.loads(l)
        if r["step"] <= 3:
            prompts[(i, r["episode"], r["step"])] = r["prompt"]
by_game = collections.defaultdict(list)
for e in eps:
    by_game[e["task_id"]].append(e)
scene = lambda t: t.split("/")[-3].split("-")[-1]
mixed = [g for g, es in by_game.items() if 0 < sum(x["success"] for x in es) < len(es)]
print(f"episodes={len(eps)} games={len(by_game)} scenes(all)={len({scene(g) for g in by_game})}")
print(f"mixed games={len(mixed)} scenes(mixed)={len({scene(g) for g in mixed})}")
print("min steps among successes:", min(e["steps"] for e in eps if e["success"]),
      "| successes with steps<=3:", sum(1 for e in eps if e["success"] and e["steps"] <= 3))
for s in (1, 2, 3):
    tie = tot = 0; distinct = []
    for g in mixed:
        es = [e for e in by_game[g] if (e["key"][0], e["key"][1], s) in prompts]
        ps = [prompts[(e["key"][0], e["key"][1], s)] for e in es]
        distinct.append(len(set(ps)))
        for (a, pa), (b, pb) in itertools.combinations(zip(es, ps), 2):
            if a["success"] != b["success"]:  # 평가에 쓰이는 (실패, 성공) 쌍만
                tot += 1; tie += pa == pb
    print(f"step {s}: eval pairs={tot}, identical-prompt pairs={tie} ({tie/tot:.1%}), "
          f"mean distinct prompts per mixed game={sum(distinct)/len(distinct):.2f} (of ~8)")
