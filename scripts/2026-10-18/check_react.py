"""ReAct 강건성 수집 직후 정합성 점검(docs/prompt_robustness_plan_2026-10-17.md). 성공률 등 결과는 출력하지 않는다.
사용: python -I check_react.py <run dir> <game list>"""
import json, sys, collections, numpy as np
M, GL = sys.argv[1:3]
games = json.load(open(GL))
eps = [json.loads(l) for l in open(f"{M}/episodes.jsonl")]; steps = [json.loads(l) for l in open(f"{M}/steps.jsonl")]
cnt = collections.Counter(e["task_id"] for e in eps)
print(f"episodes={len(eps)} (expect {len(games)}) all in list={set(cnt)==set(games)} each once={set(cnt.values())=={1}} ids contiguous={[e['episode'] for e in eps]==list(range(len(eps)))}")
per = collections.Counter(r["episode"] for r in steps)
print(f"steps rows={len(steps)} match episodes.steps={all(per[e['episode']]==e['steps'] for e in eps)} max steps={max(e['steps'] for e in eps)}")
nh = nan = inf = 0
for e in eps:
    h = np.load(f"{M}/hidden_shards/ep{e['episode']:05d}.npy"); assert h.shape[0] == e["steps"]
    nh += h.shape[0]; nan += int(np.isnan(h).sum()); inf += int(np.isinf(h).sum()); shp = h.shape[1:]
print(f"hidden rows={nh} == steps {nh==len(steps)} shape={shp} NaN={nan} inf={inf}")
try:
    m = np.load(f"{M}/hidden.npy", mmap_mode="r"); print(f"merged hidden.npy rows={m.shape[0]} == {m.shape[0]==nh}")
except FileNotFoundError:
    print("merged hidden.npy: missing")
