# 파일럿 기술 통계(탐색, 검정 아님): 같은 20게임에서 모델별 행동 패턴
import json, collections
games = set(json.load(open("results/games_pilot_models_first20.json")))
runs = {"Qwen2.5-3B": "data/main_3b/Qwen2.5-3B-Instruct", "Qwen3-4B": "data/pilot_models/Qwen3-4B",
        "Llama-3.2-3B": "data/pilot_models/Llama-3.2-3B-Instruct", "SmolLM3-3B": "data/pilot_models/SmolLM3-3B",
        "Phi-3.5-mini": "data/pilot_models/Phi-3.5-mini-instruct"}
for name, d in runs.items():
    eps = [e for e in map(json.loads, open(f"{d}/episodes.jsonl")) if e["task_id"] in games]
    keep = {e["episode"] for e in eps}
    st = collections.defaultdict(list)
    for r in map(json.loads, open(f"{d}/steps.jsonl")):
        if r["episode"] in keep: st[r["episode"]].append(r)
    n = rep = nothing = 0; kinds = collections.Counter(); uniq = []
    for ep, rows in st.items():
        rows.sort(key=lambda r: r["step"]); acts = [r["action"] for r in rows]
        uniq.append(len(set(acts)) / len(acts))
        for i, r in enumerate(rows):
            n += 1; rep += acts[i] in acts[max(0, i - 6):i]
            nothing += not r.get("ok", True)
            kinds[acts[i].split()[0]] += 1
    top = ", ".join(f"{k} {v/n:.0%}" for k, v in kinds.most_common(5))
    print(f"{name:13s} success={sum(e['success'] for e in eps)}/{len(eps)} steps={n} repeat_in_last6={rep/n:.0%} "
          f"nothing_happens={nothing/n:.0%} unique_action_ratio={sum(uniq)/len(uniq):.2f} | top verbs: {top}")
